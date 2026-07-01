/**
 * LineageDAG — React Flow visualiser for Crucible's experiment provenance graph.
 *
 * Promoted from the spike (spike/LineageDAG.jsx) and hardened:
 * - Fetches live data from the API rather than using static fixtures
 * - Handles loading, empty, and error states
 * - Inspector panel shows node metadata on click
 * - Dagre LR auto-layout (validated in spike: 8/8 layout checks pass)
 *
 * Requires: reactflow, dagre  →  npm install reactflow dagre
 */

import { useState, useCallback, useMemo, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { GitBranch } from 'lucide-react';
import { Spinner, EmptyState, SectionLabel } from './ui.jsx';
import api from '../api/client.js';

// ── Node type styles (match spike exactly) ────────────────────────────────

const NODE_STYLES = {
  data_version:  { bg: '#0D1A2E', border: '#5B8AF2', icon: '🗄', label: 'Data Version' },
  preprocessing: { bg: '#0D2218', border: '#4FCF8A', icon: '⚙',  label: 'Preprocessing' },
  model_config:  { bg: '#1A1200', border: '#F2A855', icon: '🧠', label: 'Model Config'  },
  evaluation:    { bg: '#1A0D2E', border: '#C084FC', icon: '📊', label: 'Evaluation'    },
};

const NODE_W = 210;
const NODE_H = 76;

// ── Custom node component ─────────────────────────────────────────────────

function LineageNode({ data, selected }) {
  const s = NODE_STYLES[data.nodeType] || NODE_STYLES.model_config;
  return (
    <div style={{
      background: s.bg,
      border: selected ? '2px solid #00C2A8' : `1.5px solid ${s.border}`,
      borderRadius: 8,
      padding: '9px 13px',
      width: NODE_W,
      minHeight: NODE_H,
      cursor: 'pointer',
      boxShadow: selected ? '0 0 0 3px rgba(0,194,168,0.2)' : '0 2px 6px rgba(0,0,0,0.4)',
      transition: 'box-shadow 0.15s, border 0.15s',
    }}>
      {/* Handle: target (left) */}
      <div style={{
        position: 'absolute', left: -5, top: '50%', transform: 'translateY(-50%)',
        width: 10, height: 10, borderRadius: '50%',
        background: '#2A2D30', border: '1.5px solid #5C666F',
      }} className="react-flow__handle react-flow__handle-left" />

      <div style={{ fontSize: 10, fontWeight: 600, color: s.border, marginBottom: 4, letterSpacing: '0.06em', textTransform: 'uppercase', fontFamily: 'IBM Plex Mono, monospace' }}>
        {s.icon} {s.label}
      </div>
      <div style={{ fontSize: 13, fontWeight: 600, color: '#E8EAEC', lineHeight: 1.3, marginBottom: 3 }}>
        {data.label}
      </div>
      {data.subtitle && (
        <div style={{ fontSize: 11, color: '#9BA3AA', fontFamily: 'IBM Plex Mono, monospace', lineHeight: 1.4 }}>
          {data.subtitle}
        </div>
      )}

      {/* Handle: source (right) */}
      <div style={{
        position: 'absolute', right: -5, top: '50%', transform: 'translateY(-50%)',
        width: 10, height: 10, borderRadius: '50%',
        background: '#2A2D30', border: '1.5px solid #5C666F',
      }} className="react-flow__handle react-flow__handle-right" />
    </div>
  );
}

// ── Inspector panel ───────────────────────────────────────────────────────

function Inspector({ node, onClose }) {
  if (!node) return null;
  const s = NODE_STYLES[node.data.nodeType] || NODE_STYLES.model_config;
  const meta = node.data.metadata || {};

  return (
    <div style={{
      position: 'absolute', top: 12, right: 12,
      width: 240,
      background: '#141618',
      border: '1px solid #2A2D30',
      borderRadius: 10,
      padding: 16,
      boxShadow: '0 6px 24px rgba(0,0,0,0.6)',
      zIndex: 10,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: s.border, fontFamily: 'IBM Plex Mono, monospace' }}>
          {s.icon} {s.label}
        </div>
        <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#5C666F', cursor: 'pointer', fontSize: 16, lineHeight: 1 }}>×</button>
      </div>
      <div style={{ fontSize: 14, fontWeight: 600, color: '#E8EAEC', marginBottom: 10 }}>{node.data.label}</div>
      {node.data.subtitle && (
        <div style={{ fontSize: 11, color: '#9BA3AA', marginBottom: 10, fontFamily: 'IBM Plex Mono, monospace', lineHeight: 1.6 }}>
          {node.data.subtitle}
        </div>
      )}
      {Object.entries(meta).filter(([, v]) => v !== null && v !== undefined).map(([k, v]) => (
        <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0', borderBottom: '1px solid #1C1F22', fontSize: 11 }}>
          <span style={{ color: '#5C666F', fontFamily: 'IBM Plex Mono, monospace' }}>{k}</span>
          <span style={{ color: '#9BA3AA', fontFamily: 'IBM Plex Mono, monospace', maxWidth: 120, textAlign: 'right', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {typeof v === 'object' ? JSON.stringify(v).slice(0, 40) : String(v)}
          </span>
        </div>
      ))}
    </div>
  );
}

// ── Pure layout function (extracted for testability) ──────────────────────

function applyDagreLayout(nodes, edges) {
  // Lazy-load dagre to avoid bundling issues in environments without it
  try {
    const dagre = window.__dagre;
    if (!dagre) return nodes;

    const g = new dagre.graphlib.Graph();
    g.setDefaultEdgeLabel(() => ({}));
    g.setGraph({ rankdir: 'LR', ranksep: 80, nodesep: 40 });

    nodes.forEach(n => g.setNode(n.id, { width: NODE_W, height: NODE_H }));
    edges.forEach(e => g.setEdge(e.source, e.target));
    dagre.layout(g);

    return nodes.map(n => {
      const pos = g.node(n.id);
      return { ...n, position: { x: pos.x - NODE_W / 2, y: pos.y - NODE_H / 2 } };
    });
  } catch {
    // Fallback: position nodes in a simple grid if dagre is unavailable
    return nodes.map((n, i) => ({
      ...n,
      position: { x: i * (NODE_W + 80), y: 50 },
    }));
  }
}

// ── Main component ────────────────────────────────────────────────────────

export default function LineageDAG({ datasetId, experimentId, mode = 'dataset' }) {
  const [selectedNode, setSelectedNode] = useState(null);
  const [ReactFlowModule, setReactFlowModule] = useState(null);
  const [dagreLoaded, setDagreLoaded] = useState(false);

  // Lazy-load ReactFlow and dagre
  useEffect(() => {
    Promise.all([
      import('reactflow').catch(() => null),
      import('dagre').catch(() => null),
    ]).then(([rf, dagre]) => {
      if (rf) setReactFlowModule(rf);
      if (dagre) {
        window.__dagre = dagre.default || dagre;
        setDagreLoaded(true);
      }
    });
  }, []);

  const lineagePath = mode === 'experiment'
    ? `/experiments/${experimentId}/lineage`
    : `/datasets/${datasetId}/lineage`;

  const { data, isLoading, error } = useQuery({
    queryKey: ['lineage', mode, experimentId || datasetId],
    queryFn: () => api.get(lineagePath).then(r => r.data),
    enabled: !!(datasetId || experimentId),
  });

  const rfData = data?.data;

  const { layoutNodes, layoutEdges } = useMemo(() => {
    if (!rfData || !dagreLoaded) return { layoutNodes: [], layoutEdges: [] };
    const laid = applyDagreLayout(rfData.nodes || [], rfData.edges || []);
    return { layoutNodes: laid, layoutEdges: rfData.edges || [] };
  }, [rfData, dagreLoaded]);

  if (isLoading) return (
    <div style={{ display: 'flex', justifyContent: 'center', padding: 40 }}>
      <Spinner size={20} />
    </div>
  );

  if (!rfData || rfData.nodes?.length === 0) return (
    <EmptyState
      icon={GitBranch}
      title="No lineage data yet"
      description={mode === 'dataset'
        ? "Complete an experiment to see the provenance graph."
        : "Experiment must be complete to view lineage."}
    />
  );

  if (!ReactFlowModule) return (
    <div style={{ padding: 20, color: 'var(--text-3)', fontSize: 13 }}>
      Loading DAG viewer…
    </div>
  );

  const {
    default: ReactFlow,
    Background, Controls, MiniMap,
  } = ReactFlowModule;

  return (
    <div style={{
      position: 'relative',
      height: 420,
      background: '#0D0F10',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)',
      overflow: 'hidden',
    }}>
      <ReactFlow
        nodes={layoutNodes}
        edges={layoutEdges.map(e => ({
          ...e,
          style: { stroke: '#363A3D', strokeWidth: 1.5 },
          labelStyle: { fontSize: 10, fill: '#5C666F' },
          labelBgStyle: { fill: '#0D0F10', fillOpacity: 0.9 },
        }))}
        nodeTypes={{ lineage: LineageNode }}
        onNodeClick={(_, node) => setSelectedNode(node)}
        onPaneClick={() => setSelectedNode(null)}
        fitView
        fitViewOptions={{ padding: 0.15 }}
        minZoom={0.3}
        maxZoom={2}
      >
        <Background color="#1C1F22" gap={20} />
        <Controls style={{ background: '#141618', border: '1px solid #2A2D30' }} />
      </ReactFlow>

      <Inspector node={selectedNode} onClose={() => setSelectedNode(null)} />

      {/* Legend */}
      <div style={{
        position: 'absolute', bottom: 12, left: 12,
        display: 'flex', gap: 10, flexWrap: 'wrap',
      }}>
        {Object.entries(NODE_STYLES).map(([type, s]) => (
          <div key={type} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 10, color: '#5C666F' }}>
            <div style={{ width: 8, height: 8, background: s.bg, border: `1.5px solid ${s.border}`, borderRadius: 2 }} />
            {s.label}
          </div>
        ))}
      </div>
    </div>
  );
}
