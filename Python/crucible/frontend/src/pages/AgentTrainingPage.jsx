import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  GitBranch, Package, Upload, Play, CheckCircle2,
  XCircle, Clock, Sparkles, FileJson, Archive, Trash2, Award, Loader2,
} from 'lucide-react';
import { agentTraining } from '../api/client.js';
import { PageHeader, Card, Button, Spinner, SectionLabel, EmptyState } from '../components/ui.jsx';

const TABS = ['Traces', 'Training Data', 'Registry'];

export default function AgentTrainingPage() {
  const [tab, setTab] = useState('Traces');

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 32 }}>
      <PageHeader
        title="Agent Training Pipeline"
        subtitle="Capture real agent sessions → convert to training data → fine-tune → export & import elsewhere"
      />

      <div style={{ display: 'flex', gap: 2, marginBottom: 24, borderBottom: '1px solid var(--border)' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            style={{
              padding: '10px 20px', background: 'none', border: 'none', cursor: 'pointer',
              fontSize: 13, fontWeight: 500,
              color: tab === t ? 'var(--accent)' : 'var(--text-3)',
              borderBottom: tab === t ? '2px solid var(--accent)' : '2px solid transparent',
              marginBottom: -1,
            }}>
            {t}
          </button>
        ))}
      </div>

      {tab === 'Traces' && <TracesTab />}
      {tab === 'Training Data' && <TrainingDataTab />}
      {tab === 'Registry' && <RegistryTab />}
    </div>
  );
}

// ── Traces Tab ─────────────────────────────────────────────────────────────────

function TracesTab() {
  const qc = useQueryClient();
  const [agentTypeFilter, setAgentTypeFilter] = useState('');

  const { data, isLoading } = useQuery({
    queryKey: ['agent-traces', agentTypeFilter],
    queryFn: () => agentTraining.listTraces(agentTypeFilter ? { agent_type: agentTypeFilter } : {}),
  });

  const scoreMut = useMutation({
    mutationFn: () => agentTraining.scoreTraces(50),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['agent-traces'] }),
  });

  const traces = data?.data || [];
  const pending = traces.filter(t => t.score_pending).length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <Card style={{ padding: '14px 18px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
          <p style={{ margin: 0, fontSize: 13, color: 'var(--text-3)', lineHeight: 1.6 }}>
            Run an agent goal with <code style={{ fontFamily: 'var(--font-mono)', background: 'var(--bg-3)', padding: '1px 5px', borderRadius: 4 }}>capture: true</code> on{' '}
            <code style={{ fontFamily: 'var(--font-mono)', background: 'var(--bg-3)', padding: '1px 5px', borderRadius: 4 }}>POST /agent/run</code> or{' '}
            <code style={{ fontFamily: 'var(--font-mono)', background: 'var(--bg-3)', padding: '1px 5px', borderRadius: 4 }}>/agent/multi/run</code> to populate this list.
          </p>
          <div style={{ display: 'flex', gap: 8, flexShrink: 0 }}>
            <select value={agentTypeFilter} onChange={e => setAgentTypeFilter(e.target.value)}
              style={{ background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 6, padding: '6px 10px', fontSize: 12 }}>
              <option value="">All types</option>
              <option value="react">ReAct</option>
              <option value="multi">Multi-agent</option>
            </select>
            <Button variant="primary" onClick={() => scoreMut.mutate()} loading={scoreMut.isPending} disabled={pending === 0}>
              <Sparkles size={13} style={{ marginRight: 5 }} /> Score pending ({pending})
            </Button>
          </div>
        </div>
      </Card>

      {isLoading && <Spinner />}

      {!isLoading && traces.length === 0 && (
        <EmptyState icon={GitBranch} title="No traces captured yet"
          description="Captured agent sessions will appear here, ready to convert into fine-tuning data." />
      )}

      {traces.length > 0 && (
        <Card style={{ padding: 0, overflow: 'hidden' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
                {['Type', 'Goal', 'Tool calls', 'Status', 'Quality', 'Captured'].map(h => (
                  <th key={h} style={{ padding: '10px 14px', fontSize: 11, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.05em', fontFamily: 'var(--font-mono)' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {traces.map(t => (
                <tr key={t.id} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td style={{ padding: '10px 14px' }}>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: t.agent_type === 'multi' ? 'var(--blue)' : 'var(--accent)' }}>
                      {t.agent_type}
                    </span>
                  </td>
                  <td style={{ padding: '10px 14px', maxWidth: 360, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{t.goal}</td>
                  <td style={{ padding: '10px 14px', color: 'var(--text-3)' }}>{t.n_tool_calls}</td>
                  <td style={{ padding: '10px 14px' }}>
                    {t.succeeded
                      ? <span style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--green)' }}><CheckCircle2 size={13} /> ok</span>
                      : <span style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--red)' }}><XCircle size={13} /> failed</span>}
                  </td>
                  <td style={{ padding: '10px 14px' }}>
                    {t.score_pending
                      ? <span style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--text-3)' }}><Clock size={12} /> pending</span>
                      : <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: t.quality_score >= 0.7 ? 'var(--green)' : t.quality_score >= 0.4 ? '#F39C12' : 'var(--red)' }}>
                          {t.quality_score != null ? t.quality_score.toFixed(2) : '—'}
                        </span>}
                  </td>
                  <td style={{ padding: '10px 14px', color: 'var(--text-3)', fontSize: 11 }}>
                    {t.created_at ? new Date(t.created_at).toLocaleString() : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  );
}

// ── Training Data Tab ─────────────────────────────────────────────────────────

function TrainingDataTab() {
  const [format, setFormat] = useState('alpaca');
  const [minGap, setMinGap] = useState(0.15);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleConvert() {
    setLoading(true);
    try {
      const data = await agentTraining.getTrainingData(format, minGap);
      setResult(data);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <Card>
        <SectionLabel>Convert traces to training data</SectionLabel>
        <p style={{ fontSize: 12, color: 'var(--text-3)', lineHeight: 1.7, marginBottom: 16 }}>
          Use the output directly as the request body for{' '}
          <code style={{ fontFamily: 'var(--font-mono)', background: 'var(--bg-3)', padding: '1px 5px', borderRadius: 4 }}>POST /fine-tuning/jobs</code>{' '}
          (alpaca/sharegpt) or{' '}
          <code style={{ fontFamily: 'var(--font-mono)', background: 'var(--bg-3)', padding: '1px 5px', borderRadius: 4 }}>POST /fine-tuning/jobs/dpo</code>{' '}
          (dpo).
        </p>
        <div style={{ display: 'flex', gap: 12, alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div>
            <SectionLabel>Format</SectionLabel>
            <select value={format} onChange={e => setFormat(e.target.value)}
              style={{ background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 6, padding: '7px 10px', fontSize: 13 }}>
              <option value="alpaca">Alpaca (SFT)</option>
              <option value="sharegpt">ShareGPT (SFT)</option>
              <option value="dpo">DPO pairs</option>
            </select>
          </div>
          {format === 'dpo' && (
            <div>
              <SectionLabel>Min score gap</SectionLabel>
              <input type="number" step="0.05" min="0" max="1" value={minGap}
                onChange={e => setMinGap(parseFloat(e.target.value))}
                style={{ width: 90, background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 6, padding: '7px 10px', fontSize: 13 }} />
            </div>
          )}
          <Button variant="primary" onClick={handleConvert} loading={loading}>
            <FileJson size={13} style={{ marginRight: 5 }} /> Convert
          </Button>
        </div>
      </Card>

      {result && (
        <Card>
          <div style={{ display: 'flex', gap: 24, marginBottom: 14 }}>
            <div><SectionLabel>Format</SectionLabel><span style={{ fontSize: 14, fontWeight: 700 }}>{result.format}</span></div>
            <div><SectionLabel>Samples</SectionLabel><span style={{ fontSize: 14, fontWeight: 700, color: 'var(--accent)' }}>{result.n_samples}</span></div>
            {result.stats && (
              <>
                <div><SectionLabel>Pairs</SectionLabel><span style={{ fontSize: 14 }}>{result.stats.n_pairs}</span></div>
                <div><SectionLabel>Skipped (unscored)</SectionLabel><span style={{ fontSize: 14, color: 'var(--text-3)' }}>{result.stats.n_traces_skipped_unscored}</span></div>
              </>
            )}
          </div>
          <SectionLabel>Preview (first 3)</SectionLabel>
          <pre style={{
            background: 'var(--bg-3)', border: '1px solid var(--border)', borderRadius: 8,
            padding: 14, fontSize: 11, fontFamily: 'var(--font-mono)', overflow: 'auto', maxHeight: 360,
          }}>
            {JSON.stringify(result.samples.slice(0, 3), null, 2)}
          </pre>
        </Card>
      )}
    </div>
  );
}

// ── Registry Tab ───────────────────────────────────────────────────────────────

function RegistryTab() {
  const qc = useQueryClient();
  const [importing, setImporting] = useState(false);
  const [importMsg, setImportMsg] = useState('');

  const { data: agents, isLoading } = useQuery({
    queryKey: ['registered-agents'],
    queryFn: agentTraining.listAgents,
  });

  const archiveMut = useMutation({
    mutationFn: agentTraining.archiveAgent,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['registered-agents'] }),
  });

  const benchmarkMut = useMutation({
    mutationFn: agentTraining.benchmarkAgent,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['registered-agents'] }),
  });

  async function handleImport(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setImporting(true);
    setImportMsg('');
    try {
      const data = await agentTraining.importBundle(file);
      setImportMsg(`Imported "${data.name}" successfully.`);
      qc.invalidateQueries({ queryKey: ['registered-agents'] });
    } catch (err) {
      setImportMsg(err.response?.data?.detail || 'Import failed.');
    } finally {
      setImporting(false);
      e.target.value = '';
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <Card style={{ padding: '14px 18px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
          <p style={{ margin: 0, fontSize: 13, color: 'var(--text-3)' }}>
            Import a <code style={{ fontFamily: 'var(--font-mono)', background: 'var(--bg-3)', padding: '1px 5px', borderRadius: 4 }}>.crucible</code> bundle to register a fine-tuned agent.
          </p>
          <label>
            <input type="file" accept=".crucible" onChange={handleImport} style={{ display: 'none' }} />
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: 6, cursor: 'pointer',
              padding: '8px 14px', borderRadius: 6, fontSize: 13, fontWeight: 600,
              background: 'var(--accent)', color: '#000',
            }}>
              {importing ? <Loader2 size={13} style={{ animation: 'spin 0.8s linear infinite' }} /> : <Upload size={13} />} Import bundle
            </span>
          </label>
        </div>
        {importMsg && <p style={{ fontSize: 12, marginTop: 10, color: importMsg.includes('success') ? 'var(--green)' : 'var(--red)' }}>{importMsg}</p>}
      </Card>

      {isLoading && <Spinner />}

      {!isLoading && (!agents || agents.length === 0) && (
        <EmptyState icon={Package} title="No agents registered"
          description="Import a .crucible bundle, or export one after fine-tuning on captured traces." />
      )}

      {agents && agents.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {agents.map(a => (
            <Card key={a.id} style={{ padding: '16px 20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <Archive size={15} style={{ color: 'var(--accent)' }} />
                    <span style={{ fontSize: 15, fontWeight: 700 }}>{a.name}</span>
                    <span style={{
                      fontSize: 10, padding: '2px 8px', borderRadius: 10, fontFamily: 'var(--font-mono)',
                      background: a.training_method === 'dpo' ? 'rgba(155,89,182,0.15)' : 'rgba(52,152,219,0.15)',
                      color: a.training_method === 'dpo' ? '#9B59B6' : 'var(--blue)',
                    }}>
                      {a.training_method.toUpperCase()}
                    </span>
                  </div>
                  {a.description && <p style={{ fontSize: 12, color: 'var(--text-3)', margin: '4px 0 0' }}>{a.description}</p>}
                  <div style={{ display: 'flex', gap: 16, marginTop: 8, fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>
                    <span>base: {a.base_model}</span>
                    <span>traces: {a.n_training_traces}</span>
                    {a.has_benchmark && <span style={{ color: 'var(--green)', display: 'flex', alignItems: 'center', gap: 3 }}><Award size={11} /> benchmarked</span>}
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 8 }}>
                  <Button variant="ghost" onClick={() => benchmarkMut.mutate(a.name)} loading={benchmarkMut.isPending && benchmarkMut.variables === a.name}>
                    <Play size={13} style={{ marginRight: 4 }} /> Benchmark
                  </Button>
                  <Button variant="ghost" onClick={() => archiveMut.mutate(a.name)}>
                    <Trash2 size={13} />
                  </Button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
