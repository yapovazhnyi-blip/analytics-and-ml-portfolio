import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Activity, X, AlertTriangle, CheckCircle2, AlertCircle } from 'lucide-react';
import { drift, datasets } from '../api/client.js';
import { Button, Spinner, SectionLabel } from './ui.jsx';

// ── Severity helpers ──────────────────────────────────────────────────────────

const SEVERITY_STYLE = {
  stable:      { color: '#2ECC71', bg: 'rgba(46,204,113,0.1)',  label: 'Stable',      icon: CheckCircle2 },
  slight:      { color: '#F39C12', bg: 'rgba(243,156,18,0.1)',  label: 'Slight',      icon: AlertTriangle },
  significant: { color: '#E67E22', bg: 'rgba(230,126,34,0.1)',  label: 'Significant', icon: AlertTriangle },
  critical:    { color: '#E74C3C', bg: 'rgba(231,76,60,0.1)',   label: 'Critical',    icon: AlertCircle },
};

function SeverityBadge({ severity, size = 'sm' }) {
  const s = SEVERITY_STYLE[severity] || SEVERITY_STYLE.stable;
  const Icon = s.icon;
  const pad = size === 'lg' ? '8px 16px' : '3px 8px';
  const fs  = size === 'lg' ? 14 : 11;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      padding: pad, borderRadius: 6, fontSize: fs, fontWeight: 600,
      color: s.color, background: s.bg, border: `1px solid ${s.color}40`,
      fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.04em',
    }}>
      <Icon size={fs} />
      {s.label}
    </span>
  );
}

// ── PSI bar ───────────────────────────────────────────────────────────────────

function PSIBar({ psi }) {
  if (psi == null) return <span style={{ fontSize: 11, color: 'var(--text-3)' }}>—</span>;
  const pct = Math.min(psi / 0.3 * 100, 100);
  const color = psi < 0.10 ? '#2ECC71' : psi < 0.20 ? '#F39C12' : psi < 0.25 ? '#E67E22' : '#E74C3C';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ flex: 1, height: 6, background: 'var(--bg-4)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: 3, transition: 'width 0.3s' }} />
      </div>
      <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color, width: 40, textAlign: 'right' }}>
        {psi.toFixed(3)}
      </span>
    </div>
  );
}

// ── Main modal ────────────────────────────────────────────────────────────────

export default function DriftModal({ referenceDatasetId, onClose }) {
  const [currentId, setCurrentId] = useState('');
  const [targetCol, setTargetCol] = useState('');
  const [expandedFeature, setExpandedFeature] = useState(null);

  const { data: dsData } = useQuery({
    queryKey: ['datasets-list'],
    queryFn: () => datasets.list({ page_size: 100 }).then(r => r.data),
  });

  const driftMut = useMutation({
    mutationFn: () => drift.check(referenceDatasetId, parseInt(currentId), targetCol || null),
  });

  const report = driftMut.data;
  const availableDatasets = (dsData || []).filter(
    d => d.id !== referenceDatasetId && d.status === 'ready'
  );

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 1000,
      background: 'rgba(0,0,0,0.7)', display: 'flex',
      alignItems: 'flex-start', justifyContent: 'center',
      padding: '60px 24px', overflowY: 'auto',
    }} onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div style={{
        background: 'var(--bg-2)', border: '1px solid var(--border)',
        borderRadius: 'var(--radius)', width: '100%', maxWidth: 760,
        padding: 32, position: 'relative',
      }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 24 }}>
          <div>
            <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: 'var(--text-1)', display: 'flex', alignItems: 'center', gap: 10 }}>
              <Activity size={20} style={{ color: 'var(--accent)' }} /> Drift Analysis
            </h2>
            <p style={{ margin: '4px 0 0', fontSize: 13, color: 'var(--text-3)' }}>
              Compare this dataset against a newer version to detect distributional shifts.
            </p>
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--text-3)', cursor: 'pointer', padding: 4 }}>
            <X size={18} />
          </button>
        </div>

        {/* Controls */}
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 12, marginBottom: 16 }}>
          <div>
            <SectionLabel>Compare against dataset</SectionLabel>
            <select value={currentId} onChange={e => setCurrentId(e.target.value)}
              style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 'var(--radius-sm)', padding: '8px 12px', fontSize: 13 }}>
              <option value="">Select current dataset…</option>
              {availableDatasets.map(d => (
                <option key={d.id} value={d.id}>{d.name}</option>
              ))}
            </select>
          </div>
          <div>
            <SectionLabel>Target column (optional)</SectionLabel>
            <input value={targetCol} onChange={e => setTargetCol(e.target.value)}
              placeholder="e.g. churn" style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 'var(--radius-sm)', padding: '8px 12px', fontSize: 13, boxSizing: 'border-box' }} />
          </div>
        </div>

        <Button variant="primary" disabled={!currentId} loading={driftMut.isPending}
          onClick={() => driftMut.mutate()}>
          <Activity size={14} style={{ marginRight: 6 }} /> Run drift analysis
        </Button>

        {/* Results */}
        {report && (
          <div style={{ marginTop: 24 }}>
            {/* Summary row */}
            <div style={{
              display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)',
              gap: 12, marginBottom: 24,
              padding: 16, background: 'var(--bg-3)', borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border)',
            }}>
              {[
                { label: 'Overall',         value: <SeverityBadge severity={report.severity} size="lg" /> },
                { label: 'Features checked', value: report.n_features_checked },
                { label: 'Features drifted', value: report.n_features_drifted },
                { label: 'Mean PSI',         value: report.overall_psi?.toFixed(3) },
              ].map(({ label, value }) => (
                <div key={label}>
                  <div style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>{label}</div>
                  <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-1)' }}>{value}</div>
                </div>
              ))}
            </div>

            {/* PSI legend */}
            <div style={{ display: 'flex', gap: 16, marginBottom: 16 }}>
              {Object.entries(report.thresholds || {}).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: 'var(--text-3)' }}>
                  <SeverityBadge severity={k} />
                  <span>{'<'} {v}</span>
                </div>
              ))}
            </div>

            {/* Feature table */}
            <div style={{ border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
              <div style={{
                display: 'grid', gridTemplateColumns: '180px 100px 1fr 80px',
                padding: '8px 14px', background: 'var(--bg-3)',
                fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-3)',
                textTransform: 'uppercase', letterSpacing: '0.06em',
                borderBottom: '1px solid var(--border)',
              }}>
                <span>Feature</span><span>Type</span><span>PSI / strength</span><span>Severity</span>
              </div>
              {report.features.map((f, i) => (
                <div key={f.feature}>
                  <div onClick={() => setExpandedFeature(expandedFeature === f.feature ? null : f.feature)}
                    style={{
                      display: 'grid', gridTemplateColumns: '180px 100px 1fr 80px',
                      padding: '10px 14px', alignItems: 'center', cursor: 'pointer',
                      background: i % 2 === 0 ? 'transparent' : 'var(--bg-3)',
                      borderBottom: '1px solid var(--border)',
                    }}
                    onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-3)'}
                    onMouseLeave={e => e.currentTarget.style.background = i % 2 === 0 ? 'transparent' : 'var(--bg-3)'}
                  >
                    <span style={{ fontSize: 13, fontFamily: 'var(--font-mono)', color: 'var(--text-1)', fontWeight: f.has_drift ? 600 : 400 }}>
                      {f.feature}
                    </span>
                    <span style={{ fontSize: 11, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>{f.dtype}</span>
                    <PSIBar psi={f.psi} />
                    <SeverityBadge severity={f.severity} />
                  </div>

                  {/* Expanded detail */}
                  {expandedFeature === f.feature && (
                    <div style={{ padding: '12px 14px', background: 'var(--bg-2)', borderBottom: '1px solid var(--border)' }}>
                      {f.dtype === 'numeric' ? (
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, fontSize: 12 }}>
                          {[
                            ['Ref mean', f.reference_mean?.toFixed(2)],
                            ['Cur mean', f.current_mean?.toFixed(2)],
                            ['KS stat', f.ks_stat?.toFixed(3)],
                            ['KS p-value', f.ks_pvalue?.toFixed(3)],
                          ].map(([label, value]) => (
                            <div key={label}>
                              <div style={{ color: 'var(--text-3)', marginBottom: 2 }}>{label}</div>
                              <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-1)', fontWeight: 600 }}>{value ?? '—'}</div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div>
                          <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 8 }}>Top shifted categories</div>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                            {f.top_shifted_categories.map(c => (
                              <div key={c.category} style={{ display: 'flex', gap: 12, fontSize: 12 }}>
                                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-1)', width: 100 }}>{c.category}</span>
                                <span style={{ color: 'var(--text-3)' }}>ref: {c.reference_pct}%</span>
                                <span style={{ color: 'var(--text-2)' }}>cur: {c.current_pct}%</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {driftMut.isError && (
          <div style={{ marginTop: 16, padding: 14, background: 'rgba(231,76,60,0.1)', borderRadius: 8, color: '#E74C3C', fontSize: 13 }}>
            {driftMut.error?.response?.data?.detail || 'Drift analysis failed'}
          </div>
        )}
      </div>
    </div>
  );
}
