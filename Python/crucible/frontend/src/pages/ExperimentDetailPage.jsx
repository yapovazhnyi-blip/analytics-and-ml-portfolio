import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation } from '@tanstack/react-query';
import { ArrowLeft, Download, Cpu } from 'lucide-react';
import api, { experiments as experimentsApi } from '../api/client.js';
import {
  PageHeader, StatusBadge, Button,
  Card, Spinner, SectionLabel,
} from '../components/ui.jsx';
import LineageDAG from '../components/LineageDAG.jsx';

const API_BASE = '/api/v1';

async function getExperiment(id) {
  return experimentsApi.get(id);
}

// ── Live progress via WebSocket ────────────────────────────────────────────

function LiveProgress({ jobId }) {
  const [messages, setMessages] = useState([]);
  const [connected, setConnected] = useState(false);
  const [done, setDone] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    if (!jobId || done) return;
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${protocol}://${window.location.host}/ws/experiments/${jobId}/progress`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onmessage = e => {
      const msg = JSON.parse(e.data);
      setMessages(prev => [...prev.slice(-50), msg]);
      if (msg.type === 'complete' || msg.type === 'job_status') {
        setDone(true);
      }
    };
    ws.onclose = () => { setConnected(false); };
    return () => ws.close();
  }, [jobId]);

  if (!jobId) return null;

  const lastTrial = [...messages].reverse().find(m => m.type === 'trial');
  const complete = messages.find(m => m.type === 'complete');

  return (
    <Card style={{ padding: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <SectionLabel>Training progress</SectionLabel>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: 11, color: connected ? 'var(--green)' : 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>
          <div style={{ width: 6, height: 6, borderRadius: '50%', background: connected ? 'var(--green)' : 'var(--text-3)' }} />
          {connected ? 'live' : done ? 'done' : 'connecting…'}
        </div>
      </div>

      {lastTrial && (
        <div style={{ marginBottom: 16 }}>
          <div style={{ display: 'flex', gap: 32, marginBottom: 10 }}>
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 3, fontFamily: 'var(--font-mono)' }}>TRIAL</div>
              <div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text-1)' }}>
                {lastTrial.trial} <span style={{ fontSize: 14, color: 'var(--text-3)' }}>/ {lastTrial.total_trials}</span>
              </div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 3, fontFamily: 'var(--font-mono)' }}>BEST SCORE</div>
              <div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>
                {(lastTrial.best_score * 100).toFixed(2)}%
              </div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 3, fontFamily: 'var(--font-mono)' }}>LEADING</div>
              <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-1)', marginTop: 4, fontFamily: 'var(--font-mono)' }}>
                {lastTrial.best_family?.replace(/_/g, ' ')}
              </div>
            </div>
          </div>

          {/* Progress bar */}
          <div style={{ height: 4, background: 'var(--bg-4)', borderRadius: 2 }}>
            <div style={{
              height: '100%',
              width: `${(lastTrial.trial / lastTrial.total_trials) * 100}%`,
              background: 'var(--accent)',
              borderRadius: 2,
              transition: 'width 0.3s ease',
            }} />
          </div>
        </div>
      )}

      {/* Trial log */}
      <div style={{
        background: 'var(--bg)',
        borderRadius: 'var(--radius)',
        border: '1px solid var(--border)',
        padding: '10px 12px',
        maxHeight: 180,
        overflowY: 'auto',
        fontFamily: 'var(--font-mono)',
        fontSize: 11,
      }}>
        {messages.length === 0 ? (
          <span style={{ color: 'var(--text-3)' }}>Waiting for first trial…</span>
        ) : (
          [...messages].reverse().slice(0, 20).map((m, i) => {
            if (m.type === 'trial') {
              return (
                <div key={i} style={{ color: 'var(--text-2)', padding: '1px 0' }}>
                  <span style={{ color: 'var(--text-3)' }}>trial {m.trial}</span>
                  {' · '}
                  <span style={{ color: 'var(--text-1)' }}>{m.family}</span>
                  {' · '}
                  <span style={{ color: m.score >= m.best_score ? 'var(--accent)' : 'var(--text-2)' }}>
                    {(m.score * 100).toFixed(3)}%
                  </span>
                </div>
              );
            }
            if (m.type === 'complete') {
              return (
                <div key={i} style={{ color: 'var(--green)', padding: '2px 0', fontWeight: 600 }}>
                  ✓ Complete — {m.best_family} · {(m.best_score * 100).toFixed(3)}% · {m.elapsed_secs?.toFixed(1)}s
                </div>
              );
            }
            if (m.type === 'error') {
              return <div key={i} style={{ color: 'var(--red)' }}>✗ {m.message}</div>;
            }
            return null;
          })
        )}
      </div>
    </Card>
  );
}

// ── Fairness analysis panel ────────────────────────────────────────────────

const SEVERITY_COLOR = {
  acceptable:  'var(--green)',
  marginal:    '#F39C12',
  significant: '#E67E22',
  severe:      'var(--red)',
};

function FairnessPanel({ experimentId }) {
  const [attrs, setAttrs]     = useState('');
  const [posClass, setPosClass] = useState(1);
  const [report, setReport]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState('');

  async function runFairness() {
    const list = attrs.split(',').map(s => s.trim()).filter(Boolean);
    if (!list.length) { setError('Enter at least one column name.'); return; }
    setError('');
    setLoading(true);
    try {
      const resp = await api.post(`/experiments/${experimentId}/fairness`, {
        protected_attributes: list,
        positive_class: posClass,
      });
      setReport(resp.data.data);
    } catch (e) {
      setError(e.response?.data?.detail || 'Fairness analysis failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card>
      <SectionLabel>Fairness Analysis</SectionLabel>
      <p style={{ fontSize: 12, color: 'var(--text-3)', marginBottom: 12, lineHeight: 1.6 }}>
        Computes demographic parity, equal opportunity, equalized odds, and disparate impact
        across groups defined by protected attribute columns in your dataset.
      </p>

      {!report && (
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'flex-end' }}>
          <div style={{ flex: '1 1 200px' }}>
            <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.06em', fontFamily: 'var(--font-mono)' }}>
              Protected attributes (comma-separated)
            </div>
            <input value={attrs} onChange={e => setAttrs(e.target.value)}
              placeholder="gender, age_group, region"
              style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 'var(--radius-sm)', padding: '7px 10px', fontSize: 13, boxSizing: 'border-box' }} />
          </div>
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.06em', fontFamily: 'var(--font-mono)' }}>Positive class</div>
            <input type="number" value={posClass} onChange={e => setPosClass(Number(e.target.value))}
              style={{ width: 70, background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 'var(--radius-sm)', padding: '7px 10px', fontSize: 13 }} />
          </div>
          <Button variant="primary" onClick={runFairness} loading={loading}>
            Run analysis
          </Button>
        </div>
      )}

      {error && <p style={{ color: 'var(--red)', fontSize: 12, marginTop: 8 }}>{error}</p>}

      {report && (
        <div>
          {/* Summary row */}
          <div style={{ display: 'flex', gap: 16, marginBottom: 16, padding: '10px 14px', background: 'var(--bg-3)', borderRadius: 8, border: '1px solid var(--border)' }}>
            <div>
              <div style={{ fontSize: 10, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.06em', fontFamily: 'var(--font-mono)', marginBottom: 3 }}>Overall</div>
              <span style={{ fontWeight: 700, fontSize: 16, color: SEVERITY_COLOR[report.overall_severity] || 'var(--text-1)', textTransform: 'capitalize' }}>{report.overall_severity}</span>
            </div>
            <div>
              <div style={{ fontSize: 10, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.06em', fontFamily: 'var(--font-mono)', marginBottom: 3 }}>Samples</div>
              <span style={{ fontWeight: 700, fontSize: 16, color: 'var(--text-1)' }}>{report.n_samples}</span>
            </div>
            <div>
              <div style={{ fontSize: 10, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.06em', fontFamily: 'var(--font-mono)', marginBottom: 3 }}>Flagged</div>
              <span style={{ fontWeight: 700, fontSize: 16, color: report.n_attributes_flagged > 0 ? 'var(--red)' : 'var(--green)' }}>{report.n_attributes_flagged}/{report.metrics.length}</span>
            </div>
            <button onClick={() => setReport(null)}
              style={{ marginLeft: 'auto', background: 'none', border: 'none', color: 'var(--text-3)', cursor: 'pointer', fontSize: 12, alignSelf: 'center' }}>
              Re-run ↺
            </button>
          </div>

          {/* Per-attribute metrics */}
          {report.metrics.map(m => (
            <div key={m.attribute} style={{ marginBottom: 16, padding: '12px 14px', background: 'var(--bg-3)', borderRadius: 8, border: `1px solid ${SEVERITY_COLOR[m.severity] || 'var(--border)'}40` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
                <span style={{ fontWeight: 600, fontSize: 13, fontFamily: 'var(--font-mono)', color: 'var(--text-1)' }}>{m.attribute}</span>
                <span style={{ fontSize: 11, fontWeight: 700, color: SEVERITY_COLOR[m.severity], textTransform: 'uppercase', letterSpacing: '0.06em' }}>{m.severity}</span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 12 }}>
                {[
                  ['Dem. Parity Δ', m.demographic_parity_diff?.toFixed(3), '< 0.10'],
                  ['Equal Opp. Δ',  m.equal_opportunity_diff?.toFixed(3),  '< 0.10'],
                  ['Eq. Odds Δ',    m.equalized_odds_diff?.toFixed(3),     '< 0.10'],
                  ['Disp. Impact',  m.disparate_impact_ratio?.toFixed(3),  '> 0.80'],
                ].map(([label, val, threshold]) => (
                  <div key={label}>
                    <div style={{ fontSize: 10, color: 'var(--text-3)', marginBottom: 3, textTransform: 'uppercase', letterSpacing: '0.05em', fontFamily: 'var(--font-mono)' }}>{label}</div>
                    <div style={{ fontSize: 15, fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text-1)' }}>{val ?? '—'}</div>
                    <div style={{ fontSize: 10, color: 'var(--text-3)' }}>target {threshold}</div>
                  </div>
                ))}
              </div>

              {/* Group breakdown */}
              <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 6 }}>
                Privileged: <strong style={{ color: 'var(--text-2)' }}>{m.privileged_group}</strong>
                {' · '}Unprivileged: <strong style={{ color: 'var(--text-2)' }}>{m.unprivileged_group}</strong>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 8 }}>
                {m.group_stats.map(gs => (
                  <div key={gs.group_value} style={{ padding: '6px 10px', background: 'var(--bg-2)', borderRadius: 6, border: '1px solid var(--border)' }}>
                    <div style={{ fontSize: 12, fontWeight: 600, fontFamily: 'var(--font-mono)', color: 'var(--text-1)', marginBottom: 4 }}>{gs.group_value}</div>
                    <div style={{ fontSize: 11, color: 'var(--text-3)' }}>n = {gs.n_samples}</div>
                    <div style={{ fontSize: 11, color: 'var(--text-3)' }}>sel. rate {(gs.selection_rate * 100).toFixed(1)}%</div>
                    <div style={{ fontSize: 11, color: 'var(--text-3)' }}>recall {(gs.recall * 100).toFixed(1)}%</div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

// ── SHAP importance chart ──────────────────────────────────────────────────

function SHAPChart({ importance }) {
  if (!importance || importance.length === 0) return null;
  const max = Math.max(...importance.map(f => f.mean_abs_shap));

  return (
    <Card style={{ padding: 20 }}>
      <SectionLabel>Feature importance (SHAP)</SectionLabel>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {importance.slice(0, 15).map((f, i) => (
          <div key={f.feature} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ width: 22, textAlign: 'right', fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', flexShrink: 0 }}>
              {f.rank}
            </div>
            <div style={{ width: 140, fontFamily: 'var(--font-mono)', fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flexShrink: 0 }}>
              {f.feature}
            </div>
            <div style={{ flex: 1, height: 6, background: 'var(--bg-4)', borderRadius: 3 }}>
              <div style={{
                height: '100%',
                width: `${(f.mean_abs_shap / max) * 100}%`,
                background: i === 0 ? 'var(--accent)' : `rgba(0, 194, 168, ${0.9 - i * 0.05})`,
                borderRadius: 3,
                transition: 'width 0.4s ease',
              }} />
            </div>
            <div style={{ width: 70, textAlign: 'right', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-2)', flexShrink: 0 }}>
              {f.mean_abs_shap.toFixed(4)}
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────

export default function ExperimentDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['experiment', Number(id)],
    queryFn: () => getExperiment(Number(id)),
    refetchInterval: (data) =>
      data?.data?.status === 'running' ? 3000 : false,
  });

  if (isLoading) return (
    <div style={{ display: 'flex', justifyContent: 'center', paddingTop: 80 }}>
      <Spinner size={24} />
    </div>
  );

  const exp = data?.data;
  if (!exp) return <div style={{ padding: 28, color: 'var(--red)' }}>Experiment not found.</div>;

  const isRunning = exp.status === 'running';
  const isComplete = exp.status === 'complete';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <PageHeader
        title={exp.name}
        subtitle={
          <span style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <StatusBadge status={exp.status} />
            {exp.lifecycle_stage && exp.lifecycle_stage !== 'candidate' && (
              <LifecycleBadge stage={exp.lifecycle_stage} />
            )}
            <span style={{ fontSize: 12, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>
              {exp.task_type} · target: {exp.target_column}
            </span>
          </span>
        }
        action={
          <div style={{ display: 'flex', gap: 8 }}>
            {isComplete && (
              <>
                <Button
                  variant="primary"
                  onClick={() => {
                    window.open(`/api/v1/experiments/${exp.id}/deploy`, '_blank');
                  }}
                >
                  <Download size={13} style={{ marginRight: 4 }} /> Deploy
                </Button>
                <Button
                  variant="ghost"
                  title="Export to ONNX (3-10× faster CPU inference)"
                  onClick={async () => {
                    try {
                      const resp = await api.post(`/experiments/${exp.id}/export/onnx`);
                      const { onnx_path, model_size_kb, n_features } = resp.data.data;
                      window.open(`/api/v1/experiments/${exp.id}/export/onnx/download`, '_blank');
                    } catch (e) {
                      alert(e.response?.data?.detail || 'ONNX export failed');
                    }
                  }}
                >
                  <Cpu size={13} style={{ marginRight: 4 }} /> ONNX
                </Button>
                <Button
                  variant="ghost"
                  title="Download Model Card (HTML) — EU AI Act documentation"
                  onClick={() => window.open(`/api/v1/experiments/${exp.id}/model-card?format=html`, '_blank')}
                >
                  📋 Model Card
                </Button>
              </>
            )}
            <Button variant="ghost" onClick={() => navigate(-1)}>
              <ArrowLeft size={13} /> Back
            </Button>
          </div>
        }
      />

      <div style={{ flex: 1, overflow: 'auto', padding: '24px 28px', display: 'flex', flexDirection: 'column', gap: 16 }}>

        {/* Live progress */}
        {(isRunning || exp.job_id) && (
          <LiveProgress jobId={exp.job_id} />
        )}

        {/* Results summary */}
        {isComplete && (
          <Card style={{ padding: 20 }}>
            <SectionLabel>Best model</SectionLabel>
            <div style={{ display: 'flex', gap: 40, flexWrap: 'wrap' }}>
              <Stat label="Family" value={exp.best_model_family?.replace(/_/g, ' ')} mono />
              <Stat label={exp.scoring_metric?.toUpperCase() || 'CV Score'} value={exp.best_score != null ? `${(exp.best_score * 100).toFixed(3)}%` : '—'} accent />
              <Stat label="Trials" value={exp.n_trials_completed} mono />
              <Stat label="Pruned" value={exp.n_trials_pruned} mono />
              <Stat label="Duration" value={exp.training_duration_secs ? `${exp.training_duration_secs.toFixed(1)}s` : '—'} mono />
            </div>
            {(exp.calibration_applied || exp.pruner_type) && (
              <div style={{ display: 'flex', gap: 8, marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--border)' }}>
                {exp.calibration_applied && (
                  <InfoChip
                    label={`Calibrated · ${exp.calibration_method}`}
                    title={
                      exp.calibration_method === 'isotonic'
                        ? 'Isotonic regression calibration applied (≥1000 training rows) — predicted probabilities match empirical frequencies.'
                        : 'Sigmoid (Platt scaling) calibration applied — predicted probabilities match empirical frequencies.'
                    }
                  />
                )}
                {exp.pruner_type && (
                  <InfoChip
                    label={`Pruner · ${exp.pruner_type}`}
                    title={
                      exp.pruner_type === 'hyperband'
                        ? 'HyperbandPruner — multi-fidelity successive halving across CV folds, more aggressive early stopping of weak trials.'
                        : 'MedianPruner — prunes trials scoring below the running median at each CV fold.'
                    }
                  />
                )}
              </div>
            )}
          </Card>
        )}

        {/* Holdout metrics */}
        {isComplete && exp.holdout_metrics?.length > 0 && (
          <Card style={{ padding: 20 }}>
            <SectionLabel>Holdout metrics</SectionLabel>
            <div style={{ display: 'flex', gap: 32 }}>
              {exp.holdout_metrics.map(m => (
                <Stat
                  key={m.metric}
                  label={m.metric.replace('holdout_', '').toUpperCase()}
                  value={m.metric.includes('mae') || m.metric.includes('rmse')
                    ? m.value.toFixed(4)
                    : `${(m.value * 100).toFixed(3)}%`}
                  accent={m.metric.includes('roc_auc') || m.metric.includes('r2')}
                />
              ))}
            </div>
          </Card>
        )}

        {/* SHAP importance */}
        {isComplete && exp.feature_importance?.length > 0 && (
          <SHAPChart importance={exp.feature_importance} />
        )}

        {/* Fairness analysis */}
        {isComplete && <FairnessPanel experimentId={exp.id} />}

        {/* Lineage DAG */}
        {isComplete && (
          <div>
            <SectionLabel style={{ marginBottom: 12 }}>Experiment lineage</SectionLabel>
            <LineageDAG experimentId={exp.id} mode="experiment" />
          </div>
        )}

        {/* MLflow link */}
        {exp.mlflow_run_id && (
          <Card style={{ padding: 12 }}>
            <div style={{ fontSize: 12, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>
              MLflow run: <span style={{ color: 'var(--accent)' }}>{exp.mlflow_run_id}</span>
            </div>
          </Card>
        )}

        {/* Error */}
        {exp.status === 'error' && exp.error_message && (
          <Card style={{ padding: 16, border: '1px solid var(--red)', background: 'var(--red-dim)' }}>
            <div style={{ fontSize: 13, color: 'var(--red)' }}>
              <strong>Training failed:</strong> {exp.error_message}
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value, accent, mono }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
        {label}
      </span>
      <span style={{
        fontSize: 18, fontWeight: 700,
        fontFamily: mono || accent ? 'var(--font-mono)' : 'var(--font-sans)',
        color: accent ? 'var(--accent)' : 'var(--text-1)',
      }}>
        {value ?? '—'}
      </span>
    </div>
  );
}

const LIFECYCLE_COLORS = {
  production: { bg: 'rgba(46,204,113,0.15)', fg: 'var(--green)' },
  archived:   { bg: 'rgba(127,140,141,0.15)', fg: 'var(--text-3)' },
  candidate:  { bg: 'rgba(52,152,219,0.15)', fg: 'var(--blue)' },
};

function LifecycleBadge({ stage }) {
  const c = LIFECYCLE_COLORS[stage] || LIFECYCLE_COLORS.candidate;
  return (
    <span style={{
      fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 10,
      background: c.bg, color: c.fg, fontFamily: 'var(--font-mono)', textTransform: 'uppercase',
    }} title="Model registry lifecycle stage — set by the retraining pipeline or manual promotion.">
      {stage}
    </span>
  );
}

function InfoChip({ label, title }) {
  return (
    <span title={title} style={{
      fontSize: 11, padding: '4px 10px', borderRadius: 6,
      background: 'var(--bg-3)', color: 'var(--text-2)',
      fontFamily: 'var(--font-mono)', cursor: 'help',
      border: '1px solid var(--border)',
    }}>
      {label}
    </span>
  );
}
