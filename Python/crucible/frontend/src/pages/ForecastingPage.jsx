import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { TrendingUp, Plus, Trash2, BarChart3, AlertCircle } from 'lucide-react';
import { forecasting, datasets } from '../api/client.js';
import {
  PageHeader, Card, Button, Spinner, SectionLabel, StatusBadge, EmptyState,
} from '../components/ui.jsx';

// ── Forecast line chart (pure SVG) ───────────────────────────────────────────

function ForecastChart({ historical, forecast }) {
  if (!forecast?.length) return null;

  const W = 560, H = 200, PL = 48, PR = 16, PT = 16, PB = 32;
  const iW = W - PL - PR, iH = H - PT - PB;

  const histPoints = (historical || []).slice(-24); // last 24 history points
  const allValues = [
    ...histPoints.map(p => p.value),
    ...forecast.map(p => p.predicted),
    ...forecast.map(p => p.upper),
  ].filter(v => isFinite(v));

  if (!allValues.length) return null;

  const minV = Math.min(...allValues) * 0.95;
  const maxV = Math.max(...allValues) * 1.05;
  const range = maxV - minV || 1;

  const totalPoints = histPoints.length + forecast.length;
  const toX = i => PL + (i / Math.max(totalPoints - 1, 1)) * iW;
  const toY = v => PT + (1 - (v - minV) / range) * iH;

  const histPath = histPoints.map((p, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(p.value).toFixed(1)}`).join(' ');
  const fcStart = histPoints.length;
  const fcPath  = forecast.map((p, i) => `${i === 0 ? 'M' : 'L'}${toX(fcStart + i).toFixed(1)},${toY(p.predicted).toFixed(1)}`).join(' ');

  // Confidence interval area
  const upperPath = forecast.map((p, i) => `${i === 0 ? 'M' : 'L'}${toX(fcStart + i).toFixed(1)},${toY(p.upper).toFixed(1)}`).join(' ');
  const lowerPath = forecast.slice().reverse().map((p, i) => `L${toX(fcStart + forecast.length - 1 - i).toFixed(1)},${toY(p.lower).toFixed(1)}`).join(' ');

  // Y axis ticks
  const ticks = Array.from({length: 4}, (_, i) => minV + (range * i) / 3);

  return (
    <svg width={W} height={H} style={{ overflow: 'visible', width: '100%' }}>
      {/* Grid lines */}
      {ticks.map((v, i) => (
        <g key={i}>
          <line x1={PL} x2={W - PR} y1={toY(v)} y2={toY(v)} stroke="var(--border)" strokeWidth={0.5} />
          <text x={PL - 6} y={toY(v) + 4} textAnchor="end" fontSize={9} fill="var(--text-3)">
            {v.toFixed(0)}
          </text>
        </g>
      ))}

      {/* Confidence band */}
      {forecast.length > 0 && (
        <path d={`${upperPath} ${lowerPath} Z`} fill="var(--accent)" fillOpacity={0.1} />
      )}

      {/* Historical line */}
      {histPoints.length > 0 && (
        <path d={histPath} fill="none" stroke="var(--text-3)" strokeWidth={1.5} />
      )}

      {/* Forecast line */}
      {forecast.length > 0 && (
        <>
          {/* Vertical separator */}
          <line x1={toX(fcStart)} x2={toX(fcStart)} y1={PT} y2={H - PB}
            stroke="var(--border)" strokeDasharray="4 3" strokeWidth={1} />
          <path d={fcPath} fill="none" stroke="var(--accent)" strokeWidth={2} strokeDasharray="6 3" />
          <circle cx={toX(totalPoints - 1)} cy={toY(forecast[forecast.length - 1].predicted)}
            r={4} fill="var(--accent)" />
        </>
      )}

      {/* X-axis labels — first and last forecast date */}
      {forecast.length > 0 && (
        <>
          <text x={toX(fcStart)} y={H - 4} textAnchor="middle" fontSize={9} fill="var(--text-3)">
            {forecast[0]?.date}
          </text>
          <text x={toX(totalPoints - 1)} y={H - 4} textAnchor="middle" fontSize={9} fill="var(--text-3)">
            {forecast[forecast.length - 1]?.date}
          </text>
        </>
      )}
    </svg>
  );
}

// ── Metric badge ──────────────────────────────────────────────────────────────

function MetricBadge({ label, value, unit = '' }) {
  return (
    <div style={{ background: 'var(--bg-3)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '12px 16px', minWidth: 100 }}>
      <div style={{ fontSize: 10, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4, fontFamily: 'var(--font-mono)' }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>
        {value != null ? `${value.toFixed(2)}${unit}` : '—'}
      </div>
    </div>
  );
}

// ── Submit form ───────────────────────────────────────────────────────────────

function SubmitForm({ onSubmit, loading }) {
  const [datasetId, setDatasetId] = useState('');
  const [dateCol, setDateCol] = useState('date');
  const [targetCol, setTargetCol] = useState('');
  const [horizon, setHorizon] = useState(12);
  const [freq, setFreq] = useState('auto');
  const [nTrials, setNTrials] = useState(20);

  const { data: dsList } = useQuery({
    queryKey: ['datasets-list-fc'],
    queryFn: () => datasets.list({ page_size: 100 }).then(r => r.data?.filter?.(d => d.status === 'ready') ?? []),
  });

  const { data: families } = useQuery({
    queryKey: ['forecasting-families'],
    queryFn: () => forecasting.families(),
  });

  const inp = (val, set, type = 'text', rest = {}) => (
    <input type={type} value={val} onChange={e => set(type === 'number' ? Number(e.target.value) : e.target.value)} {...rest}
      style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 'var(--radius-sm)', padding: '7px 10px', fontSize: 13, boxSizing: 'border-box' }} />
  );

  function handleSubmit() {
    if (!datasetId || !dateCol || !targetCol) return;
    onSubmit({ dataset_id: parseInt(datasetId), date_column: dateCol, target_column: targetCol, horizon, frequency: freq, n_trials: nTrials });
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div>
        <SectionLabel>Dataset</SectionLabel>
        <select value={datasetId} onChange={e => setDatasetId(e.target.value)}
          style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 'var(--radius-sm)', padding: '7px 10px', fontSize: 13 }}>
          <option value="">Select a dataset…</option>
          {(dsList || []).map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
        </select>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <div>
          <SectionLabel>Date column</SectionLabel>
          {inp(dateCol, setDateCol, 'text', { placeholder: 'date' })}
        </div>
        <div>
          <SectionLabel>Target column</SectionLabel>
          {inp(targetCol, setTargetCol, 'text', { placeholder: 'sales, revenue, …' })}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
        <div>
          <SectionLabel>Horizon</SectionLabel>
          {inp(horizon, setHorizon, 'number', { min: 1, max: 365 })}
          <p style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 3 }}>Steps to forecast</p>
        </div>
        <div>
          <SectionLabel>Frequency</SectionLabel>
          <select value={freq} onChange={e => setFreq(e.target.value)}
            style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 'var(--radius-sm)', padding: '7px 10px', fontSize: 13 }}>
            <option value="auto">Auto-detect</option>
            <option value="D">Daily (D)</option>
            <option value="W">Weekly (W)</option>
            <option value="MS">Monthly (MS)</option>
            <option value="QS">Quarterly (QS)</option>
            <option value="YS">Yearly (YS)</option>
          </select>
        </div>
        <div>
          <SectionLabel>Optuna trials</SectionLabel>
          {inp(nTrials, setNTrials, 'number', { min: 1, max: 100 })}
        </div>
      </div>

      {families && (
        <div style={{ padding: '10px 14px', background: 'var(--bg-3)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)', fontSize: 12, color: 'var(--text-3)' }}>
          Available families: {Object.keys(families).join(' · ')}
        </div>
      )}

      <Button variant="primary" disabled={!datasetId || !dateCol || !targetCol} loading={loading} onClick={handleSubmit}>
        <TrendingUp size={14} style={{ marginRight: 6 }} /> Start forecast
      </Button>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

const TABS = ['Jobs', 'New Forecast'];

export default function ForecastingPage() {
  const qc = useQueryClient();
  const [tab, setTab] = useState('Jobs');
  const [selectedJobId, setSelectedJobId] = useState(null);

  const { data: jobsData, isLoading } = useQuery({
    queryKey: ['forecast-jobs'],
    queryFn: () => forecasting.list(),
    refetchInterval: 3000,
  });

  const jobs = jobsData?.data ?? [];

  const submitMut = useMutation({
    mutationFn: (payload) => forecasting.submit(payload),
    onSuccess: (job) => {
      qc.invalidateQueries({ queryKey: ['forecast-jobs'] });
      setSelectedJobId(job.job_id);
      setTab('Jobs');
    },
  });

  const deleteMut = useMutation({
    mutationFn: (jobId) => forecasting.delete(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['forecast-jobs'] });
      setSelectedJobId(null);
    },
  });

  const { data: selectedJob } = useQuery({
    queryKey: ['forecast-job', selectedJobId],
    queryFn: () => forecasting.get(selectedJobId),
    enabled: !!selectedJobId,
    refetchInterval: (data) => data?.status === 'running' ? 2000 : false,
  });

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 32 }}>
      <PageHeader
        title="Forecasting"
        subtitle="AutoARIMA · Exponential Smoothing · (Prophet · LSTM optional)"
      />

      <div style={{ display: 'flex', gap: 2, marginBottom: 24, borderBottom: '1px solid var(--border)' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 20px', background: 'none', border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: 500,
            color: tab === t ? 'var(--accent)' : 'var(--text-3)',
            borderBottom: tab === t ? '2px solid var(--accent)' : '2px solid transparent', marginBottom: -1,
          }}>{t}</button>
        ))}
      </div>

      {/* ── JOBS TAB ── */}
      {tab === 'Jobs' && (
        <div style={{ display: 'grid', gridTemplateColumns: selectedJobId ? '1fr 420px' : '1fr', gap: 20 }}>
          <Card style={{ padding: 0 }}>
            <div style={{
              display: 'grid', gridTemplateColumns: '2fr 80px 80px 80px 80px 44px',
              padding: '8px 16px', borderBottom: '1px solid var(--border)',
              fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-3)',
              textTransform: 'uppercase', letterSpacing: '0.06em',
            }}>
              <span>Target</span><span>Status</span><span>MAPE%</span><span>RMSE</span><span>Best</span><span />
            </div>

            {isLoading && <div style={{ padding: 24, textAlign: 'center' }}><Spinner /></div>}
            {!isLoading && jobs.length === 0 && (
              <EmptyState icon={<TrendingUp size={32} />} title="No forecasts yet"
                description="Use the New Forecast tab to create a forecasting job."
                action={<Button variant="primary" onClick={() => setTab('New Forecast')}><Plus size={14} style={{ marginRight: 6 }} />New forecast</Button>} />
            )}

            {jobs.map((j, i) => (
              <div key={j.job_id} onClick={() => setSelectedJobId(j.job_id === selectedJobId ? null : j.job_id)}
                style={{
                  display: 'grid', gridTemplateColumns: '2fr 80px 80px 80px 80px 44px',
                  padding: '11px 16px', alignItems: 'center', cursor: 'pointer',
                  borderBottom: '1px solid var(--border)',
                  background: j.job_id === selectedJobId ? 'var(--bg-3)' : i % 2 === 0 ? 'transparent' : 'var(--bg-3)',
                }}
                onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-3)'}
                onMouseLeave={e => e.currentTarget.style.background = j.job_id === selectedJobId ? 'var(--bg-3)' : i % 2 === 0 ? 'transparent' : 'var(--bg-3)'}
              >
                <div>
                  <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-1)', fontFamily: 'var(--font-mono)' }}>{j.target_column}</div>
                  <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>horizon {j.horizon} · {j.date_column}</div>
                </div>
                <StatusBadge status={j.status} />
                <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-2)' }}>
                  {j.cv_mape != null ? `${j.cv_mape.toFixed(1)}%` : '—'}
                </span>
                <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-2)' }}>
                  {j.cv_rmse != null ? j.cv_rmse.toFixed(2) : '—'}
                </span>
                <span style={{ fontSize: 11, color: 'var(--accent)', fontFamily: 'var(--font-mono)' }}>
                  {j.best_family_display || '—'}
                </span>
                <button onClick={e => { e.stopPropagation(); if (confirm('Delete?')) deleteMut.mutate(j.job_id); }}
                  style={{ background: 'none', border: 'none', color: 'var(--text-3)', cursor: 'pointer', padding: 6, display: 'flex' }}
                  onMouseEnter={e => e.currentTarget.style.color = 'var(--red)'}
                  onMouseLeave={e => e.currentTarget.style.color = 'var(--text-3)'}>
                  <Trash2 size={13} />
                </button>
              </div>
            ))}
          </Card>

          {selectedJobId && selectedJob && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {/* Status + metrics */}
              <Card>
                <div style={{ display: 'flex', gap: 4, alignItems: 'center', marginBottom: 14 }}>
                  <StatusBadge status={selectedJob.status} />
                  {selectedJob.status === 'running' && <Spinner size={12} style={{ marginLeft: 8 }} />}
                </div>
                {selectedJob.status === 'succeeded' && (
                  <>
                    <div style={{ fontSize: 13, color: 'var(--text-2)', marginBottom: 10 }}>
                      Best model: <strong style={{ color: 'var(--accent)' }}>{selectedJob.best_family_display}</strong>
                      {' · '}{selectedJob.n_trials_completed} trials
                      {' · '}{selectedJob.elapsed_secs?.toFixed(0)}s
                    </div>
                    <div style={{ display: 'flex', gap: 10 }}>
                      <MetricBadge label="CV MAPE" value={selectedJob.cv_mape} unit="%" />
                      <MetricBadge label="CV RMSE" value={selectedJob.cv_rmse} />
                      <MetricBadge label="CV MAE" value={selectedJob.cv_mae} />
                    </div>
                  </>
                )}
                {selectedJob.error_message && (
                  <div style={{ marginTop: 10, padding: '8px 12px', background: 'rgba(231,76,60,0.1)', borderRadius: 6, fontSize: 12, color: 'var(--red)', display: 'flex', gap: 8 }}>
                    <AlertCircle size={13} style={{ marginTop: 2, flexShrink: 0 }} />
                    {selectedJob.error_message}
                  </div>
                )}
              </Card>

              {/* Forecast chart */}
              {selectedJob.forecast?.length > 0 && (
                <Card>
                  <SectionLabel>Forecast — next {selectedJob.horizon} periods</SectionLabel>
                  <ForecastChart historical={[]} forecast={selectedJob.forecast} />
                  <div style={{ marginTop: 12, display: 'flex', gap: 16, fontSize: 11, color: 'var(--text-3)' }}>
                    <span style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                      <svg width={20} height={4}><line x1={0} y1={2} x2={20} y2={2} stroke="var(--accent)" strokeWidth={2} strokeDasharray="6 3" /></svg>
                      Forecast
                    </span>
                    <span style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                      <svg width={20} height={10}><rect x={0} y={0} width={20} height={10} fill="var(--accent)" fillOpacity={0.15} /></svg>
                      95% confidence
                    </span>
                  </div>
                </Card>
              )}

              {/* Forecast table */}
              {selectedJob.forecast?.length > 0 && (
                <Card style={{ padding: 0, maxHeight: 260, overflowY: 'auto' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', padding: '8px 14px', borderBottom: '1px solid var(--border)', fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-3)', textTransform: 'uppercase', position: 'sticky', top: 0, background: 'var(--bg-2)' }}>
                    <span>Date</span><span style={{ textAlign: 'right' }}>Predicted</span><span style={{ textAlign: 'right' }}>Lower</span><span style={{ textAlign: 'right' }}>Upper</span>
                  </div>
                  {selectedJob.forecast.map((row, i) => (
                    <div key={i} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', padding: '7px 14px', fontSize: 12, fontFamily: 'var(--font-mono)', background: i % 2 === 0 ? 'transparent' : 'var(--bg-3)', borderBottom: '1px solid var(--border)' }}>
                      <span style={{ color: 'var(--text-2)' }}>{row.date}</span>
                      <span style={{ textAlign: 'right', color: 'var(--accent)', fontWeight: 600 }}>{row.predicted?.toFixed(2)}</span>
                      <span style={{ textAlign: 'right', color: 'var(--text-3)' }}>{row.lower?.toFixed(2)}</span>
                      <span style={{ textAlign: 'right', color: 'var(--text-3)' }}>{row.upper?.toFixed(2)}</span>
                    </div>
                  ))}
                </Card>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── NEW FORECAST TAB ── */}
      {tab === 'New Forecast' && (
        <Card>
          <SubmitForm onSubmit={payload => submitMut.mutate(payload)} loading={submitMut.isPending} />
          {submitMut.isError && (
            <div style={{ marginTop: 14, padding: '10px 14px', background: 'rgba(231,76,60,0.1)', borderRadius: 8, color: 'var(--red)', fontSize: 13 }}>
              {submitMut.error?.response?.data?.detail?.errors?.join('; ') ||
               submitMut.error?.response?.data?.detail || 'Submission failed'}
            </div>
          )}
        </Card>
      )}
    </div>
  );
}
