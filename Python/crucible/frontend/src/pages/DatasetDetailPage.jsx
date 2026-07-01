import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation } from '@tanstack/react-query';
import {
  ArrowLeft, Play, AlertTriangle, CheckCircle2,
  BarChart3, Table2, GitBranch, Info,
} from 'lucide-react';
import { datasets } from '../api/client.js';
import {
  PageHeader, StatusBadge, Button,
  Card, Spinner, SectionLabel, EmptyState,
} from '../components/ui.jsx';
import ExperimentsPage from './ExperimentsPage.jsx';
import LineageDAG from '../components/LineageDAG.jsx';
import DataContractPanel from '../components/DataContractPanel.jsx';

// ── Helpers ────────────────────────────────────────────────────────────────

function Pill({ children, color = 'text-3' }) {
  const cols = {
    'text-3': { bg: 'var(--bg-4)', text: 'var(--text-3)' },
    red:      { bg: 'var(--red-dim)', text: 'var(--red)' },
    amber:    { bg: 'var(--amber-dim)', text: 'var(--amber)' },
    green:    { bg: 'var(--green-dim)', text: 'var(--green)' },
    blue:     { bg: 'var(--blue-dim)', text: 'var(--blue)' },
  };
  const c = cols[color] || cols['text-3'];
  return (
    <span style={{
      display: 'inline-block',
      padding: '2px 7px',
      borderRadius: 20,
      background: c.bg,
      color: c.text,
      fontSize: 11,
      fontWeight: 500,
      fontFamily: 'var(--font-mono)',
    }}>
      {children}
    </span>
  );
}

function Stat({ label, value }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      <span style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
        {label}
      </span>
      <span style={{ fontSize: 20, fontWeight: 600, fontFamily: 'var(--font-mono)', color: 'var(--text-1)' }}>
        {value ?? '—'}
      </span>
    </div>
  );
}

// ── Schema tab ─────────────────────────────────────────────────────────────

function SchemaTab({ ds }) {
  const cols = ds.data?.schema_columns;
  if (!cols || cols.length === 0) {
    return <EmptyState icon={Table2} title="No schema available" description="Schema is inferred after upload completes." />;
  }
  return (
    <Card>
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 120px 70px',
        padding: '8px 16px',
        borderBottom: '1px solid var(--border)',
        fontSize: 11, fontWeight: 600, letterSpacing: '0.06em',
        textTransform: 'uppercase', color: 'var(--text-3)',
        fontFamily: 'var(--font-mono)',
      }}>
        <span>Column</span><span>Type</span><span>Nullable</span>
      </div>
      {cols.map((col, i) => (
        <div key={col.name} style={{
          display: 'grid',
          gridTemplateColumns: '1fr 120px 70px',
          padding: '9px 16px',
          borderBottom: i < cols.length - 1 ? '1px solid var(--border)' : 'none',
          alignItems: 'center',
        }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>{col.name}</span>
          <Pill color="text-3">{col.dtype}</Pill>
          <span style={{ fontSize: 12, color: col.nullable ? 'var(--amber)' : 'var(--text-3)' }}>
            {col.nullable ? 'yes' : 'no'}
          </span>
        </div>
      ))}
    </Card>
  );
}

// ── Profile tab ────────────────────────────────────────────────────────────

function ProfileTab({ datasetId, isReady }) {
  const [targetCol, setTargetCol] = useState('');
  const [timeCol, setTimeCol] = useState('');

  const { data: dsData } = useQuery({
    queryKey: ['dataset', datasetId],
    queryFn: () => datasets.get(datasetId),
  });
  const cols = dsData?.data?.schema_columns?.map(c => c.name) ?? [];

  const profileMut = useMutation({
    mutationFn: () => datasets.profile(datasetId, {
      target_column: targetCol || undefined,
      time_column: timeCol || undefined,
    }),
  });

  const report = profileMut.data?.data;

  const inputStyle = {
    background: 'var(--bg-3)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    color: 'var(--text-1)',
    padding: '6px 10px',
    fontSize: 13,
    width: '100%',
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Config panel */}
      <Card style={{ padding: 20 }}>
        <SectionLabel>Run configuration</SectionLabel>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr auto', gap: 12, alignItems: 'flex-end' }}>
          <div>
            <label style={{ fontSize: 12, color: 'var(--text-2)', display: 'block', marginBottom: 5 }}>
              Target column (optional)
            </label>
            <select
              value={targetCol}
              onChange={e => setTargetCol(e.target.value)}
              style={inputStyle}
            >
              <option value="">— none —</option>
              {cols.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div>
            <label style={{ fontSize: 12, color: 'var(--text-2)', display: 'block', marginBottom: 5 }}>
              Time column (for temporal leakage)
            </label>
            <select
              value={timeCol}
              onChange={e => setTimeCol(e.target.value)}
              style={inputStyle}
            >
              <option value="">— none —</option>
              {cols.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <Button
            onClick={() => profileMut.mutate()}
            loading={profileMut.isPending}
            disabled={!isReady || profileMut.isPending}
          >
            <Play size={13} />
            Run profile
          </Button>
        </div>
        {!isReady && (
          <p style={{ fontSize: 12, color: 'var(--amber)', marginTop: 10 }}>
            Dataset must be in "ready" state to run profiling.
          </p>
        )}
        {profileMut.isError && (
          <p style={{ fontSize: 12, color: 'var(--red)', marginTop: 10 }}>
            {profileMut.error?.response?.data?.detail || 'Profiling failed'}
          </p>
        )}
      </Card>

      {/* Results */}
      {profileMut.isPending && (
        <div style={{ display: 'flex', justifyContent: 'center', padding: 40 }}>
          <Spinner size={24} />
        </div>
      )}

      {report && <ProfileResults report={report} />}
    </div>
  );
}

// ── Profile results ────────────────────────────────────────────────────────

function ProfileResults({ report }) {
  const hasWarnings = report.warnings?.length > 0;
  const hasLeakage = report.leakage_findings?.length > 0;
  const hasMissingness = report.missingness?.length > 0;
  const hasCorrelations = report.high_correlations?.length > 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Summary stats */}
      <Card style={{ padding: 20 }}>
        <SectionLabel>Summary</SectionLabel>
        <div style={{ display: 'flex', gap: 40 }}>
          <Stat label="Rows" value={report.n_rows?.toLocaleString()} />
          <Stat label="Columns" value={report.n_columns} />
          <Stat label="Leakage" value={hasLeakage ? report.leakage_findings.length : '0'} />
          <Stat label="Missing cols" value={report.missingness?.length ?? '0'} />
          <Stat label="Duration" value={`${report.duration_secs}s`} />
        </div>
      </Card>

      {/* Target analysis */}
      {report.target_analysis && (
        <Card style={{ padding: 20 }}>
          <SectionLabel>Target — {report.target_analysis.column}</SectionLabel>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 12 }}>
            <Pill color="blue">{report.target_analysis.task_type}</Pill>
            <Pill color="text-3">{report.target_analysis.n_unique} unique values</Pill>
            {report.target_analysis.is_imbalanced && <Pill color="amber">imbalanced</Pill>}
            {report.target_analysis.is_skewed && <Pill color="amber">skewed ({report.target_analysis.skewness})</Pill>}
          </div>
          {report.target_analysis.imbalance_warning && (
            <Warning text={report.target_analysis.imbalance_warning} />
          )}
          {report.target_analysis.skewness_warning && (
            <Warning text={report.target_analysis.skewness_warning} />
          )}
          {report.target_analysis.class_distribution?.length > 0 && (
            <ClassBalance distribution={report.target_analysis.class_distribution} />
          )}
        </Card>
      )}

      {/* Leakage findings */}
      {hasLeakage && (
        <Card style={{ padding: 20 }}>
          <SectionLabel>Leakage findings</SectionLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {report.leakage_findings.map((f, i) => (
              <div key={i} style={{
                padding: '12px 14px',
                background: 'var(--red-dim)',
                border: '1px solid var(--red)',
                borderRadius: 'var(--radius)',
              }}>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 5 }}>
                  <AlertTriangle size={13} color="var(--red)" />
                  <span style={{ fontWeight: 600, fontSize: 13, color: 'var(--red)' }}>
                    {f.leakage_type} leakage
                  </span>
                  <Pill color="red">{f.severity}</Pill>
                  {f.column && <span style={{ fontSize: 12, color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>{f.column}</span>}
                </div>
                <p style={{ fontSize: 12, color: 'var(--text-2)', lineHeight: 1.6 }}>{f.rationale}</p>
                <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>
                  {f.metric_name}: {f.metric_value.toFixed(3)} (threshold: {f.threshold})
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Missingness */}
      {hasMissingness && (
        <Card style={{ padding: 20 }}>
          <SectionLabel>Missingness</SectionLabel>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 90px 80px 120px', gap: 1 }}>
            {['Column', 'Missing %', 'Count', 'Type'].map(h => (
              <div key={h} style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.06em', padding: '4px 0', marginBottom: 6 }}>{h}</div>
            ))}
            {report.missingness.map((m, i) => (
              <>
                <span key={`col-${i}`} style={{ fontFamily: 'var(--font-mono)', fontSize: 13, padding: '6px 0', borderTop: '1px solid var(--border)' }}>{m.column}</span>
                <span key={`rate-${i}`} style={{ fontSize: 13, color: m.missing_rate >= 0.5 ? 'var(--red)' : m.missing_rate >= 0.1 ? 'var(--amber)' : 'var(--text-2)', padding: '6px 0', borderTop: '1px solid var(--border)', fontFamily: 'var(--font-mono)' }}>
                  {(m.missing_rate * 100).toFixed(1)}%
                </span>
                <span key={`count-${i}`} style={{ fontSize: 13, color: 'var(--text-2)', fontFamily: 'var(--font-mono)', padding: '6px 0', borderTop: '1px solid var(--border)' }}>{m.missing_count.toLocaleString()}</span>
                <div key={`type-${i}`} style={{ padding: '6px 0', borderTop: '1px solid var(--border)' }}>
                  <Pill color={m.likely_systematic ? 'amber' : 'text-3'}>
                    {m.likely_systematic ? `systematic ↔ ${m.correlated_with}` : 'random'}
                  </Pill>
                </div>
              </>
            ))}
          </div>
        </Card>
      )}

      {/* High correlations */}
      {hasCorrelations && (
        <Card style={{ padding: 20 }}>
          <SectionLabel>High pairwise correlations (|r| ≥ 0.90)</SectionLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {report.high_correlations.map((p, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>{p.col_a}</span>
                <span style={{ color: 'var(--text-3)', fontSize: 12 }}>↔</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>{p.col_b}</span>
                <CorrelationBar value={p.correlation} />
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-2)' }}>
                  r = {p.correlation}
                </span>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Column stats */}
      {report.column_stats?.length > 0 && (
        <Card style={{ padding: 20 }}>
          <SectionLabel>Column statistics</SectionLabel>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, fontFamily: 'var(--font-mono)' }}>
              <thead>
                <tr>
                  {['Column', 'Type', 'Unique', 'Null%', 'Min', 'Median', 'Max', 'Skew'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '6px 12px', fontSize: 10, fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase', color: 'var(--text-3)', borderBottom: '1px solid var(--border)', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {report.column_stats.map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ padding: '7px 12px', color: 'var(--text-1)' }}>{s.column}</td>
                    <td style={{ padding: '7px 12px', color: 'var(--text-3)' }}>{s.dtype}</td>
                    <td style={{ padding: '7px 12px', color: 'var(--text-2)' }}>{s.n_unique.toLocaleString()}</td>
                    <td style={{ padding: '7px 12px', color: s.null_rate > 0.1 ? 'var(--amber)' : 'var(--text-2)' }}>
                      {(s.null_rate * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: '7px 12px', color: 'var(--text-2)' }}>{fmt4(s.min)}</td>
                    <td style={{ padding: '7px 12px', color: 'var(--text-2)' }}>{fmt4(s.median)}</td>
                    <td style={{ padding: '7px 12px', color: 'var(--text-2)' }}>{fmt4(s.max)}</td>
                    <td style={{ padding: '7px 12px', color: s.skewness != null && Math.abs(s.skewness) > 1 ? 'var(--amber)' : 'var(--text-2)' }}>
                      {fmt4(s.skewness)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Warnings */}
      {hasWarnings && (
        <Card style={{ padding: 16 }}>
          {report.warnings.map((w, i) => (
            <div key={i} style={{ display: 'flex', gap: 8, fontSize: 12, color: 'var(--text-2)', padding: '4px 0' }}>
              <Info size={13} color="var(--text-3)" style={{ marginTop: 2, flexShrink: 0 }} />
              {w}
            </div>
          ))}
        </Card>
      )}
    </div>
  );
}

function Warning({ text }) {
  return (
    <div style={{
      display: 'flex', gap: 8, padding: '8px 10px',
      background: 'var(--amber-dim)', border: '1px solid var(--amber)',
      borderRadius: 'var(--radius)', marginBottom: 8,
    }}>
      <AlertTriangle size={13} color="var(--amber)" style={{ marginTop: 2, flexShrink: 0 }} />
      <span style={{ fontSize: 12, color: 'var(--text-2)', lineHeight: 1.6 }}>{text}</span>
    </div>
  );
}

function ClassBalance({ distribution }) {
  return (
    <div style={{ display: 'flex', gap: 12, marginTop: 8, flexWrap: 'wrap' }}>
      {distribution.map(c => (
        <div key={c.label} style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          <span style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>{c.label}</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 14, fontWeight: 600 }}>
            {(c.proportion * 100).toFixed(1)}%
          </span>
          <div style={{ height: 3, width: 60, background: 'var(--bg-4)', borderRadius: 2 }}>
            <div style={{ height: '100%', width: `${c.proportion * 100}%`, background: 'var(--accent)', borderRadius: 2 }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function CorrelationBar({ value }) {
  return (
    <div style={{ flex: 1, maxWidth: 120, height: 4, background: 'var(--bg-4)', borderRadius: 2 }}>
      <div style={{
        height: '100%',
        width: `${value * 100}%`,
        background: value >= 0.99 ? 'var(--red)' : 'var(--amber)',
        borderRadius: 2,
      }} />
    </div>
  );
}

function fmt4(v) {
  if (v == null) return '—';
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 1) return v.toFixed(2);
  return v.toFixed(4);
}

// ── Main page ──────────────────────────────────────────────────────────────

const TABS = ['Schema', 'Profile', 'Experiments', 'Lineage', 'Contract', 'Advisor'];

export default function DatasetDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [tab, setTab] = useState('Schema');

  const { data, isLoading, error } = useQuery({
    queryKey: ['dataset', Number(id)],
    queryFn: () => datasets.get(Number(id)),
  });

  if (isLoading) return (
    <div style={{ display: 'flex', justifyContent: 'center', paddingTop: 80 }}>
      <Spinner size={24} />
    </div>
  );
  if (error || !data?.data) return (
    <div style={{ padding: 28, color: 'var(--red)' }}>
      Failed to load dataset.
    </div>
  );

  const ds = data;
  const info = ds.data;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <PageHeader
        title={info.name}
        subtitle={
          <span style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <StatusBadge status={info.status} />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-3)' }}>
              {info.source_type} · {info.row_count?.toLocaleString() ?? '?'} rows · {info.column_count ?? '?'} cols
            </span>
          </span>
        }
        action={
          <Button variant="ghost" onClick={() => navigate('/datasets')}>
            <ArrowLeft size={13} />
            Back
          </Button>
        }
      />

      {/* Tabs */}
      <div style={{
        display: 'flex',
        gap: 0,
        padding: '0 28px',
        borderBottom: '1px solid var(--border)',
        flexShrink: 0,
      }}>
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              background: 'none',
              border: 'none',
              padding: '10px 16px',
              fontSize: 13,
              fontWeight: tab === t ? 600 : 400,
              color: tab === t ? 'var(--text-1)' : 'var(--text-3)',
              borderBottom: tab === t ? '2px solid var(--accent)' : '2px solid transparent',
              cursor: 'pointer',
              transition: 'all 0.12s',
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: 'auto', padding: '24px 28px' }}>
        {tab === 'Schema' && <SchemaTab ds={ds} />}
        {tab === 'Profile' && <ProfileTab datasetId={Number(id)} isReady={info.status === 'ready'} />}
        {tab === 'Experiments' && <ExperimentsPage datasetId={Number(id)} />}
        {tab === 'Lineage' && <LineageDAG datasetId={Number(id)} mode="dataset" />}
        {tab === 'Contract' && (
          <DataContractPanel
            datasetId={Number(id)}
            allDatasets={[]}
          />
        )}
        {tab === 'Advisor' && (
          <AdvisorPanel
            datasetId={Number(id)}
            columns={info.schema_columns?.map(c => c.name) ?? []}
          />
        )}
      </div>
    </div>
  );
}
