import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Sparkles, AlertTriangle, Info, ChevronDown, ChevronRight } from 'lucide-react';
import { Button, Spinner, Card, SectionLabel, EmptyState } from './ui.jsx';
import { datasets } from '../api/client.js';

async function runAdvisor(datasetId, targetColumn, timeColumn) {
  return datasets.advise(datasetId, {
    target_column: targetColumn || undefined,
    time_column: timeColumn || undefined,
  });
}

const SEVERITY_STYLES = {
  high:   { bg: 'var(--red-dim)',   border: 'var(--red)',   text: 'var(--red)',   icon: AlertTriangle },
  medium: { bg: 'var(--amber-dim)', border: 'var(--amber)', text: 'var(--amber)', icon: AlertTriangle },
  low:    { bg: 'var(--blue-dim)',  border: 'var(--blue)',  text: 'var(--blue)',  icon: Info },
  info:   { bg: 'var(--bg-4)',      border: 'var(--border-2)', text: 'var(--text-2)', icon: Info },
};

const CATEGORY_LABELS = {
  leakage:     'Leakage',
  missingness: 'Missingness',
  imbalance:   'Imbalance',
  correlation: 'Correlation',
  general:     'General',
};

function SuggestionCard({ suggestion }) {
  const [expanded, setExpanded] = useState(false);
  const s = SEVERITY_STYLES[suggestion.severity] || SEVERITY_STYLES.info;
  const Icon = s.icon;

  return (
    <div style={{
      background: s.bg,
      border: `1px solid ${s.border}`,
      borderRadius: 'var(--radius)',
      overflow: 'hidden',
    }}>
      <button
        onClick={() => setExpanded(e => !e)}
        style={{
          width: '100%', background: 'none', border: 'none',
          padding: '11px 14px', cursor: 'pointer',
          display: 'flex', alignItems: 'center', gap: 10, textAlign: 'left',
        }}
      >
        <Icon size={13} color={s.text} style={{ flexShrink: 0 }} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-1)' }}>
              {suggestion.title}
            </span>
            <span style={{
              fontSize: 10, fontWeight: 600, padding: '1px 6px',
              borderRadius: 10, background: s.border + '30',
              color: s.text, fontFamily: 'var(--font-mono)',
              textTransform: 'uppercase', letterSpacing: '0.05em',
            }}>
              {CATEGORY_LABELS[suggestion.category] || suggestion.category}
            </span>
            {suggestion.column && (
              <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>
                {suggestion.column}
              </span>
            )}
          </div>
        </div>
        {expanded
          ? <ChevronDown size={13} color="var(--text-3)" />
          : <ChevronRight size={13} color="var(--text-3)" />
        }
      </button>

      {expanded && (
        <div style={{ padding: '0 14px 14px', borderTop: `1px solid ${s.border}40` }}>
          <p style={{ fontSize: 13, color: 'var(--text-2)', lineHeight: 1.7, margin: '10px 0 8px' }}>
            {suggestion.explanation}
          </p>
          <div style={{
            padding: '8px 10px',
            background: 'rgba(0,0,0,0.2)',
            borderRadius: 'var(--radius-sm)',
            borderLeft: `3px solid ${s.border}`,
          }}>
            <div style={{ fontSize: 10, fontWeight: 600, color: s.text, marginBottom: 4, fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              Recommended action
            </div>
            <p style={{ fontSize: 12, color: 'var(--text-1)', lineHeight: 1.6, margin: 0 }}>
              {suggestion.action}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

export default function AdvisorPanel({ datasetId, columns = [] }) {
  const [targetCol, setTargetCol] = useState('');
  const [timeCol, setTimeCol] = useState('');
  const [triggered, setTriggered] = useState(false);
  const [runKey, setRunKey] = useState(0);

  const { data, isLoading, error, isFetching } = useQuery({
    queryKey: ['advisor', datasetId, targetCol, timeCol, runKey],
    queryFn: () => runAdvisor(datasetId, targetCol, timeCol),
    enabled: triggered,
    retry: false,
  });

  const result = data?.data;

  const inputStyle = {
    background: 'var(--bg-3)', border: '1px solid var(--border)',
    borderRadius: 'var(--radius)', color: 'var(--text-1)',
    padding: '6px 10px', fontSize: 13, width: '100%',
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Config */}
      <Card style={{ padding: 20 }}>
        <SectionLabel>Claude advisor</SectionLabel>
        <p style={{ fontSize: 13, color: 'var(--text-2)', marginBottom: 14, lineHeight: 1.6 }}>
          Runs the full profiling suite and asks Claude to analyse the findings,
          then surfaces specific, actionable recommendations.
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr auto', gap: 10, alignItems: 'flex-end' }}>
          <div>
            <label style={{ fontSize: 12, color: 'var(--text-2)', display: 'block', marginBottom: 4 }}>
              Target column (optional)
            </label>
            <select value={targetCol} onChange={e => setTargetCol(e.target.value)} style={inputStyle}>
              <option value="">— none —</option>
              {columns.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div>
            <label style={{ fontSize: 12, color: 'var(--text-2)', display: 'block', marginBottom: 4 }}>
              Time column (for temporal leakage)
            </label>
            <select value={timeCol} onChange={e => setTimeCol(e.target.value)} style={inputStyle}>
              <option value="">— none —</option>
              {columns.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <Button
            onClick={() => { setTriggered(true); setRunKey(k => k + 1); }}
            loading={isFetching}
            disabled={isFetching}
          >
            <Sparkles size={13} />
            Ask Claude
          </Button>
        </div>
        {!import.meta.env.VITE_ANTHROPIC_KEY_SET && result?.error?.includes('not configured') && (
          <p style={{ fontSize: 12, color: 'var(--amber)', marginTop: 10 }}>
            Set ANTHROPIC_API_KEY in your .env to enable the advisor.
          </p>
        )}
      </Card>

      {/* Loading */}
      {isFetching && (
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', padding: '16px 0', color: 'var(--text-3)', fontSize: 13 }}>
          <Spinner size={16} />
          Profiling dataset and asking Claude…
        </div>
      )}

      {/* Error */}
      {!isFetching && error && (
        <Card style={{ padding: 16, border: '1px solid var(--red)', background: 'var(--red-dim)' }}>
          <p style={{ fontSize: 13, color: 'var(--red)' }}>{error.message}</p>
        </Card>
      )}

      {/* API not configured */}
      {!isFetching && result?.error && (
        <Card style={{ padding: 16, border: '1px solid var(--amber)', background: 'var(--amber-dim)' }}>
          <p style={{ fontSize: 13, color: 'var(--amber)' }}>{result.error}</p>
        </Card>
      )}

      {/* Suggestions */}
      {!isFetching && result?.suggestions?.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <SectionLabel>{result.suggestions.length} suggestion{result.suggestions.length !== 1 ? 's' : ''}</SectionLabel>
            {result.model && (
              <span style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>
                {result.model} · {result.used_tokens} tokens
              </span>
            )}
          </div>
          {[...result.suggestions]
            .sort((a, b) => {
              const order = { high: 0, medium: 1, low: 2, info: 3 };
              return (order[a.severity] ?? 4) - (order[b.severity] ?? 4);
            })
            .map((s, i) => <SuggestionCard key={i} suggestion={s} />)
          }
        </div>
      )}

      {/* Clean bill of health */}
      {!isFetching && result?.suggestions?.length === 0 && !result?.error && (
        <Card style={{ padding: 20, textAlign: 'center' }}>
          <div style={{ fontSize: 24, marginBottom: 8 }}>✓</div>
          <div style={{ fontSize: 14, color: 'var(--green)', fontWeight: 600 }}>Dataset looks clean</div>
          <p style={{ fontSize: 13, color: 'var(--text-2)', marginTop: 6 }}>
            No significant issues found. You're ready to train.
          </p>
        </Card>
      )}
    </div>
  );
}
