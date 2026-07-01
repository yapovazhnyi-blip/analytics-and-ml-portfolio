import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  ShieldCheck, ShieldAlert, ShieldX, RefreshCw, Trash2, PlayCircle, Plus, ChevronDown, ChevronRight
} from 'lucide-react';
import { contracts, datasets } from '../api/client.js';
import { Card, Button, Spinner, SectionLabel, StatusBadge } from './ui.jsx';

// ── Severity / status helpers ─────────────────────────────────────────────────

const DTYPE_COLOR = {
  numeric:     '#3498DB',
  categorical: '#9B59B6',
  boolean:     '#27AE60',
  datetime:    '#E67E22',
  text:        '#95A5A6',
};

function ViolationRow({ v }) {
  const [open, setOpen] = useState(false);
  const isError = v.severity === 'error';
  return (
    <div style={{
      borderBottom: '1px solid var(--border)',
      background: isError ? 'rgba(231,76,60,0.04)' : 'rgba(243,156,18,0.04)',
    }}>
      <div onClick={() => v.examples?.length && setOpen(o => !o)}
        style={{ display: 'grid', gridTemplateColumns: '140px 130px 1fr 80px', gap: 8, padding: '8px 12px', alignItems: 'center', cursor: v.examples?.length ? 'pointer' : 'default' }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-1)', fontWeight: 500 }}>{v.column}</span>
        <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>{v.check}</span>
        <span style={{ fontSize: 12, color: 'var(--text-2)' }}>{v.observed}</span>
        <span style={{ fontSize: 11, fontWeight: 700, color: isError ? 'var(--red)' : '#F39C12', textTransform: 'uppercase' }}>
          {v.severity} · {v.n_rows_failed.toLocaleString()} rows
        </span>
      </div>
      {open && v.examples?.length > 0 && (
        <div style={{ padding: '4px 12px 10px 156px', fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>
          Examples: {v.examples.slice(0, 5).join(', ')}
        </div>
      )}
    </div>
  );
}

function ColumnRow({ col }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ borderBottom: '1px solid var(--border)' }}>
      <div onClick={() => setOpen(o => !o)}
        style={{ display: 'grid', gridTemplateColumns: '180px 100px 1fr', gap: 8, padding: '7px 12px', cursor: 'pointer', alignItems: 'center' }}
        onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-3)'}
        onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {open ? <ChevronDown size={11} style={{ color: 'var(--text-3)', flexShrink: 0 }} /> : <ChevronRight size={11} style={{ color: 'var(--text-3)', flexShrink: 0 }} />}
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-1)' }}>{col.name}</span>
        </div>
        <span style={{
          fontSize: 10, fontWeight: 600, padding: '2px 6px', borderRadius: 4,
          background: `${DTYPE_COLOR[col.dtype_family] || '#95A5A6'}20`,
          color: DTYPE_COLOR[col.dtype_family] || '#95A5A6',
          fontFamily: 'var(--font-mono)', textTransform: 'uppercase',
        }}>{col.dtype_family}</span>
        <span style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>
          {col.nullable ? `nullable ≤ ${(col.max_null_rate * 100).toFixed(0)}%` : 'non-null'}
          {col.min_val != null && ` · [${col.min_val.toFixed(2)}, ${col.max_val.toFixed(2)}]`}
          {col.allowed_values && ` · ${col.allowed_values.length} categories`}
        </span>
      </div>
      {open && (
        <div style={{ padding: '6px 12px 10px 28px', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, fontSize: 11, color: 'var(--text-3)' }}>
          {col.min_val != null && <div>Min: <strong style={{ color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>{col.min_val.toFixed(4)}</strong></div>}
          {col.max_val != null && <div>Max: <strong style={{ color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>{col.max_val.toFixed(4)}</strong></div>}
          {col.max_null_rate != null && <div>Max null: <strong style={{ color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>{(col.max_null_rate * 100).toFixed(1)}%</strong></div>}
          {col.allowed_values && (
            <div style={{ gridColumn: '1/-1' }}>
              Categories: <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-2)' }}>
                {col.allowed_values.slice(0, 10).join(', ')}{col.allowed_values.length > 10 ? ` +${col.allowed_values.length - 10} more` : ''}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function DataContractPanel({ datasetId, allDatasets = [] }) {
  const qc = useQueryClient();
  const [validateAgainst, setValidateAgainst] = useState('');
  const [tolerance, setTolerance] = useState(0.10);
  const [validationResult, setValidationResult] = useState(null);

  const { data: contract, isLoading, refetch } = useQuery({
    queryKey: ['contract', datasetId],
    queryFn:  () => contracts.get(datasetId),
    retry:    false,
  });

  const generateMut = useMutation({
    mutationFn: () => contracts.generate(datasetId, { tolerance }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['contract', datasetId] }); },
  });

  const validateMut = useMutation({
    mutationFn: () => contracts.validate(datasetId, parseInt(validateAgainst)),
    onSuccess: setValidationResult,
  });

  const deleteMut = useMutation({
    mutationFn: () => contracts.delete(datasetId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['contract', datasetId] });
      setValidationResult(null);
    },
  });

  return (
    <Card>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
        <SectionLabel style={{ margin: 0 }}>Data Contract</SectionLabel>
        <div style={{ display: 'flex', gap: 8 }}>
          {contract && (
            <Button variant="ghost" onClick={() => deleteMut.mutate()} title="Delete contract">
              <Trash2 size={13} />
            </Button>
          )}
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <input type="number" value={tolerance} min={0} max={0.5} step={0.05}
              onChange={e => setTolerance(parseFloat(e.target.value))}
              style={{ width: 60, background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 6, padding: '4px 8px', fontSize: 12, textAlign: 'center' }}
              title="Numeric tolerance (0–0.5)" />
            <Button variant="primary" onClick={() => generateMut.mutate()} loading={generateMut.isPending}>
              {contract ? <><RefreshCw size={12} style={{ marginRight: 4 }} />Regenerate</> : <><Plus size={12} style={{ marginRight: 4 }} />Generate</>}
            </Button>
          </div>
        </div>
      </div>

      {isLoading && <div style={{ padding: 16, textAlign: 'center' }}><Spinner /></div>}

      {!contract && !isLoading && (
        <p style={{ fontSize: 13, color: 'var(--text-3)', lineHeight: 1.6 }}>
          No contract yet. Generate one to capture expectations about this dataset's structure,
          value ranges, and allowed categories. New data batches can then be validated against it.
        </p>
      )}

      {contract && (
        <>
          {/* Contract summary */}
          <div style={{ display: 'flex', gap: 16, marginBottom: 14, padding: '8px 12px', background: 'var(--bg-3)', borderRadius: 8 }}>
            <div><span style={{ fontSize: 10, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.06em', fontFamily: 'var(--font-mono)' }}>Columns</span>
              <div style={{ fontWeight: 700, fontSize: 15, color: 'var(--text-1)' }}>{contract.n_cols}</div></div>
            <div><span style={{ fontSize: 10, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.06em', fontFamily: 'var(--font-mono)' }}>Reference rows</span>
              <div style={{ fontWeight: 700, fontSize: 15, color: 'var(--text-1)' }}>{contract.n_rows_reference?.toLocaleString()}</div></div>
            <div><span style={{ fontSize: 10, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.06em', fontFamily: 'var(--font-mono)' }}>Tolerance</span>
              <div style={{ fontWeight: 700, fontSize: 15, color: 'var(--text-1)' }}>±{(contract.tolerance * 100).toFixed(0)}%</div></div>
            {contract.target_column && (
              <div><span style={{ fontSize: 10, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.06em', fontFamily: 'var(--font-mono)' }}>Target</span>
                <div style={{ fontWeight: 700, fontSize: 15, color: 'var(--accent)', fontFamily: 'var(--font-mono)' }}>{contract.target_column}</div></div>
            )}
          </div>

          {/* Column list */}
          <div style={{ border: '1px solid var(--border)', borderRadius: 8, overflow: 'hidden', marginBottom: 16 }}>
            <div style={{ display: 'grid', gridTemplateColumns: '180px 100px 1fr', gap: 8, padding: '6px 12px', background: 'var(--bg-3)', borderBottom: '1px solid var(--border)', fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              <span>Column</span><span>Type</span><span>Constraints</span>
            </div>
            {contract.columns?.map(col => <ColumnRow key={col.name} col={col} />)}
          </div>

          {/* Validate section */}
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: validationResult ? 12 : 0 }}>
            <select value={validateAgainst} onChange={e => setValidateAgainst(e.target.value)}
              style={{ flex: 1, background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 6, padding: '7px 10px', fontSize: 13 }}>
              <option value="">Validate another dataset against this contract…</option>
              {allDatasets.filter(d => d.id !== datasetId && d.status === 'ready').map(d => (
                <option key={d.id} value={d.id}>{d.name}</option>
              ))}
            </select>
            <Button variant="ghost" disabled={!validateAgainst} loading={validateMut.isPending}
              onClick={() => validateMut.mutate()}>
              <PlayCircle size={13} style={{ marginRight: 4 }} /> Validate
            </Button>
          </div>
        </>
      )}

      {/* Validation result */}
      {validationResult && (
        <div style={{ marginTop: 12, border: `1px solid ${validationResult.passed ? 'var(--green)' : 'var(--red)'}40`, borderRadius: 8, overflow: 'hidden' }}>
          <div style={{
            padding: '10px 14px',
            background: validationResult.passed ? 'rgba(46,204,113,0.08)' : 'rgba(231,76,60,0.08)',
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              {validationResult.passed
                ? <ShieldCheck size={16} style={{ color: 'var(--green)' }} />
                : <ShieldX size={16} style={{ color: 'var(--red)' }} />}
              <span style={{ fontWeight: 700, fontSize: 14, color: validationResult.passed ? 'var(--green)' : 'var(--red)' }}>
                {validationResult.passed ? 'Contract passed' : 'Contract violated'}
              </span>
            </div>
            <span style={{ fontSize: 12, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>
              {validationResult.n_errors} error{validationResult.n_errors !== 1 ? 's' : ''}
              {validationResult.n_warnings > 0 ? ` · ${validationResult.n_warnings} warning${validationResult.n_warnings !== 1 ? 's' : ''}` : ''}
              {' · '}{validationResult.n_rows?.toLocaleString()} rows
            </span>
          </div>

          {validationResult.violations?.length > 0 && (
            <div style={{ borderTop: '1px solid var(--border)' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '140px 130px 1fr 80px', gap: 8, padding: '6px 12px', background: 'var(--bg-3)', fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.06em', borderBottom: '1px solid var(--border)' }}>
                <span>Column</span><span>Check</span><span>Observed</span><span>Severity</span>
              </div>
              {validationResult.violations.map((v, i) => <ViolationRow key={i} v={v} />)}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
