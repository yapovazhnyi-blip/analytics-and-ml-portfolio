import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Zap, GitCompare, BarChart3, Plus, Trash2 } from 'lucide-react';
import api from '../api/client.js';
import { PageHeader, Card, Button, Spinner, SectionLabel, EmptyState } from '../components/ui.jsx';

// ── Score badge ───────────────────────────────────────────────────────────────

function ScoreBadge({ label, pct, size = 'md' }) {
  const color = pct >= 80 ? '#2ECC71' : pct >= 60 ? '#F39C12' : '#E74C3C';
  const dim = size === 'lg' ? { fontSize: 36, label: 11 } : { fontSize: 22, label: 10 };
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {label && (
        <span style={{ fontSize: dim.label, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          {label}
        </span>
      )}
      <span style={{ fontSize: dim.fontSize, fontWeight: 700, color, fontFamily: 'var(--font-mono)' }}>
        {pct}<span style={{ fontSize: dim.fontSize * 0.55 }}>%</span>
      </span>
    </div>
  );
}

// ── Criterion row ─────────────────────────────────────────────────────────────

function CriterionRow({ criterion, score_pct, explanation, winner }) {
  const color = score_pct >= 80 ? '#2ECC71' : score_pct >= 60 ? '#F39C12' : '#E74C3C';
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '120px 60px 1fr', gap: 12, padding: '8px 0', borderBottom: '1px solid var(--border)', alignItems: 'start' }}>
      <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-2)', textTransform: 'capitalize', paddingTop: 2 }}>
        {criterion} {winner && <span style={{ color: 'var(--accent)', marginLeft: 4 }}>↑</span>}
      </span>
      <span style={{ fontSize: 14, fontWeight: 700, color, fontFamily: 'var(--font-mono)' }}>{score_pct}%</span>
      <span style={{ fontSize: 12, color: 'var(--text-3)', lineHeight: 1.5 }}>{explanation}</span>
    </div>
  );
}

// ── Rubric selector ───────────────────────────────────────────────────────────

const RUBRICS = ['accuracy', 'helpfulness', 'safety', 'format', 'conciseness', 'completeness'];

function RubricSelector({ selected, onChange }) {
  return (
    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
      {RUBRICS.map(r => {
        const active = selected.includes(r);
        return (
          <button key={r} onClick={() => onChange(active ? selected.filter(x => x !== r) : [...selected, r])}
            style={{
              padding: '4px 10px', fontSize: 12, borderRadius: 4, cursor: 'pointer',
              fontFamily: 'var(--font-mono)', border: '1px solid',
              background: active ? 'var(--accent)' : 'var(--bg-3)',
              color: active ? '#000' : 'var(--text-2)',
              borderColor: active ? 'var(--accent)' : 'var(--border)',
            }}>
            {r}
          </button>
        );
      })}
    </div>
  );
}

const TA = ({ value, onChange, placeholder, rows = 4 }) => (
  <textarea value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder} rows={rows}
    style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 'var(--radius-sm)', padding: '10px 14px', fontSize: 13, lineHeight: 1.6, resize: 'vertical', fontFamily: 'var(--font-sans)', boxSizing: 'border-box' }} />
);

// ── Main page ─────────────────────────────────────────────────────────────────

const TABS = ['Judge', 'Batch', 'Compare', 'Hallucination'];

export default function EvaluationPage() {
  const [tab, setTab] = useState('Judge');
  const [rubrics, setRubrics] = useState(['accuracy', 'helpfulness', 'safety']);

  // ── Judge state
  const [jInput, setJInput] = useState('');
  const [jOutput, setJOutput] = useState('');
  const [jRef, setJRef] = useState('');
  const [judgeResult, setJudgeResult] = useState(null);

  // ── Batch state
  const [batchRows, setBatchRows] = useState([{ input: '', output: '' }]);
  const [batchResult, setBatchResult] = useState(null);

  // ── Compare state
  const [cInput, setCInput] = useState('');
  const [cA, setCA] = useState('');
  const [cB, setCB] = useState('');
  const [cLabelA, setCLabelA] = useState('Version A');
  const [cLabelB, setCLabelB] = useState('Version B');
  const [compareResult, setCompareResult] = useState(null);

  // ── Hallucination state
  const [hAnswer, setHAnswer] = useState('');
  const [hChunks, setHChunks] = useState(['']);
  const [hThreshold, setHThreshold] = useState(0.5);
  const [hallucinationResult, setHallucinationResult] = useState(null);

  const judgeMut = useMutation({
    mutationFn: () => api.post('/evaluation/judge', {
      input_text: jInput, output_text: jOutput,
      reference_text: jRef || undefined,
      rubric_names: rubrics.length ? rubrics : undefined,
    }).then(r => r.data.data),
    onSuccess: setJudgeResult,
  });

  const batchMut = useMutation({
    mutationFn: () => api.post('/evaluation/batch', {
      samples: batchRows.map(r => ({ input_text: r.input, output_text: r.output })),
      rubric_names: rubrics.length ? rubrics : undefined,
    }).then(r => r.data.data),
    onSuccess: setBatchResult,
  });

  const compareMut = useMutation({
    mutationFn: () => api.post('/evaluation/compare', {
      input_text: cInput, output_a: cA, output_b: cB,
      label_a: cLabelA, label_b: cLabelB,
      rubric_names: rubrics.length ? rubrics : undefined,
    }).then(r => r.data.data),
    onSuccess: setCompareResult,
  });

  const hallucinationMut = useMutation({
    mutationFn: () => api.post('/evaluation/hallucination', {
      answer: hAnswer,
      context_chunks: hChunks.filter(c => c.trim()),
      threshold: hThreshold,
    }).then(r => r.data.data),
    onSuccess: setHallucinationResult,
  });

  return (
    <div style={{ maxWidth: 900, margin: '0 auto', padding: 32 }}>
      <PageHeader title="LLM Evaluation" subtitle="Score any LLM output using Claude as judge" />

      {/* Rubric selector — shared across LLM-judge tabs only */}
      {tab !== 'Hallucination' && (
      <Card style={{ marginBottom: 20 }}>
        <SectionLabel>Evaluation Criteria</SectionLabel>
        <RubricSelector selected={rubrics} onChange={setRubrics} />
        <p style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 8 }}>
          Select the criteria to evaluate. Requires <code style={{ fontFamily: 'var(--font-mono)' }}>ANTHROPIC_API_KEY</code>.
        </p>
      </Card>
      )}

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 2, marginBottom: 24, borderBottom: '1px solid var(--border)' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 20px', background: 'none', border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: 500,
            color: tab === t ? 'var(--accent)' : 'var(--text-3)',
            borderBottom: tab === t ? '2px solid var(--accent)' : '2px solid transparent',
            marginBottom: -1,
          }}>{t}</button>
        ))}
      </div>

      {/* ── JUDGE TAB ── */}
      {tab === 'Judge' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div>
            <SectionLabel>Input (prompt / question)</SectionLabel>
            <TA value={jInput} onChange={setJInput} placeholder="What is machine learning?" rows={3} />
          </div>
          <div>
            <SectionLabel>Output to evaluate</SectionLabel>
            <TA value={jOutput} onChange={setJOutput} placeholder="The model's response to evaluate…" rows={5} />
          </div>
          <div>
            <SectionLabel>Reference answer <span style={{ color: 'var(--text-3)', fontWeight: 400 }}>(optional — ideal answer for comparison)</span></SectionLabel>
            <TA value={jRef} onChange={setJRef} placeholder="Ideal answer for reference-based scoring…" rows={3} />
          </div>
          <Button variant="primary" onClick={() => judgeMut.mutate()} loading={judgeMut.isPending}
            disabled={!jInput.trim() || !jOutput.trim()}>
            <Zap size={14} style={{ marginRight: 6 }} /> Evaluate
          </Button>

          {judgeResult && (
            <Card>
              <div style={{ display: 'flex', gap: 24, marginBottom: 20, alignItems: 'flex-end' }}>
                <ScoreBadge label="Overall" pct={judgeResult.overall_pct} size="lg" />
                {judgeResult.scores.map(s => (
                  <ScoreBadge key={s.criterion} label={s.criterion} pct={s.score_pct} />
                ))}
              </div>
              {judgeResult.scores.map(s => (
                <CriterionRow key={s.criterion} {...s} />
              ))}
              {judgeResult.model && (
                <p style={{ marginTop: 12, fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>
                  {judgeResult.model} · {judgeResult.input_tokens}↑ {judgeResult.output_tokens}↓ tokens
                </p>
              )}
            </Card>
          )}
        </div>
      )}

      {/* ── BATCH TAB ── */}
      {tab === 'Batch' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {batchRows.map((row, i) => (
            <Card key={i} style={{ padding: '14px 16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                <span style={{ fontSize: 12, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>Sample {i + 1}</span>
                {batchRows.length > 1 && (
                  <button onClick={() => setBatchRows(batchRows.filter((_, j) => j !== i))}
                    style={{ background: 'none', border: 'none', color: 'var(--text-3)', cursor: 'pointer', display: 'flex' }}>
                    <Trash2 size={13} />
                  </button>
                )}
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div>
                  <SectionLabel>Input</SectionLabel>
                  <TA value={row.input} rows={3} placeholder="Prompt / question"
                    onChange={v => setBatchRows(batchRows.map((r, j) => j === i ? { ...r, input: v } : r))} />
                </div>
                <div>
                  <SectionLabel>Output</SectionLabel>
                  <TA value={row.output} rows={3} placeholder="Model response"
                    onChange={v => setBatchRows(batchRows.map((r, j) => j === i ? { ...r, output: v } : r))} />
                </div>
              </div>
            </Card>
          ))}

          <div style={{ display: 'flex', gap: 10 }}>
            <Button variant="ghost" onClick={() => setBatchRows([...batchRows, { input: '', output: '' }])}>
              <Plus size={14} style={{ marginRight: 6 }} /> Add sample
            </Button>
            <Button variant="primary" loading={batchMut.isPending}
              disabled={batchRows.some(r => !r.input.trim() || !r.output.trim())}
              onClick={() => batchMut.mutate()}>
              <BarChart3 size={14} style={{ marginRight: 6 }} /> Run batch
            </Button>
          </div>

          {batchResult && (
            <Card>
              <SectionLabel>Aggregate Results — {batchResult.n_samples} samples</SectionLabel>
              <div style={{ display: 'flex', gap: 20, marginBottom: 20 }}>
                <ScoreBadge label="Overall" pct={Math.round(batchResult.overall_mean * 100)} size="lg" />
                {Object.entries(batchResult.criterion_means).map(([name, val]) => (
                  <ScoreBadge key={name} label={name} pct={Math.round(val * 100)} />
                ))}
              </div>
              {batchResult.results.map((r, i) => {
                const pct = Math.round((r.overall_score ?? 0) * 100);
                const col = pct >= 80 ? '#2ECC71' : pct >= 60 ? '#F39C12' : '#E74C3C';
                return (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--border)', fontSize: 13 }}>
                    <span style={{ color: 'var(--text-2)' }}>Sample {i + 1}</span>
                    <span style={{ fontWeight: 700, color: col, fontFamily: 'var(--font-mono)' }}>{pct}%</span>
                  </div>
                );
              })}
            </Card>
          )}
        </div>
      )}

      {/* ── COMPARE TAB ── */}
      {tab === 'Compare' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div>
            <SectionLabel>Input</SectionLabel>
            <TA value={cInput} onChange={setCInput} placeholder="The same prompt given to both versions…" rows={3} />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
                <SectionLabel style={{ margin: 0 }}>Output</SectionLabel>
                <input value={cLabelA} onChange={e => setCLabelA(e.target.value)}
                  style={{ background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--accent)', borderRadius: 4, padding: '2px 8px', fontSize: 12, fontFamily: 'var(--font-mono)', width: 110 }} />
              </div>
              <TA value={cA} onChange={setCA} placeholder={`${cLabelA} response…`} rows={6} />
            </div>
            <div>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
                <SectionLabel style={{ margin: 0 }}>Output</SectionLabel>
                <input value={cLabelB} onChange={e => setCLabelB(e.target.value)}
                  style={{ background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--blue)', borderRadius: 4, padding: '2px 8px', fontSize: 12, fontFamily: 'var(--font-mono)', width: 110 }} />
              </div>
              <TA value={cB} onChange={setCB} placeholder={`${cLabelB} response…`} rows={6} />
            </div>
          </div>
          <Button variant="primary" loading={compareMut.isPending}
            disabled={!cInput.trim() || !cA.trim() || !cB.trim()}
            onClick={() => compareMut.mutate()}>
            <GitCompare size={14} style={{ marginRight: 6 }} /> Compare
          </Button>

          {compareResult && (
            <Card>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
                <div style={{ display: 'flex', gap: 24 }}>
                  <div>
                    <span style={{ fontSize: 11, color: 'var(--text-3)', display: 'block', marginBottom: 4, fontFamily: 'var(--font-mono)' }}>{compareResult.label_a}</span>
                    <span style={{ fontSize: 32, fontWeight: 700, color: 'var(--accent)', fontFamily: 'var(--font-mono)' }}>{compareResult.result_a.overall_pct}%</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', color: 'var(--text-3)' }}>vs</div>
                  <div>
                    <span style={{ fontSize: 11, color: 'var(--text-3)', display: 'block', marginBottom: 4, fontFamily: 'var(--font-mono)' }}>{compareResult.label_b}</span>
                    <span style={{ fontSize: 32, fontWeight: 700, color: 'var(--blue)', fontFamily: 'var(--font-mono)' }}>{compareResult.result_b.overall_pct}%</span>
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  {compareResult.winner ? (
                    <span style={{ fontSize: 18, fontWeight: 700, color: 'var(--green)' }}>
                      🏆 {compareResult.winner} wins
                    </span>
                  ) : (
                    <span style={{ fontSize: 16, color: 'var(--text-3)' }}>Tie</span>
                  )}
                </div>
              </div>

              <SectionLabel>Per Criterion</SectionLabel>
              {compareResult.result_a.scores.map((sa, i) => {
                const sb = compareResult.result_b.scores[i];
                const winner = compareResult.per_criterion?.[sa.criterion];
                return (
                  <div key={sa.criterion} style={{ display: 'grid', gridTemplateColumns: '120px 1fr 60px 60px', gap: 12, padding: '8px 0', borderBottom: '1px solid var(--border)', alignItems: 'center' }}>
                    <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-2)', textTransform: 'capitalize' }}>{sa.criterion}</span>
                    <div style={{ height: 6, background: 'var(--bg-4)', borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${sa.score_pct}%`, background: 'var(--accent)', borderRadius: 3 }} />
                    </div>
                    <span style={{ fontSize: 13, fontWeight: 700, fontFamily: 'var(--font-mono)', color: winner === compareResult.label_a ? 'var(--green)' : 'var(--text-2)', textAlign: 'right' }}>{sa.score_pct}%</span>
                    <span style={{ fontSize: 13, fontWeight: 700, fontFamily: 'var(--font-mono)', color: winner === compareResult.label_b ? 'var(--green)' : 'var(--text-2)', textAlign: 'right' }}>{sb?.score_pct ?? '—'}%</span>
                  </div>
                );
              })}
            </Card>
          )}
        </div>
      )}

      {/* ── HALLUCINATION TAB ── */}
      {tab === 'Hallucination' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

          {/* Info banner */}
          <div style={{ background: 'var(--bg-3)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '10px 14px', fontSize: 12, color: 'var(--text-2)', lineHeight: 1.6 }}>
            <strong style={{ color: 'var(--text-1)' }}>NLI Faithfulness Scorer</strong> — checks each sentence in the answer against your context chunks using a local cross-encoder model (<code style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}>cross-encoder/nli-deberta-v3-small</code>, ~185 MB).
            {' '}The model downloads automatically on first use. No API key required. Inference: ~2–4 s on CPU.
          </div>

          {/* Answer */}
          <div>
            <SectionLabel>Answer to check</SectionLabel>
            <TA value={hAnswer} onChange={setHAnswer} rows={6}
              placeholder="Paste the LLM-generated answer you want to check for hallucinations…" />
          </div>

          {/* Context chunks */}
          <div>
            <SectionLabel>Context chunks <span style={{ color: 'var(--text-3)', fontWeight: 400 }}>(source passages the answer should be grounded in)</span></SectionLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {hChunks.map((chunk, i) => (
                <div key={i} style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                  <span style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', paddingTop: 10, minWidth: 20 }}>#{i + 1}</span>
                  <TA value={chunk} rows={3}
                    placeholder={`Context passage ${i + 1}…`}
                    onChange={v => setHChunks(hChunks.map((c, j) => j === i ? v : c))} />
                  {hChunks.length > 1 && (
                    <button onClick={() => setHChunks(hChunks.filter((_, j) => j !== i))}
                      style={{ background: 'none', border: 'none', color: 'var(--text-3)', cursor: 'pointer', paddingTop: 8 }}>
                      <Trash2 size={14} />
                    </button>
                  )}
                </div>
              ))}
              <Button variant="ghost" style={{ alignSelf: 'flex-start' }}
                onClick={() => setHChunks([...hChunks, ''])}>
                <Plus size={14} style={{ marginRight: 6 }} /> Add chunk
              </Button>
            </div>
          </div>

          {/* Threshold slider */}
          <div>
            <SectionLabel>
              Entailment threshold — <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>{hThreshold.toFixed(2)}</span>
              <span style={{ fontWeight: 400, color: 'var(--text-3)', marginLeft: 8 }}>
                (sentences with score ≥ this are grounded; lower = more permissive)
              </span>
            </SectionLabel>
            <input type="range" min="0.1" max="0.9" step="0.05"
              value={hThreshold}
              onChange={e => setHThreshold(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--accent)' }} />
          </div>

          <Button variant="primary" loading={hallucinationMut.isPending}
            disabled={!hAnswer.trim() || !hChunks.some(c => c.trim())}
            onClick={() => { setHallucinationResult(null); hallucinationMut.mutate(); }}>
            <Zap size={14} style={{ marginRight: 6 }} /> Check for hallucinations
          </Button>

          {hallucinationMut.isError && (
            <div style={{ fontSize: 12, color: 'var(--red)', padding: '8px 12px', background: 'rgba(231,76,60,0.08)', borderRadius: 'var(--radius)' }}>
              {hallucinationMut.error?.response?.data?.detail || 'Scoring failed — check backend logs.'}
            </div>
          )}

          {/* Results */}
          {hallucinationResult && (
            <Card>
              {/* Score header */}
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: 32, marginBottom: 20 }}>
                <div style={{ textAlign: 'center', minWidth: 90 }}>
                  <div style={{
                    fontSize: 44, fontWeight: 800, fontFamily: 'var(--font-mono)', lineHeight: 1,
                    color: hallucinationResult.faithfulness_pct >= 80 ? '#2ECC71'
                         : hallucinationResult.faithfulness_pct >= 50 ? '#F39C12' : '#E74C3C',
                  }}>
                    {hallucinationResult.faithfulness_pct}<span style={{ fontSize: 22 }}>%</span>
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 4, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                    Faithful
                  </div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, flex: 1 }}>
                  {[
                    ['Grounded sentences', `${hallucinationResult.grounded_count} / ${hallucinationResult.total_count}`],
                    ['Hallucination rate', `${Math.round(hallucinationResult.hallucination_rate * 100)}%`],
                    ['Model', hallucinationResult.model_id],
                    ['Inference time', `${hallucinationResult.inference_ms} ms`],
                  ].map(([label, val]) => (
                    <div key={label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, borderBottom: '1px solid var(--border)', paddingBottom: 4 }}>
                      <span style={{ color: 'var(--text-3)' }}>{label}</span>
                      <span style={{ color: 'var(--text-1)', fontFamily: label === 'Model' ? 'var(--font-mono)' : 'inherit', fontSize: label === 'Model' ? 10 : 12 }}>{val}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Per-sentence breakdown */}
              <SectionLabel>Sentence-level breakdown</SectionLabel>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {hallucinationResult.sentences.map((s, i) => (
                  <div key={i} style={{
                    borderRadius: 'var(--radius-sm)',
                    border: `1px solid ${s.is_grounded ? 'rgba(46,204,113,0.3)' : 'rgba(231,76,60,0.3)'}`,
                    background: s.is_grounded ? 'rgba(46,204,113,0.06)' : 'rgba(231,76,60,0.06)',
                    padding: '10px 14px',
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                      <span style={{
                        fontSize: 10, fontFamily: 'var(--font-mono)', fontWeight: 700,
                        color: s.is_grounded ? '#2ECC71' : '#E74C3C',
                        textTransform: 'uppercase', letterSpacing: '0.06em',
                      }}>
                        {s.is_grounded ? '✓ Grounded' : '✗ Hallucinated'}
                      </span>
                      <span style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>
                        p={s.entailment_score.toFixed(3)}
                      </span>
                    </div>
                    <p style={{ fontSize: 13, color: 'var(--text-1)', margin: '0 0 0', lineHeight: 1.5 }}>
                      {s.sentence}
                    </p>
                    {s.is_grounded && s.best_chunk_snippet && (
                      <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-3)', lineHeight: 1.5, borderTop: '1px solid var(--border)', paddingTop: 6 }}>
                        <span style={{ color: 'var(--text-2)', fontWeight: 500 }}>Supported by chunk #{s.best_chunk_index + 1}: </span>
                        "{s.best_chunk_snippet}"
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {hallucinationResult.error && (
                <p style={{ marginTop: 12, fontSize: 12, color: 'var(--red)' }}>⚠ {hallucinationResult.error}</p>
              )}
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
