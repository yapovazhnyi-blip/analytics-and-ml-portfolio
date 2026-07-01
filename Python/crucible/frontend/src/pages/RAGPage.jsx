import { useRef, useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Upload, FileText, Trash2, Search, Zap, BarChart3,
  ChevronDown, ChevronUp, AlertCircle, CheckCircle2,
} from 'lucide-react';
import { rag } from '../api/client.js';
import {
  PageHeader, Card, Button, Spinner, SectionLabel, StatusBadge, EmptyState,
} from '../components/ui.jsx';

// ── Metric badge ──────────────────────────────────────────────────────────────

function MetricBadge({ label, value, description }) {
  const pct = Math.round((value ?? 0) * 100);
  const color = pct >= 80 ? '#2ECC71' : pct >= 60 ? '#F39C12' : '#E74C3C';
  return (
    <div style={{
      background: 'var(--bg-3)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius)', padding: '16px 20px',
      display: 'flex', flexDirection: 'column', gap: 4, minWidth: 140,
    }}>
      <span style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</span>
      <span style={{ fontSize: 32, fontWeight: 700, color, fontFamily: 'var(--font-mono)' }}>
        {pct}<span style={{ fontSize: 16 }}>%</span>
      </span>
      {description && <span style={{ fontSize: 11, color: 'var(--text-3)' }}>{description}</span>}
    </div>
  );
}

// ── Citation card ─────────────────────────────────────────────────────────────

function CitationCard({ citation, chunk }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div style={{
      background: 'var(--bg-3)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-sm)', padding: '10px 14px', marginBottom: 8,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>
          {citation.source_name} · chunk {citation.chunk_index}
        </span>
        {chunk && (
          <button onClick={() => setExpanded(e => !e)}
            style={{ background: 'none', border: 'none', color: 'var(--text-3)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4, fontSize: 11 }}>
            {expanded ? <><ChevronUp size={12} /> Hide</> : <><ChevronDown size={12} /> Show context</>}
          </button>
        )}
      </div>
      {expanded && chunk && (
        <p style={{ marginTop: 8, fontSize: 12, color: 'var(--text-2)', lineHeight: 1.6, borderTop: '1px solid var(--border)', paddingTop: 8 }}>
          {chunk.text}
        </p>
      )}
    </div>
  );
}

// ── Document row ──────────────────────────────────────────────────────────────

function DocumentRow({ doc, onDelete, onSelect, selected }) {
  return (
    <div
      onClick={() => onSelect(doc.document_id)}
      style={{
        display: 'grid', gridTemplateColumns: '1fr 100px 80px 80px 44px',
        padding: '12px 16px', alignItems: 'center', cursor: 'pointer',
        borderBottom: '1px solid var(--border)',
        background: selected ? 'var(--bg-3)' : 'transparent',
        transition: 'background 0.1s',
      }}
      onMouseEnter={e => { if (!selected) e.currentTarget.style.background = 'var(--bg-3)'; }}
      onMouseLeave={e => { if (!selected) e.currentTarget.style.background = 'transparent'; }}
    >
      <div>
        <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-1)' }}>{doc.name}</div>
        <div style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginTop: 2 }}>{doc.filename}</div>
      </div>
      <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-2)' }}>
        {doc.chunk_count} chunks
      </span>
      <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-2)' }}>
        {doc.chunk_strategy}
      </span>
      <StatusBadge status={doc.status} />
      <button
        onClick={e => { e.stopPropagation(); if (confirm(`Delete "${doc.name}"?`)) onDelete(doc.document_id); }}
        style={{ background: 'none', border: 'none', color: 'var(--text-3)', cursor: 'pointer', padding: 6, borderRadius: 'var(--radius-sm)', display: 'flex', alignItems: 'center' }}
        onMouseEnter={e => { e.currentTarget.style.color = 'var(--red)'; }}
        onMouseLeave={e => { e.currentTarget.style.color = 'var(--text-3)'; }}
      >
        <Trash2 size={13} />
      </button>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

const TABS = ['Query', 'Documents', 'Evaluate'];

export default function RAGPage() {
  const qc = useQueryClient();
  const fileRef = useRef(null);
  const [tab, setTab] = useState('Query');
  const [selectedDocId, setSelectedDocId] = useState(null);
  const [question, setQuestion] = useState('');
  const [queryResult, setQueryResult] = useState(null);
  const [uploadError, setUploadError] = useState('');
  const [chunkStrategy, setChunkStrategy] = useState('paragraph');
  const [evalCsv, setEvalCsv] = useState('');
  const [evalResult, setEvalResult] = useState(null);
  const [evalDocId, setEvalDocId] = useState('');
  const evalFileRef = useRef(null);

  const { data: docsData, isLoading } = useQuery({
    queryKey: ['rag-documents'],
    queryFn: () => rag.listDocuments(),
    refetchInterval: (data) => {
      const docs = data?.data ?? [];
      return docs.some(d => d.status === 'indexing') ? 2000 : false;
    },
  });

  const docs = docsData?.data ?? [];

  const uploadMut = useMutation({
    mutationFn: ({ file, name }) =>
      rag.uploadDocument(file, name, chunkStrategy),
    onSuccess: () => {
      setUploadError('');
      qc.invalidateQueries({ queryKey: ['rag-documents'] });
    },
    onError: (err) => {
      setUploadError(err.response?.data?.detail || 'Upload failed');
    },
  });

  const deleteMut = useMutation({
    mutationFn: (docId) => rag.deleteDocument(docId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['rag-documents'] });
      if (selectedDocId) setSelectedDocId(null);
    },
  });

  const queryMut = useMutation({
    mutationFn: ({ q, docId }) =>
      docId ? rag.queryDocument(docId, q) : rag.query(q),
    onSuccess: (data) => setQueryResult(data),
  });

  const evalMut = useMutation({
    mutationFn: (cases) =>
      rag.evaluate(cases, evalDocId ? [evalDocId] : null),
    onSuccess: (data) => setEvalResult(data),
  });

  function handleFileChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    uploadMut.mutate({ file, name: file.name.replace(/\.[^.]+$/, '') });
    e.target.value = '';
  }

  function handleQuery(e) {
    e.preventDefault();
    if (!question.trim()) return;
    const readyDoc = docs.find(d => d.document_id === selectedDocId && d.status === 'ready');
    queryMut.mutate({ q: question, docId: readyDoc ? selectedDocId : null });
  }

  function handleEvalCsv(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => setEvalCsv(ev.target.result);
    reader.readAsText(file);
    e.target.value = '';
  }

  function runEvaluation() {
    const lines = evalCsv.trim().split('\n').filter(Boolean);
    if (lines.length < 2) return;
    const header = lines[0].toLowerCase();
    const qIdx = header.split(',').findIndex(h => h.includes('question'));
    const gtIdx = header.split(',').findIndex(h => h.includes('ground') || h.includes('truth') || h.includes('answer'));
    if (qIdx < 0) { alert('CSV must have a "question" column'); return; }
    const cases = lines.slice(1).map(line => {
      const cols = line.split(',');
      return { question: cols[qIdx]?.trim() || '', ground_truth: gtIdx >= 0 ? cols[gtIdx]?.trim() : null };
    }).filter(c => c.question);
    evalMut.mutate(cases);
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 32 }}>
      <PageHeader
        title="RAG Pipeline"
        subtitle={`${docs.length} document${docs.length !== 1 ? 's' : ''} indexed`}
        actions={
          tab === 'Documents' && (
            <>
              <select
                value={chunkStrategy}
                onChange={e => setChunkStrategy(e.target.value)}
                style={{ background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 'var(--radius-sm)', padding: '6px 10px', fontSize: 13 }}
              >
                <option value="paragraph">Paragraph chunks</option>
                <option value="sentence">Sentence chunks</option>
                <option value="fixed">Fixed-size chunks</option>
              </select>
              <Button variant="primary" onClick={() => fileRef.current?.click()} loading={uploadMut.isPending}>
                <Upload size={14} style={{ marginRight: 6 }} /> Upload document
              </Button>
              <input ref={fileRef} type="file" accept=".pdf,.docx,.txt,.md,.rst" onChange={handleFileChange} style={{ display: 'none' }} />
            </>
          )
        }
      />

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

      {/* ── QUERY TAB ── */}
      {tab === 'Query' && (
        <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: 24 }}>
          {/* Document filter */}
          <Card style={{ padding: 0, height: 'fit-content' }}>
            <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)' }}>
              <SectionLabel>Filter by document</SectionLabel>
            </div>
            <div
              onClick={() => setSelectedDocId(null)}
              style={{
                padding: '10px 16px', cursor: 'pointer', fontSize: 13,
                background: !selectedDocId ? 'var(--bg-3)' : 'transparent',
                color: !selectedDocId ? 'var(--accent)' : 'var(--text-2)',
                borderBottom: '1px solid var(--border)',
              }}
            >
              All documents
            </div>
            {docs.filter(d => d.status === 'ready').map(d => (
              <div
                key={d.document_id}
                onClick={() => setSelectedDocId(d.document_id)}
                style={{
                  padding: '10px 16px', cursor: 'pointer', fontSize: 12,
                  fontFamily: 'var(--font-mono)',
                  background: selectedDocId === d.document_id ? 'var(--bg-3)' : 'transparent',
                  color: selectedDocId === d.document_id ? 'var(--accent)' : 'var(--text-2)',
                  borderBottom: '1px solid var(--border)',
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}
                title={d.name}
              >
                {d.name}
              </div>
            ))}
            {docs.filter(d => d.status === 'ready').length === 0 && (
              <div style={{ padding: '16px', fontSize: 12, color: 'var(--text-3)', textAlign: 'center' }}>
                No ready documents.<br />Upload one in Documents tab.
              </div>
            )}
          </Card>

          {/* Query panel */}
          <div>
            <form onSubmit={handleQuery} style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
              <input
                value={question}
                onChange={e => setQuestion(e.target.value)}
                placeholder="Ask a question about your documents…"
                style={{
                  flex: 1, background: 'var(--bg-3)', border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)', padding: '10px 14px',
                  color: 'var(--text-1)', fontSize: 14,
                }}
              />
              <Button type="submit" variant="primary" loading={queryMut.isPending}>
                <Search size={14} style={{ marginRight: 6 }} /> Ask
              </Button>
            </form>

            {queryMut.isPending && (
              <div style={{ display: 'flex', gap: 12, alignItems: 'center', color: 'var(--text-3)', padding: '20px 0' }}>
                <Spinner size={16} /> Retrieving context and generating answer…
              </div>
            )}

            {queryResult && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                {/* Answer */}
                <Card>
                  <SectionLabel>Answer</SectionLabel>
                  {queryResult.error ? (
                    <div style={{ color: 'var(--red)', fontSize: 13, display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                      <AlertCircle size={14} style={{ marginTop: 2, flexShrink: 0 }} />
                      {queryResult.error}
                    </div>
                  ) : (
                    <>
                      <p style={{ fontSize: 14, lineHeight: 1.7, color: 'var(--text-1)', margin: 0 }}>
                        {queryResult.answer}
                      </p>
                      {queryResult.model && (
                        <div style={{ marginTop: 12, fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', display: 'flex', gap: 16 }}>
                          <span>model: {queryResult.model}</span>
                          <span>{queryResult.input_tokens}↑ {queryResult.output_tokens}↓ tokens</span>
                        </div>
                      )}
                    </>
                  )}
                </Card>

                {/* Citations */}
                {queryResult.citations?.length > 0 && (
                  <div>
                    <SectionLabel>Sources</SectionLabel>
                    {queryResult.citations.map((c, i) => (
                      <CitationCard
                        key={i}
                        citation={c}
                        chunk={queryResult.retrieved_chunks?.[i]}
                      />
                    ))}
                  </div>
                )}

                {/* Retrieved chunks */}
                {queryResult.retrieved_chunks?.length > 0 && (
                  <details>
                    <summary style={{ fontSize: 12, color: 'var(--text-3)', cursor: 'pointer', fontFamily: 'var(--font-mono)' }}>
                      Retrieved chunks ({queryResult.retrieved_chunks.length})
                    </summary>
                    <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
                      {queryResult.retrieved_chunks.map((c, i) => (
                        <div key={i} style={{ background: 'var(--bg-3)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '10px 14px' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                            <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>{c.source} · chunk {c.chunk_index}</span>
                            <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>score {c.score}</span>
                          </div>
                          <p style={{ fontSize: 12, color: 'var(--text-2)', lineHeight: 1.6, margin: 0 }}>{c.text}</p>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            )}

            {!queryResult && !queryMut.isPending && docs.length === 0 && (
              <EmptyState
                icon={<FileText size={32} />}
                title="No documents indexed"
                description="Upload documents in the Documents tab, then ask questions about them here."
              />
            )}
          </div>
        </div>
      )}

      {/* ── DOCUMENTS TAB ── */}
      {tab === 'Documents' && (
        <Card style={{ padding: 0 }}>
          {uploadError && (
            <div style={{ padding: '10px 16px', background: 'var(--red-dim)', color: 'var(--red)', fontSize: 13, borderBottom: '1px solid var(--border)', display: 'flex', gap: 8 }}>
              <AlertCircle size={14} style={{ marginTop: 2 }} />{uploadError}
            </div>
          )}
          <div style={{
            display: 'grid', gridTemplateColumns: '1fr 100px 80px 80px 44px',
            padding: '8px 16px', borderBottom: '1px solid var(--border)',
            fontSize: 11, fontWeight: 600, letterSpacing: '0.06em',
            textTransform: 'uppercase', color: 'var(--text-3)',
            fontFamily: 'var(--font-mono)',
          }}>
            <span>Document</span><span>Chunks</span><span>Strategy</span><span>Status</span><span />
          </div>
          {isLoading && <div style={{ padding: 24, textAlign: 'center' }}><Spinner /></div>}
          {!isLoading && docs.length === 0 && (
            <EmptyState
              icon={<FileText size={32} />}
              title="No documents yet"
              description="Upload a PDF, DOCX, TXT, or Markdown file to get started."
              action={
                <Button variant="primary" onClick={() => fileRef.current?.click()}>
                  <Upload size={14} style={{ marginRight: 6 }} /> Upload document
                </Button>
              }
            />
          )}
          {docs.map(doc => (
            <DocumentRow
              key={doc.document_id}
              doc={doc}
              onDelete={(id) => deleteMut.mutate(id)}
              onSelect={setSelectedDocId}
              selected={selectedDocId === doc.document_id}
            />
          ))}
        </Card>
      )}

      {/* ── EVALUATE TAB ── */}
      {tab === 'Evaluate' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          <Card>
            <SectionLabel>Test Set</SectionLabel>
            <p style={{ fontSize: 13, color: 'var(--text-2)', marginBottom: 16, lineHeight: 1.6 }}>
              Upload a CSV with a <code style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>question</code> column
              (and optionally a <code style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>ground_truth</code> column).
              The pipeline will answer each question and score faithfulness, answer relevancy, and context precision.
              Requires <code style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>ANTHROPIC_API_KEY</code>.
            </p>
            <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 16 }}>
              <Button variant="ghost" onClick={() => evalFileRef.current?.click()}>
                <Upload size={14} style={{ marginRight: 6 }} />
                {evalCsv ? 'Replace CSV' : 'Upload test CSV'}
              </Button>
              <input ref={evalFileRef} type="file" accept=".csv" onChange={handleEvalCsv} style={{ display: 'none' }} />
              {evalCsv && (
                <span style={{ fontSize: 12, color: 'var(--green)', display: 'flex', gap: 6, alignItems: 'center' }}>
                  <CheckCircle2 size={13} />
                  {evalCsv.trim().split('\n').length - 1} questions loaded
                </span>
              )}
            </div>

            <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
              <select
                value={evalDocId}
                onChange={e => setEvalDocId(e.target.value)}
                style={{ background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 'var(--radius-sm)', padding: '6px 10px', fontSize: 13 }}
              >
                <option value="">All documents</option>
                {docs.filter(d => d.status === 'ready').map(d => (
                  <option key={d.document_id} value={d.document_id}>{d.name}</option>
                ))}
              </select>
              <Button
                variant="primary"
                onClick={runEvaluation}
                loading={evalMut.isPending}
                disabled={!evalCsv}
              >
                <Zap size={14} style={{ marginRight: 6 }} />
                Run evaluation
              </Button>
            </div>

            {evalMut.isPending && (
              <div style={{ marginTop: 16, fontSize: 13, color: 'var(--text-3)', display: 'flex', gap: 8, alignItems: 'center' }}>
                <Spinner size={14} />
                Evaluating… (each question makes 3 API calls — may take a minute)
              </div>
            )}
          </Card>

          {evalResult && (
            <>
              {/* Aggregate scores */}
              <div>
                <SectionLabel>Aggregate Scores</SectionLabel>
                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 8 }}>
                  <MetricBadge label="Faithfulness" value={evalResult.faithfulness_mean} description="Answer grounded in context" />
                  <MetricBadge label="Answer Relevancy" value={evalResult.answer_relevancy_mean} description="Answer addresses question" />
                  <MetricBadge label="Context Precision" value={evalResult.context_precision_mean} description="Retrieved chunks are useful" />
                  <MetricBadge label="Overall" value={evalResult.overall_mean} description="Harmonic mean of all three" />
                </div>
                <p style={{ marginTop: 12, fontSize: 12, color: 'var(--text-3)' }}>
                  {evalResult.n_samples} questions evaluated · {evalResult.n_errors} errors
                </p>
              </div>

              {/* Per-sample results */}
              <Card style={{ padding: 0 }}>
                <div style={{
                  display: 'grid', gridTemplateColumns: '2fr 90px 90px 90px',
                  padding: '8px 16px', borderBottom: '1px solid var(--border)',
                  fontSize: 11, fontWeight: 600, letterSpacing: '0.06em',
                  textTransform: 'uppercase', color: 'var(--text-3)', fontFamily: 'var(--font-mono)',
                }}>
                  <span>Question</span><span style={{ textAlign: 'right' }}>Faith.</span><span style={{ textAlign: 'right' }}>Relev.</span><span style={{ textAlign: 'right' }}>Prec.</span>
                </div>
                {evalResult.samples.map((s, i) => (
                  <div key={i} style={{
                    display: 'grid', gridTemplateColumns: '2fr 90px 90px 90px',
                    padding: '10px 16px', borderBottom: i < evalResult.samples.length - 1 ? '1px solid var(--border)' : 'none',
                    background: i % 2 === 0 ? 'transparent' : 'var(--bg-3)',
                    alignItems: 'center',
                  }}>
                    <div>
                      <div style={{ fontSize: 13, color: 'var(--text-1)' }}>{s.question}</div>
                      {s.error && <div style={{ fontSize: 11, color: 'var(--red)', marginTop: 2 }}>{s.error}</div>}
                    </div>
                    {['faithfulness', 'answer_relevancy', 'context_precision'].map(metric => {
                      const v = s[metric];
                      const pct = Math.round((v ?? 0) * 100);
                      const col = pct >= 80 ? 'var(--green)' : pct >= 60 ? 'var(--amber)' : 'var(--red)';
                      return (
                        <span key={metric} style={{ textAlign: 'right', fontFamily: 'var(--font-mono)', fontSize: 13, fontWeight: 600, color: col }}>
                          {pct}%
                        </span>
                      );
                    })}
                  </div>
                ))}
              </Card>
            </>
          )}
        </div>
      )}
    </div>
  );
}
