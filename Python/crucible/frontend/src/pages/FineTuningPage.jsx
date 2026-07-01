import { useState, useEffect, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Cpu, Plus, Trash2, ChevronDown, ChevronUp, CheckCircle2, AlertCircle, Loader2, Zap } from 'lucide-react';
import { fineTuning } from '../api/client.js';
import { PageHeader, Card, Button, Spinner, SectionLabel, StatusBadge, EmptyState } from '../components/ui.jsx';

// ── Loss curve mini-chart ─────────────────────────────────────────────────────

function LossCurve({ points }) {
  if (!points.length) return null;
  const W = 260, H = 60, PAD = 4;
  const losses = points.map(p => p.loss);
  const minL = Math.min(...losses), maxL = Math.max(...losses);
  const range = maxL - minL || 1;
  const toX = i => PAD + (i / Math.max(points.length - 1, 1)) * (W - PAD * 2);
  const toY = v => PAD + (1 - (v - minL) / range) * (H - PAD * 2);
  const d = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(p.loss).toFixed(1)}`).join(' ');
  return (
    <svg width={W} height={H} style={{ overflow: 'visible' }}>
      <path d={d} fill="none" stroke="var(--accent)" strokeWidth="1.5" strokeLinejoin="round" />
      {points.length > 1 && (
        <circle cx={toX(points.length - 1)} cy={toY(losses[losses.length - 1])} r="3" fill="var(--accent)" />
      )}
    </svg>
  );
}

// ── Job row ───────────────────────────────────────────────────────────────────

function JobRow({ job, onDelete, onSelect, selected }) {
  const elapsed = job.elapsed_secs ? `${Math.round(job.elapsed_secs)}s` : '—';
  return (
    <div onClick={() => onSelect(job.job_id)}
      style={{
        display: 'grid', gridTemplateColumns: '2fr 120px 80px 80px 60px 44px',
        padding: '12px 16px', alignItems: 'center', cursor: 'pointer',
        borderBottom: '1px solid var(--border)',
        background: selected ? 'var(--bg-3)' : 'transparent',
      }}
      onMouseEnter={e => { if (!selected) e.currentTarget.style.background = 'var(--bg-3)'; }}
      onMouseLeave={e => { if (!selected) e.currentTarget.style.background = 'transparent'; }}
    >
      <div>
        <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-1)', fontFamily: 'var(--font-mono)' }}>
          {job.base_model_id}
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>
          {job.n_samples} samples · {job.epochs} epoch{job.epochs !== 1 ? 's' : ''} · {job.dataset_format}
        </div>
      </div>
      <StatusBadge status={job.status} />
      <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-2)' }}>
        {job.final_loss != null ? job.final_loss.toFixed(4) : '—'}
      </span>
      <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-2)' }}>
        {elapsed}
      </span>
      <span style={{ fontSize: 11, color: 'var(--text-3)' }}>{job.total_steps ?? '—'} steps</span>
      <button onClick={e => { e.stopPropagation(); if (confirm('Delete job?')) onDelete(job.job_id); }}
        style={{ background: 'none', border: 'none', color: 'var(--text-3)', cursor: 'pointer', padding: 6, display: 'flex' }}
        onMouseEnter={e => e.currentTarget.style.color = 'var(--red)'}
        onMouseLeave={e => e.currentTarget.style.color = 'var(--text-3)'}>
        <Trash2 size={13} />
      </button>
    </div>
  );
}

// ── Progress panel ────────────────────────────────────────────────────────────

function TrainingProgress({ jobId }) {
  const [steps, setSteps] = useState([]);
  const [status, setStatus] = useState('connecting');
  const wsRef = useRef(null);

  useEffect(() => {
    if (!jobId) return;
    setSteps([]);
    setStatus('connecting');

    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const token = localStorage.getItem('crucible_access_token');
    const url = `${proto}//localhost:8000/ws/fine-tuning/${jobId}${token ? `?token=${token}` : ''}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen  = () => setStatus('running');
    ws.onclose = () => setStatus(prev => prev === 'complete' ? 'complete' : 'disconnected');
    ws.onerror = () => setStatus('error');

    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === 'step') {
        setSteps(prev => [...prev, { step: msg.step, loss: msg.loss, lr: msg.lr, epoch: msg.epoch }]);
      } else if (msg.type === 'complete') {
        setStatus('complete');
      } else if (msg.type === 'error') {
        setStatus('error');
      }
    };

    return () => ws.close();
  }, [jobId]);

  if (!jobId) return null;

  const last = steps[steps.length - 1];
  return (
    <Card>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <SectionLabel>Training Progress</SectionLabel>
        <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color:
          status === 'complete' ? 'var(--green)' : status === 'error' ? 'var(--red)' : 'var(--accent)' }}>
          {status === 'connecting' ? '⋯ connecting' : status === 'running' ? '⚡ live' :
           status === 'complete' ? '✓ complete' : status === 'error' ? '✗ error' : '○ disconnected'}
        </span>
      </div>

      {steps.length > 0 && (
        <>
          <div style={{ display: 'flex', gap: 24, marginBottom: 14 }}>
            {[
              ['Step', `${last.step}`],
              ['Loss', last.loss.toFixed(4)],
              ['LR', last.lr.toExponential(2)],
              ['Epoch', last.epoch.toFixed(2)],
            ].map(([label, val]) => (
              <div key={label}>
                <div style={{ fontSize: 10, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 2 }}>{label}</div>
                <div style={{ fontSize: 15, fontWeight: 600, fontFamily: 'var(--font-mono)', color: 'var(--text-1)' }}>{val}</div>
              </div>
            ))}
          </div>
          <LossCurve points={steps} />
        </>
      )}

      {steps.length === 0 && status === 'running' && (
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', color: 'var(--text-3)', fontSize: 13 }}>
          <Spinner size={14} /> Waiting for first step…
        </div>
      )}
    </Card>
  );
}

// ── Submit form ───────────────────────────────────────────────────────────────

const EXAMPLE_ALPACA = JSON.stringify([
  { instruction: "Summarise this in one sentence.", input: "The quick brown fox jumps over the lazy dog.", output: "A fox jumped over a dog." },
  { instruction: "What is the capital of France?", input: "", output: "Paris." },
  { instruction: "Translate to Spanish.", input: "Good morning.", output: "Buenos días." },
], null, 2);

const EXAMPLE_SHAREGPT = JSON.stringify([
  { conversations: [{ from: "human", value: "What is machine learning?" }, { from: "gpt", value: "Machine learning is a field of AI where systems learn from data." }] },
  { conversations: [{ from: "human", value: "Name a programming language." }, { from: "gpt", value: "Python is a popular programming language." }] },
], null, 2);

function SubmitForm({ onSubmit, loading }) {
  const [modelId, setModelId] = useState('mock-phi');
  const [format, setFormat] = useState('alpaca');
  const [samplesText, setSamplesText] = useState(EXAMPLE_ALPACA);
  const [rank, setRank] = useState(16);
  const [alpha, setAlpha] = useState(32);
  const [epochs, setEpochs] = useState(1);
  const [lr, setLr] = useState(0.0002);
  const [batchSize, setBatchSize] = useState(4);
  const [parseError, setParseError] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);

  function handleFormatChange(f) {
    setFormat(f);
    setSamplesText(f === 'alpaca' ? EXAMPLE_ALPACA : EXAMPLE_SHAREGPT);
  }

  function handleSubmit() {
    setParseError('');
    let samples;
    try {
      samples = JSON.parse(samplesText);
      if (!Array.isArray(samples)) throw new Error('Must be a JSON array');
    } catch (e) {
      setParseError(`Invalid JSON: ${e.message}`);
      return;
    }
    onSubmit({ model_id: modelId, samples, dataset_format: format, lora: { rank, alpha, dropout: 0.05, target_modules: ['q_proj', 'v_proj', 'k_proj', 'o_proj'] }, epochs, learning_rate: lr, batch_size: batchSize });
  }

  const inp = (val, set, type = 'text', rest = {}) => (
    <input type={type} value={val} onChange={e => set(type === 'number' ? Number(e.target.value) : e.target.value)} {...rest}
      style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 'var(--radius-sm)', padding: '7px 10px', fontSize: 13, boxSizing: 'border-box' }} />
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 12 }}>
        <div>
          <SectionLabel>Base model ID</SectionLabel>
          {inp(modelId, setModelId, 'text', { placeholder: 'microsoft/phi-2  or  mock-phi (test)' })}
          <p style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 4 }}>
            HuggingFace model hub path. Use <code style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>mock-phi</code> to test without downloading.
          </p>
        </div>
        <div>
          <SectionLabel>Dataset format</SectionLabel>
          <select value={format} onChange={e => handleFormatChange(e.target.value)}
            style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 'var(--radius-sm)', padding: '7px 10px', fontSize: 13 }}>
            <option value="alpaca">Alpaca (instruction / output)</option>
            <option value="sharegpt">ShareGPT (conversations)</option>
          </select>
        </div>
      </div>

      <div>
        <SectionLabel>Training samples (JSON array)</SectionLabel>
        <textarea value={samplesText} onChange={e => setSamplesText(e.target.value)} rows={8}
          style={{ width: '100%', background: 'var(--bg-3)', border: `1px solid ${parseError ? 'var(--red)' : 'var(--border)'}`, color: 'var(--text-1)', borderRadius: 'var(--radius-sm)', padding: '10px 14px', fontSize: 12, fontFamily: 'var(--font-mono)', resize: 'vertical', boxSizing: 'border-box', lineHeight: 1.5 }} />
        {parseError && <p style={{ fontSize: 12, color: 'var(--red)', marginTop: 4 }}>{parseError}</p>}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
        <div><SectionLabel>Epochs</SectionLabel>{inp(epochs, setEpochs, 'number', { min: 1, max: 100 })}</div>
        <div><SectionLabel>Learning rate</SectionLabel>{inp(lr, setLr, 'number', { min: 0.000001, max: 0.1, step: 0.0001 })}</div>
        <div><SectionLabel>Batch size</SectionLabel>{inp(batchSize, setBatchSize, 'number', { min: 1, max: 64 })}</div>
      </div>

      <button onClick={() => setShowAdvanced(!showAdvanced)}
        style={{ background: 'none', border: 'none', color: 'var(--text-3)', cursor: 'pointer', fontSize: 12, display: 'flex', alignItems: 'center', gap: 6, padding: 0 }}>
        {showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />} LoRA hyperparameters
      </button>

      {showAdvanced && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, padding: '14px', background: 'var(--bg-3)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
          <div>
            <SectionLabel>Rank (r)</SectionLabel>
            {inp(rank, setRank, 'number', { min: 1, max: 256 })}
            <p style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 3 }}>Higher = more expressive but more memory. Default: 16</p>
          </div>
          <div>
            <SectionLabel>Alpha (α)</SectionLabel>
            {inp(alpha, setAlpha, 'number', { min: 1 })}
            <p style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 3 }}>Scaling = α/r. Convention: 2× rank. Default: 32</p>
          </div>
        </div>
      )}

      <Button variant="primary" onClick={handleSubmit} loading={loading}>
        <Zap size={14} style={{ marginRight: 6 }} /> Start fine-tuning
      </Button>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

const TABS = ['Jobs', 'New Job (SFT)', 'New Job (DPO)'];

// ── DPO Form ───────────────────────────────────────────────────────────────

function DPOForm() {
  const [modelId, setModelId]     = useState('mock-phi');
  const [beta, setBeta]           = useState(0.1);
  const [epochs, setEpochs]       = useState(1);
  const [batchSize, setBatchSize] = useState(2);
  const [lr, setLr]               = useState(5e-5);
  const [rawJson, setRawJson]     = useState(
    JSON.stringify([
      { prompt: 'Explain gradient descent.', chosen: 'Gradient descent minimises a loss function by iteratively moving in the direction of steepest descent.', rejected: 'It makes the model smarter.' },
      { prompt: 'What is overfitting?', chosen: 'Overfitting occurs when a model learns the training data too well, including noise, and fails to generalise.', rejected: 'The model is too good.' },
    ], null, 2)
  );
  const [result, setResult]   = useState(null);
  const [error, setError]     = useState('');
  const [loading, setLoading] = useState(false);

  async function handleSubmit() {
    setError(''); setResult(null);
    let samples;
    try { samples = JSON.parse(rawJson); } catch { setError('Invalid JSON.'); return; }
    if (!Array.isArray(samples)) { setError('Samples must be a JSON array.'); return; }
    setLoading(true);
    try {
      const { dpoSubmit } = await import('../api/client.js');
      const data = await dpoSubmit({ model_id: modelId, samples, beta, epochs, batch_size: batchSize, learning_rate: lr });
      setResult(data);
    } catch (e) {
      const detail = e.response?.data?.detail;
      setError(detail?.errors?.join('; ') || detail || 'Submission failed');
    } finally { setLoading(false); }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <Card>
        <SectionLabel>DPO — Direct Preference Optimisation</SectionLabel>
        <p style={{ fontSize: 12, color: 'var(--text-3)', lineHeight: 1.7, marginBottom: 16 }}>
          DPO trains the model to prefer <strong style={{ color: 'var(--green)' }}>chosen</strong> responses over <strong style={{ color: 'var(--red)' }}>rejected</strong> ones, without a reward model.
          Each sample must have <code style={{ fontFamily: 'var(--font-mono)', background: 'var(--bg-3)', padding: '1px 5px', borderRadius: 4 }}>prompt</code>,{' '}
          <code style={{ fontFamily: 'var(--font-mono)', background: 'var(--bg-3)', padding: '1px 5px', borderRadius: 4 }}>chosen</code>, and{' '}
          <code style={{ fontFamily: 'var(--font-mono)', background: 'var(--bg-3)', padding: '1px 5px', borderRadius: 4 }}>rejected</code> fields.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
          {[
            ['Model ID', modelId, setModelId, 'text', 'mock-phi'],
            ['Beta (KL penalty)', beta, setBeta, 'number', 0.1],
            ['Epochs', epochs, setEpochs, 'number', 1],
            ['Batch size', batchSize, setBatchSize, 'number', 2],
            ['Learning rate', lr, setLr, 'number', 5e-5],
          ].map(([label, val, setter, type, placeholder]) => (
            <div key={label}>
              <SectionLabel>{label}</SectionLabel>
              <input type={type} value={val} placeholder={String(placeholder)}
                onChange={e => setter(type === 'number' ? parseFloat(e.target.value) : e.target.value)}
                style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 6, padding: '7px 10px', fontSize: 13, boxSizing: 'border-box' }} />
            </div>
          ))}
        </div>

        <SectionLabel>Preference samples (JSON array)</SectionLabel>
        <textarea value={rawJson} onChange={e => setRawJson(e.target.value)} rows={12}
          style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 6, padding: '10px 12px', fontSize: 12, fontFamily: 'var(--font-mono)', resize: 'vertical', boxSizing: 'border-box' }} />

        <div style={{ marginTop: 14, display: 'flex', gap: 10 }}>
          <Button variant="primary" onClick={handleSubmit} loading={loading}>Submit DPO job</Button>
        </div>
        {error && <div style={{ marginTop: 12, padding: '10px 14px', background: 'rgba(231,76,60,0.1)', borderRadius: 8, color: 'var(--red)', fontSize: 13 }}>{error}</div>}
      </Card>

      {result && (
        <Card style={{ padding: '14px 16px' }}>
          <div style={{ display: 'flex', gap: 16 }}>
            <div><SectionLabel>Job ID</SectionLabel><code style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--accent)' }}>{result.job_id}</code></div>
            <div><SectionLabel>Method</SectionLabel><span style={{ fontSize: 13, color: 'var(--text-1)', textTransform: 'uppercase', fontWeight: 700 }}>{result.method}</span></div>
            <div><SectionLabel>Status</SectionLabel><span style={{ fontSize: 13, color: 'var(--green)' }}>{result.status}</span></div>
            <div><SectionLabel>Samples</SectionLabel><span style={{ fontSize: 13, color: 'var(--text-1)' }}>{result.n_samples}</span></div>
          </div>
          <p style={{ fontSize: 12, color: 'var(--text-3)', marginTop: 8 }}>
            Switch to the Jobs tab to monitor progress.
          </p>
        </Card>
      )}
    </div>
  );
}

export default function FineTuningPage() {
  const qc = useQueryClient();
  const [tab, setTab] = useState('Jobs');
  const [selectedJobId, setSelectedJobId] = useState(null);

  const { data: jobsData, isLoading } = useQuery({
    queryKey: ['fine-tuning-jobs'],
    queryFn: () => fineTuning.list(),
    refetchInterval: 3000,
  });

  const jobs = jobsData?.data ?? [];

  const submitMut = useMutation({
    mutationFn: (payload) => fineTuning.submit(payload),
    onSuccess: (job) => {
      qc.invalidateQueries({ queryKey: ['fine-tuning-jobs'] });
      setSelectedJobId(job.job_id);
      setTab('Jobs');
    },
  });

  const deleteMut = useMutation({
    mutationFn: (jobId) => fineTuning.delete(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['fine-tuning-jobs'] });
      setSelectedJobId(null);
    },
  });

  const selectedJob = jobs.find(j => j.job_id === selectedJobId);

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 32 }}>
      <PageHeader
        title="Fine-Tuning Studio"
        subtitle={`${jobs.length} job${jobs.length !== 1 ? 's' : ''} · LoRA / QLoRA · SFT`}
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

      {/* ── JOBS TAB ── */}
      {tab === 'Jobs' && (
        <div style={{ display: 'grid', gridTemplateColumns: selectedJobId ? '1fr 380px' : '1fr', gap: 20 }}>
          <div>
            <Card style={{ padding: 0 }}>
              <div style={{
                display: 'grid', gridTemplateColumns: '2fr 120px 80px 80px 60px 44px',
                padding: '8px 16px', borderBottom: '1px solid var(--border)',
                fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-3)',
                textTransform: 'uppercase', letterSpacing: '0.06em',
              }}>
                <span>Model</span><span>Status</span><span>Loss</span><span>Time</span><span>Steps</span><span />
              </div>

              {isLoading && <div style={{ padding: 24, textAlign: 'center' }}><Spinner /></div>}

              {!isLoading && jobs.length === 0 && (
                <EmptyState
                  icon={<Cpu size={32} />}
                  title="No fine-tuning jobs yet"
                  description="Use the New Job tab to start training a LoRA adapter."
                  action={<Button variant="primary" onClick={() => setTab('New Job')}><Plus size={14} style={{ marginRight: 6 }} />New job</Button>}
                />
              )}

              {jobs.map(j => (
                <JobRow key={j.job_id} job={j}
                  selected={j.job_id === selectedJobId}
                  onSelect={id => setSelectedJobId(id === selectedJobId ? null : id)}
                  onDelete={id => deleteMut.mutate(id)} />
              ))}
            </Card>
          </div>

          {selectedJobId && selectedJob && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <TrainingProgress jobId={selectedJob.status === 'running' ? selectedJobId : null} />

              {selectedJob.status !== 'running' && (
                <Card>
                  <SectionLabel>Job Details</SectionLabel>
                  {[
                    ['Job ID',     selectedJob.job_id],
                    ['Model',      selectedJob.base_model_id],
                    ['Status',     selectedJob.status],
                    ['Samples',    selectedJob.n_samples],
                    ['Epochs',     selectedJob.epochs],
                    ['Final loss', selectedJob.final_loss?.toFixed(4) ?? '—'],
                    ['Total steps', selectedJob.total_steps ?? '—'],
                    ['Elapsed',    selectedJob.elapsed_secs ? `${Math.round(selectedJob.elapsed_secs)}s` : '—'],
                  ].map(([k, v]) => (
                    <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 0', borderBottom: '1px solid var(--border)', fontSize: 12 }}>
                      <span style={{ color: 'var(--text-3)' }}>{k}</span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-1)' }}>{v}</span>
                    </div>
                  ))}
                  {selectedJob.error_message && (
                    <div style={{ marginTop: 10, padding: '8px 12px', background: 'rgba(231,76,60,0.1)', borderRadius: 6, fontSize: 12, color: 'var(--red)' }}>
                      {selectedJob.error_message}
                    </div>
                  )}
                  {selectedJob.hub_url && (
                    <a href={selectedJob.hub_url} target="_blank" rel="noopener noreferrer"
                      style={{ display: 'block', marginTop: 10, fontSize: 12, color: 'var(--accent)' }}>
                      View on HuggingFace Hub ↗
                    </a>
                  )}
                </Card>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── NEW JOB (SFT) TAB ── */}
      {tab === 'New Job (SFT)' && (
        <Card>
          <SubmitForm onSubmit={payload => submitMut.mutate(payload)} loading={submitMut.isPending} />
          {submitMut.isError && (
            <div style={{ marginTop: 14, padding: '10px 14px', background: 'rgba(231,76,60,0.1)', borderRadius: 8, color: 'var(--red)', fontSize: 13 }}>
              {submitMut.error?.response?.data?.detail?.errors?.join('; ') ||
               submitMut.error?.response?.data?.detail ||
               'Submission failed'}
            </div>
          )}
        </Card>
      )}

      {/* ── NEW JOB (DPO) TAB ── */}
      {tab === 'New Job (DPO)' && <DPOForm />}
    </div>
  );
}
