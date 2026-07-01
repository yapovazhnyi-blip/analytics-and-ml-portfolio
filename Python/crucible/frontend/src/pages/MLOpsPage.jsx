import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Activity, Cloud, RefreshCw, Database, Search, CheckCircle2, XCircle,
  Clock, Zap, Server, Plus, Play, Trash2, ChevronRight, AlertTriangle,
  ArrowUpCircle, GitCommit,
} from 'lucide-react';
import { cloud, jobQueue, retraining, datasets } from '../api/client.js';
import { PageHeader, Card, Button, Spinner, SectionLabel, EmptyState, StatusBadge } from '../components/ui.jsx';

const TABS = ['Jobs', 'Cloud Training', 'Retraining Pipeline'];

export default function MLOpsPage() {
  const [tab, setTab] = useState('Jobs');

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 32 }}>
      <PageHeader
        title="MLOps"
        subtitle="Background jobs, cloud training submission, and automated retraining"
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

      {tab === 'Jobs' && <JobsTab />}
      {tab === 'Cloud Training' && <CloudTrainingTab />}
      {tab === 'Retraining Pipeline' && <RetrainingTab />}
    </div>
  );
}

// ── Jobs Tab ───────────────────────────────────────────────────────────────────

function JobsTab() {
  const [lookupId, setLookupId] = useState('');
  const [lookupResult, setLookupResult] = useState(null);
  const [lookupError, setLookupError] = useState('');

  const { data: recentJobs, isLoading: jobsLoading, refetch: refetchJobs } = useQuery({
    queryKey: ['jobs-recent'],
    queryFn: () => jobQueue.listRecent(50),
    refetchInterval: 5000,
  });

  const { data: cacheStats, refetch: refetchCache } = useQuery({
    queryKey: ['cache-stats'],
    queryFn: jobQueue.cacheStats,
    refetchInterval: 10000,
  });

  async function handleLookup() {
    setLookupError('');
    setLookupResult(null);
    try {
      const data = await jobQueue.getStatus(lookupId.trim());
      setLookupResult(data);
    } catch (err) {
      setLookupError(err.response?.data?.detail || 'Job not found.');
    }
  }

  const statusColor = {
    completed: 'var(--green)', running: 'var(--accent)',
    retrying: '#F39C12', failed: 'var(--red)', queued: 'var(--text-3)',
  };
  const statusIcon = {
    completed: CheckCircle2, running: Activity, retrying: RefreshCw,
    failed: XCircle, queued: Clock,
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Cache stats widget */}
      <Card>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
          <SectionLabel style={{ margin: 0 }}>Result cache (profiling & SHAP)</SectionLabel>
          <Button variant="ghost" onClick={() => refetchCache()}><RefreshCw size={13} /></Button>
        </div>
        {cacheStats ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
            {['profiling', 'shap'].map(key => {
              const s = cacheStats[key];
              if (!s) return null;
              return (
                <div key={key} style={{ background: 'var(--bg-3)', borderRadius: 8, padding: '12px 16px' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-3)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)', marginBottom: 6 }}>{key}</div>
                  <div style={{ display: 'flex', gap: 16, fontSize: 13 }}>
                    <span>hits: <strong style={{ color: 'var(--green)' }}>{s.hits}</strong></span>
                    <span>misses: <strong style={{ color: 'var(--text-3)' }}>{s.misses}</strong></span>
                    <span>hit rate: <strong style={{ color: 'var(--accent)' }}>{(s.hit_rate * 100).toFixed(0)}%</strong></span>
                    <span>size: <strong>{s.size}/{s.max_size}</strong></span>
                  </div>
                </div>
              );
            })}
          </div>
        ) : <Spinner />}
      </Card>

      {/* Job lookup */}
      <Card>
        <SectionLabel>Look up a job by ID</SectionLabel>
        <div style={{ display: 'flex', gap: 8, marginBottom: lookupResult || lookupError ? 14 : 0 }}>
          <input value={lookupId} onChange={e => setLookupId(e.target.value)}
            placeholder="e.g. a1b2c3d4e5f6g7h8"
            onKeyDown={e => e.key === 'Enter' && handleLookup()}
            style={{ flex: 1, background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 6, padding: '8px 12px', fontSize: 13, fontFamily: 'var(--font-mono)' }} />
          <Button variant="primary" onClick={handleLookup} disabled={!lookupId.trim()}>
            <Search size={13} style={{ marginRight: 5 }} /> Look up
          </Button>
        </div>
        {lookupError && <p style={{ fontSize: 12, color: 'var(--red)' }}>{lookupError}</p>}
        {lookupResult && (
          <pre style={{ background: 'var(--bg-3)', border: '1px solid var(--border)', borderRadius: 8, padding: 12, fontSize: 11, fontFamily: 'var(--font-mono)', overflow: 'auto' }}>
            {JSON.stringify(lookupResult, null, 2)}
          </pre>
        )}
      </Card>

      {/* Recent jobs (in-memory queue backend only) */}
      <Card>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
          <SectionLabel style={{ margin: 0 }}>Recent jobs</SectionLabel>
          <Button variant="ghost" onClick={() => refetchJobs()}><RefreshCw size={13} /></Button>
        </div>
        <p style={{ fontSize: 11, color: 'var(--text-3)', marginTop: -8, marginBottom: 12 }}>
          Empty when running with the ARQ/Redis job queue backend — ARQ doesn't expose a list-all-jobs API; look up individual jobs by ID above instead.
        </p>
        {jobsLoading && <Spinner />}
        {!jobsLoading && (!recentJobs || recentJobs.length === 0) && (
          <EmptyState icon={Activity} title="No recent jobs" description="Background jobs (e.g. RAG document indexing) will appear here." />
        )}
        {recentJobs && recentJobs.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {recentJobs.map(j => {
              const Icon = statusIcon[j.status] || Clock;
              return (
                <div key={j.job_id} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 12px', background: 'var(--bg-3)', borderRadius: 6, fontSize: 12 }}>
                  <Icon size={13} style={{ color: statusColor[j.status] || 'var(--text-3)', flexShrink: 0 }} />
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>{j.job_id}</span>
                  <span>{j.function_name}</span>
                  <span style={{ marginLeft: 'auto', color: statusColor[j.status] }}>{j.status}</span>
                  {j.elapsed_secs != null && <span style={{ color: 'var(--text-3)' }}>{j.elapsed_secs}s</span>}
                </div>
              );
            })}
          </div>
        )}
      </Card>
    </div>
  );
}

// ── Cloud Training Tab (SageMaker) ──────────────────────────────────────────────

function CloudTrainingTab() {
  const [form, setForm] = useState({
    dataset_id: '', target_column: '', task_type: 'classification',
    role_arn: 'mock-role', s3_bucket: 'demo-bucket', s3_prefix: 'crucible/training',
    region: 'us-east-1', instance_type: 'ml.m5.xlarge', experiment_name: 'sagemaker-job',
  });
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const { data: dsResp } = useQuery({ queryKey: ['datasets-for-sagemaker'], queryFn: () => datasets.list({ page_size: 100 }) });
  const { data: instanceTypes } = useQuery({ queryKey: ['sagemaker-instance-types'], queryFn: cloud.instanceTypes });

  const submitMut = useMutation({
    mutationFn: () => cloud.submitSageMaker({ ...form, dataset_id: parseInt(form.dataset_id, 10) }),
    onSuccess: (data) => { setResult(data); setError(''); },
    onError: (err) => setError(err.response?.data?.detail || 'Submission failed.'),
  });

  const datasetOptions = dsResp?.data || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <Card>
        <SectionLabel>Submit a SageMaker training job</SectionLabel>
        <p style={{ fontSize: 12, color: 'var(--text-3)', lineHeight: 1.7, marginBottom: 16 }}>
          Use <code style={{ fontFamily: 'var(--font-mono)', background: 'var(--bg-3)', padding: '1px 5px', borderRadius: 4 }}>role_arn: "mock-role"</code> to simulate a complete job in ~2 seconds without AWS credentials —
          useful for testing this flow. With real AWS credentials, set a real IAM role ARN to submit an actual SageMaker training job.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <div>
            <SectionLabel>Dataset</SectionLabel>
            <select value={form.dataset_id} onChange={e => setForm(f => ({ ...f, dataset_id: e.target.value }))}
              style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 6, padding: '7px 10px', fontSize: 13, boxSizing: 'border-box' }}>
              <option value="">Select a dataset…</option>
              {datasetOptions.map(d => <option key={d.id} value={d.id}>{d.name} (#{d.id})</option>)}
            </select>
          </div>
          <div>
            <SectionLabel>Target column</SectionLabel>
            <input value={form.target_column} onChange={e => setForm(f => ({ ...f, target_column: e.target.value }))}
              placeholder="label"
              style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 6, padding: '7px 10px', fontSize: 13, boxSizing: 'border-box' }} />
          </div>
          <div>
            <SectionLabel>Task type</SectionLabel>
            <select value={form.task_type} onChange={e => setForm(f => ({ ...f, task_type: e.target.value }))}
              style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 6, padding: '7px 10px', fontSize: 13, boxSizing: 'border-box' }}>
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
            </select>
          </div>
          <div>
            <SectionLabel>Instance type</SectionLabel>
            <select value={form.instance_type} onChange={e => setForm(f => ({ ...f, instance_type: e.target.value }))}
              style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 6, padding: '7px 10px', fontSize: 13, boxSizing: 'border-box' }}>
              {instanceTypes ? Object.entries(instanceTypes).map(([key, val]) => (
                <option key={key} value={val}>{key} — {val}</option>
              )) : <option value={form.instance_type}>{form.instance_type}</option>}
            </select>
          </div>
          <div>
            <SectionLabel>IAM role ARN</SectionLabel>
            <input value={form.role_arn} onChange={e => setForm(f => ({ ...f, role_arn: e.target.value }))}
              style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 6, padding: '7px 10px', fontSize: 13, fontFamily: 'var(--font-mono)', boxSizing: 'border-box' }} />
          </div>
          <div>
            <SectionLabel>S3 bucket</SectionLabel>
            <input value={form.s3_bucket} onChange={e => setForm(f => ({ ...f, s3_bucket: e.target.value }))}
              style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 6, padding: '7px 10px', fontSize: 13, fontFamily: 'var(--font-mono)', boxSizing: 'border-box' }} />
          </div>
        </div>

        <div style={{ marginTop: 14 }}>
          <Button variant="primary" onClick={() => submitMut.mutate()} loading={submitMut.isPending}
            disabled={!form.dataset_id || !form.target_column}>
            <Server size={13} style={{ marginRight: 5 }} /> Submit job
          </Button>
        </div>
        {error && <p style={{ fontSize: 12, color: 'var(--red)', marginTop: 10 }}>{error}</p>}
      </Card>

      {result && (
        <Card style={{ padding: '14px 18px' }}>
          <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', fontSize: 13 }}>
            <div><SectionLabel>Job name</SectionLabel><code style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>{result.job_name}</code></div>
            <div><SectionLabel>Status</SectionLabel><span style={{ color: 'var(--green)' }}>{result.status}</span></div>
            <div><SectionLabel>Instance</SectionLabel>{result.instance_type}</div>
            <div><SectionLabel>Training time</SectionLabel>{result.training_seconds}s</div>
          </div>
          {result.model_s3_uri && (
            <p style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 10, fontFamily: 'var(--font-mono)' }}>{result.model_s3_uri}</p>
          )}
        </Card>
      )}
    </div>
  );
}

// ── Retraining Pipeline Tab ──────────────────────────────────────────────────────

function RetrainingTab() {
  const qc = useQueryClient();
  const [showCreate, setShowCreate] = useState(false);
  const [selectedPolicy, setSelectedPolicy] = useState(null);

  const { data: policies, isLoading } = useQuery({
    queryKey: ['retraining-policies'],
    queryFn: retraining.listPolicies,
  });

  if (selectedPolicy) {
    return <PolicyDetail policy={selectedPolicy} onBack={() => setSelectedPolicy(null)} />;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <Card style={{ padding: '14px 18px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <p style={{ margin: 0, fontSize: 13, color: 'var(--text-3)' }}>
            Drift check → conditional retrain → promotion, automated. Borrows MLflow's candidate/production/archived vocabulary.
          </p>
          <Button variant="primary" onClick={() => setShowCreate(s => !s)}>
            <Plus size={13} style={{ marginRight: 5 }} /> New policy
          </Button>
        </div>
      </Card>

      {showCreate && <CreatePolicyForm onCreated={() => { setShowCreate(false); qc.invalidateQueries({ queryKey: ['retraining-policies'] }); }} />}

      {isLoading && <Spinner />}
      {!isLoading && (!policies || policies.length === 0) && (
        <EmptyState icon={GitCommit} title="No retraining policies yet"
          description="Create a policy to watch a dataset for drift and automatically retrain when it occurs." />
      )}

      {policies && policies.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {policies.map(p => (
            <Card key={p.id} style={{ padding: '14px 18px', cursor: 'pointer' }} onClick={() => setSelectedPolicy(p)}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: 14, fontWeight: 700 }}>{p.name}</span>
                    <span style={{ fontSize: 10, padding: '2px 7px', borderRadius: 10, background: p.is_active ? 'rgba(46,204,113,0.15)' : 'rgba(127,140,141,0.15)', color: p.is_active ? 'var(--green)' : 'var(--text-3)' }}>
                      {p.is_active ? 'active' : 'inactive'}
                    </span>
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginTop: 4 }}>
                    target: {p.target_column} · trigger: {p.drift_severity_trigger} · margin: +{p.promotion_margin}
                    {p.check_interval_hours && ` · every ${p.check_interval_hours}h`}
                  </div>
                </div>
                <ChevronRight size={16} style={{ color: 'var(--text-3)' }} />
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}

function CreatePolicyForm({ onCreated }) {
  const [form, setForm] = useState({
    name: '', reference_dataset_id: '', target_column: '', task_type: 'classification',
    drift_severity_trigger: 'significant', promotion_margin: 0.02, n_trials: 15, cv_folds: 3,
    check_interval_hours: '',
  });
  const [error, setError] = useState('');
  const { data: dsResp } = useQuery({ queryKey: ['datasets-for-policy'], queryFn: () => datasets.list({ page_size: 100 }) });

  const createMut = useMutation({
    mutationFn: () => retraining.createPolicy({
      ...form,
      reference_dataset_id: parseInt(form.reference_dataset_id, 10),
      promotion_margin: parseFloat(form.promotion_margin),
      n_trials: parseInt(form.n_trials, 10),
      cv_folds: parseInt(form.cv_folds, 10),
      check_interval_hours: form.check_interval_hours ? parseFloat(form.check_interval_hours) : null,
    }),
    onSuccess: onCreated,
    onError: (err) => setError(err.response?.data?.detail || 'Failed to create policy.'),
  });

  const datasetOptions = dsResp?.data || [];

  return (
    <Card>
      <SectionLabel>New policy</SectionLabel>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 8 }}>
        <div>
          <SectionLabel>Name</SectionLabel>
          <input value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
            style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 6, padding: '7px 10px', fontSize: 13, boxSizing: 'border-box' }} />
        </div>
        <div>
          <SectionLabel>Reference dataset (drift baseline)</SectionLabel>
          <select value={form.reference_dataset_id} onChange={e => setForm(f => ({ ...f, reference_dataset_id: e.target.value }))}
            style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 6, padding: '7px 10px', fontSize: 13, boxSizing: 'border-box' }}>
            <option value="">Select…</option>
            {datasetOptions.map(d => <option key={d.id} value={d.id}>{d.name} (#{d.id})</option>)}
          </select>
        </div>
        <div>
          <SectionLabel>Target column</SectionLabel>
          <input value={form.target_column} onChange={e => setForm(f => ({ ...f, target_column: e.target.value }))}
            style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 6, padding: '7px 10px', fontSize: 13, boxSizing: 'border-box' }} />
        </div>
        <div>
          <SectionLabel>Task type</SectionLabel>
          <select value={form.task_type} onChange={e => setForm(f => ({ ...f, task_type: e.target.value }))}
            style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 6, padding: '7px 10px', fontSize: 13, boxSizing: 'border-box' }}>
            <option value="classification">Classification</option>
            <option value="regression">Regression</option>
          </select>
        </div>
        <div>
          <SectionLabel>Drift trigger severity</SectionLabel>
          <select value={form.drift_severity_trigger} onChange={e => setForm(f => ({ ...f, drift_severity_trigger: e.target.value }))}
            style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 6, padding: '7px 10px', fontSize: 13, boxSizing: 'border-box' }}>
            <option value="slight">Slight</option>
            <option value="significant">Significant</option>
            <option value="critical">Critical</option>
          </select>
        </div>
        <div>
          <SectionLabel>Promotion margin</SectionLabel>
          <input type="number" step="0.01" value={form.promotion_margin} onChange={e => setForm(f => ({ ...f, promotion_margin: e.target.value }))}
            style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 6, padding: '7px 10px', fontSize: 13, boxSizing: 'border-box' }} />
        </div>
        <div>
          <SectionLabel>Check interval (hours, optional)</SectionLabel>
          <input type="number" value={form.check_interval_hours} onChange={e => setForm(f => ({ ...f, check_interval_hours: e.target.value }))}
            placeholder="leave blank for manual-trigger only"
            style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 6, padding: '7px 10px', fontSize: 13, boxSizing: 'border-box' }} />
        </div>
      </div>
      <div style={{ marginTop: 14 }}>
        <Button variant="primary" onClick={() => createMut.mutate()} loading={createMut.isPending}
          disabled={!form.name || !form.reference_dataset_id || !form.target_column}>
          Create policy
        </Button>
      </div>
      {error && <p style={{ fontSize: 12, color: 'var(--red)', marginTop: 8 }}>{error}</p>}
    </Card>
  );
}

function PolicyDetail({ policy, onBack }) {
  const qc = useQueryClient();
  const [runResult, setRunResult] = useState(null);

  const { data: runs, isLoading } = useQuery({
    queryKey: ['policy-runs', policy.id],
    queryFn: () => retraining.listRuns(policy.id),
  });

  const runMut = useMutation({
    mutationFn: () => retraining.runPolicy(policy.id),
    onSuccess: (data) => {
      setRunResult(data);
      qc.invalidateQueries({ queryKey: ['policy-runs', policy.id] });
    },
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <button onClick={onBack} style={{ background: 'none', border: 'none', color: 'var(--accent)', cursor: 'pointer', fontSize: 13, textAlign: 'left', padding: 0 }}>
        ← Back to policies
      </button>

      <Card style={{ padding: '14px 18px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <span style={{ fontSize: 16, fontWeight: 700 }}>{policy.name}</span>
            <p style={{ fontSize: 12, color: 'var(--text-3)', margin: '4px 0 0' }}>{policy.description}</p>
          </div>
          <Button variant="primary" onClick={() => runMut.mutate()} loading={runMut.isPending}>
            <Play size={13} style={{ marginRight: 5 }} /> Run now
          </Button>
        </div>
      </Card>

      {runResult && (
        <Card style={{ padding: '14px 18px' }}>
          <SectionLabel>Latest run result</SectionLabel>
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 12, marginTop: 8 }}>
            <span>drift detected: <strong style={{ color: runResult.drift_detected ? '#F39C12' : 'var(--green)' }}>{String(runResult.drift_detected)}</strong></span>
            <span>retrain triggered: <strong>{String(runResult.retrain_triggered)}</strong></span>
            {runResult.retrain_triggered && <span>promoted: <strong style={{ color: runResult.promoted ? 'var(--green)' : 'var(--text-3)' }}>{String(runResult.promoted)}</strong></span>}
          </div>
          {runResult.promotion_reason && <p style={{ fontSize: 12, color: 'var(--text-3)', marginTop: 8 }}>{runResult.promotion_reason}</p>}
          {runResult.steps && (
            <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 4 }}>
              {runResult.steps.map((s, i) => (
                <div key={i} style={{ display: 'flex', gap: 8, fontSize: 11, fontFamily: 'var(--font-mono)' }}>
                  <span style={{ color: s.status === 'completed' ? 'var(--green)' : s.status === 'failed' ? 'var(--red)' : 'var(--text-3)' }}>
                    {s.status === 'completed' ? '✓' : s.status === 'failed' ? '✗' : '○'}
                  </span>
                  <span>{s.step}</span>
                  <span style={{ color: 'var(--text-3)' }}>{s.detail}</span>
                </div>
              ))}
            </div>
          )}
        </Card>
      )}

      <Card>
        <SectionLabel>Run history</SectionLabel>
        {isLoading && <Spinner />}
        {runs && runs.length === 0 && <p style={{ fontSize: 12, color: 'var(--text-3)' }}>No runs yet.</p>}
        {runs && runs.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 8 }}>
            {runs.map(r => (
              <div key={r.id} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 12px', background: 'var(--bg-3)', borderRadius: 6, fontSize: 12 }}>
                {r.status === 'completed' ? <CheckCircle2 size={13} style={{ color: 'var(--green)' }} /> :
                 r.status === 'failed' ? <XCircle size={13} style={{ color: 'var(--red)' }} /> :
                 <Clock size={13} style={{ color: 'var(--text-3)' }} />}
                <span>{new Date(r.created_at).toLocaleString()}</span>
                {r.drift_detected && <AlertTriangle size={12} style={{ color: '#F39C12' }} />}
                {r.promoted && <ArrowUpCircle size={12} style={{ color: 'var(--green)' }} />}
                <span style={{ marginLeft: 'auto', color: 'var(--text-3)' }}>{r.elapsed_secs?.toFixed(1)}s</span>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
