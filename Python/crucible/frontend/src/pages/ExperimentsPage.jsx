import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { FlaskConical, Play, Trash2, ChevronRight, Plus } from 'lucide-react';
import { datasets, connectors, experiments } from '../api/client.js'
import {
  PageHeader, StatusBadge, Button,
  Card, EmptyState, Spinner, SectionLabel,
} from '../components/ui.jsx';

async function createExperiment(body) {
  return experiments.create(body);
}

async function listExperiments(datasetId) {
  return experiments.list(datasetId);
}

async function deleteExperiment(id) {
  return experiments.delete(id);
}

function fmtScore(score, metric) {
  if (score == null) return '—';
  return `${(score * 100).toFixed(2)}%`;
}

function FamilyBadge({ family }) {
  const COLORS = {
    random_forest: 'var(--green)',
    gradient_boosting: 'var(--accent)',
    logistic_regression: 'var(--blue)',
    ridge: 'var(--blue)',
    svm: 'var(--amber)',
    knn: 'var(--text-2)',
  };
  const color = COLORS[family] || 'var(--text-2)';
  const labels = {
    random_forest: 'Random Forest',
    gradient_boosting: 'Gradient Boosting',
    logistic_regression: 'Logistic Reg.',
    ridge: 'Ridge',
    svm: 'SVM',
    knn: 'k-NN',
  };
  return (
    <span style={{
      fontSize: 12, fontFamily: 'var(--font-mono)',
      color, fontWeight: 500,
    }}>
      {labels[family] || family}
    </span>
  );
}

function CreateExperimentPanel({ datasetId, columns, onCreated, onCancel }) {
  const [form, setForm] = useState({
    name: `run_${Date.now().toString(36)}`,
    target_column: columns[0] || '',
    task_type: 'classification',
    n_trials: 20,
    cv_folds: 3,
    run_shap: true,
  });
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const inputStyle = {
    background: 'var(--bg-3)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    color: 'var(--text-1)',
    padding: '6px 10px',
    fontSize: 13,
    width: '100%',
  };

  async function submit() {
    setLoading(true);
    setError(null);
    try {
      await createExperiment({ ...form, dataset_id: datasetId });
      onCreated?.();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card style={{ padding: 20, marginBottom: 20 }}>
      <SectionLabel>New experiment</SectionLabel>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
        <div>
          <label style={{ fontSize: 12, color: 'var(--text-2)', display: 'block', marginBottom: 4 }}>Name</label>
          <input value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))} style={inputStyle} />
        </div>
        <div>
          <label style={{ fontSize: 12, color: 'var(--text-2)', display: 'block', marginBottom: 4 }}>Target column</label>
          <select value={form.target_column} onChange={e => setForm(f => ({ ...f, target_column: e.target.value }))} style={inputStyle}>
            {columns.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div>
          <label style={{ fontSize: 12, color: 'var(--text-2)', display: 'block', marginBottom: 4 }}>Task type</label>
          <select value={form.task_type} onChange={e => setForm(f => ({ ...f, task_type: e.target.value }))} style={inputStyle}>
            <option value="classification">Classification</option>
            <option value="regression">Regression</option>
          </select>
        </div>
        <div>
          <label style={{ fontSize: 12, color: 'var(--text-2)', display: 'block', marginBottom: 4 }}>Optuna trials</label>
          <input type="number" value={form.n_trials} min={2} max={100}
            onChange={e => setForm(f => ({ ...f, n_trials: Number(e.target.value) }))} style={inputStyle} />
        </div>
        <div>
          <label style={{ fontSize: 12, color: 'var(--text-2)', display: 'block', marginBottom: 4 }}>CV folds</label>
          <input type="number" value={form.cv_folds} min={2} max={10}
            onChange={e => setForm(f => ({ ...f, cv_folds: Number(e.target.value) }))} style={inputStyle} />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, paddingTop: 22 }}>
          <input type="checkbox" checked={form.run_shap}
            onChange={e => setForm(f => ({ ...f, run_shap: e.target.checked }))} id="shap" />
          <label htmlFor="shap" style={{ fontSize: 13, color: 'var(--text-2)', cursor: 'pointer' }}>
            Run SHAP after training
          </label>
        </div>
      </div>
      {error && <div style={{ color: 'var(--red)', fontSize: 12, marginBottom: 10 }}>{error}</div>}
      <div style={{ display: 'flex', gap: 8 }}>
        <Button onClick={submit} loading={loading}>
          <Play size={13} /> Start training
        </Button>
        <Button variant="ghost" onClick={onCancel}>Cancel</Button>
      </div>
    </Card>
  );
}

export default function ExperimentsPage({ datasetId: propDatasetId }) {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [showCreate, setShowCreate] = useState(false);

  // Accept datasetId as prop or from URL
  const { id: urlId } = useParams();
  const datasetId = propDatasetId || Number(urlId);

  const { data: dsData } = useQuery({
    queryKey: ['dataset', datasetId],
    queryFn: () => datasets.get(datasetId),
    enabled: !!datasetId,
  });

  const { data: expData, isLoading, refetch } = useQuery({
    queryKey: ['experiments', datasetId],
    queryFn: () => listExperiments(datasetId),
    enabled: !!datasetId,
    refetchInterval: (data) => {
      const hasRunning = data?.data?.some(e => e.status === 'running');
      return hasRunning ? 3000 : false;
    },
  });

  const cols = dsData?.data?.schema_columns?.map(c => c.name) ?? [];
  const exps = expData?.data ?? [];

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <SectionLabel>Experiments ({exps.length})</SectionLabel>
        <Button size="sm" onClick={() => setShowCreate(s => !s)}>
          <Plus size={12} /> New experiment
        </Button>
      </div>

      {showCreate && (
        <CreateExperimentPanel
          datasetId={datasetId}
          columns={cols}
          onCreated={() => { setShowCreate(false); refetch(); }}
          onCancel={() => setShowCreate(false)}
        />
      )}

      {isLoading ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: 40 }}>
          <Spinner size={20} />
        </div>
      ) : exps.length === 0 ? (
        <EmptyState
          icon={FlaskConical}
          title="No experiments yet"
          description="Start an AutoML training run to find the best model for your dataset."
          action={<Button size="sm" onClick={() => setShowCreate(true)}><Plus size={12} />New experiment</Button>}
        />
      ) : (
        <Card>
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 110px 140px 120px 90px 44px',
            padding: '8px 16px',
            borderBottom: '1px solid var(--border)',
            fontSize: 11, fontWeight: 600, letterSpacing: '0.06em',
            textTransform: 'uppercase', color: 'var(--text-3)',
            fontFamily: 'var(--font-mono)',
          }}>
            <span>Name</span><span>Status</span><span>Best model</span>
            <span>Score</span><span>Target</span><span />
          </div>

          {exps.map((exp, i) => (
            <div
              key={exp.id}
              onClick={() => navigate(`/experiments/${exp.id}`)}
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 110px 140px 120px 90px 44px',
                padding: '11px 16px',
                borderBottom: i < exps.length - 1 ? '1px solid var(--border)' : 'none',
                alignItems: 'center',
                cursor: 'pointer',
              }}
              onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-3)'}
              onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
            >
              <span style={{ fontWeight: 500, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}>
                {exp.name}
                {exp.lifecycle_stage && exp.lifecycle_stage !== 'candidate' && (
                  <span style={{
                    fontSize: 9, fontWeight: 700, padding: '1px 6px', borderRadius: 8,
                    fontFamily: 'var(--font-mono)', textTransform: 'uppercase',
                    background: exp.lifecycle_stage === 'production' ? 'rgba(46,204,113,0.15)' : 'rgba(127,140,141,0.15)',
                    color: exp.lifecycle_stage === 'production' ? 'var(--green)' : 'var(--text-3)',
                  }}>
                    {exp.lifecycle_stage}
                  </span>
                )}
              </span>
              <StatusBadge status={exp.status} />
              <span>{exp.best_model_family ? <FamilyBadge family={exp.best_model_family} /> : <span style={{ color: 'var(--text-3)', fontSize: 12 }}>—</span>}</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--accent)' }}>
                {fmtScore(exp.best_score, exp.scoring_metric)}
              </span>
              <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>
                {exp.target_column}
              </span>
              <button
                onClick={async e => {
                  e.stopPropagation();
                  if (confirm(`Delete "${exp.name}"?`)) {
                    await deleteExperiment(exp.id);
                    refetch();
                  }
                }}
                style={{ background: 'none', border: 'none', color: 'var(--text-3)', cursor: 'pointer', padding: 6, borderRadius: 'var(--radius-sm)' }}
                onMouseEnter={e => { e.currentTarget.style.background = 'var(--red-dim)'; e.currentTarget.style.color = 'var(--red)'; }}
                onMouseLeave={e => { e.currentTarget.style.background = 'none'; e.currentTarget.style.color = 'var(--text-3)'; }}
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </Card>
      )}
    </div>
  );
}
