import axios from 'axios';

const api = axios.create({
  baseURL: '/api/v1',
  headers: { 'Content-Type': 'application/json' },
});

// Unwrap the consistent { data, meta } envelope
const unwrap = (r) => r.data;

// ── Datasets ──────────────────────────────────────────────────────────────

export const datasets = {
  list: (params = {}) =>
    api.get('/datasets', { params }).then(unwrap),

  get: (id) =>
    api.get(`/datasets/${id}`).then(unwrap),

  upload: (file, name) => {
    const form = new FormData();
    form.append('file', file);
    if (name) form.append('name', name);
    return api.post('/datasets/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then(unwrap);
  },

  fromSql: (body) =>
    api.post('/datasets/from-sql', body).then(unwrap),

  delete: (id) =>
    api.delete(`/datasets/${id}`),

  profile: (id, body = {}) =>
    api.post(`/datasets/${id}/profile`, body).then(unwrap),

  advise: (id, body = {}) =>
    api.post(`/datasets/${id}/profile/advise`, body).then(unwrap),
};

// ── Connectors ────────────────────────────────────────────────────────────

export const connectors = {
  list: () =>
    api.get('/connectors').then(unwrap),

  get: (id) =>
    api.get(`/connectors/${id}`).then(unwrap),

  createSql: (body) =>
    api.post('/connectors/sql', body).then(unwrap),

  createOAuth: (body) =>
    api.post('/connectors/oauth', body).then(unwrap),

  // One-shot ingest: runs a query against BigQuery and creates a dataset
  // directly (unlike createSql/createOAuth, this does not save a reusable
  // connector record — it returns the new dataset).
  ingestBigQuery: (body) =>
    api.post('/connectors/bigquery', body).then(unwrap),
};

// ── Experiments ───────────────────────────────────────────────────────────
export const experiments = {
  create: (body) => api.post('/experiments', body).then(unwrap),
  get: (id) => api.get(`/experiments/${id}`).then(unwrap),
  list: (datasetId) =>
    api.get('/experiments', { params: { dataset_id: datasetId, page_size: 50 } }).then(unwrap),
  delete: (id) => api.delete(`/experiments/${id}`).then(unwrap),

  test: (id) =>
    api.post(`/connectors/${id}/test`).then(unwrap),

  // Returns the URL to open in a popup/new tab — the backend redirects
  // the user to the provider's consent page. Not a fetch — just a URL builder.
  authorizeUrl: (id, redirectUri) => {
    const base = `/api/v1/connectors/${id}/authorize`;
    return redirectUri ? `${base}?redirect_uri=${encodeURIComponent(redirectUri)}` : base;
  },

  delete: (id) =>
    api.delete(`/connectors/${id}`),
};

// ── RAG ───────────────────────────────────────────────────────────────────

export const rag = {
  listDocuments: (page = 1, pageSize = 20) =>
    api.get(`/rag/documents?page=${page}&page_size=${pageSize}`).then(r => r.data),

  getDocument: (documentId) =>
    api.get(`/rag/documents/${documentId}`).then(unwrap),

  uploadDocument: (file, name = '', chunkStrategy = 'paragraph', datasetId = null) => {
    const form = new FormData();
    form.append('file', file);
    form.append('name', name || '');
    form.append('chunk_strategy', chunkStrategy);
    if (datasetId) form.append('dataset_id', String(datasetId));
    return api.post('/rag/documents', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then(unwrap);
  },

  deleteDocument: (documentId) =>
    api.delete(`/rag/documents/${documentId}`),

  query: (question, k = 5) =>
    api.post('/rag/query', { question, k }).then(unwrap),

  queryDocument: (documentId, question, k = 5) =>
    api.post(`/rag/query/${documentId}`, { question, k }).then(unwrap),

  evaluate: (cases, documentIds = null, k = 5) =>
    api.post('/rag/evaluate', { cases, document_ids: documentIds, k }).then(unwrap),
};

// ── Health ────────────────────────────────────────────────────────────────

export const health = {
  live: () => api.get('/health/live').then(unwrap),
  ready: () => api.get('/health/ready').then(unwrap),
};

export default api;

// ── Drift Detection ──────────────────────────────────────────────────────────

export const drift = {
  check: (referenceDatasetId, currentDatasetId, targetCol = null) =>
    api.post('/drift/check', {
      reference_dataset_id: referenceDatasetId,
      current_dataset_id:   currentDatasetId,
      target_col:           targetCol || undefined,
    }).then(r => r.data.data),

  presets: () =>
    api.get('/drift/presets').then(r => r.data.data),
};

// ── Fine-Tuning ──────────────────────────────────────────────────────────────

export const fineTuning = {
  submit: (payload) =>
    api.post('/fine-tuning/jobs', payload).then(r => r.data.data),

  list: (page = 1, pageSize = 20) =>
    api.get(`/fine-tuning/jobs?page=${page}&page_size=${pageSize}`).then(r => r.data),

  get: (jobId) =>
    api.get(`/fine-tuning/jobs/${jobId}`).then(r => r.data.data),

  delete: (jobId) =>
    api.delete(`/fine-tuning/jobs/${jobId}`),
};

// ── Forecasting ──────────────────────────────────────────────────────────────

export const forecasting = {
  families: () =>
    api.get('/forecasting/families').then(r => r.data.data),

  submit: (payload) =>
    api.post('/forecasting/jobs', payload).then(r => r.data.data),

  list: (page = 1, pageSize = 20) =>
    api.get(`/forecasting/jobs?page=${page}&page_size=${pageSize}`).then(r => r.data),

  get: (jobId) =>
    api.get(`/forecasting/jobs/${jobId}`).then(r => r.data.data),

  delete: (jobId) =>
    api.delete(`/forecasting/jobs/${jobId}`),
};

// ── Agent ────────────────────────────────────────────────────────────────────

export const agent = {
  tools: () =>
    api.get('/agent/tools').then(r => r.data.data),

  run: (goal) =>
    api.post('/agent/run', { goal }).then(r => r.data.data),

  createStream: (goal) =>
    api.post('/agent/run/stream-id', { goal }).then(r => r.data.data),
};

// ── A/B Testing ──────────────────────────────────────────────────────────────

export const abTesting = {
  run: (experiment_a_id, experiment_b_id, confidence_level = 0.95) =>
    api.post('/ab-test/', { experiment_a_id, experiment_b_id, confidence_level })
       .then(r => r.data.data),

  power: (payload) =>
    api.post('/ab-test/power', payload).then(r => r.data.data),

  methods: () =>
    api.get('/ab-test/methods').then(r => r.data.data),
};

// ── Data Contracts ───────────────────────────────────────────────────────────

export const contracts = {
  generate: (datasetId, payload = {}) =>
    api.post(`/datasets/${datasetId}/contracts/generate`, payload).then(r => r.data.data),

  get: (datasetId) =>
    api.get(`/datasets/${datasetId}/contracts`).then(r => r.data.data),

  validate: (referenceDatasetId, incomingDatasetId) =>
    api.post(`/datasets/${referenceDatasetId}/contracts/validate`,
      { dataset_id: incomingDatasetId }).then(r => r.data.data),

  delete: (datasetId) =>
    api.delete(`/datasets/${datasetId}/contracts`),
};

// Add DPO submit to fineTuning (extend existing)
export const dpoSubmit = (payload) =>
  api.post('/fine-tuning/jobs/dpo', payload).then(r => r.data.data);

// ── BYOK — user API key management ───────────────────────────────────────────

export const apiKeys = {
  store:  (key) =>
    api.put('/auth/api-keys', { anthropic_api_key: key }).then(r => r.data.data),
  status: () =>
    api.get('/auth/api-keys/status').then(r => r.data.data),
  delete: () =>
    api.delete('/auth/api-keys'),
};

// ── Agent Training Pipeline ───────────────────────────────────────────────────

export const agentTraining = {
  captureTrace: (session, agentType) =>
    api.post('/agents/traces/capture', { session, agent_type: agentType }).then(r => r.data.data),

  listTraces: (params = {}) =>
    api.get('/agents/traces', { params }).then(r => r.data),

  scoreTraces: (limit = 50) =>
    api.post('/agents/traces/score', null, { params: { limit } }).then(r => r.data.data),

  getTrainingData: (format = 'alpaca', minScoreGap = 0.15) =>
    api.get('/agents/traces/training-data', { params: { format, min_score_gap: minScoreGap } })
      .then(r => r.data.data),

  exportBundle: (payload) =>
    api.post('/agents/export', payload).then(r => r.data.data),

  importBundle: (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/agents/import', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then(r => r.data.data);
  },

  listAgents: () =>
    api.get('/agents').then(r => r.data.data),

  getAgent: (name) =>
    api.get(`/agents/${encodeURIComponent(name)}`).then(r => r.data.data),

  archiveAgent: (name) =>
    api.delete(`/agents/${encodeURIComponent(name)}`),

  benchmarkAgent: (name) =>
    api.post(`/agents/${encodeURIComponent(name)}/benchmark`).then(r => r.data.data),
};

// ── Cloud (SageMaker, LLM/tracking providers) ─────────────────────────────────

export const cloud = {
  submitSageMaker: (payload) =>
    api.post('/cloud/sagemaker/submit', payload).then(r => r.data.data),

  instanceTypes: () =>
    api.get('/cloud/sagemaker/instance-types').then(r => r.data.data),

  llmProviders: () =>
    api.get('/cloud/llm-providers').then(r => r.data.data),

  trackingProviders: () =>
    api.get('/cloud/tracking-providers').then(r => r.data.data),
};

// ── Background job queue monitoring ───────────────────────────────────────────

export const jobQueue = {
  getStatus: (jobId) =>
    api.get(`/jobs/${encodeURIComponent(jobId)}`).then(r => r.data.data),

  listRecent: (limit = 50) =>
    api.get('/jobs', { params: { limit } }).then(r => r.data.data),

  cacheStats: () =>
    api.get('/cache/stats').then(r => r.data.data),
};

// ── Retraining pipeline ─────────────────────────────────────────────────────

export const retraining = {
  listPolicies: () =>
    api.get('/retraining/policies').then(r => r.data.data),

  getPolicy: (id) =>
    api.get(`/retraining/policies/${id}`).then(r => r.data.data),

  createPolicy: (payload) =>
    api.post('/retraining/policies', payload).then(r => r.data.data),

  updatePolicy: (id, payload) =>
    api.patch(`/retraining/policies/${id}`, payload).then(r => r.data.data),

  deletePolicy: (id) =>
    api.delete(`/retraining/policies/${id}`),

  runPolicy: (id, currentDatasetId) =>
    api.post(`/retraining/policies/${id}/run`, currentDatasetId ? { current_dataset_id: currentDatasetId } : {})
      .then(r => r.data.data),

  listRuns: (policyId, limit = 20) =>
    api.get(`/retraining/policies/${policyId}/runs`, { params: { limit } }).then(r => r.data.data),

  getRun: (runId) =>
    api.get(`/retraining/runs/${runId}`).then(r => r.data.data),

  promoteExperiment: (experimentId) =>
    api.post(`/experiments/${experimentId}/promote`).then(r => r.data.data),
};
