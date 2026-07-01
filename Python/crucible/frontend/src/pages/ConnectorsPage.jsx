import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Plug, Trash2, CheckCircle2, XCircle, Plus, X } from 'lucide-react';
import { connectors } from '../api/client.js';
import {
  PageHeader, StatusBadge, Button,
  Card, EmptyState, Spinner, SectionLabel,
} from '../components/ui.jsx';

// ── Create connector modal ─────────────────────────────────────────────────

function BigQueryProgress() {
  const [stage, setStage] = useState(0);
  const stages = [
    'Authenticating with Google Cloud…',
    'Submitting query to BigQuery…',
    'Waiting for query results…',
    'Downloading data…',
    'Processing and saving dataset…',
  ];

  useEffect(() => {
    // Advance through stages on a realistic schedule.
    // Stages are illustrative — actual progress isn't measurable from the
    // frontend since BigQuery gives no streaming progress signal.
    const timings = [1500, 3000, 5000, 8000];
    const timers = timings.map((ms, i) =>
      setTimeout(() => setStage(i + 1), ms)
    );
    return () => timers.forEach(clearTimeout);
  }, []);

  const pct = Math.min(90, (stage / stages.length) * 100 + 10);

  return (
    <div style={{ marginTop: 12, padding: '12px 16px', background: 'var(--bg-3)', borderRadius: 'var(--radius)', border: '1px solid var(--border)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
        <span style={{ fontSize: 12, color: 'var(--text-2)' }}>{stages[stage]}</span>
        <span style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>{Math.round(pct)}%</span>
      </div>
      <div style={{ height: 4, background: 'var(--bg-4)', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{
          height: '100%',
          width: `${pct}%`,
          background: 'var(--accent)',
          borderRadius: 2,
          transition: 'width 0.8s ease',
        }} />
      </div>
      <p style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 8 }}>
        Large queries can take 15–30 seconds. Do not close this window.
      </p>
    </div>
  );
}

function CreateModal({ onClose, onCreated }) {
  const [mode, setMode] = useState('sql');  // 'sql' | 'oauth' | 'bigquery'
  const [form, setForm] = useState({
    name: '', connector_type: 'sql_postgres', db_url: '',
    base_url: '', client_id: '', client_secret: '',
    oauth2_flow: 'client_credentials',
    token_url: '', auth_url: '',
    token_body_format: 'json', response_format: 'json',
    client_auth: 'body', use_pkce: false,
    // BigQuery (one-shot ingest)
    bq_project_id: '', bq_query: '', bq_credentials_json: '',
    bq_location: 'US', bq_max_rows: 500000, bq_dataset_name: '',
  });
  const [error, setError] = useState(null);
  const qc = useQueryClient();

  const createMut = useMutation({
    mutationFn: () => {
      if (mode === 'bigquery') {
        return connectors.ingestBigQuery({
          project_id: form.bq_project_id,
          query: form.bq_query,
          credentials_json: form.bq_credentials_json.trim() || null,
          location: form.bq_location || 'US',
          max_rows: Number(form.bq_max_rows) || 500000,
          dataset_name: form.bq_dataset_name || form.name,
        });
      }
      return mode === 'sql'
        ? connectors.createSql({ name: form.name, connector_type: form.connector_type, db_url: form.db_url })
        : connectors.createOAuth({
            name: form.name, base_url: form.base_url,
            client_id: form.client_id, client_secret: form.client_secret,
            oauth2_flow: form.oauth2_flow, scopes: [],
            token_url: form.token_url, auth_url: form.auth_url || undefined,
            token_body_format: form.token_body_format,
            response_format: form.response_format,
            client_auth: form.client_auth, use_pkce: form.use_pkce,
          });
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['connectors'] });
      qc.invalidateQueries({ queryKey: ['datasets'] });
      onCreated?.();
      onClose();
    },
    onError: (e) => setError(e.response?.data?.detail || 'Failed to create connector'),
  });

  const field = (label, key, type = 'text', opts = null) => (
    <div key={key}>
      <label style={{ fontSize: 12, color: 'var(--text-2)', display: 'block', marginBottom: 4 }}>{label}</label>
      {type === 'textarea' ? (
        <textarea
          value={form[key]}
          onChange={e => setForm(f => ({ ...f, [key]: e.target.value }))}
          style={{ ...inputStyle, minHeight: 80, fontFamily: 'var(--font-mono, monospace)', resize: 'vertical' }}
          placeholder={opts?.placeholder || ''}
          spellCheck={false}
        />
      ) : opts ? (
        <select
          value={form[key]}
          onChange={e => setForm(f => ({ ...f, [key]: e.target.value }))}
          style={inputStyle}
        >
          {opts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
        </select>
      ) : (
        <input
          type={type}
          value={form[key]}
          onChange={e => setForm(f => ({ ...f, [key]: type === 'checkbox' ? e.target.checked : e.target.value }))}
          style={inputStyle}
          placeholder={type === 'password' ? '••••••••' : ''}
        />
      )}
    </div>
  );

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
    <div style={{
      position: 'fixed', inset: 0,
      background: 'rgba(0,0,0,0.6)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      zIndex: 100,
    }}>
      <div style={{
        background: 'var(--bg-2)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)',
        width: '100%', maxWidth: 520,
        padding: 24,
        boxShadow: '0 8px 40px rgba(0,0,0,0.5)',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <h2 style={{ fontSize: 15, fontWeight: 600 }}>New connector</h2>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--text-3)', cursor: 'pointer' }}>
            <X size={18} />
          </button>
        </div>

        {/* Mode toggle */}
        <div style={{ display: 'flex', gap: 6, marginBottom: 20 }}>
          {[['sql', 'SQL Database'], ['oauth', 'REST + OAuth2'], ['bigquery', 'BigQuery']].map(([m, l]) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              style={{
                padding: '5px 12px',
                borderRadius: 'var(--radius)',
                border: '1px solid',
                borderColor: mode === m ? 'var(--accent)' : 'var(--border)',
                background: mode === m ? 'var(--accent-dim)' : 'transparent',
                color: mode === m ? 'var(--accent)' : 'var(--text-2)',
                fontSize: 12, fontWeight: 500, cursor: 'pointer',
              }}
            >
              {l}
            </button>
          ))}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {mode !== 'bigquery' && field('Name', 'name')}

          {mode === 'sql' && <>
            {field('Database type', 'connector_type', 'text', [
              ['sql_postgres', 'PostgreSQL'],
              ['sql_sqlite', 'SQLite'],
            ])}
            {field('Connection URL', 'db_url')}
          </>}

          {mode === 'oauth' && <>
            {field('Base URL', 'base_url')}
            {field('Client ID', 'client_id')}
            {field('Client secret', 'client_secret', 'password')}
            {field('Flow', 'oauth2_flow', 'text', [
              ['client_credentials', 'Client Credentials'],
              ['authorization_code', 'Authorization Code'],
            ])}
            {field('Token URL', 'token_url')}
            {form.oauth2_flow === 'authorization_code' && field('Auth URL', 'auth_url')}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <div>{field('Body format', 'token_body_format', 'text', [['json', 'JSON'], ['form', 'Form']])}</div>
              <div>{field('Client auth', 'client_auth', 'text', [['body', 'Body'], ['basic', 'Basic']])}</div>
            </div>
          </>}

          {mode === 'bigquery' && <>
            {field('Dataset name (in Crucible)', 'bq_dataset_name')}
            {field('Google Cloud project ID', 'bq_project_id')}
            {field('SQL query', 'bq_query', 'textarea', {
              placeholder: 'SELECT col1, col2\nFROM `bigquery-public-data.dataset.table`\nLIMIT 500000',
            })}
            {field('Service account JSON', 'bq_credentials_json', 'textarea', {
              placeholder: 'Paste the full service-account key JSON here.\nLeave empty to use GOOGLE_APPLICATION_CREDENTIALS on the server.',
            })}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <div>{field('Location', 'bq_location')}</div>
              <div>{field('Max rows', 'bq_max_rows', 'number')}</div>
            </div>
            <p style={{ fontSize: 11, color: 'var(--text-3)', lineHeight: 1.5, margin: 0 }}>
              Runs the query against BigQuery and creates a dataset directly. Always include a
              LIMIT — the cap is 5,000,000 rows. Credentials are sent over your local network to
              the backend and used only for this query; they are not stored.
            </p>
          </>}
        </div>

        {error && (
          <div style={{ marginTop: 12, fontSize: 12, color: 'var(--red)' }}>{error}</div>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 20 }}>
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button onClick={() => createMut.mutate()} loading={createMut.isPending}>
            {mode === 'bigquery' ? 'Import from BigQuery' : 'Create connector'}
          </Button>

          {mode === 'bigquery' && createMut.isPending && <BigQueryProgress />}
        </div>
      </div>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────

export default function ConnectorsPage() {
  const qc = useQueryClient();
  const [showCreate, setShowCreate] = useState(false);
  // Track which connectors are mid-authorization (waiting for callback)
  const [authorizingIds, setAuthorizingIds] = useState(new Set());
  // Track test results per connector id
  const [testResults, setTestResults] = useState({});

  const { data, isLoading } = useQuery({
    queryKey: ['connectors'],
    queryFn: () => connectors.list(),
    // Poll every 3 seconds while any connector is mid-authorization so the
    // row updates to "active" automatically once the OAuth callback completes,
    // without the user needing to manually refresh.
    refetchInterval: authorizingIds.size > 0 ? 3000 : false,
  });

  // When a previously-authorizing connector becomes active, stop polling it
  const items = data?.data ?? [];
  const prevItems = items;
  useState(() => {
    items.forEach(c => {
      if (c.status === 'active' && authorizingIds.has(c.id)) {
        setAuthorizingIds(prev => {
          const next = new Set(prev);
          next.delete(c.id);
          return next;
        });
      }
    });
  });

  const deleteMut = useMutation({
    mutationFn: (id) => connectors.delete(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['connectors'] }),
  });

  const testMut = useMutation({
    mutationFn: async (id) => {
      const result = await connectors.test(id);
      setTestResults(prev => ({ ...prev, [id]: result.data }));
      qc.invalidateQueries({ queryKey: ['connectors'] });
      return result;
    },
  });

  function handleAuthorize(connector) {
    // Open the authorization redirect in a new tab. The backend will redirect
    // the browser to the provider's consent page. After the user approves,
    // the provider redirects to /oauth/callback which stores the token and
    // marks the connector active. The polling above detects this automatically.
    const authorizeUrl = `/api/v1/connectors/${connector.id}/authorize`;
    window.open(authorizeUrl, '_blank', 'width=600,height=700');
    setAuthorizingIds(prev => new Set([...prev, connector.id]));
  }

  // Whether a connector needs the "Authorize" button instead of "Test"
  function needsAuthorization(c) {
    return (
      c.connector_type === 'rest_oauth2' &&
      c.oauth2_flow === 'authorization_code' &&
      c.status !== 'active'
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <PageHeader
        title="Connectors"
        subtitle={items.length > 0 ? `${items.length} configured` : 'No connectors configured'}
        action={
          <Button onClick={() => setShowCreate(true)}>
            <Plus size={14} />
            New connector
          </Button>
        }
      />

      <div style={{ flex: 1, overflow: 'auto', padding: '24px 28px' }}>
        {isLoading ? (
          <div style={{ display: 'flex', justifyContent: 'center', paddingTop: 60 }}>
            <Spinner size={24} />
          </div>
        ) : items.length === 0 ? (
          <EmptyState
            icon={Plug}
            title="No connectors yet"
            description="Connect a SQL database or REST API to pull data into Crucible."
            action={<Button onClick={() => setShowCreate(true)}><Plus size={13} />New connector</Button>}
          />
        ) : (
          <Card>
            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 120px 100px 1fr 44px',
              padding: '8px 16px',
              borderBottom: '1px solid var(--border)',
              fontSize: 11, fontWeight: 600, letterSpacing: '0.06em',
              textTransform: 'uppercase', color: 'var(--text-3)',
              fontFamily: 'var(--font-mono)',
            }}>
              <span>Name</span><span>Type</span><span>Status</span><span></span><span></span>
            </div>
            {items.map((c, i) => (
              <div key={c.id} style={{
                display: 'grid',
                gridTemplateColumns: '1fr 120px 100px 1fr 44px',
                padding: '12px 16px',
                borderBottom: i < items.length - 1 ? '1px solid var(--border)' : 'none',
                alignItems: 'center',
                gap: 8,
              }}>
                {/* Name + URL */}
                <div>
                  <div style={{ fontWeight: 500, fontSize: 13 }}>{c.name}</div>
                  {c.base_url && (
                    <div style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginTop: 2 }}>
                      {c.base_url}
                    </div>
                  )}
                </div>

                {/* Type */}
                <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-2)' }}>
                  {c.connector_type}
                </span>

                {/* Status — shows a spinner while awaiting OAuth callback */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <StatusBadge status={authorizingIds.has(c.id) ? 'pending' : c.status} />
                  {authorizingIds.has(c.id) && (
                    <Spinner size={11} />
                  )}
                </div>

                {/* Action column — context-sensitive */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {needsAuthorization(c) ? (
                    // Authorization-code OAuth connector that hasn't been authorized yet.
                    // Opens the provider's consent page in a popup and polls for completion.
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <Button
                        size="sm"
                        variant="primary"
                        onClick={() => handleAuthorize(c)}
                        disabled={authorizingIds.has(c.id)}
                      >
                        {authorizingIds.has(c.id) ? 'Waiting for auth…' : '🔑 Authorize'}
                      </Button>
                      {authorizingIds.has(c.id) && (
                        <span style={{ fontSize: 11, color: 'var(--text-3)' }}>
                          Complete in the popup
                        </span>
                      )}
                    </div>
                  ) : (
                    // All other connectors: show a test button with inline result feedback.
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => testMut.mutate(c.id)}
                        loading={testMut.isPending && testMut.variables === c.id}
                      >
                        Test connection
                      </Button>
                      {testResults[c.id] && (
                        <span style={{
                          fontSize: 11,
                          color: testResults[c.id].success ? 'var(--green)' : 'var(--red)',
                          fontFamily: 'var(--font-mono)',
                        }}>
                          {testResults[c.id].success
                            ? `✓ ${testResults[c.id].latency_ms}ms`
                            : `✗ ${testResults[c.id].message?.slice(0, 40)}`
                          }
                        </span>
                      )}
                    </div>
                  )}
                </div>

                {/* Delete */}
                <button
                  onClick={() => { if (confirm(`Delete "${c.name}"?`)) deleteMut.mutate(c.id); }}
                  style={{
                    background: 'none', border: 'none',
                    color: 'var(--text-3)', cursor: 'pointer',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    padding: 6, borderRadius: 'var(--radius-sm)',
                  }}
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

      {showCreate && <CreateModal onClose={() => setShowCreate(false)} />}
    </div>
  );
}
