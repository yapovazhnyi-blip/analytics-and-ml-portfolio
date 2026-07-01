import { useState, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { Database, Upload, Trash2, ChevronRight, Activity } from 'lucide-react';
import { datasets } from '../api/client.js';
import DriftModal from '../components/DriftModal.jsx';
import {
  PageHeader, StatusBadge, Button,
  EmptyState, Spinner, Card, SectionLabel,
} from '../components/ui.jsx';

function fmt(n) {
  if (n == null) return '—';
  return n.toLocaleString();
}

function RelativeTime({ iso }) {
  if (!iso) return null;
  const d = new Date(iso + (iso.endsWith('Z') ? '' : 'Z'));
  const diff = (Date.now() - d) / 1000;
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`;
  return `${Math.round(diff / 86400)}d ago`;
}

export default function DatasetsPage() {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const fileRef = useRef();
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [driftDatasetId, setDriftDatasetId] = useState(null);

  const { data, isLoading } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => datasets.list({ page_size: 50 }),
  });

  const deleteMut = useMutation({
    mutationFn: (id) => datasets.delete(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['datasets'] }),
  });

  async function handleFileChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    try {
      await datasets.upload(file, file.name.replace(/\.[^.]+$/, ''));
      qc.invalidateQueries({ queryKey: ['datasets'] });
    } catch (err) {
      setUploadError(err.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  }

  const items = data?.data ?? [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <PageHeader
        title="Datasets"
        subtitle={items.length > 0 ? `${items.length} dataset${items.length !== 1 ? 's' : ''}` : 'No datasets yet'}
        action={
          <>
            <input
              ref={fileRef}
              type="file"
              accept=".csv,.parquet"
              style={{ display: 'none' }}
              onChange={handleFileChange}
            />
            <Button
              onClick={() => fileRef.current?.click()}
              loading={uploading}
              disabled={uploading}
            >
              <Upload size={14} />
              Upload file
            </Button>
          </>
        }
      />

      {uploadError && (
        <div style={{
          margin: '12px 28px 0',
          padding: '10px 14px',
          background: 'var(--red-dim)',
          border: '1px solid var(--red)',
          borderRadius: 'var(--radius)',
          color: 'var(--red)',
          fontSize: 13,
        }}>
          {uploadError}
        </div>
      )}

      <div style={{ flex: 1, overflow: 'auto', padding: '24px 28px' }}>
        {isLoading ? (
          <div style={{ display: 'flex', justifyContent: 'center', paddingTop: 60 }}>
            <Spinner size={24} />
          </div>
        ) : items.length === 0 ? (
          <EmptyState
            icon={Database}
            title="No datasets yet"
            description="Upload a CSV or Parquet file to get started, or connect a data source."
            action={
              <Button onClick={() => fileRef.current?.click()}>
                <Upload size={13} />
                Upload your first dataset
              </Button>
            }
          />
        ) : (
          <Card>
            {/* Table header */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 80px 80px 100px 90px 44px',
              padding: '10px 16px',
              borderBottom: '1px solid var(--border)',
              fontSize: 11,
              fontWeight: 600,
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
              color: 'var(--text-3)',
              fontFamily: 'var(--font-mono)',
            }}>
              <span>Name</span>
              <span>Rows</span>
              <span>Cols</span>
              <span>Status</span>
              <span>Added</span>
              <span />
            </div>

            {items.map((ds, i) => (
              <div
                key={ds.id}
                onClick={() => navigate(`/datasets/${ds.id}`)}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 80px 80px 100px 90px 44px',
                  padding: '12px 16px',
                  borderBottom: i < items.length - 1 ? '1px solid var(--border)' : 'none',
                  alignItems: 'center',
                  cursor: 'pointer',
                  transition: 'background 0.1s',
                }}
                onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-3)'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, minWidth: 0 }}>
                  <Database size={14} color="var(--text-3)" strokeWidth={1.5} style={{ flexShrink: 0 }} />
                  <div style={{ minWidth: 0 }}>
                    <div style={{
                      fontWeight: 500,
                      fontSize: 13,
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                    }}>
                      {ds.name}
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>
                      {ds.source_type}
                    </div>
                  </div>
                </div>

                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-2)' }}>
                  {fmt(ds.row_count)}
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-2)' }}>
                  {fmt(ds.column_count)}
                </span>

                <StatusBadge status={ds.status} />

                <span style={{ fontSize: 12, color: 'var(--text-3)' }}>
                  <RelativeTime iso={ds.created_at} />
                </span>

                {/* Drift analysis button */}
                {ds.status === 'ready' && (
                  <button
                    onClick={e => { e.stopPropagation(); setDriftDatasetId(ds.id); }}
                    title="Drift analysis"
                    style={{
                      background: 'none', border: 'none', color: 'var(--text-3)',
                      cursor: 'pointer', display: 'flex', alignItems: 'center',
                      justifyContent: 'center', padding: 6, borderRadius: 'var(--radius-sm)',
                    }}
                    onMouseEnter={e => { e.currentTarget.style.color = 'var(--accent)'; }}
                    onMouseLeave={e => { e.currentTarget.style.color = 'var(--text-3)'; }}
                  >
                    <Activity size={13} />
                  </button>
                )}

                <button
                  onClick={e => {
                    e.stopPropagation();
                    if (confirm(`Delete "${ds.name}"?`)) deleteMut.mutate(ds.id);
                  }}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: 'var(--text-3)',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: 6,
                    borderRadius: 'var(--radius-sm)',
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

      {/* Drift analysis modal */}
      {driftDatasetId && (
        <DriftModal
          referenceDatasetId={driftDatasetId}
          onClose={() => setDriftDatasetId(null)}
        />
      )}
    </div>
  );
}
