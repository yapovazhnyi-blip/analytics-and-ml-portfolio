import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Key, ShieldCheck, ShieldAlert, Trash2, Eye, EyeOff } from 'lucide-react';
import { apiKeys } from '../api/client.js';
import { Card, Button, Spinner, SectionLabel } from './ui.jsx';

export default function APIKeyPanel() {
  const qc = useQueryClient();
  const [inputKey, setInputKey]   = useState('');
  const [show, setShow]           = useState(false);
  const [feedback, setFeedback]   = useState('');

  const { data: status, isLoading } = useQuery({
    queryKey: ['api-key-status'],
    queryFn:  apiKeys.status,
    retry:    false,
  });

  const storeMut = useMutation({
    mutationFn: () => apiKeys.store(inputKey.trim()),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['api-key-status'] });
      setInputKey('');
      setFeedback('Key saved successfully.');
      setTimeout(() => setFeedback(''), 3000);
    },
    onError: (e) => setFeedback(e.response?.data?.detail || 'Failed to save key.'),
  });

  const deleteMut = useMutation({
    mutationFn: apiKeys.delete,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['api-key-status'] });
      setFeedback('Key removed.');
      setTimeout(() => setFeedback(''), 3000);
    },
  });

  const sourceColor = {
    user:   'var(--green)',
    server: 'var(--accent)',
    none:   'var(--red)',
  };

  return (
    <Card>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
        <Key size={16} style={{ color: 'var(--accent)' }} />
        <SectionLabel style={{ margin: 0 }}>Anthropic API Key</SectionLabel>
      </div>

      {isLoading && <Spinner />}

      {status && (
        <div style={{ marginBottom: 16, padding: '10px 14px', background: 'var(--bg-3)', borderRadius: 8, display: 'flex', alignItems: 'center', gap: 10 }}>
          {status.active_source !== 'none'
            ? <ShieldCheck size={16} style={{ color: sourceColor[status.active_source] }} />
            : <ShieldAlert size={16} style={{ color: 'var(--red)' }} />}
          <div>
            <div style={{ fontSize: 13, fontWeight: 600, color: sourceColor[status.active_source] }}>
              {status.active_source === 'user'   && 'Using your personal key'}
              {status.active_source === 'server' && 'Using shared server key'}
              {status.active_source === 'none'   && 'No API key configured'}
            </div>
            {status.preview && (
              <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-3)', marginTop: 2 }}>
                {status.preview}
              </div>
            )}
            {status.active_source === 'server' && (
              <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>
                Add your own key so you control your usage and billing.
              </div>
            )}
          </div>
          {status.has_user_key && (
            <Button variant="ghost" onClick={() => deleteMut.mutate()}
              loading={deleteMut.isPending}
              style={{ marginLeft: 'auto' }}
              title="Remove your stored key">
              <Trash2 size={13} />
            </Button>
          )}
        </div>
      )}

      {/* Key input */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <div style={{ position: 'relative', flex: 1 }}>
          <input
            type={show ? 'text' : 'password'}
            value={inputKey}
            onChange={e => setInputKey(e.target.value)}
            placeholder="sk-ant-api03-..."
            style={{
              width: '100%', boxSizing: 'border-box',
              background: 'var(--bg-3)', border: '1px solid var(--border)',
              color: 'var(--text-1)', borderRadius: 'var(--radius-sm)',
              padding: '8px 36px 8px 12px', fontSize: 13,
              fontFamily: 'var(--font-mono)',
            }}
          />
          <button
            onClick={() => setShow(s => !s)}
            style={{
              position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)',
              background: 'none', border: 'none', cursor: 'pointer',
              color: 'var(--text-3)', padding: 0, lineHeight: 0,
            }}>
            {show ? <EyeOff size={14} /> : <Eye size={14} />}
          </button>
        </div>
        <Button
          variant="primary"
          onClick={() => storeMut.mutate()}
          loading={storeMut.isPending}
          disabled={inputKey.trim().length < 10}>
          Save key
        </Button>
      </div>

      {feedback && (
        <p style={{ fontSize: 12, color: feedback.includes('success') || feedback.includes('saved') ? 'var(--green)' : 'var(--red)', marginTop: 8 }}>
          {feedback}
        </p>
      )}

      <p style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 10, lineHeight: 1.6 }}>
        Your key is encrypted with AES-128 before storage. It is never returned in any
        API response. Get your key at{' '}
        <a href="https://console.anthropic.com" target="_blank" rel="noreferrer"
          style={{ color: 'var(--accent)' }}>
          console.anthropic.com
        </a>. New accounts receive ~$5 in free credits.
      </p>
    </Card>
  );
}
