import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext.jsx';
import { Loader2 } from 'lucide-react';

export default function LoginPage() {
  const { login, register } = useAuth();
  const navigate = useNavigate();
  const [mode, setMode] = useState('login');   // 'login' | 'register'
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      if (mode === 'login') {
        await login(email, password);
      } else {
        await register(email, password);
      }
      navigate('/datasets');
    } catch (err) {
      setError(err.response?.data?.detail || 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{
      minHeight: '100vh',
      width: '100%',
      background: 'var(--bg-1)',
      display: 'flex',
      alignItems: 'flex',
      justifyContent: 'center',
      padding: 24,
    }}>
      <div style={{ width: '100%', maxWidth: 400 }}>
        {/* Logo */}
        <div style={{ textAlign: 'center', marginBottom: 40 }}>
          <span style={{ fontSize: 48 }}>⚗</span>
          <h1 style={{ margin: '8px 0 4px', fontSize: 28, fontWeight: 700, color: 'var(--text-1)' }}>
            Crucible
          </h1>
          <p style={{ margin: 0, fontSize: 14, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>
            ML Experimentation Platform
          </p>
        </div>

        {/* Card */}
        <div style={{
          background: 'var(--bg-2)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius)',
          padding: '32px',
        }}>
          {/* Mode toggle */}
          <div style={{ display: 'flex', gap: 2, marginBottom: 24, background: 'var(--bg-3)', padding: 3, borderRadius: 8 }}>
            {['login', 'register'].map(m => (
              <button key={m} onClick={() => { setMode(m); setError(''); }}
                style={{
                  flex: 1, padding: '7px 0', fontSize: 13, fontWeight: 500,
                  border: 'none', borderRadius: 6, cursor: 'pointer',
                  background: mode === m ? 'var(--bg-2)' : 'transparent',
                  color: mode === m ? 'var(--text-1)' : 'var(--text-3)',
                  boxShadow: mode === m ? '0 1px 3px rgba(0,0,0,0.2)' : 'none',
                  textTransform: 'capitalize',
                }}>
                {m === 'login' ? 'Sign in' : 'Create account'}
              </button>
            ))}
          </div>

          {/* Error */}
          {error && (
            <div style={{
              marginBottom: 16, padding: '10px 14px', borderRadius: 8,
              background: 'rgba(231,76,60,0.1)', border: '1px solid rgba(231,76,60,0.3)',
              color: '#E74C3C', fontSize: 13, lineHeight: 1.5,
            }}>
              {error}
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div>
              <label style={{ display: 'block', fontSize: 12, color: 'var(--text-3)', marginBottom: 6, fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                Email
              </label>
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="you@example.com"
                required
                autoFocus
                style={{
                  width: '100%', background: 'var(--bg-3)',
                  border: '1px solid var(--border)',
                  borderRadius: 8, padding: '10px 14px',
                  color: 'var(--text-1)', fontSize: 14,
                  boxSizing: 'border-box',
                  outline: 'none',
                }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: 12, color: 'var(--text-3)', marginBottom: 6, fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                Password {mode === 'register' && <span style={{ fontWeight: 400, textTransform: 'none' }}>(min. 8 characters)</span>}
              </label>
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder={mode === 'register' ? 'Choose a strong password' : 'Enter your password'}
                required
                minLength={mode === 'register' ? 8 : undefined}
                style={{
                  width: '100%', background: 'var(--bg-3)',
                  border: '1px solid var(--border)',
                  borderRadius: 8, padding: '10px 14px',
                  color: 'var(--text-1)', fontSize: 14,
                  boxSizing: 'border-box',
                  outline: 'none',
                }}
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              style={{
                marginTop: 4,
                width: '100%', padding: '11px 0',
                background: loading ? 'var(--bg-3)' : 'var(--accent)',
                color: loading ? 'var(--text-3)' : '#000',
                border: 'none', borderRadius: 8,
                fontSize: 14, fontWeight: 600, cursor: loading ? 'not-allowed' : 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                transition: 'background 0.15s',
              }}
            >
              {loading
                ? <><Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} /> Working…</>
                : mode === 'login' ? 'Sign in' : 'Create account'
              }
            </button>
          </form>

          {mode === 'register' && (
            <p style={{ marginTop: 16, fontSize: 12, color: 'var(--text-3)', textAlign: 'center', lineHeight: 1.6 }}>
              The first registered account becomes the admin.
            </p>
          )}
        </div>
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
