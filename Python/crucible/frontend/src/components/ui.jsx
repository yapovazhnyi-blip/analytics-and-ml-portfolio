import React from 'react';
import { Loader2 } from 'lucide-react';

// ── Status badge ───────────────────────────────────────────────────────────

const STATUS_COLORS = {
  ready:       { bg: 'var(--green-dim)',  text: 'var(--green)',  dot: 'var(--green)'  },
  active:      { bg: 'var(--green-dim)',  text: 'var(--green)',  dot: 'var(--green)'  },
  ingesting:   { bg: 'var(--blue-dim)',   text: 'var(--blue)',   dot: 'var(--blue)'   },
  pending:     { bg: 'var(--amber-dim)',  text: 'var(--amber)',  dot: 'var(--amber)'  },
  running:     { bg: 'var(--blue-dim)',   text: 'var(--blue)',   dot: 'var(--blue)'   },
  error:       { bg: 'var(--red-dim)',    text: 'var(--red)',    dot: 'var(--red)'    },
  unconfigured:{ bg: 'var(--bg-4)',       text: 'var(--text-3)', dot: 'var(--text-3)' },
};

export function StatusBadge({ status }) {
  const colors = STATUS_COLORS[status] || STATUS_COLORS.unconfigured;
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 5,
      padding: '2px 8px',
      borderRadius: 20,
      background: colors.bg,
      color: colors.text,
      fontSize: 11,
      fontWeight: 500,
      fontFamily: 'var(--font-mono)',
      letterSpacing: '0.03em',
    }}>
      <span style={{
        width: 5, height: 5,
        borderRadius: '50%',
        background: colors.dot,
        flexShrink: 0,
      }} />
      {status}
    </span>
  );
}

// ── Page header ────────────────────────────────────────────────────────────

export function PageHeader({ title, subtitle, action }) {
  return (
    <div style={{
      padding: '24px 28px 20px',
      borderBottom: '1px solid var(--border)',
      display: 'flex',
      alignItems: 'flex-start',
      justifyContent: 'space-between',
      gap: 16,
      flexShrink: 0,
    }}>
      <div>
        <h1 style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-1)', marginBottom: 3 }}>
          {title}
        </h1>
        {subtitle && (
          <p style={{ fontSize: 13, color: 'var(--text-2)' }}>{subtitle}</p>
        )}
      </div>
      {action && <div>{action}</div>}
    </div>
  );
}

// ── Card ───────────────────────────────────────────────────────────────────

export function Card({ children, style = {} }) {
  return (
    <div style={{
      background: 'var(--bg-2)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)',
      ...style,
    }}>
      {children}
    </div>
  );
}

// ── Spinner ────────────────────────────────────────────────────────────────

export function Spinner({ size = 16 }) {
  return (
    <Loader2
      size={size}
      color="var(--accent)"
      strokeWidth={2}
      style={{ animation: 'spin 0.8s linear infinite' }}
    />
  );
}

// ── Button ─────────────────────────────────────────────────────────────────

export function Button({
  children, onClick, variant = 'primary',
  size = 'md', disabled = false, loading = false, style = {},
}) {
  const base = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 7,
    border: 'none',
    borderRadius: 'var(--radius)',
    fontWeight: 500,
    cursor: disabled || loading ? 'not-allowed' : 'pointer',
    opacity: disabled ? 0.5 : 1,
    transition: 'all 0.12s',
    fontFamily: 'var(--font-sans)',
    fontSize: size === 'sm' ? 12 : 13,
    padding: size === 'sm' ? '5px 10px' : '7px 14px',
    ...style,
  };

  const variants = {
    primary:  { background: 'var(--accent)',  color: '#000' },
    ghost:    { background: 'var(--bg-4)',    color: 'var(--text-1)', border: '1px solid var(--border-2)' },
    danger:   { background: 'var(--red-dim)', color: 'var(--red)',    border: '1px solid var(--red)' },
  };

  return (
    <button onClick={onClick} disabled={disabled || loading} style={{ ...base, ...variants[variant] }}>
      {loading && <Spinner size={13} />}
      {children}
    </button>
  );
}

// ── Empty state ────────────────────────────────────────────────────────────

export function EmptyState({ icon: Icon, title, description, action }) {
  // Accept either a pre-rendered JSX element (<Cpu size={32} />) or a
  // component reference (Cpu). Lucide icons are forwardRef objects so
  // typeof === 'object' — React.isValidElement is the correct discriminator.
  const iconEl = Icon
    ? (React.isValidElement(Icon)
        ? Icon
        : <Icon size={32} strokeWidth={1} />)
    : null;
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '60px 24px',
      gap: 12,
      color: 'var(--text-3)',
      textAlign: 'center',
    }}>
      {iconEl}
      <div style={{ fontSize: 14, fontWeight: 500, color: 'var(--text-2)' }}>{title}</div>
      {description && <div style={{ fontSize: 13, maxWidth: 320 }}>{description}</div>}
      {action && <div style={{ marginTop: 8 }}>{action}</div>}
    </div>
  );
}

// ── Section label ──────────────────────────────────────────────────────────

export function SectionLabel({ children }) {
  return (
    <div style={{
      fontSize: 11,
      fontWeight: 600,
      letterSpacing: '0.08em',
      textTransform: 'uppercase',
      color: 'var(--text-3)',
      marginBottom: 10,
      fontFamily: 'var(--font-mono)',
    }}>
      {children}
    </div>
  );
}

// Inject spin keyframe once
const style = document.createElement('style');
style.textContent = '@keyframes spin { from { transform: rotate(0deg) } to { transform: rotate(360deg) } }';
document.head.appendChild(style);
