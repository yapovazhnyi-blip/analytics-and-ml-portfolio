import { Outlet, NavLink, useLocation } from 'react-router-dom';
import { Database, Plug, FlaskConical, BookOpen, CheckSquare, Cpu, TrendingUp, Bot, GitCompare, Settings, GraduationCap, Server } from 'lucide-react';

const NAV = [
  { to: '/datasets',    icon: Database,      label: 'Datasets' },
  { to: '/connectors',  icon: Plug,          label: 'Connectors' },
  { to: '/experiments', icon: FlaskConical,  label: 'Experiments' },
  { to: '/rag',         icon: BookOpen,      label: 'RAG Pipeline' },
  { to: '/fine-tuning', icon: Cpu,           label: 'Fine-Tuning' },
  { to: '/settings',    icon: Settings,     label: 'Settings' },
  { to: '/ab-testing',  icon: GitCompare,   label: 'A/B Testing' },
  { to: '/agent',       icon: Bot,          label: 'Agent' },
  { to: '/agent-training', icon: GraduationCap, label: 'Agent Training' },
  { to: '/mlops',       icon: Server,        label: 'MLOps' },
  { to: '/forecasting', icon: TrendingUp,    label: 'Forecasting' },
  { to: '/evaluation',  icon: CheckSquare,   label: 'Evaluation' },
];

export default function Layout() {
  return (
    <div style={{ display: 'flex', height: '100%', width: '100%' }}>
      {/* Sidebar */}
      <nav style={{
        width: 220,
        flexShrink: 0,
        background: 'var(--bg-2)',
        borderRight: '1px solid var(--border)',
        display: 'flex',
        flexDirection: 'column',
        padding: '0',
      }}>
        {/* Logo */}
        <div style={{
          padding: '20px 20px 16px',
          borderBottom: '1px solid var(--border)',
          display: 'flex',
          alignItems: 'center',
          gap: 10,
        }}>
          <FlaskConical size={18} color="var(--accent)" strokeWidth={1.5} />
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 15,
            fontWeight: 600,
            letterSpacing: '0.02em',
            color: 'var(--text-1)',
          }}>
            crucible
          </span>
        </div>

        {/* Nav items */}
        <div style={{ padding: '8px 8px', flex: 1 }}>
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              style={({ isActive }) => ({
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                padding: '8px 12px',
                borderRadius: 'var(--radius)',
                color: isActive ? 'var(--text-1)' : 'var(--text-2)',
                background: isActive ? 'var(--bg-4)' : 'transparent',
                fontSize: 13,
                fontWeight: isActive ? 500 : 400,
                marginBottom: 2,
                transition: 'all 0.12s',
                textDecoration: 'none',
                borderLeft: isActive ? '2px solid var(--accent)' : '2px solid transparent',
              })}
            >
              <Icon size={15} strokeWidth={1.5} />
              {label}
            </NavLink>
          ))}
        </div>

        {/* Footer */}
        <div style={{
          padding: '12px 20px',
          borderTop: '1px solid var(--border)',
          fontSize: 11,
          color: 'var(--text-3)',
          fontFamily: 'var(--font-mono)',
        }}>
          phase 1 · v0.1.0
        </div>
      </nav>

      {/* Content */}
      <main style={{
        flex: 1,
        overflow: 'auto',
        display: 'flex',
        flexDirection: 'column',
      }}>
        <Outlet />
      </main>
    </div>
  );
}
