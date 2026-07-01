import { Settings, Cpu, BarChart3 } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import APIKeyPanel from '../components/APIKeyPanel.jsx';
import { cloud } from '../api/client.js';
import { PageHeader, Card, SectionLabel, Spinner } from '../components/ui.jsx';

export default function SettingsPage() {
  return (
    <div style={{ maxWidth: 640, margin: '0 auto', padding: 32 }}>
      <PageHeader
        title="Settings"
        subtitle="API keys, preferences, and account configuration"
      />

      <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

        {/* API key */}
        <APIKeyPanel />

        {/* LLM & tracking providers */}
        <ProviderCard
          title="LLM provider"
          icon={Cpu}
          queryKey="llm-providers"
          queryFn={cloud.llmProviders}
          footnote="Selected via LLM_PROVIDER on the server — not switchable per-request. Bedrock uses AWS IAM credentials instead of an API key."
        />
        <ProviderCard
          title="Experiment tracking provider"
          icon={BarChart3}
          queryKey="tracking-providers"
          queryFn={cloud.trackingProviders}
          footnote="Selected via TRACKING_BACKEND on the server. Every completed training run is logged here automatically."
        />

        {/* How BYOK works */}
        <Card style={{ padding: '14px 16px' }}>
          <SectionLabel>How key resolution works</SectionLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontSize: 12, color: 'var(--text-3)', lineHeight: 1.7 }}>
            {[
              ['1 — Your key (preferred)', 'When you store a personal key here, all Claude calls (advisor, agents, RAG evaluation, LLM judge, DPO training) use it. Usage and billing go to your Anthropic account.'],
              ['2 — Server key (fallback)', 'If no personal key is stored, the platform uses its shared server-level key. This key may have rate limits across all users.'],
              ['3 — No key', 'If neither is configured, Claude-powered features return a clear error. Non-Claude features (profiling, AutoML, SHAP, fairness, A/B testing) work normally.'],
            ].map(([label, desc]) => (
              <div key={label} style={{ display: 'flex', gap: 12 }}>
                <span style={{ fontWeight: 700, color: 'var(--accent)', minWidth: 180, flexShrink: 0, fontFamily: 'var(--font-mono)', fontSize: 11 }}>{label}</span>
                <span>{desc}</span>
              </div>
            ))}
          </div>
        </Card>

        {/* Cost estimate */}
        <Card style={{ padding: '14px 16px' }}>
          <SectionLabel>Cost reference (Claude Haiku 4.5)</SectionLabel>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 12, color: 'var(--text-3)' }}>
            {[
              ['Dataset advisor query', '~$0.001'],
              ['Agent run (10 steps)', '~$0.01'],
              ['RAG Q&A (per question)', '~$0.002'],
              ['LLM evaluation (per sample)', '~$0.003'],
              ['DPO training (mock mode)', '$0.00'],
              ['AutoML / profiling / SHAP', '$0.00'],
            ].map(([label, cost]) => (
              <div key={label} style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 0', borderBottom: '1px solid var(--border)' }}>
                <span>{label}</span>
                <span style={{ fontFamily: 'var(--font-mono)', color: cost === '$0.00' ? 'var(--green)' : 'var(--text-2)', fontWeight: 600 }}>{cost}</span>
              </div>
            ))}
          </div>
          <p style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 10 }}>
            New Anthropic accounts receive ~$5 free credits — enough for hundreds of agent runs.
          </p>
        </Card>

      </div>
    </div>
  );
}

function ProviderCard({ title, icon: Icon, queryKey, queryFn, footnote }) {
  const { data: providers, isLoading } = useQuery({ queryKey: [queryKey], queryFn });

  return (
    <Card style={{ padding: '14px 16px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <Icon size={15} style={{ color: 'var(--accent)' }} />
        <SectionLabel style={{ margin: 0 }}>{title}</SectionLabel>
      </div>
      {isLoading && <Spinner size={14} />}
      {providers && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {providers.map(p => (
            <div key={p.provider} style={{
              display: 'flex', alignItems: 'center', gap: 10, padding: '8px 12px',
              borderRadius: 6, fontSize: 12,
              background: p.active ? 'rgba(46,204,113,0.08)' : 'var(--bg-3)',
              border: p.active ? '1px solid rgba(46,204,113,0.3)' : '1px solid transparent',
            }}>
              <span style={{
                width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
                background: p.active ? 'var(--green)' : 'var(--border)',
              }} />
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: p.active ? 700 : 400, color: p.active ? 'var(--green)' : 'var(--text-2)' }}>
                {p.provider}
              </span>
              {p.is_free && <span style={{ fontSize: 9, padding: '1px 6px', borderRadius: 8, background: 'rgba(52,152,219,0.15)', color: 'var(--blue)' }}>FREE TIER</span>}
              <span style={{ color: 'var(--text-3)', marginLeft: 'auto', textAlign: 'right', maxWidth: 360 }}>{p.description}</span>
            </div>
          ))}
        </div>
      )}
      <p style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 10 }}>{footnote}</p>
    </Card>
  );
}
