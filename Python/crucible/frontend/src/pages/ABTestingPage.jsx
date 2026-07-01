import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { GitCompare, Zap, BarChart3, Trophy, Minus } from 'lucide-react';
import { abTesting } from '../api/client.js';
import { PageHeader, Card, Button, Spinner, SectionLabel, EmptyState } from '../components/ui.jsx';

// ── Helpers ───────────────────────────────────────────────────────────────────

const SEVERITY_COLOR = { negligible: 'var(--text-3)', small: '#F39C12', medium: '#E67E22', large: '#E74C3C' };
const WINNER_COLOR   = { A: 'var(--accent)', B: 'var(--blue)', null: 'var(--text-3)' };

function Pct({ value, decimals = 1 }) {
  return <span>{value != null ? `${(value * 100).toFixed(decimals)}%` : '—'}</span>;
}

function StatBox({ label, value, sub, accent }) {
  return (
    <div style={{ background: 'var(--bg-3)', border: '1px solid var(--border)', borderRadius: 8, padding: '12px 16px', textAlign: 'center' }}>
      <div style={{ fontSize: 10, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.06em', fontFamily: 'var(--font-mono)', marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'var(--font-mono)', color: accent || 'var(--text-1)' }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

// ── Result view ───────────────────────────────────────────────────────────────

function ABResult({ result }) {
  const { winner, is_significant, p_value, score_a, score_b, absolute_diff,
          relative_diff_pct, ci_lower, ci_upper, effect_size, effect_size_label,
          statistical_test, n_samples, recommendation,
          experiment_a_name, experiment_b_name, metric, confidence_level } = result;

  const winnerColor = WINNER_COLOR[winner] || 'var(--text-3)';
  const alpha = 1 - confidence_level;
  const ciPct = Math.round(confidence_level * 100);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Verdict banner */}
      <Card style={{ padding: '20px 24px', border: `1px solid ${is_significant ? winnerColor : 'var(--border)'}40` }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            {is_significant ? (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                  <Trophy size={18} style={{ color: winnerColor }} />
                  <span style={{ fontSize: 20, fontWeight: 700, color: winnerColor }}>
                    Experiment {winner} wins
                  </span>
                </div>
                <p style={{ margin: 0, fontSize: 13, color: 'var(--text-3)' }}>
                  p = {p_value.toFixed(4)} (α = {alpha.toFixed(2)}) · {statistical_test} · n = {n_samples}
                </p>
              </>
            ) : (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                  <Minus size={18} style={{ color: 'var(--text-3)' }} />
                  <span style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-2)' }}>No significant difference</span>
                </div>
                <p style={{ margin: 0, fontSize: 13, color: 'var(--text-3)' }}>
                  p = {p_value.toFixed(4)} ≥ α = {alpha.toFixed(2)} · {statistical_test} · n = {n_samples}
                </p>
              </>
            )}
          </div>
          <div style={{ textAlign: 'right', fontSize: 12, color: 'var(--text-3)' }}>
            <div>effect size: <strong style={{ color: SEVERITY_COLOR[effect_size_label] || 'var(--text-3)' }}>{effect_size_label}</strong></div>
            <div>Cohen's h = {effect_size.toFixed(3)}</div>
          </div>
        </div>
      </Card>

      {/* Score comparison */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: 16, alignItems: 'center' }}>
        <Card style={{ padding: '16px 20px' }}>
          <div style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginBottom: 4 }}>
            Experiment A — {experiment_a_name || `#${result.experiment_a_id}`}
          </div>
          <div style={{ fontSize: 36, fontWeight: 700, fontFamily: 'var(--font-mono)', color: winner === 'A' ? 'var(--accent)' : 'var(--text-2)' }}>
            {(score_a * 100).toFixed(2)}<span style={{ fontSize: 16 }}>%</span>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-3)', marginTop: 4, textTransform: 'capitalize' }}>{metric}</div>
        </Card>

        <div style={{ textAlign: 'center', fontSize: 13, color: 'var(--text-3)' }}>
          <div>vs</div>
          <div style={{ fontSize: 11, marginTop: 4, fontFamily: 'var(--font-mono)', color: absolute_diff > 0 ? 'var(--green)' : absolute_diff < 0 ? 'var(--red)' : 'var(--text-3)' }}>
            {absolute_diff > 0 ? '+' : ''}{(absolute_diff * 100).toFixed(2)}pp
          </div>
        </div>

        <Card style={{ padding: '16px 20px' }}>
          <div style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginBottom: 4 }}>
            Experiment B — {experiment_b_name || `#${result.experiment_b_id}`}
          </div>
          <div style={{ fontSize: 36, fontWeight: 700, fontFamily: 'var(--font-mono)', color: winner === 'B' ? 'var(--blue)' : 'var(--text-2)' }}>
            {(score_b * 100).toFixed(2)}<span style={{ fontSize: 16 }}>%</span>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-3)', marginTop: 4, textTransform: 'capitalize' }}>{metric}</div>
        </Card>
      </div>

      {/* Stats row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
        <StatBox label="p-value" value={p_value.toFixed(4)}
          sub={is_significant ? 'significant' : 'not significant'}
          accent={is_significant ? 'var(--green)' : 'var(--text-3)'} />
        <StatBox label={`${ciPct}% CI (diff)`}
          value={`[${(ci_lower*100).toFixed(1)}, ${(ci_upper*100).toFixed(1)}]pp`} />
        <StatBox label="Relative lift"
          value={`${relative_diff_pct >= 0 ? '+' : ''}${relative_diff_pct.toFixed(1)}%`}
          accent={relative_diff_pct > 0 ? 'var(--green)' : relative_diff_pct < 0 ? 'var(--red)' : undefined} />
        <StatBox label="Test" value={statistical_test} sub={`n = ${n_samples}`} />
      </div>

      {/* Recommendation */}
      <Card style={{ padding: '14px 16px', background: 'var(--bg-3)' }}>
        <SectionLabel>Recommendation</SectionLabel>
        <p style={{ margin: 0, fontSize: 13, color: 'var(--text-2)', lineHeight: 1.7 }}>{recommendation}</p>
      </Card>
    </div>
  );
}

// ── Power analysis panel ──────────────────────────────────────────────────────

function PowerPanel() {
  const [baseline, setBaseline] = useState(0.80);
  const [effect, setEffect]     = useState(0.03);
  const [alpha, setAlpha]       = useState(0.05);
  const [power, setPower]       = useState(0.80);
  const [currentN, setCurrentN] = useState(500);
  const [result, setResult]     = useState(null);

  const powerMut = useMutation({
    mutationFn: () => abTesting.power({ baseline_rate: baseline, minimum_effect: effect, alpha, power, current_n: currentN }),
    onSuccess: setResult,
  });

  const inp = (val, set, step = 0.01) => (
    <input type="number" value={val} step={step} onChange={e => set(parseFloat(e.target.value))}
      style={{ width: '100%', background: 'var(--bg-4)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 'var(--radius-sm)', padding: '7px 10px', fontSize: 13, boxSizing: 'border-box' }} />
  );

  return (
    <Card>
      <SectionLabel>Power Analysis</SectionLabel>
      <p style={{ fontSize: 12, color: 'var(--text-3)', marginBottom: 14, lineHeight: 1.6 }}>
        Calculate the sample size needed to detect an effect, or the minimum detectable effect (MDE) for your current holdout size.
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 14 }}>
        <div><SectionLabel>Baseline rate</SectionLabel>{inp(baseline, setBaseline)}</div>
        <div><SectionLabel>Min. effect (MDE)</SectionLabel>{inp(effect, setEffect, 0.001)}</div>
        <div><SectionLabel>Current holdout n</SectionLabel>
          <input type="number" value={currentN} step={10} onChange={e => setCurrentN(parseInt(e.target.value))}
            style={{ width: '100%', background: 'var(--bg-4)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 'var(--radius-sm)', padding: '7px 10px', fontSize: 13, boxSizing: 'border-box' }} />
        </div>
        <div><SectionLabel>Alpha (α)</SectionLabel>{inp(alpha, setAlpha, 0.01)}</div>
        <div><SectionLabel>Power (1-β)</SectionLabel>{inp(power, setPower, 0.05)}</div>
      </div>
      <Button variant="primary" onClick={() => powerMut.mutate()} loading={powerMut.isPending}>
        <BarChart3 size={14} style={{ marginRight: 6 }} /> Calculate
      </Button>

      {result && (
        <div style={{ marginTop: 16, display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
          <StatBox label="Required n" value={result.required_n.toLocaleString()}
            sub={result.is_adequately_powered ? '✓ You have enough' : '✗ Need more data'}
            accent={result.is_adequately_powered ? 'var(--green)' : 'var(--red)'} />
          <StatBox label="Current MDE" value={`${(result.current_mde * 100).toFixed(2)}pp`}
            sub={`with n = ${result.current_n}`} accent="var(--accent)" />
          <StatBox label="Status"
            value={result.is_adequately_powered ? 'Powered' : 'Underpowered'}
            accent={result.is_adequately_powered ? 'var(--green)' : 'var(--red)'} />
        </div>
      )}
    </Card>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

const TABS = ['Compare', 'Power Analysis'];

export default function ABTestingPage() {
  const [tab, setTab] = useState('Compare');
  const [expAId, setExpAId] = useState('');
  const [expBId, setExpBId] = useState('');
  const [confLevel, setConfLevel] = useState(0.95);
  const [result, setResult] = useState(null);

  const abMut = useMutation({
    mutationFn: () => abTesting.run(parseInt(expAId), parseInt(expBId), confLevel),
    onSuccess: setResult,
  });

  return (
    <div style={{ maxWidth: 900, margin: '0 auto', padding: 32 }}>
      <PageHeader
        title="A/B Testing"
        subtitle="McNemar · Wilcoxon · Bootstrap · Power analysis"
      />

      <div style={{ display: 'flex', gap: 2, marginBottom: 24, borderBottom: '1px solid var(--border)' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 20px', background: 'none', border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: 500,
            color: tab === t ? 'var(--accent)' : 'var(--text-3)',
            borderBottom: tab === t ? '2px solid var(--accent)' : '2px solid transparent', marginBottom: -1,
          }}>{t}</button>
        ))}
      </div>

      {tab === 'Compare' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <Card>
            <SectionLabel>Select experiments to compare</SectionLabel>
            <p style={{ fontSize: 12, color: 'var(--text-3)', marginBottom: 14, lineHeight: 1.6 }}>
              Both experiments must be trained on the same dataset and target column. The holdout split is reproduced identically for a fair paired comparison.
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 180px', gap: 12, alignItems: 'flex-end' }}>
              <div>
                <SectionLabel>Experiment A ID (control)</SectionLabel>
                <input value={expAId} onChange={e => setExpAId(e.target.value)} placeholder="e.g. 1"
                  style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 'var(--radius-sm)', padding: '8px 12px', fontSize: 14, boxSizing: 'border-box' }} />
              </div>
              <div>
                <SectionLabel>Experiment B ID (challenger)</SectionLabel>
                <input value={expBId} onChange={e => setExpBId(e.target.value)} placeholder="e.g. 2"
                  style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 'var(--radius-sm)', padding: '8px 12px', fontSize: 14, boxSizing: 'border-box' }} />
              </div>
              <div>
                <SectionLabel>Confidence level</SectionLabel>
                <select value={confLevel} onChange={e => setConfLevel(parseFloat(e.target.value))}
                  style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-1)', borderRadius: 'var(--radius-sm)', padding: '8px 12px', fontSize: 14 }}>
                  <option value={0.90}>90%</option>
                  <option value={0.95}>95% (standard)</option>
                  <option value={0.99}>99% (high-stakes)</option>
                </select>
              </div>
            </div>
            <div style={{ marginTop: 14, display: 'flex', gap: 10 }}>
              <Button variant="primary" loading={abMut.isPending}
                disabled={!expAId || !expBId || expAId === expBId}
                onClick={() => abMut.mutate()}>
                <GitCompare size={14} style={{ marginRight: 6 }} /> Run A/B test
              </Button>
              {result && (
                <Button variant="ghost" onClick={() => setResult(null)}>Clear</Button>
              )}
            </div>
            {abMut.isError && (
              <div style={{ marginTop: 12, padding: '10px 14px', background: 'rgba(231,76,60,0.1)', borderRadius: 8, color: 'var(--red)', fontSize: 13 }}>
                {abMut.error?.response?.data?.detail || 'A/B test failed'}
              </div>
            )}
          </Card>

          {result && <ABResult result={result} />}

          {!result && !abMut.isPending && (
            <EmptyState
              icon={<GitCompare size={32} />}
              title="No comparison yet"
              description="Enter two experiment IDs above and click Run A/B test to compare them with McNemar's statistical test."
            />
          )}
        </div>
      )}

      {tab === 'Power Analysis' && <PowerPanel />}
    </div>
  );
}
