import { useState, useRef, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Bot, Send, Wrench, CheckCircle2, AlertCircle, ChevronDown, ChevronRight, Zap, RotateCcw } from 'lucide-react';
import { agent } from '../api/client.js';
import { PageHeader, Card, Button, Spinner, SectionLabel } from '../components/ui.jsx';

// ── Example goals ─────────────────────────────────────────────────────────────

const EXAMPLES = [
  "List all available datasets and describe their contents",
  "What datasets do I have? Run profiling on dataset 1 and report any data quality issues",
  "Train a classification model on my largest dataset — use the first binary column as the target",
  "Analyse dataset 1, profile it for quality issues, then train a model and explain the top features",
  "Search the indexed documents for information about machine learning best practices",
];

const MULTI_EXAMPLES = [
  "Analyse my datasets, train a classification model on the 'label' column, and deploy it",
  "Find my latest dataset, check data quality, train a churn prediction model",
  "Profile dataset 1, train a model to predict revenue, then generate a deployment package",
];

// ── Event renderer ────────────────────────────────────────────────────────────

function ToolCallBadge({ tool, input }) {
  const [open, setOpen] = useState(false);
  const hasInput = input && Object.keys(input).length > 0;
  return (
    <div style={{
      display: 'inline-flex', flexDirection: 'column',
      background: 'var(--bg-3)', border: '1px solid var(--border)',
      borderRadius: 6, overflow: 'hidden', fontSize: 12,
      marginTop: 6, maxWidth: '100%',
    }}>
      <button onClick={() => hasInput && setOpen(o => !o)}
        style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 10px', background: 'none', border: 'none', cursor: hasInput ? 'pointer' : 'default', color: 'var(--accent)' }}>
        <Wrench size={12} />
        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{tool}</span>
        {hasInput && (open ? <ChevronDown size={11} /> : <ChevronRight size={11} />)}
      </button>
      {open && hasInput && (
        <pre style={{ margin: 0, padding: '0 10px 8px', fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
          {JSON.stringify(input, null, 2)}
        </pre>
      )}
    </div>
  );
}

function ToolResultBubble({ tool, result, isError }) {
  const [open, setOpen] = useState(false);
  const preview = result.split('\n')[0].slice(0, 80) + (result.length > 80 ? '…' : '');
  const hasMore = result.includes('\n') || result.length > 80;
  return (
    <div style={{
      background: isError ? 'rgba(231,76,60,0.08)' : 'var(--bg-2)',
      border: `1px solid ${isError ? 'rgba(231,76,60,0.3)' : 'var(--border)'}`,
      borderRadius: 6, padding: '6px 10px', marginTop: 4, fontSize: 12,
    }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 6 }}>
        {isError
          ? <AlertCircle size={12} style={{ color: 'var(--red)', marginTop: 1, flexShrink: 0 }} />
          : <CheckCircle2 size={12} style={{ color: 'var(--green)', marginTop: 1, flexShrink: 0 }} />
        }
        <span style={{ color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>{preview}</span>
      </div>
      {hasMore && !open && (
        <button onClick={() => setOpen(true)}
          style={{ background: 'none', border: 'none', color: 'var(--accent)', fontSize: 11, cursor: 'pointer', padding: '2px 0 0 18px' }}>
          show full output ↓
        </button>
      )}
      {open && (
        <pre style={{ margin: '6px 0 0 18px', fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
          {result}
        </pre>
      )}
    </div>
  );
}

function AgentMessage({ event }) {
  if (event.type === 'thinking') {
    return (
      <div style={{ padding: '10px 0', borderBottom: '1px solid var(--border)' }}>
        <p style={{ margin: 0, fontSize: 14, color: 'var(--text-2)', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>{event.text}</p>
      </div>
    );
  }
  if (event.type === 'tool_call') {
    return (
      <div style={{ padding: '6px 0' }}>
        <ToolCallBadge tool={event.tool} input={event.input} />
      </div>
    );
  }
  if (event.type === 'tool_result') {
    return (
      <div style={{ padding: '4px 0 8px' }}>
        <ToolResultBubble tool={event.tool} result={event.result} isError={event.is_error} />
      </div>
    );
  }
  if (event.type === 'final_answer') {
    return (
      <div style={{
        marginTop: 12, padding: '14px 16px',
        background: 'rgba(var(--accent-rgb, 30,215,96),0.08)',
        border: '1px solid rgba(var(--accent-rgb, 30,215,96),0.25)',
        borderRadius: 8,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
          <Bot size={14} style={{ color: 'var(--accent)' }} />
          <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--accent)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Final answer</span>
        </div>
        <p style={{ margin: 0, fontSize: 14, color: 'var(--text-1)', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{event.text}</p>
      </div>
    );
  }
  if (event.type === 'error') {
    return (
      <div style={{ padding: '10px 14px', background: 'rgba(231,76,60,0.1)', border: '1px solid rgba(231,76,60,0.3)', borderRadius: 8, color: 'var(--red)', fontSize: 13 }}>
        {event.message}
      </div>
    );
  }
  return null;
}

// ── Tool reference panel ──────────────────────────────────────────────────────

function ToolsPanel({ tools }) {
  const [open, setOpen] = useState(false);
  if (!tools?.length) return null;
  return (
    <Card style={{ padding: '12px 16px' }}>
      <button onClick={() => setOpen(o => !o)}
        style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%', background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-2)' }}>{tools.length} available tools</span>
        {open ? <ChevronDown size={13} style={{ color: 'var(--text-3)' }} /> : <ChevronRight size={13} style={{ color: 'var(--text-3)' }} />}
      </button>
      {open && (
        <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
          {tools.map(t => (
            <div key={t.name}>
              <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>{t.name}</span>
              <p style={{ margin: '2px 0 0', fontSize: 11, color: 'var(--text-3)', lineHeight: 1.5 }}>
                {t.description.split('.')[0]}.
              </p>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function AgentPage() {
  const [goal, setGoal]         = useState('');
  const [events, setEvents]     = useState([]);
  const [running, setRunning]   = useState(false);
  const [stats, setStats]       = useState(null);
  const [mode, setMode]         = useState('react');   // 'react' | 'multi'
  const bottomRef               = useRef(null);
  const wsRef                   = useRef(null);

  const { data: toolsData } = useQuery({
    queryKey: ['agent-tools'],
    queryFn: () => agent.tools(),
  });

  // Auto-scroll to bottom as events stream in
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [events]);

  async function handleRun() {
    if (!goal.trim() || running) return;
    setEvents([]);
    setStats(null);
    setRunning(true);
    const start = Date.now();

    try {
      if (mode === 'multi') {
        // Multi-agent: synchronous call, convert events
        const resp = await api.post('/agent/multi/run', { goal });
        const session = resp.data.data;
        setEvents(session.events.map(e => ({
          type:       e.type === 'supervisor'  ? 'thinking'     :
                      e.type === 'specialist'  ? 'tool_result'  :
                      e.type === 'finished'    ? 'final_answer' : 'error',
          text:       e.type === 'supervisor'  ? `[Supervisor → ${e.routing_to}] ${e.reasoning}` :
                      e.type === 'specialist'  ? `[${e.agent}] ${e.output}` :
                      e.type === 'finished'    ? e.answer : e.message,
          tool:       e.type === 'specialist'  ? e.agent : undefined,
          result:     e.type === 'specialist'  ? e.output : undefined,
          is_error:   false,
        })));
        setStats({ elapsed: session.elapsed_secs?.toFixed(1), toolCalls: session.steps });
      } else {
        // Single ReAct agent with WebSocket streaming
        const { session_id } = await agent.createStream(goal);
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const token = localStorage.getItem('crucible_access_token');
        const url   = `${proto}//localhost:8000/ws/agent/${session_id}${token ? `?token=${token}` : ''}`;
        let toolCalls = 0;
        await new Promise((resolve, reject) => {
          const ws = new WebSocket(url);
          wsRef.current = ws;
          ws.onmessage = (e) => {
            const msg = JSON.parse(e.data);
            setEvents(prev => [...prev, msg]);
            if (msg.type === 'tool_call') toolCalls++;
            if (msg.type === 'final_answer' || msg.type === 'error')
              setStats({ elapsed: ((Date.now() - start) / 1000).toFixed(1), toolCalls });
          };
          ws.onerror = () => reject(new Error('WebSocket error'));
          ws.onclose = () => resolve();
        });
      }
    } catch (err) {
      try {
        const session = await agent.run(goal);
        setEvents(session.events || []);
        setStats({ elapsed: session.elapsed_secs?.toFixed(1), toolCalls: session.n_tool_calls });
      } catch (e2) {
        setEvents([{ type: 'error', message: e2.message || 'Agent request failed' }]);
      }
    } finally {
      setRunning(false);
    }
  }

  function handleReset() {
    setEvents([]);
    setStats(null);
    setGoal('');
    wsRef.current?.close();
  }

  const toolCallCount = events.filter(e => e.type === 'tool_call').length;
  const isDone = events.some(e => e.type === 'final_answer' || e.type === 'error');

  return (
    <div style={{ maxWidth: 900, margin: '0 auto', padding: 32 }}>
      <PageHeader
        title="Crucible Agent"
        subtitle="Natural language → ML workflow · ReAct · Supervisor + Specialist · LangGraph"
      />

      {/* Mode switcher */}
      <div style={{ display: 'flex', gap: 2, marginBottom: 20, borderBottom: '1px solid var(--border)' }}>
        {[['react', 'ReAct (single agent)'], ['multi', 'Multi-agent (LangGraph)']].map(([m, label]) => (
          <button key={m} onClick={() => { setMode(m); setEvents([]); setStats(null); setGoal(''); }}
            style={{ padding: '8px 18px', background: 'none', border: 'none', cursor: 'pointer',
              fontSize: 13, fontWeight: 500,
              color: mode === m ? 'var(--accent)' : 'var(--text-3)',
              borderBottom: mode === m ? '2px solid var(--accent)' : '2px solid transparent', marginBottom: -1 }}>
            {label}
          </button>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 260px', gap: 20, alignItems: 'start' }}>
        {/* Left: chat area */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

          {/* Goal input */}
          <Card>
            <SectionLabel>What would you like the agent to do?</SectionLabel>
            <textarea
              value={goal}
              onChange={e => setGoal(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleRun(); }}
              placeholder="e.g. List all datasets, profile the largest one, and tell me if there are any data quality issues"
              rows={3}
              disabled={running}
              style={{
                width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)',
                color: 'var(--text-1)', borderRadius: 'var(--radius-sm)', padding: '10px 14px',
                fontSize: 14, resize: 'vertical', fontFamily: 'var(--font-sans)', lineHeight: 1.5,
                boxSizing: 'border-box', opacity: running ? 0.6 : 1,
              }}
            />
            <div style={{ display: 'flex', gap: 8, marginTop: 10, justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: 11, color: 'var(--text-3)' }}>Cmd+Enter to run</span>
              <div style={{ display: 'flex', gap: 8 }}>
                {(events.length > 0 || stats) && (
                  <Button variant="ghost" onClick={handleReset}>
                    <RotateCcw size={13} style={{ marginRight: 4 }} /> Reset
                  </Button>
                )}
                <Button variant="primary" onClick={handleRun} loading={running} disabled={!goal.trim()}>
                  <Zap size={14} style={{ marginRight: 6 }} /> Run agent
                </Button>
              </div>
            </div>
          </Card>

          {/* Stats bar */}
          {(running || stats) && (
            <div style={{ display: 'flex', gap: 16, padding: '8px 14px', background: 'var(--bg-3)', borderRadius: 8, border: '1px solid var(--border)', fontSize: 12, alignItems: 'center' }}>
              {running && <><Spinner size={12} /><span style={{ color: 'var(--accent)' }}>Running…</span></>}
              {stats && <span style={{ color: 'var(--green)' }}>✓ Completed in {stats.elapsed}s</span>}
              <span style={{ color: 'var(--text-3)', marginLeft: 'auto' }}>
                {toolCallCount} tool call{toolCallCount !== 1 ? 's' : ''}
              </span>
            </div>
          )}

          {/* Event trace */}
          {events.length > 0 && (
            <Card>
              <SectionLabel>Agent trace</SectionLabel>
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                {events.map((ev, i) => <AgentMessage key={i} event={ev} />)}
              </div>
              <div ref={bottomRef} />
            </Card>
          )}

          {/* Empty state */}
          {events.length === 0 && !running && (
            <Card style={{ textAlign: 'center', padding: '32px 24px' }}>
              <Bot size={40} style={{ color: 'var(--text-3)', margin: '0 auto 12px' }} />
              <p style={{ color: 'var(--text-3)', fontSize: 14, margin: 0 }}>
                The agent will reason step-by-step, calling Crucible tools to complete your goal.
                Try one of the examples →
              </p>
            </Card>
          )}
        </div>

        {/* Right: tools + examples */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <ToolsPanel tools={toolsData?.tools} />

          <Card style={{ padding: '12px 16px' }}>
            <SectionLabel>Example goals</SectionLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {EXAMPLES.map((ex, i) => (
                <button key={i} onClick={() => setGoal(ex)}
                  style={{
                    background: 'none', border: '1px solid var(--border)', borderRadius: 6,
                    padding: '7px 10px', cursor: 'pointer', textAlign: 'left',
                    fontSize: 11, color: 'var(--text-3)', lineHeight: 1.4,
                    transition: 'border-color 0.15s, color 0.15s',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--accent)'; e.currentTarget.style.color = 'var(--text-2)'; }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text-3)'; }}
                >
                  {ex}
                </button>
              ))}
            </div>
          </Card>

          <Card style={{ padding: '12px 16px' }}>
            <SectionLabel>How it works</SectionLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: 11, color: 'var(--text-3)', lineHeight: 1.6 }}>
              {[
                ['Reason', 'Claude Haiku reads your goal and decides which tool to call'],
                ['Act', 'The tool runs against the real Crucible API'],
                ['Observe', 'Claude reads the result and plans the next step'],
                ['Repeat', 'Up to 10 tool calls per session'],
              ].map(([label, desc]) => (
                <div key={label} style={{ display: 'flex', gap: 8 }}>
                  <span style={{ fontWeight: 600, color: 'var(--accent)', width: 52, flexShrink: 0 }}>{label}</span>
                  <span>{desc}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
