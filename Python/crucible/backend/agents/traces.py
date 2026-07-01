"""
Trace Collector — captures completed agent sessions as training data material.

WHAT GETS CAPTURED
--------------------
Every completed ReAct agent run (POST /agent/run, /agent/run/stream-id) and
every completed multi-agent run (POST /agent/multi/run) can be captured as
an AgentTrace row: the goal, every tool call and its result, the final
answer, and timing/success metadata.

WHY THIS MATTERS FOR THE TRAINING PIPELINE
---------------------------------------------
A trace is a real example of "given this goal, here is a correct sequence
of tool calls that accomplishes it." This is exactly the shape of data
needed to fine-tune a smaller, cheaper model to do the same job — instead
of relying on Claude's general reasoning ability for every single agent
call, a model fine-tuned on Crucible's own successful traces learns the
platform's specific tool vocabulary and typical workflows directly.

CAPTURE IS OPT-IN PER REQUEST
--------------------------------
Capturing is not automatic for every agent call — it is triggered explicitly
via capture=True on the run request, or via the dedicated
POST /agent/traces/capture endpoint that accepts an already-completed
session dict. This avoids silently logging every user interaction; trace
capture is a deliberate choice tied to building a training dataset.

QUALITY SCORING IS LAZY
--------------------------
Traces are stored with score_pending=True at capture time — scoring requires
an LLM call (via the LLM evaluation framework) and is not worth doing
synchronously on every capture. POST /agent/traces/score runs the scoring
pass over all pending traces in a batch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from models.agent_trace import AgentTrace


@dataclass
class TraceCaptureResult:
    trace_id: int
    agent_type: str
    succeeded: bool


class TraceCollector:
    """Persists completed agent sessions as AgentTrace rows."""

    def __init__(self, db):
        self.db = db

    async def capture_from_dict(self, session_dict: dict, agent_type: str) -> TraceCaptureResult:
        """
        Captures a trace from a session's to_dict() output.

        Works for both AgentSession.to_dict() (agent_type="react") and
        MultiAgentSession.to_dict() (agent_type="multi") — both produce
        a dict with 'goal', 'final_answer', and some form of event/step list.
        """
        goal = session_dict.get("goal", "")
        final_answer = session_dict.get("final_answer", "")
        error = session_dict.get("error") or None
        elapsed = session_dict.get("elapsed_secs", 0.0)

        if agent_type == "react":
            events = session_dict.get("events", [])
            n_tool_calls = session_dict.get("n_tool_calls", 0)
        else:   # "multi"
            events = session_dict.get("events", [])
            n_tool_calls = sum(1 for e in events if e.get("type") == "specialist")

        succeeded = error is None and bool(final_answer)

        trace = AgentTrace(
            agent_type=agent_type,
            goal=goal,
            final_answer=final_answer,
            steps_json=json.dumps(events),
            n_tool_calls=n_tool_calls,
            elapsed_secs=float(elapsed or 0.0),
            succeeded=succeeded,
            error_message=error,
            score_pending=True,
        )
        self.db.add(trace)
        await self.db.flush()
        await self.db.refresh(trace)

        return TraceCaptureResult(
            trace_id=trace.id, agent_type=agent_type, succeeded=succeeded
        )

    async def score_pending_traces(self, api_key: str = "", limit: int = 50) -> dict:
        """
        Scores pending traces using the LLM evaluation framework.

        Each trace's (goal, final_answer) pair is scored against the
        "accuracy" and "helpfulness" rubrics. The mean score becomes
        AgentTrace.quality_score.

        Returns a summary dict: {scored: N, errors: N}.
        """
        from sqlalchemy import select
        from evaluation.judge import LLMJudge

        result = await self.db.execute(
            select(AgentTrace).where(AgentTrace.score_pending == True).limit(limit)  # noqa: E712
        )
        traces = result.scalars().all()

        judge = LLMJudge(api_key=api_key)
        scored, errors = 0, 0

        for trace in traces:
            if not trace.final_answer:
                trace.quality_score = 0.0
                trace.score_pending = False
                scored += 1
                continue
            try:
                judge_result = await judge.evaluate(
                    input_text=trace.goal,
                    output_text=trace.final_answer,
                    rubric_names=["accuracy", "helpfulness"],
                )
                if judge_result.error:
                    errors += 1
                    continue
                trace.quality_score = judge_result.overall_score
                trace.score_pending = False
                scored += 1
            except Exception:
                errors += 1

        return {"scored": scored, "errors": errors, "total_checked": len(traces)}
