"""
Multi-agent LangGraph — Supervisor + Specialist pattern.

ARCHITECTURE
------------

    User Goal
        │
    ┌───▼───────┐
    │ Supervisor │  ← Claude Haiku reads full state, decides next step
    └───┬───────┘
        │ routes to one of:
   ┌────┴──────────────────────────┐
   │           │                   │
   ▼           ▼                   ▼
[Dataset   [Model              [Deployer]
 Analyst]   Trainer]
   │           │                   │
   └────┬──────────────────────────┘
        │ all return to supervisor
    ┌───▼───────┐
    │ Supervisor │  ← reads specialist output, decides next step
    └───────────┘
        │
    (FINISH when goal complete)

SUPERVISOR DESIGN
-----------------
The supervisor is the only node that calls the Anthropic API. It receives:
  1. The original goal
  2. All previous specialist outputs (via messages)
  3. The current context (what has been discovered / done)

It responds with a JSON routing decision:
  {"next": "dataset_analyst" | "model_trainer" | "deployer" | "FINISH",
   "reasoning": "...",
   "final_answer": "..." (only if next == FINISH)}

WHY STRUCTURED OUTPUT FROM SUPERVISOR
--------------------------------------
The supervisor must output structured JSON so the routing function
can extract the `next` field without text parsing. We enforce this by:
  1. Including explicit JSON schema in the system prompt
  2. Asking Claude to respond ONLY with JSON (no markdown fences)
  3. Using a robust parser that handles common formatting variations

MOCK MODE
---------
When ANTHROPIC_API_KEY is absent or starts with "mock", the supervisor
uses a simple rule-based router instead of calling Claude. This makes
all multi-agent tests fast, deterministic, and free.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator

import httpx
from langgraph.graph import StateGraph, END

from agents.state import (
    MultiAgentState, AgentContext,
    ROUTE_SUPERVISOR, ROUTE_DATASET_ANALYST, ROUTE_MODEL_TRAINER,
    ROUTE_DEPLOYER, ROUTE_END, ALL_SPECIALISTS, MAX_STEPS,
)
from agents.specialists import (
    dataset_analyst_node, model_trainer_node, deployer_node,
)
from config import settings


# ── Event types ────────────────────────────────────────────────────────────────

@dataclass
class SupervisorEvent:
    reasoning: str
    routing_to: str

@dataclass
class SpecialistEvent:
    agent: str
    output: str

@dataclass
class FinishedEvent:
    answer: str
    steps: int
    elapsed: float

@dataclass
class ErrorEvent:
    message: str


MultiAgentEvent = SupervisorEvent | SpecialistEvent | FinishedEvent | ErrorEvent


# ── Session result ─────────────────────────────────────────────────────────────

@dataclass
class MultiAgentSession:
    goal: str
    events: list = field(default_factory=list)
    final_answer: str = ""
    steps: int = 0
    elapsed: float = 0.0
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return not self.error and bool(self.final_answer)

    def to_dict(self) -> dict:
        serialised = []
        for e in self.events:
            if isinstance(e, SupervisorEvent):
                serialised.append({"type": "supervisor", "reasoning": e.reasoning, "routing_to": e.routing_to})
            elif isinstance(e, SpecialistEvent):
                serialised.append({"type": "specialist", "agent": e.agent, "output": e.output})
            elif isinstance(e, FinishedEvent):
                serialised.append({"type": "finished", "answer": e.answer, "steps": e.steps})
            elif isinstance(e, ErrorEvent):
                serialised.append({"type": "error", "message": e.message})
        return {
            "goal":         self.goal,
            "final_answer": self.final_answer,
            "steps":        self.steps,
            "elapsed_secs": round(self.elapsed, 2),
            "error":        self.error,
            "events":       serialised,
        }


# ── Supervisor prompt ──────────────────────────────────────────────────────────

SUPERVISOR_SYSTEM = """You are the Supervisor of the Crucible multi-agent ML platform.

You coordinate three specialist agents:
  dataset_analyst  — discovers datasets, analyses schema, runs profiling and anomaly detection
  model_trainer    — submits AutoML experiments, monitors training, retrieves results
  deployer         — generates deployment packages (FastAPI + Docker + K8s), exports ONNX

Your job: read the user's goal and the conversation history, then decide which specialist to run next, or whether the goal is complete.

ROUTING RULES:
1. Always run dataset_analyst first to confirm data availability.
2. Run model_trainer only after dataset_analyst has confirmed a dataset_id in context.
3. Run deployer only after model_trainer reports training_done = true.
4. Route to FINISH when the goal is complete and you can provide a full summary.
5. If a specialist returns an error, route back to the same specialist with context, or FINISH with explanation.

You MUST respond with ONLY valid JSON. No markdown, no explanation outside the JSON:
{
  "next": "<dataset_analyst | model_trainer | deployer | FINISH>",
  "reasoning": "<one sentence explaining your decision>",
  "final_answer": "<complete summary for the user, only if next == FINISH>"
}"""


# ── Supervisor node ────────────────────────────────────────────────────────────

async def supervisor_node(state: MultiAgentState) -> dict:
    """
    The supervisor reads the full state and decides which specialist to invoke next.
    Returns a partial state update with next_agent set.
    """
    if state.get("step_count", 0) >= MAX_STEPS:
        return {
            "next_agent":   ROUTE_END,
            "final_answer": f"Reached step limit ({MAX_STEPS}). Stopping.",
            "step_count":   state.get("step_count", 0) + 1,
        }

    # Build a context summary for Claude
    ctx = state.get("context", {})
    ctx_summary = json.dumps({k: v for k, v in ctx.items() if v}, indent=2)

    # Build the user message
    history = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in state.get("messages", [])[-6:]    # last 3 turns to keep prompt short
        if isinstance(m, dict) and "content" in m
    )
    user_msg = (
        f"GOAL: {state['goal']}\n\n"
        f"CURRENT CONTEXT:\n{ctx_summary}\n\n"
        f"RECENT HISTORY:\n{history or '(none yet)'}\n\n"
        "What should happen next?"
    )

    # Mock mode — use api_key from state if available, fall back to settings
    api_key = (
        state.get("_api_key") or
        getattr(settings, "anthropic_api_key", "") or
        ""
    )
    if not api_key or api_key.startswith("mock"):
        decision = _mock_supervisor(ctx, state.get("goal", ""))
    else:
        decision = await _call_supervisor(api_key, user_msg)

    next_agent = decision.get("next", ROUTE_END)
    if next_agent == "FINISH":
        next_agent = ROUTE_END

    return {
        "next_agent":   next_agent,
        "active_agent": ROUTE_SUPERVISOR,
        "step_count":   state.get("step_count", 0) + 1,
        "final_answer": decision.get("final_answer", ""),
        "messages":     [{"role": "assistant", "content":
                          f"[Supervisor] → {next_agent}: {decision.get('reasoning', '')}"}],
    }


async def _call_supervisor(api_key: str, user_msg: str) -> dict:
    """Calls Claude Haiku for routing decisions."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model":       "claude-haiku-4-5-20251001",
                    "max_tokens":  300,
                    "temperature": 0,
                    "system":      SUPERVISOR_SYSTEM,
                    "messages":    [{"role": "user", "content": user_msg}],
                },
            )
            resp.raise_for_status()
            text = resp.json()["content"][0]["text"].strip()
            # Strip markdown code fences if present
            text = re.sub(r"^```json\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            return json.loads(text)
    except Exception as exc:
        return {"next": ROUTE_END, "reasoning": f"Supervisor error: {exc}", "final_answer": f"Supervisor error: {exc}"}


def _mock_supervisor(ctx: AgentContext, goal: str) -> dict:
    """
    Rule-based supervisor for testing — no API call required.
    Mirrors the real supervisor's routing logic deterministically.
    """
    if not ctx.get("dataset_id"):
        return {"next": "dataset_analyst", "reasoning": "No dataset discovered yet."}
    if not ctx.get("training_done") and ("train" in goal.lower() or "model" in goal.lower() or "predict" in goal.lower()):
        return {"next": "model_trainer", "reasoning": "Dataset ready, starting training."}
    if ctx.get("training_done") and not ctx.get("deployment_done") and "deploy" in goal.lower():
        return {"next": "deployer", "reasoning": "Training complete, generating deployment."}
    answer = _build_summary(ctx)
    return {"next": "FINISH", "reasoning": "Goal complete.", "final_answer": answer}


def _build_summary(ctx: AgentContext) -> str:
    parts = ["Multi-agent workflow complete."]
    if ctx.get("dataset_name"):
        parts.append(f"Dataset: {ctx['dataset_name']} (ID {ctx.get('dataset_id')})")
    if ctx.get("best_model"):
        parts.append(f"Best model: {ctx['best_model']} (score: {ctx.get('best_score', 'unknown')})")
    if ctx.get("deployment_url"):
        parts.append(f"Deployment: {ctx['deployment_url']}")
    return " | ".join(parts)


# ── Graph routing ──────────────────────────────────────────────────────────────

def route_from_supervisor(state: MultiAgentState) -> str:
    """Conditional edge: read next_agent from state and return the node name."""
    next_agent = state.get("next_agent", ROUTE_END)
    if next_agent == ROUTE_END or next_agent == "FINISH":
        return END
    return next_agent


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_multi_agent_graph(db):
    """
    Constructs the LangGraph StateGraph.

    The graph has:
      - One supervisor node (calls Claude for routing)
      - Three specialist nodes (deterministic, no LLM calls)
      - Conditional edges from supervisor to specialists (or END)
      - Unconditional edges from specialists back to supervisor

    db is injected at build time (once per session) and shared by all nodes
    via closure. This avoids passing the DB session through LangGraph state.
    """

    async def _dataset_analyst(state: MultiAgentState) -> dict:
        return await dataset_analyst_node(state, db)

    async def _model_trainer(state: MultiAgentState) -> dict:
        return await model_trainer_node(state, db)

    async def _deployer(state: MultiAgentState) -> dict:
        return await deployer_node(state, db)

    g = StateGraph(MultiAgentState)

    # Add nodes
    g.add_node(ROUTE_SUPERVISOR,      supervisor_node)
    g.add_node(ROUTE_DATASET_ANALYST, _dataset_analyst)
    g.add_node(ROUTE_MODEL_TRAINER,   _model_trainer)
    g.add_node(ROUTE_DEPLOYER,        _deployer)

    # Entry point
    g.set_entry_point(ROUTE_SUPERVISOR)

    # Supervisor → specialists (conditional) or END
    g.add_conditional_edges(
        ROUTE_SUPERVISOR,
        route_from_supervisor,
        {
            ROUTE_DATASET_ANALYST: ROUTE_DATASET_ANALYST,
            ROUTE_MODEL_TRAINER:   ROUTE_MODEL_TRAINER,
            ROUTE_DEPLOYER:        ROUTE_DEPLOYER,
            END:                   END,
        },
    )

    # Specialists always return to supervisor
    for specialist in ALL_SPECIALISTS:
        g.add_edge(specialist, ROUTE_SUPERVISOR)

    return g.compile()


# ── Runner ─────────────────────────────────────────────────────────────────────

class MultiAgentRunner:
    """Runs a multi-agent session and collects events."""

    def __init__(self, db, api_key: str = ""):
        self.db      = db
        self._api_key = api_key

    async def run(self, goal: str) -> MultiAgentSession:
        session = MultiAgentSession(goal=goal)
        start = time.monotonic()

        initial_state: MultiAgentState = {
            "messages":     [{"role": "user", "content": goal}],
            "goal":         goal,
            "active_agent": ROUTE_SUPERVISOR,
            "next_agent":   ROUTE_SUPERVISOR,
            "step_count":   0,
            "context":      {},
            "agent_output": "",
            "final_answer": "",
            "error":        "",
            "_api_key":     self._api_key or "",   # threads through to supervisor_node
        }

        try:
            graph = build_multi_agent_graph(self.db)
            async for chunk in graph.astream(initial_state):
                for node_name, node_state in chunk.items():
                    if node_name == ROUTE_SUPERVISOR:
                        session.events.append(SupervisorEvent(
                            reasoning=_extract_reasoning(node_state),
                            routing_to=node_state.get("next_agent", "?"),
                        ))
                        if node_state.get("final_answer"):
                            session.final_answer = node_state["final_answer"]
                    elif node_name in ALL_SPECIALISTS:
                        session.events.append(SpecialistEvent(
                            agent=node_name,
                            output=node_state.get("agent_output", ""),
                        ))
                    session.steps = node_state.get("step_count", session.steps)

        except Exception as exc:
            session.error = str(exc)
            session.events.append(ErrorEvent(message=str(exc)))

        session.elapsed = time.monotonic() - start
        if not session.final_answer and not session.error:
            session.final_answer = "Workflow complete."

        session.events.append(FinishedEvent(
            answer=session.final_answer,
            steps=session.steps,
            elapsed=session.elapsed,
        ))
        return session

    async def stream(self, goal: str) -> AsyncGenerator[MultiAgentEvent, None]:
        """Yields events as they are produced by the graph."""
        session = await self.run(goal)
        for event in session.events:
            yield event


def _extract_reasoning(node_state: dict) -> str:
    msgs = node_state.get("messages", [])
    for m in reversed(msgs):
        if isinstance(m, dict) and "Supervisor" in str(m.get("content", "")):
            content = m["content"]
            arrow_idx = content.find("→")
            if arrow_idx != -1:
                return content[arrow_idx:].strip()
            return content[:100]
    return ""
