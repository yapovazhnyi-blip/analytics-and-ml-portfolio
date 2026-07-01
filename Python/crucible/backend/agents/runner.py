"""
ReAct Agent Runner — Reason + Act loop using Anthropic's native tool_use API.

HOW IT WORKS
------------
Unlike the classic ReAct pattern that parses "Thought/Action/Observation"
from plain text, this implementation uses Anthropic's structured tool_use
API, which:
  1. Claude receives a list of tool schemas
  2. When Claude decides to use a tool, the API returns a `tool_use` block
     with the tool name and structured inputs — no text parsing needed
  3. We execute the tool and return a `tool_result` block
  4. The loop continues until Claude returns a final `text` block without
     any tool_use blocks (it's done reasoning)

ADVANTAGES OVER TEXT PARSING
------------------------------
- No brittle regex to extract "Action: [tool]" from generated text
- Structured inputs — Claude always passes valid JSON inputs
- Native API support — Anthropic optimises for this pattern

MAX STEPS
---------
Hard limit of 10 tool calls per agent session. This prevents:
  a) Runaway loops (a bug causes the agent to keep calling get_experiment_status)
  b) Excessive API costs ($0.001 × 10 = $0.01 max per session with Haiku)

The agent can complete most ML workflows in 3-6 steps:
  1. list_datasets
  2. run_profiling
  3. start_experiment
  4. get_experiment_status (may call 2-3 times while polling)
  5. get_experiment_results
  6. generate_deployment

SYSTEM PROMPT DESIGN
---------------------
The system prompt gives the agent:
  1. Its role and what Crucible is
  2. Explicit guidance on when to use each tool category
  3. Sequencing rules ("always run_profiling before start_experiment")
  4. Format expectations (concise final answers, not verbose tool summaries)

Temperature = 0 for reproducible, deterministic tool selection.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator

import httpx

from agents.tools import ToolExecutor, tools_for_api, ToolResult
from config import settings


# ── Event types for streaming ─────────────────────────────────────────────────

@dataclass
class ThinkingEvent:
    text: str

@dataclass
class ToolCallEvent:
    tool_name: str
    tool_input: dict

@dataclass
class ToolResultEvent:
    tool_name: str
    result: str
    is_error: bool

@dataclass
class FinalAnswerEvent:
    text: str

@dataclass
class ErrorEvent:
    message: str


AgentEvent = ThinkingEvent | ToolCallEvent | ToolResultEvent | FinalAnswerEvent | ErrorEvent


# ── Session result ────────────────────────────────────────────────────────────

@dataclass
class AgentSession:
    goal: str
    events: list[AgentEvent] = field(default_factory=list)
    final_answer: str = ""
    n_tool_calls: int = 0
    elapsed_secs: float = 0.0
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and bool(self.final_answer)

    def to_dict(self) -> dict:
        events = []
        for e in self.events:
            if isinstance(e, ThinkingEvent):
                events.append({"type": "thinking", "text": e.text})
            elif isinstance(e, ToolCallEvent):
                events.append({"type": "tool_call", "tool": e.tool_name, "input": e.tool_input})
            elif isinstance(e, ToolResultEvent):
                events.append({"type": "tool_result", "tool": e.tool_name, "result": e.result, "is_error": e.is_error})
            elif isinstance(e, FinalAnswerEvent):
                events.append({"type": "final_answer", "text": e.text})
            elif isinstance(e, ErrorEvent):
                events.append({"type": "error", "message": e.message})
        return {
            "goal":          self.goal,
            "final_answer":  self.final_answer,
            "n_tool_calls":  self.n_tool_calls,
            "elapsed_secs":  round(self.elapsed_secs, 2),
            "error":         self.error,
            "events":        events,
        }


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Crucible Agent — an expert ML engineer assistant with direct access to the Crucible ML platform.

Crucible is an end-to-end ML experimentation platform. You can use tools to:
- Discover and analyse datasets
- Run deep data profiling (missing values, leakage detection, distributions)
- Train AutoML experiments across 8 model families with Optuna hyperparameter search
- Query indexed documents using RAG
- Generate production deployment packages

TOOL SEQUENCING RULES:
1. Always call list_datasets first when the user mentions a dataset by name — confirm the ID.
2. Always call run_profiling before start_experiment — catch data quality issues early.
3. After start_experiment, poll get_experiment_status every step until status is 'completed'.
4. Only call get_experiment_results and generate_deployment when status is 'completed'.
5. For document questions, call list_rag_documents first to confirm documents exist.

COMMUNICATION STYLE:
- Be concise. Don't repeat tool outputs verbatim — summarise the key findings.
- Flag any data quality warnings (missing data, leakage) before proceeding.
- State what you're doing before each tool call.
- When finished, give a clear summary of what was accomplished and any next steps.
- If a step fails, explain why and suggest an alternative approach.

LIMITS:
- You have a maximum of 10 tool calls per session. Use them efficiently.
- If a task would require more than 10 calls, focus on the most important steps first."""


# ── ReAct Runner ─────────────────────────────────────────────────────────────

class ReActRunner:
    """Executes the ReAct reasoning loop using Anthropic's tool_use API."""

    MAX_STEPS = 10
    MODEL     = "claude-haiku-4-5-20251001"

    def __init__(self, db, api_key: str = ""):
        self.db      = db
        self.executor = ToolExecutor(db)
        self._api_key = api_key   # user-resolved key; falls back to settings below

    async def run(self, goal: str) -> AgentSession:
        """
        Runs the full ReAct loop and returns an AgentSession with all events.
        Non-streaming version — collects all events then returns.
        """
        session = AgentSession(goal=goal)
        start = time.monotonic()

        async for event in self._stream(goal):
            session.events.append(event)
            if isinstance(event, FinalAnswerEvent):
                session.final_answer = event.text
            elif isinstance(event, ToolCallEvent):
                session.n_tool_calls += 1
            elif isinstance(event, ErrorEvent):
                session.error = event.message

        session.elapsed_secs = time.monotonic() - start
        return session

    async def stream(self, goal: str) -> AsyncGenerator[AgentEvent, None]:
        """Streaming version — yields events as they occur."""
        async for event in self._stream(goal):
            yield event

    async def _stream(self, goal: str) -> AsyncGenerator[AgentEvent, None]:
        if not settings.anthropic_api_key and not self._api_key:
            yield ErrorEvent("ANTHROPIC_API_KEY not configured.")
            return

        api_key = self._api_key or settings.anthropic_api_key

        messages = [{"role": "user", "content": goal}]
        tools = tools_for_api()
        step = 0

        async with httpx.AsyncClient(timeout=60.0) as client:
            while step < self.MAX_STEPS:
                step += 1

                # ── Call Claude ───────────────────────────────────────────────
                try:
                    resp = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                        },
                        json={
                            "model":      self.MODEL,
                            "max_tokens": 2000,
                            "temperature": 0,
                            "system":     SYSTEM_PROMPT,
                            "tools":      tools,
                            "messages":   messages,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except httpx.HTTPStatusError as exc:
                    yield ErrorEvent(f"Claude API error {exc.response.status_code}: {exc.response.text[:200]}")
                    return
                except Exception as exc:
                    yield ErrorEvent(f"Network error: {exc}")
                    return

                content_blocks = data.get("content", [])
                stop_reason    = data.get("stop_reason", "")

                # ── Emit text blocks ──────────────────────────────────────────
                for block in content_blocks:
                    if block.get("type") == "text" and block.get("text"):
                        yield ThinkingEvent(text=block["text"])

                # ── Check if done ─────────────────────────────────────────────
                tool_uses = [b for b in content_blocks if b.get("type") == "tool_use"]

                if not tool_uses or stop_reason == "end_turn":
                    # No tool calls — extract final answer from text blocks
                    final = " ".join(
                        b["text"] for b in content_blocks
                        if b.get("type") == "text" and b.get("text")
                    ).strip()
                    yield FinalAnswerEvent(text=final or "Task completed.")
                    return

                # ── Execute tool calls ────────────────────────────────────────
                # Add the assistant's turn (with tool_use blocks) to history
                messages.append({"role": "assistant", "content": content_blocks})

                tool_results = []
                for tool_use_block in tool_uses:
                    tool_name  = tool_use_block["name"]
                    tool_input = tool_use_block.get("input", {})
                    tool_id    = tool_use_block["id"]

                    yield ToolCallEvent(tool_name=tool_name, tool_input=tool_input)

                    result: ToolResult = await self.executor.execute(tool_name, tool_input)

                    yield ToolResultEvent(
                        tool_name=tool_name,
                        result=result.content,
                        is_error=result.is_error,
                    )

                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": tool_id,
                        "content":     result.content,
                        "is_error":    result.is_error,
                    })

                # Add tool results to conversation so Claude can read them
                messages.append({"role": "user", "content": tool_results})

        # Reached max steps
        yield FinalAnswerEvent(
            text=f"Reached the maximum of {self.MAX_STEPS} tool calls. "
                 "The task may require more steps — try a more specific goal."
        )
