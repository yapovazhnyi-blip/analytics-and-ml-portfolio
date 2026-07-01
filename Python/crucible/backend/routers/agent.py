"""
Agent router — /api/v1/agent

Endpoints:
  POST /agent/run        — run a full agent session, return complete trace
  GET  /agent/tools      — list available tools with descriptions
  WS   /ws/agent/{id}    — stream agent events in real time
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from agents.runner import (
    ReActRunner,
    ThinkingEvent, ToolCallEvent, ToolResultEvent, FinalAnswerEvent, ErrorEvent,
)
from agents.tools import tools_for_api
from auth.dependencies import get_current_user, validate_ws_token
from config import settings
from database import get_db, AsyncSessionLocal
from schemas.common import DataResponse

router    = APIRouter(prefix="/agent", tags=["agent"], dependencies=[Depends(get_current_user)])
ws_router = APIRouter(tags=["agent-ws"])

# In-memory store: session_id → AgentSession (for WebSocket reconnect)
_sessions: dict[str, object] = {}


# ── Schemas ───────────────────────────────────────────────────────────────────

class AgentRunRequest(BaseModel):
    goal: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural language description of what the agent should do.",
        examples=["Analyse dataset 1 and train a classification model on the 'churn' column"],
    )
    capture: bool = Field(
        default=False,
        description=(
            "If true, captures this session as an AgentTrace for the training "
            "pipeline once it completes. See POST /agents/traces/capture."
        ),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/tools")
async def list_tools():
    """Returns all available agent tools with their descriptions and input schemas."""
    return DataResponse(data={"tools": tools_for_api(), "count": len(tools_for_api())})


@router.post("/run")
async def run_agent(
    request: Request,
    body: AgentRunRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Runs a full ReAct agent session synchronously and returns the complete trace.

    The agent iterates up to 10 tool-use steps, calling Crucible's own API
    to analyse datasets, run experiments, and generate deployments.

    Uses the authenticated user's Anthropic API key (BYOK) if stored;
    falls back to the server-level ANTHROPIC_API_KEY.

    Set capture=true to persist this session as an AgentTrace for the
    agent training pipeline (see POST /agents/traces/training-data).
    """
    from auth.key_manager import get_anthropic_key
    api_key = await get_anthropic_key(current_user, require=False) or ""
    runner  = ReActRunner(db, api_key=api_key)
    session = await runner.run(body.goal)
    session_dict = session.to_dict()

    if body.capture:
        from agents.traces import TraceCollector
        await TraceCollector(db).capture_from_dict(session_dict, "react")

    return DataResponse(data=session_dict)


@router.post("/run/stream-id")
async def create_stream_session(
    body: AgentRunRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Creates an agent session and returns a session ID for WebSocket streaming.
    Connect to /ws/agent/{session_id} to receive events as they happen.
    """
    from auth.key_manager import get_anthropic_key
    api_key = await get_anthropic_key(current_user, require=False) or ""

    session_id = f"ag-{uuid.uuid4().hex[:16]}"
    _sessions[session_id] = {"goal": body.goal, "status": "pending", "queue": asyncio.Queue()}

    asyncio.create_task(_run_agent_background(session_id, body.goal, api_key))
    return DataResponse(data={"session_id": session_id})


async def _run_agent_background(session_id: str, goal: str, api_key: str = "") -> None:
    """Runs the agent and pushes events to the session queue."""
    session_data = _sessions.get(session_id)
    if not session_data:
        return

    queue: asyncio.Queue = session_data["queue"]
    session_data["status"] = "running"

    async with AsyncSessionLocal() as db:
        runner = ReActRunner(db, api_key=api_key)
        async for event in runner.stream(goal):
            await queue.put(event)

    session_data["status"] = "done"
    await queue.put(None)  # sentinel


# ── WebSocket streaming ───────────────────────────────────────────────────────

@ws_router.websocket("/ws/agent/{session_id}")
async def agent_stream_ws(
    session_id: str,
    websocket: WebSocket,
    token: Optional[str] = Query(None),
):
    """
    Streams agent events for a session created via POST /agent/run/stream-id.

    Message types:
      {"type": "thinking",     "text": "..."}
      {"type": "tool_call",    "tool": "...", "input": {...}}
      {"type": "tool_result",  "tool": "...", "result": "...", "is_error": false}
      {"type": "final_answer", "text": "..."}
      {"type": "error",        "message": "..."}
    """
    if not settings.disable_auth:
        async with AsyncSessionLocal() as db:
            user = await validate_ws_token(token, db)
        if not user:
            await websocket.close(code=1008)
            return

    await websocket.accept()

    session_data = _sessions.get(session_id)
    if not session_data:
        await websocket.send_json({"type": "error", "message": f"Session '{session_id}' not found."})
        await websocket.close()
        return

    queue: asyncio.Queue = session_data["queue"]

    try:
        while True:
            event = await asyncio.wait_for(queue.get(), timeout=120)
            if event is None:
                break

            if isinstance(event, ThinkingEvent):
                await websocket.send_json({"type": "thinking", "text": event.text})
            elif isinstance(event, ToolCallEvent):
                await websocket.send_json({"type": "tool_call", "tool": event.tool_name, "input": event.tool_input})
            elif isinstance(event, ToolResultEvent):
                await websocket.send_json({"type": "tool_result", "tool": event.tool_name, "result": event.result, "is_error": event.is_error})
            elif isinstance(event, FinalAnswerEvent):
                await websocket.send_json({"type": "final_answer", "text": event.text})
            elif isinstance(event, ErrorEvent):
                await websocket.send_json({"type": "error", "message": event.message})
                break

    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        # Clean up session after a delay
        asyncio.create_task(_cleanup_session(session_id))


async def _cleanup_session(session_id: str, delay: float = 60.0) -> None:
    await asyncio.sleep(delay)
    _sessions.pop(session_id, None)


# ── Multi-agent endpoint ──────────────────────────────────────────────────────

class MultiAgentRequest(BaseModel):
    goal: str = Field(
        ..., min_length=1, max_length=2000,
        description=(
            "Natural language goal for the multi-agent system. "
            "Mention data, modelling, and deployment needs explicitly. "
            "Example: 'Analyse my datasets, train a churn classifier, and deploy it.'"
        ),
        examples=["Analyse my datasets, train a classification model on the 'label' column, and generate a deployment."],
    )
    capture: bool = Field(
        default=False,
        description="If true, persists this session as an AgentTrace for the training pipeline.",
    )


@router.post("/multi/run")
async def run_multi_agent(
    request: Request,
    body: MultiAgentRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Runs a multi-agent workflow using LangGraph Supervisor + Specialist pattern.

    Three specialists collaborate under a supervisor:
      - Dataset Analyst  → discovery, profiling, anomaly detection
      - Model Trainer    → AutoML experiments, results, fairness
      - Deployer         → deployment package, ONNX export

    The supervisor (Claude Haiku) reads the goal and each specialist's output,
    then decides which specialist to invoke next, or when the goal is complete.

    Returns the complete event trace: supervisor reasoning, specialist outputs,
    and the final answer.

    Set capture=true to persist this session as an AgentTrace for the
    agent training pipeline.
    """
    from agents.multi_agent import MultiAgentRunner
    from auth.key_manager import get_anthropic_key
    api_key = await get_anthropic_key(current_user, require=False) or ""
    runner = MultiAgentRunner(db, api_key=api_key)
    session = await runner.run(body.goal)
    session_dict = session.to_dict()

    if body.capture:
        from agents.traces import TraceCollector
        await TraceCollector(db).capture_from_dict(session_dict, "multi")

    return DataResponse(data=session_dict)
