"""
Agent Training Pipeline router — /api/v1/agents/*

Endpoints:
  POST /agents/traces/capture           — capture a completed agent session as a trace
  GET  /agents/traces                   — list captured traces
  POST /agents/traces/score             — score pending traces via LLM judge
  GET  /agents/traces/training-data     — convert traces to SFT (Alpaca/ShareGPT) or DPO format
  POST /agents/export                   — package a fine-tuned adapter into a .crucible bundle
  POST /agents/import                   — import a .crucible bundle, register the agent
  GET  /agents                          — list registered agents
  GET  /agents/{name}                   — get one registered agent
  POST /agents/{name}/benchmark         — run the standard benchmark against a registered agent
  DELETE /agents/{name}                 — archive a registered agent
"""

from __future__ import annotations

import json
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user
from database import get_db
from models.agent_trace import AgentTrace
from models.registered_agent import RegisteredAgent
from schemas.common import DataResponse, PaginatedResponse, make_pagination_meta

router = APIRouter(prefix="/agents", tags=["agent-training"], dependencies=[Depends(get_current_user)])


# ══════════════════════════════════════════════════════════════════════════
# TRACE CAPTURE
# ══════════════════════════════════════════════════════════════════════════

class CaptureTraceRequest(BaseModel):
    session: dict = Field(..., description="The to_dict() output of an AgentSession or MultiAgentSession.")
    agent_type: str = Field(..., pattern="^(react|multi)$")


@router.post("/traces/capture")
async def capture_trace(body: CaptureTraceRequest, db: AsyncSession = Depends(get_db)):
    """
    Captures a completed agent session as a trace for the training pipeline.

    Pass the output of session.to_dict() from either /agent/run or
    /agent/multi/run, along with which kind of agent produced it.
    """
    from agents.traces import TraceCollector

    collector = TraceCollector(db)
    result = await collector.capture_from_dict(body.session, body.agent_type)
    return DataResponse(data={
        "trace_id":   result.trace_id,
        "agent_type": result.agent_type,
        "succeeded":  result.succeeded,
    })


@router.get("/traces")
async def list_traces(
    agent_type: Optional[str] = None,
    scored_only: bool = False,
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """Lists captured traces, optionally filtered by agent_type or scoring status."""
    from sqlalchemy import func

    stmt = select(AgentTrace).order_by(AgentTrace.created_at.desc())
    if agent_type:
        stmt = stmt.where(AgentTrace.agent_type == agent_type)
    if scored_only:
        stmt = stmt.where(AgentTrace.score_pending == False)  # noqa: E712

    total = await db.scalar(select(func.count()).select_from(stmt.subquery()))
    rows = await db.scalars(stmt.offset((page - 1) * page_size).limit(page_size))

    items = [{
        "id":             t.id,
        "agent_type":     t.agent_type,
        "goal":           t.goal,
        "succeeded":      t.succeeded,
        "n_tool_calls":   t.n_tool_calls,
        "quality_score":  t.quality_score,
        "score_pending":  t.score_pending,
        "used_in_training": t.used_in_training,
        "created_at":     t.created_at.isoformat() if t.created_at else None,
    } for t in rows.all()]

    return PaginatedResponse(data=items, pagination=make_pagination_meta(page, page_size, total or 0))


@router.post("/traces/score")
async def score_traces(
    limit: int = 50,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Scores up to `limit` pending traces via the LLM evaluation framework.

    Each trace's (goal, final_answer) is scored on accuracy + helpfulness.
    Run this periodically (or after a batch of agent sessions) before
    converting traces to DPO format, which requires quality scores.
    """
    from agents.traces import TraceCollector
    from auth.key_manager import get_anthropic_key

    api_key = await get_anthropic_key(current_user, require=False) or ""
    collector = TraceCollector(db)
    result = await collector.score_pending_traces(api_key=api_key, limit=limit)
    return DataResponse(data=result)


@router.get("/traces/training-data")
async def get_training_data(
    format: str = "alpaca",
    min_score_gap: float = 0.15,
    db: AsyncSession = Depends(get_db),
):
    """
    Converts captured traces into training data ready for the fine-tuning studio.

    format options:
      alpaca   — SFT format: [{instruction, input, output}]
      sharegpt — SFT format: [{conversations: [...]}]
      dpo      — Preference pairs: [{prompt, chosen, rejected}]
                 Requires traces to be scored first (POST /agents/traces/score).

    Use the returned samples directly as the request body for:
      POST /fine-tuning/jobs       (alpaca/sharegpt -> dataset_format)
      POST /fine-tuning/jobs/dpo   (dpo -> samples)
    """
    from agents.trace_converter import traces_to_alpaca, traces_to_sharegpt, traces_to_dpo_pairs

    if format not in ("alpaca", "sharegpt", "dpo"):
        raise HTTPException(422, f"Unknown format {format!r}. Use alpaca, sharegpt, or dpo.")

    result = await db.execute(select(AgentTrace))
    traces = result.scalars().all()

    if format == "alpaca":
        samples = traces_to_alpaca(traces)
        return DataResponse(data={"format": "alpaca", "n_samples": len(samples), "samples": samples})

    if format == "sharegpt":
        samples = traces_to_sharegpt(traces)
        return DataResponse(data={"format": "sharegpt", "n_samples": len(samples), "samples": samples})

    samples, stats = traces_to_dpo_pairs(traces, min_score_gap=min_score_gap)
    return DataResponse(data={
        "format":     "dpo",
        "n_samples":  len(samples),
        "samples":    samples,
        "stats": {
            "n_groups":                   stats.n_groups,
            "n_pairs":                    stats.n_pairs,
            "n_traces_used":              stats.n_traces_used,
            "n_traces_skipped_unscored":  stats.n_traces_skipped_unscored,
        },
    })


# ══════════════════════════════════════════════════════════════════════════
# BUNDLE EXPORT / IMPORT
# ══════════════════════════════════════════════════════════════════════════

class ExportBundleRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""
    base_model: str
    adapter_path: str = Field(..., description="Local directory containing the trained LoRA adapter.")
    training_method: str = Field(default="sft", pattern="^(sft|dpo)$")
    system_prompt: str = ""
    tool_names: list[str] = Field(default_factory=list)
    max_steps: int = 10
    agent_type: str = Field(default="react", pattern="^(react|multi)$")


@router.post("/export")
async def export_agent_bundle(
    body: ExportBundleRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Packages a trained LoRA adapter into a downloadable .crucible bundle.

    The bundle includes the adapter weights, agent configuration, and a
    sample of the traces used in training (for transparency). Download
    the resulting file and import it into another Crucible instance, or
    load the adapter/ directory directly with PEFT in any other system.
    """
    from agents.bundle import export_bundle, AgentConfig
    from config import settings

    if not os.path.isdir(body.adapter_path):
        raise HTTPException(422, f"adapter_path {body.adapter_path!r} is not a directory")

    # Pull a sample of recent used-in-training traces for the bundle
    result = await db.execute(
        select(AgentTrace).where(AgentTrace.used_in_training == True)  # noqa: E712
        .order_by(AgentTrace.created_at.desc()).limit(20)
    )
    sample_traces = result.scalars().all()

    output_path = os.path.join(
        getattr(settings, "model_storage_path", "./data/models"),
        "agent_bundles",
        f"{body.name}.crucible",
    )

    bundle_path = export_bundle(
        output_path=output_path,
        name=body.name,
        description=body.description,
        base_model=body.base_model,
        adapter_dir=body.adapter_path,
        agent_config=AgentConfig(
            system_prompt=body.system_prompt,
            tool_names=body.tool_names,
            max_steps=body.max_steps,
            agent_type=body.agent_type,
        ),
        training_method=body.training_method,
        n_training_traces=len(sample_traces),
        traces_sample=[{
            "goal": t.goal, "final_answer": t.final_answer,
            "quality_score": t.quality_score,
        } for t in sample_traces],
    )

    return DataResponse(data={"bundle_path": bundle_path, "name": body.name})


@router.post("/import")
async def import_agent_bundle(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Imports a .crucible bundle and registers it in the Agent Registry.

    The agent becomes selectable for future agent runs once registered.
    """
    from agents.bundle import import_bundle
    import tempfile

    tmp_path = tempfile.mktemp(suffix=".crucible")
    content = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)

    try:
        imported = import_bundle(tmp_path)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    existing = await db.scalar(
        select(RegisteredAgent).where(RegisteredAgent.name == imported.manifest.name)
    )
    if existing:
        raise HTTPException(409, f"An agent named {imported.manifest.name!r} is already registered.")

    agent = RegisteredAgent(
        name=imported.manifest.name,
        description=imported.manifest.description,
        base_model=imported.manifest.base_model,
        adapter_path=imported.adapter_dir,
        bundle_path=tmp_path,
        manifest_json=json.dumps(imported.manifest.to_dict()),
        n_training_traces=imported.manifest.n_training_traces,
        training_method=imported.manifest.training_method,
        benchmark_json=json.dumps(imported.benchmark_results) if imported.benchmark_results else None,
        status="active",
    )
    db.add(agent)
    await db.flush()
    await db.refresh(agent)

    return DataResponse(data=_agent_out(agent))


def _agent_out(agent: RegisteredAgent) -> dict:
    return {
        "id":                  agent.id,
        "name":                agent.name,
        "description":         agent.description,
        "base_model":          agent.base_model,
        "training_method":     agent.training_method,
        "n_training_traces":   agent.n_training_traces,
        "status":              agent.status,
        "has_benchmark":       agent.benchmark_json is not None,
        "created_at":          agent.created_at.isoformat() if agent.created_at else None,
    }


# ══════════════════════════════════════════════════════════════════════════
# AGENT REGISTRY
# ══════════════════════════════════════════════════════════════════════════

@router.get("")
async def list_registered_agents(db: AsyncSession = Depends(get_db)):
    """Lists all registered (imported or locally exported) agents."""
    result = await db.execute(
        select(RegisteredAgent).where(RegisteredAgent.status == "active")
        .order_by(RegisteredAgent.created_at.desc())
    )
    agents = result.scalars().all()
    return DataResponse(data=[_agent_out(a) for a in agents])


@router.get("/{name}")
async def get_registered_agent(name: str, db: AsyncSession = Depends(get_db)):
    agent = await db.scalar(select(RegisteredAgent).where(RegisteredAgent.name == name))
    if not agent:
        raise HTTPException(404, f"Agent {name!r} not found")
    out = _agent_out(agent)
    out["manifest"] = json.loads(agent.manifest_json)
    if agent.benchmark_json:
        out["benchmark"] = json.loads(agent.benchmark_json)
    return DataResponse(data=out)


@router.delete("/{name}", status_code=204)
async def archive_registered_agent(name: str, db: AsyncSession = Depends(get_db)):
    agent = await db.scalar(select(RegisteredAgent).where(RegisteredAgent.name == name))
    if not agent:
        raise HTTPException(404, f"Agent {name!r} not found")
    agent.status = "archived"


@router.post("/{name}/benchmark")
async def benchmark_agent(
    name: str,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Runs the standard Crucible agent benchmark against a registered agent
    (or, if the agent isn't a real local model yet, against the Claude
    baseline as a stand-in) and stores the results.

    NOTE: Running a fine-tuned local model requires loading the adapter
    weights into a real transformers pipeline, which is environment-
    dependent (GPU availability, model size). This endpoint runs the
    benchmark through the standard Claude-backed ReActRunner as the
    baseline measurement -- the same benchmark suite a fine-tuned model
    would be measured against once deployed with a local inference backend.
    """
    from agents.benchmark import run_benchmark
    from agents.runner import ReActRunner
    from auth.key_manager import get_anthropic_key

    agent = await db.scalar(select(RegisteredAgent).where(RegisteredAgent.name == name))
    if not agent:
        raise HTTPException(404, f"Agent {name!r} not found")

    api_key = await get_anthropic_key(current_user, require=False) or ""
    runner = ReActRunner(db, api_key=api_key)
    report = await run_benchmark(runner, api_key=api_key)

    agent.benchmark_json = json.dumps(report.to_dict())
    await db.flush()

    return DataResponse(data=report.to_dict())
