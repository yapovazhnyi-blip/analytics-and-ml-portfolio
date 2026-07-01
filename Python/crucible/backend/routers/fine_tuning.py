"""
Fine-tuning router — /api/v1/fine-tuning

Endpoints:
  POST /fine-tuning/jobs              — submit a new fine-tuning job
  GET  /fine-tuning/jobs              — list all jobs
  GET  /fine-tuning/jobs/{job_id}     — get job status and results
  DELETE /fine-tuning/jobs/{job_id}   — cancel or delete a job
  WS   /ws/fine-tuning/{job_id}       — stream live training progress
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user, validate_ws_token
from config import settings
from database import get_db, AsyncSessionLocal
from fine_tuning.config import FineTuningConfig, LoRAConfig
from fine_tuning.trainer import SFTTrainer, StepEvent, EpochEvent, CompleteEvent, ErrorEvent
from models.fine_tune_job import FineTuneJob
from schemas.common import DataResponse, PaginatedResponse, make_pagination_meta

router = APIRouter(
    prefix="/fine-tuning",
    tags=["fine-tuning"],
    dependencies=[Depends(get_current_user)],
)

ws_router = APIRouter(tags=["fine-tuning-ws"])

# ── Active job queues ─────────────────────────────────────────────────────────
# Maps job_id → asyncio.Queue (same sentinel pattern as AutoML)
_job_queues: dict[str, asyncio.Queue] = {}


# ── Request / response schemas ─────────────────────────────────────────────────

class LoRAConfigIn(BaseModel):
    rank: int           = Field(default=16, ge=1, le=256)
    alpha: int          = Field(default=32, ge=1)
    dropout: float      = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: list[str] = Field(
        default=["q_proj", "v_proj", "k_proj", "o_proj"]
    )


class FineTuningRequest(BaseModel):
    model_id: str = Field(
        ...,
        description="HuggingFace model hub ID, e.g. 'microsoft/phi-2'. "
                    "Use 'mock-phi' in tests to skip model download.",
    )
    samples: list[dict] = Field(
        ...,
        min_length=1,
        max_length=50_000,
        description="Training samples in Alpaca or ShareGPT format.",
    )
    dataset_format: str = Field(
        default="alpaca",
        description="'alpaca' or 'sharegpt'",
    )
    lora: LoRAConfigIn = Field(default_factory=LoRAConfigIn)
    use_qlora: bool    = Field(
        default=False,
        description="4-bit quantisation. Requires bitsandbytes + CUDA GPU.",
    )
    epochs: int            = Field(default=1,    ge=1, le=100)
    learning_rate: float   = Field(default=2e-4, gt=0)
    batch_size: int        = Field(default=4,    ge=1, le=64)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    max_seq_len: int       = Field(default=512,  ge=32, le=32768)
    push_to_hub: bool      = False
    hub_model_id: Optional[str] = None


class FineTuneJobOut(BaseModel):
    job_id: str
    base_model_id: str
    method: str
    dataset_format: str
    n_samples: int
    epochs: int
    learning_rate: float
    batch_size: int
    use_qlora: bool
    status: str
    adapter_path: Optional[str] = None
    final_loss: Optional[float] = None
    total_steps: Optional[int] = None
    elapsed_secs: Optional[float] = None
    hub_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str

    model_config = {"from_attributes": True}


def _job_out(job: FineTuneJob) -> FineTuneJobOut:
    return FineTuneJobOut(
        job_id=job.job_id,
        base_model_id=job.base_model_id,
        method=job.method,
        dataset_format=job.dataset_format,
        n_samples=job.n_samples,
        epochs=job.epochs,
        learning_rate=job.learning_rate,
        batch_size=job.batch_size,
        use_qlora=job.use_qlora,
        status=job.status,
        adapter_path=job.adapter_path,
        final_loss=job.final_loss,
        total_steps=job.total_steps,
        elapsed_secs=job.elapsed_secs,
        hub_url=job.hub_url,
        error_message=job.error_message,
        created_at=job.created_at.isoformat(),
    )


# ── Submit job ────────────────────────────────────────────────────────────────

@router.post("/jobs", response_model=DataResponse[FineTuneJobOut], status_code=201)
async def submit_fine_tuning(
    body: FineTuningRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Submits a fine-tuning job.

    Returns immediately with status='running'. Connect to
    /ws/fine-tuning/{job_id} to stream step-level progress.

    Model download happens on first call for each model_id.
    Use model_id='mock-<anything>' (e.g. 'mock-phi') to skip download
    and run a simulated training loop — useful for testing the WebSocket.
    """
    # Build and validate config
    cfg = FineTuningConfig(
        model_id=body.model_id,
        dataset_format=body.dataset_format,
        method="sft",
        use_qlora=body.use_qlora,
        lora=LoRAConfig(
            rank=body.lora.rank,
            alpha=body.lora.alpha,
            dropout=body.lora.dropout,
            target_modules=body.lora.target_modules,
        ),
        epochs=body.epochs,
        learning_rate=body.learning_rate,
        batch_size=body.batch_size,
        gradient_accumulation_steps=body.gradient_accumulation_steps,
        max_seq_len=body.max_seq_len,
        output_dir=str(Path(settings.model_storage_path) / "fine_tuning"),
        push_to_hub=body.push_to_hub,
        hub_model_id=body.hub_model_id,
    )

    config_errors = cfg.validate()
    if config_errors:
        raise HTTPException(422, detail={"errors": config_errors})

    job_id = f"ft-{uuid.uuid4().hex[:16]}"
    queue: asyncio.Queue = asyncio.Queue()
    _job_queues[job_id] = queue

    # Create DB record
    job = FineTuneJob(
        job_id=job_id,
        base_model_id=body.model_id,
        method="sft",
        dataset_format=body.dataset_format,
        n_samples=len(body.samples),
        lora_config_json=json.dumps({
            "rank": body.lora.rank,
            "alpha": body.lora.alpha,
            "dropout": body.lora.dropout,
            "target_modules": body.lora.target_modules,
        }),
        epochs=body.epochs,
        learning_rate=body.learning_rate,
        batch_size=body.batch_size,
        use_qlora=body.use_qlora,
        status="running",
    )
    db.add(job)
    await db.flush()
    await db.refresh(job)

    # Launch background task
    asyncio.create_task(_train_background(
        job_id=job_id,
        db_id=job.id,
        config=cfg,
        samples=body.samples,
        queue=queue,
    ))

    return DataResponse(data=_job_out(job))


async def _train_background(
    job_id: str,
    db_id: int,
    config: FineTuningConfig,
    samples: list[dict],
    queue: asyncio.Queue,
) -> None:
    """Background coroutine — runs the SFTTrainer and updates the DB record."""
    trainer = SFTTrainer(config=config, job_id=job_id, progress_queue=queue)
    output_dir = str(Path(config.output_dir))

    result = await trainer.run(samples=samples, output_dir=output_dir)

    async with AsyncSessionLocal() as session:
        job = await session.get(FineTuneJob, db_id)
        if job:
            if result.succeeded:
                job.status = "succeeded"
                job.adapter_path = result.adapter_path
                job.final_loss = result.final_loss
                job.total_steps = result.total_steps
                job.elapsed_secs = result.elapsed_secs
                job.hub_url = result.hub_url
            else:
                job.status = "failed"
                job.error_message = result.error
            await session.commit()

    # Clean up the queue after a delay (allow late WS connections to drain)
    await asyncio.sleep(30)
    _job_queues.pop(job_id, None)


# ── List / get / delete ───────────────────────────────────────────────────────

@router.get("/jobs", response_model=PaginatedResponse[FineTuneJobOut])
async def list_jobs(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import func
    stmt = select(FineTuneJob).order_by(FineTuneJob.created_at.desc())
    total = await db.scalar(select(func.count()).select_from(stmt.subquery()))
    rows  = await db.scalars(stmt.offset((page - 1) * page_size).limit(page_size))
    return PaginatedResponse(
        data=[_job_out(j) for j in rows.all()],
        pagination=make_pagination_meta(page, page_size, total or 0),
    )


@router.get("/jobs/{job_id}", response_model=DataResponse[FineTuneJobOut])
async def get_job(job_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(FineTuneJob).where(FineTuneJob.job_id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404, f"Fine-tuning job '{job_id}' not found")
    return DataResponse(data=_job_out(job))


@router.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(FineTuneJob).where(FineTuneJob.job_id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404, f"Fine-tuning job '{job_id}' not found")
    if job.status == "running":
        raise HTTPException(422, "Cannot delete a running job. Wait for it to complete.")
    await db.delete(job)


# ── WebSocket progress stream ─────────────────────────────────────────────────

@ws_router.websocket("/ws/fine-tuning/{job_id}")
async def fine_tuning_progress_ws(
    job_id: str,
    websocket: WebSocket,
    token: Optional[str] = Query(None),
):
    """
    Streams real-time training progress for a fine-tuning job.

    Message format:
      {"type": "step",     "step": 10, "total": 100, "loss": 1.23, "lr": 0.0002, "epoch": 0.5}
      {"type": "complete", "adapter_path": "...", "final_loss": 0.45, "total_steps": 100}
      {"type": "error",    "message": "..."}

    Connection closes automatically when training finishes.
    """
    if not settings.disable_auth:
        async with AsyncSessionLocal() as db:
            user = await validate_ws_token(token, db)
        if not user:
            await websocket.close(code=1008)
            return

    await websocket.accept()

    queue = _job_queues.get(job_id)
    if not queue:
        # Job may have already completed — send a status-not-found message
        await websocket.send_json({"type": "error", "message": f"No active stream for job '{job_id}'"})
        await websocket.close()
        return

    try:
        while True:
            event = await asyncio.wait_for(queue.get(), timeout=300)
            if event is None:   # sentinel
                break

            if isinstance(event, StepEvent):
                await websocket.send_json({
                    "type":        "step",
                    "step":        event.step,
                    "total":       event.total_steps,
                    "epoch":       round(event.epoch, 3),
                    "loss":        round(event.loss, 4),
                    "lr":          event.learning_rate,
                    "elapsed":     round(event.elapsed_secs, 1),
                })
            elif isinstance(event, CompleteEvent):
                await websocket.send_json({
                    "type":         "complete",
                    "adapter_path": event.adapter_path,
                    "total_steps":  event.total_steps,
                    "final_loss":   round(event.final_loss, 4),
                    "elapsed":      round(event.elapsed_secs, 1),
                    "hub_url":      event.hub_url,
                })
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


# ── DPO endpoint ──────────────────────────────────────────────────────────────

class DPORequest(BaseModel):
    model_id: str = Field(
        ...,
        description="HuggingFace model hub ID. Use 'mock-phi' to test without download.",
    )
    samples: list[dict] = Field(
        ..., min_length=1, max_length=50_000,
        description="Preference pairs: [{prompt, chosen, rejected}]",
    )
    sft_model_path: Optional[str] = Field(
        None,
        description="Path to a prior SFT adapter to use as the reference model. "
                    "If None, the base model is its own reference.",
    )
    beta: float = Field(default=0.1, gt=0.0, le=2.0,
                        description="KL penalty. Lower = stays closer to reference model.")
    lora: Optional[LoRAConfigIn] = None
    epochs: int          = Field(default=1, ge=1, le=50)
    learning_rate: float = Field(default=5e-5, gt=0)
    batch_size: int      = Field(default=2, ge=1, le=32)


@router.post("/jobs/dpo", response_model=DataResponse[FineTuneJobOut], status_code=201)
async def submit_dpo(
    body: DPORequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Submits a DPO (Direct Preference Optimisation) fine-tuning job.

    DPO trains the model to prefer "chosen" responses over "rejected" ones
    without a separate reward model. Each training sample must contain:
      - prompt:   The input question or context
      - chosen:   The preferred (better) response
      - rejected: The dispreferred (worse) response

    Returns immediately. Use the standard /fine-tuning/jobs/{job_id} endpoint
    to poll status and retrieve results.

    Use model_id='mock-<anything>' to run a simulated training loop without
    downloading a model — useful for testing the full pipeline.
    """
    from fine_tuning.config import DPOConfig, LoRAConfig as LRC
    from fine_tuning.dpo_trainer import DPOTrainer, validate_dpo_samples

    # Validate samples
    errors = validate_dpo_samples(body.samples)
    if errors:
        raise HTTPException(422, detail={"errors": errors})

    lora_in = body.lora or LoRAConfigIn()
    cfg = DPOConfig(
        model_id=body.model_id,
        sft_model_path=body.sft_model_path,
        beta=body.beta,
        lora=LRC(
            rank=lora_in.rank,
            alpha=lora_in.alpha,
            dropout=lora_in.dropout,
            target_modules=lora_in.target_modules,
        ),
        epochs=body.epochs,
        learning_rate=body.learning_rate,
        batch_size=body.batch_size,
        output_dir=str(Path(settings.model_storage_path) / "fine_tuning"),
    )
    config_errors = cfg.validate()
    if config_errors:
        raise HTTPException(422, detail={"errors": config_errors})

    job_id = f"dpo-{uuid.uuid4().hex[:14]}"
    queue: asyncio.Queue = asyncio.Queue()
    _job_queues[job_id] = queue

    job = FineTuneJob(
        job_id=job_id,
        base_model_id=body.model_id,
        method="dpo",
        dataset_format="dpo",
        n_samples=len(body.samples),
        lora_config_json=json.dumps({
            "rank": lora_in.rank, "alpha": lora_in.alpha,
            "dropout": lora_in.dropout, "beta": body.beta,
        }),
        epochs=body.epochs,
        learning_rate=body.learning_rate,
        batch_size=body.batch_size,
        use_qlora=False,
        status="running",
    )
    db.add(job)
    await db.flush()
    await db.refresh(job)

    asyncio.create_task(_dpo_background(
        job_id=job_id,
        db_id=job.id,
        config=cfg,
        samples=body.samples,
        queue=queue,
    ))
    return DataResponse(data=_job_out(job))


async def _dpo_background(
    job_id: str, db_id: int, config, samples: list[dict], queue: asyncio.Queue
) -> None:
    from fine_tuning.dpo_trainer import DPOTrainer
    from database import AsyncSessionLocal
    trainer = DPOTrainer(config=config, job_id=job_id, progress_queue=queue)
    result = await trainer.run(samples=samples, output_dir=config.output_dir)
    async with AsyncSessionLocal() as session:
        job = await session.get(FineTuneJob, db_id)
        if job:
            if result.succeeded:
                job.status = "succeeded"
                job.adapter_path = result.adapter_path
                job.final_loss = result.final_loss
                job.total_steps = result.total_steps
                job.elapsed_secs = result.elapsed_secs
            else:
                job.status = "failed"
                job.error_message = result.error
            await session.commit()
    await asyncio.sleep(30)
    _job_queues.pop(job_id, None)
