"""
Job manager for Crucible Phase 2.

Manages background training jobs. Each job:
  - Runs TrainingRunner in a thread executor (non-blocking for FastAPI)
  - Reports progress via asyncio.Queue (sentinel-based, from WebSocket spike)
  - Stores final state so clients can poll even after WebSocket disconnects

Design: in-memory for Phase 1. Phase 2 could persist job state to Redis
or the database for durability across restarts. The interface is the same
either way — callers use start_job() and get_job() only.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Optional

import pandas as pd

from training.runner import (
    ProgressReporter,
    TrainingConfig,
    TrainingResult,
    TrainingRunner,
    TrainingError,
)
from explainability.shap_runner import SHAPRunner
from config import settings


class JobStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    COMPLETE = "complete"
    ERROR    = "error"


@dataclass
class Job:
    job_id: str
    experiment_id: int
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[TrainingResult] = None
    shap_importance: Optional[list[dict]] = None
    error: Optional[str] = None
    progress_messages: list[dict] = field(default_factory=list)


_SENTINEL = object()

# Module-level job registry — maps job_id → Job
_jobs: dict[str, Job] = {}


def get_job(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)


def list_jobs_for_experiment(experiment_id: int) -> list[Job]:
    return [j for j in _jobs.values() if j.experiment_id == experiment_id]


async def start_job(
    experiment_id: int,
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    config: TrainingConfig,
    feature_names: list[str],
    run_shap: bool = True,
) -> str:
    """
    Kicks off a training job in the background.

    Returns job_id immediately — caller stores this in the Experiment record.
    Training runs in a thread pool, progress flows via asyncio.Queue.
    """
    job_id = uuid.uuid4().hex
    job = Job(job_id=job_id, experiment_id=experiment_id)
    _jobs[job_id] = job

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    reporter = ProgressReporter(loop, queue)

    # Launch training in background thread
    async def run():
        job.status = JobStatus.RUNNING
        job.started_at = time.time()

        try:
            runner = TrainingRunner(model_storage_path=settings.model_storage_path)
            result: TrainingResult = await loop.run_in_executor(
                None,
                _run_training,
                runner, df, target_column, task_type, config, reporter,
                f"exp_{experiment_id}",
            )
            job.result = result

            # Run SHAP on best model
            if run_shap and result.artifact_path:
                try:
                    shap_result = await loop.run_in_executor(
                        None,
                        _run_shap,
                        result.artifact_path,
                        df[feature_names].values[:200],  # background sample
                        df[feature_names].values[:100],  # explain sample
                        feature_names,
                        result.best_family,
                    )
                    job.shap_importance = shap_result
                except Exception as exc:
                    # SHAP failure is non-fatal
                    job.progress_messages.append({
                        "type": "warning",
                        "message": f"SHAP failed: {exc}",
                    })

            job.status = JobStatus.COMPLETE
        except Exception as exc:
            job.status = JobStatus.ERROR
            job.error = str(exc)
            reporter.send(TrainingError(message=str(exc)))
        finally:
            job.finished_at = time.time()
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    asyncio.ensure_future(run())

    # Background queue drainer — stores messages in job for late subscribers
    async def drain():
        while True:
            msg = await queue.get()
            if msg is _SENTINEL:
                break
            if hasattr(msg, "to_dict"):
                job.progress_messages.append(msg.to_dict())

    asyncio.ensure_future(drain())
    return job_id


async def stream_job_progress(job_id: str) -> AsyncIterator[dict]:
    """
    Yields progress messages for a job.

    Replays already-received messages first (for clients that connect late),
    then yields live messages until the job completes.
    """
    job = _jobs.get(job_id)
    if not job:
        return

    # Replay past messages
    for msg in job.progress_messages:
        yield msg

    # If already done, yield final state and stop
    if job.status in (JobStatus.COMPLETE, JobStatus.ERROR):
        yield {"type": "job_status", "status": job.status.value, "error": job.error}
        return

    # Wait for new messages — poll since we can't get a fresh queue reference
    seen = len(job.progress_messages)
    while job.status == JobStatus.RUNNING:
        await asyncio.sleep(0.2)
        new_msgs = job.progress_messages[seen:]
        for msg in new_msgs:
            yield msg
            seen += 1

    yield {"type": "job_status", "status": job.status.value, "error": job.error}


# ── Thread-callable wrappers ───────────────────────────────────────────────

def _run_training(
    runner: TrainingRunner,
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    config: TrainingConfig,
    reporter: ProgressReporter,
    experiment_name: str,
) -> TrainingResult:
    return runner.run(
        df=df,
        target_column=target_column,
        task_type=task_type,
        config=config,
        reporter=reporter,
        experiment_name=experiment_name,
    )


def _run_shap(
    artifact_path: str,
    X_background: any,
    X_explain: any,
    feature_names: list[str],
    family_name: str,
) -> list[dict]:
    import joblib
    import numpy as np
    from pathlib import Path
    from config import settings

    # Validate the artifact path is inside the model storage directory.
    # Prevents path traversal if artifact_path is ever derived from
    # user-supplied input (defence-in-depth — it comes from the DB, but
    # the DB value could be tampered with in theory).
    resolved = Path(artifact_path).resolve()
    allowed_root = Path(settings.model_storage_path).resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError:
        raise ValueError(
            f"Model artifact path {artifact_path!r} is outside the "
            f"allowed model storage directory {allowed_root}. "
            "Refusing to load."
        )

    model = joblib.load(artifact_path)
    runner = SHAPRunner(background_size=50, max_explain_rows=100)
    result = runner.explain(
        model=model,
        X_background=X_background.astype(float),
        X_explain=X_explain.astype(float),
        feature_names=feature_names,
        family_name=family_name,
    )
    return result.to_importance_dict()
