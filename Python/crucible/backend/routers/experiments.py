"""
Experiments router — /api/v1/experiments + WebSocket /ws/experiments/{id}/progress

Endpoints:
  POST /experiments           — create and start a training job
  GET  /experiments           — list experiments (optionally by dataset)
  GET  /experiments/{id}      — full experiment result with SHAP importance
  GET  /experiments/{id}/status — lightweight status check (no heavy JSON)
  DELETE /experiments/{id}    — cancel or remove an experiment

WebSocket:
  WS /ws/experiments/{job_id}/progress — live training progress stream
"""

import json
import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from auth.dependencies import get_current_user
from fastapi import Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import get_db
from models import Dataset, Experiment
from schemas.common import DataResponse, PaginatedResponse, make_pagination_meta
from schemas.experiment import (
    ExperimentCreate, ExperimentOut, ExperimentSummary, HoldoutMetrics, FeatureImportanceOut,
)
from jobs.manager import (
    JobStatus, get_job, list_jobs_for_experiment, start_job, stream_job_progress,
)
from training.runner import TrainingConfig
from profiling.runner import ProfileRunner


router = APIRouter(tags=["experiments"], dependencies=[Depends(get_current_user)])
ws_router = APIRouter(tags=["websocket"])


# ── Helpers ────────────────────────────────────────────────────────────────

def _exp_to_summary(exp: Experiment) -> ExperimentSummary:
    return ExperimentSummary(
        id=exp.id,
        name=exp.name,
        dataset_id=exp.dataset_id,
        target_column=exp.target_column,
        task_type=exp.task_type,
        status=exp.status,
        best_model_family=exp.best_model_family,
        best_score=exp.best_score,
        scoring_metric=exp.scoring_metric,
        lifecycle_stage=exp.lifecycle_stage,
        created_at=exp.created_at.isoformat(),
    )


def _exp_to_out(exp: Experiment, job_id: Optional[str] = None) -> ExperimentOut:
    holdout = []
    calibration_applied = None
    calibration_method = None
    pruner_type = None
    if exp.results_json:
        try:
            results = json.loads(exp.results_json)
            hm = results.get("holdout_metrics", {})
            holdout = [HoldoutMetrics(metric=k, value=v) for k, v in hm.items()]
            calibration_applied = results.get("calibration_applied")
            calibration_method = results.get("calibration_method")
            pruner_type = results.get("pruner_type")
        except Exception:
            pass

    importance = []
    if exp.shap_json:
        try:
            shap_data = json.loads(exp.shap_json)
            importance = [FeatureImportanceOut(**item) for item in shap_data[:20]]
        except Exception:
            pass

    return ExperimentOut(
        id=exp.id,
        name=exp.name,
        dataset_id=exp.dataset_id,
        target_column=exp.target_column,
        task_type=exp.task_type,
        status=exp.status,
        best_model_family=exp.best_model_family,
        best_score=exp.best_score,
        scoring_metric=exp.scoring_metric,
        n_trials_completed=exp.n_trials_completed,
        n_trials_pruned=exp.n_trials_pruned,
        training_duration_secs=exp.training_duration_secs,
        holdout_metrics=holdout,
        feature_importance=importance,
        error_message=exp.error_message,
        mlflow_run_id=exp.mlflow_run_id,
        calibration_applied=calibration_applied,
        calibration_method=calibration_method,
        pruner_type=pruner_type,
        lifecycle_stage=exp.lifecycle_stage,
        created_at=exp.created_at.isoformat(),
        job_id=job_id,
    )


# ── Create experiment ──────────────────────────────────────────────────────

@router.post("/experiments", response_model=DataResponse[ExperimentOut], status_code=201)
async def create_experiment(body: ExperimentCreate, db: AsyncSession = Depends(get_db)):
    """
    Creates an experiment record and starts a background training job.

    Returns immediately with status='running' and a job_id the client
    can use to connect via WebSocket for live progress.
    """
    ds = await db.get(Dataset, body.dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset {body.dataset_id} not found")
    if ds.status != "ready":
        raise HTTPException(status_code=422, detail=f"Dataset not ready (status: {ds.status})")
    if not ds.file_path:
        raise HTTPException(status_code=422, detail="Dataset has no local file")

    # Load the DataFrame
    try:
        df = ProfileRunner.load_dataframe(ds.file_path, ds.source_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {exc}")

    if body.target_column not in df.columns:
        raise HTTPException(status_code=422, detail=f"Column '{body.target_column}' not in dataset")

    # Identify numeric feature columns
    feature_cols = [
        c for c in df.columns
        if c != body.target_column and df[c].dtype.kind in "iufcb"
    ]
    if not feature_cols:
        raise HTTPException(status_code=422, detail="No numeric feature columns found")

    # Create the experiment record
    exp = Experiment(
        name=body.name,
        dataset_id=body.dataset_id,
        target_column=body.target_column,
        task_type=body.task_type,
        training_config=json.dumps({
            "n_trials": body.n_trials,
            "cv_folds": body.cv_folds,
            "timeout_secs": body.timeout_secs,
        }),
        status="running",
    )
    db.add(exp)
    await db.flush()
    await db.refresh(exp)

    config = TrainingConfig(
        n_trials=body.n_trials,
        cv_folds=body.cv_folds,
        timeout_secs=body.timeout_secs,
    )

    # Kick off background job
    job_id = await start_job(
        experiment_id=exp.id,
        df=df,
        target_column=body.target_column,
        task_type=body.task_type,
        config=config,
        feature_names=feature_cols,
        run_shap=body.run_shap,
    )

    # Store job_id in preprocessing_config for retrieval
    exp.preprocessing_config = json.dumps({"job_id": job_id, "feature_cols": feature_cols})
    await db.flush()

    # Background task: update DB record when training finishes
    import asyncio
    asyncio.ensure_future(_update_experiment_on_complete(exp.id, job_id))

    return DataResponse(data=_exp_to_out(exp, job_id=job_id))


async def _update_experiment_on_complete(exp_id: int, job_id: str) -> None:
    """Polls job status and updates the Experiment record when done."""
    from database import SessionFactory

    while True:
        await asyncio.sleep(2)
        job = get_job(job_id)
        if not job:
            break
        if job.status in (JobStatus.COMPLETE, JobStatus.ERROR):
            async with SessionFactory() as session:
                exp = await session.get(Experiment, exp_id)
                if not exp:
                    break
                if job.status == JobStatus.COMPLETE and job.result:
                    r = job.result
                    exp.status = "complete"
                    exp.best_model_family = r.best_family
                    exp.best_score = r.best_cv_score
                    exp.scoring_metric = r.scoring_metric
                    exp.n_trials_completed = r.n_trials_completed
                    exp.n_trials_pruned = r.n_trials_pruned
                    exp.training_duration_secs = r.elapsed_secs
                    exp.model_artifact_path = r.artifact_path
                    exp.mlflow_run_id = r.mlflow_run_id
                    exp.results_json = json.dumps({
                        "holdout_metrics":      r.holdout_metrics,
                        "best_params":          r.best_params,
                        "calibration_applied":  r.calibration_applied,
                        "calibration_method":   r.calibration_method,
                        "pruner_type":          r.pruner_type,
                    })
                    if job.shap_importance:
                        exp.shap_json = json.dumps(job.shap_importance)
                else:
                    exp.status = "error"
                    exp.error_message = job.error or "Training failed"
                await session.commit()
            break


# ── List experiments ───────────────────────────────────────────────────────

@router.get("/experiments", response_model=PaginatedResponse[ExperimentSummary])
async def list_experiments(
    dataset_id: Optional[int] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Experiment).order_by(Experiment.created_at.desc())
    if dataset_id is not None:
        stmt = stmt.where(Experiment.dataset_id == dataset_id)

    total = await db.scalar(select(func.count()).select_from(stmt.subquery()))
    rows = await db.scalars(stmt.offset((page - 1) * page_size).limit(page_size))

    return PaginatedResponse(
        data=[_exp_to_summary(e) for e in rows.all()],
        pagination=make_pagination_meta(page, page_size, total or 0),
    )


@router.get("/experiments/cursor")
async def list_experiments_cursor(
    dataset_id: Optional[int] = Query(default=None),
    cursor: Optional[str] = Query(default=None, description="Opaque cursor from a previous response's next_cursor."),
    limit: int = Query(default=20, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """
    Cursor-paginated (keyset) experiment listing — for programmatic clients
    paging through large result sets at constant speed regardless of depth.

    First request: omit `cursor`.
    Subsequent requests: pass the `next_cursor` from the previous response.
    Stop when `has_more` is false or `next_cursor` is null.

    Use this instead of the offset-based /experiments endpoint when:
      - Paging through more than a few thousand rows
      - Building an export/sync job that must not skip or duplicate rows
        under concurrent writes
    """
    from schemas.cursor_pagination import paginate_by_cursor

    stmt = select(Experiment)
    if dataset_id is not None:
        stmt = stmt.where(Experiment.dataset_id == dataset_id)

    try:
        page = await paginate_by_cursor(db, stmt, Experiment, cursor, limit)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    return DataResponse(data={
        "items":       [_exp_to_summary(e) for e in page.items],
        "next_cursor": page.next_cursor,
        "has_more":    page.has_more,
    })


# ── Get single experiment ──────────────────────────────────────────────────

@router.get("/experiments/{exp_id}", response_model=DataResponse[ExperimentOut])
async def get_experiment(exp_id: int, db: AsyncSession = Depends(get_db)):
    exp = await db.get(Experiment, exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found")

    job_id = None
    if exp.preprocessing_config:
        try:
            job_id = json.loads(exp.preprocessing_config).get("job_id")
        except Exception:
            pass

    # Sync from job manager if still running
    if exp.status == "running" and job_id:
        job = get_job(job_id)
        if job and job.status == JobStatus.COMPLETE and job.result:
            r = job.result
            exp.status = "complete"
            exp.best_model_family = r.best_family
            exp.best_score = r.best_cv_score
            exp.scoring_metric = r.scoring_metric
            exp.n_trials_completed = r.n_trials_completed
            exp.n_trials_pruned = r.n_trials_pruned
            exp.training_duration_secs = r.elapsed_secs
            exp.model_artifact_path = r.artifact_path
            exp.mlflow_run_id = r.mlflow_run_id
            exp.results_json = json.dumps({
                "holdout_metrics":      r.holdout_metrics,
                "best_params":          r.best_params,
                "calibration_applied":  r.calibration_applied,
                "calibration_method":   r.calibration_method,
                "pruner_type":          r.pruner_type,
            })
            if job.shap_importance:
                exp.shap_json = json.dumps(job.shap_importance)

    return DataResponse(data=_exp_to_out(exp, job_id=job_id))


# ── Delete experiment ──────────────────────────────────────────────────────

@router.delete("/experiments/{exp_id}", status_code=204)
async def delete_experiment(exp_id: int, db: AsyncSession = Depends(get_db)):
    exp = await db.get(Experiment, exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found")
    import os
    if exp.model_artifact_path and os.path.exists(exp.model_artifact_path):
        os.remove(exp.model_artifact_path)
    await db.delete(exp)


# ── WebSocket progress ─────────────────────────────────────────────────────

@ws_router.websocket("/ws/experiments/{job_id}/progress")
async def experiment_progress_ws(
    job_id: str,
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="JWT access token (required when auth is enabled)"),
):
    """
    Streams live training progress for a job.

    Authentication: pass the JWT as ?token=<access_token> since browsers
    cannot send Authorization headers with WebSocket connections.

    In dev mode (DISABLE_AUTH=true) the token parameter is ignored.

    Replays historical messages for late-connecting clients, then
    streams new messages until training completes.

    Message types: trial | complete | error | warning | job_status
    """
    from auth.dependencies import validate_ws_token
    from database import SessionFactory

    if not settings.disable_auth:
        async with SessionFactory() as db:
            user = await validate_ws_token(token, db)
        if not user:
            await websocket.close(code=1008)   # 1008 = Policy Violation
            return

    await websocket.accept()
    try:
        async for msg in stream_job_progress(job_id):
            await websocket.send_json(msg)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
