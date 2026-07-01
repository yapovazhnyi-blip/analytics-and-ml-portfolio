"""
Forecasting router — /api/v1/forecasting

Endpoints:
  POST /forecasting/jobs         — submit a forecasting job
  GET  /forecasting/jobs         — list all jobs
  GET  /forecasting/jobs/{id}    — status, metrics, and forecast data
  DELETE /forecasting/jobs/{id}  — delete a completed job
  GET  /forecasting/families     — list available families
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user
from config import settings
from database import get_db, AsyncSessionLocal
from models.dataset import Dataset
from models.forecast_job import ForecastJob
from profiling.runner import ProfileRunner
from schemas.common import DataResponse, PaginatedResponse, make_pagination_meta
from training.time_series.config import TimeSeriesConfig
from training.time_series.families import FORECASTING_FAMILIES, FAMILY_DISPLAY_TS
from training.time_series.runner import TimeSeriesRunner

router = APIRouter(
    prefix="/forecasting",
    tags=["forecasting"],
    dependencies=[Depends(get_current_user)],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class ForecastingRequest(BaseModel):
    dataset_id: int
    date_column: str   = Field(..., min_length=1)
    target_column: str = Field(..., min_length=1)
    horizon: int       = Field(default=12, ge=1, le=365)
    frequency: str     = Field(default="auto", description="auto | D | W | MS | QS | YS | H")
    n_trials: int      = Field(default=20, ge=1, le=100)
    families: Optional[list[str]] = Field(
        None, description="Restrict to these families. None = all available."
    )


class ForecastJobOut(BaseModel):
    job_id: str
    dataset_id: Optional[int]
    date_column: str
    target_column: str
    horizon: int
    frequency: str
    n_trials: int
    status: str
    best_family: Optional[str] = None
    best_family_display: Optional[str] = None
    cv_mape: Optional[float] = None
    cv_rmse: Optional[float] = None
    cv_mae: Optional[float] = None
    n_trials_completed: Optional[int] = None
    elapsed_secs: Optional[float] = None
    forecast: Optional[list[dict]] = None
    error_message: Optional[str] = None
    created_at: str


def _job_out(job: ForecastJob) -> ForecastJobOut:
    forecast = None
    if job.forecast_json:
        try:
            forecast = json.loads(job.forecast_json)
        except Exception:
            forecast = []

    return ForecastJobOut(
        job_id=job.job_id,
        dataset_id=job.dataset_id,
        date_column=job.date_column,
        target_column=job.target_column,
        horizon=job.horizon,
        frequency=job.frequency,
        n_trials=job.n_trials,
        status=job.status,
        best_family=job.best_family,
        best_family_display=FAMILY_DISPLAY_TS.get(job.best_family or "", job.best_family),
        cv_mape=job.cv_mape,
        cv_rmse=job.cv_rmse,
        cv_mae=job.cv_mae,
        n_trials_completed=job.n_trials_completed,
        elapsed_secs=job.elapsed_secs,
        forecast=forecast,
        error_message=job.error_message,
        created_at=job.created_at.isoformat(),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/families")
async def list_families():
    """Lists the forecasting families available in this environment."""
    return DataResponse(data={
        name: {
            "display": FAMILY_DISPLAY_TS.get(name, name),
            "available": True,
        }
        for name in FORECASTING_FAMILIES
    })


@router.post("/jobs", response_model=DataResponse[ForecastJobOut], status_code=201)
async def submit_forecast(
    body: ForecastingRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Submits a time series forecasting job.

    Runs AutoARIMA and/or Exponential Smoothing (and optionally Prophet/LSTM)
    with Optuna hyperparameter search and expanding-window temporal CV.

    Returns immediately — poll /forecasting/jobs/{job_id} for status.
    """
    dataset = await db.get(Dataset, body.dataset_id)
    if not dataset:
        raise HTTPException(404, f"Dataset {body.dataset_id} not found")
    if dataset.status != "ready":
        raise HTTPException(422, f"Dataset not ready (status: {dataset.status})")

    # Validate family names
    if body.families:
        unknown = set(body.families) - set(FORECASTING_FAMILIES)
        if unknown:
            raise HTTPException(422, {
                "errors": [f"Unknown families: {unknown}"],
                "available": list(FORECASTING_FAMILIES),
            })

    cfg = TimeSeriesConfig(
        date_column=body.date_column,
        target_column=body.target_column,
        horizon=body.horizon,
        frequency=body.frequency,
        n_trials=body.n_trials,
        families=body.families,
    )
    errors = cfg.validate()
    if errors:
        raise HTTPException(422, {"errors": errors})

    job_id = f"fc-{uuid.uuid4().hex[:16]}"
    job = ForecastJob(
        job_id=job_id,
        dataset_id=body.dataset_id,
        date_column=body.date_column,
        target_column=body.target_column,
        horizon=body.horizon,
        frequency=body.frequency,
        n_trials=body.n_trials,
        status="running",
    )
    db.add(job)
    await db.flush()
    await db.refresh(job)

    asyncio.create_task(_forecast_background(job.id, dataset.file_path, dataset.source_type, cfg))
    return DataResponse(data=_job_out(job))


async def _forecast_background(
    db_id: int,
    file_path: str,
    source_type: str,
    cfg: TimeSeriesConfig,
) -> None:
    """Loads data, runs the runner, updates the DB record."""
    import asyncio
    loop = asyncio.get_event_loop()

    async with AsyncSessionLocal() as session:
        job = await session.get(ForecastJob, db_id)
        if not job:
            return

        try:
            df = await loop.run_in_executor(
                None, ProfileRunner.load_dataframe, file_path, source_type
            )
            runner = TimeSeriesRunner(cfg, job.job_id)
            output_dir = str(Path(settings.model_storage_path) / "forecasting")
            result = await runner.run(df, output_dir)

            if result.succeeded:
                job.status = "succeeded"
                job.best_family = result.best_family
                job.cv_mape = result.cv_mape
                job.cv_rmse = result.cv_rmse
                job.cv_mae = result.cv_mae
                job.n_trials_completed = result.n_trials_completed
                job.elapsed_secs = result.elapsed_secs
                job.artifact_path = result.artifact_path
                job.forecast_json = result.forecast.to_json(orient="records") \
                    if result.forecast is not None and len(result.forecast) > 0 else "[]"
            else:
                job.status = "failed"
                job.error_message = result.error

        except Exception as exc:
            job.status = "failed"
            job.error_message = str(exc)

        await session.commit()


@router.get("/jobs", response_model=PaginatedResponse[ForecastJobOut])
async def list_jobs(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import func
    stmt = select(ForecastJob).order_by(ForecastJob.created_at.desc())
    total = await db.scalar(select(func.count()).select_from(stmt.subquery()))
    rows = await db.scalars(stmt.offset((page - 1) * page_size).limit(page_size))
    return PaginatedResponse(
        data=[_job_out(j) for j in rows.all()],
        pagination=make_pagination_meta(page, page_size, total or 0),
    )


@router.get("/jobs/{job_id}", response_model=DataResponse[ForecastJobOut])
async def get_job(job_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ForecastJob).where(ForecastJob.job_id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404, f"Forecast job '{job_id}' not found")
    return DataResponse(data=_job_out(job))


@router.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ForecastJob).where(ForecastJob.job_id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404, f"Forecast job '{job_id}' not found")
    await db.delete(job)
