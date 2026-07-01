"""Anomaly detection router — /api/v1/datasets/{id}/anomaly"""

from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user
from database import get_db
from models.dataset import Dataset
from schemas.common import DataResponse

router = APIRouter(tags=["anomaly"], dependencies=[Depends(get_current_user)])


class AnomalyRequest(BaseModel):
    contamination: float = Field(
        default=0.05, gt=0.0, lt=0.5,
        description="Expected fraction of anomalies (0.01–0.49). Default 0.05 = 5%.",
    )
    algorithms: list[str] = Field(
        default=["isolation_forest", "lof"],
        description="Algorithms to run. Options: isolation_forest, lof, ocsvm.",
    )
    exclude_columns: list[str] = Field(
        default_factory=list,
        description="Columns to exclude (e.g. ID columns, target column).",
    )
    top_n: int = Field(default=20, ge=1, le=500)


@router.post("/datasets/{dataset_id}/anomaly")
async def detect_anomalies(
    dataset_id: int,
    body: AnomalyRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Runs unsupervised anomaly detection on a dataset.

    Algorithms:
      isolation_forest — best general-purpose detector (default)
      lof              — good for density-based clusters
      ocsvm            — compact normal class (max 10,000 rows)

    Results include per-algorithm scores, a majority-vote consensus,
    and the top-N most anomalous rows for review.

    No target column required — fully unsupervised.
    """
    import asyncio
    from anomaly.runner import AnomalyRunner
    from profiling.runner import ProfileRunner

    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(404, f"Dataset {dataset_id} not found")
    if ds.status != "ready":
        raise HTTPException(422, f"Dataset not ready (status: {ds.status})")
    if not ds.file_path:
        raise HTTPException(422, "Dataset has no file path.")

    valid_algos = {"isolation_forest", "lof", "ocsvm"}
    unknown = set(body.algorithms) - valid_algos
    if unknown:
        raise HTTPException(422, f"Unknown algorithms: {unknown}. Valid: {valid_algos}")

    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(
        None, ProfileRunner.load_dataframe, ds.file_path, ds.source_type
    )

    runner = AnomalyRunner(
        contamination=body.contamination,
        algorithms=body.algorithms,
        top_n=body.top_n,
    )
    report = await runner.run(df, dataset_id, body.exclude_columns)

    if not report.succeeded:
        raise HTTPException(422, f"Anomaly detection failed: {report.error}")

    return DataResponse(data=report.to_dict())
