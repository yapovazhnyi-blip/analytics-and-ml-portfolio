"""
Drift detection router — /api/v1/drift

Endpoints:
  POST /drift/check    — compare two datasets, return drift report
  GET  /drift/presets  — list severity threshold presets for the UI
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user
from database import get_db
from drift.detector import compare_datasets, PSI_NEGLIGIBLE, PSI_SLIGHT, PSI_SIGNIFICANT
from models.dataset import Dataset
from profiling.runner import ProfileRunner
from schemas.common import DataResponse

router = APIRouter(
    prefix="/drift",
    tags=["drift"],
    dependencies=[Depends(get_current_user)],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class DriftCheckRequest(BaseModel):
    reference_dataset_id: int = Field(
        ...,
        description="ID of the reference (training) dataset.",
    )
    current_dataset_id: int = Field(
        ...,
        description="ID of the current (new production) dataset to compare.",
    )
    target_col: Optional[str] = Field(
        None,
        description="Target column to exclude from drift analysis (not a feature).",
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/check")
async def check_drift(
    body: DriftCheckRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Compares two datasets for distributional drift.

    Numeric features: PSI (Population Stability Index) + KS test.
    Categorical features: chi-squared frequency test.

    Severity interpretation (PSI):
      stable      < 0.10  — no action needed
      slight      < 0.20  — monitor
      significant < 0.25  — consider retraining
      critical    ≥ 0.25  — retrain immediately

    Returns per-feature statistics sorted by severity (worst first).
    """
    # Load both datasets
    ref_ds = await db.get(Dataset, body.reference_dataset_id)
    cur_ds = await db.get(Dataset, body.current_dataset_id)

    if not ref_ds:
        raise HTTPException(404, f"Reference dataset {body.reference_dataset_id} not found")
    if not cur_ds:
        raise HTTPException(404, f"Current dataset {body.current_dataset_id} not found")

    if ref_ds.status != "ready":
        raise HTTPException(422, f"Reference dataset is not ready (status: {ref_ds.status})")
    if cur_ds.status != "ready":
        raise HTTPException(422, f"Current dataset is not ready (status: {cur_ds.status})")

    # Load DataFrames (run in executor — I/O bound)
    import asyncio
    loop = asyncio.get_event_loop()

    def _load():
        ref_df = ProfileRunner.load_dataframe(ref_ds.file_path, ref_ds.source_type)
        cur_df = ProfileRunner.load_dataframe(cur_ds.file_path, cur_ds.source_type)
        return ref_df, cur_df

    try:
        ref_df, cur_df = await loop.run_in_executor(None, _load)
    except Exception as exc:
        raise HTTPException(500, f"Failed to load dataset files: {exc}")

    # Run drift detection (CPU-bound — executor)
    report = await loop.run_in_executor(
        None,
        compare_datasets,
        ref_df, cur_df,
        body.reference_dataset_id,
        body.current_dataset_id,
        body.target_col,
    )

    return DataResponse(data={
        **report.to_dict(),
        "reference_name": ref_ds.name,
        "current_name":   cur_ds.name,
        "thresholds": {
            "negligible":  PSI_NEGLIGIBLE,
            "slight":      PSI_SLIGHT,
            "significant": PSI_SIGNIFICANT,
        },
    })


@router.get("/presets")
async def drift_presets():
    """Returns the PSI severity thresholds used in drift analysis."""
    return DataResponse(data={
        "psi_thresholds": {
            "stable":      {"max": PSI_NEGLIGIBLE,  "label": "No action needed"},
            "slight":      {"max": PSI_SLIGHT,      "label": "Monitor closely"},
            "significant": {"max": PSI_SIGNIFICANT, "label": "Consider retraining"},
            "critical":    {"min": PSI_SIGNIFICANT, "label": "Retrain immediately"},
        },
        "tests": {
            "numeric":     ["PSI (Population Stability Index)", "KS Test (Kolmogorov-Smirnov)"],
            "categorical": ["Chi-Squared frequency test"],
        },
    })
