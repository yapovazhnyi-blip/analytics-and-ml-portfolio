"""
Fairness router — /api/v1/experiments/{id}/fairness

Endpoints:
  POST /experiments/{id}/fairness   — run fairness analysis, return report
  GET  /experiments/{id}/fairness   — return cached fairness report if available

The fairness analysis is stateless — it recomputes on each POST by
reproducing the holdout split and running model.predict().
Results are cached in Experiment.fairness_json for subsequent GET requests.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user
from database import get_db
from fairness.analyzer import FairnessAnalyzer
from models.experiment import Experiment
from schemas.common import DataResponse

router = APIRouter(tags=["fairness"], dependencies=[Depends(get_current_user)])


# ── Schemas ───────────────────────────────────────────────────────────────────

class FairnessRequest(BaseModel):
    protected_attributes: list[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description=(
            "Column names that define protected groups. Must exist in the dataset. "
            "E.g. ['gender', 'age_group', 'region']"
        ),
    )
    positive_class: int = Field(
        default=1,
        description=(
            "Which label counts as the positive (favourable) outcome. "
            "For binary classification: 1 (default). "
            "For loan approval encoded as 0/1, the approved class is typically 1."
        ),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/experiments/{experiment_id}/fairness")
async def run_fairness(
    experiment_id: int,
    body: FairnessRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Runs fairness analysis for a completed experiment.

    Reproduces the holdout split (same random_state as training), generates
    predictions from the saved model, and computes four fairness metrics
    per protected attribute:
      - Demographic Parity Difference
      - Equal Opportunity Difference
      - Equalized Odds Difference
      - Disparate Impact Ratio

    Severity thresholds follow the EEOC four-fifths rule for disparate impact
    and standard ML fairness literature for the difference metrics.

    Results are cached in the experiment record for fast retrieval via GET.

    Args:
        protected_attributes: Dataset columns that define protected groups.
        positive_class:        The label that counts as the favourable outcome.

    Requires: experiment status = "completed", model artifact present.
    """
    import json

    exp = await db.get(Experiment, experiment_id)
    if not exp:
        raise HTTPException(404, f"Experiment {experiment_id} not found")
    if exp.status != "completed":
        raise HTTPException(422, f"Experiment not completed (status: {exp.status})")
    if not exp.model_artifact_path:
        raise HTTPException(422, "Experiment has no model artifact. Re-run the experiment.")
    if not exp.target_column:
        raise HTTPException(422, "Experiment has no target column recorded.")

    # Fetch dataset for file path and source type
    from models.dataset import Dataset
    dataset = await db.get(Dataset, exp.dataset_id)
    if not dataset:
        raise HTTPException(404, f"Dataset {exp.dataset_id} not found")
    if not dataset.file_path:
        raise HTTPException(422, "Dataset file path not available.")

    analyzer = FairnessAnalyzer()
    report = await analyzer.analyze(
        artifact_path=exp.model_artifact_path,
        dataset_path=dataset.file_path,
        source_type=dataset.source_type,
        target_column=exp.target_column,
        protected_attributes=body.protected_attributes,
        task_type=exp.task_type or "classification",
        experiment_id=experiment_id,
        positive_class=body.positive_class,
    )

    if not report.succeeded:
        raise HTTPException(422, f"Fairness analysis failed: {report.error}")

    # Cache the result in the experiment record
    if hasattr(exp, "fairness_json"):
        exp.fairness_json = json.dumps(report.to_dict())

    return DataResponse(data=report.to_dict())


@router.get("/experiments/{experiment_id}/fairness")
async def get_fairness(
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns the cached fairness report for an experiment.
    Returns 404 if no fairness analysis has been run yet (use POST to run one).
    """
    import json

    exp = await db.get(Experiment, experiment_id)
    if not exp:
        raise HTTPException(404, f"Experiment {experiment_id} not found")

    if not hasattr(exp, "fairness_json") or not exp.fairness_json:
        raise HTTPException(
            404,
            "No fairness analysis found for this experiment. "
            "Run POST /experiments/{id}/fairness first."
        )

    return DataResponse(data=json.loads(exp.fairness_json))
