"""
A/B Testing router — /api/v1/ab-test

Endpoints:
  POST /ab-test               — run statistical comparison of two experiments
  POST /ab-test/power         — compute sample size / MDE for a target effect
  GET  /ab-test/methods       — list available statistical tests and their use cases
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ab_testing.analyzer import ABTestAnalyzer
from ab_testing.engine import power_analysis
from auth.dependencies import get_current_user
from database import get_db
from models.experiment import Experiment
from schemas.common import DataResponse

router = APIRouter(
    prefix="/ab-test",
    tags=["ab-testing"],
    dependencies=[Depends(get_current_user)],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class ABTestRequest(BaseModel):
    experiment_a_id: int = Field(..., description="The control experiment (baseline).")
    experiment_b_id: int = Field(..., description="The challenger experiment.")
    confidence_level: float = Field(
        default=0.95,
        ge=0.80, le=0.99,
        description=(
            "Statistical confidence level (1 - alpha). "
            "0.95 → 5% false positive rate. Use 0.99 for high-stakes decisions."
        ),
    )


class PowerRequest(BaseModel):
    baseline_rate: float = Field(
        ..., gt=0, lt=1,
        description="Current model's metric rate (e.g. 0.85 for 85% accuracy).",
    )
    minimum_effect: float = Field(
        ..., gt=0, lt=1,
        description="Smallest improvement worth detecting (e.g. 0.02 for a 2pp lift).",
    )
    alpha: float = Field(default=0.05, gt=0, lt=0.5)
    power: float = Field(default=0.80, ge=0.5, le=0.99)
    current_n: int = Field(default=500, ge=10, description="Current holdout set size.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/methods")
async def list_methods():
    """Describes the available statistical tests and when each is appropriate."""
    return DataResponse(data={
        "tests": {
            "mcnemar": {
                "name":  "McNemar's Test",
                "use_case": "Binary/multi-class classification",
                "description": (
                    "Uses the paired disagreement structure of two classifiers on the same test set. "
                    "Only cases where A and B disagree contribute to the statistic — "
                    "this paired structure gives ~2× the power of unpaired tests."
                ),
                "assumption": "Both models evaluated on the same samples.",
            },
            "wilcoxon": {
                "name": "Wilcoxon Signed-Rank Test",
                "use_case": "Regression (MAE, RMSE comparison)",
                "description": (
                    "Non-parametric test that ranks absolute prediction errors. "
                    "Tests whether model B's errors are systematically smaller than A's. "
                    "No normality assumption — robust to outliers."
                ),
                "assumption": "Paired samples (same test set).",
            },
            "bootstrap": {
                "name": "Bootstrap Permutation Test",
                "use_case": "Any metric (fallback)",
                "description": (
                    "Resamples the test set predictions 2000 times with replacement. "
                    "Computes the metric difference on each resample. "
                    "The p-value is the fraction of resamples where B does not beat A. "
                    "Works for any metric but requires more computation."
                ),
                "assumption": "None (distribution-free).",
            },
        }
    })


@router.post("/")
async def run_ab_test(
    body: ABTestRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Runs a statistical A/B test comparing two completed experiments.

    Both experiments must be trained on the same dataset and target column.
    The holdout split is reproduced identically (random_state=42, test_size=0.2)
    to ensure both models are evaluated on the same samples.

    Statistical test selection:
      - Classification → McNemar's test (most powerful for paired classifiers)
      - Regression     → Wilcoxon signed-rank test

    Returns: winner, p-value, confidence interval on the difference,
    effect size, and a plain-English recommendation.
    """
    if body.experiment_a_id == body.experiment_b_id:
        raise HTTPException(422, "experiment_a_id and experiment_b_id must be different.")

    # Load both experiments
    exp_a = await db.get(Experiment, body.experiment_a_id)
    exp_b = await db.get(Experiment, body.experiment_b_id)

    if not exp_a:
        raise HTTPException(404, f"Experiment {body.experiment_a_id} not found.")
    if not exp_b:
        raise HTTPException(404, f"Experiment {body.experiment_b_id} not found.")
    if exp_a.status != "completed":
        raise HTTPException(422, f"Experiment A (id={body.experiment_a_id}) is not completed.")
    if exp_b.status != "completed":
        raise HTTPException(422, f"Experiment B (id={body.experiment_b_id}) is not completed.")
    if not exp_a.model_artifact_path:
        raise HTTPException(422, f"Experiment A has no model artifact.")
    if not exp_b.model_artifact_path:
        raise HTTPException(422, f"Experiment B has no model artifact.")

    # Both experiments must share the same dataset and target
    if exp_a.dataset_id != exp_b.dataset_id:
        raise HTTPException(
            422,
            f"Experiments use different datasets "
            f"(A: dataset {exp_a.dataset_id}, B: dataset {exp_b.dataset_id}). "
            "A/B testing requires both models to be evaluated on the same holdout set."
        )
    if exp_a.target_column != exp_b.target_column:
        raise HTTPException(
            422,
            f"Experiments have different target columns "
            f"(A: '{exp_a.target_column}', B: '{exp_b.target_column}'). "
            "Both experiments must predict the same target."
        )

    # Load dataset info for file path
    from models.dataset import Dataset
    dataset = await db.get(Dataset, exp_a.dataset_id)
    if not dataset or not dataset.file_path:
        raise HTTPException(404, f"Dataset {exp_a.dataset_id} not found or has no file.")

    task_type = exp_a.task_type or "classification"

    analyzer = ABTestAnalyzer()
    result = await analyzer.analyze(
        exp_a_id=body.experiment_a_id,
        exp_b_id=body.experiment_b_id,
        artifact_path_a=exp_a.model_artifact_path,
        artifact_path_b=exp_b.model_artifact_path,
        dataset_path=dataset.file_path,
        source_type=dataset.source_type,
        target_column=exp_a.target_column,
        task_type=task_type,
        confidence_level=body.confidence_level,
    )

    # Enrich with experiment names for display
    response_data = result.to_dict()
    response_data["experiment_a_name"] = exp_a.name
    response_data["experiment_b_name"] = exp_b.name
    response_data["dataset_name"]      = dataset.name
    response_data["target_column"]     = exp_a.target_column
    response_data["task_type"]         = task_type

    return DataResponse(data=response_data)


@router.post("/power")
async def run_power_analysis(body: PowerRequest):
    """
    Computes sample size requirements and minimum detectable effect (MDE).

    Given your baseline metric and the smallest improvement you care about,
    returns the sample size needed to detect that improvement with the
    requested statistical power.

    Also computes the MDE achievable with your current sample size — this
    tells you the smallest effect that would be statistically detectable
    given your current test set.

    Use this BEFORE running an A/B test to know whether your holdout set
    is large enough to detect the expected improvement.
    """
    result = power_analysis(
        baseline_rate=body.baseline_rate,
        minimum_effect=body.minimum_effect,
        alpha=body.alpha,
        power=body.power,
        current_n=body.current_n,
    )
    return DataResponse(data=result.to_dict())
