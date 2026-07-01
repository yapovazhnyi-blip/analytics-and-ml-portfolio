from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class ExperimentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    dataset_id: int
    target_column: str
    task_type: str = Field(..., description="classification | regression")
    n_trials: int = Field(default=20, ge=2, le=100)
    cv_folds: int = Field(default=3, ge=2, le=10)
    timeout_secs: Optional[float] = Field(default=None, ge=10)
    run_shap: bool = True


class ExperimentSummary(BaseModel):
    id: int
    name: str
    dataset_id: int
    target_column: str
    task_type: str
    status: str
    best_model_family: Optional[str] = None
    best_score: Optional[float] = None
    scoring_metric: Optional[str] = None
    lifecycle_stage: Optional[str] = None      # "candidate" | "production" | "archived"
    created_at: str

    model_config = {"from_attributes": True}


class HoldoutMetrics(BaseModel):
    metric: str
    value: float


class FeatureImportanceOut(BaseModel):
    feature: str
    mean_abs_shap: float
    rank: int


class ExperimentOut(BaseModel):
    id: int
    name: str
    dataset_id: int
    target_column: str
    task_type: str
    status: str
    best_model_family: Optional[str] = None
    best_score: Optional[float] = None
    scoring_metric: Optional[str] = None
    n_trials_completed: Optional[int] = None
    n_trials_pruned: Optional[int] = None
    training_duration_secs: Optional[float] = None
    holdout_metrics: list[HoldoutMetrics] = []
    feature_importance: list[FeatureImportanceOut] = []
    error_message: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    calibration_applied: Optional[bool] = None
    calibration_method: Optional[str] = None   # "isotonic" | "sigmoid" | None
    pruner_type: Optional[str] = None          # "median" | "hyperband"
    lifecycle_stage: Optional[str] = None      # "candidate" | "production" | "archived"
    created_at: str
    job_id: Optional[str] = None

    model_config = {"from_attributes": True}
