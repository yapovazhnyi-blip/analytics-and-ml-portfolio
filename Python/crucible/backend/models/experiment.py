"""
Experiment model — one record per AutoML training run.

An experiment links a dataset + preprocessing config + model config
to evaluation results. This is the node in the lineage DAG that
connects data decisions to model outcomes.

Phase 2 will add the full SHAP and Optuna result JSON.
Phase 3 will add the lineage DAG edges (parent experiment references).
"""

from typing import Optional
import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base, TimestampMixin


class Experiment(Base, TimestampMixin):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(primary_key=True)

    name: Mapped[str] = mapped_column(sa.String(255), nullable=False)

    # Foreign key to the dataset used for training
    dataset_id: Mapped[int] = mapped_column(
        sa.Integer,
        sa.ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # The target column name within the dataset
    target_column: Mapped[str] = mapped_column(sa.String(255), nullable=False)

    # Task type — drives model family selection
    task_type: Mapped[str] = mapped_column(
        sa.String(20), nullable=False
    )  # "classification" | "regression"

    # Preprocessing config — JSON snapshot of choices made before training
    # e.g. {"impute_strategy": "mean", "scale": true, "drop_cols": ["id"]}
    preprocessing_config: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)

    # Training config — Optuna search settings
    # e.g. {"n_trials": 30, "cv_folds": 3, "timeout_secs": 300}
    training_config: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)

    # Results — populated after training completes
    best_model_family: Mapped[Optional[str]] = mapped_column(sa.String(50), nullable=True)
    best_score: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    scoring_metric: Mapped[Optional[str]] = mapped_column(sa.String(50), nullable=True)
    n_trials_completed: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    n_trials_pruned: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    training_duration_secs: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)

    # Full Optuna study results — JSON, populated after training
    results_json: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    fairness_json: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)

    # SHAP importance results — JSON, populated after explanation
    shap_json: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)

    # Path to the serialised model artifact (joblib)
    model_artifact_path: Mapped[Optional[str]] = mapped_column(sa.String(1024), nullable=True)

    # Run status
    status: Mapped[str] = mapped_column(
        sa.String(20), nullable=False, default="pending"
    )  # "pending" | "running" | "complete" | "error"

    error_message: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)

    # MLflow run ID — for cross-referencing with the MLflow UI (Phase 2)
    mlflow_run_id: Mapped[Optional[str]] = mapped_column(sa.String(64), nullable=True)

    # Model registry-style lifecycle stage, set by the retraining pipeline
    # (or manually via POST /experiments/{id}/promote). Mirrors MLflow Model
    # Registry's vocabulary: "candidate" (default) | "production" | "archived"
    lifecycle_stage: Mapped[str] = mapped_column(sa.String(20), nullable=False, default="candidate")

    def __repr__(self) -> str:
        return f"<Experiment id={self.id} name={self.name!r} status={self.status}>"
