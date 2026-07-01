"""
Retraining Pipeline models — Airflow-shaped automation built on Crucible's
existing drift detection and training infrastructure.

THE PIPELINE
------------
  1. DRIFT CHECK  — compare the policy's reference (production training)
                     dataset against a current data batch using the existing
                     drift/detector.py (PSI + KS + chi-squared).
  2. GATE         — if drift severity meets or exceeds drift_severity_trigger,
                     proceed to retraining. Otherwise the run stops here,
                     recorded as "no drift, skipped."
  3. RETRAIN      — submit a normal AutoML experiment (same code path as
                     POST /experiments) against the current dataset.
  4. PROMOTION    — compare the new candidate's holdout score against the
                     current production model's score. If it beats it by at
                     least promotion_margin, the new model is promoted:
                       old production experiment → lifecycle_stage="archived"
                       new experiment            → lifecycle_stage="production"
                     Otherwise the candidate is kept as lifecycle_stage="candidate"
                     and the existing production model keeps serving.

This mirrors MLflow's Model Registry stage vocabulary (candidate/production/
archived) deliberately — Crucible already integrates MLflow for tracking, so
reusing its promotion vocabulary keeps the mental model consistent.

WHY HIGHER-SCORE-WINS WORKS FOR BOTH CLASSIFICATION AND REGRESSION
----------------------------------------------------------------------
Crucible's Optuna studies always use direction="maximize" (training/runner.py)
— for metrics that are naturally "lower is better" (RMSE, MAE), the scoring
function negates them before optimisation. This means best_score is ALWAYS
oriented so that higher = better, regardless of task type or metric. The
promotion comparison `new_score >= old_score + margin` is therefore valid
universally without needing per-metric direction logic.
"""

from __future__ import annotations
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base, TimestampMixin


class RetrainingPolicy(Base, TimestampMixin):
    """
    A named, reusable retraining policy: "watch this dataset/target for
    drift, and if it drifts enough, retrain and conditionally promote."

    reference_dataset_id is the drift baseline — the data the CURRENT
    production model was trained on. After a successful promotion, this
    is updated to the new production experiment's dataset, so the next
    drift check compares against the latest production baseline.

    latest_dataset_id is the "freshest available data" pointer. Crucible
    has no continuous data stream, so this models the realistic separation
    between *when* to check (the schedule) and *what counts as current data*
    (an external system, or a user, updating this pointer when new data
    lands). If unset, the scheduler compares the reference against itself
    — a safe no-op that reports "stable" rather than erroring.
    """

    __tablename__ = "retraining_policies"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(sa.String(100), unique=True, nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)

    reference_dataset_id: Mapped[int] = mapped_column(
        sa.Integer, sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    latest_dataset_id: Mapped[Optional[int]] = mapped_column(
        sa.Integer, sa.ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )

    target_column: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    task_type: Mapped[str] = mapped_column(sa.String(20), nullable=False)  # classification | regression

    # "slight" | "significant" | "critical" — minimum drift severity that triggers retraining
    drift_severity_trigger: Mapped[str] = mapped_column(sa.String(20), nullable=False, default="significant")

    # Absolute score improvement required for promotion (best_score is always "higher is better")
    promotion_margin: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.02)

    # Retraining config
    n_trials: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=15)
    cv_folds: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=3)

    # If set, the scheduler runs this policy automatically every N hours.
    # If None, the policy is manual-trigger only (POST /retraining/policies/{id}/run).
    check_interval_hours: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)

    # Currently-promoted production experiment for this policy (None until first promotion)
    production_experiment_id: Mapped[Optional[int]] = mapped_column(
        sa.Integer, sa.ForeignKey("experiments.id", ondelete="SET NULL"), nullable=True
    )

    is_active: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=True)

    def __repr__(self) -> str:
        return f"<RetrainingPolicy id={self.id} name={self.name!r} active={self.is_active}>"


class RetrainingRun(Base, TimestampMixin):
    """
    One execution of a RetrainingPolicy's pipeline — the audit trail.

    steps_json records each pipeline stage's outcome in order, mirroring
    what an Airflow task-instance log would show: which steps ran, which
    were skipped, and why.
    """

    __tablename__ = "retraining_runs"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)
    policy_id: Mapped[int] = mapped_column(
        sa.Integer, sa.ForeignKey("retraining_policies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    current_dataset_id: Mapped[int] = mapped_column(sa.Integer, nullable=False)

    status: Mapped[str] = mapped_column(sa.String(20), nullable=False, default="running")  # running|completed|failed

    drift_checked: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=False)
    drift_detected: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=False)
    drift_report_json: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)

    retrain_triggered: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=False)
    new_experiment_id: Mapped[Optional[int]] = mapped_column(
        sa.Integer, sa.ForeignKey("experiments.id", ondelete="SET NULL"), nullable=True
    )

    promoted: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=False)
    promotion_reason: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)

    steps_json: Mapped[str] = mapped_column(sa.Text, nullable=False, default="[]")
    error_message: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    elapsed_secs: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)

    def __repr__(self) -> str:
        return f"<RetrainingRun id={self.id} policy_id={self.policy_id} status={self.status!r}>"
