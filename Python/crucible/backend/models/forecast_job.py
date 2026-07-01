"""ORM model for time series forecasting jobs."""
from __future__ import annotations
from typing import Optional
import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column
from models.base import Base, TimestampMixin


class ForecastJob(Base, TimestampMixin):
    __tablename__ = "forecast_jobs"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(sa.String(64), unique=True, nullable=False, index=True)

    dataset_id: Mapped[int] = mapped_column(sa.Integer, sa.ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)
    date_column: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    target_column: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    horizon: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=12)
    frequency: Mapped[str] = mapped_column(sa.String(10), nullable=False, default="auto")
    n_trials: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=20)

    status: Mapped[str] = mapped_column(sa.String(20), nullable=False, default="pending")
    best_family: Mapped[Optional[str]] = mapped_column(sa.String(50), nullable=True)
    cv_mape: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    cv_rmse: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    cv_mae: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    n_trials_completed: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    elapsed_secs: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    forecast_json: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    artifact_path: Mapped[Optional[str]] = mapped_column(sa.String(1024), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
