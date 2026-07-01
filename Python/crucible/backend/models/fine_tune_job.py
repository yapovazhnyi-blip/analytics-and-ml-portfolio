"""ORM model for tracking fine-tuning jobs."""

from __future__ import annotations
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base, TimestampMixin


class FineTuneJob(Base, TimestampMixin):
    """
    Tracks a LoRA / QLoRA fine-tuning job from submission to completion.

    One record per training run. The adapter weights are stored on disk
    at adapter_path; this table stores the metadata needed to surface
    jobs in the UI and reconstruct the configuration for replay.

    Status transitions:
      pending → running → succeeded
                       → failed
    """

    __tablename__ = "fine_tune_jobs"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)

    # Unique job identifier used for WebSocket routing and directory naming
    job_id: Mapped[str] = mapped_column(
        sa.String(64), unique=True, nullable=False, index=True
    )

    # ── Model ────────────────────────────────────────────────────────────
    base_model_id: Mapped[str] = mapped_column(sa.String(512), nullable=False)
    method: Mapped[str] = mapped_column(sa.String(20), nullable=False, default="sft")

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset_format: Mapped[str] = mapped_column(sa.String(20), nullable=False)
    n_samples: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)

    # ── LoRA config (stored as JSON) ─────────────────────────────────────
    lora_config_json: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)

    # ── Training hyperparams ─────────────────────────────────────────────
    epochs: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=1)
    learning_rate: Mapped[float] = mapped_column(sa.Float, nullable=False)
    batch_size: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=4)
    use_qlora: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=False)

    # ── Results ──────────────────────────────────────────────────────────
    status: Mapped[str] = mapped_column(
        sa.String(20), nullable=False, default="pending"
    )  # pending | running | succeeded | failed

    adapter_path: Mapped[Optional[str]] = mapped_column(sa.String(1024), nullable=True)
    final_loss: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    total_steps: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    elapsed_secs: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    hub_url: Mapped[Optional[str]] = mapped_column(sa.String(512), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
