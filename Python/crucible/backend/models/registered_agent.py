"""ORM model for the Agent Registry — fine-tuned agents available for use."""

from __future__ import annotations
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base, TimestampMixin


class RegisteredAgent(Base, TimestampMixin):
    """
    A registered, importable/exportable agent — the Agent Registry entry.

    Each row maps a human-readable agent name to a .crucible bundle
    (LoRA adapter + agent config + benchmark results), making it
    selectable from /agent/run via the optional agent_id parameter.

    LIFECYCLE
    ---------
    1. Traces are captured (AgentTrace rows) from real usage
    2. Traces are converted to SFT/DPO training data
    3. The fine-tuning studio trains a LoRA adapter on that data
    4. export_bundle() packages the adapter + config into a .crucible file
    5. A RegisteredAgent row is created pointing at the bundle
    6. The agent becomes selectable for future /agent/run calls
    7. import_bundle() lets a DIFFERENT Crucible instance (or any
       PEFT-compatible system) load the same bundle
    """

    __tablename__ = "registered_agents"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)

    name: Mapped[str] = mapped_column(sa.String(100), unique=True, nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)

    base_model: Mapped[str] = mapped_column(sa.String(200), nullable=False)
    adapter_path: Mapped[Optional[str]] = mapped_column(sa.String(1024), nullable=True)
    bundle_path: Mapped[Optional[str]] = mapped_column(sa.String(1024), nullable=True)

    # Full manifest (tool config, system prompt, training metadata) as JSON
    manifest_json: Mapped[str] = mapped_column(sa.Text, nullable=False)

    n_training_traces: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    training_method: Mapped[str] = mapped_column(sa.String(20), nullable=False, default="sft")  # "sft" | "dpo"

    # Benchmark results from the last evaluation run, as JSON (may be null
    # if the agent has not been benchmarked yet)
    benchmark_json: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)

    status: Mapped[str] = mapped_column(sa.String(20), nullable=False, default="active")  # active | archived

    def __repr__(self) -> str:
        return f"<RegisteredAgent id={self.id} name={self.name!r} method={self.training_method!r}>"
