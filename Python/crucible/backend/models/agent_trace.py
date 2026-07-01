"""ORM model for captured agent traces — raw material for fine-tuning data."""

from __future__ import annotations
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base, TimestampMixin


class AgentTrace(Base, TimestampMixin):
    """
    A captured record of one completed agent session — either a single
    ReAct agent run or a multi-agent (LangGraph supervisor) run.

    Traces are the raw material for Crucible's agent training pipeline:
      capture (this table) → convert to SFT/DPO format → fine-tune → export

    quality_score is populated lazily (not at capture time) by running the
    trace's final_answer through the LLM evaluation framework. A trace
    without a quality_score has not yet been scored — score_pending=True.
    """

    __tablename__ = "agent_traces"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)

    # "react" (single ReActRunner) | "multi" (LangGraph supervisor+specialists)
    agent_type: Mapped[str] = mapped_column(sa.String(20), nullable=False, index=True)

    goal: Mapped[str] = mapped_column(sa.Text, nullable=False)
    final_answer: Mapped[str] = mapped_column(sa.Text, nullable=False, default="")

    # Full structured event trace (tool calls, results, reasoning) as JSON.
    # This is what to_alpaca() / to_sharegpt() parse to build training samples.
    steps_json: Mapped[str] = mapped_column(sa.Text, nullable=False)

    n_tool_calls: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    elapsed_secs: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    succeeded: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=True)
    error_message: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)

    # Populated lazily via LLMJudge — None means "not yet scored"
    quality_score: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    score_pending: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=True)

    # Set once this trace has been included in an exported training dataset,
    # to avoid re-using the same trace across multiple training runs unintentionally
    used_in_training: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=False)

    def __repr__(self) -> str:
        return f"<AgentTrace id={self.id} type={self.agent_type!r} goal={self.goal[:40]!r}>"
