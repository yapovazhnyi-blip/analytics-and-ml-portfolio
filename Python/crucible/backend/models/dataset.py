"""
Dataset model — represents a single ingested dataset in Crucible.

A dataset is the result of a connector run: raw data that has been
pulled from a source (CSV, SQL, REST API) and stored locally for
profiling and training. One connector can produce many datasets
(e.g. each sync of a REST API creates a new versioned dataset).
"""

from typing import Optional
import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base, TimestampMixin


class Dataset(Base, TimestampMixin):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Human-readable name, e.g. "titanic_v2" or "churn_2024_03"
    name: Mapped[str] = mapped_column(sa.String(255), nullable=False, index=True)

    # Source type — drives which connector was used
    source_type: Mapped[str] = mapped_column(
        sa.String(50), nullable=False
    )  # "csv" | "parquet" | "sql" | "rest_api"

    # Where the raw file lives on disk (for csv/parquet)
    # Null for SQL/API sources where data is queried on demand
    file_path: Mapped[Optional[str]] = mapped_column(sa.String(1024), nullable=True)

    # Schema snapshot — JSON array of {name, dtype, nullable} dicts
    # Stored at ingest time so profiling doesn't need to re-read the file
    schema_json: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    contract_json: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)   # data contract

    # Row/column counts — populated after ingest completes
    row_count: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    column_count: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)

    # SHA-256 of the raw file content — used for lineage DAG versioning
    content_hash: Mapped[Optional[str]] = mapped_column(sa.String(64), nullable=True)

    # Ingest status — drives UI state
    status: Mapped[str] = mapped_column(
        sa.String(20), nullable=False, default="pending"
    )  # "pending" | "ingesting" | "ready" | "error"

    error_message: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)

    def __repr__(self) -> str:
        return f"<Dataset id={self.id} name={self.name!r} status={self.status}>"
