"""
Dataset schemas — Pydantic models for the /datasets API.

These are deliberately separate from the ORM model (models/dataset.py).
The ORM model describes how data is stored; these schemas describe what
the API accepts and returns. They are not always the same shape — for
example, the ORM model has internal fields like content_hash and
file_path that the API doesn't expose directly.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── Column schema (part of dataset schema snapshot) ───────────────────────

class ColumnSchema(BaseModel):
    name: str
    dtype: str
    nullable: bool


# ── Response schemas (what the API returns) ────────────────────────────────

class DatasetOut(BaseModel):
    """Full dataset representation returned by GET /datasets/{id}."""
    id: int
    name: str
    source_type: str
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    status: str
    error_message: Optional[str] = None
    schema_columns: Optional[list[ColumnSchema]] = None
    created_at: str  # ISO 8601

    model_config = {"from_attributes": True}


class DatasetSummary(BaseModel):
    """Lightweight representation used in list responses."""
    id: int
    name: str
    source_type: str
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    status: str
    created_at: str

    model_config = {"from_attributes": True}


# ── Request schemas (what the API accepts) ─────────────────────────────────

class DatasetUploadMeta(BaseModel):
    """
    Metadata accompanying a file upload (CSV/Parquet).
    The file itself arrives as a multipart form field.
    """
    name: str = Field(..., min_length=1, max_length=255, description="Human-readable dataset name")
    description: Optional[str] = Field(None, max_length=1024)


class DatasetFromSQL(BaseModel):
    """Request body for creating a dataset from a SQL connector."""
    name: str = Field(..., min_length=1, max_length=255)
    connector_id: int = Field(..., description="ID of a configured SQL connector")
    query: str = Field(
        ...,
        min_length=10,
        max_length=10_000,
        description=(
            "SQL SELECT query to run. Must start with SELECT or WITH. "
            "DDL and DML statements are not permitted."
        ),
    )

    def model_post_init(self, __context) -> None:
        """
        Validates the query is a safe read-only SELECT statement.

        This is not a substitute for database-level permissions — the
        connector's credentials should also be read-only. This is a
        defence-in-depth measure that catches obvious misuse early and
        gives the user a clear error message rather than a database error.
        """
        import re
        stripped = self.query.strip().upper()

        # Must begin with SELECT or WITH (CTE)
        if not (stripped.startswith("SELECT") or stripped.startswith("WITH")):
            raise ValueError(
                "Only SELECT and WITH (CTE) queries are permitted. "
                f"Query starts with: {self.query.strip()[:30]!r}"
            )

        # Block DDL and DML keywords that have no place in a read query
        forbidden = re.compile(
            r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|"
            r"EXEC|EXECUTE|GRANT|REVOKE|COPY|LOAD|MERGE|REPLACE)\b",
            re.IGNORECASE,
        )
        match = forbidden.search(self.query)
        if match:
            raise ValueError(
                f"Forbidden SQL keyword '{match.group()}' detected. "
                "Only read-only SELECT queries are permitted."
            )


class DatasetFromAPI(BaseModel):
    """Request body for creating a dataset from a REST API connector."""
    name: str = Field(..., min_length=1, max_length=255)
    connector_id: int = Field(..., description="ID of a configured REST/OAuth2 connector")
    endpoint: str = Field(..., description="API endpoint path, e.g. /v1/records")
    params: Optional[dict] = Field(
        None,
        description=(
            "Query parameters. Special keys (prefixed _) control pagination: "
            "_strategy (link|cursor|offset|none), _records_key, "
            "_cursor_field, _page_size."
        ),
    )
