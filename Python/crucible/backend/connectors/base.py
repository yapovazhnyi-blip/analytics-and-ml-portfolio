"""
Base connector interface for Crucible.

Every connector — file, SQL, REST — implements this protocol.
The router layer works against this interface, never against concrete
connector classes directly. This makes it straightforward to add new
source types without touching the router code.

IngestResult is the standard return type from any connector's ingest()
call. The datasets router reads from this to populate the Dataset ORM
record, so the shape must be consistent regardless of connector type.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, AsyncIterator

import pandas as pd


@dataclass
class ColumnInfo:
    name: str
    dtype: str       # string representation of the pandas dtype
    nullable: bool   # True if column contains any NaN/None values


@dataclass
class IngestResult:
    """
    Standard return type from any connector's ingest() method.

    All fields are optional — a connector populates what it can.
    For example, a streaming REST connector may not know row_count
    until it finishes paginating.
    """
    source_type: str                         # "csv" | "parquet" | "sql" | "rest_api"
    file_path: Optional[str] = None          # local path where data was saved
    storage_key: Optional[str] = None        # backend-agnostic storage key
    content_hash: Optional[str] = None       # SHA-256 of raw content
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    columns: list[ColumnInfo] = field(default_factory=list)
    error: Optional[str] = None              # set if ingest failed

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_schema_json(self) -> str:
        """Serialise column info to JSON for storage in Dataset.schema_json."""
        import json
        return json.dumps([
            {"name": c.name, "dtype": c.dtype, "nullable": c.nullable}
            for c in self.columns
        ])


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def infer_columns(df: pd.DataFrame) -> list[ColumnInfo]:
    """Extract column metadata from a DataFrame."""
    return [
        ColumnInfo(
            name=col,
            dtype=str(df[col].dtype),
            nullable=bool(df[col].isna().any()),
        )
        for col in df.columns
    ]


class BaseConnector(ABC):
    """
    Abstract base for all Crucible data source connectors.

    Subclasses implement ingest() and optionally stream_chunks()
    for large datasets that cannot fit in memory.
    """

    @abstractmethod
    async def ingest(self, **kwargs) -> IngestResult:
        """
        Pull data from the source and return an IngestResult.

        For file connectors: reads and saves the file, returns metadata.
        For SQL connectors: runs a query, saves result as Parquet, returns metadata.
        For REST connectors: paginates the API, saves result as Parquet, returns metadata.
        """
        ...

    async def stream_chunks(
        self,
        chunk_size: int = 1000,
        **kwargs,
    ) -> AsyncIterator[pd.DataFrame]:
        """
        Yields data in chunks without loading everything into memory.

        Default implementation loads everything and yields in one chunk.
        Override in connectors where true streaming is possible (SQL).
        """
        result = await self.ingest(**kwargs)
        if result.file_path:
            df = pd.read_parquet(result.file_path)
            for i in range(0, len(df), chunk_size):
                yield df.iloc[i:i + chunk_size]
