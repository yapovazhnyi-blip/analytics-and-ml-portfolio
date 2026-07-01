"""
File connector for Crucible — handles CSV and Parquet uploads.

Responsibilities:
  1. Save raw bytes to the dataset storage directory with a UUID filename
  2. Compute SHA-256 hash (used for lineage DAG versioning in Phase 3)
  3. Infer schema from a sample read — dtype per column, nullability
  4. Count rows without loading the full file into memory where possible
  5. Return a structured IngestResult

Design decisions:
  - Parquet: row count is read from metadata (zero I/O on data rows)
  - CSV: row count is computed from a full read; for very large CSVs
    this is acceptable in Phase 1. Phase 2 can add chunked counting.
  - Schema inference always reads only the first SCHEMA_SAMPLE_ROWS rows
    so a 10M-row CSV doesn't OOM the server during upload.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

from connectors.base import BaseConnector, IngestResult, infer_columns, sha256_of_bytes

# How many rows to read for schema inference on large files
SCHEMA_SAMPLE_ROWS = 50_000


class FileConnector(BaseConnector):
    """
    Ingests a CSV or Parquet file from raw bytes.

    Usage:
        connector = FileConnector(storage_dir="/data/datasets")
        result = await connector.ingest(
            content=file_bytes,
            filename="sales_2024.csv",
            dataset_name="sales_q1",
        )
    """

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def ingest(
        self,
        content: bytes,
        filename: str,
        dataset_name: Optional[str] = None,
        **kwargs,
    ) -> IngestResult:
        """
        Save file to disk, compute hash, infer schema.

        Args:
            content:      Raw file bytes
            filename:     Original filename (used to detect extension)
            dataset_name: Human label (unused in storage, for logging only)
        """
        ext = Path(filename).suffix.lower()
        if ext not in (".csv", ".parquet"):
            return IngestResult(
                source_type="unknown",
                error=f"Unsupported file type '{ext}'. Supported: .csv, .parquet",
            )

        source_type = ext.lstrip(".")  # "csv" or "parquet"

        # ── Save via StorageBackend ────────────────────────────────────────
        safe_name = f"{uuid.uuid4().hex}{ext}"
        storage_key = f"datasets/{safe_name}"

        try:
            from storage.factory import get_storage
            backend = get_storage()
            backend.write(storage_key, content)
            # For local backend, resolve to the actual path for downstream operations
            # (profiling, training) that still use pathlib directly
            if hasattr(backend, "_resolve"):
                file_path = backend._resolve(storage_key)
            else:
                # S3 or other backend: download to local tmp for schema inference
                import tempfile
                tmp_dir = Path(tempfile.mkdtemp())
                file_path = tmp_dir / safe_name
                backend.download_file(storage_key, str(file_path))
        except Exception as exc:
            return IngestResult(source_type=source_type, error=f"Failed to write file: {exc}")

        # ── Hash ───────────────────────────────────────────────────────────
        content_hash = sha256_of_bytes(content)

        # ── Schema + row count ─────────────────────────────────────────────
        try:
            if source_type == "parquet":
                result = self._ingest_parquet(file_path, content_hash)
            else:
                result = self._ingest_csv(file_path, content_hash)
            # Store the storage key (not just local path) for backend portability
            if result.file_path:
                result.storage_key = storage_key
            return result
        except Exception as exc:
            return IngestResult(
                source_type=source_type,
                file_path=str(file_path),
                content_hash=content_hash,
                error=f"Schema inference failed: {exc}",
            )

    def _ingest_parquet(self, file_path: Path, content_hash: str) -> IngestResult:
        """
        Parquet metadata is in the file footer — no need to read data rows
        to get row count or schema. This is O(1) regardless of file size.
        """
        pf = pq.ParquetFile(file_path)
        meta = pf.metadata
        row_count = meta.num_rows

        # Read schema from metadata — no data scan required
        schema = pf.schema_arrow
        columns = []
        for i, name in enumerate(schema.names):
            field = schema.field(name)
            # Arrow types → pandas dtype approximation for display
            arrow_type = str(field.type)
            dtype = _arrow_to_pandas_dtype(arrow_type)
            columns.append(
                # nullable: Arrow schema encodes nullability in the type
                __import__("connectors.base", fromlist=["ColumnInfo"]).ColumnInfo(
                    name=name,
                    dtype=dtype,
                    nullable=field.nullable,
                )
            )

        return IngestResult(
            source_type="parquet",
            file_path=str(file_path),
            content_hash=content_hash,
            row_count=row_count,
            column_count=len(columns),
            columns=columns,
        )

    def _ingest_csv(self, file_path: Path, content_hash: str) -> IngestResult:
        """
        CSV has no metadata footer — we must read the file.

        Schema inference uses SCHEMA_SAMPLE_ROWS rows.
        Row count requires a full read (we reuse the sample when small enough,
        do a second lightweight pass for large files).
        """
        from connectors.base import ColumnInfo

        # Sample read for schema
        sample_df = pd.read_csv(file_path, nrows=SCHEMA_SAMPLE_ROWS)
        columns = infer_columns(sample_df)
        n_sample = len(sample_df)

        if n_sample < SCHEMA_SAMPLE_ROWS:
            # File fit in sample — we already have the full row count
            row_count = n_sample
        else:
            # Count remaining rows without loading data into memory:
            # read only the index column (fast) and sum chunk lengths
            row_count = sum(
                len(chunk)
                for chunk in pd.read_csv(
                    file_path,
                    usecols=[0],           # only first column — minimises I/O
                    chunksize=100_000,
                )
            )

        return IngestResult(
            source_type="csv",
            file_path=str(file_path),
            content_hash=content_hash,
            row_count=row_count,
            column_count=len(columns),
            columns=columns,
        )


def _arrow_to_pandas_dtype(arrow_type: str) -> str:
    """
    Maps Arrow type strings to readable pandas dtype labels.
    Arrow types are verbose (e.g. "timestamp[us, tz=UTC]") — we simplify.
    """
    t = arrow_type.lower()
    if t.startswith("int"):
        return "int64"
    if t.startswith("uint"):
        return "uint64"
    if t.startswith("float") or t.startswith("double"):
        return "float64"
    if t.startswith("bool"):
        return "bool"
    if t.startswith("timestamp"):
        return "datetime64"
    if t.startswith("date"):
        return "date"
    if t.startswith("list"):
        return "list"
    if t.startswith("struct"):
        return "struct"
    return "object"
