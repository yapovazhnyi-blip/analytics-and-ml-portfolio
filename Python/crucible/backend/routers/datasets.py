"""
Datasets router — /api/v1/datasets

Endpoints:
  GET    /datasets          — paginated list of all datasets
  GET    /datasets/{id}     — single dataset with full schema
  POST   /datasets/upload   — ingest a CSV or Parquet file
  POST   /datasets/from-sql — create dataset from a SQL connector
  POST   /datasets/from-api — create dataset from a REST API connector
  DELETE /datasets/{id}     — delete dataset and its file

All heavy lifting is delegated to connector classes, not done inline.
"""

import json
import os
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from auth.dependencies import get_current_user
from fastapi import Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from connectors import FileConnector, SQLConnector
from database import get_db
from models import Dataset
from schemas.common import DataResponse, PaginatedResponse, make_pagination_meta
from schemas.dataset import DatasetFromAPI, DatasetFromSQL, DatasetOut, DatasetSummary

router = APIRouter(prefix="/datasets", tags=["datasets"], dependencies=[Depends(get_current_user)])


# ── Helper: ORM → schema ───────────────────────────────────────────────────

def _dataset_out(ds: Dataset) -> DatasetOut:
    schema_columns = None
    if ds.schema_json:
        try:
            schema_columns = json.loads(ds.schema_json)
        except json.JSONDecodeError:
            pass
    return DatasetOut(
        id=ds.id,
        name=ds.name,
        source_type=ds.source_type,
        row_count=ds.row_count,
        column_count=ds.column_count,
        status=ds.status,
        error_message=ds.error_message,
        schema_columns=schema_columns,
        created_at=ds.created_at.isoformat(),
    )


def _dataset_summary(ds: Dataset) -> DatasetSummary:
    return DatasetSummary(
        id=ds.id,
        name=ds.name,
        source_type=ds.source_type,
        row_count=ds.row_count,
        column_count=ds.column_count,
        status=ds.status,
        created_at=ds.created_at.isoformat(),
    )


# ── List ───────────────────────────────────────────────────────────────────

@router.get("", response_model=PaginatedResponse[DatasetSummary])
async def list_datasets(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Dataset).order_by(Dataset.created_at.desc())
    if status:
        stmt = stmt.where(Dataset.status == status)

    total = await db.scalar(select(func.count()).select_from(stmt.subquery()))
    rows = await db.scalars(stmt.offset((page - 1) * page_size).limit(page_size))

    return PaginatedResponse(
        data=[_dataset_summary(ds) for ds in rows.all()],
        pagination=make_pagination_meta(page, page_size, total or 0),
    )


@router.get("/cursor")
async def list_datasets_cursor(
    cursor: str | None = Query(default=None, description="Opaque cursor from a previous response's next_cursor."),
    limit: int = Query(default=20, ge=1, le=200),
    status: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """
    Cursor-paginated (keyset) dataset listing — constant-speed pagination
    regardless of how deep into the result set you are.

    First request: omit `cursor`. Pass the returned `next_cursor` to get
    the next page. Stop when `has_more` is false.
    """
    from schemas.cursor_pagination import paginate_by_cursor

    stmt = select(Dataset)
    if status:
        stmt = stmt.where(Dataset.status == status)

    try:
        page = await paginate_by_cursor(db, stmt, Dataset, cursor, limit)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    return DataResponse(data={
        "items":       [_dataset_summary(ds) for ds in page.items],
        "next_cursor": page.next_cursor,
        "has_more":    page.has_more,
    })


# ── Get single ─────────────────────────────────────────────────────────────

@router.get("/{dataset_id}", response_model=DataResponse[DatasetOut])
async def get_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    return DataResponse(data=_dataset_out(ds))


# ── Upload CSV / Parquet ───────────────────────────────────────────────────

@router.post("/upload", response_model=DataResponse[DatasetOut], status_code=201)
async def upload_dataset(
    file: UploadFile = File(...),
    name: Annotated[str, Form()] = "",
    db: AsyncSession = Depends(get_db),
):
    """
    Ingests a CSV or Parquet file via the FileConnector.

    Security controls applied here:
      1. Extension allowlist — only .csv and .parquet accepted
      2. File size cap — rejects files over MAX_UPLOAD_BYTES before reading
         the full content into memory (prevents OOM on giant uploads)
      3. Magic-bytes check — Parquet files must start with PAR1; CSV files
         must not start with Parquet or other binary magic bytes
    """
    MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB

    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".csv", ".parquet"):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Allowed: .csv, .parquet",
        )

    # Read with size guard — avoids reading a 10GB file into RAM
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024*1024)} MB.",
        )

    # Magic-bytes validation — extension alone is not trustworthy
    # Parquet magic: first 4 bytes are b"PAR1"
    if ext == ".parquet" and not content[:4] == b"PAR1":
        raise HTTPException(
            status_code=422,
            detail="File does not appear to be a valid Parquet file (missing PAR1 magic bytes).",
        )
    if ext == ".csv" and content[:4] == b"PAR1":
        raise HTTPException(
            status_code=422,
            detail="File appears to be Parquet but was uploaded with a .csv extension.",
        )

    dataset_name = name.strip() or os.path.splitext(filename)[0]
    connector = FileConnector(storage_dir=settings.dataset_storage_path)
    result = await connector.ingest(content=content, filename=filename, dataset_name=dataset_name)

    ds = Dataset(
        name=dataset_name,
        source_type=result.source_type,
        file_path=result.file_path,
        content_hash=result.content_hash,
        row_count=result.row_count,
        column_count=result.column_count,
        schema_json=result.to_schema_json() if result.columns else None,
        status="ready" if result.succeeded else "error",
        error_message=result.error,
    )
    db.add(ds)
    await db.flush()
    await db.refresh(ds)
    return DataResponse(data=_dataset_out(ds))


# ── From SQL connector ─────────────────────────────────────────────────────

@router.post("/from-sql", response_model=DataResponse[DatasetOut], status_code=201)
async def dataset_from_sql(body: DatasetFromSQL, db: AsyncSession = Depends(get_db)):
    """
    Runs a SQL query against a configured connector and saves the result.
    The connector's db_url is retrieved from the Connector record.
    """
    from models import Connector

    connector_rec = await db.get(Connector, body.connector_id)
    if not connector_rec:
        raise HTTPException(status_code=404, detail=f"Connector {body.connector_id} not found")
    if connector_rec.connector_type not in ("sql_postgres", "sql_sqlite"):
        raise HTTPException(status_code=422, detail="Connector is not a SQL connector")
    if not connector_rec.db_url_encrypted:
        raise HTTPException(status_code=422, detail="Connector has no database URL configured")

    # TODO Phase 1: decrypt db_url_encrypted before passing here
    from storage.encryption import decrypt
    db_url = decrypt(connector_rec.db_url_encrypted)

    # Create a pending record immediately so the UI has something to show
    ds = Dataset(name=body.name, source_type="sql", status="ingesting")
    db.add(ds)
    await db.flush()
    await db.refresh(ds)

    # Run the query via SQLConnector
    sql_connector = SQLConnector(db_url=db_url, storage_dir=settings.dataset_storage_path)
    try:
        result = await sql_connector.ingest(query=body.query)
    finally:
        await sql_connector.close()

    ds.file_path = result.file_path
    ds.content_hash = result.content_hash
    ds.row_count = result.row_count
    ds.column_count = result.column_count
    ds.schema_json = result.to_schema_json() if result.columns else None
    ds.status = "ready" if result.succeeded else "error"
    ds.error_message = result.error

    await db.flush()
    await db.refresh(ds)
    return DataResponse(data=_dataset_out(ds))


# ── From REST API connector ────────────────────────────────────────────────

@router.post("/from-api", response_model=DataResponse[DatasetOut], status_code=201)
async def dataset_from_api(body: DatasetFromAPI, db: AsyncSession = Depends(get_db)):
    """
    Fetches data from a REST API connector and creates a dataset.

    Uses the ConnectorFactory to instantiate the right connector from the
    stored credentials, then paginates through all results and saves them
    as Parquet. The connector handles authentication, pagination strategy,
    and rate limiting transparently.

    Pagination config can be provided in body.params under special keys:
      _strategy: "link" | "cursor" | "offset" | "none"
      _records_key: JSON key containing the array of records
      _cursor_field: dot-notation path to the next-cursor value
      _page_size: records per page (default 100)
    """
    from models import Connector as ConnectorModel
    from connectors.factory import ConnectorFactory
    from connectors.rest_connector import PaginationConfig

    connector_rec = await db.get(ConnectorModel, body.connector_id)
    if not connector_rec:
        raise HTTPException(status_code=404, detail=f"Connector {body.connector_id} not found")
    if connector_rec.connector_type not in ("rest_oauth2", "rest_api_key"):
        raise HTTPException(status_code=422, detail="Connector is not a REST connector")
    if connector_rec.status != "active":
        raise HTTPException(
            status_code=422,
            detail=(
                f"Connector is not active (status: {connector_rec.status}). "
                "For OAuth2 connectors, visit GET /connectors/{id}/authorize first."
            ),
        )

    # Extract pagination config from params (prefixed with _)
    params = dict(body.params or {})
    pagination = PaginationConfig(
        strategy=params.pop("_strategy", "link"),
        records_key=params.pop("_records_key", None),
        cursor_field=params.pop("_cursor_field", None),
        page_size=int(params.pop("_page_size", 100)),
    )

    # Create a pending record immediately so the UI can track state
    ds = Dataset(name=body.name, source_type="rest_api", status="ingesting")
    db.add(ds)
    await db.flush()
    await db.refresh(ds)

    try:
        rest_connector = ConnectorFactory.rest(connector_rec, pagination=pagination)
        result = await rest_connector.ingest(endpoint=body.endpoint, params=params)
    except Exception as exc:
        ds.status = "error"
        ds.error_message = str(exc)
        await db.flush()
        await db.refresh(ds)
        return DataResponse(data=_dataset_out(ds))
    finally:
        try:
            await rest_connector.close()
        except Exception:
            pass

    ds.file_path = result.file_path
    ds.content_hash = result.content_hash
    ds.row_count = result.row_count
    ds.column_count = result.column_count
    ds.schema_json = result.to_schema_json() if result.columns else None
    ds.status = "ready" if result.succeeded else "error"
    ds.error_message = result.error

    await db.flush()
    await db.refresh(ds)
    return DataResponse(data=_dataset_out(ds))


# ── Delete ─────────────────────────────────────────────────────────────────

@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    if ds.file_path and os.path.exists(ds.file_path):
        os.remove(ds.file_path)
    await db.delete(ds)
