"""
Connectors router — /api/v1/connectors

Endpoints:
  GET    /connectors          — list all configured connectors
  GET    /connectors/{id}     — single connector (no sensitive fields)
  POST   /connectors/sql      — create a SQL connector
  POST   /connectors/oauth    — create an OAuth2 connector
  POST   /connectors/{id}/test — test the connection
  DELETE /connectors/{id}     — remove connector

Security note: client_secret, db_url, and tokens are never returned
in responses. They're write-only — accepted on create, encrypted at
rest, and decrypted only when the connector is actually used.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from auth.dependencies import get_current_user
from fastapi import Depends
from pydantic import BaseModel, Field
from typing import Optional
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import Connector
from schemas.common import DataResponse, PaginatedResponse, make_pagination_meta
from schemas.connector import (
    ConnectorOut,
    ConnectorSummary,
    ConnectorTestResult,
    OAuthConnectorCreate,
    SQLConnectorCreate,
)

router = APIRouter(prefix="/connectors", tags=["connectors"], dependencies=[Depends(get_current_user)])


# ── Helper: ORM → schema ───────────────────────────────────────────────────

def _connector_out(c: Connector) -> ConnectorOut:
    return ConnectorOut(
        id=c.id,
        name=c.name,
        connector_type=c.connector_type,
        status=c.status,
        base_url=c.base_url,
        client_id=c.client_id,
        oauth2_flow=c.oauth2_flow,
        token_url=c.token_url,
        auth_url=c.auth_url,
        use_pkce=c.use_pkce,
        last_used_at=c.last_used_at,
        created_at=c.created_at.isoformat(),
    )


def _connector_summary(c: Connector) -> ConnectorSummary:
    return ConnectorSummary(
        id=c.id,
        name=c.name,
        connector_type=c.connector_type,
        status=c.status,
        created_at=c.created_at.isoformat(),
    )


# ── List ───────────────────────────────────────────────────────────────────

@router.get("", response_model=PaginatedResponse[ConnectorSummary])
async def list_connectors(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Connector).order_by(Connector.created_at.desc())
    total = await db.scalar(select(func.count()).select_from(stmt.subquery()))
    rows = await db.scalars(stmt.offset((page - 1) * page_size).limit(page_size))
    connectors = rows.all()

    return PaginatedResponse(
        data=[_connector_summary(c) for c in connectors],
        pagination=make_pagination_meta(page, page_size, total or 0),
    )


# ── Get single ─────────────────────────────────────────────────────────────

@router.get("/{connector_id}", response_model=DataResponse[ConnectorOut])
async def get_connector(connector_id: int, db: AsyncSession = Depends(get_db)):
    c = await db.get(Connector, connector_id)
    if not c:
        raise HTTPException(status_code=404, detail=f"Connector {connector_id} not found")
    return DataResponse(data=_connector_out(c))


# ── Create SQL connector ───────────────────────────────────────────────────

@router.post("/sql", response_model=DataResponse[ConnectorOut], status_code=201)
async def create_sql_connector(
    body: SQLConnectorCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Stores a SQL connection string encrypted at rest using Fernet.

    If ENCRYPTION_KEY is not set, falls back to plaintext with a log
    warning — this keeps Phase 1 development frictionless while the
    secure path activates automatically when the key is present.
    """
    from storage.encryption import encrypt
    c = Connector(
        name=body.name,
        connector_type=body.connector_type,
        db_url_encrypted=encrypt(body.db_url),
        status="unconfigured",
    )
    db.add(c)
    await db.flush()
    await db.refresh(c)
    return DataResponse(data=_connector_out(c))


# ── Create OAuth2 connector ────────────────────────────────────────────────

@router.post("/oauth", response_model=DataResponse[ConnectorOut], status_code=201)
async def create_oauth_connector(
    body: OAuthConnectorCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Stores an OAuth2 connector configuration.

    The five provider variance flags (token_body_format, response_format,
    client_auth, use_pkce, token_url) are stored so the correct OAuth2
    connector class can be instantiated at connection time without
    per-provider subclasses — validated in the OAuth2 spike.
    """
    from storage.encryption import encrypt
    c = Connector(
        name=body.name,
        connector_type="rest_oauth2",
        base_url=body.base_url,
        client_id=body.client_id,
        client_secret_encrypted=encrypt(body.client_secret),
        oauth2_flow=body.oauth2_flow,
        # OAuth2 endpoint URLs — stored explicitly, not derived from base_url,
        # because they are almost never the same host (e.g. GitHub: base_url is
        # api.github.com but token_url is github.com/login/oauth/access_token)
        token_url=body.token_url,
        auth_url=body.auth_url,
        token_body_format=body.token_body_format,
        response_format=body.response_format,
        client_auth=body.client_auth,
        use_pkce=body.use_pkce,
        status="unconfigured",
    )
    db.add(c)
    await db.flush()
    await db.refresh(c)
    return DataResponse(data=_connector_out(c))


# ── Test connection ────────────────────────────────────────────────────────

@router.post("/{connector_id}/test", response_model=DataResponse[ConnectorTestResult])
async def test_connector(connector_id: int, db: AsyncSession = Depends(get_db)):
    """
    Tests connectivity for a configured connector.

    SQL connectors: runs SELECT 1 against the configured database.
    REST connectors: makes an authenticated GET to the base URL.
    Returns latency so the UI can show connection quality.

    The connector status is updated to "active" on success or "error"
    on failure, so users can see at a glance whether a connector is
    working without needing to run a dataset ingest.
    """
    import time
    from connectors.factory import ConnectorFactory

    c = await db.get(Connector, connector_id)
    if not c:
        raise HTTPException(status_code=404, detail=f"Connector {connector_id} not found")

    t0 = time.monotonic()
    success = False
    message = ""

    try:
        if c.connector_type in ("sql_postgres", "sql_sqlite"):
            sql_connector = ConnectorFactory.sql(c)
            ok, err = await sql_connector.test_connection()
            await sql_connector.close()
            success = ok
            message = "Connection successful" if ok else (err or "Connection failed")

        elif c.connector_type in ("rest_oauth2", "rest_api_key"):
            rest_connector = ConnectorFactory.rest(c)
            ok, err = await rest_connector.test_connection()
            await rest_connector.close()
            success = ok
            message = "Connection successful" if ok else (err or "Connection failed")

        else:
            # File connectors don't need a connectivity test
            success = True
            message = "File connector — no connectivity test needed"

    except Exception as exc:
        success = False
        message = str(exc)

    latency_ms = round((time.monotonic() - t0) * 1000, 2)

    # Update connector status to reflect the test result
    c.status = "active" if success else "error"
    c.last_used_at = time.time()

    result = ConnectorTestResult(
        success=success,
        message=message,
        latency_ms=latency_ms,
    )
    return DataResponse(data=result)


# ── Delete ─────────────────────────────────────────────────────────────────

@router.delete("/{connector_id}", status_code=204)
async def delete_connector(connector_id: int, db: AsyncSession = Depends(get_db)):
    c = await db.get(Connector, connector_id)
    if not c:
        raise HTTPException(status_code=404, detail=f"Connector {connector_id} not found")
    await db.delete(c)


# ── BigQuery connector ────────────────────────────────────────────────────────

class BigQueryConnectorRequest(BaseModel):
    project_id: str = Field(..., description="Google Cloud project ID.")
    query: str = Field(
        ...,
        description=(
            "SQL query to execute. Use LIMIT to avoid full-table scans. "
            "Example: SELECT * FROM `project.dataset.table` LIMIT 10000"
        ),
    )
    credentials_json: Optional[str] = Field(
        None,
        description=(
            "Service account JSON as a string. Leave empty to use Application "
            "Default Credentials (requires GOOGLE_APPLICATION_CREDENTIALS env var)."
        ),
    )
    location: str = Field(default="US", description="BigQuery dataset location.")
    max_rows: int = Field(
        default=1_000_000, ge=1, le=5_000_000,
        description="Maximum rows to fetch. Prevents accidental full-table pulls.",
    )
    dataset_name: str = Field(
        ..., description="Name for the created dataset in Crucible."
    )


@router.post("/bigquery", status_code=201)
async def ingest_bigquery(
    body: BigQueryConnectorRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Executes a SQL query against Google BigQuery and ingests the result
    as a Crucible dataset.

    Authentication:
      - Provide credentials_json (service account JSON string), OR
      - Set GOOGLE_APPLICATION_CREDENTIALS env var (Application Default Credentials)

    For large tables always include a LIMIT clause — the default cap is 1M rows.
    Results are saved through the configured StorageBackend (local or S3).
    """
    from connectors.bigquery_connector import BigQueryConnector, BigQueryConfig
    from models.dataset import Dataset

    config = BigQueryConfig(
        project_id=body.project_id,
        query=body.query,
        credentials_json=body.credentials_json,
        location=body.location,
        max_rows=body.max_rows,
    )

    connector = BigQueryConnector(config)
    result = await connector.ingest()

    if result.error:
        raise HTTPException(422, f"BigQuery ingestion failed: {result.error}")

    ds = Dataset(
        name=body.dataset_name,
        source_type="bigquery",
        file_path=result.file_path,
        schema_json=result.to_schema_json(),
        row_count=result.row_count,
        column_count=result.column_count,
        content_hash=result.content_hash,
        status="ready",
    )
    db.add(ds)
    await db.flush()
    await db.refresh(ds)

    return DataResponse(data={
        "id":           ds.id,
        "name":         ds.name,
        "source_type":  ds.source_type,
        "row_count":    ds.row_count,
        "column_count": ds.column_count,
        "status":       ds.status,
        "query":        body.query[:200],
    })
