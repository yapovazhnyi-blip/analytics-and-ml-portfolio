"""
BigQuery connector tests.

All BigQuery client calls are mocked — no real GCP credentials or network
calls required. We mock at the `google.cloud.bigquery.Client` level so the
connector's business logic (query dispatch, DataFrame conversion, storage)
is fully tested while the actual BigQuery API is simulated.
"""

from __future__ import annotations

import json
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_client(df: pd.DataFrame):
    """Returns a mock BigQuery client whose query().result().to_dataframe() returns df."""
    mock_result = MagicMock()
    mock_result.to_dataframe.return_value = df

    mock_job = MagicMock()
    mock_job.result.return_value = mock_result

    mock_client = MagicMock()
    mock_client.query.return_value = mock_job
    return mock_client


SAMPLE_DF = pd.DataFrame({
    "user_id":  [1, 2, 3, 4, 5],
    "revenue":  [100.0, 250.5, 75.0, 420.0, 99.9],
    "country":  ["GB", "US", "DE", "US", "FR"],
    "churned":  [0, 0, 1, 0, 1],
})


# ══════════════════════════════════════════════════════════════════════════
# BigQueryConfig
# ══════════════════════════════════════════════════════════════════════════

class TestBigQueryConfig:

    def test_default_location_is_us(self):
        from connectors.bigquery_connector import BigQueryConfig
        cfg = BigQueryConfig(project_id="my-proj", query="SELECT 1")
        assert cfg.location == "US"

    def test_default_max_rows(self):
        from connectors.bigquery_connector import BigQueryConfig
        cfg = BigQueryConfig(project_id="my-proj", query="SELECT 1")
        assert cfg.max_rows == 1_000_000

    def test_credentials_json_optional(self):
        from connectors.bigquery_connector import BigQueryConfig
        cfg = BigQueryConfig(project_id="p", query="SELECT 1")
        assert cfg.credentials_json is None


# ══════════════════════════════════════════════════════════════════════════
# BigQueryConnector — unit tests (mocked client)
# ══════════════════════════════════════════════════════════════════════════

class TestBigQueryConnector:

    def _make_connector(self, **kwargs):
        from connectors.bigquery_connector import BigQueryConnector, BigQueryConfig
        cfg = BigQueryConfig(project_id="test-proj", query="SELECT *", **kwargs)
        return BigQueryConnector(cfg)

    @pytest.mark.asyncio
    async def test_ingest_returns_result(self, tmp_path):
        from connectors.bigquery_connector import BigQueryConnector, BigQueryConfig
        from storage.local import LocalStorage
        from storage.factory import StorageContext
        cfg = BigQueryConfig(project_id="p", query="SELECT *")
        conn = BigQueryConnector(cfg)
        with StorageContext(LocalStorage(str(tmp_path))):
            with patch.object(conn, "_make_client", return_value=_make_mock_client(SAMPLE_DF)):
                result = await conn.ingest()
        assert result.error is None
        assert result.source_type == "bigquery"

    @pytest.mark.asyncio
    async def test_ingest_correct_row_count(self, tmp_path):
        from connectors.bigquery_connector import BigQueryConnector, BigQueryConfig
        from storage.local import LocalStorage
        from storage.factory import StorageContext
        conn = BigQueryConnector(BigQueryConfig(project_id="p", query="SELECT *"))
        with StorageContext(LocalStorage(str(tmp_path))):
            with patch.object(conn, "_make_client", return_value=_make_mock_client(SAMPLE_DF)):
                result = await conn.ingest()
        assert result.row_count == len(SAMPLE_DF)

    @pytest.mark.asyncio
    async def test_ingest_correct_column_count(self, tmp_path):
        from connectors.bigquery_connector import BigQueryConnector, BigQueryConfig
        from storage.local import LocalStorage
        from storage.factory import StorageContext
        conn = BigQueryConnector(BigQueryConfig(project_id="p", query="SELECT *"))
        with StorageContext(LocalStorage(str(tmp_path))):
            with patch.object(conn, "_make_client", return_value=_make_mock_client(SAMPLE_DF)):
                result = await conn.ingest()
        assert result.column_count == len(SAMPLE_DF.columns)

    @pytest.mark.asyncio
    async def test_ingest_schema_has_all_columns(self, tmp_path):
        from connectors.bigquery_connector import BigQueryConnector, BigQueryConfig
        from storage.local import LocalStorage
        from storage.factory import StorageContext
        conn = BigQueryConnector(BigQueryConfig(project_id="p", query="SELECT *"))
        with StorageContext(LocalStorage(str(tmp_path))):
            with patch.object(conn, "_make_client", return_value=_make_mock_client(SAMPLE_DF)):
                result = await conn.ingest()
        col_names = {c.name for c in result.columns}
        assert col_names == set(SAMPLE_DF.columns)

    @pytest.mark.asyncio
    async def test_ingest_content_hash_is_set(self, tmp_path):
        from connectors.bigquery_connector import BigQueryConnector, BigQueryConfig
        from storage.local import LocalStorage
        from storage.factory import StorageContext
        conn = BigQueryConnector(BigQueryConfig(project_id="p", query="SELECT *"))
        with StorageContext(LocalStorage(str(tmp_path))):
            with patch.object(conn, "_make_client", return_value=_make_mock_client(SAMPLE_DF)):
                result = await conn.ingest()
        assert result.content_hash is not None
        assert len(result.content_hash) == 64

    @pytest.mark.asyncio
    async def test_file_is_saved(self, tmp_path):
        from connectors.bigquery_connector import BigQueryConnector, BigQueryConfig
        from storage.local import LocalStorage
        from storage.factory import StorageContext
        from pathlib import Path
        conn = BigQueryConnector(BigQueryConfig(project_id="p", query="SELECT *"))
        with StorageContext(LocalStorage(str(tmp_path))):
            with patch.object(conn, "_make_client", return_value=_make_mock_client(SAMPLE_DF)):
                result = await conn.ingest()
        assert result.file_path is not None
        assert Path(result.file_path).exists()

    @pytest.mark.asyncio
    async def test_saved_file_is_readable_csv(self, tmp_path):
        from connectors.bigquery_connector import BigQueryConnector, BigQueryConfig
        from storage.local import LocalStorage
        from storage.factory import StorageContext
        conn = BigQueryConnector(BigQueryConfig(project_id="p", query="SELECT *"))
        with StorageContext(LocalStorage(str(tmp_path))):
            with patch.object(conn, "_make_client", return_value=_make_mock_client(SAMPLE_DF)):
                result = await conn.ingest()
        df_loaded = pd.read_csv(result.file_path)
        assert list(df_loaded.columns) == list(SAMPLE_DF.columns)
        assert len(df_loaded) == len(SAMPLE_DF)

    @pytest.mark.asyncio
    async def test_max_rows_limit_applied(self, tmp_path):
        from connectors.bigquery_connector import BigQueryConnector, BigQueryConfig
        from storage.local import LocalStorage
        from storage.factory import StorageContext
        big_df = pd.DataFrame({"x": range(1000)})
        conn = BigQueryConnector(BigQueryConfig(project_id="p", query="SELECT *", max_rows=10))
        with StorageContext(LocalStorage(str(tmp_path))):
            with patch.object(conn, "_make_client", return_value=_make_mock_client(big_df)):
                result = await conn.ingest()
        assert result.row_count == 10

    @pytest.mark.asyncio
    async def test_client_error_returns_error_result(self, tmp_path):
        from connectors.bigquery_connector import BigQueryConnector, BigQueryConfig
        from storage.local import LocalStorage
        from storage.factory import StorageContext
        conn = BigQueryConnector(BigQueryConfig(project_id="p", query="SELECT *"))
        mock_client = MagicMock()
        mock_client.query.side_effect = Exception("quota exceeded")
        with StorageContext(LocalStorage(str(tmp_path))):
            with patch.object(conn, "_make_client", return_value=mock_client):
                result = await conn.ingest()
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_empty_result_handled(self, tmp_path):
        from connectors.bigquery_connector import BigQueryConnector, BigQueryConfig
        from storage.local import LocalStorage
        from storage.factory import StorageContext
        empty_df = pd.DataFrame({"col": pd.Series([], dtype="object")})
        conn = BigQueryConnector(BigQueryConfig(project_id="p", query="SELECT *"))
        with StorageContext(LocalStorage(str(tmp_path))):
            with patch.object(conn, "_make_client", return_value=_make_mock_client(empty_df)):
                result = await conn.ingest()
        assert result.row_count == 0








# ══════════════════════════════════════════════════════════════════════════
# BigQuery _make_client — credential paths
# ══════════════════════════════════════════════════════════════════════════

class TestBigQueryCredentials:

    def test_service_account_json_used_when_provided(self):
        from connectors.bigquery_connector import BigQueryConnector, BigQueryConfig
        fake_sa = json.dumps({
            "type": "service_account", "project_id": "test-proj",
            "private_key_id": "key123",
            "private_key": "-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----\n",
            "client_email": "sa@test-proj.iam.gserviceaccount.com",
            "client_id": "123456",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        })
        cfg = BigQueryConfig(project_id="test-proj", query="SELECT 1",
                             credentials_json=fake_sa)
        conn = BigQueryConnector(cfg)
        # Patch at the google.oauth2 level since _make_client imports locally
        with patch("google.oauth2.service_account.Credentials.from_service_account_info") as mock_creds, \
             patch("google.cloud.bigquery.Client") as mock_client_cls:
            mock_creds.return_value = MagicMock()
            mock_client_cls.return_value = MagicMock()
            conn._make_client()
            mock_creds.assert_called_once()

    def test_adc_used_when_no_credentials_json(self):
        from connectors.bigquery_connector import BigQueryConnector, BigQueryConfig
        cfg = BigQueryConfig(project_id="test-proj", query="SELECT 1")
        conn = BigQueryConnector(cfg)
        with patch("google.cloud.bigquery.Client") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()
            conn._make_client()
            # Called with project only (no credentials= kwarg for ADC)
            call_kwargs = mock_client_cls.call_args
            assert call_kwargs.kwargs.get("credentials") is None or \
                   "credentials" not in call_kwargs.kwargs


# ══════════════════════════════════════════════════════════════════════════
# API endpoint
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def bq_client(tmp_path):
    import sys, importlib, database as db_mod
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool
    from fastapi.testclient import TestClient
    from storage.factory import StorageContext
    from storage.local import LocalStorage

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    db_mod.engine = engine; db_mod.SessionFactory = factory; db_mod.AsyncSessionLocal = factory
    if "main" in sys.modules:
        importlib.reload(sys.modules["main"])
    import main as m
    with StorageContext(LocalStorage(str(tmp_path))):
        with TestClient(m.app, raise_server_exceptions=True) as c:
            yield c


class TestBigQueryAPI:

    def test_bigquery_endpoint_exists(self, bq_client):
        """Endpoint must return 422 on bad request (not 404)."""
        resp = bq_client.post("/api/v1/connectors/bigquery", json={})
        assert resp.status_code == 422   # missing required fields

    def test_bigquery_ingest_success(self, bq_client):
        """Full pipeline: mock BQ client → dataset created in DB."""
        with patch("connectors.bigquery_connector.BigQueryConnector._make_client") as mock_make:
            mock_make.return_value = _make_mock_client(SAMPLE_DF)
            resp = bq_client.post("/api/v1/connectors/bigquery", json={
                "project_id":   "test-proj",
                "query":        "SELECT * FROM dataset.table LIMIT 100",
                "dataset_name": "bq_test",
            })
        assert resp.status_code == 201
        data = resp.json()["data"]
        assert data["source_type"] == "bigquery"
        assert data["row_count"] == len(SAMPLE_DF)
        assert data["name"] == "bq_test"

    def test_bigquery_ingest_failure_returns_422(self, bq_client):
        """BQ query errors must surface as HTTP 422."""
        with patch("connectors.bigquery_connector.BigQueryConnector._make_client") as mock_make:
            mock_client = MagicMock()
            mock_client.query.side_effect = Exception("Access denied")
            mock_make.return_value = mock_client
            resp = bq_client.post("/api/v1/connectors/bigquery", json={
                "project_id":   "test-proj",
                "query":        "SELECT *",
                "dataset_name": "bq_fail",
            })
        assert resp.status_code == 422

    def test_schema_columns_captured(self):
        from connectors.base import infer_columns
        cols = infer_columns(SAMPLE_DF)
        col_names = {c.name for c in cols}
        assert "revenue" in col_names
        assert "country" in col_names
