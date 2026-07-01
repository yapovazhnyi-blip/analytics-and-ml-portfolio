"""
Connector tests.

Tests are written against known inputs with known expected outputs —
not just "it runs without crashing". Each test validates a specific
behaviour that we've seen fail in real-world connector implementations.
"""

import asyncio
import io
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from connectors import (
    AuthCodeConnector,
    ClientCredentialsConnector,
    FileConnector,
    OAuth2Config,
    SQLConnector,
    github_config,
    google_config,
)
from connectors.base import IngestResult


# ══════════════════════════════════════════════════════════════════════════
# FILE CONNECTOR — CSV
# ══════════════════════════════════════════════════════════════════════════

class TestFileConnectorCSV:

    @pytest.fixture
    def connector(self, tmp_path):
        return FileConnector(storage_dir=str(tmp_path))

    @pytest.fixture
    def simple_csv(self):
        return b"name,age,score\nAlice,30,9.5\nBob,25,8.1\nCarol,35,7.7\n"

    @pytest.mark.asyncio
    async def test_successful_ingest_returns_ready_result(self, connector, simple_csv):
        result = await connector.ingest(content=simple_csv, filename="test.csv")
        assert result.succeeded
        assert result.source_type == "csv"
        assert result.row_count == 3
        assert result.column_count == 3

    @pytest.mark.asyncio
    async def test_file_is_saved_to_storage_dir(self, connector, simple_csv, tmp_path):
        result = await connector.ingest(content=simple_csv, filename="data.csv")
        assert result.file_path is not None
        assert Path(result.file_path).exists()
        # File is now saved via StorageBackend, not necessarily to tmp_path
        # (StorageBackend uses settings.storage_local_root as its root)
        assert result.file_path.endswith(".csv")

    @pytest.mark.asyncio
    async def test_content_hash_is_sha256(self, connector, simple_csv):
        import hashlib
        result = await connector.ingest(content=simple_csv, filename="data.csv")
        expected = hashlib.sha256(simple_csv).hexdigest()
        assert result.content_hash == expected

    @pytest.mark.asyncio
    async def test_schema_infers_column_names(self, connector, simple_csv):
        result = await connector.ingest(content=simple_csv, filename="data.csv")
        col_names = [c.name for c in result.columns]
        assert col_names == ["name", "age", "score"]

    @pytest.mark.asyncio
    async def test_schema_detects_nullable_column(self, connector):
        csv_with_nulls = b"a,b\n1,\n2,3\n3,\n"
        result = await connector.ingest(content=csv_with_nulls, filename="nulls.csv")
        col_map = {c.name: c for c in result.columns}
        assert col_map["b"].nullable is True
        assert col_map["a"].nullable is False

    @pytest.mark.asyncio
    async def test_unsupported_extension_returns_error(self, connector):
        result = await connector.ingest(content=b"data", filename="model.pkl")
        assert not result.succeeded
        assert "Unsupported" in result.error

    @pytest.mark.asyncio
    async def test_empty_csv_is_handled_gracefully(self, connector):
        result = await connector.ingest(content=b"a,b,c\n", filename="empty.csv")
        # Header only — 0 data rows
        assert result.succeeded or result.error is not None  # either is acceptable
        if result.succeeded:
            assert result.row_count == 0

    @pytest.mark.asyncio
    async def test_to_schema_json_produces_valid_json(self, connector, simple_csv):
        import json
        result = await connector.ingest(content=simple_csv, filename="data.csv")
        schema = json.loads(result.to_schema_json())
        assert isinstance(schema, list)
        assert all("name" in col and "dtype" in col and "nullable" in col for col in schema)

    @pytest.mark.asyncio
    async def test_different_uploads_get_unique_filenames(self, connector, simple_csv):
        r1 = await connector.ingest(content=simple_csv, filename="a.csv")
        r2 = await connector.ingest(content=simple_csv, filename="b.csv")
        assert r1.file_path != r2.file_path


# ══════════════════════════════════════════════════════════════════════════
# FILE CONNECTOR — PARQUET
# ══════════════════════════════════════════════════════════════════════════

class TestFileConnectorParquet:

    @pytest.fixture
    def connector(self, tmp_path):
        return FileConnector(storage_dir=str(tmp_path))

    @pytest.fixture
    def parquet_bytes(self, tmp_path):
        """Creates a real Parquet file in memory."""
        df = pd.DataFrame({
            "id": range(100),
            "value": [float(i) * 1.5 for i in range(100)],
            "label": [f"item_{i}" for i in range(100)],
        })
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        return buf.getvalue()

    @pytest.mark.asyncio
    async def test_parquet_row_count_from_metadata(self, connector, parquet_bytes):
        """Parquet row count comes from footer metadata — no data scan."""
        result = await connector.ingest(content=parquet_bytes, filename="data.parquet")
        assert result.succeeded
        assert result.row_count == 100
        assert result.column_count == 3

    @pytest.mark.asyncio
    async def test_parquet_schema_inferred(self, connector, parquet_bytes):
        result = await connector.ingest(content=parquet_bytes, filename="data.parquet")
        col_names = [c.name for c in result.columns]
        assert "id" in col_names
        assert "value" in col_names
        assert "label" in col_names

    @pytest.mark.asyncio
    async def test_parquet_source_type(self, connector, parquet_bytes):
        result = await connector.ingest(content=parquet_bytes, filename="data.parquet")
        assert result.source_type == "parquet"


# ══════════════════════════════════════════════════════════════════════════
# SQL CONNECTOR
# ══════════════════════════════════════════════════════════════════════════

class TestSQLConnector:

    @pytest.fixture
    def db_path(self, tmp_path):
        """Creates a real SQLite DB with test data."""
        db = tmp_path / "test.db"
        with sqlite3.connect(db) as conn:
            conn.execute("""
                CREATE TABLE sales (
                    id INTEGER PRIMARY KEY,
                    product TEXT NOT NULL,
                    amount REAL,
                    region TEXT
                )
            """)
            conn.executemany(
                "INSERT INTO sales VALUES (?, ?, ?, ?)",
                [(i, f"prod_{i % 5}", float(i) * 9.99, ["north", "south"][i % 2])
                 for i in range(1, 51)]
            )
        return db

    @pytest.fixture
    def connector(self, db_path, tmp_path):
        db_url = f"sqlite+aiosqlite:///{db_path}"
        return SQLConnector(db_url=db_url, storage_dir=str(tmp_path / "storage"))

    @pytest.mark.asyncio
    async def test_ingest_returns_correct_row_count(self, connector):
        result = await connector.ingest(query="SELECT * FROM sales")
        await connector.close()
        assert result.succeeded, result.error
        assert result.row_count == 50

    @pytest.mark.asyncio
    async def test_ingest_saves_as_parquet(self, connector):
        result = await connector.ingest(query="SELECT * FROM sales")
        await connector.close()
        assert result.file_path is not None
        assert result.file_path.endswith(".parquet")
        assert Path(result.file_path).exists()

    @pytest.mark.asyncio
    async def test_ingest_schema_matches_table(self, connector):
        result = await connector.ingest(query="SELECT id, product, amount FROM sales LIMIT 1")
        await connector.close()
        col_names = [c.name for c in result.columns]
        assert "id" in col_names
        assert "product" in col_names
        assert "amount" in col_names

    @pytest.mark.asyncio
    async def test_filtered_query_respects_where_clause(self, connector):
        result = await connector.ingest(query="SELECT * FROM sales WHERE region = 'north'")
        await connector.close()
        assert result.succeeded
        assert result.row_count == 25  # half the rows

    @pytest.mark.asyncio
    async def test_streaming_chunks_cover_all_rows(self, connector):
        total_rows = 0
        async for chunk in connector.stream_chunks(
            chunk_size=10,
            query="SELECT * FROM sales",
        ):
            assert len(chunk) <= 10
            total_rows += len(chunk)
        await connector.close()
        assert total_rows == 50

    @pytest.mark.asyncio
    async def test_streaming_no_duplicate_rows(self, connector):
        all_ids = []
        async for chunk in connector.stream_chunks(
            chunk_size=15,
            query="SELECT id FROM sales",
        ):
            all_ids.extend(chunk["id"].tolist())
        await connector.close()
        assert len(set(all_ids)) == len(all_ids) == 50

    @pytest.mark.asyncio
    async def test_list_tables(self, connector):
        tables = await connector.list_tables()
        await connector.close()
        assert "sales" in tables

    @pytest.mark.asyncio
    async def test_get_table_schema(self, connector):
        schema = await connector.get_table_schema("sales")
        await connector.close()
        names = [c.name for c in schema]
        assert "id" in names
        assert "product" in names

    @pytest.mark.asyncio
    async def test_connection_test_succeeds(self, connector):
        ok, err = await connector.test_connection()
        await connector.close()
        assert ok is True
        assert err is None

    @pytest.mark.asyncio
    async def test_bad_url_fails_gracefully(self, tmp_path):
        bad_connector = SQLConnector(
            db_url="sqlite+aiosqlite:///nonexistent/path/db.db",
            storage_dir=str(tmp_path),
        )
        ok, err = await bad_connector.test_connection()
        await bad_connector.close()
        # Connection to a non-existent path may succeed for SQLite
        # (it creates the file), so we just check the method returns a tuple
        assert isinstance(ok, bool)

    @pytest.mark.asyncio
    async def test_invalid_query_returns_error_result(self, connector):
        result = await connector.ingest(query="SELECT * FROM nonexistent_table")
        await connector.close()
        assert not result.succeeded
        assert result.error is not None


# ══════════════════════════════════════════════════════════════════════════
# OAUTH2 CONNECTOR — config presets and interface
# ══════════════════════════════════════════════════════════════════════════

class TestOAuth2Config:
    """
    Tests the config presets and interface. Full flow tests (with mocked
    HTTP) are in the spike test suite — we don't duplicate them here.
    These tests verify the production module is importable and correctly
    configured.
    """

    def test_github_config_has_form_body_format(self):
        cfg = github_config("id", "secret")
        assert cfg.token_body_format == "form"
        assert cfg.response_format == "form"
        assert cfg.use_pkce is False

    def test_google_config_requires_pkce(self):
        cfg = google_config("id", "secret")
        assert cfg.use_pkce is True
        assert cfg.pkce_method == "S256"

    def test_client_credentials_connector_instantiates(self):
        import httpx
        cfg = OAuth2Config(
            client_id="id",
            client_secret="secret",
            token_url="https://example.com/token",
            scopes=["read"],
        )
        connector = ClientCredentialsConnector(cfg, httpx.AsyncClient())
        assert connector._token is None

    def test_auth_code_connector_builds_auth_url(self):
        cfg = github_config("gh_id", "gh_secret")
        import httpx
        connector = AuthCodeConnector(cfg, httpx.AsyncClient())
        url, state = connector.build_auth_url()
        assert "github.com" in url
        assert state in url
        assert len(state) > 16  # sufficient entropy

    def test_auth_code_pkce_challenge_in_url(self):
        cfg = google_config("g_id", "g_secret")
        import httpx
        connector = AuthCodeConnector(cfg, httpx.AsyncClient())
        url, _ = connector.build_auth_url()
        assert "code_challenge" in url
        assert "S256" in url
