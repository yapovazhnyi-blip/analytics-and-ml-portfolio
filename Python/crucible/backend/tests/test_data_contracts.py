"""
Data contracts tests.

Tests cover:
  - Contract generation for numeric, categorical, boolean, and nullable columns
  - Validation: clean data passes, violations are caught with correct messages
  - Edge cases: empty dataframe, all-null column, high-cardinality skips,
    extra columns in incoming data, missing columns
  - ContractViolation structure and serialisation
  - API endpoints: generate, get, validate, delete
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def reference_df():
    """Clean reference DataFrame with numeric, categorical, and nullable columns."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "age":     np.random.randint(18, 70, n).astype(float),
        "income":  np.random.uniform(20_000, 120_000, n),
        "city":    np.random.choice(["London", "Paris", "Berlin"], n),
        "churned": np.random.randint(0, 2, n),
        "notes":   [None if i % 10 == 0 else f"note_{i}" for i in range(n)],
    })


@pytest.fixture
def clean_contract(reference_df):
    from data_contracts.schema import generate_contract
    return generate_contract(reference_df, dataset_id=1, dataset_name="ref")


# ══════════════════════════════════════════════════════════════════════════
# CONTRACT GENERATION
# ══════════════════════════════════════════════════════════════════════════

class TestGenerate:

    def test_returns_data_contract(self, reference_df):
        from data_contracts.schema import generate_contract, DataContract
        c = generate_contract(reference_df, 1, "ref")
        assert isinstance(c, DataContract)

    def test_correct_column_count(self, reference_df):
        from data_contracts.schema import generate_contract
        c = generate_contract(reference_df, 1, "ref")
        assert len(c.columns) == len(reference_df.columns)

    def test_numeric_column_has_range(self, reference_df):
        from data_contracts.schema import generate_contract
        c = generate_contract(reference_df, 1, "ref", tolerance=0.10)
        age = next(col for col in c.columns if col.name == "age")
        assert age.dtype_family == "numeric"
        assert age.min_val is not None
        assert age.max_val is not None
        # With 10% tolerance, min should be below observed min
        assert age.min_val < reference_df["age"].min()
        assert age.max_val > reference_df["age"].max()

    def test_categorical_column_has_allowed_values(self, reference_df):
        from data_contracts.schema import generate_contract
        c = generate_contract(reference_df, 1, "ref")
        city = next(col for col in c.columns if col.name == "city")
        assert city.dtype_family == "categorical"
        assert city.allowed_values is not None
        assert set(city.allowed_values) == {"London", "Paris", "Berlin"}

    def test_high_cardinality_skips_allowed_values(self):
        from data_contracts.schema import generate_contract
        df = pd.DataFrame({"id": [f"user_{i}" for i in range(200)]})
        c = generate_contract(df, 1, "ref", max_categories=50)
        id_col = c.columns[0]
        assert id_col.allowed_values is None   # 200 unique values → skipped

    def test_nullable_column_has_nullable_true(self, reference_df):
        from data_contracts.schema import generate_contract
        c = generate_contract(reference_df, 1, "ref")
        notes = next(col for col in c.columns if col.name == "notes")
        assert notes.nullable is True
        assert notes.max_null_rate > 0

    def test_nonnull_column_has_nullable_false(self, reference_df):
        from data_contracts.schema import generate_contract
        c = generate_contract(reference_df, 1, "ref")
        age = next(col for col in c.columns if col.name == "age")
        assert age.nullable is False

    def test_target_column_recorded(self, reference_df):
        from data_contracts.schema import generate_contract
        c = generate_contract(reference_df, 1, "ref", target_column="churned")
        assert c.target_column == "churned"

    def test_serialisation_roundtrip(self, reference_df):
        from data_contracts.schema import generate_contract, DataContract
        c = generate_contract(reference_df, 1, "ref")
        restored = DataContract.from_json(c.to_json())
        assert len(restored.columns) == len(c.columns)
        for a, b in zip(c.columns, restored.columns):
            assert a.name == b.name
            assert a.dtype_family == b.dtype_family

    def test_empty_dataframe(self):
        from data_contracts.schema import generate_contract
        df = pd.DataFrame({"x": pd.Series([], dtype=float)})
        c = generate_contract(df, 1, "empty")
        assert len(c.columns) == 1

    def test_tolerance_zero_produces_tight_range(self):
        from data_contracts.schema import generate_contract
        df = pd.DataFrame({"v": [10.0, 20.0, 30.0]})
        c = generate_contract(df, 1, "tight", tolerance=0.0)
        v = c.columns[0]
        # With tolerance=0, buffer is at least 1e-6 (numeric stability)
        assert v.min_val <= 10.0
        assert v.max_val >= 30.0


# ══════════════════════════════════════════════════════════════════════════
# VALIDATION — CLEAN DATA PASSES
# ══════════════════════════════════════════════════════════════════════════

class TestValidationPasses:

    def test_clean_data_passes(self, reference_df, clean_contract):
        from data_contracts.schema import validate_dataframe
        result = validate_dataframe(reference_df, clean_contract, dataset_id=1)
        assert result.passed
        assert result.n_errors == 0

    def test_all_rows_checked(self, reference_df, clean_contract):
        from data_contracts.schema import validate_dataframe
        result = validate_dataframe(reference_df, clean_contract, dataset_id=1)
        assert result.n_rows == len(reference_df)

    def test_extra_columns_allowed(self, reference_df, clean_contract):
        """Incoming data may have MORE columns than the contract defines."""
        from data_contracts.schema import validate_dataframe
        df_extra = reference_df.copy()
        df_extra["new_feature"] = 42
        result = validate_dataframe(df_extra, clean_contract, dataset_id=1)
        assert result.passed

    def test_self_validation_passes(self, reference_df, clean_contract):
        """Validating against itself must always pass."""
        from data_contracts.schema import validate_dataframe
        result = validate_dataframe(reference_df, clean_contract, dataset_id=1)
        assert result.passed
        assert result.n_violations == 0


# ══════════════════════════════════════════════════════════════════════════
# VALIDATION — VIOLATIONS CAUGHT
# ══════════════════════════════════════════════════════════════════════════

class TestValidationViolations:

    def test_value_above_max_caught(self, reference_df, clean_contract):
        from data_contracts.schema import validate_dataframe
        df_bad = reference_df.copy()
        df_bad.loc[0, "income"] = 999_999_999    # way above max
        result = validate_dataframe(df_bad, clean_contract, dataset_id=2)
        assert not result.passed
        assert any(v.column == "income" and v.check == "max_value" for v in result.violations)

    def test_value_below_min_caught(self, reference_df, clean_contract):
        from data_contracts.schema import validate_dataframe
        df_bad = reference_df.copy()
        df_bad.loc[0, "age"] = -100   # below min
        result = validate_dataframe(df_bad, clean_contract, dataset_id=2)
        assert not result.passed
        assert any(v.column == "age" and v.check == "min_value" for v in result.violations)

    def test_unknown_category_caught(self, reference_df, clean_contract):
        from data_contracts.schema import validate_dataframe
        df_bad = reference_df.copy()
        df_bad.loc[0, "city"] = "Tokyo"    # not in [London, Paris, Berlin]
        result = validate_dataframe(df_bad, clean_contract, dataset_id=2)
        assert not result.passed
        city_violations = [v for v in result.violations
                           if v.column == "city" and v.check == "allowed_values"]
        assert len(city_violations) == 1
        assert "Tokyo" in str(city_violations[0].examples)

    def test_missing_column_caught(self, reference_df, clean_contract):
        from data_contracts.schema import validate_dataframe
        df_missing = reference_df.drop(columns=["age"])
        result = validate_dataframe(df_missing, clean_contract, dataset_id=2)
        assert not result.passed
        assert any(v.column == "age" and v.check == "column_exists"
                   for v in result.violations)

    def test_null_rate_too_high_is_warning(self, reference_df, clean_contract):
        """Excess nulls are a warning, not an error."""
        from data_contracts.schema import validate_dataframe
        df_bad = reference_df.copy()
        # Insert many nulls into notes (which had ~10% observed)
        df_bad["notes"] = None   # 100% null
        result = validate_dataframe(df_bad, clean_contract, dataset_id=2)
        null_violations = [v for v in result.violations
                           if v.column == "notes" and v.check == "max_null_rate"]
        assert len(null_violations) == 1
        assert null_violations[0].severity == "warning"
        # Warnings alone should not fail validation
        errors = [v for v in result.violations if v.severity == "error"]
        assert result.passed == (len(errors) == 0)

    def test_violation_has_example_values(self, reference_df, clean_contract):
        from data_contracts.schema import validate_dataframe
        df_bad = reference_df.copy()
        df_bad.loc[:5, "income"] = 1_000_000
        result = validate_dataframe(df_bad, clean_contract, dataset_id=2)
        income_v = next(v for v in result.violations if v.column == "income")
        assert len(income_v.examples) > 0

    def test_result_serialisable(self, reference_df, clean_contract):
        from data_contracts.schema import validate_dataframe
        result = validate_dataframe(reference_df, clean_contract, dataset_id=1)
        json.dumps(result.to_dict())


# ══════════════════════════════════════════════════════════════════════════
# EDGE CASES
# ══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_all_null_column_in_incoming(self, reference_df):
        from data_contracts.schema import generate_contract, validate_dataframe
        c = generate_contract(reference_df, 1, "ref")
        df_bad = reference_df.copy()
        df_bad["age"] = None
        result = validate_dataframe(df_bad, c, dataset_id=2)
        # Age is non-nullable in contract → should catch it
        assert not result.passed

    def test_multiple_violations_all_reported(self, reference_df, clean_contract):
        from data_contracts.schema import validate_dataframe
        df_bad = reference_df.copy()
        df_bad.loc[0, "age"]    = -100
        df_bad.loc[1, "income"] = 9_999_999
        df_bad.loc[2, "city"]   = "Tokyo"
        result = validate_dataframe(df_bad, clean_contract, dataset_id=2)
        assert result.n_violations >= 3

    def test_empty_incoming_dataframe(self, clean_contract):
        from data_contracts.schema import validate_dataframe
        empty_df = pd.DataFrame({"age": [], "income": [], "city": [],
                                  "churned": [], "notes": []})
        result = validate_dataframe(empty_df, clean_contract, dataset_id=2)
        # Zero rows — nothing to violate
        assert result.n_rows == 0


# ══════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def dc_client():
    import sys, importlib, database as db_mod
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool
    from fastapi.testclient import TestClient

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    db_mod.engine = engine
    db_mod.SessionFactory = factory
    db_mod.AsyncSessionLocal = factory
    if "main" in sys.modules:
        importlib.reload(sys.modules["main"])
    import main as m
    with TestClient(m.app, raise_server_exceptions=True) as c:
        yield c


def _upload_csv(client, df, name="test"):
    csv_bytes = df.to_csv(index=False).encode()
    resp = client.post(
        "/api/v1/datasets/upload",
        files={"file": (f"{name}.csv", csv_bytes, "text/csv")},
        data={"name": name},
    )
    assert resp.status_code == 201
    return resp.json()["data"]["id"]


class TestDataContractsAPI:

    def _ref_df(self):
        np.random.seed(0)
        n = 100
        return pd.DataFrame({
            "age":   np.random.uniform(20, 60, n),
            "score": np.random.uniform(0, 1, n),
            "tier":  np.random.choice(["A","B","C"], n),
        })

    def test_generate_contract(self, dc_client):
        ds_id = _upload_csv(dc_client, self._ref_df())
        resp = dc_client.post(f"/api/v1/datasets/{ds_id}/contracts/generate", json={})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "columns" in data
        assert len(data["columns"]) == 3

    def test_get_contract_after_generate(self, dc_client):
        ds_id = _upload_csv(dc_client, self._ref_df())
        dc_client.post(f"/api/v1/datasets/{ds_id}/contracts/generate", json={})
        resp = dc_client.get(f"/api/v1/datasets/{ds_id}/contracts")
        assert resp.status_code == 200
        assert "columns" in resp.json()["data"]

    def test_get_contract_without_generate_returns_404(self, dc_client):
        ds_id = _upload_csv(dc_client, self._ref_df())
        resp = dc_client.get(f"/api/v1/datasets/{ds_id}/contracts")
        assert resp.status_code == 404

    def test_validate_clean_data_passes(self, dc_client):
        ref_df = self._ref_df()
        ds_id = _upload_csv(dc_client, ref_df, "ref")
        dc_client.post(f"/api/v1/datasets/{ds_id}/contracts/generate", json={})
        resp = dc_client.post(f"/api/v1/datasets/{ds_id}/contracts/validate",
                               json={"dataset_id": ds_id})
        assert resp.status_code == 200
        assert resp.json()["data"]["passed"] is True

    def test_validate_bad_data_fails(self, dc_client):
        ref_df = self._ref_df()
        ds_id  = _upload_csv(dc_client, ref_df, "ref")
        dc_client.post(f"/api/v1/datasets/{ds_id}/contracts/generate", json={})

        # Upload dirty version
        bad_df = ref_df.copy()
        bad_df.loc[0:5, "age"] = 9999   # way above max
        bad_df.loc[0:2, "tier"] = "ZZZZ"  # unknown category
        bad_id = _upload_csv(dc_client, bad_df, "bad")

        resp = dc_client.post(f"/api/v1/datasets/{ds_id}/contracts/validate",
                               json={"dataset_id": bad_id})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["passed"] is False
        assert data["n_violations"] > 0

    def test_delete_contract(self, dc_client):
        ds_id = _upload_csv(dc_client, self._ref_df())
        dc_client.post(f"/api/v1/datasets/{ds_id}/contracts/generate", json={})
        del_resp = dc_client.delete(f"/api/v1/datasets/{ds_id}/contracts")
        assert del_resp.status_code == 204
        get_resp = dc_client.get(f"/api/v1/datasets/{ds_id}/contracts")
        assert get_resp.status_code == 404

    def test_generate_nonexistent_dataset(self, dc_client):
        resp = dc_client.post("/api/v1/datasets/9999/contracts/generate", json={})
        assert resp.status_code == 404

    def test_validate_without_contract_returns_422(self, dc_client):
        ds_id = _upload_csv(dc_client, self._ref_df())
        resp = dc_client.post(f"/api/v1/datasets/{ds_id}/contracts/validate",
                               json={"dataset_id": ds_id})
        assert resp.status_code == 422

    def test_contract_includes_tolerance(self, dc_client):
        ds_id = _upload_csv(dc_client, self._ref_df())
        resp = dc_client.post(f"/api/v1/datasets/{ds_id}/contracts/generate",
                               json={"tolerance": 0.20})
        assert resp.status_code == 200
        assert resp.json()["data"]["tolerance"] == 0.20
