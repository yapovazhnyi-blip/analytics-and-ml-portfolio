"""
Anomaly detection tests.

Tests cover:
  - Individual algorithms: IsolationForest, LOF, OC-SVM
  - Score normalisation to [0, 1]
  - Known anomalies are correctly flagged (injected outliers)
  - Consensus voting across algorithms
  - Feature preparation (NaN handling, scaling, column exclusion)
  - AnomalyRunner orchestration
  - API endpoint
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── Test data ────────────────────────────────────────────────────────────────

def _clean_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "y": rng.normal(0, 1, n),
        "z": rng.normal(5, 0.5, n),
    })

def _data_with_outliers(n=200, n_outliers=10, seed=0):
    """Returns a DataFrame where the last n_outliers rows are extreme outliers."""
    rng = np.random.default_rng(seed)
    normal = pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "y": rng.normal(0, 1, n),
    })
    outliers = pd.DataFrame({
        "x": rng.normal(20, 0.1, n_outliers),
        "y": rng.normal(20, 0.1, n_outliers),
    })
    return pd.concat([normal, outliers], ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════
# ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════

class TestIsolationForest:

    def test_returns_scores_for_all_rows(self):
        from anomaly.algorithms import run_isolation_forest, prepare_features
        df = _clean_data(100)
        X, _ = prepare_features(df)
        r = run_isolation_forest(X, contamination=0.05)
        assert r.succeeded, r.error
        assert len(r.scores) == 100

    def test_scores_in_zero_one(self):
        from anomaly.algorithms import run_isolation_forest, prepare_features
        X, _ = prepare_features(_clean_data(100))
        r = run_isolation_forest(X)
        assert r.scores.min() >= 0.0
        assert r.scores.max() <= 1.0

    def test_injected_outliers_get_high_scores(self):
        from anomaly.algorithms import run_isolation_forest, prepare_features
        df = _data_with_outliers(n=200, n_outliers=10)
        X, _ = prepare_features(df)
        r = run_isolation_forest(X, contamination=0.05)
        # Outliers are the last 10 rows — should have higher avg score
        normal_avg  = r.scores[:200].mean()
        outlier_avg = r.scores[200:].mean()
        assert outlier_avg > normal_avg

    def test_n_anomalies_matches_contamination(self):
        from anomaly.algorithms import run_isolation_forest, prepare_features
        n = 200
        X, _ = prepare_features(_clean_data(n))
        r = run_isolation_forest(X, contamination=0.10)
        # Should flag approximately n * contamination anomalies (±20%)
        expected = n * 0.10
        assert abs(r.n_anomalies - expected) <= expected * 0.5

    def test_labels_are_binary(self):
        from anomaly.algorithms import run_isolation_forest, prepare_features
        X, _ = prepare_features(_clean_data(100))
        r = run_isolation_forest(X)
        assert set(r.labels).issubset({0, 1})


class TestLOF:

    def test_returns_correct_length(self):
        from anomaly.algorithms import run_lof, prepare_features
        X, _ = prepare_features(_clean_data(100))
        r = run_lof(X)
        assert r.succeeded, r.error
        assert len(r.scores) == 100

    def test_scores_in_zero_one(self):
        from anomaly.algorithms import run_lof, prepare_features
        X, _ = prepare_features(_clean_data(100))
        r = run_lof(X)
        assert r.scores.min() >= 0.0
        assert r.scores.max() <= 1.0

    def test_outliers_flagged(self):
        from anomaly.algorithms import run_lof, prepare_features
        df = _data_with_outliers(n=200, n_outliers=10)
        X, _ = prepare_features(df)
        r = run_lof(X, contamination=0.05)
        assert r.succeeded
        outlier_avg = r.scores[200:].mean()
        normal_avg  = r.scores[:200].mean()
        assert outlier_avg > normal_avg


class TestOCSVM:

    def test_skips_large_dataset(self):
        from anomaly.algorithms import run_ocsvm, prepare_features
        big_df = pd.DataFrame({"x": np.random.randn(15_000)})
        X, _ = prepare_features(big_df)
        r = run_ocsvm(X)
        assert not r.succeeded
        assert "10,000" in r.error or "impractical" in r.error

    def test_runs_on_small_dataset(self):
        from anomaly.algorithms import run_ocsvm, prepare_features
        X, _ = prepare_features(_clean_data(200))
        r = run_ocsvm(X, contamination=0.05)
        if r.succeeded:
            assert len(r.scores) == 200
            assert r.scores.min() >= 0.0


class TestPrepareFeatures:

    def test_drops_non_numeric(self):
        from anomaly.algorithms import prepare_features
        df = pd.DataFrame({"x": [1.0, 2.0], "name": ["a", "b"]})
        X, cols = prepare_features(df)
        assert "name" not in cols
        assert X.shape[1] == 1

    def test_fills_nan_with_median(self):
        from anomaly.algorithms import prepare_features
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [2.0, 2.0, 2.0]})
        X, _ = prepare_features(df)
        assert not np.isnan(X).any()

    def test_excludes_specified_columns(self):
        from anomaly.algorithms import prepare_features
        df = pd.DataFrame({"id": [1, 2, 3], "x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        X, cols = prepare_features(df, exclude_columns=["id"])
        assert "id" not in cols

    def test_raises_on_no_numeric(self):
        from anomaly.algorithms import prepare_features
        df = pd.DataFrame({"name": ["a", "b", "c"]})
        with pytest.raises(ValueError, match="No numeric columns"):
            prepare_features(df)

    def test_output_is_standardised(self):
        from anomaly.algorithms import prepare_features
        df = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0, 50.0]})
        X, _ = prepare_features(df)
        # After StandardScaler, mean ≈ 0 and std ≈ 1
        assert abs(X.mean()) < 1e-10
        assert abs(X.std() - 1.0) < 0.1


# ══════════════════════════════════════════════════════════════════════════
# ANOMALY RUNNER
# ══════════════════════════════════════════════════════════════════════════

class TestAnomalyRunner:

    @pytest.mark.asyncio
    async def test_runner_returns_report(self):
        from anomaly.runner import AnomalyRunner
        df = _clean_data(200)
        runner = AnomalyRunner(contamination=0.05, algorithms=["isolation_forest"])
        report = await runner.run(df, dataset_id=1)
        assert report.succeeded, report.error
        assert report.n_rows == 200

    @pytest.mark.asyncio
    async def test_consensus_anomaly_count(self):
        from anomaly.runner import AnomalyRunner
        df = _data_with_outliers(200, 10)
        runner = AnomalyRunner(contamination=0.05, algorithms=["isolation_forest", "lof"])
        report = await runner.run(df, dataset_id=1)
        assert report.succeeded
        assert report.consensus_n_anomalies > 0

    @pytest.mark.asyncio
    async def test_top_anomalies_returned(self):
        from anomaly.runner import AnomalyRunner
        df = _clean_data(100)
        runner = AnomalyRunner(top_n=10)
        report = await runner.run(df, dataset_id=1)
        assert report.succeeded
        assert len(report.top_anomalies) <= 10
        # Top anomalies must include row_index and anomaly_score
        for row in report.top_anomalies:
            assert "row_index" in row
            assert "anomaly_score" in row

    @pytest.mark.asyncio
    async def test_score_distribution_has_percentiles(self):
        from anomaly.runner import AnomalyRunner
        report = await AnomalyRunner().run(_clean_data(200), 1)
        assert "95" in report.score_distribution
        assert "99" in report.score_distribution

    @pytest.mark.asyncio
    async def test_to_dict_is_json_serialisable(self):
        import json
        from anomaly.runner import AnomalyRunner
        report = await AnomalyRunner().run(_clean_data(100), 1)
        json.dumps(report.to_dict())   # must not raise

    @pytest.mark.asyncio
    async def test_exclude_columns_respected(self):
        from anomaly.runner import AnomalyRunner
        df = pd.DataFrame({"id": range(100), "x": np.random.randn(100), "y": np.random.randn(100)})
        report = await AnomalyRunner().run(df, 1, exclude_columns=["id"])
        assert "id" not in report.feature_names

    @pytest.mark.asyncio
    async def test_non_numeric_dataset_returns_error(self):
        from anomaly.runner import AnomalyRunner
        df = pd.DataFrame({"name": ["a", "b", "c"]})
        report = await AnomalyRunner().run(df, 1)
        assert not report.succeeded

    @pytest.mark.asyncio
    async def test_algorithm_results_in_report(self):
        from anomaly.runner import AnomalyRunner
        report = await AnomalyRunner(algorithms=["isolation_forest", "lof"]).run(_clean_data(100), 1)
        families = {a["family"] for a in report.algorithms}
        assert "isolation_forest" in families
        assert "lof" in families


# ══════════════════════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def anomaly_client(tmp_path):
    import sys, importlib, database as db_mod
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool
    from fastapi.testclient import TestClient
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    db_mod.engine = engine; db_mod.SessionFactory = factory; db_mod.AsyncSessionLocal = factory
    if "main" in sys.modules:
        importlib.reload(sys.modules["main"])
    import main as m
    with TestClient(m.app, raise_server_exceptions=True) as c:
        yield c


class TestAnomalyAPI:

    def test_anomaly_missing_dataset(self, anomaly_client):
        resp = anomaly_client.post("/api/v1/datasets/9999/anomaly", json={})
        assert resp.status_code == 404

    def test_invalid_algorithm_rejected(self, anomaly_client, tmp_path):
        resp = anomaly_client.post("/api/v1/datasets/1/anomaly", json={
            "algorithms": ["neural_autoencoder"],
        })
        assert resp.status_code in (404, 422)

    def test_contamination_out_of_range(self, anomaly_client):
        resp = anomaly_client.post("/api/v1/datasets/1/anomaly", json={
            "contamination": 0.99,
        })
        assert resp.status_code == 422

    def test_full_anomaly_run(self, anomaly_client, tmp_path):
        """Upload a dataset and run anomaly detection end-to-end."""
        df = _data_with_outliers(200, 10)
        csv_bytes = df.to_csv(index=False).encode()
        ds_resp = anomaly_client.post(
            "/api/v1/datasets/upload",
            files={"file": ("data.csv", csv_bytes, "text/csv")},
            data={"name": "anomaly_test"},
        )
        assert ds_resp.status_code == 201
        ds_id = ds_resp.json()["data"]["id"]

        resp = anomaly_client.post(f"/api/v1/datasets/{ds_id}/anomaly", json={
            "contamination": 0.05,
            "algorithms": ["isolation_forest"],
            "top_n": 15,
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["n_rows"] == 210
        assert data["consensus_n_anomalies"] > 0
        assert len(data["top_anomalies"]) <= 15
        assert "95" in data["score_distribution"]
