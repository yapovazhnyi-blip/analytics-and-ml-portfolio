"""
Time series forecasting tests.

Tests are structured in layers:
  1. Config validation
  2. Metric functions (MAPE, RMSE, MAE)
  3. Frequency inference
  4. Temporal cross-validation splits
  5. Individual families (ARIMA, Exponential Smoothing)
  6. Full TimeSeriesRunner end-to-end
  7. API endpoints

All tests use synthetic data — no real datasets needed.
Families are tested only when their dependencies are available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


# ── Synthetic data fixtures ───────────────────────────────────────────────────

@pytest.fixture
def monthly_series():
    """
    Synthetic monthly series: trend + seasonality + noise.
    60 observations = 5 years of monthly data.
    """
    rng = np.random.default_rng(42)
    n = 60
    t = np.arange(n)
    trend    = t * 2.0
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    noise    = rng.normal(0, 2, n)
    values   = 100 + trend + seasonal + noise

    dates = pd.date_range(start="2019-01-01", periods=n, freq="MS")
    return pd.Series(values, index=dates)


@pytest.fixture
def daily_df(monthly_series):
    """DataFrame with date + target columns (for runner tests)."""
    df = pd.DataFrame({
        "date":   monthly_series.index.strftime("%Y-%m-%d"),
        "sales":  monthly_series.values,
        "region": "EU",   # extra column that should be ignored
    })
    return df


# ══════════════════════════════════════════════════════════════════════════
# CONFIG VALIDATION
# ══════════════════════════════════════════════════════════════════════════

class TestTimeSeriesConfig:

    def test_valid_config(self):
        from training.time_series.config import TimeSeriesConfig
        cfg = TimeSeriesConfig(date_column="date", target_column="sales")
        assert cfg.validate() == []

    def test_missing_date_column(self):
        from training.time_series.config import TimeSeriesConfig
        cfg = TimeSeriesConfig(date_column="", target_column="sales")
        errors = cfg.validate()
        assert any("date" in e for e in errors)

    def test_missing_target_column(self):
        from training.time_series.config import TimeSeriesConfig
        cfg = TimeSeriesConfig(date_column="date", target_column="")
        errors = cfg.validate()
        assert any("target" in e for e in errors)

    def test_invalid_horizon(self):
        from training.time_series.config import TimeSeriesConfig
        cfg = TimeSeriesConfig(date_column="date", target_column="sales", horizon=0)
        assert any("horizon" in e for e in cfg.validate())

    def test_defaults(self):
        from training.time_series.config import TimeSeriesConfig
        cfg = TimeSeriesConfig(date_column="date", target_column="sales")
        assert cfg.horizon == 12
        assert cfg.frequency == "auto"
        assert cfg.n_trials == 20


# ══════════════════════════════════════════════════════════════════════════
# METRIC FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

class TestMetrics:

    def test_mape_perfect_forecast(self):
        from training.time_series.runner import mape
        actual = np.array([100., 200., 300.])
        assert mape(actual, actual) == 0.0

    def test_mape_50_pct_error(self):
        from training.time_series.runner import mape
        actual    = np.array([100., 100.])
        predicted = np.array([50.,  150.])
        # |100-50|/100 = 0.5, |100-150|/100 = 0.5 → mean = 0.5 → 50%
        assert abs(mape(actual, predicted) - 50.0) < 0.01

    def test_mape_returns_999_for_zero_actuals(self):
        from training.time_series.runner import mape
        assert mape(np.array([0., 0.]), np.array([1., 2.])) == 999.0

    def test_rmse_perfect(self):
        from training.time_series.runner import rmse
        y = np.array([1., 2., 3.])
        assert rmse(y, y) == 0.0

    def test_rmse_known_value(self):
        from training.time_series.runner import rmse
        # errors = [1, 1] → RMSE = sqrt(1) = 1
        assert abs(rmse(np.array([0., 0.]), np.array([1., -1.])) - 1.0) < 1e-6

    def test_mae_perfect(self):
        from training.time_series.runner import mae
        y = np.array([5., 10., 15.])
        assert mae(y, y) == 0.0

    def test_mae_known_value(self):
        from training.time_series.runner import mae
        assert abs(mae(np.array([0., 0.]), np.array([2., 4.])) - 3.0) < 1e-6


# ══════════════════════════════════════════════════════════════════════════
# FREQUENCY INFERENCE
# ══════════════════════════════════════════════════════════════════════════

class TestFrequencyInference:

    def test_daily_series(self):
        from training.time_series.runner import infer_frequency
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        assert infer_frequency(pd.Series(dates)) == "D"

    def test_weekly_series(self):
        from training.time_series.runner import infer_frequency
        dates = pd.date_range("2023-01-01", periods=20, freq="W")
        assert infer_frequency(pd.Series(dates)) == "W"

    def test_monthly_series(self):
        from training.time_series.runner import infer_frequency
        dates = pd.date_range("2023-01-01", periods=24, freq="MS")
        assert infer_frequency(pd.Series(dates)) == "MS"

    def test_hourly_series(self):
        from training.time_series.runner import infer_frequency
        dates = pd.date_range("2023-01-01", periods=48, freq="h")
        assert infer_frequency(pd.Series(dates)) == "h"


# ══════════════════════════════════════════════════════════════════════════
# TEMPORAL CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════

class TestTemporalCV:

    def test_returns_correct_number_of_splits(self):
        from training.time_series.runner import temporal_cv_splits
        splits = temporal_cv_splits(n=100, n_splits=3, horizon=10)
        assert len(splits) == 3

    def test_train_always_precedes_val(self):
        from training.time_series.runner import temporal_cv_splits
        for train_idx, val_idx in temporal_cv_splits(n=100, n_splits=3, horizon=10):
            assert train_idx.max() < val_idx.min()

    def test_val_size_matches_horizon(self):
        from training.time_series.runner import temporal_cv_splits
        horizon = 12
        for _, val_idx in temporal_cv_splits(n=120, n_splits=3, horizon=horizon):
            assert len(val_idx) <= horizon

    def test_training_sets_grow_across_splits(self):
        from training.time_series.runner import temporal_cv_splits
        splits = temporal_cv_splits(n=100, n_splits=3, horizon=10)
        train_sizes = [len(t) for t, _ in splits]
        assert train_sizes == sorted(train_sizes)   # strictly increasing

    def test_no_future_leakage(self):
        """Train indices must always be strictly less than validation indices."""
        from training.time_series.runner import temporal_cv_splits
        for train_idx, val_idx in temporal_cv_splits(n=200, n_splits=5, horizon=20):
            overlap = set(train_idx) & set(val_idx)
            assert len(overlap) == 0

    def test_short_series_returns_single_split(self):
        from training.time_series.runner import temporal_cv_splits
        # Too short for 3 splits
        splits = temporal_cv_splits(n=20, n_splits=3, horizon=5)
        assert len(splits) >= 1


# ══════════════════════════════════════════════════════════════════════════
# ARIMA FAMILY
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not __import__("training.time_series.families", fromlist=["ARIMA_AVAILABLE"]).ARIMA_AVAILABLE,
    reason="pmdarima not installed"
)
class TestARIMAFamily:

    @pytest.fixture
    def trial(self):
        """Fixed-parameter replay trial."""
        from training.time_series.runner import _ReplayTrial
        return _ReplayTrial({"arima_max_p": 2, "arima_max_q": 2,
                             "arima_seasonal": False, "arima_d": 1})

    def test_arima_fit_returns_model(self, trial, monthly_series):
        from training.time_series.families import _arima_family
        family = _arima_family(trial)
        fitted = family["fit"](monthly_series)
        assert fitted is not None

    def test_arima_predict_correct_length(self, trial, monthly_series):
        from training.time_series.families import _arima_family
        family = _arima_family(trial)
        fitted = family["fit"](monthly_series)
        preds = family["predict"](fitted, 12)
        assert len(preds) == 12

    def test_arima_forecast_finite(self, trial, monthly_series):
        from training.time_series.families import _arima_family
        family = _arima_family(trial)
        fitted = family["fit"](monthly_series)
        preds = family["predict"](fitted, 6)
        assert np.all(np.isfinite(preds))


# ══════════════════════════════════════════════════════════════════════════
# EXPONENTIAL SMOOTHING FAMILY
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not __import__("training.time_series.families", fromlist=["EXP_SMOOTHING_AVAILABLE"]).EXP_SMOOTHING_AVAILABLE,
    reason="statsmodels not installed"
)
class TestExpSmoothingFamily:

    @pytest.fixture
    def trial(self):
        from training.time_series.runner import _ReplayTrial
        return _ReplayTrial({"ets_trend": "add", "ets_seasonal": "add", "ets_damped": False})

    def test_ets_fit_returns_model(self, trial, monthly_series):
        from training.time_series.families import _exp_smoothing_family
        family = _exp_smoothing_family(trial)
        fitted = family["fit"](monthly_series)
        assert fitted is not None

    def test_ets_predict_correct_length(self, trial, monthly_series):
        from training.time_series.families import _exp_smoothing_family
        family = _exp_smoothing_family(trial)
        fitted = family["fit"](monthly_series)
        preds = family["predict"](fitted, 12)
        assert len(preds) == 12

    def test_ets_forecast_finite(self, trial, monthly_series):
        from training.time_series.families import _exp_smoothing_family
        family = _exp_smoothing_family(trial)
        fitted = family["fit"](monthly_series)
        preds = family["predict"](fitted, 6)
        assert np.all(np.isfinite(preds))

    def test_ets_no_trend_variant(self, monthly_series):
        from training.time_series.families import _exp_smoothing_family
        from training.time_series.runner import _ReplayTrial
        trial = _ReplayTrial({"ets_trend": None, "ets_seasonal": None, "ets_damped": False})
        family = _exp_smoothing_family(trial)
        fitted = family["fit"](monthly_series)
        preds = family["predict"](fitted, 6)
        assert len(preds) == 6


# ══════════════════════════════════════════════════════════════════════════
# FULL RUNNER
# ══════════════════════════════════════════════════════════════════════════

class TestTimeSeriesRunner:

    @pytest.mark.asyncio
    async def test_runner_succeeds_on_monthly_data(self, daily_df, tmp_path):
        from training.time_series.config import TimeSeriesConfig
        from training.time_series.runner import TimeSeriesRunner

        cfg = TimeSeriesConfig(
            date_column="date",
            target_column="sales",
            horizon=6,
            n_trials=4,
            n_cv_splits=2,
        )
        runner = TimeSeriesRunner(cfg, "test-fc-001")
        result = await runner.run(daily_df, str(tmp_path))

        assert result.succeeded, result.error
        assert result.best_family in ("arima", "exp_smoothing", "prophet", "lstm")
        assert result.cv_mape >= 0

    @pytest.mark.asyncio
    async def test_runner_forecast_has_correct_length(self, daily_df, tmp_path):
        from training.time_series.config import TimeSeriesConfig
        from training.time_series.runner import TimeSeriesRunner

        horizon = 8
        cfg = TimeSeriesConfig(
            date_column="date", target_column="sales",
            horizon=horizon, n_trials=3, n_cv_splits=2,
        )
        runner = TimeSeriesRunner(cfg, "test-fc-002")
        result = await runner.run(daily_df, str(tmp_path))

        assert result.succeeded
        assert len(result.forecast) == horizon

    @pytest.mark.asyncio
    async def test_runner_forecast_has_required_columns(self, daily_df, tmp_path):
        from training.time_series.config import TimeSeriesConfig
        from training.time_series.runner import TimeSeriesRunner

        cfg = TimeSeriesConfig(
            date_column="date", target_column="sales",
            horizon=6, n_trials=3, n_cv_splits=2,
        )
        runner = TimeSeriesRunner(cfg, "test-fc-003")
        result = await runner.run(daily_df, str(tmp_path))

        assert result.succeeded
        cols = result.forecast.columns.tolist()
        assert "date" in cols
        assert "predicted" in cols
        assert "lower" in cols
        assert "upper" in cols

    @pytest.mark.asyncio
    async def test_runner_fails_on_missing_date_column(self, daily_df, tmp_path):
        from training.time_series.config import TimeSeriesConfig
        from training.time_series.runner import TimeSeriesRunner

        cfg = TimeSeriesConfig(
            date_column="nonexistent", target_column="sales",
            horizon=6, n_trials=2, n_cv_splits=2,
        )
        runner = TimeSeriesRunner(cfg, "test-fc-004")
        result = await runner.run(daily_df, str(tmp_path))

        assert not result.succeeded
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_runner_to_dict_serialisable(self, daily_df, tmp_path):
        import json as json_mod
        from training.time_series.config import TimeSeriesConfig
        from training.time_series.runner import TimeSeriesRunner

        cfg = TimeSeriesConfig(
            date_column="date", target_column="sales",
            horizon=6, n_trials=3, n_cv_splits=2,
        )
        runner = TimeSeriesRunner(cfg, "test-fc-005")
        result = await runner.run(daily_df, str(tmp_path))

        assert result.succeeded
        d = result.to_dict()
        # Should be JSON-serialisable
        json_mod.dumps(d)

    @pytest.mark.asyncio
    async def test_family_filter_respected(self, daily_df, tmp_path):
        from training.time_series.config import TimeSeriesConfig
        from training.time_series.runner import TimeSeriesRunner
        from training.time_series.families import EXP_SMOOTHING_AVAILABLE

        if not EXP_SMOOTHING_AVAILABLE:
            pytest.skip("statsmodels not available")

        cfg = TimeSeriesConfig(
            date_column="date", target_column="sales",
            horizon=6, n_trials=3, n_cv_splits=2,
            families=["exp_smoothing"],
        )
        runner = TimeSeriesRunner(cfg, "test-fc-006")
        result = await runner.run(daily_df, str(tmp_path))

        assert result.succeeded
        assert result.best_family == "exp_smoothing"


# ══════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def fc_client():
    """
    TestClient with in-memory SQLite — same pattern as test_skeleton.py.
    Uses `with TestClient(...) as c: yield c` to trigger the app's lifespan
    so init_db() runs and all tables are created before requests are made.
    """
    import sys, importlib, database as db_mod
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool
    from fastapi.testclient import TestClient

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    db_mod.engine = engine
    db_mod.SessionFactory = factory
    db_mod.AsyncSessionLocal = factory

    if "main" in sys.modules:
        importlib.reload(sys.modules["main"])
    import main as app_module

    with TestClient(app_module.app, raise_server_exceptions=True) as c:
        yield c


class TestForecastingAPI:

    def _seed_dataset(self, client, tmp_path) -> int:
        """Uploads a synthetic time series CSV and returns the dataset ID."""
        import io
        rng = np.random.default_rng(0)
        n = 60
        dates = pd.date_range("2019-01-01", periods=n, freq="MS").strftime("%Y-%m-%d")
        values = 100 + np.arange(n) * 2 + rng.normal(0, 3, n)
        csv_bytes = f"date,sales\n" + "\n".join(f"{d},{v:.2f}" for d, v in zip(dates, values))

        resp = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("ts.csv", csv_bytes.encode(), "text/csv")},
            data={"name": "ts_test"},
        )
        assert resp.status_code == 201
        return resp.json()["data"]["id"]

    def test_list_families(self, fc_client):
        resp = fc_client.get("/api/v1/forecasting/families")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert isinstance(data, dict)
        assert len(data) >= 1   # at least one family must be available

    def test_submit_job(self, fc_client, tmp_path):
        dataset_id = self._seed_dataset(fc_client, tmp_path)
        resp = fc_client.post("/api/v1/forecasting/jobs", json={
            "dataset_id":     dataset_id,
            "date_column":    "date",
            "target_column":  "sales",
            "horizon":        6,
            "n_trials":       3,
        })
        assert resp.status_code == 201
        data = resp.json()["data"]
        assert "job_id" in data
        assert data["status"] == "running"

    def test_get_job_by_id(self, fc_client, tmp_path):
        dataset_id = self._seed_dataset(fc_client, tmp_path)
        job_id = fc_client.post("/api/v1/forecasting/jobs", json={
            "dataset_id": dataset_id, "date_column": "date",
            "target_column": "sales", "horizon": 6, "n_trials": 3,
        }).json()["data"]["job_id"]
        resp = fc_client.get(f"/api/v1/forecasting/jobs/{job_id}")
        assert resp.status_code == 200

    def test_get_nonexistent_job_returns_404(self, fc_client):
        resp = fc_client.get("/api/v1/forecasting/jobs/fc-doesnotexist")
        assert resp.status_code == 404

    def test_list_jobs(self, fc_client, tmp_path):
        dataset_id = self._seed_dataset(fc_client, tmp_path)
        fc_client.post("/api/v1/forecasting/jobs", json={
            "dataset_id": dataset_id, "date_column": "date",
            "target_column": "sales", "horizon": 6, "n_trials": 2,
        })
        resp = fc_client.get("/api/v1/forecasting/jobs")
        assert resp.status_code == 200
        assert len(resp.json()["data"]) >= 1

    def test_invalid_date_column_returns_error(self, fc_client, tmp_path):
        dataset_id = self._seed_dataset(fc_client, tmp_path)
        resp = fc_client.post("/api/v1/forecasting/jobs", json={
            "dataset_id": dataset_id, "date_column": "",
            "target_column": "sales", "horizon": 6, "n_trials": 2,
        })
        assert resp.status_code == 422

    def test_unknown_family_returns_422(self, fc_client, tmp_path):
        dataset_id = self._seed_dataset(fc_client, tmp_path)
        resp = fc_client.post("/api/v1/forecasting/jobs", json={
            "dataset_id": dataset_id, "date_column": "date",
            "target_column": "sales", "horizon": 6, "n_trials": 2,
            "families": ["xgboost_for_time_series_does_not_exist"],
        })
        assert resp.status_code == 422
