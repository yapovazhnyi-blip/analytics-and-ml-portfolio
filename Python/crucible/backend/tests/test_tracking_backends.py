"""
Tracking Backend tests.

Tests cover:
  - TrackedRun dataclass / succeeded property
  - get_tracking_backend() factory resolution (mlflow default, wandb, none)
  - MLflowBackend: successful log_run, missing mlflow handled, exception handled
  - WandBBackend: successful log_run (mocked SDK), missing API key, missing wandb,
    exception handled, wandb.finish() always called (even on error)
  - NullTrackingBackend: always returns provider="none", no error
  - Integration: TrainingRunner.run() uses the configured backend and stores its run_id
  - API: /cloud/tracking-providers reflects the active backend
"""

from __future__ import annotations

import sys
import types
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch


# ══════════════════════════════════════════════════════════════════════════
# TRACKED RUN
# ══════════════════════════════════════════════════════════════════════════

class TestTrackedRun:

    def test_succeeded_true_when_run_id_present(self):
        from tracking.base import TrackedRun
        r = TrackedRun(run_id="abc123", provider="mlflow")
        assert r.succeeded is True

    def test_succeeded_false_when_no_run_id(self):
        from tracking.base import TrackedRun
        r = TrackedRun(run_id=None, provider="mlflow")
        assert r.succeeded is False

    def test_succeeded_false_when_error_set(self):
        from tracking.base import TrackedRun
        r = TrackedRun(run_id="abc", provider="wandb", error="something failed")
        assert r.succeeded is False


# ══════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════

class TestTrackingFactory:

    def test_default_is_mlflow(self):
        from tracking.base import get_tracking_backend
        from tracking.mlflow_backend import MLflowBackend
        from config import settings
        original = settings.tracking_backend
        try:
            settings.tracking_backend = "mlflow"
            backend = get_tracking_backend()
        finally:
            settings.tracking_backend = original
        assert isinstance(backend, MLflowBackend)
        assert backend.provider_name == "mlflow"

    def test_wandb_selected_via_settings(self):
        from tracking.base import get_tracking_backend
        from tracking.wandb_backend import WandBBackend
        from config import settings
        original = settings.tracking_backend
        try:
            settings.tracking_backend = "wandb"
            backend = get_tracking_backend()
        finally:
            settings.tracking_backend = original
        assert isinstance(backend, WandBBackend)
        assert backend.provider_name == "wandb"

    def test_none_selected_via_settings(self):
        from tracking.base import get_tracking_backend
        from tracking.null_backend import NullTrackingBackend
        from config import settings
        original = settings.tracking_backend
        try:
            settings.tracking_backend = "none"
            backend = get_tracking_backend()
        finally:
            settings.tracking_backend = original
        assert isinstance(backend, NullTrackingBackend)

    def test_unknown_value_falls_back_to_mlflow(self):
        from tracking.base import get_tracking_backend
        from tracking.mlflow_backend import MLflowBackend
        from config import settings
        original = settings.tracking_backend
        try:
            settings.tracking_backend = "nonexistent_provider"
            backend = get_tracking_backend()
        finally:
            settings.tracking_backend = original
        assert isinstance(backend, MLflowBackend)


# ══════════════════════════════════════════════════════════════════════════
# MLFLOW BACKEND
# ══════════════════════════════════════════════════════════════════════════

class TestMLflowBackend:

    def test_provider_name(self):
        from tracking.mlflow_backend import MLflowBackend
        assert MLflowBackend().provider_name == "mlflow"

    def test_successful_log_run(self, tmp_path):
        from tracking.mlflow_backend import MLflowBackend

        artifact = tmp_path / "model.joblib"
        artifact.write_bytes(b"fake model bytes")

        mock_run = MagicMock()
        mock_run.info.run_id = "run-abc-123"
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.get_tracking_uri.return_value = "sqlite:///./data/mlflow.db"

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            backend = MLflowBackend()
            result = backend.log_run(
                run_name="test-run",
                params={"family": "xgboost", "n_estimators": 100},
                metrics={"accuracy": 0.91},
                artifact_path=str(artifact),
            )

        assert result.succeeded
        assert result.run_id == "run-abc-123"
        assert result.provider == "mlflow"
        # Non-http tracking URI (sqlite) must not produce a browsable run_url
        assert result.run_url is None

    def test_http_tracking_uri_produces_run_url(self, tmp_path):
        from tracking.mlflow_backend import MLflowBackend

        mock_run = MagicMock()
        mock_run.info.run_id = "run-xyz"
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.get_tracking_uri.return_value = "http://mlflow.internal:5000"

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = MLflowBackend().log_run(run_name="r", params={}, metrics={})

        assert result.run_url is not None
        assert "mlflow.internal" in result.run_url

    def test_mlflow_not_installed_returns_error(self):
        from tracking.mlflow_backend import MLflowBackend

        # Simulate ImportError by removing mlflow from sys.modules and
        # making the import fail
        with patch.dict(sys.modules, {"mlflow": None}):
            result = MLflowBackend().log_run(run_name="r", params={}, metrics={})

        assert not result.succeeded
        assert "not installed" in result.error

    def test_exception_during_logging_caught(self):
        from tracking.mlflow_backend import MLflowBackend

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.side_effect = Exception("connection refused")

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = MLflowBackend().log_run(run_name="r", params={}, metrics={})

        assert not result.succeeded
        assert "connection refused" in result.error

    def test_non_numeric_metrics_silently_dropped(self):
        from tracking.mlflow_backend import MLflowBackend

        mock_run = MagicMock()
        mock_run.info.run_id = "run-1"
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.get_tracking_uri.return_value = "sqlite:///x.db"

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = MLflowBackend().log_run(
                run_name="r", params={}, metrics={"accuracy": 0.9, "label": "not-a-number"},
            )

        assert result.succeeded
        logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
        assert "accuracy" in logged_metrics
        assert "label" not in logged_metrics


# ══════════════════════════════════════════════════════════════════════════
# W&B BACKEND
# ══════════════════════════════════════════════════════════════════════════

class TestWandBBackend:

    def test_provider_name(self):
        from tracking.wandb_backend import WandBBackend
        assert WandBBackend().provider_name == "wandb"

    def test_missing_api_key_returns_error_without_calling_sdk(self):
        from tracking.wandb_backend import WandBBackend

        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            result = WandBBackend(api_key="").log_run(run_name="r", params={}, metrics={})

        assert not result.succeeded
        assert "API key" in result.error
        mock_wandb.init.assert_not_called()

    def test_wandb_not_installed_returns_error(self):
        from tracking.wandb_backend import WandBBackend

        with patch.dict(sys.modules, {"wandb": None}):
            result = WandBBackend(api_key="fake-key").log_run(run_name="r", params={}, metrics={})

        assert not result.succeeded
        assert "not installed" in result.error

    def test_successful_log_run(self, tmp_path):
        from tracking.wandb_backend import WandBBackend

        artifact = tmp_path / "model.joblib"
        artifact.write_bytes(b"fake model")

        mock_run = MagicMock()
        mock_run.id = "wandb-run-123"
        mock_run.get_url.return_value = "https://wandb.ai/me/crucible/runs/wandb-run-123"

        mock_wandb = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Artifact.return_value = MagicMock()

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            result = WandBBackend(api_key="fake-key", project="crucible-test").log_run(
                run_name="test-run",
                params={"family": "xgboost"},
                metrics={"accuracy": 0.9},
                artifact_path=str(artifact),
                tags={"env": "test"},
            )

        assert result.succeeded
        assert result.run_id == "wandb-run-123"
        assert result.provider == "wandb"
        assert "wandb.ai" in result.run_url
        mock_wandb.login.assert_called_once()
        mock_wandb.init.assert_called_once()
        mock_wandb.log.assert_called_once()
        mock_wandb.finish.assert_called_once()

    def test_finish_called_even_on_exception(self):
        """wandb.finish() must run even if logging metrics raises mid-way."""
        from tracking.wandb_backend import WandBBackend

        mock_run = MagicMock()
        mock_run.id = "run-err"
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.log.side_effect = Exception("network error")

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            result = WandBBackend(api_key="fake-key").log_run(
                run_name="r", params={}, metrics={"x": 1.0},
            )

        assert not result.succeeded
        mock_wandb.finish.assert_called_once()

    def test_finish_not_called_if_init_never_succeeded(self):
        """If wandb.init() itself raises, there's no run to finish()."""
        from tracking.wandb_backend import WandBBackend

        mock_wandb = MagicMock()
        mock_wandb.init.side_effect = Exception("auth failed")

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            result = WandBBackend(api_key="fake-key").log_run(run_name="r", params={}, metrics={})

        assert not result.succeeded
        mock_wandb.finish.assert_not_called()

    def test_project_passed_to_init(self):
        from tracking.wandb_backend import WandBBackend

        mock_run = MagicMock()
        mock_run.id = "r1"
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            WandBBackend(api_key="key", project="my-custom-project").log_run(
                run_name="r", params={}, metrics={},
            )

        call_kwargs = mock_wandb.init.call_args.kwargs
        assert call_kwargs["project"] == "my-custom-project"


# ══════════════════════════════════════════════════════════════════════════
# NULL BACKEND
# ══════════════════════════════════════════════════════════════════════════

class TestNullBackend:

    def test_always_returns_none_provider(self):
        from tracking.null_backend import NullTrackingBackend
        result = NullTrackingBackend().log_run(run_name="r", params={"a": 1}, metrics={"b": 2.0})
        assert result.provider == "none"
        assert result.run_id is None
        assert result.error is None   # explicit no-op, not a failure

    def test_provider_name(self):
        from tracking.null_backend import NullTrackingBackend
        assert NullTrackingBackend().provider_name == "none"


# ══════════════════════════════════════════════════════════════════════════
# INTEGRATION — TRAINING RUNNER USES THE CONFIGURED BACKEND
# ══════════════════════════════════════════════════════════════════════════

class TestTrainingRunnerIntegration:

    def test_training_run_stores_tracking_run_id(self, tmp_path):
        """
        TrainingResult.mlflow_run_id is populated from whichever backend
        get_tracking_backend() resolves to — using "none" here keeps the
        test fast and avoids needing a real MLflow store.
        """
        from training.runner import TrainingRunner, TrainingConfig
        from config import settings

        original = settings.tracking_backend
        try:
            settings.tracking_backend = "none"
            rng = np.random.default_rng(0)
            df = pd.DataFrame({
                "a": rng.normal(size=200), "b": rng.normal(size=200),
            })
            df["target"] = (df["a"] > 0).astype(int)

            cfg = TrainingConfig(n_trials=2, cv_folds=2, families=["logistic_regression"])
            runner = TrainingRunner(model_storage_path=tmp_path)
            result = runner.run(
                df=df, target_column="target", task_type="classification",
                config=cfg, experiment_name="tracking_integration_test",
            )
        finally:
            settings.tracking_backend = original

        # NullTrackingBackend always returns run_id=None — confirms the
        # pipeline calls the configured backend rather than always MLflow
        assert result.mlflow_run_id is None

    def test_training_run_with_mocked_mlflow_stores_run_id(self, tmp_path):
        from training.runner import TrainingRunner, TrainingConfig
        from config import settings

        mock_run = MagicMock()
        mock_run.info.run_id = "integration-run-id"
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.get_tracking_uri.return_value = "sqlite:///x.db"

        original = settings.tracking_backend
        try:
            settings.tracking_backend = "mlflow"
            with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
                rng = np.random.default_rng(1)
                df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
                df["target"] = (df["a"] > 0).astype(int)

                cfg = TrainingConfig(n_trials=2, cv_folds=2, families=["logistic_regression"])
                runner = TrainingRunner(model_storage_path=tmp_path)
                result = runner.run(
                    df=df, target_column="target", task_type="classification",
                    config=cfg, experiment_name="mlflow_integration_test",
                )
        finally:
            settings.tracking_backend = original

        assert result.mlflow_run_id == "integration-run-id"


# ══════════════════════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tracking_client():
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


class TestTrackingProvidersAPI:

    def test_endpoint_lists_all_three_providers(self, tracking_client):
        resp = tracking_client.get("/api/v1/cloud/tracking-providers")
        assert resp.status_code == 200
        providers = resp.json()["data"]
        names = {p["provider"] for p in providers}
        assert names == {"mlflow", "wandb", "none"}

    def test_mlflow_is_active_by_default(self, tracking_client):
        resp = tracking_client.get("/api/v1/cloud/tracking-providers")
        providers = resp.json()["data"]
        mlflow_entry = next(p for p in providers if p["provider"] == "mlflow")
        assert mlflow_entry["active"] is True

    def test_only_one_provider_marked_active(self, tracking_client):
        resp = tracking_client.get("/api/v1/cloud/tracking-providers")
        providers = resp.json()["data"]
        active_count = sum(1 for p in providers if p["active"])
        assert active_count == 1
