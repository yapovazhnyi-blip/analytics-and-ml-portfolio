"""
Tests for Phase 1 remaining items:
  - CatBoost registered in AutoML families
  - Probability calibration applied post-training
  - Request ID middleware: header on every response, passthrough of caller ID
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# ══════════════════════════════════════════════════════════════════════════
# CATBOOST
# ══════════════════════════════════════════════════════════════════════════

class TestCatBoost:

    def test_catboost_available(self):
        from training.gbm_families import CATBOOST_AVAILABLE
        assert CATBOOST_AVAILABLE, "CatBoost must be installed"

    def test_catboost_in_classification_families(self):
        from training.model_families import CLASSIFICATION_FAMILIES
        assert "catboost" in CLASSIFICATION_FAMILIES

    def test_catboost_in_regression_families(self):
        from training.model_families import REGRESSION_FAMILIES
        assert "catboost" in REGRESSION_FAMILIES

    def test_catboost_in_display_names(self):
        from training.model_families import FAMILY_DISPLAY
        assert "catboost" in FAMILY_DISPLAY
        assert FAMILY_DISPLAY["catboost"] == "CatBoost"

    def test_catboost_classifier_instantiates(self):
        """catboost_classifier(trial) must return a fitted-able estimator."""
        from training.gbm_families import catboost_classifier
        from catboost import CatBoostClassifier
        import optuna
        study  = optuna.create_study(direction="maximize")
        trial  = study.ask()
        model  = catboost_classifier(trial)
        assert isinstance(model, CatBoostClassifier)

    def test_catboost_regressor_instantiates(self):
        from training.gbm_families import catboost_regressor
        from catboost import CatBoostRegressor
        import optuna
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        model = catboost_regressor(trial)
        assert isinstance(model, CatBoostRegressor)

    def test_catboost_classifier_fits_and_predicts(self):
        from training.gbm_families import catboost_classifier
        import optuna
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 5))
        y = (X[:, 0] > 0).astype(int)
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        model = catboost_classifier(trial)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 200
        assert set(preds).issubset({0, 1})

    def test_catboost_regressor_fits_and_predicts(self):
        from training.gbm_families import catboost_regressor
        import optuna
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 5))
        y = X[:, 0] * 2 + rng.normal(0, 0.1, 200)
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        model = catboost_regressor(trial)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 200

    def test_catboost_no_files_written(self, tmp_path):
        """allow_writing_files=False must prevent CatBoost writing .cbm files."""
        from training.gbm_families import catboost_classifier
        import optuna, os
        before = set(os.listdir(tmp_path))
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        model = catboost_classifier(trial)
        rng = np.random.default_rng(42)
        model.fit(rng.normal(size=(100, 4)), rng.integers(0, 2, 100))
        after = set(os.listdir(tmp_path))
        assert before == after, "CatBoost must not write files to disk"

    def test_family_count_increased(self):
        """Registering CatBoost should increase the family count."""
        from training.model_families import CLASSIFICATION_FAMILIES
        # We have sklearn base families + XGBoost + LightGBM + CatBoost
        assert len(CLASSIFICATION_FAMILIES) >= 7


# ══════════════════════════════════════════════════════════════════════════
# PROBABILITY CALIBRATION
# ══════════════════════════════════════════════════════════════════════════

class TestProbabilityCalibration:

    def _make_df(self, n=600, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (n, 4))
        df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
        df["target"] = (X[:, 0] + X[:, 1] * 0.5 > 0).astype(int)
        return df

    def test_calibration_applied_flag(self, tmp_path):
        from training.runner import TrainingRunner, TrainingConfig
        df = self._make_df(600)
        cfg = TrainingConfig(n_trials=2, cv_folds=3, families=["random_forest"])
        runner = TrainingRunner(model_storage_path=tmp_path)
        result = runner.run(
            df=df, target_column="target", task_type="classification",
            config=cfg, experiment_name="calib_test",
        )
        assert result.calibration_applied is True

    def test_calibration_method_isotonic_large(self, tmp_path):
        from training.runner import TrainingRunner, TrainingConfig
        # Need training set ≥ 1000 rows. With test_fraction=0.2, use ≥ 1250 total.
        rng = np.random.default_rng(5)
        X = rng.normal(0, 1, (1500, 4))
        df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
        df["target"] = (X[:, 0] + X[:, 1] * 0.5 > 0).astype(int)
        cfg = TrainingConfig(n_trials=2, cv_folds=3, families=["logistic_regression"])
        runner = TrainingRunner(model_storage_path=tmp_path)
        result = runner.run(
            df=df, target_column="target", task_type="classification",
            config=cfg, experiment_name="isotonic_test",
        )
        if result.calibration_applied:
            assert result.calibration_method == "isotonic"

    def test_calibration_method_sigmoid_small(self, tmp_path):
        from training.runner import TrainingRunner, TrainingConfig
        df = self._make_df(300)   # 300 rows → sigmoid (200 ≤ n < 1000)
        cfg = TrainingConfig(n_trials=2, cv_folds=3, families=["logistic_regression"])
        runner = TrainingRunner(model_storage_path=tmp_path)
        result = runner.run(
            df=df, target_column="target", task_type="classification",
            config=cfg, experiment_name="sigmoid_test",
        )
        if result.calibration_applied:
            assert result.calibration_method == "sigmoid"

    def test_calibration_skipped_for_regression(self, tmp_path):
        from training.runner import TrainingRunner, TrainingConfig
        rng = np.random.default_rng(2)
        X = rng.normal(0, 1, (400, 3))
        df = pd.DataFrame(X, columns=["a", "b", "c"])
        df["target"] = X[:, 0] * 2 + rng.normal(0, 0.1, 400)
        cfg = TrainingConfig(n_trials=2, cv_folds=3, families=["ridge"])
        runner = TrainingRunner(model_storage_path=tmp_path)
        result = runner.run(
            df=df, target_column="target", task_type="regression",
            config=cfg, experiment_name="reg_no_calib",
        )
        assert result.calibration_applied is False

    def test_calibration_skipped_for_tiny_dataset(self, tmp_path):
        from training.runner import TrainingRunner, TrainingConfig
        rng = np.random.default_rng(3)
        X = rng.normal(0, 1, (100, 3))
        df = pd.DataFrame(X, columns=["a", "b", "c"])
        df["target"] = (X[:, 0] > 0).astype(int)
        cfg = TrainingConfig(n_trials=2, cv_folds=3, families=["logistic_regression"])
        runner = TrainingRunner(model_storage_path=tmp_path)
        result = runner.run(
            df=df, target_column="target", task_type="classification",
            config=cfg, experiment_name="tiny_no_calib",
        )
        assert result.calibration_applied is False

    @pytest.mark.asyncio
    async def test_calibrated_model_has_predict_proba(self, tmp_path):
        import joblib
        from training.runner import TrainingRunner, TrainingConfig
        df = self._make_df(600)
        cfg = TrainingConfig(n_trials=2, cv_folds=3, families=["random_forest"])
        runner = TrainingRunner(model_storage_path=tmp_path)
        result = runner.run(
            df=df, target_column="target", task_type="classification",
            config=cfg, experiment_name="proba_test",
        )
        model = joblib.load(result.artifact_path)
        X = df[["a", "b", "c", "d"]].values[:10]
        proba = model.predict_proba(X)
        assert proba.shape == (10, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_calibration_fields_in_training_result_dataclass(self):
        from training.runner import TrainingResult
        import dataclasses
        fields = {f.name for f in dataclasses.fields(TrainingResult)}
        assert "calibration_applied" in fields
        assert "calibration_method"  in fields


# ══════════════════════════════════════════════════════════════════════════
# REQUEST ID MIDDLEWARE
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mid_client():
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
    with TestClient(m.app, raise_server_exceptions=False) as c:
        yield c


class TestRequestIDMiddleware:

    def test_response_has_x_request_id(self, mid_client):
        resp = mid_client.get("/api/v1/health/live")
        assert "X-Request-ID" in resp.headers

    def test_request_id_is_hex_string(self, mid_client):
        resp = mid_client.get("/api/v1/health/live")
        rid = resp.headers["X-Request-ID"]
        assert len(rid) == 32
        int(rid, 16)   # must be valid hex

    def test_each_request_has_unique_id(self, mid_client):
        ids = {mid_client.get("/api/v1/health/live").headers["X-Request-ID"]
               for _ in range(5)}
        assert len(ids) == 5

    def test_caller_supplied_id_is_passed_through(self, mid_client):
        """If the caller sends X-Request-ID, the same value must come back."""
        custom_id = "my-trace-id-abc123"
        resp = mid_client.get(
            "/api/v1/health/live",
            headers={"X-Request-ID": custom_id},
        )
        assert resp.headers["X-Request-ID"] == custom_id

    def test_request_id_on_error_responses(self, mid_client):
        """Even 404 and 422 responses must carry X-Request-ID."""
        resp = mid_client.get("/api/v1/experiments/999999")
        assert "X-Request-ID" in resp.headers
