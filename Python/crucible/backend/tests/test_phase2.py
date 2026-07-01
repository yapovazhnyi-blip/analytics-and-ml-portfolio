"""
Phase 2 tests — AutoML pipeline, SHAP, job manager, experiments endpoint.

Training tests use small datasets (100 rows, 2-3 features) and minimal
trial counts (n_trials=3) so the suite stays fast. The goal is to
validate the interface and data flow, not benchmark training quality.
"""

import io
import json
import time

import numpy as np
import pandas as pd
import pytest

from training.runner import TrainingConfig, TrainingResult, TrainingRunner
from explainability.shap_runner import SHAPRunner
from training.model_families import get_families, CLASSIFICATION_FAMILIES, REGRESSION_FAMILIES


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def clf_df():
    """Small binary classification dataset."""
    rng = np.random.default_rng(42)
    n = 150
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    y = (X1 + X2 > 0).astype(int)
    return pd.DataFrame({"feat_a": X1, "feat_b": X2, "target": y})


@pytest.fixture
def reg_df():
    """Small regression dataset."""
    rng = np.random.default_rng(7)
    n = 150
    X = rng.normal(0, 1, n)
    y = X * 2.5 + rng.normal(0, 0.5, n)
    return pd.DataFrame({"feat_x": X, "price": y})


@pytest.fixture
def fast_config():
    return TrainingConfig(n_trials=4, cv_folds=2, test_fraction=0.2)


# ══════════════════════════════════════════════════════════════════════════
# MODEL FAMILIES
# ══════════════════════════════════════════════════════════════════════════

class TestModelFamilies:

    def test_classification_families_count(self):
        # 5 sklearn + optional XGBoost + optional LightGBM + optional CatBoost + optional Keras MLP
        from training.keras_families import KERAS_AVAILABLE
        from training.gbm_families import XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE, CATBOOST_AVAILABLE
        expected = (5 + int(XGBOOST_AVAILABLE) + int(LIGHTGBM_AVAILABLE)
                    + int(CATBOOST_AVAILABLE) + int(KERAS_AVAILABLE))
        assert len(CLASSIFICATION_FAMILIES) == expected

    def test_regression_families_count(self):
        from training.keras_families import KERAS_AVAILABLE
        from training.gbm_families import XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE, CATBOOST_AVAILABLE
        expected = (5 + int(XGBOOST_AVAILABLE) + int(LIGHTGBM_AVAILABLE)
                    + int(CATBOOST_AVAILABLE) + int(KERAS_AVAILABLE))
        assert len(REGRESSION_FAMILIES) == expected

    @pytest.mark.skipif(
        not __import__("training.gbm_families", fromlist=["XGBOOST_AVAILABLE"]).XGBOOST_AVAILABLE,
        reason="xgboost not installed"
    )
    def test_xgboost_family_trains_and_predicts(self, clf_df):
        """XGBoost classifier trains on a small dataset and produces valid predictions."""
        import optuna, numpy as np
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        from training.gbm_families import xgb_classifier

        X = clf_df[["feat_a", "feat_b"]].values.astype(np.float32)
        y = clf_df["target"].values

        def objective(trial):
            model = xgb_classifier(trial)
            model.fit(X, y)
            return model.score(X, y)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=2)
        assert study.best_value > 0.0

    @pytest.mark.skipif(
        not __import__("training.gbm_families", fromlist=["LIGHTGBM_AVAILABLE"]).LIGHTGBM_AVAILABLE,
        reason="lightgbm not installed"
    )
    def test_lightgbm_family_trains_and_predicts(self, clf_df):
        """LightGBM classifier trains on a small dataset and produces valid predictions."""
        import optuna, numpy as np
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        from training.gbm_families import lgbm_classifier

        X = clf_df[["feat_a", "feat_b"]].values.astype(np.float32)
        y = clf_df["target"].values

        def objective(trial):
            model = lgbm_classifier(trial)
            model.fit(X, y)
            return model.score(X, y)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=2)
        assert study.best_value > 0.0

    @pytest.mark.skipif(
        not __import__("training.gbm_families", fromlist=["XGBOOST_AVAILABLE"]).XGBOOST_AVAILABLE,
        reason="xgboost not installed"
    )
    def test_xgboost_in_training_runner(self, clf_df, tmp_path):
        """XGBoost participates in the full TrainingRunner experiment loop."""
        from training.runner import TrainingRunner, TrainingConfig
        config = TrainingConfig(n_trials=3, cv_folds=2, families=["xgboost"])
        runner = TrainingRunner(model_storage_path=str(tmp_path))
        result = runner.run(clf_df, target_column="target", task_type="classification", config=config)
        assert isinstance(result, __import__("training.runner", fromlist=["TrainingResult"]).TrainingResult)
        assert result.best_family == "xgboost"
        assert result.best_cv_score > 0.0
        assert result.artifact_path  # model was saved

    @pytest.mark.skipif(
        not __import__("training.gbm_families", fromlist=["LIGHTGBM_AVAILABLE"]).LIGHTGBM_AVAILABLE,
        reason="lightgbm not installed"
    )
    def test_lightgbm_in_training_runner(self, clf_df, tmp_path):
        """LightGBM participates in the full TrainingRunner experiment loop."""
        from training.runner import TrainingRunner, TrainingConfig, TrainingResult
        config = TrainingConfig(n_trials=3, cv_folds=2, families=["lightgbm"])
        runner = TrainingRunner(model_storage_path=str(tmp_path))
        result = runner.run(clf_df, target_column="target", task_type="classification", config=config)
        assert isinstance(result, TrainingResult)
        assert result.best_family == "lightgbm"
        assert result.best_cv_score > 0.0
        assert result.artifact_path

    def test_get_families_classification(self):
        fams = get_families("classification")
        assert "random_forest" in fams
        assert "logistic_regression" in fams

    def test_get_families_regression(self):
        fams = get_families("regression")
        assert "ridge" in fams
        assert "random_forest" in fams

    def test_each_clf_family_produces_pipeline(self, clf_df):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        X = clf_df[["feat_a", "feat_b"]].values
        y = clf_df["target"].values

        for name, fn in CLASSIFICATION_FAMILIES.items():
            study = optuna.create_study(direction="maximize")
            def obj(trial):
                p = fn(trial)
                p.fit(X[:100], y[:100])
                return float(p.score(X[100:], y[100:]))
            study.optimize(obj, n_trials=1)
            assert study.best_value is not None, f"Family {name} failed"


# ══════════════════════════════════════════════════════════════════════════
# TRAINING RUNNER
# ══════════════════════════════════════════════════════════════════════════

class TestTrainingRunner:

    def test_classification_returns_result(self, clf_df, fast_config, tmp_path):
        runner = TrainingRunner(model_storage_path=str(tmp_path))
        result = runner.run(
            df=clf_df, target_column="target",
            task_type="classification", config=fast_config,
        )
        assert isinstance(result, TrainingResult)
        assert result.best_family in CLASSIFICATION_FAMILIES
        assert 0.0 <= result.best_cv_score <= 1.0
        assert result.elapsed_secs > 0

    def test_regression_returns_result(self, reg_df, fast_config, tmp_path):
        runner = TrainingRunner(model_storage_path=str(tmp_path))
        result = runner.run(
            df=reg_df, target_column="price",
            task_type="regression", config=fast_config,
        )
        assert result.best_family in REGRESSION_FAMILIES
        assert result.scoring_metric == "r2"

    def test_model_artifact_is_saved(self, clf_df, fast_config, tmp_path):
        runner = TrainingRunner(model_storage_path=str(tmp_path))
        result = runner.run(df=clf_df, target_column="target",
                            task_type="classification", config=fast_config)
        from pathlib import Path
        assert Path(result.artifact_path).exists()
        assert result.artifact_path.endswith(".joblib")

    def test_holdout_metrics_computed(self, clf_df, fast_config, tmp_path):
        runner = TrainingRunner(model_storage_path=str(tmp_path))
        result = runner.run(df=clf_df, target_column="target",
                            task_type="classification", config=fast_config)
        assert "holdout_accuracy" in result.holdout_metrics
        assert 0.0 <= result.holdout_metrics["holdout_accuracy"] <= 1.0

    def test_feature_names_recorded(self, clf_df, fast_config, tmp_path):
        runner = TrainingRunner(model_storage_path=str(tmp_path))
        result = runner.run(df=clf_df, target_column="target",
                            task_type="classification", config=fast_config)
        assert set(result.feature_names) == {"feat_a", "feat_b"}

    def test_saved_model_predicts(self, clf_df, fast_config, tmp_path):
        """Artifact must be loadable and produce predictions."""
        import joblib
        runner = TrainingRunner(model_storage_path=str(tmp_path))
        result = runner.run(df=clf_df, target_column="target",
                            task_type="classification", config=fast_config)
        model = joblib.load(result.artifact_path)
        preds = model.predict(clf_df[result.feature_names].values[:5])
        assert len(preds) == 5

    def test_progress_reporter_called(self, clf_df, tmp_path):
        """ProgressReporter must receive trial messages during training."""
        import asyncio
        from training.runner import ProgressReporter, TrialProgress

        messages = []
        loop = asyncio.new_event_loop()
        queue = asyncio.Queue()

        class CapturingReporter(ProgressReporter):
            def send(self, msg):
                messages.append(msg)

        reporter = CapturingReporter(loop, queue)
        config = TrainingConfig(n_trials=3, cv_folds=2)
        runner = TrainingRunner(model_storage_path=str(tmp_path))
        runner.run(df=clf_df, target_column="target",
                   task_type="classification", config=config,
                   reporter=reporter)
        loop.close()

        trial_msgs = [m for m in messages if isinstance(m, TrialProgress)]
        assert len(trial_msgs) >= 1

    def test_non_numeric_columns_dropped(self, tmp_path):
        """Non-numeric features should be silently dropped, not crash."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "numeric": rng.normal(0, 1, 100),
            "text_col": ["cat"] * 50 + ["dog"] * 50,  # non-numeric
            "target": rng.integers(0, 2, 100),
        })
        config = TrainingConfig(n_trials=2, cv_folds=2)
        runner = TrainingRunner(model_storage_path=str(tmp_path))
        result = runner.run(df=df, target_column="target",
                            task_type="classification", config=config)
        assert "numeric" in result.feature_names
        assert "text_col" not in result.feature_names


# ══════════════════════════════════════════════════════════════════════════
# SHAP RUNNER
# ══════════════════════════════════════════════════════════════════════════

class TestSHAPRunner:

    @pytest.fixture
    def trained_rf(self, clf_df, fast_config, tmp_path):
        runner = TrainingRunner(model_storage_path=str(tmp_path))
        result = runner.run(df=clf_df, target_column="target",
                            task_type="classification", config=fast_config)
        import joblib
        model = joblib.load(result.artifact_path)
        return model, clf_df[result.feature_names].values, result.feature_names, result.best_family

    TREE_FAMILIES = {"random_forest", "gradient_boosting"}
    LINEAR_FAMILIES = {"logistic_regression", "ridge"}
    KERNEL_FAMILIES = {"svm", "knn"}

    def test_tree_explainer_selected_for_rf(self, trained_rf):
        model, X, feature_names, family = trained_rf
        runner = SHAPRunner(background_size=20, max_explain_rows=30)
        result = runner.explain(model, X[:80], X[80:], feature_names, family_name=family)
        # Explainer type must match the family that actually won
        if family in self.TREE_FAMILIES:
            assert result.explainer_type == "tree"
        elif family in self.LINEAR_FAMILIES:
            assert result.explainer_type == "linear"
        else:
            assert result.explainer_type == "kernel"

    def test_importance_covers_all_features(self, trained_rf):
        model, X, feature_names, family = trained_rf
        runner = SHAPRunner(background_size=20, max_explain_rows=30)
        result = runner.explain(model, X[:80], X[80:], feature_names, family_name=family)
        assert set(f.feature for f in result.importance) == set(feature_names)

    def test_importance_ranks_are_sequential(self, trained_rf):
        model, X, feature_names, family = trained_rf
        runner = SHAPRunner(background_size=20, max_explain_rows=30)
        result = runner.explain(model, X[:80], X[80:], feature_names, family_name=family)
        ranks = sorted(f.rank for f in result.importance)
        assert ranks == list(range(1, len(feature_names) + 1))

    def test_to_importance_dict_serialisable(self, trained_rf):
        model, X, feature_names, family = trained_rf
        runner = SHAPRunner(background_size=20, max_explain_rows=30)
        result = runner.explain(model, X[:80], X[80:], feature_names, family_name=family)
        d = result.to_importance_dict()
        json.dumps(d)  # must not raise

    def test_large_explain_set_is_sampled(self, trained_rf):
        model, X, feature_names, family = trained_rf
        runner = SHAPRunner(background_size=10, max_explain_rows=20)
        # Force kernel path by pretending unknown family
        result = runner.explain(model, X[:80], X[80:100], feature_names, family_name="knn")
        assert result.n_samples_explained <= 20


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENTS API
# ══════════════════════════════════════════════════════════════════════════

class TestExperimentsAPI:
    """Integration tests using the FastAPI test client."""

    @pytest.fixture
    def client_with_dataset(self, tmp_path, monkeypatch):
        import sys
        import importlib
        import database
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
        from sqlalchemy.pool import StaticPool

        test_engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        test_factory = async_sessionmaker(test_engine, expire_on_commit=False)
        database.engine = test_engine
        database.SessionFactory = test_factory
        database.AsyncSessionLocal = test_factory

        import config
        monkeypatch.setattr(config.settings, "dataset_storage_path", str(tmp_path / "datasets"))
        monkeypatch.setattr(config.settings, "model_storage_path", str(tmp_path / "models"))
        (tmp_path / "datasets").mkdir()
        (tmp_path / "models").mkdir()

        if "main" in sys.modules:
            importlib.reload(sys.modules["main"])
        import main as app_module

        from fastapi.testclient import TestClient
        with TestClient(app_module.app, raise_server_exceptions=False) as c:
            yield c

    def _upload_dataset(self, client, tmp_path):
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame({
            "feat_a": rng.normal(0, 1, n),
            "feat_b": rng.normal(0, 1, n),
            "target": rng.integers(0, 2, n),
        })
        csv_bytes = df.to_csv(index=False).encode()
        resp = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("train.csv", io.BytesIO(csv_bytes), "text/csv")},
            data={"name": "test_train"},
        )
        assert resp.status_code == 201
        return resp.json()["data"]["id"]

    def test_list_experiments_empty(self, client_with_dataset):
        resp = client_with_dataset.get("/api/v1/experiments")
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    def test_create_experiment_returns_201(self, client_with_dataset, tmp_path):
        ds_id = self._upload_dataset(client_with_dataset, tmp_path)
        resp = client_with_dataset.post("/api/v1/experiments", json={
            "name": "test_run",
            "dataset_id": ds_id,
            "target_column": "target",
            "task_type": "classification",
            "n_trials": 3,
            "cv_folds": 2,
            "run_shap": False,
        })
        assert resp.status_code == 201
        body = resp.json()["data"]
        assert body["name"] == "test_run"
        assert body["status"] == "running"
        assert body["job_id"] is not None

    def test_get_experiment_after_create(self, client_with_dataset, tmp_path):
        ds_id = self._upload_dataset(client_with_dataset, tmp_path)
        create = client_with_dataset.post("/api/v1/experiments", json={
            "name": "get_test",
            "dataset_id": ds_id,
            "target_column": "target",
            "task_type": "classification",
            "n_trials": 2,
            "cv_folds": 2,
            "run_shap": False,
        })
        exp_id = create.json()["data"]["id"]
        get = client_with_dataset.get(f"/api/v1/experiments/{exp_id}")
        assert get.status_code == 200
        assert get.json()["data"]["id"] == exp_id

    def test_create_with_wrong_target_column_returns_422(self, client_with_dataset, tmp_path):
        ds_id = self._upload_dataset(client_with_dataset, tmp_path)
        resp = client_with_dataset.post("/api/v1/experiments", json={
            "name": "bad_target",
            "dataset_id": ds_id,
            "target_column": "nonexistent_column",
            "task_type": "classification",
            "n_trials": 2,
            "cv_folds": 2,
        })
        assert resp.status_code == 422

    def test_create_with_bad_dataset_returns_404(self, client_with_dataset):
        resp = client_with_dataset.post("/api/v1/experiments", json={
            "name": "bad_ds",
            "dataset_id": 99999,
            "target_column": "target",
            "task_type": "classification",
            "n_trials": 2,
            "cv_folds": 2,
        })
        assert resp.status_code == 404

    def test_list_experiments_by_dataset(self, client_with_dataset, tmp_path):
        ds_id = self._upload_dataset(client_with_dataset, tmp_path)
        client_with_dataset.post("/api/v1/experiments", json={
            "name": "exp_1",
            "dataset_id": ds_id,
            "target_column": "target",
            "task_type": "classification",
            "n_trials": 2, "cv_folds": 2,
        })
        resp = client_with_dataset.get(f"/api/v1/experiments?dataset_id={ds_id}")
        assert resp.json()["pagination"]["total"] >= 1

    def test_delete_experiment(self, client_with_dataset, tmp_path):
        ds_id = self._upload_dataset(client_with_dataset, tmp_path)
        create = client_with_dataset.post("/api/v1/experiments", json={
            "name": "del_me",
            "dataset_id": ds_id,
            "target_column": "target",
            "task_type": "classification",
            "n_trials": 2, "cv_folds": 2,
        })
        exp_id = create.json()["data"]["id"]
        del_resp = client_with_dataset.delete(f"/api/v1/experiments/{exp_id}")
        assert del_resp.status_code == 204
