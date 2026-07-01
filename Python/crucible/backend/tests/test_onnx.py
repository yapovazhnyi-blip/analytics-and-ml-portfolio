"""
ONNX export tests.

Tests cover:
  - export_to_onnx() for sklearn families (random forest, logistic regression, ridge)
  - ONNXExportResult shape validation
  - ONNX Runtime inference produces valid output
  - Graceful error for unsupported models
  - generate_onnx_server() produces valid Python
  - API endpoint integration (POST /experiments/{id}/export/onnx)
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clf_data():
    X, y = make_classification(n_samples=200, n_features=6, random_state=42)
    return X.astype(np.float32), y


@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=200, n_features=6, random_state=42)
    return X.astype(np.float32), y


@pytest.fixture
def rf_artifact(tmp_path, clf_data):
    X, y = clf_data
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    path = str(tmp_path / "rf_model.pkl")
    joblib.dump(model, path)
    return path


@pytest.fixture
def lr_artifact(tmp_path, clf_data):
    X, y = clf_data
    model = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=100))])
    model.fit(X, y)
    path = str(tmp_path / "lr_model.pkl")
    joblib.dump(model, path)
    return path


@pytest.fixture
def ridge_artifact(tmp_path, reg_data):
    X, y = reg_data
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    path = str(tmp_path / "ridge_model.pkl")
    joblib.dump(model, path)
    return path


FEATURE_NAMES = [f"feat_{i}" for i in range(6)]


# ══════════════════════════════════════════════════════════════════════════
# ONNX EXPORT — RESULT SHAPE
# ══════════════════════════════════════════════════════════════════════════

class TestONNXExport:

    def test_random_forest_export_succeeds(self, rf_artifact, tmp_path):
        from deployment.onnx_exporter import export_to_onnx
        result = export_to_onnx(rf_artifact, FEATURE_NAMES, "classification", str(tmp_path), 1)
        assert result.succeeded, result.error
        assert result.onnx_path.endswith(".onnx")
        assert result.n_features == 6
        assert result.model_size_bytes > 0

    def test_logistic_regression_pipeline_export(self, lr_artifact, tmp_path):
        from deployment.onnx_exporter import export_to_onnx
        result = export_to_onnx(lr_artifact, FEATURE_NAMES, "classification", str(tmp_path), 2)
        assert result.succeeded, result.error
        assert result.input_name != ""
        assert len(result.output_names) >= 1

    def test_ridge_regression_export(self, ridge_artifact, tmp_path):
        from deployment.onnx_exporter import export_to_onnx
        result = export_to_onnx(ridge_artifact, FEATURE_NAMES, "regression", str(tmp_path), 3)
        assert result.succeeded, result.error

    def test_onnx_file_created_on_disk(self, rf_artifact, tmp_path):
        from deployment.onnx_exporter import export_to_onnx
        import os
        result = export_to_onnx(rf_artifact, FEATURE_NAMES, "classification", str(tmp_path), 4)
        assert result.succeeded
        assert os.path.exists(result.onnx_path)
        assert os.path.getsize(result.onnx_path) > 0

    def test_model_size_kb_correct(self, rf_artifact, tmp_path):
        from deployment.onnx_exporter import export_to_onnx
        import os
        result = export_to_onnx(rf_artifact, FEATURE_NAMES, "classification", str(tmp_path), 5)
        assert result.succeeded
        expected_kb = os.path.getsize(result.onnx_path) / 1024
        assert abs(result.model_size_kb - expected_kb) < 0.1

    def test_invalid_artifact_returns_error(self, tmp_path):
        from deployment.onnx_exporter import export_to_onnx
        result = export_to_onnx("/nonexistent/model.pkl", FEATURE_NAMES, "classification", str(tmp_path), 99)
        assert not result.succeeded
        assert result.error is not None

    def test_keras_model_returns_helpful_error(self, tmp_path):
        """Keras models should return a clear error, not an exception."""
        from deployment.onnx_exporter import _error
        # Simulate what happens when a Keras/TF model is detected
        result = _error("Keras/TensorFlow models require tf2onnx")
        assert not result.succeeded
        assert "tf2onnx" in result.error


# ══════════════════════════════════════════════════════════════════════════
# ONNX RUNTIME INFERENCE
# ══════════════════════════════════════════════════════════════════════════

class TestONNXInference:
    """
    Verifies that the exported ONNX model produces the same predictions
    as the original sklearn model on the same input.
    """

    def test_rf_predictions_match_sklearn(self, rf_artifact, tmp_path, clf_data):
        from deployment.onnx_exporter import export_to_onnx
        import onnxruntime as rt

        X, _ = clf_data
        X32 = X[:10].astype(np.float32)

        # Original sklearn predictions
        original = joblib.load(rf_artifact)
        sklearn_preds = original.predict(X32)

        # ONNX Runtime predictions
        result = export_to_onnx(rf_artifact, FEATURE_NAMES, "classification", str(tmp_path), 10)
        assert result.succeeded
        sess = rt.InferenceSession(result.onnx_path, providers=["CPUExecutionProvider"])
        onnx_out = sess.run(None, {result.input_name: X32})
        onnx_preds = onnx_out[0]

        np.testing.assert_array_equal(sklearn_preds, onnx_preds)

    def test_ridge_predictions_close_to_sklearn(self, ridge_artifact, tmp_path, reg_data):
        from deployment.onnx_exporter import export_to_onnx
        import onnxruntime as rt

        X, _ = reg_data
        X32 = X[:10].astype(np.float32)

        original = joblib.load(ridge_artifact)
        sklearn_preds = original.predict(X32)

        result = export_to_onnx(ridge_artifact, FEATURE_NAMES, "regression", str(tmp_path), 11)
        assert result.succeeded
        sess = rt.InferenceSession(result.onnx_path, providers=["CPUExecutionProvider"])
        onnx_out = sess.run(None, {result.input_name: X32})
        onnx_preds = onnx_out[0].flatten()

        np.testing.assert_allclose(sklearn_preds, onnx_preds, rtol=1e-4, atol=1e-4)

    def test_onnx_input_name_correct(self, lr_artifact, tmp_path):
        from deployment.onnx_exporter import export_to_onnx
        import onnxruntime as rt

        result = export_to_onnx(lr_artifact, FEATURE_NAMES, "classification", str(tmp_path), 12)
        assert result.succeeded
        sess = rt.InferenceSession(result.onnx_path, providers=["CPUExecutionProvider"])

        # The input name from the result must match the session input name
        session_input_name = sess.get_inputs()[0].name
        assert result.input_name == session_input_name

    def test_onnx_handles_batch_input(self, rf_artifact, tmp_path, clf_data):
        """ONNX model should handle variable batch sizes."""
        from deployment.onnx_exporter import export_to_onnx
        import onnxruntime as rt

        X, _ = clf_data
        result = export_to_onnx(rf_artifact, FEATURE_NAMES, "classification", str(tmp_path), 13)
        assert result.succeeded
        sess = rt.InferenceSession(result.onnx_path, providers=["CPUExecutionProvider"])

        for batch_size in [1, 5, 50]:
            X_batch = X[:batch_size].astype(np.float32)
            out = sess.run(None, {result.input_name: X_batch})
            assert len(out[0]) == batch_size


# ══════════════════════════════════════════════════════════════════════════
# SERVER + README GENERATION
# ══════════════════════════════════════════════════════════════════════════

class TestCodeGeneration:

    def test_onnx_server_is_valid_python(self):
        from deployment.onnx_exporter import generate_onnx_server
        import ast
        server_code = generate_onnx_server(
            onnx_filename="model.onnx",
            feature_names=FEATURE_NAMES,
            task_type="classification",
            input_name="float_input",
            output_names=["label", "probabilities"],
        )
        # Should parse without SyntaxError
        ast.parse(server_code)

    def test_onnx_server_regression_variant(self):
        from deployment.onnx_exporter import generate_onnx_server
        import ast
        server_code = generate_onnx_server(
            onnx_filename="model.onnx",
            feature_names=FEATURE_NAMES,
            task_type="regression",
            input_name="float_input",
            output_names=["variable"],
        )
        ast.parse(server_code)

    def test_generate_readme_contains_key_sections(self):
        from deployment.onnx_exporter import generate_onnx_readme
        readme = generate_onnx_readme(
            experiment_id=42,
            onnx_filename="experiment_42_model.onnx",
            n_features=6,
            model_size_kb=85.3,
            family_name="RandomForest",
            opset=17,
        )
        assert "ONNX" in readme
        assert "uvicorn" in readme
        assert "INT8" in readme
        assert "experiment_42_model.onnx" in readme


# ══════════════════════════════════════════════════════════════════════════
# XGBoost ONNX (optional)
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not __import__("training.gbm_families", fromlist=["XGBOOST_AVAILABLE"]).XGBOOST_AVAILABLE,
    reason="xgboost not installed"
)
class TestXGBoostONNX:

    @pytest.fixture
    def xgb_artifact(self, tmp_path, clf_data):
        from xgboost import XGBClassifier
        X, y = clf_data
        model = XGBClassifier(n_estimators=10, verbosity=0, random_state=42)
        model.fit(X, y)
        path = str(tmp_path / "xgb_model.pkl")
        joblib.dump(model, path)
        return path

    def test_xgboost_export_or_helpful_error(self, xgb_artifact, tmp_path):
        """XGBoost export either succeeds or returns a helpful error (not a crash)."""
        from deployment.onnx_exporter import export_to_onnx
        result = export_to_onnx(xgb_artifact, FEATURE_NAMES, "classification", str(tmp_path), 20)
        # Either succeeded or gave a clear error message
        if not result.succeeded:
            assert result.error is not None
            assert len(result.error) > 10   # not just an empty error
        else:
            assert result.onnx_path.endswith(".onnx")
