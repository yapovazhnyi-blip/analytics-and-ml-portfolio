"""
SHAP explainability runner for Crucible Phase 2.

Promoted from spike/shap_spike.py with one addition: the explainer
auto-selects based on model family name (cheaper than isinstance checks
when the model is loaded from disk via joblib and class identity may differ).

Three explainer paths (validated in spike):
  tree   → RandomForest, GradientBoosting — fast, exact
  linear → Logistic, Ridge — fast, exact
  kernel → SVM, KNN — slow, approximate (k-means background + row sampling)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import shap
from sklearn.pipeline import Pipeline


@dataclass
class FeatureImportance:
    feature: str
    mean_abs_shap: float
    rank: int


@dataclass
class SHAPResult:
    explainer_type: str
    shap_values: np.ndarray        # (n_samples, n_features)
    feature_names: list[str]
    importance: list[FeatureImportance]
    n_samples_explained: int
    elapsed_seconds: float
    background_size: Optional[int] = None
    warning: Optional[str] = None

    def to_importance_dict(self) -> list[dict]:
        return [
            {"feature": f.feature, "mean_abs_shap": f.mean_abs_shap, "rank": f.rank}
            for f in self.importance
        ]


class SHAPRunner:

    def __init__(
        self,
        background_size: int = 50,
        max_explain_rows: int = 200,
    ):
        self.background_size = background_size
        self.max_explain_rows = max_explain_rows

    def explain(
        self,
        model: Any,
        X_background: np.ndarray,
        X_explain: np.ndarray,
        feature_names: list[str],
        family_name: str = "",
    ) -> SHAPResult:
        """
        Compute SHAP values. family_name is used to select the explainer
        without relying on isinstance (works with deserialized joblib models).

        Four explainer paths:
          tree   — TreeExplainer for Random Forest / Gradient Boosting (fast, exact)
          linear — LinearExplainer for Logistic Regression / Ridge (fast, exact)
          deep   — DeepExplainer for Keras / TensorFlow neural networks
                   Uses DeepLIFT backpropagation — faster than KernelExplainer
                   and exact for neural networks.
          kernel — KernelExplainer for everything else (slow, approximate)
        """
        estimator = _unwrap(model)
        t0 = time.monotonic()

        X_bg_t  = _transform(model, X_background)
        X_exp_t = _transform(model, X_explain)

        tree_families   = {"random_forest", "gradient_boosting", "xgboost", "lightgbm"}
        linear_families = {"logistic_regression", "ridge"}
        keras_families  = {"keras_mlp"}

        if family_name in tree_families or _is_tree(estimator):
            return self._tree(estimator, X_exp_t, feature_names, t0)
        elif family_name in linear_families or _is_linear(estimator):
            return self._linear(estimator, X_bg_t, X_exp_t, feature_names, t0)
        elif family_name in keras_families or _is_keras(estimator):
            return self._deep(estimator, X_bg_t, X_exp_t, feature_names, t0)
        else:
            return self._kernel(model, X_bg_t, X_exp_t, feature_names, t0)

    def _tree(self, estimator, X, feature_names, t0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp = shap.TreeExplainer(estimator)
            vals = exp.shap_values(X)

        if isinstance(vals, list):
            arr = np.stack(vals, axis=-1)
        else:
            arr = np.array(vals)

        return SHAPResult(
            explainer_type="tree",
            shap_values=arr,
            feature_names=feature_names,
            importance=_rank(arr, feature_names),
            n_samples_explained=len(X),
            elapsed_seconds=time.monotonic() - t0,
        )

    def _linear(self, estimator, X_bg, X_exp, feature_names, t0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp = shap.LinearExplainer(estimator, X_bg)
            vals = exp.shap_values(X_exp)

        if isinstance(vals, list):
            arr = np.stack(vals, axis=-1)
        else:
            arr = np.array(vals)

        return SHAPResult(
            explainer_type="linear",
            shap_values=arr,
            feature_names=feature_names,
            importance=_rank(arr, feature_names),
            n_samples_explained=len(X_exp),
            elapsed_seconds=time.monotonic() - t0,
        )

    def _deep(self, estimator, X_bg, X_exp, feature_names, t0):
        """
        SHAP DeepExplainer for Keras/TensorFlow neural networks.

        DeepExplainer uses DeepLIFT backpropagation — faster than
        KernelExplainer and exact for neural networks (not approximate).

        SciKeras wraps the Keras model in a KerasClassifier. DeepExplainer
        needs the inner Keras Sequential model (stored as estimator.model_
        after fitting), not the SciKeras wrapper. The wrapper does not
        have .output or .input, which DeepExplainer requires.
        """
        import warnings

        # Extract inner Keras model from SciKeras wrapper.
        # model_ (with trailing underscore) is the sklearn convention for
        # attributes set after fit() — SciKeras stores the fitted Keras
        # Sequential model here.
        keras_model = getattr(estimator, "model_", estimator)

        # Safety: if model is not yet built (no input layer), fall back to
        # KernelExplainer rather than crashing with "no defined outputs" error.
        try:
            _ = keras_model.output
        except AttributeError:
            return self._kernel(estimator, X_bg, X_exp, feature_names, t0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp = shap.DeepExplainer(keras_model, X_bg)
            vals = exp.shap_values(X_exp)

        # DeepExplainer returns a list for multi-class, array for binary
        if isinstance(vals, list):
            arr = np.stack(vals, axis=-1)
        else:
            arr = np.array(vals)

        return SHAPResult(
            explainer_type="deep",
            shap_values=arr,
            feature_names=feature_names,
            importance=_rank(arr, feature_names),
            n_samples_explained=len(X_exp),
            elapsed_seconds=time.monotonic() - t0,
        )

    def _kernel(self, model, X_bg, X_exp, feature_names, t0):
        warning = None
        bg_size = min(self.background_size, len(X_bg))
        background = shap.kmeans(X_bg, bg_size)

        n_orig = len(X_exp)
        if n_orig > self.max_explain_rows:
            idx = np.random.default_rng(42).choice(n_orig, self.max_explain_rows, replace=False)
            X_exp = X_exp[idx]
            warning = (
                f"Explain set sampled to {self.max_explain_rows} rows "
                f"(was {n_orig}) — KernelExplainer is O(n²)."
            )

        predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp = shap.KernelExplainer(predict_fn, background)
            vals = exp.shap_values(X_exp, silent=True)

        if isinstance(vals, list):
            arr = np.stack(vals, axis=-1)
        else:
            arr = np.array(vals)

        return SHAPResult(
            explainer_type="kernel",
            shap_values=arr,
            feature_names=feature_names,
            importance=_rank(arr, feature_names),
            n_samples_explained=len(X_exp),
            elapsed_seconds=time.monotonic() - t0,
            background_size=bg_size,
            warning=warning,
        )


# ── Helpers ────────────────────────────────────────────────────────────────

def _unwrap(model):
    return model[-1] if isinstance(model, Pipeline) else model


def _transform(model, X):
    if not isinstance(model, Pipeline):
        return X
    X_t = X.copy()
    for _, step in model.steps[:-1]:
        X_t = step.transform(X_t)
    return X_t


def _is_tree(est) -> bool:
    """
    Detects tree-based estimators by class name.
    Covers sklearn, XGBoost, and LightGBM without isinstance() checks
    (which can fail on joblib-deserialized models if the class is not
    importable in the current environment).
    """
    name = type(est).__name__.lower()
    return (
        "forest"   in name or
        "boosting" in name or
        "tree"     in name or
        "xgb"      in name or   # XGBClassifier, XGBRegressor
        "lgbm"     in name       # LGBMClassifier, LGBMRegressor
    )


def _is_linear(est) -> bool:
    name = type(est).__name__.lower()
    return "logistic" in name or "ridge" in name or "linear" in name


def _is_keras(est) -> bool:
    """
    Detects SciKeras wrappers (KerasClassifier, KerasRegressor) and raw
    Keras Sequential/Functional models.

    Uses class name string matching rather than isinstance() because
    joblib-deserialized models may fail isinstance() checks if the class
    is not importable in the current environment.
    """
    name = type(est).__name__.lower()
    return "keras" in name or "sequential" in name or "functional" in name


def _rank(arr: np.ndarray, feature_names: list[str]) -> list[FeatureImportance]:
    if arr.ndim == 3:
        vals = np.abs(arr).mean(axis=(0, 2))
    else:
        vals = np.abs(arr).mean(axis=0)

    ranked = sorted(zip(feature_names, vals), key=lambda x: x[1], reverse=True)
    return [
        FeatureImportance(feature=name, mean_abs_shap=round(float(v), 6), rank=i + 1)
        for i, (name, v) in enumerate(ranked)
    ]
