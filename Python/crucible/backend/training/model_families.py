"""
Model family registry for Crucible's AutoML pipeline.

Promoted and hardened from the Optuna spike (spike/optuna_spike.py).

Five families, each with:
  - suggest(trial) → fitted sklearn Pipeline ready to train
  - DISPLAY_NAME   → shown in UI and stored in experiment record
  - SUPPORTS_PROBA → whether predict_proba is available (affects SHAP path)

Design rule: every family must produce a sklearn Pipeline that includes
any required preprocessing (scaling for distance-based models). The
AutoML runner calls fit/predict without knowing which family it has.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import optuna
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC


# ── Classification families ────────────────────────────────────────────────

def _clf_random_forest(trial: optuna.Trial) -> Pipeline:
    return Pipeline([("model", RandomForestClassifier(
        n_estimators=trial.suggest_int("rf_n_estimators", 50, 300),
        max_depth=trial.suggest_int("rf_max_depth", 3, 20),
        min_samples_split=trial.suggest_int("rf_min_samples_split", 2, 20),
        max_features=trial.suggest_categorical("rf_max_features", ["sqrt", "log2"]),
        class_weight="balanced",
        random_state=42, n_jobs=-1,
    ))])


def _clf_gradient_boosting(trial: optuna.Trial) -> Pipeline:
    return Pipeline([("model", GradientBoostingClassifier(
        n_estimators=trial.suggest_int("gb_n_estimators", 50, 300),
        max_depth=trial.suggest_int("gb_max_depth", 2, 8),
        learning_rate=trial.suggest_float("gb_lr", 1e-3, 0.3, log=True),
        subsample=trial.suggest_float("gb_subsample", 0.6, 1.0),
        random_state=42,
    ))])


def _clf_logistic(trial: optuna.Trial) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=trial.suggest_float("lr_C", 1e-4, 10.0, log=True),
            solver=trial.suggest_categorical("lr_solver", ["lbfgs", "saga"]),
            class_weight="balanced",
            max_iter=1000, random_state=42,
        )),
    ])


def _clf_svm(trial: optuna.Trial) -> Pipeline:
    kernel = trial.suggest_categorical("svm_kernel", ["rbf", "linear"])
    params: dict[str, Any] = {
        "C": trial.suggest_float("svm_C", 1e-3, 10.0, log=True),
        "kernel": kernel,
        "class_weight": "balanced",
        "probability": True,
        "random_state": 42,
    }
    if kernel == "rbf":
        params["gamma"] = trial.suggest_float("svm_gamma", 1e-4, 1.0, log=True)
    return Pipeline([("scaler", StandardScaler()), ("model", SVC(**params))])


def _clf_knn(trial: optuna.Trial) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=trial.suggest_int("knn_n_neighbors", 3, 30),
            weights=trial.suggest_categorical("knn_weights", ["uniform", "distance"]),
            metric=trial.suggest_categorical("knn_metric", ["euclidean", "manhattan"]),
            n_jobs=-1,
        )),
    ])


# ── Regression families ────────────────────────────────────────────────────

def _reg_random_forest(trial: optuna.Trial) -> Pipeline:
    return Pipeline([("model", RandomForestRegressor(
        n_estimators=trial.suggest_int("rf_n_estimators", 50, 300),
        max_depth=trial.suggest_int("rf_max_depth", 3, 20),
        min_samples_split=trial.suggest_int("rf_min_samples_split", 2, 20),
        max_features=trial.suggest_categorical("rf_max_features", ["sqrt", "log2"]),
        random_state=42, n_jobs=-1,
    ))])


def _reg_gradient_boosting(trial: optuna.Trial) -> Pipeline:
    return Pipeline([("model", GradientBoostingRegressor(
        n_estimators=trial.suggest_int("gb_n_estimators", 50, 300),
        max_depth=trial.suggest_int("gb_max_depth", 2, 8),
        learning_rate=trial.suggest_float("gb_lr", 1e-3, 0.3, log=True),
        subsample=trial.suggest_float("gb_subsample", 0.6, 1.0),
        random_state=42,
    ))])


def _reg_ridge(trial: optuna.Trial) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=trial.suggest_float("ridge_alpha", 1e-3, 100.0, log=True))),
    ])


def _reg_svm(trial: optuna.Trial) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(
            C=trial.suggest_float("svm_C", 1e-3, 10.0, log=True),
            kernel=trial.suggest_categorical("svm_kernel", ["rbf", "linear"]),
            gamma=trial.suggest_float("svm_gamma", 1e-4, 1.0, log=True),
        )),
    ])


def _reg_knn(trial: optuna.Trial) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(
            n_neighbors=trial.suggest_int("knn_n_neighbors", 3, 30),
            weights=trial.suggest_categorical("knn_weights", ["uniform", "distance"]),
            n_jobs=-1,
        )),
    ])


# ── Family registries (task-specific) ─────────────────────────────────────

# Keras / TensorFlow families — registered only if TF is installed.
from training.keras_families import (
    KERAS_AVAILABLE,
    keras_mlp_classifier,
    keras_mlp_regressor,
)

# XGBoost and LightGBM — registered only if the libraries are installed.
# Both are optional: if absent, the sklearn families still cover all tasks.
from training.gbm_families import (
    XGBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    CATBOOST_AVAILABLE,
    xgb_classifier,
    xgb_regressor,
    lgbm_classifier,
    lgbm_regressor,
    catboost_classifier,
    catboost_regressor,
)

CLASSIFICATION_FAMILIES: dict[str, Callable] = {
    "random_forest":       _clf_random_forest,
    "gradient_boosting":   _clf_gradient_boosting,
    "logistic_regression": _clf_logistic,
    "svm":                 _clf_svm,
    "knn":                 _clf_knn,
    **( {"xgboost":   xgb_classifier}       if XGBOOST_AVAILABLE   else {} ),
    **( {"lightgbm":  lgbm_classifier}      if LIGHTGBM_AVAILABLE  else {} ),
    **( {"catboost":  catboost_classifier}   if CATBOOST_AVAILABLE  else {} ),
    **( {"keras_mlp": keras_mlp_classifier} if KERAS_AVAILABLE     else {} ),
}

REGRESSION_FAMILIES: dict[str, Callable] = {
    "random_forest":     _reg_random_forest,
    "gradient_boosting": _reg_gradient_boosting,
    "ridge":             _reg_ridge,
    "svm":               _reg_svm,
    "knn":               _reg_knn,
    **( {"xgboost":   xgb_regressor}       if XGBOOST_AVAILABLE   else {} ),
    **( {"lightgbm":  lgbm_regressor}      if LIGHTGBM_AVAILABLE  else {} ),
    **( {"catboost":  catboost_regressor}   if CATBOOST_AVAILABLE  else {} ),
    **( {"keras_mlp": keras_mlp_regressor} if KERAS_AVAILABLE     else {} ),
}

FAMILY_DISPLAY = {
    "random_forest":       "Random Forest",
    "gradient_boosting":   "Gradient Boosting",
    "logistic_regression": "Logistic Regression",
    "ridge":               "Ridge Regression",
    "svm":                 "SVM",
    "knn":                 "k-Nearest Neighbours",
    "xgboost":             "XGBoost",
    "lightgbm":            "LightGBM",
    "catboost":            "CatBoost",
    "keras_mlp":           "Neural Network (MLP)",
}


def get_families(task_type: str) -> dict[str, Callable]:
    if task_type == "regression":
        return REGRESSION_FAMILIES
    return CLASSIFICATION_FAMILIES
