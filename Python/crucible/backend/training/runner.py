"""
TrainingRunner for Crucible Phase 2.

Orchestrates the full AutoML training loop:
  1. Preprocess features (numeric imputation, optional scaling)
  2. Run Optuna multi-family search with MedianPruner
  3. Evaluate best model on a holdout set
  4. Report progress via ProgressReporter (WebSocket bridge)
  5. Persist model artifact with joblib
  6. Log metrics to MLflow (if tracking URI configured)

Runs in a background thread via run_in_executor — never call from
async context directly. The progress queue is drained by the WebSocket
handler to stream live updates to the browser.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.preprocessing import LabelEncoder

from training.model_families import get_families, FAMILY_DISPLAY

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Progress message types (mirror the WebSocket spike) ───────────────────

@dataclass
class TrialProgress:
    type: str = "trial"
    trial: int = 0
    total_trials: int = 0
    family: str = ""
    score: float = 0.0
    best_score: float = 0.0
    best_family: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TrainingComplete:
    type: str = "complete"
    best_family: str = ""
    best_score: float = 0.0
    scoring_metric: str = ""
    n_trials: int = 0
    n_pruned: int = 0
    elapsed_secs: float = 0.0
    artifact_path: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TrainingError:
    type: str = "error"
    message: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ── Training configuration ─────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    n_trials: int = 30
    cv_folds: int = 3
    timeout_secs: Optional[float] = None
    test_fraction: float = 0.2
    scoring: Optional[str] = None       # None → auto-select by task type
    random_state: int = 42
    families: Optional[list[str]] = None  # None → use all registered families
                                           # List → restrict search to these names
                                           # e.g. ["xgboost", "lightgbm"] for fast GBM-only runs
    pruner_type: str = "median"           # "median" | "hyperband"
                                           # median:    prunes trials below the running median
                                           #            at each CV fold (good default, robust)
                                           # hyperband: multi-fidelity successive-halving pruner.
                                           #            More aggressive — allocates more of the
                                           #            trial budget to promising configurations
                                           #            by killing bad trials earlier and with a
                                           #            geometric (reduction_factor) schedule.
                                           #            Better for larger n_trials (50+) where the
                                           #            extra trials spent on early-promising
                                           #            configs pay off.


# ── Training result ────────────────────────────────────────────────────────

@dataclass
class TrainingResult:
    best_family: str
    best_params: dict
    best_cv_score: float
    scoring_metric: str
    holdout_metrics: dict[str, float]
    n_trials_completed: int
    n_trials_pruned: int
    elapsed_secs: float
    artifact_path: str
    feature_names: list[str]
    task_type: str
    mlflow_run_id: Optional[str] = None
    calibration_applied: bool = False
    calibration_method: Optional[str] = None   # "isotonic" | "sigmoid" | None
    pruner_type: str = "median"                 # "median" | "hyperband"


# ── Progress reporter (thread-safe bridge to event loop) ──────────────────

class ProgressReporter:
    """Sends messages from training thread to async event loop via queue."""

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        self._loop = loop
        self._queue = queue

    def send(self, msg) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, msg)


# ── The runner ─────────────────────────────────────────────────────────────

class TrainingRunner:

    def __init__(self, model_storage_path: str):
        self.model_storage_path = Path(model_storage_path)
        self.model_storage_path.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        config: TrainingConfig,
        reporter: Optional[ProgressReporter] = None,
        experiment_name: str = "crucible_run",
    ) -> TrainingResult:
        """
        Synchronous training loop. Runs in a background thread.
        Reports progress via reporter (if provided) for WebSocket streaming.
        """
        t0 = time.monotonic()

        # ── 1. Prepare data ────────────────────────────────────────────────
        feature_cols = [c for c in df.columns if c != target_column]
        df_clean = self._preprocess(df, feature_cols, target_column)
        # Update feature_cols: preprocessing may have dropped non-numeric columns
        feature_cols = [c for c in feature_cols if c in df_clean.columns]

        X = df_clean[feature_cols].values
        y_raw = df_clean[target_column].values

        # Encode classification targets to int
        le = None
        if task_type == "classification":
            le = LabelEncoder()
            y = le.fit_transform(y_raw.astype(str))
        else:
            y = y_raw.astype(float)

        # Train/holdout split (stratified for classification)
        split_idx = max(1, int(len(X) * (1 - config.test_fraction)))
        X_train, X_holdout = X[:split_idx], X[split_idx:]
        y_train, y_holdout = y[:split_idx], y[split_idx:]

        # ── 2. Optuna search ───────────────────────────────────────────────
        scoring = config.scoring or _default_scoring(task_type)
        all_families = get_families(task_type)

        # Apply optional family filter from TrainingConfig.families.
        # Useful for: isolated tests, user-selected families in the UI,
        # or running a quick GBM-only search on large datasets.
        if config.families:
            unknown = set(config.families) - set(all_families)
            if unknown:
                raise ValueError(
                    f"Unknown model families: {unknown}. "
                    f"Available: {sorted(all_families)}"
                )
            families = {k: v for k, v in all_families.items() if k in config.families}
        else:
            families = all_families

        best_score = -np.inf
        best_family = ""
        trial_count = 0

        if config.pruner_type == "hyperband":
            # HyperbandPruner: multi-fidelity successive halving.
            # "Resource" here is the CV fold index (0..cv_folds-1) — the pruner
            # decides, after each fold, whether a trial has enough promise to
            # continue to the next fold or should be killed early.
            # reduction_factor=2 means roughly half of trials are pruned at
            # each rung, following the classic Hyperband geometric schedule.
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=config.cv_folds,
                reduction_factor=2,
            )
        else:
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        sampler = optuna.samplers.TPESampler(seed=config.random_state)
        study = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler)

        cv = (
            StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
            if task_type == "classification"
            else KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
        )

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_score, best_family, trial_count
            trial_count += 1

            family_name = trial.suggest_categorical("model_family", list(families.keys()))
            pipeline = families[family_name](trial)

            fold_scores = []
            for step, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                X_tr, X_val = X_train[tr_idx], X_train[val_idx]
                y_tr, y_val = y_train[tr_idx], y_train[val_idx]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipeline.fit(X_tr, y_tr)
                score = _score_pipeline(pipeline, X_val, y_val, scoring)
                fold_scores.append(score)
                trial.report(np.mean(fold_scores), step)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            cv_score = float(np.mean(fold_scores))

            if cv_score > best_score:
                best_score = cv_score
                best_family = family_name

            if reporter:
                reporter.send(TrialProgress(
                    trial=trial_count,
                    total_trials=config.n_trials,
                    family=family_name,
                    score=round(cv_score, 4),
                    best_score=round(best_score, 4),
                    best_family=best_family,
                ))

            return cv_score

        study.optimize(
            objective,
            n_trials=config.n_trials,
            timeout=config.timeout_secs,
            show_progress_bar=False,
        )

        # ── 3. Retrain best on full training data ──────────────────────────
        best_trial = study.best_trial
        best_params = {k: v for k, v in best_trial.params.items() if k != "model_family"}
        best_fam = best_trial.params["model_family"]

        # Re-create pipeline with best params via a fresh trial replay
        best_pipeline = _replay_pipeline(families[best_fam], best_trial)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_pipeline.fit(X_train, y_train)

        # ── 4. Evaluate on holdout ─────────────────────────────────────────
        holdout_metrics = {}
        if len(X_holdout) > 0:
            holdout_metrics = _compute_holdout_metrics(
                best_pipeline, X_holdout, y_holdout, task_type, scoring
            )

        # ── 4b. Probability calibration (classification only) ─────────────────
        # Tree ensembles and SVMs are notoriously miscalibrated — a model that
        # says P(churn)=0.85 may only be correct 60% of the time at that threshold.
        # Isotonic regression (for ≥1000 samples) or Platt scaling (sigmoid, smaller
        # datasets) applies a post-hoc monotone transform so predicted probabilities
        # match empirical frequencies. This has no effect on ranking/AUC but
        # significantly improves calibration metrics like Brier score and log-loss.
        # CalibratedClassifierCV uses 5-fold CV internally to avoid leakage.
        calibration_applied = False
        calibration_method  = None
        if task_type == "classification" and len(X_train) >= 200:
            try:
                from sklearn.calibration import CalibratedClassifierCV
                method = "isotonic" if len(X_train) >= 1000 else "sigmoid"
                calibrated = CalibratedClassifierCV(best_pipeline, method=method, cv=5)
                calibrated.fit(X_train, y_train)
                # Re-evaluate on holdout with calibrated probabilities
                if len(X_holdout) > 0:
                    holdout_metrics = _compute_holdout_metrics(
                        calibrated, X_holdout, y_holdout, task_type, scoring
                    )
                best_pipeline       = calibrated
                calibration_applied = True
                calibration_method  = method
            except Exception:
                pass   # calibration is best-effort; fall back to uncalibrated model

        # ── 5. Save model artifact ─────────────────────────────────────────
        artifact_name = f"{experiment_name}_{uuid.uuid4().hex[:8]}.joblib"
        artifact_path = str(self.model_storage_path / artifact_name)
        joblib.dump(best_pipeline, artifact_path)

        # ── 6. Experiment tracking (provider-agnostic) ──────────────────────
        # Uses whichever backend settings.tracking_backend selects —
        # MLflow (default, self-hostable) or Weights & Biases (cloud SaaS).
        # See tracking/base.py for the abstraction and why it exists
        # alongside an already-working MLflow integration.
        from tracking.base import get_tracking_backend
        tracked = get_tracking_backend().log_run(
            run_name=experiment_name,
            params={"family": best_fam, **best_params},
            metrics={scoring: best_score, **holdout_metrics},
            artifact_path=artifact_path,
        )
        mlflow_run_id = tracked.run_id   # field name kept for DB/schema compatibility;
                                          # holds whichever provider's run_id was returned

        elapsed = time.monotonic() - t0
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

        result = TrainingResult(
            best_family=best_fam,
            best_params=best_params,
            best_cv_score=round(study.best_value, 4),
            scoring_metric=scoring,
            holdout_metrics=holdout_metrics,
            n_trials_completed=len(completed),
            n_trials_pruned=len(pruned),
            elapsed_secs=round(elapsed, 2),
            artifact_path=artifact_path,
            feature_names=feature_cols,
            task_type=task_type,
            mlflow_run_id=mlflow_run_id,
            calibration_applied=calibration_applied,
            calibration_method=calibration_method,
            pruner_type=config.pruner_type,
        )

        if reporter:
            reporter.send(TrainingComplete(
                best_family=best_fam,
                best_score=round(study.best_value, 4),
                scoring_metric=scoring,
                n_trials=len(completed),
                n_pruned=len(pruned),
                elapsed_secs=round(elapsed, 2),
                artifact_path=artifact_path,
            ))

        return result

    def _preprocess(
        self, df: pd.DataFrame, feature_cols: list[str], target_col: str
    ) -> pd.DataFrame:
        """
        Minimal preprocessing: drop rows where target is null, impute
        numeric features with median. Phase 2 can add a proper
        preprocessing pipeline configurator.
        """
        out = df.dropna(subset=[target_col]).copy()
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(out[col]):
                median = out[col].median()
                out[col] = out[col].fillna(median if pd.notna(median) else 0)
            else:
                # Drop non-numeric features — Phase 2 adds encoding options
                out = out.drop(columns=[col])
        return out


# ── Helpers ────────────────────────────────────────────────────────────────

def _default_scoring(task_type: str) -> str:
    return "roc_auc" if task_type == "classification" else "r2"


def _score_pipeline(pipeline, X_val, y_val, scoring: str) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if scoring == "roc_auc":
            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba(X_val)
                n_classes = proba.shape[1]
                if n_classes == 2:
                    return roc_auc_score(y_val, proba[:, 1])
                return roc_auc_score(y_val, proba, multi_class="ovr")
            return accuracy_score(y_val, pipeline.predict(X_val))
        elif scoring == "accuracy":
            return accuracy_score(y_val, pipeline.predict(X_val))
        elif scoring == "r2":
            return r2_score(y_val, pipeline.predict(X_val))
        elif scoring == "neg_rmse":
            pred = pipeline.predict(X_val)
            return -float(np.sqrt(mean_squared_error(y_val, pred)))
        return pipeline.score(X_val, y_val)


def _replay_pipeline(family_fn: Callable, trial: optuna.Trial):
    """Re-create a pipeline using stored param values from a completed trial."""
    stored_params = dict(trial.params)

    class ReplayTrial:
        def suggest_int(self, name, *args, **kwargs):
            return stored_params[name]
        def suggest_float(self, name, *args, **kwargs):
            return stored_params[name]
        def suggest_categorical(self, name, *args, **kwargs):
            return stored_params[name]

    return family_fn(ReplayTrial())


def _compute_holdout_metrics(
    pipeline, X_holdout, y_holdout, task_type: str, primary_scoring: str
) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = pipeline.predict(X_holdout)

    if task_type == "classification":
        metrics = {"holdout_accuracy": round(float(accuracy_score(y_holdout, pred)), 4)}
        try:
            metrics["holdout_f1"] = round(float(f1_score(y_holdout, pred, average="weighted", zero_division=0)), 4)
        except Exception:
            pass
        if hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba(X_holdout)
                metrics["holdout_roc_auc"] = round(float(
                    roc_auc_score(y_holdout, proba[:, 1] if proba.shape[1] == 2 else proba, multi_class="ovr")
                ), 4)
            except Exception:
                pass
    else:
        metrics = {
            "holdout_r2":   round(float(r2_score(y_holdout, pred)), 4),
            "holdout_mae":  round(float(mean_absolute_error(y_holdout, pred)), 4),
            "holdout_rmse": round(float(np.sqrt(mean_squared_error(y_holdout, pred))), 4),
        }
    return metrics


# Note: MLflow logging logic previously lived here as _log_to_mlflow().
# It now lives in tracking/mlflow_backend.py as part of the provider-agnostic
# TrackingBackend abstraction (see tracking/base.py) — the behaviour is
# identical, only the entry point moved.
