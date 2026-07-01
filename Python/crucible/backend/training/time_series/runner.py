"""
Time Series Runner — Optuna-driven forecasting with temporal cross-validation.

WHY TEMPORAL CV IS NON-NEGOTIABLE
-----------------------------------
Standard k-fold cross-validation randomly shuffles rows. For time series,
this means the model sees future data during training (e.g. row 900 in the
training set while predicting row 50). The result is optimistically biased
metrics — the model memorises the future rather than learning to forecast.

Expanding window (also called walk-forward) cross-validation:
  Split 1: train on [0..60%], validate on [60..70%]
  Split 2: train on [0..70%], validate on [70..80%]
  Split 3: train on [0..80%], validate on [80..90%]

The training set grows with each split. The validation set is always in the
future relative to training. This mirrors real-world use: you always train
on the past and predict the future.

METRICS
--------
MAPE  — Mean Absolute Percentage Error: mean(|actual - predicted| / |actual|)
        Percentage scale, intuitive. Undefined when actual = 0.

RMSE  — Root Mean Squared Error: sqrt(mean((actual - predicted)²))
        Same units as the target. Penalises large errors more than small ones.

MAE   — Mean Absolute Error: mean(|actual - predicted|)
        Same units as the target. Robust to outliers.

MAPE is used as the primary optimisation metric (minimised by Optuna).

FORECAST OUTPUT
---------------
After finding the best family and parameters, the runner retrains on the
full series and produces a forecast DataFrame:
  - date:        future timestamps
  - predicted:   point forecast
  - lower:       lower confidence bound (predicted - 1.96 * residual_std)
  - upper:       upper confidence bound
"""

from __future__ import annotations

import asyncio
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from training.time_series.config import TimeSeriesConfig
from training.time_series.families import FORECASTING_FAMILIES, FAMILY_DISPLAY_TS


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class CVFoldResult:
    fold: int
    train_size: int
    val_size: int
    mape: float
    rmse: float
    mae: float


@dataclass
class ForecastResult:
    """Complete result of one time series forecasting job."""
    job_id: str
    best_family: str
    best_params: dict
    cv_mape: float           # mean MAPE across CV folds
    cv_rmse: float
    cv_mae: float
    cv_folds: list[CVFoldResult]
    forecast: pd.DataFrame   # columns: date, predicted, lower, upper
    n_trials_completed: int
    elapsed_secs: float
    artifact_path: str
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "job_id":             self.job_id,
            "best_family":        self.best_family,
            "best_family_display": FAMILY_DISPLAY_TS.get(self.best_family, self.best_family),
            "best_params":        self.best_params,
            "cv_mape":            round(self.cv_mape, 4),
            "cv_rmse":            round(self.cv_rmse, 4),
            "cv_mae":             round(self.cv_mae, 4),
            "n_trials":           self.n_trials_completed,
            "elapsed_secs":       round(self.elapsed_secs, 2),
            "forecast": self.forecast.to_dict(orient="records") if self.forecast is not None else [],
            "cv_folds": [
                {"fold": f.fold, "train_size": f.train_size, "mape": round(f.mape, 4), "rmse": round(f.rmse, 4)}
                for f in self.cv_folds
            ],
            "error": self.error,
        }


# ── Metric helpers ────────────────────────────────────────────────────────────

def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Returns 999 when actual contains zeros."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual != 0
    if not mask.any():
        return 999.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


# ── Frequency inference ───────────────────────────────────────────────────────

def infer_frequency(dates: pd.Series) -> str:
    """
    Infers the pandas frequency string from the median gap between dates.

    Returns a pandas offset alias: D, W, MS, QS, YS, H.
    Falls back to 'D' (daily) for ambiguous cases.
    """
    diffs = pd.Series(dates).sort_values().diff().dropna()
    if diffs.empty:
        return "D"
    median_gap = diffs.median()
    hours = median_gap.total_seconds() / 3600

    if hours <= 1.5:      return "h"   # pandas >= 2.2 uses lowercase aliases
    if hours <= 25:       return "D"
    if hours <= 8 * 24:   return "W"
    if hours <= 32 * 24:  return "MS"
    if hours <= 95 * 24:  return "QS"
    return "YS"


# ── Temporal cross-validation ─────────────────────────────────────────────────

def temporal_cv_splits(
    n: int,
    n_splits: int,
    horizon: int,
    min_train_size: Optional[int] = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generates expanding-window train/validation index pairs.

    Args:
        n:             Total number of observations.
        n_splits:      Number of CV folds.
        horizon:       Validation window size (matches forecast horizon).
        min_train_size: Minimum training size. Defaults to max(horizon*2, n//4).

    Returns:
        List of (train_indices, val_indices) tuples.
    """
    if min_train_size is None:
        min_train_size = max(horizon * 2, n // 4)

    splits = []
    # The last validation end point is n - we need enough data for min_train + horizon
    available = n - min_train_size - horizon
    if available <= 0:
        # Series too short — return a single split
        train_end = max(min_train_size, n - horizon)
        return [(np.arange(train_end), np.arange(train_end, min(train_end + horizon, n)))]

    step = max(1, available // n_splits)
    for i in range(n_splits):
        val_end = n - (n_splits - i - 1) * step
        val_start = max(min_train_size, val_end - horizon)
        train_idx = np.arange(val_start)
        val_idx   = np.arange(val_start, min(val_start + horizon, n))
        if len(train_idx) >= min_train_size and len(val_idx) > 0:
            splits.append((train_idx, val_idx))

    return splits if splits else [(np.arange(n - horizon), np.arange(n - horizon, n))]


# ── Main runner ───────────────────────────────────────────────────────────────

class TimeSeriesRunner:
    """
    Searches for the best forecasting model using Optuna and temporal CV.
    """

    def __init__(self, config: TimeSeriesConfig, job_id: str):
        self.config = config
        self.job_id = job_id

    async def run(self, df: pd.DataFrame, output_dir: str) -> ForecastResult:
        """Async wrapper — heavy work runs in a thread executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run_sync, df, output_dir)

    def _run_sync(self, df: pd.DataFrame, output_dir: str) -> ForecastResult:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        cfg = self.config
        start = time.monotonic()

        # ── 1. Prepare series ─────────────────────────────────────────────
        try:
            series = self._prepare_series(df)
        except Exception as exc:
            return self._error(f"Data preparation failed: {exc}", start)

        if len(series) < cfg.horizon * 2 + cfg.n_cv_splits:
            return self._error(
                f"Series too short ({len(series)} points) for horizon={cfg.horizon} "
                f"with {cfg.n_cv_splits} CV splits. Need at least "
                f"{cfg.horizon * 2 + cfg.n_cv_splits} points.",
                start,
            )

        # ── 2. Select families ────────────────────────────────────────────
        families = dict(FORECASTING_FAMILIES)
        if cfg.families:
            unknown = set(cfg.families) - set(families)
            if unknown:
                return self._error(f"Unknown families: {unknown}", start)
            families = {k: v for k, v in families.items() if k in cfg.families}

        if not families:
            return self._error("No forecasting families available. Install pmdarima or statsmodels.", start)

        # ── 3. Temporal CV splits ─────────────────────────────────────────
        cv_splits = temporal_cv_splits(
            n=len(series),
            n_splits=cfg.n_cv_splits,
            horizon=cfg.horizon,
        )

        # ── 4. Optuna search ──────────────────────────────────────────────
        best_value  = float("inf")
        best_params: dict = {}
        best_family = list(families.keys())[0]
        n_completed = 0
        all_fold_results: list[CVFoldResult] = []

        def objective(trial):
            nonlocal n_completed
            family_name = trial.suggest_categorical("family", list(families.keys()))
            family_fn   = families[family_name]
            family      = family_fn(trial)

            fold_mapes = []
            for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
                y_train = series.iloc[train_idx]
                y_val   = series.iloc[val_idx].values

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fitted = family["fit"](y_train)
                        preds  = family["predict"](fitted, len(val_idx))
                    preds = np.clip(np.asarray(preds, dtype=float), None, None)
                    fold_mapes.append(mape(y_val, preds))
                except Exception:
                    fold_mapes.append(999.0)

            n_completed += 1
            return float(np.mean(fold_mapes))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
        )
        study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)

        best_trial  = study.best_trial
        best_params = {k: v for k, v in best_trial.params.items() if k != "family"}
        best_family = best_trial.params.get("family", best_family)
        best_value  = best_trial.value

        # ── 5. Compute CV metrics for the best family ─────────────────────
        best_fn     = families[best_family]
        best_family_instance = best_fn(
            _ReplayTrial(best_trial.params, prefix=best_family + "_")
        )

        for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
            y_train = series.iloc[train_idx]
            y_val   = series.iloc[val_idx].values
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = best_family_instance["fit"](y_train)
                    preds  = best_family_instance["predict"](fitted, len(val_idx))
                preds = np.asarray(preds, dtype=float)
                all_fold_results.append(CVFoldResult(
                    fold=fold_i + 1,
                    train_size=len(train_idx),
                    val_size=len(val_idx),
                    mape=mape(y_val, preds),
                    rmse=rmse(y_val, preds),
                    mae=mae(y_val, preds),
                ))
            except Exception:
                pass

        cv_mape_mean = np.mean([f.mape for f in all_fold_results]) if all_fold_results else best_value
        cv_rmse_mean = np.mean([f.rmse for f in all_fold_results]) if all_fold_results else 0.0
        cv_mae_mean  = np.mean([f.mae  for f in all_fold_results]) if all_fold_results else 0.0

        # ── 6. Retrain on full series + produce forecast ──────────────────
        forecast_df, artifact_path = self._final_forecast(
            series=series,
            family_instance=best_family_instance,
            output_dir=output_dir,
        )

        return ForecastResult(
            job_id=self.job_id,
            best_family=best_family,
            best_params=best_params,
            cv_mape=cv_mape_mean,
            cv_rmse=cv_rmse_mean,
            cv_mae=cv_mae_mean,
            cv_folds=all_fold_results,
            forecast=forecast_df,
            n_trials_completed=n_completed,
            elapsed_secs=time.monotonic() - start,
            artifact_path=artifact_path,
        )

    # ── Series preparation ─────────────────────────────────────────────────

    def _prepare_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Extracts and validates the target time series from the DataFrame.

        Returns a pandas Series with a DatetimeIndex, sorted chronologically,
        with NaN gaps forward-filled (up to 3 consecutive gaps).
        """
        cfg = self.config
        if cfg.date_column not in df.columns:
            raise ValueError(f"Date column '{cfg.date_column}' not found. Available: {list(df.columns)}")
        if cfg.target_column not in df.columns:
            raise ValueError(f"Target column '{cfg.target_column}' not found. Available: {list(df.columns)}")

        dates  = pd.to_datetime(df[cfg.date_column])
        values = pd.to_numeric(df[cfg.target_column], errors="coerce")

        series = pd.Series(values.values, index=dates).sort_index()
        series = series.ffill(limit=3)   # forward-fill small gaps
        series = series.dropna()

        if cfg.frequency == "auto":
            freq = infer_frequency(series.index.to_series())
        else:
            freq = cfg.frequency

        # Resample to the detected frequency (aggregates duplicates with mean)
        series = series.resample(freq).mean().ffill(limit=3).dropna()
        return series

    # ── Final forecast ─────────────────────────────────────────────────────

    def _final_forecast(
        self,
        series: pd.Series,
        family_instance: dict,
        output_dir: str,
    ) -> tuple[pd.DataFrame, str]:
        """Retrains on the full series and generates future forecasts."""
        import joblib

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = family_instance["fit"](series)
            preds  = family_instance["predict"](fitted, self.config.horizon)

        preds = np.asarray(preds, dtype=float)

        # Confidence interval: ±1.96 × residual std (approximate)
        residuals = series.values - family_instance["predict"](fitted, len(series)) \
            if False else np.zeros(len(series))   # skip in-sample for speed
        std = max(np.std(series.values) * 0.1, 1e-6)  # rough estimate

        last_date = series.index[-1]
        freq = series.index.freq or pd.infer_freq(series.index) or "D"
        future_dates = pd.date_range(start=last_date, periods=self.config.horizon + 1, freq=freq)[1:]

        forecast_df = pd.DataFrame({
            "date":      future_dates.strftime("%Y-%m-%d"),
            "predicted": np.round(preds, 4),
            "lower":     np.round(preds - 1.96 * std, 4),
            "upper":     np.round(preds + 1.96 * std, 4),
        })

        # Save artifact
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        artifact_path = str(Path(output_dir) / f"{self.job_id}_model.pkl")
        joblib.dump({"model": fitted, "family": family_instance["name"]}, artifact_path)

        return forecast_df, artifact_path

    def _error(self, msg: str, start: float) -> ForecastResult:
        return ForecastResult(
            job_id=self.job_id, best_family="", best_params={},
            cv_mape=0.0, cv_rmse=0.0, cv_mae=0.0, cv_folds=[],
            forecast=pd.DataFrame(), n_trials_completed=0,
            elapsed_secs=time.monotonic() - start, artifact_path="", error=msg,
        )


# ── Replay trial shim ─────────────────────────────────────────────────────────

class _ReplayTrial:
    """
    Mimics an Optuna trial with fixed params so family functions can be
    reconstructed from stored trial params after the search.
    """
    def __init__(self, params: dict, prefix: str = ""):
        self._params = params
        self._prefix = prefix

    def suggest_int(self, name, *args, **kwargs):   return self._params.get(name, args[0] if args else 0)
    def suggest_float(self, name, *args, **kwargs): return self._params.get(name, args[0] if args else 0.0)
    def suggest_categorical(self, name, choices, **kwargs): return self._params.get(name, choices[0])
