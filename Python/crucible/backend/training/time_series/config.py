"""
Time series forecasting configuration.

FORECASTING VS CLASSIFICATION/REGRESSION
-----------------------------------------
The AutoML pipeline handles tabular tasks where each row is an independent
observation. Time series is fundamentally different: observations are
ordered in time and each row depends on its neighbours. This requires:

  1. A date/time column that defines the ordering.
  2. Temporal cross-validation — random splits would leak future data
     into the training set, producing optimistically biased metrics.
  3. Different evaluation metrics — MAPE, RMSE, MAE instead of accuracy/F1.
  4. A forecast horizon — how many periods ahead to predict.

SUPPORTED FAMILIES
-------------------
arima             AutoARIMA (pmdarima) — automatically selects ARIMA(p,d,q)
                  orders using AIC/BIC. The statistical baseline. Fast,
                  interpretable, excellent for linear trends and seasonality.

exp_smoothing     Holt-Winters Exponential Smoothing (statsmodels) — weighted
                  average of past observations with exponentially decaying
                  weights. Handles trend and multiplicative/additive seasonality.
                  Often beats ARIMA on short series.

prophet           Facebook Prophet — additive decomposition model with piecewise
                  linear/logistic trends, Fourier-series seasonality, and holiday
                  effects. Robust to missing data and outliers. Requires the
                  'prophet' package (optional — large download).

lstm              Keras LSTM — sequence model for complex non-linear patterns.
                  Uses sliding window input. Requires TensorFlow (optional).

FREQUENCY STRINGS
-----------------
  "D"   — daily
  "W"   — weekly
  "MS"  — monthly (month start)
  "QS"  — quarterly (quarter start)
  "YS"  — yearly (year start)
  "H"   — hourly
  "auto" — infer from median gap between consecutive dates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TimeSeriesConfig:
    """Configuration for a time series forecasting job."""

    # ── Required ──────────────────────────────────────────────────────────
    date_column: str            # Column containing timestamps
    target_column: str          # Column to forecast

    # ── Forecast settings ─────────────────────────────────────────────────
    horizon: int        = 12    # How many steps ahead to forecast
    frequency: str      = "auto"  # Pandas offset alias or "auto"

    # ── Search settings ───────────────────────────────────────────────────
    n_trials: int       = 20    # Optuna trials per family
    n_cv_splits: int    = 3     # Expanding-window cross-validation splits

    # ── Family filter ─────────────────────────────────────────────────────
    families: Optional[list[str]] = None  # None = all available

    def validate(self) -> list[str]:
        errors = []
        if not self.date_column:
            errors.append("date_column is required")
        if not self.target_column:
            errors.append("target_column is required")
        if self.horizon < 1:
            errors.append(f"horizon must be >= 1, got {self.horizon}")
        if self.n_trials < 1:
            errors.append(f"n_trials must be >= 1, got {self.n_trials}")
        if self.n_cv_splits < 1:
            errors.append(f"n_cv_splits must be >= 1, got {self.n_cv_splits}")
        return errors
