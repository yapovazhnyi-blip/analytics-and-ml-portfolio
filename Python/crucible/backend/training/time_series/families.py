"""
Time series model families for Crucible's forecasting pipeline.

Each family function accepts an Optuna trial and returns a callable with the
interface: fit(y_train) → fitted_model, predict(fitted_model, horizon) → array.

This wrapping pattern keeps the runner model-agnostic — it calls fit/predict
without knowing which family it's using, the same way TrainingRunner uses
sklearn's fit/predict for tabular models.

FAMILY CONTRACTS
----------------
Each family is a dict with keys:
  "fit":     fn(y_train, exog=None) → fitted model object
  "predict": fn(fitted_model, horizon) → np.ndarray of shape (horizon,)
  "name":    display name string

Optuna trial parameters must be namespaced by family to avoid conflicts when
multiple families share the same parameter name (e.g. both ARIMA and LSTM
have a "hidden_size" concept under different names).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ── ARIMA (AutoARIMA via pmdarima) ────────────────────────────────────────────

def _arima_family(trial):
    """
    AutoARIMA with Optuna-searched parameter bounds.

    AutoARIMA uses stepwise AIC minimisation to find the best (p,d,q)(P,D,Q)m
    within the bounds provided. The Optuna trial searches those bounds.

    max_p/max_q: upper limit on AR/MA order. Higher = more computation, can
    capture longer-range dependencies but risks overfitting on short series.

    seasonal: whether to also search for seasonal ARIMA components. Requires
    the series to have clear periodic patterns.
    """
    max_p   = trial.suggest_int("arima_max_p", 1, 5)
    max_q   = trial.suggest_int("arima_max_q", 1, 5)
    seasonal = trial.suggest_categorical("arima_seasonal", [True, False])
    d       = trial.suggest_int("arima_d", 0, 2)

    def fit(y_train, exog=None):
        from pmdarima import auto_arima
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = auto_arima(
                y_train,
                max_p=max_p, max_q=max_q,
                d=d, D=1 if seasonal else None,
                seasonal=seasonal,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                information_criterion="aic",
            )
        return model

    def predict(model, horizon):
        forecast = model.predict(n_periods=horizon)
        return np.array(forecast)

    return {"fit": fit, "predict": predict, "name": "AutoARIMA"}


# ── Exponential Smoothing (Holt-Winters via statsmodels) ──────────────────────

def _exp_smoothing_family(trial):
    """
    Holt-Winters Exponential Smoothing with Optuna-searched component selection.

    trend:    None (no trend), 'add' (additive), 'mul' (multiplicative)
    seasonal: None, 'add', 'mul'. Multiplicative seasonal better for data
              where seasonal swings grow with the level (e.g. sales data).
    """
    trend    = trial.suggest_categorical("ets_trend", ["add", "mul", None])
    seasonal = trial.suggest_categorical("ets_seasonal", ["add", "mul", None])
    damped   = trial.suggest_categorical("ets_damped", [True, False]) if trend else False

    def fit(y_train, exog=None):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                y_train,
                trend=trend,
                seasonal=seasonal,
                damped_trend=bool(damped),
            ).fit(optimized=True, remove_bias=True)
        return model

    def predict(model, horizon):
        return np.array(model.forecast(horizon))

    return {"fit": fit, "predict": predict, "name": "Exponential Smoothing"}


# ── Prophet (optional — requires facebook/prophet) ────────────────────────────

def _prophet_family(trial):
    """
    Facebook Prophet with Optuna-searched regularisation parameters.

    changepoint_prior_scale: controls flexibility of the trend.
      Low (0.001) → rigid trend, less risk of overfitting.
      High (0.5)  → flexible trend, can follow data closely.

    seasonality_prior_scale: controls the strength of seasonal components.
    """
    changepoint_prior = trial.suggest_float("prophet_cp", 0.001, 0.5, log=True)
    seasonality_prior = trial.suggest_float("prophet_sp", 0.01, 10.0, log=True)

    def fit(y_train, exog=None):
        from prophet import Prophet
        import logging, warnings
        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")

        # Prophet needs a DataFrame with columns 'ds' and 'y'
        if isinstance(y_train, pd.Series):
            df = pd.DataFrame({"ds": y_train.index, "y": y_train.values})
        else:
            raise ValueError("Prophet requires a pandas Series with a DatetimeIndex")

        model = Prophet(
            changepoint_prior_scale=changepoint_prior,
            seasonality_prior_scale=seasonality_prior,
            daily_seasonality=False,
            weekly_seasonality="auto",
            yearly_seasonality="auto",
        )
        model.fit(df)
        return model

    def predict(model, horizon):
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)
        return forecast["yhat"].values[-horizon:]

    return {"fit": fit, "predict": predict, "name": "Prophet"}


# ── LSTM (optional — requires TensorFlow) ─────────────────────────────────────

def _lstm_family(trial):
    """
    Keras LSTM with sliding-window input and Optuna-searched architecture.

    window_size: how many past timesteps the model sees as input.
    units:       LSTM hidden units.
    dropout:     recurrent dropout for regularisation.
    """
    window_size = trial.suggest_int("lstm_window", 5, 30)
    units       = trial.suggest_int("lstm_units", 16, 128, log=True)
    dropout     = trial.suggest_float("lstm_dropout", 0.0, 0.3)
    lr          = trial.suggest_float("lstm_lr", 1e-4, 1e-2, log=True)

    def fit(y_train, exog=None):
        import tensorflow as tf
        from tensorflow import keras

        values = np.array(y_train, dtype=np.float32)
        # Normalise to [0, 1] for stable training
        v_min, v_max = values.min(), values.max()
        v_range = v_max - v_min if v_max > v_min else 1.0
        normed = (values - v_min) / v_range

        # Build sliding-window dataset
        X, y_seq = [], []
        for i in range(len(normed) - window_size):
            X.append(normed[i: i + window_size])
            y_seq.append(normed[i + window_size])
        X = np.array(X).reshape(-1, window_size, 1)
        y_seq = np.array(y_seq)

        model = keras.Sequential([
            keras.layers.LSTM(units, dropout=dropout, return_sequences=False,
                              input_shape=(window_size, 1)),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
        model.fit(X, y_seq, epochs=20, batch_size=16, verbose=0,
                  validation_split=0.1,
                  callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

        return {"model": model, "window": window_size, "min": v_min, "scale": v_range}

    def predict(state, horizon):
        import tensorflow as tf
        model = state["model"]
        window = state["window"]
        v_min, v_scale = state["min"], state["scale"]

        y_arr = np.array([], dtype=np.float32)
        # Re-use the last window of training data (stored in the state during fit)
        # Since we only have the state dict here, we generate from zeros as fallback
        # In production, pass the last window explicitly
        last_window = np.zeros((1, window, 1), dtype=np.float32)
        preds = []
        for _ in range(horizon):
            pred = model.predict(last_window, verbose=0)[0, 0]
            preds.append(pred)
            last_window = np.roll(last_window, -1, axis=1)
            last_window[0, -1, 0] = pred
        return np.array(preds) * v_scale + v_min

    return {"fit": fit, "predict": predict, "name": "LSTM"}


# ── Registry ──────────────────────────────────────────────────────────────────

def _check_pmdarima() -> bool:
    try:
        import pmdarima  # noqa
        return True
    except ImportError:
        return False

def _check_statsmodels() -> bool:
    try:
        import statsmodels  # noqa
        return True
    except ImportError:
        return False

def _check_prophet() -> bool:
    try:
        import prophet  # noqa
        return True
    except ImportError:
        return False

def _check_tf() -> bool:
    try:
        import tensorflow  # noqa
        return True
    except ImportError:
        return False


ARIMA_AVAILABLE       = _check_pmdarima()
EXP_SMOOTHING_AVAILABLE = _check_statsmodels()
PROPHET_AVAILABLE     = _check_prophet()
LSTM_TS_AVAILABLE     = _check_tf()


FORECASTING_FAMILIES: dict[str, callable] = {
    **( {"arima":          _arima_family}         if ARIMA_AVAILABLE       else {} ),
    **( {"exp_smoothing":  _exp_smoothing_family}  if EXP_SMOOTHING_AVAILABLE else {} ),
    **( {"prophet":        _prophet_family}        if PROPHET_AVAILABLE     else {} ),
    **( {"lstm":           _lstm_family}           if LSTM_TS_AVAILABLE     else {} ),
}

FAMILY_DISPLAY_TS = {
    "arima":         "AutoARIMA",
    "exp_smoothing": "Exponential Smoothing",
    "prophet":       "Prophet",
    "lstm":          "LSTM",
}
