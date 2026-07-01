"""
Keras/TensorFlow model families for Crucible's AutoML pipeline.

These integrate with the existing Optuna search loop through SciKeras,
which wraps Keras models in sklearn-compatible KerasClassifier /
KerasRegressor wrappers. Once wrapped, a neural network looks identical
to a Random Forest from the TrainingRunner's perspective.

ARCHITECTURE — MULTILAYER PERCEPTRON (MLP)
-------------------------------------------
The MLP is the foundational deep learning architecture:

    Input → [Dense → BatchNorm → Dropout] × n_layers → Output

Each hidden layer:
  Dense(units)        — fully connected linear transformation
  BatchNormalization  — normalises layer outputs, stabilises training,
                        acts as a regulariser
  Dropout(rate)       — randomly zeroes rate% of neurons during training,
                        forces the network to learn redundant representations
                        rather than memorising specific neuron paths

WHY BATCHNORM + DROPOUT TOGETHER
---------------------------------
BatchNorm and Dropout have a complex interaction — BatchNorm reduces the
need for Dropout because it already provides regularisation. We keep both
but with a lower Dropout rate (0.1–0.3) when BatchNorm is active. This
is the modern standard rather than heavy Dropout alone (the old approach).

EARLY STOPPING
--------------
Neural networks are prone to overfitting if trained for too many epochs.
Early stopping monitors validation loss each epoch and stops training when
validation loss stops improving for `patience` consecutive epochs. This
automatically finds the right number of epochs without a grid search.

OPTUNA HYPERPARAMETER SPACE
-----------------------------
  n_layers:    1–4 hidden layers
  units_*:     32–512 units per layer (log-scale — powers of 2 are common)
  dropout:     0.0–0.4 dropout rate
  learning_rate: 1e-4 to 1e-2 (log-scale)
  batch_size:  32, 64, 128, 256
  activation:  relu, elu (both work well; elu has smoother gradients)

SCIKERAS INTEGRATION
--------------------
SciKeras provides KerasClassifier/KerasRegressor that:
  - Accept sklearn's fit(X, y) / predict(X) / predict_proba(X) interface
  - Handle train/validation splits internally for early stopping
  - Return self from fit() for compatibility with sklearn Pipelines
  - Serialise correctly with joblib (model weights + architecture)

OPTIONAL IMPORT
---------------
TensorFlow is large (~500MB). The entire module is wrapped so if TF is
not installed, KERAS_AVAILABLE = False and the calling code can skip
these families gracefully without crashing.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np

# Suppress TF startup messages — they are informational noise in a server
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks as keras_callbacks
    from scikeras.wrappers import KerasClassifier, KerasRegressor
    KERAS_AVAILABLE = True
    tf.get_logger().setLevel("ERROR")
except ImportError:
    KERAS_AVAILABLE = False


# ── MLP builder functions ──────────────────────────────────────────────────

def _build_mlp_classifier(
    n_layers: int,
    units_0: int,
    units_1: int,
    units_2: int,
    units_3: int,
    dropout: float,
    activation: str,
    meta: dict,          # SciKeras injects this — contains n_features_in_, n_classes_, etc.
    **kwargs,            # absorb compile_kwargs, params, and any future SciKeras additions
) -> "keras.Model":
    """
    Builds a Keras MLP for binary or multi-class classification.

    SciKeras 0.13 passes metadata via a 'meta' dict rather than as
    individual keyword arguments. n_features_in_ and n_classes_ live
    in meta and are only available after SciKeras has seen the training
    data (computed inside _initialize before _build_keras_model is called).

    The Optuna hyperparameters (n_layers, units_*, dropout, activation)
    are passed via KerasClassifier(model__<name>=value) and arrive as
    individual kwargs alongside meta.
    """
    n_features_in_ = meta["n_features_in_"]
    n_classes_ = meta.get("n_classes_", 2)
    unit_list = [units_0, units_1, units_2, units_3]

    model = keras.Sequential()
    model.add(layers.Input(shape=(n_features_in_,)))

    for i in range(n_layers):
        model.add(layers.Dense(unit_list[i], use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(activation))
        if dropout > 0:
            model.add(layers.Dropout(dropout))

    if n_classes_ == 2:
        model.add(layers.Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    else:
        model.add(layers.Dense(n_classes_, activation="softmax"))
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=loss,
        metrics=metrics,
    )
    return model


def _build_mlp_regressor(
    n_layers: int,
    units_0: int,
    units_1: int,
    units_2: int,
    units_3: int,
    dropout: float,
    activation: str,
    meta: dict,
    **kwargs,
) -> "keras.Model":
    """
    Builds a Keras MLP for regression. Uses meta['n_features_in_']
    for the input layer shape — see _build_mlp_classifier for details.
    """
    n_features_in_ = meta["n_features_in_"]
    unit_list = [units_0, units_1, units_2, units_3]

    model = keras.Sequential()
    model.add(layers.Input(shape=(n_features_in_,)))

    for i in range(n_layers):
        model.add(layers.Dense(unit_list[i], use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(activation))
        if dropout > 0:
            model.add(layers.Dropout(dropout))

    model.add(layers.Dense(1, activation="linear"))
    model.compile(optimizer=keras.optimizers.Adam(), loss="mse", metrics=["mae"])
    return model


# ── Optuna trial functions ─────────────────────────────────────────────────

def keras_mlp_classifier(trial) -> "KerasClassifier":
    """
    Returns a SciKeras KerasClassifier with Optuna-sampled hyperparameters.

    The model builder is passed as a callable so SciKeras can reconstruct
    the model after each fit (for cross-validation, each fold rebuilds).

    Early stopping monitors val_loss with patience=5 epochs.
    The validation split inside SciKeras is 10% of training data.
    """
    if not KERAS_AVAILABLE:
        raise ImportError("TensorFlow is required for Keras model families")

    n_layers      = trial.suggest_int("keras_clf_n_layers", 1, 4)
    units_0       = trial.suggest_int("keras_clf_units_0", 32, 512, log=True)
    units_1       = trial.suggest_int("keras_clf_units_1", 32, 512, log=True)
    units_2       = trial.suggest_int("keras_clf_units_2", 32, 256, log=True)
    units_3       = trial.suggest_int("keras_clf_units_3", 16, 128, log=True)
    dropout       = trial.suggest_float("keras_clf_dropout", 0.0, 0.4)
    learning_rate = trial.suggest_float("keras_clf_lr", 1e-4, 1e-2, log=True)
    batch_size    = trial.suggest_categorical("keras_clf_batch_size", [32, 64, 128, 256])
    activation    = trial.suggest_categorical("keras_clf_activation", ["relu", "elu"])

    early_stop = keras_callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=0,
    )

    return KerasClassifier(
        model=_build_mlp_classifier,
        model__n_layers=n_layers,
        model__units_0=units_0,
        model__units_1=units_1,
        model__units_2=units_2,
        model__units_3=units_3,
        model__dropout=dropout,
        model__activation=activation,
        optimizer=keras.optimizers.Adam,
        optimizer__learning_rate=learning_rate,
        loss="binary_crossentropy",
        batch_size=batch_size,
        epochs=100,               # early stopping will stop before this
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0,
    )


def keras_mlp_regressor(trial) -> "KerasRegressor":
    """
    Returns a SciKeras KerasRegressor with Optuna-sampled hyperparameters.
    """
    if not KERAS_AVAILABLE:
        raise ImportError("TensorFlow is required for Keras model families")

    n_layers      = trial.suggest_int("keras_reg_n_layers", 1, 4)
    units_0       = trial.suggest_int("keras_reg_units_0", 32, 512, log=True)
    units_1       = trial.suggest_int("keras_reg_units_1", 32, 512, log=True)
    units_2       = trial.suggest_int("keras_reg_units_2", 32, 256, log=True)
    units_3       = trial.suggest_int("keras_reg_units_3", 16, 128, log=True)
    dropout       = trial.suggest_float("keras_reg_dropout", 0.0, 0.4)
    learning_rate = trial.suggest_float("keras_reg_lr", 1e-4, 1e-2, log=True)
    batch_size    = trial.suggest_categorical("keras_reg_batch_size", [32, 64, 128, 256])
    activation    = trial.suggest_categorical("keras_reg_activation", ["relu", "elu"])

    early_stop = keras_callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=0,
    )

    return KerasRegressor(
        model=_build_mlp_regressor,
        model__n_layers=n_layers,
        model__units_0=units_0,
        model__units_1=units_1,
        model__units_2=units_2,
        model__units_3=units_3,
        model__dropout=dropout,
        model__activation=activation,
        optimizer=keras.optimizers.Adam,
        optimizer__learning_rate=learning_rate,
        loss="mse",
        batch_size=batch_size,
        epochs=100,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0,
    )
