"""
Fairness Analyzer — reproduces the holdout split, makes predictions,
and delegates to the metric functions.

WHY REPRODUCE THE SPLIT INSTEAD OF STORING PREDICTIONS
-------------------------------------------------------
Storing 100,000 predictions alongside every experiment would add significant
storage overhead and a DB schema change. Since the random_state is fixed
(42), the train/test split is deterministic — given the same dataset and
the same split parameters, we always get the same holdout set.

This means we can reproduce the exact holdout predictions any time by:
  1. Loading the full dataset
  2. Reproducing the split (same random_state, test_size)
  3. Running model.predict() on the test portion

The result is identical to what was computed during training.

HOW PROTECTED ATTRIBUTES ARE HANDLED
--------------------------------------
Protected attributes (gender, age_group, race, etc.) must be columns in
the same dataset used for training. They do NOT need to be feature columns
— the model may not even have access to them during training. The fairness
analysis loads the full dataset, extracts the protected attribute values for
the holdout rows, and compares model predictions across groups.

This matches real-world practice: credit models often exclude race by law
but are still analysed for racial disparate impact using the full customer
records that include race.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
import pandas as pd

from fairness.metrics import compute_fairness, FairnessReport


class FairnessAnalyzer:
    """
    Loads a trained Crucible model artifact and computes fairness metrics
    on the holdout set for specified protected attributes.
    """

    TEST_SIZE    = 0.2    # must match TrainingRunner's default
    RANDOM_STATE = 42     # must match TrainingRunner's default

    async def analyze(
        self,
        artifact_path: str,
        dataset_path: str,
        source_type: str,
        target_column: str,
        protected_attributes: list[str],
        task_type: str,
        experiment_id: int,
        positive_class: int = 1,
    ) -> FairnessReport:
        """
        Async wrapper — CPU-bound work runs in a thread executor.

        Args:
            artifact_path:         Path to the joblib model file.
            dataset_path:          Path to the dataset file.
            source_type:           "csv" | "parquet" | "sql" (for loader selection).
            target_column:         Column the model predicts.
            protected_attributes:  Columns to evaluate fairness across.
            task_type:             "classification" or "regression".
            experiment_id:         For labelling the report.
            positive_class:        Which label is the positive class.

        Returns:
            FairnessReport with metrics or an error description.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._analyze_sync,
            artifact_path,
            dataset_path,
            source_type,
            target_column,
            protected_attributes,
            task_type,
            experiment_id,
            positive_class,
        )

    def _analyze_sync(
        self,
        artifact_path: str,
        dataset_path: str,
        source_type: str,
        target_column: str,
        protected_attributes: list[str],
        task_type: str,
        experiment_id: int,
        positive_class: int,
    ) -> FairnessReport:
        """Blocking implementation — runs in thread pool."""
        try:
            # ── 1. Load model ─────────────────────────────────────────────
            import joblib
            model = joblib.load(artifact_path)

            # ── 2. Load dataset ───────────────────────────────────────────
            from profiling.runner import ProfileRunner
            df = ProfileRunner.load_dataframe(dataset_path, source_type)

            # ── 3. Validate columns ───────────────────────────────────────
            missing_protected = [c for c in protected_attributes if c not in df.columns]
            if missing_protected:
                return _error(
                    experiment_id=experiment_id,
                    task_type=task_type,
                    msg=f"Protected attribute columns not found in dataset: {missing_protected}. "
                        f"Available columns: {list(df.columns)}",
                )

            if target_column not in df.columns:
                return _error(experiment_id, task_type,
                              f"Target column '{target_column}' not found in dataset.")

            # ── 4. Reproduce the holdout split ────────────────────────────
            from sklearn.model_selection import train_test_split

            # Drop rows with missing target
            df = df.dropna(subset=[target_column])

            # Separate feature columns from protected/target columns
            # Protected attributes are NOT removed from features — the model
            # may or may not have used them during training.
            feature_cols = [
                c for c in df.columns
                if c != target_column
            ]

            X = df[feature_cols]
            y = df[target_column]

            # Encode categoricals the same way as TrainingRunner
            X = _encode_categoricals(X)
            y_encoded = _encode_target(y, task_type)

            # Reproduce the exact same split
            _, X_test, _, y_test, _, idx_test = train_test_split(
                X, y_encoded, df.index,
                test_size=self.TEST_SIZE,
                random_state=self.RANDOM_STATE,
            )

            # ── 5. Generate predictions on the holdout set ────────────────
            y_pred = model.predict(X_test)

            # ── 6. Extract protected attribute values for holdout rows ────
            protected_df = df.loc[idx_test, protected_attributes].copy()

            # ── 7. Compute fairness metrics ───────────────────────────────
            return compute_fairness(
                experiment_id=experiment_id,
                y_true=np.array(y_test),
                y_pred=np.array(y_pred),
                protected_df=protected_df,
                task_type=task_type,
                positive_class=positive_class,
            )

        except Exception as exc:
            return _error(experiment_id, task_type, str(exc))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """
    Simple label-encoding for categorical columns.
    Must match the encoding used in TrainingRunner to ensure the model
    receives the same feature representation.
    """
    X = X.copy()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype("category").cat.codes
    return X.fillna(0)


def _encode_target(y: pd.Series, task_type: str) -> pd.Series:
    """Encodes the target in the same way TrainingRunner does."""
    if task_type == "classification" and y.dtype == object:
        return y.astype("category").cat.codes
    return pd.to_numeric(y, errors="coerce").fillna(0)


def _error(experiment_id: int, task_type: str, msg: str) -> FairnessReport:
    from fairness.metrics import FairnessReport
    return FairnessReport(
        experiment_id=experiment_id,
        task_type=task_type,
        n_samples=0,
        protected_attributes=[],
        error=msg,
    )
