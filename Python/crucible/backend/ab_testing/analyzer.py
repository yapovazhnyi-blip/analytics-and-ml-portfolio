"""
A/B Test Analyzer — loads both model artifacts, reproduces the holdout split,
and delegates to the statistical engine.

WHY RELOAD MODELS INSTEAD OF COMPARING STORED METRICS
------------------------------------------------------
Comparing stored scalar metrics (accuracy_a vs accuracy_b) using a t-test
treats the holdout accuracy as a single observation with unknown variance.
You'd need many repeated holdout evaluations to estimate that variance.

McNemar's test is much more powerful because it uses the per-sample prediction
structure. Samples where both models agree provide no information — they don't
help us distinguish the models. Only samples where the models disagree matter.
This paired structure gives McNemar's test roughly 2× the power of an
unpaired chi-squared test of equal proportions.

The cost: we must reload both joblib artifacts and rerun model.predict().
For typical Crucible models (sklearn, XGBoost, LightGBM) this takes <1 second.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
import pandas as pd

from ab_testing.engine import compare_experiments, ABTestResult


class ABTestAnalyzer:
    """Loads two experiment artifacts and runs the A/B comparison."""

    TEST_SIZE    = 0.2
    RANDOM_STATE = 42

    async def analyze(
        self,
        exp_a_id: int,
        exp_b_id: int,
        artifact_path_a: str,
        artifact_path_b: str,
        dataset_path: str,
        source_type: str,
        target_column: str,
        task_type: str,
        confidence_level: float = 0.95,
    ) -> ABTestResult:
        """Async wrapper — CPU-bound work runs in a thread executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._analyze_sync,
            exp_a_id, exp_b_id,
            artifact_path_a, artifact_path_b,
            dataset_path, source_type,
            target_column, task_type,
            confidence_level,
        )

    def _analyze_sync(
        self,
        exp_a_id, exp_b_id,
        artifact_path_a, artifact_path_b,
        dataset_path, source_type,
        target_column, task_type,
        confidence_level,
    ) -> ABTestResult:
        """Blocking implementation."""
        try:
            import joblib
            from profiling.runner import ProfileRunner
            from sklearn.model_selection import train_test_split

            model_a = joblib.load(artifact_path_a)
            model_b = joblib.load(artifact_path_b)

            df = ProfileRunner.load_dataframe(dataset_path, source_type)
            df = df.dropna(subset=[target_column])

            feature_cols = [c for c in df.columns if c != target_column]
            X = df[feature_cols]
            y = df[target_column]

            # Encode to match TrainingRunner
            X = _encode(X)
            y = _encode_target(y, task_type)

            _, X_test, _, y_test = train_test_split(
                X, y,
                test_size=self.TEST_SIZE,
                random_state=self.RANDOM_STATE,
            )

            preds_a = model_a.predict(X_test)
            preds_b = model_b.predict(X_test)

            return compare_experiments(
                exp_a_id=exp_a_id,
                exp_b_id=exp_b_id,
                y_true=np.array(y_test),
                preds_a=np.array(preds_a),
                preds_b=np.array(preds_b),
                task_type=task_type,
                confidence_level=confidence_level,
            )

        except Exception as exc:
            # Return a minimal result with an error description
            from ab_testing.engine import ABTestResult
            return ABTestResult(
                experiment_a_id=exp_a_id, experiment_b_id=exp_b_id,
                metric="error", score_a=0.0, score_b=0.0,
                absolute_diff=0.0, relative_diff_pct=0.0,
                p_value=1.0, confidence_level=confidence_level,
                ci_lower=0.0, ci_upper=0.0,
                is_significant=False, winner=None, winner_id=None,
                effect_size=0.0, effect_size_label="negligible",
                statistical_test="error", n_samples=0,
                recommendation=f"Analysis failed: {exc}",
            )


def _encode(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype("category").cat.codes
    return X.fillna(0)

def _encode_target(y: pd.Series, task_type: str) -> pd.Series:
    if task_type == "classification" and y.dtype == object:
        return y.astype("category").cat.codes
    return pd.to_numeric(y, errors="coerce").fillna(0)
