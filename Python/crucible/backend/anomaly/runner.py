"""
Anomaly detection runner — orchestrates multiple algorithms and aggregates results.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from anomaly.algorithms import (
    run_isolation_forest, run_lof, run_ocsvm,
    prepare_features, AnomalyFamilyResult,
)


@dataclass
class AnomalyReport:
    dataset_id: int
    n_rows: int
    n_features: int
    feature_names: list[str]
    contamination: float
    algorithms: list[dict]                    # per-algorithm results
    consensus_labels: list[int]               # 1=anomaly if majority vote
    consensus_n_anomalies: int
    top_anomalies: list[dict]                 # top-N most anomalous rows
    score_distribution: dict                  # percentile summary of scores
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "dataset_id":            self.dataset_id,
            "n_rows":                self.n_rows,
            "n_features":            self.n_features,
            "feature_names":         self.feature_names,
            "contamination":         self.contamination,
            "algorithms":            self.algorithms,
            "consensus_n_anomalies": self.consensus_n_anomalies,
            "consensus_anomaly_rate": round(
                self.consensus_n_anomalies / max(self.n_rows, 1), 4
            ),
            "top_anomalies":         self.top_anomalies,
            "score_distribution":    self.score_distribution,
            "error":                 self.error,
        }


class AnomalyRunner:
    """
    Runs multiple anomaly detection algorithms on a dataset and
    aggregates results via majority vote.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        algorithms: list[str] = None,
        top_n: int = 20,
    ):
        self.contamination = contamination
        self.algorithms = algorithms or ["isolation_forest", "lof"]
        self.top_n = top_n

    async def run(
        self,
        df: pd.DataFrame,
        dataset_id: int,
        exclude_columns: list[str] = None,
    ) -> AnomalyReport:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._run_sync, df, dataset_id, exclude_columns
        )

    def _run_sync(
        self,
        df: pd.DataFrame,
        dataset_id: int,
        exclude_columns: list[str] = None,
    ) -> AnomalyReport:
        try:
            X, feature_names = prepare_features(df, exclude_columns)
        except ValueError as exc:
            return AnomalyReport(
                dataset_id=dataset_id, n_rows=len(df), n_features=0,
                feature_names=[], contamination=self.contamination,
                algorithms=[], consensus_labels=[], consensus_n_anomalies=0,
                top_anomalies=[], score_distribution={}, error=str(exc),
            )

        results: list[AnomalyFamilyResult] = []

        if "isolation_forest" in self.algorithms:
            results.append(run_isolation_forest(X, self.contamination))
        if "lof" in self.algorithms:
            results.append(run_lof(X, self.contamination))
        if "ocsvm" in self.algorithms:
            results.append(run_ocsvm(X, self.contamination))

        # Only use successful results for consensus
        good = [r for r in results if r.succeeded and len(r.labels) == len(X)]

        if not good:
            return AnomalyReport(
                dataset_id=dataset_id, n_rows=len(df), n_features=len(feature_names),
                feature_names=feature_names, contamination=self.contamination,
                algorithms=[self._result_dict(r) for r in results],
                consensus_labels=[], consensus_n_anomalies=0,
                top_anomalies=[], score_distribution={},
                error="All algorithms failed. " + "; ".join(r.error for r in results if r.error),
            )

        # Majority vote consensus
        vote_matrix = np.stack([r.labels for r in good], axis=1)
        consensus = (vote_matrix.mean(axis=1) >= 0.5).astype(int)

        # Average score across successful algorithms
        score_matrix = np.stack([r.scores for r in good], axis=1)
        avg_scores   = score_matrix.mean(axis=1)

        # Top-N most anomalous
        top_idx = np.argsort(avg_scores)[::-1][:self.top_n]
        top_rows = []
        for idx in top_idx:
            row = {"row_index": int(idx), "anomaly_score": round(float(avg_scores[idx]), 4)}
            for col in feature_names[:5]:          # include first 5 features for context
                col_idx = feature_names.index(col)
                if col in df.columns:
                    row[col] = float(df[col].iloc[idx]) if pd.notna(df[col].iloc[idx]) else None
            top_rows.append(row)

        # Score distribution percentiles
        score_dist = {
            str(p): round(float(np.percentile(avg_scores, p)), 4)
            for p in [50, 75, 90, 95, 99]
        }

        return AnomalyReport(
            dataset_id=dataset_id,
            n_rows=len(df),
            n_features=len(feature_names),
            feature_names=feature_names,
            contamination=self.contamination,
            algorithms=[self._result_dict(r) for r in results],
            consensus_labels=consensus.tolist(),
            consensus_n_anomalies=int(consensus.sum()),
            top_anomalies=top_rows,
            score_distribution=score_dist,
        )

    @staticmethod
    def _result_dict(r: AnomalyFamilyResult) -> dict:
        return {
            "family":             r.family,
            "n_anomalies":        r.n_anomalies,
            "contamination_used": r.contamination_used,
            "threshold_score":    round(r.threshold_score, 4),
            "params":             r.params,
            "error":              r.error,
            "succeeded":          r.succeeded,
        }
