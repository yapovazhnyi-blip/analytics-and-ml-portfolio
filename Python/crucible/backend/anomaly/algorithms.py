"""
Anomaly Detection — unsupervised algorithms for finding unusual data points.

WHEN TO USE ANOMALY DETECTION
-------------------------------
Anomaly detection is appropriate when:
  - No labelled "anomaly" column exists (unsupervised)
  - The task is to flag unusual rows for review, not classify them
  - The proportion of anomalies is small and unknown (< 10%)

Examples: fraud transaction detection, equipment failure prediction,
data quality auditing, network intrusion detection.

THREE ALGORITHMS
----------------

Isolation Forest (default, recommended)
  Builds an ensemble of random trees. Anomalous points are isolated in
  fewer splits because they are rare and different from the majority.
  Anomaly score = average path length across trees (shorter = more anomalous).
  Time complexity: O(n log n). Works well on high-dimensional data.
  Key hyperparameter: contamination — expected fraction of anomalies (default 0.05)

Local Outlier Factor (LOF)
  Compares the local density of each point to its neighbours. A point is
  an outlier if its density is much lower than its neighbours' densities.
  Score = ratio of average local density of neighbours to point's density.
  Works well for datasets with varying density clusters.
  Downside: no predict() on new data (transductive, not inductive).
  Key hyperparameter: n_neighbors (default 20)

One-Class SVM
  Learns a decision boundary that encloses the "normal" data region.
  Points outside the boundary are anomalies. Appropriate for data with
  a compact, well-defined normal class.
  Key hyperparameter: nu — upper bound on fraction of anomalies (0 < nu < 1)
  Note: Scales poorly to large datasets (> 50,000 rows). Use IsolationForest.

ANOMALY SCORE CONVENTION
-------------------------
All algorithms return a score in [0, 1] where:
  1.0 = most anomalous (furthest from normal distribution)
  0.0 = most normal

This is achieved by normalising the raw algorithm scores (which have
different ranges and signs) to a consistent [0, 1] interval.

CONTAMINATION
-------------
contamination is the expected fraction of anomalies in the data.
It is used as a threshold for binary labelling: samples with scores
above the contamination-th percentile are flagged as anomalies.
For real data where contamination is unknown, 0.05 is a safe default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


# ── Algorithm families ────────────────────────────────────────────────────────

@dataclass
class AnomalyFamilyResult:
    family: str
    scores: np.ndarray          # per-sample anomaly score in [0, 1]
    labels: np.ndarray          # 1 = anomaly, 0 = normal
    n_anomalies: int
    contamination_used: float
    threshold_score: float      # score above which a sample is flagged
    params: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


def run_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.05,
    n_estimators: int = 100,
    random_state: int = 42,
) -> AnomalyFamilyResult:
    """
    Isolation Forest anomaly detection.

    contamination controls the threshold used to label samples:
      anomaly if score > percentile(scores, 1 - contamination)
    """
    try:
        clf = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X)

        # Raw scores: negative = anomalous (sklearn convention)
        raw_scores = clf.decision_function(X)

        # Normalise to [0, 1] where 1 = most anomalous
        scores = _normalise(raw_scores, invert=True)
        threshold = float(np.percentile(scores, (1 - contamination) * 100))
        labels = (scores >= threshold).astype(int)

        return AnomalyFamilyResult(
            family="isolation_forest",
            scores=scores,
            labels=labels,
            n_anomalies=int(labels.sum()),
            contamination_used=contamination,
            threshold_score=threshold,
            params={"n_estimators": n_estimators, "contamination": contamination},
        )
    except Exception as exc:
        return AnomalyFamilyResult(
            family="isolation_forest",
            scores=np.array([]), labels=np.array([]),
            n_anomalies=0, contamination_used=contamination,
            threshold_score=0.0, error=str(exc),
        )


def run_lof(
    X: np.ndarray,
    contamination: float = 0.05,
    n_neighbors: int = 20,
) -> AnomalyFamilyResult:
    """
    Local Outlier Factor (LOF) anomaly detection.

    LOF is transductive — it cannot score new samples after fitting.
    Use novelty=False (default) for in-sample scoring.
    """
    try:
        n_neighbors = min(n_neighbors, len(X) - 1)
        clf = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=False,
            n_jobs=-1,
        )
        clf.fit_predict(X)   # fit + compute scores in one pass

        # negative_outlier_factor_: negative = outlier
        raw_scores = clf.negative_outlier_factor_
        scores = _normalise(raw_scores, invert=True)
        threshold = float(np.percentile(scores, (1 - contamination) * 100))
        labels = (scores >= threshold).astype(int)

        return AnomalyFamilyResult(
            family="lof",
            scores=scores,
            labels=labels,
            n_anomalies=int(labels.sum()),
            contamination_used=contamination,
            threshold_score=threshold,
            params={"n_neighbors": n_neighbors, "contamination": contamination},
        )
    except Exception as exc:
        return AnomalyFamilyResult(
            family="lof",
            scores=np.array([]), labels=np.array([]),
            n_anomalies=0, contamination_used=contamination,
            threshold_score=0.0, error=str(exc),
        )


def run_ocsvm(
    X: np.ndarray,
    contamination: float = 0.05,
    nu: Optional[float] = None,
    kernel: str = "rbf",
) -> AnomalyFamilyResult:
    """
    One-Class SVM anomaly detection.

    nu is an upper bound on the fraction of anomalies. Defaults to contamination.
    Note: OC-SVM scales O(n²) — skip for datasets > 10,000 rows.
    """
    if len(X) > 10_000:
        return AnomalyFamilyResult(
            family="ocsvm",
            scores=np.array([]), labels=np.array([]),
            n_anomalies=0, contamination_used=contamination,
            threshold_score=0.0,
            error=(
                f"One-Class SVM skipped: dataset has {len(X)} rows. "
                "OC-SVM scales O(n²) and is impractical above 10,000 rows. "
                "Use Isolation Forest instead."
            ),
        )
    try:
        from sklearn.svm import OneClassSVM
        nu_val = nu if nu is not None else contamination

        clf = OneClassSVM(nu=nu_val, kernel=kernel)
        clf.fit(X)

        raw_scores = clf.decision_function(X)
        scores = _normalise(raw_scores, invert=True)
        threshold = float(np.percentile(scores, (1 - contamination) * 100))
        labels = (scores >= threshold).astype(int)

        return AnomalyFamilyResult(
            family="ocsvm",
            scores=scores,
            labels=labels,
            n_anomalies=int(labels.sum()),
            contamination_used=contamination,
            threshold_score=threshold,
            params={"nu": nu_val, "kernel": kernel},
        )
    except Exception as exc:
        return AnomalyFamilyResult(
            family="ocsvm",
            scores=np.array([]), labels=np.array([]),
            n_anomalies=0, contamination_used=contamination,
            threshold_score=0.0, error=str(exc),
        )


# ── Normalisation helper ──────────────────────────────────────────────────────

def _normalise(scores: np.ndarray, invert: bool = False) -> np.ndarray:
    """Normalises an array to [0, 1]. If invert=True, flips so higher = more anomalous."""
    s = scores.astype(float)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return np.zeros_like(s)
    normalised = (s - mn) / (mx - mn)
    return 1 - normalised if invert else normalised


# ── Preprocessing ─────────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame, exclude_columns: list[str] = None) -> np.ndarray:
    """
    Prepares a numeric feature matrix for anomaly detection.

    - Drops excluded columns (e.g. ID, target)
    - Drops non-numeric columns
    - Fills NaN with column median
    - Scales to zero mean, unit variance (important for LOF and OC-SVM)
    """
    exclude = set(exclude_columns or [])
    numeric = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in exclude if c in df.columns], errors="ignore"
    )
    if numeric.empty:
        raise ValueError("No numeric columns available for anomaly detection.")
    filled = numeric.fillna(numeric.median())
    return StandardScaler().fit_transform(filled), list(numeric.columns)
