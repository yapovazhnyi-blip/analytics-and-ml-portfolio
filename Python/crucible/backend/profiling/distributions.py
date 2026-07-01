"""
Distribution analyser for Crucible's profiling layer.

Focuses on the target variable — the most important column for any
ML experiment. Answers:
  - Classification: are classes balanced? Imbalance > 4:1 is a common
    failure mode that AutoML tools handle poorly without guidance.
  - Regression: is the target skewed? Heavy skew (|skewness| > 1) often
    benefits from log transformation before training.
  - Both: what are the basic statistics (mean, std, min, max, percentiles)?

Also computes per-column basic statistics for all numeric features,
surfaced in the profiling UI as a column-by-column overview.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ── Output types ───────────────────────────────────────────────────────────

@dataclass
class ClassDistribution:
    label: str
    count: int
    proportion: float


@dataclass
class TargetAnalysis:
    column: str
    task_type: str           # "classification" | "regression" | "unknown"
    n_unique: int
    null_count: int

    # Classification fields
    class_distribution: list[ClassDistribution] = field(default_factory=list)
    imbalance_ratio: Optional[float] = None   # majority / minority count
    is_imbalanced: bool = False               # True if ratio > 4.0

    # Regression fields
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    skewness: Optional[float] = None
    is_skewed: bool = False                   # True if |skewness| > 1.0

    @property
    def imbalance_warning(self) -> Optional[str]:
        if self.is_imbalanced and self.imbalance_ratio:
            return (
                f"Class imbalance detected — majority class is {self.imbalance_ratio:.1f}x "
                f"the minority class. Consider oversampling (SMOTE), undersampling, "
                f"or class_weight='balanced'."
            )
        return None

    @property
    def skewness_warning(self) -> Optional[str]:
        if self.is_skewed and self.skewness is not None:
            direction = "right" if self.skewness > 0 else "left"
            return (
                f"Target is {direction}-skewed (skewness = {self.skewness:.2f}). "
                f"Consider log or Box-Cox transformation before training."
            )
        return None


@dataclass
class ColumnStats:
    column: str
    dtype: str
    n_unique: int
    null_count: int
    null_rate: float
    # Numeric only
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    p25: Optional[float] = None
    median: Optional[float] = None
    p75: Optional[float] = None
    max: Optional[float] = None
    skewness: Optional[float] = None


@dataclass
class DistributionReport:
    target_analysis: Optional[TargetAnalysis]
    column_stats: list[ColumnStats]
    n_rows: int
    n_columns: int


# ── Classification threshold ────────────────────────────────────────────────
# A column is treated as classification target if it has <= this many unique
# values OR if its dtype is object/bool. Otherwise it's regression.
CLASSIFICATION_UNIQUE_THRESHOLD = 20


# ── Main function ──────────────────────────────────────────────────────────

def analyse_distributions(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
) -> DistributionReport:
    """
    Compute distribution statistics for all columns and optionally
    perform deep analysis on the target column.

    Args:
        df:            Input DataFrame
        target_column: Name of the target variable, if known

    Returns:
        DistributionReport with target analysis and per-column stats
    """
    column_stats = [_compute_column_stats(df, col) for col in df.columns]

    target_analysis = None
    if target_column and target_column in df.columns:
        target_analysis = _analyse_target(df[target_column], target_column)

    return DistributionReport(
        target_analysis=target_analysis,
        column_stats=column_stats,
        n_rows=len(df),
        n_columns=len(df.columns),
    )


def _analyse_target(series: pd.Series, column: str) -> TargetAnalysis:
    """Deep analysis of the target variable."""
    n_unique = int(series.nunique())
    null_count = int(series.isna().sum())
    is_numeric = pd.api.types.is_numeric_dtype(series)
    is_categorical = (
        not is_numeric
        or n_unique <= CLASSIFICATION_UNIQUE_THRESHOLD
        or series.dtype == bool
    )

    if is_categorical:
        return _classification_target(series, column, n_unique, null_count)
    else:
        return _regression_target(series, column, n_unique, null_count)


def _classification_target(
    series: pd.Series,
    column: str,
    n_unique: int,
    null_count: int,
) -> TargetAnalysis:
    clean = series.dropna()
    counts = clean.value_counts()
    total = len(clean)

    distribution = [
        ClassDistribution(
            label=str(label),
            count=int(count),
            proportion=round(count / total, 4),
        )
        for label, count in counts.items()
    ]

    imbalance_ratio = None
    is_imbalanced = False
    if len(counts) >= 2:
        imbalance_ratio = round(float(counts.iloc[0] / counts.iloc[-1]), 2)
        is_imbalanced = bool(imbalance_ratio > 4.0)

    return TargetAnalysis(
        column=column,
        task_type="classification",
        n_unique=n_unique,
        null_count=null_count,
        class_distribution=distribution,
        imbalance_ratio=imbalance_ratio,
        is_imbalanced=is_imbalanced,
    )


def _regression_target(
    series: pd.Series,
    column: str,
    n_unique: int,
    null_count: int,
) -> TargetAnalysis:
    clean = series.dropna().astype(float)

    try:
        from scipy import stats as scipy_stats
        skewness = round(float(scipy_stats.skew(clean)), 4)
    except Exception:
        skewness = round(float(clean.skew()), 4)

    return TargetAnalysis(
        column=column,
        task_type="regression",
        n_unique=n_unique,
        null_count=null_count,
        mean=round(float(clean.mean()), 4),
        std=round(float(clean.std()), 4),
        min=round(float(clean.min()), 4),
        max=round(float(clean.max()), 4),
        skewness=skewness,
        is_skewed=abs(skewness) > 1.0,
    )


def _compute_column_stats(df: pd.DataFrame, col: str) -> ColumnStats:
    """Basic statistics for a single column."""
    series = df[col]
    n = len(series)
    null_count = int(series.isna().sum())

    stats = ColumnStats(
        column=col,
        dtype=str(series.dtype),
        n_unique=int(series.nunique()),
        null_count=null_count,
        null_rate=round(null_count / n, 4) if n > 0 else 0.0,
    )

    if pd.api.types.is_numeric_dtype(series):
        clean = series.dropna().astype(float)
        if len(clean) > 0:
            stats.mean = round(float(clean.mean()), 4)
            stats.std = round(float(clean.std()), 4)
            stats.min = round(float(clean.min()), 4)
            stats.p25 = round(float(clean.quantile(0.25)), 4)
            stats.median = round(float(clean.median()), 4)
            stats.p75 = round(float(clean.quantile(0.75)), 4)
            stats.max = round(float(clean.max()), 4)
            stats.skewness = round(float(clean.skew()), 4)

    return stats
