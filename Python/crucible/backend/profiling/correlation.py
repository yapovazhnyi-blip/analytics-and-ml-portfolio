"""
Correlation analyser for Crucible's profiling layer.

Two distinct concerns, kept separate:
  1. High pairwise correlation   — two features that say the same thing
                                   (r ≥ 0.90). One can usually be dropped.
  2. Multicollinearity via VIF   — one feature is a near-linear combination
                                   of several others. Pure pairwise correlation
                                   misses this case. VIF > 10 is the classic
                                   threshold (literature varies from 5 to 10).

Why VIF matters: a model can have no high pairwise correlations but still
suffer from multicollinearity if three or more features jointly explain
each other. VIF catches this; pairwise correlation doesn't.

Both findings are returned as structured results for the Claude advisor
to generate specific, actionable suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ── Output types ───────────────────────────────────────────────────────────

@dataclass
class PairwiseCorrelation:
    col_a: str
    col_b: str
    correlation: float
    method: str  # "pearson" | "spearman"


@dataclass
class VIFResult:
    column: str
    vif: float
    severe: bool  # True if VIF > 10


@dataclass
class CorrelationReport:
    high_pairs: list[PairwiseCorrelation]     # |r| >= threshold
    vif_results: list[VIFResult]              # all computed VIF values
    columns_analysed: int
    skipped_columns: list[str]                # non-numeric, skipped


# ── Main function ──────────────────────────────────────────────────────────

def analyse_correlations(
    df: pd.DataFrame,
    correlation_threshold: float = 0.90,
    vif_threshold: float = 10.0,
    max_columns_for_vif: int = 50,
) -> CorrelationReport:
    """
    Compute pairwise correlations and VIF for all numeric columns.

    Args:
        df:                     Input DataFrame
        correlation_threshold:  |r| above which a pair is flagged (default 0.90)
        vif_threshold:          VIF above which a column is flagged (default 10)
        max_columns_for_vif:    Skip VIF for wide datasets — it's O(n·p²) and
                                becomes slow beyond ~50 columns

    Returns:
        CorrelationReport with high-correlation pairs and VIF values.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    non_numeric = [c for c in df.columns if c not in numeric_df.columns]

    if numeric_df.empty or len(numeric_df.columns) < 2:
        return CorrelationReport(
            high_pairs=[],
            vif_results=[],
            columns_analysed=len(numeric_df.columns),
            skipped_columns=non_numeric,
        )

    # Drop columns that are entirely null — can't compute correlations
    numeric_df = numeric_df.dropna(axis=1, how="all")

    high_pairs = _find_high_pairs(numeric_df, correlation_threshold)
    vif_results = _compute_vif(numeric_df, vif_threshold, max_columns_for_vif)

    return CorrelationReport(
        high_pairs=high_pairs,
        vif_results=vif_results,
        columns_analysed=len(numeric_df.columns),
        skipped_columns=non_numeric,
    )


def _find_high_pairs(
    df: pd.DataFrame,
    threshold: float,
) -> list[PairwiseCorrelation]:
    """
    Returns unique pairs where |Pearson r| >= threshold.
    Uses Pearson — fast and sufficient for flagging purposes. The Claude
    advisor can note when Spearman might be more appropriate (non-linear).
    """
    corr = df.corr(method="pearson").abs()
    cols = list(corr.columns)
    pairs = []

    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1:]:
            val = corr.loc[col_a, col_b]
            if pd.notna(val) and val >= threshold:
                pairs.append(PairwiseCorrelation(
                    col_a=col_a,
                    col_b=col_b,
                    correlation=round(float(val), 4),
                    method="pearson",
                ))

    return sorted(pairs, key=lambda p: p.correlation, reverse=True)


def _compute_vif(
    df: pd.DataFrame,
    threshold: float,
    max_cols: int,
) -> list[VIFResult]:
    """
    Variance Inflation Factor per column.

    VIF for column j = 1 / (1 - R²_j), where R²_j is the R² from
    regressing column j on all other columns. A VIF of 10 means ~90%
    of column j's variance is explained by the other columns.

    We implement this directly with numpy rather than importing statsmodels,
    keeping dependencies lean.
    """
    cols = list(df.columns)

    if len(cols) > max_cols:
        return []  # too many columns — skip silently, router will note this

    # Fill nulls with column median for the regression
    X = df.fillna(df.median()).values.astype(float)
    n, p = X.shape

    if n <= p:
        return []  # underdetermined — skip

    results = []
    for j in range(p):
        y = X[:, j]
        X_others = np.delete(X, j, axis=1)

        # Add intercept
        X_aug = np.column_stack([np.ones(n), X_others])

        try:
            # Least squares: β = (XᵀX)⁻¹ Xᵀy
            coeffs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
            y_hat = X_aug @ coeffs
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
        except Exception:
            vif = float("nan")

        results.append(VIFResult(
            column=cols[j],
            vif=round(vif, 2) if not np.isnan(vif) and not np.isinf(vif) else vif,
            severe=vif > threshold,
        ))

    return sorted(results, key=lambda v: v.vif if np.isfinite(v.vif) else 1e9, reverse=True)
