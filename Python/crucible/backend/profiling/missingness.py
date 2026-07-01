"""
Missingness analyser for Crucible's profiling layer.

Computes per-column missing rates and distinguishes systematic missingness
(correlated with other columns — suggests a data collection process issue)
from random missingness (no pattern — typically safe to impute).

The distinction matters because:
  - Random missingness → mean/median imputation is usually fine
  - Systematic missingness → imputation can introduce bias; Claude advisor
    will suggest investigating the root cause before imputing

Output is a list of MissingnessResult, one per column that has any nulls.
Columns with zero nulls are omitted — they don't need attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MissingnessResult:
    column: str
    missing_count: int
    missing_rate: float       # 0.0 – 1.0
    likely_systematic: bool   # True if missingness correlates with another column
    correlated_with: Optional[str] = None  # the column driving systematic missingness
    correlation_strength: Optional[float] = None

    @property
    def severity(self) -> str:
        """Human-readable severity for the UI."""
        if self.missing_rate >= 0.5:
            return "high"
        if self.missing_rate >= 0.1:
            return "medium"
        return "low"


def analyse_missingness(
    df: pd.DataFrame,
    systematic_threshold: float = 0.30,
) -> list[MissingnessResult]:
    """
    Analyse per-column missingness across the whole DataFrame.

    Args:
        df:                    Input DataFrame
        systematic_threshold:  Correlation strength above which missingness
                               is flagged as likely systematic (default 0.30)

    Returns:
        One MissingnessResult per column that has at least one null.
        Columns with zero nulls are not included — no noise in the output.
    """
    results = []
    n = len(df)
    if n == 0:
        return results

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        if missing_count == 0:
            continue

        missing_rate = missing_count / n
        miss_indicator = df[col].isna().astype(int)

        # Check if missingness correlates with any numeric column
        # (excluding the column itself if it's numeric)
        best_corr = 0.0
        best_col: Optional[str] = None

        for other in numeric_cols:
            if other == col:
                continue
            try:
                other_series = df[other].fillna(df[other].median())
                corr = float(abs(miss_indicator.corr(other_series)))
                if corr > best_corr:
                    best_corr = corr
                    best_col = other
            except Exception:
                continue

        likely_systematic = best_corr >= systematic_threshold

        results.append(MissingnessResult(
            column=col,
            missing_count=missing_count,
            missing_rate=round(missing_rate, 4),
            likely_systematic=likely_systematic,
            correlated_with=best_col if likely_systematic else None,
            correlation_strength=round(best_corr, 4) if likely_systematic else None,
        ))

    return results
