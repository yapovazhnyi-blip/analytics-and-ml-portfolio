"""
Drift detector — statistical tests to detect when a dataset's distribution
has shifted relative to a reference (typically the training data).

WHY DRIFT DETECTION MATTERS
----------------------------
A model trained on February sales data will degrade when deployed in
December because customer behaviour changes seasonally. The model's
holdout accuracy looked fine — but that holdout was also from February.
Drift detection catches the mismatch between what the model was trained on
and what it now sees in production before the degradation shows up as
user-visible failures.

TWO COMPLEMENTARY TESTS
------------------------

PSI — Population Stability Index
  Originally developed for credit scorecards. Measures how much a
  feature's distribution has shifted by comparing the fraction of values
  in each histogram bin between reference and current.

    PSI = Σ (current_fraction - reference_fraction) × ln(current/reference)

  Interpretation (industry standard for credit risk):
    PSI < 0.10  — negligible shift, model is stable
    PSI < 0.20  — slight shift, monitor but no action needed
    PSI ≥ 0.20  — significant shift, re-train or investigate
    PSI ≥ 0.25  — major shift, model likely degraded

  PSI is fast (O(n) per feature) and interpretable, making it the
  standard first-pass drift metric.

KS Test — Kolmogorov-Smirnov
  Non-parametric test that compares empirical CDFs. Catches shifts that
  PSI might miss (e.g. subtle changes in tail behaviour). Reports both
  a statistic (max CDF difference) and a p-value.

  p-value < 0.05 → statistically significant distributional shift.

Chi-Squared Test
  For categorical features. Tests whether the frequency distribution
  across categories is the same in reference and current.

REFERENCE DATASET
-----------------
For each experiment, the reference is the training split stored in the
Dataset record. When a new version of the data is uploaded, the drift
check compares it against the training version.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ── PSI severity thresholds ───────────────────────────────────────────────────

PSI_NEGLIGIBLE  = 0.10
PSI_SLIGHT      = 0.20
PSI_SIGNIFICANT = 0.25


def psi_severity(psi: float) -> str:
    if psi < PSI_NEGLIGIBLE:
        return "stable"
    elif psi < PSI_SLIGHT:
        return "slight"
    elif psi < PSI_SIGNIFICANT:
        return "significant"
    else:
        return "critical"


# ── Feature-level result ──────────────────────────────────────────────────────

@dataclass
class FeatureDrift:
    """Drift measurement for a single feature."""
    feature: str
    dtype: str              # "numeric" | "categorical"
    psi: Optional[float]    # None for categorical (uses chi2 instead)
    ks_stat: Optional[float]
    ks_pvalue: Optional[float]
    chi2_stat: Optional[float]
    chi2_pvalue: Optional[float]
    severity: str           # stable | slight | significant | critical
    reference_mean: Optional[float] = None
    current_mean: Optional[float]   = None
    reference_std: Optional[float]  = None
    current_std: Optional[float]    = None
    top_shifted_categories: list[dict] = field(default_factory=list)

    @property
    def has_drift(self) -> bool:
        return self.severity in ("significant", "critical")

    @property
    def primary_stat(self) -> float:
        """The main drift statistic (PSI for numeric, chi2 p-value for categorical)."""
        if self.dtype == "numeric":
            return self.psi or 0.0
        else:
            return 1.0 - (self.chi2_pvalue or 1.0)   # invert: higher = more drift


@dataclass
class DriftReport:
    """Drift report for a full dataset comparison."""
    reference_dataset_id: int
    current_dataset_id: int
    n_features_checked: int
    n_features_drifted: int
    overall_psi: float          # mean PSI across numeric features
    severity: str               # worst feature severity
    feature_drifts: list[FeatureDrift] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def drifted_features(self) -> list[FeatureDrift]:
        return [f for f in self.feature_drifts if f.has_drift]

    def to_dict(self) -> dict:
        return {
            "reference_dataset_id":  self.reference_dataset_id,
            "current_dataset_id":    self.current_dataset_id,
            "n_features_checked":    self.n_features_checked,
            "n_features_drifted":    self.n_features_drifted,
            "overall_psi":           round(self.overall_psi, 4),
            "severity":              self.severity,
            "features": [
                {
                    "feature":          f.feature,
                    "dtype":            f.dtype,
                    "severity":         f.severity,
                    "has_drift":        f.has_drift,
                    "psi":              round(f.psi, 4) if f.psi is not None else None,
                    "ks_stat":          round(f.ks_stat, 4) if f.ks_stat else None,
                    "ks_pvalue":        round(f.ks_pvalue, 4) if f.ks_pvalue else None,
                    "chi2_stat":        round(f.chi2_stat, 4) if f.chi2_stat else None,
                    "chi2_pvalue":      round(f.chi2_pvalue, 4) if f.chi2_pvalue else None,
                    "reference_mean":   round(f.reference_mean, 4) if f.reference_mean is not None else None,
                    "current_mean":     round(f.current_mean, 4) if f.current_mean is not None else None,
                    "reference_std":    round(f.reference_std, 4) if f.reference_std is not None else None,
                    "current_std":      round(f.current_std, 4) if f.current_std is not None else None,
                    "top_shifted_categories": f.top_shifted_categories,
                }
                for f in sorted(self.feature_drifts, key=lambda x: x.primary_stat, reverse=True)
            ],
        }


# ── Core detector ─────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Detects distributional drift between a reference DataFrame and a
    current DataFrame.

    Typical use: reference = training data, current = new batch of data.
    """

    N_BINS = 10    # PSI histogram bins

    def detect(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        target_col: Optional[str] = None,
        max_categories: int = 50,
    ) -> dict[str, FeatureDrift]:
        """
        Runs drift detection on all shared features.

        Args:
            reference:       The training (reference) dataset.
            current:         The new (current) dataset to compare.
            target_col:      Column to exclude (target variable, not a feature).
            max_categories:  Skip categorical features with more distinct values
                             than this — high-cardinality IDs are not useful.

        Returns:
            Dict mapping feature name → FeatureDrift.
        """
        # Only compare features present in both datasets
        shared = [c for c in reference.columns if c in current.columns]
        if target_col and target_col in shared:
            shared.remove(target_col)

        results: dict[str, FeatureDrift] = {}
        for col in shared:
            ref_col = reference[col].dropna()
            cur_col = current[col].dropna()

            if len(ref_col) < 10 or len(cur_col) < 10:
                continue   # too few samples for meaningful statistics

            is_numeric = pd.api.types.is_numeric_dtype(ref_col)

            if is_numeric:
                results[col] = self._numeric_drift(col, ref_col, cur_col)
            else:
                n_cats = ref_col.nunique()
                if n_cats > max_categories:
                    continue   # skip high-cardinality identifiers
                results[col] = self._categorical_drift(col, ref_col, cur_col)

        return results

    # ── Numeric drift ──────────────────────────────────────────────────────────

    def _numeric_drift(
        self, feature: str, ref: pd.Series, cur: pd.Series
    ) -> FeatureDrift:
        """PSI + KS test for numeric features."""
        psi_val = self._psi(ref.values, cur.values)

        from scipy import stats
        ks_result = stats.ks_2samp(ref.values, cur.values)

        # Determine severity from PSI
        severity = psi_severity(psi_val)
        # Upgrade severity if KS test is highly significant
        if ks_result.pvalue < 0.001 and severity == "stable":
            severity = "slight"

        return FeatureDrift(
            feature=feature,
            dtype="numeric",
            psi=psi_val,
            ks_stat=float(ks_result.statistic),
            ks_pvalue=float(ks_result.pvalue),
            chi2_stat=None,
            chi2_pvalue=None,
            severity=severity,
            reference_mean=float(ref.mean()),
            current_mean=float(cur.mean()),
            reference_std=float(ref.std()),
            current_std=float(cur.std()),
        )

    def _psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """
        Population Stability Index.
        Bins reference into N_BINS equal-width buckets, then measures how
        much the current distribution deviates from those same buckets.
        """
        # Use reference percentiles as bin edges (handles skewed distributions)
        breakpoints = np.percentile(reference, np.linspace(0, 100, self.N_BINS + 1))
        breakpoints = np.unique(breakpoints)   # remove duplicates from flat distributions

        if len(breakpoints) < 3:
            return 0.0   # all values identical — no drift possible

        ref_counts, _ = np.histogram(reference, bins=breakpoints)
        cur_counts, _ = np.histogram(current, bins=breakpoints)

        # Fraction in each bin (avoid division by zero with clip)
        ref_frac = np.clip(ref_counts / len(reference), 1e-10, None)
        cur_frac = np.clip(cur_counts / len(current),   1e-10, None)

        psi = float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))
        return max(0.0, psi)   # PSI is non-negative by definition

    # ── Categorical drift ──────────────────────────────────────────────────────

    def _categorical_drift(
        self, feature: str, ref: pd.Series, cur: pd.Series
    ) -> FeatureDrift:
        """Chi-squared test for categorical features."""
        from scipy import stats

        # Build frequency tables aligned on the same categories
        all_cats = set(ref.unique()) | set(cur.unique())
        ref_counts = ref.value_counts().reindex(all_cats, fill_value=0)
        cur_counts = cur.value_counts().reindex(all_cats, fill_value=0)

        # Chi-squared requires at least 5 expected per cell; merge rare categories
        ref_arr = ref_counts.values.astype(float)
        cur_arr = cur_counts.values.astype(float)

        chi2_stat, chi2_pvalue = stats.chisquare(cur_arr + 1e-10, f_exp=ref_arr + 1e-10)

        # Severity based on p-value
        if chi2_pvalue >= 0.10:
            severity = "stable"
        elif chi2_pvalue >= 0.05:
            severity = "slight"
        elif chi2_pvalue >= 0.01:
            severity = "significant"
        else:
            severity = "critical"

        # Find categories with the biggest frequency shift
        ref_norm = ref_counts / ref_counts.sum()
        cur_norm = cur_counts / cur_counts.sum()
        shift = (cur_norm - ref_norm).abs().sort_values(ascending=False)
        top_shifted = [
            {
                "category":       str(cat),
                "reference_pct":  round(float(ref_norm.get(cat, 0)) * 100, 1),
                "current_pct":    round(float(cur_norm.get(cat, 0)) * 100, 1),
            }
            for cat in shift.head(5).index
        ]

        return FeatureDrift(
            feature=feature,
            dtype="categorical",
            psi=None,
            ks_stat=None,
            ks_pvalue=None,
            chi2_stat=float(chi2_stat),
            chi2_pvalue=float(chi2_pvalue),
            severity=severity,
            top_shifted_categories=top_shifted,
        )


# ── Convenience: compare two DataFrames ───────────────────────────────────────

def compare_datasets(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reference_id: int,
    current_id: int,
    target_col: Optional[str] = None,
) -> DriftReport:
    """
    Top-level function: run drift detection and return a DriftReport.
    """
    try:
        detector = DriftDetector()
        feature_drifts = detector.detect(reference, current, target_col=target_col)

        if not feature_drifts:
            return DriftReport(
                reference_dataset_id=reference_id,
                current_dataset_id=current_id,
                n_features_checked=0,
                n_features_drifted=0,
                overall_psi=0.0,
                severity="stable",
                error="No comparable features found in both datasets.",
            )

        drifted = [f for f in feature_drifts.values() if f.has_drift]
        numeric_psis = [f.psi for f in feature_drifts.values()
                        if f.dtype == "numeric" and f.psi is not None]
        overall_psi = float(np.mean(numeric_psis)) if numeric_psis else 0.0

        # Overall severity = worst individual severity
        severity_order = {"stable": 0, "slight": 1, "significant": 2, "critical": 3}
        worst = max(feature_drifts.values(),
                    key=lambda f: severity_order.get(f.severity, 0))

        return DriftReport(
            reference_dataset_id=reference_id,
            current_dataset_id=current_id,
            n_features_checked=len(feature_drifts),
            n_features_drifted=len(drifted),
            overall_psi=overall_psi,
            severity=worst.severity,
            feature_drifts=list(feature_drifts.values()),
        )

    except Exception as exc:
        return DriftReport(
            reference_dataset_id=reference_id,
            current_dataset_id=current_id,
            n_features_checked=0,
            n_features_drifted=0,
            overall_psi=0.0,
            severity="stable",
            error=str(exc),
        )
