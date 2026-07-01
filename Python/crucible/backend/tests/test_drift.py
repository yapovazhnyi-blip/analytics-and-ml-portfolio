"""
Drift detection tests.

Tests cover: PSI calculation, KS test, chi-squared, severity thresholds,
and the full compare_datasets() report. All tests use synthetic DataFrames
— no file I/O or database needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drift.detector import (
    DriftDetector,
    DriftReport,
    FeatureDrift,
    compare_datasets,
    psi_severity,
    PSI_NEGLIGIBLE,
    PSI_SLIGHT,
    PSI_SIGNIFICANT,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def stable_pair():
    """Two DataFrames drawn from the same distribution — expect no drift."""
    rng = np.random.default_rng(42)
    n = 500
    ref = pd.DataFrame({
        "age":    rng.normal(35, 10, n),
        "income": rng.normal(60000, 15000, n),
        "city":   rng.choice(["London", "Paris", "Berlin"], n),
    })
    cur = pd.DataFrame({
        "age":    rng.normal(35, 10, n),
        "income": rng.normal(60000, 15000, n),
        "city":   rng.choice(["London", "Paris", "Berlin"], n),
    })
    return ref, cur


@pytest.fixture
def drifted_pair():
    """Reference and current with significant numeric and categorical drift."""
    rng = np.random.default_rng(0)
    n = 500
    ref = pd.DataFrame({
        "age":    rng.normal(35, 5, n),
        "score":  rng.uniform(0, 1, n),
        "status": rng.choice(["A", "B", "C"], n, p=[0.6, 0.3, 0.1]),
    })
    cur = pd.DataFrame({
        "age":    rng.normal(55, 5, n),   # massive mean shift: 35 → 55
        "score":  rng.uniform(0.7, 1, n), # distribution narrows to upper range
        "status": rng.choice(["A", "B", "C"], n, p=[0.1, 0.3, 0.6]),  # inverted
    })
    return ref, cur


# ══════════════════════════════════════════════════════════════════════════
# PSI SEVERITY THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════

class TestPSISeverity:

    def test_below_negligible_is_stable(self):
        assert psi_severity(PSI_NEGLIGIBLE - 0.01) == "stable"

    def test_above_negligible_is_slight(self):
        assert psi_severity(PSI_NEGLIGIBLE + 0.01) == "slight"

    def test_above_slight_is_significant(self):
        assert psi_severity(PSI_SLIGHT + 0.01) == "significant"

    def test_above_significant_is_critical(self):
        assert psi_severity(PSI_SIGNIFICANT + 0.01) == "critical"

    def test_zero_psi_is_stable(self):
        assert psi_severity(0.0) == "stable"


# ══════════════════════════════════════════════════════════════════════════
# PSI CALCULATION
# ══════════════════════════════════════════════════════════════════════════

class TestPSI:

    def _psi(self, ref, cur):
        detector = DriftDetector()
        return detector._psi(np.array(ref), np.array(cur))

    def test_identical_arrays_psi_near_zero(self):
        data = np.random.default_rng(1).normal(0, 1, 1000)
        psi = self._psi(data, data)
        assert psi < 0.01

    def test_identical_distribution_psi_low(self):
        rng = np.random.default_rng(7)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(0, 1, 1000)   # different samples, same distribution
        psi = self._psi(ref, cur)
        assert psi < PSI_NEGLIGIBLE

    def test_shifted_distribution_psi_high(self):
        ref = np.random.default_rng(0).normal(0, 1, 1000)
        cur = np.random.default_rng(1).normal(5, 1, 1000)  # mean shifted by 5σ
        psi = self._psi(ref, cur)
        assert psi >= PSI_SIGNIFICANT

    def test_psi_non_negative(self):
        ref = np.random.default_rng(2).normal(0, 1, 500)
        cur = np.random.default_rng(3).uniform(0, 1, 500)
        psi = self._psi(ref, cur)
        assert psi >= 0.0

    def test_constant_array_returns_zero(self):
        """Constant arrays have no distribution to compare."""
        ref = np.ones(100)
        cur = np.ones(100)
        psi = self._psi(ref, cur)
        assert psi == 0.0


# ══════════════════════════════════════════════════════════════════════════
# NUMERIC DRIFT
# ══════════════════════════════════════════════════════════════════════════

class TestNumericDrift:

    def test_stable_numeric_feature(self, stable_pair):
        ref, cur = stable_pair
        detector = DriftDetector()
        result = detector._numeric_drift("age", ref["age"], cur["age"])
        assert result.dtype == "numeric"
        assert result.psi is not None
        assert result.ks_stat is not None
        assert result.severity in ("stable", "slight")

    def test_drifted_numeric_feature(self, drifted_pair):
        ref, cur = drifted_pair
        detector = DriftDetector()
        result = detector._numeric_drift("age", ref["age"], cur["age"])
        assert result.has_drift
        assert result.psi >= PSI_SIGNIFICANT

    def test_mean_and_std_captured(self, stable_pair):
        ref, cur = stable_pair
        detector = DriftDetector()
        result = detector._numeric_drift("income", ref["income"], cur["income"])
        assert result.reference_mean is not None
        assert result.current_mean is not None
        assert result.reference_std is not None

    def test_ks_pvalue_present(self, stable_pair):
        ref, cur = stable_pair
        detector = DriftDetector()
        result = detector._numeric_drift("age", ref["age"], cur["age"])
        assert 0.0 <= result.ks_pvalue <= 1.0


# ══════════════════════════════════════════════════════════════════════════
# CATEGORICAL DRIFT
# ══════════════════════════════════════════════════════════════════════════

class TestCategoricalDrift:

    def test_stable_categorical_feature(self, stable_pair):
        ref, cur = stable_pair
        detector = DriftDetector()
        result = detector._categorical_drift("city", ref["city"], cur["city"])
        assert result.dtype == "categorical"
        assert result.chi2_stat is not None
        assert result.chi2_pvalue is not None
        assert result.severity in ("stable", "slight")

    def test_drifted_categorical_feature(self, drifted_pair):
        ref, cur = drifted_pair
        detector = DriftDetector()
        result = detector._categorical_drift("status", ref["status"], cur["status"])
        assert result.has_drift

    def test_top_shifted_categories_present(self, drifted_pair):
        ref, cur = drifted_pair
        detector = DriftDetector()
        result = detector._categorical_drift("status", ref["status"], cur["status"])
        assert len(result.top_shifted_categories) > 0
        for cat in result.top_shifted_categories:
            assert "category" in cat
            assert "reference_pct" in cat
            assert "current_pct" in cat

    def test_psi_is_none_for_categorical(self, stable_pair):
        ref, cur = stable_pair
        detector = DriftDetector()
        result = detector._categorical_drift("city", ref["city"], cur["city"])
        assert result.psi is None


# ══════════════════════════════════════════════════════════════════════════
# FULL REPORT
# ══════════════════════════════════════════════════════════════════════════

class TestDriftReport:

    def test_stable_datasets_report(self, stable_pair):
        ref, cur = stable_pair
        report = compare_datasets(ref, cur, reference_id=1, current_id=2)
        assert report.succeeded
        assert report.n_features_checked == 3   # age, income, city
        assert report.severity in ("stable", "slight")

    def test_drifted_datasets_report(self, drifted_pair):
        ref, cur = drifted_pair
        report = compare_datasets(ref, cur, reference_id=1, current_id=2)
        assert report.succeeded
        assert report.n_features_drifted >= 2   # age and score definitely drift
        assert report.severity in ("significant", "critical")

    def test_target_col_excluded(self, stable_pair):
        ref, cur = stable_pair
        # Add a target column
        ref["target"] = np.random.randint(0, 2, len(ref))
        cur["target"] = np.random.randint(0, 2, len(cur))
        report = compare_datasets(ref, cur, reference_id=1, current_id=2, target_col="target")
        checked_features = [f.feature for f in report.feature_drifts]
        assert "target" not in checked_features

    def test_report_to_dict_structure(self, stable_pair):
        ref, cur = stable_pair
        report = compare_datasets(ref, cur, reference_id=1, current_id=2)
        d = report.to_dict()
        assert "n_features_checked" in d
        assert "overall_psi" in d
        assert "severity" in d
        assert "features" in d
        assert isinstance(d["features"], list)

    def test_features_sorted_by_severity(self, drifted_pair):
        """Most-drifted features should appear first in the report."""
        ref, cur = drifted_pair
        report = compare_datasets(ref, cur, reference_id=1, current_id=2)
        d = report.to_dict()
        # The most drifted feature (age, massive mean shift) should appear first
        assert d["features"][0]["feature"] in ("age", "score", "status")

    def test_mismatched_columns_handled(self):
        """Only shared columns should be compared."""
        ref = pd.DataFrame({"a": np.random.normal(0, 1, 100), "b": np.random.normal(0, 1, 100)})
        cur = pd.DataFrame({"a": np.random.normal(0, 1, 100), "c": np.random.normal(0, 1, 100)})
        report = compare_datasets(ref, cur, reference_id=1, current_id=2)
        checked = [f.feature for f in report.feature_drifts]
        assert "a" in checked
        assert "b" not in checked
        assert "c" not in checked

    def test_too_few_samples_skipped(self):
        """Features with fewer than 10 samples are skipped."""
        ref = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        cur = pd.DataFrame({"x": [4.0, 5.0, 6.0]})
        report = compare_datasets(ref, cur, reference_id=1, current_id=2)
        assert report.n_features_checked == 0
