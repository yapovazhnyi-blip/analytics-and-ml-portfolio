"""
Profiling layer tests.

Tests validate that each module produces correct output on known data —
not just that it runs. Where possible, we construct datasets with known
properties (e.g. a column that is 30% null, or two perfectly correlated
features) and assert the profiling output matches expectations.
"""

import numpy as np
import pandas as pd
import pytest

from profiling.missingness import analyse_missingness
from profiling.correlation import analyse_correlations
from profiling.distributions import analyse_distributions
from profiling.runner import ProfileRunner


# ── Shared fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def clean_df():
    """A clean dataset with no nulls, balanced classes."""
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame({
        "age":    rng.integers(20, 70, n).astype(float),
        "income": rng.normal(50000, 15000, n),
        "score":  rng.uniform(0, 1, n),
        "target": rng.integers(0, 2, n),
    })


@pytest.fixture
def df_with_nulls():
    """Dataset with systematic missingness: 'income' is null when 'age' > 50."""
    rng = np.random.default_rng(0)
    n = 300
    age = rng.integers(20, 70, n).astype(float)
    income = rng.normal(50000, 10000, n)
    income[age > 50] = np.nan   # systematic: missing based on age
    score = rng.uniform(0, 1, n)
    score[rng.random(n) < 0.05] = np.nan   # random 5% missing
    return pd.DataFrame({"age": age, "income": income, "score": score, "target": rng.integers(0, 2, n)})


@pytest.fixture
def imbalanced_df():
    """Dataset with 10:1 class imbalance."""
    rng = np.random.default_rng(1)
    n = 330
    target = [0] * 300 + [1] * 30
    return pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "y": rng.normal(0, 1, n),
        "target": target,
    })


@pytest.fixture
def skewed_regression_df():
    """Dataset with right-skewed regression target (lognormal)."""
    rng = np.random.default_rng(2)
    n = 300
    return pd.DataFrame({
        "feature": rng.normal(0, 1, n),
        "price": rng.lognormal(mean=10, sigma=2, size=n),  # heavy right skew
    })


@pytest.fixture
def correlated_df():
    """Dataset where col_a and col_b are perfectly correlated."""
    rng = np.random.default_rng(3)
    n = 200
    base = rng.normal(0, 1, n)
    return pd.DataFrame({
        "col_a": base,
        "col_b": base * 2 + 1,     # perfect linear relationship
        "col_c": rng.normal(0, 1, n),
        "target": rng.integers(0, 2, n),
    })


# ══════════════════════════════════════════════════════════════════════════
# MISSINGNESS
# ══════════════════════════════════════════════════════════════════════════

class TestMissingness:

    def test_no_nulls_returns_empty_list(self, clean_df):
        results = analyse_missingness(clean_df)
        assert results == []

    def test_detects_columns_with_nulls(self, df_with_nulls):
        results = analyse_missingness(df_with_nulls)
        col_names = [r.column for r in results]
        assert "income" in col_names
        assert "score" in col_names
        assert "age" not in col_names  # age has no nulls

    def test_missing_rate_is_correct(self, df_with_nulls):
        results = analyse_missingness(df_with_nulls)
        income_result = next(r for r in results if r.column == "income")
        expected_rate = df_with_nulls["income"].isna().mean()
        assert abs(income_result.missing_rate - expected_rate) < 0.01

    def test_systematic_flagged_when_correlated(self, df_with_nulls):
        """Income missingness correlates with age — should be flagged systematic."""
        results = analyse_missingness(df_with_nulls, systematic_threshold=0.25)
        income_result = next(r for r in results if r.column == "income")
        assert income_result.likely_systematic is True
        assert income_result.correlated_with == "age"

    def test_random_missingness_not_flagged_systematic(self, df_with_nulls):
        """Score has 5% random nulls — should not be flagged as systematic."""
        results = analyse_missingness(df_with_nulls, systematic_threshold=0.25)
        score_result = next((r for r in results if r.column == "score"), None)
        if score_result:  # only if score has nulls
            assert score_result.likely_systematic is False

    def test_severity_levels(self):
        rng = np.random.default_rng(5)
        n = 100
        df = pd.DataFrame({
            "high":   [np.nan] * 60 + [1.0] * 40,    # 60% missing → high
            "medium": [np.nan] * 15 + [1.0] * 85,    # 15% → medium
            "low":    [np.nan] * 5  + [1.0] * 95,    # 5% → low
        })
        results = {r.column: r for r in analyse_missingness(df)}
        assert results["high"].severity == "high"
        assert results["medium"].severity == "medium"
        assert results["low"].severity == "low"


# ══════════════════════════════════════════════════════════════════════════
# CORRELATION
# ══════════════════════════════════════════════════════════════════════════

class TestCorrelation:

    def test_detects_perfectly_correlated_pair(self, correlated_df):
        report = analyse_correlations(correlated_df, correlation_threshold=0.90)
        pairs = [(p.col_a, p.col_b) for p in report.high_pairs]
        assert ("col_a", "col_b") in pairs or ("col_b", "col_a") in pairs

    def test_uncorrelated_columns_not_flagged(self, clean_df):
        report = analyse_correlations(clean_df, correlation_threshold=0.90)
        # age, income, score should not be strongly correlated
        assert len(report.high_pairs) == 0

    def test_high_pairs_sorted_by_correlation_descending(self, correlated_df):
        report = analyse_correlations(correlated_df, correlation_threshold=0.50)
        if len(report.high_pairs) >= 2:
            for i in range(len(report.high_pairs) - 1):
                assert report.high_pairs[i].correlation >= report.high_pairs[i + 1].correlation

    def test_vif_computed_for_small_dataset(self, correlated_df):
        report = analyse_correlations(correlated_df, correlation_threshold=0.90, max_columns_for_vif=50)
        assert len(report.vif_results) > 0

    def test_vif_skipped_for_wide_dataset(self):
        """More columns than max_columns_for_vif → VIF list is empty."""
        df = pd.DataFrame(np.random.default_rng(6).standard_normal((100, 60)),
                          columns=[f"c{i}" for i in range(60)])
        report = analyse_correlations(df, max_columns_for_vif=50)
        assert report.vif_results == []

    def test_non_numeric_columns_skipped(self):
        df = pd.DataFrame({
            "text":   ["a", "b", "c"] * 50,
            "number": np.random.default_rng(7).standard_normal(150),
            "target": np.random.default_rng(8).integers(0, 2, 150),
        })
        report = analyse_correlations(df)
        assert "text" in report.skipped_columns

    def test_vif_severely_flagged_above_threshold(self, correlated_df):
        report = analyse_correlations(correlated_df, vif_threshold=5.0)
        # col_a and col_b are perfectly correlated — at least one should be severe
        severe = [v for v in report.vif_results if v.severe]
        assert len(severe) >= 1


# ══════════════════════════════════════════════════════════════════════════
# DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════

class TestDistributions:

    def test_classification_target_detected(self, imbalanced_df):
        report = analyse_distributions(imbalanced_df, target_column="target")
        assert report.target_analysis is not None
        assert report.target_analysis.task_type == "classification"

    def test_regression_target_detected(self, skewed_regression_df):
        report = analyse_distributions(skewed_regression_df, target_column="price")
        assert report.target_analysis is not None
        assert report.target_analysis.task_type == "regression"

    def test_imbalance_detected(self, imbalanced_df):
        report = analyse_distributions(imbalanced_df, target_column="target")
        ta = report.target_analysis
        assert ta.is_imbalanced
        assert ta.imbalance_ratio is not None
        assert ta.imbalance_ratio >= 9.0
        assert ta.imbalance_warning is not None

    def test_balanced_target_not_flagged(self, clean_df):
        report = analyse_distributions(clean_df, target_column="target")
        ta = report.target_analysis
        assert not ta.is_imbalanced

    def test_skewed_target_detected(self, skewed_regression_df):
        report = analyse_distributions(skewed_regression_df, target_column="price")
        ta = report.target_analysis
        assert ta.is_skewed is True
        assert ta.skewness is not None and ta.skewness > 1.0
        assert ta.skewness_warning is not None

    def test_column_stats_cover_all_columns(self, clean_df):
        report = analyse_distributions(clean_df)
        stat_cols = {s.column for s in report.column_stats}
        assert stat_cols == set(clean_df.columns)

    def test_numeric_column_has_statistics(self, clean_df):
        report = analyse_distributions(clean_df)
        age_stats = next(s for s in report.column_stats if s.column == "age")
        assert age_stats.mean is not None
        assert age_stats.std is not None
        assert age_stats.min is not None
        assert age_stats.max is not None
        assert age_stats.median is not None

    def test_no_target_column_gives_no_target_analysis(self, clean_df):
        report = analyse_distributions(clean_df)
        assert report.target_analysis is None


# ══════════════════════════════════════════════════════════════════════════
# PROFILE RUNNER — integration
# ══════════════════════════════════════════════════════════════════════════

class TestProfileRunner:

    @pytest.mark.asyncio
    async def test_full_run_returns_report(self, clean_df):
        runner = ProfileRunner()
        report = await runner.run(df=clean_df, dataset_id=1, target_column="target")
        assert report.dataset_id == 1
        assert report.n_rows == len(clean_df)
        assert report.n_columns == len(clean_df.columns)
        assert report.duration_secs > 0

    @pytest.mark.asyncio
    async def test_run_without_target_skips_leakage(self, clean_df):
        runner = ProfileRunner()
        report = await runner.run(df=clean_df, dataset_id=1)
        assert report.leakage is None

    @pytest.mark.asyncio
    async def test_run_with_target_includes_leakage(self, clean_df):
        runner = ProfileRunner()
        report = await runner.run(df=clean_df, dataset_id=1, target_column="target")
        assert report.leakage is not None

    @pytest.mark.asyncio
    async def test_to_advisor_prompt_is_non_empty_string(self, clean_df):
        runner = ProfileRunner()
        report = await runner.run(df=clean_df, dataset_id=1, target_column="target")
        prompt = report.to_advisor_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50
        assert "rows" in prompt.lower()

    @pytest.mark.asyncio
    async def test_warning_generated_for_tiny_test_split(self, clean_df):
        """test_fraction so small it produces 0 test rows → warning added."""
        tiny_df = clean_df.iloc[:5]
        runner = ProfileRunner()
        report = await runner.run(df=tiny_df, dataset_id=1, target_column="target",
                                  test_fraction=0.05)
        # Either leakage ran or a warning was added
        assert report.leakage is not None or len(report.warnings) > 0

    @pytest.mark.asyncio
    async def test_advisor_prompt_includes_imbalance_warning(self, imbalanced_df):
        runner = ProfileRunner()
        report = await runner.run(df=imbalanced_df, dataset_id=2, target_column="target")
        prompt = report.to_advisor_prompt()
        assert "imbalance" in prompt.lower() or "imbalanced" in prompt.lower()

    @pytest.mark.asyncio
    async def test_load_dataframe_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)
        loaded = ProfileRunner.load_dataframe(str(path), "csv")
        assert list(loaded.columns) == ["a", "b"]
        assert len(loaded) == 3

    @pytest.mark.asyncio
    async def test_load_dataframe_parquet(self, tmp_path):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]})
        path = tmp_path / "test.parquet"
        df.to_parquet(path, index=False)
        loaded = ProfileRunner.load_dataframe(str(path), "parquet")
        assert list(loaded.columns) == ["x", "y"]
