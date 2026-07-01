"""
A/B Testing Engine — statistical comparison of two trained ML models.

THE CORE QUESTION
-----------------
Given model A with holdout accuracy 0.871 and model B with holdout accuracy
0.884, is model B genuinely better or did it get lucky on this particular
test split?

Answering this correctly requires a proper statistical test, not a simple
"higher is better" comparison. With a 500-sample holdout set, a 1.3pp
difference is easily within the noise floor of random sampling variation.

THREE TESTS IMPLEMENTED
-----------------------

McNemar's Test (for classification, when predictions are available)
  The gold standard for comparing two classifiers on the same test set.
  It looks only at the cases where the classifiers disagree — if model A
  got 30 samples right that B got wrong, and B got 40 samples right that
  A got wrong, there's evidence B is better. Uses a chi-squared statistic
  on the (b, c) off-diagonal elements of the disagreement contingency table.
  Advantage: uses the paired structure of the predictions (same test samples)
  for higher statistical power than unpaired tests.
  Reference: McNemar, Q. (1947). Note on the sampling error of the difference
  between correlated proportions or percentages.

Bootstrap Test (metric-agnostic fallback)
  Resamples the test set predictions with replacement N times and computes
  the metric difference for each resample. The p-value is the fraction of
  bootstrap samples where B does not beat A. The confidence interval is
  the 2.5th–97.5th percentile of the bootstrap distribution of differences.
  Advantage: works for any metric (accuracy, F1, R², MAE) without
  distributional assumptions. Uses the same indices for both models on each
  resample, preserving the paired structure.

Wilcoxon Signed-Rank Test (for regression)
  A non-parametric test for comparing paired samples. Ranks the absolute
  differences between models' errors and tests whether the signed ranks
  differ from zero. Used instead of a paired t-test because residuals are
  not normally distributed in general.

POWER ANALYSIS
--------------
Given a test set of n samples, the minimum detectable effect (MDE) tells
you the smallest true improvement you could detect with a given Type I
error rate (alpha) and statistical power (1 - beta).

If your test set is small (n < 300), you may not be able to detect
differences of less than 3–5pp even if they're real. Power analysis
surfaces this limitation explicitly so practitioners know when they need
more data or to accept a higher Type I error rate.

EFFECT SIZE
-----------
Statistical significance does not imply practical significance. A 0.1pp
accuracy improvement may be statistically significant with n=100,000 but
operationally irrelevant. Cohen's h (for proportions) and Cohen's d
(for continuous metrics) classify effect sizes as:
  negligible  < 0.10
  small       0.10–0.30
  medium      0.30–0.50
  large       > 0.50
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ABTestResult:
    experiment_a_id: int
    experiment_b_id: int
    metric: str

    score_a: float
    score_b: float
    absolute_diff: float      # score_b - score_a  (positive = B is better)
    relative_diff_pct: float  # % improvement of B over A

    p_value: float
    confidence_level: float
    ci_lower: float           # lower bound of 95% CI on the difference
    ci_upper: float           # upper bound of 95% CI on the difference

    is_significant: bool
    winner: Optional[str]     # "A" | "B" | None (no significant difference)
    winner_id: Optional[int]

    effect_size: float
    effect_size_label: str    # negligible | small | medium | large

    statistical_test: str     # "mcnemar" | "bootstrap" | "wilcoxon"
    n_samples: int
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "experiment_a_id":  self.experiment_a_id,
            "experiment_b_id":  self.experiment_b_id,
            "metric":           self.metric,
            "score_a":          round(self.score_a, 4),
            "score_b":          round(self.score_b, 4),
            "absolute_diff":    round(self.absolute_diff, 4),
            "relative_diff_pct": round(self.relative_diff_pct, 2),
            "p_value":          round(self.p_value, 4),
            "confidence_level": self.confidence_level,
            "ci_lower":         round(self.ci_lower, 4),
            "ci_upper":         round(self.ci_upper, 4),
            "is_significant":   self.is_significant,
            "winner":           self.winner,
            "winner_id":        self.winner_id,
            "effect_size":      round(self.effect_size, 4),
            "effect_size_label": self.effect_size_label,
            "statistical_test": self.statistical_test,
            "n_samples":        self.n_samples,
            "recommendation":   self.recommendation,
        }


@dataclass
class PowerAnalysisResult:
    baseline_rate: float
    minimum_effect: float
    alpha: float
    power: float
    required_n: int
    current_n: int
    current_mde: float        # MDE achievable with current_n
    is_adequately_powered: bool

    def to_dict(self) -> dict:
        return {
            "baseline_rate":          round(self.baseline_rate, 4),
            "minimum_effect":         round(self.minimum_effect, 4),
            "alpha":                  self.alpha,
            "power":                  self.power,
            "required_n":             self.required_n,
            "current_n":              self.current_n,
            "current_mde":            round(self.current_mde, 4),
            "current_mde_pct":        round(self.current_mde * 100, 2),
            "is_adequately_powered":  self.is_adequately_powered,
        }


# ── Effect size ───────────────────────────────────────────────────────────────

def _cohen_h(p1: float, p2: float) -> float:
    """Cohen's h: effect size for comparing two proportions."""
    return float(2 * abs(math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2))))

def _effect_label(h: float) -> str:
    h = abs(h)
    if h < 0.10: return "negligible"
    if h < 0.30: return "small"
    if h < 0.50: return "medium"
    return "large"


# ══════════════════════════════════════════════════════════════════════════
# CLASSIFICATION TESTS
# ══════════════════════════════════════════════════════════════════════════

def mcnemar_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    confidence_level: float = 0.95,
) -> dict:
    """
    McNemar's test for paired nominal data.

    Builds the 2×2 contingency table of A/B disagreements and applies
    the chi-squared test with continuity correction (Yates's correction)
    for small sample robustness.

    Returns a dict with p_value, ci_lower, ci_upper (on accuracy difference),
    effect_size, and statistical_test = "mcnemar".
    """
    a_correct = (preds_a == y_true)
    b_correct = (preds_b == y_true)

    # Disagreement counts: b=only A correct, c=only B correct
    b_count = int(np.sum(a_correct & ~b_correct))
    c_count = int(np.sum(~a_correct & b_correct))
    n_disagree = b_count + c_count

    if n_disagree == 0:
        # Identical predictions — no test possible
        return {
            "p_value": 1.0,
            "ci_lower": 0.0, "ci_upper": 0.0,
            "effect_size": 0.0,
            "statistical_test": "mcnemar",
        }

    # McNemar statistic with continuity correction
    statistic = (abs(b_count - c_count) - 1) ** 2 / n_disagree
    p_value = float(stats.chi2.sf(statistic, df=1))

    # Confidence interval on the accuracy difference (Agresti-Min method)
    n = len(y_true)
    acc_a = float(np.mean(a_correct))
    acc_b = float(np.mean(b_correct))
    alpha = 1 - confidence_level
    z = stats.norm.ppf(1 - alpha / 2)

    # Standard error of the difference in paired proportions
    se = math.sqrt((b_count + c_count - (b_count - c_count) ** 2 / n) / n ** 2)
    diff = acc_b - acc_a
    ci_lower = diff - z * se
    ci_upper = diff + z * se

    effect_size = _cohen_h(acc_a, acc_b)

    return {
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "effect_size": effect_size,
        "statistical_test": "mcnemar",
    }


# ══════════════════════════════════════════════════════════════════════════
# BOOTSTRAP TEST (metric-agnostic)
# ══════════════════════════════════════════════════════════════════════════

def bootstrap_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    metric_fn,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Bootstrap test for any scalar metric.

    On each bootstrap iteration, the same resampled indices are applied to
    both models' predictions, preserving the paired structure. The distribution
    of metric(B) - metric(A) over bootstrap samples gives a CI and a p-value.

    p-value = fraction of bootstrap samples where B does NOT beat A.
    CI      = percentile CI of the bootstrap difference distribution.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            score_a = metric_fn(y_true[idx], preds_a[idx])
            score_b = metric_fn(y_true[idx], preds_b[idx])
            diffs[i] = score_b - score_a
        except Exception:
            diffs[i] = 0.0

    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(diffs, alpha / 2 * 100))
    ci_upper = float(np.percentile(diffs, (1 - alpha / 2) * 100))
    p_value  = float(np.mean(diffs <= 0))   # fraction where B is NOT better

    score_a = float(metric_fn(y_true, preds_a))
    score_b = float(metric_fn(y_true, preds_b))
    effect_size = _cohen_h(max(0, score_a), max(0, score_b))

    return {
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "effect_size": effect_size,
        "statistical_test": "bootstrap",
    }


# ══════════════════════════════════════════════════════════════════════════
# REGRESSION TEST
# ══════════════════════════════════════════════════════════════════════════

def wilcoxon_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    confidence_level: float = 0.95,
) -> dict:
    """
    Wilcoxon signed-rank test for regression: compares |errors| paired by sample.

    Tests whether model B's absolute errors are systematically smaller than A's.
    Non-parametric — no normality assumption on residuals.
    """
    errors_a = np.abs(y_true - preds_a)
    errors_b = np.abs(y_true - preds_b)
    diffs    = errors_a - errors_b   # positive = B is better (smaller error)

    if np.all(diffs == 0):
        return {"p_value": 1.0, "ci_lower": 0.0, "ci_upper": 0.0,
                "effect_size": 0.0, "statistical_test": "wilcoxon"}

    try:
        _, p_value = stats.wilcoxon(diffs, alternative="greater")
    except Exception:
        p_value = 1.0

    # Bootstrap CI on the MAE difference
    n = len(y_true)
    rng = np.random.default_rng(42)
    mae_diffs = []
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        mae_diffs.append(np.mean(errors_a[idx]) - np.mean(errors_b[idx]))

    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(mae_diffs, alpha / 2 * 100))
    ci_upper = float(np.percentile(mae_diffs, (1 - alpha / 2) * 100))

    # Effect size: rank-biserial correlation
    w_stat, _ = stats.wilcoxon(diffs)
    n_nonzero = np.sum(diffs != 0)
    r = 1 - 4 * w_stat / (n_nonzero * (n_nonzero + 1)) if n_nonzero > 0 else 0.0

    return {
        "p_value": float(p_value),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "effect_size": abs(r),
        "statistical_test": "wilcoxon",
    }


# ══════════════════════════════════════════════════════════════════════════
# POWER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def power_analysis(
    baseline_rate: float,
    minimum_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
    current_n: int = 100,
) -> PowerAnalysisResult:
    """
    Two-proportion z-test power analysis.

    Given a baseline performance rate and the minimum practically
    significant effect, calculates:
      1. The minimum sample size needed to detect that effect.
      2. The minimum detectable effect (MDE) with the current sample size.

    Args:
        baseline_rate:  Current model's metric (e.g. 0.85 for 85% accuracy).
        minimum_effect: The smallest improvement worth detecting (e.g. 0.02).
        alpha:          Type I error rate (probability of false positive).
        power:          1 - beta (probability of detecting a real effect).
        current_n:      The current holdout set size.

    Returns:
        PowerAnalysisResult with required_n and current_mde.
    """
    p2 = min(0.9999, baseline_rate + minimum_effect)
    p1 = baseline_rate

    # Required n per group (z-test for two proportions)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)

    # Pooled standard error under H0
    p_bar = (p1 + p2) / 2
    se_h0 = math.sqrt(2 * p_bar * (1 - p_bar))
    se_h1 = math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))

    required_n = math.ceil(
        ((z_alpha * se_h0 + z_beta * se_h1) / abs(p2 - p1)) ** 2
    )

    # MDE with current sample size (binary search)
    def _power_at_effect(effect: float) -> float:
        """Compute statistical power for a given effect size and current_n."""
        p2_ = min(0.9999, baseline_rate + effect)
        se0 = math.sqrt(2 * ((baseline_rate + p2_) / 2) * (1 - (baseline_rate + p2_) / 2))
        se1 = math.sqrt(baseline_rate * (1 - baseline_rate) + p2_ * (1 - p2_))
        z_crit = z_alpha * se0 / math.sqrt(current_n) if current_n > 0 else 0
        z_score = (abs(p2_ - baseline_rate) - z_crit) / (se1 / math.sqrt(current_n)) if current_n > 0 else 0
        return stats.norm.cdf(z_score)

    # Binary search for MDE
    lo, hi = 0.001, 0.5
    for _ in range(40):
        mid = (lo + hi) / 2
        if _power_at_effect(mid) >= power:
            hi = mid
        else:
            lo = mid
    current_mde = round(hi, 4)

    return PowerAnalysisResult(
        baseline_rate=baseline_rate,
        minimum_effect=minimum_effect,
        alpha=alpha,
        power=power,
        required_n=required_n,
        current_n=current_n,
        current_mde=current_mde,
        is_adequately_powered=(current_n >= required_n),
    )


# ══════════════════════════════════════════════════════════════════════════
# MAIN COMPARISON FUNCTION
# ══════════════════════════════════════════════════════════════════════════

def compare_experiments(
    exp_a_id: int,
    exp_b_id: int,
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    task_type: str,
    metric: str = "auto",
    confidence_level: float = 0.95,
) -> ABTestResult:
    """
    Compares two models on the same test set using the appropriate statistical test.

    Selects the test automatically:
      classification + binary labels → McNemar's test (most powerful)
      classification + probability labels → bootstrap on accuracy
      regression → Wilcoxon signed-rank test

    Args:
        exp_a_id / exp_b_id: IDs for labelling the result.
        y_true:              Ground truth labels/values (holdout set).
        preds_a / preds_b:   Model A and B predictions on the same samples.
        task_type:           "classification" or "regression".
        metric:              "auto" | "accuracy" | "mae" | "rmse"
        confidence_level:    Confidence level for the CI (default 0.95).

    Returns:
        ABTestResult with statistical test output and a plain-English recommendation.
    """
    n = len(y_true)
    alpha = 1 - confidence_level

    if task_type == "regression":
        # Primary metric: MAE (lower = better, so B wins when diff < 0)
        mae_a = float(np.mean(np.abs(y_true - preds_a)))
        mae_b = float(np.mean(np.abs(y_true - preds_b)))
        test_result = wilcoxon_test(y_true, preds_a, preds_b, confidence_level)
        score_a, score_b = mae_a, mae_b
        actual_metric = "mae"
        # For regression, B wins if MAE_B < MAE_A (lower error is better)
        b_is_better = mae_b < mae_a
        abs_diff = mae_b - mae_a
        rel_diff_pct = abs_diff / max(mae_a, 1e-10) * 100

    else:
        # Classification: accuracy as primary metric
        acc_a = float(np.mean(preds_a == y_true))
        acc_b = float(np.mean(preds_b == y_true))
        test_result = mcnemar_test(y_true, preds_a, preds_b, confidence_level)
        score_a, score_b = acc_a, acc_b
        actual_metric = "accuracy"
        b_is_better = acc_b > acc_a
        abs_diff = acc_b - acc_a
        rel_diff_pct = abs_diff / max(acc_a, 1e-10) * 100

    p_value   = test_result["p_value"]
    ci_lower  = test_result["ci_lower"]
    ci_upper  = test_result["ci_upper"]
    eff_size  = test_result["effect_size"]
    stat_test = test_result["statistical_test"]

    is_significant = p_value < alpha
    winner: Optional[str] = None
    winner_id: Optional[int] = None

    if is_significant:
        if b_is_better:
            winner, winner_id = "B", exp_b_id
        else:
            winner, winner_id = "A", exp_a_id

    eff_label = _effect_label(eff_size)

    recommendation = _make_recommendation(
        score_a=score_a, score_b=score_b, abs_diff=abs_diff,
        p_value=p_value, is_significant=is_significant,
        winner=winner, metric=actual_metric, task_type=task_type,
        n_samples=n, eff_label=eff_label, confidence_level=confidence_level,
        ci_lower=ci_lower, ci_upper=ci_upper,
    )

    return ABTestResult(
        experiment_a_id=exp_a_id,
        experiment_b_id=exp_b_id,
        metric=actual_metric,
        score_a=score_a,
        score_b=score_b,
        absolute_diff=abs_diff,
        relative_diff_pct=rel_diff_pct,
        p_value=p_value,
        confidence_level=confidence_level,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        is_significant=is_significant,
        winner=winner,
        winner_id=winner_id,
        effect_size=eff_size,
        effect_size_label=eff_label,
        statistical_test=stat_test,
        n_samples=n,
        recommendation=recommendation,
    )


def _make_recommendation(
    score_a, score_b, abs_diff, p_value, is_significant,
    winner, metric, task_type, n_samples, eff_label,
    confidence_level, ci_lower, ci_upper,
) -> str:
    alpha = 1 - confidence_level
    ci_pct = int(confidence_level * 100)

    if not is_significant:
        return (
            f"No statistically significant difference detected (p={p_value:.3f}, α={alpha:.2f}). "
            f"The observed {metric} difference of {abs(abs_diff):.3f} is within the noise floor "
            f"for a test set of {n_samples} samples. "
            f"The {ci_pct}% CI on the difference is [{ci_lower:.3f}, {ci_upper:.3f}]. "
            f"Deploy either model based on non-metric criteria (latency, size, maintainability). "
            f"To detect differences this small reliably, increase the test set size or run cross-validation."
        )

    dir_word = "improvement" if (winner == "B") else "regression"
    better   = "B" if winner == "B" else "A"
    worse    = "A" if winner == "B" else "B"

    return (
        f"Experiment {better} shows a statistically significant {dir_word} over {worse} "
        f"(p={p_value:.4f} < α={alpha:.2f}). "
        f"The {metric} difference is {abs(abs_diff):.4f} "
        f"({ci_pct}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]). "
        f"Effect size: {eff_label} (Cohen's h = {abs_diff:.3f}). "
        f"Recommendation: deploy Experiment {better}. "
        f"Continue monitoring in production — holdout metrics may not generalise if the data distribution shifts."
    )
