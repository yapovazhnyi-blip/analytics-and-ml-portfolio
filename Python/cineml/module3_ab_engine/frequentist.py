"""
Module 3 — Frequentist A/B testing.

Implements:
  - Two-proportion z-test (CTR, conversion)
  - Welch's t-test (continuous metrics)
  - Sequential probability ratio test (SPRT) for peeking-safe monitoring
  - Sample size / power calculator

References:
  Kohavi, Tang, Xu (2020) — Trustworthy Online Controlled Experiments
  Wald (1945) — Sequential Analysis
"""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)


# ── Data containers ────────────────────────────────────────────────────────────

@dataclass
class ABGroups:
    control: pd.Series      # metric values for control group
    treatment: pd.Series    # metric values for treatment group
    metric_name: str = "metric"
    is_binary: bool = True  # True = proportion test, False = means test


@dataclass
class FrequentistResult:
    metric_name: str
    control_mean: float
    treatment_mean: float
    absolute_lift: float
    relative_lift_pct: float
    p_value: float
    test_statistic: float
    ci_lower: float          # 95% CI on the difference
    ci_upper: float
    significant: bool
    alpha: float
    test_type: str
    n_control: int
    n_treatment: int


# ── Two-proportion z-test ─────────────────────────────────────────────────────

def proportion_test(groups: ABGroups, alpha: float = 0.05) -> FrequentistResult:
    """
    Two-proportion z-test for binary metrics (CTR, completion rate, etc.).
    H0: p_control == p_treatment
    """
    n_c, n_t = len(groups.control), len(groups.treatment)
    p_c = groups.control.mean()
    p_t = groups.treatment.mean()

    # Pooled proportion under H0
    p_pool = (groups.control.sum() + groups.treatment.sum()) / (n_c + n_t)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))

    z = (p_t - p_c) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Delta-method 95% CI
    se_delta = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    diff = p_t - p_c
    ci_lower = diff - 1.96 * se_delta
    ci_upper = diff + 1.96 * se_delta

    return FrequentistResult(
        metric_name=groups.metric_name,
        control_mean=p_c,
        treatment_mean=p_t,
        absolute_lift=diff,
        relative_lift_pct=(diff / p_c * 100) if p_c > 0 else 0.0,
        p_value=p_value,
        test_statistic=z,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        significant=p_value < alpha,
        alpha=alpha,
        test_type="two-proportion z-test",
        n_control=n_c,
        n_treatment=n_t,
    )


# ── Welch's t-test ────────────────────────────────────────────────────────────

def means_test(groups: ABGroups, alpha: float = 0.05) -> FrequentistResult:
    """Welch's t-test for continuous metrics (watch time, dwell, etc.)."""
    t_stat, p_value = stats.ttest_ind(groups.treatment, groups.control, equal_var=False)

    mu_c = groups.control.mean()
    mu_t = groups.treatment.mean()
    diff = mu_t - mu_c

    n_c, n_t = len(groups.control), len(groups.treatment)
    se = np.sqrt(groups.control.var() / n_c + groups.treatment.var() / n_t)
    ci_lower = diff - 1.96 * se
    ci_upper = diff + 1.96 * se

    return FrequentistResult(
        metric_name=groups.metric_name,
        control_mean=mu_c,
        treatment_mean=mu_t,
        absolute_lift=diff,
        relative_lift_pct=(diff / mu_c * 100) if mu_c != 0 else 0.0,
        p_value=p_value,
        test_statistic=t_stat,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        significant=p_value < alpha,
        alpha=alpha,
        test_type="welch-t-test",
        n_control=n_c,
        n_treatment=n_t,
    )


# ── Sequential testing (SPRT) ─────────────────────────────────────────────────

def sprt_monitor(
    control_stream: pd.Series,
    treatment_stream: pd.Series,
    mde: float = 0.02,
    alpha: float = 0.05,
    power: float = 0.80,
) -> pd.DataFrame:
    """
    Run a Sequential Probability Ratio Test over time.
    Allows valid inference at any peek — no p-hacking inflation.

    Returns a DataFrame with columns: [n, llr, decision]
    where decision ∈ {null, reject_h0, accept_h0}
    """
    A = np.log((1 - alpha) / power)          # lower boundary
    B = np.log((1 - power) / alpha)           # upper boundary — negate for convention

    combined = pd.DataFrame({
        "value": pd.concat([control_stream, treatment_stream]).values,
        "arm": ["control"] * len(control_stream) + ["treatment"] * len(treatment_stream),
    }).sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle time order

    p0 = control_stream.mean()
    p1 = p0 + mde

    llr_cumulative = 0.0
    records = []

    for i, row in combined.iterrows():
        x = row["value"]
        if row["arm"] == "treatment":
            llr_cumulative += x * np.log(p1 / (p0 + 1e-12)) + (1 - x) * np.log((1 - p1) / (1 - p0 + 1e-12))

        if llr_cumulative >= -A:
            decision = "reject_h0"
        elif llr_cumulative <= B:
            decision = "accept_h0"
        else:
            decision = "continue"

        records.append({"n": i + 1, "llr": llr_cumulative, "decision": decision})

        if decision != "continue":
            break

    return pd.DataFrame(records)


# ── Sample size calculator ────────────────────────────────────────────────────

def required_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict[str, int | float]:
    """
    Compute required n per arm for a two-proportion test.

    Args:
        baseline_rate : expected control CTR
        mde           : minimum detectable effect (absolute)
        alpha         : type-I error rate
        power         : 1 - type-II error rate
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    p1, p2 = baseline_rate, baseline_rate + mde
    pooled = (p1 + p2) / 2

    n = (
        (z_alpha * np.sqrt(2 * pooled * (1 - pooled)) + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        / (mde ** 2)
    )
    return {
        "n_per_arm": int(np.ceil(n)),
        "n_total": int(np.ceil(n)) * 2,
        "baseline_rate": baseline_rate,
        "mde": mde,
        "alpha": alpha,
        "power": power,
    }


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_frequentist_analysis(events: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Run the full frequentist battery on streaming event data.

    Args:
        events: DataFrame with columns [arm, event_type, dwell_ms, user_id]
    """
    impressions = events[events["event_type"] == "impression"]
    clicks = events[events["event_type"].isin(["click", "completion", "skip", "watchlist_add"])]

    # Build per-user metrics
    user_ctr = (
        impressions.assign(clicked=impressions["user_id"].isin(clicks["user_id"]))
        .groupby(["user_id", "arm"])["clicked"]
        .mean()
        .reset_index()
    )
    ctrl = user_ctr[user_ctr["arm"] == "control"]["clicked"]
    trt = user_ctr[user_ctr["arm"] == "treatment"]["clicked"]

    ctr_result = proportion_test(
        ABGroups(control=ctrl, treatment=trt, metric_name="CTR", is_binary=True), alpha=alpha
    )

    dwell = events[events["event_type"] == "click"].copy()
    dwell_ctrl = dwell[dwell["arm"] == "control"]["dwell_ms"] / 1000
    dwell_trt = dwell[dwell["arm"] == "treatment"]["dwell_ms"] / 1000
    dwell_result = means_test(
        ABGroups(control=dwell_ctrl, treatment=dwell_trt, metric_name="avg_dwell_seconds", is_binary=False)
    )

    sample_info = required_sample_size(baseline_rate=ctrl.mean(), mde=0.01)

    return {
        "ctr": ctr_result,
        "dwell": dwell_result,
        "sample_size_recommendation": sample_info,
    }
