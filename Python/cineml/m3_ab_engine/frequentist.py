"""
m3_ab_engine/frequentist.py
Frequentist A/B testing: t-test, z-test, and sequential testing (mSPRT).

All functions are pure (no side effects) and return typed result objects
so they can be consumed by the FastAPI service and Streamlit dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.stats as stats


@dataclass
class FrequentistResult:
    metric: str
    control_mean: float
    treatment_mean: float
    relative_lift: float          # (treatment - control) / control
    t_statistic: float
    p_value: float
    ci_lower: float               # 95% CI on the difference
    ci_upper: float
    is_significant: bool
    power: float
    sample_size_control: int
    sample_size_treatment: int


def two_sample_ttest(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> FrequentistResult:
    """Welch's t-test for difference in means."""
    t_stat, p_val = stats.ttest_ind(treatment, control, equal_var=False, alternative=alternative)
    n_c, n_t = len(control), len(treatment)
    mu_c, mu_t = control.mean(), treatment.mean()
    se = np.sqrt(control.var(ddof=1) / n_c + treatment.var(ddof=1) / n_t)
    df = (control.var(ddof=1) / n_c + treatment.var(ddof=1) / n_t) ** 2 / (
        (control.var(ddof=1) / n_c) ** 2 / (n_c - 1)
        + (treatment.var(ddof=1) / n_t) ** 2 / (n_t - 1)
    )
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_lower = (mu_t - mu_c) - t_crit * se
    ci_upper = (mu_t - mu_c) + t_crit * se

    # Post-hoc power
    ncp = abs(mu_t - mu_c) / se
    power = 1 - stats.t.cdf(t_crit, df, loc=ncp) + stats.t.cdf(-t_crit, df, loc=ncp)

    return FrequentistResult(
        metric="mean_difference",
        control_mean=float(mu_c),
        treatment_mean=float(mu_t),
        relative_lift=float((mu_t - mu_c) / mu_c) if mu_c != 0 else 0.0,
        t_statistic=float(t_stat),
        p_value=float(p_val),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        is_significant=bool(p_val < alpha),
        power=float(power),
        sample_size_control=n_c,
        sample_size_treatment=n_t,
    )


def required_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Compute required per-arm sample size for a two-proportion z-test.
    Uses the standard formula from Cohen (1988).
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p1 = baseline_rate
    p2 = baseline_rate * (1 + minimum_detectable_effect)
    p_bar = (p1 + p2) / 2
    n = ((z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * np.sqrt(
        p1 * (1 - p1) + p2 * (1 - p2)
    )) / (p2 - p1)) ** 2
    return int(np.ceil(n))


# ── Sequential testing (mSPRT) ────────────────────────────────────────────────

@dataclass
class SequentialTestState:
    """Mutable state for the mixture Sequential Probability Ratio Test."""
    n_control: int = 0
    n_treatment: int = 0
    sum_control: float = 0.0
    sum_treatment: float = 0.0
    sum_sq_control: float = 0.0
    sum_sq_treatment: float = 0.0
    log_lambda: float = 0.0     # log mixture likelihood ratio
    stopped: bool = False
    decision: str | None = None


def update_sequential_test(
    state: SequentialTestState,
    control_batch: np.ndarray,
    treatment_batch: np.ndarray,
    alpha: float = 0.05,
    tau: float = 1.0,            # mixing variance for mSPRT
) -> SequentialTestState:
    """
    Update the sequential test state with a new batch of observations.
    Implements a simplified Gaussian mSPRT (Johari et al. 2015).

    Early-stop when log_lambda > log(1/alpha) or < log(alpha).
    """
    state.n_control += len(control_batch)
    state.n_treatment += len(treatment_batch)
    state.sum_control += control_batch.sum()
    state.sum_treatment += treatment_batch.sum()
    state.sum_sq_control += (control_batch ** 2).sum()
    state.sum_sq_treatment += (treatment_batch ** 2).sum()

    if state.n_control < 2 or state.n_treatment < 2:
        return state

    mu_c = state.sum_control / state.n_control
    mu_t = state.sum_treatment / state.n_treatment
    var_c = max(state.sum_sq_control / state.n_control - mu_c ** 2, 1e-9)
    var_t = max(state.sum_sq_treatment / state.n_treatment - mu_t ** 2, 1e-9)
    se = np.sqrt(var_c / state.n_control + var_t / state.n_treatment)

    # log mixture likelihood ratio increment
    diff = mu_t - mu_c
    state.log_lambda += (diff ** 2 / (2 * se ** 2 * (1 + tau ** 2 / se ** 2)))

    threshold = np.log(1 / alpha)
    if state.log_lambda > threshold:
        state.stopped = True
        state.decision = "reject_null"
    elif state.log_lambda < -threshold:
        state.stopped = True
        state.decision = "accept_null"

    return state
