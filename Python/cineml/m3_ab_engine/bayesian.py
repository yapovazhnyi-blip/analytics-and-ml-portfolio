"""
m3_ab_engine/bayesian.py
Bayesian A/B testing via Beta-Binomial conjugate model and PyMC for
continuous outcomes. Computes posterior distributions, credible intervals,
and probability of treatment being better.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.stats as stats


@dataclass
class BetaBinomialResult:
    metric: str
    alpha_control: float       # posterior Beta(alpha, beta) parameters
    beta_control: float
    alpha_treatment: float
    beta_treatment: float
    posterior_mean_control: float
    posterior_mean_treatment: float
    prob_treatment_better: float   # P(θ_t > θ_c | data)
    credible_interval_lower: float # 95% HDI on (θ_t - θ_c)
    credible_interval_upper: float
    expected_loss_control: float
    expected_loss_treatment: float
    n_samples: int = 100_000


def beta_binomial_test(
    control_successes: int,
    control_trials: int,
    treatment_successes: int,
    treatment_trials: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_samples: int = 100_000,
    seed: int = 42,
) -> BetaBinomialResult:
    """
    Conjugate Beta-Binomial Bayesian test for binary outcomes (e.g. CTR).

    Posterior:
        θ_c | data ~ Beta(alpha + s_c, beta + f_c)
        θ_t | data ~ Beta(alpha + s_t, beta + f_t)
    """
    rng = np.random.default_rng(seed)

    a_c = prior_alpha + control_successes
    b_c = prior_beta + (control_trials - control_successes)
    a_t = prior_alpha + treatment_successes
    b_t = prior_beta + (treatment_trials - treatment_successes)

    samples_c = rng.beta(a_c, b_c, size=n_samples)
    samples_t = rng.beta(a_t, b_t, size=n_samples)
    diff_samples = samples_t - samples_c

    prob_better = float((samples_t > samples_c).mean())
    hdi_lower, hdi_upper = _hdi(diff_samples, credible_mass=0.95)

    # Expected loss — decision-theoretic stopping criterion
    loss_c = float(np.maximum(samples_t - samples_c, 0).mean())   # regret of picking control
    loss_t = float(np.maximum(samples_c - samples_t, 0).mean())   # regret of picking treatment

    return BetaBinomialResult(
        metric="conversion_rate",
        alpha_control=a_c, beta_control=b_c,
        alpha_treatment=a_t, beta_treatment=b_t,
        posterior_mean_control=float(a_c / (a_c + b_c)),
        posterior_mean_treatment=float(a_t / (a_t + b_t)),
        prob_treatment_better=prob_better,
        credible_interval_lower=float(hdi_lower),
        credible_interval_upper=float(hdi_upper),
        expected_loss_control=loss_c,
        expected_loss_treatment=loss_t,
        n_samples=n_samples,
    )


def _hdi(samples: np.ndarray, credible_mass: float = 0.95) -> tuple[float, float]:
    """Compute Highest Density Interval via sorted-window method."""
    sorted_s = np.sort(samples)
    n = len(sorted_s)
    n_included = int(np.floor(credible_mass * n))
    n_intervals = n - n_included
    interval_widths = sorted_s[n_included:] - sorted_s[:n_intervals]
    min_idx = int(np.argmin(interval_widths))
    return float(sorted_s[min_idx]), float(sorted_s[min_idx + n_included])


# ── PyMC continuous outcome model (optional — heavier dependency) ─────────────

def pymc_continuous_test(
    control_obs: np.ndarray,
    treatment_obs: np.ndarray,
    n_samples: int = 2000,
    n_chains: int = 2,
) -> dict:
    """
    Bayesian t-test for continuous outcomes (e.g. completion rate, watch time)
    using PyMC with weakly informative priors.

    Returns posterior summary dict with keys:
        mu_control, mu_treatment, diff, prob_treatment_better, rope_probability
    """
    try:
        import pymc as pm
        import arviz as az
    except ImportError:
        raise ImportError("Install pymc and arviz: pip install pymc arviz")

    pooled_sd = np.std(np.concatenate([control_obs, treatment_obs]), ddof=1)

    with pm.Model():
        mu_c = pm.Normal("mu_control", mu=control_obs.mean(), sigma=pooled_sd * 2)
        mu_t = pm.Normal("mu_treatment", mu=treatment_obs.mean(), sigma=pooled_sd * 2)
        sigma_c = pm.HalfNormal("sigma_control", sigma=pooled_sd)
        sigma_t = pm.HalfNormal("sigma_treatment", sigma=pooled_sd)

        pm.Normal("obs_control", mu=mu_c, sigma=sigma_c, observed=control_obs)
        pm.Normal("obs_treatment", mu=mu_t, sigma=sigma_t, observed=treatment_obs)

        diff = pm.Deterministic("diff", mu_t - mu_c)

        trace = pm.sample(
            n_samples,
            chains=n_chains,
            progressbar=False,
            return_inferencedata=True,
        )

    summary = az.summary(trace, var_names=["mu_control", "mu_treatment", "diff"])
    diff_samples = trace.posterior["diff"].values.flatten()
    prob_better = float((diff_samples > 0).mean())

    rope = 0.01 * pooled_sd   # region of practical equivalence = 1% of SD
    rope_prob = float((np.abs(diff_samples) < rope).mean())

    return {
        "mu_control": float(summary.loc["mu_control", "mean"]),
        "mu_treatment": float(summary.loc["mu_treatment", "mean"]),
        "diff_mean": float(summary.loc["diff", "mean"]),
        "diff_hdi_lower": float(summary.loc["diff", "hdi_3%"]),
        "diff_hdi_upper": float(summary.loc["diff", "hdi_97%"]),
        "prob_treatment_better": prob_better,
        "rope_probability": rope_prob,
    }
