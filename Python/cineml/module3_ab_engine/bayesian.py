"""
Module 3 — Bayesian A/B Testing.

Implements:
  - Beta-Binomial conjugate model (fast, closed-form)
  - PyMC hierarchical model (for more complex metrics)
  - Credible intervals and posterior probability of improvement

References:
  Gelman et al. — Bayesian Data Analysis
  VWO whitepaper — Bayesian A/B Testing
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class BayesianResult:
    metric_name: str
    control_posterior_mean: float
    treatment_posterior_mean: float
    prob_treatment_better: float        # P(theta_t > theta_c)
    expected_lift: float
    credible_interval_lower: float
    credible_interval_upper: float
    posterior_samples_control: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    posterior_samples_treatment: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


# ── Beta-Binomial conjugate ───────────────────────────────────────────────────

class BetaBinomialTest:
    """
    Closed-form Bayesian test for binary metrics.

    Prior: Beta(alpha_0, beta_0) — defaults to uninformative Beta(1, 1)
    Posterior: Beta(alpha_0 + successes, beta_0 + failures)
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0, n_samples: int = 100_000):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.n_samples = n_samples

    def fit(
        self,
        n_control: int,
        k_control: int,
        n_treatment: int,
        k_treatment: int,
        metric_name: str = "conversion",
        credible_mass: float = 0.95,
    ) -> BayesianResult:
        """
        Args:
            n_*: total observations per arm
            k_*: successes (clicks, completions, etc.) per arm
        """
        alpha_c = self.prior_alpha + k_control
        beta_c = self.prior_beta + (n_control - k_control)

        alpha_t = self.prior_alpha + k_treatment
        beta_t = self.prior_beta + (n_treatment - k_treatment)

        # Monte Carlo sampling
        rng = np.random.default_rng(42)
        samples_c = rng.beta(alpha_c, beta_c, size=self.n_samples)
        samples_t = rng.beta(alpha_t, beta_t, size=self.n_samples)

        prob_better = float((samples_t > samples_c).mean())
        diff_samples = samples_t - samples_c
        ci_lo = float(np.percentile(diff_samples, (1 - credible_mass) / 2 * 100))
        ci_hi = float(np.percentile(diff_samples, (1 + credible_mass) / 2 * 100))

        return BayesianResult(
            metric_name=metric_name,
            control_posterior_mean=alpha_c / (alpha_c + beta_c),
            treatment_posterior_mean=alpha_t / (alpha_t + beta_t),
            prob_treatment_better=prob_better,
            expected_lift=float(diff_samples.mean()),
            credible_interval_lower=ci_lo,
            credible_interval_upper=ci_hi,
            posterior_samples_control=samples_c,
            posterior_samples_treatment=samples_t,
        )


# ── PyMC hierarchical model ───────────────────────────────────────────────────

def run_pymc_model(
    events: pd.DataFrame,
    metric_col: str = "clicked",
    n_samples: int = 2000,
    n_tune: int = 1000,
) -> BayesianResult:
    """
    Hierarchical Beta-Binomial model in PyMC.
    Allows partial pooling across arms if extended to multiple variants.

    Requires: pip install pymc
    """
    try:
        import pymc as pm
        import arviz as az
    except ImportError:
        raise ImportError("Install pymc and arviz: pip install pymc arviz")

    ctrl = events[events["arm"] == "control"][metric_col].astype(int)
    trt = events[events["arm"] == "treatment"][metric_col].astype(int)

    with pm.Model() as model:
        # Hyperpriors
        mu = pm.Beta("mu", alpha=2, beta=2)
        kappa = pm.Gamma("kappa", alpha=1, beta=0.1)

        # Per-arm conversion rates
        theta_c = pm.Beta("theta_control", alpha=mu * kappa, beta=(1 - mu) * kappa)
        theta_t = pm.Beta("theta_treatment", alpha=mu * kappa, beta=(1 - mu) * kappa)

        # Likelihood
        pm.Binomial("obs_control", n=len(ctrl), p=theta_c, observed=ctrl.sum())
        pm.Binomial("obs_treatment", n=len(trt), p=theta_t, observed=trt.sum())

        trace = pm.sample(n_samples, tune=n_tune, return_inferencedata=True, progressbar=False)

    samples_c = trace.posterior["theta_control"].values.flatten()
    samples_t = trace.posterior["theta_treatment"].values.flatten()
    diff = samples_t - samples_c

    return BayesianResult(
        metric_name=metric_col,
        control_posterior_mean=float(samples_c.mean()),
        treatment_posterior_mean=float(samples_t.mean()),
        prob_treatment_better=float((diff > 0).mean()),
        expected_lift=float(diff.mean()),
        credible_interval_lower=float(np.percentile(diff, 2.5)),
        credible_interval_upper=float(np.percentile(diff, 97.5)),
        posterior_samples_control=samples_c,
        posterior_samples_treatment=samples_t,
    )


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_bayesian_analysis(events: pd.DataFrame) -> dict[str, BayesianResult]:
    """Run Beta-Binomial model on CTR and completion rate from event logs."""
    impressions = events[events["event_type"] == "impression"]
    clicks = events[events["event_type"] == "click"]
    completions = events[events["event_type"] == "completion"]

    def _counts(df: pd.DataFrame, event_df: pd.DataFrame) -> tuple[int, int, int, int]:
        ctrl_users = set(df[df["arm"] == "control"]["user_id"])
        trt_users = set(df[df["arm"] == "treatment"]["user_id"])
        n_c, n_t = len(ctrl_users), len(trt_users)
        k_c = event_df[event_df["user_id"].isin(ctrl_users)]["user_id"].nunique()
        k_t = event_df[event_df["user_id"].isin(trt_users)]["user_id"].nunique()
        return n_c, k_c, n_t, k_t

    tester = BetaBinomialTest()

    n_c, k_c, n_t, k_t = _counts(impressions, clicks)
    ctr_result = tester.fit(n_c, k_c, n_t, k_t, metric_name="CTR")

    n_c2, k_c2, n_t2, k_t2 = _counts(clicks, completions)
    comp_result = tester.fit(n_c2, k_c2, n_t2, k_t2, metric_name="completion_rate")

    return {"ctr": ctr_result, "completion_rate": comp_result}
