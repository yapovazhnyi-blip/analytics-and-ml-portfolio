"""
module3_ab_engine/api/main.py
==============================
FastAPI service exposing A/B testing endpoints.

Endpoints:
  GET  /health
  POST /frequentist   — run z-test / t-test on provided event data
  POST /bayesian      — run Beta-Binomial model
  GET  /sample-size   — required n for given baseline, MDE, alpha, power

Usage:
    uvicorn module3_ab_engine.api.main:app --host 0.0.0.0 --port 8002
"""
import sys
from pathlib import Path

# Ensure repo root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="CineML A/B Engine API",
    description="Frequentist + Bayesian A/B testing service",
    version="1.0.0",
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class FrequentistRequest(BaseModel):
    n_control:   int
    k_control:   int    # successes in control
    n_treatment: int
    k_treatment: int    # successes in treatment
    alpha:       float = 0.05


class BayesianRequest(BaseModel):
    n_control:   int
    k_control:   int
    n_treatment: int
    k_treatment: int


class SampleSizeRequest(BaseModel):
    baseline_rate: float
    mde:           float
    alpha:         float = 0.05
    power:         float = 0.80


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "ab-api"}


@app.post("/frequentist")
async def frequentist(req: FrequentistRequest):
    from module3_ab_engine.frequentist import proportion_test, ABGroups
    import numpy as np
    import pandas as pd

    # Reconstruct binary series from counts
    ctrl = pd.Series([1] * req.k_control + [0] * (req.n_control - req.k_control))
    trt  = pd.Series([1] * req.k_treatment + [0] * (req.n_treatment - req.k_treatment))

    result = proportion_test(
        ABGroups(control=ctrl, treatment=trt, metric_name="conversion"),
        alpha=req.alpha,
    )
    return {
        "control_mean":       result.control_mean,
        "treatment_mean":     result.treatment_mean,
        "absolute_lift":      result.absolute_lift,
        "relative_lift_pct":  result.relative_lift_pct,
        "p_value":            result.p_value,
        "ci_lower":           result.ci_lower,
        "ci_upper":           result.ci_upper,
        "significant":        result.significant,
        "test_type":          result.test_type,
    }


@app.post("/bayesian")
async def bayesian(req: BayesianRequest):
    from module3_ab_engine.bayesian import BetaBinomialTest

    tester = BetaBinomialTest()
    result = tester.fit(
        req.n_control, req.k_control,
        req.n_treatment, req.k_treatment,
    )
    return {
        "control_posterior_mean":   result.control_posterior_mean,
        "treatment_posterior_mean": result.treatment_posterior_mean,
        "prob_treatment_better":    result.prob_treatment_better,
        "expected_lift":            result.expected_lift,
        "credible_interval_lower":  result.credible_interval_lower,
        "credible_interval_upper":  result.credible_interval_upper,
    }


@app.get("/sample-size")
async def sample_size(
    baseline_rate: float,
    mde:           float,
    alpha:         float = 0.05,
    power:         float = 0.80,
):
    from module3_ab_engine.frequentist import required_sample_size
    return required_sample_size(baseline_rate, mde, alpha, power)
