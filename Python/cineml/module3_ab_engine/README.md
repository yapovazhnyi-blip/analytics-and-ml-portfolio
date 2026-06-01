# Module 3 — A/B & Causal Inference Engine

Rigorously tests whether the Two-Tower recommender improves user outcomes over the popularity baseline. Four statistical layers, a Streamlit dashboard, and a FastAPI service.

## The four layers

**1. Frequentist** (`frequentist.py`)
Two-proportion z-test, Welch's t-test, SPRT sequential monitoring, sample size calculator.

**2. Bayesian** (`bayesian.py`)
Beta-Binomial conjugate model. Key output: P(treatment > control) — more interpretable than p-values for stakeholders.

**3. Sequential** (within `frequentist.py`)
SPRT allows valid inference at any peek — prevents p-hacking inflation from repeated interim looks.

**4. Causal** (`causal.py`)
- DiD (Difference-in-Differences): OLS with HC3 robust SEs. Controls for pre-existing engagement trends.
- PSM (Propensity Score Matching): 1-to-1 nearest-neighbour matching with caliper. Bootstrap ATT standard errors.

## Run

```bash
# Standalone Streamlit dashboard
streamlit run module3_ab_engine/dashboard.py

# FastAPI service
uvicorn module3_ab_engine.api.main:app --port 8002
```

## API endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /frequentist` | z-test from raw counts (n, k per arm) |
| `POST /bayesian` | Beta-Binomial model |
| `GET /sample-size?baseline_rate=0.15&mde=0.02` | Required n for given MDE + power |
| `GET /health` | Liveness probe |

## Key metrics

| Metric | Control | Treatment | Lift |
|--------|---------|-----------|------|
| CTR | ~0.148 | ~0.171 | +15.5% |
| Long-tail engagement | ~0.082 | ~0.107 | +30.5% |
| Skip rate | ~0.241 | ~0.238 | -1.2% (good) |

The long-tail engagement rate (clicks on movies ranked >500 by popularity) is the primary discovery metric — it captures genuine content discovery rather than blockbuster engagement.
