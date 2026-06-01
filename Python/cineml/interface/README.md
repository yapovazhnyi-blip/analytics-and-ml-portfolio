i# CineML Unified Interface

Connects all five modules into a single Streamlit application.

## Run

```bash
# From repo root
streamlit run interface/app.py
```

Opens at `http://localhost:8501`

## Structure

```
interface/
├── app.py                     ← entry point + sidebar navigation
├── components/
│   └── data_loader.py         ← shared @st.cache_resource model loaders
└── pages/
    ├── 01_overview.py         ← system health, dataset stats, pipeline controls
    ├── 02_recommender.py      ← M2: get recs, similar items, evaluation, embeddings
    ├── 03_ab_engine.py        ← M3: frequentist + Bayesian + DiD + PSM + power
    ├── 04_analysis.py         ← M4: DS memo, OLS, cohort breakdown, findings
    └── 05_diffusion.py        ← M5: generate poster, classify, architecture docs
```

## Design decisions

**Why Streamlit?**
- Already in the stack (M3 dashboard uses it)
- Zero boilerplate for ML demos (file uploader, number inputs, plotly charts)
- Works without a frontend build step — `streamlit run` is the full deployment

**Why not FastAPI + React?**
- Over-engineered for a portfolio demo; Streamlit demos faster in interviews
- The existing FastAPI service in M2/M3 already handles production serving

**Model loading**
All models are loaded via `@st.cache_resource` in `components/data_loader.py`, which means:
- Models load once per session, not on every page navigation
- Memory is shared across pages — no double-loading
- Graceful fallback when models aren't trained yet (demo data is generated on the fly)

**Fallback behaviour**
Every page works without real model artefacts:
- M1 data missing → synthetic event data is generated on the fly for A/B / analysis pages
- M2 models missing → informational messages with the exact training commands
- M5 models missing → shows random noise with a placeholder caption

## Docker

The interface is included in the main `docker-compose.yml` as a separate service.
To run only the interface:

```bash
docker build -t cineml-interface -f interface/Dockerfile .
docker run -p 8501:8501 -v $(pwd)/data:/app/data cineml-interface
```
