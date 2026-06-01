# CineML Platform

> **End-to-End ML Research & Experimentation System for Streaming Content Intelligence**

CineML simulates the core ML systems a streaming company needs to operate — from data ingestion and personalised recommendations, through rigorous A/B testing and causal analysis, to generative artwork for content promotion. Everything is built around a single shared dataset (MovieLens 25M + TMDB + synthetic event logs) so each module feeds the next.

---

## Architecture

```
MovieLens 25M  ──┐
TMDB API       ──┼──▶  M1 Data Pipeline  ──▶  Parquet / BigQuery
Event Simulator──┘          │
                            │  ratings · movies · events · metadata
                            ▼
                     M2 Personalisation Engine
                     ALS · Two-Tower (PyTorch) · FastAPI :8001
                            │
                            │  ranked recommendations (treatment arm)
                            ▼
                     M3 A/B & Causal Engine ◀── event logs
                     Frequentist · Bayesian · DiD · PSM
                     FastAPI :8002 · Streamlit :8503
                            │
                            │  experiment results
                            ▼
                     M4 Analysis Memo
                     OLS · Cohort · DS investigation

                     M5 Diffusion + ViT  ◀── TMDB poster images
                     DDPM from scratch · ViT-B/16 · Gradio :7860
```

---

## Modules

| # | Module | Key tech | Port |
|---|--------|----------|------|
| 1 | [Data Curation Pipeline](./module1_data_pipeline/) | BigQuery, DVC, TMDB API | — |
| 2 | [Personalisation Engine](./module2_recommender/) | PyTorch Two-Tower, ALS, FastAPI | 8001 |
| 3 | [A/B & Causal Inference Engine](./module3_ab_engine/) | PyMC, DiD, PSM, Streamlit | 8002 / 8503 |
| 4 | [Content Discovery Memo](./module4_analysis_memo/) | statsmodels OLS, cohort analysis | — |
| 5 | [Diffusion + ViT](./module5_diffusion_vit/) | DDPM from scratch, ViT-B/16, Gradio | 7860 |

**Unified interface** (Streamlit) runs at `localhost:8501` — pipeline controls, Docker management, and all five modules in one place.

---

## Quick start

```bash
git clone https://github.com/yourname/cineml
cd cineml
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
# Add TMDB_API_KEY and optionally GCP_PROJECT

# Launch the interface (works immediately with demo data)
streamlit run interface/app.py
```

Run the full pipeline from the interface, or via CLI:

```bash
python module1_data_pipeline/fetch_movielens.py
python module1_data_pipeline/event_simulator.py
python module2_recommender/train.py --model als
python module2_recommender/train.py --model two-tower --epochs 20
```

Start all Docker services:

```bash
docker compose up --build -d
```

---

## Services

| Service | URL | Description |
|---------|-----|-------------|
| Unified interface | `localhost:8501` | Streamlit — full platform control |
| Recommender API | `localhost:8001/docs` | FastAPI — ALS + Two-Tower endpoints |
| A/B Engine API | `localhost:8002/docs` | FastAPI — statistical test endpoints |
| A/B Dashboard | `localhost:8503` | Streamlit — live A/B analysis |
| Diffusion Demo | `localhost:7860` | Gradio — poster generation + ViT |

---

## What it demonstrates

**Role A — ML Researcher**: DDPM implemented from scratch (Ho et al. 2020), DDIM sampling (Song et al. 2021), cross-attention genre conditioning (same mechanism as Stable Diffusion), ViT fine-tuning, Attention Rollout explainability, MLflow tracking, FID evaluation.

**Role B — Senior DS (Personalisation)**: Two-Tower neural model with BPR loss, ALS on implicit feedback, NDCG/MAP/Hit@k evaluation, causal inference (DiD + PSM), Bayesian A/B testing, content discovery metric design, Netflix-style investigation memos.

**Role C — Full-Stack DS (Experimentation)**: Production FastAPI services, Docker Compose orchestration, frequentist + Bayesian + sequential testing, BigQuery pipelines, Streamlit dashboards, end-to-end system design.

---

## References

- Hu, Koren, Volinsky (2008) — Collaborative Filtering for Implicit Feedback
- Rendle et al. (2009) — BPR: Bayesian Personalised Ranking
- Kohavi, Tang, Xu (2020) — Trustworthy Online Controlled Experiments
- Ho, Jain, Abbeel (2020) — Denoising Diffusion Probabilistic Models
- Song et al. (2021) — Denoising Diffusion Implicit Models
- Dosovitskiy et al. (2021) — An Image is Worth 16×16 Words
- Abnar & Zuidema (2020) — Quantifying Attention Flow in Transformers
