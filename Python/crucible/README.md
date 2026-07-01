<div align="center">

# ⚗️ Crucible

### Unified ML Experimentation Platform

*From raw data to production-ready AI — in one place*

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![License](https://img.shields.io/badge/License-Commercial-red?style=flat-square)

[Overview](#overview) · [Features](#features) · [Architecture](#architecture) · [Tech Stack](#tech-stack) · [Project Structure](#project-structure) · [Roadmap](#roadmap)

</div>

---

## Overview

Crucible is a full-stack ML experimentation platform that unifies the entire machine learning workflow — from data ingestion to model deployment — in a single authenticated interface.

The core idea: building an ML system from scratch requires assembling a data engineer, ML engineer, DevOps, and analyst, spending weeks on infrastructure, and months getting all tools to work together. Most ML initiatives either never reach production or cost significantly more than planned. Crucible solves this by providing ready-made infrastructure where the entire path from raw data to a working AI solution is already assembled.

> **Status:** Active development. Core ML pipeline is functional. SaaS features (billing, multi-tenancy, public API) are in progress.

---

## Features

### ✅ Implemented

| Module | Description |
|---|---|
| **Authentication** | JWT-based auth, bcrypt hashing, RBAC (admin / contributor / viewer), BYOK API key encryption |
| **Dataset Connectors** | CSV/Parquet upload, PostgreSQL, REST+OAuth2, Google BigQuery one-shot import |
| **Data Profiling** | Missing value analysis, correlation detection, temporal leakage detection, target distribution |
| **AutoML** | 9 model families, Optuna hyperparameter search, cross-validation, live WebSocket progress |
| **Explainability** | SHAP values for best model, feature importance charts |
| **Data Contracts** | AI-generated data expectations using Claude, contract validation |
| **RAG Pipeline** | Document indexing, hybrid BM25+dense retrieval, natural language Q&A with source citations |
| **Hallucination Scoring** | NLI-based faithfulness scoring (DeBERTa), per-sentence grounding, no API cost |
| **LLM Evaluation** | LLM-as-judge evaluation, custom rubrics |
| **AI Agents** | LangGraph-based agent creation, tool registration, trace recording |
| **Fine-tuning** | LoRA / QLoRA / DPO fine-tuning configuration and job management |
| **Drift Detection** | PSI, KS-test, chi-squared drift monitoring |
| **A/B Testing** | Experiment design and statistical significance testing |
| **Deployment Generator** | Code generation for AWS SageMaker, ECS Fargate, Google Cloud Run |
| **MLflow Integration** | Experiment tracking, model registry, run comparison |
| **Lineage DAG** | Visual data and experiment lineage graph |
| **Model Cards** | Auto-generated model documentation |
| **Fairness Analysis** | Protected attribute analysis, disparate impact detection |

### 🔧 In Progress

- Public REST API with API key authentication
- Anomaly detection module for financial data streams
- Forecasting module (Prophet / statsmodels)
- PostgreSQL migration (currently SQLite in development)
- Redis shared cache backend

### 📋 Planned

- SaaS billing (Stripe, tiered plans)
- Multi-tenancy (organizations, team roles)
- Real-time budget anomaly alerts
- Webhook integrations (Slack, email)
- CLI (`crucible train --dataset data.csv --target churn`)
- Model registry with promotion workflow (candidate → staging → production)
- RAG pipeline using existing connected datasets as knowledge base

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     React 18 / Vite                         │
│         TanStack Query · React Flow · Recharts              │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP / WebSocket
┌──────────────────────────▼──────────────────────────────────┐
│                  FastAPI (async, uvloop)                     │
│    24 routers · JWT auth · Rate limiting · OpenTelemetry    │
└─────────┬────────────────┬──────────────┬───────────────────┘
          │                │              │
┌─────────▼──────┐ ┌───────▼──────┐ ┌────▼──────────┐
│  SQLite /      │ │    Redis     │ │    MLflow      │
│  PostgreSQL    │ │  + ARQ       │ │    Tracking    │
│  (Alembic, 9   │ │  Worker      │ │                │
│   migrations)  │ │              │ │                │
└────────────────┘ └──────────────┘ └────────────────┘
          │
┌─────────▼──────────────────────────────────────────────────┐
│                    ML / AI Layer                            │
│  scikit-learn · XGBoost · LightGBM · CatBoost · Optuna     │
│  SHAP · PyTorch · Transformers · LangGraph · Anthropic      │
└────────────────────────────────────────────────────────────┘
```

**Key architectural decisions:**

- **In-process training** — AutoML runs in a thread pool within the backend process, streaming progress via `asyncio.Queue` over WebSocket. Simple and effective for the current scale.
- **Standalone migration runner** — `migrate.py` runs as a separate process before uvicorn starts, avoiding the `asyncio.run()` conflict with uvloop.
- **BYOK encryption** — All user API keys (Anthropic, etc.) and connector credentials (DB URLs, OAuth secrets) encrypted at rest with Fernet (AES-128-CBC + HMAC).
- **Content-hash caching** — Profiling and SHAP results are cached by dataset content hash. Re-running on the same data returns instantly.

---

## Tech Stack

### Backend
| Layer | Technology |
|---|---|
| Framework | FastAPI, uvicorn (uvloop), Python 3.11 |
| Database | SQLAlchemy 2 (async), Alembic, SQLite → PostgreSQL |
| Queue | Redis, ARQ |
| Auth | python-jose (JWT), bcrypt, Fernet |
| Rate limiting | slowapi |
| Observability | OpenTelemetry, structlog |

### ML / AI
| Area | Technology |
|---|---|
| AutoML | scikit-learn, XGBoost, LightGBM, CatBoost, Optuna |
| Explainability | SHAP |
| Deep learning | PyTorch, Transformers (HuggingFace) |
| LLM | Anthropic Claude (BYOK) |
| Agents | LangGraph |
| RAG | BM25 + dense hybrid retrieval |
| NLI scoring | cross-encoder/nli-deberta-v3-small |
| Experiment tracking | MLflow |

### Frontend
| Area | Technology |
|---|---|
| Framework | React 18, Vite |
| State / Data | TanStack Query, Axios (with auth interceptor) |
| Visualization | Recharts, React Flow |
| Real-time | WebSocket (native browser API) |

### Infrastructure
| Area | Technology |
|---|---|
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Storage | Local filesystem → S3 |
| Cloud targets | AWS SageMaker, ECS Fargate, Google Cloud Run |

---

## Project Structure

```
crucible/
│
├── backend/
│   ├── alembic/                  # 9 database migrations
│   │   └── versions/
│   ├── agents/                   # LangGraph agent runtime
│   ├── anomaly/                  # Anomaly detection (in progress)
│   ├── auth/                     # JWT, bcrypt, Fernet BYOK, RBAC
│   │   ├── dependencies.py       # FastAPI dependency injection
│   │   ├── jwt.py
│   │   ├── key_manager.py        # Fernet encryption for user keys
│   │   └── passwords.py
│   ├── caching/                  # LRU + TTL in-memory cache (→ Redis)
│   ├── connectors/               # Data source connectors
│   │   ├── bigquery_connector.py
│   │   ├── file_connector.py
│   │   ├── rest_connector.py
│   │   └── sql_connector.py
│   ├── deployment/               # Deployment code generator
│   │   └── generator.py          # SageMaker / ECS / Cloud Run templates
│   ├── drift/                    # PSI, KS-test, chi-squared drift detection
│   ├── evaluation/               # LLM-as-judge + NLI hallucination scorer
│   │   ├── hallucination_scorer.py  # DeBERTa NLI, local inference
│   │   └── runner.py
│   ├── fine_tuning/              # LoRA / QLoRA / DPO job management
│   ├── forecasting/              # Time series forecasting (in progress)
│   ├── jobs/                     # Background job manager
│   │   └── manager.py
│   ├── models/                   # SQLAlchemy ORM models (12 tables)
│   ├── observability/            # OpenTelemetry tracing setup
│   ├── profiling/                # Dataset profiling engine
│   │   └── runner.py             # Leakage detection, stats, correlations
│   ├── rag/                      # RAG pipeline
│   │   ├── indexer.py            # Document chunking and indexing
│   │   └── retriever.py          # Hybrid BM25 + dense retrieval
│   ├── retraining/               # Automated retraining policies
│   ├── routers/                  # 24 FastAPI routers
│   │   ├── ab_testing.py
│   │   ├── agent_training.py
│   │   ├── anomaly.py
│   │   ├── api_keys.py
│   │   ├── auth.py
│   │   ├── cloud.py
│   │   ├── connectors.py
│   │   ├── data_contracts.py
│   │   ├── datasets.py
│   │   ├── deployment.py (phase3.py)
│   │   ├── drift.py
│   │   ├── evaluation.py
│   │   ├── experiments.py        # AutoML + WebSocket progress
│   │   ├── fairness.py
│   │   ├── fine_tuning.py
│   │   ├── forecasting.py
│   │   ├── jobs.py
│   │   ├── model_cards.py
│   │   ├── profiling.py
│   │   ├── rag.py
│   │   └── retraining.py
│   ├── schemas/                  # Pydantic v2 request/response schemas
│   ├── training/                 # AutoML training engine
│   │   ├── runner.py             # Optuna + 9 model families
│   │   └── time_series/
│   ├── tests/                    # 1,033 automated tests
│   ├── config.py                 # Pydantic settings
│   ├── database.py               # Async SQLAlchemy engine + session
│   ├── main.py                   # FastAPI app, middleware, router registration
│   ├── migrate.py                # Standalone Alembic runner (pre-uvicorn)
│   └── requirements.txt
│
├── frontend/
│   └── src/
│       ├── api/
│       │   └── client.js         # Axios instance with JWT interceptor
│       ├── components/           # Shared UI components
│       │   ├── AdvisorPanel.jsx
│       │   ├── Layout.jsx
│       │   ├── LineageDAG.jsx
│       │   └── ui.jsx            # Design system primitives
│       ├── context/
│       │   └── AuthContext.jsx   # JWT storage and refresh logic
│       ├── pages/                # 16 application pages
│       │   ├── ABTestingPage.jsx
│       │   ├── AgentTrainingPage.jsx
│       │   ├── ConnectorsPage.jsx
│       │   ├── DatasetDetailPage.jsx
│       │   ├── DatasetsPage.jsx
│       │   ├── EvaluationPage.jsx
│       │   ├── ExperimentDetailPage.jsx
│       │   ├── ExperimentsPage.jsx
│       │   ├── FineTuningPage.jsx
│       │   ├── ForecastingPage.jsx
│       │   ├── LoginPage.jsx
│       │   ├── MLOpsPage.jsx
│       │   ├── RAGPage.jsx
│       │   └── SettingsPage.jsx
│       ├── store/
│       │   └── queryClient.js    # TanStack Query config (staleTime, retry)
│       ├── App.jsx               # Router setup
│       ├── index.css             # Design tokens (CSS variables)
│       └── main.jsx
│
├── docs/
│   ├── DOCUMENTATION_UA.md       # Full Ukrainian technical documentation
│   ├── CONNECTORS.md             # Data connector setup guide
│   ├── Crucible_Documentation_UA.pdf
│   └── Crucible_Offer_CRM_UA.pdf # Commercial offer (Construction CRM)
│
├── docker-compose.yml            # 5-service orchestration
├── .env.example                  # Environment variable template
└── README.md
```

---

## Getting Started

> ⚠️ **Note:** This repository is source-available. You may view and study the code, but deployment or commercial use requires a license. See [License](#license).

### Prerequisites

- Docker Desktop
- 8 GB RAM (16 GB recommended for ML training)

### Run locally

```bash
# 1. Clone the repository
git clone https://github.com/yapovazhnyi-blip/analytics-and-ml-portfolio
cd analytics-and-ml-portfolio/crucible

# 2. Copy and configure environment
cp .env.example .env
# Edit .env — set SECRET_KEY, ANTHROPIC_API_KEY (optional, for AI features)

# 3. Start all services
docker compose up

# 4. Open in browser
# Frontend:  http://localhost:5173
# API docs:  http://localhost:8000/docs
# MLflow:    http://localhost:5001
```

**First run:** the first registered account is automatically assigned the `admin` role.

### Services started by Docker Compose

| Service | Port | Description |
|---|---|---|
| `crucible-frontend` | 5173 | React / Vite dev server |
| `crucible-backend` | 8000 | FastAPI application |
| `crucible-worker` | — | ARQ background worker |
| `crucible-redis` | 6379 | Redis (cache + queue) |
| `crucible-mlflow` | 5001 | MLflow tracking server |

---

## Database Schema

12 tables managed by Alembic migrations:

`users` · `datasets` · `connectors` · `experiments` · `rag_documents` · `fine_tune_jobs` · `forecast_jobs` · `user_api_keys` · `agent_traces` · `registered_agents` · `retraining_policies` · `retraining_runs`

---

## Security

- **Passwords:** bcrypt (rounds=12)
- **JWT:** HS256, access token 24h + refresh 30d, type claim prevents token substitution
- **Encryption at rest:** Fernet (AES-128-CBC + HMAC) for API keys and connector credentials
- **File uploads:** extension allowlist + magic bytes validation
- **SQL:** SQLAlchemy ORM with parameterized queries throughout
- **Rate limiting:** 120 req/min global, 10 req/min on auth endpoints
- **Security headers:** `X-Content-Type-Options`, `X-Frame-Options`, `HSTS`

Current security score: **6.5/10** — solid foundation, production hardening in progress (JWT revocation, SSRF protection, ModelScan, LLM Guard, audit log).

---

## Roadmap

### Phase 0 — Stabilization *(current)*
WebSocket proxy fix, stuck experiment cleanup, health endpoint correction

### Phase 1 — Stable MVP
Full AutoML flow with live progress, Forecasting module, Anomaly detection, RAG end-to-end verification, MLOps drift + retraining pages

### Phase 2 — Quality, Security & Public API
- **Security:** ModelScan for model artifacts, LLM Guard middleware, JWT revocation, SSRF protection, audit log
- **Performance:** PostgreSQL, Redis shared cache, `--workers 4`, GZip compression, DB indexes
- **Public API:** `cruc_` prefixed API keys, Settings UI for key management, Admin user access panel
- **Features:** RAG on connected datasets, Giskard scan after AutoML, Agent system end-to-end

### Phase 3 — Productization
Stripe billing, multi-tenancy, S3 storage, production deployment, CDN, advanced LLM security (promptfoo CI, Garak, FuzzyAI)

### Phase 4 — Expansion
Webhooks, CLI, model registry with promotion flow, collaboration features, multimodal datasets

---

## Tests

```bash
cd backend
pytest tests/ -v
# 1,033 tests
```

---

## License

This project is **source-available** under a commercial license.

- ✅ You may view and study the source code
- ✅ You may run it locally for personal evaluation
- ❌ Commercial use, deployment, or redistribution requires a written license agreement

For licensing inquiries, contact the repository owner.

---

<div align="center">

Built with ⚗️ by [@yapovazhnyi-blip](https://github.com/yapovazhnyi-blip)

</div>
