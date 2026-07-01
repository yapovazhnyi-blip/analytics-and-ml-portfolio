# Crucible — Full Documentation

This document explains what Crucible is, why it exists, and how every piece of it works — in plain language, for someone reading the codebase for the first time. It assumes you can program but doesn't assume you already know FastAPI, SQLAlchemy, Optuna, SHAP, LoRA, or any of the other specific tools used here. Where a concept needs background, that background is given inline.

For the *theory* behind the technologies (what is SHAP, what is a LoRA adapter, what is async/await actually doing) — see the companion file **`docs/theory-and-qa.html`**, which is an interactive reference with a built-in Q&A you can ask anything.

---

## Table of Contents

1. [What This Is, and Why](#1-what-this-is-and-why)
2. [The Big Picture — How a Request Flows Through the System](#2-the-big-picture)
3. [Data Layer — Getting Data In](#3-data-layer)
4. [ML Core — Training, Explaining, Calibrating](#4-ml-core)
5. [Agents and LLM Integration](#5-agents-and-llm-integration)
6. [Fine-Tuning and RAG](#6-fine-tuning-and-rag)
7. [MLOps Automation](#7-mlops-automation)
8. [Platform and Infrastructure](#8-platform-and-infrastructure)
9. [The Database Layer](#9-the-database-layer)
10. [The API Surface](#10-the-api-surface)
11. [The Frontend](#11-the-frontend)
12. [Testing Strategy](#12-testing-strategy)
13. [Catalog of Key Design Decisions](#13-catalog-of-key-design-decisions)

---

## 1. What This Is, and Why

Crucible is a single platform that covers the full lifecycle of a machine learning model and, more recently, of an AI agent: get data in, understand the data, train a model on it, explain why the model makes the predictions it makes, deploy it, watch it in production, and automatically retrain it when the world changes. Separately, it can capture how an AI agent solves real problems on this platform and use that experience to train a smaller, cheaper, specialised agent.

Most learning projects stop at "train a model in a notebook." Most production ML platforms (the kind a real company would build) have to additionally answer questions a notebook never has to answer: *Where does this run? Who can use it? What happens when training fails halfway through? How do I know if the model is still good six months later? How much does each AI call cost, and who pays for it?* Crucible answers all of these, not just the modelling question, because that's the actual job of an ML platform.

**Who this is for, concretely:** the README frames this as a portfolio demonstration. That means every piece of it is built to the standard of "would a senior engineer reviewing this code find it credible," not to the standard of "does the demo work once." That's why there are 1,024 automated tests, why error paths are handled explicitly instead of silently swallowed, and why design decisions are documented inline rather than left for someone to guess at later.

---

## 2. The Big Picture

Here is what happens, end to end, when someone uses the most complete path through the system — upload a dataset, train a model, watch it get used and retrained automatically:

```
1. UPLOAD
   Browser → POST /datasets/upload → connectors/file_connector.py
   → saves the file via storage/ (local disk or S3)
   → computes a SHA-256 content hash (for lineage + caching keys)
   → creates a Dataset row in the database

2. PROFILE
   Browser → POST /datasets/{id}/profile → profiling/runner.py
   → runs missingness, correlation/VIF, leakage, and distribution analysis
   → result is cached (caching/cache.py) so a second request is instant
   → (optional) Claude advisor reads the profile and suggests next steps

3. TRAIN
   Browser → POST /experiments → routers/experiments.py
   → starts a background job (jobs/manager.py) — the request returns immediately
   → training/runner.py runs an Optuna search across up to 9 model families
   → the winning model gets probability-calibrated (if classification)
   → SHAP explains the winning model
   → results are logged to MLflow or Weights & Biases (tracking/)
   → progress streams back to the browser over a WebSocket as it happens

4. DEPLOY OR AUTOMATE
   Either: deployment/ generates a downloadable FastAPI+Docker+K8s package
   Or:     retraining/ watches this dataset for drift and retrains automatically
           when the data changes enough, promoting the new model only if
           it's actually better
```

A second, parallel path exists for **agents**: an agent (`agents/runner.py` or `agents/multi_agent.py`) calls Claude, which in turn calls tools that hit this same API (list datasets, run profiling, start training, generate a deployment). Every agent session can optionally be captured (`agents/traces.py`), turned into training data (`agents/trace_converter.py`), and used to fine-tune a smaller agent (`fine_tuning/`) that does the same job without needing Claude at all.

**The one architectural idea that repeats everywhere:** Crucible never hard-codes a single vendor for anything that has more than one reasonable provider. Storage can be a local disk or S3. The LLM can be Claude direct, AWS Bedrock, or a local Ollama model. The job queue can be in-process or Redis-backed. Tracking can be MLflow or Weights & Biases. Every one of these is implemented as an abstract base class with exactly one no-infrastructure default and one production alternative, selected by a single config value. This is explained in detail in [§13](#13-catalog-of-key-design-decisions).

---

## 3. Data Layer

**Directories:** `connectors/`, `profiling/`, `data_contracts/`, `storage/`, `drift/`, `anomaly/`

### `connectors/` — getting data in, from anywhere

Every data source — a CSV upload, a Parquet file, a SQL database, a REST API behind OAuth2, a BigQuery table — implements the same contract: `async def ingest(...) -> IngestResult`. `IngestResult` carries the file path, row/column counts, a content hash, and a schema. This means every downstream system (profiling, training, the deployment generator) is written once against `IngestResult` and never needs to know or care which connector produced it.

- `base.py` — defines `IngestResult` and the `BaseConnector` contract, plus `infer_columns()`, which sniffs column types (numeric, categorical, datetime) from a pandas DataFrame.
- `file_connector.py` — CSV and Parquet uploads.
- `sql_connector.py` — arbitrary SQL databases via SQLAlchemy, with the connection string encrypted at rest (Fernet — see [§8](#8-platform--infrastructure)).
- `oauth_connector.py` — REST APIs behind OAuth2, with pagination handling and token refresh.
- `bigquery_connector.py` — runs a SQL query against Google BigQuery using the Storage Read API (10–50× faster than the REST API for large tables), with a hard 1,000,000-row safety cap so an unbounded query can't silently pull an entire warehouse table into memory.

**Why a common `ingest() → IngestResult` contract matters:** without it, every feature that touches data (profiling, training, the lineage DAG) would need a different code path per source type. With it, adding a tenth connector (say, Snowflake) means writing one new file that returns the same shape of result — nothing else in the system changes.

### `profiling/` — understanding the data before training on it

Most AutoML tools train first and let you discover data problems from a bad score. Crucible profiles first. `runner.py` orchestrates four analyses:

- **`missingness.py`** — distinguishes *systematic* missingness (a column is null whenever another specific condition holds — usually a real signal, like "shipping_date is null whenever the order was cancelled") from *random* missingness (just noise). This distinction matters because systematic missingness, if ignored, often becomes a leakage vector — the model learns "this field being empty means cancelled" rather than learning anything about cancellation itself.
- **`correlation.py`** — pairwise correlation *and* VIF (Variance Inflation Factor), which catches multicollinearity that pairwise correlation alone misses (three columns can each have low pairwise correlation with each other while still being linearly dependent as a group).
- **`leakage.py`** — three distinct leakage detectors: feature leakage (a feature suspiciously predicts the target almost perfectly), temporal leakage (a feature's values change in a way that couldn't have been known before the prediction time, for time-series data), and ID-column leakage (a column that looks like an identifier but happens to encode target ordering — e.g. row IDs assigned in a way correlated with the outcome).
- **`distributions.py`** — class imbalance for classification, skewness for regression targets, both of which silently break naive AutoML if not flagged.

A `ProfileReport`'s result is cached (`caching/cache.py`) keyed on `(content_hash, target_column, params)` — re-profiling the same dataset with the same parameters returns instantly instead of recomputing.

### `data_contracts/` — locking in what "valid data" means

Once you trust a dataset, `data_contracts/schema.py` can auto-generate a `DataContract`: numeric ranges (with a tolerance margin) and allowed categorical values, inferred from the observed data. Future batches of data can be validated against this contract before they're used for retraining — this is what catches "someone uploaded a CSV with a column renamed or a currency switched from USD to cents" before it silently corrupts a model.

### `storage/` — where files actually live

`StorageBackend` is an abstract base with two implementations: `LocalStorage` (writes to disk, used by default) and `S3Storage` (boto3, presigned URLs for downloads, streaming uploads so large files don't need to fit in memory). `get_storage()` is a cached factory that returns whichever backend `settings.storage_backend` selects. Every other module that needs to read or write a file goes through this interface — nothing else in the codebase calls `open()` directly on a dataset or model file.

### `drift/` — has the data changed?

`detector.py`'s `compare_datasets()` takes a reference dataset and a current one and runs three statistical tests: PSI (Population Stability Index, for numeric features), KS (Kolmogorov-Smirnov, also numeric), and chi-squared (for categorical features). Each feature gets a severity rating (`stable` / `slight` / `significant` / `critical`); the overall report rolls these up. This is the input to the automated retraining pipeline (§7).

### `anomaly/` — finding the rows that don't belong

Three unsupervised algorithms — Isolation Forest, Local Outlier Factor, and One-Class SVM — each score every row for "how unusual is this," and `runner.py` combines them by majority vote so no single algorithm's blind spot (e.g. Isolation Forest struggles when anomalies form a tight cluster) determines the final answer alone. One-Class SVM is automatically skipped above 10,000 rows, because its O(n²) memory cost makes it impractical at that scale — the system tells you why rather than just running out of memory.

---

## 4. ML Core

**Directories:** `training/`, `explainability/`, `fairness/`, `model_cards/`

### `training/` — the AutoML engine

This is the heart of the platform. `runner.py`'s `TrainingRunner.run()` does, in order:

1. **Split** the data into train/holdout (the holdout is never touched during hyperparameter search — it exists only for an honest final evaluation).
2. **Search.** An Optuna study (TPE sampler — Tree-structured Parzen Estimator, a Bayesian optimisation method that models which hyperparameter regions look promising based on trials so far, rather than searching randomly or exhaustively) runs across every registered model family. Each trial's cross-validation score is reported back to Optuna fold-by-fold, which lets the pruner (MedianPruner by default, or HyperbandPruner — see [§13](#13-catalog-of-key-design-decisions)) kill a clearly-bad trial after the first fold instead of wasting compute finishing it.
3. **Model families** (`model_families.py`, `gbm_families.py`, `keras_families.py`): Random Forest, Gradient Boosting, Logistic Regression/Ridge, SVM, k-NN, XGBoost, LightGBM, CatBoost, and an optional Keras MLP — nine in total for classification, eight for regression. Each is registered behind an availability flag (`XGBOOST_AVAILABLE`, etc.) so the platform still works if a given library isn't installed; only that one family is skipped.
4. **Calibrate.** For classification, if the training set has ≥200 rows, `CalibratedClassifierCV` is applied post-hoc — isotonic regression above 1,000 rows, Platt scaling (sigmoid) below. This matters because tree ensembles are notoriously overconfident: a raw model might say "85% probability of churn" when it's actually right only 60% of the time at that score. Calibration fixes that without touching the ranking/AUC.
5. **Save** the winning pipeline (`joblib`) and **log** the run to whichever tracking backend is configured (§7).

### `explainability/` — why did the model decide that?

`shap_runner.py` picks the right SHAP explainer for the model type automatically: `TreeExplainer` for tree-based models (exact and fast), `LinearExplainer` for linear models, and `KernelExplainer` (with k-means background summarisation to keep it tractable) for anything else. The result is a ranked feature importance list stored alongside the experiment.

### `fairness/` — does the model treat groups differently?

`analyzer.py` reproduces the *exact* holdout split used during training (same `random_state`, same `test_size`) so fairness metrics are computed on the same data the reported accuracy came from — comparing fairness on a different split than the one that produced the headline metric would be comparing two different things. It computes four standard fairness metrics (demographic parity difference, equal opportunity difference, equalized odds difference, disparate impact ratio) against the EEOC's "4/5ths rule" threshold, a real legal standard used in U.S. employment discrimination law.

### `model_cards/` — documenting the model for non-engineers

`generator.py` assembles a structured `ModelCard` from data that already exists — no new computation, just collecting what training, fairness, and the data contract already produced — and `renderer.py` turns it into Markdown or a styled HTML report. The format follows Mitchell et al.'s "Model Cards for Model Reporting" (Google, 2019) and is structured to satisfy what the EU AI Act requires for high-risk AI system documentation: intended use, limitations, training data summary, performance, and a fairness assessment (or an explicit note that one hasn't been run).

---

## 5. Agents and LLM Integration

**Directories:** `agents/`, `llm/`, `evaluation/`, `advisor/`

### `llm/` — provider-agnostic LLM access

`base.py` defines `LLMBackend` with one method, `complete()`, and a normalised `LLMResponse`. Three implementations: `AnthropicBackend` (direct API), `BedrockBackend` (AWS, translates Anthropic's message format to Bedrock's Converse API format), and `OpenAICompatBackend` (covers Groq, Ollama, OpenRouter — anything speaking the OpenAI Chat Completions format, with bidirectional translation between Anthropic's and OpenAI's very different tool-calling conventions). `resolve_backend()` picks one based on `settings.llm_provider`.

### `agents/` — the ReAct agent and the multi-agent system

**`runner.py`** implements ReAct (Reason + Act): Claude is given a goal and a list of tools (`tools.py` — nine of them, covering dataset discovery, profiling, training, and deployment), and iterates up to 10 steps, calling tools and reading their results until it produces a final answer. `tools.py`'s `ToolExecutor` is the layer that actually calls Crucible's own services — the agent doesn't have special access; it uses the same code paths a human clicking through the UI would.

**`multi_agent.py`** and **`state.py`** implement a different pattern using LangGraph: a *supervisor* (which calls Claude to decide what to do next) routes work to one of three *specialists* — Dataset Analyst, Model Trainer, Deployer — each of which is a deterministic function with no LLM call of its own. The supervisor reasons in natural language about strategy; the specialists execute mechanically. This split exists because it makes the expensive, non-deterministic part (Claude's reasoning) small and isolated, while the bulk of the actual work is fast, free, and fully testable without mocking an LLM.

**`traces.py`, `trace_converter.py`, `bundle.py`, `benchmark.py`** — the agent training pipeline, covered in depth in [§7](#7-mlops-automation), since conceptually it's the agent-side equivalent of model retraining.

### `evaluation/` — LLM-as-judge

`judge.py`'s `LLMJudge` scores a piece of text against named rubrics (accuracy, helpfulness, safety, or custom ones) by asking Claude to rate it and return structured JSON. This is what scores agent traces before they become DPO training pairs, and what the agent benchmark uses to score "answer quality" alongside the mechanical "did it call the right tool" check.

### `advisor/` — the data quality advisor

`claude.py` takes a profiling report, turns it into a structured prompt, and asks Claude for severity-ranked, specific suggestions ("column X is 40% null and looks systematic — investigate before training" rather than a generic "check your data for missing values").

---

## 6. Fine-Tuning and RAG

**Directories:** `fine_tuning/`, `rag/`

### `fine_tuning/` — training smaller models on Crucible's own data

Two training methods, both using LoRA (Low-Rank Adaptation — instead of updating all of a model's weights, you train a small pair of low-rank matrices that get added to the original weights, which is dramatically cheaper and produces a small, portable "adapter" file instead of a full new copy of the model):

- **`trainer.py`** — Supervised Fine-Tuning (SFT). Takes Alpaca or ShareGPT-format instruction/response pairs and trains the model to reproduce that pattern.
- **`dpo_trainer.py`** — Direct Preference Optimisation. Takes `(prompt, chosen, rejected)` triplets and trains the model to prefer the chosen response over the rejected one, without needing a separate reward model (the technique that made RLHF require three training phases; DPO collapses it to one). The mock-mode loss curve deliberately starts near `ln(2) ≈ 0.693` — the theoretically correct value for a model with no preference information, since that's the cross-entropy of a coin-flip between two equally-likely options.

Both trainers support a `mock-` model ID prefix that simulates training without downloading real weights — this is how the test suite (and a from-scratch clone of this repo with no GPU) can exercise the full pipeline.

### `rag/` — retrieval-augmented generation

A standard but complete RAG pipeline: `chunker.py` (three chunking strategies), `embedder.py` (fastembed, ONNX-based, no GPU required), `vector_store.py` (ChromaDB), `retriever.py` (hybrid: BM25 keyword search + dense vector search, combined via Reciprocal Rank Fusion so neither pure-keyword nor pure-semantic search dominates), `generator.py` (Claude, grounded in retrieved chunks), and `evaluator.py` (RAGAS-style metrics: faithfulness, answer relevancy, context precision).

---

## 7. MLOps Automation

**Directories:** `retraining/`, `jobs/`, `caching/`, `tracking/`, `cloud/`

### `retraining/` — the scheduled retraining pipeline

This is Crucible's answer to "why don't you just use Airflow." `pipeline.py`'s `run_pipeline()` executes four steps for one `RetrainingPolicy`:

1. **Drift check** — `drift.detector.compare_datasets()` between the policy's reference dataset (what the current production model was trained on) and a "current" dataset (the latest data batch).
2. **Gate** — if drift severity doesn't meet the policy's configured trigger threshold, the run stops here, logged as "no drift, skipped."
3. **Retrain** — if drift is significant enough, a normal training job is submitted (the exact same `jobs.manager.start_job()` call `POST /experiments` uses), so a pipeline-triggered retrain produces an ordinary, fully-visible Experiment record.
4. **Promotion** — the new candidate's score is compared against the current production model's score. If it wins by at least the configured margin, the old model is archived, the new one is marked production, and the policy's drift baseline updates to the new model's training data (so the *next* drift check compares against the current baseline, not a stale one).

Every step's outcome is appended to a `steps_json` audit log — the same transparency an Airflow task-instance log gives you, without the multi-component Airflow deployment. `scheduler.py` wires this to `APScheduler` for genuine periodic execution inside the same process (with a documented single-worker-process constraint — see [§13](#13-catalog-of-key-design-decisions)).

The agent-side equivalent lives in `agents/`: `traces.py` captures real agent sessions, `trace_converter.py` turns them into SFT or DPO training data (pairing the best- and worst-scored response to the same goal for DPO), `bundle.py` packages a trained adapter into a portable `.crucible` ZIP (standard PEFT format inside — loadable by any `transformers`/`peft` installation, not just Crucible), and `benchmark.py` runs a fixed set of goals through an agent to measure whether it actually does the right thing, not just whether it produces *an* answer.

### `jobs/` — background work

Two distinct patterns, used for different reasons:

- **`manager.py`** — a bespoke job tracker built specifically for AutoML training, because training needs *live, granular progress* (which trial is running, current best score) streamed to a WebSocket. This is intentionally not generalised into the abstraction below — forcing live-progress training onto a generic interface would have meant losing that granularity.
- **`queue.py`** — `JobQueueBackend`, with an `InMemoryJobQueue` default (bounded concurrency via a semaphore, retry with exponential backoff) and an `ArqJobQueue` (Redis-backed, survives process restarts). Used for simpler fire-and-forget work like RAG document indexing, where retry-on-failure matters more than live progress.

### `caching/` — not recomputing the expensive stuff

`cache.py`'s `LRUTTLCache` is a thread-safe in-memory cache with per-entry TTL and LRU eviction. Profiling results are cached for an hour; the cache key is `(content_hash, target_column, time_column, test_fraction)` — using the dataset's *content hash* rather than its database ID means a re-uploaded file with identical bytes shares a cache entry even under a different ID.

### `tracking/` — MLflow or Weights & Biases

`TrackingBackend.log_run()` covers the whole lifecycle (start a run, log params, log metrics, log an artifact, end the run) in one call, because that's how training actually uses it — once, at the end of a completed run. `MLflowBackend` is the self-hostable default (already running via `docker-compose.yml`); `WandBBackend` is the cloud-SaaS alternative for teams that already use it.

### `cloud/` — SageMaker

`sagemaker.py`'s `SageMakerTrainingRunner` uploads data to S3, submits a real SageMaker training job using a pre-built sklearn container, and polls until completion. A `role_arn` starting with `mock-` (or containing AWS's own documentation placeholder account ID `000000000000`) triggers a 2-second simulated run, so the whole flow is testable and demoable without AWS credentials.

---

## 8. Platform and Infrastructure

**Directories:** `auth/`, `observability/`, `deployment/`, cross-cutting concerns in `main.py`

### `auth/` — who's allowed to do what, and whose API key gets used

JWT authentication (`jwt.py`, `dependencies.py`) with three roles (viewer, contributor, admin). `passwords.py` hashes with bcrypt directly (not via `passlib`, which has version-compatibility issues with recent bcrypt releases on some platforms).

**BYOK (Bring Your Own Key)** — `key_manager.py`'s `get_anthropic_key(user, require=True)` resolves, in order: the user's own Anthropic key (Fernet-encrypted in the database) → the server's fallback key → a clear HTTP 422 error. Every one of the six places Crucible calls Claude resolves its key this way, so a user with their own key controls their own spend and rate limits, and a misconfigured server doesn't silently use someone else's billing.

### `observability/` — seeing what's happening in production

`tracing.py` wires up OpenTelemetry: automatic spans for every HTTP request (via `FastAPIInstrumentor`), plus manual spans (`start_span()`) around the two operations actually worth measuring in detail — agent tool calls and the profiling pipeline. Exports to the console (default, zero infrastructure), an OTLP collector (Jaeger, Grafana Tempo, etc.), or nowhere (disabled). Separately, `main.py`'s `RequestIDMiddleware` stamps every request with a UUID (or reuses one the caller supplied), returned as `X-Request-ID`, so a single ID can be grepped across every log line touched by that request.

### `deployment/` — turning a trained model into a running service

`generator.py` produces a downloadable ZIP: a FastAPI server exposing `POST /predict`, a Dockerfile, pinned `requirements.txt` (the exact versions used during training, not whatever's latest), an OpenAPI spec, and a Kubernetes Deployment manifest with health probes and resource limits already filled in. `onnx_exporter.py` separately converts the trained sklearn/XGBoost/LightGBM pipeline to ONNX format for 3–10× faster CPU inference, validating the converted model's predictions match the original before handing it back.

### Deploying Crucible itself — three documented paths

This is a separate question from deploying one *trained model* (above) — it's about running the whole platform somewhere public. All three paths use the exact same Docker image; only the orchestration layer changes (full walkthrough in **`DEPLOYMENT.md`**):

- **Render** (`render.yaml`) — connect a repo, get a public URL in minutes, free tier, no AWS account needed. The trade-off: the free tier sleeps after 15 minutes idle, with a real cold-start delay (10–50s) on the next request.
- **Google Cloud Run** — also scales to zero, but with a more generous always-free allowance and no forced sleep behaviour. Pairs naturally with the existing BigQuery connector.
- **AWS ECS Fargate** (`infra/ecs/task-definition.json`, `infra/ecs/service-definition.json`) — the production-realistic path. No free tier, but real load balancing (an Application Load Balancer health-checks and routes across running tasks), auto-scaling, and a `deploymentCircuitBreaker` that automatically halts and rolls back a bad rollout before it can fully replace a healthy service.

The deploy pipeline itself has a deliberate asymmetry: `.github/workflows/deploy.yml` builds and pushes the Docker image to GHCR automatically on every push to `main` (a safe, reversible action), while `.github/workflows/deploy-ecs.yml` is `workflow_dispatch`-only — a human has to explicitly trigger an actual production deploy from the Actions tab. AWS authentication uses OIDC (a short-lived, signed token proving "this is really GitHub Actions, running this workflow"), so no permanent `AWS_SECRET_ACCESS_KEY` needs to be stored anywhere at all.

### Cross-cutting: `main.py`

This is where the FastAPI app is assembled: CORS, the security headers middleware, the request ID middleware, the rate limiter (`slowapi` — global default of 120 requests/minute, with stricter per-endpoint limits on login and agent runs), and the `lifespan` function that runs once at startup (initialise the database, start the OpenTelemetry SDK, start the retraining scheduler) and once at shutdown (stop the scheduler, dispose the database connection pool).

---

## 9. The Database Layer

Twelve tables, managed by SQLAlchemy (async) and versioned with Alembic (9 migrations from the initial schema to today). The schema lives in `models/`, one file per table:

| Table | Purpose |
|---|---|
| `datasets` | Every ingested dataset — file path, content hash, schema, row/column counts |
| `connectors` | Saved connection configs (SQL, OAuth) with credentials Fernet-encrypted |
| `experiments` | Every training run — best model, score, holdout metrics, calibration info, `lifecycle_stage` |
| `rag_documents` | Indexed documents for the RAG pipeline |
| `users` / `user_api_keys` | Auth + BYOK |
| `fine_tune_jobs` | SFT and DPO training jobs |
| `forecast_jobs` | Time-series forecasting runs |
| `agent_traces` | Captured agent sessions, for the agent training pipeline |
| `registered_agents` | Imported/exported fine-tuned agents (the Agent Registry) |
| `retraining_policies` / `retraining_runs` | The scheduled retraining pipeline and its audit history |

**Why async SQLAlchemy:** every database call in a request handler is `await`-ed, so the event loop is free to serve other requests while waiting on I/O. This matters specifically because Crucible's request handlers frequently wait on something slow (a file read, a network call to S3) — a synchronous ORM would block the whole server during that wait.

**Why Alembic instead of `create_all()` in production:** `create_all()` is idempotent but can't *alter* an existing table — if you add a column to a model, `create_all()` silently does nothing on a database that already has that table. Alembic generates a versioned migration script that actually runs `ALTER TABLE`. The one exception is in-memory SQLite (used by the entire test suite), where there's no existing schema to alter, so `database.py`'s `init_db()` detects `:memory:` and uses `create_all()` directly — no migration history needed for a database that's destroyed at the end of every test.

A genuine bug surfaced during this build is worth recording here because it's instructive: **SQLite stores `server_default=func.now()` timestamps at second resolution as plain TEXT**, but SQLAlchemy's bind-parameter formatting for a Python `datetime` includes microseconds. Comparing the two directly in a `WHERE` clause (needed for cursor-based pagination) silently matched nothing, because SQLite compares TEXT-affinity columns lexicographically and `"...16:30:00"` is not equal to `"...16:30:00.000000"` as strings. The fix (`schemas/cursor_pagination.py`) detects the SQLite dialect and normalises both sides to second-resolution text via `strftime()` before comparing; PostgreSQL needs no such workaround because it stores genuine microsecond-precision timestamps that round-trip exactly.

---

## 10. The API Surface

24 FastAPI routers, each scoped to one functional area, all mounted under `/api/v1`. The full interactive list (with request/response schemas) is always available at `/docs` when running with `DEBUG=true`. Grouped by area:

| Area | Routers |
|---|---|
| Data | `datasets`, `connectors`, `profiling`, `drift`, `anomaly`, `data_contracts` |
| ML | `experiments`, `fairness`, `model_cards`, `evaluation` |
| Agents | `agent`, `agent_training` |
| Fine-tuning / RAG | `fine_tuning`, `rag`, `forecasting` |
| MLOps | `retraining`, `jobs`, `cloud` |
| Platform | `auth`, `api_keys`, `health` |
| Comparison | `ab_testing` |

Every router that touches user data depends on `get_current_user` (disabled in dev via `DISABLE_AUTH=true`). Every response uses one of two consistent envelope shapes: `DataResponse[T]` (`{"data": ...}`) for single resources, `PaginatedResponse[T]` (`{"data": [...], "pagination": {...}}`) for lists — this consistency is what lets the frontend's API client (`api/client.js`) use the same `.then(r => r.data.data)` unwrapping pattern everywhere instead of special-casing each endpoint.

---

## 11. The Frontend

React + Vite, 15 pages, using TanStack Query for all server state (no manual `useEffect`-driven fetching) and React Router for navigation. The shared design system lives in `components/ui.jsx` — `Card`, `Button`, `Spinner`, `EmptyState`, `StatusBadge`, `PageHeader`, `SectionLabel` — used consistently so every page looks like it belongs to the same product rather than each being styled independently.

| Page | What it's for |
|---|---|
| `DatasetsPage` / `DatasetDetailPage` | Upload, browse, profile datasets |
| `ConnectorsPage` | SQL / OAuth connector management |
| `ExperimentsPage` / `ExperimentDetailPage` | Training results, SHAP, calibration/pruner badges, lifecycle stage |
| `RAGPage` | Document upload, indexing, Q&A |
| `FineTuningPage` | SFT and DPO job submission |
| `ForecastingPage` | Time-series forecasting |
| `AgentPage` | ReAct single-agent and multi-agent chat interface |
| `AgentTrainingPage` | Trace capture, training-data conversion, agent registry |
| `MLOpsPage` | Job/cache monitoring, SageMaker submission, retraining policies |
| `EvaluationPage` | LLM-as-judge evaluation runs |
| `ABTestingPage` | Statistical comparison between two experiments |
| `SettingsPage` | BYOK key management, active LLM/tracking provider display |
| `LoginPage` | JWT auth |

---

## 12. Testing Strategy

**1,024 tests in the default suite, plus 10 separate real-server integration tests.** The split matters:

The default suite uses FastAPI's `TestClient`, which calls the ASGI app directly in-process — fast (the whole suite runs in under three minutes), but it never opens a real network socket. This is the right tool for testing business logic, and it's where the overwhelming majority of coverage lives: every router, every model, every algorithm, every edge case.

`tests/test_integration_real_server.py` does something different: it spawns a genuine `uvicorn` subprocess bound to a real port and talks to it with the `requests` library over real HTTP, including a real multipart upload and a real WebSocket handshake. This exists because some classes of bug are invisible to an in-process test client — `slowapi`'s rate limiter reads the *real* client IP from the ASGI connection info, which `TestClient` always fakes as the literal string `"testclient"`. A rate-limiting bug that only manifests with genuinely different client IPs would never be caught by the 1,024 fast tests, no matter how many were written. These 10 tests are excluded from the default `pytest` run (`pytest.ini`'s `addopts = -m "not integration"`) and run as a separate, slower CI job after the fast suite passes.

**Mock mode, used consistently across every external dependency:** SageMaker, BigQuery, the DPO trainer, W&B, MLflow — every integration with something slow, paid, or requiring credentials supports a recognisable "mock" trigger (a `mock-` prefixed ID, an empty API key, a patched-out client) that simulates realistic behaviour without the real dependency. This is what makes the full test suite runnable on a machine with no AWS account, no GPU, and no paid API keys.

---

## 13. Catalog of Key Design Decisions

This section collects the recurring architectural choices that show up across multiple subsystems, in one place, since understanding any one of them helps explain several others.

### The four-times-repeated provider abstraction

`StorageBackend` (Local/S3), `LLMBackend` (Anthropic/Bedrock/OpenAI-compatible), `JobQueueBackend` (in-memory/ARQ), `TrackingBackend` (MLflow/W&B). Each follows the identical shape: one abstract interface, one zero-infrastructure default, one production alternative, selected by a single config value, resolved through a factory function (`get_storage()`, `resolve_backend()`, `get_job_queue()`, `get_tracking_backend()`). This isn't four separate decisions — it's the same judgment applied consistently, because provider choice is an organisational decision, not a technical one. A platform that hard-codes one vendor forces every team that adopts it to either migrate their existing infrastructure or run a second, unintegrated system alongside it.

### BYOK resolution order, used identically everywhere Claude gets called

User's own key (encrypted) → server's fallback key → a clear error. Never a silent failure, never an ambiguous "it didn't work."

### Mock mode as a first-class feature, not a testing hack

Every expensive or credentialed external call — SageMaker, the DPO/SFT trainers, BigQuery — has a documented, recognisable way to simulate it. This isn't only for the test suite; it's also how someone evaluating this project can see the *entire* pipeline run end-to-end without an AWS account or a GPU.

### Calibration thresholds are not arbitrary

200 rows minimum before calibrating at all (below that, 5-fold cross-validation inside `CalibratedClassifierCV` would have under 40 samples per fold — too few for a stable calibration curve). 1,000 rows as the cutoff between sigmoid (Platt scaling, a 2-parameter fit, works on smaller samples) and isotonic regression (non-parametric, needs more data to learn a reliable monotone mapping).

### "Higher score always wins" is true by construction, not by convention

Crucible's Optuna studies always use `direction="maximize"`. For metrics that are naturally "lower is better" (RMSE, MAE), the scoring function negates them before they reach Optuna. This single fact is *why* the retraining pipeline's promotion logic (`new_score >= old_score + margin`) can be one line of code that works correctly for both classification and regression without a metric-direction lookup table — a smaller, more fragile design would have needed to special-case it.

### Two different background-work patterns, deliberately not unified

AutoML training (`jobs/manager.py`) needs live, granular, per-trial progress streamed over a WebSocket — it has its own bespoke tracker. Simpler fire-and-forget work (RAG indexing, SageMaker submission) uses the generic `JobQueueBackend`. These were kept separate on purpose: forcing live-progress training onto the generic interface (which only exposes coarse queued/running/completed/failed states) would have silently removed real, tested functionality just to make a cleaner architecture diagram.

### SageMaker's mock trigger matches the AWS documentation's own placeholder

`role_arn` starting with `mock-`, or containing `000000000000` (the account ID AWS's own example ARNs always use), triggers mock mode. Someone copy-pasting an AWS docs example to try the feature gets a working simulated result instead of a confusing authentication error.

### The retraining pipeline's drift baseline updates after every promotion

After a new model is promoted, the policy's `reference_dataset_id` becomes the newly-promoted model's training data — so the *next* drift check compares against the current production baseline, not whatever dataset happened to be first. Without this, drift severity would be measured against an increasingly stale reference, eventually triggering retraining on every check regardless of whether anything had actually changed recently.

### The `.crucible` bundle format wraps a standard, not a custom format

The `adapter/` directory inside a `.crucible` ZIP is exactly what `PeftModel.save_pretrained()` produces — any system with `transformers` and `peft` installed can load it with `PeftModel.from_pretrained(base_model, "adapter/")` without ever knowing Crucible exists. The manifest and bundled trace sample are additive transparency, not a lock-in mechanism.

### Logged production lessons, not just decisions

Two bugs found during this build are documented inline at their fix site rather than silently corrected, because they're genuinely instructive: the `pytest.ini` vs. `pyproject.toml` config-priority conflict (pytest.ini silently wins when both exist, so settings added only to `pyproject.toml` never took effect), and `requirements.txt` missing three unconditionally-imported packages (`python-jose`, `bcrypt`, `slowapi`) that happened to already be installed in the development sandbox — meaning the bug was invisible until checked against a truly clean install. Both are the kind of mistake that's easy to make and worth knowing to watch for.
