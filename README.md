Hi, I'm Yaroslav (Yaro) Povazhnyi
Data Analyst · ML Engineer · Full-Stack DS
SQL · Python · BigQuery · PyTorch · Tableau · Docker
I build end-to-end data systems — from raw ingestion and SQL modelling through ML training, causal inference, and production deployment. My work spans analytical deep-dives, AutoML tooling, and research-grade ML platforms.

Skills & Tools
AreaTechnologiesLanguagesPython · SQLML / Deep LearningPyTorch · scikit-learn · XGBoost · LightGBM · PyCaret · implicitStatisticsscipy · statsmodels · PyMC · Bayesian inference · causal inferenceDatabasesBigQuery · PostgreSQL · MySQL · MongoDB · SQLiteServing & InfraFastAPI · Docker · Docker Compose · uvicornVisualisationTableau · Streamlit · Gradio · Plotly · Google SheetsTrackingMLflow · DVCPython Librariespandas · NumPy · Matplotlib · seaborn · pyarrowWorkflowGit · GitHub Actions · CI/CD

Featured Projects

CineML Platform — End-to-End ML Research & Experimentation System
PyTorch · FastAPI · Streamlit · Docker Compose · BigQuery · MLflow · DDPM · ViT
A full-stack ML platform that simulates the core systems a streaming company needs to operate — from data ingestion and personalised recommendations through rigorous A/B testing, causal analysis, and generative artwork. Five interconnected modules built around a single data universe (MovieLens 25M + TMDB + synthetic event logs).
What it does:

M1 — Data Pipeline: Ingests MovieLens 25M + TMDB metadata + poster images. Generates synthetic streaming event logs (impressions, clicks, completions) with a position-decay CTR simulator. Uploads to BigQuery with DAY partitioning and clustering for cost-efficient analytics.
M2 — Personalisation Engine: Two-Tower neural recommender (PyTorch, BPR loss, cosine similarity) + ALS matrix factorisation on implicit feedback. Full offline evaluation: NDCG@k, MAP@k, Hit@k, Novelty, Catalogue Coverage. Served via FastAPI at :8001.
M3 — A/B & Causal Inference Engine: Four statistical layers — frequentist z-test + Welch's t-test, Bayesian Beta-Binomial (P(treatment > control)), SPRT sequential monitoring, DiD via OLS (HC3 SEs) + Propensity Score Matching with bootstrap ATT. FastAPI at :8002, Streamlit dashboard at :8503.
M4 — Analysis Memo: Netflix-style DS investigation — problem framing, long-tail discovery metric design, OLS with HC3 controls, cohort breakdown (power/casual/lurker), findings, forward-looking measurement plan.
M5 — Diffusion + ViT: DDPM implemented from scratch (Ho et al. 2020) with DDIM sampling (Song et al. 2021). Class-conditional U-Net with cross-attention genre conditioning — the same mechanism used in Stable Diffusion. ViT-B/16 fine-tuned on TMDB posters with Attention Rollout explainability (Abnar & Zuidema 2020). Gradio demo at :7860.

Deployed services (Docker Compose):
ServicePortDescriptionRecommender API:8001FastAPI — /recommend, /similar, /batch-recommendA/B Engine API:8002FastAPI — /frequentist, /bayesian, /sample-sizeA/B Dashboard:8503Streamlit — live statistical analysisDiffusion Demo:7860Gradio — poster generation + ViT classifierUnified Interface:8501Streamlit — full platform control + pipeline runner

📁 Python/cineml/ · streamlit run interface/app.py > ['https://github.com/yapovazhnyi-blip/analytics-and-ml-portfolio/blob/ecb8e3e746eddcc17759556bcd04c1f6a72d8827/Python/cineml/README.md']


AutoML Studio — Zero-Code Machine Learning App
Python · Streamlit · PyCaret · SQL · BigQuery · MongoDB
A full-stack AutoML web application that lets anyone train and compare ML models without writing code.
What it does:

Connects to 7 data sources: CSV, Excel, PostgreSQL, MySQL, MariaDB, BigQuery, MongoDB, SQLite
Cleans missing data column-by-column with configurable fill strategies
Persists cleaned data as a durable artifact for full training provenance
Benchmarks 15–25 ML algorithms with k-fold cross-validation using PyCaret
Produces a ranked leaderboard, model visualisations (Confusion Matrix, ROC/AUC, Feature Importance), and a downloadable production .pkl model

Key technical decisions:

Lazy PyCaret imports keep startup fast despite 50+ transitive dependencies
File-based plot capture handles PyCaret's cross-version inconsistency in plot_model() return types


📁 Python/automl-studio/ · streamlit run app.py ['https://github.com/yapovazhnyi-blip/analytics-and-ml-portfolio/blob/ecb8e3e746eddcc17759556bcd04c1f6a72d8827/Python/automl_studio/README.md']


End-to-End Sales Analytics — BigQuery SQL
BigQuery SQL · Google Sheets
A structured SQL analytics workflow covering the full pipeline from raw schema to KPI reporting.
What it does:

Renames and standardises raw field names for consistency
Runs data quality checks before any aggregation
Calculates revenue metrics and merges Orders and Products datasets
Builds temporary analytical tables for downstream BI consumption

Skills demonstrated: CTEs · JOINs · window functions · aggregations · schema design · data quality validation

📁 SQL/end-to-end-sales-analytics-bigquery/ ['https://github.com/yapovazhnyi-blip/analytics-and-ml-portfolio/tree/ecb8e3e746eddcc17759556bcd04c1f6a72d8827/SQL/end-to-end-sales-analytics-bigquery']


Drone Usage Analytics — Tableau Story
Tableau Public
An interactive data storytelling dashboard covering drone usage patterns in the Russia–Ukraine war, built for a non-technical audience.

Structures complex event data into a narrative-driven Tableau Story
Uses layered filters and calculated fields to enable audience-led exploration
Designed for clarity — every visual serves a specific analytical question


🔗 View on Tableau Public ['https://public.tableau.com/app/profile/yaro.povazhnyi2155/viz/DroneusageinwarbetweenRussiaandUkraine/Story1']


Tableau Visualisation Case Study
Tableau Public
An interactive dashboard demonstrating structured visual insight design — layout hierarchy, colour encoding, and KPI storytelling.

🔗 View on Tableau Public ['https://public.tableau.com/app/profile/yaro.povazhnyi2155/viz/TestProject_17678821298320/Story1']


CSV Data Exploration — Google Sheets Pipeline
Google Sheets · CSV
A data cleaning and structuring workflow built entirely in Google Sheets: filtering, aggregation, category-level revenue analysis, and automated data preparation.

📁 SpreadSheets/GoogleTables/csv-data-exploration/ ['https://github.com/yapovazhnyi-blip/analytics-and-ml-portfolio/tree/ecb8e3e746eddcc17759556bcd04c1f6a72d8827/SpreadSheets/GoogleTables/csv-data-exploration']


Repository Structure
analytics-and-ml-portfolio/
│
├── Python/
│   ├── cineml/                    # CineML Platform — 5-module ML system
│   │   ├── module1_data_pipeline/ #   BigQuery + TMDB + event simulator
│   │   ├── module2_recommender/   #   Two-Tower neural model + ALS + FastAPI
│   │   ├── module3_ab_engine/     #   Frequentist + Bayesian + DiD + PSM
│   │   ├── module4_analysis_memo/ #   Netflix-style DS investigation
│   │   ├── module5_diffusion_vit/ #   DDPM from scratch + ViT + Gradio
│   │   ├── interface/             #   Unified Streamlit platform
│   │   └── docker-compose.yml     #   4 production services
│   │
│   └── automl-studio/             # AutoML Studio — Streamlit + PyCaret
│
├── SQL/
│   └── end-to-end-sales-analytics-bigquery/   # BigQuery SQL pipeline
│
├── SpreadSheets/
│   └── GoogleTables/
│       └── csv-data-exploration/  # Google Sheets analytics workflow
│
└── Visualization/                 # Tableau dashboards and visual case studies
