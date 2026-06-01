# Hi, I'm Yaroslav (Yaro) Povazhnyi

### Data Analyst · ML Engineer · Full-Stack DS

**SQL · Python · BigQuery · PyTorch · Tableau · Docker**

I build end-to-end data systems — from raw ingestion and SQL modelling through ML training, causal inference, and production API deployment. My work spans analytical deep-dives, AutoML tooling, and research-grade ML platforms with live Docker services.

---

## Skills & Tools

| Area | Technologies |
|------|-------------|
| **Languages** | Python · SQL |
| **ML / Deep Learning** | PyTorch · scikit-learn · XGBoost · LightGBM · PyCaret · implicit |
| **Statistics** | scipy · statsmodels · PyMC · Bayesian inference · causal inference (DiD, PSM) |
| **Databases** | BigQuery · PostgreSQL · MySQL · MongoDB · SQLite |
| **Serving & Infra** | FastAPI · Docker · Docker Compose · uvicorn |
| **Visualisation** | Tableau · Streamlit · Gradio · Plotly |
| **Tracking** | MLflow · DVC |
| **Python Libraries** | pandas · NumPy · Matplotlib · seaborn · pyarrow |
| **Workflow** | Git · GitHub Actions · CI/CD |

---

## Featured Projects

---

### CineML Platform — End-to-End ML Research & Experimentation System

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://github.com/yapovazhnyi-blip/analytics-and-ml-portfolio/tree/main/Python/cineml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-ee4c2c?logo=pytorch&logoColor=white)](https://github.com/yapovazhnyi-blip/analytics-and-ml-portfolio/tree/main/Python/cineml)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)](https://github.com/yapovazhnyi-blip/analytics-and-ml-portfolio/tree/main/Python/cineml)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://github.com/yapovazhnyi-blip/analytics-and-ml-portfolio/tree/main/Python/cineml)

A full-stack ML platform built around a single data universe (MovieLens 25M + TMDB + synthetic event logs). Five interconnected modules that simulate the core ML systems a streaming company operates — from data ingestion through production-deployed APIs.

| Module | What it does | Key tech |
|--------|-------------|----------|
| **M1 Data Pipeline** | MovieLens 25M + TMDB API + synthetic event simulator → BigQuery | DVC · pyarrow · BigQuery |
| **M2 Personalisation** | Two-Tower neural recommender (BPR loss) + ALS · served via FastAPI `:8001` | PyTorch · implicit · FastAPI |
| **M3 A/B Engine** | Frequentist · Bayesian · SPRT · DiD · PSM · FastAPI `:8002` + dashboard `:8503` | PyMC · statsmodels · Streamlit |
| **M4 Analysis Memo** | Netflix-style DS investigation — long-tail discovery metric, OLS, cohort analysis | statsmodels · OLS · HC3 |
| **M5 Diffusion + ViT** | DDPM from scratch (Ho 2020) + DDIM + ViT-B/16 fine-tune + Attention Rollout · Gradio `:7860` | PyTorch · HuggingFace · Gradio |

> 📁 **[Python/cineml/](https://github.com/yapovazhnyi-blip/analytics-and-ml-portfolio/tree/main/Python/cineml)** · `streamlit run interface/app.py` · `docker compose up -d`

---

### AutoML Studio — Zero-Code Machine Learning App

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://github.com/yapovazhnyi-blip/analytics-and-ml-portfolio/tree/main/Python/automl_studio)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-ff4b4b?logo=streamlit&logoColor=white)](https://github.com/yapovazhnyi-blip/analytics-and-ml-portfolio/tree/main/Python/automl_studio)
[![PyCaret](https://img.shields.io/badge/PyCaret-AutoML-orange)](https://github.com/yapovazhnyi-blip/analytics-and-ml-portfolio/tree/main/Python/automl_studio)

A full-stack AutoML web app — connect any data source, clean, and benchmark 15–25 ML algorithms with k-fold cross-validation without writing a single line of code.

- Connects to 7 data sources: CSV, Excel, PostgreSQL, MySQL, MariaDB, BigQuery, MongoDB, SQLite
- Column-by-column missing data handling with configurable fill strategies
- Ranked leaderboard with Confusion Matrix, ROC/AUC, Feature Importance plots
- One-click downloadable production `.pkl` model with full training provenance

> 📁 **[Python/automl_studio/](https://github.com/yapovazhnyi-blip/analytics-and-ml-portfolio/tree/main/Python/automl_studio)** · [README](https://github.com/yapovazhnyi-blip/analytics-and-ml-portfolio/blob/main/Python/automl_studio/README.md) · `streamlit run app.py`

---

### End-to-End Sales Analytics — BigQuery SQL

[![BigQuery](https://img.shields.io/badge/BigQuery-SQL-4285F4?logo=google-cloud&logoColor=white)](https://github.com/yapovazhnyi-blip/analytics-and-ml-portfolio/tree/main/SQL/end-to-end-sales-analytics-bigquery)

A structured SQL analytics workflow — raw schema to KPI reporting.

- Data quality checks before aggregation, field standardisation, CTE-based pipeline
- Revenue metrics, Orders × Products joins, temporary analytical tables for BI

> 📁 **[SQL/end-to-end-sales-analytics-bigquery/](https://github.com/yapovazhnyi-blip/analytics-and-ml-portfolio/tree/main/SQL/end-to-end-sales-analytics-bigquery)** · [README](https://github.com/yapovazhnyi-blip/analytics-and-ml-portfolio/blob/main/SQL/end-to-end-sales-analytics-bigquery/docs/README.md)

---

### Drone Usage Analytics — Tableau Story

[![Tableau](https://img.shields.io/badge/Tableau-Public-E97627?logo=tableau&logoColor=white)](https://public.tableau.com/app/profile/yaro.povazhnyi2155/viz/DroneusageinwarbetweenRussiaandUkraine/Story1)

Interactive data storytelling dashboard covering drone usage patterns in the Russia–Ukraine war — narrative-driven Tableau Story with layered filters and calculated fields for audience-led exploration.

> 📁 **[Visualization/](https://github.com/yapovazhnyi-blip/analytics-and-ml-portfolio/tree/main/Visualization)** · 🔗 [View on Tableau Public](https://public.tableau.com/app/profile/yaro.povazhnyi2155/viz/DroneusageinwarbetweenRussiaandUkraine/Story1)

---

### Tableau Visualisation Case Study

[![Tableau](https://img.shields.io/badge/Tableau-Public-E97627?logo=tableau&logoColor=white)](https://public.tableau.com/app/profile/yaro.povazhnyi2155/viz/TestProject_17678821298320/Story1)

Interactive dashboard demonstrating layout hierarchy, colour encoding, and KPI storytelling for structured visual insight design.

> 🔗 [View on Tableau Public](https://public.tableau.com/app/profile/yaro.povazhnyi2155/viz/TestProject_17678821298320/Story1)

---

### CSV Data Exploration — Google Sheets Pipeline

[![Sheets](https://img.shields.io/badge/Google_Sheets-Analytics-34A853?logo=google-sheets&logoColor=white)](https://github.com/yapovazhnyi-blip/analytics-and-ml-portfolio/tree/main/SpreadSheets/GoogleTables/csv-data-exploration)

Data cleaning and structuring workflow in Google Sheets — filtering, aggregation, category-level revenue analysis, automated data preparation.

> 📁 **[SpreadSheets/GoogleTables/csv-data-exploration/](https://github.com/yapovazhnyi-blip/analytics-and-ml-portfolio/tree/main/SpreadSheets/GoogleTables/csv-data-exploration)** · [README](https://github.com/yapovazhnyi-blip/analytics-and-ml-portfolio/blob/main/SpreadSheets/GoogleTables/csv-data-exploration/docs/README.md)

---

## Repository Structure

```
analytics-and-ml-portfolio/
│
├── Python/
│   ├── cineml/                         # CineML — 5-module ML platform
│   │   ├── module1_data_pipeline/      #   BigQuery + TMDB + event simulator
│   │   ├── module2_recommender/        #   Two-Tower neural model + ALS + FastAPI :8001
│   │   ├── module3_ab_engine/          #   Frequentist + Bayesian + DiD + PSM + FastAPI :8002
│   │   ├── module4_analysis_memo/      #   Netflix-style DS investigation
│   │   ├── module5_diffusion_vit/      #   DDPM from scratch + ViT + Gradio :7860
│   │   ├── interface/                  #   Unified Streamlit platform :8501
│   │   └── docker-compose.yml          #   4 production Docker services
│   │
│   └── automl_studio/                  # AutoML Studio — Streamlit + PyCaret
│
├── SQL/
│   └── end-to-end-sales-analytics-bigquery/
│
├── SpreadSheets/
│   └── GoogleTables/
│       └── csv-data-exploration/
│
└── Visualization/                      # Tableau dashboards
```

---

## Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Yaroslav_Povazhnyi-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yaroslav-povazhnyi-617285289/)
[![Tableau](https://img.shields.io/badge/Tableau_Public-Portfolio-E97627?logo=tableau&logoColor=white)](https://public.tableau.com/app/profile/yaro.povazhnyi2155)
