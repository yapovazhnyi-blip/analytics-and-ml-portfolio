# 👋 Hi, I'm Yaroslav (Yaro) Povazhnyi

### Data Analyst & ML Engineer  
**SQL · BigQuery · Python · Tableau · Machine Learning**

I build end-to-end data workflows — from raw data ingestion and cleaning through SQL modelling, visualisation, and ML-powered automation. My work focuses on turning messy datasets into clear, business-ready insights and deployable models.

---

## 🛠️ Skills & Tools

| Area | Technologies |
|---|---|
| **Languages** | Python · SQL |
| **Databases** | BigQuery · PostgreSQL · MySQL · MongoDB · SQLite |
| **ML / AutoML** | PyCaret · scikit-learn · XGBoost · LightGBM |
| **Visualisation** | Tableau · Google Sheets |
| **Python Libraries** | Streamlit · pandas · NumPy · Matplotlib |
| **Workflow** | Git · Docker |

---

## 🚀 Featured Projects

---

### ⚗️ AutoML Studio — Zero-Code Machine Learning App
**Python · Streamlit · PyCaret · SQL · BigQuery · MongoDB**

A full-stack AutoML web application that lets anyone train and compare ML models without writing code.

**What it does:**
- Connects to 7 data sources: CSV, Excel, PostgreSQL, MySQL, MariaDB, BigQuery, MongoDB, SQLite
- Cleans missing data column-by-column with configurable fill strategies
- Persists cleaned data as a durable artifact (CSV file or database temp table)
- Benchmarks 15–25 ML algorithms with k-fold cross-validation using PyCaret
- Produces a ranked leaderboard, model visualisations (Confusion Matrix, ROC/AUC, Feature Importance), and a downloadable production `.pkl` model

**Key technical decisions:**
- Lazy PyCaret imports keep startup fast despite 50+ transitive dependencies
- Temp artifact persistence gives full training provenance — the exact dataset the model trained on is always recoverable
- File-based plot capture handles PyCaret's cross-version inconsistency in `plot_model()` return types

> 📁 [`Python/automl-studio/`](.Python/automl_studio) · `streamlit run app.py`

---

### 📊 End-to-End Sales Analytics — BigQuery SQL
**BigQuery SQL · Google Sheets**

A structured SQL analytics workflow covering the full pipeline from raw schema to KPI reporting.

**What it does:**
- Renames and standardises raw field names for consistency
- Runs data quality checks before any aggregation
- Calculates revenue metrics and merges Orders and Products datasets
- Builds temporary analytical tables for downstream BI consumption

**Skills demonstrated:** CTEs · JOINs · aggregations · schema design · data quality validation

> 📁 [`SQL/end-to-end-sales-analytics-bigquery/`](./SQL/end-to-end-sales-analytics-bigquery/docs/README.md)

---

### 🚁 Drone Usage Analytics — Tableau Story
**Tableau Public**

An interactive data storytelling dashboard covering drone usage patterns in the Russia–Ukraine war, built for a non-technical audience.

**What it does:**
- Structures complex event data into a narrative-driven Tableau Story
- Uses layered filters and calculated fields to enable audience-led exploration
- Designed for clarity first — every visual serves a specific analytical question

> 🔗 [View on Tableau Public](https://public.tableau.com/app/profile/yaro.povazhnyi2155/viz/DroneusageinwarbetweenRussiaandUkraine/Story1)

---

### 📈 Tableau Visualisation Case Study
**Tableau Public**

An interactive dashboard demonstrating structured visual insight design — layout hierarchy, colour encoding, and KPI storytelling.

> 🔗 [View on Tableau Public](https://public.tableau.com/app/profile/yaro.povazhnyi2155/viz/TestProject_17678821298320/Story1)

---

### 📂 CSV Data Exploration — Google Sheets Pipeline
**Google Sheets · CSV**

A data cleaning and structuring workflow built entirely in Google Sheets, covering filtering, aggregation, category-level revenue analysis, and automated data preparation.

> 📁 [`SpreadSheets/GoogleTables/csv-data-exploration/`](./SpreadSheets/GoogleTables/csv-data-exploration/docs/README.md)

---

## 🗂️ Repository Structure

```
yaro-povazhnyi2155/
│
├── Python/
│   └── automl-studio/          # AutoML Studio — Streamlit + PyCaret app
│
├── SQL/
│   └── end-to-end-sales-analytics-bigquery/   # BigQuery SQL pipeline
│
├── SpreadSheets/
│   └── GoogleTables/
│       └── csv-data-exploration/              # Google Sheets analytics
│
└── Visualization/              # Tableau dashboards and visual case studies
```

---

## 📊 Analytics Focus

- End-to-end data workflows (ingest → clean → model → visualise)
- SQL joins, window functions, and KPI modelling
- AutoML and model selection pipelines
- Dashboard UX and data storytelling
- Reproducible analytics with documented provenance

---

## 📫 Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Yaroslav_Povazhnyi-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yaroslav-povazhnyi-617285289/)
[![Tableau](https://img.shields.io/badge/Tableau_Public-Portfolio-E97627?logo=tableau&logoColor=white)](https://public.tableau.com/app/profile/yaro.povazhnyi2155)
