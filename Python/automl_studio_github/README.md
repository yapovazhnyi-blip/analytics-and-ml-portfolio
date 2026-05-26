**Connect your data · Clean it · Compare every model · Ship the best one**

A zero-code AutoML web application built with [Streamlit](https://streamlit.io) and [PyCaret](https://pycaret.org).  
Load data from any source, handle missing values with full transparency, benchmark every ML algorithm, and download a production-ready model — no data science experience required.

![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit&logoColor=white)
![PyCaret](https://img.shields.io/badge/PyCaret-3.3.2-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Features](#-features)
- [Five-Step Workflow](#-five-step-workflow)
- [Quick Start](#-quick-start)
- [Data Source Setup](#-data-source-setup)
- [Data Cleaning](#-data-cleaning)
- [Training Configuration](#-training-configuration)
- [Model Visualisations](#-model-visualisations)
- [Results & Downloads](#-results--downloads)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Known Limitations](#-known-limitations)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## ✨ Features

| Feature | Details |
|---|---|
| **7 data sources** | CSV, Excel, PostgreSQL, MySQL, MariaDB, BigQuery, MongoDB, SQLite |
| **Smart data cleaning** | Per-column null handling: mean / median / mode / N/A / exclude / drop rows |
| **Temp artifact persistence** | Cleaned data saved to a CSV file (file sources) or a temp table/collection (DB sources) |
| **Auto task detection** | Classification vs Regression detected automatically from the target column |
| **Full model comparison** | 15–25 algorithms benchmarked with stratified k-fold cross-validation |
| **Visual leaderboard** | Score bars, KPI metric tiles, 👑 crown for the winner |
| **Model visualisations** | Confusion Matrix, ROC/AUC, Feature Importance, Classification Report (and regression equivalents) |
| **Three downloads** | Results CSV · Best model `.pkl` · Cleaned data CSV |
| **Inference code snippet** | Ready-to-run Python code generated with actual model name and score |

---

## 🔄 Five-Step Workflow

```
01 · CONNECT
     Load data from CSV, Excel, or any supported database
          ↓
02 · CLEAN
     Fix nulls per column → cleaned data saved as a durable artifact
     ┌─────────────────────────────────────────────────────────────┐
     │ CSV / Excel / Demo  →  ~/automl_studio_output/cleaned.csv  │
     │ PostgreSQL / MySQL  →  table automl_cleaned_<ts> in same DB │
     │ SQLite              →  table automl_cleaned_<ts> in .db     │
     │ BigQuery            →  <project>.<dataset>.automl_cleaned_  │
     │ MongoDB             →  collection automl_cleaned_<ts>       │
     └─────────────────────────────────────────────────────────────┘
          ↓
03 · CONFIGURE
     Target column · task type · train split · CV folds
          ↓
04 · TRAIN
     PyCaret setup() + compare_models() benchmarks all algorithms
          ↓
05 · RESULTS
     Ranked leaderboard · KPI tiles · plots · downloads · code snippet
```

---

## 🚀 Quick Start

### Requirements

> ⚠️ **Python 3.9, 3.10, or 3.11 required.** PyCaret does not support Python 3.12+.

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/automl-studio.git
cd automl-studio

# 2. Create a virtual environment with a compatible Python version
python3.11 -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

### Windows — Execution Policy

If you see a PowerShell error about running scripts being disabled:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Docker

```bash
docker compose up --build
# Open http://localhost:8501
```

---

## 🔌 Data Source Setup

### CSV / Excel
Upload any `.csv`, `.xlsx`, or `.xls` file via the sidebar. Choose the correct separator for CSV files.

### PostgreSQL
```
Host:     your-host          Port: 5432
Database: mydb               User: postgres
Password: ••••
Query:    SELECT * FROM my_table LIMIT 50000;
```
Cleaned artifact → table `automl_cleaned_<timestamp>` in the same database.

### MySQL / MariaDB
```
Host:     your-host          Port: 3306
Database: mydb               User: root
Password: ••••
Query:    SELECT * FROM my_table LIMIT 50000;
```
Cleaned artifact → table `automl_cleaned_<timestamp>` in the same database.

### Google BigQuery
1. Create a GCP service account with **BigQuery Data Viewer** and **BigQuery Data Editor** roles
2. Download the JSON key file
3. Enter your Project ID and Dataset name
4. Upload the JSON key and write your SQL query

Cleaned artifact → `<project>.<dataset>.automl_cleaned_<timestamp>` BigQuery table.

### MongoDB
```
URI:        mongodb://localhost:27017
Database:   mydb
Collection: mycollection
Filter:     {}
```
Cleaned artifact → collection `automl_cleaned_<timestamp>` in the same database.

### SQLite
Upload your `.db` or `.sqlite` file directly.  
Cleaned artifact → table `automl_cleaned_<timestamp>` written back into the same file.

---

## 🧹 Data Cleaning

After loading, the app scans every column for missing values and presents a decision row for each.

| Strategy | Numeric | Text | When to use |
|---|:---:|:---:|---|
| Fill with mean | ✅ | ❌ | Normally distributed columns |
| Fill with median | ✅ | ❌ | Columns with outliers or skewed distributions |
| Fill with mode | ✅ | ✅ | Discrete integers, categorical text |
| Fill with "N/A" | ❌ | ✅ | When "missing" should be an explicit category |
| Exclude column | ✅ | ✅ | Column has > 50% missing or low predictive value |
| Drop rows | ✅ | ✅ | < 5% missing and dataset is large enough |

**Why the artifact matters:**  
After clicking *Apply Cleaning*, the cleaned DataFrame is saved to a durable location outside Python memory. This gives you a verifiable record of exactly what the model was trained on — shareable with teammates and queryable independently of the app.

---

## ⚙️ Training Configuration

| Option | Description |
|---|---|
| **Target column** | The column to predict |
| **Task type** | Auto-detected or manually set to Classification / Regression |
| **Models to compare** | All (15–25 models) · Top 10 · Top 5 |
| **Train split %** | Fraction of data used for training (60–90%) |
| **Normalize features** | Z-score normalisation — essential for KNN, SVM, linear models |
| **CV folds** | 3, 5, or 10-fold stratified cross-validation |
| **n_jobs** | CPU parallelism (-1 = all cores) |

### Tips for small datasets
- Use **Top 5** models to reduce training time and failure risk
- Use **3 CV folds** instead of 5 or 10
- Keep **Train split %** at 80 or above

---

## 📈 Model Visualisations

Plots are generated lazily — only when you click a button — because each one re-evaluates the model on the hold-out test set.

**Classification:**

| Plot | What to look for |
|---|---|
| Confusion Matrix | Diagonal should be dark. Off-diagonal cells are errors. |
| ROC / AUC Curve | AUC > 0.9 is strong. Curve should hug the top-left corner. |
| Feature Importance | Dominant single feature (> 60%) may indicate data leakage. |
| Classification Report | Low recall = model misses that class. Low precision = over-predicts it. |

**Regression:**

| Plot | What to look for |
|---|---|
| Residuals | Should be randomly scattered around zero — any pattern is a problem. |
| Predicted vs Actual | Dots should cluster tightly along the diagonal. |
| Feature Importance | Same leakage check as classification. |
| Learning Curve | Train score >> test score = overfitting. Both low = underfitting. |

---

## 📦 Results & Downloads

After training completes the results dashboard provides:

- **KPI tiles** — Best Accuracy / R², AUC / RMSE, models tested, training time
- **Visual leaderboard** — every model ranked with a score bar and a 👑 for the winner
- **Full comparison table** — all CV metrics with green gradient on the primary metric
- **Results CSV** — full leaderboard as a spreadsheet
- **Best model `.pkl`** — the winning model including its preprocessing pipeline
- **Cleaned data CSV** — the exact dataset the model was trained on (file sources)

### Using the downloaded model

```python
from pycaret.classification import load_model, predict_model  # or .regression
import pandas as pd

# Load model — includes the full preprocessing pipeline
model = load_model("best_automl_model")

# Predict on new raw data (no preprocessing needed)
new_data    = pd.read_csv("your_new_data.csv")
predictions = predict_model(model, data=new_data)
print(predictions.head())
```

---

## 🗂️ Project Structure

```
automl-studio/
│
├── app.py                        # Entire Streamlit application
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Python 3.11 container
├── docker-compose.yml            # One-command local deployment
│
├── .github/
│   └── workflows/
│       └── ci.yml                # CI: syntax check + import validation
│
├── .gitignore                    # Keeps secrets, models, data out of git
├── LICENSE                       # MIT
└── README.md                     # This file
```

**Output files** (created at runtime, not committed to git):

```
~/automl_studio_output/
├── best_automl_model.pkl         # Trained model + preprocessing pipeline
├── automl_cleaned_<ts>.csv       # Cleaned dataset artifact
├── confusion_matrix.png          # Generated plots
├── auc.png
└── ...
```

---

## 📦 Requirements

```
pycaret[full]==3.3.2
streamlit>=1.35.0
pandas>=1.5.0,<2.2
numpy>=1.23.0,<2
sqlalchemy>=2.0
psycopg2-binary>=2.9
mysql-connector-python>=8.0
pymongo>=4.6
google-cloud-bigquery>=3.0
google-auth>=2.0
pandas-gbq>=0.19
openpyxl>=3.1
xgboost
lightgbm
```

Install everything:

```bash
pip install -r requirements.txt
```

---

## ⚠️ Known Limitations

| Limitation | Workaround |
|---|---|
| Python 3.12+ not supported | Use Python 3.11 or the Docker image |
| Large datasets (> 100k rows) slow to train | Use **Top 5** models mode |
| Target column with many unique classes | Check you selected the right column — high cardinality usually means it's an ID field |
| Feature Importance unavailable for some models | Use Confusion Matrix or AUC instead |
| BigQuery write-back requires Editor role | Grant **BigQuery Data Editor** to the service account |

---

## 🔧 Troubleshooting

**`missing ScriptRunContext` warnings**  
You ran the file with `python app.py` instead of `streamlit run app.py`.

**`No module named 'pycaret'`**  
Your virtual environment isn't active. Run `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (macOS/Linux) first.

**`fix_imbalance / remove_outliers` parameter errors**  
These were removed in PyCaret 3.3. Remove them from the `setup()` call.

**`test_size should be >= number of classes`**  
Your target column likely has too many unique values — it may be an ID or free-text column, not a class label.

**`Estimator xgboost Not Available`**  
Install it: `pip install xgboost lightgbm --upgrade`

**Plot not saving to disk**  
Ensure the app has write access to `~/automl_studio_output/`. On Windows this is `C:\Users\<your-username>\automl_studio_output\`.

---

## 📄 License

[MIT](LICENSE) — free to use, modify, and distribute.

---

## 🙏 Built With

[Streamlit](https://streamlit.io) · [PyCaret](https://pycaret.org) · [pandas](https://pandas.pydata.org) · [scikit-learn](https://scikit-learn.org) · [XGBoost](https://xgboost.ai) · [LightGBM](https://lightgbm.readthedocs.io) · [SQLAlchemy](https://www.sqlalchemy.org) · [PyMongo](https://pymongo.readthedocs.io) · [pandas-gbq](https://pandas-gbq.readthedocs.io)
