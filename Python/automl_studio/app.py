"""
AutoML Studio - app.py

Five-step workflow
    1. CONNECT        - load data from CSV/Excel or 6 supported database engines
    2. CLEAN          - per-column null handling (fill / exclude / drop rows)
                        -> saves the cleaned result as TEMPORARY ARTIFACT
                           - CSV / Excel / Demo -> /automl_cleaned_<ts>.csv
                           - PostgreSQL / MySQL / - table automl_cleaned_<ts> in same DB
                             MariaDB / SQLite
                           - BigQuery - table automl_cleaned_<ts> in same dataset
                           - MongoDB - collection automl_cleaned_<ts> in same DB
    3. CONFIGURE      - pick target column, task type, train/test split, CV folds
    4. TRAIN          - PyCaret setup() + compare_models() benchmarks all algorithms
    5. RESULTS        - ranked leaderboard, KPI tiles, model/CSV download, code snippet


The temporary artifact explained
    After the user clicks "Apply Cleaning" the cleaned DataFrame is persisted
    outside Python memory so it can be:
        a) Re-loaded from disk/DB independently of the app process
        b) Inspected or shared with teammates before modelling begins
        c) Used as the definitive input to PyCaret (guarantees model was trained on exactly what the user approved, not a silent in-memory copy)
    File sources get a downloadable CSV button; DB sources get the table/collection name printed with a "drop" button.
"""

# Libraries
import io
import os
import time
import tempfile
import traceback
import datetime
from pathlib import Path


import streamlit as st
import pandas as pd
import numpy as np

# Step 1 - Page config
st.set_page_config(
    page_title="AutoML Studio",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Step 2 - Global CSS
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

        :root {
            --bg:      #0d0f14;
            --bg2:     #13161e;
            --bg3:     #1a1f2c;
            --border:  #252b3b;
            --accent:  #00e5a0;
            --accent2: #6c63ff;
            --accent3: #ff6b6b;
            --text:    #e8eaf0;
            --muted:   #7b82a0;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: var(--bg) !important;
            color: var(--text) !important;
            font-family: 'DM Sans', sans-serif;
        }
        [data-testid="stSidebar"] {
            background: var(--bg2) !important;
            border-right: 1px solid var(--border);
        }
        h1, h2, h3 { font-family: 'Space Mono', monospace; }

        /* -- Hero banner -- */
        /* ::before creates two soft radial glows with no JS needed */
        .hero {
            background: linear-gradient(135deg, #0d0f14 0%, #13161e 50%, #0d1a14 100%);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 2.5rem 3rem;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }
        .hero::before {
            content: '';
            position: absolute;
            top: -50%; left: -50%;
            width: 200%; height: 200%;
            background:
                radial-gradient(circle at 30% 40%, rgba(0,229,160,0.05) 0%, transparent 50%),
                radial-gradient(circle at 70% 60%, rgba(108,99,255,0.05) 0%, transparent 50%);
            pointer-events: none;
        }
        .hero h1 { font-size: 2.4rem; color: var(--accent); margin: 0 0 .4rem; }
        .hero p { color: var(--muted); margin: 0; font-size: 1.05rem; }

        /* -- Reusable card -- */
        .card {
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        .card-title {
            font-family: 'Space Mono', monospace;
            font-size: .75rem;
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: .12em;
            margin-bottom: .8rem;
        }

        /* -- KPI metric tiles -- */
        .metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
        .metric {
            background: var(--bg3);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1rem 1.4rem;
            flex: 1; min-width: 120px;
            text-align: center;
        }
        .metric-val { font-family: 'Space Mono', monospace; font-size: 1.6rem; color: var(--accent); }
        .metric-lbl { font-size: .75rem; color: var(--muted); margin-top: .2rem; }

        /* -- Model leaderboard rows -- */
        .model-row {
            display: flex; align-items: center;
            background: var(--bg3);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: .75rem 1rem;
            margin-bottom: .5rem;
            gap: 1rem;
        }
        .model-rank { font-family: 'Space Mono', monospace; color: var(--muted); font-size: .85rem; width: 28px; }
        .model-name { flex: 1; font-weight: 500; }
        .model-score { font-family: 'Space Mono', monospace; font-size: 1rem; color: var(--accent); min-width: 70px; text-align: right; }
        .model-bar-wrap { width: 120px; height: 6px; background: var(--border); border-radius: 3px; }
        .model-bar { height: 100%; border-radius: 3px; background: linear-gradient(90deg, var(--accent2), var(--accent)); }

        /* -- Task-type badges -- */
        .badge {
            display: inline-block;
            font-family: 'Space Mono', monospace;
            font-size: .65rem; padding: .2rem .6rem;
            border-radius: 4px; text-transform: uppercase; letter-spacing: .08em;
        }
        .badge-cls  { background: rgba(108,99,255,.2);  color: var(--accent2); border: 1px solid rgba(108,99,255,.3); }
        .badge-reg  { background: rgba(0,229,160,.15);  color: var(--accent);  border: 1px solid rgba(0,229,160,.25); }
        .badge-file { background: rgba(0,229,160,.12);  color: var(--accent);  border: 1px solid rgba(0,229,160,.2); }
        .badge-db   { background: rgba(108,99,255,.12); color: var(--accent2); border: 1px solid rgba(108,99,255,.2); }

        /* -- Override default blue Streamlit buttons -- */
        .stButton>button {
            background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
            color: #0d0f14 !important;
            border: none !important;
            border-radius: 8px !important;
            font-family: 'Space Mono', monospace !important;
            font-weight: 700 !important;
            font-size: .8rem !important;
            letter-spacing: .06em !important;
            padding: .6rem 1.4rem !important;
            transition: opacity .2s !important;
        }
        .stButton>button:hover { opacity: .85 !important; }

        /* -- Dark form inputs -- */
        .stSelectbox>div>div,
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input {
            background: var(--bg3) !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
            border-radius: 8px !important;
        }
        .stFileUploader {
            background: var(--bg3) !important;
            border: 2px dashed var(--border) !important;
            border-radius: 12px !important;
        }
        [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
        div[data-testid="stExpander"] {
            background: var(--bg2) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
        }

        /* -- Step progress indicator -- */
        .step-indicator { display: flex; gap: .5rem; align-items: center; margin-bottom: 2rem; }
        .step {
            font-family: 'Space Mono', monospace;
            font-size: .7rem; padding: .3rem .8rem;
            border-radius: 20px; border: 1px solid var(--border); color: var(--muted);
        }
        .step.active { background: rgba(0,229,160,.15); color: var(--accent); border-color: rgba(0,229,160,.4); }
        .step.done   { background: rgba(0,229,160,.08); color: var(--muted); border-color: var(--border); text-decoration: line-through; }
        .step-sep { color: var(--border); }

        /* -- Info pill tags -- */
        .info-tag {
            display: inline-flex; align-items: center; gap: .4rem;
            font-size: .8rem; color: var(--muted);
            background: var(--bg3); border: 1px solid var(--border);
            padding: .25rem .7rem; border-radius: 20px;
            margin-right: .4rem; margin-bottom: .4rem;
        }
        .dot   { width: 7px; height: 7px; border-radius: 50%; }
        .dot-g { background: var(--accent); }
        .dot-p { background: var(--accent2); }
        .dot-r { background: var(--accent3); }

        /* -- Cleaning panel -- */
        .clean-summary {
            background: rgba(0,229,160,.07);
            border: 1px solid rgba(0,229,160,.2);
            border-radius: 10px;
            padding: 1rem 1.25rem;
            margin-top: 1rem;
            font-size: .85rem;
        }
        .clean-log-item {
            display: flex; align-items: center; gap: .5rem;
            font-size: .8rem; color: var(--muted); padding: .2rem 0;
        }
        .clean-ok   { color: var(--accent)  !important; }
        .clean-excl { color: var(--accent3) !important; }

        /* -- Visualisation placeholder -- */
        .viz-placeholder {
            background: var(--bg3);
            border: 1px dashed var(--border);
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            color: var(--muted);
            font-size: .88rem;
            margin-top: .5rem;
        }

        /* -- Temp-artifact card -- */
        /* Highlighted card shown after cleaning showing where the data was saved */
        .artifact-card {
            background: linear-gradient(135deg, rgba(108,99,255,.08) 0%, rgba(0,229,160,.06) 100%);
            border: 1px solid rgba(108,99,255,.3);
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin-top: 1rem;
        }
        .artifact-title {
            font-family: 'Space Mono', monospace;
            font-size: .75rem;
            color: var(--accent2);
            text-transform: uppercase;
            letter-spacing: .12em;
            margin-bottom: .6rem;
        }
        .artifact-path {
            font-family: 'Space Mono', monospace;
            font-size: .82rem;
            color: var(--accent);
            background: var(--bg3);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: .4rem .8rem;
            display: inline-block;
            margin: .3rem 0;
            word-break: break-all;
        }
        .artifact-meta {
            font-size: .78rem;
            color: var(--muted);
            margin-top: .4rem;
        }
    </style>
    """, unsafe_allow_html=True)


# Step 3 - Session state initialization
for key, val in {
    "df":                 None,
    "df_clean":           None,
    "source_type":        "file",
    "conn_info":          {},
    "cleaning_decisions": {},
    "cleaning_applied":   False,
    "artifact_type":      None,
    "artifact_location":  None,
    "artifact_ts":        None,
    "target":             None,
    "task_type":          None,
    "compare_results":    None,
    "best_model":         None,
    "source_label":       "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# Step 4 - Helper: Reset downstream state
def _reset_downstream():
    """Clear all cleaning and training session state."""
    st.session_state.cleaning_applied   = False
    st.session_state.cleaning_decisions = {}
    st.session_state.df_clean           = None
    st.session_state.artifact_type      = None
    st.session_state.artifact_location  = None
    st.session_state.artifact_ts        = None
    st.session_state.target             = None
    st.session_state.compare_results    = None
    st.session_state.best_model         = None


# Step 5 - Helper: Persist cleaned artifact
def _persist_clean_artifact(clean_df: pd.DataFrame) -> tuple:
    """
    Save clean_df to the appropriate storage backend.

    Returns
    -------
    (artifact_type, artifact_location)
        artifact_type:     "csv" | "sql_table" | "mongo_collection" | "sqlite_table"
        artifact_location: file path, or table name, or collection name
    """
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.artifact_ts = ts
    src = st.session_state.source_type

    # -- File / Demo -------------------
    # Write a plain CSV to the system temp directory.
    # Users can download the file using the button rendered in the UI.
    if src in ("file", "demo"):
        _out_dir = os.path.join(os.path.expanduser("~"), "automl_studio_output")
        os.makedirs(_out_dir, exist_ok=True)
        path = os.path.join(_out_dir, f"automl_cleaned_{ts}.csv")
        clean_df.to_csv(path, index=False)
        return "csv", path

    # -- PostgreSQL / MySQL / MariaDB -----------------
    # Use the stored SQLAlchemy connection string to write the cleaned DataFrame
    # back as a new table. if_exists="replace" makes re-runs idempotent.
    # The table name uses a timestamp suffix to avoid collisions with existing
    # tables and to make it easy to identify and clean up later.
    if src in ("postgresql", "mysql"):
        from sqlalchemy import create_engine
        table_name = f"automl_cleaned_{ts}"
        conn_str   = st.session_state.conn_info.get("conn_str", "")
        eng        = create_engine(conn_str)
        clean_df.to_sql(table_name, eng, if_exists="replace", index=False)
        return "sql_table", table_name

    # -- SQLite --------------------
    # Connect to the same tempfile path that was written during upload, write
    # the cleaned table into it alongside the original data.
    if src == "sqlite":
        import sqlite3
        table_name = f"automl_cleaned_{ts}"
        db_path    = st.session_state.conn_info.get("db_path", "")
        conn       = sqlite3.connect(db_path)
        clean_df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
        return "sqlite_table", table_name

    # -- BigQuery --------------------
    # pandas-gbq's to_gbq() writes a DataFrame directly to BigQuery.
    # The destination table is project.dataset.automl_cleaned_<ts>.
    # "replace" table disposition drops and recreates on re-run.
    if src == "bigquery":
        import pandas_gbq
        from google.oauth2 import service_account
        info    = st.session_state.conn_info
        project = info.get("project", "")
        dataset = info.get("dataset", "automl_temp")
        table   = f"{dataset}.automl_cleaned_{ts}"
        creds   = service_account.Credentials.from_service_account_file(
            info.get("cred_path", "")
        )
        pandas_gbq.to_gbq(
            clean_df,
            destination_table = table,
            project_id        = project,
            credentials       = creds,
            if_exists         = "replace",
        )
        return "sql_table", f"{project}.{table}"

    # -- MongoDB ----------------
    # Convert the DataFrame to a list of dicts (documents) and insert into a
    # new collection. We use insert_many for efficiency on large datasets.
    if src == "mongodb":
        from pymongo import MongoClient
        info      = st.session_state.conn_info
        client    = MongoClient(info.get("uri", ""))
        coll_name = f"automl_cleaned_{ts}"
        db_handle = client[info.get("db", "")]
        records   = clean_df.to_dict(orient="records")
        db_handle[coll_name].insert_many(records)
        return "mongo_collection", coll_name

    # Fallback - should never reach here
    return "csv", "/tmp/automl_cleaned_fallback.csv"


# Step 6 - Hero banner
st.markdown("""
<div class="hero">
    <h1>⚗️ AutoML Studio</h1>
    <p>Connect · Clean &amp; persist · Compare every model · Ship the best one</p>
</div>
""", unsafe_allow_html=True)

# Step 7 - Step progress indicator
step = 1
if st.session_state.df is not None:                step = 2
if st.session_state.cleaning_applied:              step = 3
if st.session_state.target:                        step = 4
if st.session_state.compare_results is not None:   step = 5

def _step_pill(n, label):
    cls = "done" if n < step else ("active" if n == step else "step")
    return f'<span class="step {cls}">{label}</span>'

st.markdown(f"""
<div class="step-indicator">
    {_step_pill(1,"01 · Connect")}
    <span class="step-sep">›</span>
    {_step_pill(2,"02 · Clean")}
    <span class="step-sep">›</span>
    {_step_pill(3,"03 · Configure")}
    <span class="step-sep">›</span>
    {_step_pill(4,"04 · Train")}
    <span class="step-sep">›</span>
    {_step_pill(5,"05 · Results")}
</div>
""", unsafe_allow_html=True)


# Step 8 - Sidebar: Data source connectors
with st.sidebar:
    st.markdown('<div class="card-title">📡 Data Source</div>', unsafe_allow_html=True)

    source = st.selectbox("Connect via", [
        "📂 Upload CSV / Excel",
        "🐘 PostgreSQL",
        "🐬 MySQL",
        "🦭 MariaDB",
        "☁️ BigQuery",
        "🍃 MongoDB",
        "🗄️ SQLite",
    ], label_visibility="collapsed")

    st.divider()

    # -- CSV / Excel -----------------------
    # Streamlit's file_uploader returns a BytesIO-like object; pandas reads it
    # directly without a temp file. The artifact will be a /tmp CSV.
    if source == "📂 Upload CSV / Excel":
        uploaded = st.file_uploader("Drop file here", type=["csv", "xlsx", "xls"])
        sep = st.selectbox("CSV separator", [",", ";", "\t", "|"], index=0)
        if uploaded and st.button("Load File ▶"):
            try:
                df = (pd.read_csv(uploaded, sep=sep)
                      if uploaded.name.endswith(".csv")
                      else pd.read_excel(uploaded))
                st.session_state.df           = df
                st.session_state.source_type  = "file"
                st.session_state.conn_info    = {}
                st.session_state.source_label = f"📂 {uploaded.name}"
                _reset_downstream()
                st.success(f"Loaded {len(df):,} rows × {len(df.columns)} cols")
            except Exception as e:
                st.error(f"Parse error: {e}")

    # -- PostgreSQL -----------------------
    # Connection string: postgresql+psycopg2://user:password@host:port/dbname
    # conn_info stores the full string so _persist_clean_artifact() can
    # create_engine() to write the cleaned table back to the same DB.
    elif source == "🐘 PostgreSQL":
        host     = st.text_input("Host", "localhost")
        port     = st.number_input("Port", value=5432, step=1)
        dbname   = st.text_input("Database")
        user     = st.text_input("User")
        password = st.text_input("Password", type="password")
        query    = st.text_area("SQL Query", "SELECT * FROM your_table LIMIT 10000;")
        if st.button("Connect & Load ▶"):
            try:
                from sqlalchemy import create_engine
                conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
                eng = create_engine(conn_str)
                df  = pd.read_sql(query, eng)
                st.session_state.df           = df
                st.session_state.source_type  = "postgresql"
                st.session_state.conn_info    = {"conn_str": conn_str, "dbname": dbname}
                st.session_state.source_label = f"🐘 {dbname}"
                _reset_downstream()
                st.success(f"Loaded {len(df):,} rows")
            except Exception as e:
                st.error(str(e))

    # -- MySQL / MariaDB -----------------------
    # mysql-connector-python driver via SQLAlchemy.
    # Both MySQL and MariaDB share the same wire protocol so one block handles both.
    elif source in ("🐬 MySQL", "🦭 MariaDB"):
        host     = st.text_input("Host", "localhost")
        port     = st.number_input("Port", value=3306, step=1)
        dbname   = st.text_input("Database")
        user     = st.text_input("User")
        password = st.text_input("Password", type="password")
        query    = st.text_area("SQL Query", "SELECT * FROM your_table LIMIT 10000;")
        if st.button("Connect & Load ▶"):
            try:
                from sqlalchemy import create_engine
                conn_str = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{dbname}"
                eng = create_engine(conn_str)
                df  = pd.read_sql(query, eng)
                st.session_state.df           = df
                st.session_state.source_type  = "mysql"
                st.session_state.conn_info    = {"conn_str": conn_str, "dbname": dbname}
                st.session_state.source_label = f"🐬 {dbname}"
                _reset_downstream()
                st.success(f"Loaded {len(df):,} rows")
            except Exception as e:
                st.error(str(e))

    # -- BigQuery -----------------------
    # Service account JSON is written to a tempfile (google-auth needs a path).
    # conn_info stores the cred_path, project, and dataset for write-back.
    elif source == "☁️ BigQuery":
        project   = st.text_input("GCP Project ID")
        dataset   = st.text_input("Dataset (for temp table)", "automl_temp")
        cred_file = st.file_uploader("Service Account JSON", type=["json"])
        query     = st.text_area("SQL Query",
                                  "SELECT * FROM `your_project.your_dataset.your_table` LIMIT 10000")
        if st.button("Connect & Load ▶"):
            try:
                import json, tempfile as _tf
                from google.cloud import bigquery
                from google.oauth2 import service_account
                creds_json = json.load(cred_file)
                tmp = _tf.NamedTemporaryFile("w", suffix=".json", delete=False)
                json.dump(creds_json, tmp); tmp.close()
                cred_path = tmp.name
                creds  = service_account.Credentials.from_service_account_file(cred_path)
                client = bigquery.Client(project=project, credentials=creds)
                df = client.query(query).to_dataframe()
                st.session_state.df           = df
                st.session_state.source_type  = "bigquery"
                st.session_state.conn_info    = {
                    "project":   project,
                    "dataset":   dataset,
                    "cred_path": cred_path,
                }
                st.session_state.source_label = f"☁️ {project}"
                _reset_downstream()
                st.success(f"Loaded {len(df):,} rows")
            except Exception as e:
                st.error(str(e))

    # -- MongoDB -----------------------
    # Fetches documents, flattens to DataFrame, drops ObjectId "_id" field.
    # conn_info stores uri + db name so _persist_clean_artifact() can
    # connect and insert_many() into a new collection.
    elif source == "🍃 MongoDB":
        uri        = st.text_input("Connection URI", "mongodb://localhost:27017")
        dbname     = st.text_input("Database")
        collection = st.text_input("Collection")
        limit      = st.number_input("Row limit", value=10000, step=1000)
        query_str  = st.text_input("Filter (JSON)", "{}")
        if st.button("Connect & Load ▶"):
            try:
                from pymongo import MongoClient
                import json
                filt   = json.loads(query_str)
                client = MongoClient(uri)
                cursor = client[dbname][collection].find(filt).limit(int(limit))
                df = pd.DataFrame(list(cursor))
                if "_id" in df.columns:
                    df.drop(columns=["_id"], inplace=True)
                st.session_state.df           = df
                st.session_state.source_type  = "mongodb"
                st.session_state.conn_info    = {"uri": uri, "db": dbname}
                st.session_state.source_label = f"🍃 {dbname}.{collection}"
                _reset_downstream()
                st.success(f"Loaded {len(df):,} rows")
            except Exception as e:
                st.error(str(e))

    # -- SQLite -----------------------
    # The uploaded .db file is written to a named tempfile so sqlite3 (which
    # needs a filesystem path) can open it. The same tempfile path is stored
    # in conn_info so _persist_clean_artifact() can write the cleaned table
    # back into the exact same file.
    elif source == "🗄️ SQLite":
        db_file = st.file_uploader(
            "Upload .db / .sqlite file", type=["db", "sqlite", "sqlite3"]
        )
        query = st.text_area("SQL Query", "SELECT * FROM your_table LIMIT 10000;")
        if db_file and st.button("Connect & Load ▶"):
            try:
                import sqlite3
                tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
                tmp.write(db_file.read()); tmp.close()
                db_path = tmp.name
                conn    = sqlite3.connect(db_path)
                df      = pd.read_sql(query, conn); conn.close()
                st.session_state.df           = df
                st.session_state.source_type  = "sqlite"
                st.session_state.conn_info    = {"db_path": db_path}
                st.session_state.source_label = f"🗄️ {db_file.name}"
                _reset_downstream()
                st.success(f"Loaded {len(df):,} rows")
            except Exception as e:
                st.error(str(e))

    # -- Demo datasets -----------------------
    # Titanic: real public CSV - binary classification (Survived 0/1).
    # House Prices: synthetically generated regression dataset.
    # Both use source_type="demo" which routes to the /tmp CSV artifact path.
    st.divider()
    st.markdown('<div class="card-title">🧪 Quick Demo</div>', unsafe_allow_html=True)
    demo_ds = st.selectbox(
        "Dataset",
        ["Titanic (Classification)", "House Prices (Regression)"],
    )
    if st.button("Load Demo ▶"):
        if "Titanic" in demo_ds:
            url = ("https://raw.githubusercontent.com/"
                   "datasciencedojo/datasets/master/titanic.csv")
            df  = pd.read_csv(url)
            st.session_state.source_label = "🧪 Titanic Demo"
        else:
            np.random.seed(42)
            n = 500
            df = pd.DataFrame({
                "sqft":         np.random.randint(600, 4000, n),
                "bedrooms":     np.random.randint(1, 6, n),
                "bathrooms":    np.random.randint(1, 4, n),
                "age":          np.random.randint(0, 60, n),
                "garage":       np.random.randint(0, 3, n),
                "neighborhood": np.random.choice(["A", "B", "C", "D"], n),
            })
            df["price"] = (
                df["sqft"]     * 120
              + df["bedrooms"] * 8000
              - df["age"]      * 500
              + df["garage"]   * 12000
              + np.random.normal(0, 20000, n)
            ).astype(int)
            st.session_state.source_label = "🧪 House Prices Demo"

        st.session_state.df          = df
        st.session_state.source_type = "demo"
        st.session_state.conn_info   = {}
        _reset_downstream()
        st.rerun()


# Step 9 - Landing state (no data loaded)
df = st.session_state.df

if df is None:
    col1, col2, col3, col4 = st.columns(4)
    for col, icon, title, body in [
        (col1, "🔌", "Connect",
         "Upload CSV/Excel or plug into Postgres, MySQL, MariaDB, BigQuery, MongoDB, SQLite."),
        (col2, "🧹", "Clean & Persist",
         "Fix nulls per column. Cleaned data is saved to a temp CSV file or a DB temp table."),
        (col3, "⚙️", "Configure",
         "Pick your target column. AutoML detects classification vs regression automatically."),
        (col4, "🏆", "Compare",
         "PyCaret benchmarks 15–25 algorithms with k-fold CV and ranks them by the best metric."),
    ]:
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center; min-height:160px;">
                <div style="font-size:2rem; margin-bottom:.5rem;">{icon}</div>
                <div style="font-family:'Space Mono', monospace; font-size:.85rem;
                            color:var(--accent); margin-bottom:.5rem">{title}</div>
                <div style="font-size:.85rem; color:var(--muted); line-height:1.6">{body}</div>
            </div>
            """, unsafe_allow_html=True)
    st.stop()


# Step 10 - Data preview
total_nulls = df.isnull().sum().sum()

st.markdown(f"""
<div class="card">
    <div class="card-title">📊 Dataset — {st.session_state.source_label}</div>
    <div>
        <span class="info-tag"><span class="dot dot-g"></span> {len(df):,} rows</span>
        <span class="info-tag"><span class="dot dot-p"></span> {len(df.columns)} columns</span>
        <span class="info-tag"><span class="dot dot-r"></span> {total_nulls:,} null values</span>
        <span class="info-tag"><span class="dot dot-p"></span>source: {st.session_state.source_type}</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab_prev, tab_info, tab_stats = st.tabs(["📋 Preview", "ℹ️ Column Info", "📈 Quick Stats"])

with tab_prev:
    st.dataframe(df.head(200), use_container_width=True, height=280)

with tab_info:
    info_df = pd.DataFrame({
        "dtype":    df.dtypes.astype(str),
        "non-null": df.notnull().sum(),
        "nulls":    df.isnull().sum(),
        "null %":   (df.isnull().mean() * 100).round(1),
        "unique":   df.nunique(),
    })
    st.dataframe(info_df, use_container_width=True)

with tab_stats:
    st.dataframe(df.describe(include="all").T.round(3), use_container_width=True)

st.divider()


# Step 11 - DATA CLEANING + TEMP ARTIFACT PERSISTENCE
#
# Purpose
# ---------
# Null values in training data cause two problems:
#   1. Many sklearn estimators raise ValueError on NaN inputs
#   2. Silent imputation by PyCaret hides data quality issues from the user
# This section surfaces every null column, lets the user choose a strategy,
# then saves the result to a durable artifact before any training begins.
#
# Cleaning strategies
# --------------------
# Numeric columns (int / float):
#   • Fill with mean    - arithmetic average; sensitive to outliers
#   • Fill with median  - middle value; robust to heavy tails / outliers
#   • Fill with mode    - most frequent value; good for discrete integers
#   • Exclude column    - drop the whole column from training data
#   • Drop rows         - remove every row with a null in this column
#
# Text / categorical columns (object / bool / category):
#   • Fill with "N/A"   - literal string; creates explicit "unknown" category
#   • Fill with mode    - most frequent text value
#   • Exclude column    - drop the whole column from training data
#   • Drop rows         - remove every row with a null in this column
#
# Temp artifact storage
# -----------------------
# Source type      Artifact type        Location
# --------------------------------------------------
# file / demo      CSV file             /tmp/automl_cleaned_<ts>.csv
#                                       (downloadable from UI)
# postgresql       SQL table            automl_cleaned_<ts>  in same database
# mysql / mariadb  SQL table            automl_cleaned_<ts>  in same database
# sqlite           SQLite table         automl_cleaned_<ts>  in same .db file
# bigquery         BigQuery table       <project>.<dataset>.automl_cleaned_<ts>
# mongodb          Collection           automl_cleaned_<ts>  in same database
#
# PyCaret's setup() is called with df_clean — the cleaned in-memory DataFrame
# (equivalent content to the persisted artifact). The artifact's role is to
# be a verifiable, shareable snapshot of exactly what the model was trained on.

null_cols = df.columns[df.isnull().any()].tolist()

st.markdown(
    '<div class="card-title" style="font-size:.9rem;">🧹 Data Cleaning</div>',
    unsafe_allow_html=True
)

if not null_cols:
    # -- No nulls -> auto-advance (no artifact needed, just use raw df) -----------
    st.markdown("""
    <div class="clean-summary">
        ✅ <strong>No missing values found.</strong> Dataset is complete - no cleaning needed.
    </div>
    """, unsafe_allow_html=True)
    if not st.session_state.cleaning_applied:
        st.session_state.df_clean           = df.copy()
        st.session_state.cleaning_applied   = True
        st.session_state.cleaning_decisions = {}
        # Still persist a clean artifact so the provenance record exists
        try:
            a_type, a_loc = _persist_clean_artifact(df.copy())
            st.session_state.artifact_type     = a_type
            st.session_state.artifact_location = a_loc
        except Exception:
            pass  # artifact persistence is non-blocking

else:
    # -- Null summary pills -----------
    affected_pct = len(null_cols) / len(df.columns) * 100
    st.markdown(f"""
    <div style="display:flex; gap:1.5rem; margin-bottom:1.2rem; flex-wrap:wrap;">
        <span class="info-tag"><span class="dot dot-r"></span>{len(null_cols)} column(s) with nulls</span>
        <span class="info-tag"><span class="dot dot-p"></span>{total_nulls:,} total missing cells</span>
        <span class="info-tag"><span class="dot dot-g"></span>{affected_pct:.1f}% of columns affected</span>
    </div>
    """, unsafe_allow_html=True)

    # -- Per-column decision table -----------
    # Carry forward previous decisions from session_state so choices survive
    # re-renders triggered by any widget interaction on this page.
    decisions = st.session_state.cleaning_decisions.copy()

    # Column header labels
    hc1, hc2, hc3, hc4, hc5 = st.columns([2.2, 1, 1.2, 1.5, 3])
    for hcol, lbl in zip([hc1, hc2, hc3, hc4, hc5],
                         ["Column", "Type", "Nulls", "% Missing", "Action"]):
        hcol.markdown(
            f'<div style="font-size:.7rem;color:var(--muted);font-family:Space Mono,'
            f'monospace;text-transform:uppercase;letter-spacing:.1em;">{lbl}</div>',
            unsafe_allow_html=True
        )

    for col_name in null_cols:
        null_count = int(df[col_name].isnull().sum())
        null_pct   = df[col_name].isnull().mean() * 100
        dtype_str  = str(df[col_name].dtype)
        is_numeric = pd.api.types.is_numeric_dtype(df[col_name])

        # Build option list and sensible default per dtype
        if is_numeric:
            options = [
                "Fill with mean",    # best default for normally distributed cols
                "Fill with median",  # better when outliers are present
                "Fill with mode",    # best for discrete / integer-valued cols
                "Exclude column",    # drop entire column from training
                "Drop rows",         # remove rows with a null in this column
            ]
            default = decisions.get(col_name, "Fill with mean")
        else:
            options = [
                "Fill with N/A",     # explicit "unknown" category
                "Fill with mode",    # most frequent text value
                "Exclude column",    # drop entire column from training
                "Drop rows",         # remove rows with a null in this column
            ]
            default = decisions.get(col_name, "Fill with N/A")

        # Guard against stale session state after dtype changes between datasets
        if default not in options:
            default = options[0]

        c1, c2, c3, c4, c5 = st.columns([2.2, 1, 1.2, 1.5, 3])

        with c1:
            # title= HTML attribute provides a tooltip for truncated long names
            st.markdown(
                f'<div style="font-family:Space Mono,monospace;font-size:.78rem;'
                f'color:var(--text);padding-top:.45rem;overflow:hidden;'
                f'text-overflow:ellipsis;white-space:nowrap;" title="{col_name}">'
                f'{col_name}</div>', unsafe_allow_html=True
            )
        with c2:
            badge_color = "var(--accent3)" if not is_numeric else "var(--accent)"
            st.markdown(
                f'<div style="font-size:.68rem;color:{badge_color};background:var(--bg2);'
                f'border:1px solid var(--border);padding:.15rem .45rem;border-radius:4px;'
                f'margin-top:.35rem;display:inline-block;">{dtype_str}</div>',
                unsafe_allow_html=True
            )
        with c3:
            st.markdown(
                f'<div style="font-family:Space Mono,monospace;font-size:.8rem;'
                f'color:var(--accent3);padding-top:.45rem;">{null_count:,}</div>',
                unsafe_allow_html=True
            )
        with c4:
            # Severity colour: orange < 20%, purple 20-50%, red > 50%
            bar_w     = min(int(null_pct), 100)
            bar_color = (
                "var(--accent3)" if null_pct > 50
                else "var(--accent2)" if null_pct > 20
                else "#ff9f43"
            )
            st.markdown(
                f'<div style="padding-top:.5rem;">'
                f'<div style="height:5px;width:100%;background:var(--border);border-radius:3px;'
                f'margin-bottom:.2rem;">'
                f'<div style="height:100%;width:{bar_w}%;background:{bar_color};'
                f'border-radius:3px;"></div></div>'
                f'<div style="font-size:.68rem;color:var(--muted);">{null_pct:.1f}%</div>'
                f'</div>', unsafe_allow_html=True
            )
        with c5:
            # Unique key per column so Streamlit tracks each widget independently
            chosen = st.selectbox(
                label            = f"action_{col_name}",
                options          = options,
                index            = options.index(default),
                key              = f"clean_{col_name}",
                label_visibility = "collapsed",
            )
            decisions[col_name] = chosen

    # Persist updated decisions immediately so re-runs keep all choices
    st.session_state.cleaning_decisions = decisions

    # -- Cleaning plan preview ----------------------------
    with st.expander("👁️  Preview cleaning plan"):
        excluded  = [c for c, a in decisions.items() if a == "Exclude column"]
        drop_rows = [c for c, a in decisions.items() if a == "Drop rows"]
        filled    = {c: a for c, a in decisions.items()
                     if a not in ("Exclude column", "Drop rows")}

        if excluded:
            st.markdown(
                f"**Columns to be dropped:** `{'`, `'.join(excluded)}`  \n"
                f"*These columns will be removed entirely before training.*"
            )
        if drop_rows:
            st.markdown(
                f"**Rows dropped for nulls in:** `{'`, `'.join(drop_rows)}`  \n"
                f"*Any row with a null in these columns will be excluded.*"
            )
        if filled:
            st.markdown(
                "**Fill strategy per column:**\n"
                + "\n".join(f"- `{c}` → {a}" for c, a in filled.items())
            )

        # Explain the artifact that will be created
        src = st.session_state.source_type
        if src in ("file", "demo"):
            st.info("💾 A **CSV file** will be saved to `/tmp/` after cleaning.")
        elif src == "bigquery":
            proj = st.session_state.conn_info.get("project", "your-project")
            ds   = st.session_state.conn_info.get("dataset", "automl_temp")
            st.info(f"💾 A **BigQuery table** will be created in `{proj}.{ds}`.")
        elif src == "mongodb":
            dbn = st.session_state.conn_info.get("db", "your_db")
            st.info(f"💾 A **MongoDB collection** will be created in `{dbn}`.")
        elif src == "sqlite":
            st.info("💾 A **SQLite table** will be written into the same `.db` file.")
        else:
            dbn = st.session_state.conn_info.get("dbname", "your_db")
            st.info(f"💾 A **temporary table** will be created in `{dbn}`.")

    # -- Action buttons ----------------------------
    col_btn1, col_btn2, _ = st.columns([1.8, 1.8, 4])
    with col_btn1:
        apply_clicked = st.button("✅  Apply Cleaning & Save Artifact",
                                  use_container_width=True)
    with col_btn2:
        if st.session_state.cleaning_applied:
            if st.button("↩️  Reset Cleaning", use_container_width=True):
                st.session_state.cleaning_applied   = False
                st.session_state.df_clean           = None
                st.session_state.cleaning_decisions = {}
                st.session_state.artifact_type      = None
                st.session_state.artifact_location  = None
                st.session_state.artifact_ts        = None
                st.session_state.target             = None
                st.session_state.compare_results    = None
                st.session_state.best_model         = None
                st.rerun()

    # -- Core cleaning logic ----------------------------
    if apply_clicked:
        working   = df.copy()  # always work on a copy - never mutate raw df
        log_lines = []

        for col_name, action in decisions.items():

            if action == "Exclude column":
                # Drop entire column - it won't appear in training at all
                if col_name in working.columns:
                    working.drop(columns=[col_name], inplace=True)
                    log_lines.append(("excl", col_name, "column excluded from training"))

            elif action == "Drop rows":
                # Remove every row that has a null in this specific column.
                # Applied sequentially: if multiple columns use "Drop rows",
                # the union of all affected rows is removed across passes.
                before = len(working)
                working.dropna(subset=[col_name], inplace=True)
                after  = len(working)
                log_lines.append(("excl", col_name, f"{before - after:,} rows removed"))

            elif action == "Fill with N/A":
                # Replace nulls with the literal string "N/A".
                # Tree-based models encode this as a distinct "missing" bucket.
                working[col_name] = working[col_name].fillna("N/A")
                log_lines.append(("ok", col_name, 'filled -> "N/A"'))

            elif action == "Fill with mean":
                # Arithmetic mean - best for normally distributed numeric cols.
                # Sensitive to outliers; use median if the distribution is skewed.
                mean_val = working[col_name].mean()
                working[col_name] = working[col_name].fillna(round(mean_val, 4))
                log_lines.append(("ok", col_name, f"filled -> mean ({mean_val:.4f})"))

            elif action == "Fill with median":
                # Median = middle value after sorting.
                # Robust to extreme values; preferred when outliers are likely.
                median_val = working[col_name].median()
                working[col_name] = working[col_name].fillna(round(median_val, 4))
                log_lines.append(("ok", col_name, f"filled -> median ({median_val:.4f})"))

            elif action == "Fill with mode":
                # Mode = most frequent value. Works for both text and discrete numeric.
                # mode() returns a Series in case of ties; we take the first value.
                mode_series = working[col_name].mode()
                if not mode_series.empty:
                    mode_val = mode_series.iloc[0]
                    working[col_name] = working[col_name].fillna(mode_val)
                    log_lines.append(("ok", col_name, f"filled -> mode ({mode_val})"))

        # -- Persist the cleaned artifact ------------------
        # This is the key step that differentiates this feature from a plain
        # in-memory clean. The artifact gives the user (and auditors) a durable,
        # inspectable record of exactly what data the model was trained on.
        artifact_error = None
        try:
            a_type, a_loc = _persist_clean_artifact(working)
            st.session_state.artifact_type     = a_type
            st.session_state.artifact_location = a_loc
        except Exception as e:
            artifact_error = str(e)

        # Store cleaned DataFrame for downstream steps (configure, train)
        st.session_state.df_clean         = working
        st.session_state.cleaning_applied = True
        # Invalidate any previous training results - they were on different data
        st.session_state.target          = None
        st.session_state.compare_results = None
        st.session_state.best_model      = None

        # Build the styled log banner
        log_html = "".join(
            f'<div class="clean-log-item">'
            f'<span class="{"clean-ok" if t == "ok" else "clean-excl"}">'
            f'{"✔" if t == "ok" else "✖"}</span>'
            f'<code style="color:var(--accent2);font-size:.78rem;">{c}</code>'
            f'<span>{msg}</span></div>'
            for t, c, msg in log_lines
        )
        remaining = working.isnull().sum().sum()
        st.markdown(f"""
        <div class="clean-summary">
            <strong>Cleaning applied - {len(log_lines)} column(s) processed</strong>
            <div style="margin-top:.6rem;">{log_html}</div>
            <div style="margin-top:.8rem;font-size:.8rem;color:var(--muted);">
                Shape after cleaning:
                <span style="color:var(--accent);">{len(working):,} rows × {len(working.columns)} columns</span>
                · remaining nulls:
                <span style="color:{'var(--accent3)' if remaining else 'var(--accent)'};">{remaining:,}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if artifact_error:
            st.warning(f"⚠️ Artifact could not be saved: {artifact_error}")

        st.rerun()

    # -- Show artifact card once cleaning is applied -----------
    if st.session_state.cleaning_applied and st.session_state.df_clean is not None:
        clean_df  = st.session_state.df_clean
        remaining = clean_df.isnull().sum().sum()
        a_type    = st.session_state.artifact_type
        a_loc     = st.session_state.artifact_location
        src       = st.session_state.source_type

        # Status summary
        st.markdown(f"""
        <div class="clean-summary">
          ✅ <strong>Cleaning applied.</strong>
          Dataset: <span style="color:var(--accent)">{len(clean_df):,} rows × {len(clean_df.columns)} cols</span>
          · remaining nulls: <span style="color:{'var(--accent3)' if remaining else 'var(--accent)'};">{remaining:,}</span>.
          Use <em>Reset Cleaning</em> above to change decisions.
        </div>
        """, unsafe_allow_html=True)

        # -- Artifact card ----------------
        # Renders different content based on whether the artifact is a file or DB object
        if a_loc:
            if a_type == "csv":
                # File artifact - show path and download button
                st.markdown(f"""
                <div class="artifact-card">
                  <div class="artifact-title">💾 Cleaned Data Artifact — Temporary CSV File</div>
                  <div class="artifact-path">{a_loc}</div>
                  <div class="artifact-meta">
                    File written to the system temp directory.
                    Download it below to keep a permanent copy.
                    This file and <code>df_clean</code> are identical — PyCaret trains on both.
                  </div>
                </div>
                """, unsafe_allow_html=True)
                # Stream the cleaned CSV from the temp file for download
                try:
                    with open(a_loc, "rb") as fh:
                        st.download_button(
                            "⬇️  Download Cleaned CSV",
                            data      = fh,
                            file_name = Path(a_loc).name,
                            mime      = "text/csv",
                        )
                except FileNotFoundError:
                    st.warning("Temp file not found - it may have been cleared. Re-apply cleaning.")

            elif a_type in ("sql_table", "sqlite_table"):
                # Database table artifact
                db_label = {
                    "postgresql": "PostgreSQL",
                    "mysql":      "MySQL / MariaDB",
                    "sqlite":     "SQLite",
                    "bigquery":   "BigQuery",
                }.get(src, "Database")
                dbname = st.session_state.conn_info.get(
                    "dbname", st.session_state.conn_info.get("db", "")
                )
                st.markdown(f"""
                <div class="artifact-card">
                  <div class="artifact-title">💾 Cleaned Data Artifact — {db_label} Temp Table</div>
                  <div class="artifact-path">{a_loc}</div>
                  <div class="artifact-meta">
                    Table created in <strong>{dbname or src}</strong>.
                    Query it directly: <code>SELECT * FROM {a_loc} LIMIT 10;</code><br>
                    PyCaret trains on the in-memory <code>df_clean</code> (identical content).
                    Drop this table manually when you no longer need it.
                  </div>
                </div>
                """, unsafe_allow_html=True)
                # Offer a "drop table" button so users can clean up easily
                if st.button(f"🗑️  Drop temp table `{a_loc}`"):
                    try:
                        if src in ("postgresql", "mysql"):
                            from sqlalchemy import create_engine, text
                            eng = create_engine(
                                st.session_state.conn_info.get("conn_str", "")
                            )
                            with eng.connect() as con:
                                con.execute(text(f"DROP TABLE IF EXISTS {a_loc}"))
                                con.commit()
                        elif src == "sqlite":
                            import sqlite3
                            conn = sqlite3.connect(
                                st.session_state.conn_info.get("db_path", "")
                            )
                            conn.execute(f"DROP TABLE IF EXISTS {a_loc}")
                            conn.commit(); conn.close()
                        st.session_state.artifact_location = None
                        st.success(f"Table `{a_loc}` dropped.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not drop table: {e}")

            elif a_type == "mongo_collection":
                # MongoDB collection artifact
                dbn = st.session_state.conn_info.get("db", "")
                st.markdown(f"""
                <div class="artifact-card">
                  <div class="artifact-title">💾 Cleaned Data Artifact — MongoDB Collection</div>
                  <div class="artifact-path">{dbn}.{a_loc}</div>
                  <div class="artifact-meta">
                    Collection created in database <strong>{dbn}</strong>.
                    PyCaret trains on the in-memory <code>df_clean</code> (identical content).
                    Drop the collection when no longer needed.
                  </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"🗑️  Drop collection `{a_loc}`"):
                    try:
                        from pymongo import MongoClient
                        client = MongoClient(
                            st.session_state.conn_info.get("uri", "")
                        )
                        client[dbn].drop_collection(a_loc)
                        st.session_state.artifact_location = None
                        st.success(f"Collection `{a_loc}` dropped.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not drop collection: {e}")


# GATE - require cleaning before configure / train
# st.stop() halts all further execution so configure/train/results only render
# once cleaning is explicitly approved. This enforces step order and ensures
# PyCaret always trains on the user-verified, artifact-matched dataset.
if not st.session_state.cleaning_applied:
    st.stop()

# From this point the cleaned DataFrame is the single source of truth
df = st.session_state.df_clean

st.divider()


# Step 12 - Configure
# Four controls in one row plus an expandable advanced-options panel.
# Auto-detect heuristic for task type:
#   object dtype OR <= 20 unique values -> Classification
#   otherwise                           -> Regression
col_a, col_b, col_c, col_d = st.columns([2, 1.5, 1.5, 1.5])

with col_a:
    target = st.selectbox("🎯 Target Column", ["— select —"] + list(df.columns))

with col_b:
    if target != "— select —":
        nuniq     = df[target].nunique()
        auto_task = "Classification" if (
            df[target].dtype == object or nuniq <= 20
        ) else "Regression"
    else:
        auto_task = "Classification"

    task_type = st.selectbox(
        "Task Type", ["Auto-detect", "Classification", "Regression"], index=0
    )
    if task_type == "Auto-detect" and target != "— select —":
        task_type = auto_task
    

with col_c:
    n_models = st.selectbox("Models to compare", ["All", "Top 10", "Top 5"], index=0)

with col_d:
    train_size = st.slider("Train split %", 60, 90, 80)
    
if task_type == "Classification" and target != "- select -":
    n_cls = df[target].nunique()
    test_rows = int(len(df) * (1 - train_size / 100))
    if n_cls > test_rows:
        st.warning(
            f"⚠️ Target has {n_cls} unique classes but test split only has "
            f"{test_rows} rows. Lower Train split % or pick a different target column."
        )

# -- Advanced PyCaret options -------------------
with st.expander("⚙️  Advanced PyCaret Options"):
    col_e, col_f, col_g = st.columns(3)

    with col_e:
        normalize = st.checkbox("Normalize features", value=True)
        # Z-score normalisation - essential for KNN, SVM, linear models.
        # Tree-based models (RF, XGBoost) are scale-invariant; it doesn't hurt them.
     
    with col_f:
        fold_n = st.selectbox("CV folds", [3, 5, 10], index=1)
        # Stratified k-fold cross-validation. More folds -> more reliable
        # estimates but proportionally longer training time. 5 is the standard.

    with col_g:
        n_jobs = st.number_input("n_jobs (parallelism)", value=-1, step=1)
        # -1 = use all available CPU cores.
        # Set to 1 for easier debugging or on machines with limited RAM.


# Step 13 - Train
# Clicking "Run AutoML" triggers the full PyCaret pipeline:
#   setup()          - preprocess df_clean: encode, impute any residual nulls,
#                      normalise, split into train/test
#   compare_models() - train every algorithm with k-fold CV; return best model
#   pull()           - retrieve the leaderboard DataFrame PyCaret printed
#   save_model()     - serialise model + preprocessing pipeline to .pkl
#                      (pipeline means raw data can be passed at inference time)
#
# PyCaret is imported LAZILY (inside the handler, not at the top of the file)
# to keep startup time fast - it imports 50+ packages on load.
st.divider()

if target == "— select —":
    st.info("👆 Select a target column above to continue.")
    st.stop()

st.session_state.target    = target
st.session_state.task_type = task_type

if st.button("🚀  Run AutoML — Compare All Models"):
    progress_area = st.empty()
    log_area      = st.empty()

    def _log(msg):
        """Render a styled live log line."""
        log_area.markdown(
            f'<div style="background:var(--bg3);border:1px solid var(--border);'
            f'border-radius:8px;padding:.6rem 1rem;font-family:Space Mono,monospace;'
            f'font-size:.75rem;color:var(--accent);">⚡ {msg}</div>',
            unsafe_allow_html=True
        )
        time.sleep(0.1)

    try:
        _log("Importing PyCaret ...")

        # Lazy import dispatched by task type
        if task_type == "Classification":
            from pycaret.classification import (
                setup, compare_models, pull, save_model
            )
        else:
            from pycaret.regression import (
                setup, compare_models, pull, save_model
            )

        _log("Setting up PyCaret environment ...")
        with progress_area:
            with st.spinner("PyCaret setup in progress ..."):
                if task_type == "Classification":
                    class_counts = df[target].value_counts()
                    valid_classes = class_counts[class_counts >= 2].index
                    removed = len(df) - len(df[df[target].isin(valid_classes)])
                    df = df[df[target].isin(valid_classes)].reset_index(drop=True)
                    if removed > 0:
                        st.warning(
                            f"⚠️ Removed {removed} row(s) whose target class had fewer than "
                            f"2 samples - required for stratified splitting."
                        )
                    
                    n_classes = df[target].nunique()
                    test_rows = int(len(df) * (1 - train_size / 100))
                    if test_rows < n_classes:
                        st.error(
                            f"❌ Cannot train: your dataset has **{n_classes} classes** but the "
                            f"test split only contains **{test_rows} rows** — each class needs "
                            f"at least one row in the test set.\n\n"
                            f"**Options:**\n"
                            f"- Lower the Train split % (currently {train_size}%) so the test set is larger\n"
                            f"- Check you selected the right target column — a column with {n_classes} "
                            f"unique values is likely an ID or free-text field, not a class label\n"
                            f"- Use a larger dataset"
                        )
                        st.stop()
                    
                setup(
                    data              = df,
                    target            = target,
                    train_size        = train_size / 100,
                    normalize         = normalize,
                    fold              = fold_n,
                    n_jobs            = n_jobs,
                    verbose           = False,
                    session_id        = 42,
                )

        _log("Comparing models - this may take several minutes ...")

        # Build optional include list for Top 5 / Top 10 modes
        include_list = None
        if n_models == "Top 5":
            include_list = ["lr", "dt", "rf", "lightgbm", "gbc"]
        elif n_models == "Top 10":
            include_list = (
                ["lr","dt","rf","et","lightgbm","gbc","ada","knn","nb","lda"]
                if task_type == "Classification"
                else ["lr","lasso","ridge","dt","rf","et","lightgbm","knn","gbr","br"]
            )

        with progress_area:
            with st.spinner("Training & comparing all models ..."):
                kwargs = dict(verbose=False, errors="ignore")
                # errors="ignore" skips any algorithm that fails (missing dep,
                # convergence failure, etc.) without crashing the whole run
                if include_list:
                    kwargs["include"] = include_list
                best = compare_models(**kwargs)

        results_df = pull()
        st.session_state.compare_results = results_df
        st.session_state.best_model      = best

        # save_model serialises model + pipeline into a single .pkl
        _model_dir = os.path.join(os.path.expanduser("~"), "automl_studio_output")
        os.makedirs(_model_dir, exist_ok=True)
        save_model(best, os.path.join(_model_dir, "best_automl_model"))

        _log("✅ Done! Best model saved.")
        progress_area.empty()
        log_area.empty()
        st.rerun()

    except Exception as e:
        progress_area.empty()
        log_area.error(f"Error: {e}\n\n{traceback.format_exc()}")


# Step 14 - Results Dashboard
# Renders when compare_results is populated:
#   task badge -> KPI metric tiles -> ranked leaderboard -> full comparison table
#   -> download buttons (results CSV + model .pkl + cleaned CSV) -> code snippet
if st.session_state.compare_results is not None:
    results = st.session_state.compare_results
    a_type  = st.session_state.artifact_type
    a_loc   = st.session_state.artifact_location

    st.divider()
    st.markdown(
        '<div class="card-title" style="font-size:.9rem;color:var(--accent);">'
        '🏆 Model Comparison Results</div>',
        unsafe_allow_html=True
    )

    # Task-type badge
    task_badge = (
        f'<span class="badge badge-cls">{task_type}</span>'
        if task_type == "Classification"
        else f'<span class="badge badge-reg">{task_type}</span>'
    )
    st.markdown(task_badge, unsafe_allow_html=True)

    # -- KPI tiles ------------------------
    if results.empty:
        st.error("❌ No models were successfully trained. This usually means:\n\n"
            "- All algorithms failed silently (try switching from 'All' to 'Top 5' models)\n"
            "- The dataset is too small for the number of CV folds "
            f"(currently {st.session_state.get('fold_n', 5)} folds — try 3)\n"
            "- The target column has too few samples per class\n\n"
            "Reset and try again with different settings."
        )
        st.stop()
    
    best_row         = results.iloc[0]
    primary_metric   = "Accuracy" if task_type == "Classification" else "R2"
    secondary_metric = "AUC"      if task_type == "Classification" else "RMSE"
    prim_val = best_row.get(primary_metric,   best_row.iloc[1] if len(best_row) > 1 else 0)
    sec_val  = best_row.get(secondary_metric, best_row.iloc[2] if len(best_row) > 2 else 0)
    tt_val   = best_row.get("TT (Sec)", 0)

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric">
        <div class="metric-val">{prim_val:.4f}</div>
        <div class="metric-lbl">Best {primary_metric}</div>
      </div>
      <div class="metric">
        <div class="metric-val">{sec_val:.4f}</div>
        <div class="metric-lbl">Best {secondary_metric}</div>
      </div>
      <div class="metric">
        <div class="metric-val">{len(results)}</div>
        <div class="metric-lbl">Models Tested</div>
      </div>
      <div class="metric">
        <div class="metric-val">{tt_val:.2f}s</div>
        <div class="metric-lbl">Best Model TT</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # -- Ranked leaderboard ------------------------
    score_col = primary_metric if primary_metric in results.columns else results.columns[1]
    max_score = results[score_col].max() if score_col in results.columns else 1
    min_score = results[score_col].min() if score_col in results.columns else 0

    html_rows = ""
    for i, (_, row) in enumerate(results.iterrows()):
        model_name = row.get("Model", row.name) if "Model" in results.columns else str(row.name)
        score      = row.get(score_col, 0)
        bar_pct    = int(((score - min_score) / max(max_score - min_score, 1e-9)) * 100)
        crown      = "👑 " if i == 0 else ""
        html_rows += (
            f'<div class="model-row">'
            f'<div class="model-rank">#{i+1}</div>'
            f'<div class="model-name">{crown}{model_name}</div>'
            f'<div class="model-bar-wrap"><div class="model-bar" style="width:{bar_pct}%"></div></div>'
            f'<div class="model-score">{score:.4f}</div>'
            f'</div>'
        )
    st.markdown(html_rows, unsafe_allow_html=True)

    # -- Full comparison table -----------------------------------------
    with st.expander("📊 Full Comparison Table"):
        st.dataframe(
            results.style.background_gradient(
                subset=[score_col] if score_col in results.columns else [],
                cmap="Greens"
            ),
            use_container_width=True
        )

    # -- Model Visualisations ------------------------------------------
    # PyCaret's plot_model() generates matplotlib figures for the best model.
    # All plots are rendered lazily - only when the user clicks a button -
    # because each plot re-runs inference on the hold-out test set (2-5 sec).
    #
    # plot_model() saves the figure to a temp PNG file by default. We capture
    # it as a matplotlib Figure object instead (save=False) so st.pyplot()
    # can render it inline without touching the filesystem.
    #
    # Classification plots available here:
    #   confusion_matrix - true/false positive matrix on the test set
    #   auc              - ROC curve with AUC score per class
    #   feature          - feature importance (top 10 columns by weight)
    #   class_report     - precision / recall / F1 per class as a heatmap
    #
    # Regression plots available here:
    #   residuals        - residuals vs fitted values; should be random scatter
    #   error            - actual vs predicted values; should hug the diagonal
    #   feature          - feature importance (top 10 columns by weight)
    #   learning         - train vs test score across increasing training sizes

    st.divider()
    st.markdown(
        '<div class="card-title" style="font-size:.9rem;">📈 Model Visualisations</div>',
        unsafe_allow_html=True
    )
    st.caption(
        "Each plot re-evaluates the model on the hold-out test set. "
        "Click a button to generate it — results appear below."
    )

    # Define which plots are available per task type
    # Each entry: (button_label, pycaret_plot_id, description_shown_on_hover)
    if task_type == "Classification":
        plot_options = [
            ("🔲 Confusion Matrix",     "confusion_matrix",
             "True vs predicted labels — reveals exactly which classes get confused"),
            ("📉 ROC / AUC Curve",      "auc",
             "ROC curve with AUC score; higher = better class separation"),
            ("🏅 Feature Importance",   "feature",
             "Top 10 features by importance weight used by the model"),
            ("📋 Classification Report","class_report",
             "Precision, Recall, F1-score per class as a heatmap"),
        ]
    else:  # Regression
        plot_options = [
            ("〰️ Residuals Plot",     "residuals",
             "Residuals vs fitted values — random scatter = good fit"),
            ("🎯 Predicted vs Actual", "error",
             "Predicted vs actual values — dots should hug the diagonal"),
            ("🏅 Feature Importance",  "feature",
             "Top 10 features by importance weight used by the model"),
            ("📚 Learning Curve",      "learning",
             "Train vs test score across increasing training-set sizes"),
        ]

    # Render buttons in a row.
    # We keep track of which plot the user selected via session_state so the
    # selection survives the page re-run triggered by clicking the button.
    if "active_plot" not in st.session_state:
        st.session_state.active_plot = None

    btn_cols = st.columns(len(plot_options))
    for col, (label, plot_id, _) in zip(btn_cols, plot_options):
        with col:
            if st.button(label, use_container_width=True, key=f"plot_{plot_id}"):
                # Toggle: clicking the active plot again hides it
                st.session_state.active_plot = (
                    None if st.session_state.active_plot == plot_id else plot_id
                )

    # Find the currently selected plot metadata (if any)
    active      = st.session_state.active_plot
    active_meta = next(
        ((lbl, pid, desc) for lbl, pid, desc in plot_options if pid == active),
        None
    )

    if active_meta:
        plot_label, plot_id, plot_desc = active_meta

        # Description caption beneath the buttons
        st.caption(f"**{plot_label.split(' ', 1)[1]}** - {plot_desc}")

        # Spinner wraps the plot_model() call which re-runs inference internally
        with st.spinner(f"Generating {plot_label} ..."):
            try:
                # Lazy import - module is already in sys.modules from training
                if task_type == "Classification":
                    from pycaret.classification import plot_model
                else:
                    from pycaret.regression import plot_model
                
                import matplotlib.pyplot as plt
                
                _out_dir = os.path.join(os.path.expanduser("~"), "automl_studio_output")
                os.makedirs(_out_dir, exist_ok=True)
                               
                _before = set(os.listdir(_out_dir))
                
                _orig_cwd = os.getcwd()
                os.chdir(_out_dir)
                try:
                   plot_model(
                       st.session_state.best_model,
                       plot = plot_id,
                       save = True,
                   )
                finally:
                   # Always restore orginal cwd even if plot_model raieses
                   os.chdir(_orig_cwd)
                   
                _after = set(os.listdir(_out_dir))
                _new = {f for f in (_after - _before) if f.endswith(".png")}
                
                if _new:
                    _dest = os.path.join(_out_dir, _new.pop())
                    st.image(_dest, use_container_width=True)
                    with open(_dest, "rb") as fh:
                        st.download_button(
                            f"⬇️  Download {plot_label.split(' ', 1)[1]} (PNG)",
                            data      = fh,
                            file_name = f"automl_{plot_id}.png",
                            mime      = "image/png",
                        )
                
                else:
                    fig = plt.gcf()
                    if fig is not None and len(fig.get_axes()) > 0:
                        _dest = os.path.join(_out_dir, f"{plot_id}.png")
                        fig.savefig(_dest, dpi=150, bbox_inches="tight")
                        plt.close("all")
                        st.image(_dest, use_container_width=True)
                        with open(_dest, "rb") as fh:
                            st.download_button(
                                f"⬇️  Download {plot_label.split(' ', 1)[1]} (PNG)",
                                data = f,
                                file_name = f"automl_{plot_id}.png",
                                mine = "image/png",
                            )
                    else:
                        plt.close("all")
                        st.warning(
                            f"Plot `{plot_id}` could not be generated. "
                            "This model type may not support this visualisation - try a different one."
                        )
                        
            except Exception as e:
                err = str(e)
                if "feature_importances_" in err or "coef_" in err:
                    st.warning(
                        "**Feature Importance** is not available for this model type = it requires "
                        "a tree-based or liner estimator. Try the confusion Matrix or AUC plot instead."
                    )
                else:
                    st.error(
                        f"Could not generate `{plot_id}` plot: {e}\n\n"
                        "This usually means the model type doesn't support this visualisation "
                        "or a required optional dependency is missing."
                    )
    else:
        # Placeholder shown before any button is clicked
        st.markdown("""
        <div style="background:var(--bg3);border:1px dashed var(--border);
                    border-radius:10px;padding:2rem;text-align:center;
                    color:var(--muted);font-size:.88rem;margin-top:.5rem;">
          Click a button above to generate a plot
        </div>
        """, unsafe_allow_html=True)

    # -- Download buttons ---------------------------------------------
    st.divider()
    col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)

    with col_dl1:
        # Leaderboard results as CSV
        csv_buf = io.StringIO()
        results.to_csv(csv_buf)
        st.download_button(
            "⬇️  Results CSV",
            data                = csv_buf.getvalue(),
            file_name           = "automl_comparison.csv",
            mime                = "text/csv",
            use_container_width = True,
        )

    with col_dl2:
        # Best model .pkl (PyCaret save_model appends .pkl automatically)
        _model_dir = os.path.join(os.path.expanduser("~"), "automl_studio_output")
        model_path = Path(os.path.join(_model_dir, "best_automl_model.pkl"))
        if model_path.exists():
            with open(model_path, "rb") as fh:
                st.download_button(
                    "⬇️  Best Model (.pkl)",
                    data                = fh,
                    file_name           = "best_automl_model.pkl",
                    mime                = "application/octet-stream",
                    use_container_width = True,
                )

    with col_dl3:
        # Cleaned data CSV (only for file/demo sources; DB sources have a table)
        if a_type == "csv" and a_loc:
            try:
                with open(a_loc, "rb") as fh:
                    st.download_button(
                        "⬇️  Cleaned Data CSV",
                        data                = fh,
                        file_name           = "automl_cleaned_data.csv",
                        mime                = "text/csv",
                        use_container_width = True,
                    )
            except FileNotFoundError:
                st.caption("Cleaned CSV temp file not found.")
        elif a_type in ("sql_table", "sqlite_table", "mongo_collection"):
            st.markdown(
                f'<div style="font-size:.75rem;color:var(--muted);padding-top:.5rem;">'
                f'Cleaned data → <code style="color:var(--accent2);">{a_loc}</code></div>',
                unsafe_allow_html=True
            )

    with col_dl4:
        if st.button("🔄  Reset & Start Over", use_container_width=True):
            for k in ["df", "df_clean", "target", "task_type", "compare_results",
                      "best_model", "source_label", "conn_info",
                      "artifact_type", "artifact_location", "artifact_ts"]:
                st.session_state[k] = None if k not in ("source_label", "conn_info") else (
                    "" if k == "source_label" else {}
                )
            st.session_state["cleaning_applied"]   = False
            st.session_state["cleaning_decisions"] = {}
            st.session_state["source_type"]        = "file"
            st.rerun()

    # -- Inference code snippet -----------------------------------------------
    best_name = (results.iloc[0].get("Model", "Best Model")
                 if "Model" in results.columns else "Best Model")
    task_mod          = "classification" if task_type == "Classification" else "regression"
    score_col_display = score_col
    artifact_comment  = (
        f"# Trained on: {a_loc}" if a_loc
        else "# Trained on: in-memory cleaned DataFrame"
    )

    with st.expander("💻 Code — Load & Use Your Model"):
        st.code(f"""
from pycaret.{task_mod} import load_model, predict_model
import pandas as pd

# Load model — includes the full preprocessing pipeline so you can pass
# raw data directly without any manual feature engineering
model = load_model("best_automl_model")   # path to your downloaded .pkl

# Run predictions on new raw data
new_data    = pd.read_csv("your_new_data.csv")
predictions = predict_model(model, data=new_data)
print(predictions.head())

# ── Training provenance ────────────────────────────────
# Best model : {best_name}
# {score_col_display:12s} : {prim_val:.4f}
{artifact_comment}
""", language="python")