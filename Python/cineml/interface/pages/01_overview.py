"""
Page 1 -- Overview
System health, dataset stats, live pipeline controls, Docker Compose panel.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _bootstrap import ROOT   # sets up sys.path; ROOT = cineml/

import pandas as pd
import plotly.express as px
import streamlit as st

# Direct imports -- no 'interface.components.' prefix
from data_loader import data_status, load_events, load_movies, load_ratings
from pipeline_runner import run_script, render_run_history
from docker_manager import render_docker_panel

st.title("CineML Platform")
st.caption("End-to-end ML research & experimentation system for streaming content intelligence")
st.divider()

# ── Module status ──────────────────────────────────────────────────────────────
st.subheader("Module status")
status = data_status()

STEPS = [
    ("M1", "Data pipeline",   ["ratings", "movies", "events"]),
    ("M2", "Recommender",     ["als", "two_tower"]),
    ("M3", "A/B engine",      ["events"]),
    ("M4", "Analysis memo",   ["ratings", "events"]),
    ("M5", "Diffusion + ViT", ["vit", "ddpm"]),
]
cols = st.columns(5)
for col, (badge, label, deps) in zip(cols, STEPS):
    ready = all(status.get(d, False) for d in deps)
    col.metric(label=f"{badge}  {label}", value="✅ Ready" if ready else "⚠️ Pending",
               help=f"Requires: {', '.join(deps)}")
if st.button("🔄 Refresh status"):
    st.cache_data.clear()
    st.rerun()

st.divider()

# ── Dataset overview ───────────────────────────────────────────────────────────
st.subheader("Dataset overview")
col_l, col_r = st.columns(2)

with col_l:
    ratings = load_ratings()
    if ratings is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ratings",    f"{len(ratings)/1e6:.1f}M")
        m2.metric("Users",      f"{ratings['userId'].nunique():,}")
        m3.metric("Movies",     f"{ratings['movieId'].nunique():,}")
        m4.metric("Avg rating", f"{ratings['rating'].mean():.2f}")
        fig = px.histogram(
            ratings.sample(min(50_000, len(ratings)), random_state=42),
            x="rating", nbins=9, title="Rating distribution (sample)",
            color_discrete_sequence=["#6366f1"],
        )
        fig.update_layout(showlegend=False, height=220, margin=dict(t=36, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run **Fetch MovieLens** below to populate this section.")

with col_r:
    events = load_events()
    n_clicks = (events["event_type"] == "click").sum()
    n_imps   = (events["event_type"] == "impression").sum()
    ecols = st.columns(4)
    ecols[0].metric("Events", f"{len(events):,}")
    ecols[1].metric("Users",  f"{events['user_id'].nunique():,}")
    ecols[2].metric("CTR",    f"{n_clicks / max(n_imps, 1):.3f}")
    ecols[3].metric("Arms",   "50 / 50")
    breakdown = events.groupby(["arm", "event_type"]).size().reset_index(name="count")
    fig2 = px.bar(breakdown, x="event_type", y="count", color="arm", barmode="group",
                  title="Events by type & arm",
                  color_discrete_map={"control": "#94a3b8", "treatment": "#6366f1"}, height=220)
    fig2.update_layout(margin=dict(t=36, b=0, l=0, r=0), legend=dict(title=""))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Pipeline controls ──────────────────────────────────────────────────────────
st.subheader("Pipeline controls")
st.caption("Click Run to execute a script. Live output streams below each button.")

with st.expander("📦  M1 — Data Curation Pipeline", expanded=not status["ratings"]):
    c1, c2 = st.columns(2)
    tmdb_limit = c1.number_input("TMDB movie limit", 100, 50_000, 5_000, 500, key="tmdb_limit")
    n_users    = c2.number_input("Simulator users",  1_000, 100_000, 10_000, 1_000, key="sim_n")
    events_pu  = c2.number_input("Events per user",  10, 200, 50, 10, key="sim_epu")
    st.markdown("---")
    run_script("fetch_movielens", "Fetch MovieLens 25M",
               "module1_data_pipeline/fetch_movielens.py",
               description="Downloads ~250 MB, cleans ratings + movies Parquet files",
               warn_long=True)
    run_script("fetch_tmdb", "Fetch TMDB metadata & posters",
               "module1_data_pipeline/fetch_tmdb.py",
               args=["--limit", str(tmdb_limit)],
               description=f"Metadata + posters for {tmdb_limit:,} movies via TMDB API")
    run_script("simulate_events", "Simulate streaming events",
               "module1_data_pipeline/event_simulator.py",
               args=["--n-users", str(n_users), "--events-per-user", str(events_pu)],
               description=f"{n_users:,} users x {events_pu} events, A/B arm split")
    run_script("load_bq", "Load to BigQuery",
               "module1_data_pipeline/bigquery_loader.py", args=["--all"],
               description="Uploads Parquet files to BQ (needs GCP credentials in .env)")

with st.expander("🎯  M2 — Personalisation Engine", expanded=not status["als"]):
    c1, c2 = st.columns(2)
    tt_epochs = c1.number_input("Two-Tower epochs", 5, 100, 20, 5, key="tt_ep")
    c2.number_input("ALS iterations", 10, 100, 30, 10, key="als_it")
    st.markdown("---")
    run_script("train_als", "Train ALS (matrix factorisation)",
               "module2_recommender/train.py", args=["--model", "als"],
               description="Fast CPU baseline ~2 min")
    run_script("train_tt", "Train Two-Tower neural model",
               "module2_recommender/train.py",
               args=["--model", "two-tower", "--epochs", str(tt_epochs)],
               description="GPU recommended. BPR loss, cosine similarity scoring",
               warn_long=True)

with st.expander("📊  M3 — A/B & Causal Engine"):
    st.info("M3 reads the event logs from M1 -- no training needed. "
            "Open the A/B Engine page to run the live analysis.")

with st.expander("🔬  M4 — Analysis Memo"):
    run_script("export_memo", "Export analysis memo",
               "module4_analysis_memo/analysis_memo.py",
               description="Runs the full DS investigation and prints findings")

with st.expander("🎨  M5 — Diffusion + ViT", expanded=not status["ddpm"]):
    c1, c2, c3 = st.columns(3)
    ddpm_ds    = c1.selectbox("Dataset", ["mnist", "tmdb"], key="ddpm_ds")
    ddpm_ep    = c2.number_input("Epochs", 5, 500, 20, 5, key="ddpm_ep")
    ddpm_size  = c3.number_input("Image size", 32, 128, 32, 32, key="ddpm_sz")
    vit_ep     = c1.number_input("ViT epochs", 3, 50, 10, 1, key="vit_ep")
    st.markdown("---")
    run_script("train_ddpm", f"Train DDPM ({ddpm_ds.upper()})",
               "module5_diffusion_vit/diffusion/train.py",
               args=["--dataset", ddpm_ds, "--epochs", str(ddpm_ep),
                     "--image-size", str(ddpm_size)],
               description="MNIST: CPU ~20 min. TMDB: GPU recommended.", warn_long=True)
    run_script("finetune_vit", "Fine-tune ViT genre classifier",
               "module5_diffusion_vit/vit/finetune.py",
               args=["--data-dir", "data/raw/tmdb/posters",
                     "--epochs", str(vit_ep), "--output-dir", "data/models/vit_genre"],
               description=f"Linear probe -> full ViT-B/16. {vit_ep} epochs.", warn_long=True)

st.divider()

# ── Docker panel ───────────────────────────────────────────────────────────────
render_docker_panel()

st.divider()

# ── Run history ────────────────────────────────────────────────────────────────
with st.expander("📋  Session run history"):
    render_run_history()

st.divider()
st.subheader("Data flow")
st.code("""
MovieLens 25M  --+
TMDB API       --+--> M1 Data Pipeline --> Parquet / BigQuery
Event Simulator--+         |
                           v
                    M2 Personalisation Engine  (ALS + Two-Tower)
                           | ranked recommendations
                           v
                    M3 A/B & Causal Engine  <-- treatment arm recs
                    (Frequentist / Bayesian / DiD / PSM)
                           | results
                           v
                    M4 Analysis Memo  (OLS + cohort + findings)

                    M5 Diffusion + ViT  <-- TMDB poster images
""", language="text")
