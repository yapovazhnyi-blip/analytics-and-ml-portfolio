"""
Module 3 — Streamlit A/B Testing Dashboard.

Run: streamlit run module3_ab_engine/dashboard.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))

from module3_ab_engine.frequentist import run_frequentist_analysis, required_sample_size
from module3_ab_engine.bayesian import run_bayesian_analysis

st.set_page_config(
    page_title="CineML A/B Engine",
    page_icon="🎬",
    layout="wide",
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🎬 CineML — A/B & Causal Inference Dashboard")
st.markdown(
    "**Control**: popularity-based ranking &nbsp;|&nbsp; "
    "**Treatment**: Two-Tower neural recommender"
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")
alpha = st.sidebar.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)
n_sample = st.sidebar.number_input("Sample n (for simulation)", 1000, 100_000, 10_000, 1000)
mde = st.sidebar.slider("MDE (absolute)", 0.001, 0.05, 0.01, 0.001, format="%.3f")

# ── Data loading / simulation ──────────────────────────────────────────────────
@st.cache_data
def load_or_simulate_events(n: int) -> pd.DataFrame:
    events_path = Path("data/processed/streaming_events.parquet")
    if events_path.exists():
        df = pd.read_parquet(events_path)
        return df.sample(min(n, len(df)), random_state=42)

    # Fallback: quick simulation for demo
    rng = np.random.default_rng(42)
    n_users = n
    records = []
    for uid in range(n_users):
        arm = "treatment" if uid % 2 == 0 else "control"
        ctr = rng.beta(3, 17) * (1.15 if arm == "treatment" else 1.0)
        clicked = rng.random() < ctr
        records.append({
            "user_id": uid, "movie_id": rng.integers(1, 5000),
            "event_type": "impression", "arm": arm,
            "dwell_ms": int(rng.lognormal(7, 1)),
            "timestamp": pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(rng.integers(0, 365))),
        })
        if clicked:
            records.append({**records[-1], "event_type": "click",
                            "dwell_ms": int(rng.lognormal(8, 1.5))})
    return pd.DataFrame(records)


events = load_or_simulate_events(n_sample)

# ── KPI Cards ──────────────────────────────────────────────────────────────────
imps = events[events["event_type"] == "impression"]
clicks = events[events["event_type"] == "click"]

ctrl_users = set(imps[imps["arm"] == "control"]["user_id"])
trt_users = set(imps[imps["arm"] == "treatment"]["user_id"])
ctrl_ctr = clicks[clicks["user_id"].isin(ctrl_users)]["user_id"].nunique() / len(ctrl_users)
trt_ctr = clicks[clicks["user_id"].isin(trt_users)]["user_id"].nunique() / len(trt_users)
lift = (trt_ctr - ctrl_ctr) / ctrl_ctr * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Control CTR", f"{ctrl_ctr:.3f}", help="Control arm click-through rate")
col2.metric("Treatment CTR", f"{trt_ctr:.3f}", delta=f"{trt_ctr - ctrl_ctr:+.3f}")
col3.metric("Relative Lift", f"{lift:+.1f}%")
col4.metric("Total Users", f"{imps['user_id'].nunique():,}")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_freq, tab_bayes, tab_power, tab_did = st.tabs(
    ["📊 Frequentist", "🎲 Bayesian", "⚡ Sample Size", "📐 DiD / Causal"]
)

# ── Frequentist tab ────────────────────────────────────────────────────────────
with tab_freq:
    st.subheader("Frequentist Test Results")
    results = run_frequentist_analysis(events, alpha=alpha)
    ctr_r = results["ctr"]

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"""
| Metric | Value |
|--------|-------|
| Control CTR | `{ctr_r.control_mean:.4f}` |
| Treatment CTR | `{ctr_r.treatment_mean:.4f}` |
| Absolute lift | `{ctr_r.absolute_lift:+.4f}` |
| Relative lift | `{ctr_r.relative_lift_pct:+.1f}%` |
| p-value | `{ctr_r.p_value:.4f}` |
| 95% CI | `[{ctr_r.ci_lower:.4f}, {ctr_r.ci_upper:.4f}]` |
| Significant? | {'✅ Yes' if ctr_r.significant else '❌ No'} |
""")
    with col_r:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Control", x=["CTR"], y=[ctr_r.control_mean],
                             error_y=dict(type="data", array=[ctr_r.ci_upper - ctr_r.absolute_lift])))
        fig.add_trace(go.Bar(name="Treatment", x=["CTR"], y=[ctr_r.treatment_mean]))
        fig.update_layout(barmode="group", title="CTR by Arm", height=350)
        st.plotly_chart(fig, use_container_width=True)

# ── Bayesian tab ───────────────────────────────────────────────────────────────
with tab_bayes:
    st.subheader("Bayesian Beta-Binomial Analysis")
    with st.spinner("Sampling posteriors …"):
        bayes = run_bayesian_analysis(events)
    ctr_b = bayes["ctr"]

    st.metric(
        "P(Treatment > Control)",
        f"{ctr_b.prob_treatment_better:.1%}",
        help="Posterior probability that treatment CTR exceeds control CTR"
    )

    # Posterior plot
    diff = ctr_b.posterior_samples_treatment - ctr_b.posterior_samples_control
    fig = px.histogram(
        diff, nbins=100,
        labels={"value": "CTR difference (treatment − control)"},
        title="Posterior distribution of CTR lift",
        color_discrete_sequence=["#6366f1"],
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="no effect")
    fig.add_vline(x=float(np.percentile(diff, 2.5)), line_dash="dot", line_color="gray")
    fig.add_vline(x=float(np.percentile(diff, 97.5)), line_dash="dot", line_color="gray",
                  annotation_text="95% CI")
    st.plotly_chart(fig, use_container_width=True)

# ── Sample Size / Power tab ────────────────────────────────────────────────────
with tab_power:
    st.subheader("Sample Size & Power Calculator")
    _baseline_default = float(min(max(round(ctrl_ctr, 3), 0.01), 0.99))
    baseline = st.number_input("Baseline CTR", 0.01, 0.99, _baseline_default, 0.01, format="%.3f")
    power_target = st.slider("Target power (1-β)", 0.60, 0.99, 0.80, 0.01)

    mde_range = np.linspace(0.001, 0.05, 50)
    n_required = [required_sample_size(baseline, m, alpha, power_target)["n_per_arm"]
                  for m in mde_range]

    fig = px.line(x=mde_range, y=n_required,
                  labels={"x": "Minimum Detectable Effect (absolute)", "y": "n per arm"},
                  title="Required sample size vs MDE")
    fig.add_hline(y=n_sample // 2, line_dash="dash", annotation_text="Current n/arm")
    st.plotly_chart(fig, use_container_width=True)

    info = required_sample_size(baseline, mde, alpha, power_target)
    st.info(f"For MDE={mde:.3f}: need **{info['n_per_arm']:,}** per arm "
            f"(**{info['n_total']:,}** total)")

# ── DiD tab ────────────────────────────────────────────────────────────────────
with tab_did:
    st.subheader("Difference-in-Differences")
    st.markdown("""
    DiD controls for pre-existing trends between arms.
    Here we use the synthetic event log split at the experiment mid-point.

    **Model**: `CTR = β₀ + β₁·treated + β₂·post + β₃·(treated×post) + ε`

    The coefficient **β₃** is the DiD estimate — the causal effect of treatment
    *after* controlling for time trends and baseline group differences.
    """)

    try:
        from module3_ab_engine.causal import build_did_panel, difference_in_differences
        mid_date = str(events["timestamp"].median().date())
        panel = build_did_panel(events, mid_date)
        did_result = difference_in_differences(panel)

        cols = st.columns(3)
        cols[0].metric("DiD Estimate", f"{did_result.did_estimate:+.4f}")
        cols[1].metric("p-value", f"{did_result.p_value:.4f}")
        cols[2].metric("Significant?", "✅ Yes" if did_result.p_value < alpha else "❌ No")

        fig = go.Figure()
        periods = ["Pre", "Post"]
        for arm, (pre, post) in [
            ("Control", (did_result.pre_control_mean, did_result.post_control_mean)),
            ("Treatment", (did_result.pre_treatment_mean, did_result.post_treatment_mean)),
        ]:
            fig.add_trace(go.Scatter(x=periods, y=[pre, post], mode="lines+markers", name=arm))
        fig.update_layout(title="Parallel trends plot", yaxis_title="CTR", height=350)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"DiD requires streaming_events.parquet. Error: {e}")
