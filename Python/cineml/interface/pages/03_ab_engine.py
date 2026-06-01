"""
Page 3 — A/B & Causal Inference Engine (M3)
Full statistical analysis: frequentist, Bayesian, causal.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _bootstrap import ROOT, COMPONENTS  # noqa: E402

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import load_events

st.title("A/B & Causal Inference Engine")
st.caption("Module 3 — Frequentist · Bayesian · DiD · Propensity Score Matching")
st.divider()

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading event data…"):
    events = load_events()

# ── Config sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**A/B Config**")
    alpha       = st.slider("Significance level α", 0.01, 0.10, 0.05, 0.01, key="ab_alpha")
    sample_n    = st.slider("Event sample size", 5_000, min(50_000, len(events)),
                             min(20_000, len(events)), 5_000, key="ab_n")
    mde_input   = st.slider("MDE (absolute)", 0.005, 0.05, 0.01, 0.005,
                             format="%.3f", key="ab_mde")

events_sample = events.sample(sample_n, random_state=42)
imps = events_sample[events_sample["event_type"] == "impression"]
clicks = events_sample[events_sample["event_type"] == "click"]

ctrl_users = set(imps[imps["arm"] == "control"]["user_id"])
trt_users  = set(imps[imps["arm"] == "treatment"]["user_id"])
ctrl_ctr   = clicks[clicks["user_id"].isin(ctrl_users)]["user_id"].nunique() / max(len(ctrl_users), 1)
trt_ctr    = clicks[clicks["user_id"].isin(trt_users)]["user_id"].nunique()  / max(len(trt_users), 1)
lift_abs   = trt_ctr - ctrl_ctr
lift_rel   = lift_abs / ctrl_ctr * 100 if ctrl_ctr > 0 else 0.0

# ── KPI row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Control CTR",   f"{ctrl_ctr:.4f}")
k2.metric("Treatment CTR", f"{trt_ctr:.4f}", delta=f"{lift_abs:+.4f}")
k3.metric("Relative lift", f"{lift_rel:+.1f}%")
k4.metric("Control n",     f"{len(ctrl_users):,}")
k5.metric("Treatment n",   f"{len(trt_users):,}")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_freq, tab_bayes, tab_causal, tab_power = st.tabs(
    ["📊 Frequentist", "🎲 Bayesian", "📐 Causal (DiD + PSM)", "⚡ Sample Size"]
)

# ── Frequentist ────────────────────────────────────────────────────────────────
with tab_freq:
    try:
        from module3_ab_engine.frequentist import (
            ABGroups, proportion_test, means_test, run_frequentist_analysis
        )

        results = run_frequentist_analysis(events_sample, alpha=alpha)
        ctr_r   = results["ctr"]
        dwell_r = results["dwell"]

        st.subheader("Click-through rate (two-proportion z-test)")
        col_t, col_v = st.columns([1, 1])
        with col_t:
            sig_color = "🟢" if ctr_r.significant else "🔴"
            st.markdown(f"""
| Statistic | Value |
|-----------|-------|
| Control CTR | `{ctr_r.control_mean:.4f}` |
| Treatment CTR | `{ctr_r.treatment_mean:.4f}` |
| Absolute lift | `{ctr_r.absolute_lift:+.4f}` |
| Relative lift | `{ctr_r.relative_lift_pct:+.1f}%` |
| z-statistic | `{ctr_r.test_statistic:.3f}` |
| p-value | `{ctr_r.p_value:.4f}` |
| 95% CI | `[{ctr_r.ci_lower:+.4f}, {ctr_r.ci_upper:+.4f}]` |
| Significant? | {sig_color} `{'Yes' if ctr_r.significant else 'No'}` at α={alpha} |
""")
        with col_v:
            fig = go.Figure()
            # CI plot
            for arm, mean, ci_lo, ci_hi, color in [
                ("Control", ctr_r.control_mean,
                 ctr_r.control_mean - 1.96 * (ctr_r.control_mean * (1 - ctr_r.control_mean) / max(ctr_r.n_control, 1)) ** 0.5,
                 ctr_r.control_mean + 1.96 * (ctr_r.control_mean * (1 - ctr_r.control_mean) / max(ctr_r.n_control, 1)) ** 0.5,
                 "#94a3b8"),
                ("Treatment", ctr_r.treatment_mean,
                 ctr_r.treatment_mean - 1.96 * (ctr_r.treatment_mean * (1 - ctr_r.treatment_mean) / max(ctr_r.n_treatment, 1)) ** 0.5,
                 ctr_r.treatment_mean + 1.96 * (ctr_r.treatment_mean * (1 - ctr_r.treatment_mean) / max(ctr_r.n_treatment, 1)) ** 0.5,
                 "#6366f1"),
            ]:
                fig.add_trace(go.Scatter(
                    x=[arm, arm], y=[ci_lo, ci_hi],
                    mode="lines", line=dict(color=color, width=2),
                    showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    x=[arm], y=[mean], mode="markers",
                    marker=dict(size=12, color=color),
                    name=arm,
                ))
            fig.update_layout(title="CTR with 95% CI", height=280,
                               yaxis_title="CTR", margin=dict(t=40,b=0,l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Average dwell time (Welch's t-test)")
        st.markdown(f"""
Dwell lift: `{dwell_r.absolute_lift:+.1f}s` &nbsp;|&nbsp;
p-value: `{dwell_r.p_value:.4f}` &nbsp;|&nbsp;
Significant: `{'Yes ✅' if dwell_r.significant else 'No ❌'}`
""")
    except ImportError as e:
        st.error(f"Could not import frequentist module: {e}")


# ── Bayesian ───────────────────────────────────────────────────────────────────
with tab_bayes:
    try:
        from module3_ab_engine.bayesian import BetaBinomialTest, run_bayesian_analysis

        with st.spinner("Sampling posteriors (100k draws)…"):
            bayes = run_bayesian_analysis(events_sample)
        ctr_b = bayes["ctr"]

        c1, c2, c3 = st.columns(3)
        c1.metric("P(treatment > control)", f"{ctr_b.prob_treatment_better:.1%}")
        c2.metric("Expected lift", f"{ctr_b.expected_lift:+.4f}")
        c3.metric("95% credible interval",
                   f"[{ctr_b.credible_interval_lower:+.4f}, {ctr_b.credible_interval_upper:+.4f}]")

        # Posterior distribution
        diff = ctr_b.posterior_samples_treatment - ctr_b.posterior_samples_control
        fig = px.histogram(diff, nbins=120,
                            labels={"value": "CTR difference (treatment − control)"},
                            title="Posterior distribution of CTR lift",
                            color_discrete_sequence=["#6366f1"], height=300)
        fig.add_vline(x=0, line_dash="dash", line_color="red",
                       annotation_text="null", annotation_position="top right")
        fig.add_vline(x=float(np.percentile(diff, 2.5)), line_dash="dot",
                       line_color="gray", annotation_text="2.5%")
        fig.add_vline(x=float(np.percentile(diff, 97.5)), line_dash="dot",
                       line_color="gray", annotation_text="97.5%")
        fig.update_layout(showlegend=False, margin=dict(t=40,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"**Interpretation**: There is a {ctr_b.prob_treatment_better:.1%} probability "
            f"that the Two-Tower recommender improves CTR over the popularity baseline. "
            f"Under a 95% credible interval the true lift is between "
            f"{ctr_b.credible_interval_lower:+.3f} and {ctr_b.credible_interval_upper:+.3f}."
        )
    except ImportError as e:
        st.error(f"Could not import Bayesian module: {e}")


# ── Causal ─────────────────────────────────────────────────────────────────────
with tab_causal:
    try:
        from module3_ab_engine.causal import (
            build_did_panel, difference_in_differences, PropensityScoreMatching
        )

        st.subheader("Difference-in-Differences")
        st.markdown(
            "Controls for pre-existing engagement trends between arms. "
            "The DiD estimate (β₃) isolates the causal effect of the Two-Tower recommender."
        )

        mid_date = str(events_sample["timestamp"].median().date())
        panel = build_did_panel(events_sample, mid_date)

        with st.spinner("Fitting DiD OLS model…"):
            did = difference_in_differences(panel)

        d1, d2, d3 = st.columns(3)
        d1.metric("DiD estimate (β₃)", f"{did.did_estimate:+.4f}")
        d2.metric("HC3 std error",      f"{did.std_error:.4f}")
        d3.metric("p-value",
                   f"{did.p_value:.4f}",
                   delta="significant ✅" if did.p_value < alpha else "not significant ❌",
                   delta_color="normal" if did.p_value < alpha else "inverse")

        # Parallel trends plot
        fig = go.Figure()
        for arm, (pre, post), color in [
            ("Control",   (did.pre_control_mean,   did.post_control_mean),   "#94a3b8"),
            ("Treatment", (did.pre_treatment_mean, did.post_treatment_mean), "#6366f1"),
        ]:
            fig.add_trace(go.Scatter(
                x=["Pre-experiment", "Post-experiment"],
                y=[pre, post], mode="lines+markers", name=arm,
                line=dict(color=color, width=2),
                marker=dict(size=9),
            ))
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray",
                       annotation_text="experiment start")
        fig.update_layout(title="Parallel trends plot", height=320,
                           yaxis_title="CTR", margin=dict(t=40,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "**Parallel trends assumption**: if the pre-period slopes were equal, "
            "any post-period divergence is attributable to the treatment."
        )

        st.divider()
        st.subheader("Propensity Score Matching")
        st.markdown(
            "Corrects for selection bias — power users may be more likely to be "
            "in the treatment arm. PSM matches each treated user to the most "
            "similar control user by estimated P(T=1|X)."
        )

        if st.button("Run PSM (may take ~10s)"):
            user_level = (
                events_sample.groupby("user_id")
                .agg(
                    n_impressions=("event_id", "count"),
                    arm=("arm", "first"),
                )
                .reset_index()
            )
            user_level["n_clicks"] = user_level["user_id"].map(
                clicks.groupby("user_id").size()
            ).fillna(0)
            user_level["ctr"] = (user_level["n_clicks"] / user_level["n_impressions"].clip(lower=1))
            user_level["treated"] = (user_level["arm"] == "treatment").astype(int)

            psm = PropensityScoreMatching(caliper=0.1)
            with st.spinner("Running PSM…"):
                psm_result = psm.fit_transform(
                    user_level,
                    treatment_col="treated",
                    covariate_cols=["n_impressions"],
                    outcome_col="ctr",
                )

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("ATT (causal effect)", f"{psm_result.att:+.4f}")
            p2.metric("Bootstrap SE",         f"{psm_result.std_error:.4f}")
            p3.metric("p-value",              f"{psm_result.p_value:.4f}")
            p4.metric("Balance improved",
                       "✅ Yes" if psm_result.balance_improved else "❌ No")
            st.caption(
                f"Matched {psm_result.n_treated} treated users to {psm_result.n_matched_control} controls. "
                f"Std mean diff before matching: {psm_result.standardised_mean_diff_before:.3f} → "
                f"after: {psm_result.standardised_mean_diff_after:.3f}"
            )

    except ImportError as e:
        st.error(f"Could not import causal module: {e}")


# ── Sample Size ────────────────────────────────────────────────────────────────
with tab_power:
    try:
        from module3_ab_engine.frequentist import required_sample_size
        import numpy as np

        st.subheader("Sample size & power calculator")

        # Clamp default to [0.01, 0.99] in case real CTR is outside the typical range
        _baseline_default = float(min(max(round(ctrl_ctr, 3), 0.01), 0.99))
        baseline = st.number_input("Baseline CTR", 0.01, 0.99, _baseline_default,
                                    0.01, format="%.3f", key="pw_base")
        power_t  = st.slider("Target power (1−β)", 0.60, 0.99, 0.80, 0.01, key="pw_power")

        mde_range = np.linspace(0.002, 0.05, 60)
        n_vals    = [required_sample_size(baseline, m, alpha, power_t)["n_per_arm"]
                     for m in mde_range]

        fig = px.line(x=mde_range, y=n_vals,
                       labels={"x": "Minimum detectable effect (absolute)", "y": "n per arm"},
                       title="Required sample size vs MDE",
                       color_discrete_sequence=["#6366f1"], height=320)
        fig.add_hline(y=len(ctrl_users), line_dash="dash",
                       annotation_text=f"Current n/arm ≈ {len(ctrl_users):,}")
        fig.update_layout(margin=dict(t=40,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

        info = required_sample_size(baseline, mde_input, alpha, power_t)
        st.info(
            f"For **MDE = {mde_input:.3f}**, α = {alpha}, power = {power_t:.0%}:  "
            f"need **{info['n_per_arm']:,}** per arm (**{info['n_total']:,}** total).  "
            f"You currently have **{len(ctrl_users):,}** control users "
            f"{'✅ sufficient' if len(ctrl_users) >= info['n_per_arm'] else '❌ — need more data'}."
        )
    except ImportError as e:
        st.error(f"Could not import frequentist module: {e}")
