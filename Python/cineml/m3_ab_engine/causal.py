"""
m3_ab_engine/causal.py
Causal inference methods:
    1. Difference-in-Differences (DiD)
    2. Propensity Score Matching (PSM)

Used to complement A/B results with quasi-experimental analysis
when randomisation is imperfect or post-hoc analysis is required.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ── Difference-in-Differences ─────────────────────────────────────────────────

@dataclass
class DiDResult:
    att: float                   # Average Treatment Effect on the Treated
    std_error: float
    t_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    pre_period_parallel_test: float   # p-value for parallel trends


def difference_in_differences(
    df: pd.DataFrame,
    outcome_col: str,
    unit_col: str,        # user_id or cohort_id
    time_col: str,        # pre=0 / post=1
    treatment_col: str,   # 0=control / 1=treatment
    covariates: list[str] | None = None,
) -> DiDResult:
    """
    Estimate ATT via two-way FE DiD regression:

        Y_it = α + β·Post_t + γ·Treat_i + δ·(Post_t × Treat_i) + ε_it

    δ is the DiD estimator (ATT).
    """
    df = df.copy()
    df["_post_x_treat"] = df[time_col] * df[treatment_col]

    formula_parts = [
        outcome_col, "~",
        time_col, "+",
        treatment_col, "+",
        "_post_x_treat",
    ]
    if covariates:
        formula_parts += ["+", " + ".join(covariates)]
    formula = " ".join(formula_parts)

    model = smf.ols(formula, data=df).fit(cov_type="HC3")

    att = float(model.params["_post_x_treat"])
    se = float(model.bse["_post_x_treat"])
    t = float(model.tvalues["_post_x_treat"])
    p = float(model.pvalues["_post_x_treat"])
    ci = model.conf_int().loc["_post_x_treat"].values

    # Parallel trends: estimate DiD on pre-period only using half the pre window
    pre_df = df[df[time_col] == 0].copy()
    if len(pre_df) > 10:
        pre_df["_fake_post"] = (pre_df.index % 2).astype(int)
        pre_df["_fake_interact"] = pre_df["_fake_post"] * pre_df[treatment_col]
        placebo_model = smf.ols(
            f"{outcome_col} ~ _fake_post + {treatment_col} + _fake_interact",
            data=pre_df,
        ).fit()
        parallel_p = float(placebo_model.pvalues.get("_fake_interact", 1.0))
    else:
        parallel_p = 1.0

    return DiDResult(
        att=att,
        std_error=se,
        t_statistic=t,
        p_value=p,
        ci_lower=float(ci[0]),
        ci_upper=float(ci[1]),
        pre_period_parallel_test=parallel_p,
    )


# ── Propensity Score Matching ─────────────────────────────────────────────────

@dataclass
class PSMResult:
    att: float
    std_error: float
    matched_control_n: int
    matched_treatment_n: int
    smd_before: dict[str, float]   # Standardised Mean Differences before matching
    smd_after: dict[str, float]    # SMD after matching (balance check)


def propensity_score_matching(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    covariate_cols: list[str],
    caliper: float = 0.05,         # max PS distance for a match
    n_neighbours: int = 1,
    seed: int = 42,
) -> PSMResult:
    """
    1-to-N nearest-neighbour PSM with caliper.
    Propensity score estimated via logistic regression.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    df = df.dropna(subset=covariate_cols + [outcome_col, treatment_col]).copy()

    X = df[covariate_cols].values
    T = df[treatment_col].values.astype(int)
    Y = df[outcome_col].values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Propensity score
    lr = LogisticRegression(max_iter=500, random_state=seed)
    lr.fit(X_scaled, T)
    ps = lr.predict_proba(X_scaled)[:, 1]
    df["_ps"] = ps

    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]

    ps_treated = ps[treated_idx].reshape(-1, 1)
    ps_control = ps[control_idx].reshape(-1, 1)

    # Nearest-neighbour matching
    nn = NearestNeighbors(n_neighbors=n_neighbours, metric="euclidean")
    nn.fit(ps_control)
    distances, matched_control_positions = nn.kneighbors(ps_treated)

    valid_mask = distances[:, 0] <= caliper
    matched_treated_idx = treated_idx[valid_mask]
    matched_control_idx = control_idx[matched_control_positions[valid_mask, 0]]

    att = float(Y[matched_treated_idx].mean() - Y[matched_control_idx].mean())
    se = float(np.std(Y[matched_treated_idx] - Y[matched_control_idx], ddof=1)
               / np.sqrt(len(matched_treated_idx)))

    def smd(group1: np.ndarray, group2: np.ndarray) -> float:
        pooled_sd = np.sqrt((group1.var(ddof=1) + group2.var(ddof=1)) / 2)
        return float(abs(group1.mean() - group2.mean()) / pooled_sd) if pooled_sd > 0 else 0.0

    smd_before = {col: smd(X[T == 1, i], X[T == 0, i]) for i, col in enumerate(covariate_cols)}
    smd_after = {
        col: smd(
            X[matched_treated_idx, i],
            X[matched_control_idx, i],
        )
        for i, col in enumerate(covariate_cols)
    }

    return PSMResult(
        att=att,
        std_error=se,
        matched_control_n=len(matched_control_idx),
        matched_treatment_n=len(matched_treated_idx),
        smd_before=smd_before,
        smd_after=smd_after,
    )
