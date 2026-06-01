"""
Module 3 — Causal Inference Methods.

Implements:
  1. Difference-in-Differences (DiD) — pre/post comparison controlling for trends
  2. Propensity Score Matching (PSM) — adjusts for selection bias in observational data

Use-case in CineML:
  - DiD: estimate effect of recommender launch controlling for seasonal watch trends
  - PSM: correct for power-users being more likely to be in treatment

References:
  Angrist & Pischke (2009) — Mostly Harmless Econometrics
  Rosenbaum & Rubin (1983) — The Central Role of the Propensity Score
"""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


@dataclass
class DiffInDiffResult:
    did_estimate: float     # ATT — average treatment effect on the treated
    std_error: float
    t_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    pre_control_mean: float
    pre_treatment_mean: float
    post_control_mean: float
    post_treatment_mean: float
    n_obs: int


@dataclass
class PSMResult:
    att: float              # Average Treatment Effect on Treated
    std_error: float
    p_value: float
    n_treated: int
    n_matched_control: int
    standardised_mean_diff_before: float
    standardised_mean_diff_after: float
    balance_improved: bool


# ── Difference-in-Differences ─────────────────────────────────────────────────

def difference_in_differences(
    panel: pd.DataFrame,
    outcome_col: str = "ctr",
    treatment_col: str = "treated",
    post_col: str = "post",
    covariates: list[str] | None = None,
) -> DiffInDiffResult:
    """
    Estimate DiD using OLS: Y = β0 + β1·treated + β2·post + β3·(treated×post) + ε

    Args:
        panel        : Long-format DataFrame with one row per (user, period)
        outcome_col  : Name of the outcome variable
        treatment_col: Binary indicator for treatment group membership
        post_col     : Binary indicator for post-treatment period
        covariates   : Additional control variables

    Returns: DiffInDiffResult with β3 = DiD estimate
    """
    df = panel.copy()
    df["interaction"] = df[treatment_col] * df[post_col]

    covariates_str = ""
    if covariates:
        covariates_str = " + " + " + ".join(covariates)

    formula = f"{outcome_col} ~ {treatment_col} + {post_col} + interaction{covariates_str}"
    result = smf.ols(formula, data=df).fit(cov_type="HC3")  # Heteroscedasticity-robust SEs

    did_est = result.params["interaction"]
    se = result.bse["interaction"]
    t_stat = result.tvalues["interaction"]
    p_val = result.pvalues["interaction"]
    ci = result.conf_int().loc["interaction"]

    # Group means for sanity check
    pre_ctrl = df[(df[post_col] == 0) & (df[treatment_col] == 0)][outcome_col].mean()
    pre_trt = df[(df[post_col] == 0) & (df[treatment_col] == 1)][outcome_col].mean()
    post_ctrl = df[(df[post_col] == 1) & (df[treatment_col] == 0)][outcome_col].mean()
    post_trt = df[(df[post_col] == 1) & (df[treatment_col] == 1)][outcome_col].mean()

    naive_did = (post_trt - pre_trt) - (post_ctrl - pre_ctrl)
    log.info("Naive DiD: %.4f  |  OLS DiD: %.4f", naive_did, did_est)

    return DiffInDiffResult(
        did_estimate=float(did_est),
        std_error=float(se),
        t_statistic=float(t_stat),
        p_value=float(p_val),
        ci_lower=float(ci[0]),
        ci_upper=float(ci[1]),
        pre_control_mean=float(pre_ctrl),
        pre_treatment_mean=float(pre_trt),
        post_control_mean=float(post_ctrl),
        post_treatment_mean=float(post_trt),
        n_obs=len(df),
    )


# ── Propensity Score Matching ─────────────────────────────────────────────────

class PropensityScoreMatching:
    """
    1-to-1 nearest-neighbour matching on estimated propensity scores.

    Propensity score e(X) = P(T=1 | X) estimated via logistic regression.
    Matching: for each treated unit, find the closest control unit (by PS)
    within a caliper of 0.1 × std(logit(PS)).
    """

    def __init__(self, caliper: float = 0.1, n_neighbours: int = 1):
        self.caliper = caliper
        self.n_neighbours = n_neighbours
        self._scaler = StandardScaler()
        self._ps_model = LogisticRegression(max_iter=500, C=1.0)
        self.propensity_scores_: np.ndarray | None = None
        self.matched_indices_: dict[int, list[int]] | None = None

    def fit_transform(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        covariate_cols: list[str],
        outcome_col: str,
    ) -> PSMResult:
        """
        Estimate propensity scores, match, and compute ATT.

        Args:
            df            : DataFrame with treatment, covariates, and outcome
            treatment_col : Binary treatment indicator
            covariate_cols: Confounders to condition on
            outcome_col   : Outcome to estimate effect on
        """
        X = self._scaler.fit_transform(df[covariate_cols])
        T = df[treatment_col].values
        Y = df[outcome_col].values

        self._ps_model.fit(X, T)
        ps = self._ps_model.predict_proba(X)[:, 1]
        self.propensity_scores_ = ps

        # Balance check before matching
        smd_before = self._smd(X[T == 1], X[T == 0])

        # Caliper on logit(PS)
        logit_ps = np.log(ps / (1 - ps + 1e-9))
        caliper_threshold = self.caliper * logit_ps.std()

        treated_idx = np.where(T == 1)[0]
        control_idx = np.where(T == 0)[0]

        matched_pairs: list[tuple[int, int]] = []
        used_controls: set[int] = set()

        for i in treated_idx:
            distances = np.abs(logit_ps[i] - logit_ps[control_idx])
            candidates = [(j, d) for j, d in zip(control_idx, distances)
                          if d <= caliper_threshold and j not in used_controls]
            if not candidates:
                continue
            best_j = min(candidates, key=lambda x: x[1])[0]
            matched_pairs.append((i, best_j))
            used_controls.add(best_j)

        if not matched_pairs:
            raise ValueError("No matched pairs found — try increasing the caliper.")

        treated_matched = np.array([p[0] for p in matched_pairs])
        control_matched = np.array([p[1] for p in matched_pairs])

        # Balance check after matching
        smd_after = self._smd(X[treated_matched], X[control_matched])

        # ATT = mean(Y_treated) - mean(Y_matched_control)
        att = float(Y[treated_matched].mean() - Y[control_matched].mean())

        # Bootstrap SE for the ATT
        rng = np.random.default_rng(42)
        boot_atts = []
        for _ in range(500):
            boot_idx = rng.choice(len(matched_pairs), size=len(matched_pairs), replace=True)
            bt = treated_matched[boot_idx]
            bc = control_matched[boot_idx]
            boot_atts.append(Y[bt].mean() - Y[bc].mean())
        se = float(np.std(boot_atts))
        p_val = float(2 * (1 - stats.norm.cdf(abs(att / (se + 1e-9)))))

        log.info(
            "PSM: %d treated matched from %d controls  ATT=%.4f  p=%.4f",
            len(treated_matched), len(control_idx), att, p_val,
        )

        return PSMResult(
            att=att,
            std_error=se,
            p_value=p_val,
            n_treated=len(treated_matched),
            n_matched_control=len(control_matched),
            standardised_mean_diff_before=float(smd_before.mean()),
            standardised_mean_diff_after=float(smd_after.mean()),
            balance_improved=float(smd_after.mean()) < float(smd_before.mean()),
        )

    @staticmethod
    def _smd(X_treated: np.ndarray, X_control: np.ndarray) -> np.ndarray:
        """Standardised mean difference per covariate (lower = better balance)."""
        pooled_std = np.sqrt((X_treated.var(axis=0) + X_control.var(axis=0)) / 2 + 1e-9)
        return np.abs(X_treated.mean(axis=0) - X_control.mean(axis=0)) / pooled_std


# ── Convenience: build panel from event logs ──────────────────────────────────

def build_did_panel(events: pd.DataFrame, split_date: str) -> pd.DataFrame:
    """
    Build a user-level panel DataFrame suitable for DiD.

    Pre-period: before split_date
    Post-period: on or after split_date
    """
    events["date"] = pd.to_datetime(events["timestamp"]).dt.date
    split = pd.Timestamp(split_date).date()

    imps = events[events["event_type"] == "impression"].copy()
    clicks = events[events["event_type"] == "click"].copy()

    # User-period level CTR
    user_arm = events.drop_duplicates("user_id")[["user_id", "arm"]]

    records = []
    for period_label, mask in [("pre", imps["date"] < split), ("post", imps["date"] >= split)]:
        period_imps = imps[mask]
        period_clicks = clicks[clicks["date"] < split if period_label == "pre" else clicks["date"] >= split]
        for user_id, arm in user_arm.itertuples(index=False):
            n_imps = (period_imps["user_id"] == user_id).sum()
            n_clicks = (period_clicks["user_id"] == user_id).sum()
            ctr = n_clicks / n_imps if n_imps > 0 else 0
            records.append({
                "user_id": user_id,
                "period": period_label,
                "post": int(period_label == "post"),
                "treated": int(arm == "treatment"),
                "ctr": ctr,
                "n_impressions": n_imps,
            })

    return pd.DataFrame(records)
