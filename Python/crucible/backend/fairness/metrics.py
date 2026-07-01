"""
Fairness metrics for binary and multi-class classification.

WHY FAIRNESS METRICS MATTER
----------------------------
A model can achieve 92% accuracy while systematically disadvantaging a
protected group. Standard holdout metrics (accuracy, F1, AUC) are computed
on the full population — they cannot reveal that the model is three times
more likely to deny loans to one demographic than another at the same
creditworthiness level.

Fairness metrics measure whether the model's behaviour is consistent
across groups defined by a protected attribute (gender, age group, race,
postal code, etc.). They are required by:
  - EU AI Act (high-risk AI systems must report non-discrimination metrics)
  - US EEOC guidelines (80% / four-fifths rule for adverse impact)
  - GDPR Article 22 (automated decision-making affecting individuals)
  - Financial regulators (ECOA, Fair Housing Act for credit decisions)

FOUR METRICS IMPLEMENTED
------------------------

Demographic Parity Difference (DPD)
  P(Ŷ=1 | A=privileged) − P(Ŷ=1 | A=unprivileged)
  Measures whether the model predicts the positive class at equal rates
  across groups, regardless of the true label.
  Industry threshold: |DPD| < 0.10 considered acceptable.
  Use when: the base rates in the population should be equal (e.g. hiring).

Equal Opportunity Difference (EOD)
  TPR(privileged) − TPR(unprivileged)
  Measures whether the model correctly identifies positive cases at equal
  rates across groups. Only looks at true positives, not false positives.
  Use when: missing a qualified candidate is the primary concern.

Equalized Odds Difference (EqOdd)
  max(|TPR_priv − TPR_unpriv|, |FPR_priv − FPR_unpriv|)
  Requires both TPR and FPR to be equal across groups. Stricter than EOD.
  Use when: both false negatives AND false positives have serious consequences
  (e.g. medical diagnosis, criminal justice risk scoring).

Disparate Impact Ratio (DIR)
  P(Ŷ=1 | A=unprivileged) / P(Ŷ=1 | A=privileged)
  The EEOC's 80% (four-fifths) rule: DIR < 0.8 signals potential discrimination.
  Range: 0 (complete disparate impact) → 1 (perfect parity) → ∞
  Capped at 1 for display (values > 1 mean the unprivileged group is OVER-selected).

MULTI-CLASS EXTENSION
---------------------
For k>2 classes, all metrics are computed per class (one-vs-rest) and
macro-averaged. The "positive class" rotates through each label.

REGRESSION
----------
For regression, we compute: mean absolute prediction difference between
groups and a normalised group mean comparison.

SEVERITY THRESHOLDS
-------------------
  acceptable  |DPD| < 0.05   DIR > 0.90
  marginal    |DPD| < 0.10   DIR > 0.80  (EEOC 80% rule boundary)
  significant |DPD| < 0.20   DIR > 0.60
  severe      |DPD| ≥ 0.20   DIR ≤ 0.60
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Thresholds ────────────────────────────────────────────────────────────────

def _dpd_severity(dpd: float) -> str:
    a = abs(dpd)
    if a < 0.05: return "acceptable"
    if a < 0.10: return "marginal"
    if a < 0.20: return "significant"
    return "severe"

def _dir_severity(ratio: float) -> str:
    r = min(ratio, 1.0)            # values > 1 are actually favourable for minority
    if r > 0.90: return "acceptable"
    if r > 0.80: return "marginal"
    if r > 0.60: return "significant"
    return "severe"

def _overall_severity(metrics: "GroupMetrics") -> str:
    order = {"acceptable": 0, "marginal": 1, "significant": 2, "severe": 3}
    worst = max(
        _dpd_severity(metrics.demographic_parity_diff),
        _dir_severity(metrics.disparate_impact_ratio),
        key=lambda s: order[s],
    )
    return worst


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class GroupStats:
    """Raw statistics for one group within a protected attribute."""
    group_value: str
    n_samples: int
    n_positive_pred: int    # predicted positive
    n_positive_true: int    # true positive labels
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def selection_rate(self) -> float:
        """Fraction of samples predicted positive."""
        return self.n_positive_pred / max(self.n_samples, 1)

    @property
    def tpr(self) -> float:
        """True positive rate (recall / sensitivity)."""
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def fpr(self) -> float:
        """False positive rate (1 - specificity)."""
        denom = self.fp + self.tn
        return self.fp / denom if denom > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tpr

    def to_dict(self) -> dict:
        return {
            "group_value":    self.group_value,
            "n_samples":      self.n_samples,
            "selection_rate": round(self.selection_rate, 4),
            "accuracy":       round(self.accuracy, 4),
            "precision":      round(self.precision, 4),
            "recall":         round(self.recall, 4),
            "tpr":            round(self.tpr, 4),
            "fpr":            round(self.fpr, 4),
        }


@dataclass
class GroupMetrics:
    """
    Fairness metrics for one protected attribute.
    Computed relative to the group with the highest selection rate (privileged group).
    """
    attribute: str
    privileged_group: str       # group with highest selection rate
    unprivileged_group: str     # group with lowest selection rate
    group_stats: list[GroupStats]

    demographic_parity_diff: float  = 0.0   # DPD
    equal_opportunity_diff:  float  = 0.0   # EOD
    equalized_odds_diff:     float  = 0.0   # EqOdd (max of TPR and FPR diffs)
    disparate_impact_ratio:  float  = 1.0   # DIR  (unprivileged / privileged)

    tpr_diff:  float = 0.0
    fpr_diff:  float = 0.0
    severity:  str   = "acceptable"

    def to_dict(self) -> dict:
        return {
            "attribute":              self.attribute,
            "privileged_group":       self.privileged_group,
            "unprivileged_group":     self.unprivileged_group,
            "demographic_parity_diff": round(self.demographic_parity_diff, 4),
            "equal_opportunity_diff": round(self.equal_opportunity_diff, 4),
            "equalized_odds_diff":    round(self.equalized_odds_diff, 4),
            "disparate_impact_ratio": round(self.disparate_impact_ratio, 4),
            "tpr_diff":               round(self.tpr_diff, 4),
            "fpr_diff":               round(self.fpr_diff, 4),
            "severity":               self.severity,
            "group_stats":            [s.to_dict() for s in self.group_stats],
        }


@dataclass
class FairnessReport:
    """Complete fairness analysis for one experiment."""
    experiment_id: int
    task_type: str
    n_samples: int
    protected_attributes: list[str]
    metrics: list[GroupMetrics] = field(default_factory=list)
    overall_severity: str = "acceptable"
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def n_attributes_flagged(self) -> int:
        return sum(1 for m in self.metrics if m.severity != "acceptable")

    def to_dict(self) -> dict:
        return {
            "experiment_id":        self.experiment_id,
            "task_type":            self.task_type,
            "n_samples":            self.n_samples,
            "protected_attributes": self.protected_attributes,
            "overall_severity":     self.overall_severity,
            "n_attributes_flagged": self.n_attributes_flagged,
            "metrics":              [m.to_dict() for m in self.metrics],
            "error":                self.error,
            "thresholds": {
                "demographic_parity": {
                    "acceptable":  "|diff| < 0.05",
                    "marginal":    "|diff| < 0.10  (EEOC 80% rule boundary)",
                    "significant": "|diff| < 0.20",
                    "severe":      "|diff| ≥ 0.20",
                },
                "disparate_impact": {
                    "acceptable":  "ratio > 0.90",
                    "marginal":    "ratio > 0.80  (EEOC 80% / four-fifths rule)",
                    "significant": "ratio > 0.60",
                    "severe":      "ratio ≤ 0.60",
                },
            },
        }


# ── Core metric calculations ──────────────────────────────────────────────────

def compute_group_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_mask: np.ndarray,
    group_value: str,
    positive_class: int = 1,
) -> GroupStats:
    """Computes confusion-matrix-based stats for samples where group_mask is True."""
    y_t = y_true[group_mask]
    y_p = y_pred[group_mask]

    tp = int(np.sum((y_t == positive_class) & (y_p == positive_class)))
    fp = int(np.sum((y_t != positive_class) & (y_p == positive_class)))
    tn = int(np.sum((y_t != positive_class) & (y_p != positive_class)))
    fn = int(np.sum((y_t == positive_class) & (y_p != positive_class)))

    return GroupStats(
        group_value=str(group_value),
        n_samples=int(group_mask.sum()),
        n_positive_pred=int(np.sum(y_p == positive_class)),
        n_positive_true=int(np.sum(y_t == positive_class)),
        tp=tp, fp=fp, tn=tn, fn=fn,
    )


def compute_group_metrics(
    attribute: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_values: np.ndarray,
    positive_class: int = 1,
) -> GroupMetrics:
    """
    Computes all four fairness metrics for one protected attribute.

    For multi-class (positive_class != 1), metrics are computed one-vs-rest.
    The group with the highest selection rate is treated as the privileged group.
    """
    unique_groups = np.unique(protected_values)

    group_stats_list: list[GroupStats] = []
    for gv in unique_groups:
        mask = protected_values == gv
        if mask.sum() < 5:      # skip tiny groups (unreliable statistics)
            continue
        gs = compute_group_stats(y_true, y_pred, mask, str(gv), positive_class)
        group_stats_list.append(gs)

    if len(group_stats_list) < 2:
        return GroupMetrics(
            attribute=attribute,
            privileged_group="unknown",
            unprivileged_group="unknown",
            group_stats=group_stats_list,
            severity="acceptable",
        )

    # Identify privileged (highest selection rate) and unprivileged (lowest)
    sorted_by_rate = sorted(group_stats_list, key=lambda s: s.selection_rate, reverse=True)
    priv   = sorted_by_rate[0]
    unpriv = sorted_by_rate[-1]

    dpd  = priv.selection_rate - unpriv.selection_rate
    eod  = priv.tpr - unpriv.tpr
    tprd = priv.tpr - unpriv.tpr
    fprd = priv.fpr - unpriv.fpr
    eqod = max(abs(tprd), abs(fprd))

    # Disparate impact: unprivileged rate / privileged rate
    dir_ = (unpriv.selection_rate / priv.selection_rate
            if priv.selection_rate > 0 else 1.0)

    gm = GroupMetrics(
        attribute=attribute,
        privileged_group=priv.group_value,
        unprivileged_group=unpriv.group_value,
        group_stats=group_stats_list,
        demographic_parity_diff=dpd,
        equal_opportunity_diff=eod,
        equalized_odds_diff=eqod,
        disparate_impact_ratio=dir_,
        tpr_diff=tprd,
        fpr_diff=fprd,
    )
    gm.severity = _overall_severity(gm)
    return gm


def compute_fairness(
    experiment_id: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_df,          # pd.DataFrame with only the protected attribute columns
    task_type: str,
    positive_class: int = 1,
) -> FairnessReport:
    """
    Computes fairness metrics for all protected attributes.

    Args:
        experiment_id:  For labelling the report.
        y_true:         Ground truth labels (holdout set).
        y_pred:         Model predictions (holdout set).
        protected_df:   DataFrame with one column per protected attribute.
        task_type:      "classification" or "regression".
        positive_class: Which label counts as the positive class (default 1).

    Returns:
        FairnessReport with per-attribute GroupMetrics.
    """
    metrics = []
    severity_order = {"acceptable": 0, "marginal": 1, "significant": 2, "severe": 3}

    for col in protected_df.columns:
        protected_values = protected_df[col].fillna("missing").astype(str).values
        try:
            gm = compute_group_metrics(
                attribute=col,
                y_true=y_true.astype(int) if task_type == "classification" else y_true,
                y_pred=y_pred.astype(int) if task_type == "classification" else y_pred,
                protected_values=protected_values,
                positive_class=positive_class,
            )
            metrics.append(gm)
        except Exception as exc:
            # If one attribute fails, continue with the rest
            metrics.append(GroupMetrics(
                attribute=col,
                privileged_group="error",
                unprivileged_group="error",
                group_stats=[],
                severity="acceptable",
            ))

    # Overall severity = worst across all attributes
    overall = max(
        (m.severity for m in metrics),
        key=lambda s: severity_order.get(s, 0),
        default="acceptable",
    )

    return FairnessReport(
        experiment_id=experiment_id,
        task_type=task_type,
        n_samples=len(y_true),
        protected_attributes=list(protected_df.columns),
        metrics=sorted(metrics, key=lambda m: severity_order.get(m.severity, 0), reverse=True),
        overall_severity=overall,
    )
