"""
Model Card Generator — assembles structured documentation from Crucible experiment data.

WHAT MODEL CARDS ARE
---------------------
A model card is a structured document that describes a trained ML model:
what it does, what data it was trained on, how it performs across different
groups, and what its limitations are.

Model cards were introduced by Mitchell et al. (2019) "Model Cards for
Model Reporting" (Google). They are now required or recommended by:
  - EU AI Act (2024): high-risk AI systems must document training data,
    performance metrics, and fairness assessment
  - US NIST AI Risk Management Framework: recommends model documentation
    for transparency and accountability
  - Financial regulators (ECB, FCA): model risk management guidelines
    require model documentation for credit and risk models
  - ISO/IEC 42001: AI management system standard includes model documentation

WHAT IS AUTO-GENERATED HERE
-----------------------------
Crucible populates the card from data already computed during training:

  From Experiment record:
    → task type, best model family, CV score, holdout metrics, feature importances

  From Dataset record:
    → dataset name, row count, column count, source type

  From fairness_json (if fairness analysis was run):
    → demographic parity, equal opportunity, disparate impact per attribute

  From contract_json (if data contract was generated):
    → column constraints that were active during training

  Generated automatically:
    → limitations section (always present with drift warning)
    → intended use (inferred from task type)
    → ethical considerations (depends on whether fairness was run)

SCHEMA
------
Based on Google's Model Card Toolkit schema with EU AI Act additions.
All fields are Optional — missing data is handled gracefully.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class MetricEntry:
    name: str
    value: float
    split: str = "holdout"   # "cv" | "holdout"
    notes: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "value": round(self.value, 4),
                "split": self.split, "notes": self.notes}


@dataclass
class FairnessEntry:
    attribute: str
    demographic_parity_diff: float
    equal_opportunity_diff: float
    disparate_impact_ratio: float
    severity: str
    privileged_group: str
    unprivileged_group: str

    def to_dict(self) -> dict:
        return {
            "attribute":               self.attribute,
            "demographic_parity_diff": round(self.demographic_parity_diff, 4),
            "equal_opportunity_diff":  round(self.equal_opportunity_diff, 4),
            "disparate_impact_ratio":  round(self.disparate_impact_ratio, 4),
            "severity":                self.severity,
            "privileged_group":        self.privileged_group,
            "unprivileged_group":      self.unprivileged_group,
        }


@dataclass
class FeatureImportance:
    feature: str
    mean_abs_shap: float

    def to_dict(self) -> dict:
        return {"feature": self.feature, "mean_abs_shap": round(self.mean_abs_shap, 4)}


@dataclass
class ModelCard:
    # ── Identity ──────────────────────────────────────────────────────────
    experiment_id: int
    model_name: str
    model_family: str
    task_type: str
    created_at: str
    crucible_version: str = "1.0.0"

    # ── Training data ─────────────────────────────────────────────────────
    dataset_id: Optional[int] = None
    dataset_name: Optional[str] = None
    dataset_source_type: Optional[str] = None
    n_training_rows: Optional[int] = None
    n_features: Optional[int] = None
    target_column: Optional[str] = None
    feature_names: list[str] = field(default_factory=list)

    # ── Training config ───────────────────────────────────────────────────
    n_trials: Optional[int] = None
    cv_folds: Optional[int] = None
    best_params: dict = field(default_factory=dict)

    # ── Performance ───────────────────────────────────────────────────────
    cv_score: Optional[float] = None
    cv_metric: str = "accuracy"
    metrics: list[MetricEntry] = field(default_factory=list)

    # ── Fairness ──────────────────────────────────────────────────────────
    fairness_assessed: bool = False
    fairness_overall_severity: Optional[str] = None
    fairness_entries: list[FairnessEntry] = field(default_factory=list)

    # ── Explainability ────────────────────────────────────────────────────
    explainability_method: str = "SHAP"
    feature_importances: list[FeatureImportance] = field(default_factory=list)

    # ── Data contract ─────────────────────────────────────────────────────
    contract_version: Optional[str] = None
    contract_n_columns: Optional[int] = None
    contract_tolerance: Optional[float] = None

    # ── Limitations and use ───────────────────────────────────────────────
    intended_use: str = ""
    out_of_scope: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    ethical_considerations: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "experiment_id":            self.experiment_id,
            "model_name":               self.model_name,
            "model_family":             self.model_family,
            "task_type":                self.task_type,
            "created_at":               self.created_at,
            "crucible_version":         self.crucible_version,
            "dataset": {
                "id":           self.dataset_id,
                "name":         self.dataset_name,
                "source_type":  self.dataset_source_type,
                "n_rows":       self.n_training_rows,
                "n_features":   self.n_features,
                "target_column": self.target_column,
                "feature_names": self.feature_names[:20],
            },
            "training": {
                "n_trials":     self.n_trials,
                "cv_folds":     self.cv_folds,
                "best_params":  self.best_params,
            },
            "performance": {
                "cv_score":  round(self.cv_score, 4) if self.cv_score else None,
                "cv_metric": self.cv_metric,
                "metrics":   [m.to_dict() for m in self.metrics],
            },
            "fairness": {
                "assessed":         self.fairness_assessed,
                "overall_severity": self.fairness_overall_severity,
                "attributes":       [f.to_dict() for f in self.fairness_entries],
            },
            "explainability": {
                "method":              self.explainability_method,
                "feature_importances": [fi.to_dict() for fi in self.feature_importances[:10]],
            },
            "data_contract": {
                "version":    self.contract_version,
                "n_columns":  self.contract_n_columns,
                "tolerance":  self.contract_tolerance,
            },
            "intended_use":            self.intended_use,
            "out_of_scope":            self.out_of_scope,
            "limitations":             self.limitations,
            "ethical_considerations":  self.ethical_considerations,
            "recommendations":         self.recommendations,
        }


def generate_model_card(
    experiment,          # Experiment ORM model
    dataset,             # Dataset ORM model (may be None)
) -> ModelCard:
    """
    Generates a ModelCard from a completed Crucible experiment.

    Reads: experiment.results_json, experiment.fairness_json, dataset.contract_json
    All fields are extracted defensively — missing data produces empty/None values.
    """

    # ── Parse results_json ────────────────────────────────────────────────
    results = {}
    if experiment.results_json:
        try:
            results = json.loads(experiment.results_json)
        except (json.JSONDecodeError, TypeError):
            pass

    # ── Parse fairness_json ───────────────────────────────────────────────
    fairness_data = {}
    if hasattr(experiment, "fairness_json") and experiment.fairness_json:
        try:
            fairness_data = json.loads(experiment.fairness_json)
        except (json.JSONDecodeError, TypeError):
            pass

    # ── Parse contract_json ───────────────────────────────────────────────
    contract_data = {}
    if dataset and hasattr(dataset, "contract_json") and dataset.contract_json:
        try:
            contract_data = json.loads(dataset.contract_json)
        except (json.JSONDecodeError, TypeError):
            pass

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics = []
    holdout = results.get("holdout_metrics", {})
    if holdout:
        for name, value in holdout.items():
            if isinstance(value, (int, float)):
                metrics.append(MetricEntry(name=name, value=float(value), split="holdout"))

    # ── Feature importances ───────────────────────────────────────────────
    importances = []
    for fi in results.get("feature_importances", [])[:10]:
        if isinstance(fi, dict) and "feature" in fi:
            importances.append(FeatureImportance(
                feature=fi["feature"],
                mean_abs_shap=float(fi.get("mean_abs_shap", fi.get("importance", 0))),
            ))

    feature_names = results.get("feature_names", [])

    # ── Fairness entries ──────────────────────────────────────────────────
    fairness_entries = []
    for m in fairness_data.get("metrics", []):
        fairness_entries.append(FairnessEntry(
            attribute=m.get("attribute", ""),
            demographic_parity_diff=float(m.get("demographic_parity_diff", 0)),
            equal_opportunity_diff=float(m.get("equal_opportunity_diff", 0)),
            disparate_impact_ratio=float(m.get("disparate_impact_ratio", 1)),
            severity=m.get("severity", "unknown"),
            privileged_group=m.get("privileged_group", ""),
            unprivileged_group=m.get("unprivileged_group", ""),
        ))

    # ── Infer intended use from task type ─────────────────────────────────
    task_type = experiment.task_type or "classification"
    intended_use = _intended_use(task_type, experiment.target_column)

    # ── Limitations ───────────────────────────────────────────────────────
    limitations = _build_limitations(
        task_type=task_type,
        n_rows=dataset.row_count if dataset else None,
        fairness_assessed=bool(fairness_data),
        holdout_metrics=holdout,
    )

    # ── Ethical considerations ────────────────────────────────────────────
    ethical = _build_ethical(fairness_data, fairness_entries)

    # ── Recommendations ───────────────────────────────────────────────────
    recommendations = _build_recommendations(fairness_entries, holdout)

    model_name = (
        f"{getattr(experiment, 'best_model_family', None) or getattr(experiment, 'best_family', None) or 'Unknown'}"
        f" — Experiment {experiment.id}"
    )

    return ModelCard(
        experiment_id=experiment.id,
        model_name=model_name,
        model_family=getattr(experiment, "best_model_family", None) or
                     getattr(experiment, "best_family", "Unknown") or "Unknown",
        task_type=task_type,
        created_at=experiment.created_at.isoformat()
                   if hasattr(experiment, "created_at") and experiment.created_at
                   else datetime.now(timezone.utc).isoformat(),

        # Dataset
        dataset_id=dataset.id if dataset else None,
        dataset_name=dataset.name if dataset else None,
        dataset_source_type=dataset.source_type if dataset else None,
        n_training_rows=dataset.row_count if dataset else None,
        n_features=len(feature_names) or (dataset.column_count - 1 if dataset and dataset.column_count else None),
        target_column=experiment.target_column,
        feature_names=feature_names,

        # Training
        n_trials=getattr(experiment, "n_trials_completed", None) or getattr(experiment, "n_trials", None),
        best_params=results.get("best_params", {}),

        # Performance
        cv_score=getattr(experiment, "best_cv_score", None) or getattr(experiment, "best_score", None),
        cv_metric=_cv_metric(task_type),
        metrics=metrics,

        # Fairness
        fairness_assessed=bool(fairness_data),
        fairness_overall_severity=fairness_data.get("overall_severity"),
        fairness_entries=fairness_entries,

        # Explainability
        feature_importances=importances,

        # Contract
        contract_version=contract_data.get("version"),
        contract_n_columns=contract_data.get("n_cols"),
        contract_tolerance=contract_data.get("tolerance"),

        # Documentation
        intended_use=intended_use,
        out_of_scope=_out_of_scope(task_type),
        limitations=limitations,
        ethical_considerations=ethical,
        recommendations=recommendations,
    )


# ── Text generation helpers ───────────────────────────────────────────────────

def _cv_metric(task_type: str) -> str:
    return "accuracy" if task_type == "classification" else "r2"


def _intended_use(task_type: str, target_column: Optional[str]) -> str:
    target = f" the '{target_column}' column" if target_column else ""
    if task_type == "classification":
        return (
            f"This model is intended to classify{target} from structured tabular features. "
            "It is suitable for batch scoring and real-time inference pipelines where "
            "a human reviewer is available to validate high-stakes decisions."
        )
    if task_type == "regression":
        return (
            f"This model is intended to predict{target} (a continuous value) from "
            "structured tabular features. It is suitable for forecasting and scoring "
            "pipelines where predictions inform but do not automatically trigger actions."
        )
    return (
        f"This model is intended to analyse{target} using automated machine learning. "
        "Review the performance metrics and fairness assessment before production deployment."
    )


def _out_of_scope(task_type: str) -> list[str]:
    return [
        "Production deployment without human review of high-stakes decisions.",
        "Use on populations or data distributions substantially different from the training data.",
        "Causal inference — this model identifies correlations, not causal relationships.",
        "Real-time decisions where data drift has not been monitored.",
    ]


def _build_limitations(
    task_type: str,
    n_rows: Optional[int],
    fairness_assessed: bool,
    holdout_metrics: dict,
) -> list[str]:
    lims = []
    if n_rows and n_rows < 1000:
        lims.append(
            f"Small training set ({n_rows} rows). Performance estimates may be "
            "unreliable. Consider collecting more data before production deployment."
        )
    if not fairness_assessed:
        lims.append(
            "Fairness analysis was not performed. The model may exhibit disparate "
            "performance across demographic groups. Run POST /experiments/{id}/fairness "
            "before deploying in regulated contexts."
        )
    if task_type == "classification":
        acc = holdout_metrics.get("accuracy")
        if acc and acc < 0.80:
            lims.append(
                f"Holdout accuracy is {acc:.1%}. Consider whether this is "
                "sufficient for the intended use case before deployment."
            )
    lims.append(
        "Model performance may degrade over time as the data distribution shifts. "
        "Monitor input feature distributions and prediction distributions using "
        "the drift detection endpoint (POST /drift/check)."
    )
    lims.append(
        "SHAP explanations describe feature contributions for the training distribution. "
        "They may not accurately reflect contributions for individual predictions "
        "on out-of-distribution samples."
    )
    return lims


def _build_ethical(fairness_data: dict, entries: list[FairnessEntry]) -> list[str]:
    if not fairness_data:
        return [
            "No fairness assessment has been conducted. Before deploying this model "
            "in contexts that affect individuals (credit, hiring, healthcare, etc.), "
            "run a fairness analysis across all relevant protected attributes using "
            "POST /experiments/{id}/fairness."
        ]

    severe = [e for e in entries if e.severity == "severe"]
    significant = [e for e in entries if e.severity == "significant"]

    notes = []
    if severe:
        attrs = ", ".join(e.attribute for e in severe)
        notes.append(
            f"SEVERE fairness concern: significant disparity detected for {attrs}. "
            "Deployment in regulated contexts (EU AI Act Article 10, US ECOA) "
            "without remediation may expose the deploying organisation to legal risk."
        )
    if significant:
        attrs = ", ".join(e.attribute for e in significant)
        notes.append(
            f"SIGNIFICANT fairness concern: notable disparity detected for {attrs}. "
            "Review group-level metrics before production deployment."
        )
    if not severe and not significant:
        severity = fairness_data.get("overall_severity", "acceptable")
        notes.append(
            f"Fairness assessment completed. Overall severity: {severity}. "
            "Continue monitoring for disparity drift in production."
        )
    return notes


def _build_recommendations(entries: list[FairnessEntry], holdout_metrics: dict) -> list[str]:
    recs = [
        "Monitor input feature distributions in production using the drift detection endpoint.",
        "Set up automated retraining alerts when drift exceeds the training distribution thresholds.",
        "Retain the data contract (POST /datasets/{id}/contracts/generate) to validate future data batches.",
    ]
    if any(e.severity in ("severe", "significant") for e in entries):
        recs.insert(0,
            "Address fairness disparities before production deployment. "
            "Consider resampling, reweighting, or post-processing calibration."
        )
    return recs
