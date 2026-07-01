"""
Leakage detection spike for Crucible's data profiling layer.

Three leakage classes covered:

  1. Feature leakage     — a feature is suspiciously correlated with the target
                           post-split (e.g. a derived column that encodes the answer)
  2. Temporal leakage    — a time-indexed dataset has future data bleeding into
                           train rows (train rows with timestamps > any test row)
  3. ID / index leakage  — a high-cardinality string/int column that is effectively
                           a row identifier leaks through (near-zero duplicate rate,
                           near-perfect target prediction)

Design principle: return structured findings with severity + rationale,
not just True/False. The Claude advisor layer consumes these findings and
generates actionable suggestions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder


# ── Output types ───────────────────────────────────────────────────────────

class LeakageSeverity(str, Enum):
    HIGH   = "high"    # almost certainly leakage
    MEDIUM = "medium"  # suspicious, investigate
    LOW    = "low"     # minor signal, informational


@dataclass
class LeakageFinding:
    leakage_type: str            # "feature" | "temporal" | "id"
    severity: LeakageSeverity
    column: Optional[str]        # None for dataset-level findings
    rationale: str               # human-readable explanation
    metric_name: str             # what was measured
    metric_value: float          # the actual value
    threshold: float             # threshold that was breached


@dataclass
class LeakageReport:
    findings: list[LeakageFinding] = field(default_factory=list)
    columns_checked: int = 0
    rows_checked: int = 0

    @property
    def has_leakage(self) -> bool:
        return bool(self.findings)

    @property
    def high_severity(self) -> list[LeakageFinding]:
        return [f for f in self.findings if f.severity == LeakageSeverity.HIGH]

    def summary(self) -> str:
        if not self.findings:
            return "No leakage detected."
        lines = [f"⚠ {len(self.findings)} leakage finding(s):"]
        for f in self.findings:
            col = f.column or "dataset"
            lines.append(f"  [{f.severity.upper()}] {f.leakage_type} — {col}: {f.rationale}")
        return "\n".join(lines)


# ── Leakage detectors ──────────────────────────────────────────────────────

class LeakageDetector:
    """
    Detects three classes of data leakage.

    Thresholds are deliberately conservative — the goal is to flag for human
    review, not to auto-drop columns. False positives are preferable to
    missed leakage in an ML experimentation context.
    """

    def __init__(
        self,
        feature_corr_threshold: float = 0.95,   # Pearson/Spearman |r| for feature leakage
        id_duplicate_rate_max: float = 0.01,     # max duplicate rate to flag as ID column
        id_predictive_threshold: float = 0.90,   # CV accuracy to flag ID as predictive
    ):
        self.feature_corr_threshold = feature_corr_threshold
        self.id_duplicate_rate_max = id_duplicate_rate_max
        self.id_predictive_threshold = id_predictive_threshold

    def check(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target_col: str,
        time_col: Optional[str] = None,
    ) -> LeakageReport:
        """
        Full leakage check. Runs all three detectors.

        Args:
            train:      training split
            test:       test split
            target_col: name of the target column
            time_col:   name of timestamp column, if any (enables temporal check)
        """
        report = LeakageReport(
            columns_checked=len(train.columns),
            rows_checked=len(train) + len(test),
        )

        feature_cols = [c for c in train.columns if c not in {target_col, time_col} and time_col]
        feature_cols_all = [c for c in train.columns if c != target_col]

        report.findings.extend(
            self._check_feature_leakage(train, target_col, feature_cols_all)
        )
        if time_col:
            report.findings.extend(
                self._check_temporal_leakage(train, test, time_col)
            )
        report.findings.extend(
            self._check_id_leakage(train, target_col, feature_cols_all)
        )

        return report

    # ── Feature leakage ───────────────────────────────────────────────────

    def _check_feature_leakage(
        self, train: pd.DataFrame, target_col: str, feature_cols: list[str]
    ) -> list[LeakageFinding]:
        """
        Flags features with suspiciously high correlation with the target.

        Uses Pearson for continuous targets, Spearman for ordinal, and
        Cramér's V for categorical target × categorical feature pairs.

        A legitimate feature can have high correlation — this flags for review,
        not automatic removal.
        """
        findings = []
        target = train[target_col]
        target_is_categorical = target.dtype == "object" or target.nunique() < 10

        for col in feature_cols:
            if train[col].dtype == "object":
                continue  # skip raw strings — encode separately if needed
            if pd.api.types.is_datetime64_any_dtype(train[col]):
                continue  # skip datetime columns — not correlatable as-is

            series = train[col].dropna()
            aligned_target = target.loc[series.index]

            if target_is_categorical:
                corr, _ = stats.spearmanr(
                    LabelEncoder().fit_transform(aligned_target.astype(str)),
                    series,
                )
                method = "Spearman"
            else:
                corr, _ = stats.pearsonr(aligned_target.astype(float), series)
                method = "Pearson"

            abs_corr = abs(corr)
            if abs_corr >= self.feature_corr_threshold:
                findings.append(LeakageFinding(
                    leakage_type="feature",
                    severity=LeakageSeverity.HIGH if abs_corr > 0.99 else LeakageSeverity.MEDIUM,
                    column=col,
                    rationale=(
                        f"{method} correlation with target is {abs_corr:.3f} — "
                        f"above threshold {self.feature_corr_threshold}. "
                        f"This column may encode the target or be derived from it."
                    ),
                    metric_name=f"{method.lower()}_correlation",
                    metric_value=abs_corr,
                    threshold=self.feature_corr_threshold,
                ))

        return findings

    # ── Temporal leakage ──────────────────────────────────────────────────

    def _check_temporal_leakage(
        self, train: pd.DataFrame, test: pd.DataFrame, time_col: str
    ) -> list[LeakageFinding]:
        """
        Checks if any train rows have timestamps after the earliest test row.

        This is the canonical temporal leakage pattern: a random split on a
        time-series dataset allows future information to contaminate training.
        """
        findings = []

        train_times = pd.to_datetime(train[time_col], errors="coerce").dropna()
        test_times  = pd.to_datetime(test[time_col],  errors="coerce").dropna()

        if train_times.empty or test_times.empty:
            return findings

        test_min  = test_times.min()
        train_max = train_times.max()

        # Any train row with timestamp > min test timestamp is future leakage
        leaked_rows = (train_times > test_min).sum()

        if leaked_rows > 0:
            leak_rate = leaked_rows / len(train_times)
            findings.append(LeakageFinding(
                leakage_type="temporal",
                severity=LeakageSeverity.HIGH if leak_rate > 0.05 else LeakageSeverity.MEDIUM,
                column=time_col,
                rationale=(
                    f"{leaked_rows} train rows ({leak_rate:.1%}) have timestamps "
                    f"after the earliest test timestamp ({test_min.date()}). "
                    f"This indicates a random split was used on time-series data. "
                    f"Use a chronological split instead."
                ),
                metric_name="leaked_row_rate",
                metric_value=leak_rate,
                threshold=0.0,
            ))

        return findings

    # ── ID / index leakage ────────────────────────────────────────────────

    def _check_id_leakage(
        self, train: pd.DataFrame, target_col: str, feature_cols: list[str]
    ) -> list[LeakageFinding]:
        """
        Flags columns that look like row identifiers — near-unique values
        that nonetheless predict the target (because the target was encoded
        into the ID during data generation, e.g. customer_id sorted by outcome).

        Two-stage check:
          1. Near-zero duplicate rate → candidate ID column
          2. Cross-validated RF accuracy above threshold → confirms predictive leakage
        """
        findings = []
        target = train[target_col]
        target_is_categorical = target.dtype == "object" or target.nunique() < 20

        for col in feature_cols:
            n = len(train[col].dropna())
            if n == 0:
                continue

            duplicate_rate = 1 - train[col].nunique() / n
            if duplicate_rate > self.id_duplicate_rate_max:
                continue  # not ID-like enough

            # It looks like an ID — now check if it predicts the target
            col_vals = train[col].dropna()
            aligned_target = target.loc[col_vals.index]

            try:
                X = col_vals.values.reshape(-1, 1)
                if target_is_categorical:
                    y = LabelEncoder().fit_transform(aligned_target.astype(str))
                    model = RandomForestClassifier(n_estimators=20, random_state=42)
                    scoring = "accuracy"
                    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                else:
                    y = aligned_target.astype(float).values
                    model = RandomForestRegressor(n_estimators=20, random_state=42)
                    scoring = "r2"
                    cv = 3

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
                cv_score = scores.mean()

                if cv_score >= self.id_predictive_threshold:
                    findings.append(LeakageFinding(
                        leakage_type="id",
                        severity=LeakageSeverity.HIGH,
                        column=col,
                        rationale=(
                            f"Column has near-zero duplicate rate ({duplicate_rate:.3f}) "
                            f"suggesting it is an ID, but achieves {cv_score:.3f} CV {scoring} "
                            f"when used alone to predict the target. "
                            f"The ID may encode ordering or grouping correlated with the target."
                        ),
                        metric_name=f"cv_{scoring}",
                        metric_value=cv_score,
                        threshold=self.id_predictive_threshold,
                    ))
            except Exception:
                pass  # skip columns that can't be used as features

        return findings
