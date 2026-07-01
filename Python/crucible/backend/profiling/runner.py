"""
ProfileRunner — orchestrates Crucible's profiling suite.

Runs all four profiling modules against a DataFrame and returns a single
ProfileReport. The profiling router delegates entirely to this class —
no profiling logic lives in the router.

Modules orchestrated:
  1. MissingnessAnalyser  → per-column null rates + systematic detection
  2. CorrelationAnalyser  → pairwise correlation + VIF multicollinearity
  3. LeakageDetector      → feature / temporal / ID leakage (promoted from spike)
  4. DistributionAnalyser → target analysis + per-column stats

The runner also prepares structured output suitable for the Claude advisor
API call — a compact text summary that fits comfortably in a prompt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from profiling.missingness import MissingnessResult, analyse_missingness
from profiling.correlation import CorrelationReport, analyse_correlations
from profiling.distributions import DistributionReport, analyse_distributions
from profiling.leakage import LeakageDetector, LeakageReport


@dataclass
class ProfileReport:
    dataset_id: int
    n_rows: int
    n_columns: int

    missingness: list[MissingnessResult]
    correlation: CorrelationReport
    distributions: DistributionReport
    leakage: Optional[LeakageReport]          # None if no target column given

    duration_secs: float
    warnings: list[str] = field(default_factory=list)

    def to_advisor_prompt(self) -> str:
        """
        Compact text summary for the Claude advisor API call.
        Keeps the most actionable signals — not a full data dump.
        """
        lines = [
            f"Dataset: {self.n_rows} rows × {self.n_columns} columns",
            "",
        ]

        # Missingness
        high_miss = [m for m in self.missingness if m.missing_rate >= 0.10]
        if high_miss:
            lines.append("Significant missing data:")
            for m in high_miss[:5]:
                systematic = " (systematic)" if m.likely_systematic else ""
                lines.append(f"  - {m.column}: {m.missing_rate:.1%} missing{systematic}")
        else:
            lines.append("Missingness: no columns above 10% missing.")

        # Correlation
        if self.correlation.high_pairs:
            lines.append(f"\nHigh pairwise correlations (|r| ≥ 0.90):")
            for p in self.correlation.high_pairs[:5]:
                lines.append(f"  - {p.col_a} ↔ {p.col_b}: r={p.correlation}")
        severe_vif = [v for v in self.correlation.vif_results if v.severe]
        if severe_vif:
            lines.append(f"\nMulticollinearity (VIF > 10):")
            for v in severe_vif[:5]:
                lines.append(f"  - {v.column}: VIF={v.vif:.1f}")

        # Target analysis
        if self.distributions.target_analysis:
            ta = self.distributions.target_analysis
            lines.append(f"\nTarget ({ta.column}, {ta.task_type}):")
            if ta.imbalance_warning:
                lines.append(f"  ⚠ {ta.imbalance_warning}")
            if ta.skewness_warning:
                lines.append(f"  ⚠ {ta.skewness_warning}")
            if ta.task_type == "classification" and ta.class_distribution:
                top = ta.class_distribution[:3]
                lines.append(f"  Classes: " + ", ".join(
                    f"{c.label}={c.proportion:.1%}" for c in top
                ))

        # Leakage
        if self.leakage and self.leakage.has_leakage:
            lines.append(f"\nLeakage findings ({len(self.leakage.findings)}):")
            for f in self.leakage.findings[:3]:
                col = f.column or "dataset"
                lines.append(f"  ⚠ [{f.severity.upper()}] {f.leakage_type} — {col}")
                lines.append(f"    {f.rationale}")

        return "\n".join(lines)


class ProfileRunner:
    """
    Runs the full Crucible profiling suite against a DataFrame.

    Usage:
        runner = ProfileRunner()
        report = await runner.run(
            df=my_dataframe,
            dataset_id=42,
            target_column="survived",
            time_column="date",
            test_fraction=0.2,
        )
    """

    def __init__(
        self,
        correlation_threshold: float = 0.90,
        missingness_systematic_threshold: float = 0.30,
        leakage_feature_threshold: float = 0.95,
    ):
        self.correlation_threshold = correlation_threshold
        self.missingness_systematic_threshold = missingness_systematic_threshold
        self.leakage_feature_threshold = leakage_feature_threshold

    async def run(
        self,
        df: pd.DataFrame,
        dataset_id: int,
        target_column: Optional[str] = None,
        time_column: Optional[str] = None,
        test_fraction: float = 0.2,
    ) -> ProfileReport:
        """
        Run all profiling modules and return a unified ProfileReport.

        Profiling is CPU-bound but fast enough for Phase 1 to run
        synchronously in the request/response cycle. Phase 2 will
        move this to a background job with WebSocket progress for
        large datasets.
        """
        from observability.tracing import start_span

        with start_span("profiling.run", {
            "dataset_id": dataset_id, "n_rows": len(df), "n_columns": len(df.columns),
        }):
            return await self._run_inner(df, dataset_id, target_column, time_column, test_fraction)

    async def _run_inner(
        self,
        df: pd.DataFrame,
        dataset_id: int,
        target_column: Optional[str] = None,
        time_column: Optional[str] = None,
        test_fraction: float = 0.2,
    ) -> ProfileReport:
        t0 = time.monotonic()
        warnings = []

        # ── 1. Missingness ─────────────────────────────────────────────────
        missingness = analyse_missingness(
            df,
            systematic_threshold=self.missingness_systematic_threshold,
        )

        # ── 2. Correlations + VIF ──────────────────────────────────────────
        if len(df.columns) > 50:
            warnings.append(
                f"VIF computation skipped — dataset has {len(df.columns)} columns "
                f"(limit: 50). Pairwise correlations still computed."
            )

        correlation = analyse_correlations(
            df,
            correlation_threshold=self.correlation_threshold,
        )

        # ── 3. Distributions ───────────────────────────────────────────────
        distributions = analyse_distributions(df, target_column=target_column)

        # ── 4. Leakage detection ───────────────────────────────────────────
        leakage: Optional[LeakageReport] = None
        if target_column and target_column in df.columns:
            split_idx = max(1, int(len(df) * (1 - test_fraction)))
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()

            if len(test_df) > 0:
                detector = LeakageDetector(
                    feature_corr_threshold=self.leakage_feature_threshold,
                )
                leakage = detector.check(
                    train=train_df,
                    test=test_df,
                    target_col=target_column,
                    time_col=time_column,
                )
            else:
                warnings.append(
                    "Leakage detection skipped — dataset too small to split "
                    f"(test_fraction={test_fraction} produced 0 test rows)."
                )

        return ProfileReport(
            dataset_id=dataset_id,
            n_rows=len(df),
            n_columns=len(df.columns),
            missingness=missingness,
            correlation=correlation,
            distributions=distributions,
            leakage=leakage,
            duration_secs=round(time.monotonic() - t0, 3),
            warnings=warnings,
        )

    @staticmethod
    def load_dataframe(file_path: str, source_type: str) -> pd.DataFrame:
        """
        Load a dataset file into a DataFrame.
        Centralised here so all profiling paths use the same loading logic.
        """
        if source_type in ("csv", "bigquery"):
            return pd.read_csv(file_path)
        elif source_type in ("parquet", "sql", "rest_api"):
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Cannot load source type '{source_type}'")
