"""
m1_data_pipeline/quality_checks.py
Automated data quality validation with a simple pass/fail report.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from m1_data_pipeline.config import cfg


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class QualityReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def print(self) -> None:
        print("\n── Data Quality Report ─────────────────────────────────────")
        for r in self.results:
            icon = "✓" if r.passed else "✗"
            print(f"  {icon}  {r.name:<45} {r.detail}")
        print("────────────────────────────────────────────────────────────")
        status = "ALL CHECKS PASSED" if self.passed else "SOME CHECKS FAILED"
        print(f"  {status}\n")


def check_ratings(df: pd.DataFrame) -> list[CheckResult]:
    results = []

    # Row count
    results.append(CheckResult(
        "ratings row count ≥ 20M",
        len(df) >= 20_000_000,
        f"actual={len(df):,}",
    ))

    # No nulls in key columns
    nulls = df[["user_id", "movie_id", "rating"]].isnull().sum().sum()
    results.append(CheckResult("no nulls in ratings key cols", nulls == 0, f"nulls={nulls}"))

    # Rating range
    in_range = df["rating"].between(0.5, 5.0).all()
    results.append(CheckResult("rating values in [0.5, 5.0]", bool(in_range)))

    # Duplicates
    dupes = df.duplicated(subset=["user_id", "movie_id", "timestamp"]).sum()
    results.append(CheckResult(
        "no duplicate (user, movie, ts) rows",
        dupes == 0,
        f"dupes={dupes}",
    ))

    return results


def check_events(df: pd.DataFrame) -> list[CheckResult]:
    results = []

    results.append(CheckResult(
        "events row count ≥ 1M",
        len(df) >= 1_000_000,
        f"actual={len(df):,}",
    ))

    valid_types = {"impression", "click", "completion", "skip"}
    invalid = ~df["event_type"].isin(valid_types)
    results.append(CheckResult(
        "all event_type values are valid",
        not invalid.any(),
        f"invalid={invalid.sum()}",
    ))

    # Clicks never exceed impressions per session
    counts = df.groupby(["session_id", "event_type"]).size().unstack(fill_value=0)
    if "impression" in counts and "click" in counts:
        bad = (counts["click"] > counts["impression"]).sum()
        results.append(CheckResult(
            "clicks ≤ impressions per session",
            bad == 0,
            f"violations={bad}",
        ))

    return results


def run_checks() -> QualityReport:
    report = QualityReport()

    # ── Ratings ───────────────────────────────────────────────────────────────
    ratings_path = cfg.processed_dir / "ratings.parquet"
    if ratings_path.exists():
        ratings = pd.read_parquet(ratings_path)
        for r in check_ratings(ratings):
            report.add(r)
    else:
        report.add(CheckResult("ratings.parquet exists", False, "file missing"))

    # ── Events ────────────────────────────────────────────────────────────────
    events_path = cfg.events_dir / "streaming_events.parquet"
    if events_path.exists():
        events = pd.read_parquet(events_path)
        for r in check_events(events):
            report.add(r)
    else:
        report.add(CheckResult("streaming_events.parquet exists", False, "file missing"))

    return report


if __name__ == "__main__":
    report = run_checks()
    report.print()
    sys.exit(0 if report.passed else 1)
