"""
Batch evaluation runner — evaluates many (input, output) pairs concurrently.

This runner wraps the LLMJudge to:
  - Process test cases concurrently (bounded by max_concurrency)
  - Aggregate scores across all samples
  - Track error rates
  - Support both single-output and A/B comparison modes

COST ESTIMATE
-------------
Claude Haiku pricing (approximate):
  - Each judge call: ~1,500 input tokens + ~200 output tokens
  - Cost per sample: ~$0.0004 (with 3 criteria)
  - 100 samples, 3 criteria: ~$0.04

Use Haiku (default) for routine evaluation, Sonnet for high-stakes decisions.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from evaluation.judge import LLMJudge, JudgeResult, ComparisonResult


# ── Input types ───────────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    input_text: str
    output_text: str
    reference_text: Optional[str] = None
    label: Optional[str] = None   # arbitrary tag for grouping


@dataclass
class ComparisonSample:
    input_text: str
    output_a: str
    output_b: str
    reference_text: Optional[str] = None


# ── Batch results ─────────────────────────────────────────────────────────────

@dataclass
class BatchReport:
    """Aggregate results across all evaluated samples."""
    results: list[JudgeResult] = field(default_factory=list)
    rubric_names: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.results)

    @property
    def n_errors(self) -> int:
        return sum(1 for r in self.results if not r.succeeded)

    @property
    def overall_mean(self) -> float:
        ok = [r.overall_score for r in self.results if r.succeeded]
        return round(sum(ok) / len(ok), 4) if ok else 0.0

    def criterion_mean(self, criterion: str) -> float:
        scores = []
        for r in self.results:
            if not r.succeeded:
                continue
            for s in r.scores:
                if s.criterion == criterion:
                    scores.append(s.normalised)
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    def to_dict(self) -> dict:
        criterion_means = {
            c: self.criterion_mean(c) for c in self.rubric_names
        }
        return {
            "n_samples":      self.n_samples,
            "n_errors":       self.n_errors,
            "overall_mean":   self.overall_mean,
            "criterion_means": criterion_means,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class ComparisonReport:
    """Aggregate results for A/B comparison across many samples."""
    comparisons: list[ComparisonResult] = field(default_factory=list)
    label_a: str = "A"
    label_b: str = "B"

    @property
    def n_samples(self) -> int:
        return len(self.comparisons)

    @property
    def wins_a(self) -> int:
        return sum(1 for c in self.comparisons if c.winner == self.label_a)

    @property
    def wins_b(self) -> int:
        return sum(1 for c in self.comparisons if c.winner == self.label_b)

    @property
    def ties(self) -> int:
        return sum(1 for c in self.comparisons if c.winner is None)

    @property
    def overall_winner(self) -> Optional[str]:
        if self.wins_a > self.wins_b:
            return self.label_a
        elif self.wins_b > self.wins_a:
            return self.label_b
        return None  # tie

    @property
    def score_diff_mean(self) -> float:
        """Mean percentage-point difference (A - B)."""
        diffs = [c.score_diff_pct for c in self.comparisons]
        return round(sum(diffs) / len(diffs), 1) if diffs else 0.0

    def to_dict(self) -> dict:
        return {
            "n_samples":      self.n_samples,
            "label_a":        self.label_a,
            "label_b":        self.label_b,
            "wins_a":         self.wins_a,
            "wins_b":         self.wins_b,
            "ties":           self.ties,
            "overall_winner": self.overall_winner,
            "score_diff_mean": self.score_diff_mean,
            "comparisons": [
                {
                    "input":    c.input_text[:200],
                    "winner":   c.winner,
                    "score_a":  c.result_a.overall_pct,
                    "score_b":  c.result_b.overall_pct,
                    "diff":     c.score_diff_pct,
                    "per_criterion": c.per_criterion_winner(),
                }
                for c in self.comparisons
            ],
        }


# ── Runner ────────────────────────────────────────────────────────────────────

class EvalRunner:
    """
    Runs LLM-as-judge evaluation over batches of samples.

    Two modes:
      run_batch()   — evaluate a list of (input, output) pairs
      run_comparison() — compare two outputs per input (A/B test)
    """

    def __init__(
        self,
        judge: Optional[LLMJudge] = None,
        max_concurrency: int = 3,
    ):
        self.judge = judge or LLMJudge()
        self.max_concurrency = max_concurrency

    async def run_batch(
        self,
        samples: list[EvalSample],
        rubric_names: Optional[list[str]] = None,
        custom_rubrics: Optional[list[dict]] = None,
    ) -> BatchReport:
        """
        Evaluates a list of samples concurrently.

        Args:
            samples:        List of EvalSample (input + output + optional reference).
            rubric_names:   Predefined rubric names. Defaults to general bundle.
            custom_rubrics: Additional custom rubrics.

        Returns:
            BatchReport with aggregate scores and per-sample details.
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _one(sample: EvalSample) -> JudgeResult:
            async with semaphore:
                return await self.judge.evaluate(
                    input_text=sample.input_text,
                    output_text=sample.output_text,
                    rubric_names=rubric_names,
                    custom_rubrics=custom_rubrics,
                    reference_text=sample.reference_text,
                )

        results = await asyncio.gather(*[_one(s) for s in samples])

        from evaluation.rubrics import resolve_rubrics
        resolved = resolve_rubrics(names=rubric_names, custom=custom_rubrics)
        rubric_names_resolved = [r.name for r in resolved]

        return BatchReport(
            results=list(results),
            rubric_names=rubric_names_resolved,
        )

    async def run_comparison(
        self,
        samples: list[ComparisonSample],
        label_a: str = "A",
        label_b: str = "B",
        rubric_names: Optional[list[str]] = None,
        custom_rubrics: Optional[list[dict]] = None,
    ) -> ComparisonReport:
        """
        Compares output A vs output B across multiple samples.

        Each comparison evaluates A and B independently (not head-to-head)
        to avoid positional bias in the judge's responses.
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _one(sample: ComparisonSample) -> ComparisonResult:
            async with semaphore:
                return await self.judge.compare(
                    input_text=sample.input_text,
                    output_a=sample.output_a,
                    output_b=sample.output_b,
                    label_a=label_a,
                    label_b=label_b,
                    rubric_names=rubric_names,
                    custom_rubrics=custom_rubrics,
                    reference_text=sample.reference_text,
                )

        comparisons = await asyncio.gather(*[_one(s) for s in samples])
        return ComparisonReport(
            comparisons=list(comparisons),
            label_a=label_a,
            label_b=label_b,
        )
