"""
Agent Benchmark — a fixed set of Crucible goals used to evaluate any agent
(Claude-backed or a fine-tuned local model) on a consistent, repeatable basis.

WHY A BENCHMARK SUITE
------------------------
"The agent works" is not a testable claim on its own. A benchmark suite
turns it into a measurable one: a fixed list of representative goals, each
with an expected tool (so you can check the agent did the RIGHT kind of
work, not just produced SOME answer), run through the agent, and scored.

This is what makes "test on platform" in the training pipeline concrete:
after fine-tuning an agent on captured traces, run it through this same
benchmark and compare the score against the Claude-backed baseline. If the
fine-tuned model's benchmark score is close to baseline at a fraction of
the cost, that's the evidence the fine-tuning was worthwhile.

SCORING DIMENSIONS
---------------------
Each benchmark case is scored on two axes:
  tool_correctness — did the agent call (at least one of) the expected
                      tool(s) for this goal? (binary: 1.0 or 0.0)
  answer_quality    — LLM-judged quality of the final answer against the
                      goal, using the "accuracy" and "helpfulness" rubrics
                      (continuous: 0.0-1.0, via the existing LLMJudge)

The combined score per case is the average of these two. The benchmark
report's overall_score is the mean across all cases.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BenchmarkCase:
    """One benchmark goal with its expected tool(s)."""
    name: str
    goal: str
    expected_tools: list[str]      # passes if ANY of these tools was called


# Standard Crucible agent benchmark — covers the core workflows
STANDARD_BENCHMARK: list[BenchmarkCase] = [
    BenchmarkCase(
        name="list_datasets",
        goal="What datasets are available in Crucible?",
        expected_tools=["list_datasets"],
    ),
    BenchmarkCase(
        name="dataset_info",
        goal="Tell me about the schema and column types of dataset 1.",
        expected_tools=["get_dataset_info"],
    ),
    BenchmarkCase(
        name="run_profiling",
        goal="Profile dataset 1 and tell me about any data quality issues.",
        expected_tools=["run_profiling"],
    ),
    BenchmarkCase(
        name="start_experiment",
        goal="Train a classification model on dataset 1 to predict the 'label' column.",
        expected_tools=["start_experiment"],
    ),
    BenchmarkCase(
        name="check_experiment_status",
        goal="What is the status of experiment 1?",
        expected_tools=["get_experiment_status"],
    ),
]


@dataclass
class BenchmarkCaseResult:
    case_name: str
    goal: str
    tool_correctness: float
    answer_quality: float
    combined_score: float
    tools_called: list[str]
    final_answer: str
    passed: bool

    def to_dict(self) -> dict:
        return {
            "case_name":        self.case_name,
            "goal":              self.goal,
            "tool_correctness":  round(self.tool_correctness, 2),
            "answer_quality":    round(self.answer_quality, 2),
            "combined_score":    round(self.combined_score, 2),
            "tools_called":      self.tools_called,
            "passed":            self.passed,
        }


@dataclass
class BenchmarkReport:
    cases: list[BenchmarkCaseResult] = field(default_factory=list)
    overall_score: float = 0.0
    n_passed: int = 0
    n_total: int = 0
    elapsed_secs: float = 0.0

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 3),
            "n_passed":      self.n_passed,
            "n_total":       self.n_total,
            "pass_rate":     round(self.n_passed / max(self.n_total, 1), 3),
            "elapsed_secs":  round(self.elapsed_secs, 2),
            "cases":         [c.to_dict() for c in self.cases],
        }


PASS_THRESHOLD = 0.5   # combined_score >= this counts as a pass


async def run_benchmark(
    runner,                     # ReActRunner instance (has .run(goal) -> AgentSession)
    cases: Optional[list[BenchmarkCase]] = None,
    api_key: str = "",
) -> BenchmarkReport:
    """
    Runs the agent through each benchmark case and scores the results.

    Args:
        runner: Any object with an async .run(goal) -> session method where
                session has .events (list of tool_call dicts) and .final_answer.
        cases:  Benchmark cases to run. Defaults to STANDARD_BENCHMARK.
        api_key: Anthropic key for the LLM-judged answer_quality scoring.
                 If empty, answer_quality defaults to 0.5 (neutral) for all cases.

    Returns:
        BenchmarkReport with per-case and aggregate scores.
    """
    from evaluation.judge import LLMJudge

    cases = cases or STANDARD_BENCHMARK
    judge = LLMJudge(api_key=api_key) if api_key else None

    start = time.monotonic()
    results = []

    for case in cases:
        session = await runner.run(case.goal)
        session_dict = session.to_dict() if hasattr(session, "to_dict") else {}

        tools_called = [
            e.get("tool", "") for e in session_dict.get("events", [])
            if e.get("type") == "tool_call"
        ]
        tool_correctness = 1.0 if any(t in case.expected_tools for t in tools_called) else 0.0

        final_answer = session_dict.get("final_answer", "")
        if judge and final_answer:
            try:
                judge_result = await judge.evaluate(
                    input_text=case.goal,
                    output_text=final_answer,
                    rubric_names=["accuracy", "helpfulness"],
                )
                answer_quality = judge_result.overall_score if not judge_result.error else 0.5
            except Exception:
                answer_quality = 0.5
        else:
            answer_quality = 0.5 if final_answer else 0.0

        combined = (tool_correctness + answer_quality) / 2
        results.append(BenchmarkCaseResult(
            case_name=case.name,
            goal=case.goal,
            tool_correctness=tool_correctness,
            answer_quality=answer_quality,
            combined_score=combined,
            tools_called=tools_called,
            final_answer=final_answer,
            passed=combined >= PASS_THRESHOLD,
        ))

    elapsed = time.monotonic() - start
    n_passed = sum(1 for r in results if r.passed)
    overall = sum(r.combined_score for r in results) / max(len(results), 1)

    return BenchmarkReport(
        cases=results,
        overall_score=overall,
        n_passed=n_passed,
        n_total=len(results),
        elapsed_secs=elapsed,
    )
