"""
Tests for the LLM evaluation framework.

All Claude API calls are mocked — tests run offline and cost nothing.
Coverage: rubric resolution, score parsing, edge cases, batch runner,
comparison logic, and API endpoint structure.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, patch


# ══════════════════════════════════════════════════════════════════════════
# RUBRICS
# ══════════════════════════════════════════════════════════════════════════

class TestRubrics:

    def test_predefined_rubrics_available(self):
        from evaluation.rubrics import PREDEFINED
        for name in ("accuracy", "helpfulness", "safety", "format",
                     "conciseness", "completeness"):
            assert name in PREDEFINED

    def test_resolve_by_name(self):
        from evaluation.rubrics import resolve_rubrics
        rubrics = resolve_rubrics(names=["accuracy", "helpfulness"])
        assert len(rubrics) == 2
        assert rubrics[0].name == "accuracy"

    def test_resolve_unknown_name_raises(self):
        from evaluation.rubrics import resolve_rubrics
        with pytest.raises(ValueError, match="Unknown rubric"):
            resolve_rubrics(names=["nonexistent"])

    def test_resolve_defaults_to_general_bundle(self):
        from evaluation.rubrics import resolve_rubrics, BUNDLE_GENERAL
        rubrics = resolve_rubrics()
        assert len(rubrics) == len(BUNDLE_GENERAL)

    def test_resolve_custom_rubric(self):
        from evaluation.rubrics import resolve_rubrics
        rubrics = resolve_rubrics(custom=[{
            "name": "tone",
            "description": "Is the tone professional?",
            "guidance": "5=very professional, 1=unprofessional",
            "weight": 2.0,
        }])
        assert len(rubrics) == 1
        assert rubrics[0].name == "tone"
        assert rubrics[0].weight == 2.0

    def test_resolve_combined_named_and_custom(self):
        from evaluation.rubrics import resolve_rubrics
        rubrics = resolve_rubrics(
            names=["accuracy"],
            custom=[{"name": "custom1", "description": "d", "guidance": "g"}],
        )
        assert len(rubrics) == 2

    def test_rubric_weight_default_is_one(self):
        from evaluation.rubrics import ACCURACY
        assert ACCURACY.weight == 1.0


# ══════════════════════════════════════════════════════════════════════════
# JUDGE — SCORE PARSING
# ══════════════════════════════════════════════════════════════════════════

class TestJudgeParsing:

    def _make_judge(self):
        from evaluation.judge import LLMJudge
        return LLMJudge()

    def _make_rubrics(self, names):
        from evaluation.rubrics import resolve_rubrics
        return resolve_rubrics(names=names)

    def test_parse_valid_json(self):
        judge = self._make_judge()
        rubrics = self._make_rubrics(["accuracy", "helpfulness"])
        raw = json.dumps({"scores": [
            {"criterion": "accuracy", "score": 4, "explanation": "Mostly correct."},
            {"criterion": "helpfulness", "score": 5, "explanation": "Very helpful."},
        ]})
        scores = judge._parse(raw, rubrics)
        assert len(scores) == 2
        assert scores[0].criterion == "accuracy"
        assert scores[0].score == 4
        assert scores[1].score == 5

    def test_parse_clamps_score_to_valid_range(self):
        judge = self._make_judge()
        rubrics = self._make_rubrics(["accuracy"])
        # Score of 7 should be clamped to 5
        raw = json.dumps({"scores": [
            {"criterion": "accuracy", "score": 7, "explanation": "Too high."},
        ]})
        scores = judge._parse(raw, rubrics)
        assert scores[0].score == 5

    def test_parse_extracts_json_from_surrounding_text(self):
        judge = self._make_judge()
        rubrics = self._make_rubrics(["accuracy"])
        raw = 'Here is my evaluation:\n{"scores": [{"criterion": "accuracy", "score": 3, "explanation": "Ok."}]}'
        scores = judge._parse(raw, rubrics)
        assert scores[0].score == 3

    def test_parse_fallback_on_invalid_json(self):
        """Falls back to neutral score (3) when JSON is unparseable."""
        judge = self._make_judge()
        rubrics = self._make_rubrics(["accuracy"])
        scores = judge._parse("not json at all", rubrics)
        assert len(scores) == 1
        assert scores[0].score == 3

    def test_parse_missing_criterion_gets_neutral(self):
        """When a criterion is missing from the response, use score=3."""
        judge = self._make_judge()
        rubrics = self._make_rubrics(["accuracy", "helpfulness"])
        raw = json.dumps({"scores": [
            {"criterion": "accuracy", "score": 4, "explanation": "Good."},
            # helpfulness missing
        ]})
        scores = judge._parse(raw, rubrics)
        assert len(scores) == 2
        helpfulness = next(s for s in scores if s.criterion == "helpfulness")
        assert helpfulness.score == 3


# ══════════════════════════════════════════════════════════════════════════
# JUDGE — EVALUATE
# ══════════════════════════════════════════════════════════════════════════

class TestJudgeEvaluate:

    def _mock_response(self, scores_dict: dict[str, int]) -> tuple[str, str, int, int]:
        raw = json.dumps({"scores": [
            {"criterion": k, "score": v, "explanation": f"{k} rated {v}."}
            for k, v in scores_dict.items()
        ]})
        return raw, "claude-haiku-mock", 500, 100

    @pytest.mark.asyncio
    async def test_evaluate_returns_judge_result(self):
        from evaluation.judge import LLMJudge
        judge = LLMJudge()

        with patch("evaluation.judge.settings") as ms, \
             patch.object(judge, "_call", new_callable=AsyncMock) as mock_call:
            ms.anthropic_api_key = "test-key"
            mock_call.return_value = self._mock_response({"accuracy": 4, "helpfulness": 5, "safety": 5})

            result = await judge.evaluate(
                input_text="What is Python?",
                output_text="Python is a programming language.",
                rubric_names=["accuracy", "helpfulness", "safety"],
            )

        assert result.succeeded
        assert len(result.scores) == 3
        assert result.overall_score > 0

    @pytest.mark.asyncio
    async def test_evaluate_without_api_key_returns_error(self):
        from evaluation.judge import LLMJudge
        judge = LLMJudge()
        with patch("evaluation.judge.settings") as ms:
            ms.anthropic_api_key = None
            result = await judge.evaluate("q", "a")
        assert not result.succeeded
        assert "ANTHROPIC_API_KEY" in result.error

    @pytest.mark.asyncio
    async def test_overall_score_is_weighted_average(self):
        """With weight=2 on accuracy, accuracy dominates the overall score."""
        from evaluation.judge import LLMJudge
        judge = LLMJudge()

        with patch("evaluation.judge.settings") as ms, \
             patch.object(judge, "_call", new_callable=AsyncMock) as mock_call:
            ms.anthropic_api_key = "test-key"
            mock_call.return_value = self._mock_response({"accuracy": 5, "tone": 1})

            result = await judge.evaluate(
                input_text="q",
                output_text="a",
                custom_rubrics=[
                    {"name": "accuracy", "description": "Accurate?", "guidance": "5=yes, 1=no", "weight": 2.0},
                    {"name": "tone", "description": "Good tone?", "guidance": "5=yes, 1=no", "weight": 1.0},
                ],
            )

        # Weighted: (5*2 + 1*1) / (2+1) normalised = (10+1)/3 / 4 ≈ (3.67/4) ≈ 0.917
        # but actually: accuracy normalised = (5-1)/4 = 1.0, tone normalised = 0.0
        # weighted = (1.0*2 + 0.0*1) / 3 = 0.667
        assert result.succeeded
        assert result.overall_score > 0.5   # accuracy dominates

    @pytest.mark.asyncio
    async def test_criterion_score_normalised_range(self):
        from evaluation.judge import LLMJudge
        judge = LLMJudge()

        with patch("evaluation.judge.settings") as ms, \
             patch.object(judge, "_call", new_callable=AsyncMock) as mock_call:
            ms.anthropic_api_key = "test-key"
            mock_call.return_value = self._mock_response({"accuracy": 1, "helpfulness": 5, "safety": 3})

            result = await judge.evaluate("q", "a",
                                          rubric_names=["accuracy", "helpfulness", "safety"])

        for s in result.scores:
            assert 0.0 <= s.normalised <= 1.0


# ══════════════════════════════════════════════════════════════════════════
# COMPARISON
# ══════════════════════════════════════════════════════════════════════════

class TestComparison:

    @pytest.mark.asyncio
    async def test_compare_returns_winner(self):
        from evaluation.judge import LLMJudge
        judge = LLMJudge()

        call_count = [0]
        async def _mock_call(prompt):
            call_count[0] += 1
            # First call (output A) gets high scores, second (B) gets low
            if call_count[0] == 1:
                raw = json.dumps({"scores": [
                    {"criterion": "accuracy", "score": 5, "explanation": "Perfect."},
                ]})
            else:
                raw = json.dumps({"scores": [
                    {"criterion": "accuracy", "score": 2, "explanation": "Poor."},
                ]})
            return raw, "claude-mock", 100, 50

        with patch("evaluation.judge.settings") as ms, \
             patch.object(judge, "_call", new_callable=AsyncMock, side_effect=_mock_call):
            ms.anthropic_api_key = "test-key"
            result = await judge.compare(
                "What is Python?",
                output_a="Python is a language.",
                output_b="Python is a fruit.",
                rubric_names=["accuracy"],
            )

        assert result.winner == "A"
        assert result.score_diff_pct > 0

    @pytest.mark.asyncio
    async def test_compare_tie_when_close(self):
        from evaluation.judge import LLMJudge
        judge = LLMJudge()

        async def _mock_call(prompt):
            raw = json.dumps({"scores": [
                {"criterion": "accuracy", "score": 4, "explanation": "Good."},
            ]})
            return raw, "claude-mock", 100, 50

        with patch("evaluation.judge.settings") as ms, \
             patch.object(judge, "_call", new_callable=AsyncMock, side_effect=_mock_call):
            ms.anthropic_api_key = "test-key"
            result = await judge.compare("q", "a1", "a2", rubric_names=["accuracy"])

        # Both get score 4 — difference is 0, so winner is None (tie)
        assert result.winner is None

    def test_comparison_report_winner(self):
        from evaluation.judge import ComparisonResult, JudgeResult, CriterionScore
        from evaluation.runner import ComparisonReport

        def _result(score_val):
            s = CriterionScore("accuracy", score_val, "ok", 1.0)
            return JudgeResult("q", "a", None, scores=[s])

        comparisons = [
            ComparisonResult("q", _result(5), _result(2), "X", "Y"),
            ComparisonResult("q", _result(5), _result(2), "X", "Y"),
            ComparisonResult("q", _result(2), _result(5), "X", "Y"),
        ]
        report = ComparisonReport(comparisons, label_a="X", label_b="Y")
        assert report.wins_a == 2
        assert report.wins_b == 1
        assert report.overall_winner == "X"


# ══════════════════════════════════════════════════════════════════════════
# BATCH RUNNER
# ══════════════════════════════════════════════════════════════════════════

class TestBatchRunner:

    @pytest.mark.asyncio
    async def test_batch_aggregates_correctly(self):
        from evaluation.runner import EvalRunner, EvalSample
        from evaluation.judge import LLMJudge

        judge = LLMJudge()
        runner = EvalRunner(judge=judge, max_concurrency=2)

        samples = [
            EvalSample("What is ML?", "ML learns from data."),
            EvalSample("What is AI?", "AI simulates intelligence."),
        ]

        async def _mock_eval(*args, **kwargs):
            from evaluation.judge import JudgeResult, CriterionScore
            s = CriterionScore("accuracy", 4, "Good.", 1.0)
            return JudgeResult(kwargs.get("input_text","q"), kwargs.get("output_text","a"), None, scores=[s])

        with patch.object(judge, "evaluate", new_callable=AsyncMock, side_effect=_mock_eval):
            report = await runner.run_batch(samples, rubric_names=["accuracy"])

        assert report.n_samples == 2
        assert report.n_errors == 0
        assert abs(report.criterion_mean("accuracy") - 0.75) < 0.01  # score 4 → 0.75

    @pytest.mark.asyncio
    async def test_batch_error_excluded_from_mean(self):
        from evaluation.runner import EvalRunner, EvalSample
        from evaluation.judge import LLMJudge, JudgeResult, CriterionScore

        judge = LLMJudge()
        runner = EvalRunner(judge=judge)

        call_count = [0]
        async def _mock_eval(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return JudgeResult("q","a",None,scores=[CriterionScore("accuracy",5,"Good.",1.0)])
            return JudgeResult("q","a",None,error="API error")

        with patch.object(judge, "evaluate", new_callable=AsyncMock, side_effect=_mock_eval):
            report = await runner.run_batch(
                [EvalSample("q1","a1"), EvalSample("q2","a2")],
                rubric_names=["accuracy"],
            )

        assert report.n_errors == 1
        assert report.criterion_mean("accuracy") == 1.0   # only score 5 counted


# ══════════════════════════════════════════════════════════════════════════
# RESULT SERIALISATION
# ══════════════════════════════════════════════════════════════════════════

class TestSerialisaton:

    def test_judge_result_to_dict(self):
        from evaluation.judge import JudgeResult, CriterionScore
        r = JudgeResult(
            input_text="q",
            output_text="a",
            reference_text=None,
            scores=[CriterionScore("accuracy", 4, "Good.", 1.0)],
            model="claude-haiku",
            input_tokens=100,
            output_tokens=50,
        )
        d = r.to_dict()
        assert "overall_score" in d
        assert "overall_pct" in d
        assert len(d["scores"]) == 1
        assert d["scores"][0]["criterion"] == "accuracy"
        assert d["scores"][0]["score_pct"] == 75   # (4-1)/4 * 100

    def test_batch_report_to_dict(self):
        from evaluation.judge import JudgeResult, CriterionScore
        from evaluation.runner import BatchReport
        s = CriterionScore("accuracy", 3, "ok", 1.0)
        r = JudgeResult("q","a",None,scores=[s])
        report = BatchReport(results=[r], rubric_names=["accuracy"])
        d = report.to_dict()
        assert d["n_samples"] == 1
        assert "criterion_means" in d
        assert "accuracy" in d["criterion_means"]
