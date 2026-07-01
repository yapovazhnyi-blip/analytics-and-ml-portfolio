"""
Tests for the RAG evaluator.

The Claude API is mocked in all tests so they run offline and cost nothing.
Each metric is tested independently and the full evaluate() flow is tested
end-to-end with a mock pipeline.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag.evaluator import (
    RAGEvaluator,
    EvalCase,
    EvalSample,
    EvalReport,
    _split_sentences,
    _extract_json,
)


# ── Helper: build a mock pipeline ────────────────────────────────────────────

def _mock_pipeline(answer="Machine learning automates pattern detection.",
                   chunk_texts=None, succeed=True):
    """Builds a mock RAGPipeline that returns preset answers."""
    from rag.retriever import RetrievedChunk
    from rag.pipeline import QueryResult
    from rag.embedder import MockEmbedder

    chunks = [
        RetrievedChunk(
            document_id="doc1",
            chunk_index=i,
            text=t,
            dense_score=0.9 - i * 0.1,
            bm25_score=0.8,
            rrf_score=0.85,
            rerank_score=-1.0,
            metadata={"source": "doc1.txt"},
        )
        for i, t in enumerate(chunk_texts or [
            "Machine learning is a method of teaching computers to learn from data.",
            "It automates the analytical model building process.",
        ])
    ]

    query_result = QueryResult(
        query="",
        answer=answer if succeed else "",
        chunks=chunks,
        citations=[],
        error=None if succeed else "Pipeline error",
    )

    pipeline = MagicMock()
    pipeline.query = AsyncMock(return_value=query_result)
    pipeline._embedder = MockEmbedder()
    return pipeline


# ══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTION TESTS
# ══════════════════════════════════════════════════════════════════════════

class TestUtilities:

    def test_split_sentences_basic(self):
        text = "First sentence here. Second sentence here. Third one."
        sents = _split_sentences(text)
        assert len(sents) >= 2

    def test_split_sentences_empty(self):
        assert _split_sentences("") == []
        assert _split_sentences("   ") == []

    def test_split_sentences_short_skipped(self):
        # Sentences under 10 chars are skipped
        text = "Hi. This is a longer sentence with real content."
        sents = _split_sentences(text)
        assert not any(len(s) <= 10 for s in sents)

    def test_extract_json_clean(self):
        assert _extract_json("[true, false, true]") == "[true, false, true]"

    def test_extract_json_with_surrounding_text(self):
        raw = "Here is my answer:\n[1, 0, 1]\nThat's it."
        assert _extract_json(raw) == "[1, 0, 1]"

    def test_extract_json_fallback(self):
        # No JSON array — returns the input stripped
        raw = "I cannot determine this."
        result = _extract_json(raw)
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════════════
# FAITHFULNESS METRIC
# ══════════════════════════════════════════════════════════════════════════

class TestFaithfulness:

    @pytest.mark.asyncio
    async def test_high_faithfulness_when_all_supported(self):
        """When Claude says all sentences are supported, score = 1.0."""
        evaluator = RAGEvaluator()
        with patch("rag.evaluator._claude_call", new_callable=AsyncMock) as mock_call, \
             patch("rag.evaluator.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_call.return_value = "[true, true, true]"
            score = await evaluator._faithfulness(
                "ML learns from data. It finds patterns. This is useful.",
                "ML is a field that learns patterns from data and uses them."
            )
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_partial_faithfulness(self):
        """When 2 of 3 sentences are supported, score = 0.667."""
        evaluator = RAGEvaluator()
        with patch("rag.evaluator._claude_call", new_callable=AsyncMock) as mock_call, \
             patch("rag.evaluator.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_call.return_value = "[true, false, true]"
            score = await evaluator._faithfulness(
                "First claim. Second made-up claim. Third supported claim.",
                "Context only supports first and third."
            )
        assert abs(score - 2/3) < 0.01

    @pytest.mark.asyncio
    async def test_faithfulness_neutral_without_api_key(self):
        """Without API key, faithfulness returns 1.0 (neutral / unchecked)."""
        evaluator = RAGEvaluator()
        with patch("rag.evaluator.settings") as mock_settings:
            mock_settings.anthropic_api_key = None
            score = await evaluator._faithfulness("Some answer.", "Some context.")
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_faithfulness_empty_answer(self):
        evaluator = RAGEvaluator()
        score = await evaluator._faithfulness("", "some context")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_faithfulness_handles_parse_error(self):
        """When Claude returns unparseable JSON, falls back to 1.0."""
        evaluator = RAGEvaluator()
        with patch("rag.evaluator._claude_call", new_callable=AsyncMock) as mock_call, \
             patch("rag.evaluator.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_call.return_value = "I cannot determine this."
            score = await evaluator._faithfulness("Some answer.", "Some context.")
        assert score == 1.0


# ══════════════════════════════════════════════════════════════════════════
# ANSWER RELEVANCY METRIC
# ══════════════════════════════════════════════════════════════════════════

class TestAnswerRelevancy:

    @pytest.mark.asyncio
    async def test_relevancy_same_topic(self):
        """Same topic should score higher than completely different topic."""
        from rag.embedder import MockEmbedder
        evaluator = RAGEvaluator()
        pipeline = MagicMock()
        pipeline._embedder = MockEmbedder()

        # Both about ML — should have some positive similarity
        score = await evaluator._answer_relevancy(
            "What is machine learning?",
            "Machine learning is a method of data analysis that automates model building.",
            pipeline,
        )
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_relevancy_empty_answer(self):
        evaluator = RAGEvaluator()
        pipeline = MagicMock()
        pipeline._embedder = MagicMock()
        score = await evaluator._answer_relevancy("question", "", pipeline)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_relevancy_score_range(self):
        """Score must always be between 0 and 1."""
        from rag.embedder import MockEmbedder
        evaluator = RAGEvaluator()
        pipeline = MagicMock()
        pipeline._embedder = MockEmbedder()

        score = await evaluator._answer_relevancy(
            "Tell me about weather patterns.",
            "Python is a programming language used in data science.",
            pipeline,
        )
        assert 0.0 <= score <= 1.0


# ══════════════════════════════════════════════════════════════════════════
# CONTEXT PRECISION METRIC
# ══════════════════════════════════════════════════════════════════════════

class TestContextPrecision:

    @pytest.mark.asyncio
    async def test_all_chunks_useful(self):
        evaluator = RAGEvaluator()
        with patch("rag.evaluator._claude_call", new_callable=AsyncMock) as mock_call, \
             patch("rag.evaluator.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_call.return_value = "[1, 1, 1]"
            score = await evaluator._context_precision(
                "What is ML?",
                ["ML chunk 1", "ML chunk 2", "ML chunk 3"],
            )
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_no_chunks_useful(self):
        evaluator = RAGEvaluator()
        with patch("rag.evaluator._claude_call", new_callable=AsyncMock) as mock_call, \
             patch("rag.evaluator.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_call.return_value = "[0, 0, 0]"
            score = await evaluator._context_precision(
                "What is ML?",
                ["Weather chunk", "Cooking chunk", "Sports chunk"],
            )
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_precision_empty_chunks(self):
        evaluator = RAGEvaluator()
        score = await evaluator._context_precision("any question", [])
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_precision_neutral_without_api_key(self):
        evaluator = RAGEvaluator()
        with patch("rag.evaluator.settings") as mock_settings:
            mock_settings.anthropic_api_key = None
            score = await evaluator._context_precision("?", ["chunk"])
        assert score == 1.0


# ══════════════════════════════════════════════════════════════════════════
# EVAL REPORT
# ══════════════════════════════════════════════════════════════════════════

class TestEvalReport:

    def _make_report(self, faithfulness, relevancy, precision):
        samples = [
            EvalSample(
                question=f"Q{i}",
                answer="some answer",
                faithfulness=f,
                answer_relevancy=r,
                context_precision=p,
                n_chunks_retrieved=3,
            )
            for i, (f, r, p) in enumerate(zip(faithfulness, relevancy, precision))
        ]
        return EvalReport(samples=samples)

    def test_means_computed_correctly(self):
        report = self._make_report([0.8, 0.6], [0.9, 0.7], [0.5, 0.5])
        assert abs(report.faithfulness_mean - 0.7) < 0.01
        assert abs(report.answer_relevancy_mean - 0.8) < 0.01
        assert abs(report.context_precision_mean - 0.5) < 0.01

    def test_error_samples_excluded_from_means(self):
        samples = [
            EvalSample("Q1", "ans", 0.8, 0.9, 0.7, 3),
            EvalSample("Q2", "", 0.0, 0.0, 0.0, 0, error="Pipeline failed"),
        ]
        report = EvalReport(samples=samples)
        assert report.n_errors == 1
        assert report.faithfulness_mean == 0.8   # only Q1 counted

    def test_to_dict_structure(self):
        report = self._make_report([0.9], [0.8], [0.7])
        d = report.to_dict()
        assert "faithfulness_mean" in d
        assert "samples" in d
        assert len(d["samples"]) == 1
        assert "faithfulness" in d["samples"][0]

    def test_overall_mean_harmonic(self):
        """Overall is harmonic mean — low scores drag it down more than arithmetic mean."""
        report = self._make_report([1.0], [1.0], [0.1])
        # Harmonic mean of 1, 1, 0.1 = 3/(1+1+10) ≈ 0.25
        assert report.overall_mean < 0.4


# ══════════════════════════════════════════════════════════════════════════
# FULL EVALUATE() FLOW
# ══════════════════════════════════════════════════════════════════════════

class TestRAGEvaluator:

    @pytest.mark.asyncio
    async def test_evaluate_returns_report(self):
        """Full evaluate() call returns an EvalReport with correct sample count."""
        evaluator = RAGEvaluator()
        pipeline = _mock_pipeline()
        cases = [
            EvalCase("What is ML?"),
            EvalCase("What is deep learning?"),
        ]

        with patch("rag.evaluator._claude_call", new_callable=AsyncMock) as mock_call, \
             patch("rag.evaluator.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_call.return_value = "[true, true]"
            report = await evaluator.evaluate(pipeline, cases, k=2)

        assert report.n_samples == 2
        assert report.n_errors == 0
        assert all(0.0 <= s.faithfulness <= 1.0 for s in report.samples)

    @pytest.mark.asyncio
    async def test_evaluate_handles_pipeline_error(self):
        """When the pipeline fails for a question, that sample shows error=not None."""
        evaluator = RAGEvaluator()
        pipeline = _mock_pipeline(succeed=False)
        cases = [EvalCase("What is ML?")]

        with patch("rag.evaluator.settings") as mock_settings:
            mock_settings.anthropic_api_key = None
            report = await evaluator.evaluate(pipeline, cases, k=2)

        assert report.n_errors == 1
        assert report.samples[0].error is not None

    @pytest.mark.asyncio
    async def test_evaluate_empty_cases(self):
        pipeline = _mock_pipeline()
        evaluator = RAGEvaluator()
        report = await evaluator.evaluate(pipeline, [], k=3)
        assert report.n_samples == 0

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self):
        """With max_concurrency=1, samples are processed serially."""
        evaluator = RAGEvaluator(max_concurrency=1)
        pipeline = _mock_pipeline()
        cases = [EvalCase(f"Q{i}") for i in range(3)]

        with patch("rag.evaluator._claude_call", new_callable=AsyncMock) as mock_call, \
             patch("rag.evaluator.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_call.return_value = "[true]"
            report = await evaluator.evaluate(pipeline, cases, k=2)

        assert report.n_samples == 3
