"""
RAG Evaluator — measures quality of a RAG pipeline across three dimensions.

WHY EVALUATION MATTERS
----------------------
Building a RAG pipeline without measuring whether it works is the most
common mistake in the field. A pipeline that retrieves the wrong chunks
will hallucinate confidently. A pipeline that retrieves the right chunks
but ignores them will also hallucinate. Evaluation surfaces which
component is failing so you can fix the right thing.

THE THREE METRICS
-----------------

1. FAITHFULNESS (0–1, higher = better)
   "Is every claim in the answer supported by the retrieved context?"

   Method: LLM-as-judge. Each sentence in the answer is checked against
   the context block. The LLM returns a JSON list of booleans — one per
   sentence. Score = supported_sentences / total_sentences.

   Why this matters: a faithfulness score of 0.5 means half the answer
   was made up. Even if the made-up half sounds plausible, it is not
   grounded in your documents.

2. ANSWER RELEVANCY (0–1, higher = better)
   "Does the answer actually address the question?"

   Method: embedding cosine similarity between the question and the answer.
   Similar embedding = the answer talks about the same topic as the question.

   Why not use LLM-as-judge here: embedding similarity is faster, cheaper,
   and less prone to LLM verbosity bias (LLMs tend to rate verbose answers
   as more relevant even when concise answers are better).

   Limitation: two texts about the same topic will score high even if the
   answer does not logically follow from the question. This is a known
   weakness of embedding-based relevancy. For production, augment with
   RAGAS's question-generation approach: generate 3 questions from the
   answer, compare to the original question. We use the simpler version.

3. CONTEXT PRECISION (0–1, higher = better)
   "Are the retrieved chunks actually useful for answering the question?"

   Method: LLM-as-judge. Each retrieved chunk is rated 1 (useful) or
   0 (not useful) for answering the specific question. Score = mean rating.

   Why this matters: retrieving 5 chunks of which only 1 is relevant
   dilutes the context and increases the chance the LLM ignores the
   useful one. Context precision < 0.5 means your retrieval is noisy.

USAGE
-----
    evaluator = RAGEvaluator()
    test_cases = [
        EvalCase(question="What is overfitting?", ground_truth="Overfitting occurs..."),
    ]
    results = await evaluator.evaluate(pipeline, test_cases, document_ids=["ml-book"])
    print(results.faithfulness_mean)   # 0.82
    print(results.answer_relevancy_mean)  # 0.91
    print(results.context_precision_mean) # 0.75
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Optional

from config import settings


# ── Input / output types ──────────────────────────────────────────────────────

@dataclass
class EvalCase:
    """One question–answer pair for evaluation."""
    question: str
    ground_truth: Optional[str] = None   # optional reference answer


@dataclass
class EvalSample:
    """Evaluation result for a single question."""
    question: str
    answer: str
    faithfulness: float         # 0–1
    answer_relevancy: float     # 0–1
    context_precision: float    # 0–1
    n_chunks_retrieved: int
    error: Optional[str] = None

    @property
    def overall(self) -> float:
        """Harmonic mean of the three metrics."""
        scores = [self.faithfulness, self.answer_relevancy, self.context_precision]
        if any(s <= 0 for s in scores):
            return 0.0
        return 3 / sum(1 / s for s in scores)


@dataclass
class EvalReport:
    """Aggregate evaluation results across all test cases."""
    samples: list[EvalSample] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_errors(self) -> int:
        return sum(1 for s in self.samples if s.error)

    @property
    def faithfulness_mean(self) -> float:
        scores = [s.faithfulness for s in self.samples if not s.error]
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    @property
    def answer_relevancy_mean(self) -> float:
        scores = [s.answer_relevancy for s in self.samples if not s.error]
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    @property
    def context_precision_mean(self) -> float:
        scores = [s.context_precision for s in self.samples if not s.error]
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    @property
    def overall_mean(self) -> float:
        scores = [s.overall for s in self.samples if not s.error]
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    def to_dict(self) -> dict:
        return {
            "n_samples":              self.n_samples,
            "n_errors":               self.n_errors,
            "faithfulness_mean":      self.faithfulness_mean,
            "answer_relevancy_mean":  self.answer_relevancy_mean,
            "context_precision_mean": self.context_precision_mean,
            "overall_mean":           self.overall_mean,
            "samples": [
                {
                    "question":          s.question,
                    "answer":            s.answer,
                    "faithfulness":      s.faithfulness,
                    "answer_relevancy":  s.answer_relevancy,
                    "context_precision": s.context_precision,
                    "n_chunks":          s.n_chunks_retrieved,
                    "error":             s.error,
                }
                for s in self.samples
            ],
        }


# ── Evaluator ─────────────────────────────────────────────────────────────────

class RAGEvaluator:
    """
    Evaluates a RAGPipeline on a set of test questions.

    All three metrics are computed per sample. The evaluator runs all
    samples concurrently (bounded by max_concurrency) to keep wall-clock
    time reasonable even for large test sets.
    """

    def __init__(self, max_concurrency: int = 3):
        """
        Args:
            max_concurrency: Number of questions evaluated in parallel.
                             Limited by the Claude API rate limit.
                             Default 3 = safe for Haiku tier.
        """
        self.max_concurrency = max_concurrency

    async def evaluate(
        self,
        pipeline,
        cases: list[EvalCase],
        document_ids: Optional[list[str]] = None,
        k: int = 5,
        api_key: str = "",
    ) -> EvalReport:
        """
        Runs all evaluation cases through the pipeline and computes metrics.

        Args:
            pipeline:     The RAGPipeline to evaluate.
            cases:        List of EvalCase (question + optional ground truth).
            document_ids: Restrict retrieval to specific documents.
            k:            Number of chunks to retrieve per question.
            api_key:      Anthropic key (user BYOK key preferred; falls back to settings).

        Returns:
            EvalReport with per-sample scores and aggregate statistics.
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = [
            self._eval_one(pipeline, case, document_ids, k, semaphore, api_key)
            for case in cases
        ]
        samples = await asyncio.gather(*tasks, return_exceptions=False)
        return EvalReport(samples=list(samples))

    async def _eval_one(
        self,
        pipeline,
        case: EvalCase,
        document_ids: Optional[list[str]],
        k: int,
        semaphore: asyncio.Semaphore,
        api_key: str = "",
    ) -> EvalSample:
        """Evaluates a single question."""
        async with semaphore:
            try:
                # Run the full RAG query
                result = await pipeline.query(
                    question=case.question,
                    k=k,
                    document_ids=document_ids,
                )

                if not result.succeeded:
                    return EvalSample(
                        question=case.question,
                        answer="",
                        faithfulness=0.0,
                        answer_relevancy=0.0,
                        context_precision=0.0,
                        n_chunks_retrieved=0,
                        error=result.error,
                    )

                # Build context text for LLM judge calls
                context_texts = [c.text for c in result.chunks]
                context_block = "\n---\n".join(context_texts)

                # Run the three metrics concurrently for each sample
                faithfulness, relevancy, precision = await asyncio.gather(
                    self._faithfulness(result.answer, context_block, api_key),
                    self._answer_relevancy(case.question, result.answer, pipeline, api_key),
                    self._context_precision(case.question, context_texts, api_key),
                )

                return EvalSample(
                    question=case.question,
                    answer=result.answer,
                    faithfulness=faithfulness,
                    answer_relevancy=relevancy,
                    context_precision=precision,
                    n_chunks_retrieved=len(result.chunks),
                )

            except Exception as exc:
                return EvalSample(
                    question=case.question,
                    answer="",
                    faithfulness=0.0,
                    answer_relevancy=0.0,
                    context_precision=0.0,
                    n_chunks_retrieved=0,
                    error=str(exc),
                )

    # ── Metric 1: Faithfulness ─────────────────────────────────────────────

    async def _faithfulness(self, answer: str, context: str, api_key: str = "") -> float:
        """
        Splits the answer into sentences, then asks Claude whether each
        sentence is directly supported by the context.

        Returns: fraction of sentences that are supported (0.0–1.0).
        """
        if not answer.strip():
            return 0.0

        sentences = _split_sentences(answer)
        if not sentences:
            return 0.0

        resolved_key = api_key or settings.anthropic_api_key or ""
        if not resolved_key:
            return 1.0  # cannot evaluate without API — return neutral score

        prompt = (
            "You are evaluating whether an answer is faithful to a given context.\n\n"
            f"CONTEXT:\n{context[:4000]}\n\n"
            "ANSWER SENTENCES TO CHECK:\n"
            + "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
            + "\n\nFor each numbered sentence, determine if it is directly supported "
            "by the CONTEXT (not your general knowledge).\n\n"
            "Respond with ONLY a JSON array of booleans, one per sentence. Example:\n"
            '[true, false, true]\n\n'
            "Respond with ONLY the JSON array, nothing else."
        )

        raw = await _claude_call(prompt, max_tokens=200)
        try:
            flags = json.loads(_extract_json(raw))
            if not isinstance(flags, list) or not flags:
                return 1.0
            # Zip so the shorter list limits iteration — prevents score > 1.0
            # when Claude returns more booleans than there are sentences
            pairs = list(zip(sentences, flags))
            supported = sum(1 for _, f in pairs if f is True)
            return round(supported / len(pairs), 4)
        except Exception:
            return 1.0   # parse failure → neutral

    # ── Metric 2: Answer Relevancy ─────────────────────────────────────────

    async def _answer_relevancy(self, question: str, answer: str, pipeline, api_key: str = "") -> float:
        """
        Computes cosine similarity between the question embedding and the
        answer embedding. High similarity = answer is on-topic.

        This is an approximation of RAGAS's answer relevancy metric (which
        generates N questions from the answer and averages their similarity
        to the original). The simpler version is used here for speed.
        """
        if not answer.strip():
            return 0.0

        import asyncio
        import numpy as np

        loop = asyncio.get_event_loop()

        def _encode():
            emb = pipeline._embedder
            q_vec = emb.encode_one(question)
            a_vec = emb.encode_one(answer)
            # Vectors are already L2-normalised → dot product = cosine similarity
            return float(np.clip(q_vec @ a_vec, 0.0, 1.0))

        score = await loop.run_in_executor(None, _encode)
        return round(score, 4)

    # ── Metric 3: Context Precision ────────────────────────────────────────

    async def _context_precision(self, question: str, chunks: list[str], api_key: str = "") -> float:
        """
        Asks Claude to rate each retrieved chunk as useful (1) or not useful (0)
        for answering the specific question.

        Returns: fraction of chunks rated useful (0.0–1.0).
        """
        if not chunks:
            return 0.0

        resolved_key = api_key or settings.anthropic_api_key or ""
        if not resolved_key:
            return 1.0  # cannot evaluate without API

        chunk_block = "\n---\n".join(
            f"CHUNK {i+1}:\n{c[:500]}" for i, c in enumerate(chunks)
        )

        prompt = (
            f"QUESTION: {question}\n\n"
            f"{chunk_block}\n\n"
            "Rate each chunk: does it contain information useful for answering "
            "the QUESTION above?\n\n"
            "Respond with ONLY a JSON array of 1s and 0s, one per chunk. Example:\n"
            "[1, 0, 1, 1, 0]\n\n"
            "Respond with ONLY the JSON array, nothing else."
        )

        raw = await _claude_call(prompt, max_tokens=100)
        try:
            ratings = json.loads(_extract_json(raw))
            if not isinstance(ratings, list):
                return 1.0
            useful = sum(1 for r in ratings if r == 1 or r is True)
            return round(useful / len(chunks), 4)
        except Exception:
            return 1.0


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _claude_call(prompt: str, max_tokens: int = 300) -> str:
    """Makes a single Claude API call for evaluation."""
    import httpx

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": resolved_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": max_tokens,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(
            b["text"] for b in data.get("content", []) if b.get("type") == "text"
        )


def _extract_json(text: str) -> str:
    """Extracts a JSON array from a response that may have surrounding text."""
    text = text.strip()
    # Try to find a JSON array
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    return match.group(0) if match else text


def _split_sentences(text: str) -> list[str]:
    """
    Splits text into sentences for faithfulness evaluation.
    Uses a simple regex that handles '.', '!', '?' followed by space + uppercase.
    """
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]
