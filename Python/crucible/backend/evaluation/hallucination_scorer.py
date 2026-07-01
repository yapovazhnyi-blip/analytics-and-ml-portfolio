"""
NLI-based hallucination / faithfulness scorer.

Uses a local cross-encoder model (cross-encoder/nli-deberta-v3-small, ~185 MB)
to measure how well a generated answer is grounded in provided context chunks.

ALGORITHM
---------
1. Split the generated answer into individual sentences.
2. For each sentence (hypothesis), pair it with each context chunk (premise).
3. Run the NLI model on all (premise, hypothesis) pairs in a single batched
   forward pass.
4. A sentence is "grounded" if the max entailment probability across all
   chunks exceeds the threshold (default 0.5).
5. faithfulness_score = grounded_sentences / total_sentences

This is the same algorithm RAGAS uses for its faithfulness metric — it
treats hallucination as failure of entailment: if no context chunk supports
a claim, that claim is likely hallucinated.

WHY NLI INSTEAD OF LLM-AS-JUDGE FOR FAITHFULNESS
-------------------------------------------------
LLM-as-judge gives holistic quality scores, but NLI has specific advantages
for faithfulness measurement:

  - Sentence-level granularity: shows *exactly which sentences* are hallucinated,
    not just an aggregate score. A RAG response that is 90% grounded but
    contains one fabricated citation needs that specific sentence highlighted.

  - No API cost or latency: runs entirely on local CPU in 2–4 seconds with
    no external calls. No ANTHROPIC_API_KEY required.

  - Deterministic: same input → same score every time, enabling reproducible
    evaluation runs.

  - Explicit probability: the entailment score is a calibrated probability
    from an NLI-trained model, not an integer rubric score converted to 0–1.

MODEL LOADING
-------------
cross-encoder/nli-deberta-v3-small (~185 MB) is downloaded from HuggingFace
Hub on first use and cached in ~/.cache/huggingface/. Subsequent requests
within the same container lifecycle reuse the in-memory singleton — loading
does NOT happen at app startup, only on the first scoring request.

Inference: ~2–4 seconds on a modern CPU for a typical RAG answer (~5–10
sentences) checked against 5–10 context chunks.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class SentenceVerdict:
    """Groundedness verdict for a single sentence in the generated answer."""
    sentence: str
    is_grounded: bool
    entailment_score: float        # 0–1: max entailment probability across all chunks
    best_chunk_index: int          # index of the most-supporting chunk; -1 if ungrounded
    best_chunk_snippet: str        # first 120 chars of the best supporting chunk


@dataclass
class HallucinationResult:
    """Complete NLI faithfulness result for one (answer, context_chunks) pair."""
    faithfulness_score: float      # 0–1: grounded_count / total_count
    grounded_count: int
    total_count: int
    sentences: list[SentenceVerdict] = field(default_factory=list)
    model_id: str = ""
    inference_ms: int = 0
    error: Optional[str] = None

    @property
    def hallucination_rate(self) -> float:
        """Fraction of sentences NOT supported by any context chunk."""
        return round(1.0 - self.faithfulness_score, 4)

    @property
    def faithfulness_pct(self) -> int:
        return round(self.faithfulness_score * 100)


# ── Singleton scorer ──────────────────────────────────────────────────────────

class NLIFaithfulnessScorer:
    """
    Process-level singleton for NLI hallucination scoring.

    The HuggingFace model is loaded lazily on the first scoring request —
    it does NOT slow down application startup. Once loaded the model object
    is kept in memory and reused for all subsequent calls.

    Thread safety: score() is synchronous and CPU-bound. It is always called
    via asyncio.to_thread() from the FastAPI router, so it never blocks the
    event loop.
    """

    MODEL_ID = "cross-encoder/nli-deberta-v3-small"
    DEFAULT_THRESHOLD = 0.5

    _instance: Optional["NLIFaithfulnessScorer"] = None
    _tokenizer = None
    _model = None
    _entailment_idx: int = 2  # updated from model.config.id2label after load

    @classmethod
    def get(cls) -> "NLIFaithfulnessScorer":
        """Return the process-level singleton, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_loaded(self) -> None:
        """
        Load the NLI model into memory if not already done.

        Called on the first scoring request. Subsequent calls are no-ops
        (the model object check is a simple None guard, no lock needed for
        the singleton use case — the GIL serialises the first-load path in
        practice, and scoring is always called from a thread pool thread,
        not the event loop).
        """
        if self._model is not None:
            return

        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
            )
        except ImportError as exc:
            raise RuntimeError(
                "The 'transformers' and 'torch' packages are required for NLI "
                "hallucination scoring. Ensure both are listed in requirements.txt "
                "(not commented out) and rebuild the Docker image."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_ID)
        self._model.eval()

        # Read the entailment label index from the model's own config —
        # the label order differs between NLI checkpoints and hardcoding is fragile.
        id2label: dict = self._model.config.id2label
        self._entailment_idx = next(
            (int(i) for i, lbl in id2label.items() if str(lbl).lower() == "entailment"),
            2,  # safe fallback for nli-deberta-v3-small (confirmed entailment=2)
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def score(
        self,
        answer: str,
        context_chunks: list[str],
        threshold: float = DEFAULT_THRESHOLD,
    ) -> HallucinationResult:
        """
        Score how well the generated answer is grounded in the context chunks.

        Args:
            answer:          LLM-generated text to evaluate.
            context_chunks:  Retrieved / reference passages the answer should
                             be grounded in (list of plain strings).
            threshold:       Entailment probability ≥ threshold → sentence is
                             grounded. Default 0.5.

        Returns:
            HallucinationResult with faithfulness_score and per-sentence
            verdicts. On error, faithfulness_score=0.0 and error is populated.
        """
        if not answer.strip():
            return HallucinationResult(
                faithfulness_score=0.0, grounded_count=0, total_count=0,
                error="answer is empty.",
            )
        clean_chunks = [c.strip() for c in context_chunks if c.strip()]
        if not clean_chunks:
            return HallucinationResult(
                faithfulness_score=0.0, grounded_count=0, total_count=0,
                error="No non-empty context chunks provided.",
            )

        try:
            self._ensure_loaded()
        except RuntimeError as exc:
            return HallucinationResult(
                faithfulness_score=0.0, grounded_count=0, total_count=0,
                error=str(exc),
            )

        sentences = _split_sentences(answer)
        if not sentences:
            return HallucinationResult(
                faithfulness_score=0.0, grounded_count=0, total_count=0,
                error="No sentences could be extracted from the answer.",
            )

        # Truncate chunks to keep tokenised length well under 512.
        # DeBERTa encodes the (premise, hypothesis) pair jointly; a premise
        # longer than ~300 characters leaves little room for the hypothesis.
        clean_chunks = [c[:600] for c in clean_chunks]

        import torch
        import numpy as np

        t0 = time.monotonic()

        n, m = len(sentences), len(clean_chunks)

        # Build all (premise, hypothesis) pairs in a flat list for batched
        # tokenisation and a single forward pass.
        premises = [chunk for sent in sentences for chunk in clean_chunks]
        hypotheses = [sent for sent in sentences for _ in clean_chunks]

        encoding = self._tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self._model(**encoding).logits     # (n*m, 3)
            probs = torch.softmax(logits, dim=1).numpy()  # (n*m, 3)

        # Reshape → (n_sentences, n_chunks, 3_labels)
        probs_3d = probs.reshape(n, m, 3)

        entailment = probs_3d[:, :, self._entailment_idx]  # (n, m)
        best_chunk_per_sent = entailment.argmax(axis=1)     # (n,)
        max_entailment_per_sent = entailment.max(axis=1)    # (n,)

        inference_ms = round((time.monotonic() - t0) * 1000)

        verdicts: list[SentenceVerdict] = []
        for i, sent in enumerate(sentences):
            prob = float(max_entailment_per_sent[i])
            best_idx = int(best_chunk_per_sent[i])
            grounded = prob >= threshold
            verdicts.append(SentenceVerdict(
                sentence=sent,
                is_grounded=grounded,
                entailment_score=round(prob, 4),
                best_chunk_index=best_idx if grounded else -1,
                best_chunk_snippet=(
                    (clean_chunks[best_idx][:120] + "…")
                    if grounded and best_idx < len(clean_chunks)
                    else ""
                ),
            ))

        grounded_count = sum(1 for v in verdicts if v.is_grounded)
        faithfulness = grounded_count / len(verdicts)

        return HallucinationResult(
            faithfulness_score=round(faithfulness, 4),
            grounded_count=grounded_count,
            total_count=len(verdicts),
            sentences=verdicts,
            model_id=self.MODEL_ID,
            inference_ms=inference_ms,
        )


# ── Sentence splitter ─────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences suitable for NLI evaluation.

    Strategy:
      1. Split on sentence-ending punctuation (.!?) followed by whitespace
         and an uppercase letter/digit/quote — the most reliable heuristic
         without a full NLP pipeline (no nltk dependency).
      2. Also split on newlines, so bullet points and numbered lists each
         become separate evaluation units.
      3. Filter out fragments shorter than 8 characters (punctuation
         artefacts, list markers, etc.) that NLI would not handle usefully.
    """
    text = re.sub(r"\r\n|\r", "\n", text.strip())

    # Split on sentence-ending punctuation
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'\d(])', text)

    sentences: list[str] = []
    for part in parts:
        for line in part.split("\n"):
            line = line.strip().strip("-•·*›> ")
            if len(line) >= 8:
                sentences.append(line)

    return sentences
