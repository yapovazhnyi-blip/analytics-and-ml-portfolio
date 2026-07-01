"""
Evaluation router — /api/v1/evaluation

Endpoints:
  POST /evaluation/judge          — evaluate a single (input, output) pair
  POST /evaluation/batch          — evaluate multiple pairs
  POST /evaluation/compare        — A/B compare two outputs on the same input
  GET  /evaluation/rubrics        — list all predefined rubrics
  POST /evaluation/hallucination  — NLI-based faithfulness / hallucination scoring
"""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter
from auth.dependencies import get_current_user
from auth.key_manager import get_anthropic_key
from fastapi import Depends
from pydantic import BaseModel, Field

from evaluation.judge import LLMJudge
from evaluation.runner import EvalRunner, EvalSample, ComparisonSample
from evaluation.rubrics import PREDEFINED
from schemas.common import DataResponse

router = APIRouter(prefix="/evaluation", tags=["evaluation"], dependencies=[Depends(get_current_user)])


# ── Request / response schemas ────────────────────────────────────────────────

class CustomRubricIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    description: str = Field(..., min_length=1, max_length=500)
    guidance: str = Field(..., min_length=1, max_length=500)
    weight: float = Field(default=1.0, ge=0.1, le=5.0)


class JudgeRequest(BaseModel):
    input_text: str = Field(..., min_length=1, max_length=5000)
    output_text: str = Field(..., min_length=1, max_length=8000)
    reference_text: Optional[str] = Field(None, max_length=5000)
    rubric_names: Optional[list[str]] = Field(
        None,
        description="Predefined rubric names. Defaults to ['accuracy','helpfulness','safety'].",
    )
    custom_rubrics: Optional[list[CustomRubricIn]] = Field(
        None,
        description="Custom evaluation criteria alongside predefined ones.",
    )


class BatchSampleIn(BaseModel):
    input_text: str = Field(..., min_length=1, max_length=5000)
    output_text: str = Field(..., min_length=1, max_length=8000)
    reference_text: Optional[str] = Field(None, max_length=5000)
    label: Optional[str] = Field(None, max_length=100)


class BatchRequest(BaseModel):
    samples: list[BatchSampleIn] = Field(..., min_length=1, max_length=50)
    rubric_names: Optional[list[str]] = None
    custom_rubrics: Optional[list[CustomRubricIn]] = None


class CompareRequest(BaseModel):
    input_text: str = Field(..., min_length=1, max_length=5000)
    output_a: str = Field(..., min_length=1, max_length=8000,
                          description="First output to compare (labelled A).")
    output_b: str = Field(..., min_length=1, max_length=8000,
                          description="Second output to compare (labelled B).")
    label_a: str = Field(default="A", max_length=50)
    label_b: str = Field(default="B", max_length=50)
    reference_text: Optional[str] = Field(None, max_length=5000)
    rubric_names: Optional[list[str]] = None
    custom_rubrics: Optional[list[CustomRubricIn]] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rubrics_to_dicts(custom: Optional[list[CustomRubricIn]]) -> Optional[list[dict]]:
    if not custom:
        return None
    return [c.model_dump() for c in custom]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/rubrics")
async def list_rubrics():
    """
    Returns all predefined evaluation rubrics with their descriptions.
    Use these names in rubric_names when calling the judge endpoints.
    """
    return DataResponse(data={
        name: {
            "description": r.description,
            "guidance":    r.guidance,
            "weight":      r.weight,
        }
        for name, r in PREDEFINED.items()
    })


@router.post("/judge")
async def judge_single(body: JudgeRequest, current_user=Depends(get_current_user)):
    """
    Evaluates a single (input, output) pair against the specified rubrics.

    Returns per-criterion scores (1–5) with explanations, and an overall
    weighted score. Requires ANTHROPIC_API_KEY.

    Example rubric_names: ["accuracy", "helpfulness", "safety"]
    Example custom_rubrics: [{"name": "tone", "description": "Is the tone professional?",
                               "guidance": "5=very professional, 1=unprofessional"}]
    """
    api_key = await get_anthropic_key(current_user, require=False) or ""
    judge = LLMJudge(api_key=api_key)
    result = await judge.evaluate(
        input_text=body.input_text,
        output_text=body.output_text,
        rubric_names=body.rubric_names,
        custom_rubrics=_rubrics_to_dicts(body.custom_rubrics),
        reference_text=body.reference_text,
    )
    return DataResponse(data=result.to_dict())


@router.post("/batch")
async def judge_batch(body: BatchRequest):
    """
    Evaluates multiple (input, output) pairs concurrently.

    Runs up to 3 judge calls simultaneously. For 10 samples with 3 criteria,
    expect ~10–15 seconds total. Returns aggregate scores and per-sample details.
    """
    runner = EvalRunner(max_concurrency=3)
    samples = [
        EvalSample(
            input_text=s.input_text,
            output_text=s.output_text,
            reference_text=s.reference_text,
            label=s.label,
        )
        for s in body.samples
    ]
    report = await runner.run_batch(
        samples=samples,
        rubric_names=body.rubric_names,
        custom_rubrics=_rubrics_to_dicts(body.custom_rubrics),
    )
    return DataResponse(data=report.to_dict())


@router.post("/compare")
async def compare_outputs(body: CompareRequest, current_user=Depends(get_current_user)):
    """
    Compares two outputs (A and B) on the same input.

    Both outputs are evaluated independently to avoid positional bias.
    Returns per-criterion winners and an overall winner based on score gap.
    A gap of less than 5 percentage points is considered a tie.

    Use this to:
      - Compare two prompt versions on the same test case
      - Compare two model families (e.g. GPT-4 vs Claude)
      - Verify that a fine-tuned model beats the base model
    """
    api_key = await get_anthropic_key(current_user, require=False) or ""
    judge = LLMJudge(api_key=api_key)
    comparison = await judge.compare(
        input_text=body.input_text,
        output_a=body.output_a,
        output_b=body.output_b,
        label_a=body.label_a,
        label_b=body.label_b,
        rubric_names=body.rubric_names,
        custom_rubrics=_rubrics_to_dicts(body.custom_rubrics),
        reference_text=body.reference_text,
    )
    return DataResponse(data={
        "input":           body.input_text[:200],
        "winner":          comparison.winner,
        "score_diff_pct":  comparison.score_diff_pct,
        "label_a":         body.label_a,
        "label_b":         body.label_b,
        "result_a":        comparison.result_a.to_dict(),
        "result_b":        comparison.result_b.to_dict(),
        "per_criterion":   comparison.per_criterion_winner(),
    })


# ── Hallucination scoring — NLI-based ─────────────────────────────────────────

class HallucinationRequest(BaseModel):
    answer: str = Field(
        ..., min_length=1, max_length=8000,
        description="The LLM-generated answer to evaluate for hallucinations.",
    )
    context_chunks: list[str] = Field(
        ..., min_length=1, max_length=20,
        description=(
            "Retrieved / reference text passages the answer should be grounded in. "
            "Each chunk is a plain string (no length restriction, but >600 chars are "
            "truncated internally to respect the model's 512-token limit)."
        ),
    )
    threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description=(
            "Entailment probability ≥ threshold → sentence is considered grounded. "
            "Lower values accept weaker support; higher values are stricter."
        ),
    )


class SentenceVerdictOut(BaseModel):
    sentence: str
    is_grounded: bool
    entailment_score: float   # 0–1
    best_chunk_index: int     # -1 if ungrounded
    best_chunk_snippet: str


class HallucinationOut(BaseModel):
    faithfulness_score: float   # 0–1
    faithfulness_pct: int
    hallucination_rate: float   # 1 - faithfulness_score
    grounded_count: int
    total_count: int
    sentences: list[SentenceVerdictOut]
    model_id: str
    inference_ms: int
    error: Optional[str] = None


@router.post("/hallucination", response_model=DataResponse[HallucinationOut])
async def score_hallucination(body: HallucinationRequest):
    """
    NLI-based hallucination / faithfulness scoring.

    Splits the answer into sentences and checks each one against every
    provided context chunk using a local NLI cross-encoder model
    (cross-encoder/nli-deberta-v3-small, ~185 MB).

    A sentence is considered grounded if the max entailment probability
    across all chunks exceeds the threshold (default 0.5). The
    faithfulness_score is the fraction of grounded sentences.

    The model is downloaded from HuggingFace Hub on the first call and
    cached for subsequent requests. Inference runs on CPU in ~2–4 seconds.
    No API key is required.
    """
    from evaluation.hallucination_scorer import NLIFaithfulnessScorer

    scorer = NLIFaithfulnessScorer.get()
    result = await asyncio.to_thread(
        scorer.score, body.answer, body.context_chunks, body.threshold,
    )

    return DataResponse(data=HallucinationOut(
        faithfulness_score=result.faithfulness_score,
        faithfulness_pct=result.faithfulness_pct,
        hallucination_rate=result.hallucination_rate,
        grounded_count=result.grounded_count,
        total_count=result.total_count,
        sentences=[
            SentenceVerdictOut(
                sentence=s.sentence,
                is_grounded=s.is_grounded,
                entailment_score=s.entailment_score,
                best_chunk_index=s.best_chunk_index,
                best_chunk_snippet=s.best_chunk_snippet,
            )
            for s in result.sentences
        ],
        model_id=result.model_id,
        inference_ms=result.inference_ms,
        error=result.error,
    ))
