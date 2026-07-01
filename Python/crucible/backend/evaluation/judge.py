"""
LLM-as-judge — scores any LLM output against configurable evaluation rubrics.

HOW LLM-AS-JUDGE WORKS
-----------------------
Instead of hand-writing evaluation functions for every possible output type,
we ask a stronger LLM (the "judge") to score a weaker model's output.

The judge receives:
  - The input (user question or task)
  - The output (what the model being evaluated produced)
  - An optional reference (ideal answer or ground truth)
  - A rubric (list of criteria with scoring guidance)

The judge returns a score 1–5 per criterion with a brief explanation.

WHY THIS WORKS WELL
-------------------
LLMs are trained on enormous amounts of human-written text, including text
that evaluates and critiques other text (reviews, peer feedback, academic
grading). This makes them surprisingly good at mimicking human judgment.

Meta-evaluation studies (Zheng et al. 2023 "Judging LLM-as-a-Judge with
MT-Bench") show GPT-4-class models agree with human raters at >80% on
most evaluation tasks — comparable to the agreement rate between two
human raters on the same task.

LIMITATIONS
-----------
- Positional bias: judges tend to prefer the first of two options.
  Mitigation: compare A vs B and B vs A, take the majority.
- Verbosity bias: longer outputs are rated higher even if not better.
  Mitigation: include explicit conciseness rubric; provide guidance.
- Self-enhancement bias: models rate outputs similar to their own style higher.
  Mitigation: use a different model family as judge when possible.
- Cost: each judge call uses tokens. Keep rubrics targeted.

PROMPT DESIGN
-------------
The judge prompt is structured to force a specific JSON output format:
  {"scores": [{"criterion": "...", "score": 1-5, "explanation": "..."}]}

Temperature=0 ensures deterministic scoring for the same input.
The system prompt establishes the judge role and constraints explicitly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from config import settings
from evaluation.rubrics import Rubric, resolve_rubrics


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class CriterionScore:
    """Score for one rubric criterion."""
    criterion: str
    score: int           # 1–5
    explanation: str
    weight: float = 1.0

    @property
    def normalised(self) -> float:
        """Score normalised to 0–1."""
        return (self.score - 1) / 4.0


@dataclass
class JudgeResult:
    """Complete evaluation result for one (input, output) pair."""
    input_text: str
    output_text: str
    reference_text: Optional[str]
    scores: list[CriterionScore] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and bool(self.scores)

    @property
    def overall_score(self) -> float:
        """Weighted average of normalised criterion scores."""
        if not self.scores:
            return 0.0
        total_weight = sum(s.weight for s in self.scores)
        if total_weight == 0:
            return 0.0
        return sum(s.normalised * s.weight for s in self.scores) / total_weight

    @property
    def overall_pct(self) -> int:
        return round(self.overall_score * 100)

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 4),
            "overall_pct":   self.overall_pct,
            "scores": [
                {
                    "criterion":   s.criterion,
                    "score":       s.score,
                    "score_pct":   round(s.normalised * 100),
                    "explanation": s.explanation,
                    "weight":      s.weight,
                }
                for s in self.scores
            ],
            "model":        self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error":        self.error,
        }


@dataclass
class ComparisonResult:
    """A/B comparison between two outputs on the same input."""
    input_text: str
    result_a: JudgeResult
    result_b: JudgeResult
    label_a: str = "A"
    label_b: str = "B"

    @property
    def winner(self) -> Optional[str]:
        """Returns label of the winner, or None if tied."""
        if not self.result_a.succeeded or not self.result_b.succeeded:
            return None
        diff = self.result_a.overall_score - self.result_b.overall_score
        if abs(diff) < 0.05:   # < 5% difference = tie
            return None
        return self.label_a if diff > 0 else self.label_b

    @property
    def score_diff_pct(self) -> int:
        """Percentage point difference (A - B)."""
        return self.result_a.overall_pct - self.result_b.overall_pct

    def per_criterion_winner(self) -> dict[str, Optional[str]]:
        """Which output wins on each criterion."""
        result = {}
        scores_a = {s.criterion: s.score for s in self.result_a.scores}
        scores_b = {s.criterion: s.score for s in self.result_b.scores}
        for criterion in scores_a:
            a, b = scores_a.get(criterion, 0), scores_b.get(criterion, 0)
            if a > b:
                result[criterion] = self.label_a
            elif b > a:
                result[criterion] = self.label_b
            else:
                result[criterion] = None  # tie
        return result


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert evaluator of AI-generated text outputs.
Your task is to evaluate a given output against specific criteria.

Rules:
- Score each criterion from 1 (worst) to 5 (best) using the provided guidance.
- Be objective and consistent.
- Base your evaluation ONLY on the output and reference provided, not on your preferences.
- Keep explanations brief (1–2 sentences per criterion).
- You MUST respond with ONLY a valid JSON object — no other text, no markdown fences.

Required JSON format:
{
  "scores": [
    {"criterion": "criterion_name", "score": 1-5, "explanation": "brief reason"}
  ]
}"""


# ── Judge ─────────────────────────────────────────────────────────────────────

class LLMJudge:
    """
    Evaluates LLM outputs using Claude as the judge model.

    The judge is model-agnostic about what it evaluates — it can score
    outputs from any LLM, any prompt version, or any pipeline stage.
    """

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str = ""):
        self.model    = model
        self._api_key = api_key   # resolved by the caller; falls back to settings in evaluate()

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        rubric_names: Optional[list[str]] = None,
        custom_rubrics: Optional[list[dict]] = None,
        reference_text: Optional[str] = None,
    ) -> JudgeResult:
        """
        Evaluates a single (input, output) pair against the specified rubrics.

        Args:
            input_text:      The prompt / question given to the model.
            output_text:     The model's response to evaluate.
            rubric_names:    Predefined rubric names (e.g. ["accuracy", "helpfulness"]).
                             Defaults to ["accuracy", "helpfulness", "safety"].
            custom_rubrics:  Custom rubric dicts: [{name, description, guidance, weight}].
            reference_text:  Optional ideal answer for reference-based scoring.

        Returns:
            JudgeResult with per-criterion scores and overall weighted score.
        """
        resolved_key = self._api_key or settings.anthropic_api_key or ""
        if not resolved_key:
            return JudgeResult(
                input_text=input_text,
                output_text=output_text,
                reference_text=reference_text,
                error="ANTHROPIC_API_KEY not configured.",
            )
        self._api_key = resolved_key

        rubrics = resolve_rubrics(names=rubric_names, custom=custom_rubrics)
        prompt = self._build_prompt(input_text, output_text, reference_text, rubrics)

        try:
            raw, model, in_tok, out_tok = await self._call(prompt)
            scores = self._parse(raw, rubrics)
            return JudgeResult(
                input_text=input_text,
                output_text=output_text,
                reference_text=reference_text,
                scores=scores,
                model=model,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
        except Exception as exc:
            return JudgeResult(
                input_text=input_text,
                output_text=output_text,
                reference_text=reference_text,
                error=str(exc),
            )

    async def compare(
        self,
        input_text: str,
        output_a: str,
        output_b: str,
        label_a: str = "A",
        label_b: str = "B",
        rubric_names: Optional[list[str]] = None,
        custom_rubrics: Optional[list[dict]] = None,
        reference_text: Optional[str] = None,
    ) -> ComparisonResult:
        """
        Evaluates two outputs on the same input and returns a ComparisonResult.

        Both outputs are evaluated independently (not head-to-head) to avoid
        positional bias. The winner is determined by comparing overall scores.
        """
        import asyncio
        result_a, result_b = await asyncio.gather(
            self.evaluate(input_text, output_a, rubric_names, custom_rubrics, reference_text),
            self.evaluate(input_text, output_b, rubric_names, custom_rubrics, reference_text),
        )
        return ComparisonResult(
            input_text=input_text,
            result_a=result_a,
            result_b=result_b,
            label_a=label_a,
            label_b=label_b,
        )

    # ── Prompt building ───────────────────────────────────────────────────────

    def _build_prompt(
        self,
        input_text: str,
        output_text: str,
        reference_text: Optional[str],
        rubrics: list[Rubric],
    ) -> str:
        criteria_block = "\n".join(
            f"- {r.name}: {r.description}\n  Scoring: {r.guidance}"
            for r in rubrics
        )

        ref_block = ""
        if reference_text:
            ref_block = f"\nREFERENCE ANSWER:\n{reference_text[:2000]}\n"

        return (
            f"INPUT:\n{input_text[:2000]}\n\n"
            f"OUTPUT TO EVALUATE:\n{output_text[:3000]}\n"
            f"{ref_block}\n"
            f"EVALUATION CRITERIA:\n{criteria_block}\n\n"
            "Evaluate the OUTPUT against each criterion and respond with the JSON object."
        )

    # ── Claude API call ───────────────────────────────────────────────────────

    async def _call(self, prompt: str) -> tuple[str, str, int, int]:
        """Returns (raw_text, model, input_tokens, output_tokens)."""
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 1000,
                    "temperature": 0,
                    "system": _SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            data = resp.json()

        raw = "".join(
            b["text"] for b in data.get("content", []) if b.get("type") == "text"
        )
        usage = data.get("usage", {})
        return raw, data.get("model", self.model), usage.get("input_tokens", 0), usage.get("output_tokens", 0)

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse(self, raw: str, rubrics: list[Rubric]) -> list[CriterionScore]:
        """
        Parses the judge's JSON response into CriterionScore objects.

        Falls back gracefully when the response is malformed:
        - Extracts JSON from surrounding text if present
        - Clamps scores to 1–5
        - Uses 3 (neutral) for missing criteria
        """
        text = raw.strip()
        # Extract JSON object if surrounded by text or markdown
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Total parse failure — return neutral scores
            return [
                CriterionScore(criterion=r.name, score=3,
                               explanation="Parse failed", weight=r.weight)
                for r in rubrics
            ]

        raw_scores = data.get("scores", [])
        scored = {
            item["criterion"]: item
            for item in raw_scores
            if isinstance(item, dict) and "criterion" in item
        }

        results = []
        for rubric in rubrics:
            item = scored.get(rubric.name, {})
            raw_score = item.get("score", 3)
            try:
                score = max(1, min(5, int(raw_score)))
            except (TypeError, ValueError):
                score = 3

            results.append(CriterionScore(
                criterion=rubric.name,
                score=score,
                explanation=item.get("explanation", "No explanation provided."),
                weight=rubric.weight,
            ))

        return results
