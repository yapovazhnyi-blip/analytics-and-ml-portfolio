"""
Claude data quality advisor for Crucible Phase 3.

Feeds the profiling report summary to the Claude API and returns
structured, actionable suggestions the UI can surface alongside
the profiling results.

Design:
  - Uses the profiling report's to_advisor_prompt() for the data summary
  - Response is streamed and parsed into typed AdvisorSuggestion objects
  - Falls back gracefully when ANTHROPIC_API_KEY is not configured
  - Rate-limited to one call per profile run — not called automatically
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from config import settings


@dataclass
class AdvisorSuggestion:
    category: str           # "missingness" | "leakage" | "imbalance" | "correlation" | "general"
    severity: str           # "high" | "medium" | "low" | "info"
    title: str
    explanation: str
    action: str             # concrete next step
    column: Optional[str] = None


@dataclass
class AdvisorResponse:
    suggestions: list[AdvisorSuggestion] = field(default_factory=list)
    raw_text: str = ""
    model: str = ""
    used_tokens: int = 0
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


SYSTEM_PROMPT = """You are a senior data scientist reviewing a dataset profiling report for an ML experiment.

Your job is to provide specific, actionable suggestions based on the profiling findings.
Each suggestion must be grounded in the data — no generic advice.

Respond ONLY with a JSON array of suggestion objects. No preamble, no markdown, no explanation outside the JSON.

Each object must have exactly these fields:
{
  "category": one of: missingness | leakage | imbalance | correlation | general,
  "severity": one of: high | medium | low | info,
  "title": short title (under 10 words),
  "explanation": what the issue is and why it matters (2-3 sentences),
  "action": concrete next step the user should take (1-2 sentences),
  "column": column name if specific to one column, else null
}

Rules:
- Only flag what is actually present in the report. Do not invent issues.
- Prioritise HIGH severity findings first.
- Maximum 6 suggestions. Focus on the most impactful.
- "action" must be specific: name the technique, the column, the threshold.
- If the data looks clean, say so with a single info-level suggestion."""


async def get_advisor_suggestions(
    profile_prompt: str,
    api_key: str = "",
) -> AdvisorResponse:
    """
    Calls the Claude API with the profile summary and returns suggestions.

    api_key: resolved by the caller via llm.resolver.resolve_api_key().
    Falls back to settings.anthropic_api_key if empty.
    """
    resolved_key = api_key or settings.anthropic_api_key or ""
    if not resolved_key:
        return AdvisorResponse(
            error="No API key configured. Set ANTHROPIC_API_KEY in .env or add your own key via Settings → API Keys."
        )

    # Cap prompt length to control API costs and prevent context exhaustion.
    # 8,000 characters covers any realistic profiling report.
    MAX_PROMPT_CHARS = 8_000
    safe_prompt = profile_prompt[:MAX_PROMPT_CHARS]
    if len(profile_prompt) > MAX_PROMPT_CHARS:
        safe_prompt += "\n\n[Profile truncated — dataset has many columns]"

    try:
        import httpx

        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1500,
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Here is the profiling report for a dataset I'm about to use for ML training. "
                        "Provide your suggestions:\n\n"
                        + safe_prompt
                    ),
                }
            ],
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": resolved_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        raw_text = "".join(
            block["text"] for block in data.get("content", [])
            if block.get("type") == "text"
        )
        model = data.get("model", "")
        used_tokens = data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get("output_tokens", 0)

        suggestions = _parse_suggestions(raw_text)

        return AdvisorResponse(
            suggestions=suggestions,
            raw_text=raw_text,
            model=model,
            used_tokens=used_tokens,
        )

    except Exception as exc:
        return AdvisorResponse(error=f"Advisor call failed: {exc}")


def _parse_suggestions(raw: str) -> list[AdvisorSuggestion]:
    """
    Parses the JSON array from Claude's response.
    Strips any accidental markdown fences before parsing.
    """
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        items = json.loads(clean)
        if not isinstance(items, list):
            return []

        suggestions = []
        for item in items[:6]:
            if not isinstance(item, dict):
                continue
            suggestions.append(AdvisorSuggestion(
                category=str(item.get("category", "general")),
                severity=str(item.get("severity", "info")),
                title=str(item.get("title", "Suggestion")),
                explanation=str(item.get("explanation", "")),
                action=str(item.get("action", "")),
                column=item.get("column"),
            ))
        return suggestions
    except (json.JSONDecodeError, Exception):
        return []
