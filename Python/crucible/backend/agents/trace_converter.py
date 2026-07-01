"""
Trace-to-Training-Data Converter.

Converts AgentTrace rows into the formats Crucible's fine-tuning studio
already accepts (see fine_tuning/formatter.py for the Alpaca/ShareGPT
loaders, and fine_tuning/dpo_trainer.py for the DPO sample format).

SFT CONVERSION
--------------
Each trace becomes one training example. The "instruction" is the goal,
and the "output" is a reconstructed reasoning trace showing the tool calls
and final answer — this teaches the fine-tuned model to reproduce the same
tool-use pattern for similar goals.

Alpaca format:
  {"instruction": goal, "input": "", "output": reasoning_trace_text}

ShareGPT format:
  {"conversations": [
      {"from": "human", "value": goal},
      {"from": "gpt", "value": reasoning_trace_text},
  ]}

DPO CONVERSION
--------------
DPO needs (prompt, chosen, rejected) triplets — two different responses to
the SAME prompt, one better than the other. Traces don't naturally come in
pairs, so traces_to_dpo_pairs() groups traces by goal similarity and pairs
the highest-quality-scored trace (chosen) against the lowest-quality-scored
trace (rejected) for each group.

Two traces are considered "the same goal" if their goal strings match after
lightly normalising whitespace and case — this catches near-duplicate runs
of the same benchmark goal without requiring exact string equality.

ONLY SUCCESSFUL, SCORED TRACES ARE USED
------------------------------------------
Traces with succeeded=False are excluded from SFT (we don't want to teach
the model to reproduce failures). Traces with quality_score=None (not yet
scored) are excluded from DPO pairing, since pairing requires comparing
scores.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass

from models.agent_trace import AgentTrace


def _normalise_goal(goal: str) -> str:
    """Normalises a goal string for grouping near-duplicate goals."""
    return re.sub(r"\s+", " ", goal.strip().lower())


def _reconstruct_reasoning_text(steps_json: str, final_answer: str) -> str:
    """
    Reconstructs a readable reasoning trace from the structured event list.

    Produces text like:
      "I'll check available datasets first.
       [Tool: list_datasets()] → Found 3 datasets: ...
       [Tool: run_profiling(dataset_id=1)] → Missingness: 2% ...
       Based on this analysis, ..."

    This is what the fine-tuned model learns to generate — a thinking-then-
    acting pattern that mirrors the original Claude-driven agent's behaviour.
    """
    try:
        events = json.loads(steps_json)
    except (json.JSONDecodeError, TypeError):
        events = []

    parts = []
    for event in events:
        etype = event.get("type", "")
        if etype == "thinking":
            parts.append(event.get("text", "").strip())
        elif etype == "tool_call":
            tool = event.get("tool", "")
            inp  = event.get("input", {})
            args = ", ".join(f"{k}={v!r}" for k, v in inp.items())
            parts.append(f"[Tool: {tool}({args})]")
        elif etype == "tool_result":
            result = str(event.get("result", ""))[:300]   # truncate long results
            parts.append(f"→ {result}")
        elif etype == "specialist":
            agent = event.get("agent", "")
            out   = str(event.get("output", ""))[:300]
            parts.append(f"[{agent}] {out}")
        elif etype == "supervisor":
            reasoning = event.get("reasoning", "")
            routing   = event.get("routing_to", "")
            parts.append(f"[Supervisor → {routing}] {reasoning}")

    if final_answer:
        parts.append(final_answer.strip())

    return "\n".join(p for p in parts if p)


# ── SFT conversion ──────────────────────────────────────────────────────────

def traces_to_alpaca(traces: list[AgentTrace]) -> list[dict]:
    """Converts successful traces to Alpaca-format SFT samples."""
    samples = []
    for t in traces:
        if not t.succeeded:
            continue
        reasoning = _reconstruct_reasoning_text(t.steps_json, t.final_answer)
        if not reasoning:
            continue
        samples.append({
            "instruction": t.goal,
            "input":       "",
            "output":      reasoning,
        })
    return samples


def traces_to_sharegpt(traces: list[AgentTrace]) -> list[dict]:
    """Converts successful traces to ShareGPT-format SFT samples."""
    samples = []
    for t in traces:
        if not t.succeeded:
            continue
        reasoning = _reconstruct_reasoning_text(t.steps_json, t.final_answer)
        if not reasoning:
            continue
        samples.append({
            "conversations": [
                {"from": "human", "value": t.goal},
                {"from": "gpt",   "value": reasoning},
            ]
        })
    return samples


# ── DPO conversion ──────────────────────────────────────────────────────────

@dataclass
class DPOPairingStats:
    n_groups: int
    n_pairs: int
    n_traces_used: int
    n_traces_skipped_unscored: int


def traces_to_dpo_pairs(
    traces: list[AgentTrace],
    min_score_gap: float = 0.15,
) -> tuple[list[dict], DPOPairingStats]:
    """
    Groups traces by normalised goal and pairs the best vs worst scored
    trace in each group into a DPO (prompt, chosen, rejected) sample.

    Args:
        traces: AgentTrace rows (any agent_type, any succeeded value —
                a failed trace can legitimately be the "rejected" example).
        min_score_gap: Minimum quality_score difference required to form
                a pair. Pairs with too small a gap are skipped — DPO learns
                nothing useful from two nearly-identical-quality responses,
                and a noisy score difference could even teach the wrong
                preference.

    Returns:
        (dpo_samples, stats) — dpo_samples is a list of
        {"prompt": ..., "chosen": ..., "rejected": ...} dicts ready for
        fine_tuning/dpo_trainer.py.
    """
    scored = [t for t in traces if t.quality_score is not None]
    skipped_unscored = len(traces) - len(scored)

    groups: dict[str, list[AgentTrace]] = defaultdict(list)
    for t in scored:
        groups[_normalise_goal(t.goal)].append(t)

    samples = []
    n_traces_used = 0

    for goal_key, group in groups.items():
        if len(group) < 2:
            continue   # need at least 2 traces of the same goal to form a pair

        group_sorted = sorted(group, key=lambda t: t.quality_score, reverse=True)
        best, worst = group_sorted[0], group_sorted[-1]

        gap = best.quality_score - worst.quality_score
        if gap < min_score_gap:
            continue   # not enough of a quality difference to be a useful signal

        chosen_text   = _reconstruct_reasoning_text(best.steps_json, best.final_answer)
        rejected_text = _reconstruct_reasoning_text(worst.steps_json, worst.final_answer)
        if not chosen_text or not rejected_text:
            continue

        samples.append({
            "prompt":   best.goal,    # use the original (un-normalised) goal text
            "chosen":   chosen_text,
            "rejected": rejected_text,
        })
        n_traces_used += 2

    stats = DPOPairingStats(
        n_groups=len(groups),
        n_pairs=len(samples),
        n_traces_used=n_traces_used,
        n_traces_skipped_unscored=skipped_unscored,
    )
    return samples, stats
