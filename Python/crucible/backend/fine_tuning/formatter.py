"""
Dataset formatting — converts raw training data into tokenised format
for SFT (Supervised Fine-Tuning).

TWO SUPPORTED FORMATS
---------------------

ALPACA FORMAT
  Used by Stanford Alpaca (2023) and many subsequent instruction datasets.
  Each sample has three fields:
    instruction: the task description
    input:       optional additional context (empty string if not needed)
    output:      the ideal response

  Example:
    {"instruction": "Translate to French.", "input": "Hello world", "output": "Bonjour le monde"}

  Formatted as:
    ### Instruction:
    Translate to French.

    ### Input:
    Hello world

    ### Response:
    Bonjour le monde

SHAREGPT FORMAT
  Used by ShareGPT and Vicuna datasets. Models multi-turn conversations.
  Each sample has a "conversations" list with alternating human/gpt turns.

  Example:
    {"conversations": [
      {"from": "human", "value": "What is Python?"},
      {"from": "gpt",   "value": "Python is a programming language..."}
    ]}

  Formatted as standard chat format (matches most modern model tokenisers):
    <|user|>What is Python?<|end|>
    <|assistant|>Python is a programming language...<|end|>

WHY FORMAT MATTERS
------------------
The model has been pre-trained on specific prompt patterns. If you fine-tune
with a different pattern, the model learns to respond to the new pattern but
the pre-trained chat template is broken. Always use the base model's native
chat template when available (tokenizer.apply_chat_template).

When the tokeniser doesn't have a chat template (older models), use the
Alpaca template — it's simple and the model learns it quickly.
"""

from __future__ import annotations

import json
from typing import Union


# ── Alpaca template ────────────────────────────────────────────────────────────

_ALPACA_WITH_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

_ALPACA_NO_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


def format_alpaca(sample: dict) -> str:
    """
    Formats one Alpaca-format sample as a plain text string.

    Handles the two common Alpaca variants:
    - with input field: instruction + input → response
    - without input:    instruction alone → response
    """
    instruction = sample.get("instruction", "").strip()
    inp         = sample.get("input", "").strip()
    output      = sample.get("output", "").strip()

    if not instruction:
        raise ValueError(f"Alpaca sample missing 'instruction' field: {sample!r}")
    if not output:
        raise ValueError(f"Alpaca sample missing 'output' field: {sample!r}")

    if inp:
        return _ALPACA_WITH_INPUT.format(instruction=instruction, input=inp, output=output)
    else:
        return _ALPACA_NO_INPUT.format(instruction=instruction, output=output)


# ── ShareGPT template ──────────────────────────────────────────────────────────

def format_sharegpt(sample: dict, tokenizer=None) -> str:
    """
    Formats one ShareGPT-format sample.

    If a tokenizer is provided and has a chat_template, uses
    tokenizer.apply_chat_template() for the model's native format.
    Otherwise falls back to a simple human/assistant template.
    """
    conversations = sample.get("conversations", [])
    if not conversations:
        raise ValueError(f"ShareGPT sample missing 'conversations': {sample!r}")

    # Normalise role names (ShareGPT uses 'human'/'gpt', some use 'user'/'assistant')
    _role_map = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}
    messages = []
    for turn in conversations:
        role = _role_map.get(turn.get("from", ""), "user")
        content = turn.get("value", "")
        messages.append({"role": role, "content": content})

    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass  # fall through to generic template

    # Generic multi-turn template
    parts = []
    for msg in messages:
        if msg["role"] == "user":
            parts.append(f"<|user|>{msg['content']}<|end|>")
        else:
            parts.append(f"<|assistant|>{msg['content']}<|end|>")
    return "\n".join(parts)


# ── Unified formatter ──────────────────────────────────────────────────────────

def format_sample(sample: dict, dataset_format: str, tokenizer=None) -> str:
    """
    Formats one training sample to a string based on the dataset format.

    Args:
        sample:         Dict containing the training example.
        dataset_format: "alpaca" or "sharegpt".
        tokenizer:      Optional tokenizer for native chat template.

    Returns:
        Plain text string ready for tokenisation.
    """
    if dataset_format == "alpaca":
        return format_alpaca(sample)
    elif dataset_format == "sharegpt":
        return format_sharegpt(sample, tokenizer)
    else:
        raise ValueError(f"Unknown dataset_format: {dataset_format!r}. Use 'alpaca' or 'sharegpt'")


def validate_samples(
    samples: list[dict],
    dataset_format: str,
    max_errors: int = 5,
) -> list[str]:
    """
    Validates a list of training samples.

    Returns a list of error messages (empty = all valid).
    Stops after max_errors to avoid flooding logs on large bad datasets.
    """
    errors = []
    for i, sample in enumerate(samples):
        if len(errors) >= max_errors:
            errors.append(f"... and more errors (stopped at {max_errors})")
            break
        try:
            format_sample(sample, dataset_format)
        except (ValueError, KeyError) as exc:
            errors.append(f"Sample {i}: {exc}")
    return errors


def load_jsonl(path: str) -> list[dict]:
    """Loads a JSONL file where each line is one training sample."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples
