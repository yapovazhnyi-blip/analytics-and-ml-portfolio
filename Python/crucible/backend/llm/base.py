"""
LLM Backend Abstraction.

WHY ABSTRACT THE LLM CALLS
----------------------------
Crucible currently calls the Anthropic API directly in 6+ places. This means:
  - Switching provider requires editing 6 files
  - Testing requires mocking the exact Anthropic request format
  - Cost comparison across providers requires running each provider's code path

Abstracting into a backend interface fixes all three:
  - One place to switch provider (config or environment variable)
  - Tests mock the backend, not the raw httpx call
  - New providers (Bedrock, Ollama, Groq) require adding one file

PROVIDER LANDSCAPE
------------------
  Anthropic (direct)
    Models: Claude Haiku 4.5, Sonnet 4.6, Opus 4.x
    Cost:   $1/$5 per MTok (Haiku), $3/$15 (Sonnet), $15/$75 (Opus)
    Use:    Production deployments with Anthropic's trust and safety stack

  AWS Bedrock
    Same Claude models, billed through AWS
    Cost:   Same token rates + 10% regional surcharge for non-global endpoints
    Use:    Enterprises with AWS contracts, data residency requirements (EU),
            AWS-managed IAM instead of API keys
    Auth:   IAM roles (no API key — uses boto3 credentials chain)

  OpenAI-compatible (covers Groq, Ollama, OpenRouter, Together AI, Anyscale)
    Models: Llama 3, Mixtral, Gemma, local models via Ollama
    Cost:   Groq free tier, Ollama local (zero cost), OpenRouter pay-per-use
    Use:    Local development without API spend, testing, self-hosted deployments
    Note:   Message format differs from Anthropic — must translate tool_use

MESSAGE FORMAT DIFFERENCES
---------------------------
Anthropic uses its own request format:
  {"model": "...", "messages": [...], "tools": [...], "system": "..."}

OpenAI-compatible APIs use a different format:
  {"model": "...", "messages": [{"role": "system", ...}, ...]}

The OpenAICompatBackend translates between the two formats internally.

BACKEND RESOLUTION
------------------
resolve_backend(user) reads from settings:
  LLM_PROVIDER=anthropic       → AnthropicBackend (default)
  LLM_PROVIDER=bedrock         → BedrockBackend
  LLM_PROVIDER=openai_compat   → OpenAICompatBackend
  LLM_PROVIDER=ollama          → OpenAICompatBackend (preset Ollama config)
  LLM_PROVIDER=groq            → OpenAICompatBackend (preset Groq config)

Per-user BYOK keys override the provider's key for all providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """Normalised response from any LLM backend."""
    content: str                      # text content of the response
    model: str                        # model ID that was used
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list[dict] = None     # tool_use blocks (Anthropic format)
    raw: dict = None                  # full raw response (for debugging)

    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []
        if self.raw is None:
            self.raw = {}


class LLMBackend(ABC):
    """
    Abstract base class for LLM provider backends.

    All backends expose a single async method: complete().
    The caller always uses Anthropic-style messages — backends translate
    internally if the provider uses a different format.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] = None,
        max_tokens: int = 2000,
        temperature: float = 0,
    ) -> LLMResponse:
        """
        Sends a completion request to the provider.

        Args:
            messages:   Anthropic-format messages [{"role": "user", "content": "..."}].
            system:     System prompt (Anthropic-style string).
            tools:      Anthropic-format tool definitions (for tool-use/agents).
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature (0 = deterministic).

        Returns:
            LLMResponse with normalised content and token counts.
        """
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        """Returns the model identifier being used."""
        ...


def resolve_backend(api_key: str = "", user=None) -> LLMBackend:
    """
    Factory — returns the appropriate backend based on settings.

    Resolution order:
      1. settings.llm_provider determines the backend type
      2. api_key (BYOK user key) overrides the provider's own key
      3. settings.anthropic_api_key / AWS credentials are the fallback

    Provider selection:
      LLM_PROVIDER=anthropic      → AnthropicBackend     (default)
      LLM_PROVIDER=bedrock        → BedrockBackend
      LLM_PROVIDER=openai_compat  → OpenAICompatBackend  (requires LLM_BASE_URL)
      LLM_PROVIDER=ollama         → OpenAICompatBackend  (localhost:11434)
      LLM_PROVIDER=groq           → OpenAICompatBackend  (api.groq.com)
    """
    from config import settings
    provider = (getattr(settings, "llm_provider", "") or "anthropic").lower()

    if provider == "bedrock":
        from llm.bedrock import BedrockBackend
        return BedrockBackend(
            model_id=getattr(settings, "llm_model", "anthropic.claude-haiku-4-5-20251001-v1:0"),
            region=getattr(settings, "aws_region", "us-east-1"),
        )

    if provider in ("openai_compat", "ollama", "groq", "openrouter", "together"):
        from llm.openai_compat import OpenAICompatBackend

        presets = {
            "ollama":     ("http://localhost:11434/v1", "llama3"),
            "groq":       ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
            "openrouter": ("https://openrouter.ai/api/v1", "meta-llama/llama-3-8b-instruct"),
            "together":   ("https://api.together.xyz/v1", "meta-llama/Llama-3-8b-chat-hf"),
        }
        base_url, default_model = presets.get(provider, ("", ""))
        return OpenAICompatBackend(
            base_url=getattr(settings, "llm_base_url", "") or base_url,
            model=getattr(settings, "llm_model", "") or default_model,
            api_key=api_key or getattr(settings, "llm_api_key", "") or "",
        )

    # Default: Anthropic direct
    from llm.anthropic_backend import AnthropicBackend
    resolved_key = api_key or getattr(settings, "anthropic_api_key", "") or ""
    return AnthropicBackend(
        api_key=resolved_key,
        model=getattr(settings, "llm_model", "claude-haiku-4-5-20251001"),
    )
