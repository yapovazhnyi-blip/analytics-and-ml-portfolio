"""Anthropic backend — wraps the existing direct API calls."""

from __future__ import annotations

from typing import Optional

import httpx

from llm.base import LLMBackend, LLMResponse


class AnthropicBackend(LLMBackend):
    """
    Calls the Anthropic API directly using httpx.

    This is the default backend and the only one that supports Anthropic's
    full feature set including prompt caching, extended thinking, and
    native tool_use (which the agent system relies on).
    """

    BASE_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, api_key: str = "", model: str = "claude-haiku-4-5-20251001"):
        self._api_key = api_key
        self._model   = model

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] = None,
        max_tokens: int = 2000,
        temperature: float = 0,
    ) -> LLMResponse:
        if not self._api_key:
            return LLMResponse(
                content="",
                model=self._model,
                raw={"error": "No Anthropic API key configured."},
            )

        body: dict = {
            "model":       self._model,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "messages":    messages,
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = tools

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                self.BASE_URL,
                headers={
                    "x-api-key":         self._api_key,
                    "anthropic-version": self.API_VERSION,
                    "content-type":      "application/json",
                },
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        # Extract text and tool_use blocks
        text_parts  = []
        tool_calls  = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block["text"])
            elif block.get("type") == "tool_use":
                tool_calls.append(block)

        usage = data.get("usage", {})
        return LLMResponse(
            content="".join(text_parts),
            model=data.get("model", self._model),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            tool_calls=tool_calls,
            raw=data,
        )
