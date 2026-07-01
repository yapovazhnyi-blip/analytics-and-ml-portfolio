"""
OpenAI-compatible backend — covers Groq, Ollama, OpenRouter, Together AI.

These providers all expose an endpoint that speaks the OpenAI Chat Completions
API format. One backend implementation covers all of them.

FREE/LOCAL OPTIONS
------------------
  Ollama (local, zero cost)
    Download: https://ollama.com
    Run:      ollama serve && ollama pull llama3
    URL:      http://localhost:11434/v1
    Models:   llama3, mistral, gemma, codellama, qwen, phi3
    Use:      Local development and demos without any API spend.
              Crucible's agent and advisor features work with local models —
              quality is lower than Claude but sufficient for testing.

  Groq (cloud, free tier available)
    URL:      https://api.groq.com/openai/v1
    Models:   llama-3.3-70b-versatile, mixtral-8x7b
    Cost:     Free tier (6,000 tokens/minute), paid plans available
    Use:      Fast inference for testing — Groq's LPU hardware is 10-20× faster
              than GPU-based inference for the same model.

  OpenRouter (cloud, many free models)
    URL:      https://openrouter.ai/api/v1
    Models:   100+ models including free-tier Llama, Gemma, Mistral
    Cost:     Pay-per-use, many models have a free tier
    Use:      Model comparison and experimentation.

MESSAGE FORMAT DIFFERENCES FROM ANTHROPIC
------------------------------------------
Anthropic format (what Crucible uses internally):
  {"role": "user",      "content": "Hello"}
  {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]}
  {"role": "assistant", "content": [{"type": "tool_use", "id": "...", ...}]}
  {"role": "user",      "content": [{"type": "tool_result", ...}]}

OpenAI format:
  {"role": "user",      "content": "Hello"}
  {"role": "assistant", "content": "Hi", "tool_calls": [...]}
  {"role": "tool",      "content": "...", "tool_call_id": "..."}

This backend translates between the two formats transparently.
Tool support (function calling) is available on Groq and most OpenRouter models.
Ollama supports function calling on llama3 and phi3.
"""

from __future__ import annotations

import json
from typing import Optional

import httpx

from llm.base import LLMBackend, LLMResponse


class OpenAICompatBackend(LLMBackend):
    """
    Backend for any OpenAI Chat Completions-compatible provider.

    Translates Anthropic-format messages to OpenAI format, calls the endpoint,
    and translates the response back to LLMResponse.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 60.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._model    = model
        self._api_key  = api_key
        self._timeout  = timeout

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
        # Convert Anthropic messages → OpenAI format
        openai_messages = self._to_openai_messages(messages, system)

        body: dict = {
            "model":       self._model,
            "messages":    openai_messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }

        if tools:
            body["tools"] = self._to_openai_tools(tools)
            body["tool_choice"] = "auto"

        headers = {
            "content-type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        return self._parse_response(data)

    def _to_openai_messages(self, messages: list[dict], system: str) -> list[dict]:
        """Converts Anthropic-format messages to OpenAI format."""
        result = []
        if system:
            result.append({"role": "system", "content": system})

        for msg in messages:
            role    = msg["role"]
            content = msg.get("content", "")

            if isinstance(content, str):
                result.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Anthropic content blocks → OpenAI format
                text_parts = []
                tool_calls = []
                tool_results = []

                for block in content:
                    btype = block.get("type", "")
                    if btype == "text":
                        text_parts.append(block["text"])
                    elif btype == "tool_use":
                        tool_calls.append({
                            "id":       block["id"],
                            "type":     "function",
                            "function": {
                                "name":      block["name"],
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })
                    elif btype == "tool_result":
                        tool_results.append({
                            "role":         "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content":      str(block.get("content", "")),
                        })

                if tool_calls:
                    result.append({
                        "role":       "assistant",
                        "content":    " ".join(text_parts),
                        "tool_calls": tool_calls,
                    })
                for tr in tool_results:
                    result.append(tr)
                if not tool_calls and not tool_results and text_parts:
                    result.append({"role": role, "content": " ".join(text_parts)})
            else:
                result.append({"role": role, "content": str(content)})

        return result

    def _to_openai_tools(self, anthropic_tools: list[dict]) -> list[dict]:
        """Converts Anthropic tool definitions to OpenAI function format."""
        return [
            {
                "type": "function",
                "function": {
                    "name":        t["name"],
                    "description": t.get("description", ""),
                    "parameters":  t.get("input_schema", {}),
                },
            }
            for t in anthropic_tools
        ]

    def _parse_response(self, data: dict) -> LLMResponse:
        """Parses an OpenAI-format response into LLMResponse."""
        choice  = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""

        # Convert OpenAI tool_calls → Anthropic tool_use format
        tool_calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({
                "type":  "tool_use",
                "id":    tc.get("id", ""),
                "name":  fn.get("name", ""),
                "input": args,
            })

        usage = data.get("usage", {})
        return LLMResponse(
            content=content,
            model=data.get("model", self._model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            tool_calls=tool_calls,
            raw=data,
        )
