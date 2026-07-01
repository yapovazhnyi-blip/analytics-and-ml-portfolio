"""
AWS Bedrock backend — calls Claude through AWS Bedrock using boto3.

WHY BEDROCK INSTEAD OF DIRECT ANTHROPIC
-----------------------------------------
Enterprises that have AWS Enterprise Agreements often route all AI API calls
through AWS Bedrock for:

  Centralised billing  — all AI spend appears in a single AWS bill alongside
    compute, storage, and data transfer. Finance teams can see the full cloud
    spend in one place.

  IAM-based access control — instead of distributing API keys, access is
    controlled via IAM roles and policies. A developer's EC2 instance or
    ECS task gets Bedrock access via its instance profile — no key rotation.

  Data residency — Bedrock regional endpoints guarantee that data does not
    leave a specific geographic region. Required for EU GDPR compliance,
    Swiss banking, and Australian health data regulations.

  VPC endpoints — Bedrock can be accessed from within a VPC without
    internet egress, satisfying PCI-DSS and SOC 2 network isolation requirements.

AUTHENTICATION
--------------
Bedrock uses the standard boto3 credentials chain — no API key needed:
  1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
  2. IAM instance profile (ECS task role, EC2 instance role)
  3. AWS config file (~/.aws/credentials)
  4. IAM role via Web Identity Token (EKS with IRSA)

MESSAGE FORMAT TRANSLATION
---------------------------
Bedrock uses a different API structure than Anthropic's direct API.
The Bedrock InvokeModel API uses the same JSON body as the Anthropic API
but via a different endpoint and signed with AWS Signature V4.

The newer Bedrock Converse API (used here) provides a unified interface
that accepts messages in a format similar to the direct Anthropic API.
"""

from __future__ import annotations

import json
from typing import Optional

from llm.base import LLMBackend, LLMResponse


class BedrockBackend(LLMBackend):
    """
    Calls Claude via AWS Bedrock's Converse API.

    Uses boto3 for authentication (IAM credentials chain).
    No API key required — authentication is via AWS credentials.

    The Converse API normalises the message format across all Bedrock-hosted
    models, making it easier to swap between Claude, Llama, and Titan.
    """

    def __init__(
        self,
        model_id: str = "anthropic.claude-haiku-4-5-20251001-v1:0",
        region: str = "us-east-1",
    ):
        self._model_id = model_id
        self._region   = region

    @property
    def model(self) -> str:
        return self._model_id

    def _make_client(self):
        """Isolated for mocking in tests."""
        import boto3
        return boto3.client("bedrock-runtime", region_name=self._region)

    async def complete(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] = None,
        max_tokens: int = 2000,
        temperature: float = 0,
    ) -> LLMResponse:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._complete_sync, messages, system, tools, max_tokens, temperature
        )

    def _complete_sync(
        self,
        messages: list[dict],
        system: str,
        tools: Optional[list[dict]],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Synchronous Bedrock API call — runs in thread pool."""
        try:
            client = self._make_client()

            # Convert Anthropic message format to Bedrock Converse format
            # Bedrock Converse uses {"role": "user", "content": [{"text": "..."}]}
            bedrock_messages = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    bedrock_content = [{"text": content}]
                elif isinstance(content, list):
                    # Handle tool results and text blocks
                    bedrock_content = []
                    for block in content:
                        if block.get("type") == "text":
                            bedrock_content.append({"text": block["text"]})
                        elif block.get("type") == "tool_result":
                            bedrock_content.append({
                                "toolResult": {
                                    "toolUseId": block.get("tool_use_id", ""),
                                    "content":   [{"text": str(block.get("content", ""))}],
                                }
                            })
                        elif block.get("type") == "tool_use":
                            bedrock_content.append({
                                "toolUse": {
                                    "toolUseId": block.get("id", ""),
                                    "name":      block.get("name", ""),
                                    "input":     block.get("input", {}),
                                }
                            })
                else:
                    bedrock_content = [{"text": str(content)}]

                bedrock_messages.append({
                    "role":    msg["role"],
                    "content": bedrock_content,
                })

            kwargs: dict = {
                "modelId":            self._model_id,
                "messages":           bedrock_messages,
                "inferenceConfig": {
                    "maxTokens":   max_tokens,
                    "temperature": temperature,
                },
            }

            if system:
                kwargs["system"] = [{"text": system}]

            if tools:
                kwargs["toolConfig"] = {
                    "tools": [
                        {
                            "toolSpec": {
                                "name":        t["name"],
                                "description": t.get("description", ""),
                                "inputSchema": {"json": t.get("input_schema", {})},
                            }
                        }
                        for t in tools
                    ]
                }

            response = client.converse(**kwargs)

            # Parse response
            output_msg = response.get("output", {}).get("message", {})
            content_blocks = output_msg.get("content", [])

            text_parts = []
            tool_calls = []
            for block in content_blocks:
                if "text" in block:
                    text_parts.append(block["text"])
                elif "toolUse" in block:
                    # Convert Bedrock toolUse → Anthropic tool_use format
                    tu = block["toolUse"]
                    tool_calls.append({
                        "type":  "tool_use",
                        "id":    tu.get("toolUseId", ""),
                        "name":  tu.get("name", ""),
                        "input": tu.get("input", {}),
                    })

            usage = response.get("usage", {})
            return LLMResponse(
                content="".join(text_parts),
                model=self._model_id,
                input_tokens=usage.get("inputTokens", 0),
                output_tokens=usage.get("outputTokens", 0),
                tool_calls=tool_calls,
                raw=response,
            )

        except Exception as exc:
            return LLMResponse(
                content="",
                model=self._model_id,
                raw={"error": str(exc)},
            )
