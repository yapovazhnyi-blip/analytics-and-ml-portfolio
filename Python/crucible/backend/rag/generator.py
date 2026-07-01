"""
RAG Generator — answer synthesis using Claude with retrieved context.

HOW RAG GENERATION WORKS
-------------------------
Standard LLM generation: model answers from training memory alone.
RAG generation: model reads retrieved chunks as context, then answers.

The retrieved chunks are injected into the system prompt as "context you
may use to answer the question". The model reads them the same way a
human reads reference documents before answering. This grounds the answer
in the actual content of your documents rather than the model's (possibly
outdated or hallucinated) training memory.

PROMPT STRUCTURE
----------------
System prompt establishes the model's role and constraints:
  - Answer ONLY from the provided context
  - If context is insufficient, say so explicitly
  - Cite the source document for each claim

User message:
  [CONTEXT]
  Source: document_name, chunk 3:
  {chunk text}
  ---
  Source: document_name, chunk 7:
  {chunk text}
  [/CONTEXT]

  Question: {user's question}

This structure makes it easy for the model to distinguish context from
question, and produces answers that are grounded and attributable.

ATTRIBUTION / CITATIONS
-----------------------
The generator returns both the answer text and a list of source citations
(document names and chunk indices). The UI displays these alongside the
answer so users can verify claims.

TEMPERATURE
-----------
Temperature=0 for RAG generation. The model's job here is to read and
summarise, not to be creative. Deterministic output is more trustworthy
for factual question-answering than a creative interpretation.

COST CONTROL
------------
Input tokens = system prompt + context chunks + question.
Context chunks dominate: k=5 chunks × 250 words × ~1.3 tokens/word ≈ 1,625 tokens.
Full RAG call ≈ 2,000–3,000 input tokens + 300–500 output tokens.
At claude-haiku pricing this is approximately $0.001 per query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from config import settings


# ── Response types ────────────────────────────────────────────────────────────

@dataclass
class Citation:
    """A reference to a source chunk used to generate the answer."""
    document_id: str
    chunk_index: int
    source_name: str    # human-readable filename


@dataclass
class GeneratorResponse:
    """The complete output of one RAG query."""
    answer: str
    citations: list[Citation] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def cost_estimate_usd(self) -> float:
        """
        Rough cost estimate using Claude Haiku pricing.
        Actual cost depends on the model used — this is a lower bound.
        """
        # claude-haiku-4-5 pricing (approximate): $0.25/M input, $1.25/M output
        return (self.input_tokens * 0.00000025) + (self.output_tokens * 0.00000125)


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a helpful assistant answering questions about documents.

Rules:
- Answer ONLY using information from the [CONTEXT] provided.
- If the context does not contain enough information to answer, say clearly: \
"The provided documents do not contain enough information to answer this question."
- Never make up facts not present in the context.
- Be concise and direct. Do not pad your answer.
- When making a specific claim, you may note which source it came from.

Format:
- Plain prose for explanations.
- Use bullet points only for lists or step-by-step instructions.
- Do not repeat the question back to the user."""


# ── Generator ─────────────────────────────────────────────────────────────────

class Generator:
    """
    Calls the Claude API with retrieved chunks as context to generate answers.

    Requires ANTHROPIC_API_KEY in the environment.
    Falls back gracefully when the key is absent.
    """

    # Use Haiku by default — fast and cheap for RAG which is input-heavy.
    # Swap for claude-sonnet-4-6 when higher reasoning quality is needed.
    DEFAULT_MODEL = "claude-haiku-4-5-20251001"
    MAX_CONTEXT_CHARS = 12_000   # ~3,000 tokens — well within Haiku's 200k context

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    async def generate(
        self,
        query: str,
        chunks: list,
        max_tokens: int = 512,
        api_key: str = "",
    ) -> GeneratorResponse:
        """Generates a grounded answer. Uses api_key if provided, else server key."""
        resolved_key = api_key or settings.anthropic_api_key or ""
        if not resolved_key:
            return GeneratorResponse(
                answer="",
                error="No API key configured. Add your key via Settings → API Keys.",
            )

        if not chunks:
            return GeneratorResponse(
                answer="The provided documents do not contain enough information to answer this question.",
                citations=[],
            )

        # Build context block from retrieved chunks
        context_block, citations = self._build_context(chunks)

        user_message = (
            f"[CONTEXT]\n{context_block}\n[/CONTEXT]\n\n"
            f"Question: {query}"
        )

        try:
            import httpx

            payload = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 0,
                "system": _SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_message}],
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

            answer = "".join(
                block["text"]
                for block in data.get("content", [])
                if block.get("type") == "text"
            )
            usage = data.get("usage", {})

            return GeneratorResponse(
                answer=answer.strip(),
                citations=citations,
                model=data.get("model", self.model),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            )

        except Exception as exc:
            return GeneratorResponse(
                answer="",
                error=f"Generation failed: {exc}",
            )

    def _build_context(
        self, chunks: list
    ) -> tuple[str, list[Citation]]:
        """
        Formats retrieved chunks into a numbered context block.

        Truncates at MAX_CONTEXT_CHARS to stay within token limits.
        Returns (context_text, citations).
        """
        parts = []
        citations = []
        total_chars = 0

        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", chunk.document_id)
            header = f"Source {i}: {source} (chunk {chunk.chunk_index})"
            body = chunk.text.strip()

            entry = f"{header}\n{body}\n---"
            if total_chars + len(entry) > self.MAX_CONTEXT_CHARS:
                break

            parts.append(entry)
            total_chars += len(entry)
            citations.append(Citation(
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                source_name=source,
            ))

        return "\n".join(parts), citations
