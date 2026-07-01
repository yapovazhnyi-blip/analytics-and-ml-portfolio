"""
RAG Pipeline — orchestrates the full ingest and query workflows.

This is the only class the router needs to interact with. It wires
together chunker → embedder → vector_store for ingest, and
embedder → retriever → generator for query.

INGEST FLOW
-----------
1. extract_text(file_path) — pulls plain text from PDF/DOCX/TXT
2. chunk_text(text, strategy) — splits into overlapping chunks
3. embedder.encode([c.text for c in chunks]) — batch-embeds all chunks
4. vector_store.add_chunks(...) — persists chunks + embeddings to ChromaDB

QUERY FLOW
----------
1. retriever.retrieve(query, k) — hybrid BM25 + dense search with RRF fusion
2. generator.generate(query, chunks) — Claude synthesises answer from context
3. Return GeneratorResponse with answer + citations

CONFIGURATION
-------------
All defaults are tunable at construction time. The router uses these
defaults for the standard /rag/query endpoint. Advanced users can
override via query parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rag.chunker import chunk_file, TextChunk
from rag.embedder import Embedder, get_embedder
from rag.vector_store import VectorStore
from rag.retriever import Retriever, RetrievedChunk
from rag.generator import Generator, GeneratorResponse


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class RAGConfig:
    chunk_strategy: str    = "paragraph"   # paragraph | fixed | sentence
    max_chars: int         = 1500          # max characters per chunk
    overlap_chars: int     = 150           # overlap between adjacent chunks
    retrieval_k: int       = 5             # chunks to retrieve per query
    candidate_multiplier: int = 4          # k * multiplier candidates before RRF
    use_reranker: bool     = False         # cross-encoder reranking (requires ST)
    generator_model: str   = Generator.DEFAULT_MODEL
    max_answer_tokens: int = 512


@dataclass
class IngestResult:
    """Result of ingesting one document into the RAG pipeline."""
    document_id: str
    chunk_count: int
    strategy: str
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass
class QueryResult:
    """Full result of one RAG query."""
    query: str
    answer: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    citations: list             = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline for Crucible.

    Usage:
        pipeline = RAGPipeline(vector_store_dir="/data/rag")

        # Ingest a document
        result = await pipeline.ingest("report.pdf", document_id="q3-report")

        # Query across all documents
        answer = await pipeline.query("What were the key findings?")

        # Query within specific documents
        answer = await pipeline.query("...", document_ids=["q3-report"])
    """

    def __init__(
        self,
        vector_store_dir: str,
        config: Optional[RAGConfig] = None,
        embedder: Optional[Embedder] = None,
    ):
        self.config = config or RAGConfig()
        self._embedder = embedder or get_embedder()
        self._vector_store = VectorStore(vector_store_dir)
        self._retriever = Retriever(
            vector_store=self._vector_store,
            embedder=self._embedder,
            use_reranker=self.config.use_reranker,
            candidate_multiplier=self.config.candidate_multiplier,
        )
        self._generator = Generator(model=self.config.generator_model)

    # ── Ingest ────────────────────────────────────────────────────────────────

    async def ingest(
        self,
        file_path: str,
        document_id: str,
    ) -> IngestResult:
        """
        Ingests a document: extract → chunk → embed → store.

        This is the async wrapper for use in FastAPI route handlers.
        The heavy work (embedding) runs in a thread via run_in_executor
        to avoid blocking the event loop.
        """
        import asyncio

        try:
            # Chunking is fast (pure Python string processing) — run inline
            chunks = chunk_file(
                file_path,
                strategy=self.config.chunk_strategy,
                max_chars=self.config.max_chars,
                overlap_chars=self.config.overlap_chars,
            )

            if not chunks:
                return IngestResult(
                    document_id=document_id,
                    chunk_count=0,
                    strategy=self.config.chunk_strategy,
                    error="No text could be extracted from the file.",
                )

            # Embedding is CPU-bound (ONNX inference) — run in thread pool
            texts = [c.text for c in chunks]
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, self._embedder.encode, texts
            )

            # Add document_id to each chunk's metadata before storing
            metadatas = []
            for c in chunks:
                meta = dict(c.metadata)
                meta["document_id"] = document_id
                metadatas.append(meta)

            # Storage is fast — run inline
            self._vector_store.add_chunks(
                document_id=document_id,
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            return IngestResult(
                document_id=document_id,
                chunk_count=len(chunks),
                strategy=self.config.chunk_strategy,
            )

        except Exception as exc:
            return IngestResult(
                document_id=document_id,
                chunk_count=0,
                strategy=self.config.chunk_strategy,
                error=str(exc),
            )

    # ── Query ─────────────────────────────────────────────────────────────────

    async def query(
        self,
        question: str,
        k: Optional[int] = None,
        document_ids: Optional[list[str]] = None,
    ) -> QueryResult:
        """
        Queries the RAG pipeline: retrieve → generate.

        Args:
            question:     The user's question in plain text.
            k:            Number of chunks to retrieve (default from config).
            document_ids: Restrict to specific documents (None = all).

        Returns:
            QueryResult with answer, citations, and retrieved chunks.
        """
        import asyncio

        k = k or self.config.retrieval_k

        try:
            # Retrieve relevant chunks (embedding + BM25 + RRF)
            # Embedding call is CPU-bound — run in thread pool
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(
                None,
                lambda: self._retriever.retrieve(
                    query=question,
                    k=k,
                    document_ids=document_ids,
                ),
            )

            # Generate answer (API call — already async)
            gen_response = await self._generator.generate(
                query=question,
                chunks=chunks,
                max_tokens=self.config.max_answer_tokens,
            )

            if not gen_response.succeeded:
                return QueryResult(
                    query=question,
                    answer="",
                    chunks=chunks,
                    error=gen_response.error,
                )

            return QueryResult(
                query=question,
                answer=gen_response.answer,
                chunks=chunks,
                citations=gen_response.citations,
                model=gen_response.model,
                input_tokens=gen_response.input_tokens,
                output_tokens=gen_response.output_tokens,
            )

        except Exception as exc:
            return QueryResult(
                query=question,
                answer="",
                error=str(exc),
            )

    # ── Document management ───────────────────────────────────────────────────

    def delete_document(self, document_id: str) -> bool:
        """Removes all chunks for a document. Returns True if found."""
        return self._vector_store.delete_document(document_id)

    def list_documents(self) -> list[str]:
        """Returns document_ids of all indexed documents."""
        return self._vector_store.list_documents()

    def chunk_count(self, document_id: str) -> int:
        """Returns how many chunks are stored for a document."""
        return self._vector_store.count_chunks(document_id)

    def warm_up(self) -> None:
        """Pre-loads the embedding model to avoid cold-start latency."""
        self._embedder.warm_up()
