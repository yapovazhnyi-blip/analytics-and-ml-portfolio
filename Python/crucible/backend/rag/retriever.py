"""
RAG Retriever — hybrid BM25 + dense retrieval with Reciprocal Rank Fusion.

WHY HYBRID RETRIEVAL
--------------------
Pure semantic (dense) search: excellent at paraphrased queries, misses
exact keyword matches. A query for "RFC 7231" finds chunks about HTTP
semantics — correct topic, wrong document if you need the specific RFC.

Pure keyword (BM25) search: exact matches but misses paraphrased queries.
"How does HTTP caching work?" finds only chunks containing those exact words.

Hybrid retrieval combines both. For each query:
  1. Dense search retrieves top-K candidates by cosine similarity.
  2. BM25 retrieves top-K candidates by term-frequency score.
  3. Reciprocal Rank Fusion merges the two ranked lists.
  4. Optional: a cross-encoder reranker re-scores the merged top-N.

In practice, hybrid retrieval beats either approach alone by 5–15%
on most question-answering benchmarks.

RECIPROCAL RANK FUSION (RRF)
-----------------------------
RRF is a simple, parameter-free fusion algorithm:
    score(doc) = Σ 1 / (k + rank_in_list_i)

where k=60 is a constant that dampens the influence of top-ranked items
from either list. Documents appearing in both lists get a double bonus.
Documents only in one list get a partial score.

RRF was chosen over score normalisation (e.g. CombSUM) because:
  - No score calibration needed — ranks are used, not raw scores.
  - Dense cosine similarities (0..1) and BM25 scores (unbounded) are
    on completely different scales. Normalisation would require knowing
    the score distribution, which changes per corpus.
  - RRF was shown in the original Cormack 2009 paper to be competitive
    with manually tuned score fusion despite having only one parameter.

BM25 INDEX
----------
BM25 (Best Match 25) is a classic TF-IDF variant. rank_bm25 provides
a pure-Python implementation. Each retriever instance maintains one
BM25Index per document set.

The BM25 index is built in-memory from the chunk texts — it is not
persisted because: (a) it rebuilds in <100ms even for 10,000 chunks,
(b) the chunk texts are already stored in ChromaDB, so there is no
data duplication risk.

RERANKING
---------
Cross-encoder reranking is optional because it requires a separate model
download (~66 MB for ms-marco-MiniLM-L-6-v2). When enabled, the merged
top-20 candidates are re-scored by a model that sees the (query, chunk)
pair together — far more accurate than the bi-encoder that embeds query
and chunk independently.

The reranker is not installed by default. Enable it with:
    pip install sentence-transformers
    retriever = Retriever(..., use_reranker=True)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from rag.vector_store import VectorStore, SearchResult
from rag.embedder import Embedder


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A retrieved chunk with its combined relevance score."""
    document_id: str
    chunk_index: int
    text: str
    dense_score: float    # original cosine similarity from vector search
    bm25_score: float     # original BM25 score
    rrf_score: float      # combined Reciprocal Rank Fusion score
    rerank_score: float   # cross-encoder score if reranking was applied (else -1)
    metadata: dict = field(default_factory=dict)

    @property
    def final_score(self) -> float:
        """The score to use for ordering results."""
        return self.rerank_score if self.rerank_score >= 0 else self.rrf_score


# ── Retriever ─────────────────────────────────────────────────────────────────

class Retriever:
    """
    Hybrid retriever combining vector similarity and BM25 keyword matching.

    Usage:
        retriever = Retriever(vector_store, embedder)
        chunks = retriever.retrieve("what is gradient descent?", k=5)
    """

    RRF_K = 60           # RRF constant — dampens the influence of top ranks
    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        use_reranker: bool = False,
        candidate_multiplier: int = 4,
    ):
        """
        Args:
            vector_store:        ChromaDB wrapper from rag.vector_store.
            embedder:            fastembed wrapper from rag.embedder.
            use_reranker:        If True, applies cross-encoder reranking.
                                 Requires sentence-transformers installed.
            candidate_multiplier: Retrieve k * multiplier candidates before
                                 RRF fusion. More candidates = better fusion
                                 but slower. Default 4: retrieve 4k then cut to k.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.use_reranker = use_reranker
        self.candidate_multiplier = candidate_multiplier
        self._reranker = None

    def retrieve(
        self,
        query: str,
        k: int = 5,
        document_ids: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieves the top-k most relevant chunks for the query.

        Args:
            query:        The user's question in plain text.
            k:            Number of chunks to return.
            document_ids: If provided, restrict retrieval to these documents.

        Returns:
            List of RetrievedChunk objects sorted by final_score descending.
        """
        candidate_k = k * self.candidate_multiplier

        # ── Step 1: Dense retrieval ──────────────────────────────────────────
        query_vec = self.embedder.encode_one(query)
        dense_results = self.vector_store.search(
            query_embedding=query_vec,
            k=candidate_k,
            document_ids=document_ids,
        )

        # ── Step 2: BM25 retrieval ───────────────────────────────────────────
        # Build BM25 index over the same candidate pool.
        # For document-scoped queries we fetch all chunks from those documents.
        all_chunks = self._fetch_chunks_for_bm25(document_ids, dense_results)
        bm25_results = self._bm25_search(query, all_chunks, candidate_k)

        # ── Step 3: RRF fusion ───────────────────────────────────────────────
        fused = self._rrf_fuse(dense_results, bm25_results)
        top_candidates = fused[:max(k * 2, 10)]  # keep extra for reranker

        # ── Step 4: Optional cross-encoder reranking ─────────────────────────
        if self.use_reranker and top_candidates:
            top_candidates = self._rerank(query, top_candidates)

        return top_candidates[:k]

    # ── BM25 ──────────────────────────────────────────────────────────────────

    def _fetch_chunks_for_bm25(
        self,
        document_ids: Optional[list[str]],
        dense_results: list[SearchResult],
    ) -> list[dict]:
        """
        Fetches chunks for BM25 indexing.

        Strategy:
        - If document_ids are specified: fetch all chunks from those docs.
          This gives BM25 access to the full vocabulary, not just the
          chunks the dense model retrieved.
        - Otherwise: use the dense results as the BM25 corpus. This is
          faster but slightly less accurate for keyword queries.
        """
        if document_ids:
            chunks = []
            for doc_id in document_ids:
                for item in self.vector_store.get_chunks(doc_id):
                    chunks.append({
                        "document_id": item["metadata"].get("document_id", doc_id),
                        "chunk_index": item["metadata"].get("chunk_index", 0),
                        "text": item["text"],
                        "metadata": item["metadata"],
                    })
            return chunks
        else:
            # Fall back to dense results as BM25 corpus
            return [
                {
                    "document_id": r.document_id,
                    "chunk_index": r.chunk_index,
                    "text": r.text,
                    "metadata": r.metadata,
                }
                for r in dense_results
            ]

    def _bm25_search(
        self,
        query: str,
        chunks: list[dict],
        k: int,
    ) -> list[tuple[dict, float]]:
        """
        Runs BM25 over the chunk corpus and returns top-k (chunk, score) pairs.

        Tokenises by lowercasing and splitting on whitespace/punctuation.
        Simple but effective for English text.
        """
        if not chunks:
            return []

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            # BM25 not available — return empty so RRF falls back to dense only
            return []

        import re

        def tokenise(text: str) -> list[str]:
            return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()

        corpus = [tokenise(c["text"]) for c in chunks]
        query_tokens = tokenise(query)

        if not any(corpus) or not query_tokens:
            return []

        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)

        ranked = sorted(
            [(chunks[i], float(scores[i])) for i in range(len(chunks))],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:k]

    # ── RRF fusion ────────────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        dense: list[SearchResult],
        bm25: list[tuple[dict, float]],
    ) -> list[RetrievedChunk]:
        """
        Merges dense and BM25 results using Reciprocal Rank Fusion.

        score(doc) = 1/(k + rank_in_dense_list) + 1/(k + rank_in_bm25_list)

        Documents appearing in both lists get contributions from both terms.
        Documents in only one list get a partial score.
        """
        rrf_scores: dict[str, float] = {}
        dense_scores: dict[str, float] = {}
        bm25_scores_map: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}

        # Dense contributions
        for rank, result in enumerate(dense):
            key = f"{result.document_id}:{result.chunk_index}"
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.RRF_K + rank + 1)
            dense_scores[key] = result.score
            chunk_data[key] = {
                "document_id": result.document_id,
                "chunk_index": result.chunk_index,
                "text": result.text,
                "metadata": result.metadata,
            }

        # BM25 contributions
        for rank, (chunk, score) in enumerate(bm25):
            key = f"{chunk['document_id']}:{chunk['chunk_index']}"
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.RRF_K + rank + 1)
            bm25_scores_map[key] = score
            if key not in chunk_data:
                chunk_data[key] = chunk

        # Build sorted result list
        results = []
        for key, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            data = chunk_data[key]
            results.append(RetrievedChunk(
                document_id=data["document_id"],
                chunk_index=data["chunk_index"],
                text=data["text"],
                dense_score=dense_scores.get(key, 0.0),
                bm25_score=bm25_scores_map.get(key, 0.0),
                rrf_score=rrf_score,
                rerank_score=-1.0,
                metadata=data.get("metadata", {}),
            ))

        return results

    # ── Reranking ─────────────────────────────────────────────────────────────

    def _load_reranker(self):
        """Lazily loads the cross-encoder reranking model."""
        if self._reranker is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.RERANK_MODEL)
        except ImportError:
            raise ImportError(
                "Reranking requires sentence-transformers: "
                "pip install sentence-transformers"
            )

    def _rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Re-scores candidates with a cross-encoder that sees (query, chunk) together.

        Cross-encoders are significantly more accurate than bi-encoders because
        they see both texts simultaneously in one forward pass, enabling
        attention between query and document tokens. The trade-off is speed:
        cross-encoders cannot pre-compute embeddings, so they must run once
        per (query, candidate) pair.

        The pattern: retrieve top-20 cheap with bi-encoder, rerank to top-5
        expensive with cross-encoder. This gives near-cross-encoder accuracy
        at near-bi-encoder throughput.
        """
        self._load_reranker()

        pairs = [(query, c.text) for c in candidates]
        scores = self._reranker.predict(pairs)

        for chunk, score in zip(candidates, scores):
            chunk.rerank_score = float(score)

        candidates.sort(key=lambda c: c.rerank_score, reverse=True)
        return candidates
