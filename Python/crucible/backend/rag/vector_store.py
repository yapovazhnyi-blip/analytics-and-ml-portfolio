"""
RAG Vector Store — ChromaDB wrapper for embedding storage and retrieval.

WHY CHROMADB
------------
ChromaDB is an embeddable vector database that persists to disk with no
server process required. You import it like a Python library:
    import chromadb
    client = chromadb.PersistentClient(path="./data/chroma")

Alternatives and why they were not chosen:
  FAISS   — faster for similarity search but in-memory only (data lost
             on restart). Would need separate Parquet/SQLite persistence
             to survive restarts.
  Pinecone — managed cloud service, requires API key and network. Adds
             latency and a subscription cost.
  Weaviate — full server deployment with Docker. Overkill for a single
             developer tool.
  pgvector — excellent for Postgres users, but adds a schema migration
             (requires vector column in a table) and means the vector
             index lives in the same database as the application data,
             complicating backups.

ChromaDB gives: persistence, cosine similarity search, metadata
filtering, document-level delete — everything the RAG pipeline needs
with zero infrastructure.

DATA MODEL
----------
Each RAG document creates one ChromaDB collection with a fixed name
derived from the document_id. All chunks of a document live in that
collection. Deleting a document deletes the entire collection.

A flat design (all chunks in one collection) was considered but rejected
because it does not support document-scoped queries ("only retrieve
from document X") without iterating all chunks.

METADATA STORED PER CHUNK
--------------------------
- document_id: str   — parent document identifier
- chunk_index: int   — position within the document (for ordering results)
- source: str        — original filename
- strategy: str      — chunking strategy used ('paragraph', 'fixed', 'sentence')
- char_start: int    — character offset in the original text
- char_end: int      — character offset end
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class SearchResult:
    """One retrieved chunk from vector similarity search."""
    document_id: str
    chunk_index: int
    text: str
    score: float            # cosine similarity 0..1 (higher = more similar)
    metadata: dict


class VectorStore:
    """
    Manages one ChromaDB persistent client for the entire RAG module.

    Each document gets its own collection named 'doc_{document_id}'.
    This allows per-document queries and clean deletes.
    """

    def __init__(self, persist_dir: str):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    def _client_(self):
        if self._client is None:
            try:
                import chromadb
            except ImportError:
                raise ImportError("chromadb is required: pip install chromadb")
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._client

    def _collection_name(self, document_id: str) -> str:
        """
        Sanitise document_id to a valid ChromaDB collection name.
        ChromaDB names must be 3–63 chars, alphanumeric + hyphens.
        """
        import re
        safe = re.sub(r"[^a-zA-Z0-9\-]", "-", document_id)[:60]
        safe = safe.strip("-") or "doc"
        return f"d-{safe}"   # prefix ensures minimum 3 chars

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        document_id: str,
        texts: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict],
    ) -> None:
        """
        Stores text chunks with their embeddings in ChromaDB.

        Embeddings are provided pre-computed by the Embedder so this
        method is pure storage — no model loading happens here.

        Args:
            document_id:  Unique identifier for the parent document.
            texts:        Original chunk texts (stored as-is, returned in search).
            embeddings:   numpy array (n_chunks, dim) — pre-computed by Embedder.
            metadatas:    List of dicts, one per chunk. Must include chunk_index.
        """
        if not texts:
            return

        client = self._client_()
        coll = client.get_or_create_collection(
            name=self._collection_name(document_id),
            metadata={"hnsw:space": "cosine"},  # cosine similarity for retrieval
        )

        # ChromaDB requires string IDs
        ids = [f"{document_id}_{m.get('chunk_index', i)}" for i, m in enumerate(metadatas)]

        # Ensure all metadata values are JSON-serialisable (ChromaDB requirement)
        safe_metas = [_safe_meta(m) for m in metadatas]

        coll.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=safe_metas,
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        document_ids: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """
        Finds the k most similar chunks to the query embedding.

        Args:
            query_embedding: Shape (dim,) — from Embedder.encode_one(query).
            k:               Number of results to return.
            document_ids:    If provided, restrict search to these documents.
                             If None, searches all documents.

        Returns:
            List of SearchResult objects sorted by score descending.
        """
        client = self._client_()
        results: list[SearchResult] = []

        # Determine which collections to search
        if document_ids:
            coll_names = [self._collection_name(d) for d in document_ids]
        else:
            coll_names = [c.name for c in client.list_collections()]

        if not coll_names:
            return []

        qvec = query_embedding.tolist()

        for coll_name in coll_names:
            try:
                coll = client.get_collection(coll_name)
            except Exception:
                continue

            n_items = coll.count()
            if n_items == 0:
                continue

            n_query = min(k, n_items)
            res = coll.query(
                query_embeddings=[qvec],
                n_results=n_query,
                include=["documents", "metadatas", "distances"],
            )

            for text, meta, dist in zip(
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
            ):
                # ChromaDB cosine distance = 1 - cosine_similarity
                score = max(0.0, 1.0 - dist)
                results.append(SearchResult(
                    document_id=meta.get("document_id", "unknown"),
                    chunk_index=meta.get("chunk_index", 0),
                    text=text,
                    score=score,
                    metadata=meta,
                ))

        # Sort by score descending and return top-k across all collections
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def get_chunks(self, document_id: str) -> list[dict]:
        """Returns all chunks for a document, sorted by chunk_index."""
        client = self._client_()
        try:
            coll = client.get_collection(self._collection_name(document_id))
        except Exception:
            return []

        res = coll.get(include=["documents", "metadatas"])
        pairs = list(zip(res["documents"], res["metadatas"]))
        pairs.sort(key=lambda p: p[1].get("chunk_index", 0))
        return [{"text": t, "metadata": m} for t, m in pairs]

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_document(self, document_id: str) -> bool:
        """
        Removes all chunks for a document by deleting its collection.
        Returns True if the collection existed, False if not found.
        """
        client = self._client_()
        coll_name = self._collection_name(document_id)
        try:
            client.delete_collection(coll_name)
            return True
        except Exception:
            return False

    # ── Inspect ───────────────────────────────────────────────────────────────

    def list_documents(self) -> list[str]:
        """Returns document_ids of all indexed documents."""
        client = self._client_()
        coll_names = [c.name for c in client.list_collections()]
        doc_ids = []
        for name in coll_names:
            try:
                coll = client.get_collection(name)
                res = coll.get(limit=1, include=["metadatas"])
                if res["metadatas"]:
                    doc_id = res["metadatas"][0].get("document_id", name)
                    doc_ids.append(doc_id)
            except Exception:
                continue
        return doc_ids

    def count_chunks(self, document_id: str) -> int:
        """Returns number of chunks stored for a document."""
        client = self._client_()
        try:
            coll = client.get_collection(self._collection_name(document_id))
            return coll.count()
        except Exception:
            return 0


# ── Helper ────────────────────────────────────────────────────────────────────

def _safe_meta(meta: dict) -> dict:
    """
    ChromaDB requires metadata values to be str, int, float, or bool.
    Converts anything else to str so storage never raises TypeError.
    """
    safe = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            safe[k] = v
        else:
            safe[k] = str(v)
    return safe
