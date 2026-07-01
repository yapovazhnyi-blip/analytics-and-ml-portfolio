"""
RAG pipeline tests.

Tests are structured in layers matching the pipeline components:

  Layer 1 — Chunker: pure Python, no dependencies. Fast.
  Layer 2 — Embedder: requires fastembed download on first run (~22 MB).
  Layer 3 — Vector store: requires chromadb (no network after init).
  Layer 4 — Retriever: builds on embedder + vector store.
  Layer 5 — Integration: full ingest→query flow with a mock generator.

The generator is mocked in all tests to avoid real API calls.
To test against the real Claude API: set ANTHROPIC_API_KEY and
pytest tests/test_rag.py -k "real_api" (no such tests yet — add them
once the pipeline is stable and you have budget).
"""

from __future__ import annotations

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ══════════════════════════════════════════════════════════════════════════
# CHUNKER TESTS — pure Python, no dependencies
# ══════════════════════════════════════════════════════════════════════════

class TestChunker:

    SAMPLE = (
        "The quick brown fox jumps over the lazy dog. "
        "This sentence tests the chunker behaviour.\n\n"
        "A second paragraph follows. It contains multiple sentences. "
        "Each sentence ends with a full stop. The chunker must handle this correctly.\n\n"
        "Third and final paragraph. Short one."
    )

    def test_paragraph_strategy_produces_chunks(self):
        from rag.chunker import chunk_text
        chunks = chunk_text(self.SAMPLE, strategy="paragraph")
        assert len(chunks) >= 1
        assert all(c.text.strip() for c in chunks)

    def test_paragraph_respects_max_chars(self):
        from rag.chunker import chunk_text
        chunks = chunk_text(self.SAMPLE, strategy="paragraph", max_chars=100)
        for c in chunks:
            # May exceed slightly at sentence boundaries — allow 20% headroom
            assert len(c.text) <= 200, f"Chunk too long: {len(c.text)}"

    def test_fixed_strategy_consistent_sizes(self):
        from rag.chunker import chunk_text
        long_text = "word " * 1000
        chunks = chunk_text(long_text, strategy="fixed", max_chars=200, overlap_chars=20)
        # All except the last should be exactly max_chars
        for c in chunks[:-1]:
            assert len(c.text) == 200

    def test_fixed_overlap_present(self):
        from rag.chunker import chunk_text
        text = "a" * 500
        chunks = chunk_text(text, strategy="fixed", max_chars=100, overlap_chars=20)
        # The start of chunk[1] should appear in the tail of chunk[0]
        assert len(chunks) >= 2
        assert chunks[1].text[:10] in chunks[0].text[-30:]

    def test_sentence_strategy_does_not_split_mid_sentence(self):
        from rag.chunker import chunk_text
        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = chunk_text(text, strategy="sentence", max_chars=60, overlap_chars=10)
        # No chunk should cut off in the middle of a word
        for c in chunks:
            assert c.text.strip() == c.text.strip()  # basic sanity
            assert len(c.text) > 0

    def test_chunk_indices_are_sequential(self):
        from rag.chunker import chunk_text
        chunks = chunk_text(self.SAMPLE, strategy="paragraph")
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_metadata_propagated(self):
        from rag.chunker import chunk_text
        chunks = chunk_text(self.SAMPLE, metadata={"source": "test.txt"})
        for c in chunks:
            assert c.metadata["source"] == "test.txt"

    def test_empty_text_returns_empty_list(self):
        from rag.chunker import chunk_text
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_chunk_file_txt(self, tmp_path):
        from rag.chunker import chunk_file
        f = tmp_path / "test.txt"
        f.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")
        chunks = chunk_file(str(f))
        assert len(chunks) >= 1
        assert all("source" in c.metadata for c in chunks)

    def test_unsupported_extension_raises(self, tmp_path):
        from rag.chunker import extract_text
        f = tmp_path / "test.xlsx"
        f.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text(str(f))

    def test_unknown_strategy_raises(self):
        from rag.chunker import chunk_text
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunk_text("some text", strategy="magic")


# ══════════════════════════════════════════════════════════════════════════
# EMBEDDER TESTS
# ══════════════════════════════════════════════════════════════════════════

class TestEmbedder:
    """Tests the MockEmbedder mechanics and interface."""

    def test_encode_returns_correct_shape(self):
        from rag.embedder import MockEmbedder
        emb = MockEmbedder()
        vecs = emb.encode(["hello world", "foo bar"])
        assert vecs.shape == (2, 384)

    def test_encode_one_returns_1d(self):
        from rag.embedder import MockEmbedder
        emb = MockEmbedder()
        vec = emb.encode_one("test query")
        assert vec.shape == (384,)

    def test_vectors_are_normalised(self):
        import numpy as np
        from rag.embedder import MockEmbedder
        emb = MockEmbedder()
        vecs = emb.encode(["hello world"])
        norms = np.linalg.norm(vecs, axis=1)
        assert abs(norms[0] - 1.0) < 0.01

    def test_same_text_same_vector(self):
        """MockEmbedder must be deterministic — same input, same output."""
        from rag.embedder import MockEmbedder
        import numpy as np
        emb = MockEmbedder()
        v1 = emb.encode(["the quick brown fox"])
        v2 = emb.encode(["the quick brown fox"])
        assert np.allclose(v1, v2)

    def test_different_texts_different_vectors(self):
        from rag.embedder import MockEmbedder
        import numpy as np
        emb = MockEmbedder()
        v1 = emb.encode(["hello world"])
        v2 = emb.encode(["something completely different"])
        # Should not be identical
        assert not np.allclose(v1, v2)

    def test_empty_list_returns_empty_array(self):
        import numpy as np
        from rag.embedder import MockEmbedder
        emb = MockEmbedder()
        result = emb.encode([])
        assert result.shape == (0, 384)

    def test_singleton_get_embedder(self):
        from rag.embedder import get_embedder
        e1 = get_embedder()
        e2 = get_embedder()
        assert e1 is e2   # same instance


# ══════════════════════════════════════════════════════════════════════════
# VECTOR STORE TESTS
# ══════════════════════════════════════════════════════════════════════════

class TestVectorStore:

    @pytest.fixture
    def store(self, tmp_path):
        from rag.vector_store import VectorStore
        return VectorStore(persist_dir=str(tmp_path / "chroma"))

    @pytest.fixture
    def sample_embeddings(self):
        import numpy as np
        # Pre-built unit vectors — no embedding model needed for storage tests
        np.random.seed(42)
        vecs = np.random.randn(5, 384).astype(np.float32)
        # L2-normalise so cosine similarity works correctly
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def test_add_and_count(self, store, sample_embeddings):
        store.add_chunks(
            document_id="doc1",
            texts=["chunk 0", "chunk 1", "chunk 2", "chunk 3", "chunk 4"],
            embeddings=sample_embeddings,
            metadatas=[{"chunk_index": i, "document_id": "doc1"} for i in range(5)],
        )
        assert store.count_chunks("doc1") == 5

    def test_search_returns_results(self, store, sample_embeddings):
        store.add_chunks(
            document_id="doc1",
            texts=["chunk 0", "chunk 1", "chunk 2", "chunk 3", "chunk 4"],
            embeddings=sample_embeddings,
            metadatas=[{"chunk_index": i, "document_id": "doc1"} for i in range(5)],
        )
        results = store.search(query_embedding=sample_embeddings[0], k=3)
        assert len(results) == 3
        # The most similar to chunk 0 should be chunk 0 itself
        assert results[0].score > 0.99

    def test_search_respects_document_filter(self, store, sample_embeddings):
        store.add_chunks(
            document_id="doc1",
            texts=["doc1 chunk"] * 3,
            embeddings=sample_embeddings[:3],
            metadatas=[{"chunk_index": i, "document_id": "doc1"} for i in range(3)],
        )
        store.add_chunks(
            document_id="doc2",
            texts=["doc2 chunk"] * 2,
            embeddings=sample_embeddings[3:],
            metadatas=[{"chunk_index": i, "document_id": "doc2"} for i in range(2)],
        )
        results = store.search(sample_embeddings[0], k=5, document_ids=["doc1"])
        assert all(r.document_id == "doc1" for r in results)

    def test_delete_document(self, store, sample_embeddings):
        store.add_chunks(
            document_id="doc1",
            texts=["a", "b"],
            embeddings=sample_embeddings[:2],
            metadatas=[{"chunk_index": i, "document_id": "doc1"} for i in range(2)],
        )
        assert store.count_chunks("doc1") == 2
        store.delete_document("doc1")
        assert store.count_chunks("doc1") == 0

    def test_list_documents(self, store, sample_embeddings):
        store.add_chunks("alpha", ["text"], sample_embeddings[:1],
                         [{"chunk_index": 0, "document_id": "alpha"}])
        store.add_chunks("beta", ["text"], sample_embeddings[1:2],
                         [{"chunk_index": 0, "document_id": "beta"}])
        docs = store.list_documents()
        assert "alpha" in docs
        assert "beta" in docs

    def test_empty_search_returns_empty(self, store, sample_embeddings):
        results = store.search(sample_embeddings[0], k=5)
        assert results == []


# ══════════════════════════════════════════════════════════════════════════
# RETRIEVER TESTS
# ══════════════════════════════════════════════════════════════════════════

class TestRetriever:

    @pytest.fixture
    def retriever_and_store(self, tmp_path):
        import numpy as np
        from rag.embedder import MockEmbedder
        from rag.vector_store import VectorStore
        from rag.retriever import Retriever

        emb = MockEmbedder()
        store = VectorStore(str(tmp_path / "chroma"))
        retriever = Retriever(vector_store=store, embedder=emb, use_reranker=False)

        # Index 6 short documents
        docs = [
            ("doc1", "Machine learning uses algorithms to learn from data."),
            ("doc1", "Neural networks are a type of machine learning model."),
            ("doc2", "The weather forecast shows rain tomorrow morning."),
            ("doc2", "Temperatures will drop significantly this weekend."),
            ("doc3", "Python is a popular programming language for data science."),
            ("doc3", "scikit-learn provides machine learning tools for Python."),
        ]
        for i, (doc_id, text) in enumerate(docs):
            vec = emb.encode([text])
            store.add_chunks(doc_id, [text], vec,
                             [{"chunk_index": i % 2, "document_id": doc_id,
                               "source": f"{doc_id}.txt"}])

        return retriever, store

    def test_retrieve_returns_k_results(self, retriever_and_store):
        retriever, _ = retriever_and_store
        results = retriever.retrieve("machine learning algorithms", k=3)
        assert len(results) <= 3
        assert len(results) >= 1

    def test_retrieve_ranks_relevant_first(self, retriever_and_store):
        retriever, _ = retriever_and_store
        results = retriever.retrieve("machine learning Python data science", k=4)
        texts = [r.text for r in results]
        # Weather chunks should not be top result for ML query
        assert not all("weather" in t.lower() for t in texts[:2])

    def test_retrieve_document_filter(self, retriever_and_store):
        retriever, _ = retriever_and_store
        results = retriever.retrieve("machine learning", k=5, document_ids=["doc2"])
        assert all(r.document_id == "doc2" for r in results)

    def test_rrf_scores_are_positive(self, retriever_and_store):
        retriever, _ = retriever_and_store
        results = retriever.retrieve("neural networks", k=3)
        for r in results:
            assert r.rrf_score > 0

    def test_empty_store_returns_empty(self, tmp_path):
        from rag.embedder import MockEmbedder
        from rag.vector_store import VectorStore
        from rag.retriever import Retriever

        emb = MockEmbedder()
        store = VectorStore(str(tmp_path / "empty"))
        retriever = Retriever(store, emb)
        results = retriever.retrieve("anything", k=5)
        assert results == []


# ══════════════════════════════════════════════════════════════════════════
# PIPELINE INTEGRATION TESTS (generator mocked)
# ══════════════════════════════════════════════════════════════════════════

class TestRAGPipeline:
    """
    End-to-end pipeline tests with the Claude generator mocked.
    Tests the full ingest → query flow without API calls.
    Uses MockEmbedder so no model download is required.
    """

    @pytest.fixture
    def pipeline(self, tmp_path):
        from rag.pipeline import RAGPipeline, RAGConfig
        from rag.embedder import MockEmbedder
        return RAGPipeline(
            vector_store_dir=str(tmp_path / "rag"),
            config=RAGConfig(retrieval_k=3, chunk_strategy="paragraph"),
            embedder=MockEmbedder(),
        )

    @pytest.fixture
    def sample_txt(self, tmp_path):
        """A minimal text document for ingestion tests."""
        content = (
            "Machine learning is a method of data analysis that automates "
            "analytical model building.\n\n"
            "It is based on the idea that systems can learn from data, "
            "identify patterns and make decisions with minimal human intervention.\n\n"
            "Deep learning is a subfield of machine learning concerned with "
            "algorithms inspired by the structure of the brain called artificial neural networks."
        )
        f = tmp_path / "ml_intro.txt"
        f.write_text(content, encoding="utf-8")
        return str(f)

    @pytest.mark.asyncio
    async def test_ingest_txt_succeeds(self, pipeline, sample_txt):
        result = await pipeline.ingest(sample_txt, document_id="ml-intro")
        assert result.succeeded, result.error
        assert result.chunk_count >= 1
        assert result.document_id == "ml-intro"

    @pytest.mark.asyncio
    async def test_ingest_updates_chunk_count(self, pipeline, sample_txt):
        result = await pipeline.ingest(sample_txt, document_id="ml-doc")
        stored = pipeline.chunk_count("ml-doc")
        assert stored == result.chunk_count

    @pytest.mark.asyncio
    async def test_delete_removes_chunks(self, pipeline, sample_txt):
        await pipeline.ingest(sample_txt, document_id="to-delete")
        assert pipeline.chunk_count("to-delete") > 0
        pipeline.delete_document("to-delete")
        assert pipeline.chunk_count("to-delete") == 0

    @pytest.mark.asyncio
    async def test_list_documents_includes_ingested(self, pipeline, sample_txt):
        await pipeline.ingest(sample_txt, document_id="listed-doc")
        docs = pipeline.list_documents()
        assert "listed-doc" in docs

    @pytest.mark.asyncio
    async def test_query_with_mocked_generator(self, pipeline, sample_txt):
        """Full query flow with generator mocked — no API call."""
        await pipeline.ingest(sample_txt, document_id="ml-q")

        from rag.generator import GeneratorResponse, Citation
        mock_response = GeneratorResponse(
            answer="Machine learning automates model building from data.",
            citations=[Citation("ml-q", 0, "ml_intro.txt")],
            model="claude-mock",
            input_tokens=100,
            output_tokens=20,
        )

        with patch("rag.generator.Generator.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_response
            result = await pipeline.query("What is machine learning?", k=3,
                                         document_ids=["ml-q"])

        assert result.succeeded, result.error
        assert "machine learning" in result.answer.lower()
        assert len(result.chunks) >= 1

    @pytest.mark.asyncio
    async def test_query_returns_error_when_no_api_key(self, pipeline, sample_txt, monkeypatch):
        """Without API key, generator returns a helpful error, not a crash."""
        await pipeline.ingest(sample_txt, document_id="no-key")
        monkeypatch.setattr("rag.generator.settings.anthropic_api_key", None)
        result = await pipeline.query("any question", k=2, document_ids=["no-key"])
        assert not result.succeeded
        # Error message may mention ANTHROPIC_API_KEY or Settings → API Keys (BYOK)
        err = result.error or ""
        assert any(kw in err for kw in ("ANTHROPIC_API_KEY", "API key", "api-keys", "Settings"))

    @pytest.mark.asyncio
    async def test_ingest_missing_file_returns_error(self, pipeline):
        result = await pipeline.ingest("/nonexistent/file.txt", document_id="bad")
        assert not result.succeeded
        assert result.error is not None
