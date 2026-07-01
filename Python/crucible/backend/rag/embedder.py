"""
RAG Embedder — converts text to dense vector representations.

WHAT EMBEDDINGS ARE
-------------------
An embedding model converts text into a dense numerical vector (e.g.
384 numbers) that captures semantic meaning. Two texts about the same
topic produce similar vectors even when they use different words. This
is what enables semantic search: a query about "how to prevent overfitting"
retrieves chunks about regularisation, dropout, and cross-validation —
not just chunks containing the word "overfitting".

WHY FASTEMBED INSTEAD OF SENTENCE-TRANSFORMERS
-----------------------------------------------
sentence-transformers is the most popular embedding library but requires
PyTorch (~2 GB). fastembed uses ONNX Runtime to serve the same models
at a fraction of the size (~50 MB for the library, ~30 MB for the model).
The model quality is identical — the weights are the same, only the
inference backend differs.

In production, fastembed's ONNX backend is also faster on CPU because
ONNX Runtime has more aggressive CPU optimisation than PyTorch in
inference mode.

MODEL: BAAI/bge-small-en-v1.5
------------------------------
- 33M parameters, 384-dimensional vectors
- Licensed under MIT
- MTEB score 62.1 (competitive with all-MiniLM-L6-v2 at 62.3)
- Quantised ONNX version: ~22 MB on disk
- Inference: ~0.5ms per chunk on a modern CPU

Models are downloaded on first use to ~/.cache/fastembed/ and cached.
This means the first call to Embedder() after installation takes
~5–10 seconds while the model downloads. Subsequent calls are instant.

INTERFACE
---------
The Embedder class wraps fastembed behind a simple encode() method that
returns numpy arrays. The RAG pipeline only calls encode() — it never
imports fastembed directly. This makes it easy to swap the backend
(e.g. to OpenAI embeddings for higher quality) without touching any
pipeline code.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ── Default model ─────────────────────────────────────────────────────────────

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIM   = 384


class Embedder:
    """
    Wraps fastembed's TextEmbedding for use in the Crucible RAG pipeline.

    The model is loaded lazily on first encode() call and cached for the
    lifetime of the instance. Creating multiple Embedder instances is safe
    but wasteful — use one instance per server process.

    Example:
        embedder = Embedder()
        vectors = embedder.encode(["hello world", "foo bar"])
        # vectors.shape == (2, 384)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.dim = DEFAULT_DIM
        self._model = None  # lazy initialisation

    def _load(self):
        """
        Downloads and initialises the ONNX embedding model.
        Called automatically on first encode().
        Downloads to ~/.cache/fastembed/ — ~22 MB for bge-small-en-v1.5.
        """
        if self._model is not None:
            return
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError(
                "fastembed is required for RAG embeddings. "
                "Run: pip install fastembed"
            )
        self._model = TextEmbedding(model_name=self.model_name)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Encodes a list of text strings into embedding vectors.

        Args:
            texts:       List of strings to embed.
            batch_size:  Number of texts processed per ONNX inference call.
                         32 is a good default for CPU — larger batches
                         improve throughput but use more memory.

        Returns:
            numpy array of shape (len(texts), self.dim).
            Each row is the normalised embedding for the corresponding text.
            Vectors are L2-normalised so cosine similarity == dot product.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        self._load()

        # fastembed.embed() returns a generator of numpy arrays
        embeddings = list(self._model.embed(texts, batch_size=batch_size))
        return np.array(embeddings, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        """
        Encodes a single string. Returns shape (dim,) not (1, dim).
        Convenience wrapper for query encoding.
        """
        return self.encode([text])[0]

    def warm_up(self) -> None:
        """
        Pre-loads the model. Call this at server startup (e.g. in the
        FastAPI lifespan) so the first real request doesn't incur the
        5–10 second model load delay.
        """
        self._load()
        # Encode a dummy string to initialise all ONNX buffers
        self.encode(["warmup"])


# ── Module-level singleton ────────────────────────────────────────────────────
# The application uses one shared Embedder instance to avoid redundant
# model loads. Import and use this:
#   from rag.embedder import get_embedder
#   vec = get_embedder().encode_one("my query")

_embedder: Optional[Embedder] = None


def get_embedder(model_name: str = DEFAULT_MODEL) -> Embedder:
    """Returns the shared module-level Embedder instance, creating it once."""
    global _embedder
    if _embedder is None or _embedder.model_name != model_name:
        _embedder = Embedder(model_name=model_name)
    return _embedder


# ── Mock embedder for testing ─────────────────────────────────────────────────

class MockEmbedder(Embedder):
    """
    A deterministic fake embedder that works without downloading any model.
    Produces consistent 384-dim vectors from text hashes.

    Use in tests where the real fastembed model is unavailable:
        from rag.embedder import MockEmbedder
        emb = MockEmbedder()

    The vectors are NOT semantically meaningful — semantic similarity tests
    will fail. Use only for testing pipeline wiring (ingest, store, retrieve
    mechanics), not embedding quality.

    For semantic similarity testing, use the real Embedder on a machine
    with internet access so the model can be downloaded.
    """

    def __init__(self):
        super().__init__(model_name="mock")
        self.dim = DEFAULT_DIM

    def _load(self):
        pass  # No model needed

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        vecs = []
        for text in texts:
            # Hash the text to a seed and generate a deterministic vector
            seed = hash(text) % (2**31)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dim).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-8   # L2-normalise
            vecs.append(vec)

        return np.array(vecs, dtype=np.float32)
