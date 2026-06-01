"""
Module 2 — ALS Matrix Factorisation on implicit feedback.

Uses the `implicit` library (GPU-accelerated ALS with Conjugate Gradient solver).
Treats rating ≥ 3.5 as a positive implicit signal; lower ratings are dropped.

References:
  Hu, Koren, Volinsky (2008) — Collaborative Filtering for Implicit Feedback Datasets
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import mean_average_precision_at_k, ndcg_at_k

log = logging.getLogger(__name__)


class ALSRecommender:
    """Thin wrapper around implicit.ALS with encode/decode helpers."""

    def __init__(
        self,
        factors: int = 128,
        regularization: float = 0.01,
        iterations: int = 50,
        alpha: float = 40.0,
        use_gpu: bool = False,
    ) -> None:
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            use_gpu=use_gpu,
        )
        self.alpha = alpha
        self._user_enc: dict[int, int] = {}
        self._item_enc: dict[int, int] = {}
        self._item_dec: dict[int, int] = {}
        self._interaction_matrix: sp.csr_matrix | None = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, ratings: pd.DataFrame) -> "ALSRecommender":
        """
        Args:
            ratings: DataFrame with columns [userId, movieId, rating]
        """
        pos = ratings[ratings["rating"] >= 3.5].copy()
        log.info("Positive interactions: %s", f"{len(pos):,}")

        # Encode IDs to contiguous integers
        users = pos["userId"].unique()
        items = pos["movieId"].unique()
        self._user_enc = {uid: i for i, uid in enumerate(users)}
        self._item_enc = {iid: i for i, iid in enumerate(items)}
        self._item_dec = {v: k for k, v in self._item_enc.items()}

        row = pos["userId"].map(self._user_enc).values
        col = pos["movieId"].map(self._item_enc).values
        # Confidence = 1 + alpha * rating (standard ALS confidence weighting)
        data = 1 + self.alpha * pos["rating"].values

        n_users = len(users)
        n_items = len(items)
        self._interaction_matrix = sp.csr_matrix((data, (row, col)), shape=(n_users, n_items))

        log.info("Fitting ALS — %d users × %d items …", n_users, n_items)
        self.model.fit(self._interaction_matrix)
        log.info("ALS training complete.")
        return self

    # ── Recommend ─────────────────────────────────────────────────────────────

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        filter_already_liked: bool = True,
    ) -> list[tuple[int, float]]:
        """
        Returns list of (original_movie_id, score) tuples.
        """
        if user_id not in self._user_enc:
            raise KeyError(f"Unknown user_id: {user_id}")

        enc_uid = self._user_enc[user_id]
        ids, scores = self.model.recommend(
            enc_uid,
            self._interaction_matrix[enc_uid],
            N=n,
            filter_already_liked_items=filter_already_liked,
        )
        return [(self._item_dec[i], float(s)) for i, s in zip(ids, scores)]

    # ── Similar items ─────────────────────────────────────────────────────────

    def similar_items(self, movie_id: int, n: int = 10) -> list[tuple[int, float]]:
        if movie_id not in self._item_enc:
            raise KeyError(f"Unknown movie_id: {movie_id}")
        enc_iid = self._item_enc[movie_id]
        ids, scores = self.model.similar_items(enc_iid, N=n + 1)
        return [(self._item_dec[i], float(s)) for i, s in zip(ids, scores) if i != enc_iid][:n]

    # ── Offline evaluation ────────────────────────────────────────────────────

    def evaluate(self, test_interactions: sp.csr_matrix, k: int = 10) -> dict[str, float]:
        """Compute NDCG@k and MAP@k on held-out interactions."""
        ndcg = ndcg_at_k(self.model, self._interaction_matrix, test_interactions, K=k)
        map_k = mean_average_precision_at_k(
            self.model, self._interaction_matrix, test_interactions, K=k
        )
        return {"ndcg_at_k": float(ndcg), "map_at_k": float(map_k), "k": k}

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        import pickle
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("ALS model saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "ALSRecommender":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
