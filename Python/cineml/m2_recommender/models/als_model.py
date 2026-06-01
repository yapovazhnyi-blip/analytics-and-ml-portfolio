"""
m2_recommender/models/als_model.py
Alternating Least Squares (ALS) on implicit feedback via the `implicit` library.
Wraps the raw ALS model with fit / recommend / evaluate helpers.
"""
from __future__ import annotations

from pathlib import Path

import implicit
import numpy as np
import pandas as pd
import scipy.sparse as sp


class ALSRecommender:
    """ALS collaborative filter on implicit (confidence-weighted) feedback."""

    def __init__(
        self,
        factors: int = 128,
        regularization: float = 0.01,
        iterations: int = 30,
        alpha: float = 40.0,   # confidence scaling c_ui = 1 + alpha * r_ui
        random_state: int = 42,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=random_state,
            use_gpu=False,
        )
        self._user_map: dict[int, int] = {}
        self._item_map: dict[int, int] = {}
        self._item_ids: list[int] = []
        self._interaction_matrix: sp.csr_matrix | None = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, ratings: pd.DataFrame) -> "ALSRecommender":
        """
        ratings: DataFrame with columns [user_id, movie_id, rating].
        Converts explicit ratings to confidence values: c = 1 + alpha * rating.
        """
        users = ratings["user_id"].unique()
        items = ratings["movie_id"].unique()
        self._user_map = {u: i for i, u in enumerate(users)}
        self._item_map = {m: i for i, m in enumerate(items)}
        self._item_ids = list(items)

        row = ratings["user_id"].map(self._user_map).values
        col = ratings["movie_id"].map(self._item_map).values
        data = (1 + self.alpha * ratings["rating"].values).astype(np.float32)

        mat = sp.coo_matrix(
            (data, (row, col)),
            shape=(len(users), len(items)),
        ).tocsr()
        self._interaction_matrix = mat

        # implicit expects item-user matrix
        self.model.fit(mat.T)
        return self

    # ── Recommend ─────────────────────────────────────────────────────────────

    def recommend(
        self, user_id: int, n: int = 10, filter_already_liked: bool = True
    ) -> list[dict]:
        if user_id not in self._user_map:
            return []
        u_idx = self._user_map[user_id]
        ids, scores = self.model.recommend(
            u_idx,
            self._interaction_matrix[u_idx],
            N=n,
            filter_already_liked_items=filter_already_liked,
        )
        return [
            {"movie_id": self._item_ids[i], "score": float(s)}
            for i, s in zip(ids, scores)
        ]

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        import pickle
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "als_model.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "ALSRecommender":
        import pickle
        with open(Path(path) / "als_model.pkl", "rb") as f:
            return pickle.load(f)
