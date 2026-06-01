"""
m2_recommender/evaluate.py
Offline evaluation framework for ranking models.

Metrics:
    NDCG@k        — Normalised Discounted Cumulative Gain
    MAP@k         — Mean Average Precision
    Recall@k      — Fraction of relevant items in top-k
    Coverage      — Fraction of catalogue items ever recommended
    Novelty       — Average self-information of recommended items
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    """Discounted Cumulative Gain @ k."""
    r = relevance[:k].astype(float)
    if r.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, r.size + 2))
    return float((r / discounts).sum())


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """NDCG@k for a single user."""
    rel = np.array([1 if i in relevant else 0 for i in recommended[:k]])
    ideal = np.ones(min(len(relevant), k))
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(rel, k) / idcg


def average_precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Average Precision @ k for a single user."""
    hits = 0
    score = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k) if relevant else 0.0


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def evaluate_recommendations(
    recommendations: dict[int, list[int]],   # user_id → ordered list of item_ids
    ground_truth: dict[int, set[int]],        # user_id → set of relevant item_ids
    catalogue_size: int,
    k: int = 10,
) -> dict[str, float]:
    """
    Compute aggregated offline metrics over all users.

    Args:
        recommendations: model output, already trimmed to top-K
        ground_truth:    held-out positive interactions per user
        catalogue_size:  total number of items (for coverage)
        k:               cutoff

    Returns:
        dict with NDCG@k, MAP@k, Recall@k, Coverage, Novelty
    """
    ndcgs, maps, recalls = [], [], []
    all_recommended: set[int] = set()

    for uid, recs in recommendations.items():
        relevant = ground_truth.get(uid, set())
        ndcgs.append(ndcg_at_k(recs, relevant, k))
        maps.append(average_precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        all_recommended.update(recs[:k])

    item_counts = pd.Series(
        [item for recs in recommendations.values() for item in recs[:k]]
    ).value_counts()
    popularity = item_counts / item_counts.sum()
    novelty = float(-popularity.map(np.log2).mean())

    return {
        f"ndcg@{k}": float(np.mean(ndcgs)),
        f"map@{k}": float(np.mean(maps)),
        f"recall@{k}": float(np.mean(recalls)),
        "coverage": len(all_recommended) / catalogue_size,
        "novelty": novelty,
    }


def train_test_split_by_time(
    ratings: pd.DataFrame,
    test_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split: most-recent `test_frac` interactions per user go to test."""
    ratings = ratings.sort_values("timestamp")
    test_idx = (
        ratings.groupby("user_id")
        .apply(lambda g: g.tail(max(1, int(len(g) * test_frac))), include_groups=False)
        .index.get_level_values(1)
    )
    train = ratings.drop(index=test_idx)
    test = ratings.loc[test_idx]
    return train.reset_index(drop=True), test.reset_index(drop=True)
