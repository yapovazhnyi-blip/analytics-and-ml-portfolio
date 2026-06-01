"""
Module 2 — Offline evaluation framework.

Metrics:
  NDCG@k   — normalised discounted cumulative gain
  MAP@k    — mean average precision
  Hit@k    — hit rate (recall@1 relaxed)
  Coverage — catalogue coverage
  Novelty  — mean self-information of recommendations

Usage:
    from evaluation import evaluate_recommender
    results = evaluate_recommender(model, test_df, k=10)
"""
import logging
from typing import Callable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Core metric functions ─────────────────────────────────────────────────────

def dcg_at_k(relevance: list[int], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    gains = np.array(relevance[:k], dtype=float)
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return float((gains / discounts).sum())


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """NDCG@k for a single user."""
    rel = [1 if r in relevant else 0 for r in recommended[:k]]
    ideal = sorted(rel, reverse=True)
    dcg = dcg_at_k(rel, k)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def ap_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Average Precision@k for a single user."""
    hits, precisions = 0, []
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            precisions.append(hits / (i + 1))
    return float(np.mean(precisions)) if precisions else 0.0


def hit_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    return float(len(set(recommended[:k]) & relevant) > 0)


def novelty(recommended: list[int], item_popularity: dict[int, float]) -> float:
    """Mean self-information (negative log popularity) of recommendations."""
    scores = [-np.log2(item_popularity.get(i, 1e-6) + 1e-9) for i in recommended]
    return float(np.mean(scores))


# ── Batch evaluation ──────────────────────────────────────────────────────────

def evaluate_recommender(
    recommend_fn: Callable[[int, int], list[int]],
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    k: int = 10,
) -> dict[str, float]:
    """
    Evaluate a recommender over all test users.

    Args:
        recommend_fn : Callable(user_id, n) → list of movie_ids
        test_df      : DataFrame with [userId, movieId, rating]
        train_df     : Training interactions (used for popularity + coverage)
        k            : Cut-off rank

    Returns: dict of metric_name → value
    """
    # Build item popularity from training set
    item_counts = train_df["movieId"].value_counts()
    total = item_counts.sum()
    item_pop = (item_counts / total).to_dict()
    catalogue = set(train_df["movieId"].unique())

    ndcgs, aps, hits, novelties = [], [], [], []
    covered_items: set[int] = set()

    test_users = test_df["userId"].unique()
    for user_id in test_users:
        relevant = set(test_df[test_df["userId"] == user_id]["movieId"].tolist())
        try:
            recs = recommend_fn(user_id, k)
        except KeyError:
            continue  # cold-start users not in train

        ndcgs.append(ndcg_at_k(recs, relevant, k))
        aps.append(ap_at_k(recs, relevant, k))
        hits.append(hit_at_k(recs, relevant, k))
        novelties.append(novelty(recs, item_pop))
        covered_items.update(recs)

    coverage = len(covered_items) / len(catalogue) if catalogue else 0.0

    results = {
        f"ndcg@{k}": float(np.mean(ndcgs)),
        f"map@{k}": float(np.mean(aps)),
        f"hit@{k}": float(np.mean(hits)),
        "novelty": float(np.mean(novelties)),
        "catalogue_coverage": coverage,
        "n_users_evaluated": len(ndcgs),
    }

    log.info("Evaluation complete: %s", results)
    return results


def print_report(results: dict[str, float]) -> None:
    print("\n── Recommender Evaluation ─────────────────────────")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<28} {v:.4f}")
        else:
            print(f"  {k:<28} {v}")
