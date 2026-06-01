"""Tests for Module 2 Two-Tower model."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pytest

from module2_recommender.models.two_tower import TwoTowerModel


@pytest.fixture
def model():
    return TwoTowerModel(n_users=100, n_items=200, n_genres=10, d_embed=16, d_model=32)


def test_forward_pass_shape(model):
    B = 8
    user_ids = torch.randint(0, 100, (B,))
    pos_item_ids = torch.randint(0, 200, (B,))
    pos_genre_ids = torch.randint(0, 10, (B, 5))
    neg_item_ids = torch.randint(0, 200, (B,))
    neg_genre_ids = torch.randint(0, 10, (B, 5))

    out = model(user_ids, pos_item_ids, pos_genre_ids, neg_item_ids, neg_genre_ids)
    assert "loss" in out
    assert out["loss"].shape == ()
    assert out["pos_scores"].shape == (B,)


def test_loss_is_positive(model):
    B = 16
    user_ids = torch.randint(0, 100, (B,))
    pos_item_ids = torch.randint(0, 200, (B,))
    pos_genre_ids = torch.zeros(B, 5, dtype=torch.long)
    neg_item_ids = torch.randint(0, 200, (B,))
    neg_genre_ids = torch.zeros(B, 5, dtype=torch.long)
    out = model(user_ids, pos_item_ids, pos_genre_ids, neg_item_ids, neg_genre_ids)
    assert out["loss"].item() >= 0


def test_score_range(model):
    user_ids = torch.tensor([0, 1, 2])
    item_ids = torch.tensor([0, 1, 2])
    genre_ids = torch.zeros(3, 5, dtype=torch.long)
    scores = model.score(user_ids, item_ids, genre_ids)
    assert scores.shape == (3,)
    assert (scores >= -1.0).all() and (scores <= 1.0).all(), "Cosine scores must be in [-1, 1]"
