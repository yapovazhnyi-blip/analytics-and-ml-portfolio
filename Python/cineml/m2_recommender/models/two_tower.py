"""
m2_recommender/models/two_tower.py
Two-Tower Neural Retrieval Model — PyTorch implementation.

Architecture:
    User tower  : user_id embedding → MLP → L2-normalised user vector
    Item tower  : item_id embedding + genre features → MLP → L2-normalised item vector
    Score       : dot product (inner product retrieval)
    Loss        : Bayesian Personalised Ranking (BPR)

Reference:
    Yi et al. (2019) "Sampling-Bias-Corrected Neural Modeling for Large Corpus
    Item Recommendations" — Google RecSys 2019.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Generic MLP with LayerNorm, GELU activations, and dropout."""

    def __init__(self, dims: list[int], dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UserTower(nn.Module):
    def __init__(
        self,
        n_users: int,
        embed_dim: int = 64,
        hidden_dims: list[int] | None = None,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        self.embedding = nn.Embedding(n_users + 1, embed_dim, padding_idx=0)
        self.mlp = MLP([embed_dim] + hidden_dims + [output_dim])

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(user_ids)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=-1)


class ItemTower(nn.Module):
    def __init__(
        self,
        n_items: int,
        n_genres: int,
        embed_dim: int = 64,
        hidden_dims: list[int] | None = None,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        self.embedding = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        # Item input = item embedding + genre features
        self.mlp = MLP([embed_dim + n_genres] + hidden_dims + [output_dim])

    def forward(
        self,
        item_ids: torch.Tensor,
        genre_features: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embedding(item_ids)
        x = torch.cat([x, genre_features], dim=-1)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """Full two-tower retrieval model."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_genres: int = 12,
        embed_dim: int = 64,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.user_tower = UserTower(n_users, embed_dim=embed_dim, output_dim=output_dim)
        self.item_tower = ItemTower(n_items, n_genres, embed_dim=embed_dim, output_dim=output_dim)

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        pos_genre_features: torch.Tensor,
        neg_genre_features: torch.Tensor,
    ) -> torch.Tensor:
        """Returns BPR loss over a batch of (user, pos_item, neg_item) triples."""
        user_vecs = self.user_tower(user_ids)                          # (B, D)
        pos_vecs  = self.item_tower(pos_item_ids, pos_genre_features)  # (B, D)
        neg_vecs  = self.item_tower(neg_item_ids, neg_genre_features)  # (B, D)

        pos_scores = (user_vecs * pos_vecs).sum(dim=-1)  # (B,)
        neg_scores = (user_vecs * neg_vecs).sum(dim=-1)  # (B,)

        # BPR loss: -mean log σ(pos - neg)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        return loss

    @torch.inference_mode()
    def score(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        genre_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dot-product scores for (user, item) pairs."""
        u = self.user_tower(user_ids)
        v = self.item_tower(item_ids, genre_features)
        return (u * v).sum(dim=-1)

    @torch.inference_mode()
    def get_user_embedding(self, user_id: int) -> torch.Tensor:
        uid = torch.tensor([user_id])
        return self.user_tower(uid).squeeze(0)

    @torch.inference_mode()
    def get_item_embeddings(
        self,
        item_ids: torch.Tensor,
        genre_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.item_tower(item_ids, genre_features)
