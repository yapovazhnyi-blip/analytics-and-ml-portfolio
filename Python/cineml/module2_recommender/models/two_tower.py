"""
Module 2 — Two-Tower Neural Recommender (PyTorch).

Architecture:
  User Tower  : user_id embedding → 2× Linear + LayerNorm + ReLU → d_model
  Item Tower  : item_id + genre embeddings → 2× Linear + LayerNorm + ReLU → d_model
  Score       : dot product (cosine similarity in inference)
  Loss        : BPR (Bayesian Personalised Ranking)

Training uses implicit feedback (interaction = positive signal).

References:
  Covington et al. (2016) — Deep Neural Networks for YouTube Recommendations
  Rendle et al. (2009) — BPR: Bayesian Personalised Ranking from Implicit Feedback
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    """Encodes a user into a fixed-size embedding vector."""

    def __init__(
        self,
        n_users: int,
        d_embed: int = 64,
        d_model: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.user_embed = nn.Embedding(n_users, d_embed, padding_idx=0)
        self.net = nn.Sequential(
            nn.Linear(d_embed, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Args: user_ids (B,) → Returns: (B, d_model)"""
        x = self.user_embed(user_ids)
        return self.net(x)


class ItemTower(nn.Module):
    """Encodes an item (movie) including genre signals."""

    def __init__(
        self,
        n_items: int,
        n_genres: int,
        d_embed: int = 64,
        d_genre: int = 16,
        d_model: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.item_embed = nn.Embedding(n_items, d_embed, padding_idx=0)
        self.genre_embed = nn.EmbeddingBag(n_genres, d_genre, mode="mean", padding_idx=0)
        in_features = d_embed + d_genre
        self.net = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        item_ids: torch.Tensor,
        genre_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            item_ids  : (B,)
            genre_ids : (B, max_genres) — padded with 0s
        Returns: (B, d_model)
        """
        item_x = self.item_embed(item_ids)
        genre_x = self.genre_embed(genre_ids)
        x = torch.cat([item_x, genre_x], dim=-1)
        return self.net(x)


class TwoTowerModel(nn.Module):
    """
    Full two-tower model.
    Scoring: cosine similarity × temperature.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_genres: int,
        d_embed: int = 64,
        d_model: int = 128,
        temperature: float = 0.07,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.user_tower = UserTower(n_users, d_embed, d_model, dropout)
        self.item_tower = ItemTower(n_items, n_genres, d_embed, 16, d_model, dropout)
        self.temperature = temperature

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        pos_genre_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        neg_genre_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        BPR forward pass.
        Returns a dict with 'loss' and 'pos_scores', 'neg_scores'.
        """
        user_emb = self.user_tower(user_ids)  # (B, D)
        pos_emb = self.item_tower(pos_item_ids, pos_genre_ids)  # (B, D)
        neg_emb = self.item_tower(neg_item_ids, neg_genre_ids)  # (B, D)

        # L2 normalise for cosine similarity
        user_emb = F.normalize(user_emb, dim=-1)
        pos_emb = F.normalize(pos_emb, dim=-1)
        neg_emb = F.normalize(neg_emb, dim=-1)

        pos_scores = (user_emb * pos_emb).sum(dim=-1) / self.temperature  # (B,)
        neg_scores = (user_emb * neg_emb).sum(dim=-1) / self.temperature  # (B,)

        # BPR loss: -log σ(pos - neg)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        return {"loss": loss, "pos_scores": pos_scores, "neg_scores": neg_scores}

    @torch.inference_mode()
    def score(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        genre_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score (user, item) pairs for inference. Returns (B,) float."""
        user_emb = F.normalize(self.user_tower(user_ids), dim=-1)
        item_emb = F.normalize(self.item_tower(item_ids, genre_ids), dim=-1)
        return (user_emb * item_emb).sum(dim=-1)

    @torch.inference_mode()
    def get_user_embedding(self, user_id: int) -> torch.Tensor:
        uid = torch.tensor([user_id])
        return F.normalize(self.user_tower(uid), dim=-1)

    @torch.inference_mode()
    def get_all_item_embeddings(
        self,
        item_ids: torch.Tensor,
        genre_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Pre-compute item embedding matrix for ANN retrieval."""
        return F.normalize(self.item_tower(item_ids, genre_ids), dim=-1)
