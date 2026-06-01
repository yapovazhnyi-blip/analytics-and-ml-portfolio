"""
Module 2 — Training entry-point.

Trains ALS and Two-Tower models, evaluates both, and saves artifacts.

Usage:
    python train.py --model als
    python train.py --model two-tower --epochs 20
    python train.py --model all
"""
import argparse
import logging
import sys
from pathlib import Path

# ── Windows UTF-8 fix ─────────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Repo root on sys.path ─────────────────────────────────────────────────────
# Required when running this script directly (python module2_recommender/train.py)
# so that sibling modules (module2_recommender, module5_diffusion_vit) are importable.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from models.als_model import ALSRecommender
from models.two_tower import TwoTowerModel
from evaluation import evaluate_recommender, print_report

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ImplicitDataset(Dataset):
    """Positive + negative sampling for BPR training."""

    def __init__(
        self,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        genre_vocab: dict[str, int],
        max_genres: int = 5,
        neg_ratio: int = 4,
    ):
        pos = ratings[ratings["rating"] >= 3.5].copy()
        self.users = torch.tensor(pos["userId"].values, dtype=torch.long)
        self.items = torch.tensor(pos["movieId"].values, dtype=torch.long)

        # Build genre lookup
        genre_map = {}
        for _, row in movies.iterrows():
            genres = row.get("genres", []) or []
            ids = [genre_vocab.get(g, 0) for g in genres[:max_genres]]
            ids += [0] * (max_genres - len(ids))
            genre_map[row["movieId"]] = ids

        self.genre_ids = torch.tensor(
            [genre_map.get(m, [0] * max_genres) for m in pos["movieId"].values],
            dtype=torch.long,
        )
        self.all_items = pos["movieId"].unique()
        self.neg_ratio = neg_ratio

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        neg_idx = np.random.choice(len(self.all_items))
        neg_item = self.all_items[neg_idx]
        return {
            "user_id": self.users[idx],
            "pos_item_id": self.items[idx],
            "pos_genre_ids": self.genre_ids[idx],
            "neg_item_id": torch.tensor(neg_item, dtype=torch.long),
            "neg_genre_ids": torch.zeros(self.genre_ids.shape[1], dtype=torch.long),
        }


# ── ALS training ──────────────────────────────────────────────────────────────

def train_als(ratings: pd.DataFrame) -> None:
    split = int(len(ratings) * 0.9)
    train, test = ratings.iloc[:split], ratings.iloc[split:]

    model = ALSRecommender(factors=128, iterations=30, alpha=40.0)
    model.fit(train)

    results = evaluate_recommender(
        lambda uid, k: [m for m, _ in model.recommend(uid, k)],
        test_df=test,
        train_df=train,
        k=10,
    )
    print_report(results)
    model.save(MODEL_DIR / "als_model.pkl")


# ── Two-Tower training ────────────────────────────────────────────────────────

def train_two_tower(ratings: pd.DataFrame, movies: pd.DataFrame, epochs: int = 20) -> None:
    genre_vocab = {
        g: i + 1
        for i, g in enumerate(
            sorted({g for genres in movies["genres"].dropna() for g in genres})
        )
    }

    n_users = ratings["userId"].max() + 1
    n_items = ratings["movieId"].max() + 1
    n_genres = len(genre_vocab) + 1

    dataset = ImplicitDataset(ratings, movies, genre_vocab)
    loader = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on %s", device)

    model = TwoTowerModel(n_users=n_users, n_items=n_items, n_genres=n_genres).to(device)
    optimiser = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                batch["user_id"],
                batch["pos_item_id"],
                batch["pos_genre_ids"],
                batch["neg_item_id"],
                batch["neg_genre_ids"],
            )
            loss = out["loss"]
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        log.info("Epoch %d/%d  loss=%.4f  lr=%.2e", epoch, epochs, avg_loss,
                 scheduler.get_last_lr()[0])
        print(f"PROGRESS:{epoch}/{epochs}:Training Two-Tower  loss={avg_loss:.4f}", flush=True)

    torch.save(model.state_dict(), MODEL_DIR / "two_tower.pt")
    log.info("Two-Tower model saved.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["als", "two-tower", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    ratings = pd.read_parquet(DATA_DIR / "ratings.parquet")
    movies = pd.read_parquet(DATA_DIR / "movies.parquet")

    if args.model in ("als", "all"):
        log.info("=== Training ALS ===")
        train_als(ratings)

    if args.model in ("two-tower", "all"):
        log.info("=== Training Two-Tower ===")
        train_two_tower(ratings, movies, epochs=args.epochs)


if __name__ == "__main__":
    main()
