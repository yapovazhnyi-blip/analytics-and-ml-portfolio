"""
m1_data_pipeline/simulate_events.py
Generate realistic synthetic streaming event logs.

Each user session produces a sequence of:
    impression → (click?) → (completion | skip?)

User behaviour is modelled with genre-affinity vectors and
popularity bias, so the resulting logs are non-trivial and
suitable for real causal inference experiments in M3.

Usage:
    python -m m1_data_pipeline.simulate_events \
        --n-users 10000 --days 90 --seed 42
"""
from __future__ import annotations

import argparse
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from m1_data_pipeline.config import cfg


GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Horror", "Romance",
    "Sci-Fi", "Thriller",
]

EVENT_TYPES = ["impression", "click", "completion", "skip"]


def _build_movie_pool(n_movies: int = 5000) -> pd.DataFrame:
    """Build a lightweight synthetic movie catalogue if real data isn't ready."""
    rng = np.random.default_rng(cfg.random_seed)
    genre_matrix = rng.dirichlet(np.ones(len(GENRES)), size=n_movies)
    popularity = rng.power(0.4, size=n_movies)   # heavy-tailed
    return pd.DataFrame({
        "movie_id": np.arange(n_movies, dtype=np.int32),
        "popularity": popularity.astype(np.float32),
        **{f"genre_{g.lower()}": genre_matrix[:, i].astype(np.float32)
           for i, g in enumerate(GENRES)},
    })


def _build_user_pool(n_users: int) -> pd.DataFrame:
    """Each user has a genre-affinity vector + activity level."""
    rng = np.random.default_rng(cfg.random_seed + 1)
    affinity = rng.dirichlet(np.ones(len(GENRES)) * 0.5, size=n_users)
    activity = rng.exponential(scale=3.0, size=n_users).clip(0.5, 20)
    return pd.DataFrame({
        "user_id": np.arange(n_users, dtype=np.int32),
        "activity_level": activity.astype(np.float32),
        **{f"genre_{g.lower()}": affinity[:, i].astype(np.float32)
           for i, g in enumerate(GENRES)},
    })


def simulate(
    n_users: int = 10_000,
    days: int = 90,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    movies = _build_movie_pool()
    users = _build_user_pool(n_users)
    start = datetime(2024, 1, 1)

    genre_cols = [c for c in movies.columns if c.startswith("genre_")]

    records: list[dict] = []

    for _, user in tqdm(users.iterrows(), total=n_users, desc="Simulating users"):
        n_sessions = int(rng.poisson(user["activity_level"] * days / 30))
        user_affinity = user[genre_cols].values.astype(np.float64)

        for _ in range(n_sessions):
            ts = start + timedelta(
                seconds=int(rng.integers(0, days * 86_400))
            )
            session_id = str(uuid.uuid4())

            # Sample 20 candidate movies weighted by popularity
            weights = movies["popularity"].values.astype(np.float64)
            weights /= weights.sum()
            candidate_idx = rng.choice(len(movies), size=20, replace=False, p=weights)
            candidates = movies.iloc[candidate_idx]

            for _, movie in candidates.iterrows():
                movie_affinity = movie[genre_cols].values.astype(np.float64)
                relevance = float(np.dot(user_affinity, movie_affinity))

                # Impression always recorded
                records.append({
                    "user_id": int(user["user_id"]),
                    "movie_id": int(movie["movie_id"]),
                    "event_type": "impression",
                    "session_id": session_id,
                    "timestamp": ts,
                    "experiment_arm": "unassigned",
                })
                ts += timedelta(seconds=int(rng.integers(1, 10)))

                # Click probability driven by affinity + popularity
                click_prob = 0.1 + 0.5 * relevance + 0.1 * float(movie["popularity"])
                click_prob = min(click_prob, 0.95)
                if rng.random() < click_prob:
                    records.append({
                        "user_id": int(user["user_id"]),
                        "movie_id": int(movie["movie_id"]),
                        "event_type": "click",
                        "session_id": session_id,
                        "timestamp": ts,
                        "experiment_arm": "unassigned",
                    })
                    ts += timedelta(seconds=int(rng.integers(5, 60)))

                    # Completion vs skip
                    completion_prob = 0.3 + 0.6 * relevance
                    event = "completion" if rng.random() < completion_prob else "skip"
                    records.append({
                        "user_id": int(user["user_id"]),
                        "movie_id": int(movie["movie_id"]),
                        "event_type": event,
                        "session_id": session_id,
                        "timestamp": ts,
                        "experiment_arm": "unassigned",
                    })
                    ts += timedelta(minutes=int(rng.integers(5, 120)))

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    out = cfg.events_dir / "streaming_events.parquet"
    df.to_parquet(out, index=False)
    print(f"[simulate] {len(df):,} events → {out}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users", type=int, default=cfg.n_synthetic_users)
    parser.add_argument("--days", type=int, default=cfg.simulation_days)
    parser.add_argument("--seed", type=int, default=cfg.random_seed)
    args = parser.parse_args()
    simulate(n_users=args.n_users, days=args.days, seed=args.seed)


if __name__ == "__main__":
    main()
