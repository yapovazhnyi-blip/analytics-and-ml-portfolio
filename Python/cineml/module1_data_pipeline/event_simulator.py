"""
Module 1 — Synthetic streaming event log simulator.

Generates realistic impression → click → completion / skip sequences
for each user-movie pair, stratified by:
  - User activity level (power / casual / lurker)
  - Movie popularity
  - Genre affinity (latent per-user taste vector)

Output schema:
    event_id, user_id, movie_id, event_type, timestamp,
    session_id, position_in_feed, dwell_ms, arm (control/treatment)

Usage:
    python event_simulator.py --n-users 10000 --events-per-user 50
"""
import argparse
import logging
import sys

# ── Windows UTF-8 fix ─────────────────────────────────────────────────────────
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import (
    CLICK_THROUGH_RATE,
    COMPLETION_RATE,
    N_EVENTS_PER_USER,
    N_USERS,
    PROCESSED_DIR,
    RANDOM_SEED,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

RNG = np.random.default_rng(RANDOM_SEED)
GENRES = ["Action", "Comedy", "Drama", "Thriller", "Sci-Fi", "Romance", "Horror", "Documentary"]
EVENT_TYPES = ["impression", "click", "completion", "skip", "watchlist_add"]


# ── User & Movie profiles ─────────────────────────────────────────────────────

def build_user_profiles(n_users: int) -> pd.DataFrame:
    """Generate synthetic user attributes."""
    activity = RNG.choice(["power", "casual", "lurker"], size=n_users, p=[0.2, 0.5, 0.3])
    # Each user has a latent genre affinity vector (softmax of noise)
    taste = RNG.dirichlet(np.ones(len(GENRES)), size=n_users)
    df = pd.DataFrame({"user_id": range(n_users), "activity_level": activity})
    for i, g in enumerate(GENRES):
        df[f"affinity_{g.lower()}"] = taste[:, i]
    return df


def build_movie_profiles(movies_df: pd.DataFrame) -> pd.DataFrame:
    """Attach synthetic popularity scores to movies."""
    df = movies_df[["movieId"]].copy().rename(columns={"movieId": "movie_id"})
    # Popularity ~ log-normal (a few blockbusters, many niche films)
    df["popularity_score"] = RNG.lognormal(mean=0, sigma=1.5, size=len(df))
    df["popularity_score"] /= df["popularity_score"].max()
    return df


# ── Event generation ──────────────────────────────────────────────────────────

def _ctr_for_user_movie(
    user: pd.Series,
    movie: pd.Series,
    position: int,
    arm: str,
) -> float:
    """Position-decay CTR with activity and arm lift."""
    base_ctr = CLICK_THROUGH_RATE * movie["popularity_score"]
    activity_mult = {"power": 1.5, "casual": 1.0, "lurker": 0.4}[user["activity_level"]]
    position_decay = 1 / (1 + 0.1 * position)
    arm_lift = 1.2 if arm == "treatment" else 1.0  # treatment = two-tower recommender
    return min(base_ctr * activity_mult * position_decay * arm_lift, 0.95)


def simulate_events(
    user_profiles: pd.DataFrame,
    movie_profiles: pd.DataFrame,
    n_events_per_user: int = N_EVENTS_PER_USER,
) -> pd.DataFrame:
    """Main simulation loop."""
    movies = movie_profiles.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    start_ts = datetime(2023, 1, 1)

    records = []
    n_total_users = len(user_profiles)
    for ui, (_, user) in enumerate(user_profiles.iterrows()):
        print(f"PROGRESS:{ui+1}/{n_total_users}:Simulating streaming events", flush=True)
        # Assign arm (50/50 split)
        arm = "treatment" if user["user_id"] % 2 == 0 else "control"
        session_id = str(uuid.uuid4())

        for event_idx in range(n_events_per_user):
            # Sample a movie (weighted by popularity)
            movie = movies.sample(1, weights=movies["popularity_score"]).iloc[0]
            position = event_idx % 20
            ts = start_ts + timedelta(
                days=RNG.integers(0, 365).item(),
                seconds=RNG.integers(0, 86400).item(),
            )

            ctr = _ctr_for_user_movie(user, movie, position, arm)
            clicked = RNG.random() < ctr
            completed = clicked and RNG.random() < COMPLETION_RATE
            skipped = clicked and not completed and RNG.random() < 0.7
            watchlisted = clicked and not completed and RNG.random() < 0.1

            # Always log impression
            records.append({
                "event_id": str(uuid.uuid4()),
                "user_id": int(user["user_id"]),
                "movie_id": int(movie["movie_id"]),
                "event_type": "impression",
                "timestamp": ts,
                "session_id": session_id,
                "position_in_feed": position,
                "dwell_ms": int(RNG.integers(100, 5000)),
                "arm": arm,
            })

            if clicked:
                records.append({**records[-1], "event_id": str(uuid.uuid4()),
                                 "event_type": "click", "dwell_ms": int(RNG.integers(500, 3000))})
            if completed:
                records.append({**records[-1], "event_id": str(uuid.uuid4()),
                                 "event_type": "completion", "dwell_ms": int(RNG.integers(3600000, 7200000))})
            elif skipped:
                records.append({**records[-1], "event_id": str(uuid.uuid4()),
                                 "event_type": "skip", "dwell_ms": int(RNG.integers(100, 30000))})
            if watchlisted:
                records.append({**records[-1], "event_id": str(uuid.uuid4()),
                                 "event_type": "watchlist_add", "dwell_ms": 0})

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    log.info("Generated %s events for %d users", f"{len(df):,}", len(user_profiles))
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Simulate streaming event logs")
    parser.add_argument("--n-users", type=int, default=N_USERS)
    parser.add_argument("--events-per-user", type=int, default=N_EVENTS_PER_USER)
    args = parser.parse_args()

    movies = pd.read_parquet(PROCESSED_DIR / "movies.parquet")
    user_profiles = build_user_profiles(args.n_users)
    movie_profiles = build_movie_profiles(movies)

    events = simulate_events(user_profiles, movie_profiles, args.events_per_user)

    out = PROCESSED_DIR / "streaming_events.parquet"
    events.to_parquet(out, index=False, compression="snappy")
    log.info("Events saved to %s", out)

    # Quick stats
    print("\n── Event Log Summary ───────────────────────────────")
    print(events["event_type"].value_counts().to_string())
    print(f"\n  CTR  : {(events['event_type']=='click').sum() / (events['event_type']=='impression').sum():.3f}")
    print(f"  Arms : {events.drop_duplicates('user_id')['arm'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
