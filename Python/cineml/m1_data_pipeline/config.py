"""
m1_data_pipeline/config.py
Central configuration for the data pipeline module.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


@dataclass
class DataConfig:
    # ── Paths ──────────────────────────────────────────────────────────────────
    raw_dir: Path = DATA_DIR / "raw"
    processed_dir: Path = DATA_DIR / "processed"
    events_dir: Path = DATA_DIR / "events"
    posters_dir: Path = DATA_DIR / "posters"

    # ── MovieLens ──────────────────────────────────────────────────────────────
    movielens_url: str = (
        "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    )
    movielens_zip: Path = field(init=False)

    # ── TMDB ──────────────────────────────────────────────────────────────────
    tmdb_api_key: str = field(default_factory=lambda: os.environ["TMDB_API_KEY"])
    tmdb_base_url: str = "https://api.themoviedb.org/3"
    tmdb_image_base: str = "https://image.tmdb.org/t/p/w342"
    tmdb_limit: int = 45_000

    # ── Synthetic events ──────────────────────────────────────────────────────
    n_synthetic_users: int = 10_000
    simulation_days: int = 90
    random_seed: int = 42

    # ── BigQuery ──────────────────────────────────────────────────────────────
    bq_project: str = field(
        default_factory=lambda: os.environ.get("BIGQUERY_PROJECT_ID", "")
    )
    bq_dataset: str = field(
        default_factory=lambda: os.environ.get("BIGQUERY_DATASET", "cineml")
    )

    def __post_init__(self) -> None:
        self.movielens_zip = self.raw_dir / "ml-25m.zip"
        for d in (
            self.raw_dir,
            self.processed_dir,
            self.events_dir,
            self.posters_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


cfg = DataConfig()
