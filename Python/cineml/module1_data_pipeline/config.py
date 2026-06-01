"""
Module 1 — Data Curation Pipeline
Central configuration for all data sources and storage targets.
"""
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

for d in [RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── MovieLens ─────────────────────────────────────────────────────────────────
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
MOVIELENS_DIR = RAW_DIR / "movielens"

# ── TMDB ──────────────────────────────────────────────────────────────────────
TMDB_API_KEY: str = os.getenv("TMDB_API_KEY", "")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"
TMDB_DIR = RAW_DIR / "tmdb"
TMDB_POSTERS_DIR = TMDB_DIR / "posters"

# ── BigQuery ──────────────────────────────────────────────────────────────────
BQ_PROJECT: str = os.getenv("GCP_PROJECT", "cineml-project")
BQ_DATASET = "cineml"
BQ_TABLES = {
    "ratings": f"{BQ_PROJECT}.{BQ_DATASET}.ratings",
    "movies": f"{BQ_PROJECT}.{BQ_DATASET}.movies",
    "events": f"{BQ_PROJECT}.{BQ_DATASET}.streaming_events",
    "tmdb_meta": f"{BQ_PROJECT}.{BQ_DATASET}.tmdb_metadata",
}

# ── Event Simulator ───────────────────────────────────────────────────────────
N_USERS = 10_000
N_EVENTS_PER_USER = 50
CLICK_THROUGH_RATE = 0.15
COMPLETION_RATE = 0.60
RANDOM_SEED = 42
