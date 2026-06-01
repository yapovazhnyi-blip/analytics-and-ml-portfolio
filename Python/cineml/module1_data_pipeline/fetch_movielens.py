"""
Module 1 — Fetch & process MovieLens 25M dataset.

Downloads ratings.csv + movies.csv, cleans, deduplicates,
and saves versioned Parquet artefacts.

Usage:
    python fetch_movielens.py [--sample 100000]
"""
import argparse
import hashlib
import io
import logging
import sys
import zipfile
from pathlib import Path

# ── Windows UTF-8 fix ─────────────────────────────────────────────────────────
# cp1251/cp1252 can't encode box-drawing characters used in the summary output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import requests

from config import MOVIELENS_DIR, MOVIELENS_URL, PROCESSED_DIR, ARTIFACTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Download ──────────────────────────────────────────────────────────────────

def download_movielens(dest: Path = MOVIELENS_DIR) -> Path:
    """Stream-download the ML-25M zip and extract to dest/."""
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "ml-25m.zip"

    if zip_path.exists():
        log.info("Archive already present — skipping download.")
        return dest

    log.info("Downloading MovieLens 25M …")
    response = requests.get(MOVIELENS_URL, stream=True, timeout=120)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    downloaded = 0
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                print(f"PROGRESS:{downloaded}/{total}:Downloading MovieLens 25M", flush=True)

    log.info("Extracting …")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)

    return dest


# ── Load & clean ──────────────────────────────────────────────────────────────

def load_ratings(source_dir: Path, sample: int | None = None) -> pd.DataFrame:
    """Load ratings.csv, optionally sample, and enforce schema."""
    path = source_dir / "ml-25m" / "ratings.csv"
    log.info("Loading ratings from %s", path)

    dtype = {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}
    df = pd.read_csv(path, dtype=dtype)

    if sample:
        df = df.sample(sample, random_state=42)

    # Basic quality gates
    assert df["rating"].between(0.5, 5.0).all(), "Unexpected rating range"
    assert df[["userId", "movieId", "rating", "timestamp"]].notna().all().all(), "Nulls in ratings"

    duplicates = df.duplicated(["userId", "movieId"]).sum()
    if duplicates:
        log.warning("Dropping %d duplicate (user, movie) pairs", duplicates)
        df = df.drop_duplicates(["userId", "movieId"], keep="last")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    log.info("Ratings loaded: %s rows", f"{len(df):,}")
    return df


def load_movies(source_dir: Path) -> pd.DataFrame:
    """Load movies.csv, parse genres into a list column."""
    path = source_dir / "ml-25m" / "movies.csv"
    df = pd.read_csv(path)
    df["genres"] = df["genres"].str.split("|")
    df["year"] = df["title"].str.extract(r"\((\d{4})\)$").astype("Int16")
    df["title_clean"] = df["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True).str.strip()
    log.info("Movies loaded: %s rows", f"{len(df):,}")
    return df


# ── Versioned save ────────────────────────────────────────────────────────────

def _md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()[:8]


def save_parquet(df: pd.DataFrame, name: str, dest: Path = PROCESSED_DIR) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{name}.parquet"
    df.to_parquet(out, index=False, engine="pyarrow", compression="snappy")
    checksum = _md5(out)
    (ARTIFACTS_DIR / f"{name}.md5").write_text(checksum)
    log.info("Saved %s  [md5=%s]", out, checksum)
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch and process MovieLens 25M")
    parser.add_argument("--sample", type=int, default=None, help="Row sample for dev mode")
    args = parser.parse_args()

    source_dir = download_movielens()
    ratings = load_ratings(source_dir, sample=args.sample)
    movies = load_movies(source_dir)

    save_parquet(ratings, "ratings")
    save_parquet(movies, "movies")

    # Print quick summary
    print("\n── Data Quality Summary ──────────────────────────")
    print(f"  Ratings : {len(ratings):>10,} rows")
    print(f"  Users   : {ratings['userId'].nunique():>10,}")
    print(f"  Movies  : {ratings['movieId'].nunique():>10,}")
    print(f"  Date range: {ratings['timestamp'].min().date()} → {ratings['timestamp'].max().date()}")
    print(f"  Movies  : {len(movies):>10,} rows")
    print(f"  Genres  : {movies['genres'].explode().nunique()} unique")


if __name__ == "__main__":
    main()
