"""
Module 1 — Fetch TMDB metadata + poster images.

Matches MovieLens movie IDs to TMDB IDs via title/year search,
then downloads metadata JSON and poster thumbnails.

Usage:
    python fetch_tmdb.py --limit 5000
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
import time
from pathlib import Path

import pandas as pd
import requests

from config import (
    PROCESSED_DIR,
    TMDB_API_KEY,
    TMDB_BASE_URL,
    TMDB_DIR,
    TMDB_IMAGE_BASE,
    TMDB_POSTERS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

TMDB_FIELDS = [
    "id", "title", "overview", "genres", "release_date",
    "vote_average", "vote_count", "popularity",
    "poster_path", "backdrop_path", "original_language",
    "runtime", "budget", "revenue",
]


# ── TMDB API helpers ──────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict) -> dict | None:
    """Rate-limited GET with simple retry."""
    params["api_key"] = TMDB_API_KEY
    for attempt in range(3):
        try:
            r = requests.get(f"{TMDB_BASE_URL}/{endpoint}", params=params, timeout=10)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 2))
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            log.debug("Attempt %d failed: %s", attempt + 1, e)
            time.sleep(1)
    return None


def search_movie(title: str, year: int | None = None) -> dict | None:
    params = {"query": title, "include_adult": False}
    if year:
        params["year"] = year
    data = _get("search/movie", params)
    if data and data.get("results"):
        return data["results"][0]
    return None


def fetch_movie_details(tmdb_id: int) -> dict | None:
    return _get(f"movie/{tmdb_id}", {"append_to_response": "credits,keywords"})


# ── Poster download ───────────────────────────────────────────────────────────

def download_poster(poster_path: str, movie_id: int) -> Path | None:
    """Download a single poster image; returns local path or None on failure."""
    TMDB_POSTERS_DIR.mkdir(parents=True, exist_ok=True)
    dest = TMDB_POSTERS_DIR / f"{movie_id}.jpg"
    if dest.exists():
        return dest
    url = f"{TMDB_IMAGE_BASE}{poster_path}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return dest
    except requests.RequestException:
        return None


# ── Main pipeline ─────────────────────────────────────────────────────────────

def fetch_tmdb_metadata(movies_df: pd.DataFrame, limit: int = 5000) -> pd.DataFrame:
    """
    For each MovieLens movie, search TMDB and pull metadata.
    Returns a DataFrame aligned to movies_df index.
    """
    TMDB_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = TMDB_DIR / "tmdb_metadata.parquet"

    # Resume from cache
    done_ids: set[int] = set()
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        done_ids = set(cached["movieId"].tolist())
        log.info("Resuming — %d already fetched", len(done_ids))
    else:
        cached = pd.DataFrame()

    subset = movies_df[~movies_df["movieId"].isin(done_ids)].head(limit)
    records = []

    n_total = len(subset)
    for i, (_, row) in enumerate(subset.iterrows()):
        print(f"PROGRESS:{i+1}/{n_total}:Fetching TMDB metadata", flush=True)
        result = search_movie(row["title_clean"], year=row.get("year"))
        if not result:
            continue
        details = fetch_movie_details(result["id"])
        if not details:
            continue

        # Poster
        poster_local = None
        if details.get("poster_path"):
            poster_local = download_poster(details["poster_path"], row["movieId"])

        records.append({
            "movieId": row["movieId"],
            "tmdb_id": details["id"],
            "tmdb_title": details.get("title"),
            "overview": details.get("overview"),
            "genres_tmdb": [g["name"] for g in details.get("genres", [])],
            "release_date": details.get("release_date"),
            "vote_average": details.get("vote_average"),
            "vote_count": details.get("vote_count"),
            "popularity": details.get("popularity"),
            "runtime": details.get("runtime"),
            "poster_path": str(poster_local) if poster_local else None,
            "original_language": details.get("original_language"),
        })

        time.sleep(0.04)  # ~25 req/s — stay under free-tier limit

    new_df = pd.DataFrame(records)
    combined = pd.concat([cached, new_df], ignore_index=True) if not cached.empty else new_df
    combined.to_parquet(cache_path, index=False)
    log.info("TMDB metadata saved: %d movies total", len(combined))
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()

    if not TMDB_API_KEY:
        raise EnvironmentError("Set TMDB_API_KEY in your .env file")

    movies = pd.read_parquet(PROCESSED_DIR / "movies.parquet")
    fetch_tmdb_metadata(movies, limit=args.limit)


if __name__ == "__main__":
    main()
