"""
m1_data_pipeline/ingest.py
Download MovieLens 25M and TMDB metadata + poster images.

Usage:
    python -m m1_data_pipeline.ingest --source movielens
    python -m m1_data_pipeline.ingest --source tmdb --limit 5000
    python -m m1_data_pipeline.ingest --source all
"""
from __future__ import annotations

import argparse
import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from m1_data_pipeline.config import cfg


# ── MovieLens ─────────────────────────────────────────────────────────────────

def download_movielens() -> None:
    """Download and unpack MovieLens 25M into data/raw/."""
    zip_path = cfg.movielens_zip
    if zip_path.exists():
        print(f"[movielens] Already downloaded: {zip_path}")
    else:
        print(f"[movielens] Downloading from {cfg.movielens_url} …")
        resp = requests.get(cfg.movielens_url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    print("[movielens] Extracting …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cfg.raw_dir)

    _process_movielens()


def _process_movielens() -> None:
    src = cfg.raw_dir / "ml-25m"
    ratings = pd.read_csv(src / "ratings.csv", dtype={"userId": "int32", "movieId": "int32"})
    ratings.columns = ["user_id", "movie_id", "rating", "timestamp"]
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings["rating"] = ratings["rating"].astype("float32")

    movies = pd.read_csv(src / "movies.csv")
    movies.columns = ["movie_id", "title", "genres"]
    movies["movie_id"] = movies["movie_id"].astype("int32")

    out_ratings = cfg.processed_dir / "ratings.parquet"
    out_movies = cfg.processed_dir / "movies.parquet"
    ratings.to_parquet(out_ratings, index=False)
    movies.to_parquet(out_movies, index=False)
    print(f"[movielens] Saved {len(ratings):,} ratings → {out_ratings}")
    print(f"[movielens] Saved {len(movies):,} movies  → {out_movies}")


# ── TMDB ──────────────────────────────────────────────────────────────────────

def fetch_tmdb(limit: int | None = None) -> None:
    """Fetch TMDB movie metadata and download poster thumbnails."""
    limit = limit or cfg.tmdb_limit
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {cfg.tmdb_api_key}"})

    records: list[dict] = []
    page = 1
    print(f"[tmdb] Fetching metadata (limit={limit}) …")

    with tqdm(total=limit) as bar:
        while len(records) < limit:
            resp = session.get(
                f"{cfg.tmdb_base_url}/movie/popular",
                params={"page": page, "language": "en-US"},
                timeout=30,
            )
            if resp.status_code == 429:
                time.sleep(2)
                continue
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break
            records.extend(results)
            bar.update(len(results))
            page += 1

    records = records[:limit]
    meta_df = pd.json_normalize(records)[
        ["id", "title", "genre_ids", "release_date", "vote_average",
         "vote_count", "popularity", "poster_path", "overview"]
    ].rename(columns={"id": "tmdb_id"})

    out_meta = cfg.processed_dir / "tmdb_metadata.parquet"
    meta_df.to_parquet(out_meta, index=False)
    print(f"[tmdb] Saved {len(meta_df):,} metadata rows → {out_meta}")

    _download_posters(meta_df, session)


def _download_posters(meta_df: pd.DataFrame, session: requests.Session) -> None:
    """Download poster images to data/posters/."""
    rows = meta_df.dropna(subset=["poster_path"])
    print(f"[tmdb] Downloading {len(rows):,} poster images …")
    for _, row in tqdm(rows.iterrows(), total=len(rows)):
        dest = cfg.posters_dir / f"{row['tmdb_id']}.jpg"
        if dest.exists():
            continue
        url = f"{cfg.tmdb_image_base}{row['poster_path']}"
        try:
            r = session.get(url, timeout=15)
            r.raise_for_status()
            dest.write_bytes(r.content)
        except requests.RequestException:
            pass  # skip failed downloads


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CineML data ingest")
    parser.add_argument(
        "--source", choices=["movielens", "tmdb", "all"], default="all"
    )
    parser.add_argument("--limit", type=int, default=None, help="TMDB record cap")
    args = parser.parse_args()

    if args.source in ("movielens", "all"):
        download_movielens()
    if args.source in ("tmdb", "all"):
        fetch_tmdb(limit=args.limit)


if __name__ == "__main__":
    main()
