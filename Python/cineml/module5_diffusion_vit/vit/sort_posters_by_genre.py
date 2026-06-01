"""
module5_diffusion_vit/vit/sort_posters_by_genre.py
====================================================
One-time script that organises flat TMDB poster downloads into
genre subfolders required by PosterDataset.

Input  : data/raw/tmdb/posters/{movieId}.jpg  (flat, from fetch_tmdb.py)
         data/raw/tmdb/tmdb_metadata.parquet   (from fetch_tmdb.py)

Output : data/raw/tmdb/posters_by_genre/{genre}/{movieId}.jpg

Usage:
    python module5_diffusion_vit/vit/sort_posters_by_genre.py
"""
import sys
import shutil
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT         = Path(__file__).resolve().parent.parent.parent
POSTERS_DIR  = ROOT / "data" / "raw" / "tmdb" / "posters"
METADATA_PATH = ROOT / "data" / "raw" / "tmdb" / "tmdb_metadata.parquet"
OUTPUT_DIR   = ROOT / "data" / "raw" / "tmdb" / "posters_by_genre"

# Map TMDB genre names to the 8 categories ViT is trained on
GENRE_MAP = {
    "Action":      "Action",
    "Adventure":   "Action",
    "Comedy":      "Comedy",
    "Drama":       "Drama",
    "Thriller":    "Thriller",
    "Crime":       "Thriller",
    "Mystery":     "Thriller",
    "Science Fiction": "Sci-Fi",
    "Fantasy":     "Sci-Fi",
    "Romance":     "Romance",
    "Horror":      "Horror",
    "Documentary": "Documentary",
    "Animation":   "Comedy",
    "Family":      "Comedy",
}

GENRES = ["Action", "Comedy", "Drama", "Thriller", "Sci-Fi", "Romance", "Horror", "Documentary"]


def main():
    if not METADATA_PATH.exists():
        log.error("Run fetch_tmdb.py first to generate tmdb_metadata.parquet")
        sys.exit(1)

    if not POSTERS_DIR.exists():
        log.error("No posters directory found at %s", POSTERS_DIR)
        sys.exit(1)

    meta = pd.read_parquet(METADATA_PATH)
    log.info("Loaded metadata: %d movies", len(meta))

    # Create output genre dirs
    for genre in GENRES:
        (OUTPUT_DIR / genre).mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    # Debug: show column names and sample to diagnose any issues
    log.info("Metadata columns: %s", list(meta.columns))
    log.info("Sample genres_tmdb: %s", meta["genres_tmdb"].dropna().head(3).tolist())
    log.info("Sample movieId: %s", meta["movieId"].head(3).tolist() if "movieId" in meta.columns else "MISSING")

    # Handle both 'movieId' and 'movie_id' column names
    id_col = "movieId" if "movieId" in meta.columns else "movie_id"

    for _, row in meta.iterrows():
        movie_id   = row.get(id_col)
        genres_raw = row.get("genres_tmdb", [])

        # genres_tmdb may come back as numpy array after parquet round-trip
        if genres_raw is None:
            genres_raw = []
        elif hasattr(genres_raw, "tolist"):
            genres_raw = genres_raw.tolist()
        elif not isinstance(genres_raw, list):
            try:
                genres_raw = list(genres_raw)
            except Exception:
                genres_raw = []

        # Find first mappable genre
        target_genre = None
        for g in genres_raw:
            if g in GENRE_MAP:
                target_genre = GENRE_MAP[g]
                break

        if target_genre is None:
            skipped += 1
            continue

        src = POSTERS_DIR / f"{movie_id}.jpg"
        if not src.exists():
            skipped += 1
            continue

        dst = OUTPUT_DIR / target_genre / f"{movie_id}.jpg"
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

        if copied % 500 == 0 and copied > 0:
            print(f"PROGRESS:{copied}/{len(meta)}:Sorting posters by genre", flush=True)

    log.info("Done — copied %d posters, skipped %d", copied, skipped)

    # Print summary
    print("\nGenre distribution:")
    for genre in GENRES:
        n = len(list((OUTPUT_DIR / genre).glob("*.jpg")))
        print(f"  {genre:<15} {n:>5} images")

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("Use this path as --data-dir when running finetune.py:")
    print(f"  python module5_diffusion_vit/vit/finetune.py --data-dir {OUTPUT_DIR} --sample-size 500")


if __name__ == "__main__":
    main()
