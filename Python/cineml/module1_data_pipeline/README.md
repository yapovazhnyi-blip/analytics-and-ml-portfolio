# Module 1 — Data Curation Pipeline

Builds the shared data universe that every other module depends on. Three outputs: ratings, movies metadata, and synthetic streaming event logs.

## What it produces

| File | Rows | Description |
|------|------|-------------|
| `data/processed/ratings.parquet` | 25M | userId, movieId, rating, timestamp |
| `data/processed/movies.parquet` | 62k | movieId, title, genres (list), year |
| `data/processed/streaming_events.parquet` | ~500k | Synthetic impressions/clicks/completions with A/B arm split |
| `data/raw/tmdb/tmdb_metadata.parquet` | 7,050 | TMDB genres, overview, runtime, poster path |

## Run

```bash
python module1_data_pipeline/fetch_movielens.py
python module1_data_pipeline/fetch_tmdb.py --limit 5000
python module1_data_pipeline/event_simulator.py --n-users 10000 --events-per-user 50
python module1_data_pipeline/bigquery_loader.py --all   # optional, needs GCP credentials
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | Central config — all paths, BQ table refs, simulator constants |
| `fetch_movielens.py` | Stream-download ML-25M, clean, MD5-version, save Parquet |
| `fetch_tmdb.py` | Rate-limited TMDB API client — metadata + poster images, resumable |
| `event_simulator.py` | Position-decay CTR model, user activity tiers, A/B arm assignment |
| `bigquery_loader.py` | Upload to BQ — DAY partition on timestamp, cluster on [arm, event_type] |

## Event log schema

```
event_id        UUID    unique per event
user_id         int     0 to N_USERS
movie_id        int     MovieLens movieId
event_type      enum    impression | click | completion | skip | watchlist_add
timestamp       datetime
session_id      UUID
position_in_feed int    rank position shown (affects CTR)
dwell_ms        int
arm             enum    control | treatment
```

## BigQuery cost notes

Events table is DAY-partitioned on `timestamp` and clustered on `[arm, event_type]`. A query like `WHERE arm='treatment' AND event_type='click'` scans ~12× less data than an unpartitioned table.
