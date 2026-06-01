# M1 — Data Curation Pipeline

Pulls **MovieLens 25M** ratings, **TMDB** metadata + poster images, and generates **synthetic streaming event logs**. Loads everything into BigQuery and local Parquet, with DVC versioning and automated quality checks.

## Outputs consumed by

| Downstream module | Artefact used |
|---|---|
| M2 Recommender | `data/processed/ratings.parquet`, `data/processed/movies.parquet` |
| M3 A/B Engine | `data/events/streaming_events.parquet` |
| M4 Analysis Memo | All of the above |
| M5 Diffusion + ViT | `data/posters/` image directory |

## Steps

```bash
# 1. Download MovieLens 25M
python m1_data_pipeline/ingest.py --source movielens

# 2. Pull TMDB metadata + posters (requires TMDB_API_KEY in .env)
python m1_data_pipeline/ingest.py --source tmdb --limit 45000

# 3. Generate synthetic streaming event logs
python m1_data_pipeline/simulate_events.py --n-users 10000 --days 90

# 4. Run quality checks
python m1_data_pipeline/quality_checks.py

# 5. Load to BigQuery (optional — requires GCP credentials)
python m1_data_pipeline/bigquery_loader.py

# Or run the full pipeline at once:
python m1_data_pipeline/pipeline.py
```

## Schema

### `ratings.parquet`
| column | type | description |
|--------|------|-------------|
| user_id | int64 | MovieLens user identifier |
| movie_id | int64 | MovieLens movie identifier |
| rating | float32 | Explicit rating 0.5–5.0 |
| timestamp | datetime64 | Rating timestamp |

### `streaming_events.parquet`
| column | type | description |
|--------|------|-------------|
| user_id | int64 | User identifier |
| movie_id | int64 | Movie identifier |
| event_type | str | impression / click / completion / skip |
| session_id | str | UUID session grouping |
| timestamp | datetime64 | Event timestamp |
| experiment_arm | str | control / treatment (set in M3) |
