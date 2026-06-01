"""
Module 1 — BigQuery loader.

Uploads processed Parquet files to BigQuery tables,
with schema enforcement and partition/cluster options.

Usage:
    python bigquery_loader.py --table ratings
    python bigquery_loader.py --all
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from config import BQ_DATASET, BQ_PROJECT, BQ_TABLES, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Schema definitions ────────────────────────────────────────────────────────

SCHEMAS: dict[str, list[bigquery.SchemaField]] = {
    "ratings": [
        bigquery.SchemaField("userId", "INTEGER"),
        bigquery.SchemaField("movieId", "INTEGER"),
        bigquery.SchemaField("rating", "FLOAT"),
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
    ],
    "movies": [
        bigquery.SchemaField("movieId", "INTEGER"),
        bigquery.SchemaField("title", "STRING"),
        bigquery.SchemaField("genres", "STRING", mode="REPEATED"),
        bigquery.SchemaField("year", "INTEGER"),
        bigquery.SchemaField("title_clean", "STRING"),
    ],
    "events": [
        bigquery.SchemaField("event_id", "STRING"),
        bigquery.SchemaField("user_id", "INTEGER"),
        bigquery.SchemaField("movie_id", "INTEGER"),
        bigquery.SchemaField("event_type", "STRING"),
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        bigquery.SchemaField("session_id", "STRING"),
        bigquery.SchemaField("position_in_feed", "INTEGER"),
        bigquery.SchemaField("dwell_ms", "INTEGER"),
        bigquery.SchemaField("arm", "STRING"),
    ],
}

PARQUET_MAP = {
    "ratings": PROCESSED_DIR / "ratings.parquet",
    "movies": PROCESSED_DIR / "movies.parquet",
    "events": PROCESSED_DIR / "streaming_events.parquet",
}

# ── Upload logic ──────────────────────────────────────────────────────────────

def ensure_dataset(client: bigquery.Client) -> None:
    dataset_ref = bigquery.Dataset(f"{BQ_PROJECT}.{BQ_DATASET}")
    dataset_ref.location = "US"
    try:
        client.create_dataset(dataset_ref, exists_ok=True)
        log.info("Dataset %s.%s ready", BQ_PROJECT, BQ_DATASET)
    except Exception as e:
        log.error("Failed to create dataset: %s", e)
        raise


def upload_table(client: bigquery.Client, table_name: str) -> None:
    parquet_path = PARQUET_MAP[table_name]
    if not parquet_path.exists():
        raise FileNotFoundError(f"Run the data pipeline first: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # Explode list columns for BQ compatibility
    if table_name == "movies" and "genres" in df.columns:
        # BQ REPEATED mode: pass as JSON array strings
        df["genres"] = df["genres"].apply(lambda x: x if isinstance(x, list) else [])

    table_id = BQ_TABLES[table_name]
    job_config = bigquery.LoadJobConfig(
        schema=SCHEMAS.get(table_name, []),
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=(table_name not in SCHEMAS),
    )

    # Partition events table by date for cost efficiency
    if table_name == "events":
        job_config.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="timestamp",
        )
        job_config.clustering_fields = ["arm", "event_type"]

    log.info("Uploading %s (%s rows) to %s …", table_name, f"{len(df):,}", table_id)
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()  # Wait for completion
    log.info("✓ %s uploaded — %s rows", table_name, f"{len(df):,}")


def main():
    parser = argparse.ArgumentParser(description="Load processed data into BigQuery")
    parser.add_argument("--table", choices=list(PARQUET_MAP.keys()), help="Single table to load")
    parser.add_argument("--all", action="store_true", help="Load all tables")
    args = parser.parse_args()

    client = bigquery.Client(project=BQ_PROJECT)
    ensure_dataset(client)

    tables_to_load = list(PARQUET_MAP.keys()) if args.all else [args.table]
    if not tables_to_load or tables_to_load == [None]:
        parser.error("Specify --table <name> or --all")

    for t in tables_to_load:
        upload_table(client, t)


if __name__ == "__main__":
    main()
