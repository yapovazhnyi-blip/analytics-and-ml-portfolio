"""
m1_data_pipeline/bigquery_loader.py
Load processed Parquet files into BigQuery tables.

Tables created:
    {dataset}.ratings
    {dataset}.movies
    {dataset}.streaming_events
    {dataset}.tmdb_metadata
"""
from __future__ import annotations

import pandas as pd
from google.cloud import bigquery

from m1_data_pipeline.config import cfg

TABLES = {
    "ratings": cfg.processed_dir / "ratings.parquet",
    "movies": cfg.processed_dir / "movies.parquet",
    "streaming_events": cfg.events_dir / "streaming_events.parquet",
    "tmdb_metadata": cfg.processed_dir / "tmdb_metadata.parquet",
}


def load_table(client: bigquery.Client, table_name: str, parquet_path) -> None:
    if not parquet_path.exists():
        print(f"[bq] Skipping {table_name} — file not found: {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)
    table_ref = f"{cfg.bq_project}.{cfg.bq_dataset}.{table_name}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
    )
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()  # block until done

    dest = job.destination
    print(f"[bq] Loaded {job.output_rows:,} rows → {dest.project}.{dest.dataset_id}.{dest.table_id}")


def main() -> None:
    if not cfg.bq_project:
        raise EnvironmentError("BIGQUERY_PROJECT_ID is not set in .env")

    client = bigquery.Client(project=cfg.bq_project)

    # Ensure dataset exists
    dataset = bigquery.Dataset(f"{cfg.bq_project}.{cfg.bq_dataset}")
    dataset.location = "US"
    client.create_dataset(dataset, exists_ok=True)
    print(f"[bq] Dataset ready: {cfg.bq_project}.{cfg.bq_dataset}")

    for table_name, path in TABLES.items():
        load_table(client, table_name, path)


if __name__ == "__main__":
    main()
