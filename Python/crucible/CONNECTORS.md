# Connecting Data to Crucible

Crucible ingests data four ways: **file upload** (CSV/Parquet), a saved
**SQL Database** connector (PostgreSQL/SQLite), a saved **REST + OAuth2**
connector, and one-shot **BigQuery** import. This guide covers the three you
asked about, step by step.

> **Prerequisite for all of them:** the database schema must be migrated.
> Every connector ends by inserting a row into the `datasets` (or
> `connectors`) table, so if migrations have not run, every connector fails
> with `no such column` / `table ... already exists`. See
> [Database migrations](#database-migrations) at the bottom first if uploads
> are failing.

---

## 1. CSV / Parquet upload

The simplest path — a direct file upload, no configuration.

1. Open the **Datasets** page.
2. Click **Upload file** (top right) or **Upload your first dataset**.
3. Choose a `.csv` or `.parquet` file and confirm.

**What happens:** the file is validated (extension allowlist, 500 MB size
cap, and a magic-bytes check so a renamed binary can't slip through), parsed
into a dataframe by the `FileConnector`, its schema and a content hash are
computed, and a `Dataset` row is created with status `ready`.

**Limits & notes**
- Allowed extensions: `.csv`, `.parquet` only.
- Maximum size: 500 MB (rejected before the file is read into memory).
- A `.csv` that is actually Parquet (or vice versa) is rejected with a clear
  message — the extension alone is not trusted.

---

## 2. PostgreSQL (SQL Database connector)

This creates a **saved, reusable** connector: Crucible stores the connection
string (encrypted at rest with Fernet) and you can run queries against it
repeatedly.

### Step 1 — Create the connector

1. Open the **Connectors** page → **New connector**.
2. Keep the **SQL Database** tab selected.
3. Fill in:
   - **Name** — any label, e.g. `Prod analytics DB`.
   - **Database type** — `PostgreSQL`.
   - **Connection URL** — use the **async driver** prefix:
     ```
     postgresql+asyncpg://USER:PASSWORD@HOST:5432/DATABASE
     ```
     The `+asyncpg` part matters — Crucible connects asynchronously and
     converts to a synchronous driver internally only when needed. A plain
     `postgresql://...` URL will fail to connect.
4. Click **Create connector**.

### Step 2 — Verify it connects

On the connector's card, click **Test**. Crucible runs a lightweight
`SELECT 1` and reports success or the exact connection error. Fix
credentials/host/firewall until this passes.

### Step 3 — Pull data in

Use the connector to run a query and save the result as a dataset (the
`/datasets/from-sql` flow). Always scope the query so you don't pull an
entire warehouse table:
```sql
SELECT customer_id, amount, created_at
FROM transactions
WHERE created_at >= '2025-01-01'
LIMIT 500000
```

**Common pitfalls**
- `postgresql://` instead of `postgresql+asyncpg://` → connection fails.
- Managed Postgres (Neon, RDS, Cloud SQL) usually requires SSL — append
  `?ssl=require` (asyncpg) if the provider mandates it.
- The host must be reachable **from inside the backend container**. If your
  DB is on the Docker host itself, use `host.docker.internal`, not
  `localhost` (inside the container `localhost` is the container).

---

## 3. BigQuery (one-shot import)

Unlike the SQL connector, BigQuery is a **one-shot ingest**: you give it a
query and it creates a dataset directly. It does **not** save a reusable
connector record. Credentials are pasted into the form and used only for that
import — they are never stored.

### Step 0 — One-time setup (libraries)

The BigQuery client library is now enabled in `requirements.txt`
(`google-cloud-bigquery`, `db-dtypes`). Because that is a dependency change
(not just code), the backend image must be **rebuilt once**:
```powershell
docker compose build --no-cache backend
docker compose up
```

### Step 1 — Get a Google Cloud service account key

1. In the [Google Cloud Console](https://console.cloud.google.com), select or
   create a project, and enable the **BigQuery API** for it.
2. **IAM & Admin → Service Accounts → Create service account.**
3. Grant it the **BigQuery Job User** role (enough to run queries; the public
   datasets are already readable).
4. Open the account → **Keys → Add key → Create new key → JSON**. A `.json`
   key file downloads.

### Step 2 — Import from the interface

1. Open the **Connectors** page → **New connector** → **BigQuery** tab.
2. Fill in:
   - **Dataset name (in Crucible)** — what the imported dataset will be called.
   - **Google Cloud project ID** — your project's ID.
   - **SQL query** — always include a `LIMIT`:
     ```sql
     SELECT pickup_datetime, passenger_count, trip_distance, fare_amount, tip_amount
     FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2018`
     WHERE pickup_datetime >= '2018-01-01' AND pickup_datetime < '2018-01-08'
     LIMIT 500000
     ```
   - **Service account JSON** — open the `.json` key file from Step 1 and
     paste its entire contents here. (Leave empty only if the server has
     `GOOGLE_APPLICATION_CREDENTIALS` configured.)
   - **Location** — `US` for the public datasets, or your dataset's region.
   - **Max rows** — safety cap (hard maximum 5,000,000).
3. Click **Import from BigQuery**.

On success the dataset appears on the **Datasets** page, ready to profile and
train on like any uploaded file.

**Good public dataset to test with**
`bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2018` — hundreds of
millions of rows, so always keep the `LIMIT`. Natural regression targets:
`fare_amount`, `tip_amount`.

**Common pitfalls**
- `403 Access Denied` → the service account is missing the BigQuery Job User
  role, or the BigQuery API isn't enabled on the project.
- Query with no `LIMIT` on a huge table → slow and may hit the row cap; always
  bound it.
- `ImportError: google.cloud.bigquery` → the Step 0 rebuild hasn't happened.

---

## Database migrations

All connectors depend on the schema being current. Migrations run as a
**separate step before the web server starts** (`python migrate.py` in the
backend container command), in their own process — which is why they're
reliable regardless of the app's async event loop.

**If uploads/connectors fail with a schema error** (`no such column:
datasets.contract_json`, or `table connectors already exists`), the database
file is in a half-migrated state from an earlier run. Reset it cleanly:

```powershell
docker compose down
docker volume rm crucible_crucible-data
docker compose up
```

On startup you should see all nine migrations apply in order, ending at
`...add_retraining_pipeline_and_lifecycle_stage`, followed by
`crucible.migrations_applied`. If migration fails, the container stops before
starting the server and prints `crucible.migrations_failed: ...` at the top of
the logs — that line is the real error to act on.

> The named volume is `crucible_crucible-data` (Docker prefixes it with the
> project folder). If `docker volume rm` says "no such volume," it's already
> gone — just run `docker compose up`.
