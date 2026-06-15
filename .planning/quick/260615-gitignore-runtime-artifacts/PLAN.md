---
quick_id: 260615-gitignore-runtime-artifacts
created: 2026-06-15
type: quick
---

# Quick Task: gitignore runtime artifacts

## Objective

Stop host-side runtime artifacts from polluting the repo while keeping the
frozen drift reference snapshots tracked for fresh-clone reproducibility.

## Tasks

1. Add to `.gitignore`:
   - `mlflow.db` / `mlflow.db-journal` — host-side SQLite MLflow store (the
     compose stack uses Postgres; this file is a stray local/test artifact).
   - `data/monitoring/forecast_vs_realized.parquet`, `data/monitoring/*_drift.html`,
     `data/monitoring/*_drift.json` — regenerated every monitoring run.
2. Keep `data/monitoring/reference/` tracked; stage the previously-untracked
   `4_reference.parquet` for consistency with the already-committed `3_reference.parquet`.
3. Verify `git check-ignore mlflow.db` reports ignored; commit.
