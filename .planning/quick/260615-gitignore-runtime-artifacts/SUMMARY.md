---
quick_id: 260615-gitignore-runtime-artifacts
status: complete
completed: 2026-06-15
type: quick
---

# Summary: gitignore runtime artifacts

## What changed

- `.gitignore`: added `mlflow.db` + `mlflow.db-journal` (host-side SQLite MLflow
  store — compose stack uses Postgres), and runtime monitoring outputs
  (`data/monitoring/forecast_vs_realized.parquet`, `*_drift.html`, `*_drift.json`).
- Staged the previously-untracked `data/monitoring/reference/4_reference.parquet`
  so both frozen reference snapshots are version-controlled for fresh-clone drift.

## Verification

- `git check-ignore mlflow.db` → ignored ✓
- `git check-ignore data/monitoring/forecast_vs_realized.parquet` and a sample
  dated `*_drift.html` → both ignored ✓
- `data/monitoring/reference/4_reference.parquet` staged ✓ (3_reference.parquet
  was already tracked)

## Notes

Reference snapshots stay tracked deliberately: drift detection needs a frozen
reference to compare against, and regenerating it requires the training feature
data, so committing the small parquet keeps a clone runnable out of the box.
