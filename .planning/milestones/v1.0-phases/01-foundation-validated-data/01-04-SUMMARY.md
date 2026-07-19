---
phase: 01-foundation-validated-data
plan: "04"
subsystem: infra-ci-pipeline-seal
tags:
  - docker-compose
  - mlflow
  - prefect
  - postgres
  - github-actions
  - ci
  - dvc
  - validate_asset
  - pipeline
  - windows11
dependency_graph:
  requires:
    - 01-02  # ingest adapters (cli.py, crypto/equity adapters)
    - 01-03  # validate_asset gate (validate/__init__.py, schemas, checks)
  provides:
    - docker-compose stack (Postgres + MLflow + Prefect) with named volumes
    - GitHub Actions CI (fixture-only, VOLFORECAST_NO_LIVE_API=1)
    - validate_asset gate wired into full ingest pipeline for both asset classes
    - processed_path() for data/processed/{cls}/{slug}.parquet
    - data/processed.dvc DVC tracking pointer
    - README.md with full-stack run documentation
  affects:
    - src/volforecast/cli.py (validate_asset gate for all assets)
    - src/volforecast/config.py (processed_path added)
    - tests/unit/test_pipeline.py (new end-to-end pipeline tests)
tech_stack:
  added:
    - docker-compose (postgres:16, ghcr.io/mlflow/mlflow:v3.13.0, prefecthq/prefect:3-latest)
    - psycopg2-binary 2.9.12 (PostgreSQL adapter for MLflow/Prefect)
    - GitHub Actions astral-sh/setup-uv@v8
  patterns:
    - postgres named volume (no bind mount — avoids Windows NTFS SQLite locking)
    - mlflow-entrypoint.sh with --allowed-hosts '*' (MLflow 3.x security middleware workaround)
    - postgres-init/01-create-databases.sh (create prefectdb alongside mlflowdb on first start)
    - validate_asset gate before processed write (fails closed — no invalid data promoted)
    - VOLFORECAST_NO_LIVE_API=1 in CI (prevents any live API call)
key_files:
  created:
    - infra/docker-compose.yml
    - infra/.env.example
    - infra/mlflow-entrypoint.sh
    - infra/postgres-init/01-create-databases.sh
    - .github/workflows/ci.yml
    - README.md
    - data/processed.dvc
    - tests/unit/test_pipeline.py
  modified:
    - src/volforecast/cli.py (validate_asset gate wired for both asset classes)
    - src/volforecast/config.py (processed_path() added)
    - data/.gitignore (DVC adds /processed)
decisions:
  - "MLflow 3.x entrypoint script: MLflow 3.x security middleware ignores --host 0.0.0.0 and
    binds to 127.0.0.1 by default when the --allowed-hosts flag is embedded in a docker-compose
    YAML bash -c string due to quote escaping issues. Fixed with a dedicated mlflow-entrypoint.sh
    where the shell quoting is unambiguous."
  - "Postgres host port 5433 (not 5432): port 5432 was already in use by another project's
    Postgres container on the dev machine. Changed host binding to 5433; Docker-internal port
    remains 5432 so MLflow/Prefect service URIs are unchanged."
  - "postgres-init script for prefectdb: Prefect server requires prefectdb to exist before
    startup. Added infra/postgres-init/01-create-databases.sh mounted to
    /docker-entrypoint-initdb.d; Postgres runs it on first volume init."
  - "validate_asset gate: both crypto and equity assets are now gated through validate_asset
    before being promoted to data/processed/. Raw parquet is always written. The gate fails
    closed — rejected assets produce a quarantine CSV but no processed parquet."
metrics:
  duration_minutes: 204
  completed_date: "2026-06-10"
  tasks_completed: 4
  tasks_total: 4
  files_created: 9
  files_modified: 3
---

# Phase 01 Plan 04: Infra + CI + Pipeline Seal Summary

**One-liner:** Postgres-backed MLflow 3.13 + Prefect 3 compose stack, GitHub Actions
fixture-only CI with `VOLFORECAST_NO_LIVE_API=1`, and validate_asset gate wired into
the full ingest pipeline (both crypto and equity) sealing `data/processed/` as a
trusted-data store, DVC-tracked.

---

## Tasks Completed

| Task | Name | Commit | Key Outputs |
|------|------|--------|-------------|
| 1 | docker-compose stack + .gitattributes | 37806b6 | infra/docker-compose.yml, infra/.env.example |
| 2 | Compose bring-up + psycopg2-binary checkpoint | df8c881 | mlflow-entrypoint.sh, postgres-init/, port 5433 fix |
| 3 | Wire validate_asset into pipeline | f3f65b0 | cli.py updated, config.py processed_path, test_pipeline.py |
| 4 | GitHub Actions CI + DVC processed tracking + README | 044ce0c | .github/workflows/ci.yml, data/processed.dvc, README.md |

---

## What Was Built

### Infrastructure (Task 1 + Task 2 fixes)

`infra/docker-compose.yml` defines a four-service stack:
- **postgres:16** — shared Postgres backend with `postgres_data` named volume and
  `pg_isready` healthcheck. Host port 5433 (5432 internal). Runs
  `infra/postgres-init/01-create-databases.sh` on first start to create `prefectdb`
  alongside `mlflowdb`.
- **mlflow-server ghcr.io/mlflow/mlflow:v3.13.0** — tracking server + model registry
  on port 5000. Uses `infra/mlflow-entrypoint.sh` which installs psycopg2-binary and
  starts MLflow with `--allowed-hosts '*'` to allow Docker-host access (MLflow 3.x
  security middleware fix). Backend URI: `postgresql://...@postgres:5432/mlflowdb`
  (synchronous psycopg2, no +asyncpg).
- **prefect-server prefecthq/prefect:3-latest** — orchestration server on port 4200.
  Backend URI: `postgresql+asyncpg://...@postgres:5432/prefectdb` (separate database).
- **prefect-worker** — polls `local-pool` work pool.

`infra/.env.example` contains placeholder credentials only; `infra/.env` is gitignored.

### Checkpoint verification evidence

- psycopg2-binary 2.9.12 confirmed installed from PyPI (source: github.com/psycopg/psycopg2,
  ~30M weekly downloads per RESEARCH Package Legitimacy Audit — pre-approved in session).
- `curl http://localhost:5000/health` returns `OK` (MLflow)
- `curl http://localhost:4200/api/health` returns `true` (Prefect)
- `docker compose -f infra/docker-compose.yml ps` shows all 4 services running

### Pipeline (Task 3)

`src/volforecast/config.py` gained `processed_path(asset, data_root)` returning
`data/processed/{asset_class}/{slug}.parquet`.

`src/volforecast/cli.py` refactored into `_ingest_single_asset()` which handles both
crypto and equity assets through a unified pipeline:
1. Fetch from adapter
2. Write raw parquet (unconditional)
3. Call `validate_asset(df, asset_class, quarantine_dir)` — BOTH asset classes
4. On success: write validated df to `processed_path` via `incremental_update`
5. On failure: quarantine CSV written, processed parquet skipped (gate fails closed), rc=1

Two offline pipeline tests in `tests/unit/test_pipeline.py`:
- `test_pipeline_promotes_clean_asset_to_processed`: clean crypto fixture → processed parquet exists, no quarantine
- `test_pipeline_quarantines_and_skips_bad_asset`: weekend-bad equity fixture → processed parquet absent, quarantine CSV exists

### CI (Task 4)

`.github/workflows/ci.yml` triggers on push + pull_request with a single `lint-and-test` job:
- `astral-sh/setup-uv@v8` with Python 3.12, enable-cache
- `uv sync --locked --dev` for reproducible installs
- `uv run ruff check .` + `uv run ruff format --check .`
- `uv run pytest tests/ -x -q` with `VOLFORECAST_NO_LIVE_API: "1"`

### DVC tracking (Task 4)

`data/processed.dvc` tracks the 5-asset processed parquet directory.
`dvc status data/processed.dvc` is clean. `dvc checkout` round-trips confirmed.
`data/.gitignore` gitignores `/raw` and `/processed` (managed by DVC).

### README (Task 4)

`README.md` with `## Running the stack` section documenting the complete workflow:
docker compose up → uv sync --dev → volforecast ingest → dvc checkout.

---

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Port 5432 conflict with existing Postgres container**
- **Found during:** Task 2 checkpoint verification
- **Issue:** `docker compose up -d` failed with `Bind for 0.0.0.0:5432 failed: port is
  already allocated`. Another project's Postgres container had port 5432 bound.
- **Fix:** Changed postgres host port binding from 5432 to 5433. Docker-internal port
  remains 5432 so all service URIs (`postgres:5432`) are unaffected.
- **Files modified:** infra/docker-compose.yml
- **Commit:** df8c881

**2. [Rule 3 - Blocking] Prefect startup fails — `database "prefectdb" does not exist`**
- **Found during:** Task 2 checkpoint verification
- **Issue:** Prefect server logged `asyncpg.exceptions.InvalidCatalogNameError:
  database "prefectdb" does not exist` and exited with `Application startup failed`.
  The compose pattern creates `mlflowdb` via `POSTGRES_DB` but not `prefectdb`.
- **Fix:** Added `infra/postgres-init/01-create-databases.sh` (mounted to
  `/docker-entrypoint-initdb.d`) that creates `prefectdb` alongside `mlflowdb` on
  first Postgres volume initialization.
- **Files modified:** infra/docker-compose.yml, infra/postgres-init/01-create-databases.sh (new)
- **Commit:** df8c881

**3. [Rule 1 - Bug] MLflow 3.x security middleware binds to 127.0.0.1 despite `--host 0.0.0.0`**
- **Found during:** Task 2 checkpoint verification (`curl http://localhost:5000/health`
  returned `curl: (52) Empty reply from server`)
- **Issue:** MLflow 3.13 introduced a security middleware that defaults to localhost-only
  binding. The `--host 0.0.0.0` flag alone is insufficient; `--allowed-hosts '*'` must
  also be set. Embedding this in the docker-compose YAML `bash -c "..."` string led to
  shell quoting issues (`'*'` literal passed, `"*"` glob-expanded in bash context).
- **Fix:** Extracted the startup command to `infra/mlflow-entrypoint.sh` where the shell
  quoting is unambiguous (`--allowed-hosts '*'` in a regular shell heredoc context).
  Changed docker-compose command to `["bash", "/mlflow-entrypoint.sh"]`.
- **Files modified:** infra/docker-compose.yml, infra/mlflow-entrypoint.sh (new)
- **Commit:** df8c881

---

## Threat Surface Scan

No new security-relevant surface beyond what is documented in the plan's threat model:
- T-04-01: Postgres credentials — only `.env.example` committed, `.env` gitignored ✓
- T-04-02: SQLite bind-mount — Postgres with named volumes used ✓
- T-04-03: CRLF line endings — `.gitattributes` enforced (from plan 01-01) ✓
- T-04-04: CI live API — `VOLFORECAST_NO_LIVE_API: "1"` in CI workflow ✓
- T-04-05: Dependency drift — `uv sync --locked` in CI ✓
- T-04-06: psycopg2-binary — package confirmed legitimate, 2.9.12 installed ✓
- T-04-07: MLflow/Prefect exposed ports — localhost-only, local dev stack only ✓

The MLflow `--allowed-hosts '*'` warning is intentional for a local dev stack (T-04-07
accepted). This configuration must NOT be used for any public/cloud deployment.

---

## Known Stubs

None — all pipeline components are fully wired and producing real output (raw and
processed parquet for all 5 assets).

---

## Self-Check

### Files exist

- infra/docker-compose.yml: FOUND
- infra/.env.example: FOUND
- infra/mlflow-entrypoint.sh: FOUND
- infra/postgres-init/01-create-databases.sh: FOUND
- .github/workflows/ci.yml: FOUND
- README.md: FOUND
- data/processed.dvc: FOUND
- tests/unit/test_pipeline.py: FOUND
- src/volforecast/cli.py (modified): FOUND
- src/volforecast/config.py (processed_path): FOUND

### Commits exist

- 37806b6: FOUND (feat(01-04): Postgres-backed MLflow + Prefect docker-compose stack)
- df8c881: FOUND (fix(01-04): resolve compose stack startup issues on Windows 11)
- f3f65b0: FOUND (feat(01-04): wire validate_asset into full pipeline + processed_path)
- 044ce0c: FOUND (feat(01-04): CI workflow, processed-data DVC tracking, README run section)

### Test results

All 29 tests pass (pytest tests/ -q; 15s; fixture-only, offline).

### Acceptance criteria verification

- `docker compose -f infra/docker-compose.yml config` exits 0: VERIFIED
- All 4 services defined with postgres_data + mlflow_artifacts named volumes: VERIFIED
- MLflow uses `postgresql://` URI (no +asyncpg): VERIFIED
- Prefect uses `postgresql+asyncpg://` against prefectdb: VERIFIED
- `infra/.env.example` contains only placeholder credentials: VERIFIED
- `.env` gitignored: VERIFIED
- `.gitattributes` contains `*.sh text eol=lf`: VERIFIED (from plan 01-01)
- docker compose stack actually runs on Windows 11 (all 4 services up): VERIFIED
  (MLflow :5000 health OK; Prefect :4200/api/health OK)
- `cli.py` calls `validate_asset` for every asset: VERIFIED (inspect.getsource assertion)
- Clean assets produce processed parquet; rejected assets do NOT: VERIFIED (pytest)
- Both pipeline tests pass offline: VERIFIED
- `uv run ruff check src tests` exits 0: VERIFIED
- `.github/workflows/ci.yml` has setup-uv, uv sync --locked, ruff, pytest, VOLFORECAST_NO_LIVE_API: VERIFIED
- `data/processed.dvc` exists, dvc status clean: VERIFIED
- README.md has `docker compose` section: VERIFIED
- Full test suite (`pytest tests/ -q`) passes: VERIFIED (29 passed)

## Self-Check: PASSED
