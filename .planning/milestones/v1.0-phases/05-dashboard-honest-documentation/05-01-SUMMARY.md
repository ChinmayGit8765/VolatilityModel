---
phase: 05-dashboard-honest-documentation
plan: "01"
subsystem: dashboard
tags: [streamlit, observability, mlops, read-only, empty-state]
dependency_graph:
  requires:
    - 04-05  # Prefect daily flow (populates FVR parquet + drift reports)
    - 04-04  # Promotion gate (volforecast-lgbm@champion alias)
    - 04-03  # Performance monitor (prediction log + FVR schema)
    - 04-02  # Drift adapter (drift JSON outputs)
    - 03-01  # FastAPI /health endpoint
  provides:
    - DASH-01: read-only Streamlit observability dashboard on http://127.0.0.1:8501
  affects:
    - infra/docker-compose.yml (new dashboard service)
    - pyproject.toml (streamlit==1.58.0 dependency added)
tech_stack:
  added:
    - streamlit==1.58.0
  patterns:
    - "data_access module pattern: pure loaders, no Streamlit import, unit-testable offline"
    - "empty-state-first design: every loader returns neutral value, never raises"
    - "read-only contract: grep-verified absence of set_registered_model_alias/to_parquet/write-open"
    - "defensive Evidently dict traversal: recursive _find_dataset_drift + _count_drifted_columns"
    - "stdlib urllib api_health probe: no extra HTTP dep, perf_counter latency measurement"
    - "slim + libgomp1 + split uv-sync + non-root appuser pattern (mirrors infra/api/Dockerfile)"
key_files:
  created:
    - src/volforecast/dashboard/__init__.py
    - src/volforecast/dashboard/data_access.py
    - src/volforecast/dashboard/app.py
    - infra/dashboard/Dockerfile
    - tests/unit/test_dashboard_data_access.py
  modified:
    - pyproject.toml (added streamlit==1.58.0)
    - infra/docker-compose.yml (added dashboard service)
    - uv.lock (resolved streamlit + transitive deps)
decisions:
  - "Stdlib urllib for api_health (no httpx/requests) — keeps zero new heavy deps, perf_counter gives latency"
  - "Defensive Evidently dict traversal (recursive search) — result.dict() schema is nested and Evidently docs do not guarantee stable top-level key paths"
  - "REPORT-ONLY caption in drift panel — locked anti-pattern per 04-CONTEXT.md; dashboard explicitly states drift never implies promotion"
  - "Rebase worktree onto main — worktree was branched from a Phase 3 commit; all Phase 4 source files (labeller, drift, promotion, prediction_log) are required by the dashboard data_access loaders"
metrics:
  duration: "~25 minutes"
  completed: "2026-06-15"
  tasks_completed: 3
  tasks_total: 3
  files_created: 5
  files_modified: 3
  tests_added: 23
  tests_total_passing: 452
---

# Phase 05 Plan 01: Dashboard & Compose Service Summary

One-liner: Read-only Streamlit observability dashboard with 4 locked panels, defensive
empty-state handling for fresh-clone first-run, and a loopback-only compose service
mirroring the api container pattern (slim + libgomp1 + non-root + split uv-sync layers).

## Tasks Completed

| Task | Name | Commits | Files |
|------|------|---------|-------|
| 1 (RED) | Failing data-access tests | 95d184f | tests/unit/test_dashboard_data_access.py, src/volforecast/dashboard/__init__.py |
| 1 (GREEN) | data_access.py implementation | 347509c | src/volforecast/dashboard/data_access.py |
| 2 | Streamlit 4-panel app + streamlit dep | f0078c4 | src/volforecast/dashboard/app.py, pyproject.toml, uv.lock |
| 3 | Dashboard Dockerfile + compose service | f91e2c8 | infra/dashboard/Dockerfile, infra/docker-compose.yml |

## What Was Built

### Task 1 - Read-only data-access layer (data_access.py, 357 lines)

Five pure loader functions with empty-state guards. All data flows through these
loaders; no Streamlit import in this module (unit-testable offline).

- **load_fvr(data_root)** reads data/monitoring/forecast_vs_realized.parquet;
  returns empty DataFrame with correct FVR_SCHEMA columns when absent or malformed.
- **latest_drift_summary(monitoring_dir)** globs *_drift.json, selects
  lexicographically latest file, traverses the Evidently result.dict() payload
  recursively to find dataset_drift bool and count drift_detected entries.
  Returns None on any error.
- **champion_info(tracking_uri)** calls MlflowClient().get_model_version_by_alias
  and get_run; wraps the entire call in try/except; returns None on any failure.
  Never calls set_registered_model_alias (read-only, grep-verified).
- **prediction_stats(data_root)** reads data/predictions/predictions.parquet;
  computes row count, max timestamp, and latest forecast_var per asset.
- **api_health(url, timeout)** uses urllib.request.urlopen with
  time.perf_counter latency; returns {reachable: False} on any exception.

Imports FVR_SCHEMA from volforecast.monitoring.labeller and
PREDICTION_LOG_SCHEMA from volforecast.serving.prediction_log (no re-declared
column lists). 23 offline unit tests covering every empty-state branch.

### Task 2 - Streamlit app (app.py, 252 lines)

Multi-section app with exactly the 4 locked panels from 05-CONTEXT.md:

1. **Forecast vs realized** - asset selector from FVR slugs (falls back to config
   slugs when empty); line chart with champion forecast_var, realized_var, and
   GARCH baseline overlay when present.
2. **Drift status** - report date, dataset_drift flag (error/success/warning),
   drifted-column count, HTML path. Prominent REPORT-ONLY caption (locked
   anti-pattern: drift never implies promotion or retraining).
3. **Live model version** - volforecast-lgbm@champion version, run_id, params,
   metrics via champion_info() loader.
4. **Service stats** - prediction-log row count, last timestamp, per-asset
   forecast_var table; live /health probe with latency from api_health().

10 st.info/st.warning calls (>=4 required). All data exclusively via
data_access loaders; no pd.read_parquet, MlflowClient, or urllib in
app.py code (only in docstring comments). Reads MLFLOW_TRACKING_URI and
API_BASE_URL from env with compose-network defaults.

Dependency: streamlit==1.58.0 added to pyproject.toml (CLAUDE.md pin).

### Task 3 - Container + compose service

**infra/dashboard/Dockerfile** (31 lines):
- python:3.12-slim + libgomp1 (LightGBM transitive dep via MLflow artifacts)
- Split uv-sync layers (mirrors WR-07 api pattern): deps layer cached, then src/ copy
- PYTHONPATH=/app/src, VOLFORECAST_ROOT=/app, VOLFORECAST_DATA_ROOT=/data
- MLFLOW_TRACKING_URI and API_BASE_URL defaults baked in
- Non-root appuser uid 1000 (T-05-04 / mirrors api Dockerfile WR-08)
- CMD: uv run --no-sync streamlit run src/volforecast/dashboard/app.py --server.port 8501 --server.address 0.0.0.0

**infra/docker-compose.yml** - new dashboard service:
- depends_on: mlflow-server (service_started) + api (service_started)
- ports: "127.0.0.1:8501:8501" (loopback-only, T-05-01)
- volumes: ../data:/data (dashboard reads FVR + drift + prediction log)
- MLFLOW_TRACKING_URI, API_BASE_URL, VOLFORECAST_DATA_ROOT env vars

## Acceptance Criteria Verification

| Criterion | Status |
|-----------|--------|
| uv run pytest tests/unit/test_dashboard_data_access.py -x -q passes (23 tests) | PASS |
| Full unit suite (452 passed, 2 skipped) | PASS |
| data_access.py imports FVR_SCHEMA from labeller, PREDICTION_LOG_SCHEMA from prediction_log | PASS |
| No set_registered_model_alias/to_parquet/write-open in data_access.py | PASS |
| No set_registered_model_alias/to_parquet/write-open in app.py | PASS |
| pyproject.toml has streamlit==1.58.0 in dependencies | PASS |
| app.py parses as valid Python | PASS |
| No direct pd.read_parquet/MlflowClient/urllib in app.py code | PASS |
| app.py reads MLFLOW_TRACKING_URI and API_BASE_URL from os.environ | PASS |
| grep -c "st.info|st.warning" app.py returns >= 4 (actual: 10) | PASS |
| compose YAML parses; dashboard service exists with 127.0.0.1:8501:8501 | PASS |
| dashboard service depends_on includes mlflow-server and api | PASS |
| Dockerfile has libgomp1, non-root user, streamlit run CMD on port 8501 | PASS |
| data/monitoring/forecast_vs_realized.parquet path in load_fvr | PASS |
| MlflowClient.get_model_version_by_alias pattern in champion_info | PASS |
| GET {API_BASE_URL}/health via urllib in api_health | PASS |
| dashboard: key in compose services | PASS |

## STRIDE Threat Mitigations Applied

| Threat ID | Mitigation | Verification |
|-----------|------------|--------------|
| T-05-01 | 127.0.0.1:8501:8501 loopback-only port binding | compose YAML, acceptance check |
| T-05-02 | No write operations in code (grep-verified) | grep acceptance criteria |
| T-05-03 | All loaders return neutral empty values; empty-state UI messages | 23 unit tests |
| T-05-04 | Non-root appuser uid 1000 in Dockerfile | Dockerfile inspection |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Worktree rebased onto main to resolve missing Phase 4 sources**
- **Found during:** Task 1 RED phase (pytest collection failed with ModuleNotFoundError)
- **Issue:** This worktree was branched from a Phase 3 commit (f5af0ca). The
  data_access.py module needed to import FVR_SCHEMA from
  volforecast.monitoring.labeller, PREDICTION_LOG_SCHEMA from
  volforecast.serving.prediction_log, and MODEL_NAME from
  volforecast.monitoring.promotion - all files that only exist on main (Phase 4).
  The worktree's src/volforecast/monitoring/ contained only __init__.py.
- **Fix:** git rebase main - cleanly rebased the worktree branch onto main
  (6c23123). No conflicts. All Phase 4 source files now present.
- **Files modified:** No source files modified; git history restructured.
- **Commit:** Rebase (no new commit; HEAD moved to 6c23123 as new base)

## Known Stubs

None - all four panels are wired to real data-access loaders. The dashboard shows
"no data yet" empty-state messages on fresh clone (intended, not a stub).

## Threat Flags

None - no new network endpoints, auth paths, file access patterns, or schema changes
beyond what the STRIDE register (threat_model in the PLAN) already covers.

## Self-Check: PASSED

All created files verified present:
- FOUND: src/volforecast/dashboard/__init__.py
- FOUND: src/volforecast/dashboard/data_access.py
- FOUND: src/volforecast/dashboard/app.py
- FOUND: infra/dashboard/Dockerfile
- FOUND: tests/unit/test_dashboard_data_access.py

All commits verified in git log:
- FOUND: 95d184f (test: RED phase - failing data-access tests)
- FOUND: 347509c (feat: GREEN phase - data_access.py implementation)
- FOUND: f0078c4 (feat: Streamlit app + streamlit==1.58.0 dep)
- FOUND: f91e2c8 (feat: dashboard Dockerfile + compose service)
