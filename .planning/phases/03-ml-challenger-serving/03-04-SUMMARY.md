---
phase: 03-ml-challenger-serving
plan: "04"
subsystem: serving
tags: [fastapi, mlflow, lightgbm, docker, prediction-log, pydantic-v2, tdd]
dependency_graph:
  requires: [03-02]
  provides: [SERVE-01, SERVE-02, SERVE-03]
  affects: [phase-04-monitoring]
tech_stack:
  added: [fastapi==0.136.3, uvicorn==0.49.0, python:3.12-slim + libgomp1 Dockerfile]
  patterns:
    - FastAPI async lifespan for model load
    - Atomic parquet append (tmp -> os.replace) with threading.Lock
    - Native LGBMRegressor bypass of pyfunc schema validation for CategoricalDtype
    - ASSET_DTYPE + reindex-to-model-columns serving pattern
key_files:
  created:
    - src/volforecast/serving/schemas.py
    - src/volforecast/serving/prediction_log.py
    - src/volforecast/serving/app.py
    - infra/api/Dockerfile
    - tests/unit/test_prediction_log.py
    - tests/integration/test_api.py
    - tests/integration/__init__.py
  modified:
    - infra/docker-compose.yml
decisions:
  - "Use native LGBMRegressor.predict() instead of pyfunc.predict() to avoid schema coercion failure on CategoricalDtype; pyfunc wrapper expects string for 'asset' but LightGBM booster requires category dtype"
  - "Added VOLFORECAST_DATA_ROOT env var to separate code root (/app) from data root (/data bind mount) in container"
  - "Reindex serving rows to full 21-column model schema (matching evaluate_per_asset) to match training pooled column union"
  - "service_started (not healthcheck) for mlflow-server depends_on since mlflow has no healthcheck"
metrics:
  duration_minutes: 119
  tasks_completed: 4
  files_created: 7
  files_modified: 1
  completed_date: 2026-06-11
---

# Phase 03 Plan 04: FastAPI Serving Slice + Prediction Log Summary

**One-liner:** FastAPI champion-model inference service (lifespan load, /health + /forecast + /forecast/{symbol}), atomic-parquet prediction log, python:3.12-slim+libgomp1 Dockerfile, and docker-compose api service — all verified live.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 (RED) | Prediction log + schemas tests | 4954507 | tests/unit/test_prediction_log.py |
| 1 (GREEN) | Prediction log + schemas implementation | acabafc | serving/schemas.py, serving/prediction_log.py |
| 2 (RED) | FastAPI app integration tests | 79ef23e | tests/integration/test_api.py |
| 2 (GREEN) | FastAPI app implementation | 80783a5 | serving/app.py |
| 3 | Dockerfile + docker-compose | 79c7d73 | infra/api/Dockerfile, infra/docker-compose.yml |
| fix | Resolve pyfunc schema mismatch + Path import | 4512f20 | serving/app.py, Dockerfile, compose |

## Verification Evidence

### /health
```json
{"status":"ok","model_name":"volforecast-lgbm","model_version":"3","alias":"champion"}
```

### /forecast (all 5 assets)
```json
{
  "forecasts": [
    {"asset":"BTC-USD","forecast_var":0.000187,"forecast_vol":0.01366,"horizon_days":1,"as_of_date":"2026-06-09","model_version":"3","alias":"champion"},
    {"asset":"ETH-USD","forecast_var":0.000612,"forecast_vol":0.02473,"horizon_days":1,"as_of_date":"2026-06-09","model_version":"3","alias":"champion"},
    {"asset":"SPY","forecast_var":0.0000517,"forecast_vol":0.00719,"horizon_days":1,"as_of_date":"2026-06-09","model_version":"3","alias":"champion"},
    {"asset":"AAPL","forecast_var":0.000161,"forecast_vol":0.01270,"horizon_days":1,"as_of_date":"2026-06-09","model_version":"3","alias":"champion"},
    {"asset":"MSFT","forecast_var":0.0000965,"forecast_vol":0.00983,"horizon_days":1,"as_of_date":"2026-06-09","model_version":"3","alias":"champion"}
  ],
  "generated_at":"2026-06-11T12:02:10Z"
}
```

### /forecast/SPY (single asset)
```json
{"forecasts":[{"asset":"SPY","forecast_var":0.0000517,"forecast_vol":0.00719,"horizon_days":1,"as_of_date":"2026-06-09","model_version":"3","alias":"champion"}],"generated_at":"2026-06-11T12:02:17Z"}
```

### /forecast/DOGE (unknown symbol)
HTTP 404

### Prediction log (after 3 calls: /forecast x1, /forecast/SPY x1, /forecast/BTC-USD x1)
```
Columns: ['timestamp_utc', 'asset', 'horizon', 'forecast_var', 'model_version', 'alias']
Rows: 7
```
Schema matches Phase 4 contract exactly.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] FileNotFoundError: config/assets.yaml missing in container**
- **Found during:** Task 3 live verification
- **Issue:** VOLFORECAST_ROOT was not set; `load_assets()` couldn't find config/assets.yaml inside the container
- **Fix:** Added `COPY config/ ./config/` to Dockerfile; set `ENV VOLFORECAST_ROOT=/app` and `ENV VOLFORECAST_DATA_ROOT=/data` (separates code root from bind-mounted data root)
- **Files modified:** infra/api/Dockerfile, infra/docker-compose.yml, src/volforecast/serving/app.py

**2. [Rule 1 - Bug] MLflow pyfunc schema validation rejects CategoricalDtype**
- **Found during:** Task 3 live verification  
- **Issue:** pyfunc.predict() enforces model signature — asset column registered as `string`, but passing `CategoricalDtype` raises `Can not safely convert category to <U0`. Passing plain `string` causes LightGBM internal error: `categorical_feature do not match`
- **Fix:** Extract native `LGBMRegressor` from pyfunc wrapper (`pyfunc._model_impl.lgb_model`); predict directly on native model with ASSET_DTYPE + reindexed to full 21-column training schema (NaN-fill missing cross-asset cols). This mirrors `evaluate_per_asset` reindex logic exactly
- **Files modified:** src/volforecast/serving/app.py
- **Why not Rule 4:** This is a serving correctness issue, not an architectural change — the native model access pattern is documented in the codebase (Pitfall 6 note) and uses the same model object

**3. [Rule 1 - Bug] Missing `from pathlib import Path` import in app.py**
- **Found during:** Task 3 live verification (NameError in container)
- **Fix:** Added `from pathlib import Path` to app.py imports
- **Files modified:** src/volforecast/serving/app.py

## FEAT-07 Compliance

`app.py::_forecast_for` imports and calls `build_features(raw_df, cross_asset_dfs=..., include_garch=True)` — the identical function used in `scripts/generate_features.py` and training. No second feature implementation exists. Cross-asset wiring (BTC rv_22 -> non-BTC; ETH rv_22 -> BTC) matches `scripts/generate_features.py` exactly.

## Security Mitigations Applied

| Threat ID | Mitigation | Status |
|-----------|-----------|--------|
| T-03-09 | symbol validated against KNOWN_ASSETS BEFORE any filesystem access | Applied |
| T-03-10 | HTTPException detail strings are generic (no stack traces, no paths) | Applied |
| T-03-11 | Only `@champion` alias loaded from project's own registry | Applied |
| T-03-12 | ports: 127.0.0.1:8000:8000 (loopback-only) | Applied |

## Test Results

- `uv run pytest tests/unit/test_prediction_log.py -q` — 8 passed
- `uv run pytest tests/integration/test_api.py -q` — 8 passed
- `uv run ruff check .` — clean
- Docker build: success (libgomp1 installed, all 247 packages, uvicorn starts)
- Live: /health 200, /forecast 5 assets, /forecast/SPY 1 asset, /forecast/DOGE 404, prediction log 7 rows with correct schema

## Known Stubs

None — all endpoints return real model predictions from volforecast-lgbm@champion (version 3).

## Threat Flags

None — no new network surfaces or auth paths introduced beyond the planned api service.

## Self-Check: PASSED

Files verified:
- src/volforecast/serving/schemas.py — exists
- src/volforecast/serving/prediction_log.py — exists
- src/volforecast/serving/app.py — exists
- infra/api/Dockerfile — exists
- infra/docker-compose.yml — api service present
- tests/unit/test_prediction_log.py — exists
- tests/integration/test_api.py — exists

Commits verified:
- 4954507 — test(03-04): add failing tests for prediction log + Pydantic schemas (RED)
- acabafc — feat(03-04): Pydantic schemas + atomic prediction-log append (GREEN)
- 79ef23e — test(03-04): add failing integration tests for FastAPI app (RED)
- 80783a5 — feat(03-04): FastAPI app with lifespan champion load + /health /forecast routes (GREEN)
- 79c7d73 — feat(03-04): serving Dockerfile + api service in docker-compose
- 4512f20 — fix(03-04): resolve pyfunc schema mismatch + missing config/Path imports
