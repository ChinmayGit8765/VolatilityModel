---
phase: 04-monitoring-orchestration-retraining
plan: "01"
subsystem: monitoring
tags: [labeller, garch-baseline, feedback-loop, idempotent-parquet, forecast-vs-realized]
dependency_graph:
  requires:
    - src/volforecast/serving/prediction_log.py  # PREDICTION_LOG_SCHEMA contract
    - src/volforecast/features/target.py         # compute_target (single source of realized-var truth)
    - src/volforecast/models/garch.py            # GARCH.forecast_path, GarchFitError
    - src/volforecast/features/estimators.py     # log_returns helper
    - src/volforecast/config.py                  # load_assets, processed_path, project_root
  provides:
    - src/volforecast/monitoring/labeller.py     # FVR table, label_forecasts entry point
    - scripts/run_labeller.py                    # CLI entry for Prefect Plan 05
    - data/monitoring/forecast_vs_realized.parquet  # runtime artifact (append-only)
  affects:
    - .planning/phases/04-monitoring-orchestration-retraining/04-03-PLAN.md  # performance monitor reads FVR
tech_stack:
  added: []
  patterns:
    - idempotent-parquet-append (atomic .tmp.parquet -> os.replace + drop_duplicates(keep="first"))
    - lazy-arch-import (GARCH imported inside functions, not at module load)
    - calendar-correct labelling via compute_target NaN-drop (no weekend row fabrication)
    - T-04-01 input validation (required columns + dtype assertion before join)
key_files:
  created:
    - src/volforecast/monitoring/labeller.py
    - scripts/run_labeller.py
    - tests/unit/test_labeller.py
  modified: []
decisions:
  - "as_of_date derived from timestamp_utc - 1 day (serve wall-clock D → as_of D-1; forecast targets RV[D])"
  - "LABEL_KEY = [asset, as_of_date, model_alias] so champion and garch_baseline rows for same date coexist"
  - "GARCH model_version sentinel = garch_1_1 (not an MLflow version number)"
  - "GarchFitError caught per-asset with log.warning; no garch row emitted for that window (T-04-03 accept)"
  - "label_forecasts raises clear FileNotFoundError on missing predictions.parquet (no silent skip)"
metrics:
  duration: "8 minutes"
  completed: "2026-06-12T16:20:11Z"
  tasks_completed: 2
  files_created: 3
  files_modified: 0
---

# Phase 04 Plan 01: Feedback-Loop Labeller Summary

**One-liner:** Idempotent forecast-vs-realized labeller joining compute_target to prediction log rows, with GARCH(1,1) live-forecast baseline rows in the same FVR table keyed on (asset, as_of_date, model_alias).

## What Was Built

### src/volforecast/monitoring/labeller.py (436 lines)

The core feedback-loop module. Key exports:

- `FVR_SCHEMA` — 7-column monitoring parquet schema: `as_of_date`, `asset`, `horizon`, `model_version`, `model_alias`, `forecast_var`, `realized_var`
- `LABEL_KEY = ["asset", "as_of_date", "model_alias"]` — composite idempotency key; allows champion and garch_baseline rows for the same date to coexist
- `append_labels(new_rows, out_path) -> int` — atomic `.tmp.parquet` → `os.replace` write with `drop_duplicates(keep="first")`; returns net-new row count
- `label_champion_forecasts(predictions, data_root) -> pd.DataFrame` — derives `as_of_date = timestamp_utc - 1 day`, joins to `compute_target`; drops NaN realized rows (no zero-fill)
- `garch_live_forecasts(asset_cfg, data_root, label_dates) -> pd.Series` — runs `GARCH(min_train=252, step=21).forecast_path` restricted to labelable dates; catches `GarchFitError` with warning, returns empty Series on failure
- `label_garch_baseline(asset_cfg, data_root) -> pd.DataFrame` — emits FVR rows with `model_alias="garch_baseline"`, `model_version="garch_1_1"`
- `label_forecasts(data_root, fvr_path) -> int` — top-level entry: gates on prediction log existence (clear `FileNotFoundError`), builds champion + GARCH rows, calls `append_labels`

### scripts/run_labeller.py

Thin CLI entry point mirroring `scripts/generate_features.py` structure. Resolves `data_root = project_root() / "data"`, calls `label_forecasts`, prints `net_new_rows=N`. Exits with code 1 on `FileNotFoundError` (missing prediction log).

### tests/unit/test_labeller.py (6 tests, all passing)

Hermetic, offline, no MLflow/network. Synthetic processed parquets and prediction logs under `tmp_path`. Uses GARCH-after-min_train dates in fixtures to ensure coexistence tests work.

Tests:
1. `test_idempotency` — second `label_forecasts` call adds 0 rows
2. `test_champion_and_garch_coexist` — both `champion` and `garch_baseline` model_alias values present, with at least one overlapping (asset, as_of_date)
3. `test_no_forward_close_produces_no_row` — last processed date yields no champion row (NaN compute_target → dropped)
4. `test_realized_join_correctness` — joined `realized_var` matches `compute_target` within float tolerance
5. `test_garch_forecast_var_decimal_units` — `forecast_var` in [1e-6, 1e-1]; ≥90% in [1e-5, 1e-2] (T-04-02 guard against 100x-scale bug)
6. `test_missing_prediction_log_raises` — `label_forecasts` raises `FileNotFoundError` with clear message

## Verification Evidence

- `uv run pytest tests/unit/test_labeller.py -q` → 6 passed
- `uv run ruff check src/volforecast/monitoring/labeller.py scripts/run_labeller.py tests/unit/test_labeller.py` → All checks passed
- `uv run ruff check src tests scripts` → All checks passed
- Import check: `from volforecast.monitoring.labeller import FVR_SCHEMA, LABEL_KEY, append_labels, label_champion_forecasts` → ok

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing test fixture alignment] Prediction dates in synth_root fixture placed after GARCH min_train window**
- **Found during:** Task 2 RED phase — test_champion_and_garch_coexist failed because champion prediction dates (first 10 labelable dates = Jan 2023) had no overlap with GARCH forecast dates (which require 252 rows = Sept 2023)
- **Fix:** Changed fixture to use `labelable_dates[255:265]` so champion predictions fall in the GARCH-forecast range
- **Files modified:** `tests/unit/test_labeller.py`
- **Commit:** 995be93

None of the plan's core requirements were missed; the deviation was a test fixture calibration.

## Threat Mitigations Applied

| Threat ID | Mitigation Applied |
|-----------|-------------------|
| T-04-01 | `label_champion_forecasts` checks required columns + asserts `forecast_var` is float dtype before join |
| T-04-02 | Test 5 asserts decimal magnitude [1e-6, 1e-1]; GARCH module's `garch_forecast_variance_decimal` does the single de-scale |
| T-04-03 | `GarchFitError` (and generic Exception) caught in `garch_live_forecasts`; warning logged, no row emitted, pipeline continues |

## Known Stubs

None. All values derive from `compute_target` (realized variance) and `GARCH.forecast_path` (forecasts). No placeholder data.

## Threat Flags

None. No new network endpoints, auth paths, or schema changes at trust boundaries beyond what was planned.

## Self-Check: PASSED

- `src/volforecast/monitoring/labeller.py` — FOUND (436 lines, > 80 minimum)
- `scripts/run_labeller.py` — FOUND
- `tests/unit/test_labeller.py` — FOUND
- Commit `a2479bc` (Task 1) — FOUND in git log
- Commit `995be93` (Task 2) — FOUND in git log
- 6/6 tests passing
- ruff clean repo-wide
