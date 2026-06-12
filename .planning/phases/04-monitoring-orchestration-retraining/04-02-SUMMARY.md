---
phase: 04-monitoring-orchestration-retraining
plan: "02"
subsystem: monitoring
tags: [evidently, drift-detection, reference-snapshot, tdd]
dependency_graph:
  requires: [04-01]
  provides: [MON-02]
  affects: [04-05-dashboard]
tech_stack:
  added:
    - "evidently==0.7.21 (pinned >=0.7.21,<0.8)"
  patterns:
    - "Evidently 0.7 Report/Dataset/DataDefinition API (not legacy ColumnMapping)"
    - "TDD RED/GREEN cycle for drift adapter"
    - "Frozen reference snapshot versioned by champion model version"
key_files:
  created:
    - src/volforecast/monitoring/drift.py
    - scripts/create_reference_snapshot.py
    - tests/unit/test_drift.py
    - data/monitoring/reference/3_reference.parquet
  modified:
    - pyproject.toml
    - uv.lock
decisions:
  - "Evidently result.save_html/save_json called on result object (report.run() return value), not on Report template — avoids Pitfall 1 AttributeError"
  - "DataDefinition(numerical_columns=[...]) always passed explicitly — avoids Pitfall 2 mis-categorisation"
  - "Drift adapter is strictly report-only: no MLflow alias flip, no retrain flag — locked anti-pattern per CONTEXT.md"
  - "Reference snapshot filename embeds champion version (3_reference.parquet); script refuses to overwrite same-version file — frozen reference contract (Pitfall 6)"
  - "champion_version='3' used directly for snapshot creation (MLflow server not running in worktree; documented in script docstring)"
  - "Evidently package legitimacy auto-approved per session instructions: evidently 0.7.21 = EvidentlyAI OSS (github.com/evidentlyai/evidently, 17k stars, ~500k downloads/month, 4+ year history)"
metrics:
  duration: "~10 minutes"
  completed: "2026-06-13"
  tasks_completed: 2
  files_created: 4
  files_modified: 2
---

# Phase 04 Plan 02: Evidently Distribution-Drift Adapter Summary

One-liner: Evidently 0.7.21 drift adapter with version-keyed frozen reference snapshot, dated HTML+JSON output, report-only (no promotion path).

## Tasks Completed

| # | Task | Commit | Key Files |
|---|------|--------|-----------|
| 1 | Add evidently dependency and create frozen reference snapshot script | 90a8644 | pyproject.toml, uv.lock, scripts/create_reference_snapshot.py, data/monitoring/reference/3_reference.parquet |
| 2 (RED) | Failing drift adapter tests | d7af5db | tests/unit/test_drift.py |
| 2 (GREEN) | Evidently distribution-drift adapter implementation | b901580 | src/volforecast/monitoring/drift.py |

## What Was Built

### `src/volforecast/monitoring/drift.py`

Thin Evidently 0.7 adapter with two public functions:

- `run_distribution_drift(reference_df, current_df, numerical_columns, output_dir, date_str) -> dict` — builds `DataDefinition(numerical_columns=...)`, wraps both frames with `Dataset.from_pandas`, runs `Report([DataDriftPreset()]).run(current_ds, reference_ds)`, calls `result.save_html` and `result.save_json` on the RESULT object (not the Report template), returns `result.dict()`.
- `select_numerical_columns(df) -> list[str]` — returns numeric-dtype columns excluding known non-feature columns (asset/date/string); callers use this to build the explicit numerical_columns list.

Module docstring declares: report-only, no promotion path, frozen reference read never written.

### `scripts/create_reference_snapshot.py`

One-time frozen reference snapshot creation. Key properties:
- Resolves champion version from MLflow registry (or accepts explicit override).
- Concatenates all 5 training feature parquets (crypto + equity).
- Keeps only numerical feature columns (`select_dtypes(include=["number"])` minus known non-feature cols).
- Writes to `data/monitoring/reference/{version}_reference.parquet`.
- Idempotent on version key: refuses to overwrite an existing same-version snapshot.
- Raises `FileNotFoundError` on missing feature parquets (no silent empty snapshot).

### `data/monitoring/reference/3_reference.parquet`

Frozen reference snapshot for champion version 3: 6,578 rows x 20 numerical feature columns (rv_5, rv_10, rv_22, rv_66, ewma_var, lagged_vol, parkinson_var, gk_var, vol_of_vol, rolling_skew, rolling_kurt, day_of_week, month, is_monday, is_friday, garch_cond_var, rv_22_eth, rv_22_btc, log_return, squared_return). Concatenated from 5 Phase 2 feature parquets.

### `tests/unit/test_drift.py`

Four offline hermetic tests (200-row synthetic frames, seeded RNG, tmp_path output):
1. `test_api_shape_files_created` — returns dict; HTML and JSON files exist and are non-empty.
2. `test_numerical_columns_honored` — `select_numerical_columns` excludes string/datetime; drift run succeeds.
3. `test_result_object_save_methods` — no `AttributeError` on happy path (result object, not Report).
4. `test_drift_detected_on_shifted_data` — shifted distribution produces different summary than identical distribution.

All 4 tests pass in 13–14s.

## TDD Gate Compliance

- RED gate: commit `d7af5db` — `test(04-02)`: 4 failing tests (ModuleNotFoundError on missing drift.py). Confirmed.
- GREEN gate: commit `b901580` — `feat(04-02)`: implementation; all 4 tests pass. Confirmed.
- REFACTOR gate: N/A — implementation was clean; no refactor step needed.

## Acceptance Criteria Status

- [x] `pyproject.toml` declares `evidently>=0.7.21,<0.8`; `import evidently` reports `0.7.21` (within `[0.7.21, 0.8)`).
- [x] Current Evidently API symbols (`Dataset`, `DataDefinition`, `Report`, `DataDriftPreset`) import successfully — no `ColumnMapping` reliance.
- [x] `scripts/create_reference_snapshot.py` parses, resolves the champion version into the snapshot filename, and refuses to overwrite an existing same-version snapshot.
- [x] Script raises `FileNotFoundError` when feature parquets are missing.
- [x] `uv run pytest tests/unit/test_drift.py -q` — 4/4 pass.
- [x] `save_html`/`save_json` invoked on `result` returned by `report.run(...)`, not on the `Report` instance (Pitfall 1 avoided).
- [x] `DataDefinition(numerical_columns=[...])` always passed explicitly (Pitfall 2 avoided).
- [x] Adapter contains no alias-flip or retrain-flag call — strictly report-only.
- [x] `uv run ruff check .` — clean repo-wide.

## Evidently Package Legitimacy Evidence (T-04-SC)

Per session `<checkpoint_handling>` instructions (auto-approved for well-known packages):
- Package name: exactly `evidently` on PyPI (not a typosquat)
- Publisher/org: EvidentlyAI (github.com/evidentlyai/evidently)
- Age: ~4 years (2021 initial release)
- Stars: ~17,000 GitHub stars
- Downloads: ~500,000/month (per RESEARCH.md, PyPI verified)
- Installed version: 0.7.21 — within pinned range `[0.7.21, 0.8)`
- Key API symbols verified importable: `from evidently import Dataset, DataDefinition, Report; from evidently.presets import DataDriftPreset`
- Verdict: LEGITIMATE — established EvidentlyAI OSS project, not a slopsquat

## Deviations from Plan

### Auto-applied adjustments

**1. [Rule 2 - Missing critical functionality] champion_version explicit override parameter**
- **Found during:** Task 1 implementation
- **Issue:** The plan calls for resolving champion version from MLflow, but the MLflow server is not running in the worktree/CI environment.
- **Fix:** Added `champion_version` parameter to `create_reference_snapshot()` so it accepts an explicit override; MLflow resolution is the default path when server is available. The snapshot was generated with `champion_version='3'` directly, matching the known champion from Phase 3 context.
- **Files modified:** `scripts/create_reference_snapshot.py`

No other deviations. Plan executed as written.

## Known Stubs

None — the drift adapter is fully wired and produces real output against real feature data.

## Threat Surface Scan

No new network endpoints, auth paths, or trust boundaries beyond what the plan's `<threat_model>` documented. All four STRIDE mitigations from the plan were implemented:

| Threat ID | Status |
|-----------|--------|
| T-04-SC | Mitigated — evidently pinned, legitimacy verified |
| T-04-04 | Mitigated — no alias flip or retrain flag in drift.py (docstring declares anti-pattern) |
| T-04-05 | Mitigated — snapshot script refuses to overwrite same-version file |
| T-04-06 | Mitigated — numerical_columns always passed explicitly to DataDefinition |

## Self-Check: PASSED

All created files exist and all commits verified:

| Item | Status |
|------|--------|
| `src/volforecast/monitoring/drift.py` | FOUND |
| `scripts/create_reference_snapshot.py` | FOUND |
| `tests/unit/test_drift.py` | FOUND |
| `data/monitoring/reference/3_reference.parquet` | FOUND |
| Commit `90a8644` (feat: evidently + snapshot) | FOUND |
| Commit `d7af5db` (test: RED) | FOUND |
| Commit `b901580` (feat: GREEN) | FOUND |
