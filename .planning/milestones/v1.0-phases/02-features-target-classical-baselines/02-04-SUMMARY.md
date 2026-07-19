---
phase: "02"
plan: "04"
subsystem: features
tags: [feature-engineering, garch-as-feature, cross-asset, no-lookahead, dvc, parquet]
dependency_graph:
  requires: ["02-01", "02-02", "02-03"]
  provides: ["build_features()", "cross_asset.as_of_join()", "data/features/*.parquet"]
  affects: ["Phase 3 ML model input contract", "Phase 4 serving codepath"]
tech_stack:
  added: []
  patterns:
    - "Walk-forward GARCH reuse for conditional-vol feature (models.garch.GARCH.forecast_path)"
    - "pd.merge_asof backward direction with pd.Timedelta('3D') tolerance for cross-asset join"
    - "Integer-count rolling windows throughout (Pitfall 3 guard)"
    - "DVC data/features pointer (mirrors Phase 1 data/processed pattern)"
key_files:
  created:
    - src/volforecast/features/cross_asset.py
    - src/volforecast/features/pipeline.py
    - tests/unit/test_range_estimators.py
    - tests/unit/test_cross_asset.py
    - tests/unit/test_pipeline_features.py
    - tests/unit/test_no_lookahead.py
    - data/features.dvc
    - scripts/generate_features.py
  modified:
    - src/volforecast/features/estimators.py
    - data/.gitignore
decisions:
  - "[FEAT-07] Single build_features() codepath — one definition, importable identically by training and serving"
  - "[GARCH-as-feature] Reuse GARCH.forecast_path() for conditional-vol feature — walk-forward forecast, never fitted on data past as-of t; truncation invariance proven"
  - "[VOL-OF-VOL] Rolling std of rv_22 over 22-period window"
  - "[SKEW/KURT] 22-period rolling window for both rolling_skew and rolling_kurt"
  - "[LAGGED-VOL] rv_22 shifted by 1 period (k=1)"
  - "[GK-EQUITY] Garman-Klass applied to equities with overnight-gap caveat documented in module docstring (Open Q #4)"
  - "[CROSS-ASSET] BTC RV22 joined onto ETH/SPY/AAPL/MSFT; ETH RV22 joined onto BTC"
  - "[DVC] data/features tracked with dvc add data/features (consistent with Phase 1 pattern)"
metrics:
  duration_minutes: 19
  completed_date: "2026-06-11"
  tasks_completed: 3
  tasks_total: 3
  files_created: 8
  files_modified: 2
  tests_added: 124
  tests_total_after: 276
---

# Phase 02 Plan 04: Feature Pipeline + Cross-Asset + Parquets Summary

Single versioned feature codepath (`build_features`) producing a 17/19-column leak-free feature matrix — multi-lookback RV, log/squared returns, EWMA, GARCH conditional vol, Parkinson, Garman-Klass, vol-of-vol, rolling skew/kurt, lagged vol, calendar, and backward as-of cross-asset features — proven no-lookahead by truncation-invariance test and persisted as 5 DVC-tracked parquet files.

## What Was Built

### Task 1: Range Estimators, Vol-of-Vol, Skew/Kurt, Lagged Vol, Calendar

Extended `src/volforecast/features/estimators.py` with seven new functions:

- **`parkinson_var(df)`**: `ln(H/L)^2 / (4 * ln(2))` row-wise; no rolling; strictly positive
- **`garman_klass_var(df)`**: `0.5*ln(H/L)^2 - (2*ln(2)-1)*ln(C/O)^2`; overnight-gap caveat for equities documented (Open Q #4)
- **`vol_of_vol(rv_series, window)`**: `rv_series.rolling(window).std()`; integer count-based
- **`rolling_skew(lr, window)`**: `lr.rolling(window).skew()`
- **`rolling_kurt(lr, window)`**: `lr.rolling(window).kurt()` (excess)
- **`lagged_vol(rv_series, k)`**: `rv_series.shift(k)`; first k rows NaN; no future leakage
- **`calendar_features(df)`**: `day_of_week` (0-6), `month` (1-12), `is_monday`, `is_friday`

Test file: `tests/unit/test_range_estimators.py` — 36 tests; all pass.

### Task 2: Cross-Asset As-Of Join with 3-Day Staleness Rule

Created `src/volforecast/features/cross_asset.py`:

- **`MAX_CROSS_ASSET_STALENESS = pd.Timedelta("3D")`** — single source of truth
- **`as_of_join(left, right, feature_cols, suffix="_xasset")`**: `pd.merge_asof(direction="backward", tolerance=3D)`
  - Pre-renames right columns with suffix BEFORE merge (prevents `_x`/`_y` collision)
  - Sorts both frames ascending and asserts `is_monotonic_increasing` (Pitfall 6 guard)
  - Normalizes unnamed index to "date" column after `reset_index()`
  - Restores UTC DatetimeIndex with name="date"

Test file: `tests/unit/test_cross_asset.py` — 23 tests; all pass.
Key tests: 2-day gap yields value; 3-day gap yields value (inclusive); 4-day gap yields NaN.

### Task 3: Single Feature Codepath + No-Lookahead Test + 5 Parquets

Created `src/volforecast/features/pipeline.py`:

**`build_features(df, cross_asset_dfs=None, include_garch=True) -> pd.DataFrame`** — the single codepath (FEAT-07):
- RV windows: 5, 10, 22, 66 (integer count-based)
- EWMA variance (lambda=0.94)
- Lagged vol (rv_22 shifted 1)
- Parkinson + Garman-Klass (row-wise)
- Vol-of-vol (std of rv_22, 22-period)
- Rolling skew + kurtosis (22-period)
- Calendar features
- GARCH conditional variance: delegates to `GARCH.forecast_path()` (reuses Plan 03 machinery; walk-forward monthly refit; never fitted past as-of t; lazy import guard)
- Cross-asset: `as_of_join` with 3-day cap when `cross_asset_dfs` provided

**GARCH-as-feature implementation**: Uses `GARCH.forecast_path()` directly. At each position t, the feature value is the one-step-ahead forecast from a model trained on `data[0..refit_pos)` where `refit_pos <= t`. Truncation invariance holds because the same refit schedule applies regardless of how many future rows exist beyond t.

**5 feature parquets generated and DVC-tracked**:

| Asset | Rows | Cols | Non-NaN GARCH | Cross-asset input |
|-------|------|------|---------------|-------------------|
| crypto/BTC-USD.parquet | 1621 | 19 | 1369 | eth_rv22_xasset |
| crypto/ETH-USD.parquet | 1621 | 19 | 1369 | btc_rv22_xasset |
| equity/SPY.parquet | 1112 | 19 | 860 | btc_rv22_xasset |
| equity/AAPL.parquet | 1112 | 19 | 860 | btc_rv22_xasset |
| equity/MSFT.parquet | 1112 | 19 | 860 | btc_rv22_xasset |

DVC command: `uv run dvc add data/features` → `data/features.dvc` pointer (5 files, ~1MB, md5 hash).

Test files:
- `tests/unit/test_pipeline_features.py` — 15 tests (column presence, GARCH units/variance check, cross-asset NaN, single codepath import)
- `tests/unit/test_no_lookahead.py` — 11 tests + 2 skipped (fixture too short) covering truncation invariance for rv/ewma/parkinson/lagged/vol-of-vol/skew/calendar and GARCH-as-feature

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Cross-asset column name collision with pandas suffixes**
- **Found during:** Task 3, first cross-asset test
- **Issue:** When `out` and the right source shared a column name (e.g., `rv_22`), `pd.merge_asof` applied pandas default `_x`/`_y` suffixes instead of the intended `_xasset` suffix
- **Fix:** Pre-rename source columns with the target suffix BEFORE passing to `merge_asof`, eliminating the collision entirely
- **Files modified:** `src/volforecast/features/cross_asset.py`
- **Commit:** 6113471

**2. [Rule 1 - Bug] Unnamed index "date" became "index" after reset_index()**
- **Found during:** Task 3, cross-asset NaN test
- **Issue:** When source DataFrames had an unnamed DatetimeIndex, `reset_index()` produced a column named "index" instead of "date", causing `sort_values("date")` to fail with KeyError
- **Fix:** Added `_normalize_date_col()` helper in `as_of_join` to normalize the date column name after reset regardless of original index name
- **Files modified:** `src/volforecast/features/cross_asset.py`
- **Commit:** 6113471

**3. [Rule 1 - Bug] GARCH conditional-vol feature: NaN on truncated series (wrong implementation)**
- **Found during:** Task 3, GARCH truncation-invariance test
- **Issue:** Initial implementation used `res.conditional_volatility` from the in-sample filtered series and assigned values by position offset. This produced NaN on truncated series because the alignment logic was incorrect and the refit schedule boundary differed.
- **Fix:** Replaced with delegation to `GARCH.forecast_path()` — the walk-forward one-step-ahead forecasts are inherently truncation-invariant because they use the same refit schedule regardless of how many future rows exist. This also eliminates a second GARCH implementation (satisfies the "no duplicate" requirement).
- **Files modified:** `src/volforecast/features/pipeline.py`
- **Commit:** 6113471

**4. [Rule 2 - Missing] Output index name set to "date" in build_features**
- **Found during:** Task 3, index match test
- **Issue:** `pd.DataFrame(..., index=df.index)` preserved the input index name as-is. Test helper had unnamed index; real processed parquets have name="date". Added `out_index = df.index.rename("date")` to ensure the output always carries the "date" index name required by `as_of_join`.
- **Files modified:** `src/volforecast/features/pipeline.py`
- **Commit:** 6113471

**5. [Rule 2 - Missing] Test helpers used unnamed index**
- **Found during:** Task 3
- **Issue:** `_ohlc()` test helpers in `test_pipeline_features.py` and `test_no_lookahead.py` used `pd.date_range(...)` without `name="date"`, causing index equality assertions to fail against the named output
- **Fix:** Added `name="date"` to all `pd.date_range(...)` calls in test helpers
- **Files modified:** `tests/unit/test_pipeline_features.py`, `tests/unit/test_no_lookahead.py`
- **Commit:** 6113471

## Known Stubs

None — all feature columns are computed from real data; no hardcoded placeholders.

## Threat Flags

None — all new code is internal data transformation (no new network endpoints, no new auth paths, no user-supplied path strings).

## DVC Tracking

`uv run dvc add data/features` was run from the worktree root after generating the 5 parquet files.

Pointer: `data/features.dvc` (md5 hash: `0ea732b69f541a6187bd608526f963ae.dir`, 5 files, 1,023,740 bytes)

The orchestrator must push the DVC cache from the main tree after merge:
```
dvc push  # or: uv run dvc push
```

## Commits

| Hash | Type | Description |
|------|------|-------------|
| 6e1c11c | feat | Range estimators, vol-of-vol, skew/kurt, lagged vol, calendar features |
| cb2ad11 | feat | Cross-asset backward as-of join with 3-day staleness rule |
| 6113471 | feat | Single build_features() codepath + no-lookahead test |
| da6fa2b | feat | Persist 5 feature parquets + DVC pointer |

## Self-Check: PASSED

All key files exist and all commits are present in git log.

| Check | Status |
|-------|--------|
| src/volforecast/features/estimators.py | FOUND |
| src/volforecast/features/cross_asset.py | FOUND |
| src/volforecast/features/pipeline.py | FOUND |
| tests/unit/test_range_estimators.py | FOUND |
| tests/unit/test_cross_asset.py | FOUND |
| tests/unit/test_pipeline_features.py | FOUND |
| tests/unit/test_no_lookahead.py | FOUND |
| data/features/crypto/BTC-USD.parquet | FOUND |
| data/features/crypto/ETH-USD.parquet | FOUND |
| data/features/equity/SPY.parquet | FOUND |
| data/features/equity/AAPL.parquet | FOUND |
| data/features/equity/MSFT.parquet | FOUND |
| data/features.dvc | FOUND |
| commit 6e1c11c | FOUND |
| commit cb2ad11 | FOUND |
| commit 6113471 | FOUND |
| commit da6fa2b | FOUND |
