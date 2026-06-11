---
phase: 02-features-target-classical-baselines
reviewed: 2026-06-11T00:00:00Z
depth: standard
files_reviewed: 22
findings:
  critical: 1
  warning: 4
  info: 4
  total: 9
status: issues_found
---

# Phase 2: Code Review Report

**Reviewed:** 2026-06-11 | **Depth:** standard | **Files:** 22 | **Status:** issues_found

## Summary

Numerical core well-structured; purge/embargo math correct; GARCH 100x scaling round-trips correctly; QLIKE Patton form correct (`qlike(x,x)==0` holds); GARCH-as-feature truncation invariance holds; merge_asof staleness semantics correct. One Critical alignment defect and four Warnings found. All 276 tests pass — every defect below is latent (suite checks counts/NaN placement, not value pairing).

## Critical

### CR-01: All classical baselines misaligned with target by one day (invalidates the published baseline bar)
**Files:** `src/volforecast/models/ewma.py:106-108`, `src/volforecast/models/garch.py:228,247`, `src/volforecast/models/har_rv.py:235`, `src/volforecast/reports/baseline.py:194-204`
Target is `compute_target[t] = RV[t+1]` (as-of t, data ≤ t). But: EWMA `forecast_path` returns `ewma_variance(lr).shift(1)` → `forecast[t] = ewma[t-1]` (data ≤ t-1); GARCH trains on `iloc[:pos]` (data ≤ pos-1); HAR-RV tail `iloc[:t]` (data ≤ t-1). All three produce a forecast of `RV[t]` from data ≤ t-1 but are scored against `RV[t+1]` — internally consistent with each other but systematically handicapped one day of information vs the Phase-3 ML model whose features are as-of t. Biases the ML-vs-baseline comparison in ML's favor. Empirically confirmed (simulated GARCH series: aligned EWMA scores better).
**Fix:** at index t each `forecast_path` must forecast `RV[t+1]` using data ≤ t: EWMA — drop the extra `.shift(1)`; GARCH — train on `iloc[:pos+1]`; HAR-RV — tail `iloc[:t+1]`. Add a VALUE-level alignment test (not count test). Regenerate reports/baseline_eval.md + CSV.

## Warnings

### WR-01: Cross-asset join silently collides for multiple sources, loses asset identity
**Files:** `src/volforecast/features/pipeline.py:256-261`, `src/volforecast/features/cross_asset.py:117-128`
`build_features` discards the dict key and every source is suffixed `_xasset`; two sources sharing a feature name produce `rv_22_xasset_x`/`_y` with dict-order-dependent names (verified). Breaks FEAT-07 column-name stability.
**Fix:** suffix per source: `as_of_join(out, source_df, feature_cols=..., suffix=f"_{asset_name}")`.

### WR-02: `qlike` floors only forecast, not realized — silent inf/nan for the shared promotion-gate metric
**File:** `src/volforecast/eval/metrics.py:76-79`
`rv_var=0` → `log(0)` → `+inf`; NaN propagates. Report path pre-filters, but Phase-4 promotion gate callers may not.
**Fix:** floor both inputs with `QLIKE_FLOOR`; raise ValueError on non-finite result.

### WR-03: Convergence/stationarity enforcement via bare `assert` (stripped under python -O)
**File:** `src/volforecast/models/garch.py:108-113,233-242`
**Fix:** dedicated exceptions (GarchConvergenceError/GarchNonStationaryError) raised unconditionally; catch those in forecast_path.

### WR-04: Overly broad `except (AssertionError, Exception)` + blanket `warnings.simplefilter("ignore")` launder real bugs into "fallbacks"
**File:** `src/volforecast/models/garch.py:104-106,235`
**Fix:** catch only the WR-03 exceptions + arch-specific errors; scope warnings filter to specific arch categories.

## Info

### IN-01: Harness silently drops final partial test window (docstring says it emits it)
`src/volforecast/eval/harness.py:114` — guard requires full step; `min(...)` is dead code. Fix guard or docstring.

### IN-02: EWMA feature (`ewma[t]`) and EWMA baseline (`ewma[t-1]`) disagree by one day on the same quantity
Corroborating symptom of CR-01; resolving CR-01 reconciles both.

### IN-03: GARCH-as-feature docstring says "filtered conditional variance" but implementation forecasts
`src/volforecast/features/pipeline.py:14-15,115-151` — no leakage, but docs misrepresent the feature. Update docstring or switch.

### IN-04: `compute_target(horizon>1)` duplicates `forward_realized_var` logic
`src/volforecast/features/target.py:66-72` vs `78-104` — have one delegate to the other.

---
_Reviewer: gsd-code-reviewer (returned inline; artifact written by orchestrator)_
