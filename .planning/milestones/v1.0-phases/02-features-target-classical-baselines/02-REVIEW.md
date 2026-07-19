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

## Fix Status (2026-06-11)

Scope: CR-01 + WR-01..WR-04 (Critical + Warning). Info findings IN-01/03/04 out of scope; IN-02 resolved as a corollary of CR-01. Suite green at every commit (`uv run pytest tests/ -q`: 297 passed, 2 skipped; ruff format + check clean).

| ID | Status | Commit | Fix |
|----|--------|--------|-----|
| CR-01 | fixed | e40b6a8, 2d17d0b | EWMA drops the extra `.shift(1)`; GARCH trains on `iloc[:pos+1]`; HAR-RV tail is `iloc[:t+1]` — forecast[t] now uses data ≤ t and is scored against RV[t+1]. Value-level alignment tests added (`tests/unit/test_baseline_alignment.py`). Reports + CSV regenerated with real data (aligned baselines score better: BTC EWMA QLIKE 2.0215 → 1.9500); feature parquets regenerated, DVC pointer committed + pushed. |
| WR-01 | fixed | 020fb6e, 2d17d0b | `build_features` passes `suffix=f"_{asset_name}"` to `as_of_join`; cross-asset columns are per-source (`rv_22_btc`, `rv_22_eth`), stable and asset-identifying. Two-source collision + dict-order-invariance regression test added. Feature parquets regenerated. |
| WR-02 | fixed | 26b7d4e | `qlike` floors BOTH inputs at `QLIKE_FLOOR` and raises `ValueError` on a non-finite result. Zero-rv, floored-equivalence, and NaN/inf-raise tests added. |
| WR-03 | fixed | d72dafe | Bare asserts replaced with unconditional raises of `GarchConvergenceError`/`GarchNonStationaryError` (base `GarchFitError`); `forecast_path` catches exactly that family for the fallback chain. Hierarchy + fallback-behavior tests added. |
| WR-04 | fixed | 89da68d | Warnings suppression scoped to arch categories (`ConvergenceWarning`, `DataScaleWarning`, `StartingValueWarning`); fallback catches narrowed to `GarchFitError` + `np.linalg.LinAlgError` (fit) and `(ValueError, LinAlgError)` (re-forecast) — real bugs (e.g. TypeError) now propagate. Bug-propagation + warning-scoping tests added. |
| IN-02 | fixed (corollary) | e40b6a8 | EWMA feature (`ewma[t]`) and EWMA baseline forecast (`ewma[t]`) now agree — same quantity, same day. |
| IN-01 | not fixed (out of scope) | — | Info: harness drops final partial window; docstring mismatch. |
| IN-03 | not fixed (out of scope) | — | Info: GARCH-as-feature docstring says "filtered" but implementation forecasts. |
| IN-04 | not fixed (out of scope) | — | Info: `compute_target(horizon>1)` duplicates `forward_realized_var`. |

_Fixer: Claude (gsd-code-fixer)_
