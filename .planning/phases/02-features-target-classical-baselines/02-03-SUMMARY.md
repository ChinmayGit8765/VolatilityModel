---
phase: "02"
plan: "03"
subsystem: models/baselines
tags: [garch, har-rv, ewma, walk-forward, evaluation, baselines, tdd]
dependency_graph:
  requires: ["02-01", "02-02"]
  provides: ["GARCH(1,1)-baseline", "HAR-RV-baseline", "three-baseline-report"]
  affects: ["02-04", "03-01"]
tech_stack:
  added:
    - "arch 8.0.0 (GARCH(1,1) fit/forecast, lazy import)"
    - "statsmodels 0.14.6 (HAR-RV OLS via sm.OLS, lazy import)"
  patterns:
    - "100x manual scaling + rescale=False + per-refit convergence/stationarity assertions"
    - "Multi-step-per-refit: forecast(horizon=step) once per monthly refit (Pitfall 5 guard)"
    - "Fallback chain: previous fit params -> EWMA; fallback_count exposed for report transparency"
    - "Corsi (2009) HAR-RV: rv_d / rv_w(rolling(5)) / rv_m(rolling(22)) — MEANS not last values"
    - "Identical folds: walk_forward_splits called once per asset, shared across all 3 models"
    - "Lazy imports: arch/statsmodels inside function bodies; feature pipeline import cost unaffected"
key_files:
  created:
    - src/volforecast/models/garch.py
    - src/volforecast/models/har_rv.py
    - tests/unit/test_garch.py
    - tests/unit/test_har_rv.py
    - tests/unit/test_report_baselines.py
  modified:
    - src/volforecast/models/__init__.py
    - src/volforecast/reports/baseline.py
    - tests/unit/test_report_e2e.py
    - reports/baseline_eval.md
    - reports/baseline_metrics.csv
decisions:
  - "GARCH_SCALE=100.0 module-level constant (never inline) — tested by grep and unit test"
  - "rescale=False to prevent hidden arch auto-rescaling — explicit and testable"
  - "forecast(horizon=step) per refit emits all step daily forecasts (Pitfall 5 guard)"
  - "Fallback: previous ARCHModelResult -> EWMA; fallback_count tracked for report"
  - "HAR-RV uses rolling MEANS for rv_w/rv_m (not last value) per Corsi (2009) spec"
  - "HAR-RV refits every step (OLS microseconds) for methodological consistency"
  - "All 3 baselines score on identical walk_forward_splits per asset (T-02-09)"
  - "test_model_column_is_ewma updated to test_model_column_contains_ewma (02-02 test updated for 02-03 multi-model world)"
metrics:
  duration: "~25 minutes"
  completed_date: "2026-06-11"
  tasks_completed: 3
  files_changed: 10
---

# Phase 2 Plan 03: GARCH(1,1) + HAR-RV Baselines Summary

**One-liner:** GARCH(1,1) with 100x scaling/convergence/stationarity assertions and HAR-RV OLS on 1/5/22 lagged RV components, wired into three-baseline identical-fold report scoring all 5 assets.

## What Was Built

Three-task TDD execution completing the classical baseline bar:

**Task 1: GARCH(1,1) baseline** (`src/volforecast/models/garch.py`)

- `GARCH_SCALE = 100.0` module constant; `fit_garch()` scales returns by 100x before fitting
- `arch_model(scaled, mean="Zero", vol="GARCH", p=1, q=1, rescale=False).fit(disp="off")`
- Per-refit assertions: `convergence_flag == 0` AND `alpha + beta < 1.0` (stationarity)
- `garch_forecast_variance_decimal()` divides by `GARCH_SCALE**2` exactly once to recover decimal variance
- `GARCH.forecast_path()` refits monthly, calls `forecast(horizon=step)` once per refit to produce all step daily forecasts (Pitfall 5 guard)
- Fallback chain: previous `ARCHModelResult` params -> EWMA; `fallback_count` attribute tracked
- `arch` imported lazily inside function bodies — feature pipeline import cost unaffected
- 13 tests covering constants, convergence, scale inversion (~1e-4 magnitude check), multi-step refit, and fallback path

**Task 2: HAR-RV baseline** (`src/volforecast/models/har_rv.py`)

- Corsi (2009) HAR-RV via `statsmodels.api.OLS`
- `build_har_features()`: `rv_d = rv.shift(1)`, `rv_w = rv.shift(1).rolling(5).mean()`, `rv_m = rv.shift(1).rolling(22).mean()` — rolling MEANS not last values (per State of the Art spec)
- `sm.add_constant(X, has_constant="add")` prepends intercept
- `fit_har()` selects training rows by label intersection with the walk-forward window
- `har_forecast()` uses last `[1, rv_d, rv_w_mean, rv_m_mean]` from history tail
- `HARRV.forecast_path()` refits every step; exposes `n_refits`
- No scaling: HAR regresses decimal variance on lagged decimal variance directly
- 17 tests covering regressor construction (rolling mean verification), unit consistency, walk-forward refit count, and finite forecasts

**Task 3: Three-baseline report** (`src/volforecast/reports/baseline.py`)

- Extended `generate_baseline_report` to score EWMA, GARCH, and HAR-RV on **identical** `walk_forward_splits` per asset (T-02-09 guard: splits computed once, shared across all three models)
- All three forecast paths pre-computed; then a single split loop collects targets + three forecast vectors, applying valid_mask once across all models
- `garch_fallbacks` column in CSV (populated for GARCH rows, empty for EWMA/HAR)
- Markdown report documents all three baselines with model descriptions and identical-folds transparency note
- 11 tests: identical n_forecasts per asset, 3 rows per asset in CSV, all metrics finite/non-negative
- Updated `test_model_column_is_ewma` -> `test_model_column_contains_ewma` in 02-02 e2e test to accommodate multi-model world

**Regenerated published bar** (`reports/baseline_eval.md`, `reports/baseline_metrics.csv`):

All 5 assets, 3 models, 15 data rows, 0 GARCH fallbacks on real data.

## Published Results

| Asset | Model | N Forecasts | RMSE | MAE | QLIKE |
|-------|-------|-------------|------|-----|-------|
| BTC-USD | EWMA | 1,344 | 1.572e-03 | 7.443e-04 | 2.0215 |
| BTC-USD | GARCH | 1,344 | 1.590e-03 | 9.387e-04 | 2.0212 |
| BTC-USD | HAR | 1,344 | 1.568e-03 | 8.563e-04 | 1.9516 |
| ETH-USD | EWMA | 1,344 | 3.003e-03 | 1.404e-03 | 1.9632 |
| ETH-USD | GARCH | 1,344 | 3.011e-03 | 1.529e-03 | 1.9644 |
| ETH-USD | HAR | 1,344 | 3.005e-03 | 1.498e-03 | 1.9560 |
| SPY | EWMA | 818 | 3.993e-04 | 1.079e-04 | 1.6616 |
| SPY | GARCH | 818 | 4.010e-04 | 1.097e-04 | 1.7789 |
| SPY | HAR | 818 | 3.995e-04 | 1.182e-04 | 1.6662 |
| AAPL | EWMA | 817 | 9.027e-04 | 3.049e-04 | 1.7182 |
| AAPL | GARCH | 817 | 9.128e-04 | 3.247e-04 | 1.7600 |
| AAPL | HAR | 817 | 9.068e-04 | 3.223e-04 | 1.6985 |
| MSFT | EWMA | 819 | 6.598e-04 | 2.643e-04 | 1.7052 |
| MSFT | GARCH | 819 | 6.685e-04 | 2.820e-04 | 1.7617 |
| MSFT | HAR | 819 | 6.612e-04 | 2.970e-04 | 1.7027 |

**Honest observations:**
- HAR-RV achieves the best QLIKE on 4 of 5 assets (BTC, ETH, AAPL, MSFT) — consistent with the literature finding that HAR beats GARCH at daily horizon
- GARCH QLIKE is competitive on BTC-USD but underperforms on equity (SPY especially)
- EWMA is competitive or best on RMSE/MAE for all assets — the simplest baseline is hard to beat on symmetric metrics
- 0 GARCH fallbacks across all assets = model converged every refit; no data quality issues

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_model_column_is_ewma in test_report_e2e.py**
- **Found during:** Task 3
- **Issue:** The 02-02 e2e test `test_model_column_is_ewma` asserted that ALL CSV rows have `model=='EWMA'`. After Plan 02-03 adds GARCH and HAR rows, this test fails as designed — it was explicitly written for the single-model 02-02 world.
- **Fix:** Renamed to `test_model_column_contains_ewma` and changed assertion to `"EWMA" in models` — verifies EWMA is present without excluding the new models. The plan spec ("EWMA row unchanged in shape") is satisfied.
- **Files modified:** `tests/unit/test_report_e2e.py`
- **Commit:** afedffc

**2. [Rule 1 - Bug] Fixed test_rv_d_is_shifted_by_1 positional vs label indexing**
- **Found during:** Task 2 (RED phase — test had a bug)
- **Issue:** The test used `X["rv_d"].iloc[i]` with integer position `i=30..34`, but `build_har_features` drops ~23 NaN rows, so positional index 30 in X != original row 30 in rv.
- **Fix:** Changed to label-based indexing using `X.index[5:10]` and `rv.index.get_loc(lbl)` to correctly verify the shift-by-1 relationship.
- **Files modified:** `tests/unit/test_har_rv.py`
- **Commit:** 20f9c1b

## Threat Mitigations Applied

| Threat ID | Mitigation |
|-----------|-----------|
| T-02-07 | GARCH_SCALE=100.0 scaling + rescale=False + convergence_flag==0 + alpha+beta<1 assert on every refit + fallback chain |
| T-02-08 | forecast variance divided by GARCH_SCALE**2 exactly once; test asserts decimal magnitude (<1.0) |
| T-02-09 | splits computed once per asset, shared across all 3 models; test_report_baselines.py asserts identical n_forecasts |

## Known Stubs

None — all three baselines produce real walk-forward forecasts on real 5-asset data. No placeholders.

## Threat Flags

None — no new network endpoints, auth paths, or schema changes introduced.

## Self-Check: PASSED

Files exist:
- FOUND: src/volforecast/models/garch.py
- FOUND: src/volforecast/models/har_rv.py
- FOUND: tests/unit/test_garch.py
- FOUND: tests/unit/test_har_rv.py
- FOUND: tests/unit/test_report_baselines.py
- FOUND: reports/baseline_eval.md
- FOUND: reports/baseline_metrics.csv

Commits exist:
- e02b30d: feat(02-03): add GARCH(1,1) baseline
- 20f9c1b: feat(02-03): add HAR-RV OLS baseline
- afedffc: feat(02-03): extend report to three baselines
