---
phase: 02-features-target-classical-baselines
plan: "02"
subsystem: evaluation
tags: [ewma, riskmetrics, walk-forward, qlike, realized-variance, baseline, report]

# Dependency graph
requires:
  - phase: 02-01
    provides: walk_forward_splits harness, compute_target, qlike/rmse/mae metrics
  - phase: 01-foundation-validated-data
    provides: processed parquet (5-asset OHLCV) via config.processed_path

provides:
  - EWMA(lambda=0.94) one-step-ahead variance forecaster (models/ewma.py)
  - Deterministic estimator helpers: log_returns, squared_returns, realized_var, ewma_variance (features/estimators.py)
  - Baseline report generator: generate_baseline_report() wiring target+harness+EWMA+metrics (reports/baseline.py)
  - Committed baseline_eval.md with per-asset EWMA RMSE/MAE/QLIKE for BTC-USD, ETH-USD, SPY, AAPL, MSFT
  - Machine-readable baseline_metrics.csv (one row per asset/model with n_forecasts)

affects:
  - 02-03 (GARCH and HAR-RV extend generate_baseline_report with new model columns)
  - 02-04 (full feature pipeline reuses log_returns, realized_var, ewma_variance from estimators.py)
  - 03 (LightGBM compared against this published bar; same metrics module)
  - 04 (promotion gate uses same harness and metrics)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Estimator module: stateless vectorized helpers over pd.Series; integer-count rolling windows (not duration strings) for consistent NaN prefix across equity/crypto calendars"
    - "EWMA forecaster: forecast_path() = ewma_variance(lr).shift(1); shift(1) is the explicit no-look-ahead guarantee"
    - "Report generator: library function (no CLI entrypoint), callable from tests (fixture) and production (real data)"
    - "Zero-target guard: drop target==0 rows alongside NaN rows before QLIKE — yfinance adjusted-close artifact"

key-files:
  created:
    - src/volforecast/features/estimators.py
    - src/volforecast/models/ewma.py
    - src/volforecast/models/__init__.py (updated)
    - src/volforecast/reports/__init__.py
    - src/volforecast/reports/baseline.py
    - reports/baseline_eval.md
    - reports/baseline_metrics.csv
    - tests/unit/test_estimators.py
    - tests/unit/test_ewma.py
    - tests/unit/test_report_e2e.py
  modified:
    - src/volforecast/models/__init__.py

key-decisions:
  - "EWMA_LAMBDA=0.94 constant lives in estimators.py and is imported by models/ewma.py — single source of truth, no duplication"
  - "forecast_path() = ewma_variance(lr).shift(1): the shift(1) makes the no-look-ahead invariant explicit and testable"
  - "Zero-target rows (identical consecutive closes) dropped before QLIKE scoring — these are yfinance adjusted-close artifacts, not genuine zero-variance observations"
  - "generate_baseline_report() takes a data_root parameter to support both test fixtures (tmp_path) and real data (project_root/data)"
  - "Atomic file writes (write to .tmp then rename) prevent partial-file reads on interruption"

patterns-established:
  - "Integer-window rolling only: realized_var(lr, window=5) uses .rolling(5) not .rolling('5D') — count-based windows give consistent NaN prefix on irregular calendars"
  - "adjust=False is mandatory for EWMA: ewm(alpha=1-lam, adjust=False) matches the RiskMetrics recursion; adjust=True diverges on short series"
  - "Baseline interface pattern: class with forecast_path(log_returns) -> pd.Series; returns full-series one-step-ahead forecasts; harness runner slices by test_idx"
  - "NaN+zero target guard in scoring loop: valid_mask = test_tgt.notna() & (test_tgt > 0.0)"

requirements-completed: [FEAT-02, EVAL-01, EVAL-03]

# Metrics
duration: 45min
completed: 2026-06-11
---

# Phase 02 Plan 02: EWMA Baseline Evaluation Summary

**EWMA(lambda=0.94) walk-forward baseline producing per-asset RMSE/MAE/QLIKE for all 5 assets via the first committed published bar (baseline_eval.md)**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-06-11T07:30:00Z
- **Completed:** 2026-06-11T08:15:00Z
- **Tasks:** 3 (TDD: RED estimators, GREEN EWMA forecaster, GREEN report)
- **Files modified:** 10 created, 1 updated

## Accomplishments

- Delivered `features/estimators.py` with all deterministic helpers (log_returns, squared_returns, realized_var with integer window, ewma_variance with adjust=False) and EWMA_LAMBDA=0.94 constant
- Delivered `models/ewma.py` with EWMA forecaster using shift(1) one-step-ahead guarantee — no per-fold refit, uniform baseline call shape ready for GARCH/HAR-RV in 02-03
- Delivered `reports/baseline.py` wiring target + walk-forward harness + EWMA forecasts + canonical metrics into a reproducible library function
- Committed `reports/baseline_eval.md` and `reports/baseline_metrics.csv` with real EWMA numbers for all 5 assets (1621/1112 rows each, min_train=252, step=21); this is the published bar Plan 02-03 extends
- 45 tests across 3 test files, all passing; ruff clean on all src/tests

## Task Commits

Each task was committed atomically:

1. **Task 1: Failing e2e test (RED) + estimators** - `adb8fff` (test)
2. **Task 2: EWMA forecaster (walk-forward compatible)** - `e1716b7` (feat)
3. **Task 3: Baseline report generator → RED→GREEN** - `cb9d543` (feat)

## Files Created/Modified

- `src/volforecast/features/estimators.py` - log_returns, squared_returns, realized_var, ewma_variance, EWMA_LAMBDA
- `src/volforecast/models/ewma.py` - EWMA class with forecast_path() returning shift(1) EWMA variance
- `src/volforecast/models/__init__.py` - exports EWMA
- `src/volforecast/reports/__init__.py` - reports package stub
- `src/volforecast/reports/baseline.py` - generate_baseline_report() library function
- `reports/baseline_eval.md` - committed published bar: per-asset EWMA RMSE/MAE/QLIKE
- `reports/baseline_metrics.csv` - machine-readable per-asset metrics (5 rows)
- `tests/unit/test_estimators.py` - 21 tests for all estimator behaviors
- `tests/unit/test_ewma.py` - 15 tests for EWMA forecaster (no-look-ahead, alignment, walk-forward)
- `tests/unit/test_report_e2e.py` - 9 e2e tests for generate_baseline_report contract

## Published Baseline Numbers

| Asset | N Forecasts | RMSE | MAE | QLIKE |
|-------|-------------|------|-----|-------|
| BTC-USD | 1,344 | 1.572e-03 | 7.443e-04 | 2.022 |
| ETH-USD | 1,344 | 3.003e-03 | 1.404e-03 | 1.963 |
| SPY | 818 | 3.993e-04 | 1.079e-04 | 1.662 |
| AAPL | 817 | 9.027e-04 | 3.049e-04 | 1.718 |
| MSFT | 819 | 6.598e-04 | 2.643e-04 | 1.705 |

*Daily decimal variance units; walk-forward min_train=252, step=21*

## Decisions Made

- EWMA_LAMBDA=0.94 constant lives in estimators.py and is imported by models/ewma.py — prevents duplication across 3 plans (02, 03, 04)
- forecast_path() = ewma_variance(lr).shift(1) — explicit shift makes the no-look-ahead invariant testable with a simple assertion
- generate_baseline_report() takes `data_root` param so the same function works in unit tests (tmp_path fixture) and production (project_root/data)
- Atomic file writes (.tmp → rename) for both outputs

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Drop zero-target rows before QLIKE scoring**
- **Found during:** Task 3 (generate real 5-asset report)
- **Issue:** SPY and AAPL had exactly 2 rows where yfinance adjusted-close produced identical consecutive closes (float64 artifact), causing target = squared log return = 0 exactly. QLIKE = +inf because log(0/h) = -inf.
- **Fix:** Changed valid_mask from `test_tgt.notna()` to `test_tgt.notna() & (test_tgt > 0.0)`. Documented in module docstring, report header, and comments.
- **Files modified:** src/volforecast/reports/baseline.py
- **Verification:** SPY QLIKE changed from inf to 1.662; all 5 assets now have finite metrics. e2e tests still pass.
- **Committed in:** cb9d543 (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (Rule 2 missing critical edge case)
**Impact on plan:** Fix is a correctness requirement — zero realized variance is not a genuine market observation but a data artifact. Dropping it is the right behavior and is consistent with the plan's Pitfall 4 guidance ("never fill NaN with zero"). No scope creep.

## Issues Encountered

- Test logic error in test_nan_prefix_length: asserted window=22 NaN count on a 20-row series (rolling(22) on 19 non-NaN values = all NaN, not 22). Fixed by using n=50 in test — deviation within Task 1 RED commit.
- ruff import ordering on auto-generated test files — fixed with `ruff check --fix`.

## Known Stubs

None — the report generator is fully wired to real data (not mocked). The committed reports/baseline_eval.md and reports/baseline_metrics.csv contain real EWMA metrics from the 5-asset processed data.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. The report generator reads from the project-root-relative data directory (config.processed_path) and writes to the project-root-relative reports/ directory — no user-supplied path strings traversal risk per T-02-06 (accepted).

## Next Phase Readiness

- Plan 02-03: extend generate_baseline_report() with GARCH(1,1) and HAR-RV model columns; EWMA row already present as the first baseline
- Plan 02-04: import log_returns, realized_var, ewma_variance from estimators.py for the full feature pipeline
- The published bar (baseline_eval.md) gives honest per-asset EWMA numbers for the Phase 3 ML model to beat

## Self-Check

- [x] reports/baseline_eval.md exists with all 5 asset slugs and QLIKE column
- [x] reports/baseline_metrics.csv has 5 data rows with asset, model, n_forecasts, rmse, mae, qlike
- [x] Task commits exist: adb8fff (test), e1716b7 (feat), cb9d543 (feat)
- [x] 45 unit tests pass; ruff check src tests exits 0

---
*Phase: 02-features-target-classical-baselines*
*Completed: 2026-06-11*
