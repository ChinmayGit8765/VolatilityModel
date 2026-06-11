---
phase: 02-features-target-classical-baselines
verified: 2026-06-11T00:00:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
---

# Phase 2: Features, Target & Classical Baselines — Verification Report

**Phase Goal:** A leak-free purged walk-forward harness scores honest classical baselines (EWMA, GARCH(1,1), HAR-RV) on a canonically defined realized-vol target — the bar ML must clear exists before ML does.
**Verified:** 2026-06-11
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (Roadmap Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Target defined once in shared module; feature pipeline is a single versioned codepath imported identically by training and serving — no skew | VERIFIED | `src/volforecast/features/target.py` owns `compute_target`, `HORIZON=1`; `src/volforecast/features/pipeline.py` has exactly one `def build_features`; FEAT-07 comment in module docstring; `uv run python -c "from volforecast.features.pipeline import build_features; print('ok')"` exits 0 |
| 2 | Feature pipeline produces multi-lookback RV (5/10/22/66), log/squared returns, lagged vol, EWMA vol, GARCH(1,1) conditional vol, Parkinson/GK, vol-of-vol, rolling skew/kurtosis, cross-asset as-of features, calendar features — every window ending strictly at as-of t | VERIFIED | BTC-USD feature parquet has 19 columns incl. rv_5/rv_10/rv_22/rv_66, ewma_var, garch_cond_var, parkinson_var, gk_var, vol_of_vol, rolling_skew/kurt, day_of_week/month/is_monday/is_friday, eth_rv22_xasset; no-lookahead truncation-invariance test (test_no_lookahead.py: 25 passed, 2 skipped on short fixtures) |
| 3 | EWMA, GARCH(1,1), and HAR-RV each produce walk-forward forecasts per asset; GARCH fitted on scaled returns with convergence asserted on every refit | VERIFIED | `GARCH_SCALE = 100.0` (line 69 garch.py); `rescale=False` (line 103); `assert res.convergence_flag == 0` (line 108); `assert alpha + beta < 1.0` (line 113); 15 rows in baseline_metrics.csv (5 assets × 3 models); all N Forecasts > 0; 0 GARCH fallbacks on real data |
| 4 | Unit test FAILS if any split is non-temporal or embargo < horizon; canonical QLIKE passes `qlike(x, x) == 0` and is used alongside RMSE/MAE | VERIFIED | `qlike(x, x) = 0.0` confirmed live (`< 1e-12: True`); test_harness.py: all 13 tests pass incl. `test_walk_forward_no_leakage_horizon5` that bites on purge-removal mutation; SUMMARY documents 16/16 splits fail on mutation |
| 5 | Evaluation report shows per-asset baseline RMSE/MAE/QLIKE — the published bar for Phase 3 | VERIFIED | `reports/baseline_eval.md` exists with 15-row table covering BTC-USD/ETH-USD/SPY/AAPL/MSFT × EWMA/GARCH/HAR; `reports/baseline_metrics.csv` has 16 lines (header + 15 data rows); all metrics finite and non-negative |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/volforecast/features/target.py` | Canonical target definition + HORIZON constant | VERIFIED | `compute_target`, `HORIZON=1`, `forward_realized_var`; module docstring explicitly states "daily VARIANCE of decimal log returns; NO annualization" |
| `src/volforecast/eval/metrics.py` | Canonical `qlike()`, `rmse()`, `mae()` in Patton variance form | VERIFIED | `QLIKE_FLOOR = 1e-10`; Patton form `ratio - np.log(ratio) - 1` on line 79; `qlike(x,x)==0` confirmed live |
| `src/volforecast/eval/harness.py` | `WalkForwardSplit` + `walk_forward_splits` with purge + embargo | VERIFIED | `def walk_forward_splits` present; `train_end = test_start - horizon` purge formula documented; embargo invariant `min(test_idx) - max(train_idx) >= horizon` enforced and tested |
| `tests/unit/test_harness.py` | Leak test asserting temporal ordering and embargo >= horizon | VERIFIED | `assert split.train_idx.max() < split.test_idx.min()` (line 42); `assert split.test_idx.min() - split.train_idx.max() >= 1` (line 47); 13 tests all pass |
| `src/volforecast/features/estimators.py` | log_returns, squared_returns, realized_var, ewma_variance, EWMA_LAMBDA, parkinson_var, garman_klass_var, vol_of_vol, rolling_skew/kurt, lagged_vol, calendar_features | VERIFIED | All 11 functions present; `EWMA_LAMBDA = 0.94` on line 58; `adjust=False` in ewma_variance; integer rolling windows throughout |
| `src/volforecast/models/ewma.py` | EWMA(lambda=0.94) forecaster with `forecast_path` | VERIFIED | `EWMA_LAMBDA` imported from estimators.py; `shift(1)` no-look-ahead explicit |
| `src/volforecast/models/garch.py` | GARCH(1,1) with GARCH_SCALE=100, rescale=False, convergence + stationarity assertions, scale inversion | VERIFIED | `GARCH_SCALE: float = 100.0` (line 69); `rescale=False` (line 103); `convergence_flag == 0` assertion (line 108); `alpha + beta < 1.0` assertion (line 113); `GARCH_SCALE**2` inversion on line 137; 0 fallbacks on real data |
| `src/volforecast/models/har_rv.py` | HAR-RV OLS on 1/5/22 lagged RV components | VERIFIED | `rolling(5).mean()` and `rolling(22).mean()` for rv_w/rv_m (lines 83-84); `sm.add_constant` present; statsmodels lazy import |
| `src/volforecast/features/cross_asset.py` | as_of_join with `pd.Timedelta("3D")` staleness tolerance | VERIFIED | `MAX_CROSS_ASSET_STALENESS = pd.Timedelta("3D")` (line 37); `direction="backward"` (line 124 — via `tolerance=MAX_CROSS_ASSET_STALENESS`); both frames sorted + monotonic assert before merge_asof |
| `src/volforecast/features/pipeline.py` | `build_features()` — single codepath | VERIFIED | `def build_features` (line 159); imports from estimators, cross_asset, models.garch; 19 feature columns in persisted parquets |
| `reports/baseline_eval.md` | Per-asset RMSE/MAE/QLIKE for 3 baselines × 5 assets | VERIFIED | 15-row table; all 5 slugs present; EWMA/GARCH/HAR columns all populated with real numeric values |
| `reports/baseline_metrics.csv` | 15 data rows, asset/model/rmse/mae/qlike | VERIFIED | 16 lines (header + 15 data rows) confirmed; schema has asset,model,n_forecasts,rmse,mae,qlike,garch_fallbacks |
| `data/features/crypto/BTC-USD.parquet` | Persisted feature matrix | VERIFIED | 1621 rows, 19 cols; all required columns present |
| `data/features/crypto/ETH-USD.parquet` | Persisted feature matrix | VERIFIED | 1621 rows, 19 cols |
| `data/features/equity/SPY.parquet` | Persisted feature matrix | VERIFIED | 1112 rows, 19 cols |
| `data/features/equity/AAPL.parquet` | Persisted feature matrix | VERIFIED | 1112 rows, 19 cols |
| `data/features/equity/MSFT.parquet` | Persisted feature matrix | VERIFIED | 1112 rows, 19 cols |
| `data/features.dvc` | DVC pointer for data/features | VERIFIED | Pointer exists; md5 hash `0ea732b69f541a6187bd608526f963ae.dir`; nfiles=5; `dvc status` reports "Data and pipelines are up to date" |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `eval/metrics.py` | qlike(x,x)==0 | Patton form `ratio - np.log(ratio) - 1` | VERIFIED | Confirmed on line 79; live execution returns 0.0 (< 1e-12) |
| `eval/harness.py` | `test_harness.py` | `train_idx.max() < test_idx.min()` + embargo assert | VERIFIED | Both assertions on lines 42 and 47 of test file |
| `reports/baseline.py` | `eval/harness.py` + `eval/metrics.py` + `features/target.py` | imports walk_forward_splits, qlike/rmse/mae, compute_target | VERIFIED | All three canonical modules imported; test_report_baselines.py: 11/11 pass |
| `models/ewma.py` | `reports/baseline.py` | EWMA forecast per walk-forward fold | VERIFIED | EWMA row present in CSV for all 5 assets; N Forecasts > 0 |
| `models/garch.py` | `arch.arch_model` | ZeroMean GARCH(1,1) on 100x scaled returns, rescale=False | VERIFIED | `arch_model(scaled, mean="Zero", vol="GARCH", p=1, q=1, rescale=False)` on line 103 |
| `reports/baseline.py` | `models/garch.py` + `models/har_rv.py` | identical folds | VERIFIED | test_report_baselines.py `TestIdenticalFoldsAcrossModels::test_same_n_forecasts_per_asset` passes |
| `features/pipeline.py` | `features/estimators.py` + `features/cross_asset.py` + `models/garch.py` | build_features imports all | VERIFIED | All three import lines present at top of pipeline.py |
| `features/cross_asset.py` | `pd.merge_asof` | backward + `pd.Timedelta("3D")` tolerance | VERIFIED | `tolerance=MAX_CROSS_ASSET_STALENESS` (line 127); `direction="backward"` (line 124) |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `reports/baseline_eval.md` | per-asset RMSE/MAE/QLIKE | `generate_baseline_report()` reads processed parquets via `config.processed_path` | Yes — BTC-USD 1,344 forecasts; SPY 818 forecasts; 0 GARCH fallbacks | FLOWING |
| `data/features/crypto/BTC-USD.parquet` | 19 feature columns | `build_features()` on `data/processed/crypto/BTC-USD.parquet` | Yes — 1621 rows, non-NaN GARCH from row 252 onward | FLOWING |
| `reports/baseline_metrics.csv` | 15 metric rows | `generate_baseline_report()` | Yes — all metrics finite, QLIKE values 1.66–2.02 range | FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `build_features` importable | `uv run python -c "from volforecast.features.pipeline import build_features; print('ok')"` | `ok` | PASS |
| `qlike(x, x) == 0` | `uv run python -c "import numpy as np; from volforecast.eval.metrics import qlike; x=np.array([0.001,0.002,0.0005]); print(abs(qlike(x,x)) < 1e-12)"` | `True` | PASS |
| BTC-USD parquet has required columns | `uv run python -c "..."` | All 7 required columns present; 1621 rows | PASS |
| Full test suite | `uv run pytest tests/ -q` | 276 passed, 2 skipped in 43.43s | PASS |
| DVC status clean | `uv run dvc status data/features.dvc` | "Data and pipelines are up to date." | PASS |
| ruff check | `uv run ruff check src tests` | "All checks passed!" | PASS |

---

### Probe Execution

Step 7c: No explicit probe scripts (`scripts/*/tests/probe-*.sh`) were declared in the PLAN files for Phase 2. SKIPPED (no probe scripts for this phase).

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| FEAT-01 | 02-01 | Target is next-period RV with canonical unit convention in one shared module | SATISFIED | `target.py` owns `compute_target` + `HORIZON=1`; docstring states decimal variance, no annualization |
| FEAT-02 | 02-02, 02-04 | Multi-lookback RV (5/10/22/66), log/squared returns, lagged vol, EWMA vol | SATISFIED | All present in `estimators.py` and `pipeline.py`; verified in feature parquets |
| FEAT-03 | 02-04 | Range estimators (Parkinson, GK), vol-of-vol, rolling skew/kurtosis | SATISFIED | `parkinson_var`, `garman_klass_var`, `vol_of_vol`, `rolling_skew`, `rolling_kurt` all in `estimators.py` |
| FEAT-04 | 02-03, 02-04 | GARCH(1,1) conditional vol as model feature | SATISFIED | `garch_cond_var` column in all 5 feature parquets; `_garch_conditional_var_feature` in pipeline.py |
| FEAT-05 | 02-04 | Cross-asset features with as-of joins + documented staleness rule | SATISFIED | `cross_asset.py`; `MAX_CROSS_ASSET_STALENESS = pd.Timedelta("3D")`; `eth_rv22_xasset` / `btc_rv22_xasset` in parquets |
| FEAT-06 | 02-04 | Calendar features (day-of-week, month, session flags) | SATISFIED | `calendar_features()` adds day_of_week, month, is_monday, is_friday; equity fixture test confirms no weekend values |
| FEAT-07 | 02-04 | Training and serving import identical versioned feature module | SATISFIED | Exactly one `def build_features` in `pipeline.py`; importable as `from volforecast.features.pipeline import build_features` |
| EVAL-01 | 02-02, 02-03 | EWMA, GARCH(1,1) (arch, convergence asserted), HAR-RV produce walk-forward forecasts | SATISFIED | 15 rows in CSV; convergence assertions in `fit_garch`; 0 fallbacks on real data |
| EVAL-02 | 02-01 | Walk-forward harness with purging + embargo >= label horizon; unit test asserts temporal split | SATISFIED | `walk_forward_splits` with purge formula `train_end = test_start - horizon`; leak test passes and bites on mutation |
| EVAL-03 | 02-01, 02-02 | One canonical QLIKE function + RMSE/MAE shared by all evaluation | SATISFIED | `eval/metrics.py` is the single source; `qlike(x,x)==0` passes live; used by baseline.py and test suite |

---

### Anti-Patterns Found

| File | Pattern Searched | Result | Severity |
|------|-----------------|--------|----------|
| All phase-2 src files | TBD/FIXME/XXX | 0 matches | Clean |
| All phase-2 src files | placeholder/coming soon/not yet implemented | 0 matches | Clean |
| `models/garch.py` | convergence assertions | Present on every refit | Clean |
| `features/estimators.py` | `adjust=False` for EWMA | Present (line ~131) | Clean |
| `reports/baseline_metrics.csv` | 15 data rows (not empty) | 15 rows confirmed | Clean |

No anti-patterns found. All source files are substantive implementations with no stubs, TODOs, or unresolved debt markers.

---

### Human Verification Required

None. All observable truths are verifiable programmatically for this batch-processing, report-generating phase. No UI components, real-time behavior, or external service integrations are present in Phase 2.

---

### Gaps Summary

No gaps. All 5 roadmap success criteria are verified:

1. Single codepath / canonical target — confirmed by import test and module structure
2. Full feature set with no-lookahead — confirmed by parquet column check and 25-passing truncation-invariance tests
3. Three classical baselines with GARCH convergence assertions — confirmed by code inspection and 276 passing tests with 0 GARCH fallbacks
4. Leak-free harness with biting leak test + qlike(x,x)==0 — confirmed live
5. Published evaluation report — confirmed by baseline_eval.md (15 rows) and baseline_metrics.csv (15 data rows)

---

_Verified: 2026-06-11_
_Verifier: Claude (gsd-verifier)_
