---
phase: "03"
plan: "03"
subsystem: eval-reporting
tags: [eval, reporting, regimes, ml-vs-baselines, qlike, honest-benchmarking]
dependency_graph:
  requires: [03-01, 03-02]
  provides: [ml-vs-baselines-report, regime-labelling-library]
  affects: [phase-05-model-card]
tech_stack:
  added: []
  patterns:
    - vol-tercile regime labelling on test-fold realized variance only (no lookahead)
    - pure render_report() function for offline smoke testing
    - native lightgbm model load via mlflow.lightgbm (avoids pyfunc categorical schema issue)
    - .values assignment to avoid DatetimeIndex alignment mismatch in groupby frames
key_files:
  created:
    - src/volforecast/eval/regimes.py
    - scripts/eval_lgbm.py
    - reports/ml_vs_baselines.md
    - reports/ml_vs_baselines.csv
    - tests/unit/test_regimes.py
    - tests/smoke/__init__.py
    - tests/smoke/test_report.py
  modified: []
decisions:
  - "Load champion via mlflow.lightgbm.load_model (native LGBMRegressor) rather than pyfunc.load_model; pyfunc schema enforcement rejects ASSET_DTYPE categorical columns"
  - "Use .values when assigning regime labels to a groupby-derived DataFrame to avoid DatetimeIndex/integer index alignment mismatch (NaN bug found and fixed)"
  - "assign_vol_terciles uses pd.cut with explicit bins — handles constant-input edge case by returning all-mid without raising"
metrics:
  duration_minutes: 18
  completed_date: "2026-06-11"
  task_count: 3
  file_count: 7
---

# Phase 03 Plan 03: ML vs Baselines Evaluation Report Summary

**One-liner:** Honest per-asset and per-regime ML-vs-baseline comparison using QLIKE/RMSE/MAE on identical walk-forward folds, with vol-tercile and calendar-year regime breakdown, stating all losses plainly.

## Tasks Completed

| Task | Name | Status | Commit |
|------|------|--------|--------|
| 1 | Regime labelling library (vol terciles + calendar year) | Done | e6e90ec |
| 2 | Report generator script and committed reports | Done | 2c7b6ee |
| 3 | Smoke test for report completeness | Done | 5a22557 |

## Deliverables

### reports/ml_vs_baselines.md
Markdown report with 4 sections:
1. Per-asset overall: LightGBM vs EWMA, GARCH, HAR-RV for all 5 assets
2. Per-vol-tercile breakdown (low/mid/high) per asset
3. Per-calendar-year breakdown (2022-2026) per asset
4. Honest findings: all 11 asset/regime combinations where LightGBM loses explicitly named

### reports/ml_vs_baselines.csv
168 rows covering `overall`, `tercile`, and `year` regime types for all 5 assets and 4 models.

### Key Honest Findings (from real data)
LightGBM significantly underperforms ALL baselines on QLIKE:
- **BTC-USD (overall):** LGBM QLIKE=3.891 vs HAR=1.931 (worse by 1.96)
- **ETH-USD (overall):** LGBM QLIKE=5.269 vs EWMA=1.934 (worse by 3.34)
- **SPY (overall):** LGBM QLIKE=2.574 vs EWMA=1.647 (worse by 0.93)
- **AAPL (overall):** LGBM QLIKE=3.940 vs HAR=1.696 (worse by 2.24)
- **MSFT (overall):** LGBM QLIKE=3.388 vs HAR=1.704 (worse by 1.68)
- High-vol regime losses are even more pronounced (LGBM QLIKE up to 13.3 for ETH-USD)

These results confirm the expected outcome: at daily frequency, LightGBM does NOT beat the classical baselines on QLIKE. The project's honest-reporting design was validated.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed index-alignment NaN in calendar-year regime labelling**
- **Found during:** Task 2 verification (year regime type missing from CSV)
- **Issue:** `assign_calendar_year()` returns a Series indexed by DatetimeIndex, but `asset_group` (after `groupby`) has integer index. Pandas alignment produced all-NaN year values.
- **Fix:** Used `.values` when assigning regime labels: `asset_group["year"] = assign_calendar_year(...).values`
- **Files modified:** `scripts/eval_lgbm.py`
- **Commit:** 2c7b6ee

**2. [Rule 1 - Bug] pyfunc champion load fails with ASSET_DTYPE categorical column**
- **Found during:** Task 2 development (investigation phase)
- **Issue:** `mlflow.pyfunc.load_model("models:/volforecast-lgbm@champion").predict(df)` rejects CategoricalDtype for the `asset` column. Passing string fails too (LightGBM internal train/valid mismatch).
- **Fix:** Load champion via `mlflow.lightgbm.load_model(f"runs:/{mv.run_id}/model")` using the run ID from the alias. Native LGBMRegressor accepts ASSET_DTYPE natively.
- **Files modified:** `scripts/eval_lgbm.py`
- **Commit:** 2c7b6ee

## Verification Evidence

- `uv run python scripts/eval_lgbm.py` exits cleanly; reports committed.
- CSV assertions: all 5 assets, all 4 models, `tercile` and `year` regime types present.
- `uv run pytest tests/ -x -q`: 378 passed, 2 skipped, 0 failures.
- `uv run ruff check .`: All checks passed.

## Self-Check: PASSED

**Checked files exist:**
- `src/volforecast/eval/regimes.py`: FOUND
- `scripts/eval_lgbm.py`: FOUND
- `reports/ml_vs_baselines.md`: FOUND
- `reports/ml_vs_baselines.csv`: FOUND
- `tests/unit/test_regimes.py`: FOUND
- `tests/smoke/test_report.py`: FOUND

**Checked commits exist:**
- e6e90ec: feat(03-03): vol-regime labelling library — FOUND
- 2c7b6ee: feat(03-03): report generator and ml-vs-baselines committed reports — FOUND
- 5a22557: test(03-03): smoke test for report completeness — FOUND
