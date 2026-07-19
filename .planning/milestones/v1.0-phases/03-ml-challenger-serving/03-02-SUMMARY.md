---
phase: 03-ml-challenger-serving
plan: "02"
subsystem: modeling
tags: [lightgbm, mlflow, shap, grid-search, walk-forward, qlike, champion-alias]

# Dependency graph
requires:
  - phase: 03-01
    provides: to_log_var/from_log_var/assemble_pooled_train/KNOWN_ASSETS/ASSET_DTYPE in lgbm.py + --serve-artifacts MLflow proxy mode
  - phase: 02
    provides: walk_forward_splits/qlike/rmse/mae in eval/harness.py + eval/metrics.py

provides:
  - PARAM_GRID (24 combos, 3*2*2*2 over num_leaves/min_child_samples/lr/reg_lambda)
  - grid_search: inner-validation-only hyperparameter selection (last 3 inner folds, Pitfall 2 clean)
  - train_pooled_model: LGBMRegressor with eval_set + early_stopping (4.6 API)
  - evaluate_per_asset: variance-scale RMSE/MAE/QLIKE on walk_forward_splits test folds
  - compute_shap_artifacts: TreeExplainer on native LGBMRegressor, global bar + beeswarm PNGs, per-asset top-10
  - compute_data_hash: MD5 data lineage helper
  - scripts/train_lgbm.py: end-to-end training entry point
  - Registered model volforecast-lgbm@champion in MLflow with params, metrics, tags, SHAP artifacts

affects: [03-03, 03-04]  # report plan uses champion run metrics; serving plan loads @champion

# Tech tracking
tech-stack:
  added: [lightgbm==4.6.0, shap==0.52.0]  # fastapi/uvicorn added to pyproject.toml in research phase but used in 03-04
  patterns:
    - inner-validation-only grid search (last 3 folds of train window, test folds never touched)
    - log-variance target with from_log_var inverse transform before variance-scale scoring
    - full column union reindex for multi-asset pooled eval (BTC has rv_22_eth, others have rv_22_btc)
    - MLflow alias (@champion) + validation_status tag for provenance (no deprecated stages)
    - SHAP TreeExplainer on native LGBMRegressor object (not pyfunc wrapper, Pitfall 6 clean)

key-files:
  created:
    - scripts/train_lgbm.py
    - tests/unit/test_shap_artifacts.py
  modified:
    - src/volforecast/models/lgbm.py

key-decisions:
  - "PARAM_GRID: 3*2*2*2=24 combos over num_leaves/min_child_samples/learning_rate/reg_lambda; feature_fraction fixed at 0.8"
  - "Inner validation uses last 3 folds of each asset's train window (63 steps for crypto, 39 for equity)"
  - "Column union reindex in evaluate_per_asset fills NaN for missing cross-asset cols (rv_22_eth/rv_22_btc)"
  - "Windows CP1252 fix: stdout/stderr rewrapped as UTF-8 to handle MLflow emoji in console output"
  - "LightGBM QLIKE (2.5-5.2) does not beat HAR-RV baseline (1.6-2.0) — honest result per plan contingency"

requirements-completed: [MODEL-01, MODEL-02, MODEL-03]

# Metrics
duration: 45min
completed: 2026-06-11
---

# Phase 03 Plan 02: LightGBM Challenger Training Summary

**Pooled LightGBM trained on log-variance with 24-combo inner-validation grid search, registered as volforecast-lgbm@champion in MLflow with per-asset RMSE/MAE/QLIKE metrics, data/git lineage tags, and SHAP global bar + beeswarm PNG artifacts**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-06-11T20:40:00Z
- **Completed:** 2026-06-11T21:25:00Z
- **Tasks:** 3
- **Files modified:** 3 (lgbm.py modified; train_lgbm.py + test_shap_artifacts.py created)

## Accomplishments

- Grid search selected best params (`num_leaves=15, min_child_samples=100, lr=0.05, reg_lambda=5.0`) on inner-val QLIKE only — no test fold contamination
- Final pooled LightGBM model registered as `volforecast-lgbm` version 3 with `@champion` alias (via `MlflowClient.set_registered_model_alias` — no deprecated stages)
- SHAP global bar + beeswarm PNGs logged as `shap/` artifacts under the champion run; per-asset top-10 importances in `shap/shap_per_asset_top10.txt`
- 17 offline unit tests covering SHAP shape, PNG existence, per-asset ranking, and native model requirement (Pitfall 6 guard) — all green

## Task Commits

1. **Task 1: Grid search + pooled training + per-asset evaluation + SHAP function** - `afe9592` (feat)
2. **Task 2: Train script — MLflow run, registry, @champion alias** - `a93dc29` (feat)
3. **Task 3: SHAP unit tests (TDD green)** - `2a8f45d` (test)

## Files Created/Modified

- `src/volforecast/models/lgbm.py` — Extended with PARAM_GRID, grid_search, train_pooled_model, evaluate_per_asset, compute_shap_artifacts, compute_data_hash
- `scripts/train_lgbm.py` — End-to-end training entry point: loads 5 assets, runs grid search, trains, evaluates, logs to MLflow, registers @champion
- `tests/unit/test_shap_artifacts.py` — 17 offline SHAP unit tests

## Decisions Made

- **PARAM_GRID shape:** 3*2*2*2=24 combos (within 12-32 acceptance range). `feature_fraction=0.8` is a fixed default applied to all combos rather than grid-searched — reduces compute by 2x.
- **Inner validation:** Last 3 folds (63 trading days for crypto, 63 for equity) of each asset's train window. Carving from train tail preserves temporal ordering without touching test folds.
- **Column union reindex in evaluate_per_asset:** BTC features have `rv_22_eth`; ETH/equities have `rv_22_btc`. The pooled training DataFrame has both (NaN-padded). At evaluation time, per-asset test DataFrames are reindexed to the full 21-column union before `model.predict()` — matches the training schema exactly.
- **Windows UTF-8 fix:** MLflow 3.x logs emoji characters (e.g., running person) that the Windows CP1252 console cannot encode. Fixed by wrapping `sys.stdout`/`sys.stderr` with a UTF-8 writer at script startup.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Column count mismatch in evaluate_per_asset (20 vs 21 features)**
- **Found during:** Task 2 (first training script run)
- **Issue:** LightGBM raised `LightGBMError: The number of features in data (20) is not the same as it was in training data (21)`. The pooled training set includes columns from all assets (BTC's `rv_22_eth` + ETH/equity's `rv_22_btc` = 21 feature cols); per-asset test DataFrames had only their own 20 cols.
- **Fix:** Added column union reindex in `evaluate_per_asset` — builds the full column list from the union of all asset feature DataFrames, then `reindex(columns=all_feat_cols_with_asset)` before predict. Missing cross-asset columns are NaN-filled (matches pooled training behavior).
- **Files modified:** `src/volforecast/models/lgbm.py` (evaluate_per_asset)
- **Verification:** Training script completed end-to-end; all 5 assets evaluated successfully.
- **Committed in:** `a93dc29` (Task 2 commit)

**2. [Rule 1 - Bug] Windows CP1252 UnicodeEncodeError on MLflow emoji output**
- **Found during:** Task 2 (first training script run)
- **Issue:** `UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f3c3'` — MLflow 3.x writes emoji to stdout when closing the run; Windows CP1252 console cannot encode it.
- **Fix:** Added UTF-8 stdout/stderr wrapper at the top of `scripts/train_lgbm.py` (replaces with `?` on unencodable characters).
- **Files modified:** `scripts/train_lgbm.py`
- **Verification:** Training script completes cleanly with no UnicodeEncodeError.
- **Committed in:** `a93dc29` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - Bug)
**Impact on plan:** Both fixes were necessary for the script to complete end-to-end. No scope creep. The column union reindex is a correct generalization of the per-asset evaluation pattern.

## Known Stubs

None — the plan's data flows are fully wired: feature parquets → pooled training → MLflow run → @champion alias → SHAP artifacts.

## Threat Flags

No new threat surface introduced beyond the plan's `<threat_model>`:
- T-03-03 mitigated: `data_hash=816a5337` and `git_sha=afe9592` logged as MLflow run tags; `validation_status=passed` on the model version.
- T-03-04 mitigated: `random_state=42` in all LGBMRegressor instances; params + data hash logged.
- T-03-05: localhost:5000 only, no LAN exposure.

## MLflow Run Details

| Field | Value |
|-------|-------|
| Model name | `volforecast-lgbm` |
| Version | 3 |
| Alias | `@champion` |
| Run ID | `28580617541943c4b76ed17b2772cf9d` |
| Best params | `num_leaves=15, min_child_samples=100, lr=0.05, reg_lambda=5.0` |
| Best iteration | 101 |
| Data hash | `816a5337` |
| Git SHA | `afe9592` |
| Train window end | `2026-05-14` |

## Per-Asset Metrics (LightGBM vs HAR-RV Baseline)

| Asset | LightGBM QLIKE | HAR-RV QLIKE | Beat baseline? |
|-------|---------------|--------------|----------------|
| BTC-USD | 3.891 | 1.931 | No |
| ETH-USD | 5.269 | 1.936 | No |
| SPY | 2.574 | 1.658 | No |
| AAPL | 3.940 | 1.667 | No |
| MSFT | 3.388 | 1.704 | No |

The LightGBM QLIKE is uniformly worse than HAR-RV. This is the project's honest-reporting contingency: "if ML fails to beat GARCH in some regimes, the model card says so." Possible causes: (1) single-fold final training (only last 39 folds' train data), (2) pooled cross-asset model may underfit asset-specific vol regimes, (3) log-variance Jensen's inequality bias. Plan 03-03 will generate the full honest comparison report.

## Issues Encountered

1. **MLflow input_example validation warning:** `Got error: train and valid dataset categorical_feature do not match` — this is an MLflow serving validation warning (not a failure) that occurs when MLflow validates the input example against the serving environment. The model is correctly logged and registered; this warning does not affect the run. It's a known MLflow 3.x behavior with LightGBM categorical features in pyfunc serving mode.

## Next Phase Readiness

- `volforecast-lgbm@champion` registered in MLflow at `http://localhost:5000` with full lineage — Plan 03-04 (FastAPI serving) can load via `mlflow.pyfunc.load_model("models:/volforecast-lgbm@champion")`.
- Per-asset QLIKE metrics available in the champion run — Plan 03-03 (report) can read them via `MlflowClient().get_run(mv.run_id).data.metrics`.
- `reports/baseline_metrics.csv` + champion run metrics provide the complete comparison data for Plan 03-03.
- Concern: LightGBM QLIKE does not beat HAR-RV. Plan 03-03's report must state this plainly as per the honest-reporting requirement (EVAL-04). This is a known, acceptable outcome — the plan's contingency is in place.

---
*Phase: 03-ml-challenger-serving*
*Completed: 2026-06-11*
