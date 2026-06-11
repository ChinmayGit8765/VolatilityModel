---
phase: 03-ml-challenger-serving
reviewed: 2026-06-12T00:00:00Z
depth: standard
files_reviewed: 18
findings:
  critical: 2
  warning: 10
  info: 6
  total: 18
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-06-12 | **Depth:** standard | **Files:** 18 | **Status:** issues_found

## Summary

Transforms, alias-only registry usage (no stages — verified), prediction-log atomicity, and SHAP-on-native-model are correct. But the evaluation pipeline — the credibility centrepiece — has two independent Criticals that invalidate every reported LightGBM metric: train-on-test contamination and a silent eval column transpose. The flagship no-leakage unit tests are vacuous (NaN asset categories) and could not catch either.

## Critical

### CR-01: Single model trained at last fold's cutoff evaluated on all earlier test folds — train-on-test contamination
**Files:** `scripts/train_lgbm.py:176-205`, `src/volforecast/models/lgbm.py:569-675` (evaluate_per_asset), `scripts/eval_lgbm.py:150-178`
One pooled model is trained on fold `min_folds-1` (training window ≈ all history minus last test window), then scored on ALL folds' test windows — earlier test windows are inside its training set. Baselines refit per step (genuine walk-forward), so the report compares in-sample ML vs out-of-sample classical. Grid-search inner-val rows are also earlier folds' test windows. CONTEXT's "identical walk-forward folds" implies per-fold retraining.
**Fix:** train one model per outer fold (shared best_params) and predict only that fold's own test window — mirroring baseline forecast_path discipline. If too slow, restrict evaluation to test windows strictly after the final training cutoff and say so in the report. Champion stays the last-fold model; metrics must come from leak-free folds. Regenerate MLflow metrics + reports.

### CR-02: Eval feature-column order differs from training; LightGBM predicts silently on swapped columns
**Files:** `src/volforecast/models/lgbm.py:606-629`, `363-377`, `scripts/eval_lgbm.py:120-125,155-159`
Training concat yields `[...base, rv_22_eth, asset, rv_22_btc]`; eval builds `[...base, rv_22_eth, rv_22_btc, asset]` — last two transposed. LGBMRegressor.predict default `validate_features=False` checks only column COUNT, so eval feeds category codes into rv_22_btc and floats into asset. Every evaluate_per_asset metric and lgbm row in ml_vs_baselines is computed on a corrupted matrix. Serving and SHAP are unaffected (they use training-order construction / signature reindex).
**Fix:** reindex eval frames to `model.feature_name_` exact order, restore ASSET_DTYPE, and call predict with `validate_features=True` everywhere. Same in eval_lgbm._collect_per_fold_rows.

## Warnings

### WR-01: assign_vol_terciles crashes on tied quantile boundaries; ±1e-30 bin padding is a float no-op
`src/volforecast/eval/regimes.py:91-101` — if q33==lo (mass at floored zero-returns) pd.cut raises on non-monotonic bins. Fix: quantile-comparison labels via np.where, or pd.qcut(duplicates="drop") with collapse fallback.

### WR-02: Zero-RV drop in eval_lgbm is dead code — floored zeros pass `<= 0` check, injecting ~+15 QLIKE outliers
`scripts/eval_lgbm.py:181-184`, `lgbm.py:640-663` — from_log_var round-trips zero to 1e-10 > 0 so the filter never fires. Fix: filter raw target, or `rv_val <= LOG_VAR_EPS*(1+1e-9)`; apply identically in evaluate_per_asset.

### WR-03: No purge between inner-train and inner-val (horizon-1 label overlap at boundary)
`lgbm.py:311-317,356-361` — fix: `inner_train_idx = train_idx[:-(n_val + horizon)]`.

### WR-04: Flagship no-leakage tests vacuous — symbols "A"/"B" not in KNOWN_ASSETS → NaN categories → empty selections always pass
`tests/unit/test_lgbm_folds.py:139-161,186-226` — fix: use KNOWN_ASSETS symbols and assert non-empty row selections.

### WR-05: Serving unwraps private MLflow internals (`pyfunc_model._model_impl.lgb_model`); three divergent model-load codepaths
`serving/app.py:90-94` vs `eval_lgbm.py` runs:/ vs train. Fix: `mlflow.lightgbm.load_model("models:/volforecast-lgbm@champion")` everywhere + `mlflow.models.get_model_info(uri).signature`.

### WR-06: Serving dropna() silently serves stale as-of rows
`serving/app.py:270-278` — full-row dropna steps back to older rows on any NaN with no freshness bound. Fix: dropna only on required columns; add staleness guard (503 or `stale: true` beyond N days).

### WR-07: Dockerfile `uv sync --frozen` before src/ copy breaks fresh builds (hatchling root project); CMD re-syncs at runtime
`infra/api/Dockerfile:13-15,40` — fix: `--no-install-project` first layer, copy src, final `uv sync`, `uv run --no-sync` in CMD.

### WR-08: API container runs as root with writable bind mount into host repo
`infra/api/Dockerfile`, compose api service — fix: non-root USER; document the tradeoff.

### WR-09: `--allowed-hosts '*'` defeats DNS-rebinding protection on the loopback-bound MLflow
`infra/mlflow-entrypoint.sh:22` — fix: `--allowed-hosts 'localhost,127.0.0.1,mlflow-server,mlflow-server:5000'` (+host.docker.internal if needed).

### WR-10: Baseline/LGBM row alignment in _collect_per_fold_rows breaks if feature/processed indices diverge
`scripts/eval_lgbm.py:176-178` vs 573-574 — fix: reindex baseline forecast series to feat_df.index before the fold loop; use positional selection consistently.

## Info

### IN-01: assemble_pooled_train dead on production path; train_pooled_model docstring false
### IN-02: train_window_end lineage tag computed from wrong fold (and only first asset)
### IN-03: feature_names parameter of compute_shap_artifacts unused
### IN-04: Dead fallback branch in serving asset-config lookup
### IN-05: test_pyfunc_wrapper_raises nearly vacuous (catches Exception)
### IN-06: grid_search docstring claims 3-fold averaging; implementation uses single contiguous block

---
_Reviewer: gsd-code-reviewer (returned inline; artifact written by orchestrator)_
