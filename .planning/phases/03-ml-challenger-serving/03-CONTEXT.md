# Phase 3: ML Challenger & Serving - Context

**Gathered:** 2026-06-11
**Status:** Ready for planning
**Mode:** Autonomous smart discuss (recommended answers auto-accepted per user directive)

<domain>
## Phase Boundary

A tracked, explainable LightGBM challenger benchmarked honestly against the Phase 2 classical bar on identical folds, served as next-day forecasts from a Dockerized FastAPI that logs every prediction. Delivers: LightGBM training on the Phase 2 feature set, MLflow run tracking + alias-based registry (@champion), SHAP artifacts, ml-vs-baselines comparison report (per asset and per regime, honest), FastAPI service in compose serving from @champion with prediction log. No drift monitoring, no retraining automation, no champion/challenger gate logic — Phase 4.

</domain>

<decisions>
## Implementation Decisions

### ML Training Design
- One pooled LightGBM across all 5 assets with asset as a categorical feature; per-asset metrics still reported
- Identical walk-forward folds as Phase 2 baselines (reuse eval/harness.py splits per asset); pooled training assembles train rows from each asset's train indices only — no cross-asset temporal leakage (train cutoff respected per asset)
- Regression on log(target variance) for scale stability; predictions inverse-transformed (exp) back to variance before QLIKE/RMSE/MAE; metrics reported on variance scale with the canonical eval/metrics.py
- Modest fixed hyperparameter grid (~20 combos) selected on validation folds only (last fold(s) of train window); test folds never used for tuning
- Reproducibility: fixed seed, params + data hash logged to MLflow

### MLflow Tracking & Registry
- Tracking server: the compose-stack MLflow at http://localhost:5000 (Postgres-backed) — set MLFLOW_TRACKING_URI
- Registry: model name `volforecast-lgbm`; alias `@champion` assigned to the initial best run; `@challenger` is Phase 4's concern
- No deprecated stages — aliases + tags only
- Regimes for the report: per-asset vol terciles (low/mid/high realized vol) and calendar year

### Serving
- FastAPI endpoints: GET /health, GET /forecast (all assets), GET /forecast/{symbol}; responses include forecast variance + vol, model_version, alias, as_of date
- Features computed via the same `build_features()` import on local processed data — single codepath (FEAT-07)
- Model loaded from registry by alias `models:/volforecast-lgbm@champion` at startup, with manual refresh endpoint or lazy reload allowed
- Prediction log: append-only parquet at `data/predictions/predictions.parquet` with columns (timestamp_utc, asset, horizon, forecast_var, model_version, alias) — the Phase 4 monitoring contract
- Service runs in docker-compose alongside MLflow stack; Dockerfile in src/volforecast/serving or infra/

### SHAP & Report
- TreeExplainer; global summary bar + beeswarm PNGs; per-asset top-10 feature importances; logged as MLflow artifacts under the registered run
- `reports/ml_vs_baselines.md` + CSV: LightGBM vs EWMA/GARCH/HAR-RV per asset and per regime; losses stated plainly

### Claude's Discretion
- Exact LightGBM param grid, log-variance epsilon handling, FastAPI module layout, Dockerfile base image (python:3.12-slim + libgomp1 per research), how compose mounts model/data

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/volforecast/features/pipeline.py::build_features` — single feature codepath, 19+ cols incl. cross-asset (`rv_22_btc` style suffixes)
- `src/volforecast/eval/harness.py::walk_forward_splits` (min_train=252, step=21, purge+embargo) and `eval/metrics.py::qlike/rmse/mae` (both inputs floored, ValueError on non-finite)
- `src/volforecast/models/{ewma,garch,har_rv}.py` — aligned baselines: forecast[t] predicts RV[t+1] from data ≤ t
- `reports/baseline_eval.md` + `baseline_metrics.csv` — the bar (HAR-RV best QLIKE on 4/5 assets)
- `scripts/generate_features.py` — regenerates data/features parquets
- infra/docker-compose.yml — postgres (5433), mlflow-server (127.0.0.1:5000), prefect-server (127.0.0.1:4200), prefect-worker; custom mlflow Dockerfile at infra/mlflow/Dockerfile

### Established Patterns
- uv-managed Python 3.12.13 (uv NOT on PATH); ruff format + check before commits; tests offline-only with fixtures; CI runs `ruff check .` on the whole repo
- lightgbm 4.6 + mlflow 3.13 + shap 0.52 already in the pinned dependency matrix (verify importable; add only if missing)
- data/ gitignored, DVC-tracked with local remote ../../dvc-cache (sibling of repo)

### Integration Points
- Inputs: data/features/*.parquet (5 assets, 19 cols), data/processed/ parquet
- Outputs consumed by Phase 4: prediction log parquet schema, registry aliases, the harness reused by the promotion gate
- MLflow tracking URI http://localhost:5000 (compose stack must be up for training/registry tasks — it is currently up on the dev machine)

</code_context>

<specifics>
## Specific Ideas

- "Beating, or honestly not beating, GARCH is the eval-rigour centrepiece" — ml_vs_baselines.md must state losses plainly per asset/regime
- MLflow aliases not stages (deprecated) — using stages would signal outdated MLOps knowledge

</specifics>

<deferred>
## Deferred Ideas

- Champion/challenger promotion gate, drift monitoring, scheduled/triggered retraining (Phase 4)
- Streamlit dashboard (Phase 5); LSTM/TFT challenger (v2); cloud deploy (v2)

</deferred>
