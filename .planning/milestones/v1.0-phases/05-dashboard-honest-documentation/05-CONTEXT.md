# Phase 05 Context — Dashboard & Honest Documentation

Source: orchestrator-captured decisions (autonomous run, 2026-06-15). Stack choices
are locked by CLAUDE.md; panels are locked by ROADMAP success criteria. This phase
is an internal observability dashboard + honest docs — function over visual design,
so no UI-SPEC design contract is needed (`--skip-ui`).

## Decisions (LOCKED)

### Dashboard (DASH-01)
- **Tech:** Streamlit 1.58 (locked by stack). Single multi-section app.
- **Deployment:** New `dashboard` service in `infra/docker-compose.yml`, port `8501`
  bound to `127.0.0.1` (consistent with api/mlflow/prefect localhost binding).
  Depends on `mlflow-server` and `api`. Reuses the project image / `uv`-installed
  package; mounts the repo read-only + `data/` like the prefect-worker so it can read
  monitoring artifacts. Set `MLFLOW_TRACKING_URI=http://mlflow-server:5000` and an
  `API_BASE_URL=http://api:8000` env var.
- **Panels (exactly the 4 success-criteria items):**
  1. **Forecast vs realized** — per-asset time series from
     `data/monitoring/forecast_vs_realized.parquet` (champion forecast vs realized
     vol; GARCH baseline line if present). Asset selector.
  2. **Drift status** — summary of the latest drift report under `data/monitoring/`
     (dataset-drift flag + count of drifted columns); link/embed of the dated HTML if
     available. Drift is REPORT-ONLY — never implies promotion (locked anti-pattern).
  3. **Live model version/alias** — query MLflow registry for `volforecast-lgbm@champion`
     (version number, run id, key params/metrics tags).
  4. **Service stats** — prediction-log stats from `data/predictions/predictions.parquet`
     (row count, last forecast timestamp, latest forecast per asset) + live `GET /health`
     from the API (status + round-trip latency).
- **Empty-state handling (CRITICAL for fresh-clone / first-run portfolio demo):** every
  panel must render a friendly "no data yet — run the daily pipeline" message instead of
  raising when its parquet/registry/endpoint is missing or empty. The dashboard must come
  up cleanly on `docker compose up` before any pipeline run.
- **No authentication** — local/internal tool, same trust boundary as the other services.
- Read-only: the dashboard NEVER writes data, never triggers retraining, never flips aliases.

### MODEL_CARD.md (DOCS-01)
- Honest, regime-segmented results sourced from `reports/ml_vs_baselines.md` +
  `reports/ml_vs_baselines.csv` + `reports/baseline_eval.md` (per-asset and per-regime
  RMSE/MAE/QLIKE for LightGBM vs EWMA / GARCH(1,1) / HAR-RV).
- MUST explicitly call out regimes/assets where ML LOSES to GARCH — honesty is a feature.
- Include: intended use (a volatility-forecasting + MLOps **demo**, explicitly NOT a
  trading strategy), training data + window, leak-free purged walk-forward methodology,
  metrics definitions (esp. QLIKE), assumptions, and known limitations (e.g. promotion
  cooldown not persisted across Prefect runs — documented v1 limitation from Phase 4).
- Pull the actual numbers from the report files; do not invent metrics.

### README (DOCS-02)
- **Architecture diagram:** Mermaid (renders natively on GitHub) showing the full
  lifecycle: ingest → validate → features → train/eval (walk-forward) → MLflow registry
  (@champion/@challenger) → FastAPI serving → prediction log → labeller → drift +
  performance monitor → promotion gate → Prefect daily flow (scheduled + drift-triggered).
  Show the docker-compose services and how data flows between them.
- **How to run:** exact `docker compose -f infra/docker-compose.yml up` instructions,
  the service ports (api 8000, mlflow 5000, prefect 4200, dashboard 8501), and the
  first-run sequence to populate state (ingest/train/register, then run the daily flow)
  so a reviewer gets a working dashboard.
- **"What I learned":** honest writeup — the real ML-vs-GARCH finding, and MLOps lessons
  (leak-free walk-forward, alias-based registry, drift→retrain loop, Windows+Docker gotchas).

## Out of scope for this phase
- No new model work, no new endpoints beyond what serving already exposes.
- No design system / theming work (Streamlit defaults are fine).
- Cloud deployment (local docker-compose only, per project constraints).

## Open assumptions (planner may refine)
- Plan granularity: ~2–3 plans (dashboard app + compose service; MODEL_CARD; README+diagram).
- If `forecast_vs_realized.parquet` is empty at build time, dashboard is validated against
  the empty-state path; real data is populated later by the live Prefect flow run.
