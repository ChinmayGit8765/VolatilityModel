# Phase 4: Monitoring, Orchestration & Retraining - Context

**Gathered:** 2026-06-12
**Status:** Ready for planning
**Mode:** Autonomous smart discuss (recommended answers auto-accepted per user directive)

<domain>
## Phase Boundary

The system closes its feedback loop: auto-arriving realized vol labels logged forecasts; distribution and performance drift raise alerts and flag retraining; a Prefect DAG runs the pipeline on schedule and on drift trigger; promotion is gated on rolling QLIKE with no-promote default and alias-flip rollback. Delivers MON-01..04 and ORCH-01..03. No dashboard (Phase 5), no new modeling.

</domain>

<decisions>
## Implementation Decisions

### Feedback Loop & Labels (MON-01)
- Labeller job joins arrived realized vol (computed from ingested processed OHLCV via the canonical target module) against the prediction log → append-only `data/monitoring/forecast_vs_realized.parquet`
- A forecast made as-of t is labelable once t+1 close exists in processed data; per-asset-class calendars respected (equity weekends/holidays simply have no labelable rows)
- Labeller is idempotent: re-running never duplicates rows (keyed on asset + as_of date + model_version)

### Drift & Alerts (MON-02..04)
- Distribution drift: Evidently 0.7+ Report (current API: Report/Dataset/DataDefinition — NOT legacy ColumnMapping) on feature and prediction distributions vs a frozen training reference snapshot; output JSON + HTML to data/monitoring/; dashboard/log only — NEVER triggers promotion
- Performance drift: rolling live QLIKE of champion vs rolling QLIKE of the GARCH baseline over a 21-day window from forecast_vs_realized; if champion underperforms GARCH by a documented threshold, raise alert + set retrain flag
- Alerts: webhook URL from env var ALERT_WEBHOOK_URL (Slack-compatible JSON POST); when unset, write structured alert records to data/monitoring/alerts.jsonl (still observable); never crash the pipeline on alert delivery failure
- Monitor taxonomy (locked): data-quality failure → pipeline hard-fail (existing Pandera gates); distribution drift → log/report only; performance degradation → alert + retrain flag

### Orchestration (ORCH-01..02)
- One Prefect 3 daily flow: ingest → validate → features → label → drift-check → (conditional on flag or schedule) train → eval → register challenger; flows in pipelines/ as thin wrappers over existing functions/scripts
- Drift-triggered retrain: simple flag check inside the daily flow (flag file or flow parameter) — no Prefect Automations in v1 (research-recommended)
- Deployment served by the compose Prefect worker (process pool, local-pool); schedule daily after market data availability
- Flows must run offline-safe in tests (mock external calls); live runs use the compose stack

### Champion/Challenger Gate (ORCH-03)
- Retrain registers the new model version with alias @challenger (never auto-champion)
- Promotion gate: challenger beats champion on rolling QLIKE over a frozen comparison window (identical rows for both, data hash + window recorded); canonical qlike from eval/metrics.py
- Default outcome: no-promote. Promotion = set_registered_model_alias flip; rollback = flip back to previous version (rehearsed in a test)
- Cooldown: minimum 7 days between promotions (documented constant)

### Claude's Discretion
- Exact threshold for performance degradation (document the choice), Evidently metric selection, flow/task naming, monitoring parquet schemas, retry policies

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/volforecast/serving/prediction_log.py` — PREDICTION_LOG_SCHEMA (timestamp_utc, asset, horizon, forecast_var, model_version, alias), atomic append
- `src/volforecast/features/target.py::compute_target` — canonical realized-variance label; `eval/metrics.py::qlike` (floored, promotion-gate-ready)
- `src/volforecast/eval/harness.py` — reusable walk-forward; `models/{ewma,garch,har_rv}.py` aligned baselines (GARCH for the live comparison)
- `scripts/train_lgbm.py` (MLflow run + registration), `scripts/eval_lgbm.py`, `scripts/generate_features.py`
- CLI: `volforecast ingest` (cache-first, validated); compose stack: postgres/mlflow(--serve-artifacts)/prefect-server/prefect-worker/api

### Established Patterns
- uv-managed Python 3.12.13 (uv NOT on PATH); ruff format+check repo-wide before commits (CI runs `ruff check .`); tests offline-only and hermetic (no data/ dependency — see tests/integration/test_api.py synthetic fixture pattern)
- MLflow aliases only; prediction log is the serving→monitoring contract

### Integration Points
- Inputs: data/predictions/predictions.parquet (live, accumulating), data/processed/, data/features/, MLflow registry volforecast-lgbm v3 @champion
- Outputs for Phase 5 dashboard: forecast_vs_realized.parquet, drift report JSON/HTML, alerts.jsonl, registry alias state
- MLFLOW_TRACKING_URI: host http://localhost:5000; in-network http://mlflow-server:5000

</code_context>

<specifics>
## Specific Ideas

- "Drift should trigger retrain+eval only, never deployment" — anti-pattern guard from research
- The honest Phase 3 result (LightGBM loses to HAR-RV/GARCH) makes the performance-drift monitor genuinely meaningful: champion underperformance vs GARCH is the real, expected signal

</specifics>

<deferred>
## Deferred Ideas

- Streamlit dashboard (Phase 5); Prefect Automations/event triggers (v2); cloud deploy (v2)

</deferred>
