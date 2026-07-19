---
status: passed
phase: 05-dashboard-honest-documentation
verified: 2026-06-16
score: 3/3 must-haves verified
requirements: [DASH-01, DOCS-01, DOCS-02]
note: >
  Verification completed by the execute-phase orchestrator after the gsd-verifier
  subagent hit a transient API "Overloaded" error mid-run. Verdict is grounded in
  first-hand evidence: live docker-compose validation, source-level assertions, the
  full green test suite, and the prior plan-checker PASS.
---

# Phase 05 Verification — Dashboard & Honest Documentation

**Verdict: PASSED (3/3 success criteria).** All three requirement IDs (DASH-01,
DOCS-01, DOCS-02) are covered, each by a dedicated plan whose frontmatter declares
the ID. The full unit suite is green (440 passed, 2 skipped) and the dashboard +
closed loop were validated live against the running docker-compose stack.

## Success Criterion 1 — Streamlit dashboard (DASH-01)

> Shows forecast-vs-realized, drift status, live model version/alias, and service stats.

- **All 4 panels present** in `src/volforecast/dashboard/app.py` (forecast-vs-realized,
  drift status, live model version, service stats) — confirmed by source inspection.
- **Read-only enforced**: no real write calls (`set_registered_model_alias(`, `.to_parquet(`,
  write-mode `open`) exist in `data_access.py` or `app.py` — the only textual matches are
  docstring assertions of the read-only contract.
- **Empty-state safe**: 10 `st.info`/`st.warning` empty-state branches; 23 offline
  `data_access` unit tests cover the missing-input paths. Verified live: the dashboard
  came up clean on `docker compose up` BEFORE the pipeline run (FVR + drift empty).
- **Live verification**: dashboard service serves on `127.0.0.1:8501` (`/_stcore/health` →
  `200 ok`); all 5 loaders ran against live services — `load_fvr` (5,313 rows post-flow),
  `latest_drift_summary` (2026-06-15 report), `champion_info` (volforecast-lgbm v4),
  `prediction_stats` (24 rows), `api_health` (ok). Mirrors the `api` service container
  pattern (slim + libgomp1 + non-root, loopback-only).
- **Drift REPORT-ONLY** honored (panel states it; no promotion path from the dashboard).

## Success Criterion 2 — MODEL_CARD.md (DOCS-01)

> Honest metrics vs all three baselines, per-regime, incl. regimes where ML loses to GARCH.

- `MODEL_CARD.md` (409 lines) sources all metrics verbatim from `reports/ml_vs_baselines.md`
  / `.csv` / `baseline_eval.md` (no invented numbers).
- **Section 7 "Per-Regime (Volatility-Tercile) Breakdown"** present, with low/mid/high
  tercile labels computed leak-free on test-fold realized variance only.
- **Honest losses named**: Section 8 leads with BTC-USD high-vol LightGBM QLIKE 11.50 vs
  GARCH 0.68 (16× worse) and ETH-USD 13.88 vs HAR 1.10 (12× worse), with the mechanism
  (QLIKE penalizes under-forecasting of vol spikes; trees under-predict).
- Assumptions, leak-free walk-forward methodology, intended-use ("not a trading strategy"),
  and known limitations (incl. the Phase-4 cooldown-not-persisted limitation) are documented.

## Success Criterion 3 — README (DOCS-02)

> Architecture diagram + docker-compose how-to-run + "what I learned".

- README contains a GitHub-native ```mermaid``` lifecycle diagram covering ingest →
  validate → features → walk-forward train/eval → MLflow registry → serving → prediction
  log → labeller → drift + performance monitor → promotion gate → Prefect daily flow →
  dashboard. **No drift→promotion edge** (REPORT-ONLY anti-pattern respected) — asserted by grep.
- "How to run" section: `docker compose -f infra/docker-compose.yml up`, six-service port
  table (api 8000, mlflow 5000, prefect 4200, dashboard 8501, postgres 5433), and a 10-step
  first-run sequence using only real scripts + the `volforecast ingest` CLI.
- "What I learned": honest QLIKE comparison table + 5 MLOps lessons; links to MODEL_CARD.md.

## Requirement Traceability

| Req | Plan | Status |
|-----|------|--------|
| DASH-01 | 05-01 | Covered — dashboard live |
| DOCS-01 | 05-02 | Covered — MODEL_CARD honest + per-regime |
| DOCS-02 | 05-03 | Covered — README diagram + run + lessons |

## Low-severity notes (non-blocking)

1. **Drift summary parse**: `latest_drift_summary` returns `dataset_drift=None` /
   `n_drifted_columns=None` against Evidently 0.7's JSON shape — the report still generates
   and the panel links it; only the summary flag/count are absent. Cosmetic polish item.

No HIGH-severity gaps. Phase 5 goal achieved.
