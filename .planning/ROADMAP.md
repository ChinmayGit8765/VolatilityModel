# Roadmap: VolForecast

## Overview

VolForecast goes from empty repo to a closed-loop volatility forecasting platform in five phases. Phase 1 settles the expensive-to-retrofit infrastructure (package, docker-compose, CI) and delivers validated, DVC-versioned cross-asset OHLCV data. Phase 2 builds the credibility centerpiece: a documented realized-vol target, a single feature codepath, and a leak-free purged walk-forward harness that scores EWMA/GARCH(1,1)/HAR-RV baselines — the bar exists before anything tries to clear it. Phase 3 adds the LightGBM challenger (tracked in MLflow with alias-based registry, benchmarked honestly on identical folds) and a Dockerized FastAPI service that logs every forecast. Phase 4 closes the loop: auto-arriving labels, Evidently drift, QLIKE-gated champion/challenger promotion, and a Prefect DAG with scheduled plus drift-triggered retraining. Phase 5 makes the system legible to reviewers via a Streamlit dashboard, an honest MODEL_CARD.md, and a README with architecture diagram.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation & Validated Data** - Reproducible local stack (package, compose, CI) ingesting and validating 2+ years of crypto + equity OHLCV under DVC (completed 2026-06-10)
- [ ] **Phase 2: Features, Target & Classical Baselines** - Documented RV target, single feature codepath, and a leak-free walk-forward harness scoring EWMA/GARCH/HAR-RV
- [ ] **Phase 3: ML Challenger & Serving** - MLflow-tracked LightGBM benchmarked on identical folds, served via Dockerized FastAPI with an append-only prediction log
- [ ] **Phase 4: Monitoring, Orchestration & Retraining** - Closed feedback loop: auto labels, drift detection, alerting, Prefect DAG, QLIKE-gated champion/challenger promotion
- [ ] **Phase 5: Dashboard & Honest Documentation** - Streamlit observability dashboard, MODEL_CARD.md with regime-segmented honest results, README with architecture diagram

## Phase Details

### Phase 1: Foundation & Validated Data
**Goal**: A reproducible local stack on Windows 11 ingests, validates, and versions 2+ years of cross-asset daily OHLCV — the trusted data layer everything downstream consumes
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: FOUND-01, FOUND-02, FOUND-03, INGEST-01, INGEST-02, INGEST-03, INGEST-04, INGEST-05
**Success Criteria** (what must be TRUE):
  1. `docker compose up` on the Windows 11 dev machine brings up a Postgres-backed MLflow tracking server and a Prefect server + worker, and `volforecast` installs as a package on Python 3.12 with the pinned dependency matrix
  2. Every push triggers GitHub Actions CI that passes lint and unit tests using fixture data only — no live API calls in CI
  3. A single command fetches 2+ years of daily OHLCV for BTC, ETH, SPY, and at least 2 large caps; re-running is cache-first incremental (no full re-download), incomplete last candles are excluded, and yfinance `auto_adjust`/rate limits are handled explicitly
  4. Data with gaps, bad ticks, stale rows, schema violations, or fabricated weekend equity bars is rejected by Pandera gates before reaching features — crypto validated against a 24/7 calendar, equities against session/holiday calendars
  5. Raw and processed datasets can be reproduced at any commit via DVC checkout
**Plans**: 4 plans
Plans:
- [x] 01-01-PLAN.md — Walking Skeleton: scaffold package + pinned deps + CLI, ingest one crypto asset (BTC) through validate -> parquet -> DVC-track (end-to-end slice)
- [x] 01-02-PLAN.md — Broaden ingestion: full BTC+ETH (ccxt) and SPY/AAPL/MSFT (yfinance) cache-first incremental, auto_adjust + retry, configurable exchange
- [x] 01-03-PLAN.md — Validation layer: crypto + equity Pandera schemas, calendar-aware gap checks (24/7 vs XNYS), stale/OHLC checks, validate_asset dispatcher with quarantine
- [x] 01-04-PLAN.md — Infra + CI seal: docker-compose (Postgres + MLflow + Prefect), GitHub Actions fixture-only CI, .gitattributes, validate_asset wired into pipeline, processed-data DVC tracking

### Phase 2: Features, Target & Classical Baselines
**Goal**: A leak-free purged walk-forward harness scores honest classical baselines (EWMA, GARCH(1,1), HAR-RV) on a canonically defined realized-vol target — the bar ML must clear exists before ML does
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: FEAT-01, FEAT-02, FEAT-03, FEAT-04, FEAT-05, FEAT-06, FEAT-07, EVAL-01, EVAL-02, EVAL-03
**Success Criteria** (what must be TRUE):
  1. The target (next-period realized vol, daily variance/vol of decimal log returns) is defined once in a shared, documented module, and the feature pipeline is a single versioned codepath imported identically by training and serving — no skew
  2. Running the feature pipeline produces multi-lookback realized vol (5/10/22/66), log/squared returns, lagged vol, EWMA vol, GARCH(1,1) conditional vol, Parkinson/Garman-Klass estimators, vol-of-vol, rolling skew/kurtosis, cross-asset as-of-joined features with a documented staleness rule, and calendar features — with every window ending strictly at as-of time
  3. EWMA, GARCH(1,1), and HAR-RV each produce walk-forward forecasts per asset, with GARCH fitted on scaled returns and convergence asserted on every refit
  4. A unit test fails if any split is non-temporal or the embargo gap is shorter than the label horizon; the canonical QLIKE function (shared by all evaluation and the future promotion gate) passes `qlike(x, x) == 0` and is used alongside RMSE/MAE
  5. An evaluation report shows per-asset baseline RMSE/MAE/QLIKE — the published bar for Phase 3
**Plans**: TBD

### Phase 3: ML Challenger & Serving
**Goal**: A tracked, explainable LightGBM challenger is benchmarked honestly against the classical bar on identical folds and served as next-day forecasts from a Dockerized API that logs every prediction
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: MODEL-01, MODEL-02, MODEL-03, EVAL-04, SERVE-01, SERVE-02, SERVE-03
**Success Criteria** (what must be TRUE):
  1. LightGBM trains on the Phase 2 feature set and is evaluated on the identical walk-forward folds as the baselines; the comparison report shows ML-vs-baseline RMSE/MAE/QLIKE per asset and per regime, stated honestly even where ML loses
  2. Every training run (params, metrics, artifacts) is visible in MLflow, and the registered model carries `@champion`/`@challenger` version aliases (no deprecated stages)
  3. SHAP explainability artifacts exist for the registered model and are inspectable
  4. Calling the FastAPI service returns next-day vol forecasts for all tracked assets from the `@champion` alias, with model-version metadata in responses and a working health endpoint, all running under docker-compose
  5. Every served forecast appends a row (timestamp, asset, horizon, forecast, model version) to the prediction log — the contract monitoring will consume
**Plans**: TBD

### Phase 4: Monitoring, Orchestration & Retraining
**Goal**: The system closes its feedback loop — auto-arriving realized vol labels logged forecasts, drift and performance degradation raise alerts and trigger retraining, and promotion is gated on rolling QLIKE with rollback as an alias flip
**Mode:** mvp
**Depends on**: Phase 3
**Requirements**: MON-01, MON-02, MON-03, MON-04, ORCH-01, ORCH-02, ORCH-03
**Success Criteria** (what must be TRUE):
  1. The forecast-vs-realized table populates automatically as realized vol arrives — no manual labelling step
  2. Evidently reports feature and prediction distribution drift on a schedule; drift informs dashboards/logs and can flag retraining but never auto-promotes a model
  3. When the live model's rolling QLIKE degrades relative to GARCH's rolling QLIKE, an alert is delivered via the configured channel (Slack webhook or email) and retraining is flagged
  4. The Prefect DAG (ingest → validate → features → train → eval → register) completes end-to-end both on schedule and when triggered by the performance-drift flag
  5. A challenger is promoted only if it beats the champion on rolling QLIKE over a frozen comparison window — default outcome is no-promote, and rollback is demonstrated as a single alias flip
**Plans**: TBD

### Phase 5: Dashboard & Honest Documentation
**Goal**: A reviewer can see the live system state at a glance and read an honest, regime-segmented account of how ML actually performed against the classical baselines
**Mode:** mvp
**Depends on**: Phase 4
**Requirements**: DASH-01, DOCS-01, DOCS-02
**Success Criteria** (what must be TRUE):
  1. The Streamlit dashboard shows forecast-vs-realized, current drift status, live model version/alias, and basic service stats
  2. MODEL_CARD.md reports honest metrics vs all three baselines with a per-regime breakdown, assumptions, and limitations — including any regimes where ML loses to GARCH
  3. README contains an architecture diagram, docker-compose how-to-run instructions, and a "what I learned" writeup
**Plans**: TBD
**UI hint**: yes

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation & Validated Data | 4/4 | Complete   | 2026-06-10 |
| 2. Features, Target & Classical Baselines | 0/TBD | Not started | - |
| 3. ML Challenger & Serving | 0/TBD | Not started | - |
| 4. Monitoring, Orchestration & Retraining | 0/TBD | Not started | - |
| 5. Dashboard & Honest Documentation | 0/TBD | Not started | - |

---
*Roadmap created: 2026-06-10*
*Granularity: coarse | Mode: mvp | Coverage: 35/35 v1 requirements*
