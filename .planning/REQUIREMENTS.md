# Requirements: VolForecast

**Defined:** 2026-06-10
**Core Value:** Honestly benchmark an ML volatility model against the correct classical baselines (GARCH(1,1)/EWMA/HAR-RV) under leak-free walk-forward evaluation, inside a genuine end-to-end MLOps lifecycle.

## v1 Requirements

### Foundation

- [x] **FOUND-01**: Project runs on Python 3.12 with pinned dependency matrix (pandas <3, numpy 2.x, mlflow 3.x, arch, lightgbm, evidently 0.7+, pandera 0.31+, prefect 3.x) installable as a package (`src/volforecast/`)
- [x] **FOUND-02**: docker-compose stack runs locally on Windows 11 (Postgres-backed MLflow tracking server; Prefect server + worker) with `.gitattributes` enforcing LF for shell scripts
- [x] **FOUND-03**: CI pipeline (GitHub Actions) runs lint + unit tests on every push using fixture data (no live API calls in CI)

### Ingestion & Validation

- [x] **INGEST-01**: Pipeline ingests 2+ years of daily OHLCV for BTC and ETH via ccxt with cache-first incremental updates and incomplete-last-candle handling
- [x] **INGEST-02**: Pipeline ingests 2+ years of daily OHLCV for SPY + at least 2 large caps via yfinance with explicit `auto_adjust` handling and rate-limit tolerance
- [x] **INGEST-03**: Per-asset-class trading calendars are respected — crypto is 24/7, equities have sessions/holidays; no fabricated weekend equity rows
- [x] **INGEST-04**: Pandera validation gates reject gaps, bad ticks, stale data, and schema violations before data reaches the feature pipeline
- [x] **INGEST-05**: Raw and processed datasets are versioned with DVC

### Features & Target

- [x] **FEAT-01**: Target is next-period realized volatility with a documented, canonical proxy definition and unit convention (daily variance/vol of decimal log returns) in one shared module
- [x] **FEAT-02**: Feature pipeline computes multi-lookback realized vol (5/10/22/66), log returns, squared returns, lagged vol, and EWMA vol
- [x] **FEAT-03**: Feature pipeline computes range-based estimators (Parkinson, Garman-Klass), vol-of-vol, and rolling skew/kurtosis from OHLC
- [x] **FEAT-04**: GARCH(1,1) conditional volatility is available as a model feature
- [x] **FEAT-05**: Cross-asset features (e.g., BTC vol as ETH input) use as-of joins with a documented staleness rule across calendar mismatches
- [x] **FEAT-06**: Calendar features (day-of-week, month, session/overnight flags for equities) are included
- [x] **FEAT-07**: Training and serving import the identical versioned feature module (single codepath — no training/serving skew)

### Baselines & Evaluation

- [x] **EVAL-01**: EWMA, GARCH(1,1) (arch, fitted on scaled returns with convergence assertions), and HAR-RV baselines produce walk-forward forecasts
- [x] **EVAL-02**: Walk-forward evaluation harness is a reusable library with purging and an embargo gap >= label horizon; a unit test asserts temporal split ordering (no random splits)
- [x] **EVAL-03**: One canonical, unit-tested QLIKE function (plus RMSE, MAE) is shared by baseline eval, ML eval, and the promotion gate
- [x] **EVAL-04**: Evaluation reports ML-vs-baseline comparison per asset and per regime, reported honestly even where ML loses

### ML Model & Tracking

- [x] **MODEL-01**: LightGBM regression model trains on the feature set and is evaluated on identical walk-forward folds as the baselines
- [x] **MODEL-02**: All runs (params, metrics, artifacts) are tracked in MLflow; models are registered with version aliases (`@champion`/`@challenger`), not deprecated stages
- [x] **MODEL-03**: SHAP explainability artifacts are produced for the registered model

### Serving

- [x] **SERVE-01**: FastAPI service serves next-day vol forecasts for all tracked assets from the champion model alias
- [x] **SERVE-02**: Service runs in Docker via docker-compose with a health endpoint and model-version metadata in responses
- [x] **SERVE-03**: Every served forecast is appended to a prediction log (timestamp, asset, horizon, forecast, model version) — the contract for monitoring

### Monitoring & Feedback

- [x] **MON-01**: Feedback-loop labeller joins arrived realized vol against logged forecasts to produce a forecast-vs-realized table automatically
- [x] **MON-02**: Evidently reports feature and prediction distribution drift on a schedule (dashboard/log only, not auto-promotion)
- [x] **MON-03**: Performance degradation (rolling live QLIKE vs GARCH's rolling QLIKE) triggers an alert and flags retraining
- [x] **MON-04**: Alerts are delivered via a configurable channel (e.g., Slack webhook or email)

### Orchestration & Retraining

- [x] **ORCH-01**: Prefect DAG runs ingest → validate → features → train → eval → register end-to-end on a schedule
- [x] **ORCH-02**: Retraining can be triggered by the performance-drift flag in addition to schedule
- [x] **ORCH-03**: Champion/challenger gate promotes the challenger only if it beats the champion on rolling QLIKE over a frozen comparison window; default outcome is no-promote; rollback is an alias flip

### Observability & Docs

- [x] **DASH-01**: Streamlit dashboard shows forecast-vs-realized, current drift status, live model version/alias, and basic service stats
- [x] **DOCS-01**: MODEL_CARD.md reports honest metrics vs all baselines, per-regime breakdown, assumptions, and limitations
- [x] **DOCS-02**: README includes architecture diagram, how-to-run (docker-compose), and a "what I learned" writeup

## v2 Requirements

### Stretch

- **STRETCH-01**: Intraday horizon (true realized vol from higher-frequency crypto data)
- **STRETCH-02**: Deep learning challenger (LSTM or small Temporal Fusion Transformer, PyTorch)
- **STRETCH-03**: Cloud deployment (AWS or Azure) of the serving stack
- **STRETCH-04**: Feast feature store integration
- **STRETCH-05**: Multi-horizon forecasts (1/5/22 days) via horizon-parameterized harness

## Out of Scope

| Feature | Reason |
|---------|--------|
| Trading strategy / PnL / signals | This is a forecasting + ML-systems demo, not a trading scheme |
| Price direction prediction | Not reliably learnable; volatility is the credible target |
| Random CV splits | Lookahead leakage — correctness violation; temporal splits enforced by test |
| Kafka / streaming infra | Daily-horizon batch loop; streaming adds cost without credibility |
| Kubernetes | docker-compose suffices for a portfolio-scale system |
| Paid data feeds | Free tiers only (ccxt public, yfinance) |
| MLflow registry stages | Deprecated since MLflow 2.9 — aliases used instead |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FOUND-01 | Phase 1 | Complete |
| FOUND-02 | Phase 1 | Complete |
| FOUND-03 | Phase 1 | Complete |
| INGEST-01 | Phase 1 | Complete |
| INGEST-02 | Phase 1 | Complete |
| INGEST-03 | Phase 1 | Complete |
| INGEST-04 | Phase 1 | Complete |
| INGEST-05 | Phase 1 | Complete |
| FEAT-01 | Phase 2 | Complete |
| FEAT-02 | Phase 2 | Complete |
| FEAT-03 | Phase 2 | Complete |
| FEAT-04 | Phase 2 | Complete |
| FEAT-05 | Phase 2 | Complete |
| FEAT-06 | Phase 2 | Complete |
| FEAT-07 | Phase 2 | Complete |
| EVAL-01 | Phase 2 | Complete |
| EVAL-02 | Phase 2 | Complete |
| EVAL-03 | Phase 2 | Complete |
| EVAL-04 | Phase 3 | Complete |
| MODEL-01 | Phase 3 | Complete |
| MODEL-02 | Phase 3 | Complete |
| MODEL-03 | Phase 3 | Complete |
| SERVE-01 | Phase 3 | Complete |
| SERVE-02 | Phase 3 | Complete |
| SERVE-03 | Phase 3 | Complete |
| MON-01 | Phase 4 | Complete |
| MON-02 | Phase 4 | Complete |
| MON-03 | Phase 4 | Complete |
| MON-04 | Phase 4 | Complete |
| ORCH-01 | Phase 4 | Complete |
| ORCH-02 | Phase 4 | Complete |
| ORCH-03 | Phase 4 | Complete |
| DASH-01 | Phase 5 | Complete |
| DOCS-01 | Phase 5 | Complete |
| DOCS-02 | Phase 5 | Complete |

**Coverage:**
- v1 requirements: 35 total
- Mapped to phases: 35
- Completed: 35
- Unmapped: 0 ✓

---
*Requirements defined: 2026-06-10*
*Last updated: 2026-06-10 after roadmap creation (traceability mapped; corrected count 31→35)*
