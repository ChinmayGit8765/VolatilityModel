# Requirements: VolForecast

**Defined:** 2026-06-10
**Core Value:** Honestly benchmark an ML volatility model against the correct classical baselines (GARCH(1,1)/EWMA/HAR-RV) under leak-free walk-forward evaluation, inside a genuine end-to-end MLOps lifecycle.

## v1 Requirements

### Foundation

- [ ] **FOUND-01**: Project runs on Python 3.12 with pinned dependency matrix (pandas <3, numpy 2.x, mlflow 3.x, arch, lightgbm, evidently 0.7+, pandera 0.31+, prefect 3.x) installable as a package (`src/volforecast/`)
- [ ] **FOUND-02**: docker-compose stack runs locally on Windows 11 (Postgres-backed MLflow tracking server; Prefect server + worker) with `.gitattributes` enforcing LF for shell scripts
- [ ] **FOUND-03**: CI pipeline (GitHub Actions) runs lint + unit tests on every push using fixture data (no live API calls in CI)

### Ingestion & Validation

- [ ] **INGEST-01**: Pipeline ingests 2+ years of daily OHLCV for BTC and ETH via ccxt with cache-first incremental updates and incomplete-last-candle handling
- [ ] **INGEST-02**: Pipeline ingests 2+ years of daily OHLCV for SPY + at least 2 large caps via yfinance with explicit `auto_adjust` handling and rate-limit tolerance
- [ ] **INGEST-03**: Per-asset-class trading calendars are respected — crypto is 24/7, equities have sessions/holidays; no fabricated weekend equity rows
- [ ] **INGEST-04**: Pandera validation gates reject gaps, bad ticks, stale data, and schema violations before data reaches the feature pipeline
- [ ] **INGEST-05**: Raw and processed datasets are versioned with DVC

### Features & Target

- [ ] **FEAT-01**: Target is next-period realized volatility with a documented, canonical proxy definition and unit convention (daily variance/vol of decimal log returns) in one shared module
- [ ] **FEAT-02**: Feature pipeline computes multi-lookback realized vol (5/10/22/66), log returns, squared returns, lagged vol, and EWMA vol
- [ ] **FEAT-03**: Feature pipeline computes range-based estimators (Parkinson, Garman-Klass), vol-of-vol, and rolling skew/kurtosis from OHLC
- [ ] **FEAT-04**: GARCH(1,1) conditional volatility is available as a model feature
- [ ] **FEAT-05**: Cross-asset features (e.g., BTC vol as ETH input) use as-of joins with a documented staleness rule across calendar mismatches
- [ ] **FEAT-06**: Calendar features (day-of-week, month, session/overnight flags for equities) are included
- [ ] **FEAT-07**: Training and serving import the identical versioned feature module (single codepath — no training/serving skew)

### Baselines & Evaluation

- [ ] **EVAL-01**: EWMA, GARCH(1,1) (arch, fitted on scaled returns with convergence assertions), and HAR-RV baselines produce walk-forward forecasts
- [ ] **EVAL-02**: Walk-forward evaluation harness is a reusable library with purging and an embargo gap >= label horizon; a unit test asserts temporal split ordering (no random splits)
- [ ] **EVAL-03**: One canonical, unit-tested QLIKE function (plus RMSE, MAE) is shared by baseline eval, ML eval, and the promotion gate
- [ ] **EVAL-04**: Evaluation reports ML-vs-baseline comparison per asset and per regime, reported honestly even where ML loses

### ML Model & Tracking

- [ ] **MODEL-01**: LightGBM regression model trains on the feature set and is evaluated on identical walk-forward folds as the baselines
- [ ] **MODEL-02**: All runs (params, metrics, artifacts) are tracked in MLflow; models are registered with version aliases (`@champion`/`@challenger`), not deprecated stages
- [ ] **MODEL-03**: SHAP explainability artifacts are produced for the registered model

### Serving

- [ ] **SERVE-01**: FastAPI service serves next-day vol forecasts for all tracked assets from the champion model alias
- [ ] **SERVE-02**: Service runs in Docker via docker-compose with a health endpoint and model-version metadata in responses
- [ ] **SERVE-03**: Every served forecast is appended to a prediction log (timestamp, asset, horizon, forecast, model version) — the contract for monitoring

### Monitoring & Feedback

- [ ] **MON-01**: Feedback-loop labeller joins arrived realized vol against logged forecasts to produce a forecast-vs-realized table automatically
- [ ] **MON-02**: Evidently reports feature and prediction distribution drift on a schedule (dashboard/log only, not auto-promotion)
- [ ] **MON-03**: Performance degradation (rolling live QLIKE vs GARCH's rolling QLIKE) triggers an alert and flags retraining
- [ ] **MON-04**: Alerts are delivered via a configurable channel (e.g., Slack webhook or email)

### Orchestration & Retraining

- [ ] **ORCH-01**: Prefect DAG runs ingest → validate → features → train → eval → register end-to-end on a schedule
- [ ] **ORCH-02**: Retraining can be triggered by the performance-drift flag in addition to schedule
- [ ] **ORCH-03**: Champion/challenger gate promotes the challenger only if it beats the champion on rolling QLIKE over a frozen comparison window; default outcome is no-promote; rollback is an alias flip

### Observability & Docs

- [ ] **DASH-01**: Streamlit dashboard shows forecast-vs-realized, current drift status, live model version/alias, and basic service stats
- [ ] **DOCS-01**: MODEL_CARD.md reports honest metrics vs all baselines, per-regime breakdown, assumptions, and limitations
- [ ] **DOCS-02**: README includes architecture diagram, how-to-run (docker-compose), and a "what I learned" writeup

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
| FOUND-01 | Phase 1 | Pending |
| FOUND-02 | Phase 1 | Pending |
| FOUND-03 | Phase 1 | Pending |
| INGEST-01 | Phase 1 | Pending |
| INGEST-02 | Phase 1 | Pending |
| INGEST-03 | Phase 1 | Pending |
| INGEST-04 | Phase 1 | Pending |
| INGEST-05 | Phase 1 | Pending |
| FEAT-01 | Phase 2 | Pending |
| FEAT-02 | Phase 2 | Pending |
| FEAT-03 | Phase 2 | Pending |
| FEAT-04 | Phase 2 | Pending |
| FEAT-05 | Phase 2 | Pending |
| FEAT-06 | Phase 2 | Pending |
| FEAT-07 | Phase 2 | Pending |
| EVAL-01 | Phase 2 | Pending |
| EVAL-02 | Phase 2 | Pending |
| EVAL-03 | Phase 2 | Pending |
| EVAL-04 | Phase 3 | Pending |
| MODEL-01 | Phase 3 | Pending |
| MODEL-02 | Phase 3 | Pending |
| MODEL-03 | Phase 3 | Pending |
| SERVE-01 | Phase 3 | Pending |
| SERVE-02 | Phase 3 | Pending |
| SERVE-03 | Phase 3 | Pending |
| MON-01 | Phase 4 | Pending |
| MON-02 | Phase 4 | Pending |
| MON-03 | Phase 4 | Pending |
| MON-04 | Phase 4 | Pending |
| ORCH-01 | Phase 4 | Pending |
| ORCH-02 | Phase 4 | Pending |
| ORCH-03 | Phase 4 | Pending |
| DASH-01 | Phase 5 | Pending |
| DOCS-01 | Phase 5 | Pending |
| DOCS-02 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 35 total
- Mapped to phases: 35
- Unmapped: 0 ✓

---
*Requirements defined: 2026-06-10*
*Last updated: 2026-06-10 after roadmap creation (traceability mapped; corrected count 31→35)*
