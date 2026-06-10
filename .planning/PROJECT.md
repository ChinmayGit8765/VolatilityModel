# VolForecast — Crypto + Stock Volatility Forecasting MLOps Platform

## What This Is

A short-horizon realized volatility forecasting system covering crypto (BTC, ETH) and equities (SPY + large caps), wrapped in a full production MLOps lifecycle: ingestion, validation, feature engineering, classical + ML models, serving, monitoring, drift detection, and automated retraining. It is a flagship ML Engineer portfolio project — a volatility *forecasting* and ML-systems skill demo, explicitly NOT a trading strategy.

## Core Value

Honestly benchmark an ML volatility model against the correct classical baseline (GARCH(1,1)/EWMA) under leak-free walk-forward evaluation, inside a genuine end-to-end MLOps lifecycle — every screening question ("deploy", "CI/CD for ML", "drift + retraining", "feature engineering at scale") gets a repo-backed answer.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Ingest daily OHLCV for crypto (BTC, ETH via ccxt/Binance) and equities (SPY + large caps via yfinance), handling 24/7 crypto vs session-gapped equity calendars
- [ ] Validate ingested data with Pandera gates (gaps, bad ticks, stale data) before it reaches features
- [ ] Reproducible feature pipeline: multi-lookback realized vol (5/10/22/66), log/squared returns, lagged vol, EWMA vol, GARCH(1,1) conditional vol as feature, Parkinson/Garman-Klass estimators, vol-of-vol, rolling skew/kurtosis, cross-asset features, calendar features
- [ ] Classical baselines: EWMA and GARCH(1,1) (arch library) with walk-forward harness
- [ ] LightGBM regression model tracked in MLflow (runs, metrics, registry, staging→prod promotion)
- [ ] Walk-forward / time-series CV only; metrics RMSE, MAE, QLIKE — each vs GARCH baseline
- [ ] FastAPI + Docker real-time inference service serving vol forecasts
- [ ] Evidently drift detection on features/predictions with alerting
- [ ] Closed feedback loop: log forecast vs realized vol (auto-arriving label), feed errors into monitoring/retraining
- [ ] Champion/challenger: promote only if challenger beats champion on rolling QLIKE
- [ ] Prefect orchestration DAG: ingest → validate → features → train → eval → register → deploy; scheduled + drift-triggered retrain
- [ ] GitHub Actions CI/CD: lint + tests + build (+ optional retrain kick-off)
- [ ] Streamlit observability dashboard: forecast-vs-realized, drift status, model version, latency
- [ ] DVC data versioning; MLflow model stages for rollback
- [ ] SHAP explainability on the gradient-boosted model
- [ ] MODEL_CARD.md with honest metrics, baselines, limitations; README with architecture diagram

### Out of Scope

- Trading strategy / signal generation — this is a forecasting + ML-systems demo, not a trading scheme
- Price direction prediction — not reliably learnable; volatility is the credible target
- Intraday horizon — stretch goal only; start daily
- Paid data feeds — free tiers only (ccxt/Binance public, yfinance)
- Deep learning (LSTM/TFT) — optional add-on, not required for v1
- Feast feature store — optional touch, deferred unless time permits
- Cloud deployment hard-requirement — local Docker first; cloud (AWS/Azure) is a deploy target once serving works

## Context

- Builder's profile: strong cloud/CI/CD/observability engineering; Financial Mathematics background; gap is the modeling/data half of ML engineering. The MLOps lifecycle is the star of this project; modeling stays pragmatic.
- Cross-asset coverage (24/7 crypto vs gapped equity sessions) is a deliberate robustness/feature-engineering talking point.
- Volatility clusters and mean-reverts, so it is genuinely learnable; ground-truth labels arrive automatically (tomorrow's realized vol is today's label), which makes the feedback loop genuine rather than simulated.
- Answers ML Engineer screening questions (Entain-style): production deployment, CI/CD for ML, monitoring/drift/retraining, feature engineering at scale.
- Honest reporting is a feature: if ML fails to beat GARCH in some regimes, the model card says so.

## Constraints

- **Tech stack**: Python; pandas/Polars, arch, LightGBM, MLflow, Evidently, Pandera, Prefect, FastAPI, Docker, Streamlit, DVC, GitHub Actions — chosen for free/open tooling and direct mapping to job-description keywords
- **Data**: free APIs only (ccxt/Binance, yfinance); pipeline must tolerate rate limits and gaps
- **Evaluation**: walk-forward / time-series CV only — random splits are a correctness violation (lookahead leakage)
- **Budget**: $0 infra requirement for local dev; cloud deploy uses free/cheap tiers
- **Timeline**: phased weekend-scale builds; Phases 0–3 alone must form a credible, shippable project
- **Platform**: Windows 11 dev machine; everything must run locally via Docker/docker-compose

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Volatility (not price) as target | Clusters/mean-reverts → learnable; respected classical baselines; automatic ground truth | — Pending |
| GARCH(1,1) + EWMA as baselines to beat | Eval-rigour centrepiece; benchmarking ML vs the right classical model is the credibility signal | — Pending |
| Walk-forward CV only | Prevents lookahead leakage; itself a correctness signal | — Pending |
| QLIKE alongside RMSE/MAE | Volatility-specific loss; quant credibility | — Pending |
| LightGBM primary model | Pragmatic, strong tabular performance; SHAP-explainable | — Pending |
| MLflow + Prefect + Evidently + Pandera stack | Standard open-source MLOps; maps to screening questions | — Pending |
| Local-first, Docker-compose; cloud later | Ship fast; cloud deploy is a phase, not a prerequisite | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-06-10 after initialization*
