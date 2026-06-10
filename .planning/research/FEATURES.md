# Feature Research

**Domain:** Short-horizon realized volatility forecasting wrapped in a full MLOps lifecycle (ML Engineer portfolio / skill-demo project)
**Researched:** 2026-06-10
**Confidence:** HIGH (MLOps lifecycle features verified against ecosystem projects and vendor docs; vol-forecasting methodology verified against academic literature; complexity estimates MEDIUM — judgment-based)

## Context: Who "Users" Are

For a portfolio project, the "users" are **hiring managers and technical interviewers** scanning a repo for credibility signals. "Table stakes" means: missing it makes the project look like a tutorial clone or — worse — methodologically wrong. "Differentiator" means: signals senior-level judgment that generic MLOps zoomcamp-style projects (iris/churn/taxi-fare + MLflow + Prefect + Evidently) lack.

Two distinct credibility audiences exist and the feature set must satisfy both:
1. **ML/MLOps reviewers** — care about lifecycle completeness (tracking, registry, serving, monitoring, retraining, CI/CD)
2. **Quant-literate reviewers** — care about methodological correctness (right baselines, right loss, no leakage, honest reporting)

## Feature Landscape

### Table Stakes (Project Not Credible Without)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Multi-source OHLCV ingestion (ccxt/Binance + yfinance) with rate-limit & gap tolerance | Every forecasting project starts here; flaky ingestion = broken everything downstream | MEDIUM | Idempotent backfill + incremental update; handle yfinance silent failures and Binance pagination |
| Data validation gates before features (Pandera) | "Garbage in" is the #1 ML failure mode; reviewers check whether validation exists at all | LOW | Schema + null/gap/stale/outlier checks; fail-loud, quarantine bad batches |
| **Explicit realized-vol target definition** | The single most-scrutinized methodological choice. With daily bars, "realized vol" is a proxy (rolling std of log returns / squared returns / range-based) — must be documented, annualization convention stated | LOW | Decide once, document in README/MODEL_CARD. Inconsistent target = entire eval invalid |
| EWMA baseline (RiskMetrics λ=0.94 style) | The minimal classical reference; trivially cheap | LOW | A few lines with pandas ewm; also a feature input |
| GARCH(1,1) baseline (arch library) | The canonical "beat this or your ML is pointless" model; its absence marks the project as naive | MEDIUM | Walk-forward refit is slow — refit weekly/monthly, forecast daily; this is a known engineering wrinkle |
| **Walk-forward / expanding-window evaluation only** | Random splits on time series = lookahead leakage = instant disqualification by any quant reviewer | MEDIUM | The eval harness is the centrepiece artifact; all models (classical + ML) run through the same harness |
| QLIKE metric alongside RMSE/MAE | QLIKE is in the only class of loss functions robust for ranking vol forecasts (Patton 2011); using it signals domain literacy | LOW | QLIKE = RV/RV̂ − log(RV/RV̂) − 1; guard against zero/near-zero forecasts |
| ML model (LightGBM) with versioned feature pipeline | The "ML" in ML Engineer; tabular GBM is the pragmatic, defensible choice | MEDIUM | Lagged RV (5/10/22/66), returns, EWMA/GARCH-as-feature, calendar features minimum |
| MLflow experiment tracking + model registry | Universal expectation for any MLOps demo; "how do you version models?" screening answer | LOW-MEDIUM | Runs, params, metrics, artifacts; registry aliases for staging→prod |
| FastAPI + Docker inference service | "Have you deployed a model?" — the most common screening question | MEDIUM | Pydantic request/response, model loaded from registry, health endpoint, docker-compose |
| Drift detection on features + predictions (Evidently) | "How do you monitor models in production?" — second most common screening question | MEDIUM | Reference vs current windows; report generation + threshold alerting |
| Orchestrated pipeline DAG (Prefect): ingest → validate → features → train → eval → register | Manual notebook steps ≠ MLOps; a scheduled DAG is the lifecycle skeleton | MEDIUM | Scheduled runs; each task idempotent and independently retryable |
| CI/CD via GitHub Actions (lint + tests + Docker build) | Baseline software-engineering hygiene; absent CI reads as "data scientist, not engineer" | LOW | ruff + pytest + docker build; tests must cover leakage-sensitive feature code |
| README with architecture diagram + reproducible local setup | First 30 seconds of repo review; docker-compose up must work | LOW | One diagram showing the full loop; quickstart that actually runs on a fresh machine |

### Differentiators (Senior-Signal Add-ons)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **HAR-RV as a third baseline** | The academic-standard daily RV benchmark (lagged 1/5/22-day RV, plain OLS); literature shows it often beats GARCH on QLIKE. Costs ~30 lines, buys outsized quant credibility | LOW | Strongly recommended addition to the PROJECT.md baseline set — it is *the* RV benchmark in the literature |
| Diebold-Mariano significance test on forecast differences | Distinguishes "ML beat GARCH" from "ML beat GARCH, p<0.05" — the difference between a claim and evidence | LOW | statsmodels/scipy implementation; report per asset and horizon |
| **Champion/challenger promotion gated on rolling QLIKE** | Automated, metric-gated promotion is the senior MLOps pattern; most portfolio projects promote manually or not at all | MEDIUM | Challenger must beat champion on rolling out-of-sample QLIKE window before registry alias flips |
| **Closed feedback loop (forecast vs auto-arriving realized label)** | Volatility's killer property: tomorrow's realized vol IS the label — the feedback loop is genuine, not simulated. Almost no portfolio project has real ground-truth arrival | MEDIUM | Log every forecast; join with realized outcome when it arrives; error stream feeds monitoring + retrain triggers |
| Drift-**triggered** retraining (not just scheduled) | Event-driven retraining is what separates "monitoring exists" from "monitoring does something" | MEDIUM | Evidently drift signal or rolling-QLIKE degradation kicks off Prefect retrain flow → challenger eval |
| Range-based estimators as features (Parkinson, Garman-Klass; optionally Rogers-Satchell/Yang-Zhang) + vol-of-vol + rolling skew/kurtosis | Literature-verified: range estimators extract intraday information from OHLC and improve RV forecasts; signals real feature-engineering depth | LOW-MEDIUM | OHLC-only formulas — free given existing data; Yang-Zhang handles overnight gaps (equities) |
| Cross-asset features + 24/7-vs-session calendar handling | Crypto (24/7) and equities (gapped sessions) in one pipeline forces real engineering: trading calendars, alignment, per-asset annualization. A deliberate interview talking point | MEDIUM-HIGH | BTC vol as SPY feature and vice versa; exchange calendars; this is where naive pipelines silently break |
| Observability dashboard (Streamlit): forecast-vs-realized, drift status, model version, rolling metrics, latency | Makes the invisible lifecycle visible in a 2-minute demo; the artifact you screen-share in interviews | MEDIUM | Read from forecast log + Evidently outputs + MLflow registry; keep it read-only |
| SHAP explainability on the GBM | "Can you explain your model?" — interpretability is increasingly a screening topic | LOW | TreeExplainer is cheap for LightGBM; global importance + a few local examples in dashboard/model card |
| **MODEL_CARD.md with honest, regime-segmented results** | "ML beats GARCH in high-vol regimes, loses in calm regimes" is a *stronger* signal than a fake clean win; honest limitations read as senior judgment | LOW | Segment results by vol regime and asset class; state failure modes explicitly |
| DVC data versioning + fully pinned reproducibility | "Can you reproduce last month's model?" — closes the reproducibility loop (data + code + model all versioned) | MEDIUM | DVC for raw/feature data; MLflow for models; lockfiles for env |
| Rollback story (MLflow stages/aliases) | Deployment maturity = being able to undo; one registry-alias flip should restore the prior champion | LOW | Document and demo the rollback procedure |
| Crypto intraday true-RV target (sum of squared 5-min returns from Binance 1m bars) | Upgrades the target from proxy to *actual* realized variance for crypto — the academically correct RV. Free via Binance; impossible free for equities | MEDIUM-HIGH | Stretch: keep daily proxy for equities, true RV for crypto; document the asymmetry |
| Serving latency/throughput metrics (p50/p95) surfaced in dashboard | Ops-literacy signal aligned with builder's observability background | LOW | FastAPI middleware timing; matches existing strengths |

### Anti-Features (Deliberately NOT Building)

| Feature | Why Requested/Tempting | Why Problematic | Alternative |
|---------|------------------------|-----------------|-------------|
| Trading strategy / backtest PnL / signal generation | "Show it makes money" feels like the payoff | Converts a credible forecasting demo into a non-credible trading scheme; invites Sharpe-ratio scrutiny the project can't survive; dilutes the MLOps story | Forecast-accuracy framing only; say so explicitly in README ("forecasting system, not a trading strategy") |
| Price/return direction prediction | The "obvious" finance ML target | Daily price direction is ~unlearnable from free daily data; near-coin-flip results destroy credibility | Volatility target — clusters, mean-reverts, genuinely learnable, auto-labeled |
| Deep learning (LSTM/TFT/transformers) in v1 | Resume keyword appeal; papers show DL can beat GARCH | Weeks of tuning for marginal/unstable gains on small daily datasets; GPU friction; distracts from lifecycle star | LightGBM v1; DL as a clearly-scoped optional challenger later (the champion/challenger machinery makes adding it trivial) |
| Feature store (Feast) in v1 | JD keyword; "feature engineering at scale" | Heavy infra for a single-model, daily-batch project; serving/training skew is manageable with a shared feature module | Single versioned feature-pipeline package imported by both training and serving; Feast as documented future work |
| Real-time streaming (Kafka, websockets, tick data) | "Real-time" sounds impressive | Massive infra cost; daily horizon needs none of it; storage/compute blows the $0 budget | Daily batch + on-demand inference API; "intraday" stays an explicit stretch goal |
| Kubernetes deployment | Production-grade keyword | Overkill for one service + one dashboard; doubles ops surface on a Windows dev box | docker-compose locally; one cloud container service (ECS/Cloud Run/Azure ACA) as the later deploy phase |
| Massive ticker universe (50+ symbols) | "More data = more impressive" | Rate-limited free APIs choke; per-asset model quality drops; nothing new demonstrated after ~5 symbols | BTC, ETH, SPY + 2–4 large caps; depth over breadth |
| AutoML / large hyperparameter sweeps | "Rigorous tuning" appeal | Walk-forward eval makes sweeps expensive; overfitting-to-backtest risk; tuning is not the skill being demonstrated | Modest Optuna search (or sensible defaults) with tuning done *inside* the walk-forward protocol, documented |
| Random K-fold CV "for more data efficiency" | Standard ML habit | Lookahead leakage — future data trains models evaluated on the past; instantly invalidates all results | Walk-forward / expanding window only; write a test that asserts temporal ordering of splits |
| Multi-horizon forecasting (1/5/22-day) in v1 | Academic papers always do it | Triples eval surface and dashboard complexity before the lifecycle works end-to-end | Ship 1-day horizon through the full loop first; add horizons after — the harness should be horizon-parameterized from day one |

## Feature Dependencies

```
Ingestion (ccxt + yfinance)
    └──requires──> nothing (start here)

Validation gates ──requires──> Ingestion
Feature pipeline ──requires──> Validation gates
Target definition ──requires──> Validation gates   (compute RV proxy from clean data)

Baselines (EWMA, GARCH, HAR-RV) ──requires──> Target definition
Walk-forward harness ──requires──> Target definition
QLIKE / RMSE / MAE / DM-test ──requires──> Walk-forward harness

LightGBM model ──requires──> Feature pipeline + Walk-forward harness
MLflow tracking/registry ──requires──> LightGBM model (something to track)

FastAPI serving ──requires──> MLflow registry + Feature pipeline (same code path!)
Forecast logging ──requires──> FastAPI serving
Feedback loop (forecast vs realized) ──requires──> Forecast logging + Ingestion (labels arrive via next day's data)

Drift detection ──requires──> Feature pipeline + Forecast logging
Champion/challenger ──requires──> MLflow registry + Walk-forward harness + rolling QLIKE
Drift-triggered retrain ──requires──> Drift detection + Prefect DAG + Champion/challenger
Prefect DAG ──requires──> all pipeline stages existing as callable tasks

Dashboard ──requires──> Forecast logging + Drift detection + MLflow registry
SHAP ──requires──> LightGBM model
MODEL_CARD ──requires──> Walk-forward results across all models

Cross-asset features ──enhances──> LightGBM model
Range estimators ──enhances──> Feature pipeline (OHLC already ingested — near-free)
Crypto intraday true-RV ──enhances──> Target definition (crypto only; stretch)
DVC ──enhances──> reproducibility of everything

Trading strategy ──conflicts──> honest-forecasting positioning (core value)
Random CV ──conflicts──> Walk-forward harness (correctness violation)
```

### Dependency Notes

- **Serving requires the same feature code as training:** the single highest-risk dependency. If the API recomputes features with different code, training/serving skew silently invalidates the whole demo. One shared, versioned feature module is a hard architectural requirement.
- **Feedback loop requires ingestion, not extra infra:** tomorrow's OHLCV ingest *is* the label-delivery mechanism. This is why the loop is genuine — design the forecast log schema so the join (forecast_date, asset, horizon → realized RV) is trivial.
- **Champion/challenger requires the walk-forward harness:** promotion decisions reuse the same eval machinery as offline benchmarking — build the harness as a library, not a script.
- **GARCH baseline is the schedule risk inside table stakes:** per-step refitting in walk-forward is computationally the heaviest piece; plan the refit-frequency compromise early.

## MVP Definition

### Launch With (v1 — Phases 0–3 must be shippable alone, per PROJECT.md)

- [ ] Ingestion (BTC, ETH, SPY + 2–3 large caps) with validation gates — everything downstream depends on it
- [ ] Documented RV target + feature pipeline (lagged RV, returns, EWMA, range estimators, calendar) — range estimators are near-free given OHLC
- [ ] EWMA + GARCH(1,1) + **HAR-RV** baselines — HAR-RV added: lowest-cost/highest-credibility item in this document
- [ ] Walk-forward harness producing QLIKE/RMSE/MAE per asset vs baselines — the credibility centrepiece
- [ ] LightGBM challenger tracked in MLflow (runs + registry) — the ML half
- [ ] FastAPI + Docker serving from registry, sharing the feature module — the "deployed a model" answer
- [ ] Honest results table in README (even if ML loses to HAR-RV/GARCH somewhere) — honest reporting is a stated feature

### Add After Validation (v1.x)

- [ ] Prefect DAG + scheduling — trigger: manual end-to-end run works twice
- [ ] Forecast logging + feedback loop — trigger: serving stable for a week of daily runs
- [ ] Evidently drift detection + alerting — trigger: forecast log has enough history for reference windows
- [ ] Champion/challenger promotion on rolling QLIKE — trigger: at least two model versions exist
- [ ] Drift-triggered retraining — trigger: drift detection + champion/challenger both proven
- [ ] Streamlit dashboard — trigger: there is real monitoring data to show
- [ ] GitHub Actions CI/CD — can land any time after tests exist; cheap, do early
- [ ] SHAP + MODEL_CARD.md + Diebold-Mariano tests — trigger: final eval numbers stable
- [ ] DVC versioning — trigger: data pipeline schema stabilizes

### Future Consideration (v2+)

- [ ] Crypto intraday true-RV target (5-min realized variance) — defer: changes target definition; do after the daily loop is trusted
- [ ] Multi-horizon forecasts (5/22-day) — defer: multiplies eval/dashboard surface
- [ ] Cloud deployment (ECS/Cloud Run/ACA) — defer per PROJECT.md: a deploy target, not a prerequisite
- [ ] Optional DL challenger (LSTM/TFT) — defer: champion/challenger machinery makes this a clean later experiment
- [ ] Feast feature store — defer: document as "what I'd add at scale" in README instead

## Feature Prioritization Matrix

| Feature | Reviewer Value | Implementation Cost | Priority |
|---------|----------------|---------------------|----------|
| Walk-forward harness + QLIKE | HIGH | MEDIUM | P1 |
| GARCH/EWMA/HAR-RV baselines | HIGH | LOW-MEDIUM | P1 |
| Ingestion + Pandera validation | HIGH | MEDIUM | P1 |
| Shared feature pipeline (train = serve) | HIGH | MEDIUM | P1 |
| LightGBM + MLflow tracking/registry | HIGH | MEDIUM | P1 |
| FastAPI + Docker serving | HIGH | MEDIUM | P1 |
| Honest README results table | HIGH | LOW | P1 |
| Prefect DAG | HIGH | MEDIUM | P2 |
| Feedback loop (forecast vs realized) | HIGH | MEDIUM | P2 |
| Evidently drift + alerting | HIGH | MEDIUM | P2 |
| Champion/challenger QLIKE gate | HIGH | MEDIUM | P2 |
| GitHub Actions CI/CD | MEDIUM | LOW | P2 |
| Streamlit dashboard | MEDIUM | MEDIUM | P2 |
| Drift-triggered retrain | HIGH | MEDIUM | P2 |
| SHAP + MODEL_CARD + DM test | MEDIUM | LOW | P2 |
| DVC versioning | MEDIUM | MEDIUM | P3 |
| Cross-asset features | MEDIUM | MEDIUM-HIGH | P3 |
| Crypto intraday true-RV | MEDIUM | MEDIUM-HIGH | P3 |
| Multi-horizon, cloud deploy, DL challenger | LOW-MEDIUM | HIGH | P3 |

## Competitor Feature Analysis

| Feature | Generic MLOps portfolio projects (zoomcamp-style: taxi/churn/iris) | Academic vol-forecasting work (papers, quant repos) | Our Approach |
|---------|---------------------------------------------------------------------|------------------------------------------------------|--------------|
| Domain target | Toy regression/classification, no domain stakes | Rigorous RV targets, intraday data, no production system | Daily RV proxy with rigorous definition; production lifecycle around it |
| Baselines | Usually none (or trivial mean) | GARCH/HAR-RV always; DM tests standard | GARCH + EWMA + HAR-RV, DM tests — academic rigor inside an MLOps repo |
| Evaluation | Random splits common (a flaw to exploit) | Walk-forward + QLIKE standard | Walk-forward + QLIKE as a *stated correctness feature* |
| Ground-truth feedback | Simulated/absent (labels never arrive) | N/A (offline studies) | Genuine auto-arriving labels — the project's structural advantage |
| Lifecycle (registry, drift, retrain, CI/CD) | Present — this is their whole value | Absent | Full lifecycle, equal to the best generic projects |
| Champion/challenger | Rare; promotion usually manual | N/A | Automated QLIKE-gated promotion — rare in both camps |
| Honest reporting | "Model achieves 0.97 accuracy" theatre | Honest but inaccessible | MODEL_CARD with regime-segmented wins *and* losses |

**Positioning conclusion:** the intersection is empty in the searched ecosystem — no found project combines correct vol-forecasting methodology (HAR-RV/GARCH baselines, QLIKE, walk-forward) with a complete MLOps lifecycle (registry, drift, champion/challenger, drift-triggered retrain). The differentiation is the *combination*, not any single feature.

## Sources

- [Patton-class robust loss / QLIKE usage and walk-forward standards in RV forecasting (arXiv 2606.09478)](https://arxiv.org/abs/2606.09478) — HIGH confidence (academic standard, multiple corroborating papers)
- [GARCH vs deep learning RV forecasting comparison, Computational Economics](https://link.springer.com/article/10.1007/s10614-024-10694-2) — MEDIUM confidence
- [Volatility Forecasting with ML and Intraday Commonality, Journal of Financial Econometrics](https://academic.oup.com/jfec/article/22/2/492/7081291) — HIGH confidence
- [HAR model overview, Portfolio Optimizer blog](https://portfoliooptimizer.io/blog/volatility-forecasting-har-model/) — MEDIUM confidence
- [Range estimators (Parkinson/Garman-Klass/Rogers-Satchell/Yang-Zhang) improving RV forecasts, Finance Research Letters](https://www.sciencedirect.com/science/article/pii/S1544612323003641) — MEDIUM confidence
- [Databricks MLOps workflow docs — champion/challenger via registry aliases](https://docs.databricks.com/aws/en/machine-learning/mlops/mlops-workflow) — HIGH confidence (official vendor docs)
- [ModelMesh — drift detection triggering Prefect retrain DAGs](https://github.com/shreeyansh17/modelmesh) — MEDIUM confidence (single repo, pattern corroborated by Databricks docs)
- [mlops-zoomcamp-project — representative generic MLOps portfolio feature set](https://github.com/KonuTech/mlops-zoomcamp-project) — MEDIUM confidence
- [dpleus/mlops — Prefect + MLflow + FastAPI + Prometheus/Grafana + Streamlit reference](https://github.com/dpleus/mlops) — MEDIUM confidence
- GitHub topic scans: [volatility-forecasting](https://github.com/topics/volatility-forecasting), [mlops-project](https://github.com/topics/mlops-project?l=python) — used for competitor gap analysis; LOW-MEDIUM confidence (search coverage, not exhaustive)

---
*Feature research for: realized volatility forecasting MLOps platform (VolForecast)*
*Researched: 2026-06-10*
