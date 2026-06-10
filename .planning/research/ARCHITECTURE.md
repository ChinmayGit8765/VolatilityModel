# Architecture Research

**Domain:** Time-series ML forecasting platform with full MLOps lifecycle (volatility forecasting: crypto + equities)
**Researched:** 2026-06-10
**Confidence:** HIGH (overall structure is well-established MLOps canon; specifics verified against current MLflow/Prefect/Evidently docs)

## Standard Architecture

Time-series ML forecasting systems with MLOps converge on a single canonical shape: a **batch-oriented pipeline loop** (ingest → validate → features → train → evaluate → register) feeding a **thin online serving layer**, with a **monitoring/feedback loop** that closes back into retraining. Because the forecast horizon is daily, this is fundamentally a *batch system with an API on top* — not a streaming system. That one realization simplifies every boundary below.

### System Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│ ORCHESTRATION (Prefect 3 flows — thin wrappers, no business logic)     │
│   daily flow: ingest → validate → features → predict → log            │
│   retrain flow (scheduled + drift-triggered): train → eval → promote  │
└──────┬─────────────────────────────────────────────────────────────────┘
       │ calls functions in src/
┌──────▼─────────────────────────────────────────────────────────────────┐
│ DATA PLANE (batch, DVC-versioned parquet)                              │
│  ┌──────────┐   ┌───────────┐   ┌───────────┐   ┌──────────────────┐  │
│  │ Ingest   │──▶│ Validate  │──▶│ Features  │──▶│ Feature/label    │  │
│  │ ccxt +   │   │ Pandera   │   │ (single   │   │ datasets         │  │
│  │ yfinance │   │ gates     │   │ codepath) │   │ (parquet, DVC)   │  │
│  └──────────┘   └───────────┘   └───────────┘   └────────┬─────────┘  │
│   raw/           validated/      features/               │            │
└───────────────────────────────────────────────────────────┼───────────┘
                                                            │
┌───────────────────────────────────────────────────────────▼───────────┐
│ MODEL PLANE                                                            │
│  ┌──────────────────┐  ┌───────────────────┐  ┌────────────────────┐  │
│  │ Baselines        │  │ LightGBM trainer  │  │ Walk-forward eval  │  │
│  │ EWMA, GARCH(1,1) │  │ (logs to MLflow)  │  │ harness (shared by │  │
│  │ (arch)           │  │                   │  │ ALL models)        │  │
│  └────────┬─────────┘  └────────┬──────────┘  └─────────┬──────────┘  │
│           └─────────────────────┴────────────────────────┘            │
│                                 │                                      │
│                    ┌────────────▼─────────────┐                        │
│                    │ MLflow Tracking+Registry │                        │
│                    │ aliases: @champion       │                        │
│                    │          @challenger     │                        │
│                    └────────────┬─────────────┘                        │
└─────────────────────────────────┼──────────────────────────────────────┘
                                  │ load models:/vol_model@champion
┌─────────────────────────────────▼──────────────────────────────────────┐
│ SERVING PLANE                                                           │
│  ┌──────────────────────────┐        ┌─────────────────────────────┐   │
│  │ FastAPI (Docker)         │───────▶│ Prediction log              │   │
│  │ /forecast /health /model │ append │ (parquet or SQLite:         │   │
│  │ imports src/features     │        │  ts, asset, forecast,       │   │
│  └──────────────────────────┘        │  model_version, latency)    │   │
└───────────────────────────────────────┴──────────┬──────────────────────┘
                                                   │ joined with realized vol
┌──────────────────────────────────────────────────▼──────────────────────┐
│ MONITORING / FEEDBACK PLANE                                              │
│  ┌────────────────────┐  ┌─────────────────────┐  ┌──────────────────┐  │
│  │ Label materializer │─▶│ Evidently drift +   │─▶│ Champion/        │  │
│  │ (realized vol at   │  │ performance reports │  │ challenger gate  │  │
│  │  t+h = auto-label) │  │ (batch, daily)      │  │ (rolling QLIKE)  │  │
│  └────────────────────┘  └──────────┬──────────┘  └────────┬─────────┘  │
│                                     │ drift signal          │ promote    │
│                                     └──────▶ retrain flow ◀─┘ alias swap │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Streamlit dashboard (read-only consumer of all stores above)       │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| `src/ingest` | Pull OHLCV from ccxt (24/7) + yfinance (session calendars); normalize to one canonical schema; write `data/raw/` parquet per asset. Owns rate-limit/retry logic. | Idempotent incremental fetch keyed by (asset, date); per-source adapter classes behind one `fetch_ohlcv()` interface |
| `src/validate` | Pandera schemas + custom checks (gaps vs expected calendar, bad ticks, stale data, OHLC consistency). Pass → `data/validated/`; fail → halt pipeline with report. | Pandera `DataFrameSchema` per source; calendar-aware gap check (crypto expects every day, equities expect trading days only) |
| `src/features` | The **single** feature codepath: realized vol (5/10/22/66), returns, lagged/EWMA vol, GARCH cond. vol as feature, Parkinson/GK, vol-of-vol, rolling moments, cross-asset, calendar features. Also materializes labels (forward realized vol). | Pure functions `df → df`; imported by training, serving, AND label materializer — never duplicated |
| `src/models` | Baselines (EWMA, GARCH via `arch`), LightGBM trainer, and the walk-forward evaluation harness. Harness is model-agnostic: every model implements the same `fit/predict` protocol. | Trainer logs params/metrics/artifacts to MLflow; registers versions; harness produces RMSE/MAE/QLIKE per fold per asset |
| MLflow Tracking + Registry | Run history, metrics, model artifacts, version lineage. Promotion via **aliases** (`@champion`, `@challenger`) — stages are deprecated since MLflow 2.9. | `mlflow server` in docker-compose (SQLite/local artifacts fine for v1); load via `models:/vol_forecaster@champion` |
| `src/serving` | FastAPI app: `/forecast` (latest features → champion model → prediction), `/health`, `/model-info`. Appends every prediction to the prediction log. | Loads model once at startup + on alias change; imports `src/features` for any on-request computation; Dockerized |
| Prediction log | The **contract between serving and monitoring**: timestamp, asset, horizon, forecast, model version, latency. | Append-only parquet (or SQLite) at `data/predictions/`; schema validated by Pandera too |
| `src/monitoring` | (a) Label materializer: once t+h passes, compute realized vol and join onto logged forecasts. (b) Evidently `Report` with drift + regression presets on features and forecast-vs-realized. (c) Alert/trigger emission. | Batch job in daily flow; Evidently ≥0.7 API: `Report([DataDriftPreset()], include_tests=True).run(reference, current)`; HTML reports saved + boolean drift flag |
| Champion/challenger gate | Compares challenger vs champion on rolling QLIKE over the live feedback window; swaps registry aliases only if challenger wins. | Pure function over the joined forecast/realized table + `MlflowClient.set_registered_model_alias()` |
| `pipelines/` (Prefect 3) | DAG definitions only: daily scoring flow, retrain flow, monitoring flow. Schedules (cron) + drift-triggered runs. **No logic** — every task calls a `src/` function. | Prefect server + worker in docker-compose; deployments with cron schedules; retrain flow also invocable ad hoc by drift trigger |
| Streamlit dashboard | Read-only: forecast vs realized, rolling QLIKE champion vs baseline, drift status, current model version, latency. | Reads prediction log, monitoring outputs, MLflow registry; zero write access |
| DVC | Versions `data/` (raw/validated/features); pairs with git commit for reproducibility. | DVC tracks data, MLflow tracks models — never double-track artifacts |
| GitHub Actions | Lint (ruff) + tests (pytest) + Docker build on PR; optional manual retrain dispatch. | CI does not need live API keys — tests run on fixture data |

## Recommended Project Structure

The planned layout is correct and matches convention. Refined version:

```
.
├── src/volforecast/          # installable package (pip install -e .) — critical for
│   ├── ingest/                #   serving + pipelines importing the SAME code
│   │   ├── crypto.py          # ccxt adapter (24/7 calendar)
│   │   ├── equity.py          # yfinance adapter (exchange calendar)
│   │   └── base.py            # canonical OHLCV schema, fetch interface
│   ├── validate/
│   │   ├── schemas.py         # Pandera DataFrameSchemas
│   │   └── checks.py          # calendar-aware gap/staleness checks
│   ├── features/
│   │   ├── volatility.py      # RV, EWMA, Parkinson/GK, vol-of-vol
│   │   ├── returns.py         # log/squared returns, rolling moments
│   │   ├── garch_feature.py   # GARCH(1,1) conditional vol as feature
│   │   ├── calendar.py        # dow/month/session features
│   │   └── build.py           # assemble feature matrix + labels
│   ├── models/
│   │   ├── baselines.py       # EWMA, GARCH(1,1) wrappers (common protocol)
│   │   ├── lgbm.py            # LightGBM train/predict + MLflow logging
│   │   ├── evaluate.py        # walk-forward harness; RMSE/MAE/QLIKE
│   │   └── registry.py        # MLflow alias promotion helpers
│   ├── serving/
│   │   ├── app.py             # FastAPI app
│   │   ├── predictor.py       # model loading (models:/...@champion)
│   │   └── logging.py         # prediction log writer
│   ├── monitoring/
│   │   ├── labels.py          # realized-vol label materializer + join
│   │   ├── drift.py           # Evidently reports + drift flag
│   │   └── promotion.py       # champion/challenger gate (rolling QLIKE)
│   └── config.py              # asset universe, horizons, paths (pydantic-settings)
├── pipelines/                 # Prefect flows — orchestration only
│   ├── daily.py               # ingest → validate → features → predict → log → monitor
│   ├── retrain.py             # features → train → walk-forward eval → register → gate
│   └── deployments.py         # schedules, drift trigger wiring
├── data/                      # DVC-tracked, gitignored
│   ├── raw/  validated/  features/  predictions/  monitoring/
├── tests/
│   ├── unit/                  # feature math, QLIKE, schema checks (fixture data)
│   └── integration/           # pipeline segments end-to-end on fixtures
├── infra/
│   ├── docker-compose.yml     # mlflow, prefect server, prefect worker, api, streamlit
│   ├── Dockerfile.api
│   └── Dockerfile.worker
├── dashboard/app.py           # Streamlit (separate from src — it's a consumer)
├── .github/workflows/ci.yml
├── notebooks/                 # exploration only; nothing imports from here
├── dvc.yaml  pyproject.toml  MODEL_CARD.md  README.md
```

### Structure Rationale

- **`src/volforecast/` as an installable package:** the serving container, Prefect worker, tests, and notebooks all `import volforecast`. This is the only reliable way to guarantee one feature codepath everywhere (training/serving skew is the #1 killer of forecasting systems).
- **`pipelines/` separate from `src/`:** flows are deployment artifacts, not library code. Keeping them logic-free means everything is unit-testable without a running Prefect server.
- **`data/` with stage subfolders:** mirrors the validation gate — a file's location encodes how much trust it has earned. DVC versions the whole tree.
- **`dashboard/` outside `src/`:** Streamlit is a read-only consumer; it should never be importable by pipeline code.

## Architectural Patterns

### Pattern 1: Single Feature Codepath (anti-skew)

**What:** Feature computation lives only in `src/volforecast/features/` as pure `DataFrame → DataFrame` functions. Training, the daily scoring flow, and the FastAPI service all import it.
**When to use:** Always, for any system where the same features are needed offline and online.
**Trade-offs:** Slightly slower serving (recompute features on request) vs. zero skew risk. At daily horizon, latency is irrelevant — take the correctness.

```python
# features/build.py — used identically by trainer and server
def build_features(ohlcv: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """All windows end strictly at `asof`. No future rows ever enter."""
```

### Pattern 2: Common Model Protocol + Shared Walk-Forward Harness

**What:** EWMA, GARCH, and LightGBM all implement `fit(train_df)` / `predict(test_df)`. One walk-forward harness (expanding or rolling origin) evaluates all of them on identical folds, producing comparable RMSE/MAE/QLIKE.
**When to use:** Whenever "ML vs classical baseline" is the headline claim — identical folds are what make the comparison honest.
**Trade-offs:** GARCH refits per fold are slow (refit per fold, not per step; or refit every k steps). Worth it — this harness IS the project's credibility.

```python
class VolModel(Protocol):
    def fit(self, train: pd.DataFrame) -> None: ...
    def predict(self, test: pd.DataFrame) -> pd.Series: ...

def walk_forward(model: VolModel, panel: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    # returns per-fold, per-asset RMSE/MAE/QLIKE — same folds for every model
```

### Pattern 3: Alias-Based Champion/Challenger (MLflow ≥ 2.9)

**What:** Registry stages are deprecated. Use model version aliases: serving always loads `models:/vol_forecaster@champion`; retraining registers a new version as `@challenger`; the promotion gate swaps the alias only if challenger beats champion on rolling QLIKE over the live feedback window.
**When to use:** This is the current official MLflow pattern — do not build on `transition_model_version_stage`.
**Trade-offs:** Alias swap is atomic and instantly picked up on next model reload; rollback = point alias back at prior version.

```python
client.set_registered_model_alias("vol_forecaster", "champion", winning_version)
# serving: mlflow.lightgbm.load_model("models:/vol_forecaster@champion")
```

### Pattern 4: Prediction Log as the Serving↔Monitoring Contract

**What:** Every forecast served is appended to an immutable log (ts, asset, horizon, forecast, model_version, features_hash, latency). The monitoring plane never queries the API — it reads the log, joins realized vol once labels mature, and computes everything from that table.
**When to use:** Any system with delayed ground truth. Here labels arrive automatically at t+h, making the feedback loop genuine.
**Trade-offs:** Eventually-consistent metrics (you wait h days for labels) — that delay is real and should be visible on the dashboard, not hidden.

### Pattern 5: Thin Orchestrator

**What:** Prefect tasks are one-line wrappers: `@task def ingest_task(): return run_ingest(cfg)`. All retry-worthy logic, IO, and computation lives in `src/`.
**When to use:** Always. The DAG should read like the architecture diagram.
**Trade-offs:** None meaningful. Bonus: the whole pipeline runs as a plain Python script for debugging, no Prefect server needed.

### Pattern 6: Dual-Calendar Ingestion Behind One Schema

**What:** Crypto (every calendar day, UTC) and equities (exchange sessions, gaps legitimate) get separate adapters and separate Pandera gap expectations, but emit one canonical OHLCV schema. Downstream feature code is calendar-aware via an `expected_calendar` attribute per asset, never via `if asset == "BTC"` branches.
**When to use:** Any cross-asset system. This is the project's deliberate robustness talking point — make the boundary explicit.
**Trade-offs:** Cross-asset features (e.g., BTC vol as SPY feature) must handle non-overlapping days explicitly (forward-fill with staleness cap, and document it).

## Data Flow

### Daily Scoring Flow (the hot path)

```
[Prefect daily schedule]
    ↓
ingest (ccxt, yfinance) → data/raw/
    ↓
Pandera validation gate ──fail──▶ halt + alert (bad data never reaches features)
    ↓ pass → data/validated/
build_features(asof=today) → data/features/
    ↓
load models:/vol_forecaster@champion → predict next-h-day vol
    ↓
append to prediction log (data/predictions/)
    ↓
monitoring: materialize labels for forecasts whose t+h has passed
    ↓
Evidently report (feature drift + forecast-vs-realized) → data/monitoring/
    ↓
drift flag? ──yes──▶ trigger retrain flow
```

### Retrain / Promotion Flow

```
[cron schedule OR drift trigger OR manual dispatch]
    ↓
rebuild features over full history (DVC snapshot)
    ↓
walk-forward eval: EWMA, GARCH(1,1), LightGBM on identical folds
    ↓
MLflow: log runs/metrics → register new LightGBM version → alias @challenger
    ↓
champion/challenger gate: rolling QLIKE on live feedback window
    ↓ challenger wins?
yes → set alias @champion to new version (serving picks up on reload)
no  → keep champion; record decision in MLflow tags
```

### Online Request Flow

```
GET /forecast?asset=BTC
    ↓
FastAPI → latest validated data → build_features(asof=now) → champion.predict()
    ↓
append to prediction log → return {forecast, model_version, asof}
```

### Key Data Flows

1. **Trust gradient:** data moves raw → validated → features, gaining trust only by passing Pandera gates. Nothing skips a stage.
2. **Closed feedback loop:** prediction log + auto-arriving realized vol → joined error table → drives drift reports, the promotion gate, the dashboard, and retrain triggers. This single joined table is the most valuable artifact in the system.
3. **Model flow:** trainer → MLflow registry → alias → serving. Serving never touches training data; trainer never touches the API.

## Suggested Build Order

Dependencies dictate the order almost completely. Each step is shippable on its own.

| Order | Component | Depends On | Why This Position |
|-------|-----------|------------|-------------------|
| 0 | Repo scaffold, `pyproject.toml`, installable package, CI (lint+test), docker-compose skeleton | — | CI from day one is cheap; retrofitting is not. Matches builder's strengths — bank the easy win first |
| 1 | Ingest (ccxt + yfinance) + Pandera validation + DVC init | 0 | Everything downstream consumes validated data; dual-calendar handling decided here ripples everywhere |
| 2 | Feature pipeline + label materialization | 1 | Features needed by baselines, ML, and serving alike; leakage discipline (windows end at asof) established here |
| 3 | Baselines (EWMA, GARCH) + walk-forward harness + QLIKE | 2 | The harness BEFORE the ML model — the bar must exist before anything tries to clear it. End of phase 3 = credible shippable project per PROJECT.md constraint |
| 4 | LightGBM + MLflow tracking/registry (aliases) | 3 | First model versions exist; evaluated on the same folds as baselines |
| 5 | FastAPI + Docker serving + prediction log | 4 (needs a registered champion), 2 (imports features) | Prediction log schema defined here is the contract monitoring builds on |
| 6 | Monitoring: label join, Evidently drift, champion/challenger gate | 5 (needs prediction log), 4 (needs registry) | Feedback loop only becomes real once predictions are being logged |
| 7 | Prefect orchestration: daily + retrain flows, schedules, drift trigger | 1–6 (wraps all of them) | Orchestrate last — thin wrappers over already-working functions; trying to orchestrate first means debugging logic through a scheduler |
| 8 | Streamlit dashboard | 5, 6 (reads their stores) | Pure consumer; trivially built once stores exist |
| 9 | Cloud deploy, MODEL_CARD.md, SHAP, README diagram | all | Polish on a working system; SHAP needs the final model |

**Critical ordering insights:**
- **Harness before ML (3 before 4):** evaluating LightGBM without the GARCH bar already in place invites motivated reasoning.
- **Serving before monitoring (5 before 6):** monitoring consumes the prediction log; the log doesn't exist until serving writes it. (A backfill script can simulate history to test monitoring without waiting weeks.)
- **Orchestration near-last (7):** Prefect wraps working functions. Inverting this is the most common MLOps build mistake.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| v1 (≤10 assets, daily) | Everything above as-is: parquet files, SQLite MLflow backend, single docker-compose host. No databases needed |
| ~100 assets / intraday stretch | Swap parquet prediction log → Postgres/Timescale; Polars for feature builds; MLflow backend → Postgres; consider Feast only here |
| Multi-user / cloud | Object storage (S3) for DVC remote + MLflow artifacts; Prefect work pools on cloud workers; API behind a load balancer |

### Scaling Priorities

1. **First bottleneck:** GARCH refits in walk-forward eval (per-asset, per-fold). Fix: refit every k steps, parallelize per asset. Hits in phase 3, not production.
2. **Second bottleneck:** yfinance rate limiting / silent data revisions. Fix: incremental fetch with local cache plus periodic full-history re-pull to catch retroactive split/dividend adjustments.

## Anti-Patterns

### Anti-Pattern 1: Duplicated Feature Logic in Serving

**What people do:** Reimplement "just the few features the API needs" inside the FastAPI service.
**Why it's wrong:** Training/serving skew — the model sees subtly different inputs online than it was trained on, and nothing flags it. The most common silent failure in deployed forecasting.
**Do this instead:** Serving imports `volforecast.features.build`. The package-install structure exists precisely for this.

### Anti-Pattern 2: Business Logic Inside Prefect Tasks

**What people do:** Write ingestion/training code directly in `@task` bodies in `pipelines/`.
**Why it's wrong:** Untestable without a Prefect server; flows become a second source of truth; debugging happens through an orchestrator UI.
**Do this instead:** Tasks are one-line wrappers over `src/` functions. The full pipeline must run as plain Python.

### Anti-Pattern 3: Building on MLflow Stages

**What people do:** Follow older tutorials using `transition_model_version_stage("Staging"/"Production")`.
**Why it's wrong:** Stages are deprecated since MLflow 2.9 and slated for removal; the modern pattern is aliases, which also allow multiple aliases per version.
**Do this instead:** `@champion`/`@challenger` aliases; serve from `models:/name@champion`.

### Anti-Pattern 4: Drift-Triggered Auto-Promotion

**What people do:** Drift detected → retrain → deploy new model automatically.
**Why it's wrong:** A model retrained during a regime break can be worse than the champion; auto-deploying it is how monitoring systems cause incidents. Vol regime shifts are exactly when this bites.
**Do this instead:** Drift triggers *retraining and evaluation* only. Promotion happens solely through the rolling-QLIKE champion/challenger gate.

### Anti-Pattern 5: One Naive Daily Index for Both Asset Classes

**What people do:** Concatenate crypto and equity frames on a shared daily index, forward-filling equity weekends.
**Why it's wrong:** Fabricates zero-vol weekend "observations" for equities, corrupts realized vol windows, and makes the gap-validation gate meaningless.
**Do this instead:** Per-asset calendars carried through the pipeline; cross-asset features handle non-overlap explicitly with documented staleness caps.

### Anti-Pattern 6: Random or Standard K-Fold CV Anywhere

**What people do:** Use sklearn's default CV for hyperparameter tuning "just for speed."
**Why it's wrong:** Lookahead leakage; with overlapping rolling-window features, even adjacent-fold contamination inflates metrics. Invalidates the whole ML-vs-GARCH claim.
**Do this instead:** Walk-forward only, everywhere — including hyperparameter tuning inside each training fold. PROJECT.md already mandates this; enforce it structurally by making the harness the only evaluation entry point.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Binance via ccxt | REST polling, incremental since-last-timestamp fetch | Public endpoints, generous limits for daily OHLCV; build retry/backoff anyway |
| yfinance | Batch download per ticker, local cache | Unofficial API: rate limits, occasional schema breaks, **retroactive adjustment of historical prices** (splits/dividends) — periodic full re-pull + DVC versioning catches silent rewrites |
| MLflow server | docker-compose service; clients via `MLFLOW_TRACKING_URI` | SQLite backend + local artifact dir fine for v1 |
| Prefect server + worker | docker-compose; deployments with cron schedules; flow image has package baked in | Prefect 3: worker polls a work pool; flow-run containers need `volforecast` installed |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| ingest ↔ validate | parquet files in `data/raw/` → `data/validated/` | Pandera gate is the boundary; validate never calls APIs |
| features ↔ models | feature/label parquet in `data/features/` | Models never read raw/validated data directly |
| models ↔ serving | MLflow registry alias (`@champion`) | Only coupling is the model URI + feature schema |
| serving ↔ monitoring | prediction log (append-only) | Monitoring never calls the API; log schema is a versioned contract |
| monitoring ↔ pipelines | drift flag file / Prefect trigger | Drift triggers retrain flow, never promotion directly |
| everything ↔ dashboard | read-only file/registry access | Dashboard has zero write paths |

## Sources

- [MLflow Model Registry Workflows — aliases, champion/challenger](https://mlflow.org/docs/latest/ml/model-registry/workflow/) (HIGH — official)
- [MLflow RFC: deprecating model registry stages (#10336)](https://github.com/mlflow/mlflow/issues/10336) (HIGH — official)
- [Prefect 3 Deployments concepts](https://docs.prefect.io/v3/concepts/deployments) and [Run flows in Docker containers](https://docs.prefect.io/v3/deploy/infrastructure-examples/docker) (HIGH — official)
- [Evidently GitHub README — current Report API](https://github.com/evidentlyai/evidently) and [DataDriftPreset docs](https://docs.evidentlyai.com/metrics/preset_data_drift) (HIGH — official)
- [AWS: Robust time series forecasting with MLOps](https://aws.amazon.com/blogs/machine-learning/robust-time-series-forecasting-with-mlops-on-amazon-sagemaker/) (MEDIUM — vendor reference architecture, pattern-level)
- [MLOps architecture for time-series experiments (Medium)](https://medium.com/machine-learning-with-market-data/mlops-for-time-series-experiments-946e9135cfeb) (LOW-MEDIUM — community, corroborates retrain-compare-promote DAG pattern)
- Volatility-modeling specifics (QLIKE, walk-forward discipline, GARCH refit cadence, calendar handling): training-data domain knowledge, consistent with `arch` library documentation and standard econometrics practice (MEDIUM)

---
*Architecture research for: VolForecast — volatility forecasting MLOps platform*
*Researched: 2026-06-10*
