# Walking Skeleton — VolForecast

**Phase:** 1
**Generated:** 2026-06-10

## Capability Proven End-to-End

> One sentence: the smallest user-visible capability that exercises the full stack.

Running `volforecast ingest --symbol BTC/USDT` fetches 2+ years of daily BTC OHLCV from a public crypto exchange via ccxt, drops the incomplete forming candle, passes a Pandera validation gate, writes `data/raw/crypto/BTC-USD.parquet`, and DVC tracks it — the entire trusted-data path (fetch → validate → store → version) proven on a single asset before broadening.

This is a data/MLOps pipeline with no UI. The "user" is the ML engineer (and the portfolio reviewer) operating the pipeline from the command line; the "interaction" is the `volforecast` CLI and the resulting versioned parquet artifact.

## Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Language / runtime | Python 3.12 (pinned floor `>=3.12`) | SHAP 0.52 forces `>=3.12`; widest verified wheel coverage across the whole stack (per RESEARCH.md Standard Stack) |
| Package layout | src layout, `src/volforecast/`, hatchling build backend | src layout resolves imports against the installed package not CWD; hatchling is uv's default for src-layout packages |
| Env / dependency manager | uv with committed `uv.lock` | `uv sync --locked` gives byte-reproducible installs locally, in Docker, and in CI; far faster than pip |
| CLI entry point | `[project.scripts] volforecast = "volforecast.cli:main"`; also `python -m volforecast.ingest` | Console script per locked decision D-Ingest; module form for CI/scripting |
| Data storage | Parquet, one file per asset, `data/raw/{asset_class}/{symbol}.parquet` and `data/processed/{asset_class}/{symbol}.parquet` | Locked decision; columnar, type-safe, DVC-versionable binary |
| Data versioning | DVC 3.x, per-subdirectory tracking (`data/raw/`, `data/processed/`), local cache remote, `.dvc` files committed to git | Locked decision; per-subdir keeps `.dvc` hashes small (RESEARCH Pitfall 7); $0 — no cloud remote for v1 |
| Crypto ingestion | ccxt, Binance default with Kraken fallback, exchange configurable via `config/assets.yaml` | Locked decision; public OHLCV needs no API key; one-line exchange swap for geo-block resilience |
| Equity ingestion | yfinance `[nospam]`, batched `yf.download(threads=False)`, explicit `auto_adjust=True`, tenacity retry/backoff | Locked decision; `threads=False` avoids the shared-global-dict race; explicit `auto_adjust` documents adjusted-return intent |
| Validation | Pandera 0.31+ (`import pandera.pandas as pa`), lazy validation, hard-fail gate, quarantine reports to `data/quarantine/` | Locked decision; schema-as-code, pytest-able, fast in-pipeline gate |
| Calendars | `exchange_calendars` XNYS for equities; continuous 24/7 `date_range` for crypto | Locked decision; no fabricated weekend equity rows |
| Backing infra (compose) | Postgres 16 backing both MLflow tracking server and Prefect server; Prefect worker; named volumes (never bind mounts) | Locked decision; SQLite-on-Windows-bind-mount corrupts (RESEARCH Pitfall 4) |
| CI | GitHub Actions, `astral-sh/setup-uv@v8`, `uv sync --locked`, ruff check + format, pytest on fixtures only — no live API calls | Locked decision; fixture-only CI is itself a portfolio talking point |
| Run command (full stack) | `docker compose -f infra/docker-compose.yml up` brings up Postgres + MLflow + Prefect; `volforecast ingest` runs host-side or in worker | No cloud deploy in v1; documented local full-stack run exercises the stack |

## Stack Touched in Phase 1

- [x] Project scaffold (pyproject.toml src layout, uv.lock, ruff config, pytest runner, package skeleton dirs)
- [x] CLI entry point — `volforecast ingest` console script wired through `[project.scripts]`
- [x] Data write — one real ccxt fetch → parquet write (`data/raw/crypto/BTC-USD.parquet`)
- [x] Data read — cache-first incremental read of existing parquet (merge-dedupe on date index)
- [x] Validation gate — one real Pandera schema enforced before any parquet write reaches `processed/`
- [x] Versioning — `dvc init` + `dvc add data/raw/` + committed `.dvc` pointer
- [x] Backing infra — docker-compose stack (Postgres + MLflow + Prefect server/worker)
- [x] CI — GitHub Actions lint + fixture-only tests on every push
- [x] Documented full-stack run command (`docker compose up` + `volforecast ingest`)

## Out of Scope (Deferred to Later Slices / Phases)

> Anything that is *not* in the skeleton. Explicit so future phases do not re-litigate Phase 1's minimalism.

- Feature engineering, target definition, RV computation — Phase 2
- Classical baselines (EWMA / GARCH / HAR-RV) and the walk-forward harness — Phase 2
- LightGBM model, MLflow run tracking/registry usage, SHAP — Phase 3 (the MLflow *server* runs in Phase 1, but no models are tracked yet)
- FastAPI serving, prediction log — Phase 3
- Evidently drift, alerting, Prefect *flows* (the Prefect server/worker run in Phase 1, but no DAG is defined yet) — Phase 4
- Streamlit dashboard, MODEL_CARD.md, README architecture diagram — Phase 5
- Cloud deploy, DVC hosted/cloud remote, intraday data, Feast — v2 / out of scope
- Binance geo-block contingency for cloud CI ingestion — revisited at cloud-deploy time (CI uses fixtures)

## Subsequent Slice Plan

Each later phase adds one vertical slice on top of this skeleton without altering its architectural decisions:

- **Phase 2:** Documented realized-vol target + single feature codepath + leak-free purged walk-forward harness scoring EWMA / GARCH(1,1) / HAR-RV baselines (consumes `data/processed/` and the `src/volforecast/` package layout)
- **Phase 3:** MLflow-tracked LightGBM challenger benchmarked on identical folds, served via Dockerized FastAPI with an append-only prediction log
- **Phase 4:** Auto-arriving labels, Evidently drift, QLIKE-gated champion/challenger promotion, Prefect DAG (ingest → validate → features → train → eval → register) with scheduled + drift-triggered retraining
- **Phase 5:** Streamlit observability dashboard, honest regime-segmented MODEL_CARD.md, README with architecture diagram
