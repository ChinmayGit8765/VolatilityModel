# Stack Research

**Domain:** Financial volatility forecasting MLOps platform (Python, time-series ML, crypto + equities)
**Researched:** 2026-06-10
**Confidence:** HIGH (all versions verified against PyPI metadata on 2026-06-10; key compatibility constraints verified from package `requires_dist`)

## Headline Findings (read these first)

1. **Pin pandas to 2.3.x, NOT pandas 3.0.** pandas 3.0 shipped 2026-01-21 with breaking changes (default `str` dtype, mandatory Copy-on-Write), and **MLflow 3.13.0 hard-pins `pandas<3`** in its dependencies (verified from PyPI metadata). The candidate stack works only on the pandas 2.x line for now. Confidence: HIGH.
2. **MLflow model registry "stages" (Staging/Production) are deprecated — use aliases.** Stages were deprecated in MLflow 2.9 and remain deprecated in 3.x. The replacement — model version aliases (`models:/VolForecast@champion`) — actually maps *better* to this project's champion/challenger requirement than stages do. PROJECT.md's "staging→prod promotion" and "MLflow model stages for rollback" requirements should be reworded to aliases (`@champion` / `@challenger`) + tags (`validation_status`). Confidence: HIGH (MLflow official docs + GitHub RFC #10336).
3. **SHAP 0.52.0 requires Python >=3.12 and numpy>=2.** This forces the project's Python floor. **Recommend Python 3.12** — every package in the stack explicitly supports it. Confidence: HIGH.
4. **Evidently 0.7.x has a completely rewritten API** (`Report` / `Dataset` / `DataDefinition`). Most blog tutorials online still show the legacy 0.4-era API (`ColumnMapping`, old `Report.run` signature) — they will not work. Use only official docs.evidentlyai.com examples. Confidence: HIGH.
5. **Pandera now requires the `import pandera.pandas as pa` namespace** (introduced 0.24; top-level access deprecated as of 0.29; current release is 0.31.1). Old-style `import pandera as pa` examples in tutorials emit warnings/break. Confidence: HIGH.
6. **yfinance crossed to 1.x** (current 1.4.1) with no breaking changes from 0.2.x, but `curl_cffi` is now a core dependency and there's a new config system replacing old patterns. Confidence: HIGH.

## Recommended Stack

### Runtime & Core Data

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.12.x | Runtime | Forced floor by SHAP 0.52 (`>=3.12`); explicitly supported by every package in this stack. 3.13 also works but 3.12 has the widest verified wheel coverage (LightGBM classifiers top out at 3.13; Evidently at 3.13). |
| pandas | >=2.3,<3 | DataFrames, time-series indexing | **Must stay <3**: MLflow 3.13 pins `pandas<3`. pandas 2.3 is the final, stable 2.x line. Revisit pandas 3 only when MLflow lifts the pin. |
| numpy | >=2,<3 | Numerics | SHAP 0.52 requires `numpy>=2`; MLflow and arch require `numpy<3`. numpy 2.x is now the ecosystem default (current: 2.4.6). |
| ccxt | 4.5.x (current 4.5.56) | Crypto OHLCV ingestion (BTC, ETH) | De-facto standard unified exchange API; public market-data endpoints need no API key. Releases near-daily — pin minor, allow patch updates. |
| yfinance | >=1.4 (current 1.4.1) | Equity OHLCV ingestion (SPY + large caps) | Standard free Yahoo data client. 1.x added `curl_cffi` as core dep (impersonates browser TLS to dodge Yahoo blocking). Install the `nospam` extra (`requests_cache` + `requests_ratelimiter`) to survive rate limits. Treat as a flaky dependency: cache raw pulls to disk and version with DVC so the pipeline never depends on Yahoo being up. |

### Modeling

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| arch | 8.0.0 | GARCH(1,1)/EWMA baselines | The canonical Python univariate volatility library (Kevin Sheppard). 8.0 requires Python >=3.10, numpy `>=1.22.3,<3` — compatible with the numpy 2 pin. |
| LightGBM | 4.6.0 | Primary ML regressor | Best tabular speed/accuracy tradeoff for small-to-mid feature matrices; native SHAP TreeExplainer support; trivial CPU training on daily-frequency data. Windows wheels ship binaries; **Docker slim images need `libgomp1` installed** (OpenMP runtime — classic build failure). |
| statsmodels | latest (arch dep, >=0.13) | Diagnostics (Ljung-Box, etc.) | Pulled in by arch anyway; use for residual diagnostics in the model card. |
| scikit-learn | >=1.5,<2 | Metrics, TimeSeriesSplit scaffolding | MLflow pins `scikit-learn<2`. Use `TimeSeriesSplit` only as a building block — write your own walk-forward harness with purge/embargo since `TimeSeriesSplit` alone doesn't handle multi-asset panels. |
| SHAP | 0.52.0 | Explainability | TreeExplainer is exact and fast for LightGBM. 0.52 forces Python >=3.12, numpy>=2 — this is the binding constraint of the whole stack. |

### MLOps Lifecycle

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| MLflow | 3.x (current 3.13.0) | Tracking + model registry | Industry-standard open registry. **Use aliases not stages**: `@champion` / `@challenger` aliases + `validation_status` tags. MLflow 3 removed Recipes, fastai/mleap flavors, and the old deployment-server CLI — none of which this project needs; `mlflow models serve` and pyfunc flavors remain. |
| Prefect | 3.x (current 3.7.4) | Orchestration DAG | Lighter than Airflow, Pythonic flows/tasks, native scheduling + event-driven runs (drift-triggered retrain). Run `prefect server` as a docker-compose service; run the worker in a container too — avoids Windows-host process-worker quirks entirely. |
| Evidently | 0.7.x (current 0.7.21) | Drift detection | OSS standard for tabular drift. Current API: `Report([DataDriftPreset()])` + `Dataset.from_pandas(df, data_definition=DataDefinition(...))`. No pandas upper-bound pin (`pandas>=1.3.5`), so it follows our pandas 2.3 pin cleanly. |
| Pandera | 0.31.1 | Data validation gates | Schema-as-code with statistical checks (ranges, monotonic dates, null policies) — better fit than Great Expectations for in-pipeline gating. **Import as `import pandera.pandas as pa`.** Also has a Polars backend if Polars is added later. |
| DVC | 3.x (current 3.67.1) | Data versioning | Versions raw OHLCV pulls + feature parquet alongside Git. For $0 budget: start with a **local directory remote**; optionally use DagsHub's free hosted DVC remote for public reproducibility. Avoid the Google Drive remote (long-standing OAuth breakage). |
| FastAPI | 0.136.x | Inference service | Standard Python serving layer; requires `pydantic>=2.9` (use Pydantic v2 models for request/response schemas). Load model via `models:/VolForecast@champion` URI at startup; expose `/forecast`, `/health`, `/model-info`. |
| uvicorn | latest | ASGI server | Standard pairing with FastAPI (`uvicorn[standard]`). |
| Streamlit | 1.58.0 | Observability dashboard | Fastest path to forecast-vs-realized charts, drift status, model version, latency panels. Pins `pandas<4`, `numpy<3` — compatible. |
| Docker / docker-compose | Docker Desktop (WSL2 backend) | Local platform | Compose services: `mlflow-server`, `prefect-server`, `prefect-worker`, `api` (FastAPI), `dashboard` (Streamlit), `postgres`. See Windows notes below. |
| PostgreSQL | 16/17 (official image) | MLflow + Prefect backend store | **Do not use SQLite on a Windows bind mount** — SQLite file locking over the WSL2/NTFS boundary causes corruption and `database is locked` errors. One Postgres container backing both MLflow and Prefect is the robust $0 fix (or at minimum keep SQLite on a named volume, never a bind mount). |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| uv | Package/env manager | 2026 standard; `uv sync` from `pyproject.toml` + `uv.lock` gives reproducible envs locally, in Docker, and in CI (`astral-sh/setup-uv` action). Far faster than pip in CI. |
| Ruff | Lint + format | Replaces flake8/isort/black with one tool; trivial CI step. |
| pytest | Tests | Unit-test feature functions (leakage tests: assert features at t use only data ≤ t) and Pandera schemas. |
| pre-commit | Hooks | Ruff + whitespace + `*.sh eol=lf` enforcement. |
| GitHub Actions | CI/CD | `actions/checkout@v5`, `astral-sh/setup-uv@v5`(+) or `actions/setup-python@v5` with Python 3.12; jobs: lint → test → docker build; optional `workflow_dispatch` retrain trigger. |

## Installation

```bash
# pyproject.toml managed via uv
uv init && uv add \
  "pandas>=2.3,<3" "numpy>=2,<3" \
  "ccxt>=4.5" "yfinance[nospam]>=1.4" \
  "arch>=8,<9" "lightgbm>=4.6,<5" "scikit-learn>=1.5,<2" "shap>=0.52" \
  "mlflow>=3.13,<4" "prefect>=3.7,<4" "evidently>=0.7.21,<0.8" \
  "pandera[pandas]>=0.31" "dvc>=3.67" \
  "fastapi>=0.136" "uvicorn[standard]" "pydantic>=2.9" \
  "streamlit>=1.58" "psycopg2-binary"

# Dev
uv add --dev ruff pytest pre-commit httpx
```

```dockerfile
# Dockerfile note for LightGBM on slim images
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| pandas 2.3 | Polars 1.41 | Only if data scale grows to intraday/tick. At daily OHLCV (~5k rows × ~15 tickers), Polars buys nothing and adds an interchange seam — arch, LightGBM sklearn API, Evidently, SHAP, and Streamlit all speak pandas/numpy natively. Drop Polars from v1; PROJECT.md's "pandas/Polars" should resolve to pandas. |
| Prefect 3 | Dagster | Dagster's asset model is arguably nicer for data pipelines, but Prefect is lighter to self-host on docker-compose, already a stated JD keyword for this project, and 3.x event triggers cover drift-triggered retraining. |
| Prefect 3 | Airflow 3 | Airflow is heavyweight for a single-machine portfolio project (scheduler + triggerer + webserver + metadata DB) and is miserable on Windows outside containers. |
| Evidently 0.7 | NannyML, whylogs | NannyML is great for performance-estimation-without-labels — but this project *has* auto-arriving labels (tomorrow's realized vol), so Evidently's simpler drift presets + the genuine feedback loop suffice. |
| Pandera | Great Expectations | GX is heavier (data contexts, stores, docs sites) and awkward to embed as a fast in-pipeline gate. Pandera schemas are plain Python, pytest-able, and Prefect-task-friendly. |
| FastAPI custom service | `mlflow models serve` / BentoML | The custom FastAPI service is deliberate portfolio surface area (request validation, model-info endpoint, latency logging). `mlflow models serve` is fine for smoke tests; BentoML adds a framework to learn without adding interview value here. |
| LightGBM | XGBoost | Near-equivalent; LightGBM is faster on CPU and already the stated choice. Don't run both in v1. |
| arch (GARCH) | statsforecast (Nixtla) | statsforecast has a fast GARCH too, but `arch` is the reference implementation with richer diagnostics — better for the model-card credibility story. |
| ccxt (Binance) | exchange-native SDKs | ccxt's unified API means swapping Binance → Kraken/Coinbase is a one-line exchange-id change — important because Binance geo-blocks some regions and many cloud-DC IPs (relevant when cloud deploy happens). |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| pandas 3.0.x | MLflow 3.13 pins `pandas<3`; also breaking changes (default `str` dtype, mandatory CoW) ripple through the ecosystem | `pandas>=2.3,<3`; revisit when MLflow lifts pin |
| MLflow registry stages (`transition_model_version_stage`) | Deprecated since 2.9; UI and docs steer to aliases; building new work on it is dead-end | Aliases `@champion`/`@challenger` + model version tags |
| Evidently <0.7 examples (`ColumnMapping`, legacy `Report`) | API fully rewritten in 0.7; most blog/Medium tutorials are stale and won't run | Current `Report`/`Dataset`/`DataDefinition` API from official docs |
| `import pandera as pa` (top-level) | Deprecated as of 0.29 in favor of per-backend namespaces | `import pandera.pandas as pa` |
| SQLite backend on Windows bind mounts (MLflow/Prefect) | File locking across WSL2/NTFS boundary → `database is locked`, corrupted runs | Postgres container in compose (or named volume at minimum) |
| Python 3.11 or earlier | SHAP 0.52 requires >=3.12 | Python 3.12 |
| Python 3.14 | LightGBM 4.6 and Evidently 0.7 don't declare 3.14 support yet | Python 3.12 (or 3.13) |
| DVC Google Drive remote | Long-standing OAuth verification breakage | Local directory remote; DagsHub free tier if a hosted remote is wanted |
| Feast feature store (v1) | Already out-of-scope in PROJECT.md; substantial infra for zero modeling gain at this scale | Versioned parquet feature files + DVC |
| `TimeSeriesSplit` as the whole eval story | Doesn't handle multi-asset panels, purging, or embargo | Custom walk-forward harness (expanding window, per-asset, with embargo), logged to MLflow |
| Pinning yfinance exactly | Yahoo breaks scrapers regularly; patches land often | `yfinance>=1.4` + raw-data caching via DVC so reruns don't need Yahoo |

## Stack Patterns by Variant

**If Binance public API is geo-blocked from your network/cloud region:**
- Use ccxt with `kraken` or `coinbase` exchange id (same unified `fetch_ohlcv` call)
- Because Binance blocks many cloud-datacenter and some country IPs; ccxt makes the swap a config value, not a rewrite

**If MLflow lifts the `pandas<3` pin (watch their release notes):**
- Migrate to pandas 3 deliberately in its own PR; test for `str` dtype inference changes in feature code and any chained-assignment patterns
- Because pandas 3 CoW will silently change copy/view behavior in rolling-window feature code

**If you later add intraday data (stretch goal):**
- Introduce Polars for the feature pipeline only, with Pandera's Polars backend for validation, converting to pandas at model boundaries
- Because data volume then justifies the interchange seam

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| mlflow 3.13.0 | pandas <3, numpy <3, scikit-learn <2, pyarrow >=4,<25 | Verified from PyPI `requires_dist` — **the binding ecosystem constraint** |
| shap 0.52.0 | Python >=3.12, numpy >=2 | Verified from PyPI — forces the Python floor and numpy 2 |
| arch 8.0.0 | Python >=3.10, numpy >=1.22.3,<3, pandas >=1.4 | Verified from PyPI — fine with numpy 2 / pandas 2.3 |
| lightgbm 4.6.0 | Python 3.7–3.13 (classifiers) | Docker slim needs `libgomp1`; SHAP TreeExplainer supports it natively |
| evidently 0.7.21 | pandas >=1.3.5 (no upper pin), numpy >=1.23, Python 3.10–3.13 | Verified from PyPI |
| prefect 3.7.4 | Python >=3.10,<3.15 | Run server + worker in containers on Windows |
| fastapi 0.136.3 | pydantic >=2.9, Python >=3.10 | Pydantic v2 only |
| streamlit 1.58.0 | pandas <4, numpy <3 | Compatible with the pandas 2.3 / numpy 2 pins |
| pandera 0.31.1 | Python >=3.10; pandas + polars backends | Use `pandera[pandas]` extra; `import pandera.pandas as pa` |
| yfinance 1.4.1 | curl_cffi >=0.15 (core dep) | curl_cffi ships Windows + manylinux wheels; works in Docker |
| dvc 3.67.1 | Python 3.9–3.14 | Windows-supported; watch CRLF in any hook scripts |

## Windows 11 + Docker Notes (project constraint)

- **Docker Desktop with WSL2 backend** is required; everything stateful goes in compose.
- **Line endings:** add `.gitattributes` with `*.sh text eol=lf` — CRLF in entrypoint scripts is the #1 "works on my machine, dies in container" failure on Windows. (HIGH confidence, well-documented.)
- **Bind-mount performance:** mounting `C:\...` code into Linux containers is slow for file-watching (Streamlit/uvicorn `--reload`). Acceptable at this project size; if it hurts, move the repo into the WSL filesystem. (MEDIUM confidence.)
- **State on named volumes, not bind mounts:** Postgres data, MLflow artifacts. Bind-mounted SQLite is the known killer (see What NOT to Use).
- **Native Windows installs all work** for the Python packages above (LightGBM, SHAP, curl_cffi, DVC ship Windows wheels) — you can develop host-side and run services in compose. (HIGH confidence from wheel availability.)

## Confidence Assessment

| Claim | Confidence | Basis |
|-------|------------|-------|
| All version numbers | HIGH | PyPI JSON metadata fetched 2026-06-10 |
| MLflow `pandas<3` pin | HIGH | PyPI `requires_dist` for mlflow 3.13.0 |
| Stages deprecated → aliases | HIGH | MLflow official docs + RFC mlflow/mlflow#10336 |
| Evidently 0.7 API shape | HIGH | docs.evidentlyai.com quickstart |
| Pandera namespace change | HIGH | pandera.readthedocs.io (0.24 intro, 0.29 deprecation) |
| pandas 3.0 breaking changes | HIGH | pandas.pydata.org whatsnew/v3.0.0 |
| yfinance 1.0 "no breaking changes" | MEDIUM | GitHub release notes via search; verify config deprecation warnings on first run |
| SQLite-on-bind-mount failures | MEDIUM | Widely reported community pattern; not from a single official doc |
| Prefect worker quirks on Windows host | LOW-MEDIUM | Community reports; containerized worker sidesteps it regardless |

## Sources

- https://pypi.org/pypi/{mlflow,prefect,evidently,pandera,lightgbm,arch,polars,yfinance,ccxt,shap,dvc,pandas,fastapi,streamlit,numpy}/json — versions + `requires_dist` constraints (HIGH)
- https://mlflow.org/docs/latest/ml/model-registry/ — aliases/tags workflow (HIGH)
- https://mlflow.org/docs/3.0.1/mlflow-3/breaking-changes — MLflow 3 removals (HIGH)
- https://github.com/mlflow/mlflow/issues/10336 — stages deprecation RFC (HIGH)
- https://pandas.pydata.org/docs/whatsnew/v3.0.0.html — pandas 3.0 (2026-01-21) breaking changes (HIGH)
- https://pandera.readthedocs.io/en/stable/ — `pandera.pandas` namespace (HIGH)
- https://docs.evidentlyai.com/quickstart_ml — current Evidently API (HIGH)
- https://github.com/ranaroussi/yfinance/releases/tag/1.0 — yfinance 1.0 notes (MEDIUM)

---
*Stack research for: VolForecast — volatility forecasting MLOps platform*
*Researched: 2026-06-10*
