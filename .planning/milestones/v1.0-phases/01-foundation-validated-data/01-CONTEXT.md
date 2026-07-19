# Phase 1: Foundation & Validated Data - Context

**Gathered:** 2026-06-10
**Status:** Ready for planning
**Mode:** Autonomous smart discuss (recommended answers auto-accepted per user directive)

<domain>
## Phase Boundary

A reproducible local stack on Windows 11 that ingests, validates, and versions 2+ years of cross-asset daily OHLCV (BTC, ETH via ccxt; SPY + 2 large caps via yfinance). Delivers: installable `volforecast` package on Python 3.12 with pinned deps, docker-compose stack (Postgres-backed MLflow + Prefect server/worker), GitHub Actions CI (lint + tests, fixtures only), cache-first incremental ingestion CLI, Pandera validation gates with per-asset-class calendars, and DVC-versioned datasets. No features, no models, no serving — those are later phases.

</domain>

<decisions>
## Implementation Decisions

### Data Storage & Versioning
- Parquet files under `data/raw/{asset_class}/{symbol}.parquet` and `data/processed/`; one file per asset
- DVC tracks `data/` directory; `.dvc` pointer files committed to git; local DVC cache (no remote required for v1)
- Cache-first incremental: resume from last stored timestamp per asset, merge-dedupe on date index, exclude incomplete last candle

### Ingestion Interface
- Console script `volforecast ingest` (also runnable as `python -m volforecast.ingest`); asset universe defined in `config/assets.yaml`
- Crypto via ccxt: Binance public API default, exchange configurable (Kraken fallback) for geo-block resilience; incomplete-last-candle exclusion explicit
- Equities: SPY + AAPL + MSFT via yfinance; pinned version; explicit `auto_adjust=True` (documented choice); retry/backoff on rate limits
- No live API calls in CI — ingestion tested against fixtures

### Validation Policy
- Pandera schemas in `src/volforecast/validate/`; hard-fail gate with quarantine report (offending rows + reason written to `data/quarantine/`)
- Equity sessions validated against `exchange_calendars` (XNYS); crypto validated against continuous 24/7 daily calendar; no fabricated weekend equity rows
- Checks: schema/dtype, OHLC consistency (high>=low, etc.), gaps vs calendar, stale rows (repeated closes), non-negative volume

### Dev Tooling
- `pyproject.toml` packaging (src layout, `src/volforecast/`), Python 3.12, pinned dependency matrix (pandas<3, numpy 2.x, mlflow 3.x, arch, lightgbm, evidently 0.7+, pandera 0.31+, prefect 3.x)
- ruff (lint) + pytest; fixtures = small committed parquet/CSV snapshots in `tests/fixtures/`
- GitHub Actions: lint + unit tests on every push; Windows-safe `.gitattributes` (`*.sh text eol=lf`)
- docker-compose: Postgres + MLflow tracking server + Prefect server + Prefect worker; named volumes (not bind mounts) for DB data

### Claude's Discretion
- Exact retry/backoff parameters, parquet partitioning details, compose service naming, fixture sizes, ruff rule selection — implementer's choice guided by research notes in .planning/research/.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- None — greenfield repo (only .planning/ and CLAUDE.md exist)

### Established Patterns
- Research findings in `.planning/research/STACK.md` (pinned versions, Windows/Docker notes: Postgres not SQLite on bind mounts, libgomp1 in slim images) and `PITFALLS.md` (yfinance rate limits, ccxt incomplete candles, calendar fabrication)

### Integration Points
- Phase 2 consumes validated processed datasets and the package layout; the canonical units/target module lands in Phase 2 but the package skeleton must accommodate `src/volforecast/{ingest,validate,features,models,serving,monitoring}`

</code_context>

<specifics>
## Specific Ideas

- Repo layout from PROJECT.md: `src/volforecast/`, `pipelines/`, `tests/`, `infra/` (docker-compose), `.github/workflows/`, `notebooks/`
- CI must be green using fixture data only — this is itself a portfolio talking point
- 2+ years of history so Phase 2 walk-forward has enough folds

</specifics>

<deferred>
## Deferred Ideas

- Feast feature store, cloud deploy, intraday data — explicitly v2/out of scope
- Binance geo-block contingency for cloud CI ingestion — revisit at cloud-deploy time (CI uses fixtures, so not blocking)

</deferred>
