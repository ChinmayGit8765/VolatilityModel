---
phase: 01-foundation-validated-data
verified: 2026-06-11T00:00:00Z
status: gaps_found
score: 4/5 must-haves verified
overrides_applied: 0
gaps:
  - truth: "Every push triggers GitHub Actions CI that passes lint and unit tests using fixture data only — no live API calls in CI"
    status: failed
    reason: >
      The CI workflow includes `uv run ruff format --check .` as a required step. Running this
      locally shows 10 source files would be reformatted (cli.py, equity.py, validate/__init__.py,
      schemas.py, conftest.py, test_checks.py, test_ingest.py, test_pipeline.py, test_schemas.py,
      ingest/__main__.py). The format check step would fail in CI as written. Additionally, the
      remote tracking branch is gone (git reports "upstream is gone") so GitHub Actions CI has
      not yet run to confirm a green build. The pytest and ruff check (lint) steps do pass locally.
    artifacts:
      - path: "src/volforecast/cli.py"
        issue: "ruff format --check would reformat (string continuation alignment)"
      - path: "src/volforecast/ingest/equity.py"
        issue: "ruff format --check would reformat"
      - path: "src/volforecast/validate/__init__.py"
        issue: "ruff format --check would reformat"
      - path: "src/volforecast/validate/schemas.py"
        issue: "ruff format --check would reformat"
      - path: "tests/conftest.py"
        issue: "ruff format --check would reformat"
      - path: "tests/unit/test_checks.py"
        issue: "ruff format --check would reformat"
      - path: "tests/unit/test_ingest.py"
        issue: "ruff format --check would reformat"
      - path: "tests/unit/test_pipeline.py"
        issue: "ruff format --check would reformat"
      - path: "tests/unit/test_schemas.py"
        issue: "ruff format --check would reformat"
    missing:
      - "Run `uv run ruff format src tests` to apply formatting then re-commit"
      - "Push to origin/main so GitHub Actions CI runs and confirms green build"
human_verification:
  - test: "docker compose -f infra/docker-compose.yml up -d (on a clean machine, not the dev machine where stack is already running)"
    expected: "All four services (postgres, mlflow-server, prefect-server, prefect-worker) start healthy; MLflow at :5000, Prefect at :4200 respond within ~60s"
    why_human: "Stack is already up on the dev machine — verified live during Plan 04 execution. A fresh bring-up on a different machine or after `docker compose down --volumes` would confirm the compose file is truly reproducible, not just idempotent on an already-initialised volume."
---

# Phase 1: Foundation & Validated Data — Verification Report

**Phase Goal:** A reproducible local stack on Windows 11 ingests, validates, and versions 2+ years of cross-asset daily OHLCV — the trusted data layer everything downstream consumes
**Verified:** 2026-06-11T00:00:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (Roadmap Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `docker compose up` brings up Postgres-backed MLflow + Prefect server + worker; `volforecast` installs on Python 3.12 with pinned dependency matrix | VERIFIED | `docker compose -f infra/docker-compose.yml ps` shows all 4 services Up/healthy. MLflow `:5000/health` returns `OK`; Prefect `:4200/api/health` returns `true`. `volforecast` installs via `uv sync --locked --dev` (770 packages in `uv.lock`). CLI `volforecast ingest --help` lists `--symbol`, `--start`, `--exchange`. |
| 2 | Every push triggers GitHub Actions CI that passes lint and unit tests using fixture data only — no live API calls in CI | FAILED | CI workflow exists at `.github/workflows/ci.yml` and correctly includes `astral-sh/setup-uv@v8`, `uv sync --locked --dev`, `ruff check .`, `ruff format --check .`, and `pytest` with `VOLFORECAST_NO_LIVE_API: "1"`. However, `uv run ruff format --check .` fails locally with 10 files needing reformatting. The remote tracking branch is gone (git: "upstream is gone") so no green CI run can be confirmed. |
| 3 | A single command fetches 2+ years of daily OHLCV for BTC, ETH, SPY, and 2 large caps; re-running is cache-first incremental; incomplete last candles excluded; `auto_adjust`/rate limits handled explicitly | VERIFIED | `data/raw/crypto/BTC-USD.parquet`: 1621 rows 2022-01-01→2026-06-09, no forming candle. `data/raw/crypto/ETH-USD.parquet`: 1621 rows. `data/raw/equity/SPY.parquet`: 1112 rows (equals XNYS sessions count for that range). AAPL + MSFT: 1112 rows each. `resume_since_ms` in `crypto.py` implements cache-first incremental. `equity.py` has `auto_adjust=True`, `threads=False`, tenacity `@retry` decorator verified by inspect. |
| 4 | Data with gaps, bad ticks, stale rows, schema violations, or fabricated weekend equity bars is rejected by Pandera gates before reaching features | VERIFIED | `validate_asset` dispatcher in `validate/__init__.py` runs: crypto_gap_check (24/7 date_range) or equity_session_check (XNYS sessions_in_range), stale_row_check, ohlc_consistency_check, and Pandera schema validation with lazy=True. Fails closed on any failure; writes quarantine CSV then raises. All 29 tests pass confirming rejection of weekend-bad fixture and OHLC-violation fixture. `cli.py` calls `validate_asset` for every asset (both crypto and equity) before writing to `data/processed/`. |
| 5 | Raw and processed datasets can be reproduced at any commit via DVC checkout | VERIFIED | `data/raw.dvc` and `data/processed.dvc` both exist (5 parquet files each, hashed). `uv run dvc status data/raw.dvc` and `data/processed.dvc` both report clean. `dvc checkout` round-trips confirmed (empty output = already up to date). Local cache remote configured at `../../dvc-cache`. |

**Score:** 4/5 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pyproject.toml` | Package definition, pinned deps, console script | VERIFIED | hatchling backend, name="volforecast", requires-python=">=3.12", all deps present (pandas, ccxt, yfinance, pandera, dvc, mlflow, prefect, tenacity, exchange-calendars), `volforecast = "volforecast.cli:main"` |
| `src/volforecast/ingest/base.py` | OHLCV column contract + incremental merge-dedupe + incomplete-candle drop | VERIFIED | 109 lines; `drop_incomplete_candles`, `candles_to_df`, `incremental_update` all present and substantive |
| `src/volforecast/ingest/crypto.py` | ccxt BTC daily OHLCV adapter with since-pagination | VERIFIED | 121 lines; since-pagination loop, `drop_incomplete_candles` wired, `VOLFORECAST_NO_LIVE_API` guard |
| `src/volforecast/ingest/equity.py` | yfinance equity adapter with auto_adjust + tenacity retry | VERIFIED | 163 lines; `_download_with_retry` with tenacity `@retry`, `auto_adjust=True`, `threads=False`, `normalize_equity_frame` |
| `src/volforecast/validate/schemas.py` | Pandera OHLCV schemas + `validate_and_quarantine` | VERIFIED | `import pandera.pandas as pa` (correct 0.31+ namespace); `crypto_ohlcv_schema`, `equity_ohlcv_schema`, `validate_and_quarantine` |
| `src/volforecast/validate/checks.py` | Calendar-aware gap/session checks, stale, OHLC | VERIFIED | 263 lines; `crypto_gap_check`, `equity_session_check` (XNYS + `sessions_in_range`), `stale_row_check`, `ohlc_consistency_check` |
| `src/volforecast/validate/__init__.py` | `validate_asset` single dispatch entry point | VERIFIED | `validate_asset(df, asset_class, quarantine_dir)` exported; routes crypto/equity to correct calendar check; fails closed |
| `config/assets.yaml` | Full 5-asset universe: BTC/USDT, ETH/USDT, SPY, AAPL, MSFT | VERIFIED | All 5 symbols present with asset_class and exchange fields |
| `tests/fixtures/crypto_sample.parquet` | Committed fixture for offline tests | VERIFIED | Exists; used by test_skeleton_e2e.py |
| `tests/fixtures/equity_sample.parquet` | Committed equity fixture | VERIFIED | Exists; used by test_ingest.py |
| `tests/fixtures/equity_bad_weekend.parquet` | Broken equity fixture with Saturday row | VERIFIED | Exists; drives test_validate_asset_equity_rejects_weekend |
| `tests/fixtures/crypto_gap.parquet` | Broken crypto fixture with missing day | VERIFIED | Exists; drives test_crypto_gap_rejected |
| `tests/unit/test_skeleton_e2e.py` | E2E fixture-driven test: ingest -> validate -> parquet | VERIFIED | 3 tests pass |
| `tests/unit/test_ingest.py` | Incremental resume + equity adapter tests | VERIFIED | 6 tests pass |
| `tests/unit/test_checks.py` | Calendar gap / weekend / stale tests | VERIFIED | 9 tests pass |
| `tests/unit/test_schemas.py` | validate_asset + schema tests | VERIFIED | 9 tests pass |
| `tests/unit/test_pipeline.py` | Pipeline promotion + quarantine gate tests | VERIFIED | 2 tests pass |
| `infra/docker-compose.yml` | Postgres + MLflow + Prefect stack | VERIFIED | 4 services defined; `postgres_data` + `mlflow_artifacts` named volumes; MLflow uses `postgresql://` (no asyncpg); Prefect uses `postgresql+asyncpg://` against separate `prefectdb` |
| `.github/workflows/ci.yml` | GitHub Actions fixture-only CI | PARTIAL | Workflow correctly includes all required steps and `VOLFORECAST_NO_LIVE_API: "1"` — but `ruff format --check .` would fail on 10 source files |
| `.gitattributes` | LF enforcement for shell/container files | VERIFIED | `*.sh text eol=lf`, `Dockerfile* text eol=lf`, `*.yml text eol=lf`, `*.yaml text eol=lf`, `*.parquet binary` |
| `data/raw.dvc` | DVC pointer for raw data (5 parquet files) | VERIFIED | Exists; md5 hash tracked; dvc status clean |
| `data/processed.dvc` | DVC pointer for processed data (5 parquet files) | VERIFIED | Exists; md5 hash tracked; dvc status clean |
| `README.md` | Full-stack run documentation | VERIFIED | `## Running the stack` section with `docker compose`, `uv sync --dev`, `volforecast ingest`, `dvc checkout` |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|-------|-----|--------|---------|
| `src/volforecast/cli.py` | `src/volforecast/ingest/crypto.py` | ingest command dispatch | WIRED | `from volforecast.ingest.crypto import fetch_crypto_ohlcv` inside `_ingest_single_asset` |
| `src/volforecast/cli.py` | `src/volforecast/ingest/equity.py` | ingest dispatch by asset_class | WIRED | `from volforecast.ingest.equity import download_equity_ohlcv` inside `_ingest_single_asset` |
| `src/volforecast/cli.py` | `src/volforecast/validate/__init__.py` | validate_asset gate before processed write | WIRED | `from volforecast.validate import ValidationError, validate_asset`; called unconditionally for both asset classes before writing processed parquet |
| `src/volforecast/ingest/crypto.py` | `src/volforecast/validate/schemas.py` | validation gate before write (Plan 01) | WIRED | `crypto_ohlcv_schema` used via `validate_and_quarantine` in cli.py; schema imported in `validate/__init__.py` |
| `pyproject.toml` | `src/volforecast/cli.py` | console script entry point | WIRED | `volforecast = "volforecast.cli:main"` in `[project.scripts]`; verified via `uv run volforecast --help` |
| `src/volforecast/validate/checks.py` | `exchange_calendars` | XNYS session calendar | WIRED | `import exchange_calendars as xcals`; `xcals.get_calendar("XNYS").sessions_in_range(...)` present and tested |
| `src/volforecast/validate/__init__.py` | `src/volforecast/validate/schemas.py` | schema selection by asset_class | WIRED | `from volforecast.validate.schemas import crypto_ohlcv_schema, equity_ohlcv_schema` |
| `infra/docker-compose.yml` | `postgres` | MLflow + Prefect backend store | WIRED | Both services reference `postgres:5432` in their connection URIs; healthcheck dependency configured |
| `.github/workflows/ci.yml` | `uv.lock` | reproducible install | WIRED | `uv sync --locked --dev` step present |

---

## Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `data/raw/crypto/BTC-USD.parquet` | 1621 rows OHLCV | ccxt Binance `fetch_ohlcv` via `fetch_crypto_ohlcv` | Yes — live data ingested 2022-01-01 to 2026-06-09 | FLOWING |
| `data/raw/equity/SPY.parquet` | 1112 rows OHLCV | yfinance `download` via `download_equity_ohlcv` | Yes — adjusted OHLCV from 2022-01-03 (first XNYS session) | FLOWING |
| `data/processed/crypto/BTC-USD.parquet` | validated OHLCV | `validate_asset` gate → `incremental_update` to processed_path | Yes — only validated rows promoted | FLOWING |
| `data/processed/equity/SPY.parquet` | validated OHLCV | `validate_asset` gate → `incremental_update` to processed_path | Yes — only XNYS-session-validated rows promoted | FLOWING |

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full test suite passes on fixtures | `uv run pytest tests/ -q` | 29 passed, 1 warning (FutureWarning in pandera, non-blocking) | PASS |
| `ruff check` (lint) passes | `uv run ruff check src tests` | All checks passed | PASS |
| `ruff format --check` passes (CI gate) | `uv run ruff format --check .` | 10 files would be reformatted | FAIL |
| volforecast CLI installs and responds | `uv run volforecast ingest --help` | Shows `--symbol`, `--start`, `--exchange` | PASS |
| DVC raw data tracked and clean | `uv run dvc status data/raw.dvc` | "Data and pipelines are up to date" | PASS |
| DVC processed data tracked and clean | `uv run dvc status data/processed.dvc` | "Data and pipelines are up to date" | PASS |
| Docker compose stack up and healthy | `docker compose -f infra/docker-compose.yml ps` | All 4 services Up/healthy; MLflow `:5000` and Prefect `:4200` responding | PASS |
| BTC data has 2+ years, no forming candle | Python validation check | 1621 rows, first=2022-01-01, last=2026-06-09 < today | PASS |
| Equity data has 2+ years trading sessions | 1112 rows = XNYS sessions 2022-01-01→2026-06-09 | Verified against exchange_calendars count | PASS |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| FOUND-01 | 01-01, 01-04 | Python 3.12 + pinned dependency matrix, installable package | SATISFIED | pyproject.toml with all pinned deps; uv.lock 770 packages; package imports cleanly |
| FOUND-02 | 01-04 | docker-compose: Postgres-backed MLflow + Prefect on Windows 11 | SATISFIED | 4 services running healthy; both UIs confirmed live during Plan 04 checkpoint |
| FOUND-03 | 01-04 | CI: lint + unit tests on push, fixture data only | BLOCKED | `ruff format --check` step fails on 10 files; CI not yet pushed/confirmed green |
| INGEST-01 | 01-01, 01-02 | BTC + ETH via ccxt, cache-first incremental, 2+ years | SATISFIED | BTC 1621 rows, ETH 1621 rows; `resume_since_ms` implements incremental; incomplete candle drop tested |
| INGEST-02 | 01-02 | SPY + 2 large caps via yfinance, explicit auto_adjust, rate-limit tolerance | SATISFIED | SPY/AAPL/MSFT 1112 rows each; `auto_adjust=True`, `threads=False`, tenacity retry confirmed by inspect |
| INGEST-03 | 01-03 | Per-asset-class calendars: crypto 24/7, equities XNYS sessions | SATISFIED | `crypto_gap_check` uses `pd.date_range(freq="D")`; `equity_session_check` uses XNYS `sessions_in_range`; tested against broken fixtures |
| INGEST-04 | 01-01, 01-03 | Pandera gates reject gaps, bad ticks, stale data, schema violations | SATISFIED | `validate_asset` runs Pandera + calendar + stale + OHLC checks; fails closed; quarantine written; 29 tests all pass |
| INGEST-05 | 01-01, 01-04 | Raw + processed datasets versioned with DVC | SATISFIED | `data/raw.dvc` (5 files) + `data/processed.dvc` (5 files); both clean; dvc checkout round-trips |

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/volforecast/cli.py` | 52-54 | `ruff format` would reformat string continuation | Warning | CI `ruff format --check` step fails |
| `src/volforecast/ingest/equity.py` | — | `ruff format` would reformat | Warning | CI `ruff format --check` step fails |
| `src/volforecast/validate/__init__.py` | — | `ruff format` would reformat | Warning | CI `ruff format --check` step fails |
| `src/volforecast/validate/schemas.py` | — | `ruff format` would reformat | Warning | CI `ruff format --check` step fails |
| `tests/conftest.py` + 5 test files | — | `ruff format` would reformat | Warning | CI `ruff format --check` step fails |

No `TBD`, `FIXME`, or `XXX` debt markers found in any source file. No placeholder returns or stub implementations found. No hardcoded empty arrays/objects returned from production code paths.

---

## Human Verification Required

### 1. Fresh docker-compose bring-up

**Test:** On a clean machine (or after `docker compose -f infra/docker-compose.yml down --volumes`), run `docker compose -f infra/docker-compose.yml up -d` and wait ~60s.
**Expected:** All four services (postgres, mlflow-server, prefect-server, prefect-worker) start healthy; `curl http://localhost:5000/health` returns `OK`; `curl http://localhost:4200/api/health` returns `true`.
**Why human:** Stack is already initialised and running on the dev machine (verified during Plan 04 execution). A reproducibility claim requires a cold start; automated verification cannot tear down an already-live stack.

---

## Gaps Summary

**1 gap blocking goal achievement:**

**Truth 2 (SC-2) — GitHub Actions CI passes lint and unit tests:**

The CI workflow is correctly defined with all required steps including `VOLFORECAST_NO_LIVE_API: "1"` for test isolation. However, `ruff format --check .` is a required CI step that would currently fail because 10 Python source files have formatting inconsistencies (mainly string continuation alignment in `cli.py` and multi-line print calls; docstring whitespace in others).

This is a one-command fix: `uv run ruff format src tests` followed by `git add` and `git commit`. The fix is trivial but the gap is real — the CI pipeline as defined would not pass in its current state. Additionally, the remote tracking branch is gone (`git push` has not been run against the GitHub remote), so no CI run result exists to observe.

**Root cause:** `ruff format` was checked only against `src tests` subdirectories during plan execution (which passes), but the CI workflow runs `uv run ruff format --check .` (the entire repo root, including `tests/conftest.py` which is at the root of `tests/`). The behaviour is equivalent since `tests/conftest.py` is under `tests/`, but the formatting differences are present regardless of path scoping.

**Fix:**
1. Run `uv run ruff format src tests` (or `uv run ruff format .`)
2. `git add src/ tests/`
3. Commit and push to origin/main
4. Confirm GitHub Actions shows green CI run

---

_Verified: 2026-06-11T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
