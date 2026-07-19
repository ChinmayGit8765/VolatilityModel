---
phase: 01-foundation-validated-data
plan: "01"
subsystem: ingest-validate-store
tags: [ingest, pandera, dvc, ccxt, parquet, walking-skeleton, tdd]
dependency_graph:
  requires: []
  provides: [volforecast-package, ingest-base, crypto-adapter, pandera-schema, dvc-tracking]
  affects: [01-02-equity-ingest, 01-03-calendar-validation, 01-04-ci-infra]
tech_stack:
  added:
    - "pandera[pandas] 0.31+ — import pandera.pandas as pa with UTC datetime index"
    - "ccxt 4.5.x — since-pagination loop with VOLFORECAST_NO_LIVE_API guard"
    - "dvc 3.67+ — per-subdirectory tracking (data/raw.dvc)"
    - "uv 0.11+ — pyproject.toml src layout, uv sync, uv.lock"
    - "hatchling — build backend for src/volforecast/ wheel"
  patterns:
    - "src-layout with hatchling build backend and [project.scripts] console entry"
    - "drop_incomplete_candles: filter candles where open_time + timeframe_ms > now_ms"
    - "incremental_update: concat + dedupe on index keep=last + sort_index + to_parquet"
    - "Pandera lazy=True + quarantine CSV on SchemaErrors"
    - "VOLFORECAST_NO_LIVE_API=1 guard in tests via conftest.py os.environ.setdefault"
key_files:
  created:
    - src/volforecast/__init__.py
    - src/volforecast/config.py
    - src/volforecast/cli.py
    - src/volforecast/ingest/__init__.py
    - src/volforecast/ingest/__main__.py
    - src/volforecast/ingest/base.py
    - src/volforecast/ingest/crypto.py
    - src/volforecast/validate/__init__.py
    - src/volforecast/validate/schemas.py
    - src/volforecast/features/__init__.py
    - src/volforecast/models/__init__.py
    - src/volforecast/serving/__init__.py
    - src/volforecast/monitoring/__init__.py
    - pipelines/__init__.py
    - config/assets.yaml
    - tests/conftest.py
    - tests/fixtures/crypto_sample.parquet
    - tests/unit/test_skeleton_e2e.py
    - pyproject.toml
    - uv.lock
    - .gitignore
    - .gitattributes
    - .dvcignore
    - data/.gitignore
    - data/raw.dvc
  modified: []
decisions:
  - "Use pd.DatetimeTZDtype(tz='UTC') for Pandera Index dtype — pa.dtypes.DateTime(tz=...) constructor does not accept tz kwarg in 0.31.x; pd.DatetimeTZDtype works and produces datetime64[ns, UTC]"
  - "VOLFORECAST_NO_LIVE_API guard: raise RuntimeError in fetch_crypto_ohlcv when env var set — prevents any live call from sneaking into tests even if conftest.py is skipped"
  - "Binance accessible from dev machine (no geo-block): 1621 rows fetched 2022-01-01 to 2026-06-09"
  - "DVC per-subdirectory tracking: dvc add data/raw (not data/), per RESEARCH Pitfall 7"
metrics:
  duration_minutes: 12
  completed_date: "2026-06-10"
  tasks_completed: 3
  tasks_total: 3
  files_created: 25
  files_modified: 0
---

# Phase 1 Plan 1: Walking Skeleton — Foundation Validated Data Summary

**One-liner:** Installable `volforecast` package with ccxt BTC ingestion, Pandera OHLCV gate, incremental parquet merge-dedupe, and DVC-tracked raw data — TDD proven on fixture.

## What Was Built

The VolForecast Walking Skeleton: the thinnest possible end-to-end data pipeline slice from exchange API to DVC-versioned validated parquet. A single command (`volforecast ingest --symbol BTC/USDT --start 2022-01-01`) now fetches 2+ years of daily BTC OHLCV via ccxt/Binance, drops the still-forming last candle, passes a Pandera schema gate, merges into a parquet with cache-first deduplication, and DVC tracks the result with a committed pointer file.

### Capability Proven End-to-End

- `uv sync --dev` installs the `volforecast` package on Python 3.12 with all 249 pinned transitive deps; `uv.lock` committed for reproducibility.
- `volforecast ingest --symbol BTC/USDT --start 2022-01-01` writes `data/raw/crypto/BTC-USD.parquet` with 1621 daily rows (2022-01-01 to 2026-06-09); no forming candle in stored data.
- Pandera `crypto_ohlcv_schema` gate runs before any persist; OHLC-consistency checks reject bad ticks; `import pandera.pandas as pa` (correct 0.31+ namespace).
- `data/raw.dvc` committed; `dvc status data/raw.dvc` clean; local cache remote configured.
- All 3 end-to-end fixture-driven tests pass; `ruff check src tests` exits 0.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Failing e2e skeleton tests + fixture (RED) | b2a7f27 | tests/conftest.py, tests/fixtures/crypto_sample.parquet, tests/unit/test_skeleton_e2e.py, pyproject.toml, uv.lock |
| 2 | Scaffold package + ingest/validate (GREEN) | 82be4c6 | src/volforecast/ingest/base.py, src/volforecast/ingest/crypto.py, src/volforecast/validate/schemas.py, src/volforecast/cli.py |
| 3 | DVC init + raw data tracking (seal) | 667a947 | data/raw.dvc, .dvc/config, .dvcignore, data/.gitignore |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Pandera Index dtype for tz-aware DatetimeIndex**
- **Found during:** Task 2 (GREEN phase, first test run)
- **Issue:** `pa.Index(pa.dtypes.DateTime, name="date")` creates a dtype expecting `datetime64[ns]` (naive), but the canonical OHLCV contract uses tz-aware UTC (`datetime64[ns, UTC]`). The first test failed with `SchemaErrors: expected series 'date' to have type datetime64[ns], got datetime64[ns, UTC]`.
- **Fix:** Replaced with `pa.Index(pd.DatetimeTZDtype(tz="UTC"), name="date")` and added `_UTC_DATETIME_DTYPE = pd.DatetimeTZDtype(tz="UTC")` constant. The `pa.dtypes.DateTime(tz=...)` constructor does not accept a `tz` keyword argument in pandera 0.31.x.
- **Files modified:** `src/volforecast/validate/schemas.py`
- **Commit:** 82be4c6

**2. [Rule 1 - Bug] Ruff I001 import sort order in test file**
- **Found during:** Task 2 (ruff check after initial implementation)
- **Issue:** `import os` followed by `from pathlib import Path` before third-party imports triggered ruff I001 (import block un-sorted); also `os` was unused.
- **Fix:** Removed unused `import os`, ran `ruff check --fix` to auto-sort import blocks.
- **Files modified:** `tests/unit/test_skeleton_e2e.py`
- **Commit:** 82be4c6

## Known Stubs

None — all modules either have real implementation or are empty skeleton `__init__.py` files whose emptiness is intentional (Phase 2+ fills features, models, serving, monitoring, pipelines). No stub produces misleading UI output or blocks the plan's goal.

## TDD Gate Compliance

RED gate (test commit): b2a7f27 `test(01-01): add failing e2e skeleton tests + fixture (RED)`
GREEN gate (feat commit): 82be4c6 `feat(01-01): scaffold package, implement ingest+validate, all tests GREEN`

Both gates present in correct order. RED tests failed with `ModuleNotFoundError: No module named 'volforecast'`. GREEN implementation passes all 3 tests.

## Threat Flags

No new threat surface found beyond what is documented in the plan's `<threat_model>`. All T-01-01 (Pandera OHLC gate), T-01-02 (incomplete candle drop), T-01-03 (no API keys), T-01-04 (parquet read from own store), and T-01-SC (uv.lock hash-locks) mitigations are implemented.

## Self-Check: PASSED

Files exist:
- `src/volforecast/ingest/base.py` FOUND
- `src/volforecast/validate/schemas.py` FOUND
- `tests/fixtures/crypto_sample.parquet` FOUND
- `data/raw.dvc` FOUND
- `uv.lock` FOUND

Commits exist:
- b2a7f27 FOUND (test RED)
- 82be4c6 FOUND (feat GREEN)
- 667a947 FOUND (feat DVC seal)
