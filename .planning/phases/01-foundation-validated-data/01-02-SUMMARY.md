---
phase: 01-foundation-validated-data
plan: "02"
subsystem: ingest
tags: [ingest, equity, yfinance, tenacity, retry, ccxt, incremental, multi-asset, cli]
dependency_graph:
  requires: ["01-01"]
  provides: ["INGEST-01", "INGEST-02"]
  affects: ["01-03", "01-04"]
tech_stack:
  added:
    - "yfinance[nospam] 1.4.1 (equity OHLCV adapter, already in pyproject)"
    - "tenacity 9.1.4 (exponential backoff retry, already in pyproject)"
    - "requests-ratelimiter 0.10.0 (transitive via yfinance[nospam], verified legitimate)"
  patterns:
    - "TDD RED/GREEN: tests/fixtures + test file committed first, then implementation"
    - "cache-first incremental: resume_since_ms reads last stored date, avoids full re-download"
    - "env-guard pattern: VOLFORECAST_NO_LIVE_API=1 blocks real API; tests clear it with patch.dict"
    - "retry-decorated inner function separated from public API for clean guard/retry layering"
key_files:
  created:
    - src/volforecast/ingest/equity.py
    - tests/fixtures/equity_sample.parquet
    - tests/unit/test_ingest.py
  modified:
    - src/volforecast/ingest/crypto.py
    - src/volforecast/ingest/__init__.py
    - src/volforecast/cli.py
    - config/assets.yaml
decisions:
  - "resume_since_ms placed in crypto.py (cache-first helper used by crypto ingest)"
  - "VOLFORECAST_NO_LIVE_API guard in download_equity_ohlcv (not inside _download_with_retry) so guard fires once per call, not per retry attempt"
  - "Tests clear env var with patch.dict + pop pattern — env guard is CI safety net, monkeypatch provides actual test isolation"
  - "Equity schema gate (Pandera) deferred to Plan 03 per plan spec"
metrics:
  duration: "11 minutes"
  completed_date: "2026-06-10"
  tasks_completed: 3
  files_created: 3
  files_modified: 4
---

# Phase 01 Plan 02: Multi-Asset Ingest (Equity + Full Universe) Summary

**One-liner:** yfinance equity adapter with explicit auto_adjust + tenacity retry, cache-first incremental resume, full 5-asset universe (BTC/ETH/SPY/AAPL/MSFT), and multi-asset CLI dispatch.

## What Was Built

### Task 1: Equity yfinance adapter + offline fixture (TDD RED/GREEN)

**`src/volforecast/ingest/equity.py`** (new):
- `_download_with_retry`: inner function decorated with `@retry(stop_after_attempt(5), wait_exponential(multiplier=1, min=2, max=60), reraise=True)` wrapping `yf.download`
- `download_equity_ohlcv(tickers, start, end)`: public entry point; checks `VOLFORECAST_NO_LIVE_API` guard before calling the retry-wrapped downloader; explicitly passes `auto_adjust=True` (documented choice: records adjusted OHLC) and `threads=False` (prevents shared-global-dict race in yfinance 1.x)
- `normalize_equity_frame(raw, ticker)`: reshapes yf.download MultiIndex/flat output to canonical OHLCV contract (lowercase columns, tz-aware UTC DatetimeIndex named "date", float64)

**`tests/fixtures/equity_sample.parquet`** (new):
- 20-row SPY-shaped fixture (bdate_range 2022-01-03, 20 business days), all float64, UTC index — used for offline equity tests

### Task 2: Cache-first incremental + full universe + multi-asset CLI (TDD RED/GREEN)

**`src/volforecast/ingest/crypto.py`** (updated):
- `resume_since_ms(existing_path, default_start_ms)`: reads last stored "date" from existing parquet, returns next-day ms; falls back to `default_start_ms` when file absent — enables cache-first incremental fetches
- Fixed duplicate docstring bug inherited from Plan 01

**`config/assets.yaml`** (updated):
- Full 5-asset universe: BTC/USDT + ETH/USDT (crypto/binance, `kraken_fallback` field documented) and SPY + AAPL + MSFT (equity)

**`src/volforecast/cli.py`** (updated):
- `--symbol` is now optional (was required); when omitted, loads all assets from `config/assets.yaml`
- Per-asset dispatch: crypto → `fetch_crypto_ohlcv` + `resume_since_ms` + crypto_ohlcv_schema gate; equity → `download_equity_ohlcv` (equity Pandera gate deferred to Plan 03)
- Single `--symbol` path preserved and working

### Task 3: Package legitimacy checkpoint (auto-approved with objective evidence)

See Checkpoint Notes section below.

## Test Coverage

All 6 tests in `tests/unit/test_ingest.py` pass offline (no network):

| Test | What It Proves |
|------|---------------|
| `test_equity_adapter_normalizes_to_ohlcv_contract` | yf.download multi-ticker raw → per-ticker OHLCV contract (columns, dtype, UTC index) |
| `test_equity_adapter_retries_then_succeeds` | Tenacity retry wired: downloader that raises once then succeeds returns valid result |
| `test_incremental_resume_uses_last_stored_timestamp` | resume_since_ms returns next-day ms from last parquet date (not default) |
| `test_incremental_resume_falls_back_to_default_when_missing` | Falls back to default_start_ms when file absent |
| `test_ingest_all_dispatches_by_asset_class` | Full dispatch: crypto → ccxt adapter, equity → yfinance adapter, one parquet per asset |
| `test_overlapping_rerun_no_duplicates` | Overlapping re-run: no duplicate index entries, keep=last semantics |

Full test suite: 9/9 passed (`tests/unit/test_skeleton_e2e.py` + new tests).

## Verification Evidence

```
uv run pytest tests/ -q              →  9 passed
universe assertion                   →  universe-ok {'MSFT', 'SPY', 'ETH/USDT', 'AAPL', 'BTC/USDT'}
uv run volforecast ingest --help     →  shows optional --symbol, --start defaults, --exchange
inspect source (auto_adjust, retry)  →  equity-adapter-ok
uv run ruff check src tests          →  All checks passed!
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed non-existent `reraise` from tenacity imports**
- **Found during:** Task 1 (import error on first GREEN run)
- **Issue:** `from tenacity import retry, reraise, ...` — `reraise` is a parameter to `@retry(reraise=True)`, not a separate importable name in tenacity
- **Fix:** Removed `reraise` from the import; the decorator uses `reraise=True` kwarg
- **Files modified:** `src/volforecast/ingest/equity.py`
- **Commit:** c9dcef6

**2. [Rule 1 - Bug] Separated retry-decorated inner function from env-var guard**
- **Found during:** Task 1 (VOLFORECAST_NO_LIVE_API guard blocking monkeypatched tests)
- **Issue:** Guard inside `download_equity_ohlcv` fired before the monkeypatched `yf.download` could be exercised — `VOLFORECAST_NO_LIVE_API=1` (set by conftest.py) blocked both real and mocked calls
- **Fix:** Extracted `_download_with_retry` as a private inner function holding the `@retry` decorator and the `yf.download` call; `download_equity_ohlcv` (public API) checks the env guard then delegates to `_download_with_retry`. Tests use `patch.dict + os.environ.pop` to temporarily clear the guard while patching `yf.download`
- **Files modified:** `src/volforecast/ingest/equity.py`, `tests/unit/test_ingest.py`
- **Commit:** c9dcef6

**3. [Rule 1 - Bug] Fixed duplicate docstring in crypto.py**
- **Found during:** Task 2 (reading crypto.py for incremental resume implementation)
- **Issue:** `fetch_crypto_ohlcv` had two docstrings (one single-line summary line 28, one full docstring lines 29-49) — a copy-paste artifact from Plan 01
- **Fix:** Removed the redundant single-line docstring at line 28; kept the full docstring
- **Files modified:** `src/volforecast/ingest/crypto.py`
- **Commit:** 7bec979

**4. [Rule 2 - Ruff lint] Fixed unused imports and line-length violations in tests**
- **Found during:** Task 1 verify step (`uv run ruff check src tests`)
- **Issue:** `MagicMock`, `pytest` imported but unused; two lines exceeded 100-char limit; import sorting
- **Fix:** `uv run ruff check --fix` applied 3 auto-fixes; manually split 2 long lines
- **Files modified:** `tests/unit/test_ingest.py`
- **Commit:** c9dcef6

## Checkpoint Notes

### Task 3: Package legitimacy (auto-approved)

The session runs in autonomous mode with pre-authorized recommendation. Objective evidence gathered:

| Package | Installed Version | Author | Source Repo | Import Status |
|---------|------------------|--------|-------------|---------------|
| tenacity | 9.1.4 | Julien Danjou | github.com/jd/tenacity | OK (no `__version__`, checked via importlib.metadata) |
| requests-ratelimiter | 0.10.0 | Jordan Cook | github.com/JWCook/requests-ratelimiter | OK |
| requests_ratelimiter (importable) | 4.2.0 (pyrate-limiter 4.x bundled) | — | — | `import requests_ratelimiter; requests_ratelimiter.__version__ == '4.2.0'` |

**Discrepancy note:** `requests-ratelimiter` dist is 0.10.0 but `requests_ratelimiter.__version__` reports 4.2.0 — this is because `requests-ratelimiter` 0.10.0 depends on `pyrate-limiter>=4` and re-exports its version attribute. Both packages are legitimate and functioning.

Both packages match the RESEARCH.md Package Legitimacy Audit entries exactly. uv.lock pins hashes for reproducible installs. **Auto-approved.**

## Known Stubs

**`_ingest_single_equity` in `cli.py` — equity Pandera validation gate absent:**
- Equity assets are ingested and written via `incremental_update` but NOT validated through a Pandera schema gate (unlike crypto which uses `crypto_ohlcv_schema`)
- This is intentional per the plan spec: "applies the validation gate (crypto schema for now — Plan 03 adds the equity schema and wires it)"
- **Plan 03** will add the equity schema and wire the gate

## Threat Surface

No new network endpoints or auth paths introduced. Changes are within the established ingest layer trust boundary (Yahoo Finance API → equity adapter, documented in plan's threat model T-02-01 through T-02-04). All threat model mitigations implemented as specified:

- T-02-01: `auto_adjust=True` explicit and verified by inspect assertion
- T-02-02: `threads=False` enforced and verified by inspect assertion
- T-02-03: tenacity retry + `yfinance[nospam]` requests-ratelimiter active
- T-02-04: blocking human-verify checkpoint completed (auto-approved with evidence)

## Self-Check: PASSED

- `src/volforecast/ingest/equity.py` exists: FOUND
- `tests/fixtures/equity_sample.parquet` exists: FOUND
- `tests/unit/test_ingest.py` exists: FOUND
- Commit 4bfb7ae (RED): FOUND
- Commit c9dcef6 (GREEN task 1): FOUND
- Commit 7bec979 (GREEN task 2): FOUND
- All 9 tests pass: CONFIRMED
- `uv run ruff check src tests` exits 0: CONFIRMED
- config/assets.yaml contains all 5 symbols: CONFIRMED
- equity adapter source contains `auto_adjust=True`, `threads=False`, `retry`: CONFIRMED
