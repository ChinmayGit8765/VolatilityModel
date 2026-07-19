---
phase: 01-foundation-validated-data
plan: "03"
subsystem: validate
tags: [pandera, exchange_calendars, data-quality, tdd]
dependency_graph:
  requires: ["01-01"]
  provides: ["validate_asset", "equity_ohlcv_schema", "crypto_ohlcv_schema"]
  affects: ["01-04"]
tech_stack:
  added: []
  patterns:
    - "Pandera 0.31+ lazy validation with quarantine (import pandera.pandas as pa)"
    - "exchange_calendars XNYS sessions_in_range for equity calendar gate"
    - "pd.date_range(freq=D) for crypto 24/7 continuous calendar"
    - "CheckResult dataclass for structured pass/fail + offending index list"
    - "validate_asset single-dispatch pattern: schema + calendar + stale + OHLC"
key_files:
  created:
    - src/volforecast/validate/checks.py
    - tests/fixtures/equity_bad_weekend.parquet
    - tests/fixtures/crypto_gap.parquet
    - tests/unit/test_checks.py
    - tests/unit/test_schemas.py
  modified:
    - src/volforecast/validate/schemas.py
    - src/volforecast/validate/__init__.py
decisions:
  - "coerce=True on Pandera schemas to accept both ns and us datetime64 precision (pandas 2.x parquet reads produce us; in-memory construction produces ns)"
  - "Calendar session validation done in checks.py equity_session_check, not embedded in equity_ohlcv_schema index check — Pandera index checks cannot express exchange-calendar set-membership with structured failure output"
  - "validate_asset runs non-Pandera checks first, then Pandera schema, aggregating all failures into one quarantine CSV — fails closed on first failure set, never silently passes partial data"
  - "CheckResult dataclass with passed/offending_index/reason returned from all check functions for clean dispatcher aggregation"
metrics:
  duration: "~35 minutes"
  completed: "2026-06-10"
  tasks_completed: 2
  files_changed: 7
requirements: [INGEST-03, INGEST-04]
---

# Phase 01 Plan 03: Validation Layer — Calendar-aware Checks + validate_asset Summary

**One-liner:** Calendar-aware OHLCV validation with XNYS equity sessions, crypto 24/7 gap detection, stale-row and OHLC consistency checks, and a single validate_asset dispatcher that quarantines and fails closed.

## What Was Built

### Task 1: Calendar-aware + OHLC + stale checks with broken-data fixtures

Implemented `src/volforecast/validate/checks.py` with four check functions, each returning a `CheckResult(passed, offending_index, reason)`:

- `crypto_gap_check(df)`: builds expected `pd.date_range(min, max, freq="D")` and reports any missing days as gaps.
- `equity_session_check(df)`: calls `xcals.get_calendar("XNYS").sessions_in_range(start, end)` (tz-naive args required — strips UTC before calling), reports extra rows (fabricated weekend/holiday bars) and missing expected sessions.
- `stale_row_check(df)`: flags when fewer than 95% of close values are non-duplicated.
- `ohlc_consistency_check(df)`: detects any row where high < low/open/close or low > open/close.

Created two broken-data fixtures:
- `tests/fixtures/equity_bad_weekend.parquet`: 8 valid XNYS session rows + one fabricated Saturday (2022-01-08) row. Dayofweek 5 confirmed present.
- `tests/fixtures/crypto_gap.parquet`: 20-day range with 2022-01-11 removed from the interior.

### Task 2: Equity schema, validate_asset dispatcher, quarantine-on-failure

Extended `src/volforecast/validate/schemas.py`:
- Added `equity_ohlcv_schema` (mirrors `crypto_ohlcv_schema` — same positivity/OHLC/index constraints).
- **Bug fix (Rule 1):** Changed `_UTC_DATETIME_DTYPE = pd.DatetimeTZDtype(tz="UTC")` (nanosecond default) to `pd.DatetimeTZDtype("us", tz="UTC")` and set `coerce=True` on both schemas. Pandas 2.x parquet reads produce `datetime64[us, UTC]` but in-memory DatetimeIndex construction (from `exchange_calendars` sessions) produces `datetime64[ns, UTC]`. Without `coerce=True`, clean DataFrames fail with WRONG_DATATYPE on the precision mismatch.

Implemented `src/volforecast/validate/__init__.py` with `validate_asset(df, asset_class, quarantine_dir)`:
1. Routes to `equity_session_check` or `crypto_gap_check` by asset_class.
2. Runs `stale_row_check` and `ohlc_consistency_check` for both.
3. Runs Pandera schema validation with `lazy=True`.
4. Aggregates all failures into one quarantine CSV (`{asset_class}_{timestamp}.csv`) under `quarantine_dir`.
5. Raises `ValidationError` (or re-raises `SchemaErrors`) after writing — fails closed, never silently passes.

Re-exports: `validate_asset`, `equity_ohlcv_schema`, `crypto_ohlcv_schema`, `validate_and_quarantine`.

## Test Coverage

| Test file | Tests | Result |
|-----------|-------|--------|
| tests/unit/test_checks.py | 9 | PASS |
| tests/unit/test_schemas.py | 9 | PASS |
| tests/unit/test_skeleton_e2e.py (pre-existing) | 3 | PASS |
| **Total** | **21** | **PASS** |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed exchange_calendars sessions_in_range rejects tz-aware timestamps**
- **Found during:** Task 1, first GREEN run
- **Issue:** `xnys.sessions_in_range(start, end)` raises `AttributeError: 'UTC' object has no attribute 'key'` when passed a tz-aware `pd.Timestamp`. The API requires tz-naive dates.
- **Fix:** Strip timezone before calling: `start = df.index.normalize().min().tz_localize(None)`
- **Files modified:** `src/volforecast/validate/checks.py`
- **Commit:** 9bde630

**2. [Rule 1 - Bug] Fixed pandas 2.x datetime64 precision mismatch in Pandera schema**
- **Found during:** Task 2, first GREEN run
- **Issue:** `pd.DatetimeTZDtype(tz="UTC")` defaults to `ns` (nanosecond) precision. Parquet reads under pandas 2.x produce `datetime64[us, UTC]`. Pandera raises `WRONG_DATATYPE` on the clean crypto_sample fixture.
- **Fix:** Changed to `pd.DatetimeTZDtype("us", tz="UTC")` and added `coerce=True` to both schemas so both `ns` (in-memory construction) and `us` (parquet reads) are accepted.
- **Files modified:** `src/volforecast/validate/schemas.py`
- **Commit:** 90c454a

**3. [Rule 1 - Bug] Fixed inline equity test fixture triggering stale_row_check**
- **Found during:** Task 2 GREEN — `test_validate_asset_clean_equity_passes` failed because the inline fixture used constant `close=152.0` for all 10 rows, correctly triggering `stale_row_check` (0% non-duplicated).
- **Fix:** Updated inline equity fixtures in `test_schemas.py` to use varied close values (`152.0 + i * 0.5`).
- **Files modified:** `tests/unit/test_schemas.py`
- **Commit:** 90c454a

## Threat Model Coverage

All mitigations from the plan's threat register were implemented:

| Threat ID | Mitigation | Status |
|-----------|-----------|--------|
| T-03-01 | crypto_gap_check + equity_session_check | Implemented + tested |
| T-03-02 | equity_session_check rejects non-XNYS index values | Implemented + tested |
| T-03-03 | Pandera positivity + ohlc_consistency_check | Implemented + tested |
| T-03-04 | stale_row_check (>95% duplicated closes) | Implemented + tested |
| T-03-06 | Gate fails closed (raises) — proven by tests | Implemented + tested |

## Known Stubs

None — all check functions are fully implemented with real calendar logic; no placeholder values or TODO paths.

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema changes at trust boundaries beyond what the plan's threat model already covers.

## Self-Check: PASSED

- `src/volforecast/validate/checks.py` exists and contains XNYS + sessions_in_range
- `src/volforecast/validate/schemas.py` uses `import pandera.pandas as pa` (verified by inspect assertion)
- `src/volforecast/validate/__init__.py` exports validate_asset
- `tests/fixtures/equity_bad_weekend.parquet` exists
- `tests/fixtures/crypto_gap.parquet` exists
- `tests/unit/test_checks.py` exists (9 tests pass)
- `tests/unit/test_schemas.py` exists (9 tests pass)
- Commits: 6eb70b5 (RED checks), 9bde630 (GREEN checks), da065f4 (RED schemas), 90c454a (GREEN schemas)
