---
phase: 01-foundation-validated-data
reviewed: 2026-06-11T00:49:10Z
depth: standard
files_reviewed: 21
files_reviewed_list:
  - src/volforecast/cli.py
  - src/volforecast/config.py
  - src/volforecast/ingest/base.py
  - src/volforecast/ingest/crypto.py
  - src/volforecast/ingest/equity.py
  - src/volforecast/ingest/__init__.py
  - src/volforecast/validate/__init__.py
  - src/volforecast/validate/checks.py
  - src/volforecast/validate/schemas.py
  - config/assets.yaml
  - infra/docker-compose.yml
  - infra/mlflow-entrypoint.sh
  - infra/postgres-init/01-create-databases.sh
  - infra/.env.example
  - .github/workflows/ci.yml
  - tests/conftest.py
  - tests/unit/test_checks.py
  - tests/unit/test_ingest.py
  - tests/unit/test_pipeline.py
  - tests/unit/test_schemas.py
  - tests/unit/test_skeleton_e2e.py
findings:
  critical: 2
  warning: 9
  info: 9
  total: 20
status: findings
---

# Phase 01: Code Review Report

**Reviewed:** 2026-06-11T00:49:10Z
**Depth:** standard
**Files Reviewed:** 21
**Status:** findings

## Narrative Findings (AI reviewer)

## Summary

The package structure, schema gates, incomplete-candle exclusion, and merge-dedupe core are solid: candle-close boundary math is correct (`open_time + timeframe_ms <= now_ms`), `incremental_update` dedupe/sort semantics are correct and tested, `.gitattributes` enforces LF on the shell scripts (verified `i/lf w/lf` in the index), `infra/.env` is gitignored, `yaml.safe_load` is used, and `VOLFORECAST_NO_LIVE_API` is enforced in both adapters and CI.

Two Critical defects remain. First, `stale_row_check` flags *global* duplicate closes rather than consecutive runs; empirically (pandas 2.3.3, 750-day cent-quantized random walks) clean equity series score 85.3% / 69.7% non-duplicated against a 95% pass threshold — real SPY/AAPL/MSFT history will be falsely quarantined and never reach `data/processed/`. Second, validation gates only the freshly fetched batch, never the merged stored dataset, so incremental crypto runs can promote a processed parquet containing seam gaps (or partial history after a prior rejection) that the gap check would reject — a silent violation of the phase's core invariant.

## Critical Issues

### CR-01: `stale_row_check` falsely quarantines clean real-world equity data — the gate will block the pipeline

**File:** `src/volforecast/validate/checks.py:178-186`
**Issue:** The check uses `df["close"].duplicated(keep=False)` over the **entire series**, so any close value occurring twice anywhere in history marks all its occurrences as "stale", and the frame fails unless >95% of closes are globally unique. Equity closes are quantized to cents, so collisions are the norm over multi-year history, not a defect. Empirical verification with the project's pinned pandas 2.3.3: a 750-day random walk around $150 rounded to cents yields **85.3%** non-duplicated (a low-volatility walk yields **69.7%**) — both far below the 95% threshold. With the required 2+ years of history, SPY/AAPL/MSFT will be rejected as "stale" by `validate_asset`, quarantined, and never promoted to `data/processed/`. The existing tests miss this because they use an 18-row all-distinct fixture and an all-identical frame — no realistic mid-density case. A stale/stuck feed manifests as *consecutive* repeated closes, which is what the check should target.
**Fix:**
```python
def stale_row_check(df: pd.DataFrame, max_run: int = 5) -> CheckResult:
    if df.empty or "close" not in df.columns:
        return CheckResult(passed=True)
    # A stale feed produces runs of consecutive identical closes.
    same_as_prev = df["close"].diff() == 0
    run_id = (~same_as_prev).cumsum()
    run_lengths = same_as_prev.groupby(run_id).cumsum() + 1
    offending_mask = run_lengths > max_run
    if not offending_mask.any():
        return CheckResult(passed=True)
    return CheckResult(
        passed=False,
        offending_index=df.index[offending_mask].tolist(),
        reason=f"Stale feed: close repeated for more than {max_run} consecutive rows",
    )
```
Then add a test with realistic cent-quantized closes (some non-consecutive duplicates) that must PASS, plus a test with a 6-row identical-close run that must FAIL.

### CR-02: Processed dataset is never re-validated after merge — incremental runs silently promote gap-ridden "validated" data

**File:** `src/volforecast/cli.py:149,166` (with `src/volforecast/ingest/crypto.py:23-61`, `src/volforecast/ingest/base.py:75-109`)
**Issue:** `validate_asset(df, ...)` gates only the freshly fetched batch; the actual stored dataset — the output of `incremental_update(processed_out_path, validated_df)` — is never validated. Two concrete failure paths:
1. **Seam gap:** `resume_since_ms` returns `last_raw_date + 1 day`. If the exchange returns its first candle later than that (outage, delisting window, Kraken history limits), the new batch is internally continuous so `crypto_gap_check` passes (it only inspects the batch's own min→max range), but the *merged* processed parquet now has a hole between the stored history and the new batch that the gap check would reject if it ever saw the full frame.
2. **Partial history after rejection:** Run 1 fetches Jan–Mar, fails validation → raw written, processed not written. Run 2 resumes from the **raw** parquet's last date, fetches Apr+ which passes → the processed parquet is created containing *only* Apr+ — the rejected Jan–Mar window is silently absent from "validated" data and nothing ever flags it.

Either way the phase invariant — "everything in `data/processed/` passed the gates" — is violated without any quarantine record. Downstream phases (features, walk-forward) consume processed parquet assuming gap-free history.
**Fix:** Validate the merged frame, not the batch. In `_ingest_single_asset`:
```python
proc_result = incremental_update(processed_out_path, validated_df)
# Gate the STORED dataset: catches seam gaps and partial history.
validate_asset(proc_result, asset_class, quarantine_dir)
```
(or merge first into a temp frame, validate, then write). At daily scale the double validation cost is negligible. Add a test: existing processed parquet ending at D, new batch starting at D+3, assert `validate_asset` rejects the merged result.

## Warnings

### WR-01: Quarantine reports collide on same-second writes and carry no asset identity

**File:** `src/volforecast/validate/__init__.py:111-112`, `src/volforecast/cli.py:146,231`
**Issue:** The CLI computes a per-asset `quarantine_path` (`{slug}_quarantine.csv`) but `_ingest_single_asset` uses only `.parent`; `validate_asset` names files `{asset_class}_{YYYYmmddTHHMMSSZ}.csv`. Timestamp resolution is one second, so BTC/USDT and ETH/USDT both failing in the same second produce the **same filename** and the second report silently overwrites the first. Even without collision, a quarantine CSV cannot be traced back to the symbol that produced it.
**Fix:** Add a `symbol_slug` parameter to `validate_asset` and name reports `{asset_class}_{slug}_{timestamp}.csv` (or honor the full `quarantine_path` the CLI already builds).

### WR-02: Docstring and inline comment state the opposite of the actual return-code contract

**File:** `src/volforecast/cli.py:90-92,157-160`
**Issue:** The `_ingest_single_asset` docstring says "Validation failures … do NOT cause a non-zero return code" and the comment at lines 157-159 says "Return 0 so the pipeline continues" — but line 160 is `return 1`, `_cmd_ingest` counts it as an error, and the process exits 1. `test_pipeline_quarantines_and_skips_bad_asset` asserts `rc == 1`, so the code is the real contract and the comments lie. A future maintainer "fixing" the code to match the comment would silently change CI/orchestration exit semantics.
**Fix:** Rewrite the docstring/comment to match behavior (validation rejection ⇒ per-asset rc 1, loop continues, overall exit 1), or introduce distinct return codes (e.g. 2 = validation rejection) if rejections should be distinguishable from fetch errors.

### WR-03: yfinance's dominant failure mode (silent empty DataFrame) defeats the retry/backoff entirely

**File:** `src/volforecast/ingest/equity.py:38-66,100`
**Issue:** `yf.download` swallows per-ticker errors and rate-limit failures, printing to stderr and returning an empty or NaN-filled frame **without raising** — so the tenacity retry (which fires only on exceptions) never triggers for the very failure it was added to handle. The eventual error is a confusing `ValueError: missing columns` from `normalize_equity_frame`, outside the retry loop. Conversely, `retry_if_exception_type(Exception)` retries programming errors (TypeError, KeyError) 5 times with waits up to 60s, masking bugs for minutes.
**Fix:** Inside `_download_with_retry`, raise a retryable error on empty results:
```python
raw = yf.download(...)
if raw.empty:
    raise OSError(f"yf.download returned empty frame for {tickers} (rate limit?)")
return raw
```
and narrow the filter to transient types (e.g. `retry_if_exception_type((OSError, ConnectionError))`).

### WR-04: `prefect-worker` polls a work pool that nothing creates — crash loop on first `docker compose up`

**File:** `infra/docker-compose.yml:89`
**Issue:** `prefect worker start --pool local-pool` errors out when the pool does not exist (Prefect 3 only auto-creates the pool when `--type` is supplied). Nothing in the repo creates `local-pool` — no init container, no documented `prefect work-pool create` step — so the worker exits immediately and `restart: unless-stopped` turns it into a crash loop.
**Fix:** `command: prefect worker start --pool local-pool --type process` (auto-creates the pool), or add a one-shot init service / documented setup step that runs `prefect work-pool create local-pool --type process`.

### WR-05: All services bound to 0.0.0.0 with default credentials and no auth — exposed to the LAN

**File:** `infra/docker-compose.yml:25-27,44-45,53-54,73-74`; `infra/mlflow-entrypoint.sh:15`
**Issue:** Published ports in `"5433:5432"` form bind to all host interfaces, and on Windows Docker Desktop forwards published ports through the host firewall. Postgres is reachable from the local network with the compose-default credentials `volforecast:volforecast` (the `:-` fallbacks mean the stack runs fine if the user never creates `infra/.env`, so the documented "edit with real credentials" step is unenforced). MLflow runs with `--allowed-hosts '*'` and no authentication; Prefect's API is likewise unauthenticated. For a local single-machine dev stack this should all be loopback-only.
**Fix:** Bind to loopback: `"127.0.0.1:5433:5432"`, `"127.0.0.1:5000:5000"`, `"127.0.0.1:4200:4200"`. Make the password required so misconfiguration fails loudly: `POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?set POSTGRES_PASSWORD in infra/.env}`. With loopback binding, `--allowed-hosts '*'` can stay or be narrowed to `localhost`.

### WR-06: MLflow container runs an unpinned `pip install` from the network on every start

**File:** `infra/mlflow-entrypoint.sh:8`
**Issue:** `pip install psycopg2-binary -q` at container startup makes the tracking server's availability depend on PyPI being reachable, installs an unpinned version (silent drift between restarts), and re-downloads on every container recreation. A transient PyPI failure leaves the stack down with a non-obvious error.
**Fix:** Build a 3-line custom image instead:
```dockerfile
FROM ghcr.io/mlflow/mlflow:v3.13.0
RUN pip install --no-cache-dir psycopg2-binary==2.9.10
```
and reference it via `build:` in compose. At minimum, pin the version in the entrypoint.

### WR-07: Config resolved relative to the installed package, data relative to cwd — split-brain paths

**File:** `src/volforecast/config.py:15-19`; `src/volforecast/cli.py:194`
**Issue:** `_CONFIG_DIR`/`_DATA_ROOT` are `Path(__file__).parent.parent.parent / ...`, which is the repo root only under a src-layout **editable** install; for a built wheel this resolves into `site-packages`' parents where no `config/assets.yaml` exists (the project explicitly claims an "installable volforecast package"). Meanwhile `_cmd_ingest` uses `Path.cwd()` as the project root for `data/` output. Run the console script from any directory other than the repo root and it reads the repo's config (editable install) while writing `data/` under the current directory — two different roots, silently.
**Fix:** Resolve both config and data from a single root — `Path.cwd()` (or a `VOLFORECAST_ROOT` env var / `--config` flag) — and raise a clear `FileNotFoundError` pointing at the expected `config/assets.yaml` location when absent.

### WR-08: Crypto path lacks the empty-fetch guard the equity path has — writes empty raw AND processed parquet, reports success

**File:** `src/volforecast/cli.py:139-167` (contrast `cli.py:132-134`)
**Issue:** The equity branch skips on `df.empty` with a warning; the crypto branch does not. A first run started "today" (only the forming candle is returned, then dropped) yields an empty DataFrame → an empty raw parquet is written, the empty frame passes every check vacuously, an **empty processed parquet** is written, and the run prints success ("Raw: stored 0 rows through NaT" — verified `NaT.date()` returns NaT under pandas 2.3.3, so it prints rather than crashes). Downstream consumers see a zero-row "validated" file. Also, `pd.concat` with an empty frame triggers a pandas 2.x FutureWarning on every same-day re-run.
**Fix:** Mirror the equity guard after the crypto fetch:
```python
if df.empty:
    print(f"  WARNING: {symbol} returned no closed candles — skipping.", file=sys.stderr)
    return 0
```

### WR-09: Equity merge can mix split/dividend adjustment bases across runs

**File:** `src/volforecast/cli.py:119,166`; `src/volforecast/ingest/base.py:105-106`
**Issue:** With `auto_adjust=True`, Yahoo rewrites the *entire* history every time a split/dividend occurs. The default flow re-downloads from `--start` and `keep="last"` replaces all overlapping rows — consistent. But if a user re-runs with a later `--start` (explicitly supported by the CLI) after a corporate action, the stored pre-`start` rows remain on the **old** adjustment basis while new rows use the new basis. A 4:1 split at the seam fabricates a −75% one-day return — a volatility explosion that no gate detects (prices stay positive, OHLC-consistent, calendar-aligned). For a volatility-forecasting project this is a direct label/feature corruption hazard.
**Fix:** For equities, always full-refresh from the asset's original inception start (ignore later `--start` for the fetch window), or detect basis drift by comparing overlapping rows: if stored close differs from re-downloaded close beyond a tolerance for the same date, replace the full history instead of merging.

## Info

### IN-01: `validate_asset` comment claims SchemaErrors is "wrapped" — it is re-raised bare; `ValidationError` missing from `__all__`

**File:** `src/volforecast/validate/__init__.py:45-50,191-195`
**Issue:** The comment says the Pandera exception is wrapped "so callers catching the broader ValidationError also get it", but `raise pandera_exc` re-raises bare `SchemaErrors`, which is not a `ValidationError` subclass — callers catching only `ValidationError` miss all Pandera failures (cli.py happens to catch both). `ValidationError` is also absent from `__all__` despite being part of the public contract.
**Fix:** `raise ValidationError(...) from pandera_exc` for a single gate exception type, and add `ValidationError` to `__all__`.

### IN-02: Quarantine report omits offending row data (context decision: "offending rows + reason")

**File:** `src/volforecast/validate/__init__.py:128-160,188`
**Issue:** Non-Pandera failures record only `check`/`reason`/index-label strings — the OHLCV values of offending rows are not captured, reducing forensic value and deviating from the Phase 1 decision ("quarantine report (offending rows + reason)").
**Fix:** Join `offending_index` back to `df` and include row values (missing-session/gap entries keep NaN columns).

### IN-03: `symbol_slug` collisions: BTC/USDT and BTC/USDC map to the same file

**File:** `src/volforecast/config.py:52-58`
**Issue:** Both normalize to `BTC-USD`, hence the same raw/processed parquet path — configuring both would silently merge two different instruments into one series. Latent today (config only has USDT pairs).
**Fix:** Assert slug uniqueness in `load_assets`, or keep the quote currency in the slug.

### IN-04: `getattr(ccxt, exchange_id)` accepts any ccxt attribute

**File:** `src/volforecast/ingest/crypto.py:98`
**Issue:** `--exchange milliseconds` (or any non-exchange attribute) produces a confusing `TypeError` deep in instantiation rather than a clear error.
**Fix:** `if exchange_id not in ccxt.exchanges: raise ValueError(f"Unknown ccxt exchange: {exchange_id}")` before the `getattr`.

### IN-05: Dead code, wrong annotation, and a doctest-shaped example that would fail

**File:** `src/volforecast/cli.py:127-129,119`; `src/volforecast/ingest/equity.py:73`; `src/volforecast/ingest/base.py:37-42`; `tests/fixtures/equity_sample.parquet`
**Issue:** (a) `if symbol not in result_dict` in cli.py is unreachable — `download_equity_ohlcv` builds the dict by comprehension over `tickers`, so the key always exists (failures raise instead). (b) `download_equity_ohlcv` annotates `end: str` but cli passes `end=None`. (c) `drop_incomplete_candles`' docstring example is doctest-formatted but uses runtime variables in the expected output — it fails the moment `--doctest-modules` is enabled. (d) `tests/fixtures/equity_sample.parquet` is committed but referenced by no test.
**Fix:** Remove the dead branch; annotate `end: str | None`; rewrite the example as prose or a valid doctest; use or delete the unused fixture.

### IN-06: `test_ingest_all_dispatches_by_asset_class` reimplements the dispatch instead of exercising `_cmd_ingest`

**File:** `tests/unit/test_ingest.py:204-287`
**Issue:** The test inlines its own dispatch loop, so the CLI's actual routing logic is never executed despite the test's name and docstring claiming "the CLI dispatch" is tested — false coverage confidence for the multi-asset path (`test_pipeline.py` only covers `_ingest_single_asset`).
**Fix:** Invoke `cli._cmd_ingest` with a monkeypatched `load_assets`/adapters and assert per-asset parquet outputs, or rename the test to reflect what it actually covers.

### IN-07: CI workflow lacks `permissions:`/`concurrency:`; duplicate runs for same-repo PRs

**File:** `.github/workflows/ci.yml:11-13`
**Issue:** Bare `on: push` + `pull_request` double-runs every commit on a same-repo PR branch; default `GITHUB_TOKEN` permissions are broader than a lint+test job needs.
**Fix:** Add `permissions: contents: read`, a `concurrency:` group keyed on the ref, and/or restrict push triggers to `main`.

### IN-08: `python -m volforecast.ingest` requires a redundant `ingest` argument

**File:** `src/volforecast/cli.py:36` (via `src/volforecast/ingest/__main__.py`)
**Issue:** `__main__.py` delegates to `cli.main()`, whose parser requires a subcommand — so the context-promised invocation `python -m volforecast.ingest` exits with an argparse error; users must type `python -m volforecast.ingest ingest --start …`.
**Fix:** In `__main__.py`, prepend `"ingest"` to `sys.argv` (or call a dedicated ingest entry function) so the module form works as documented.

### IN-09: Infra documentation error and unquoted SQL identifier in init script

**File:** `infra/.env.example:12-13`; `infra/postgres-init/01-create-databases.sh:8`
**Issue:** (a) `.env.example` claims `prefectdb` is "created automatically on first Prefect start" — it is actually created by the Postgres init script; a user reading the comment might delete the init mount believing Prefect self-provisions. (b) `OWNER $POSTGRES_USER` interpolates an unquoted identifier into SQL: a username containing `-` or quotes breaks the statement (low risk — local operator-controlled value).
**Fix:** Correct the comment; quote the identifier (`OWNER "$POSTGRES_USER"` → `OWNER \"...\"` in the heredoc) or document the username constraint.

---

## Fix Status

Fix scope: all Critical and Warning findings (Info findings deferred).
Every fix was verified with the full test suite (47 passed) and ruff check/format clean.

| Finding | Severity | Status | Commit | Notes |
|---------|----------|--------|--------|-------|
| CR-01 | Critical | Fixed | 170aed3 | stale_row_check now flags consecutive runs (>5) instead of global duplicates; cent-quantized random-walk + 6-run + boundary tests added |
| CR-02 | Critical | Fixed | 033ba46 | validate_asset now gates the MERGED processed frame (new merge_bars helper); crypto resume frontier moved from raw to processed parquet; seam-gap and post-rejection resume tests added |
| WR-01 | Warning | Fixed | 86cebfa | Quarantine reports named {asset_class}_{slug}_{timestamp}.csv; CLI passes the symbol slug |
| WR-02 | Warning | Fixed | 467692c | Docstring/comments rewritten to match the real contract: validation rejection ⇒ per-asset rc 1, loop continues, overall exit 1 |
| WR-03 | Warning | Fixed | c89e2d3 | Empty yf.download result raises retryable OSError; retry filter narrowed to (OSError, ConnectionError); empty-retry + no-retry-on-TypeError tests added |
| WR-04 | Warning | Fixed | a4affda | `prefect worker start --pool local-pool --type process` auto-creates the pool |
| WR-05 | Warning | Fixed | 749e14b | All published ports bound to 127.0.0.1; POSTGRES_PASSWORD uses `:?` (required, no default) in all three references |
| WR-06 | Warning | Fixed | 58f1fe9 | infra/mlflow/Dockerfile pins psycopg2-binary==2.9.10 at build time; entrypoint pip install removed |
| WR-07 | Warning | Fixed | a3fa113 | config and data both resolve via project_root() (VOLFORECAST_ROOT env var, else cwd); load_assets raises clear FileNotFoundError |
| WR-08 | Warning | Fixed | 04a2764 | Crypto branch skips empty fetches (rc 0, nothing written) mirroring the equity guard |
| WR-09 | Warning | Fixed | 8333c6a | effective_fetch_start re-downloads equities from min(--start, stored inception) so merges never mix adjustment bases |
| IN-01–IN-09 | Info | Deferred | — | Out of fix scope (Info tier); unchanged |

_Fixed: 2026-06-11_
_Fixer: Claude (gsd-code-fixer)_

---

_Reviewed: 2026-06-11T00:49:10Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
