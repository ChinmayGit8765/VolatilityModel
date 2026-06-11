"""VolForecast CLI entry point.

Provides the `volforecast` console script with an `ingest` subcommand that fetches
OHLCV data, validates it through the validate_asset gate (Pandera + calendar checks),
and writes both raw and validated-processed parquet files.

Usage:
    # Ingest all assets from config/assets.yaml (default — no --symbol):
    volforecast ingest --start 2022-01-01

    # Ingest a single crypto asset:
    volforecast ingest --symbol BTC/USDT --start 2022-01-01
    volforecast ingest --symbol BTC/USDT --start 2022-01-01 --exchange kraken

    # Ingest a single equity:
    volforecast ingest --symbol SPY --start 2022-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Default ingest start date — 2+ years of daily history per the plan requirements
_DEFAULT_START = "2022-01-01"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="volforecast",
        description="VolForecast — Crypto + Stock Volatility Forecasting MLOps Platform",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── ingest subcommand ──────────────────────────────────────────────────────
    ingest = subparsers.add_parser(
        "ingest",
        help="Fetch OHLCV data for all assets (or a single --symbol), validate, write parquet",
    )
    ingest.add_argument(
        "--symbol",
        default=None,
        help=(
            "Optional: ingest a single symbol (e.g. BTC/USDT or SPY). "
            "If omitted, all assets in config/assets.yaml are ingested."
        ),
    )
    ingest.add_argument(
        "--start",
        default=_DEFAULT_START,
        help=f"Start date in YYYY-MM-DD format (default: {_DEFAULT_START}). "
        "For incremental re-runs the cache-first logic takes precedence and "
        "resumes from the last stored date per asset.",
    )
    ingest.add_argument(
        "--exchange",
        default=None,
        help=(
            "ccxt exchange id override for crypto (e.g. kraken as Binance geo-block fallback). "
            "When ingesting all assets, per-asset exchange from config is used unless this "
            "flag is specified."
        ),
    )

    return parser


def _ingest_single_asset(
    asset: dict,
    since_ms: int,
    exchange_id: str,
    start: str,
    out_path: Path,
    processed_out_path: Path,
    quarantine_path: Path,
) -> int:
    """Fetch, validate via validate_asset gate, and write raw + processed parquet.

    Pipeline:
      1. Fetch raw OHLCV from adapter (crypto or equity)
      2. Write raw parquet to out_path (unconditional; raw is always written)
      3. Merge the new batch with the existing processed parquet (no write) and
         run the validate_asset gate on the MERGED frame — the dataset that will
         actually live in data/processed/, not just the fresh batch
      4. On validation success: write the validated merged frame to processed_out_path
      5. On validation failure: quarantine file written, no processed parquet

    Return codes:
      0 — success (raw and processed written), or asset skipped (empty fetch /
          unknown asset_class).
      1 — fetch error OR validation rejection.  A rejection is the gate failing
          closed as designed: the quarantine report is written, no processed
          parquet is produced, and this per-asset rc 1 makes _cmd_ingest count
          the asset as failed.  The multi-asset loop still CONTINUES to the next
          asset; the overall process then exits non-zero so CI/orchestration can
          see that at least one asset did not reach data/processed/.
    """
    from pandera.errors import SchemaErrors

    from volforecast.config import symbol_slug
    from volforecast.ingest.base import incremental_update, merge_bars
    from volforecast.validate import ValidationError, validate_asset

    symbol = asset["symbol"]
    asset_class = asset.get("asset_class", "crypto")
    slug = symbol_slug(symbol)

    # ── Step 1: Fetch raw OHLCV ─────────────────────────────────────────────
    if asset_class == "crypto":
        from volforecast.ingest.crypto import fetch_crypto_ohlcv

        print(f"  Fetching {symbol} from {exchange_id}...")
        try:
            df = fetch_crypto_ohlcv(symbol, since_ms=since_ms, exchange_id=exchange_id)
        except RuntimeError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"  ERROR fetching {symbol} from {exchange_id}: {e}", file=sys.stderr)
            return 1

        # Mirror the equity empty guard (WR-08): a run started "today" returns
        # only the still-forming candle, which is then dropped — without this
        # guard an EMPTY raw AND processed parquet would be written (all checks
        # pass vacuously on an empty frame) and the run would report success.
        if df.empty:
            print(f"  WARNING: {symbol} returned no closed candles — skipping.", file=sys.stderr)
            return 0
    elif asset_class == "equity":
        from volforecast.ingest.equity import download_equity_ohlcv, effective_fetch_start

        # WR-09: always re-download from the earliest stored date so the whole
        # history lands on ONE split/dividend adjustment basis.  Fetching only
        # from a later --start would leave stored pre-start rows on the old
        # basis after a corporate action — a fabricated discontinuity at the
        # seam that no validation gate detects.
        fetch_start = effective_fetch_start(out_path, start)
        print(f"  Fetching equity {symbol} via yfinance since {fetch_start}...")
        try:
            result_dict = download_equity_ohlcv([symbol], start=fetch_start, end=None)
        except RuntimeError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"  ERROR fetching {symbol}: {e}", file=sys.stderr)
            return 1

        if symbol not in result_dict:
            print(f"  ERROR: {symbol} not in yfinance result (empty download?)", file=sys.stderr)
            return 1

        df = result_dict[symbol]
        if df.empty:
            print(f"  WARNING: {symbol} returned empty DataFrame — skipping.", file=sys.stderr)
            return 0
    else:
        print(f"  WARNING: Unknown asset_class '{asset_class}' for {symbol} — skipping.")
        return 0

    print(f"  Fetched {len(df)} rows.")

    # ── Step 2: Write raw parquet (always) ──────────────────────────────────
    raw_result = incremental_update(out_path, df)
    print(f"  Raw: stored {len(raw_result)} rows through {raw_result.index.max().date()}.")

    # ── Step 3: validate_asset gate on the MERGED stored dataset ────────────
    # Gate the frame that will actually live in data/processed/, not just the
    # freshly fetched batch (CR-02): merging before validating catches seam
    # gaps between the stored history and the new batch (e.g. an exchange
    # returning its first candle later than requested) and partial history
    # left behind by a previously rejected run.
    candidate = merge_bars(processed_out_path, df)
    quarantine_dir = quarantine_path.parent
    print(f"  Validating merged dataset ({asset_class} schema, {len(candidate)} rows)...")
    try:
        validated_df = validate_asset(candidate, asset_class, quarantine_dir, symbol_slug=slug)
    except (ValidationError, SchemaErrors) as e:
        print(
            f"  VALIDATION FAILED: {symbol} rejected — quarantine written to {quarantine_dir}",
            file=sys.stderr,
        )
        print(f"    Details: {e}", file=sys.stderr)
        print(f"  Skipping processed write for {symbol} (gate fails closed).", file=sys.stderr)
        # The gate failing closed is EXPECTED behaviour, but it still returns 1:
        # _cmd_ingest counts this asset as failed (the loop continues to the next
        # asset) and the overall process exits non-zero so CI/orchestration can
        # detect that not every asset reached data/processed/.
        return 1
    except Exception as e:
        print(f"  ERROR: Unexpected validation error for {symbol}: {e}", file=sys.stderr)
        return 1

    # ── Step 4: Write processed parquet (only on validation success) ─────────
    # validated_df IS the merged dataset (validated above), so write it directly:
    # everything in data/processed/ has passed the gates as a whole.
    processed_out_path.parent.mkdir(parents=True, exist_ok=True)
    validated_df.to_parquet(processed_out_path)
    print(
        f"  Processed: stored {len(validated_df)} rows through {validated_df.index.max().date()}."
    )
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    """Execute the ingest subcommand.

    When --symbol is provided: ingest that single asset.
    When --symbol is omitted: load all assets from config/assets.yaml and ingest each.

    Every asset — both crypto and equity — is gated through validate_asset before
    being promoted to data/processed/.  The raw parquet is always written.

    Cache-first incremental: resume_since_ms computes the start of each crypto fetch
    from the last stored date in the existing PROCESSED (validated) parquet, falling
    back to --start when no processed parquet exists yet (first run, or every prior
    run was rejected by validation).
    """
    from volforecast.config import load_assets, processed_path, project_root, raw_path, symbol_slug
    from volforecast.ingest.crypto import resume_since_ms

    # Parse start date -> millisecond epoch (used for crypto since, and equity start string)
    try:
        start_ts = pd.Timestamp(args.start, tz="UTC")
    except ValueError as e:
        print(f"ERROR: Invalid --start date '{args.start}': {e}", file=sys.stderr)
        return 1
    default_since_ms = int(start_ts.timestamp() * 1000)
    # Single root for BOTH config and data (WR-07): VOLFORECAST_ROOT env var,
    # falling back to the current working directory.
    root = project_root()

    # Build the list of assets to ingest
    if args.symbol:
        # Single-asset mode: construct a minimal asset dict from CLI args
        # Determine asset_class heuristically: crypto symbols contain "/"
        if "/" in args.symbol:
            asset_class = "crypto"
            exchange_id = args.exchange or "binance"
        else:
            asset_class = "equity"
            exchange_id = args.exchange or "nasdaq"
        assets_to_ingest = [
            {
                "symbol": args.symbol,
                "asset_class": asset_class,
                "exchange": exchange_id,
            }
        ]
    else:
        # Multi-asset mode: load full universe from config (resolved against the
        # same root as the data/ output — see volforecast.config.project_root).
        try:
            assets_to_ingest = load_assets()
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        if not assets_to_ingest:
            print("ERROR: No assets found in config/assets.yaml", file=sys.stderr)
            return 1

    print(f"Ingesting {len(assets_to_ingest)} asset(s) from {args.start}...")
    errors = 0

    for asset in assets_to_ingest:
        symbol = asset["symbol"]
        asset_class = asset.get("asset_class", "crypto")
        exchange_id = args.exchange or asset.get("exchange", "binance")
        slug = symbol_slug(symbol)

        out_path = raw_path(asset, data_root=root / "data")
        processed_out_path = processed_path(asset, data_root=root / "data")
        quarantine_path = root / "data" / "quarantine" / f"{slug}_quarantine.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        processed_out_path.parent.mkdir(parents=True, exist_ok=True)
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n[{asset_class.upper()}] {symbol} -> {out_path.relative_to(root)}")

        # Cache-first for crypto: resume from the last VALIDATED date (processed
        # parquet), not the raw parquet (CR-02).  If a previous run was rejected
        # by validation, raw is ahead of processed; resuming from raw would
        # permanently skip the rejected window in "validated" data.  Resuming
        # from the processed frontier re-fetches that window so the merged
        # dataset can be re-validated as a whole (incremental_update dedupes
        # any overlap with raw).
        if asset_class == "crypto":
            since_ms = resume_since_ms(processed_out_path, default_since_ms)
        else:
            since_ms = default_since_ms

        rc = _ingest_single_asset(
            asset=asset,
            since_ms=since_ms,
            exchange_id=exchange_id,
            start=args.start,
            out_path=out_path,
            processed_out_path=processed_out_path,
            quarantine_path=quarantine_path,
        )

        if rc != 0:
            errors += 1

    if errors:
        print(f"\n{errors} asset(s) failed or were rejected by validation.", file=sys.stderr)
        return 1

    print(f"\nAll {len(assets_to_ingest)} asset(s) ingested successfully.")
    return 0


def main() -> None:
    """Entry point for the `volforecast` console script."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        sys.exit(_cmd_ingest(args))
    else:
        parser.print_help()
        sys.exit(1)
