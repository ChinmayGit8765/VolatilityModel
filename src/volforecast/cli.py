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
      3. Run validate_asset gate — BOTH crypto and equity schemas
      4. On validation success: write validated df to processed_out_path
      5. On validation failure: quarantine file written, no processed parquet

    Returns 0 on success (both raw and processed written), 1 on any error.
    Validation failures are logged but do NOT cause a non-zero return code unless
    the failure is unexpected — the gate failing closed is expected behaviour.
    """
    from pandera.errors import SchemaErrors

    from volforecast.ingest.base import incremental_update
    from volforecast.validate import ValidationError, validate_asset

    symbol = asset["symbol"]
    asset_class = asset.get("asset_class", "crypto")

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
    elif asset_class == "equity":
        from volforecast.ingest.equity import download_equity_ohlcv

        print(f"  Fetching equity {symbol} via yfinance since {start}...")
        try:
            result_dict = download_equity_ohlcv([symbol], start=start, end=None)
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

    # ── Step 3: validate_asset gate (crypto + equity both go through here) ──
    quarantine_dir = quarantine_path.parent
    print(f"  Validating ({asset_class} schema)...")
    try:
        validated_df = validate_asset(df, asset_class, quarantine_dir)
    except (ValidationError, SchemaErrors) as e:
        print(
            f"  VALIDATION FAILED: {symbol} rejected — quarantine written to {quarantine_dir}",
            file=sys.stderr,
        )
        print(f"    Details: {e}", file=sys.stderr)
        print(f"  Skipping processed write for {symbol} (gate fails closed).", file=sys.stderr)
        # Validation failure is NOT a pipeline error — it is expected behaviour.
        # Return 0 so the pipeline continues to next asset.
        # The caller can detect missing processed parquet to identify rejected assets.
        return 1
    except Exception as e:
        print(f"  ERROR: Unexpected validation error for {symbol}: {e}", file=sys.stderr)
        return 1

    # ── Step 4: Write processed parquet (only on validation success) ─────────
    proc_result = incremental_update(processed_out_path, validated_df)
    print(f"  Processed: stored {len(proc_result)} rows through {proc_result.index.max().date()}.")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    """Execute the ingest subcommand.

    When --symbol is provided: ingest that single asset.
    When --symbol is omitted: load all assets from config/assets.yaml and ingest each.

    Every asset — both crypto and equity — is gated through validate_asset before
    being promoted to data/processed/.  The raw parquet is always written.

    Cache-first incremental: resume_since_ms computes the start of each crypto fetch
    from the last stored date in the existing parquet, falling back to --start only when
    the parquet does not exist.
    """
    from volforecast.config import load_assets, processed_path, raw_path, symbol_slug
    from volforecast.ingest.crypto import resume_since_ms

    # Parse start date -> millisecond epoch (used for crypto since, and equity start string)
    try:
        start_ts = pd.Timestamp(args.start, tz="UTC")
    except ValueError as e:
        print(f"ERROR: Invalid --start date '{args.start}': {e}", file=sys.stderr)
        return 1
    default_since_ms = int(start_ts.timestamp() * 1000)
    project_root = Path.cwd()

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
        # Multi-asset mode: load full universe from config
        assets_to_ingest = load_assets()
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

        out_path = raw_path(asset, data_root=project_root / "data")
        processed_out_path = processed_path(asset, data_root=project_root / "data")
        quarantine_path = project_root / "data" / "quarantine" / f"{slug}_quarantine.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        processed_out_path.parent.mkdir(parents=True, exist_ok=True)
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n[{asset_class.upper()}] {symbol} -> {out_path.relative_to(project_root)}")

        # Cache-first for crypto: resume from last stored date if parquet exists
        if asset_class == "crypto":
            since_ms = resume_since_ms(out_path, default_since_ms)
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
