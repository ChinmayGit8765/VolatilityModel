"""VolForecast CLI entry point.

Provides the `volforecast` console script with an `ingest` subcommand that fetches
OHLCV data, validates it through the Pandera gate, and writes the versioned parquet.

Usage:
    volforecast ingest --symbol BTC/USDT --start 2022-01-01
    volforecast ingest --symbol BTC/USDT --start 2022-01-01 --exchange kraken
    python -m volforecast.ingest --symbol BTC/USDT --start 2022-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="volforecast",
        description="VolForecast — Crypto + Stock Volatility Forecasting MLOps Platform",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── ingest subcommand ──────────────────────────────────────────────────────
    ingest = subparsers.add_parser(
        "ingest",
        help="Fetch OHLCV data, validate, and write parquet",
    )
    ingest.add_argument(
        "--symbol",
        required=True,
        help="Trading symbol in ccxt format, e.g. BTC/USDT",
    )
    ingest.add_argument(
        "--start",
        required=True,
        help="Start date in YYYY-MM-DD format (ISO 8601)",
    )
    ingest.add_argument(
        "--exchange",
        default="binance",
        help="ccxt exchange id (default: binance; use kraken as Binance geo-block fallback)",
    )

    return parser


def _cmd_ingest(args: argparse.Namespace) -> int:
    """Execute the ingest subcommand.

    Fetches OHLCV from the specified exchange, validates through the Pandera gate,
    and writes/merges into the parquet store via incremental_update.
    """
    from volforecast.config import symbol_slug
    from volforecast.ingest.base import incremental_update
    from volforecast.ingest.crypto import fetch_crypto_ohlcv
    from volforecast.validate.schemas import crypto_ohlcv_schema, validate_and_quarantine

    # Parse start date -> millisecond epoch
    try:
        start_ts = pd.Timestamp(args.start, tz="UTC")
    except ValueError as e:
        print(f"ERROR: Invalid --start date '{args.start}': {e}", file=sys.stderr)
        return 1
    since_ms = int(start_ts.timestamp() * 1000)

    # Determine output paths
    slug = symbol_slug(args.symbol)
    # Resolve paths relative to the project root (CWD when CLI is invoked)
    project_root = Path.cwd()
    out_path = project_root / "data" / "raw" / "crypto" / f"{slug}.parquet"
    quarantine_path = project_root / "data" / "quarantine" / f"{slug}_quarantine.csv"

    print(f"Fetching {args.symbol} from {args.exchange} since {args.start}...")
    try:
        df = fetch_crypto_ohlcv(args.symbol, since_ms=since_ms, exchange_id=args.exchange)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR fetching data from {args.exchange}: {e}", file=sys.stderr)
        return 1

    print(f"Fetched {len(df)} rows. Running validation gate...")
    try:
        validated = validate_and_quarantine(df, crypto_ohlcv_schema, quarantine_path)
    except Exception as e:
        print(
            f"ERROR: Validation gate rejected data — quarantine written to {quarantine_path}",
            file=sys.stderr,
        )
        print(f"  Details: {e}", file=sys.stderr)
        return 1

    print(f"Validation passed. Writing to {out_path}...")
    result = incremental_update(out_path, validated)
    print(f"Done. Stored {len(result)} rows. Last date: {result.index.max()}")
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
