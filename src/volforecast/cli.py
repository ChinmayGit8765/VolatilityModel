"""VolForecast CLI entry point.

Provides the `volforecast` console script with an `ingest` subcommand that fetches
OHLCV data, validates it through the Pandera gate, and writes the versioned parquet.

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


def _ingest_single_crypto(
    symbol: str,
    since_ms: int,
    exchange_id: str,
    out_path: Path,
    quarantine_path: Path,
) -> int:
    """Fetch, validate, and write a single crypto asset."""
    from volforecast.ingest.base import incremental_update
    from volforecast.ingest.crypto import fetch_crypto_ohlcv
    from volforecast.validate.schemas import crypto_ohlcv_schema, validate_and_quarantine

    print(f"  Fetching {symbol} from {exchange_id}...")
    try:
        df = fetch_crypto_ohlcv(symbol, since_ms=since_ms, exchange_id=exchange_id)
    except RuntimeError as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"  ERROR fetching {symbol} from {exchange_id}: {e}", file=sys.stderr)
        return 1

    print(f"  Fetched {len(df)} rows. Validating...")
    try:
        validated = validate_and_quarantine(df, crypto_ohlcv_schema, quarantine_path)
    except Exception as e:
        print(
            f"  ERROR: Validation rejected {symbol} — quarantine at {quarantine_path}",
            file=sys.stderr,
        )
        print(f"    Details: {e}", file=sys.stderr)
        return 1

    result = incremental_update(out_path, validated)
    print(f"  Done. Stored {len(result)} rows through {result.index.max().date()}.")
    return 0


def _ingest_single_equity(
    symbol: str,
    start: str,
    out_path: Path,
) -> int:
    """Fetch, normalize, and write a single equity asset (no Pandera gate yet — Plan 03)."""
    from volforecast.ingest.base import incremental_update
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

    result = incremental_update(out_path, df)
    print(f"  Done. Stored {len(result)} rows through {result.index.max().date()}.")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    """Execute the ingest subcommand.

    When --symbol is provided: ingest that single asset.
    When --symbol is omitted: load all assets from config/assets.yaml and ingest each.

    Crypto assets use fetch_crypto_ohlcv + crypto_ohlcv_schema validation gate.
    Equity assets use download_equity_ohlcv (equity schema gate is added in Plan 03).

    Cache-first incremental: resume_since_ms computes the start of each crypto fetch
    from the last stored date in the existing parquet, falling back to --start only when
    the parquet does not exist.
    """
    from volforecast.config import load_assets, raw_path, symbol_slug
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
        quarantine_path = (
            project_root / "data" / "quarantine" / f"{slug}_quarantine.csv"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n[{asset_class.upper()}] {symbol} -> {out_path.relative_to(project_root)}")

        if asset_class == "crypto":
            # Cache-first: resume from last stored date if parquet exists
            since_ms = resume_since_ms(out_path, default_since_ms)
            rc = _ingest_single_crypto(symbol, since_ms, exchange_id, out_path, quarantine_path)
        elif asset_class == "equity":
            rc = _ingest_single_equity(symbol, args.start, out_path)
        else:
            print(f"  WARNING: Unknown asset_class '{asset_class}' for {symbol} — skipping.")
            rc = 0

        if rc != 0:
            errors += 1

    if errors:
        print(f"\n{errors} asset(s) failed. Check stderr for details.", file=sys.stderr)
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
