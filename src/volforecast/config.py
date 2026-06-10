"""Asset universe configuration and path helpers.

Loads config/assets.yaml and provides helpers for symbol normalization and
canonical data file paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Default config file location (relative to the project root)
_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
_ASSETS_YAML = _CONFIG_DIR / "assets.yaml"

# Default data root (relative to project root)
_DATA_ROOT = Path(__file__).parent.parent.parent / "data"


def load_assets(config_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load the asset universe from config/assets.yaml.

    Args:
        config_path: Path to the assets YAML file. Defaults to config/assets.yaml
                     relative to the package root.

    Returns:
        List of asset dicts, each with keys: symbol, asset_class, exchange.
    """
    path = Path(config_path) if config_path else _ASSETS_YAML
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("assets", [])


def symbol_slug(symbol: str) -> str:
    """Normalize a trading symbol to a filesystem-safe slug.

    Converts exchange-style symbols to a canonical filename stem:
    - "BTC/USDT" -> "BTC-USD"
    - "ETH/USDT" -> "ETH-USD"
    - "SPY" -> "SPY"

    Args:
        symbol: Trading symbol, e.g. "BTC/USDT".

    Returns:
        Filesystem-safe slug, e.g. "BTC-USD".
    """
    # Handle crypto pairs: BTC/USDT -> BTC-USD (strip USDT/USDC/BTC quote suffix)
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
        # Normalize stablecoin quotes to USD for consistency
        if quote in ("USDT", "USDC", "BUSD", "USD"):
            return f"{base}-USD"
        return f"{base}-{quote}"
    return symbol


def raw_path(asset: dict[str, Any], data_root: Path | str | None = None) -> Path:
    """Return the canonical raw parquet path for an asset.

    Args:
        asset: Asset dict with keys: symbol, asset_class.
        data_root: Root data directory. Defaults to data/ relative to project root.

    Returns:
        Path like data/raw/{asset_class}/{slug}.parquet
    """
    root = Path(data_root) if data_root else _DATA_ROOT
    slug = symbol_slug(asset["symbol"])
    return root / "raw" / asset["asset_class"] / f"{slug}.parquet"
