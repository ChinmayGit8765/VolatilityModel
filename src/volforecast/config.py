"""Asset universe configuration and path helpers.

Loads config/assets.yaml and provides helpers for symbol normalization and
canonical data file paths.

Path resolution: config (config/assets.yaml) and data (data/) are BOTH resolved
relative to a single project root — ``project_root()`` — so the CLI never reads
config from one directory tree while writing data into another.  The root is
the VOLFORECAST_ROOT environment variable when set, otherwise the current
working directory.  (Resolving relative to ``__file__`` would point into
site-packages for a built wheel.)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the single root directory for config and data resolution.

    Resolution order:
    1. The ``VOLFORECAST_ROOT`` environment variable, if set.
    2. The current working directory.

    Both ``config/assets.yaml`` and ``data/`` are resolved relative to this
    root, so running the console script from any directory keeps reads and
    writes in the same tree (set VOLFORECAST_ROOT to pin the project location).
    """
    env_root = os.environ.get("VOLFORECAST_ROOT")
    return Path(env_root) if env_root else Path.cwd()


def load_assets(config_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load the asset universe from config/assets.yaml.

    Args:
        config_path: Path to the assets YAML file. Defaults to
                     ``{project_root()}/config/assets.yaml``.

    Returns:
        List of asset dicts, each with keys: symbol, asset_class, exchange.

    Raises:
        FileNotFoundError: If the config file does not exist, with a message
            pointing at the expected location.
    """
    path = Path(config_path) if config_path else project_root() / "config" / "assets.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Asset config not found at {path}. Run from the project root, set the "
            "VOLFORECAST_ROOT environment variable, or pass config_path explicitly."
        )
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
        data_root: Root data directory. Defaults to ``{project_root()}/data``.

    Returns:
        Path like data/raw/{asset_class}/{slug}.parquet
    """
    root = Path(data_root) if data_root else project_root() / "data"
    slug = symbol_slug(asset["symbol"])
    return root / "raw" / asset["asset_class"] / f"{slug}.parquet"


def processed_path(asset: dict[str, Any], data_root: Path | str | None = None) -> Path:
    """Return the canonical processed parquet path for an asset.

    Processed data is written only after the asset has passed the validate_asset gate.

    Args:
        asset: Asset dict with keys: symbol, asset_class.
        data_root: Root data directory. Defaults to ``{project_root()}/data``.

    Returns:
        Path like data/processed/{asset_class}/{slug}.parquet
    """
    root = Path(data_root) if data_root else project_root() / "data"
    slug = symbol_slug(asset["symbol"])
    return root / "processed" / asset["asset_class"] / f"{slug}.parquet"
