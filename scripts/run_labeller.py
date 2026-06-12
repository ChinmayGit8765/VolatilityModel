"""Run the feedback-loop labeller against the data/ directory.

Reads:
    data/predictions/predictions.parquet   (prediction log from FastAPI)
    data/processed/{asset_class}/{slug}.parquet  (per-asset processed close data)

Writes (idempotent append):
    data/monitoring/forecast_vs_realized.parquet

Override paths with environment variables:
    VOLFORECAST_ROOT  — project root (default: cwd)

Usage::

    uv run python scripts/run_labeller.py

This script is the thin entry point the Prefect daily flow (Plan 05) calls.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow import from src/ when run directly (development / CLI usage)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from volforecast.config import project_root
from volforecast.monitoring.labeller import label_forecasts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)


def main() -> None:
    root = project_root()
    data_root = root / "data"
    fvr_path = data_root / "monitoring" / "forecast_vs_realized.parquet"

    log.info("Labeller starting — data_root=%s, fvr_path=%s", data_root, fvr_path)

    try:
        net_new = label_forecasts(data_root, fvr_path)
    except FileNotFoundError as exc:
        log.error("Labeller aborted: %s", exc)
        sys.exit(1)

    log.info("Labeller complete — %d net-new rows written to %s", net_new, fvr_path)
    print(f"net_new_rows={net_new}")


if __name__ == "__main__":
    main()
