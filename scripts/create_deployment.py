"""Register the VolForecast daily Prefect deployment against the compose Prefect server.

This script is a one-time deployment registration step that runs from the HOST shell
(not inside a container).  After registration the compose Prefect worker polls
``local-pool`` for scheduled or manually-triggered runs.

Host run command
----------------
::

    PREFECT_API_URL=http://localhost:4200/api uv run python scripts/create_deployment.py

Prerequisites
-------------
- The compose stack must be running: ``docker compose -f infra/docker-compose.yml up -d``
- The Prefect server UI should be accessible at http://localhost:4200
- The prefect-worker container must be up with ``--pool local-pool --type process``

Deployment details
------------------
- Name:          volforecast-daily
- Flow:          daily_flow (pipelines/daily_pipeline.py)
- Work pool:     local-pool  (must match compose worker ``--pool`` arg — Pitfall 4)
- Cron:          0 8 * * *  (08:00 UTC daily)

Cron timing rationale (Claude's discretion per 04-RESEARCH.md)
--------------------------------------------------------------
``0 8 * * *`` = 08:00 UTC daily gives:
  - ~8 hours after Binance daily close (00:00 UTC), so yesterday's crypto close data
    is available for labelling and drift checks.
  - Before major equity opens (NYSE/NASDAQ: 13:30 UTC), so equity data from the
    previous session is available.
  - Plenty of time for the full ingest→features→label→monitor→(retrain) pipeline to
    complete before the trading day begins.

Source resolution
-----------------
``flow.from_source(source=str(repo_root))`` records a LOCAL directory path.  This
works because the compose prefect-worker mounts the same repo directory into the
container at the same path (see infra/docker-compose.yml prefect-worker volumes).
When the worker picks up a scheduled run it resolves the entrypoint from the mounted
local filesystem, not from a git remote.

If the worker cannot find the code, the flow will appear as "Scheduled" indefinitely
(Pitfall 4 symptom).  Verify the worker volume mount in docker-compose.yml.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: allow running from project root or scripts/ directory
# ---------------------------------------------------------------------------
# Add src/ to sys.path so volforecast imports resolve when running from host
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
# Add the repo root so `pipelines.daily_pipeline` resolves
sys.path.insert(0, str(_REPO_ROOT))

from pipelines.daily_pipeline import daily_flow  # noqa: E402

# ---------------------------------------------------------------------------
# Deployment registration
# ---------------------------------------------------------------------------


def main() -> None:
    """Register the volforecast-daily deployment against the compose Prefect server."""
    # PREFECT_API_URL must point to the Prefect server — for the compose stack running
    # locally this is http://localhost:4200/api (Pitfall 4: worker uses the internal
    # docker-network URL http://prefect-server:4200/api — the HOST must use localhost).
    api_url = os.environ.get("PREFECT_API_URL", "http://localhost:4200/api")
    print(f"Registering deployment against Prefect API: {api_url}")
    print(f"Source repo root: {_REPO_ROOT}")

    daily_flow.from_source(
        # Local directory path on the volume that the worker can see.
        # The compose prefect-worker mounts the repo root at /repo (or the Windows
        # host path) so this path is resolvable inside the container.
        source=str(_REPO_ROOT),
        entrypoint="pipelines/daily_pipeline.py:daily_flow",
    ).deploy(
        name="volforecast-daily",
        # Must EXACTLY match the compose worker's --pool argument (Pitfall 4)
        work_pool_name="local-pool",
        # 08:00 UTC daily — after crypto close, before equity opens (see module docstring)
        cron="0 8 * * *",
    )

    print("\nDeployment registered successfully.")
    print("  Name:      volforecast-daily")
    print("  Pool:      local-pool")
    print("  Schedule:  0 8 * * *  (08:00 UTC daily)")
    print("\nVerify in the Prefect UI: http://localhost:4200 -> Deployments")
    print(
        "\nTrigger a manual run:"
        "\n  PREFECT_API_URL=http://localhost:4200/api "
        "uv run prefect deployment run 'volforecast-daily/volforecast-daily'"
    )


if __name__ == "__main__":
    main()
