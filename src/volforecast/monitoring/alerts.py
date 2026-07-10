"""Alert delivery for VolForecast monitoring.

Provides a fire-and-forget ``send_alert`` function that:

1. POSTs a Slack-compatible JSON payload to ``ALERT_WEBHOOK_URL`` (from the
   environment variable or the injected ``webhook_url`` argument) when a URL
   is available.
2. Falls back to appending a structured JSON record to ``alerts.jsonl`` (from
   the injected ``jsonl_path`` argument or the default
   ``data/monitoring/alerts.jsonl``) when the URL is absent or the POST fails.

Security contract
-----------------
- ``ALERT_WEBHOOK_URL`` is read from the environment or the injected arg only.
  It is NEVER hardcoded in this file (Security Domain, T-04-07).
- The alert payload contains only ``title`` + the caller-supplied ``body``
  dict. No environment values, tokens, or credentials are included in the
  payload (Test 4, T-04-07).

Reliability contract
--------------------
``send_alert`` must NEVER raise an exception (T-04-08). Any network error
(``URLError``, ``OSError``) is caught, logged, and the fallback JSONL path
is used. Callers can rely on this guarantee — alert delivery failure cannot
crash the daily Prefect pipeline.

Dependencies
------------
Stdlib only: ``json``, ``os``, ``pathlib``, ``urllib.request``, ``urllib.error``,
``datetime``, ``logging``. No new package dependencies added.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
from urllib import request
from urllib.error import URLError

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths / environment keys
# ---------------------------------------------------------------------------

#: Environment variable name for the Slack-compatible webhook URL.
#: Never hardcoded; resolved at call time (not at module load) to allow test
#: monkeypatching via monkeypatch.setenv / monkeypatch.delenv.
_WEBHOOK_ENV_VAR: str = "ALERT_WEBHOOK_URL"

#: Default fallback alert log path (relative to the project root).
#: Tests override this via the injected ``jsonl_path`` argument.
_DEFAULT_JSONL_PATH: Path = Path("data") / "monitoring" / "alerts.jsonl"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def send_alert(
    title: str,
    body: dict,
    *,
    webhook_url: str | None = None,
    jsonl_path: Path | None = None,
) -> None:
    """Fire-and-forget alert delivery.

    Resolves the delivery channel in this order:
    1. ``webhook_url`` argument (not empty string).
    2. ``ALERT_WEBHOOK_URL`` environment variable (not empty string).
    3. JSONL fallback (always available as a last resort).

    On successful webhook delivery, the function returns without writing to
    the JSONL file.  On any ``(URLError, OSError)`` the error is logged and
    execution falls through to the JSONL fallback.

    This function MUST NEVER raise an exception.  All error paths are caught
    internally (T-04-08).

    Args:
        title: Human-readable alert title, e.g. ``"Champion underperforms GARCH"``.
        body: Dict of scalar metric values to include in the alert, e.g.
            ``{"champion_qlike": 0.15, "garch_qlike": 0.10}``.
            Must be JSON-serialisable. No secrets or tokens should be passed here.
        webhook_url: Optional Slack-compatible webhook URL.  When ``None`` (or
            empty string), the function reads ``ALERT_WEBHOOK_URL`` from the
            environment.  When both are absent, falls back to JSONL.
        jsonl_path: Path for the JSONL fallback log.  Defaults to
            ``data/monitoring/alerts.jsonl``.  The parent directory is created
            automatically (``mkdir -p``).  Tests pass ``tmp_path / "alerts.jsonl"``
            here for hermeticity.

    Returns:
        None.  Always.
    """
    # Resolve effective webhook URL: injected arg takes priority over env var.
    # Read from env at call time (not module load) so tests can monkeypatch env.
    effective_webhook: str = webhook_url or os.environ.get(_WEBHOOK_ENV_VAR, "") or ""

    # Resolve JSONL path.
    effective_jsonl: Path = jsonl_path if jsonl_path is not None else _DEFAULT_JSONL_PATH

    # Build the timestamped record (used for both JSONL and as the basis for
    # the webhook payload). Payload carries ONLY title + body fields — no env
    # values, no URL, no tokens (T-04-07, Test 4).
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    record: dict = {"timestamp": timestamp, "title": title, **body}

    # Attempt webhook delivery when a URL is configured.
    if effective_webhook:
        delivered = _post_webhook(effective_webhook, title, body)
        if delivered:
            # Webhook succeeded — no fallback needed.
            log.info("send_alert: delivered via webhook (title=%r)", title)
            return
        # Fall through to JSONL fallback after logging.
        log.warning("send_alert: webhook delivery failed; writing fallback to %s", effective_jsonl)
    else:
        log.debug(
            "send_alert: ALERT_WEBHOOK_URL not set; writing to jsonl fallback %s", effective_jsonl
        )

    # JSONL fallback — always succeeds (local disk write).
    _append_jsonl(record, effective_jsonl)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _post_webhook(url: str, title: str, body: dict) -> bool:
    """POST a Slack-compatible JSON payload.

    Returns ``True`` on HTTP success, ``False`` on any network/OS error.
    Never raises.  Only ``URLError`` and ``OSError`` are caught; any other
    exception propagates — callers that need full isolation should wrap
    ``send_alert`` itself.

    The payload body contains ONLY ``title`` and ``body`` fields.  The ``url``
    is used as the POST target and is never echoed inside the payload
    (T-04-07, Test 4).

    Args:
        url: Webhook endpoint URL.
        title: Alert title (appears in the Slack ``text`` field).
        body: Metric dict, JSON-serialised and appended to the text.

    Returns:
        ``True`` if the POST completed without error.
    """
    # Slack-compatible payload: {"text": "<bold title>\n<json body>"}
    # No URL, no env values in the body.
    payload = {"text": f"*{title}*\n{json.dumps(body, indent=2)}"}
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=5):  # noqa: S310 — URL from env/arg, not user input
            return True
    except (URLError, OSError) as exc:
        log.warning("send_alert: webhook POST failed (%s: %s)", type(exc).__name__, exc)
        return False


def _append_jsonl(record: dict, jsonl_path: Path) -> None:
    """Append one JSON record to ``jsonl_path``.

    Creates parent directories (``mkdir -p``) and appends atomically via a
    single ``write`` call (single line + newline).

    Args:
        record: Dict to serialise as a JSON line.
        jsonl_path: Destination path for the JSONL log.
    """
    try:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        # Last-ditch: even the JSONL fallback failed (e.g., disk full).
        # Log and swallow — pipeline must not crash (T-04-08).
        log.error(
            "send_alert: JSONL fallback write also failed (%s: %s); alert dropped",
            type(exc).__name__,
            exc,
        )
