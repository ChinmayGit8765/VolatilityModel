"""Offline tests for the alert delivery module.

All tests are hermetic: no real webhook calls, no real filesystem state.
The ``tmp_path`` fixture supplies a clean temp directory for each test.
Monkeypatching ``urllib.request.urlopen`` simulates webhook success / failure.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError

import pytest


# ---------------------------------------------------------------------------
# Helper: read a jsonl file into a list of dicts
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# Test 1: webhook success — record delivered, NO jsonl fallback written
# ---------------------------------------------------------------------------


class TestWebhookSuccess:
    """Test 1: ALERT_WEBHOOK_URL set and POST succeeds."""

    def test_webhook_success_no_jsonl(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When the webhook POST succeeds, send_alert returns and does not write jsonl."""
        import urllib.request

        calls: list[str] = []

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        def fake_urlopen(req, timeout=None):  # noqa: ANN001, ARG001
            calls.append("opened")
            return _FakeResponse()

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        from volforecast.monitoring.alerts import send_alert

        jsonl_path = tmp_path / "alerts.jsonl"
        send_alert(
            title="Test Alert",
            body={"metric": "qlike", "value": 0.05},
            webhook_url="https://hooks.example.com/test",
            jsonl_path=jsonl_path,
        )

        # Webhook was called
        assert calls == ["opened"], "urlopen must have been called exactly once"
        # No jsonl fallback file was written
        assert not jsonl_path.exists(), "alerts.jsonl must NOT be written on webhook success"


# ---------------------------------------------------------------------------
# Test 2: webhook failure → JSONL fallback, no raise
# ---------------------------------------------------------------------------


class TestWebhookFailureFallback:
    """Test 2: ALERT_WEBHOOK_URL set but POST raises URLError → fallback to jsonl."""

    def test_webhook_failure_falls_back_to_jsonl(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When the POST raises URLError, send_alert swallows and appends to jsonl."""
        import urllib.request

        def fail_urlopen(req, timeout=None):  # noqa: ANN001, ARG001
            raise URLError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)

        from volforecast.monitoring.alerts import send_alert

        jsonl_path = tmp_path / "subdir" / "alerts.jsonl"
        send_alert(
            title="Degradation Detected",
            body={"champion_qlike": 0.12, "garch_qlike": 0.08},
            webhook_url="https://hooks.example.com/test",
            jsonl_path=jsonl_path,
        )

        # Must not raise — function completes normally
        # Must write exactly one jsonl record
        assert jsonl_path.exists(), "alerts.jsonl must be created as fallback"
        records = _read_jsonl(jsonl_path)
        assert len(records) == 1, f"Expected 1 jsonl record, got {len(records)}"

        # Record must contain title and body fields
        rec = records[0]
        assert rec["title"] == "Degradation Detected"
        assert rec["champion_qlike"] == pytest.approx(0.12)
        assert rec["garch_qlike"] == pytest.approx(0.08)
        assert "timestamp" in rec, "Record must include a timestamp"


# ---------------------------------------------------------------------------
# Test 3: env unset → JSONL fallback
# ---------------------------------------------------------------------------


class TestEnvUnsetFallback:
    """Test 3: ALERT_WEBHOOK_URL not set → fallback to jsonl without trying webhook."""

    def test_env_unset_writes_jsonl(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When ALERT_WEBHOOK_URL is absent from env, send_alert writes to jsonl."""
        monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)

        # Track if urlopen is called (it must NOT be)
        import urllib.request

        opened: list[bool] = []

        def _should_not_open(req, timeout=None):  # noqa: ANN001, ARG001
            opened.append(True)
            raise AssertionError("urlopen must not be called when ALERT_WEBHOOK_URL is unset")

        monkeypatch.setattr(urllib.request, "urlopen", _should_not_open)

        from volforecast.monitoring.alerts import send_alert

        jsonl_path = tmp_path / "alerts.jsonl"
        send_alert(
            title="Cold Start Notice",
            body={"reason": "cold_start", "n_champion": 5},
            webhook_url=None,  # explicitly None — resolver must check env
            jsonl_path=jsonl_path,
        )

        assert not opened, "urlopen must NOT be called when no webhook URL is available"
        assert jsonl_path.exists(), "jsonl fallback must be written"
        records = _read_jsonl(jsonl_path)
        assert len(records) == 1
        assert records[0]["title"] == "Cold Start Notice"
        assert records[0]["reason"] == "cold_start"


# ---------------------------------------------------------------------------
# Test 4: no secret leakage in payload
# ---------------------------------------------------------------------------


class TestNoSecretLeakage:
    """Test 4: alert payload contains only title + metric fields, no env secrets."""

    def test_payload_contains_no_env_values(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The webhook payload must carry only title + body dict, no env var values."""
        import urllib.request

        FAKE_WEBHOOK = "https://hooks.example.com/secret-token-1234"
        FAKE_SECRET = "my-secret-token-xyz"
        monkeypatch.setenv("ALERT_WEBHOOK_URL", FAKE_WEBHOOK)
        monkeypatch.setenv("SOME_SECRET", FAKE_SECRET)

        captured_bodies: list[str] = []

        class _CapturingResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        def capture_urlopen(req, timeout=None):  # noqa: ANN001, ARG001
            captured_bodies.append(req.data.decode("utf-8"))
            return _CapturingResponse()

        monkeypatch.setattr(urllib.request, "urlopen", capture_urlopen)

        from volforecast.monitoring.alerts import send_alert

        send_alert(
            title="Performance Degradation",
            body={"champion_qlike": 0.15, "garch_qlike": 0.10, "threshold": 0.10},
            webhook_url=FAKE_WEBHOOK,
            jsonl_path=tmp_path / "alerts.jsonl",
        )

        assert len(captured_bodies) == 1, "Expected exactly one POST"
        payload_str = captured_bodies[0]

        # Payload must NOT contain the secret env value
        assert FAKE_SECRET not in payload_str, "Payload must not leak SOME_SECRET"
        # Payload must NOT contain the raw webhook URL token
        # (the URL itself is used as the POST target, but must not appear in the body)
        assert "secret-token-1234" not in payload_str, "Payload must not echo the webhook URL"

        # Payload must contain the supplied metrics
        payload = json.loads(payload_str)
        assert "text" in payload, "Slack-compatible payload must have a 'text' key"
        assert "Performance Degradation" in payload["text"], "title must appear in text"
