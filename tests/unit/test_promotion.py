"""Unit tests for the champion/challenger promotion gate (ORCH-03).

All tests are hermetic and offline — NO live MLflow registry or network I/O.
The MLflow client is replaced with a stub that records all calls.

Five behaviour tests:
  1. No-promote default: challenger QLIKE >= champion QLIKE → no alias flip
  2. Promote on strict win: challenger < champion AND cooldown elapsed → flip once + tag
  3. Cooldown blocks: < 7 days since last promotion → no flip even when challenger wins
  4. Identical-rows guard: champion and challenger QLIKE scored on same frozen-window rows
  5. Rollback: rollback_champion(to_version) calls set_registered_model_alias exactly once
"""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np
import pandas as pd

from volforecast.monitoring.promotion import (
    MODEL_NAME,
    PROMOTION_COOLDOWN_DAYS,
    frozen_window_qlike,
    promote_if_better,
    rollback_champion,
    select_frozen_window,
)

# ---------------------------------------------------------------------------
# MLflow client stub (offline, no network)
# ---------------------------------------------------------------------------


class _FakeModelVersion:
    """Minimal stub for MlflowClient.get_model_version_by_alias return value."""

    def __init__(self, version: str) -> None:
        self.version = version


class FakeMlflowClient:
    """Stub MlflowClient that records set_registered_model_alias /
    get_model_version_by_alias / set_model_version_tag calls.

    champion_version: str — the version returned by get_model_version_by_alias.
    """

    def __init__(self, champion_version: str = "3") -> None:
        self._champion_version = champion_version
        self.alias_flips: list[tuple[str, str, Any]] = []  # (model_name, alias, version)
        self.tags_set: list[tuple[str, str, str, str]] = []  # (model_name, version, key, value)

    def get_model_version_by_alias(self, name: str, alias: str) -> _FakeModelVersion:
        return _FakeModelVersion(self._champion_version)

    def set_registered_model_alias(self, name: str, alias: str, version: Any) -> None:
        self.alias_flips.append((name, alias, version))

    def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        self.tags_set.append((name, version, key, value))


# ---------------------------------------------------------------------------
# Helpers: synthetic FVR DataFrames for frozen-window QLIKE tests
# ---------------------------------------------------------------------------


def _make_fvr(
    n: int = 10,
    champion_qlike_multiplier: float = 1.0,
    window_start: str = "2026-01-01",
    window_end: str = "2026-01-31",
    assets: list[str] | None = None,
) -> pd.DataFrame:
    """Build a synthetic forecast_vs_realized DataFrame.

    champion_qlike_multiplier controls how much worse the champion is
    relative to perfect forecasts:
      - 1.0 → champion uses perfect forecast (qlike ≈ 0)
      - >1  → champion forecast biased higher (worse QLIKE than challenger)

    The challenger always uses a perfect forecast (realized_var == forecast_var,
    QLIKE ≈ 0), making it strictly better when multiplier > 1.
    """
    if assets is None:
        assets = ["BTC-USD"]

    rng = np.random.default_rng(42)
    dates = pd.date_range(window_start, window_end, freq="D")[:n]
    rows = []
    for date in dates:
        for asset in assets:
            rv = float(rng.uniform(1e-4, 5e-4))
            # Challenger: perfect forecast
            rows.append(
                {
                    "as_of_date": date,
                    "asset": asset,
                    "horizon": 1,
                    "model_version": "99",
                    "model_alias": "challenger",
                    "forecast_var": rv,  # perfect
                    "realized_var": rv,
                }
            )
            # Champion: forecast biased by multiplier
            rows.append(
                {
                    "as_of_date": date,
                    "asset": asset,
                    "horizon": 1,
                    "model_version": "3",
                    "model_alias": "champion",
                    "forecast_var": rv * champion_qlike_multiplier,
                    "realized_var": rv,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: No-promote default (challenger QLIKE >= champion QLIKE)
# ---------------------------------------------------------------------------


class TestNoPromoteDefault:
    """Gate returns promoted=False when challenger does NOT strictly beat champion."""

    def test_no_promote_when_challenger_ties(self) -> None:
        """When challenger and champion have identical QLIKE, no alias flip."""
        # Both use perfect forecasts → QLIKE = 0 for both → challenger == champion
        fvr = _make_fvr(n=10, champion_qlike_multiplier=1.0)
        client = FakeMlflowClient(champion_version="3")

        today = datetime.date(2026, 2, 1)
        last_promo = datetime.date(2026, 1, 1)  # 31 days ago — cooldown not a factor

        champ_q, _ = frozen_window_qlike(fvr, "champion", "2026-01-01", "2026-01-31")
        chall_q, _ = frozen_window_qlike(fvr, "challenger", "2026-01-01", "2026-01-31")

        promoted = promote_if_better(
            challenger_qlike=chall_q,
            champion_qlike=champ_q,
            last_promotion_date=last_promo,
            today=today,
            challenger_version=99,
            client=client,
        )

        assert promoted is False
        assert client.alias_flips == [], "set_registered_model_alias must NOT be called on tie"

    def test_no_promote_when_challenger_loses(self) -> None:
        """When challenger is worse (higher QLIKE), no alias flip."""
        rng = np.random.default_rng(7)
        dates = pd.date_range("2026-01-01", periods=10, freq="D")
        rows = []
        for date in dates:
            rv = float(rng.uniform(1e-4, 5e-4))
            rows.append(
                {
                    "as_of_date": date,
                    "asset": "BTC-USD",
                    "horizon": 1,
                    "model_version": "99",
                    "model_alias": "challenger",
                    "forecast_var": rv * 2.0,  # worse
                    "realized_var": rv,
                }
            )
            rows.append(
                {
                    "as_of_date": date,
                    "asset": "BTC-USD",
                    "horizon": 1,
                    "model_version": "3",
                    "model_alias": "champion",
                    "forecast_var": rv,  # perfect
                    "realized_var": rv,
                }
            )
        fvr_reverse = pd.DataFrame(rows)
        client = FakeMlflowClient(champion_version="3")

        champ_q, _ = frozen_window_qlike(fvr_reverse, "champion", "2026-01-01", "2026-01-31")
        chall_q, _ = frozen_window_qlike(fvr_reverse, "challenger", "2026-01-01", "2026-01-31")

        assert chall_q > champ_q, "pre-condition: challenger should be worse"

        promoted = promote_if_better(
            challenger_qlike=chall_q,
            champion_qlike=champ_q,
            last_promotion_date=datetime.date(2026, 1, 1),
            today=datetime.date(2026, 2, 1),
            challenger_version=99,
            client=client,
        )

        assert promoted is False
        assert client.alias_flips == []


# ---------------------------------------------------------------------------
# Test 2: Promote on strict win (challenger QLIKE < champion AND cooldown OK)
# ---------------------------------------------------------------------------


class TestPromoteOnStrictWin:
    """Gate promotes challenger when it strictly beats champion AND cooldown elapsed."""

    def test_promotes_and_flips_alias_once(self) -> None:
        """Alias flip called exactly once; previous_champion_version tag recorded."""
        # challenger perfect (QLIKE≈0); champion biased (higher QLIKE)
        fvr = _make_fvr(n=10, champion_qlike_multiplier=2.0)
        client = FakeMlflowClient(champion_version="3")

        champ_q, _ = frozen_window_qlike(fvr, "champion", "2026-01-01", "2026-01-31")
        chall_q, _ = frozen_window_qlike(fvr, "challenger", "2026-01-01", "2026-01-31")

        assert chall_q < champ_q, "pre-condition: challenger must be better"

        promoted = promote_if_better(
            challenger_qlike=chall_q,
            champion_qlike=champ_q,
            last_promotion_date=datetime.date(2026, 1, 1),  # 31 days ago
            today=datetime.date(2026, 2, 1),
            challenger_version=99,
            client=client,
        )

        assert promoted is True

        # Exactly one alias flip to "champion" for version 99
        champion_flips = [(n, a, v) for n, a, v in client.alias_flips if a == "champion"]
        assert len(champion_flips) == 1, f"Expected 1 champion alias flip, got {champion_flips}"
        _, _, flipped_version = champion_flips[0]
        assert flipped_version == 99

    def test_previous_champion_version_tag_recorded(self) -> None:
        """previous_champion_version tag is set on the new champion version."""
        fvr = _make_fvr(n=10, champion_qlike_multiplier=2.0)
        client = FakeMlflowClient(champion_version="3")

        champ_q, _ = frozen_window_qlike(fvr, "champion", "2026-01-01", "2026-01-31")
        chall_q, _ = frozen_window_qlike(fvr, "challenger", "2026-01-01", "2026-01-31")

        promote_if_better(
            challenger_qlike=chall_q,
            champion_qlike=champ_q,
            last_promotion_date=datetime.date(2026, 1, 1),
            today=datetime.date(2026, 2, 1),
            challenger_version=99,
            client=client,
        )

        # Check that previous_champion_version tag was set on the new version (99)
        prev_tags = [
            (n, ver, k, v) for n, ver, k, v in client.tags_set if k == "previous_champion_version"
        ]
        assert len(prev_tags) == 1, f"Expected 1 previous_champion_version tag, got {prev_tags}"
        _, tagged_version, _, prev_version_value = prev_tags[0]
        assert tagged_version == str(99)
        assert prev_version_value == "3"  # the old champion

    def test_no_previous_champion_on_first_promotion(self) -> None:
        """Promotion succeeds even when get_model_version_by_alias raises (no existing champion)."""

        class NoChampionClient(FakeMlflowClient):
            def get_model_version_by_alias(self, name: str, alias: str) -> _FakeModelVersion:
                raise Exception("No champion version found")  # simulate missing alias

        fvr = _make_fvr(n=10, champion_qlike_multiplier=2.0)
        client = NoChampionClient(champion_version="3")

        champ_q, _ = frozen_window_qlike(fvr, "champion", "2026-01-01", "2026-01-31")
        chall_q, _ = frozen_window_qlike(fvr, "challenger", "2026-01-01", "2026-01-31")

        promoted = promote_if_better(
            challenger_qlike=chall_q,
            champion_qlike=champ_q,
            last_promotion_date=None,  # first ever promotion
            today=datetime.date(2026, 2, 1),
            challenger_version=99,
            client=client,
        )

        assert promoted is True
        champion_flips = [f for f in client.alias_flips if f[1] == "champion"]
        assert len(champion_flips) == 1


# ---------------------------------------------------------------------------
# Test 3: Cooldown blocks promotion
# ---------------------------------------------------------------------------


class TestCooldownBlocks:
    """Gate returns promoted=False when < PROMOTION_COOLDOWN_DAYS have elapsed."""

    def test_cooldown_blocks_even_when_challenger_wins(self) -> None:
        """Challenger wins on QLIKE but was promoted < 7 days ago — no flip."""
        fvr = _make_fvr(n=10, champion_qlike_multiplier=2.0)
        client = FakeMlflowClient(champion_version="3")

        champ_q, _ = frozen_window_qlike(fvr, "champion", "2026-01-01", "2026-01-31")
        chall_q, _ = frozen_window_qlike(fvr, "challenger", "2026-01-01", "2026-01-31")
        assert chall_q < champ_q, "pre-condition"

        today = datetime.date(2026, 2, 1)
        # Last promotion was 3 days ago — inside cooldown window
        last_promo = today - datetime.timedelta(days=3)

        promoted = promote_if_better(
            challenger_qlike=chall_q,
            champion_qlike=champ_q,
            last_promotion_date=last_promo,
            today=today,
            challenger_version=99,
            client=client,
        )

        assert promoted is False
        assert client.alias_flips == [], "Cooldown must prevent alias flip"

    def test_cooldown_exactly_at_boundary_blocks(self) -> None:
        """Exactly PROMOTION_COOLDOWN_DAYS-1 days → still blocked."""
        fvr = _make_fvr(n=10, champion_qlike_multiplier=2.0)
        client = FakeMlflowClient(champion_version="3")

        champ_q, _ = frozen_window_qlike(fvr, "champion", "2026-01-01", "2026-01-31")
        chall_q, _ = frozen_window_qlike(fvr, "challenger", "2026-01-01", "2026-01-31")

        today = datetime.date(2026, 2, 1)
        last_promo = today - datetime.timedelta(days=PROMOTION_COOLDOWN_DAYS - 1)

        promoted = promote_if_better(
            challenger_qlike=chall_q,
            champion_qlike=champ_q,
            last_promotion_date=last_promo,
            today=today,
            challenger_version=99,
            client=client,
        )

        assert promoted is False

    def test_cooldown_exactly_elapsed_allows_promotion(self) -> None:
        """Exactly PROMOTION_COOLDOWN_DAYS days ago → promotion allowed."""
        fvr = _make_fvr(n=10, champion_qlike_multiplier=2.0)
        client = FakeMlflowClient(champion_version="3")

        champ_q, _ = frozen_window_qlike(fvr, "champion", "2026-01-01", "2026-01-31")
        chall_q, _ = frozen_window_qlike(fvr, "challenger", "2026-01-01", "2026-01-31")

        today = datetime.date(2026, 2, 1)
        last_promo = today - datetime.timedelta(days=PROMOTION_COOLDOWN_DAYS)

        promoted = promote_if_better(
            challenger_qlike=chall_q,
            champion_qlike=champ_q,
            last_promotion_date=last_promo,
            today=today,
            challenger_version=99,
            client=client,
        )

        assert promoted is True


# ---------------------------------------------------------------------------
# Test 4: Identical-rows guard (frozen window selects same rows for both)
# ---------------------------------------------------------------------------


class TestIdenticalRowsGuard:
    """Champion and challenger QLIKE are computed over the SAME frozen-window rows."""

    def test_same_row_count_and_key_set(self) -> None:
        """frozen_window_qlike for champion and challenger operate on identical row sets."""
        assets = ["BTC-USD", "ETH-USD"]
        fvr = _make_fvr(n=10, champion_qlike_multiplier=2.0, assets=assets)

        window_start = "2026-01-01"
        window_end = "2026-01-31"

        # Both calls should be computed on the same (asset, as_of_date) key set
        _, champ_prov = frozen_window_qlike(fvr, "champion", window_start, window_end)
        _, chall_prov = frozen_window_qlike(fvr, "challenger", window_start, window_end)

        # Row counts must match (same window, same assets)
        assert champ_prov["n_rows"] == chall_prov["n_rows"], (
            f"Row counts differ: champion={champ_prov['n_rows']}, challenger={chall_prov['n_rows']}"
        )

        # Window bounds must be the same (both derived from the same frozen selection)
        assert champ_prov["window_start"] == chall_prov["window_start"]
        assert champ_prov["window_end"] == chall_prov["window_end"]

    def test_frozen_window_excludes_out_of_window_rows(self) -> None:
        """Rows outside [window_start, window_end] must not affect QLIKE."""
        rng = np.random.default_rng(10)
        all_dates = pd.date_range("2025-12-01", periods=60, freq="D")
        rows = []
        for date in all_dates:
            rv = float(rng.uniform(1e-4, 5e-4))
            for alias, version, multiplier in [("champion", "3", 2.0), ("challenger", "99", 1.0)]:
                rows.append(
                    {
                        "as_of_date": date,
                        "asset": "BTC-USD",
                        "horizon": 1,
                        "model_version": version,
                        "model_alias": alias,
                        "forecast_var": rv * multiplier,
                        "realized_var": rv,
                    }
                )
        fvr_wide = pd.DataFrame(rows)

        window_start = "2026-01-01"
        window_end = "2026-01-31"

        _, champ_prov = frozen_window_qlike(fvr_wide, "champion", window_start, window_end)
        _, chall_prov = frozen_window_qlike(fvr_wide, "challenger", window_start, window_end)

        # Both must be restricted to the window
        assert champ_prov["window_start"] == window_start
        assert champ_prov["window_end"] == window_end
        # Row count must match the filtered window (not the wider frame)
        frozen = select_frozen_window(fvr_wide, window_start, window_end)
        expected_n_per_alias = len(frozen[frozen["model_alias"] == "champion"])
        assert champ_prov["n_rows"] == expected_n_per_alias

    def test_single_select_frozen_window_source(self) -> None:
        """select_frozen_window filters on as_of_date range only (no alias filter).

        Both champion and challenger rows for the window must be present,
        confirming that the function is shared (not alias-specific).
        """
        fvr = _make_fvr(n=10, champion_qlike_multiplier=2.0)
        window_start = "2026-01-01"
        window_end = "2026-01-31"

        frozen = select_frozen_window(fvr, window_start, window_end)

        aliases_in_frozen = set(frozen["model_alias"].unique())
        assert "champion" in aliases_in_frozen, "champion rows must be in frozen window"
        assert "challenger" in aliases_in_frozen, "challenger rows must be in frozen window"


# ---------------------------------------------------------------------------
# Test 5: Rollback
# ---------------------------------------------------------------------------


class TestRollback:
    """rollback_champion(to_version) calls set_registered_model_alias exactly once."""

    def test_rollback_calls_alias_flip_once(self) -> None:
        """rollback_champion flips @champion to the given version exactly once."""
        client = FakeMlflowClient(champion_version="3")

        rollback_champion(to_version=3, client=client)

        assert len(client.alias_flips) == 1, (
            f"Expected exactly 1 alias flip, got {client.alias_flips}"
        )
        name, alias, version = client.alias_flips[0]
        assert name == MODEL_NAME
        assert alias == "champion"
        assert version == 3

    def test_rollback_does_not_set_any_tags(self) -> None:
        """rollback_champion must not set any model version tags (it's a pure flip)."""
        client = FakeMlflowClient(champion_version="3")
        rollback_champion(to_version=5, client=client)

        assert client.tags_set == [], "rollback must not set any version tags"

    def test_rollback_uses_model_name_constant(self) -> None:
        """rollback_champion must use the MODEL_NAME constant, not a hardcoded string."""
        client = FakeMlflowClient(champion_version="3")
        rollback_champion(to_version=2, client=client)

        name, _, _ = client.alias_flips[0]
        assert name == MODEL_NAME, f"Expected MODEL_NAME={MODEL_NAME!r}, got {name!r}"


# ---------------------------------------------------------------------------
# Constant guard tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify locked constants defined in the module."""

    def test_model_name_constant(self) -> None:
        assert MODEL_NAME == "volforecast-lgbm"

    def test_cooldown_days_constant(self) -> None:
        assert PROMOTION_COOLDOWN_DAYS == 7
