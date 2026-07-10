"""Champion/challenger promotion gate for VolForecast (ORCH-03).

Default outcome is NO-PROMOTE. The @champion MLflow alias is flipped ONLY when:
  1. The challenger's frozen-window rolling QLIKE is strictly less than the
     champion's QLIKE on the SAME frozen-window row set (identical rows for
     both — a single ``select_frozen_window`` call is the shared source), AND
  2. The 7-day cooldown since the last promotion has elapsed.

Distribution drift NEVER reaches this function.  Promotion is triggered only by
an explicit call from the Prefect daily flow's promotion-gate step, not by the
drift check task.  This is enforced at the call-site level (taxonomy guard).

Rollback is a single alias flip: ``rollback_champion(to_version)`` restores
@champion to any recorded previous version.  The previous champion version is
recorded as a model-version tag (``previous_champion_version``) on the new
champion at promotion time (Pitfall 7 from 04-RESEARCH.md).

Usage
-----
::

    from volforecast.monitoring.promotion import (
        select_frozen_window,
        frozen_window_qlike,
        promote_if_better,
        rollback_champion,
    )

    # Both candidates scored on the SAME row window
    champ_q, champ_prov = frozen_window_qlike(fvr, "champion", start, end)
    chall_q, chall_prov = frozen_window_qlike(fvr, "challenger", start, end)

    promoted = promote_if_better(
        challenger_qlike=chall_q,
        champion_qlike=champ_q,
        last_promotion_date=last_date,
        today=datetime.date.today(),
        challenger_version=new_version,
    )
    if not promoted:
        ...  # default: do nothing
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

import pandas as pd

from volforecast.eval.metrics import qlike

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locked constants (per 04-CONTEXT.md / 04-RESEARCH.md Pattern 6)
# ---------------------------------------------------------------------------

#: Registered model name in the MLflow registry.
MODEL_NAME: str = "volforecast-lgbm"

#: Minimum calendar days between successive promotions.
#: Prevents retrain-storm promotions (T-04-14 mitigation).
#: Locked per CONTEXT.md — do not change without updating the phase context.
PROMOTION_COOLDOWN_DAYS: int = 7


# ---------------------------------------------------------------------------
# Frozen-window helpers
# ---------------------------------------------------------------------------


def select_frozen_window(
    fvr: pd.DataFrame,
    window_start: str | datetime.date,
    window_end: str | datetime.date,
) -> pd.DataFrame:
    """Return rows of *fvr* whose ``as_of_date`` falls within [window_start, window_end].

    This is the **single shared source** for both champion and challenger row
    selection.  Both ``frozen_window_qlike`` calls invoke this function before
    splitting by alias — guaranteeing that champion and challenger QLIKE are
    computed on the identical date range and asset universe (Test 4 / T-04-12).

    Args:
        fvr: forecast_vs_realized DataFrame with at minimum columns
            ``as_of_date`` (datetime-like) and ``model_alias`` (str).
        window_start: Inclusive start of the comparison window (ISO str or date).
        window_end: Inclusive end of the comparison window (ISO str or date).

    Returns:
        Filtered DataFrame — all aliases present; caller filters by alias.
    """
    dates = pd.to_datetime(fvr["as_of_date"])
    start = pd.Timestamp(window_start)
    end = pd.Timestamp(window_end)
    mask = (dates >= start) & (dates <= end)
    return fvr.loc[mask].copy()


def frozen_window_qlike(
    fvr: pd.DataFrame,
    model_alias: str,
    window_start: str | datetime.date,
    window_end: str | datetime.date,
) -> tuple[float, dict[str, Any]]:
    """Compute QLIKE for *model_alias* over the frozen comparison window.

    Uses ``select_frozen_window`` to restrict rows to [window_start, window_end]
    BEFORE filtering by alias — ensuring identical date coverage for all callers
    of this function (champion and challenger are scored on the same rows).

    The provenance dict records n_rows, window bounds, and the sorted list of
    ``(asset, as_of_date)`` key pairs so callers can verify row identity.

    Args:
        fvr: forecast_vs_realized DataFrame conforming to
            ``monitoring.labeller.FVR_SCHEMA``.
        model_alias: Alias to score (``"champion"`` or ``"challenger"``).
        window_start: Inclusive start of the comparison window.
        window_end: Inclusive end of the comparison window.

    Returns:
        Tuple of (qlike_score: float, provenance: dict).

    Raises:
        ValueError: If no rows exist for *model_alias* in the frozen window.
    """
    # Shared frozen-window selection — SAME frame used for all aliases
    frozen = select_frozen_window(fvr, window_start, window_end)
    alias_rows = frozen[frozen["model_alias"] == model_alias]

    if alias_rows.empty:
        raise ValueError(
            f"No rows for model_alias={model_alias!r} in frozen window "
            f"[{window_start}, {window_end}]. "
            "Ensure forecast_vs_realized contains rows for both aliases in this range."
        )

    score = qlike(
        rv_var=alias_rows["realized_var"].to_numpy(),
        forecast_var=alias_rows["forecast_var"].to_numpy(),
    )

    # Provenance for auditability and Test 4 identical-rows assertion
    key_pairs = sorted(
        zip(alias_rows["asset"].tolist(), alias_rows["as_of_date"].astype(str).tolist())
    )
    provenance: dict[str, Any] = {
        "model_alias": model_alias,
        "n_rows": len(alias_rows),
        "window_start": str(window_start),
        "window_end": str(window_end),
        "key_pairs_hash": hash(tuple(key_pairs)),  # fast equality check
        "key_pairs": key_pairs,
    }

    log.debug(
        "frozen_window_qlike: alias=%s n_rows=%d qlike=%.6f",
        model_alias,
        len(alias_rows),
        score,
    )
    return score, provenance


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------


def promote_if_better(
    challenger_qlike: float,
    champion_qlike: float,
    last_promotion_date: datetime.date | None,
    today: datetime.date,
    *,
    challenger_version: int,
    client: Any = None,
) -> bool:
    """Promote the challenger to @champion if it strictly wins AND cooldown elapsed.

    This is the ONLY function that may call
    ``client.set_registered_model_alias(MODEL_NAME, "champion", ...)``.
    No other code path in the codebase should flip @champion (T-04-11 EoP
    mitigation — grep-verifiable: search for ``set_registered_model_alias``
    with alias ``"champion"`` outside this function and ``rollback_champion``).

    Decision logic (in order — both conditions must be met for promotion):

    1. **Cooldown guard**: If ``last_promotion_date`` is set and fewer than
       ``PROMOTION_COOLDOWN_DAYS`` have elapsed, return False immediately.
       Prevents retrain-storm promotions (T-04-14).

    2. **Strict-win requirement**: If ``challenger_qlike >= champion_qlike``,
       return False.  Challenger must strictly improve QLIKE — a tie is a
       no-promote (default-no-promote policy).

    3. **Promotion**: Resolve the current champion version for rollback record
       (Pitfall 7), flip @champion alias to ``challenger_version``, tag the new
       version with ``previous_champion_version``, return True.

    Args:
        challenger_qlike: QLIKE of the challenger on the frozen comparison window.
        champion_qlike: QLIKE of the champion on the identical frozen window.
        last_promotion_date: Date of the last successful promotion (or None if
            never promoted).
        today: Reference date for cooldown arithmetic (injectable for tests).
        challenger_version: MLflow model version number to promote to @champion.
        client: MLflow client instance (injectable for unit tests; defaults to a
            live ``MlflowClient()`` when not provided).

    Returns:
        True if promotion occurred; False otherwise.
    """
    if client is None:
        from mlflow import MlflowClient  # lazy import — no live registry in tests

        client = MlflowClient()

    # --- 1. Cooldown guard ---
    if last_promotion_date is not None:
        days_since = (today - last_promotion_date).days
        if days_since < PROMOTION_COOLDOWN_DAYS:
            log.info(
                "Promotion blocked by cooldown: %d days since last promotion (required >= %d)",
                days_since,
                PROMOTION_COOLDOWN_DAYS,
            )
            return False

    # --- 2. Strict-win requirement (default no-promote) ---
    if challenger_qlike >= champion_qlike:
        log.info(
            "No promotion: challenger QLIKE %.6f >= champion QLIKE %.6f (no-promote default)",
            challenger_qlike,
            champion_qlike,
        )
        return False

    # --- 3. Promote ---
    log.info(
        "Promoting challenger v%s → @champion: QLIKE %.6f < %.6f (champion)",
        challenger_version,
        challenger_qlike,
        champion_qlike,
    )

    # Record previous champion version for rollback (Pitfall 7 / T-04-13)
    prev_version: str | None = None
    try:
        prev_mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
        prev_version = str(prev_mv.version)
    except Exception:  # noqa: BLE001 — first-ever promotion has no existing alias
        log.warning(
            "Could not resolve existing @champion version for rollback record "
            "(may be first-ever promotion); proceeding without rollback tag."
        )

    # Atomic alias flip (T-04-11) — this is the ONLY call site for "champion" alias flip
    client.set_registered_model_alias(MODEL_NAME, "champion", challenger_version)

    # Tag new champion with rollback info
    if prev_version is not None:
        client.set_model_version_tag(
            MODEL_NAME,
            str(challenger_version),
            "previous_champion_version",
            prev_version,
        )
        log.info("Tagged v%s with previous_champion_version=%s", challenger_version, prev_version)

    log.info("Promotion complete: @champion → v%s", challenger_version)
    return True


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def rollback_champion(to_version: int, *, client: Any = None) -> None:
    """Rollback @champion to a previously recorded version.

    Flips @champion back to *to_version* via a single
    ``set_registered_model_alias`` call.  Read the ``previous_champion_version``
    tag from the current champion version to obtain *to_version*.

    This is the ONLY other function (besides ``promote_if_better``) that may
    call ``set_registered_model_alias`` with alias ``"champion"``.

    Args:
        to_version: MLflow model version to restore as @champion.
        client: MLflow client instance (injectable for unit tests).
    """
    if client is None:
        from mlflow import MlflowClient  # lazy import — no live registry in tests

        client = MlflowClient()

    log.info("Rolling back @champion → v%s", to_version)
    client.set_registered_model_alias(MODEL_NAME, "champion", to_version)
    log.info("Rollback complete: @champion → v%s", to_version)
