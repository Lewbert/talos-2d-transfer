"""App-bootstrap data migrations.

These run at startup and must NEVER raise — a broken calibration DB must
not block the app. All migrations here are idempotent.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def materialize_labscope_calibration(settings) -> dict:
    """The one-shot Labscope unwire: materialize the Zeiss table values as
    proper px_um calibration entries (provenance labscope_materialized),
    adopt the settings-row objective names in the DB, and backfill each
    settings row's px_um from the canonical value so the Objectives page
    shows the live calibration. Idempotent — existing px_um entries
    (wizard/manual) always win. Returns the store counts ({} on failure).
    """
    try:
        from talos.calibration_store import CalibrationStore
        from talos.paths import get_calibration_db_path

        rows = settings.get("objectives") or []
        names = {i: str(row.get("name") or "")
                 for i, row in enumerate(rows) if row.get("name")}
        mags = {i: float(row["mag"])
                for i, row in enumerate(rows)
                if row.get("mag") and float(row.get("mag") or 0) > 0}
        store = CalibrationStore(get_calibration_db_path())
        try:
            counts = store.materialize_labscope(names, mags)
            by_pos = {o["nosepiece_position"]: o
                      for o in store.list_objectives()}
        finally:
            store.close()
        changed = False
        for i, row in enumerate(rows):
            if float(row.get("px_um") or 0.0) > 0:
                continue
            obj = by_pos.get(i)
            if not obj:
                continue
            value = obj.get("labscope_px_per_unit")
            units = obj.get("labscope_px_per_unit_units")
            if not value or not units:
                continue
            um = CalibrationStore.canonical_um_per_px(
                float(value), str(units))
            if um and um > 0:
                rows[i]["px_um"] = round(um, 6)
                changed = True
        if changed:
            settings.data["objectives"] = rows
            settings.save()
        logger.info("Labscope materialization: %s (settings px_um %s)",
                    counts, "backfilled" if changed else "unchanged")
        return counts
    except Exception as exc:  # noqa: BLE001
        logger.warning("Labscope materialization skipped: %s", exc)
        return {}
