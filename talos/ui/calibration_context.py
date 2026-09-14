"""Shared active-calibration lookup for the UI: the scale-bar overlay,
the snapshot burn and the sample-finding workspace all need the current
objective's µm/px. Cached per nosepiece position; refreshed when the
objective selection changes (the DB hit happens once per switch, not
per frame)."""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal

from talos.calibration_store import CalibrationStore
from talos.cv.calibration import SENSOR_WIDTH_PX
from talos.models import ObjectiveCalibration
from talos.paths import get_calibration_db_path

logger = logging.getLogger(__name__)


class CalibrationContext(QObject):
    sig_changed = Signal(object)   # ObjectiveCalibration

    def __init__(self, state, parent: QObject | None = None,
                 store: CalibrationStore | None = None):
        super().__init__(parent)
        self._state = state
        self._store = store or self._open_store()
        self._cached: ObjectiveCalibration | None = None
        state.sig_objective_changed.connect(lambda _i: self.refresh())
        self.refresh()

    @staticmethod
    def _open_store() -> CalibrationStore | None:
        """A damaged calibration.db must not brick startup — the app falls
        back to the pixel-pitch estimate and says so."""
        try:
            return CalibrationStore(get_calibration_db_path())
        except Exception as exc:  # noqa: BLE001 - any DB trouble is non-fatal
            logger.error("Calibration DB unavailable (%s) — using the "
                         "pixel-pitch fallback", exc)
            return None

    def calibration(self) -> ObjectiveCalibration:
        if self._cached is None:
            self.refresh()
        return self._cached

    def refresh(self) -> None:
        calib = self._lookup()
        if calib != self._cached:
            self._cached = calib
            self.sig_changed.emit(calib)

    def um_per_px(self) -> float | None:
        calib = self.calibration()
        value = calib.um_per_px_x if calib is not None else None
        return float(value) if value and value > 0 else None

    def um_per_px_at(self, width_px: int) -> float | None:
        """The canonical value is per 4K-SENSOR pixel; convert it for a
        frame of the given width (a 1080p frame pixel covers 2× the µm)."""
        value = self.um_per_px()
        if value is None or width_px <= 0:
            return None
        return value * (SENSOR_WIDTH_PX / width_px)

    def _lookup(self) -> ObjectiveCalibration:
        pos = self._state.objective
        if self._store is not None:
            try:
                objectives = self._store.list_objectives()
                for obj in objectives:
                    if obj.get("nosepiece_position") == pos:
                        return self._store.get_active_calibration(obj["id"])
            except Exception as exc:  # noqa: BLE001
                logger.error("Calibration lookup failed (%s) — using the "
                             "pixel-pitch fallback", exc)
        # Fallback: Axiocam 2 µm pixels / magnification (5x·2^pos).
        mag = 5.0 * (2.0 ** pos)
        um_per_px = 2.0 / mag
        return ObjectiveCalibration(objective_id=-1, um_per_px_x=um_per_px,
                                    um_per_px_y=um_per_px, source="pixel_pitch")
