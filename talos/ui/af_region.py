"""The autofocus measurement region (AF ROI) as ONE piece of shared state.

Three places care about it: the right-panel AF settings, the AF detail
window, and the live-view overlay. They used to keep private copies (the
panel had a combo, the overlay read the panel's attribute, the settings key
was never loaded), which is how the panel came to claim "Full frame" while
autofocus was actually measuring the centre crop.

Semantics: ``None`` means the WHOLE frame. A tuple is the normalised
(x, y, w, h) region autofocus scores — the same value the metric path uses,
so the overlay can never disagree with what AF measures.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal

from talos.cv import af_roi

logger = logging.getLogger(__name__)

#: The region "Reset ROI" restores: the centre 2/3 × 2/3 of the frame (the
#: historical default — fewer pixels to score, away from the vignetted edge).
DEFAULT_ROI_NORM = (0.1667, 0.1667, 0.6667, 0.6667)

# The clamp rule lives in the CV layer so the autofocus service can apply the
# SAME one to a stored ROI (it used to pass the settings value through
# unsanitized and crash inside cv2 on a 1-pixel crop).
MIN_SIDE = af_roi.MIN_SIDE


def sanitize_roi(norm) -> tuple[float, float, float, float] | None:
    """Clamp an arbitrary (x, y, w, h) into the frame; None when unusable."""
    return af_roi.sanitize_roi_norm(norm)


def mirror_roi_norm(norm) -> tuple[float, float, float, float] | None:
    """The same region after the frame is rotated 180° (x' = 1 − x − w).

    Used when the camera flip changes: the region is stored in FRAME
    coordinates, so without the mirror autofocus would keep measuring the
    diagonally opposite corner of the specimen.
    """
    roi = sanitize_roi(norm)
    if roi is None:
        return None
    x, y, w, h = roi
    return sanitize_roi((1.0 - x - w, 1.0 - y - h, w, h))


class AfRegionController(QObject):
    """The AF measurement region, persisted in ``autofocus.default_roi_norm``.

    Autofocus itself reads that same settings key (``AutofocusService.
    _default_roi``), so every AF start — panel, quick action, sample-finding
    workspace — uses exactly what this controller holds and the overlay
    draws, with no call site needing to pass anything.
    """

    sig_changed = Signal(object)   # tuple | None

    def __init__(self, settings, parent: QObject | None = None):
        super().__init__(parent)
        self._settings = settings
        self._roi = sanitize_roi(settings.section("autofocus")
                                 .get("default_roi_norm"))

    # -- state ------------------------------------------------------------

    def roi(self) -> tuple[float, float, float, float] | None:
        """Normalised region, or None for the whole frame."""
        return self._roi

    def is_full_frame(self) -> bool:
        return self._roi is None

    def describe(self) -> str:
        """One-line human summary (used by the readouts)."""
        if self._roi is None:
            return "whole frame"
        x, y, w, h = self._roi
        return (f"x {x * 100:.0f} %, y {y * 100:.0f} %, "
                f"{w * 100:.0f} × {h * 100:.0f} % of the frame")

    # -- mutations --------------------------------------------------------

    def set_roi(self, norm, *, persist: bool = True) -> None:
        roi = sanitize_roi(norm)
        if roi == self._roi:
            return
        self._roi = roi
        if persist:
            self._persist()
        self.sig_changed.emit(roi)

    def use_full_frame(self, *, persist: bool = True) -> None:
        self.set_roi(None, persist=persist)

    def reset_to_default(self, *, persist: bool = True) -> None:
        self.set_roi(DEFAULT_ROI_NORM, persist=persist)

    def move_region(self, *, x=None, y=None, w=None, h=None) -> None:
        """Edit individual fields of the region (the numeric spinboxes).

        Editing any field implies ROI mode: from "whole frame" the edit
        starts from the default region so the numbers have a meaning.
        """
        base = self._roi or DEFAULT_ROI_NORM
        values = list(base)
        for index, value in enumerate((x, y, w, h)):
            if value is not None:
                values[index] = value
        self.set_roi(values)

    def _persist(self) -> None:
        cfg = self._settings.section("autofocus")
        cfg["default_roi_norm"] = list(self._roi) if self._roi else None
        self._settings.save()

    # -- helpers for the views -------------------------------------------

    @staticmethod
    def as_percent(roi) -> tuple[float, float, float, float]:
        """(x, y, w, h) normalized -> percentages, for the spinboxes."""
        x, y, w, h = roi or DEFAULT_ROI_NORM
        return (x * 100.0, y * 100.0, w * 100.0, h * 100.0)
