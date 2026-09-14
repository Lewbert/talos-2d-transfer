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

logger = logging.getLogger(__name__)

#: The region "Reset ROI" restores: the centre 2/3 × 2/3 of the frame (the
#: historical default — fewer pixels to score, away from the vignetted edge).
DEFAULT_ROI_NORM = (0.1667, 0.1667, 0.6667, 0.6667)

MIN_SIDE = 0.02          # a ROI smaller than 2 % of a side is not useful


def sanitize_roi(norm) -> tuple[float, float, float, float] | None:
    """Clamp an arbitrary (x, y, w, h) into the frame; None when unusable."""
    if norm is None:
        return None
    try:
        x, y, w, h = (float(v) for v in norm)
    except (TypeError, ValueError):
        return None
    x = min(max(x, 0.0), 1.0 - MIN_SIDE)
    y = min(max(y, 0.0), 1.0 - MIN_SIDE)
    w = min(max(w, MIN_SIDE), 1.0 - x)
    h = min(max(h, MIN_SIDE), 1.0 - y)
    return (round(x, 4), round(y, 4), round(w, 4), round(h, 4))


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
