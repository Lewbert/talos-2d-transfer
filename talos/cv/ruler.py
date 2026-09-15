"""Calibrated tick ruler: major/minor tick spacing for the live view.

Pure maths in FRAME pixels — the Qt renderer maps the offsets through the
same letterbox/pixmap scale the scale bar uses, and the snapshot burn
never sees a ruler (a ruler is a screen aid, not part of the record).

Spacing comes from the SAME 1/2/5 ladder as the scale bar
(``nice_length_um_at_most``), so the ruler and the bar can never disagree
about what a "round" length is. Ticks are laid out from the FRAME CENTRE,
which is where the crosshair sits and where the stage position is, so
"0" on the ruler is the optical axis and the labelled span in µm does not
change when the window is resized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from talos.cv.scale_bar import format_length_um, nice_length_um_at_most

#: A major tick is at most this fraction of the frame width apart. The
#: ladder then picks the largest round length below it, so majors land
#: between ~6 % and 16 % of the frame — 6 to 15 of them per axis.
MAJOR_MAX_FRACTION = 0.16

#: Minor ticks finer than this many frame pixels are not drawn (they read
#: as a solid line and cost paint time for nothing).
MIN_MINOR_PX = 3.0


def minors_per_major(major_um: float) -> int:
    """How many minor divisions a major tick is divided into.

    1 → 5 (0.2 each), 2 → 4 (0.5 each), 5 → 5 (1 each) — i.e. the minor
    step is always a round number too.
    """
    if major_um <= 0:
        return 1
    mantissa = major_um / (10.0 ** math.floor(math.log10(major_um)))
    if mantissa >= 4.0:
        return 5
    if mantissa >= 1.5:
        return 4
    return 5


@dataclass(frozen=True)
class RulerSpec:
    """Major/minor step of the ruler in µm, plus the major's label."""

    major_um: float
    minor_um: float
    minors_per_major: int
    label: str


@lru_cache(maxsize=32)
def _ruler_spec_cached(um_per_px: float, frame_shape: tuple,
                       max_major_fraction: float) -> RulerSpec | None:
    if um_per_px <= 0 or not frame_shape:
        return None
    width_px = int(frame_shape[1])
    if width_px <= 0:
        return None
    major_um = nice_length_um_at_most(um_per_px, width_px,
                                      max_fraction=max_major_fraction)
    if major_um <= 0:
        return None
    divisions = minors_per_major(major_um)
    return RulerSpec(major_um=major_um,
                     minor_um=major_um / divisions,
                     minors_per_major=divisions,
                     label=format_length_um(major_um))


def ruler_spec(um_per_px: float | None, frame_shape,
               max_major_fraction: float = MAJOR_MAX_FRACTION) \
        -> RulerSpec | None:
    """The ruler's step for this calibration/frame, or None when useless.

    Cached like ``scale_bar_layout``: both the spec and the tick offsets
    are recomputed only when the calibration, the frame shape or the
    window size actually change.
    """
    if um_per_px is None or not isinstance(um_per_px, (int, float)):
        return None
    shape = tuple(frame_shape[:2]) if frame_shape is not None else ()
    return _ruler_spec_cached(float(um_per_px), shape,
                              float(max_major_fraction))


def tick_offsets(spec: RulerSpec, um_per_px: float, frame_px: int) \
        -> tuple[list[float], list[float]]:
    """(major, minor) tick offsets in FRAME pixels, measured from the
    frame centre (negative = left/up). Minor offsets that coincide with a
    major are omitted — the renderer draws the major over them anyway."""
    if spec is None or um_per_px <= 0 or frame_px <= 0:
        return [], []
    major_px = spec.major_um / um_per_px
    minor_px = spec.minor_um / um_per_px
    if major_px <= 0:
        return [], []
    half = frame_px / 2.0
    n_major = int(half // major_px)
    majors = [i * major_px for i in range(-n_major, n_major + 1)]
    if minor_px < MIN_MINOR_PX:
        return majors, []
    n_minor = int(half // minor_px)
    ratio = max(1, int(round(spec.minors_per_major)))
    minors = [i * minor_px for i in range(-n_minor, n_minor + 1)
              if i % ratio != 0]
    return majors, minors


def tick_label(offset_um: float) -> str:
    """The µm value of a major tick, relative to the frame centre."""
    value = round(float(offset_um), 2)
    if value == 0:
        return "0"
    return f"{value:g}"
