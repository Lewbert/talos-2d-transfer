"""How stage coordinates and image coordinates are related — one place, and
the sign of every axis is in it.

Two independent things decide the relation:

**The mounting.** Which way the optics put the specimen on the sensor. The
bench was measured on 2026-09-17, with the flip in its default state: stage
+X moves a feature to the right in the frame, and stage +Y moves it UP.
``_MOUNTED`` is that statement in the form this module uses — the sign of
each stage axis as the UNFLIPPED frame sees it: ``(x, -y)``. It was
``(x, y)`` by assumption before the bench said otherwise, which put every
mosaic, map and "go to sample" move a mirror-image away in Y.

**The camera flip.** A 180° rotation of every delivered frame
(``cv2.flip(frame, -1)``, applied by each backend at its single egress), so
it negates BOTH axes of that relation. It is the operator's correction for
the bench's optics, and it is why the default build reads ``(-x, +y)``.

Why this maps more than offsets. A frame taken with the stage at ``p`` shows
the sample point whose stage coordinate is ``s`` at image offset
``a·(s - p)`` where ``a`` is the per-axis sign above. Drop that tile into a
mosaic at position ``m(p)`` and the point lands at ``m(p) + a·(s - p)``,
which must depend on ``s`` ALONE for two tiles to agree about where a
feature is — so ``m(p) - a·p`` has to be constant: **the layout is scaled by
the same ``a``**, not merely the offsets. Place tiles with the wrong sign and
every feature appears once per tile, at a different place each time: a mosaic
that looks doubled, which is exactly what shipped before this module existed
(the tiles were placed unmirrored AND rotated in place, which is wrong twice
over).

The same ``a`` governs the px→µm mapping, so "go to sample" and the map can
never disagree about which way +Y is.
"""

from __future__ import annotations

#: The mounting, as the UNFLIPPED frame sees it: (x sign, y sign).
#: Bench-measured 2026-09-17 — X as assumed, Y inverted.
_MOUNTED: tuple[int, int] = (1, -1)


def axis_signs(flip: bool) -> tuple[int, int]:
    """(sx, sy): the sign of each stage axis in the frame's coordinates.
    The flip is a 180° rotation, so it negates both."""
    sx, sy = _MOUNTED
    return (-sx, -sy) if flip else (sx, sy)


def orient(x: float, y: float, flip: bool) -> tuple[float, float]:
    """A 2-D offset in the other frame. Used for stage↔image offsets and
    for mosaic/map layout alike (see the module docstring for why one
    function serves both)."""
    sx, sy = axis_signs(flip)
    return float(x) * sx, float(y) * sy


def stage_offset(dx_um: float, dy_um: float, flip: bool) -> tuple[float, float]:
    """An image offset (µm from the frame centre) → the stage offset of the
    sample point imaged there."""
    return orient(dx_um, dy_um, flip)


def mosaic_offset(x_um: float, y_um: float, flip: bool) -> tuple[float, float]:
    """A stage coordinate → where it belongs in a mosaic/map layout."""
    return orient(x_um, y_um, flip)


__all__ = ["axis_signs", "mosaic_offset", "orient", "stage_offset"]
