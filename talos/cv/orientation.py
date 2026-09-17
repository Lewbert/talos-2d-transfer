"""The camera flip, as a coordinate transform — one place, one sign.

The flip is a 180° rotation applied to every delivered frame (the backends
apply it at their single egress). It is not cosmetic: it decides how stage
motion and image content relate, so anything that maps BETWEEN the image
and the stage has to apply it, and anything that ASSEMBLES images into a
sample-frame picture has to mirror its layout by it.

Why the layout mirrors. A frame taken with the stage at ``p`` is the scene
rotated 180° about the frame centre, so the sample point whose stage
coordinate is ``s`` (the stage position at which that point is centred in
the image) sits at image offset ``-(s - p)``. Drop that tile into a mosaic
at position ``m(p)`` and the point lands at ``m(p) - (s - p)``. For two
tiles to agree about where a feature is, that must depend on ``s`` alone —
so ``m(p) + p`` has to be constant, i.e. ``m(p) = C - p``: the layout is
mirrored. Place tiles unmirrored instead and every feature appears once per
tile, at a different place each time — a mosaic that looks doubled, which
is exactly what shipped before this module existed.

The same reasoning fixes the px→µm mapping: with the flip on, a feature
appearing to the RIGHT of the frame centre is at a smaller stage X, not a
larger one. ``flake_to_stage`` ignored the flip, so "go to sample" drove
the stage to the mirrored position.

The other half of the convention — whether an UNFLIPPED frame has stage +X
to the right — is the mounting, and it is what the flip setting exists to
correct: with the flip set the way the operator wants, an unflipped frame
is the reference. That is the assumption `flake_to_stage` has always made.
"""

from __future__ import annotations


def orient(x: float, y: float, flip: bool) -> tuple[float, float]:
    """A 2-D offset in the other frame: (x, y) with the flip, unchanged
    without it. Used for stage↔image offsets and for mosaic layout."""
    if flip:
        return -float(x), -float(y)
    return float(x), float(y)


def stage_offset(dx_um: float, dy_um: float, flip: bool) -> tuple[float, float]:
    """An image offset (µm from the frame centre) → the stage offset of the
    sample point imaged there."""
    return orient(dx_um, dy_um, flip)


def mosaic_offset(x_um: float, y_um: float, flip: bool) -> tuple[float, float]:
    """A stage coordinate → where it belongs in a mosaic/map layout."""
    return orient(x_um, y_um, flip)


__all__ = ["mosaic_offset", "orient", "stage_offset"]
