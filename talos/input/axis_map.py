"""Manual-control axis mapping: per-axis inversion + the X↔Y axis swap.

Some operators prefer an inverted jog (the precursor project exposes the
same knobs as "Invert Axes" + "Flip X/Y" checkboxes per stage). The mapping
is applied in ``InputSystem._dispatch`` — the single choke point every
manual source funnels through: keyboard, gamepad sticks / D-pad /
triggers, on-screen hold buttons and single clicks, the dialbox.

Order matters and follows the precursor project
(``transfer-stage-control/stage_control/instruments.py``): flip the AXIS
IDENTITY first, then invert the direction of the (possibly swapped) axis,
so ``invert_z`` + ``flip_xy`` behaves as the reference does.

Scope: MANUAL motion only. Position readback, autofocus, the objective
focus offsets, the flake "go to" move and the grid scan are computed
motions and are deliberately NOT mapped — inverting a computed move would
silently corrupt stored coordinates and scan geometry. The camera flip is
likewise independent: it rotates the image, never an axis.

Settings keys: ``devices.<stage>.invert_<axis>`` (``invert_r`` on the
Zolix XYR), ``devices.<stage>.flip_xy``, and ``devices.focus.invert`` for
the focus jog (triggers, keys and the dialbox holds alike).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

#: Which axes a device can invert (also the valid ``invert_<axis>`` keys).
AXES: dict[str, tuple[str, ...]] = {
    "sigmakoki": ("x", "y", "z"),
    "zolix": ("x", "y", "r"),
}


@dataclass(frozen=True)
class AxisMap:
    """Axis/direction mapping for one device. Immutable — rebuild it from
    settings rather than mutating it (see ``axis_maps``)."""

    invert: Mapping[str, bool] = field(default_factory=dict)
    flip_xy: bool = False

    def apply(self, axis: str, direction: int) -> tuple[str, int]:
        """Map a commanded (axis, direction) to the physical one.

        ``direction`` 0 (a stop) is passed through unchanged — a stop must
        address the axis it started, and the flip still applies to it.
        """
        if self.flip_xy and axis in ("x", "y"):
            axis = "y" if axis == "x" else "x"
        if self.invert.get(axis, False):
            direction = -direction
        return axis, direction



#: The neutral map (used for unknown devices).
IDENTITY = AxisMap()


def axis_map_for(settings, device_key: str) -> AxisMap:
    """The map configured for one device (IDENTITY when it has none)."""
    if device_key == "focus":
        cfg = settings.device("focus")
        return AxisMap({"z": bool(cfg.get("invert", False))})
    axes = AXES.get(device_key)
    if axes is None:
        return IDENTITY
    cfg = settings.device(device_key)
    invert = {axis: bool(cfg.get(f"invert_{axis}", False)) for axis in axes}
    return AxisMap(invert, bool(cfg.get("flip_xy", False)))


def axis_maps(settings) -> dict[str, AxisMap]:
    """Maps for every device manual input can drive."""
    return {key: axis_map_for(settings, key)
            for key in ("sigmakoki", "zolix", "focus")}
