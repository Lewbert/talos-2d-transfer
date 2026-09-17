"""Shared state for the simulated bench.

The stage and the camera are separate devices with separate worker threads,
so in the real application the only thing connecting them is the world. The
simulation needs the same connection: a simulated camera that always images
the same scene cannot show whether a SCAN stitches correctly — every tile
looks identical, and a backward or mirrored mosaic is indistinguishable
from a correct one.

So the simulated XYR stage publishes where it is, and the simulated camera
images a wafer that moves under it. `SimCamera(config={"wafer": True})`
opts in; without it the camera renders the static test scene every closed
loop suite was calibrated against.

Module-level and Qt-free on purpose: any thread may read or write it.
"""

from __future__ import annotations

#: Where the simulated XYR stage is, in µm. Written by SimZolixXYRStage on
#: every position query; read by SimCamera when it renders a wafer frame.
xy_um: tuple[float, float] = (0.0, 0.0)

#: Where the simulated focus axis is, in steps (unused by the camera today,
#: kept beside xy so a future defocus rig has one place to look).
focus_steps: int = 0


def set_xy(x_um: float, y_um: float) -> None:
    global xy_um
    xy_um = (float(x_um), float(y_um))


def get_xy() -> tuple[float, float]:
    return xy_um


def reset() -> None:
    """Back to the origin — tests call this so one test's scan cannot move
    the next test's wafer."""
    global focus_steps
    set_xy(0.0, 0.0)
    focus_steps = 0


__all__ = ["focus_steps", "get_xy", "reset", "set_xy", "xy_um"]
