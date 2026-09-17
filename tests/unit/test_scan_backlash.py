"""Backlash take-up: every move must FINISH from the same side.

Play in the drive means the same coordinate reached from −X and from +X
sits at two different physical places, so the cure is not "correct each
reversal" but "never finish a move from the wrong side" — a serpentine
whose rows alternate would otherwise land its odd rows a backlash away
from its even ones.
"""

import pytest

from talos.cv.scan import GridScanner, backlash_fix, plan_path
from talos.models import ScanParams


def test_no_correction_when_the_approach_is_already_right():
    # travelling +X into the target with a +X approach: nothing to take up
    assert backlash_fix((0.0, 0.0), (100.0, 0.0), 5.0, 1) == []
    assert backlash_fix((0.0, 50.0), (100.0, 50.0), 5.0, 1) == []
    # mirrored for a −X approach
    assert backlash_fix((100.0, 0.0), (0.0, 0.0), 5.0, -1) == []


def test_reversal_backs_off_past_the_target_and_comes_in():
    steps = backlash_fix((100.0, 0.0), (0.0, 0.0), 5.0, 1)
    assert steps == [(-5.0, 0.0)]
    # the caller then moves to (0, 0): the final motion is +X ✓


def test_both_axes_are_taken_up_in_one_intermediate_move():
    steps = backlash_fix((100.0, 60.0), (0.0, 0.0), 5.0, 1)
    assert steps == [(-5.0, -5.0)]


def test_an_axis_that_does_not_move_keeps_its_play_state():
    """A whole serpentine row shares one Y — correcting Y every waypoint
    would add a pointless back-and-forth per tile."""
    # Y is already at the target: only X needs the take-up.
    assert backlash_fix((100.0, 0.0), (0.0, 0.0), 5.0, 1) == [(-5.0, 0.0)]
    # moving right along a row: no correction at all
    assert backlash_fix((0.0, 0.0), (100.0, 0.0), 5.0, 1) == []


def test_zero_backlash_is_off():
    assert backlash_fix((100.0, 0.0), (0.0, 0.0), 0.0, 1) == []
    assert backlash_fix((100.0, 0.0), (0.0, 0.0), -3.0, 1) == []


def test_no_previous_position_means_nothing_to_take_up():
    """The first waypoint is where the operator already stands."""
    assert backlash_fix(None, (0.0, 0.0), 5.0, 1) == []


class _RecordingStage:
    """Minimal XYRStage stand-in: records every commanded position."""

    def __init__(self):
        self.moves: list[tuple[float, float]] = []
        self._pos = (0.0, 0.0)

    def move_abs_um(self, x, y, r_deg=None, speed=None):
        self.moves.append((float(x), float(y)))
        self._pos = (float(x), float(y))

    def wait_idle(self, timeout_s=120.0):
        pass

    def get_position(self):
        from talos.models import StagePosition
        return StagePosition(x_um=self._pos[0], y_um=self._pos[1])

    def stop(self):
        pass


@pytest.mark.parametrize("approach", [1, -1])
def test_the_scanner_finishes_every_move_from_the_same_side(approach, tmp_path):
    """The invariant, checked over a whole serpentine run: every commanded
    move that LANDS on a waypoint arrives along the approach direction on
    every axis it moves."""
    params = ScanParams(x0_um=0.0, y0_um=0.0, width_um=499.0, height_um=299.0,
                        overlap=0.0, serpentine=True, backlash_um=4.0,
                        backlash_approach=approach, settle_ms=0,
                        return_to_start=False)
    fov = (100.0, 100.0)
    waypoints = plan_path(params, fov)
    targets = {(round(w.x_um, 3), round(w.y_um, 3)) for w in waypoints}

    stage = _RecordingStage()
    scanner = GridScanner(stage, frame_source=None)
    scanner.run(params, tmp_path, meta={"fov_um": fov})

    prev = (0.0, 0.0)          # the start point is the first waypoint
    landings = 0
    for point in stage.moves:
        if (round(point[0], 3), round(point[1], 3)) in targets:
            landings += 1
            for axis in (0, 1):
                delta = point[axis] - prev[axis]
                if abs(delta) > 1e-9:
                    assert delta * approach > 0, (
                        f"landing on {point} arrived from the wrong side "
                        f"on axis {axis}")
        prev = point
    assert landings == len(waypoints)
    # the take-up costs one extra move per non-conforming waypoint, and
    # never more than one
    assert len(stage.moves) <= 2 * len(waypoints)
