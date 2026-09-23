"""A fault the stage can recover from is retried, not fatal.

The bench report: one zolix timeout aborted a whole scan. The timeout is
resumable — the controller lost a reply, or an axis was still moving — and
re-issuing the approach is safe because the move is ABSOLUTE: the driver
recomposes it from a fresh readback, so a duplicate is a no-op.

What must NOT be retried is the hardware answering: a limit switch, an
e-stop, a dead port. A retry loop that tries those again turns a fast, clear
failure into minutes of a jammed stage, which is worse than no retry at all.

And one case that is silently WRONG to retry: a stage that will not settle.
The re-issued move is composed from a readback taken mid-flight, so it lands
at ``target + (target - where it was)`` — the manifest stays honest but the
tile grid gains a gap and a doubled tile and nothing on screen says so.
"""

import pytest

from talos.cv.scan import GridScanner, plan_path
from talos.hal.base import (DeviceBusyError, DeviceError, DeviceTimeoutError,
                            EStopError, LimitHitError, NotConnectedError,
                            ProtocolError)
from talos.models import ScanParams, StagePosition

FOV = (100.0, 100.0)
PARAMS = dict(x0_um=0.0, y0_um=0.0, width_um=499.0, height_um=299.0,
              overlap=0.0, serpentine=True, settle_ms=0,
              return_to_start=False)
BACKLASH = dict(PARAMS, backlash_um=4.0, backlash_approach=1)


def _plan(params):
    return plan_path(ScanParams(**params), FOV)


class _FaultyStage:
    """A stage that fails on demand — by move-call number, by wait-call
    number, and by exception kind."""

    def __init__(self, move_faults=None, idle_faults=None):
        self.moves: list[tuple[float, float]] = []
        #: every commanded point, including the ones that raised
        self.attempts: list[tuple[float, float]] = []
        self.move_calls = 0
        self.idle_calls = 0
        self.move_faults = dict(move_faults or {})   # call no. -> factory
        self.idle_faults = dict(idle_faults or {})   # call no. -> factory
        self._pos = (0.0, 0.0)

    def move_abs_um(self, x, y, r_deg=None, speed=None, speed_pps=None):
        self.move_calls += 1
        self.attempts.append((float(x), float(y)))
        fault = self.move_faults.get(self.move_calls)
        if fault is not None:
            raise fault()
        self.moves.append((float(x), float(y)))
        self._pos = (float(x), float(y))

    def wait_idle(self, timeout_s=120.0):
        self.idle_calls += 1
        fault = self.idle_faults.get(self.idle_calls)
        if fault is not None:
            raise fault()

    def get_position(self):
        return StagePosition(x_um=self._pos[0], y_um=self._pos[1])

    def stop(self):
        pass


def _run(stage, tmp_path, params=None):
    scanner = GridScanner(stage, frame_source=None)
    result = scanner.run(ScanParams(**(params or PARAMS)), tmp_path,
                         meta={"fov_um": FOV})
    return scanner, result, _plan(params or PARAMS)


def _timeout():
    return DeviceTimeoutError("zolix.move_abs_um: DeviceTimeoutError: "
                              "No Modbus reply on COM3")


# --- what a retry clears --------------------------------------------------

def test_a_timeout_on_the_approach_is_retried_and_the_run_finishes(tmp_path):
    """The bench case: the run must carry on, and say that it retried."""
    stage = _FaultyStage(move_faults={3: _timeout})
    _scanner, result, waypoints = _run(stage, tmp_path)

    assert result.complete, result.message
    assert not result.stopped_early
    assert result.timing.retries == 1
    assert result.visited == result.planned == len(waypoints)
    # one move per waypoint (no backlash here) plus the ONE re-issued
    # approach: a retry must not cost a second move per waypoint after it
    assert stage.move_calls == len(waypoints) + 1


def test_a_busy_axis_is_retried(tmp_path):
    """The commonest bench flavour: the previous motion was still running
    when the next command arrived (the driver refuses that by design)."""
    stage = _FaultyStage(move_faults={2: lambda: DeviceBusyError(
        "Zolix axes are moving; wait for idle first")})
    _scanner, result, _waypoints = _run(stage, tmp_path)
    assert result.complete
    assert result.timing.retries == 1


def test_a_lost_frame_is_retried(tmp_path):
    """A CRC/truncated reply that survives the driver's own three read
    attempts: the link is bad for a moment, not the stage."""
    stage = _FaultyStage(move_faults={2: lambda: ProtocolError(
        "Read input reg 30016: Frame too short (3 attempts)")})
    _scanner, result, _waypoints = _run(stage, tmp_path)
    assert result.complete
    assert result.timing.retries == 1


def test_the_retry_shows_up_in_the_timing_summary(tmp_path):
    """The bench reads s/tile from the summary; a run that fought the link
    must not look like a slow one."""
    stage = _FaultyStage(move_faults={3: _timeout})
    _scanner, result, _waypoints = _run(stage, tmp_path)
    assert "1 retried approach(es)" in result.timing.summary
    other = GridScanner(_FaultyStage(), frame_source=None)
    clean = other.run(ScanParams(**PARAMS), tmp_path,
                      meta={"fov_um": FOV}).timing.summary
    assert "retried" not in clean


# --- what it does not ----------------------------------------------------

def test_a_fault_that_never_clears_gives_up_after_three_attempts(tmp_path):
    """Bounded: a link that is down stays down, and the operator gets a
    stopped run with the reason rather than an endless retry."""
    from talos.cv import scan as scan_mod

    attempts = scan_mod._MAX_APPROACH_ATTEMPTS
    stage = _FaultyStage(move_faults={i: _timeout for i in range(2, 40)})
    _scanner, result, _waypoints = _run(stage, tmp_path)

    assert result.stopped_early and not result.aborted
    assert stage.move_calls == 1 + attempts
    assert "No Modbus reply" in result.message
    assert result.timing.retries == attempts - 1


@pytest.mark.parametrize("fault", [
    lambda: LimitHitError("Modbus exception 0x07"),
    lambda: EStopError("Zolix emergency stop is active"),
    lambda: NotConnectedError("Zolix serial error: port closed"),
])
def test_the_hardware_answering_is_not_retried(fault, tmp_path):
    """A limit switch cannot be asked again into agreeing."""
    stage = _FaultyStage(move_faults={2: fault})
    _scanner, result, _waypoints = _run(stage, tmp_path)
    assert stage.move_calls == 2, "the move was re-issued"
    assert result.stopped_early
    assert not result.complete


def test_the_settle_timeout_is_not_retried(tmp_path):
    """Not a move fault at all: a stage that reports "moving" for 120 s is
    jammed or travelling, and re-issuing onto it lands the stage a whole
    step past the target."""
    stage = _FaultyStage(idle_faults={2: lambda: DeviceTimeoutError(
        "Zolix stage not settled after 120 s")})
    _scanner, result, _waypoints = _run(stage, tmp_path)
    assert stage.move_calls == 2, "the approach was re-issued after a settle"
    assert result.stopped_early
    assert "not settled" in result.message


def test_an_approach_that_will_not_settle_before_the_retry_is_not_re_issued(
        tmp_path):
    """The quiesce between attempts is what makes the retry safe. If it
    fails, the retry is abandoned rather than taken."""
    stage = _FaultyStage(move_faults={2: _timeout},
                         idle_faults={2: lambda: DeviceTimeoutError(
                             "Zolix stage not settled after 20 s")})
    _scanner, result, _waypoints = _run(stage, tmp_path)
    assert stage.move_calls == 2, "re-issued onto a stage that never stopped"
    assert result.stopped_early
    assert result.timing.retries == 0


def test_an_abort_is_not_retried(tmp_path):
    """Esc during a fault must end the run, not start another approach."""
    stage = _FaultyStage()
    scanner = GridScanner(stage, frame_source=None)

    def aborting_move(x, y, r_deg=None, speed=None, speed_pps=None):
        stage.move_calls += 1
        scanner.abort_requested = True
        raise _timeout()

    stage.move_abs_um = aborting_move
    result = scanner.run(ScanParams(**PARAMS), tmp_path,
                         meta={"fov_um": FOV})
    assert result.aborted
    assert stage.move_calls == 1


# --- the state a retry must not corrupt ----------------------------------

def _a_reversal(params):
    """(waypoint, prev, steps) for the first waypoint that needs a take-up."""
    from talos.cv.scan import backlash_fix

    waypoints = _plan(params)
    prev = None
    for waypoint in waypoints:
        target = (waypoint.x_um, waypoint.y_um)
        steps = backlash_fix(prev, target, params["backlash_um"],
                             params["backlash_approach"])
        if steps and prev is not None:
            return waypoint, prev, target, steps
        prev = target
    raise AssertionError("this plan has no reversal to test")


def test_a_retry_repeats_the_take_up_and_lands_from_the_same_side(tmp_path):
    """A reversal needs its take-up move every time: the second attempt is
    a fresh approach, not a jump to the target.

    This is also what proves ``prev`` was not committed by the failed
    attempt: had it advanced to the failed target, the re-derived take-up
    would be empty and the re-issue would go straight to the target.
    """
    waypoint, _prev, target, steps = _a_reversal(BACKLASH)
    stage = _FaultyStage()
    # fail the first move that lands on that waypoint's own target
    first = {"done": False}
    real = stage.move_abs_um

    def fault_once(x, y, r_deg=None, speed=None, speed_pps=None):
        if (float(x), float(y)) == target and not first["done"]:
            first["done"] = True
            stage.move_calls += 1
            stage.attempts.append((float(x), float(y)))
            raise _timeout()
        return real(x, y, r_deg, speed, speed_pps)

    stage.move_abs_um = fault_once
    _scanner, result, _waypoints = _run(stage, tmp_path, BACKLASH)

    assert result.complete
    assert result.timing.retries == 1
    at = stage.attempts.index(target)          # the move that timed out
    assert stage.attempts[at - 1] == steps[0], "no take-up before the target"
    assert stage.attempts[at + 1] == steps[0], \
        "the re-issued approach must repeat the take-up first"
    assert stage.attempts[at + 2] == target


def test_prev_is_only_committed_by_a_landed_approach(tmp_path):
    """The landing invariant still holds over the whole run when one
    approach had to be re-issued."""
    waypoint, _prev, target, _steps = _a_reversal(BACKLASH)
    stage = _FaultyStage()
    real = stage.move_abs_um
    first = {"done": False}

    def fault_once(x, y, r_deg=None, speed=None, speed_pps=None):
        if (float(x), float(y)) == target and not first["done"]:
            first["done"] = True
            stage.move_calls += 1
            stage.attempts.append((float(x), float(y)))
            raise _timeout()
        return real(x, y, r_deg, speed, speed_pps)

    stage.move_abs_um = fault_once
    _scanner, result, _waypoints = _run(stage, tmp_path, BACKLASH)
    assert result.complete
    targets = {(round(w.x_um, 3), round(w.y_um, 3)) for w in _plan(BACKLASH)}
    landings = [p for p in stage.moves
                if (round(p[0], 3), round(p[1], 3)) in targets]
    assert len(landings) == len(targets)


def test_a_retried_waypoint_is_counted_once(tmp_path):
    """Progress and the manifest are per WAYPOINT: a retry must not appear
    as a second tile."""
    stage = _FaultyStage(move_faults={3: _timeout})
    scanner = GridScanner(stage, frame_source=None)
    progress: list = []
    scanner.sig_progress.connect(lambda done, total: progress.append(done))
    result = scanner.run(ScanParams(**PARAMS), tmp_path,
                         meta={"fov_um": FOV})
    assert result.visited == result.planned
    assert progress == list(range(1, result.planned + 1)), \
        "a retried waypoint reports progress once"
    import csv
    with open(result.manifest_path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == result.planned, "one manifest row per waypoint"


def test_an_unreachable_retry_still_reports_the_fault_not_a_finish(tmp_path):
    """The three-outcome rule survives the retry: a run that gave up on a
    fault is stopped early, never "done"."""
    stage = _FaultyStage(move_faults={i: _timeout for i in range(2, 40)})
    _scanner, result, _waypoints = _run(stage, tmp_path)
    assert not result.complete and result.stopped_early
    assert not result.aborted
    assert result.visited == 1 and result.planned > 1
    assert isinstance(result.message, str) and result.message


def test_a_plain_device_error_is_not_retried(tmp_path):
    """The base class means "something the adapter could not classify" —
    a driver bug, an AttributeError. Retrying it would hide the bug."""
    stage = _FaultyStage(move_faults={2: lambda: DeviceError("zolix.move_abs_um: boom")})
    _scanner, result, _waypoints = _run(stage, tmp_path)
    assert stage.move_calls == 2
    assert result.stopped_early
