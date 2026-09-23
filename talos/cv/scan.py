"""Grid scan over the XYR stage with camera capture.

Per waypoint: move (with the optional backlash take-up) → wait_idle →
READBACK position (never the commanded value) → settle → capture frame →
hand the frame and its row to the writer thread → next move. Abortable:
the stage is stopped and the dataset closed cleanly at any point.

The frames come from a *frame source* (``grab(settle_s, timeout_s)``), not
from a camera object — see :mod:`talos.cv.frame_source` for why the scanner
must not fetch from a backend the camera worker owns.

Nothing that can be done off the scan thread is done on it: the manifest,
the PNG encode and the thumbnail are the writer thread's
(:mod:`talos.cv.frame_writer`), and identification is the detection
worker's. What is left here is the part that needs the stage to be
somewhere: the plan, the moves, the readback, the settle and the capture.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from PySide6.QtCore import QObject, Signal

from talos.cv.frame_source import DEFAULT_TIMEOUT_S
from talos.cv.frame_writer import FrameWriter
from talos.hal.base import (DeviceBusyError, DeviceError, DeviceTimeoutError,
                            ProtocolError)
from talos.hal.base import StageSpeed
from talos.models import ScanParams

#: How many captured tiles may be waiting for the detector before the scan
#: pauses. Tiles are never dropped, so without a cap a scan whose tiles are
#: identified more slowly than they are captured queues whole frames — 6 MB
#: each at 1080p, 25 MB at 4K — until the process runs out of memory. Past
#: the cap the run waits for the detector instead: slower, and bounded.
MAX_PENDING_TILES = 8
_PENDING_POLL_S = 0.05
_PENDING_TIMEOUT_S = 120.0

#: Faults a RE-ISSUED approach can clear: the previous motion was still
#: running, a reply never came, a frame was lost. All three are the link or
#: the timing, not the hardware's answer.
#:
#: Deliberately not here: a limit switch, an e-stop, a dead port, a refused
#: command. Those are the controller saying no — the same rule the driver
#: already applies when it declines to retry a Modbus exception.
_RETRYABLE_FAULTS = (DeviceTimeoutError, DeviceBusyError, ProtocolError)
#: Attempts at ONE waypoint's approach, the first included.
_MAX_APPROACH_ATTEMPTS = 3
#: Pause before a re-issue, so a controller state that clears on its own
#: (a reply still in the buffer) has the chance to.
_RETRY_BACKOFF_S = 0.4
#: How long a re-issue waits for the PREVIOUS attempt's motion to end. A
#: safety limit, not politeness: the driver composes an absolute move from a
#: fresh position readback, so commanding one onto an axis that is still
#: travelling lands at ``target + (target − mid-flight position)`` — up to a
#: whole extra step. The frame is filed at the true readback, so the
#: manifest stays honest, but the tile grid gains a gap and a doubled tile
#: and nothing on screen says so. If this wait fails, the retry is ABANDONED.
_RETRY_SETTLE_S = 20.0

#: Path orders. ``serpentine`` is the only one with a bench history.
#: ``one_way`` is the same serpentine cells walked with every row in the
#: same direction — a first-class path here because the panel asks the
#: question once ("how should the stage walk the area?") rather than twice
#: ("path... order..."), and because a name is what the settings file and
#: the map legend need. ``serpentine=False`` still means the same thing,
#: so a stored configuration needs no migration.
SERPENTINE = "serpentine"
ONE_WAY = "one_way"
SPIRAL = "spiral"
HILBERT = "hilbert"
PATHS = (SERPENTINE, ONE_WAY, SPIRAL, HILBERT)
PATH_LABELS = {
    SERPENTINE: "Serpentine",
    ONE_WAY: "One-way",
    SPIRAL: "Spiral (experimental)",
    HILBERT: "Hilbert (experimental)",
}

#: Where the operator's position sits relative to the area they typed.
#: ``centre`` is the original behaviour and stays the default.
ORIGIN_CENTRE = "centre"
ORIGIN_CORNER_FIT = "corner_fit"
ORIGIN_CORNER_PITCH = "corner_pitch"
ORIGINS = (ORIGIN_CENTRE, ORIGIN_CORNER_FIT, ORIGIN_CORNER_PITCH)
ORIGIN_LABELS = {
    ORIGIN_CENTRE: "Centre",
    ORIGIN_CORNER_FIT: "Corner",
    ORIGIN_CORNER_PITCH: "Corner (pitch)",
}

#: The slowest the scan will be asked to run, in pulses per second.
SCAN_SPEED_MIN_PPS = 10

#: Waypoints that may lose their POSITION READBACK in a row before the run
#: gives up. The bench's serial link drops and corrupts the odd frame (a
#: truncated reply, a CRC mismatch — the driver retries both), and one that
#: survives the retries should cost a single tile rather than the whole
#: scan: the tile is recorded honestly as missing and the run carries on.
#: Several in a row is not a blip, it is a dead link, and walking the rest
#: of the plan would produce nothing.
MAX_CONSECUTIVE_LOSSES = 3


def scan_speed_config(stage_cfg: dict, speed_pps: float) -> dict:
    """A COPY of the stage config carrying ONE speed for the scan.

    Scanning and *go to sample* drive the stage in fixed-steps mode, where
    the controller generates its own acceleration and deceleration ramp —
    so there is no stability case for a slow/fast pair, and the two
    speeds the manual jogs use are simply not the scan's to borrow. The
    objective's ``stage_speed_multiplier`` is a manual-jog preference and
    does not apply here either.

    Both keys the adapter can read are set to the same number, so which
    one it happens to pick cannot matter. ``speed_pps`` of 0 leaves the
    config alone: the CLI benches pass their own and predate this.

    The live settings are never mutated — the scan gets this copy.
    """
    cfg = dict(stage_cfg)
    if float(speed_pps or 0.0) <= 0.0:
        return cfg
    pps = int(max(SCAN_SPEED_MIN_PPS, round(float(speed_pps))))
    for key in ("slow_speed_pps", "fast_speed_pps"):
        cfg[key] = pps
    return cfg


@dataclass
class Waypoint:
    index: int
    x_um: float
    y_um: float
    col: int = 0
    row: int = 0


def plan_steps(params, fov_um: tuple[float, float]) -> tuple[float, float]:
    """The grid pitch: FOV × (1 − overlap), per axis."""
    return (max(fov_um[0] * (1.0 - params.overlap), 1e-3),
            max(fov_um[1] * (1.0 - params.overlap), 1e-3))


@dataclass(frozen=True)
class PlanGeometry:
    """One grid: how many tiles, where the first one is, how far apart.

    ``first_x/first_y`` are the first tile's CENTRE and ``step_x/step_y``
    the signed pitch actually used, so the two corner modes can differ
    from the requested pitch without anything downstream needing to know
    why. Signed rather than absolute: the direction the area grows lives
    here, in one place, instead of being re-applied at every use.
    """

    cols: int
    rows: int
    first_x: float
    first_y: float
    step_x: float
    step_y: float

    @property
    def count(self) -> int:
        return self.cols * self.rows


def _axis_geometry(origin: str, a0: float, size: float, fov: float,
                   pitch: float, sign: int) -> tuple[int, float, float]:
    """(count, first tile centre, signed step) along one axis."""
    if size <= 0.0 or fov <= 0.0 or pitch <= 0.0:
        return 1, a0, sign * max(pitch, 1e-3)
    if origin == ORIGIN_CORNER_FIT:
        # The operator stands on the area's CORNER: the first tile is
        # inset half a field of view so its edge is on that corner, and
        # the last tile's far edge lands exactly on the far one. The
        # count is the minimum that covers the area, which means the
        # step comes out at or below the requested pitch — more overlap
        # than asked for at the ends, never less.
        count = max(1, math.ceil((size - fov) / pitch) + 1)
        if count <= 1:
            return 1, a0 + sign * size / 2.0, sign * pitch
        return count, a0 + sign * fov / 2.0, sign * (size - fov) / (count - 1)
    if origin == ORIGIN_CORNER_PITCH:
        # Same corner, but every tile sits at exactly the requested
        # pitch: the last CENTRE lands on the far edge and the union
        # overhangs it by up to one step. The area typed is the path the
        # stage walks rather than the region the frames cover.
        count = max(1, math.ceil((size - fov / 2.0) / pitch) + 1)
        if count <= 1:
            return 1, a0 + sign * size / 2.0, sign * pitch
        return count, a0 + sign * fov / 2.0, sign * pitch
    # ORIGIN_CENTRE: the operator's position is the centre of the first
    # tile and the area grows outward from it, so half a field of view of
    # coverage sits behind where they were standing.
    #
    # The count has to allow for that overhang, which the obvious
    # ``ceil(size / pitch)`` did not: half a frame of the covered span is
    # BEHIND the origin, so the far edge is short whenever the remainder
    # falls in the last half-frame. On the bench 5× a 2000 µm area was
    # covered only to 1912 µm — an 88 µm strip the operator asked for and
    # the scan never imaged. ``(n−1)·pitch + FOV ≥ area`` was true and
    # irrelevant: it measures the span, not where the span starts.
    count = max(1, math.ceil((size - fov / 2.0) / pitch) + 1)
    return count, a0, sign * pitch


def plan_geometry(params, fov_um: tuple[float, float]) -> PlanGeometry:
    """The grid, for the run AND for every preview of it.

    One function on purpose: a preview that disagrees with the plan the
    scanner walks is worse than no preview, and the corner modes make
    that easy to get wrong by hand (the tile count depends on the field
    of view, not just on the area).
    """
    step_x, step_y = plan_steps(params, fov_um)
    origin = str(getattr(params, "origin", None) or ORIGIN_CENTRE).lower()
    if origin not in ORIGINS:
        origin = ORIGIN_CENTRE
    sign_x = -1 if int(getattr(params, "x_dir", 1) or 1) < 0 else 1
    sign_y = -1 if int(getattr(params, "y_dir", 1) or 1) < 0 else 1
    cols, first_x, step_x = _axis_geometry(
        origin, float(params.x0_um), float(params.width_um),
        float(fov_um[0]), step_x, sign_x)
    rows, first_y, step_y = _axis_geometry(
        origin, float(params.y0_um), float(params.height_um),
        float(fov_um[1]), step_y, sign_y)
    return PlanGeometry(cols=cols, rows=rows, first_x=first_x,
                        first_y=first_y, step_x=step_x, step_y=step_y)


def grid_shape(params, fov_um: tuple[float, float]) -> tuple[int, int]:
    """The plan's (cols, rows).

    Exposed for the UI: a preview must be the plan ``GridScanner.plan``
    would actually walk (a preview that disagrees with the run is worse
    than none).
    """
    geometry = plan_geometry(params, fov_um)
    return geometry.cols, geometry.rows


# ----------------------------------------------------------------------
# Path orders — every one returns the SAME (col, row) cell set, only the
# visit order differs. ``tests/unit/test_scan_plan.py`` proves the
# coverage (each cell exactly once) for every order, including the
# degenerate one-row / one-column rectangles.
# ----------------------------------------------------------------------

def _serpentine_cells(nx: int, ny: int, serpentine: bool,
                      start_axis: str) -> list[tuple[int, int]]:
    if start_axis == "y":
        cells: list[tuple[int, int]] = []
        for col in range(nx):
            back = serpentine and col % 2 == 1
            rows = range(ny - 1, -1, -1) if back else range(ny)
            cells.extend((col, row) for row in rows)
        return cells
    cells = []
    for row in range(ny):
        back = serpentine and row % 2 == 1
        cols = range(nx - 1, -1, -1) if back else range(nx)
        cells.extend((col, row) for col in cols)
    return cells


def _spiral_cells(nx: int, ny: int) -> list[tuple[int, int]]:
    """Rectangle-spiral order: ring after ring, outermost first.

    Same tile centres as the serpentine — it starts at a corner and works
    its way inward. Cheap to verify (a closed ring visits each of its
    cells once), which is why it ships.
    """
    cells: list[tuple[int, int]] = []
    top, bottom, left, right = 0, ny - 1, 0, nx - 1
    while top <= bottom and left <= right:
        for col in range(left, right + 1):
            cells.append((col, top))
        top += 1
        for row in range(top, bottom + 1):
            cells.append((right, row))
        right -= 1
        if top <= bottom:
            for col in range(right, left - 1, -1):
                cells.append((col, bottom))
            bottom -= 1
        if left <= right:
            for row in range(bottom, top - 1, -1):
                cells.append((left, row))
            left += 1
    return cells


def _hilbert_cells(nx: int, ny: int) -> list[tuple[int, int]]:
    """Hilbert order on the next power-of-two square, clipped to the grid.

    The Hilbert curve visits every cell of the square exactly once, so the
    clipped order visits every cell of the rectangle exactly once. On a
    very asymmetric area (say 200 × 2) most of the square is discarded and
    the kept cells are scattered along the curve — complete, but the
    locality that makes Hilbert worth choosing is gone. The UI labels it
    experimental for that reason.
    """
    side = 1
    while side < max(nx, ny):
        side *= 2
    order: list[tuple[int, int]] = []
    for d in range(side * side):
        x, y, t = 0, 0, d
        s = 1
        while s < side:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            # rotate the quadrant
            if ry == 0:
                if rx == 1:
                    x, y = s - 1 - x, s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        if x < nx and y < ny:
            order.append((x, y))
    return order


def plan_cells(params, fov_um: tuple[float, float]) -> list[tuple[int, int]]:
    """The visit order as (col, row) cells — pure geometry, no µm."""
    nx, ny = grid_shape(params, fov_um)
    kind = (getattr(params, "path", None) or SERPENTINE).lower()
    start_axis = (getattr(params, "start_axis", None) or "x").lower()
    if kind == SPIRAL:
        cells = _spiral_cells(nx, ny)
    elif kind == HILBERT:
        cells = _hilbert_cells(nx, ny)
    elif kind == ONE_WAY:
        cells = _serpentine_cells(nx, ny, False, start_axis)
    else:
        cells = _serpentine_cells(nx, ny, bool(params.serpentine), start_axis)
    return cells


def plan_path(params, fov_um: tuple[float, float]) -> list[Waypoint]:
    """The waypoints, in visit order, in stage µm."""
    geometry = plan_geometry(params, fov_um)
    return [
        Waypoint(index=i,
                 x_um=geometry.first_x + col * geometry.step_x,
                 y_um=geometry.first_y + row * geometry.step_y,
                 col=col, row=row)
        for i, (col, row) in enumerate(plan_cells(params, fov_um))
    ]


def backlash_fix(prev: tuple[float, float] | None,
                 target: tuple[float, float],
                 backlash_um: float,
                 approach: int = 1) -> list[tuple[float, float]]:
    """The moves to run BEFORE the final approach to ``target``.

    Play in the drive means a position reached from −X and the same
    position reached from +X differ by the backlash: tiles would land on a
    grid shifted row-to-row. The cure is to finish every move from the
    SAME side — so an axis travelling the wrong way first backs off past
    the target and comes in. Returns [] when the approach is already
    correct on both axes (or backlash is off), and the caller then moves
    straight to the target.
    """
    if backlash_um <= 0 or prev is None or target is None:
        return []
    approach = -1 if int(approach or 1) < 0 else 1
    back = float(backlash_um) * approach
    x, y = target
    px, py = prev
    # The final motion into the target is in `approach` when we come from
    # the far side; otherwise back off first (and for an axis that does not
    # move at all, leave it alone — it keeps the play state it arrived with).
    fix_x = abs(x - px) > 1e-9 and (x - px) * approach < 0
    fix_y = abs(y - py) > 1e-9 and (y - py) * approach < 0
    if not (fix_x or fix_y):
        return []
    return [(x - back if fix_x else x, y - back if fix_y else y)]


@dataclass
class ScanTiming:
    """Where a run's wall-clock went (seconds, summed over the waypoints).

    Split at the two points the operator can act on: ``command_s`` is the
    serial cost of issuing the motion, ``travel_s`` is the stage actually
    getting there (a `speed_pps` question), and ``stopped_s`` is everything
    after the controller reported the axes stopped — the readback, the
    settle and the wait for a frame (a `settle_ms` and camera question).
    """

    command_s: float = 0.0
    travel_s: float = 0.0
    stopped_s: float = 0.0
    tiles: int = 0
    #: Approaches that had to be re-issued after a resumable fault. Its own
    #: field because it is what tells a run that fought the link apart from
    #: a run that was simply slow — the failed attempt's wall-clock is
    #: inside ``command_s``, where nothing else would distinguish it.
    retries: int = 0

    @property
    def total_s(self) -> float:
        return self.command_s + self.travel_s + self.stopped_s

    @property
    def summary(self) -> str:
        """``0.21 s/tile stopped · 1.42 s/tile travel`` — the bench number."""
        if self.tiles <= 0:
            return ""
        n = float(self.tiles)
        text = (f"{self.stopped_s / n:.2f} s/tile stopped · "
                f"{self.travel_s / n:.2f} s/tile travel · "
                f"{self.command_s / n:.2f} s/tile to command")
        if self.retries:
            text += f" · {self.retries} retried approach(es)"
        return text


@dataclass
class ScanResult:
    frames: list[Path] = field(default_factory=list)
    manifest_path: Path | None = None
    aborted: bool = False
    message: str = ""
    # Waypoints the stage visited but for which nothing could be recorded:
    # no frame (no frame source, or the fetch failed), or no position to
    # file a frame under (the readback failed after the driver's retries).
    # Kept separate so a scan can never report frames it does not have.
    missing: int = 0
    #: How many waypoints the plan had, and how many the loop got through.
    #: Together they are what makes "did this run finish?" answerable: a
    #: run that stopped early because a MOVE failed is not an abort (the
    #: operator did not press anything) and it is certainly not a
    #: completed scan — and reporting it as either was a real bug.
    planned: int = 0
    visited: int = 0
    #: The writer thread could not record something (a full disk, an encode
    #: failure). It has to be its own flag, not just a message: a failure on
    #: the LAST waypoint leaves ``visited == planned``, which would otherwise
    #: read as a completed scan over a truncated dataset.
    failed: bool = False
    timing: ScanTiming = field(default_factory=ScanTiming)

    @property
    def captured(self) -> int:
        return len(self.frames)

    @property
    def complete(self) -> bool:
        """Every planned waypoint was visited, nothing aborted, and every
        frame the run took is on disk."""
        return (not self.aborted and not self.failed and self.planned > 0
                and self.visited >= self.planned)

    @property
    def stopped_early(self) -> bool:
        """Ended before the last waypoint for a reason that is NOT the
        operator's: a device error, a settle timeout, a limit, a writer
        failure."""
        return not self.aborted and not self.complete


class GridScanner(QObject):
    sig_progress = Signal(int, int)     # waypoint index, total
    #: (index, x_um, y_um, frame) — the full captured frame and the readback
    #: position it was taken at. Both signals carry the position so neither
    #: depends on the other's delivery order.
    sig_frame = Signal(int, float, float, object)
    #: (index, x_um, y_um, thumbnail) — for the scan map.
    sig_tile = Signal(int, float, float, object)
    sig_done = Signal(object)           # ScanResult
    sig_log = Signal(str)

    def __init__(self, stage, frame_source=None, parent: QObject | None = None,
                 thumb_width: int = 160, pending_tiles_fn=None,
                 max_pending_tiles: int = MAX_PENDING_TILES):
        super().__init__(parent)
        self._stage = stage            # XYRStage (worker-thread owned)
        self._frames = frame_source    # grab(settle_s, timeout_s) | None
        self._thumb_width = int(thumb_width)
        #: How many tiles are still waiting for the detector. Supplied by
        #: the app (the detection engine's queue); None (the CLI benches)
        #: means no pacing.
        self._pending_fn = pending_tiles_fn
        self._max_pending = max(1, int(max_pending_tiles))
        self.abort_requested = False

    def _thumbnail(self, frame):
        """A small RGB copy for the scan map (the map fills with tiles as
        they arrive; a full 1080p frame per tile is megabytes of QImage)."""
        width = self._thumb_width
        if not width or frame is None:
            return None
        h, w = frame.shape[:2]
        if w <= width:
            return frame.copy()
        scale = width / float(w)
        return cv2.resize(frame, (width, max(1, int(round(h * scale)))),
                          interpolation=cv2.INTER_AREA)

    def request_abort(self) -> None:
        self.abort_requested = True
        try:
            self._stage.stop()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------

    def plan(self, params: ScanParams, fov_um: tuple[float, float]) -> list[Waypoint]:
        """The waypoint grid for ``params`` — see :func:`plan_path`."""
        return plan_path(params, fov_um)

    def _wait_for_the_detector(self) -> None:
        """Pace the run to the detection queue.

        The tiles are never dropped, so a scan that captures faster than
        the identification chain runs — denoise at 4K on a slow machine —
        would hold every waiting frame in memory. Past the cap the stage
        waits instead. The timer is a safety valve: a detector that never
        drains must not hang a run with the stage parked mid-plan.
        """
        if self._pending_fn is None:
            return
        deadline = time.monotonic() + _PENDING_TIMEOUT_S
        while not self.abort_requested:
            try:
                pending = int(self._pending_fn() or 0)
            except Exception:  # noqa: BLE001 - pacing is never fatal
                return
            if pending < self._max_pending:
                return
            if time.monotonic() >= deadline:
                self.sig_log.emit(
                    f"{pending} tiles are still waiting to be identified — "
                    f"carrying on")
                return
            time.sleep(_PENDING_POLL_S)

    def _approach(self, waypoint, prev, target, backlash_um, approach,
                  speed, timing) -> float:
        """Move through the take-up steps to ``target``, retrying an
        approach a RESUMABLE fault interrupted. Returns the idle timestamp.

        The take-up is re-derived from ``prev`` on every attempt, which is
        idempotent exactly BECAUSE the caller commits ``prev``: a failed
        attempt must not advance it, or the next attempt's take-up would be
        computed from a position the stage never reached.

        Only the move/wait pair is retried — never the readback or the
        capture below it. Those have their own, different tolerance (one
        lost tile), and re-running them would spend the settle window twice.
        """
        steps = backlash_fix(prev, target, backlash_um, approach)
        for attempt in range(1, _MAX_APPROACH_ATTEMPTS + 1):
            t_attempt = time.monotonic()
            settling = False
            try:
                for step in steps:
                    if self.abort_requested:
                        raise DeviceError("scan aborted")
                    self._stage.move_abs_um(step[0], step[1], speed=speed)
                    # The take-up is a move like any other: the driver
                    # refuses to command one onto a moving axis, so the
                    # next one waits for it to land.
                    self._stage.wait_idle(timeout_s=120.0)
                if self.abort_requested:
                    raise DeviceError("scan aborted")
                self._stage.move_abs_um(target[0], target[1], speed=speed)
                t_commanded = time.monotonic()
                settling = True        # a fault from here is not a move fault
                self._stage.wait_idle(timeout_s=120.0)
                t_idle = time.monotonic()
            except DeviceError as exc:
                # A failed attempt's wall-clock is real time the stage was
                # neither commanded nor travelling, so it goes in the
                # command phase; ``timing.retries`` is what tells that apart
                # from a genuinely slow run.
                timing.command_s += time.monotonic() - t_attempt
                if (self.abort_requested
                        or (settling and isinstance(exc, DeviceTimeoutError))
                        or not isinstance(exc, _RETRYABLE_FAULTS)
                        or attempt >= _MAX_APPROACH_ATTEMPTS):
                    # A stage that never reported idle is either jammed or
                    # genuinely still travelling: re-issuing is the one
                    # action guaranteed to be wrong, so that timeout is not
                    # retried (see _RETRY_SETTLE_S's note).
                    raise
                self.sig_log.emit(
                    f"waypoint {waypoint.index}: {exc} — retrying the "
                    f"approach ({attempt + 1}/{_MAX_APPROACH_ATTEMPTS})")
                time.sleep(_RETRY_BACKOFF_S)
                if self.abort_requested:
                    raise DeviceError("scan aborted")
                self._settle_before_retry(waypoint)
                timing.retries += 1
                continue
            timing.command_s += t_commanded - t_attempt
            timing.travel_s += t_idle - t_commanded
            return t_idle
        raise DeviceError("scan aborted")        # unreachable; the loop raises

    def _settle_before_retry(self, waypoint) -> None:
        """Wait out a still-running previous attempt, or give up the retry."""
        try:
            self._stage.wait_idle(timeout_s=_RETRY_SETTLE_S)
        except DeviceError as exc:
            self.sig_log.emit(
                f"waypoint {waypoint.index}: {exc} — the stage is still "
                f"moving after the fault; not re-issuing the approach")
            raise

    def _grab(self, settle_s: float):
        if self._frames is None:
            return None
        try:
            item = self._frames.grab(settle_s, DEFAULT_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 - a capture never kills a scan
            self.sig_log.emit(f"capture failed: {exc}")
            return None
        return item[0] if item is not None else None

    def run(self, params: ScanParams, out_dir: Path,
            meta: dict | None = None) -> ScanResult:
        self.abort_requested = False
        result = ScanResult()
        out_dir = Path(out_dir)
        meta = dict(meta or {})
        fov = tuple(meta.get("fov_um") or (1000.0, 1000.0))
        waypoints = self.plan(params, fov)
        result.planned = len(waypoints)

        writer = FrameWriter(out_dir, meta)
        writer.start()

        settle_s = max(0.0, float(getattr(params, "settle_ms", 0) or 0) / 1000.0)
        backlash_um = max(0.0, float(getattr(params, "backlash_um", 0.0) or 0.0))
        approach = int(getattr(params, "backlash_approach", 1) or 1)
        # One speed, carried by the adapter's config (see
        # ``scan_speed_config``); the enum choice here is therefore
        # arbitrary, and SLOW is the one whose key that function sets.
        speed = StageSpeed.SLOW
        frame_shape: tuple | None = None
        prev: tuple[float, float] | None = None
        lost_in_a_row = 0
        try:
            for waypoint in waypoints:
                if self.abort_requested:
                    result.aborted = True
                    result.message = "aborted by user"
                    break
                self._wait_for_the_detector()
                if self.abort_requested:
                    result.aborted = True
                    result.message = "aborted by user"
                    break
                target = (waypoint.x_um, waypoint.y_um)
                # The approach may be retried; the readback and the capture
                # below it are reached only once the stage is there.
                t_idle = self._approach(waypoint, prev, target, backlash_um,
                                        approach, speed, result.timing)
                prev = target          # committed only by a LANDED approach
                try:
                    pos = self._stage.get_position()  # READBACK, not commanded
                except DeviceError as exc:
                    # The stage is where it should be, but the controller
                    # did not say so — and the manifest may not invent a
                    # position. The row is written with an EMPTY position
                    # and no frame (the same honesty as the missing-frame
                    # case) and the run carries on: one bad exchange costs
                    # one tile, not the dataset.
                    lost_in_a_row += 1
                    result.missing += 1
                    self.sig_log.emit(
                        f"waypoint {waypoint.index}: no position readback "
                        f"({exc}) — recorded as missing")
                    writer.submit_missing(waypoint.index, None, time.time())
                    result.visited = waypoint.index + 1
                    self.sig_progress.emit(waypoint.index + 1, len(waypoints))
                    if lost_in_a_row >= MAX_CONSECUTIVE_LOSSES:
                        result.message = (
                            f"{lost_in_a_row} waypoints in a row could not be "
                            f"read back ({exc})")
                        self.sig_log.emit(f"scan stopped: {result.message}")
                        break
                    continue
                lost_in_a_row = 0
                frame = self._grab(settle_s)
                result.timing.stopped_s += time.monotonic() - t_idle
                result.timing.tiles += 1
                if frame is not None:
                    # The signals are emitted HERE, on the scan thread —
                    # one emitter, and the ordering against sig_done stays
                    # trivially right. The expensive half (the encode, the
                    # manifest row and its flush) is the writer's.
                    frame_shape = tuple(frame.shape)
                    self.sig_frame.emit(waypoint.index, pos.x_um, pos.y_um,
                                        frame)
                    thumb = self._thumbnail(frame)
                    if thumb is not None:
                        self.sig_tile.emit(waypoint.index, pos.x_um, pos.y_um,
                                           thumb)
                    writer.submit_frame(
                        waypoint.index, (pos.x_um, pos.y_um, pos.r_deg), frame,
                        time.time())
                else:
                    # No frame source (or the fetch failed). The row is
                    # still written so the visited geometry is recorded —
                    # with an EMPTY frame column — but the waypoint is not
                    # counted as a frame: a touched empty file used to be
                    # reported as "Scan done: N frames" over a dataset that
                    # contained no images at all.
                    result.missing += 1
                    self.sig_log.emit(f"waypoint {waypoint.index}: no frame")
                    writer.submit_missing(
                        waypoint.index, (pos.x_um, pos.y_um, pos.r_deg),
                        time.time())
                if writer.error is not None:
                    # A disk that cannot take the frames is a fault like any
                    # other: the run stops and says why, rather than walking
                    # the rest of the plan recording nothing.
                    raise DeviceError(f"frame writer: {writer.error}")
                result.visited = waypoint.index + 1
                self.sig_progress.emit(waypoint.index + 1, len(waypoints))
        except DeviceError as exc:
            # An aborted move surfaces here as a device error (the
            # adapter refuses to submit once the abort is set) — report it
            # as an abort, not as a failure: the manifest and the UI both
            # read this flag.
            #
            # But a device error WITHOUT an abort is a real fault — a
            # move timeout, a soft limit, a serial glitch — and it must
            # not read as a completed scan. ``aborted`` stays False (the
            # operator asked for nothing) and ``visited < planned`` is
            # what tells the UI the run did not finish; ``message`` says
            # why. Reporting this as "Scan done" hid the fault behind a
            # success message, which is the worst place for one.
            result.aborted = self.abort_requested
            result.message = str(exc)
            self.sig_log.emit(f"scan stopped: {exc}")
            try:
                self._stage.stop()
            except Exception:  # noqa: BLE001
                pass
        finally:
            # Park back at the ORIGIN — where the operator was standing
            # when they pressed the button, not the first waypoint: in the
            # corner modes those differ by half a field of view, and the
            # point of the return is to put the scope back where they left
            # it. Never on an abort — the point of an abort is that the
            # stage stops moving.
            #
            # The travel happens FIRST so the writer drains its queue while
            # the stage is moving; the join then costs almost nothing.
            if (params.return_to_start and not result.aborted
                    and not self.abort_requested and waypoints):
                try:
                    self._stage.move_abs_um(params.x0_um, params.y0_um,
                                            speed=speed)
                    self._stage.wait_idle(timeout_s=120.0)
                except Exception as exc:  # noqa: BLE001
                    self.sig_log.emit(f"return to start failed: {exc}")
            # Nothing downstream may run before this: scan_output re-reads
            # the manifest from disk, and meta.json records how many frames
            # the run produced.
            writer.close()
            result.failed = writer.error is not None
            if result.failed and not result.message:
                result.message = f"frame writer: {writer.error}"
            result.frames = writer.frames
            (out_dir / "meta.json").write_text(json.dumps(
                {"params": params.__dict__,
                 "meta": meta,
                 "n_frames": len(result.frames),
                 "n_missing": result.missing,
                 "n_planned": result.planned,
                 "n_visited": result.visited,
                 "frame_shape": list(frame_shape) if frame_shape else None,
                 "aborted": result.aborted,
                 "failed": result.failed,
                 "timing_s": {"command": round(result.timing.command_s, 3),
                              "travel": round(result.timing.travel_s, 3),
                              "stopped": round(result.timing.stopped_s, 3),
                              "retries": result.timing.retries},
                 "message": result.message}, indent=2), encoding="utf-8")
        result.manifest_path = writer.manifest_path
        if not result.message and not result.aborted:
            result.message = "ok"
        self.sig_done.emit(result)
        return result
