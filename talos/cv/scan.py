"""Grid scan over the XYR stage with camera capture.

Per waypoint: move (with the optional backlash take-up) → wait_idle →
READBACK position (never the commanded value) → settle → capture frame →
manifest row. Abortable: the stage is stopped and the dataset closed
cleanly at any point.

The frames come from a *frame source* (``grab(settle_s, timeout_s)``), not
from a camera object — see :mod:`talos.cv.frame_source` for why the scanner
must not fetch from a backend the camera worker owns.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from PySide6.QtCore import QObject, Signal

from talos.cv.frame_source import DEFAULT_TIMEOUT_S
from talos.hal.base import DeviceError
from talos.hal.base import StageSpeed
from talos.models import ScanParams

MANIFEST_HEADER = ["frame", "x_um", "y_um", "r_deg", "t_unix", "objective_id", "focus_pos"]

#: Path orders. ``serpentine`` is the only one with a bench history.
SERPENTINE = "serpentine"
SPIRAL = "spiral"
HILBERT = "hilbert"
PATHS = (SERPENTINE, SPIRAL, HILBERT)
PATH_LABELS = {
    SERPENTINE: "Serpentine",
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
class ScanResult:
    frames: list[Path] = field(default_factory=list)
    manifest_path: Path | None = None
    aborted: bool = False
    message: str = ""
    # Waypoints the stage visited but for which no frame was captured
    # (no frame source, or the fetch failed). Kept separate so a scan
    # can never report frames it does not have.
    missing: int = 0

    @property
    def captured(self) -> int:
        return len(self.frames)


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
                 thumb_width: int = 160):
        super().__init__(parent)
        self._stage = stage            # XYRStage (worker-thread owned)
        self._frames = frame_source    # grab(settle_s, timeout_s) | None
        self._thumb_width = int(thumb_width)
        self.abort_requested = False

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
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifest.csv"
        manifest = open(manifest_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(manifest)
        writer.writerow(MANIFEST_HEADER)
        meta = dict(meta or {})
        fov = tuple(meta.get("fov_um") or (1000.0, 1000.0))
        waypoints = self.plan(params, fov)

        settle_s = max(0.0, float(getattr(params, "settle_ms", 0) or 0) / 1000.0)
        backlash_um = max(0.0, float(getattr(params, "backlash_um", 0.0) or 0.0))
        approach = int(getattr(params, "backlash_approach", 1) or 1)
        # One speed, carried by the adapter's config (see
        # ``scan_speed_config``); the enum choice here is therefore
        # arbitrary, and SLOW is the one whose key that function sets.
        speed = StageSpeed.SLOW
        frame_shape: tuple | None = None
        prev: tuple[float, float] | None = None
        try:
            for waypoint in waypoints:
                if self.abort_requested:
                    result.aborted = True
                    result.message = "aborted by user"
                    break
                target = (waypoint.x_um, waypoint.y_um)
                for step in backlash_fix(prev, target, backlash_um, approach):
                    if self.abort_requested:
                        break
                    self._stage.move_abs_um(step[0], step[1], speed=speed)
                if self.abort_requested:
                    result.aborted = True
                    result.message = "aborted by user"
                    break
                self._stage.move_abs_um(target[0], target[1], speed=speed)
                prev = target
                self._stage.wait_idle(timeout_s=120.0)
                pos = self._stage.get_position()  # READBACK, not commanded
                frame = self._grab(settle_s)
                frame_path = frames_dir / f"frame_{waypoint.index:05d}.png"
                if frame is not None:
                    frame_shape = tuple(frame.shape)
                    cv2.imwrite(str(frame_path),
                                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    self.sig_frame.emit(waypoint.index, pos.x_um, pos.y_um,
                                        frame)
                    thumb = self._thumbnail(frame)
                    if thumb is not None:
                        self.sig_tile.emit(waypoint.index, pos.x_um, pos.y_um,
                                           thumb)
                    result.frames.append(frame_path)
                else:
                    # No frame source (or the fetch failed). The row is
                    # still written so the visited geometry is recorded —
                    # with an EMPTY frame column — but the waypoint is not
                    # counted as a frame: a touched empty file used to be
                    # reported as "Scan done: N frames" over a dataset that
                    # contained no images at all.
                    result.missing += 1
                    self.sig_log.emit(f"waypoint {waypoint.index}: no frame")
                writer.writerow([
                    frame_path.name if frame is not None else "",
                    f"{pos.x_um:.3f}", f"{pos.y_um:.3f}",
                    f"{pos.r_deg:.4f}", f"{time.time():.3f}",
                    meta.get("objective_id", ""),
                    meta.get("focus_pos", ""),
                ])
                manifest.flush()
                self.sig_progress.emit(waypoint.index + 1, len(waypoints))
        except DeviceError as exc:
            # An aborted move surfaces here as a device error (the
            # adapter refuses to submit once the abort is set) — report it
            # as an abort, not as a failure: the manifest and the UI both
            # read this flag.
            result.aborted = self.abort_requested
            result.message = str(exc)
            self.sig_log.emit(f"scan stopped: {exc}")
            try:
                self._stage.stop()
            except Exception:  # noqa: BLE001
                pass
        finally:
            manifest.close()
            # Park back at the ORIGIN — where the operator was standing
            # when they pressed the button, not the first waypoint: in the
            # corner modes those differ by half a field of view, and the
            # point of the return is to put the scope back where they left
            # it. Never on an abort — the point of an abort is that the
            # stage stops moving.
            if (params.return_to_start and not result.aborted
                    and not self.abort_requested and waypoints):
                try:
                    self._stage.move_abs_um(params.x0_um, params.y0_um,
                                            speed=speed)
                    self._stage.wait_idle(timeout_s=120.0)
                except Exception as exc:  # noqa: BLE001
                    self.sig_log.emit(f"return to start failed: {exc}")
            (out_dir / "meta.json").write_text(json.dumps(
                {"params": params.__dict__,
                 "meta": meta,
                 "n_frames": len(result.frames),
                 "n_missing": result.missing,
                 "frame_shape": list(frame_shape) if frame_shape else None,
                 "aborted": result.aborted}, indent=2), encoding="utf-8")
        result.manifest_path = manifest_path
        if not result.message and not result.aborted:
            result.message = "ok"
        self.sig_done.emit(result)
        return result
