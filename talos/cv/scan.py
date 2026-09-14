"""Serpentine raster/grid scan over the XYR stage with camera capture.

Per waypoint: move → wait_idle → READBACK position (never the commanded
value) → capture frame → manifest row. Abortable: the stage is stopped and
the dataset closed cleanly at any point.
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

from talos.hal.base import DeviceError
from talos.hal.base import StageSpeed
from talos.models import ScanParams

MANIFEST_HEADER = ["frame", "x_um", "y_um", "r_deg", "t_unix", "objective_id", "focus_pos"]


def scale_scan_speed_config(stage_cfg: dict, multiplier: float) -> dict:
    """A COPY of the stage config with the slow/fast speeds scaled by the
    active objective's stage multiplier (floor 10 pps; the multiplier is
    clamped to [0.05, 1.0] — manual jogs apply the same scale via the
    ActionResolver). The scan must never mutate the live settings."""
    cfg = dict(stage_cfg)
    mult = max(0.05, min(1.0, float(multiplier or 1.0)))
    for key in ("slow_speed_pps", "fast_speed_pps"):
        base = float(cfg.get(key) or 0.0)
        if base > 0:
            cfg[key] = int(max(10, round(base * mult)))
    return cfg


@dataclass
class Waypoint:
    index: int
    x_um: float
    y_um: float


@dataclass
class ScanResult:
    frames: list[Path] = field(default_factory=list)
    manifest_path: Path | None = None
    aborted: bool = False
    message: str = ""
    # Waypoints the stage visited but for which no frame was captured
    # (no camera attached, or the fetch failed). Kept separate so a scan
    # can never report frames it does not have.
    missing: int = 0

    @property
    def captured(self) -> int:
        return len(self.frames)


class GridScanner(QObject):
    sig_progress = Signal(int, int)     # waypoint index, total
    sig_frame = Signal(object)          # captured frame (np.ndarray)
    sig_done = Signal(object)           # ScanResult
    sig_log = Signal(str)

    def __init__(self, stage, camera=None, parent: QObject | None = None):
        super().__init__(parent)
        self._stage = stage            # XYRStage (worker-thread owned)
        self._camera = camera          # Camera with fetch(), or None
        self.abort_requested = False

    def request_abort(self) -> None:
        self.abort_requested = True
        try:
            self._stage.stop()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------

    def plan(self, params: ScanParams, fov_um: tuple[float, float]) -> list[Waypoint]:
        """Serpentine waypoint grid: step = FOV × (1 − overlap)."""
        step_x = max(fov_um[0] * (1.0 - params.overlap), 1e-3)
        step_y = max(fov_um[1] * (1.0 - params.overlap), 1e-3)
        nx = max(1, math.ceil(params.width_um / step_x))
        ny = max(1, math.ceil(params.height_um / step_y))
        waypoints: list[Waypoint] = []
        for row in range(ny):
            serpentine = params.serpentine and row % 2 == 1
            cols = range(nx - 1, -1, -1) if serpentine else range(nx)
            for col in cols:
                waypoints.append(Waypoint(
                    index=len(waypoints),
                    x_um=params.x0_um + col * step_x,
                    y_um=params.y0_um + row * step_y))
        return waypoints

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
        waypoints = self.plan(params, meta.get("fov_um", (1000.0, 1000.0))
                              if meta else (1000.0, 1000.0))

        # Real-time ops run at 1080p — 4K is laggy with unstable framerate
        # on the Axiocam 208 (backend handles the mid-stream switch).
        if self._camera is not None and hasattr(self._camera, "set_property"):
            try:
                self._camera.set_property("resolution", 1)
            except Exception:  # noqa: BLE001
                pass

        speed = StageSpeed.SLOW if params.slow_speed else StageSpeed.FAST
        try:
            for waypoint in waypoints:
                if self.abort_requested:
                    result.aborted = True
                    result.message = "aborted by user"
                    break
                self._stage.move_abs_um(waypoint.x_um, waypoint.y_um, speed=speed)
                self._stage.wait_idle(timeout_s=120.0)
                pos = self._stage.get_position()  # READBACK, not commanded
                frame = self._camera.fetch(timeout_ms=3000.0) if self._camera else None
                frame_path = frames_dir / f"frame_{waypoint.index:05d}.png"
                if frame is not None:
                    cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    self.sig_frame.emit(frame)
                    result.frames.append(frame_path)
                else:
                    # No camera attached (or the fetch failed). The row is
                    # still written so the visited geometry is recorded —
                    # with an EMPTY frame column — but the waypoint is not
                    # counted as a frame: a touch()-ed empty file used to
                    # be reported as "Scan done: N frames" over a dataset
                    # that contained no images at all.
                    result.missing += 1
                    self.sig_log.emit(f"waypoint {waypoint.index}: no frame")
                writer.writerow([
                    frame_path.name if frame is not None else "",
                    f"{pos.x_um:.3f}", f"{pos.y_um:.3f}",
                    f"{pos.r_deg:.4f}", f"{time.time():.3f}",
                    meta.get("objective_id", "") if meta else "",
                    meta.get("focus_pos", "") if meta else "",
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
            (out_dir / "meta.json").write_text(json.dumps(
                {"params": params.__dict__,
                 "meta": meta or {},
                 "n_frames": len(result.frames),
                 "n_missing": result.missing,
                 "aborted": result.aborted}, indent=2), encoding="utf-8")
        result.manifest_path = manifest_path
        if not result.message and not result.aborted:
            result.message = "ok"
        self.sig_done.emit(result)
        return result
