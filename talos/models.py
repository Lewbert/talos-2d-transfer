"""Shared dataclasses used across the HAL, CV, and UI layers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StagePosition:
    """Zolix XYR position: raw pulses plus derived engineering units."""

    x_pulses: int = 0
    y_pulses: int = 0
    r_pulses: int = 0
    x_um: float = 0.0
    y_um: float = 0.0
    r_deg: float = 0.0

    @classmethod
    def from_telemetry(cls, payload: dict | None) -> "StagePosition":
        """Build from the manager's telemetry position.

        The proxies publish ``dataclasses.asdict`` — a plain dict — so
        consumers that expect an object (flake → stage mapping, the stage
        origin) must convert first: reading `.x_um` off the dict raised
        AttributeError.
        """
        raw = payload or {}
        return cls(
            x_pulses=int(raw.get("x_pulses", 0) or 0),
            y_pulses=int(raw.get("y_pulses", 0) or 0),
            r_pulses=int(raw.get("r_pulses", 0) or 0),
            x_um=float(raw.get("x_um", 0.0) or 0.0),
            y_um=float(raw.get("y_um", 0.0) or 0.0),
            r_deg=float(raw.get("r_deg", 0.0) or 0.0),
        )


@dataclass
class StageStatus:
    """Zolix ZC300 status (regs 30012-30015)."""

    x_moving: bool = False
    y_moving: bool = False
    r_moving: bool = False
    limit_x_pos: bool = False
    limit_x_neg: bool = False
    limit_y_pos: bool = False
    limit_y_neg: bool = False
    home_x: bool = False
    home_y: bool = False
    home_r: bool = False
    estop: bool = False
    alarm: bool = False

    @property
    def any_moving(self) -> bool:
        return self.x_moving or self.y_moving or self.r_moving


@dataclass
class FocusStatus:
    """Focus stage STATUS? response."""

    pos: int = 0
    mode: str = "IDLE"          # IDLE | CONT | TRAP | LIMIT
    v: float = 0.0
    spd: int = 0
    lim: bool = False           # any limit currently blocking a direction
    blocked_dir: str = "0"      # blocked limit direction: "0" | "+" | "-"
    slim: tuple[int, int] | None = None
    slim_on: bool = False       # SLIM enabled (bounds fetched separately)

    @property
    def is_idle(self) -> bool:
        return self.mode == "IDLE"


@dataclass
class ScanParams:
    """Grid-scan request, in stage coordinates (µm).

    ``x0_um/y0_um`` is the FIRST WAYPOINT (the operator jogs there and
    clicks "scan from here"); the rectangle grows from it by ``width_um``
    along ``x_dir`` and ``height_um`` along ``y_dir``. ``path`` selects the
    visit order over the resulting tile grid, ``start_axis`` which axis
    advances first, and ``serpentine`` whether the rows alternate direction
    (bi-directional) or all run the same way (uni-directional).
    """

    x0_um: float
    y0_um: float
    width_um: float
    height_um: float
    overlap: float = 0.10
    serpentine: bool = True
    slow_speed: bool = True
    capture: str = "camera"  # "camera" | "manual"
    path: str = "serpentine"       # serpentine | spiral | hilbert
    start_axis: str = "x"          # which axis advances first
    x_dir: int = 1                 # +1 / -1 — the direction the area grows
    y_dir: int = 1
    settle_ms: int = 200           # wait after the move, before the capture
    backlash_um: float = 0.0       # 0 = off (see cv.scan.backlash_fix)
    backlash_approach: int = 1     # the side every move finishes from
    return_to_start: bool = True   # drive back to (x0, y0) when done


@dataclass
class FlakeCandidate:
    """One detected flake, in pixels and (when calibrated) stage µm."""

    x_px: float
    y_px: float
    area_px2: float
    area_um2: float = 0.0
    x_um: float = 0.0
    y_um: float = 0.0
    score: float = 0.0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    color_class: str = "unknown"
    thumbnail: Any = None


@dataclass
class ObjectiveCalibration:
    """Effective calibration for one objective (composed at query time)."""

    objective_id: int
    name: str = ""
    nosepiece_position: int = -1
    um_per_px_x: float | None = None
    um_per_px_y: float | None = None
    jacobian_px_per_um: tuple[tuple[float, float], tuple[float, float]] | None = None
    focus_offset_steps: int | None = None
    source: str = "none"


@dataclass
class CalibrationEntry:
    """One calibration measurement stored in the calibration database."""

    objective_id: int
    kind: str  # "px_um" | "focus_offset"
    um_per_px_x: float | None = None
    um_per_px_y: float | None = None
    jacobian: list[list[float]] | None = None
    residual_px: float | None = None
    move_um: float | None = None
    focus_offset_steps: int | None = None
    ref_objective_id: int | None = None
    provenance: dict = field(default_factory=dict)
    notes: str = ""
    measured_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


@dataclass
class Job:
    """A command dispatched to a DeviceProxy."""

    job_id: int
    device: str
    method: str
    args: tuple = ()
    priority: int = 0
    created_at: float = field(default_factory=time.time)
