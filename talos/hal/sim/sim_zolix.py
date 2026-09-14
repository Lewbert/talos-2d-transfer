"""Simulated Zolix ZC300 XYR stage: mirrors the real controller's behavior
(busy rejection, estop bit, limit flags, motion-state registers)."""

from __future__ import annotations

import time
from typing import Any

from talos.hal.base import (
    AxisMask,
    CommandRejectedError,
    DeviceBusyError,
    EStopError,
    LimitHitError,
    StageSpeed,
    XYRStage,
)
from talos.hal.sim._common import move_duration, now
from talos.models import StagePosition, StageStatus


class SimZolixXYRStage(XYRStage):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config or {})
        config = self.config
        self.um_per_pulse_xy = float(config.get("um_per_pulse_xy", 0.625))
        self.um_per_pulse_r = float(config.get("um_per_pulse_r", 0.00125))
        self.slow_speed_pps = int(config.get("slow_speed_pps", 500))
        self.fast_speed_pps = int(config.get("fast_speed_pps", 2000))
        self.slow_speed_r = int(config.get("slow_speed_r", 1000))
        self.fast_speed_r = int(config.get("fast_speed_r", 10000))
        self.latency_s = float(config.get("latency_s", 0.001))
        self._x = int(config.get("initial_x", 0))
        self._y = int(config.get("initial_y", 0))
        self._r = int(config.get("initial_r", 0))
        self._moving = {"x": False, "y": False, "r": False}
        self._done_at: dict[str, float] = {}
        self._estop = False
        self._limits = {k: False for k in ("x+", "x-", "y+", "y-", "r+", "r-")}
        self._alarm = False

    @property
    def device_id(self) -> str:
        return "zolix@sim"

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # Test helpers -------------------------------------------------------

    def trigger_estop(self) -> None:
        self._estop = True
        self._moving = dict.fromkeys(self._moving, False)

    def clear_estop(self) -> None:
        self._estop = False

    def set_limit(self, key: str, value: bool = True) -> None:
        if key not in self._limits:
            raise KeyError(key)
        self._limits[key] = value

    # Motion --------------------------------------------------------------

    def _tick(self) -> None:
        for axis, moving in list(self._moving.items()):
            if moving and now() >= self._done_at[axis]:
                self._moving[axis] = False

    def _ensure_idle(self) -> None:
        self._tick()
        if any(self._moving.values()):
            raise DeviceBusyError("SimZolix: axes are moving")
        if self._estop:
            raise EStopError("SimZolix: emergency stop active")

    def _start_move(self, axis: str, steps: int, speed_pps: int) -> None:
        self._moving[axis] = True
        self._done_at[axis] = now() + self.latency_s + move_duration(steps, speed_pps)

    def move_abs_pulses(self, x: int, y: int, r: int | None = None,
                        speed_pps: int | None = None) -> None:
        self._ensure_idle()
        speed_pps = speed_pps or self.slow_speed_pps
        targets = {"x": int(x), "y": int(y)}
        if r is not None:
            targets["r"] = int(r)
        for axis, target in targets.items():
            cur = getattr(self, f"_{axis}")
            if self._limits.get(f"{axis}+") and target > cur:
                raise LimitHitError(f"SimZolix: {axis}+ limit blocks this move")
            if self._limits.get(f"{axis}-") and target < cur:
                raise LimitHitError(f"SimZolix: {axis}- limit blocks this move")
        for axis, target in targets.items():
            cur = getattr(self, f"_{axis}")
            speed = self.slow_speed_r if axis == "r" else speed_pps
            self._start_move(axis, target - cur, speed)
            setattr(self, f"_{axis}", target)

    def move_abs_um(self, x_um: float, y_um: float, r_deg: float | None = None,
                    speed: StageSpeed = StageSpeed.SLOW) -> None:
        pps = self.fast_speed_pps if speed is StageSpeed.FAST else self.slow_speed_pps
        self.move_abs_pulses(
            round(x_um / self.um_per_pulse_xy),
            round(y_um / self.um_per_pulse_xy),
            round(r_deg / self.um_per_pulse_r) if r_deg is not None else None,
            speed_pps=pps,
        )

    def move_rel_um(self, dx_um: float, dy_um: float, dr_deg: float | None = None,
                    speed: StageSpeed = StageSpeed.SLOW) -> None:
        self.move_abs_um(
            self._x * self.um_per_pulse_xy + dx_um,
            self._y * self.um_per_pulse_xy + dy_um,
            self._r * self.um_per_pulse_r + dr_deg if dr_deg is not None else None,
            speed=speed,
        )

    def move_continuous(self, axis: str, direction: int, speed_pps: int | None = None) -> None:
        """Continuous move until stop (mirrors the real controller)."""
        self._ensure_idle()
        self._moving[axis] = now() + 10.0

    def stop_axis(self, axis: str) -> None:
        self._moving[axis] = False

    def home(self, axes: AxisMask = AxisMask.ALL, mode: int | None = None) -> None:
        self._ensure_idle()
        if axes & AxisMask.X:
            self._x = 0
        if axes & AxisMask.Y:
            self._y = 0
        if axes & AxisMask.Z:
            self._r = 0

    def stop(self) -> None:
        self._moving = dict.fromkeys(self._moving, False)
        if self._estop:
            pass  # estop is latched until cleared, like the real controller

    def save_parameters(self) -> None:
        if any(self._moving.values()):
            raise DeviceBusyError("SimZolix: cannot save while moving")

    def configure_motion(self, accel_pps2: int, speeds: dict[str, int]) -> None:
        self.accel_pps2 = int(accel_pps2)
        for key in ("x", "y"):
            if key in speeds:
                self.slow_speed_pps = int(speeds[key])
        if "r" in speeds:
            self.slow_speed_r = int(speeds["r"])

    # Status ---------------------------------------------------------------

    def get_status(self) -> StageStatus:
        self._tick()
        return StageStatus(
            x_moving=self._moving["x"],
            y_moving=self._moving["y"],
            r_moving=self._moving["r"],
            limit_x_pos=self._limits["x+"],
            limit_x_neg=self._limits["x-"],
            limit_y_pos=self._limits["y+"],
            limit_y_neg=self._limits["y-"],
            estop=self._estop,
            alarm=self._alarm,
        )

    def get_position(self) -> StagePosition:
        self._tick()
        return StagePosition(
            x_pulses=self._x, y_pulses=self._y, r_pulses=self._r,
            x_um=self._x * self.um_per_pulse_xy,
            y_um=self._y * self.um_per_pulse_xy,
            r_deg=self._r * self.um_per_pulse_r,
        )

    def check_estop(self) -> bool:
        return self._estop

    def wait_idle(self, timeout_s: float = 120.0, poll_s: float = 0.01) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self.get_status().any_moving:
                return
            time.sleep(poll_s)
        raise TimeoutError(f"SimZolix not idle after {timeout_s}s")
