"""Simulated SigmaKoki XYZ transfer stage: positions, speed levels,
limit gates, and settle-based wait_idle (like the real firmware, there
is no busy flag — a moving axis just changes position)."""

from __future__ import annotations

import time
from typing import Any

from talos.hal.base import (
    Axis,
    AxisMask,
    CommandRejectedError,
    Direction,
    LimitHitError,
    XYZStage,
)
from talos.hal.sim._common import move_duration, now
from talos.hal.devices.sigmakoki import SPEED_LEVEL_TO_HZ

_UM_PER_STEP = {Axis.X: 0.5, Axis.Y: 0.5, Axis.Z: 0.25}


class SimSigmaKokiXYZStage(XYZStage):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config or {})
        config = self.config
        self.latency_s = float(config.get("latency_s", 0.001))
        self._pos = {Axis.X: 0, Axis.Y: 0, Axis.Z: 0}
        self._levels = {Axis.X: 2, Axis.Y: 2, Axis.Z: 2}
        self._moving: dict[Axis, float] = {}   # axis -> until timestamp
        self._events: list[str] = []
        self._limits = {f"{a}{s}".lower(): False
                        for a in ("x", "y", "z") for s in ("+", "-")}

    @property
    def device_id(self) -> str:
        return "sigmakoki@sim"

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # Motion --------------------------------------------------------------

    def _tick(self) -> None:
        for axis in list(self._moving):
            if now() >= self._moving[axis]:
                del self._moving[axis]

    def _limit_key(self, axis: Axis, direction: Direction) -> str:
        sign = "+" if direction is Direction.POSITIVE else "-"
        return f"{axis.value}{sign}".lower()

    def _check_limit(self, axis: Axis, direction: Direction) -> None:
        if self._limits[self._limit_key(axis, direction)]:
            self._events.append(f"EV:LIM:{axis.value}{'+' if direction is Direction.POSITIVE else '-'}")
            raise LimitHitError(f"SimSigmaKoki: {axis.value} limit blocks this direction")

    def move(self, axis: Axis, direction: Direction, level: int) -> None:
        self._check_limit(axis, direction)
        self._levels[axis] = max(0, min(5, int(level)))
        # Continuous: keep advancing position until stopped.
        self._moving[axis] = now() + 10.0

    def step(self, axis: Axis, direction: Direction, steps: int) -> int:
        if steps <= 0:
            raise CommandRejectedError("STEP count must be positive")
        self._check_limit(axis, direction)
        sign = 1 if direction is Direction.POSITIVE else -1
        self._pos[axis] += sign * steps
        speed = SPEED_LEVEL_TO_HZ[self._levels[axis]]
        self._moving[axis] = now() + self.latency_s + move_duration(steps, speed)
        return steps

    def move_rel_um(self, axis: Axis, um: float, level: int) -> int:
        steps = round(um / _UM_PER_STEP[axis])
        if steps == 0:
            return 0
        direction = Direction.POSITIVE if steps > 0 else Direction.NEGATIVE
        return self.step(axis, direction, abs(steps))

    def set_speed(self, axis: Axis, level: int) -> None:
        self._levels[axis] = max(0, min(5, int(level)))

    def home(self) -> None:
        self._pos = dict.fromkeys(self._pos, 0)

    def stop(self, axes: AxisMask | None = None) -> None:
        if axes is None or axes is AxisMask.ALL:
            self._moving.clear()
        else:
            for axis in (Axis.X, Axis.Y, Axis.Z):
                if axes & getattr(AxisMask, axis.name):
                    self._moving.pop(axis, None)

    # Status ---------------------------------------------------------------

    def get_limits(self) -> dict[str, bool]:
        return dict(self._limits)

    def get_status(self) -> dict[str, str]:
        self._tick()
        return {
            "x": str(self._pos[Axis.X]),
            "y": str(self._pos[Axis.Y]),
            "z": str(self._pos[Axis.Z]),
            "xspd": str(self._levels[Axis.X]),
            "yspd": str(self._levels[Axis.Y]),
            "zspd": str(self._levels[Axis.Z]),
        }

    def get_position(self) -> dict[Axis, int]:
        self._tick()
        return dict(self._pos)

    def wait_idle(self, timeout_s: float = 300.0, poll_s: float = 0.01) -> None:
        deadline = time.monotonic() + timeout_s
        last: dict[Axis, int] | None = None
        stable = 0
        while time.monotonic() < deadline:
            pos = self.get_position()
            if last is not None and pos == last and not self._moving:
                stable += 1
                if stable >= 3:
                    return
            else:
                stable = 0
            last = pos
            time.sleep(poll_s)
        raise TimeoutError(f"SimSigmaKoki not settled after {timeout_s}s")

    def drain_events(self) -> list[str]:
        events, self._events = self._events, []
        return events
