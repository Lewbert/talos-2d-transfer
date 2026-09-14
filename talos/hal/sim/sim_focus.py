"""Simulated focus stage: mirrors the real firmware's state machine
(TRAP moves with EV:DONE, CONT via SPD, soft limits, inactivity events).

Backlash model (for closed-loop autofocus tests):
- The STATUS? position counter is the COMMANDED axis (the real firmware
  counts steps it commanded — it cannot know the load position).
- The LOAD axis (what the camera actually sees) lags the counter by
  ``backlash_steps`` in the current move direction: after a direction
  reversal the first B commanded steps take up gear slack without moving
  the load, so load = counter − dir×B. ``load_pos`` exposes it (test/sim
  only — the real driver has no such axis).
- ``pos_noise_steps`` jitters the reported counter (simulated lost steps /
  coupling slip).
Defaults are 0 — all existing sim invariants hold unchanged.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from talos.hal.base import (
    CommandRejectedError,
    DeviceBusyError,
    FocusStage,
    LimitHitError,
)
from talos.hal.sim._common import move_duration, now
from talos.models import FocusStatus


class SimFocusStage(FocusStage):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config or {})
        config = self.config
        self.max_speed = int(config.get("max_speed", 2000))
        self.clamp_lo = int(config.get("clamp_speed_lo", 10))
        self.clamp_hi = int(config.get("clamp_speed_hi", 5000))
        self.latency_s = float(config.get("latency_s", 0.001))
        self.backlash_steps = int(config.get("backlash_steps", 0))
        self.pos_noise_steps = int(config.get("pos_noise_steps", 0))
        self._pos = int(config.get("initial_pos", 0))
        self._mode = "IDLE"
        self._speed = 0
        self._cont_accum = 0.0  # fractional CONT steps (exact integration)
        self._done_at = 0.0
        self._events: list[str] = []
        self._blocked_dir = "0"
        self._rng = np.random.default_rng(7)
        # TRAP bookkeeping: live counter interpolation between start/target.
        self._trap_start_pos = self._pos
        self._trap_target = self._pos
        self._trap_start_t = 0.0
        self._trap_duration = 1e-9
        self._move_dir = 0  # direction of the last commanded move
        # Thread-safe: closed-loop tests render frames from a producer
        # thread that polls get_status() while the controller drives moves.
        # RLock: move_* calls set_speed() while already holding the lock.
        self._lock = threading.RLock()
        slim_on = bool(config.get("slim_on", False))
        self._slim = (
            (int(config.get("slim_min", -2_000_000)), int(config.get("slim_max", 2_000_000)))
            if slim_on else None
        )

    @property
    def device_id(self) -> str:
        return "focus@sim"

    # ------------------------------------------------------------------

    def connect(self) -> None:
        with self._lock:
            self._connected = True

    def disconnect(self) -> None:
        with self._lock:
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        if self._mode == "TRAP" and now() >= self._done_at:
            self._pos = self._trap_target
            self._mode = "IDLE"
            self._speed = 0
            self._events.append(f"EV:DONE:{self._pos}")

    def _check_window(self, target: int) -> None:
        if self._slim is not None and not (self._slim[0] <= target <= self._slim[1]):
            self._mode = "LIMIT"
            self._blocked_dir = "+" if target > self._slim[1] else "-"
            self._events.append(f"EV:LIM:{self._blocked_dir}:{self._pos}")
            raise LimitHitError(f"SimFocus: target {target} outside soft limits {self._slim}")

    def _start_trap(self, target: int) -> None:
        steps = target - self._pos
        self._move_dir = 0 if steps == 0 else (1 if steps > 0 else -1)
        self._trap_start_pos = self._pos
        self._trap_target = target
        self._trap_start_t = now()
        speed_used = abs(self._speed) or self.max_speed
        self._trap_duration = self.latency_s + move_duration(abs(steps), speed_used)
        self._mode = "TRAP"
        self._done_at = self._trap_start_t + self._trap_duration

    def move_rel(self, steps: int, speed: int | None = None) -> None:
        with self._lock:
            self._tick()
            if self._mode == "TRAP":
                raise DeviceBusyError("SimFocus: already executing a TRAP move")
            if speed is not None and speed > 0:
                self.set_speed(speed)
            target = self._pos + int(steps)
            self._check_window(target)
            self._start_trap(target)

    def move_abs(self, position: int, speed: int | None = None) -> None:
        with self._lock:
            self._tick()
            if self._mode == "TRAP":
                raise DeviceBusyError("SimFocus: already executing a TRAP move")
            if speed is not None and speed > 0:
                self.set_speed(speed)
            target = int(position)
            self._check_window(target)
            self._start_trap(target)

    def set_speed(self, steps_per_s: int) -> None:
        with self._lock:
            spd = int(steps_per_s)
            if spd == 0:
                self._mode = "IDLE"
                self._speed = 0
                self._cont_accum = 0.0
                self._events.append(f"EV:STOP:{self._pos}")
                return
            if not (self.clamp_lo <= abs(spd) <= self.clamp_hi):
                raise CommandRejectedError(f"SimFocus: speed {spd} outside clamp")
            self._speed = spd  # signed: negative = downward jog
            self._move_dir = 1 if spd > 0 else -1
            if self._mode == "IDLE":
                self._mode = "CONT"
                self._cont_started = now()
                self._cont_accum = 0.0

    def zero(self) -> None:
        with self._lock:
            self._pos = 0
            self._move_dir = 0

    def stop(self) -> None:
        with self._lock:
            self._mode = "IDLE"
            self._speed = 0
            self._cont_accum = 0.0
            self._events.append(f"EV:STOP:{self._pos}")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def load_pos(self) -> int:
        """The LOAD position (sim/test only): LIVE counter minus the
        backlash lag in the current direction. This is the axis the
        camera sees — it must track the interpolated counter during TRAP
        moves, or sweep frames would render at a stale position."""
        with self._lock:
            return int(round(self._raw_counter())) \
                - self._move_dir * self.backlash_steps

    def _raw_counter(self) -> float:
        if self._mode == "TRAP":
            frac = min(1.0, (now() - self._trap_start_t)
                       / max(self._trap_duration, 1e-9))
            return self._trap_start_pos + frac * (self._trap_target - self._trap_start_pos)
        return float(self._pos)

    def _counter_pos(self) -> int:
        pos = self._raw_counter()
        if self.pos_noise_steps:
            pos += self._rng.integers(-self.pos_noise_steps,
                                      self.pos_noise_steps + 1)
        return int(pos)

    def get_status(self) -> FocusStatus:
        with self._lock:
            self._tick()
            if self._mode == "CONT":
                dt = now() - getattr(self, "_cont_started", now())
                # fractional-step accumulator: int(speed×dt) truncation
                # would silently under-speed (int(50×0.04)=1, not 2)
                self._cont_accum += self._speed * dt
                steps = int(self._cont_accum)
                self._pos += steps
                self._cont_accum -= steps
                self._cont_started = now()
            return FocusStatus(
                pos=self._counter_pos(), mode=self._mode, v=float(self._speed),
                spd=self._speed, lim=self._blocked_dir not in ("0", ""),
                blocked_dir=self._blocked_dir, slim=self._slim,
            )

    def get_soft_limits(self) -> tuple[int, int]:
        return self._slim if self._slim is not None else (-2_000_000, 2_000_000)

    def set_soft_limits(self, lo: int, hi: int) -> None:
        if lo >= hi:
            raise CommandRejectedError(f"SimFocus: min={lo} >= max={hi}")
        self._slim = (int(lo), int(hi))

    def wait_idle(self, timeout_s: float = 120.0, poll_s: float = 0.01) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.get_status().is_idle:
                return
            time.sleep(poll_s)
        raise TimeoutError(f"SimFocus not idle after {timeout_s}s")

    def drain_events(self) -> list[str]:
        with self._lock:
            events, self._events = self._events, []
            return events
