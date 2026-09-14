"""Simulated Yudian AI-828 temperature controller with a simple thermal
lag model (PV approaches SV exponentially)."""

from __future__ import annotations

import time
from typing import Any

from talos.hal.base import CommandRejectedError, TemperatureController
from talos.hal.sim._common import now


class SimYudianTempController(TemperatureController):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config or {})
        config = self.config
        self.safety_lo_c = float(config.get("safety_lo_c", -100.0))
        self.safety_hi_c = float(config.get("safety_hi_c", 400.0))
        self.tau_s = float(config.get("thermal_tau_s", 5.0))
        self._pv = float(config.get("initial_pv_c", 25.0))
        self._sv = float(config.get("initial_sv_c", 25.0))
        self._dpt = 1
        self._last_tick = now()

    @property
    def device_id(self) -> str:
        return "yudian@sim"

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def stop(self) -> None:
        pass

    def _tick(self) -> None:
        t = now()
        dt = t - self._last_tick
        self._last_tick = t
        if dt > 0 and self.tau_s > 0:
            self._pv += (self._sv - self._pv) * (1 - 2.718281828 ** (-dt / self.tau_s))
        self._pv = round(self._pv, 1)

    def read_pv(self) -> float:
        self._tick()
        return self._pv

    def read_sv(self) -> float:
        return self._sv

    def set_sv(self, temp_c: float) -> None:
        temp_c = float(temp_c)
        if not (self.safety_lo_c <= temp_c <= self.safety_hi_c):
            raise CommandRejectedError(
                f"Setpoint {temp_c}°C outside safety range "
                f"[{self.safety_lo_c}, {self.safety_hi_c}]°C"
            )
        self._sv = round(temp_c, 1)

    def read_output_percent(self) -> float:
        return max(0.0, min(100.0, (self._sv - self._pv) * 10.0))

    def read_decimal_point(self) -> int:
        return self._dpt
