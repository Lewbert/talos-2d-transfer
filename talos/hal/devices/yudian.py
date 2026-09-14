"""Yudian AI-828 temperature controller driver (Modbus RTU).

Protocol facts (researched register map + tested reference deployment):
- 9600 8N1, slave 1 (configurable).
- Reads fn 0x03 (holding), writes fn 0x06 (single).
- Registers (1-based): SV write = 1 (40001), dPt = 13, PV = 75,
  SV read = 76, MV = 77 (low byte = output %).
- Scaling: raw / 10^dPt (dPt cached at connect; default 1 → 0.1 °C).
- Values are signed 16-bit two's complement.
- Setpoint writes are validated against the configured safety range
  (default -100...+400 °C) and verified by readback.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import serial

from talos.hal.base import (
    CommandRejectedError,
    DeviceConnectionError,
    DeviceTimeoutError,
    NotConnectedError,
    ProtocolError,
    TemperatureController,
)
from talos.protocols.modbus_rtu import (
    FrameError,
    ModbusException,
    READ_HOLDING,
    build_read,
    build_write_single,
    parse_read_response,
    parse_write_response,
    to_signed16,
)

logger = logging.getLogger(__name__)

REG_SV_WRITE = 1
REG_DPT = 13
REG_PV = 75
REG_SV_READ = 76
REG_MV = 77


class YudianTempController(TemperatureController):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.port_name = config.get("port", "COM5")
        self.slave = int(config.get("slave_address", 1))
        self.timeout_s = float(config.get("timeout_s", 0.5))
        self.safety_lo_c = float(config.get("safety_lo_c", -100.0))
        self.safety_hi_c = float(config.get("safety_hi_c", 400.0))
        # Test injection point: a callable returning a serial-port-like object.
        self._serial_factory = config.get("serial_factory")
        self._ser: serial.Serial | None = None
        self._dpt: int = 1

    @property
    def device_id(self) -> str:
        return f"yudian@{self.port_name}"

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        factory = self._serial_factory or serial.Serial
        try:
            self._ser = factory(
                port=self.port_name,
                baudrate=9600,
                bytesize=8,
                parity="N",
                stopbits=1,
                timeout=self.timeout_s,
            )
        except (OSError, serial.SerialException) as exc:
            raise DeviceConnectionError(f"Yudian on {self.port_name}: {exc}") from exc
        self._connected = True  # probes below need is_connected
        try:
            self._dpt = self.read_decimal_point()
            self.read_pv()  # liveness probe
        except Exception:
            self._connected = False
            self._close_port()
            raise
        logger.info("Connected to Yudian AI-828 on %s (dPt=%d)", self.port_name, self._dpt)

    def disconnect(self) -> None:
        self._connected = False
        self._close_port()

    def _close_port(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:  # noqa: BLE001
                pass
            self._ser = None

    @property
    def is_connected(self) -> bool:
        return bool(
            self._connected
            and self._ser is not None
            and getattr(self._ser, "is_open", False)
        )

    def stop(self) -> None:
        """No motion to halt; the temperature controller is passive."""
        pass

    # ------------------------------------------------------------------
    # Values
    # ------------------------------------------------------------------

    def _scale(self, raw: int) -> float:
        return raw / (10 ** self._dpt)

    def _to_raw(self, temp_c: float) -> int:
        return round(temp_c * (10 ** self._dpt))

    def read_pv(self) -> float:
        raw = self._read_reg(REG_PV)
        return self._scale(to_signed16(raw))

    def read_sv(self) -> float:
        raw = self._read_reg(REG_SV_READ)
        return self._scale(to_signed16(raw))

    def read_output_percent(self) -> float:
        raw = self._read_reg(REG_MV)
        return float(raw & 0xFF)

    def read_decimal_point(self) -> int:
        return self._read_reg(REG_DPT)

    def set_sv(self, temp_c: float) -> None:
        """Clamp to the safety range, write, and verify by readback."""
        temp_c = float(temp_c)
        if not (self.safety_lo_c <= temp_c <= self.safety_hi_c):
            raise CommandRejectedError(
                f"Setpoint {temp_c}°C outside safety range "
                f"[{self.safety_lo_c}, {self.safety_hi_c}]°C"
            )
        raw = self._to_raw(temp_c) & 0xFFFF
        self._write_reg(REG_SV_WRITE, raw)
        readback = self.read_sv()
        if abs(readback - temp_c) > 0.1:
            raise CommandRejectedError(
                f"Yudian setpoint readback mismatch: requested {temp_c}°C, "
                f"read back {readback}°C"
            )

    # ------------------------------------------------------------------
    # Modbus I/O
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise NotConnectedError(f"Yudian on {self.port_name} is not connected")

    def _read_reg(self, reg: int) -> int:
        self._require_connected()
        resp = self._transact(build_read(self.slave, READ_HOLDING, reg, 1))
        try:
            values = parse_read_response(resp, self.slave, READ_HOLDING)
            return values[0]
        except ModbusException as exc:
            raise CommandRejectedError(f"Yudian read reg {reg}: {exc}") from exc
        except FrameError as exc:
            raise ProtocolError(f"Yudian read reg {reg}: {exc}") from exc

    def _write_reg(self, reg: int, value: int) -> None:
        self._require_connected()
        resp = self._transact(build_write_single(self.slave, reg, value))
        try:
            parse_write_response(resp, self.slave, 0x06)
        except ModbusException as exc:
            raise CommandRejectedError(f"Yudian write reg {reg}: {exc}") from exc
        except FrameError as exc:
            raise ProtocolError(f"Yudian write reg {reg}: {exc}") from exc

    def _transact(self, frame: bytes) -> bytes:
        try:
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
            self._ser.write(frame)
            self._ser.flush()
            response = bytearray()
            deadline = time.monotonic() + self.timeout_s * 2
            while time.monotonic() < deadline:
                chunk = self._ser.read(64)
                if chunk:
                    response.extend(chunk)
                    if self._response_len(frame[1], response) is not None:
                        break
                else:
                    time.sleep(0.005)
            if not response:
                raise DeviceTimeoutError(f"No Modbus reply from Yudian on {self.port_name}")
            return bytes(response)
        except OSError as exc:
            self._connected = False
            raise NotConnectedError(f"Yudian serial error: {exc}") from exc

    @staticmethod
    def _response_len(fn: int, buf: bytearray) -> int | None:
        if len(buf) < 2:
            return None
        if buf[1] == fn | 0x80:
            return 5
        if fn == 0x03:
            if len(buf) < 3:
                return None
            return 3 + buf[2] + 2
        return 8
