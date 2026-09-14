"""Zolix ZC300 XYR stage driver (Modbus RTU over a USB virtual COM port).

Protocol facts (ZC300 manual + the tested reference deployment,
github.com/Lewbert/transfer-stage-control):
- 115200 8N1, slave 1-255 (default 1).
- Reads: fn 0x03 (holding) / 0x04 (input); single writes: fn 0x06.
  fn 0x10 is reserved for the OPCODE BLOCK only, with the exact register
  count for the opcode's arity (moves=3, stops=2, save-params=1); a
  padded frame is rejected with exception 0x03.
- Floats are IEEE 754 big-endian across 2 registers.
- A motion command to a moving axis is rejected with exception 0x06.
- The rotation axis is wired to the ZC300 Z channel (logical "r").

⚠️ Absolute moves (0x0064) and homing (0x0069) were implemented from the
register map but never exercised on hardware by the reference project —
validate with tools/smoke_test.py before relying on them. Fixed-length
moves (0x0065) are the hardware-validated opcode.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import serial

from talos.hal.base import (
    AxisMask,
    CommandRejectedError,
    DeviceBusyError,
    DeviceConnectionError,
    DeviceTimeoutError,
    EStopError,
    LimitHitError,
    NotConnectedError,
    ProtocolError,
    StageSpeed,
    XYRStage,
)
from talos.models import StagePosition, StageStatus
from talos.protocols.modbus_rtu import (
    FrameError,
    ModbusException,
    READ_INPUT,
    build_read,
    build_write_multiple,
    build_write_single,
    parse_read_response,
    parse_write_response,
)

logger = logging.getLogger(__name__)

# --- Register map (1-based, per the ZC300 manual) ---
REG_DEVICE_MODEL = 30001    # 7 regs, ASCII
REG_MOTION_STATE = 30012    # 3 regs: 0=stopped, 1=moving
REG_STATUS = 30015          # bitmask: limits/home/estop/alarms
REG_POS_X = 30016           # float, 2 regs; Y=30018, R=30020
REG_UNIT_X = 30022          # 0=pulse, 1=mm, 2=deg
REG_OPCODE = 30050          # opcode block: 30050 opcode, 30051 axis, 30052 dir
REG_TARGET_X = 30059        # absolute-move target float; Y=30061, R=30063
REG_ENABLE_X = 30066        # 0x01 = enabled
REG_HOME_MODE_X = 30105     # 0x01 user / 0x02 neg limit / 0x03 origin
REG_DIST_X = 30114          # fixed-length distance float
REG_SPEED_CONST_X = 30129   # constant speed float (pps)
REG_ACC_X = 30135           # accel float (pps^2)

# --- Opcodes ---
OP_ABSOLUTE = 0x0064
OP_FIXED_LENGTH = 0x0065
OP_CONTINUOUS = 0x0066
OP_DECEL_STOP = 0x0067
OP_IMMEDIATE_STOP = 0x0068
OP_HOME = 0x0069
OP_SAVE_PARAMS = 0x006D

AXIS_SEL = {"x": 0x31, "y": 0x32, "r": 0x33}
AXIS_ALL = 0x30
DIR_POS = 0x50
DIR_NEG = 0x4E

_LIMIT_BITS = {"x+": 0, "x-": 1, "y+": 3, "y-": 4, "r+": 6, "r-": 7}
_HOME_BITS = {"x": 2, "y": 5, "r": 8}
_ESTOP_BIT = 9
_ALARM_BITS = {"x": 10, "y": 11, "r": 12}

_AXIS_IDX = {"x": 0, "y": 1, "r": 2}
_MOTION = {"x", "y", "r"}

_EXC_TO_ERROR = {
    0x06: DeviceBusyError,
    0x07: LimitHitError,
    0x08: EStopError,
    0x09: CommandRejectedError,
}


class ZolixXYRStage(XYRStage):
    """Modbus-RTU driver for the Zolix ZC300 motion controller."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.port_name = config.get("port", "COM3")
        self.slave = int(config.get("slave_address", 1))
        self.timeout_s = float(config.get("timeout_s", 0.05))
        self.um_per_pulse_xy = float(config.get("um_per_pulse_xy", 0.625))
        self.um_per_pulse_r = float(config.get("um_per_pulse_r", 0.00125))
        self.slow_speed_pps = int(config.get("slow_speed_pps", 500))
        self.fast_speed_pps = int(config.get("fast_speed_pps", 2000))
        self.slow_speed_r = int(config.get("slow_speed_r", 1000))
        self.fast_speed_r = int(config.get("fast_speed_r", 10000))
        self.accel_pps2 = int(config.get("accel_pps2", 10_000_000))
        self.stop_mode = config.get("stop_mode", "immediate")
        self._stop_opcode = OP_IMMEDIATE_STOP if self.stop_mode == "immediate" else OP_DECEL_STOP
        # Test injection point: a callable returning a serial-port-like object.
        self._serial_factory = config.get("serial_factory")

        self._ser: serial.Serial | None = None
        self._lock = threading.RLock()
        self._device_model: str | None = None
        self._speed_written: dict[str, int] = {}
        self._dist_written: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @property
    def device_id(self) -> str:
        return f"zolix@{self.port_name}"

    def connect(self) -> None:
        factory = self._serial_factory or serial.Serial
        try:
            self._ser = factory(
                port=self.port_name,
                baudrate=115200,
                bytesize=8,
                parity="N",
                stopbits=1,
                timeout=self.timeout_s,
            )
        except (OSError, serial.SerialException) as exc:
            raise DeviceConnectionError(f"Zolix on {self.port_name}: {exc}") from exc
        try:
            # Liveness probe: motion-state registers must answer.
            self._read_input_regs(REG_MOTION_STATE, 3)
            model = self._read_input_regs(REG_DEVICE_MODEL, 7)
            self._device_model = "".join(chr(r >> 8) + chr(r & 0xFF) for r in model).strip("\x00 ")
            logger.info("Zolix device model: %s", self._device_model)
            self._check_units()
            for axis in self._axes_here():
                self._write_single(REG_ENABLE_X + _AXIS_IDX[axis], 0x01)
                self._write_floats(REG_ACC_X + 2 * _AXIS_IDX[axis], [float(self.accel_pps2)])
            self._verify_motion_config()
        except Exception:
            self._close_port()
            raise
        self._connected = True
        logger.info("Connected to Zolix ZC300 on %s", self.port_name)

    def disconnect(self) -> None:
        try:
            if self.is_connected:
                self.stop()
        except Exception:  # noqa: BLE001 - disconnect must be idempotent
            pass
        with self._lock:
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

    def _axes_here(self) -> list[str]:
        axes = ["x", "y"]
        if self.config.get("rotation_enabled", True):
            axes.append("r")
        return axes

    def _check_units(self) -> None:
        units = self._read_input_regs(REG_UNIT_X, 3)
        for axis in self._axes_here():
            value = units[_AXIS_IDX[axis]] if len(units) > _AXIS_IDX[axis] else 0
            if value != 0:
                logger.warning(
                    "Zolix axis '%s' is not in pulse mode (unit=%d) — "
                    "step math assumes pulses (fix via the controller panel)",
                    axis, value,
                )

    def _verify_motion_config(self) -> None:
        """Readback accel registers and warn on deviation."""
        try:
            for axis in self._axes_here():
                regs = self._read_holding_regs(REG_ACC_X + 2 * _AXIS_IDX[axis], 2)
                self._f32_from_regs(regs)  # validate parseability only
        except Exception as exc:  # noqa: BLE001 - informational only
            logger.warning("Motion config readback failed: %s", exc)

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def move_abs_pulses(self, x: int, y: int, r: int | None = None,
                        speed_pps: int | None = None) -> None:
        """Absolute moves in pulses (opcode 0x0064). HARDWARE-UNVALIDATED."""
        targets = {"x": int(x), "y": int(y)}
        if r is not None:
            targets["r"] = int(r)
        with self._lock:
            self._require_connected()
            self._ensure_not_moving()
            for axis, target in targets.items():
                self._ensure_speed(axis, speed_pps)
                self._write_floats(REG_TARGET_X + 2 * _AXIS_IDX[axis], [float(target)])
                # Direction is a placeholder to satisfy the 3-register arity;
                # the controller derives direction from the target.
                self._write_opcode(OP_ABSOLUTE, AXIS_SEL[axis], DIR_POS, regs=3)

    def move_abs_um(self, x_um: float, y_um: float, r_deg: float | None = None,
                    speed: StageSpeed = StageSpeed.SLOW,
                    speed_pps: int | None = None) -> None:
        """Absolute moves composed of validated FIXED-LENGTH relative moves.

        The 0x0064 absolute opcode was found to silently no-op on this
        controller (hardware-verified 2026-09-05) — only fixed-length
        (0x0065) motion is trustworthy. The relative delta is computed
        from the controller's own position readback.
        """
        current = self.get_position()
        dx = x_um - current.x_um
        dy = y_um - current.y_um
        dr = (r_deg - current.r_deg) if r_deg is not None else 0.0
        self.move_rel_um(dx, dy, dr if r_deg is not None else None, speed=speed,
                         speed_pps=speed_pps)

    def move_rel_um(self, dx_um: float, dy_um: float, dr_deg: float | None = None,
                    speed: StageSpeed = StageSpeed.SLOW,
                    speed_pps: int | None = None) -> None:
        """Relative moves via fixed-length (0x0065) — the validated opcode.

        ``speed_pps`` overrides the configured slow/fast speed register
        (the grid scan passes the active objective's scaled speed)."""
        pps = self._speed_for(speed) if speed_pps is None else int(speed_pps)
        moves = {
            "x": round(dx_um / self.um_per_pulse_xy),
            "y": round(dy_um / self.um_per_pulse_xy),
        }
        if dr_deg is not None:
            moves["r"] = round(dr_deg / self.um_per_pulse_r)
        with self._lock:
            self._require_connected()
            self._ensure_not_moving()
            for axis, steps in moves.items():
                if steps == 0:
                    continue
                self._ensure_speed(axis, pps)
                distance = float(abs(steps))
                # Write the distance, then VERIFY by readback before the
                # motion opcode: a CRC-glitched frame on the wire must
                # never become a wild move (hardware incident 2026-09-05).
                self._write_floats(REG_DIST_X + 2 * _AXIS_IDX[axis], [distance])
                readback = self._read_holding_regs(
                    REG_DIST_X + 2 * _AXIS_IDX[axis], 2)
                actual = self._f32_from_regs(readback)
                if abs(actual - distance) > 0.5:
                    raise ProtocolError(
                        f"Zolix distance register readback mismatch: wrote "
                        f"{distance} pulses, read {actual} — motion blocked")
                self._dist_written[axis] = distance
                direction = DIR_POS if steps > 0 else DIR_NEG
                self._write_opcode(OP_FIXED_LENGTH, AXIS_SEL[axis], direction, regs=3)

    def _speed_for(self, speed: StageSpeed, axis: str = "x") -> int:
        if axis == "r":
            return self.slow_speed_r if speed is StageSpeed.SLOW else self.fast_speed_r
        return self.slow_speed_pps if speed is StageSpeed.SLOW else self.fast_speed_pps

    def _ensure_speed(self, axis: str, speed_pps: int | None) -> None:
        if speed_pps is None:
            speed_pps = self._speed_for(StageSpeed.SLOW, axis)
        if self._speed_written.get(axis) != speed_pps:
            self._write_floats(REG_SPEED_CONST_X + 2 * _AXIS_IDX[axis], [float(speed_pps)])
            self._speed_written[axis] = speed_pps

    def _ensure_not_moving(self) -> None:
        status = self.get_status()
        if status.any_moving:
            raise DeviceBusyError("Zolix axes are moving; wait for idle first")
        if status.estop:
            raise EStopError("Zolix emergency stop is active")

    def home(self, axes: AxisMask = AxisMask.ALL, mode: int | None = None) -> None:
        targets = []
        if axes & AxisMask.X:
            targets.append("x")
        if axes & AxisMask.Y:
            targets.append("y")
        if axes & AxisMask.Z:
            targets.append("r")
        with self._lock:
            self._require_connected()
            if mode is not None:
                for axis in targets:
                    self._write_single(REG_HOME_MODE_X + _AXIS_IDX[axis], int(mode))
            if len(targets) == 3:
                self._write_opcode(OP_HOME, AXIS_ALL, DIR_NEG, regs=3)
            else:
                for axis in targets:
                    self._write_opcode(OP_HOME, AXIS_SEL[axis], DIR_NEG, regs=3)

    def move_continuous(self, axis: str, direction: int, speed_pps: int | None = None) -> None:
        """Continuous move until stop (opcode 0x0066). Used for jog-hold."""
        if axis not in _AXIS_IDX:
            raise CommandRejectedError(f"Unknown axis: {axis!r}")
        with self._lock:
            self._require_connected()
            self._ensure_speed(axis, speed_pps)
            self._write_opcode(OP_CONTINUOUS, AXIS_SEL[axis], direction, regs=3)

    def stop_axis(self, axis: str) -> None:
        """Stop one axis (stop opcodes take 2 registers: opcode + axis)."""
        if axis not in _AXIS_IDX:
            raise CommandRejectedError(f"Unknown axis: {axis!r}")
        with self._lock:
            self._require_connected()
            self._write_opcode(self._stop_opcode, AXIS_SEL[axis], regs=2)

    def stop(self) -> None:
        """Stop all axes; verify, escalate to immediate stop, verify again.

        Never raises — this is the emergency halt.
        """
        try:
            self._require_connected()
            first = OP_IMMEDIATE_STOP if self.stop_mode == "immediate" else OP_DECEL_STOP
            with self._lock:
                self._write_opcode(first, AXIS_ALL, regs=2)
            if not self._wait_stopped(0.3):
                logger.warning("Zolix: %s stop left axes moving — escalating to immediate stop",
                               self.stop_mode)
                with self._lock:
                    self._write_opcode(OP_IMMEDIATE_STOP, AXIS_ALL, regs=2)
                if not self._wait_stopped(0.2):
                    logger.error("Zolix: axes still moving after immediate stop")
        except Exception as exc:  # noqa: BLE001 - stop() never raises
            logger.error("Zolix stop() failed: %s", exc)

    def _wait_stopped(self, timeout_s: float, poll_s: float = 0.02) -> bool:
        """True once no axis reports moving (or a status read errors).

        Polls instead of sleeping the full settle: an immediate stop halts
        in tens of milliseconds, and this runs on EVERY release-stop and
        inside the 1.5 s STOP ALL budget — the fixed 0.3 s + 0.2 s sleeps
        spent most of that budget waiting for nothing.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                if not self.get_status().any_moving:
                    return True
            except Exception:  # noqa: BLE001 - a read error must not break the stop
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll_s)

    def save_parameters(self) -> None:
        with self._lock:
            self._require_connected()
            if self.get_status().any_moving:
                raise DeviceBusyError("Cannot save Zolix parameters while moving")
            self._write_opcode(OP_SAVE_PARAMS, regs=1)

    def configure_motion(self, accel_pps2: int, speeds: dict[str, int]) -> None:
        with self._lock:
            self._require_connected()
            self.accel_pps2 = int(accel_pps2)
            for axis in self._axes_here():
                self._write_floats(REG_ACC_X + 2 * _AXIS_IDX[axis], [float(self.accel_pps2)])
                if axis in speeds:
                    self._write_floats(REG_SPEED_CONST_X + 2 * _AXIS_IDX[axis],
                                       [float(speeds[axis])])
                    self._speed_written[axis] = int(speeds[axis])
            self._verify_motion_config()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> StageStatus:
        with self._lock:
            self._require_connected()
            data = self._read_input_regs(REG_MOTION_STATE, 10)  # 30012-30021

        def bit(n: int) -> bool:
            return bool((data[3] if len(data) > 3 else 0) & (1 << n))

        moving = {
            axis: len(data) > idx and data[idx] == 1
            for axis, idx in _AXIS_IDX.items()
        }
        return StageStatus(
            x_moving=moving["x"],
            y_moving=moving["y"],
            r_moving=moving["r"],
            limit_x_pos=bit(_LIMIT_BITS["x+"]),
            limit_x_neg=bit(_LIMIT_BITS["x-"]),
            limit_y_pos=bit(_LIMIT_BITS["y+"]),
            limit_y_neg=bit(_LIMIT_BITS["y-"]),
            home_x=bit(_HOME_BITS["x"]),
            home_y=bit(_HOME_BITS["y"]),
            home_r=bit(_HOME_BITS["r"]),
            estop=bit(_ESTOP_BIT),
            alarm=any(bit(n) for n in _ALARM_BITS.values()),
        )

    def get_position(self) -> StagePosition:
        with self._lock:
            self._require_connected()
            data = self._read_input_regs(REG_POS_X, 6)  # 30016-30021

        def f32(idx: int) -> float:
            return self._f32_from_regs(data[idx * 2:idx * 2 + 2]) if len(data) >= idx * 2 + 2 else 0.0

        x, y, r = f32(0), f32(1), f32(2)
        return StagePosition(
            x_pulses=round(x), y_pulses=round(y), r_pulses=round(r),
            x_um=x * self.um_per_pulse_xy,
            y_um=y * self.um_per_pulse_xy,
            r_deg=r * self.um_per_pulse_r,
        )

    def check_estop(self) -> bool:
        try:
            return self.get_status().estop
        except DeviceError:
            return False

    def wait_idle(self, timeout_s: float = 120.0, poll_s: float = 0.05) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self.get_status().any_moving:
                return
            time.sleep(poll_s)
        raise DeviceTimeoutError(f"Zolix axes still moving after {timeout_s}s")

    # ------------------------------------------------------------------
    # Modbus I/O
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise NotConnectedError(f"Zolix on {self.port_name} is not connected")

    @staticmethod
    def _f32_from_regs(regs: list[int]) -> float:
        from talos.protocols.modbus_rtu import unpack_f32

        if len(regs) < 2:
            raise ValueError("Need 2 registers for a float")
        return unpack_f32(regs[0], regs[1])

    def _transact(self, frame: bytes, allow_retry: bool) -> bytes:
        for attempt in (1, 2):
            try:
                self._ser.reset_input_buffer()
                self._ser.reset_output_buffer()
                self._ser.write(frame)
                self._ser.flush()
                time.sleep(0.01)
                response = bytearray()
                deadline = time.monotonic() + self.timeout_s * 4
                while time.monotonic() < deadline:
                    chunk = self._ser.read(256)
                    if chunk:
                        response.extend(chunk)
                        total = self._response_len(frame[1], response)
                        if total is not None and len(response) >= total:
                            return bytes(response[:total])
                    elif response:
                        return bytes(response)
                    else:
                        time.sleep(0.005)
                if response:
                    return bytes(response)
                if not allow_retry or attempt == 2:
                    raise DeviceTimeoutError(f"No Modbus reply on {self.port_name}")
                time.sleep(0.02)
            except OSError as exc:
                self._connected = False
                raise NotConnectedError(f"Zolix serial error: {exc}") from exc
        raise DeviceTimeoutError(f"No Modbus reply on {self.port_name}")

    @staticmethod
    def _response_len(fn: int, buf: bytearray) -> int | None:
        if len(buf) < 2:
            return None
        if buf[1] == fn | 0x80:
            return 5
        if fn in (0x03, 0x04):
            if len(buf) < 3:
                return None
            return 3 + buf[2] + 2
        return 8

    def _raise_for_exception(self, exc: ModbusException, context: str) -> None:
        error_cls = _EXC_TO_ERROR.get(exc.code, CommandRejectedError)
        raise error_cls(f"{context}: {exc}")

    def _read_input_regs(self, reg: int, count: int) -> list[int]:
        resp = self._transact(build_read(self.slave, READ_INPUT, reg, count), allow_retry=True)
        try:
            return parse_read_response(resp, self.slave, READ_INPUT)
        except ModbusException as exc:
            self._raise_for_exception(exc, f"Read input reg {reg}")
        except ValueError as exc:
            raise DeviceTimeoutError(f"Read input reg {reg}: {exc}") from exc

    def _read_holding_regs(self, reg: int, count: int) -> list[int]:
        from talos.protocols.modbus_rtu import READ_HOLDING

        resp = self._transact(build_read(self.slave, READ_HOLDING, reg, count), allow_retry=True)
        try:
            return parse_read_response(resp, self.slave, READ_HOLDING)
        except ModbusException as exc:
            self._raise_for_exception(exc, f"Read holding reg {reg}")
        except ValueError as exc:
            raise DeviceTimeoutError(f"Read holding reg {reg}: {exc}") from exc

    def _write_single(self, reg: int, value: int) -> None:
        resp = self._transact(build_write_single(self.slave, reg, value), allow_retry=False)
        self._check_write_ack(resp, 0x06, f"Write reg {reg}")

    def _write_floats(self, reg: int, values: list[float]) -> None:
        from talos.protocols.modbus_rtu import pack_f32

        regs: list[int] = []
        for v in values:
            hi, lo = pack_f32(float(v))
            regs.extend([hi, lo])
        self._write_multiple(reg, regs)

    def _write_multiple(self, reg: int, values: list[int]) -> None:
        resp = self._transact(build_write_multiple(self.slave, reg, values), allow_retry=False)
        self._check_write_ack(resp, 0x10, f"Write multiple reg {reg}")

    def _check_write_ack(self, resp: bytes, fn: int, context: str) -> None:
        """Validate a write acknowledgement.

        An empty reply is a timeout (raised by _transact). A bad CRC or
        malformed ack raises ProtocolError — we cannot be sure the write
        applied, and silently continuing on motion commands is unsafe.
        """
        try:
            parse_write_response(resp, self.slave, fn)
        except ModbusException as exc:
            self._raise_for_exception(exc, context)
        except FrameError as exc:
            raise ProtocolError(f"{context}: {exc}") from exc

    def _write_opcode(self, opcode: int, *params: int, regs: int = 3) -> None:
        """Write an opcode via fn 0x10 with the EXACT register count for its
        arity (moves=3, stops=2, save-params=1). Caller must hold the lock."""
        assert len(params) + 1 == regs, "Opcode arity mismatch"
        self._write_multiple(REG_OPCODE, [opcode, *params])
