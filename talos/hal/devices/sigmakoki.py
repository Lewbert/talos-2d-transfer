"""DIY SigmaKoki XYZ transfer stage (Arduino + Autonics MD5-HD14 drivers).

ASCII line protocol at 115200 8N1 (from the live driver of the tested
reference deployment, github.com/Lewbert/transfer-stage-control):

    MV:<AXIS>:<dir>:<level>      continuous move, idempotent (re-issue
                                 updates dir/speed)
                                 → OK:MV:<AXIS>:<dir>:<level> | ERR:<AXIS>:LIMIT
    STEP:<AXIS>:<dir>:<steps>    stepped move → OK:STEP:<AXIS>:<actual>
                                 (the ack signals completion, ≤5 s)
    SPD:<AXIS>:<level>           speed level 0..5 (25..500 steps/s)
    HOME                         re-zero only (no homing motion) → OK:HOME
    STOP:<AXIS> | STOP:ALL
    LIMITS?  → L:X+:0,X-:1,Y+:0,Y-:0,Z+:0,Z-:0   (1 = triggered)
    STATUS?  → S:X:<p>,Y:<p>,Z:<p>,XSPD:<l>,YSPD:<l>,ZSPD:<l>
    Unsolicited: BOOT banner, EV:LIM:<AXIS><dir>
    Errors: ERR:BAD_AXIS, ERR:BAD_DIR, ERR:<AXIS>:BUSY, ERR:<AXIS>:LIMIT

The firmware has NO per-axis busy flag; ``wait_idle`` therefore uses
settle detection: positions stable across consecutive polls = stopped.

Speed levels: {0: 25, 1: 50, 2: 100, 3: 167, 4: 250, 5: 500} steps/s.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import serial

from talos.hal.base import (
    Axis,
    AxisMask,
    CommandRejectedError,
    DeviceBusyError,
    DeviceConnectionError,
    DeviceTimeoutError,
    Direction,
    LimitHitError,
    NotConnectedError,
    XYZStage,
)
from talos.protocols.ascii_line import LineIO, parse_kv

logger = logging.getLogger(__name__)

SPEED_LEVEL_TO_HZ = {0: 25, 1: 50, 2: 100, 3: 167, 4: 250, 5: 500}
_HZ_TO_LEVEL = ((37, 0), (75, 1), (133, 2), (208, 3), (375, 4), (10**9, 5))

_UM_PER_STEP_DEFAULT = {Axis.X: 0.5, Axis.Y: 0.5, Axis.Z: 0.25}


def hz_to_level(hz: float) -> int:
    for limit, level in _HZ_TO_LEVEL:
        if hz <= limit:
            return level
    return 5


class SigmaKokiXYZStage(XYZStage):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.port_name = config.get("port", "COM6")
        self.timeout_s = float(config.get("timeout_s", 0.3))
        self.um_per_step_xy = float(config.get("um_per_step_xy", 0.5))
        self.um_per_step_z = float(config.get("um_per_step_z", 0.25))
        self.slow_speed_hz = int(config.get("slow_speed_hz", 100))
        self.fast_speed_hz = int(config.get("fast_speed_hz", 1000))
        self.slow_speed_z = int(config.get("slow_speed_z", 99))
        self.fast_speed_z = int(config.get("fast_speed_z", 500))
        # Test injection point: a callable returning a serial-port-like object.
        self._serial_factory = config.get("serial_factory")
        self._ser: serial.Serial | None = None
        self._io: LineIO | None = None

    @property
    def device_id(self) -> str:
        return f"sigmakoki@{self.port_name}"

    def _um_per_step(self, axis: Axis) -> float:
        return self.um_per_step_z if axis is Axis.Z else self.um_per_step_xy

    def _level_for(self, axis: Axis, fast: bool) -> int:
        if axis is Axis.Z:
            return hz_to_level(self.fast_speed_z if fast else self.slow_speed_z)
        return hz_to_level(self.fast_speed_hz if fast else self.slow_speed_hz)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

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
            raise DeviceConnectionError(f"SigmaKoki on {self.port_name}: {exc}") from exc
        self._io = LineIO(self._ser, timeout_s=self.timeout_s)
        try:
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
            # Handshake: look for the BOOT/READY banner, fall back to PING.
            # The CH340's DTR toggle resets the Arduino; its boot banner can
            # take 1-2 s, so keep reading until the full deadline.
            deadline = time.monotonic() + 3.0
            saw_banner = False
            while time.monotonic() < deadline:
                line = self._io.read_line()
                if line and ("BOOT" in line or "READY" in line):
                    saw_banner = True
                    break
            if not saw_banner:
                try:
                    reply = self._io.read_reply("PONG")
                    if "PONG" not in reply.line:
                        raise TimeoutError()
                except TimeoutError as exc:
                    raise DeviceConnectionError(
                        f"SigmaKoki on {self.port_name}: no BOOT/READY banner and no PONG"
                    ) from exc
            # Post-handshake liveness check: the first command after the
            # boot banner can be swallowed by the bootloader; a successful
            # PING/PONG proves the command channel is live.
            self._io.send("PING")
            try:
                self._io.read_reply("PONG")
            except TimeoutError as exc:
                raise DeviceConnectionError(
                    f"SigmaKoki on {self.port_name}: PING liveness check failed"
                ) from exc
        except Exception:
            self._close_port()
            raise
        self._connected = True
        logger.info("Connected to SigmaKoki stage on %s", self.port_name)

    def disconnect(self) -> None:
        try:
            if self.is_connected:
                self.stop(AxisMask.ALL)
        except Exception:  # noqa: BLE001
            pass
        self._connected = False
        self._close_port()

    def _close_port(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:  # noqa: BLE001
                pass
            self._ser = None
            self._io = None

    @property
    def is_connected(self) -> bool:
        return bool(
            self._connected
            and self._ser is not None
            and getattr(self._ser, "is_open", False)
        )

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def move(self, axis: Axis, direction: Direction, level: int) -> None:
        self._require_connected()
        level = max(0, min(5, int(level)))
        dir_str = "+1" if direction is Direction.POSITIVE else "-1"
        cmd = f"MV:{axis.value}:{dir_str}:{level}"
        self._io.send(cmd)
        try:
            # The firmware answers OK:MV:<axis>:<dir>:<level> (or
            # ERR:<axis>:LIMIT). Waiting for a bare "MV" prefix never
            # matched that ack, so the reply was discarded as a stray line
            # and EVERY jog blocked this worker for the whole serial
            # timeout (0.3 s) — the stage felt sluggish while STOP stayed
            # snappy (STOP goes through _send_loose, which matches any
            # line). Hardware-verified against transfer-stage-controller.ino.
            reply = self._io.read_reply("OK:MV:")
            self._check_events(reply.events)
            if "ERR" in reply.line and "LIMIT" in reply.line:
                raise LimitHitError(f"SigmaKoki {axis.value} limit blocks this direction")
            if "ERR" in reply.line:
                self._raise_err(reply.line, cmd)
        except TimeoutError:
            # MV is idempotent and already applied by the controller; a
            # missing ack (wedged/deaf firmware) must not fail the jog.
            logger.debug("SigmaKoki MV: no ack for %r within %.2fs",
                         cmd, self.timeout_s)

    def step(self, axis: Axis, direction: Direction, steps: int) -> int:
        self._require_connected()
        steps = int(steps)
        if steps <= 0:
            raise CommandRejectedError("STEP count must be positive")
        dir_str = "+1" if direction is Direction.POSITIVE else "-1"
        cmd = f"STEP:{axis.value}:{dir_str}:{steps}"
        self._io.send(cmd)
        try:
            reply = self._io.read_reply("OK:STEP:")
        except TimeoutError as exc:
            raise DeviceTimeoutError(f"SigmaKoki STEP {axis.value}: {exc}") from exc
        self._check_events(reply.events)
        if "ERR" in reply.line:
            self._raise_err(reply.line, cmd)
        parts = reply.line.strip().split(":")
        if len(parts) < 4:
            raise CommandRejectedError(f"Bad OK:STEP reply: {reply.line!r}")
        return int(parts[3])

    def move_rel_um(self, axis: Axis, um: float, level: int) -> int:
        steps = round(um / self._um_per_step(axis))
        if steps == 0:
            return 0
        direction = Direction.POSITIVE if steps > 0 else Direction.NEGATIVE
        return self.step(axis, direction, abs(steps))

    def set_speed(self, axis: Axis, level: int) -> None:
        self._require_connected()
        level = max(0, min(5, int(level)))
        self._send_loose(f"SPD:{axis.value}:{level}")

    def home(self) -> None:
        self._require_connected()
        self._io.send("HOME")
        try:
            reply = self._io.read_reply("OK:HOME")
            self._check_events(reply.events)
        except TimeoutError as exc:
            raise DeviceTimeoutError(f"SigmaKoki HOME: {exc}") from exc

    def stop(self, axes: AxisMask | None = None) -> None:
        try:
            self._require_connected()
            if axes is None or axes is AxisMask.ALL:
                self._send_loose("STOP:ALL")
            else:
                for axis in (Axis.X, Axis.Y, Axis.Z):
                    if axes & getattr(AxisMask, axis.name):
                        self._send_loose(f"STOP:{axis.value}")
        except Exception as exc:  # noqa: BLE001 - stop() never raises
            logger.error("SigmaKoki stop() failed: %s", exc)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_limits(self) -> dict[str, bool]:
        self._require_connected()
        self._io.send("LIMITS?")
        try:
            reply = self._io.read_reply("L:")
        except TimeoutError as exc:
            raise DeviceTimeoutError(f"SigmaKoki LIMITS?: {exc}") from exc
        fields = parse_kv(reply.line[2:])
        return {f"{axis}{sign}".lower(): fields.get(f"{axis}{sign}", "0") == "1"
                for axis in ("X", "Y", "Z") for sign in ("+", "-")}

    def get_status(self) -> dict[str, str]:
        self._require_connected()
        self._io.send("STATUS?")
        try:
            reply = self._io.read_reply("S:")
        except TimeoutError as exc:
            raise DeviceTimeoutError(f"SigmaKoki STATUS?: {exc}") from exc
        self._check_events(reply.events)
        status = {"x": "0", "y": "0", "z": "0", "xspd": "0", "yspd": "0", "zspd": "0"}
        # Firmware sends uppercase keys (X, XSPD...) — normalize to lowercase.
        status.update({k.lower(): v for k, v in parse_kv(reply.line[2:]).items()})
        return status

    def get_position(self) -> dict[Axis, int]:
        return self._position_from(self.get_status())

    def get_telemetry(self) -> dict[str, Any]:
        """Status + position from ONE wire round trip.

        STATUS? carries both, so polling ``get_status`` and
        ``get_position`` separately (which the proxy used to do) sent the
        same query twice per poll cycle on an already-busy serial line.
        """
        status = self.get_status()
        return {"status": status, "position": self._position_from(status)}

    @staticmethod
    def _position_from(status: dict[str, str]) -> dict[Axis, int]:
        return {
            Axis.X: int(status.get("x", 0)),
            Axis.Y: int(status.get("y", 0)),
            Axis.Z: int(status.get("z", 0)),
        }

    def wait_idle(self, timeout_s: float = 300.0, poll_s: float = 0.05) -> None:
        """Settle detection: the firmware has no busy flag, but a moving
        stepper changes position every poll. Idle = positions stable across
        3 consecutive polls."""
        self._require_connected()
        last: dict[Axis, int] | None = None
        stable = 0
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            pos = self.get_position()
            if last is not None and pos == last:
                stable += 1
                if stable >= 3:
                    return
            else:
                stable = 0
            last = pos
            time.sleep(poll_s)
        raise DeviceTimeoutError(f"SigmaKoki stage not settled after {timeout_s}s")

    def drain_events(self) -> list[str]:
        if self._io is None:
            return []
        return self._io.drain_events()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self.is_connected or self._io is None:
            raise NotConnectedError(f"SigmaKoki on {self.port_name} is not connected")

    def _send_loose(self, cmd: str) -> None:
        """Send a command whose ack format we do not strictly require."""
        self._io.send(cmd)
        try:
            reply = self._io.read_reply("")
        except TimeoutError:
            reply = None
        if reply is not None:
            self._check_events(reply.events)
            if "ERR" in reply.line and "LIMIT" not in reply.line:
                logger.debug("SigmaKoki loose reply to %r: %s", cmd, reply.line)

    def _check_events(self, events: list[str]) -> None:
        for event in events:
            if event.startswith("EV:LIM"):
                logger.warning("SigmaKoki event: %s", event)

    def _raise_err(self, line: str, context: str) -> None:
        if "BUSY" in line:
            raise DeviceBusyError(f"{context}: {line}")
        if "LIMIT" in line:
            raise LimitHitError(f"{context}: {line}")
        raise CommandRejectedError(f"{context}: {line}")
