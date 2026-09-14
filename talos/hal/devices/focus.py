"""DIY focus stage driver (Arduino Uno/Nano + CRD5103PB driver).

⚠️ THE FOCUS STAGE HAS NO LIMIT SENSOR. Safety rules implemented here and
enforced by the autofocus controller:
- All speeds clamped to [10, max_speed] (configured max: 2000 steps/s).
- Autofocus searches are bounded inside the firmware soft limits (SLIM).
- The firmware auto-stops after 5 s of serial inactivity while moving
  (CFG:TMO) — wait_idle polls well within that window.

ASCII line protocol, 115200 8N1 (docs/hardware/focus/protocol.md):
    STATUS? → S:POS:<p>,MODE:<m>,V:<v>,SPD:<t>,LIM:<b>,SLIM:<o>
        MODE: IDLE | CONT | TRAP | LIMIT
    MOVE:<rel> → OK:MOVE:<rel>      (trapezoid at MVSPD, exact landing)
    GOTO:<abs> → OK:GOTO:<abs>
    SPD:<spd>  → OK:SPD:<applied>   (signed steps/s; 0 = ramp stop)
    STOP, ZERO, MVSPD:<n>, AWOFF:<0|1>, CUTB:<0|1>, CFG:MAX/ACC/TMO
    SLIM:<0|1> → OK:SLIM:<o>:<min>:<max>
    SLIM:SET:<min>:<max> → OK:SLIM:1:<min>:<max>
    SLIM? → SLIM:<o>:<min>:<max>
    Handshake: PING → PONG; power-up banner BOOT:FOCUSCTRL:1.0 then READY.
    Errors: ERR:UNKNOWN | BAD_FORMAT | BAD_VALUE | BUSY | LIMIT | RANGE |
            NOHW | OVERFLOW
    Events: EV:STOP:<pos> (aborted), EV:DONE:<pos> (move complete),
            EV:LIM:<+/->:<pos>, EV:TMO:<pos> (inactivity auto-stop), EV:WDT

Driver sign convention: the firmware counts + on the CW pin (D11), which
DECREASES the objective-sample distance on this rig. TALOS inverts at the
driver boundary so +speed/+position = distance INCREASE (user preference).
The flip is uniform (every motion/status/limit method), so upper layers
only ever see driver units; the wire protocol — and the firmware for other
software — is untouched.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import serial

from talos.hal.base import (
    CommandRejectedError,
    DeviceBusyError,
    DeviceConnectionError,
    DeviceError,
    DeviceTimeoutError,
    EStopError,
    FocusStage,
    LimitHitError,
    NotConnectedError,
)
from talos.models import FocusStatus
from talos.protocols.ascii_line import LineIO, parse_kv

logger = logging.getLogger(__name__)


class FocusStageDriver(FocusStage):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.port_name = config.get("port", "COM10")
        self.timeout_s = float(config.get("timeout_s", 0.3))
        # Quiet gap after every command: legacy mitigation, default OFF.
        # The firmware's command processing is FAST (hardware-verified:
        # ~1 ms replies; the precursor logged 58k+ clean exchanges at 5 Hz).
        # The historical garbling was junk bytes from an external noise
        # source folding into commands — fixed at the firmware parser
        # (byte filter + stale-line flush; source in-tree at
        # arduino_firmware/focus_controller). Keep the knob for
        # pathological links.
        self.command_gap_s = float(config.get("command_gap_s", 0.0))
        self.max_speed = int(config.get("max_speed", 2000))
        self.clamp_speed_lo = int(config.get("clamp_speed_lo", 10))
        self.clamp_speed_hi = int(config.get("clamp_speed_hi", 5000))
        self.awoff_on_exit = bool(config.get("awoff_on_exit", False))
        # Test injection point: a callable returning a serial-port-like object.
        self._serial_factory = config.get("serial_factory")
        self._ser: serial.Serial | None = None
        self._io: LineIO | None = None
        # Soft limits are CACHED: SLIM? + STATUS? back-to-back at the
        # poll rate desynchronizes the firmware's reply pairing (hardware-
        # verified: every poll garbled with ERR:UNKNOWN at 10 Hz). Limits
        # only change through set_soft_limits(), which refreshes the cache.
        self._slim_cache: tuple[int, int] | None = None
        # Consecutive garbled-command count: after enough, the firmware's
        # line discipline is desynced and ONLY a DTR reset recovers it —
        # the driver self-heals by reopening the port.
        self._garble_streak = 0

    @property
    def device_id(self) -> str:
        return f"focus@{self.port_name}"

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _open_port(self) -> None:
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
            raise DeviceConnectionError(f"Focus on {self.port_name}: {exc}") from exc
        self._io = LineIO(self._ser, timeout_s=self.timeout_s)

    def _handshake(self) -> None:
        """Banner or PING→PONG + liveness PING + SLIM cache warm-up. The
        SLIM? happens HERE, serialized — a lazy SLIM? during the first
        telemetry poll races the STATUS? pairing and desynchronizes the
        firmware's replies (hardware-verified: every poll garbled with
        ERR:UNKNOWN until a DTR reset)."""
        self._ser.reset_input_buffer()
        # The Uno resets on DTR toggle; its banner can take 1-2 s, so
        # keep reading until the full deadline.
        deadline = time.monotonic() + 3.0
        saw_banner = False
        while time.monotonic() < deadline:
            line = self._io.read_line()
            if line and ("BOOT" in line or "READY" in line):
                saw_banner = True
                break
        if not saw_banner:
            self._strict("PING", "PONG")
        # Post-handshake liveness check: the first command after a reset
        # can be garbled by USB re-enumeration; a PING/PONG proves the
        # channel is live and flushes any quirk bytes harmlessly.
        self._strict("PING", "PONG")
        try:
            self.get_soft_limits()
        except DeviceError as exc:
            logger.warning("Focus SLIM? cache warm-up failed: %s", exc)
        self._garble_streak = 0

    def connect(self) -> None:
        self._open_port()
        self._connected = True  # probes below need is_connected
        try:
            self._handshake()
        except Exception:
            self._connected = False
            self._close_port()
            raise
        logger.info("Connected to focus stage on %s", self.port_name)

    def _recover_serial(self) -> None:
        """Self-heal a desynced firmware: close (DTR reset) + reopen +
        re-handshake. Hardware-verified: a fresh reset recovers the line
        100% of the time (raw probes were always clean after their own
        DTR reset)."""
        logger.warning("Focus serial desynced (%d consecutive garbles) — "
                       "resetting the link", self._garble_streak)
        try:
            self._close_port()
            time.sleep(0.3)
            self._open_port()
            self._handshake()
            logger.info("Focus serial link reset — firmware re-synced")
        except Exception as exc:  # noqa: BLE001
            self._connected = False
            logger.error("Focus serial recovery failed: %s", exc)
            raise CommandRejectedError(f"Focus serial recovery failed: {exc}")

    def disconnect(self) -> None:
        try:
            if self.is_connected:
                self.stop()
                if self.awoff_on_exit:
                    self._strict("AWOFF:1", "OK")
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

    def move_rel(self, steps: int, speed: int | None = None) -> None:
        steps = int(steps)
        if steps == 0:
            return
        # NOTE: point-to-point moves run at MVSPD (MVSPD:<n>); SPD:<n> is the
        # continuous-jog command and would put the axis in CONT mode, making
        # MOVE fail with ERR:BUSY. Steps are negated per the sign convention
        # (driver + = distance increase; firmware + = distance decrease).
        if speed is not None and speed > 0:
            self._set_move_speed(speed)
        self._strict(f"MOVE:{-steps}", "OK")

    def move_abs(self, position: int, speed: int | None = None) -> None:
        if speed is not None and speed > 0:
            self._set_move_speed(speed)
        self._strict(f"GOTO:{-int(position)}", "OK")

    def _set_move_speed(self, speed: int) -> None:
        hi = min(self.clamp_speed_hi, 5000)
        lo = max(self.clamp_speed_lo, 10)
        if not (lo <= int(speed) <= hi):
            raise CommandRejectedError(f"Speed {speed} outside clamp [{lo}, {hi}] steps/s")
        self._strict(f"MVSPD:{int(speed)}", "OK")

    def set_speed(self, steps_per_s: int) -> None:
        """SPD:<signed> — negative speeds jog downward (CONT mode)."""
        spd = int(steps_per_s)
        if spd == 0:
            self._strict("SPD:0", "OK")
            return
        hi = min(self.clamp_speed_hi, 5000)
        lo = max(self.clamp_speed_lo, 10)
        if not (lo <= abs(spd) <= hi):
            raise CommandRejectedError(
                f"Speed magnitude {abs(spd)} outside clamp [{lo}, {hi}] steps/s")
        self._strict(f"SPD:{-spd}", "OK")

    def zero(self) -> None:
        self._strict("ZERO", "OK")

    def get_config(self) -> dict:
        """CFG? → {min_speed, max_speed, accel_sps2, tmo_ms}. Used by the
        adaptive AF passes for the predictive edge-stop math (the real
        firmware ACC)."""
        reply = self._strict("CFG?", "CFG:")
        fields = parse_kv(reply.line[4:])
        return {
            "min_speed": int(fields.get("MIN", 10)),
            "max_speed": int(fields.get("MAX", 2000)),
            "accel_sps2": int(fields.get("ACC", 20000)),
            "tmo_ms": int(fields.get("TMO", 5000)),
        }

    def stop(self) -> None:
        """Emergency halt. Never raises."""
        try:
            if self._io is None:
                return
            self._io.send("STOP")
            # STOP aborts motion; also request a ramp stop as belt-and-braces.
            self._io.send("SPD:0")
            time.sleep(0.05)
            self._io.drain_events()
        except Exception as exc:  # noqa: BLE001
            logger.error("Focus stop() failed: %s", exc)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> FocusStatus:
        reply = self._strict("STATUS?", "S:")
        fields = parse_kv(reply.line[2:])
        pos = -int(fields.get("POS", 0))  # sign convention: + = distance increase
        mode = fields.get("MODE", "IDLE")
        v = float(fields.get("V", 0))
        spd = int(fields.get("SPD", 0))
        lim_raw = fields.get("LIM", "0")
        slim_raw = fields.get("SLIM", "0")
        driver_lim = {"+": "-", "-": "+"}.get(lim_raw, lim_raw)
        return FocusStatus(
            pos=pos, mode=mode, v=v, spd=spd,
            lim=lim_raw not in ("0", ""),
            blocked_dir=driver_lim,
            slim=None,  # STATUS? carries only the on/off flag; bounds via SLIM?
            # The firmware gates EVERY limit check on this flag, and ships
            # with it OFF: the bounds are only enforced when it is set.
            slim_on=slim_raw not in ("0", ""),
        )

    def get_slim_state(self) -> bool:
        """True when the firmware is enforcing the soft limits.

        Reading the bounds is NOT evidence of enforcement (SLIM:SET
        persists them even when the check is off), so autofocus asks
        separately before trusting its own bound clamp.
        """
        return bool(self.get_status().slim_on)

    def get_soft_limits(self) -> tuple[int, int]:
        """Soft limits — cached after the first read (see __init__), then
        converted to driver units (+ = distance increase), so the driver
        bounds are (−firmware_max, −firmware_min)."""
        if self._slim_cache is None:
            reply = self._strict("SLIM?", "SLIM:")
            parts = reply.line.split(":")
            if len(parts) < 4:
                raise CommandRejectedError(f"Bad SLIM? reply: {reply.line!r}")
            self._slim_cache = (int(parts[2]), int(parts[3]))
        lo, hi = self._slim_cache
        return (-hi, -lo)

    def set_soft_limits(self, lo: int, hi: int) -> None:
        """SLIM:SET — persisted to EEPROM. Called only from the Calibration wizard.
        Driver units in, negated+swapped to firmware units on the wire."""
        if lo >= hi:
            raise CommandRejectedError(f"Bad soft limits: min={lo} >= max={hi}")
        reply = self._strict(f"SLIM:SET:{-int(hi)}:{-int(lo)}", "OK")
        # The OK line carries the applied bounds (OK:SLIM:1:<min>:<max>).
        parts = reply.line.split(":")
        if len(parts) >= 5:
            try:
                self._slim_cache = (int(parts[3]), int(parts[4]))
                return
            except ValueError:
                pass
        self._slim_cache = (int(lo), int(hi))

    def wait_idle(self, timeout_s: float = 120.0, poll_s: float = 0.05) -> None:
        """Poll STATUS? until MODE is IDLE.

        Raises DeviceTimeoutError on EV:TMO (serial-inactivity auto-stop
        aborted the move) so the caller can re-issue the move once.
        Transient poll errors (e.g. a stepper-current glitch resetting the
        Arduino mid-poll) are logged and tolerated until the deadline.
        """
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            for event in self.drain_events():
                if event.startswith("EV:TMO"):
                    raise DeviceTimeoutError(f"Focus move aborted by inactivity stop: {event}")
            try:
                status = self.get_status()
                if status.is_idle:
                    return
            except DeviceError as exc:
                last_error = exc
                logger.warning("Focus wait_idle: transient poll error: %s", exc)
                time.sleep(0.2)  # give a reset firmware time to come back
            time.sleep(poll_s)
        raise DeviceTimeoutError(
            f"Focus stage not idle after {timeout_s}s (last error: {last_error})")

    def drain_events(self) -> list[str]:
        if self._io is None:
            return []
        return self._io.drain_events()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self.is_connected or self._io is None:
            raise NotConnectedError(f"Focus on {self.port_name} is not connected")

    def _strict(self, cmd: str, prefix: str):
        """Send a command and return its reply; ERR: lines become typed errors.

        ERR:UNKNOWN is retried once: with the firmware parser fixed (in-tree
        source), it is rare — an intermittent external noise source couples
        onto the serial lines (hardware-observed, both directions) and a
        burst can still tear one command; the next one lands clean.
        """
        self._require_connected()
        for attempt in (1, 2):
            self._io.send(cmd)
            try:
                reply = self._io.read_reply(prefix)
            except TimeoutError as exc:
                # A desynced firmware often goes SILENT instead of
                # replying ERR:UNKNOWN — timeouts must count toward the
                # self-heal streak too (hardware-verified).
                self._garble_streak += 1
                if self._garble_streak >= 5:
                    self._recover_serial()
                raise DeviceTimeoutError(f"Focus {cmd}: {exc}") from exc
            self._check_events(reply.events)
            if reply.line.startswith("ERR:UNKNOWN") and attempt == 1:
                logger.warning("Focus %s: retrying after %s", cmd, reply.line)
                time.sleep(0.1)
                continue
            if reply.line.startswith("ERR:"):
                if "UNKNOWN" in reply.line:
                    # Persistent garbling = desynced line discipline: a DTR
                    # reset recovers it 100% (hardware-verified). Self-heal
                    # before the caller sees the failure.
                    self._garble_streak += 1
                    if self._garble_streak >= 5:
                        self._recover_serial()
                self._raise_err(reply.line)
            self._garble_streak = 0
            if self.command_gap_s > 0:
                time.sleep(self.command_gap_s)
            return reply
        raise CommandRejectedError(f"Focus {cmd}: retry failed")

    def _check_events(self, events: list[str], raise_tmo: bool = False) -> None:
        for event in events:
            if event.startswith("ERR:"):
                self._raise_err(event)
            if event.startswith("EV:LIM"):
                raise LimitHitError(f"Focus soft limit stop: {event}")
            if event.startswith("EV:TMO") and raise_tmo:
                raise DeviceTimeoutError(f"Focus inactivity stop: {event}")
            if event.startswith("EV:WDT"):
                logger.warning("Focus watchdog reset detected: %s", event)
            logger.debug("Focus event: %s", event)

    def _raise_err(self, line: str) -> None:
        code = line.split(":", 1)[1] if ":" in line else line
        if "BUSY" in code:
            raise DeviceBusyError(f"Focus busy: {line}")
        if "LIMIT" in code:
            raise LimitHitError(f"Focus limit: {line}")
        if "NOHW" in code:
            raise DeviceConnectionError(f"Focus no hardware: {line}")
        if "STOP" in code or "ESTOP" in code:
            raise EStopError(f"Focus stopped: {line}")
        raise CommandRejectedError(f"Focus error: {line}")
