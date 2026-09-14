"""Hardware abstraction layer: abstract base classes, exceptions, enums.

Design rules:
- Drivers are blocking and synchronous, with per-transaction timeouts.
- Drivers NEVER import Qt — they are unit-testable against fake serials.
- ``stop()`` is the emergency halt: callable at any time, never raises.
- Destructors must not touch hardware; release happens in ``disconnect()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, IntEnum, IntFlag
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DeviceError(Exception):
    """Base class for all device errors."""


class DeviceConnectionError(DeviceError):
    """Port/camera could not be opened."""


class DeviceTimeoutError(DeviceError):
    """No reply within the transaction timeout."""


class ProtocolError(DeviceError):
    """CRC/framing/format error or a malformed device response."""


class CommandRejectedError(DeviceError):
    """Device refused the command (ERR:BAD_AXIS, ERR:RANGE, Modbus exc 0x03...)."""


class DeviceBusyError(DeviceError):
    """Command refused because the device/axis is busy."""


class LimitHitError(CommandRejectedError):
    """A limit switch gates the requested motion direction."""


class EStopError(DeviceError):
    """Emergency-stop condition is active."""


class NotConnectedError(DeviceError):
    """Operation attempted while the device is disconnected."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Axis(str, Enum):
    """Linear axes. For the Zolix XYR stage, Z means the rotation axis 'r'."""

    X = "X"
    Y = "Y"
    Z = "Z"


class Direction(IntEnum):
    POSITIVE = 1
    NEGATIVE = -1


class StageSpeed(IntEnum):
    SLOW = 0
    FAST = 1


class AxisMask(IntFlag):
    X = 1
    Y = 2
    Z = 4
    ALL = X | Y | Z


# ---------------------------------------------------------------------------
# Abstract devices
# ---------------------------------------------------------------------------

class AbstractDevice(ABC):
    """Common contract for every hardware device (and every simulation)."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._connected = False

    @abstractmethod
    def connect(self) -> None:
        """Open the port / acquire the camera. Raises DeviceConnectionError."""

    @abstractmethod
    def disconnect(self) -> None:
        """Idempotent release of resources. Never raises."""

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    @abstractmethod
    def device_id(self) -> str:
        """Human-readable identity, e.g. 'zolix@COM3'."""

    @abstractmethod
    def stop(self) -> None:
        """Emergency halt. Callable at any time. Never raises."""

    def drain_events(self) -> list[str]:
        """Non-blocking pull of unsolicited device events (EV:..., BOOT...).

        Devices without an async event channel return an empty list.
        The owning proxy calls this from its poll loop.
        """
        return []


class Camera(AbstractDevice):
    """Streaming camera. All methods are called from the owning proxy thread only."""

    @abstractmethod
    def start(self) -> None:
        """Backend-specific init: allocate buffers, start the stream."""

    @abstractmethod
    def fetch(self, timeout_ms: float = 2000.0) -> np.ndarray | None:
        """Grab the next frame as RGB uint8 HxWx3, or None on timeout.

        Called in a tight acquire loop; must not raise on transient
        timeouts. CONTRACT: the returned array is a FRESH C-contiguous
        allocation each call (never a reused buffer) — consumers may keep
        it (the proxy passes frames to the GUI by reference).
        """

    @abstractmethod
    def get_properties(self) -> dict[str, Any]:
        """Current exposure_us, gain, white_balance, roi, binning, framerate..."""

    @abstractmethod
    def set_property(self, name: str, value: Any) -> None:
        """Validate and apply one camera property."""

    @abstractmethod
    def snapshot(self, path: Path, timeout_s: float = 15.0,
                 resolution: int | None = None,
                 burn: dict | None = None) -> Path:
        """Capture one frame to ``path``; returns the path.

        ``resolution``: optional temporary mode switch (0 = 4K, 1 = 1080p)
        for the capture — the live stream pauses and is restored after.
        ``burn``: optional scale-bar burn request dict
        {"um_per_px": float} — the backend picks the bar length for the
        actual frame width.
        """

    def capture_time(self) -> float | None:
        """Capture-side monotonic timestamp of the last fetched frame, or
        None when the backend cannot report one. Concrete default so
        backends without timestamps need no change."""
        return None


class FocusStage(AbstractDevice):
    """1-DOF motorized focus (Arduino + CRD5103PB). No limit sensor."""

    @abstractmethod
    def move_rel(self, steps: int, speed: int | None = None) -> None:
        """MOVE:<rel> — returns once the command is accepted (not completed)."""

    @abstractmethod
    def move_abs(self, position: int, speed: int | None = None) -> None:
        """GOTO:<abs> — trapezoid profile with exact landing."""

    @abstractmethod
    def set_speed(self, steps_per_s: int) -> None:
        """SPD:<signed>; 0 = ramp stop."""

    @abstractmethod
    def zero(self) -> None:
        """ZERO — re-zero the position counter."""

    @abstractmethod
    def get_status(self):
        """STATUS? → FocusStatus."""

    @abstractmethod
    def get_soft_limits(self) -> tuple[int, int]:
        """Read SLIM soft limits."""

    @abstractmethod
    def set_soft_limits(self, lo: int, hi: int) -> None:
        """SLIM:SET — EEPROM write; called only from the Calibration wizard."""

    @abstractmethod
    def wait_idle(self, timeout_s: float = 120.0, poll_s: float = 0.05) -> None:
        """Poll STATUS? until MODE idle / EV:DONE. Raises DeviceTimeoutError."""


class XYRStage(AbstractDevice):
    """Zolix ZC300 sample stage: X, Y linear + rotation (logical axis 'r')."""

    @abstractmethod
    def move_abs_pulses(self, x: int, y: int, r: int | None = None,
                        speed_pps: int | None = None) -> None:
        """Absolute moves in pulses (opcode 0x0064 via fn 0x10)."""

    @abstractmethod
    def move_abs_um(self, x_um: float, y_um: float, r_deg: float | None = None,
                    speed: StageSpeed = StageSpeed.SLOW) -> None:
        """Absolute moves in engineering units (µm / deg)."""

    @abstractmethod
    def move_rel_um(self, dx_um: float, dy_um: float, dr_deg: float | None = None,
                    speed: StageSpeed = StageSpeed.SLOW) -> None:
        """Relative moves in engineering units."""

    @abstractmethod
    def home(self, axes: AxisMask) -> None:
        """Opcode 0x0069."""

    @abstractmethod
    def get_position(self) -> Any:
        """Read float positions (regs 30016-30021) → StagePosition."""

    @abstractmethod
    def get_status(self) -> Any:
        """Regs 30012-30015 → StageStatus (motion, limits, estop bit 9)."""

    @abstractmethod
    def wait_idle(self, timeout_s: float = 120.0, poll_s: float = 0.05) -> None:
        """Poll until no axis is moving."""

    @abstractmethod
    def configure_motion(self, accel_pps2: int, speeds: dict[str, int]) -> None:
        """Write accel/speed registers on connect; verify by readback."""

    @abstractmethod
    def save_parameters(self) -> None:
        """Opcode 0x006D — persist parameters to controller EEPROM."""

    @abstractmethod
    def check_estop(self) -> bool:
        """Status bit 9."""


class XYZStage(AbstractDevice):
    """DIY transfer stage (SigmaKoki + Arduino + Autonics drivers)."""

    @abstractmethod
    def move(self, axis: Axis, direction: Direction, level: int) -> None:
        """MV:<axis>:<dir>:<level> — continuous move at speed level."""

    @abstractmethod
    def step(self, axis: Axis, direction: Direction, steps: int) -> int:
        """STEP:<axis>:<dir>:<steps> → returns actual steps moved."""

    @abstractmethod
    def move_rel_um(self, axis: Axis, um: float, level: int) -> int:
        """Steps = round(um / um_per_step[axis]); returns actual steps moved."""

    @abstractmethod
    def set_speed(self, axis: Axis, level: int) -> None:
        """SPD:<axis>:<level>, level 0..5."""

    @abstractmethod
    def home(self) -> None:
        """HOME — re-zero only, no homing motion."""

    @abstractmethod
    def stop(self, axes: AxisMask | None = None) -> None:
        """STOP:<axis> / STOP:ALL."""

    @abstractmethod
    def get_limits(self) -> dict[str, bool]:
        """LIMITS? → per-direction limit state, keys 'x+', 'x-', 'y+', ... (1 = triggered)."""

    @abstractmethod
    def get_status(self) -> dict[str, str]:
        """STATUS? raw parsed fields."""

    @abstractmethod
    def wait_idle(self, timeout_s: float = 300.0, poll_s: float = 0.05) -> None:
        """Poll STATUS? until not busy."""


class TemperatureController(AbstractDevice):
    """Yudian AI-828, Modbus RTU."""

    @abstractmethod
    def read_pv(self) -> float:
        """Process value (°C), reg 75 / 10^dPt."""

    @abstractmethod
    def read_sv(self) -> float:
        """Setpoint (°C), reg 76."""

    @abstractmethod
    def set_sv(self, temp_c: float) -> None:
        """Clamp to configured safety range, write reg 40001, verify readback."""

    @abstractmethod
    def read_output_percent(self) -> float:
        """Reg 77 low byte."""

    @abstractmethod
    def read_decimal_point(self) -> int:
        """Reg 13 (dPt), cached at connect."""


class Nosepiece(AbstractDevice):
    """Objective nosepiece sense/control — RESERVED.

    The current microscope cannot sense or rotate the nosepiece, so v1 ships
    ``NullNosepiece``. Future hardware = one new adapter + registry entry.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """True if the hardware can sense/control the nosepiece."""

    @abstractmethod
    def get_position(self) -> int | None:
        """Current nosepiece position, or None if unknown."""

    @abstractmethod
    def set_position(self, pos: int) -> None:
        """Rotate the nosepiece to a position."""

    def select_objective(self, objective_id: int) -> None:
        """Select an objective by id. Raises CommandRejectedError if unavailable."""
        raise CommandRejectedError("Nosepiece control not available on this microscope")


class NullNosepiece(Nosepiece):
    """v1 nosepiece: the microscope provides neither sensing nor control."""

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def device_id(self) -> str:
        return "nosepiece@none"

    def stop(self) -> None:
        pass

    def is_available(self) -> bool:
        return False

    def get_position(self) -> int | None:
        return None

    def set_position(self, pos: int) -> None:
        raise CommandRejectedError("Nosepiece control not available on this microscope")
