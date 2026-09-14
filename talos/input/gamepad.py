"""Gamepad input via raw XInput (ctypes) — the `inputs` package hangs on
this machine's gamepad setup, so we poll XInputGetState directly (60 Hz,
non-blocking).

The controller exposes a GamepadState snapshot each tick; the
ActionResolver (reference port) does all the mapping logic.
"""

from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

logger = logging.getLogger(__name__)

DEADZONE = 0.2
GAMMA = 2.2

# XInput button bitmask (WORD).
_XINPUT_GAMEPAD_DPAD_UP = 0x0001
_XINPUT_GAMEPAD_DPAD_DOWN = 0x0002
_XINPUT_GAMEPAD_DPAD_LEFT = 0x0004
_XINPUT_GAMEPAD_DPAD_RIGHT = 0x0008
_XINPUT_GAMEPAD_START = 0x0010
_XINPUT_GAMEPAD_BACK = 0x0020
_XINPUT_GAMEPAD_LEFT_SHOULDER = 0x0100
_XINPUT_GAMEPAD_RIGHT_SHOULDER = 0x0200
_XINPUT_GAMEPAD_A = 0x1000
_XINPUT_GAMEPAD_B = 0x2000
_XINPUT_GAMEPAD_X = 0x4000
_XINPUT_GAMEPAD_Y = 0x8000

_ERROR_SUCCESS = 0
_ERROR_DEVICE_NOT_CONNECTED = 1167


class _XInputGamepad(ctypes.Structure):
    _fields_ = [("wButtons", ctypes.c_ushort), ("bLeftTrigger", ctypes.c_ubyte),
                ("bRightTrigger", ctypes.c_ubyte), ("sThumbLX", ctypes.c_short),
                ("sThumbLY", ctypes.c_short), ("sThumbRX", ctypes.c_short),
                ("sThumbRY", ctypes.c_short)]


class _XInputState(ctypes.Structure):
    _fields_ = [("dwPacketNumber", ctypes.c_ulong), ("Gamepad", _XInputGamepad)]


def _load_xinput():
    for name in ("xinput1_4", "xinput9_1_0"):
        try:
            dll = getattr(ctypes.windll, name)
        except (AttributeError, OSError):
            continue
        dll.XInputGetState.argtypes = [ctypes.c_uint, ctypes.POINTER(_XInputState)]
        dll.XInputGetState.restype = ctypes.c_uint
        return dll
    return None


@dataclass
class GamepadState:
    """One tick of raw gamepad state (analog values -1..1)."""
    connected: bool = False
    left_x: float = 0.0
    left_y: float = 0.0
    right_x: float = 0.0
    right_y: float = 0.0
    left_trigger: float = 0.0   # 0..1
    right_trigger: float = 0.0  # 0..1
    dpad_up: bool = False
    dpad_down: bool = False
    dpad_left: bool = False
    dpad_right: bool = False
    button_start: bool = False
    button_back: bool = False
    button_a: bool = False
    button_b: bool = False
    button_x: bool = False
    button_y: bool = False
    button_left_shoulder: bool = False
    button_right_shoulder: bool = False
    buttons: int = 0
    edges: dict[str, bool] = field(default_factory=dict)  # button → pressed-this-tick


def normalize_axis(value: int, deadzone: float, gamma: float) -> float:
    """Map a signed 16-bit axis reading to -1..1 with deadzone + gamma."""
    v = value / 32768.0
    if abs(v) < deadzone:
        return 0.0
    magnitude = (abs(v) - deadzone) / (1.0 - deadzone)
    magnitude = magnitude ** gamma
    return magnitude if v > 0 else -magnitude


def map_button_event(event) -> tuple[str, dict] | None:
    """Pure mapping for discrete buttons (kept for tests/porting)."""
    code = event.code
    state = event.state
    if code in ("BTN_START", "BTN_SELECT") and state:
        return ("stop_all", {})
    if code == "BTN_SOUTH" and state:
        return ("preset", {"index": 0})
    if code == "BTN_EAST" and state:
        return ("preset", {"index": 1})
    if code == "BTN_WEST" and state:
        return ("preset", {"index": 2})
    if code == "BTN_NORTH" and state:
        return ("preset", {"index": 3})
    if code == "ABS_HAT0Y" and state != 0:
        return ("focus_step", {"direction": -1 if state > 0 else 1})
    return None


class GamepadController(QObject):
    """Polls XInput and publishes a GamepadState + button edges."""

    sig_state = Signal(object)         # GamepadState
    sig_connected = Signal(bool)

    def __init__(self, settings=None, poll_hz: int = 60, parent: QObject | None = None):
        super().__init__(parent)
        cfg = (settings.section("input").get("gamepad", {}) if settings else {})
        self._deadzone = float(cfg.get("deadzone", DEADZONE))
        self._gamma = float(cfg.get("gamma", GAMMA))
        self._dll = None
        self._prev_buttons = 0
        self._was_connected = False
        self._timer = QTimer(self)
        self._timer.setInterval(1000 // poll_hz)
        self._timer.timeout.connect(self._poll)

    def start(self) -> bool:
        self._dll = _load_xinput()
        if self._dll is None:
            logger.info("XInput unavailable — no gamepad support")
            return False
        self._timer.start()
        return True

    def stop(self) -> None:
        self._timer.stop()

    # ------------------------------------------------------------------

    def _poll(self) -> None:
        if self._dll is None:
            return
        state = _XInputState()
        rc = self._dll.XInputGetState(0, ctypes.byref(state))
        connected = rc == _ERROR_SUCCESS
        if connected != self._was_connected:
            self._was_connected = connected
            self.sig_connected.emit(connected)
        if not connected:
            return
        pad = state.Gamepad
        buttons = int(pad.wButtons)

        def bit(mask: int) -> bool:
            return bool(buttons & mask)

        edges = {}
        for name, mask in (
            ("start", _XINPUT_GAMEPAD_START), ("back", _XINPUT_GAMEPAD_BACK),
            ("a", _XINPUT_GAMEPAD_A), ("b", _XINPUT_GAMEPAD_B),
            ("x", _XINPUT_GAMEPAD_X), ("y", _XINPUT_GAMEPAD_Y),
            ("lb", _XINPUT_GAMEPAD_LEFT_SHOULDER),
            ("rb", _XINPUT_GAMEPAD_RIGHT_SHOULDER),
            ("dpad_up", _XINPUT_GAMEPAD_DPAD_UP),
            ("dpad_down", _XINPUT_GAMEPAD_DPAD_DOWN),
            ("dpad_left", _XINPUT_GAMEPAD_DPAD_LEFT),
            ("dpad_right", _XINPUT_GAMEPAD_DPAD_RIGHT),
        ):
            edges[name] = bit(mask) and not (self._prev_buttons & mask)
        self._prev_buttons = buttons

        self.sig_state.emit(GamepadState(
            connected=True,
            left_x=normalize_axis(pad.sThumbLX, self._deadzone, self._gamma),
            left_y=normalize_axis(pad.sThumbLY, self._deadzone, self._gamma),
            right_x=normalize_axis(pad.sThumbRX, self._deadzone, self._gamma),
            right_y=normalize_axis(pad.sThumbRY, self._deadzone, self._gamma),
            left_trigger=pad.bLeftTrigger / 255.0,
            right_trigger=pad.bRightTrigger / 255.0,
            dpad_up=bit(_XINPUT_GAMEPAD_DPAD_UP),
            dpad_down=bit(_XINPUT_GAMEPAD_DPAD_DOWN),
            dpad_left=bit(_XINPUT_GAMEPAD_DPAD_LEFT),
            dpad_right=bit(_XINPUT_GAMEPAD_DPAD_RIGHT),
            button_start=bit(_XINPUT_GAMEPAD_START),
            button_back=bit(_XINPUT_GAMEPAD_BACK),
            button_a=bit(_XINPUT_GAMEPAD_A),
            button_b=bit(_XINPUT_GAMEPAD_B),
            button_x=bit(_XINPUT_GAMEPAD_X),
            button_y=bit(_XINPUT_GAMEPAD_Y),
            button_left_shoulder=bit(_XINPUT_GAMEPAD_LEFT_SHOULDER),
            button_right_shoulder=bit(_XINPUT_GAMEPAD_RIGHT_SHOULDER),
            buttons=buttons,
            edges=edges,
        ))
