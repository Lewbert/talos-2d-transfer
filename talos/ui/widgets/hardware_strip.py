"""Bottom hardware strip: compact XYR/XYZ/Focus/Temp status (the
simplified transfer-stage-control bottom bar). Status + enable toggles
only — all control buttons live in the StageControlWindow.

The parsers use the CORRECT telemetry keys (the old stage panels read
stale ones): zolix position is ``x_pulses/.../x_um/...`` and limits are
``limit_x_pos/...`` (``any_moving`` is a property and never serialized —
moving is derived from the per-axis flags); sigmakoki position is
``dict[Axis, int]`` and its status is a ``dict[str, str]``.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from talos.hal.base import Axis
from talos.hal.devices.sigmakoki import SPEED_LEVEL_TO_HZ
from talos.ui.theme import DANGER, LED_OFF
from talos.ui.widgets.trigger_bar import TriggerBarWidget


# --- pure parsers (unit-tested) ----------------------------------------


def parse_zolix(payload: dict) -> dict:
    pos = payload.get("position") or {}
    status = payload.get("status") or {}
    return {
        "x_um": float(pos.get("x_um", 0.0)),
        "y_um": float(pos.get("y_um", 0.0)),
        "r_deg": float(pos.get("r_deg", 0.0)),
        "moving": bool(status.get("x_moving") or status.get("y_moving")
                       or status.get("r_moving")),
        "limits": {
            "x+": bool(status.get("limit_x_pos")),
            "x-": bool(status.get("limit_x_neg")),
            "y+": bool(status.get("limit_y_pos")),
            "y-": bool(status.get("limit_y_neg")),
        },
        "estop": bool(status.get("estop")),
    }


def parse_sigmakoki(payload: dict) -> dict:
    """STATUS? carries POSITIONS as x/y/z and SPEED LEVELS as xspd/yspd/
    zspd.

    The levels are a SETTING, not a motion flag: the firmware's default is
    level 2 and ``stopAxis()`` clears ``moving`` but NOT ``speed_level``
    (verified in transfer_stage_controller.ino), so a level > 0 would light
    a "moving" lamp forever. Motion is inferred from the position changing
    between polls — the same settle detection the driver's ``wait_idle``
    uses (the firmware has no per-axis busy flag).
    """
    status = payload.get("status") or {}
    pos = payload.get("position") or {}

    def steps(key: str) -> int:
        value = pos.get(getattr(Axis, key, None))
        if value is not None:
            return int(value)
        try:
            # the status dict uses lowercase keys ("x"/"y"/"z")
            return int(status.get(key.lower(), 0))
        except (TypeError, ValueError):
            return 0

    def level(key: str) -> int:
        try:
            return int(status.get(key, 0))
        except (TypeError, ValueError):
            return 0

    levels = {axis: level(f"{axis}spd") for axis in ("x", "y", "z")}
    return {"x": steps("X"), "y": steps("Y"), "z": steps("Z"),
            "levels": levels,
            "speed_hz": float(SPEED_LEVEL_TO_HZ.get(levels["z"], 0.0))}


def parse_focus(payload: dict) -> dict:
    status = payload.get("status") or {}
    return {
        "pos": int(status.get("pos", 0)),
        "mode": str(status.get("mode", "—")),
        "slim_bounds": payload.get("slim_bounds"),
        "slim_on": bool(status.get("slim_on")),
        "blocked": str(status.get("blocked_dir", "0")),
    }


def parse_yudian(payload: dict) -> dict:
    return {
        "pv": payload.get("pv"),
        "sv": payload.get("sv"),
        "out": payload.get("output_percent"),
    }


# --- widgets ------------------------------------------------------------


class _MiniDot(QWidget):
    """8px status dot (lit = triggered)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._on = False
        self.setFixedSize(8, 8)

    def set_on(self, on: bool) -> None:
        if on != self._on:
            self._on = on
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(DANGER if self._on else LED_OFF))
        painter.drawEllipse(0, 0, 8, 8)


class _Section(QFrame):
    """A strip section: title row + body. Styled by objectName (theme QSS)
    — inline stylesheets with hardcoded colours are the house anti-pattern."""

    def __init__(self, title: str, tooltip: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("strip_section")
        self._title = QLabel(title)
        self._title.setObjectName("dim")
        if tooltip:
            self.setToolTip(tooltip)
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 3, 6, 3)
        root.setSpacing(2)
        root.addWidget(self._title)
        self._body = QHBoxLayout()
        self._body.setSpacing(4)
        root.addLayout(self._body)

    def add(self, widget: QWidget, stretch: int = 0) -> None:
        self._body.addWidget(widget, stretch)


class _StageSection(_Section):
    """XYR/XYZ compact status: enable toggle, axis readout, MOV (a motion
    command is in flight / the axis is turning) and — where the hardware
    has them — limit dots and the E-STOP latch.

    Both stages behave identically: MOV lights for the same reason, the
    readout uses the same fixed-width format, and every optional indicator
    keeps its slot reserved so lighting up never re-flows the bar.
    """

    def __init__(self, title: str, device_key: str, tooltip: str,
                 manager, settings, *, limits: bool, parent=None):
        super().__init__(title, tooltip, parent)
        self._enable = QCheckBox("Enable")
        self._enable.setChecked(True)  # matches the manager's default gate
        self._enable.setToolTip(f"Enable commands for {tooltip or title}")
        self._enable.toggled.connect(
            lambda on: manager.set_enabled(device_key, on))
        self.add(self._enable)

        self._pos = QLabel("—")
        self._pos.setObjectName("readout")
        self._pos.setMinimumWidth(150)
        self._pos.setAlignment(Qt.AlignmentFlag.AlignRight
                               | Qt.AlignmentFlag.AlignVCenter)
        self.add(self._pos, stretch=1)

        self._dots: dict[str, _MiniDot] = {}
        if limits:
            for key in ("x+", "x-", "y+", "y-"):
                dot = _MiniDot(self)
                dot.setToolTip(f"{device_key} limit {key}")
                self._dots[key] = dot
                self.add(dot)
        # Reserved slots (fixed width, text toggled): showing MOV/E-STOP
        # must not move the neighbouring sections around.
        self._estop = QLabel("")
        self._estop.setObjectName("strip_estop")
        self._estop.setFixedWidth(52)
        self._estop.setToolTip("Emergency-stop bit is set on the controller")
        if limits:
            self.add(self._estop)

        self._moving = QLabel("")
        self._moving.setObjectName("strip_mov")
        self._moving.setFixedWidth(34)
        self._moving.setToolTip("The axis is turning")
        self.add(self._moving)
        self._last_pos: tuple | None = None

    def _is_moving(self, parsed: dict) -> bool:
        """zolix reports real per-axis moving bits; sigmakoki has none, so
        its motion is the position changing between polls (the driver's own
        settle detection — the speed level persists after a stop)."""
        if "moving" in parsed:
            return bool(parsed["moving"])
        pos = (parsed["x"], parsed["y"], parsed["z"])
        moved = self._last_pos is not None and pos != self._last_pos
        self._last_pos = pos
        return moved

    def set_telem(self, parsed: dict) -> None:
        if "r_deg" in parsed:  # zolix (its r_deg is the discriminator)
            self._pos.setText(
                f"{parsed['x_um']:8.1f} {parsed['y_um']:8.1f} µm "
                f"{parsed['r_deg']:6.2f}°")
            for key, dot in self._dots.items():
                dot.set_on(bool(parsed["limits"].get(key)))
            self._estop.setText("E-STOP" if parsed.get("estop") else "")
        else:  # sigmakoki (steps → µm done by the strip's caller)
            self._pos.setText(
                f"{parsed['x_um']:8.1f} {parsed['y_um']:8.1f} "
                f"{parsed['z_um']:7.1f} µm")
        self._moving.setText("MOV" if self._is_moving(parsed) else "")


def format_focus_pos(pos_steps: int, um_per_step: float) -> str:
    """The strip's FOCUS readout: µm + steps."""
    return f"{pos_steps * um_per_step:.1f} µm · {pos_steps} st"


class HardwareStrip(QWidget):
    """Shared bottom strip — one instance, owned by MainWindow."""

    def __init__(self, manager, settings, input_system=None, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._manager = manager
        self.reload_settings()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(6)

        self._xyr = _StageSection("XYR", "zolix", "Zolix XYR sample stage",
                                  manager, settings, limits=True)
        self._xyr.setMinimumWidth(240)  # equalized: the readouts'
        layout.addWidget(self._xyr, stretch=1)  # min-widths would otherwise
        self._xyz = _StageSection("XYZ", "sigmakoki",
                                  "SigmaKoki XYZ transfer stage",
                                  manager, settings, limits=False)
        self._xyz.setMinimumWidth(240)  # dominate the equal stretch
        layout.addWidget(self._xyz, stretch=1)

        focus = _Section("FOCUS", "Focus stage (no limit sensor)")
        focus.setMinimumWidth(240)
        self._focus_pos = QLabel("—")
        self._focus_pos.setObjectName("readout")
        self._focus_pos.setMinimumWidth(140)  # fits "123.4 µm · 123456 st"
        self._focus_pos.setAlignment(Qt.AlignmentFlag.AlignRight
                                     | Qt.AlignmentFlag.AlignVCenter)
        focus.add(self._focus_pos)
        self._focus_state = QLabel("—")
        self._focus_state.setObjectName("dim")
        focus.add(self._focus_state)
        self._triggers = TriggerBarWidget(settings)
        focus.add(self._triggers, stretch=1)
        layout.addWidget(focus, stretch=1)

        temp = _Section("TEMP", "Yudian AI-828 temperature controller")
        temp.setMinimumWidth(240)
        self._temp_pv = QLabel("—")
        self._temp_pv.setObjectName("readout")
        self._temp_pv.setMinimumWidth(52)
        self._temp_pv.setAlignment(Qt.AlignmentFlag.AlignRight
                                   | Qt.AlignmentFlag.AlignVCenter)
        temp.add(self._temp_pv)
        self._temp_sv = QLabel("SV —")
        self._temp_sv.setObjectName("dim")
        temp.add(self._temp_sv)
        self._temp_out = QLabel("0%")
        self._temp_out.setObjectName("dim")
        temp.add(self._temp_out)
        layout.addWidget(temp, stretch=1)

        layout.addStretch(0)
        self.setFixedHeight(56)

    def reload_settings(self) -> None:
        """Re-read the scale factors the readouts convert with.

        They are cached at construction, so a Preferences edit to
        µm-per-pulse would otherwise keep converting with the old value
        until the next restart."""
        cfg_z = self._settings.device("zolix")
        cfg_s = self._settings.device("sigmakoki")
        self._um_xy_zolix = float(cfg_z.get("um_per_pulse_xy", 0.625))
        self._um_xy_sig = float(cfg_s.get("um_per_step_xy", 0.5))
        self._um_z_sig = float(cfg_s.get("um_per_step_z", 0.25))
        self._um_focus = float(self._settings.device("focus").get("um_per_step", 0.2))
        # The trigger readout caches the focus curve + jog inversion too.
        # getattr: reload_settings() also runs from __init__, BEFORE the
        # trigger bar exists.
        triggers = getattr(self, "_triggers", None)
        if triggers is not None:
            triggers.reload_settings()

    def trigger_bar(self) -> TriggerBarWidget:
        return self._triggers

    def update_telem(self, device_key: str, payload: dict) -> None:
        # The enable gate can change from outside this checkbox (the
        # gamepad's Start button toggles the D-pad-selected stage).
        if "enabled" in payload:
            section = {"zolix": self._xyr, "sigmakoki": self._xyz}.get(device_key)
            if section is not None:
                box = section._enable
                box.blockSignals(True)
                box.setChecked(bool(payload["enabled"]))
                box.blockSignals(False)
        if device_key == "zolix":
            self._xyr.set_telem(parse_zolix(payload))
        elif device_key == "sigmakoki":
            parsed = parse_sigmakoki(payload)
            parsed["x_um"] = parsed["x"] * self._um_xy_sig
            parsed["y_um"] = parsed["y"] * self._um_xy_sig
            parsed["z_um"] = parsed["z"] * self._um_z_sig
            self._xyz.set_telem(parsed)
        elif device_key == "focus":
            parsed = parse_focus(payload)
            self._focus_pos.setText(
                format_focus_pos(parsed["pos"], self._um_focus))
            blocked = parsed["blocked"] not in ("0", "")
            self._focus_state.setText("⚠ BLOCKED" if blocked
                                      else parsed["mode"])
            # objectName swap + re-polish instead of an inline stylesheet
            # (the theme's rule wins over QStatusBar's dim descendant rule)
            self._focus_state.setObjectName("strip_warn" if blocked else "dim")
            self._focus_state.style().unpolish(self._focus_state)
            self._focus_state.style().polish(self._focus_state)
        elif device_key == "yudian":
            parsed = parse_yudian(payload)
            if parsed["pv"] is not None:
                self._temp_pv.setText(f"{parsed['pv']:.1f} °C")
            if parsed["sv"] is not None:
                self._temp_sv.setText(f"SV {parsed['sv']:.1f} °C")
            if parsed["out"] is not None:
                self._temp_out.setText(f"{parsed['out']:.0f} %")
