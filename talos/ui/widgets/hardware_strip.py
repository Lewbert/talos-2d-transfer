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
from talos.ui.theme import BORDER, DANGER, LED_OFF, OK, PANEL, WARN
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

    try:
        speed_hz = float(SPEED_LEVEL_TO_HZ.get(int(status.get("zspd", 0)), 0.0))
    except (TypeError, ValueError):
        speed_hz = 0.0
    return {"x": steps("X"), "y": steps("Y"), "z": steps("Z"),
            "speed_hz": speed_hz}


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
    """6px status dot (lit = triggered)."""

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
    """A strip section: title row + body."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame {{ border: 1px solid {BORDER}; border-radius: 4px;"
            f" background: {PANEL}; }}")
        self._title = QLabel(title)
        self._title.setObjectName("dim")
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
    """XYR/XYZ compact status: enable toggle, µm readout, limit dots."""

    def __init__(self, title: str, device_key: str, axes: list[str],
                 manager, settings, parent=None):
        super().__init__(title, parent)
        self._axes = axes
        self._enable = QCheckBox("Enable")
        self._enable.setChecked(True)  # matches the manager's default gate
        self._enable.toggled.connect(
            lambda on: manager.set_enabled(device_key, on))
        self.add(self._enable)

        self._pos = QLabel("—")
        self._pos.setObjectName("readout")
        self._pos.setMinimumWidth(120)
        self._pos.setAlignment(Qt.AlignmentFlag.AlignRight
                               | Qt.AlignmentFlag.AlignVCenter)
        self.add(self._pos)

        self._dots: dict[str, _MiniDot] = {}
        for key in ("x+", "x-", "y+", "y-"):
            dot = _MiniDot(self)
            dot.setToolTip(f"limit {key}")
            self._dots[key] = dot
            self.add(dot)
        self._estop = QLabel("E-STOP")
        self._estop.setObjectName("ok")
        self._estop.setStyleSheet(f"color: {DANGER}; font-weight: 700;")
        self._estop.hide()
        self.add(self._estop)

        self._moving = QLabel("MOV")
        self._moving.setStyleSheet(f"color: {OK};")
        self._moving.hide()
        self.add(self._moving)

    def set_telem(self, parsed: dict) -> None:
        if "r_deg" in parsed:  # zolix (its r_deg is the discriminator)
            self._pos.setText(
                f"{parsed['x_um']:.1f} · {parsed['y_um']:.1f} µm · "
                f"{parsed['r_deg']:.2f}°")
            for key, dot in self._dots.items():
                dot.set_on(bool(parsed["limits"].get(key)))
            self._estop.setVisible(bool(parsed.get("estop")))
        else:  # sigmakoki (steps → µm done by the strip's caller)
            self._pos.setText(
                f"{parsed['x_um']:.1f} · {parsed['y_um']:.1f} · "
                f"{parsed['z_um']:.1f} µm")
            for dot in self._dots.values():
                dot.set_on(False)
            self._estop.hide()
            self._moving.hide()  # no busy flag — speed level is not motion
        self._moving.setVisible(bool(parsed.get("moving")))


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

        self._xyr = _StageSection("XYR STAGE", "zolix", ["x", "y", "r"],
                                  manager, settings)
        self._xyr.setMinimumWidth(240)  # equalized: the readouts'
        layout.addWidget(self._xyr, stretch=1)  # min-widths would otherwise
        self._xyz = _StageSection("XYZ STAGE", "sigmakoki", ["x", "y", "z"],
                                  manager, settings)
        self._xyz.setMinimumWidth(240)  # dominate the equal stretch
        layout.addWidget(self._xyz, stretch=1)

        focus = _Section("FOCUS")
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

        temp = _Section("TEMP")
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

    def trigger_bar(self) -> TriggerBarWidget:
        return self._triggers

    def update_telem(self, device_key: str, payload: dict) -> None:
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
            if blocked:
                self._focus_state.setText("⚠ BLOCKED")
                self._focus_state.setStyleSheet(
                    f"color: {WARN}; font-weight: 700;")
            else:
                self._focus_state.setText(parsed["mode"])
                self._focus_state.setStyleSheet("")
        elif device_key == "yudian":
            parsed = parse_yudian(payload)
            if parsed["pv"] is not None:
                self._temp_pv.setText(f"{parsed['pv']:.1f} °C")
            if parsed["sv"] is not None:
                self._temp_sv.setText(f"SV {parsed['sv']:.1f} °C")
            if parsed["out"] is not None:
                self._temp_out.setText(f"{parsed['out']:.0f} %")
