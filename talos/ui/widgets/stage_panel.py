"""Reference-style stage panel (ported from transfer-stage-control's
gui/stage_panel.py + axis_control_buttons.py).

Per stage: enable checkbox, axis table (steps | converted | limit
circles), live speed, and a 4×2 button grid — press-and-hold =
continuous (300 ms), quick click = single step — plus STOP and ZERO.
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from talos.ui.theme import DANGER, LED_OFF, TEXT_DIM


class _LimitDot(QWidget):
    """Small circular limit indicator (lit = triggered)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._triggered = False
        self.setFixedSize(10, 10)

    def set_triggered(self, triggered: bool) -> None:
        if triggered != self._triggered:
            self._triggered = triggered
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        from PySide6.QtGui import QColor, QPainter

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(DANGER if self._triggered else LED_OFF)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(0, 0, 10, 10)


class _HoldButton(QPushButton):
    """Press = hold start, quick click = single step, release = stop."""

    sig_press = Signal()
    sig_click = Signal()
    sig_release = Signal()

    def __init__(self, text: str, long_press_ms: int = 300, parent=None):
        super().__init__(text, parent)
        self._held = False
        self._clicked_fired = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(long_press_ms)
        self._timer.timeout.connect(self._on_long)
        self.pressed.connect(self._on_press)
        self.released.connect(self._on_release)

    def _on_press(self) -> None:
        self._held = False
        self._clicked_fired = False
        self._timer.start()

    def _on_long(self) -> None:
        self._held = True
        self.sig_press.emit()

    def _on_release(self) -> None:
        self._timer.stop()
        if self._held:
            self.sig_release.emit()
        elif not self._clicked_fired:
            self._clicked_fired = True
            self.sig_click.emit()

    def cancel_hold(self) -> None:
        """End a hold early — the pointer left the button, or the window
        was hidden/ESC'd while held.

        Safety: the release signal is the ONLY thing that clears the
        input system's claim, so a hold that never releases leaves the
        axis jogging with no button pressed. `_clicked_fired` is set so
        the physical release that follows is not mistaken for a short
        click (which would fire a phantom single step).
        """
        if self._held:
            self._held = False
            self._clicked_fired = True
            self.sig_release.emit()

    def leaveEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        # Drag off the button while holding = release (dead-man behavior).
        self.cancel_hold()
        super().leaveEvent(event)


class ReferenceStagePanel(QGroupBox):
    """Reference-faithful per-stage manual control panel.

    Holds feed the InputSystem (ui_hold/ui_release) so the resolver's
    claim/priority logic sees UI buttons exactly like the reference;
    clicks go through ui_click (0.2 s cooldown).
    """

    def __init__(self, stage_id: str, title: str, axes: list[str],
                 input_system, manager, settings, state=None, parent=None):
        super().__init__(title, parent)
        self._stage_id = stage_id
        self._axes = axes
        self._input = input_system
        self._manager = manager
        self._settings = settings
        self._state = state
        self._buttons: dict[str, _HoldButton] = {}
        self._limit_dots: dict[str, _LimitDot] = {}
        self._pos_labels: dict[str, QLabel] = {}
        self._conv_labels: dict[str, QLabel] = {}

        layout = QVBoxLayout(self)

        # Enable checkbox (software gate in the manager).
        self._enable = QCheckBox("Enabled")
        self._enable.setChecked(True)
        self._enable.toggled.connect(
            lambda on: manager.set_enabled(stage_id, on))
        layout.addWidget(self._enable)

        # Axis table: axis | steps | converted | limit dots.
        grid = QGridLayout()
        grid.addWidget(QLabel(""), 0, 0)
        grid.addWidget(QLabel("Position"), 0, 1, 1, 2)
        grid.addWidget(QLabel("Limits"), 0, 3)
        for row, axis in enumerate(axes, start=1):
            grid.addWidget(QLabel(axis.upper()), row, 0)
            pos = QLabel("0")
            pos.setAlignment(Qt.AlignmentFlag.AlignRight)
            self._pos_labels[axis] = pos
            grid.addWidget(pos, row, 1)
            conv = QLabel("")
            conv.setAlignment(Qt.AlignmentFlag.AlignRight)
            self._conv_labels[axis] = conv
            grid.addWidget(conv, row, 2)
            dots = QHBoxLayout()
            for sign in ("+", "-"):
                dot = _LimitDot()
                self._limit_dots[f"{axis}{sign}"] = dot
                dots.addWidget(dot)
            dots.addStretch(1)
            grid.addLayout(dots, row, 3)
        layout.addLayout(grid)

        # Speed line.
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed:"))
        self._speed_label = QLabel("0 step/s")
        self._speed_label.setStyleSheet(f"color: {TEXT_DIM};")
        speed_row.addWidget(self._speed_label)
        speed_row.addStretch(1)
        layout.addLayout(speed_row)

        # 4×2 button grid: X± / Y± / Z±(R±) / STOP | ZERO. The jog
        # buttons FILL their stretched columns (~2.5:1 at 56 px tall —
        # the transfer-stage reference ratio); STOP/ZERO are ~1.4×
        # taller. NOTE: no AlignCenter here — an aligned grid item
        # renders at its natural sizeHint width and ignores fixed
        # widths (Qt behavior), which would leave 54 px slivers.
        buttons = QGridLayout()
        buttons.setColumnStretch(0, 1)
        buttons.setColumnStretch(1, 1)
        zr_axis = "z" if "z" in axes else "r"
        zr_pos = "Z+" if "z" in axes else "R+"
        zr_neg = "Z-" if "z" in axes else "R-"
        for col, (axis, direction, text) in enumerate((
            ("x", 1, "X+"), ("x", -1, "X-"),
            ("y", 1, "Y+"), ("y", -1, "Y-"),
            (zr_axis, 1, zr_pos), (zr_axis, -1, zr_neg),
        )):
            btn = _HoldButton(text)
            btn.setFixedHeight(56)
            btn.sig_press.connect(
                lambda a=axis, d=direction: self._on_hold(a, d))
            btn.sig_click.connect(
                lambda a=axis, d=direction: self._on_click(a, d))
            btn.sig_release.connect(
                lambda a=axis: self._on_release(a))
            self._buttons[axis] = btn
            buttons.addWidget(btn, col // 2, col % 2)
        stop = QPushButton("STOP")
        stop.setObjectName("danger")
        stop.setFixedHeight(80)
        stop.clicked.connect(lambda: manager.stop_all())
        buttons.addWidget(stop, 3, 0)
        zero = QPushButton("ZERO")
        zero.setFixedHeight(80)
        zero.clicked.connect(self._on_zero)
        buttons.addWidget(zero, 3, 1)
        layout.addLayout(buttons)

    # ------------------------------------------------------------------

    def _claim(self, axis: str) -> str:
        return f"{self._stage_id}:{axis}"

    def _on_hold(self, axis: str, direction: int) -> None:
        if self._input is not None:
            self._input.ui_hold(self._claim(axis), direction)

    def _on_release(self, axis: str) -> None:
        if self._input is not None:
            self._input.ui_release(self._claim(axis))

    def _on_click(self, axis: str, direction: int) -> None:
        if self._input is not None:
            self._input.ui_click(self._stage_id, axis, direction)

    def _motion_allowed(self) -> bool:
        """ZERO/home commands motion (Zolix re-homes, hardware-unvalidated):
        it must obey the same mode gate as every jog — before this it was
        reachable mid-scan and mid-autofocus."""
        return getattr(self._state, "mode", "MANUAL") == "MANUAL"

    def _on_zero(self) -> None:
        if not self._motion_allowed():
            mode = getattr(self._state, "mode", "?")
            QMessageBox.information(
                self, "ZERO",
                f"The {self._stage_id} stage is in use ({mode}) — "
                "re-zeroing is refused while a job owns the axes.")
            return
        answer = QMessageBox.question(
            self, "ZERO",
            f"Re-zero the {self._stage_id} stage?\n"
            "Zolix: home opcode (hardware-unvalidated — hand near STOP).\n"
            "SigmaKoki: counters only, no motion.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        if answer != QMessageBox.StandardButton.Ok:
            return
        if self._stage_id == "sigmakoki":
            self._manager.submit("sigmakoki", "home")
        else:
            self._manager.submit("zolix", "home")

    # ------------------------------------------------------------------

    def update_telem(self, payload: dict) -> None:
        """Refresh display from manager telemetry."""
        pos = payload.get("position") or {}
        status = payload.get("status") or {}
        for axis in self._axes:
            # Zolix telemetry keys are <axis>_pulses (the raw "x"/"y" keys
            # never exist, so X and Y read a permanent 0 while the scale
            # conversion below reported 0.00 µm).
            steps = int(pos.get(f"{axis}_pulses", pos.get(axis, 0)))
            self._pos_labels[axis].setText(f"{steps}")
            factor = self._factor(axis)
            if factor:
                # the R axis is a rotation — degrees, not µm
                unit = "°" if axis == "r" else "µm"
                self._conv_labels[axis].setText(
                    f"{steps * factor:.2f} {unit}")
        # Limit dots.
        if "limits" in payload:
            for key, dot in self._limit_dots.items():
                dot.set_triggered(bool(payload["limits"].get(key, False)))
        elif self._stage_id == "zolix":
            for key, dot in self._limit_dots.items():
                dot.set_triggered(bool(status.get(f"limit_{key}", False)))
        # Speed: max of status speeds (raw fields vary per device).
        speeds = []
        if self._stage_id == "sigmakoki":
            for axis in self._axes:
                try:
                    speeds.append(SPEED_LEVEL_HZ[int(status.get(f"{axis}spd", 0))])
                except (KeyError, ValueError, TypeError):
                    pass
        elif self._stage_id == "zolix":
            moving = status.get("any_moving", False)
            speeds.append(500 if moving else 0)
        self._speed_label.setText(f"{max(speeds) if speeds else 0} step/s")

    def _factor(self, axis: str) -> float:
        if self._stage_id == "zolix":
            return (float(self._settings.device("zolix").get("um_per_pulse_r", 0.00125))
                    if axis == "r"
                    else float(self._settings.device("zolix").get("um_per_pulse_xy", 0.625)))
        cfg = self._settings.device("sigmakoki")
        return (float(cfg.get("um_per_step_z", 0.25)) if axis == "z"
                else float(cfg.get("um_per_step_xy", 0.5)))


from talos.hal.devices.sigmakoki import SPEED_LEVEL_TO_HZ as SPEED_LEVEL_HZ  # noqa: E402
