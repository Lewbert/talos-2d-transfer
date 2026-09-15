"""Focus LT/RT trigger display: two mirrored bars with deadzone notches
plus the computed signed jog speed (same math the ActionResolver uses —
`focus_trigger_to_speed`). RT = focus up (right bar), LT = down."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFontMetrics, QPainter
from PySide6.QtWidgets import QWidget

from talos.input.action_resolver import focus_trigger_to_speed
from talos.ui import theme
from talos.ui.theme import LED_OFF, TEXT_DIM, WARN


class TriggerBarWidget(QWidget):
    def __init__(self, settings, parent: QWidget | None = None):
        super().__init__(parent)
        self._settings = settings
        self._load_settings()
        self._lt = 0.0
        self._rt = 0.0
        self._connected = False
        self.setMinimumWidth(130)
        # NOT a fixed height: the strip's FOCUS row is ~29 px tall, and a
        # 40 px bar was clipped there — its bottom, i.e. the LT/RT labels
        # and the jog-speed readout, never appeared. The bar lays itself
        # out from the height it is actually given.
        self.setMinimumHeight(26)
        self.setToolTip("Focus jog: gamepad LT (down) / RT (up) triggers")

    def _load_settings(self) -> None:
        cfg = self._settings.device("focus")
        self._min_speed = float(cfg.get("min_speed", 50))
        self._max_speed = float(cfg.get("max_speed", 2000))
        self._gamma = float(cfg.get("gamma", 2.2))
        self._deadzone = float(cfg.get("deadzone", 0.05))
        # The readout must follow the SAME inversion the dispatcher applies
        # (talos.input.axis_map) — the bar used to be the only place the
        # trigger direction was inverted.
        self._invert = bool(cfg.get("invert", False))

    def reload_settings(self) -> None:
        self._load_settings()
        self.update()

    def set_state(self, lt: float, rt: float) -> None:
        """Raw trigger values 0..1."""
        self._lt = max(0.0, min(float(lt), 1.0))
        self._rt = max(0.0, min(float(rt), 1.0))
        self.update()

    def set_connected(self, connected: bool) -> None:
        self._connected = connected
        if not connected:
            self._lt = self._rt = 0.0
        self.update()

    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        fm = QFontMetrics(self.font())
        # The LT/RT + speed line owns the bottom of whatever height we get;
        # the bar takes the space above it (no fixed y — see __init__).
        text_y = h - 2
        bar_h = 7
        bar_y = max(2, (text_y - fm.height() - bar_h) // 2)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(LED_OFF))
        painter.drawRoundedRect(0, bar_y, w, bar_h, 3, 3)

        # Mirrored fills: LT grows from the left, RT from the right.
        # module read (not a value import): live accent changes reach it
        if self._connected:
            painter.setBrush(QColor(WARN))
            painter.drawRoundedRect(0, bar_y, int(w * self._lt), bar_h, 3, 3)
            painter.setBrush(QColor(theme.ACCENT))
            rt_w = int(w * self._rt)
            painter.drawRoundedRect(w - rt_w, bar_y, rt_w, bar_h, 3, 3)

        # Deadzone notches (net deadzone fraction from each end).
        notch = self._deadzone / (1.0 + self._deadzone)
        painter.setBrush(QColor("#000000"))
        for x in (int(w * notch), int(w * (1 - notch))):
            painter.drawRect(x - 1, bar_y - 2, 2, bar_h + 4)

        # Labels + speed.
        painter.setPen(QColor(TEXT_DIM))
        painter.drawText(0, text_y, "LT")
        painter.drawText(w - fm.horizontalAdvance("RT"), text_y, "RT")
        speed = focus_trigger_to_speed(
            self._lt, self._rt, min_speed=self._min_speed,
            max_speed=self._max_speed, gamma=self._gamma,
            deadzone=self._deadzone, invert=self._invert)
        # The jog speed, drawn only while a trigger is actually driving the
        # focus. At rest the middle used to read "idle", which just repeated
        # the IDLE/CONT word sitting next to the bar.
        if speed:
            text = f"→ {speed:+d} sps"
            painter.setPen(QColor(theme.ACCENT))
            painter.drawText(w // 2 - fm.horizontalAdvance(text) // 2,
                             text_y, text)
        painter.end()
