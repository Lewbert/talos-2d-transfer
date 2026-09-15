"""Small round status LED with label."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from talos.ui import theme
from talos.ui.theme import DANGER, LED_OFF, OK

OFF, CONNECTING, ON, ERROR = "off", "connecting", "on", "error"

_COLORS = {
    OFF: LED_OFF,
    ON: OK,
    ERROR: DANGER,
}


def _state_color(state: str) -> str:
    # module read (not a value import): live accent changes must reach it
    return theme.ACCENT if state == CONNECTING else _COLORS.get(state, LED_OFF)


class StatusLED(QWidget):
    """A label + colored dot. State in {off, connecting, on, error}."""

    def __init__(self, text: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._state = OFF
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(6)
        self._dot = _Dot(self)
        self._label = QLabel(text, self)
        self._label.setObjectName("dim")
        layout.addWidget(self._dot)
        layout.addWidget(self._label)

    @property
    def text(self) -> str:
        return self._label.text()

    def set_state(self, state: str) -> None:
        if state != self._state:
            self._state = state
            self._dot.update()

    def state(self) -> str:
        return self._state


class _Dot(QWidget):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setFixedSize(10, 10)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        color = QColor(_state_color(self.parentWidget().state()))
        painter.setBrush(color)
        painter.drawEllipse(0, 0, 10, 10)
