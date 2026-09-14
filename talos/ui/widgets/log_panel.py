"""Ring-buffer log view with a level filter.

Throttled rendering: lines are appended to the deque immediately, but the
HTML view rebuilds at most ~4×/second — high-frequency logging (input
commands, telemetry) must never saturate the GUI thread.
"""

from __future__ import annotations

from collections import deque

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

_LEVEL_COLORS = {
    "debug": "#8b90a0",
    "info": "#d8dbe2",
    "warning": "#d29922",
    "error": "#f85149",
    "critical": "#ff6b63",
}


class LogPanel(QWidget):
    """Keeps the last ``max_lines`` log entries; filterable by level."""

    def __init__(self, max_lines: int = 500, parent: QWidget | None = None):
        super().__init__(parent)
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._levels: deque[str] = deque(maxlen=max_lines)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        top.addWidget(QLabel("Log"))
        self._filter = QComboBox()
        self._filter.addItems(["all", "debug", "info", "warning", "error"])
        self._filter.setFixedWidth(110)
        self._filter.currentTextChanged.connect(self._refresh)
        top.addWidget(self._filter)
        top.addStretch(1)
        layout.addLayout(top)

        self._view = QTextEdit()
        self._view.setReadOnly(True)
        self._view.setObjectName("log")
        layout.addWidget(self._view)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)  # append() restarts it —
        # no rebuilds while idle (the repeating form redrew ~4×/s forever)
        self._refresh_timer.setInterval(250)  # max 4 rebuilds/second
        self._refresh_timer.timeout.connect(self._refresh)

    def append(self, level: str, message: str) -> None:
        self._lines.append(message)
        self._levels.append(level.lower())
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def _refresh(self) -> None:
        selected = self._filter.currentText()
        blocks = []
        for level, message in zip(self._levels, self._lines):
            if selected != "all" and level != selected:
                continue
            color = _LEVEL_COLORS.get(level, "#d8dbe2")
            blocks.append(f'<span style="color:{color}">[{level[:4]}]</span> {message}')
        self._view.setHtml("<br>".join(blocks))
        scrollbar = self._view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
