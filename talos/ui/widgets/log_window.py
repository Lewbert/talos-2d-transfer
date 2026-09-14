"""LogWindow: the full message log as a Windows-menu-toggled window
(hidden by default — brief messages live in the status bar). Esc/close
HIDE the window and uncheck the menu action (FocusWindow pattern)."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QVBoxLayout

from talos.ui.widgets.log_panel import LogPanel


class LogWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Log")
        self._toggle_action = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.panel = LogPanel(parent=self)
        layout.addWidget(self.panel)
        self.resize(960, 420)

    def set_toggle_action(self, action) -> None:
        self._toggle_action = action

    def _hide_and_uncheck(self) -> None:
        self.hide()
        if self._toggle_action is not None:
            self._toggle_action.setChecked(False)

    def reject(self) -> None:  # Esc
        self._hide_and_uncheck()

    def closeEvent(self, event) -> None:  # noqa: N802
        event.ignore()  # never destroyed — toggled back via the menu
        self._hide_and_uncheck()
