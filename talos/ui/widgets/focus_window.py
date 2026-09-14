"""FocusWindow: the autofocus panel as a standalone, toolbar-toggled
window. Esc/close HIDE the window (and uncheck the toolbar action) —
Esc additionally issues the global STOP ALL first, like it does
everywhere else in the app."""
from __future__ import annotations

from PySide6.QtWidgets import QDialog, QVBoxLayout

from talos.ui.widgets.autofocus_panel import AutofocusPanel


class FocusWindow(QDialog):
    def __init__(self, manager, settings, state, service, live_view,
                 input_system=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Focus")
        self._toggle_action = None
        self._input = input_system
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        # The panel keeps its ROI-arming link to the navigation live view.
        self.panel = AutofocusPanel(manager, settings, state, service,
                                    live_view, parent=self)
        layout.addWidget(self.panel)
        self.resize(560, 520)

    def set_toggle_action(self, action) -> None:
        self._toggle_action = action

    def set_live_view(self, live_view) -> None:
        """Retarget the panel's ROI rubber band at the ACTIVE workspace's
        live view (the panel kept the navigation one, so selecting an ROI
        while Sample Finding was shown armed a hidden widget)."""
        self.panel.set_live_view(live_view)

    def _hide_and_uncheck(self) -> None:
        self.hide()
        if self._toggle_action is not None:
            self._toggle_action.setChecked(False)

    def reject(self) -> None:  # Esc
        # The window hides, but Esc stays the global STOP ALL: the
        # MainWindow's Esc shortcut is WindowShortcut-scoped and cannot
        # fire while this window has focus.
        if self._input is not None:
            self._input.on_escape()
        self._hide_and_uncheck()

    def closeEvent(self, event) -> None:  # noqa: N802
        event.ignore()  # never destroyed — toggled back via the toolbar
        self._hide_and_uncheck()
