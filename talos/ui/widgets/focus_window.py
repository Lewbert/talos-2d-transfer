"""FocusWindow: the autofocus panel plus the FULL autofocus settings block
in a standalone, toolbar-toggled window, so every AF knob is reachable
while watching the live view instead of having to switch to the right
panel. Esc/close HIDE the window (and uncheck the toolbar action) — Esc
additionally issues the global STOP ALL first, like everywhere else in
the app."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialog, QGroupBox, QVBoxLayout, QWidget

from talos.ui.af_region import AfRegionController
from talos.ui.widgets.autofocus_panel import AutofocusPanel
from talos.ui.widgets.control_groups import AfSettingsWidget


class FocusWindow(QDialog):
    sig_settings_changed = Signal()

    def __init__(self, manager, settings, state, service,
                 input_system=None, af_roi: AfRegionController | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Focus")
        self._toggle_action = None
        self._input = input_system
        self._settings = settings
        self._roi = af_roi or AfRegionController(settings)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Order matches the workflow: settings (measure area, window,
        # speeds) on top, then the run controls, progress and curve.
        settings_box = QGroupBox("AF settings")
        box_layout = QVBoxLayout(settings_box)
        box_layout.setContentsMargins(6, 6, 6, 6)
        self.settings_widget = AfSettingsWidget(settings, state, self._roi)
        self.settings_widget.sig_settings_changed.connect(
            self.sig_settings_changed.emit)
        box_layout.addWidget(self.settings_widget)
        layout.addWidget(settings_box)

        self.panel = AutofocusPanel(manager, settings, state, service,
                                    self._roi, parent=self)
        layout.addWidget(self.panel, stretch=1)
        self.resize(620, 760)

    # ------------------------------------------------------------------

    def set_toggle_action(self, action) -> None:
        self._toggle_action = action

    def refresh_settings(self) -> None:
        """Re-read the knobs (the right-panel instance may have edited them)."""
        self.settings_widget.refresh_from_settings()

    def showEvent(self, event) -> None:  # noqa: N802
        # Never show stale numbers: the right panel can have changed while
        # this window was hidden.
        super().showEvent(event)
        self.refresh_settings()

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
