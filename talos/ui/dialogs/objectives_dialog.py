"""ObjectivesDialog: the standalone wrapper around ObjectivesPage (the
same body lives in the Preferences dialog's Objectives & Calibration
page)."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout

from talos.ui.dialogs.objectives_page import ObjectivesPage


class ObjectivesDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self.setWindowTitle("Objectives & Calibration")
        self.resize(1100, 460)
        layout = QVBoxLayout(self)
        self.page = ObjectivesPage(settings, self)
        layout.addWidget(self.page)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_save(self) -> None:
        self.page.save()
        self.accept()
