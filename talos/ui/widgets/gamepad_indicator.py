"""Gamepad status + D-pad stage indicator (reference gui/gamepad_indicator.py)."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget


class GamepadIndicator(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self._status = QLabel("No Controller")
        self._status.setObjectName("dim")
        layout.addWidget(self._status)
        self._dpad = QLabel("")
        self._dpad.setObjectName("ok")
        layout.addWidget(self._dpad)

    def set_connected(self, connected: bool, name: str = "Xbox Controller") -> None:
        if connected:
            self._status.setText(f"Controller: {name}")
            self._restyle(self._status, "ok")
        else:
            self._status.setText("No Controller")
            self._restyle(self._status, "dim")

    @staticmethod
    def _restyle(widget, object_name: str) -> None:
        """Changing objectName does NOT re-evaluate the stylesheet — Qt
        only repolishes when asked, so the label kept its old colour
        (green text stayed grey after a disconnect and vice versa)."""
        widget.setObjectName(object_name)
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)

    def set_dpad_stage(self, stage_id: str) -> None:
        name = "SigmaKoki XYZ" if stage_id == "sigmakoki" else "Zolix XYR"
        self._dpad.setText(f"D-pad → {name}")
