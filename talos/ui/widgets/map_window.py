"""MapWindow: the scan map, larger, in its own window.

The map is the smallest panel in the Sample Finding tab and the one whose
whole point is seeing where a run went, so a double-click on it opens this
— a plain window whose only content is *the same widget*, re-parented.

Re-parenting rather than a second instance is deliberate. A copy would
mean two sets of tile thumbnails (up to 600 QImages), two view transforms,
and two chances for them to disagree about what the operator is looking
at — and the map is exactly the thing that must not disagree with itself.
One widget, one state, moved while it is wanted elsewhere.

The panel owns the swap: it hands the widget over with :meth:`adopt` and
takes it back with :meth:`release`, putting a placeholder with a way back
in the emptied slot. Closing the window returns the widget rather than
destroying it, and Esc behaves as it does in every other window here —
STOP ALL first, then hide — because a map that quietly swallowed the
global stop would be the one place the operator cannot panic in.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def _compact_button(text: str, tooltip: str) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("compact")
    button.setToolTip(tooltip)
    button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return button


class MapWindow(QDialog):
    """Hosts the shared ScanMapWidget while the operator wants it big."""

    #: The widget has been handed back and the window is hidden — the
    #: panel re-homes it (see :meth:`release`).
    sig_dismissed = Signal()

    def __init__(self, input_system=None, parent: QWidget | None = None):
        super().__init__(parent)
        self._input = input_system
        self._map: QWidget | None = None
        self.setWindowTitle("Scan map")
        self.setModal(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        self.fit_btn = _compact_button("Fit", "Reset the zoom and centre "
                                              "the plan")
        self.fit_btn.clicked.connect(self._fit)
        self.return_btn = _compact_button("Back to the panel",
                                          "Put the map back where it was")
        self.return_btn.clicked.connect(self.hide)
        row_layout.addWidget(self.fit_btn)
        row_layout.addStretch(1)
        row_layout.addWidget(QLabel("Wheel zooms · drag pans"))
        row_layout.addWidget(self.return_btn)
        layout.addWidget(row)

        self._slot = QVBoxLayout()
        layout.addLayout(self._slot, stretch=1)
        self.resize(980, 760)

    # --- the widget's tenancy ------------------------------------------

    def adopt(self, map_widget: QWidget) -> None:
        """Take the shared map widget (the panel must have released it)."""
        self._map = map_widget
        map_widget.setParent(self)
        self._slot.addWidget(map_widget)
        map_widget.setVisible(True)

    def release(self) -> QWidget | None:
        """Hand the widget back, detached from this window."""
        if self._map is None:
            return None
        widget = self._map
        self._map = None
        self._slot.removeWidget(widget)
        widget.setParent(None)
        return widget

    def _fit(self) -> None:
        if self._map is not None:
            self._map.fit()

    # --- the window contract -------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        """Closing returns the map rather than destroying it."""
        event.ignore()
        self.hide()

    def hideEvent(self, event) -> None:  # noqa: N802
        super().hideEvent(event)
        self.sig_dismissed.emit()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            # Esc is the global STOP ALL: it must do here exactly what it
            # does in every other window, then take the map away.
            if self._input is not None:
                self._input.on_escape()
            self.hide()
            event.accept()
            return
        super().keyPressEvent(event)


__all__ = ["MapWindow"]
