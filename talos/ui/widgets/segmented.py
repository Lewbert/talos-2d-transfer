"""SegmentedToggle: one row of exclusive buttons for a small choice.

The scan panel carries several settings that are really a choice between
two or three named things — which axis first, which way the area grows,
which path order. A QComboBox costs a click, a popup and a look away from
the image; a row of buttons shows every option and takes one click. That
difference is why these are buttons and the genuinely-numeric settings
are spin boxes.

Deliberately not a QComboBox subclass and deliberately not styled inline:
it carries an objectName and the QSS in ``ui/theme.py`` draws it, like
every other themed widget here.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QWidget,
)


class SegmentedToggle(QWidget):
    """A compact exclusive choice: ``[(value, label, tooltip), ...]``.

    The signal carries the VALUE, never an index — a settings file holds
    values, and an index is a thing that quietly means something else
    after the list is reordered.
    """

    sig_changed = Signal(object)

    def __init__(self, options, value=None, parent: QWidget | None = None,
                 stretch: bool = True):
        super().__init__(parent)
        self.setObjectName("segmented")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(0)
        self._buttons: dict[object, QPushButton] = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for option in options:
            item = _as_option(option)
            button = QPushButton(item[1])
            button.setObjectName("segmented_btn")
            button.setCheckable(True)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            if len(item) > 2 and item[2]:
                button.setToolTip(item[2])
            if stretch:
                button.setSizePolicy(QSizePolicy.Policy.Expanding,
                                     QSizePolicy.Policy.Fixed)
            self._group.addButton(button)
            layout.addWidget(button)
            self._buttons[item[0]] = button
            button.clicked.connect(
                lambda _checked, value=item[0]: self._on_clicked(value))
        self.set_value(value if value is not None else options[0])

    # ------------------------------------------------------------------

    def value(self):
        for candidate, button in self._buttons.items():
            if button.isChecked():
                return candidate
        return None

    def set_value(self, value) -> None:
        """Select ``value``; an unknown one selects the first option rather
        than leaving the row blank (a hand-edited settings file again)."""
        button = self._buttons.get(value)
        if button is None:
            button = next(iter(self._buttons.values()), None)
        if button is not None and not button.isChecked():
            button.setChecked(True)

    def set_options(self, options) -> None:
        """Replace the choices, keeping the current value if it survives.

        Rebuilding in place is what lets a control whose contents depend on
        another setting (the path order, the origin modes) hold its value
        across the change instead of resetting to the first entry.
        """
        current = self.value()
        for button in list(self._buttons.values()):
            self._group.removeButton(button)
            button.setParent(None)
            button.deleteLater()
        self._buttons.clear()
        layout = self.layout()
        for option in options:
            item = _as_option(option)
            button = QPushButton(item[1])
            button.setObjectName("segmented_btn")
            button.setCheckable(True)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setSizePolicy(QSizePolicy.Policy.Expanding,
                                 QSizePolicy.Policy.Fixed)
            if len(item) > 2 and item[2]:
                button.setToolTip(item[2])
            self._group.addButton(button)
            layout.addWidget(button)
            self._buttons[item[0]] = button
            button.clicked.connect(
                lambda _checked, value=item[0]: self._on_clicked(value))
        self.set_value(current if current in self._buttons else None)

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802
        super().setEnabled(enabled)
        for button in self._buttons.values():
            button.setEnabled(enabled)

    def _on_clicked(self, value) -> None:
        self.sig_changed.emit(value)


def _as_option(option) -> tuple:
    """Accept ``value``, ``(value, label)`` or ``(value, label, tooltip)``."""
    if isinstance(option, (tuple, list)):
        return tuple(option)
    return (option, str(option))


__all__ = ["SegmentedToggle"]
