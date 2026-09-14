"""CollapsibleGroup: an accordion section for the right-panel settings.

The wrapper IS the rounded-border container (styled via the
``#collapsible`` QSS rule): the header button (▾/▸ + title) sits INSIDE
the box, above the content. The wrapped QGroupBox's own chrome is
neutralized (no title, no border — the wrapper draws the frame). The
header never takes keyboard focus (the window owns the jog keys).

The collapse state survives restarts: pass ``settings`` + ``state_key``
(the workspace name) and the collapsed titles persist under
``ui.collapsed_sections[state_key]``.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class CollapsibleGroup(QWidget):
    def __init__(self, title: str, child: QWidget, parent=None,
                 collapsed: bool = False, settings=None,
                 state_key: str | None = None):
        super().__init__(parent)
        self._child = child
        self._title = title
        self._settings = settings
        self._state_key = state_key
        if settings is not None and state_key \
                and title in self._stored_collapsed():
            collapsed = True
        self.setObjectName("collapsible")
        # plain QWidgets need the styled-background attribute for the
        # QSS border/background to paint
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        if isinstance(child, QGroupBox):
            child.setTitle("")
            # the wrapper draws the frame — the inner box becomes a
            # transparent content container
            child.setStyleSheet(
                "QGroupBox { border: none; margin-top: 0; "
                "padding: 4px 2px; background: transparent; }")
        self._header = QPushButton(
            f"{'▸' if collapsed else '▾'} {title}")
        self._header.setObjectName("collapsible_header")
        self._header.setCheckable(True)
        self._header.setChecked(not collapsed)
        self._header.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._header.toggled.connect(self._on_toggled)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 4)
        layout.setSpacing(0)
        layout.addWidget(self._header)
        layout.addWidget(child)
        self._apply(not collapsed)

    def _stored_collapsed(self) -> list:
        section = self._settings.section("ui").get("collapsed_sections") or {}
        return list(section.get(self._state_key) or [])

    def _persist_collapsed(self, collapsed: bool) -> None:
        if self._settings is None or not self._state_key:
            return
        ui = self._settings.section("ui")
        data = dict(ui.get("collapsed_sections") or {})
        titles = [t for t in data.get(self._state_key, [])
                  if t != self._title]
        if collapsed:
            titles.append(self._title)
        data[self._state_key] = titles
        ui["collapsed_sections"] = data
        self._settings.save()

    def _on_toggled(self, on: bool) -> None:
        self._apply(bool(on))
        self._persist_collapsed(not on)  # user action only — the init
        # apply must not write the file at every startup

    def _apply(self, expanded: bool) -> None:
        text = self._header.text()[2:]  # strip the arrow prefix
        self._header.setText(f"{'▾' if expanded else '▸'} {text}")
        self._child.setVisible(expanded)

    @property
    def expanded(self) -> bool:
        return self._header.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self._header.setChecked(bool(expanded))
