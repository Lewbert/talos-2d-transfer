"""Navigation & Control workspace: the big camera live view on the left,
a fixed-width right column (Quick Actions + scrollable Capture / Camera /
AF / Temperature settings), and the shared bottom hardware strip (owned
by the MainWindow). The stage jog panels live in the StageControlWindow
(Windows menu); the strip shows status + enable only.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from talos.ui.widgets.collapsible import CollapsibleGroup
from talos.ui.widgets.control_groups import (
    AFGroup,
    CameraGroup,
    CaptureGroup,
    QuickActionsGroup,
    TemperatureGroup,
)
from talos.ui.widgets.live_view import LiveViewWidget


class NavigationWorkspace(QWidget):
    """Live view (left, stretch) + the fixed-width settings column."""

    def __init__(self, manager, settings, input_system=None, parent: QWidget | None = None,
                 state=None, autofocus_service=None, autogain=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: the camera live view takes all the room it can get.
        self.live_view = LiveViewWidget()
        splitter.addWidget(self.live_view)

        # Right: quick actions (fixed) + scrollable settings groups.
        # The 4px gutters keep the group outlines off the splitter
        # handle (left) and the window edge (right).
        right_col = QWidget()
        right = QVBoxLayout(right_col)
        right.setContentsMargins(4, 0, 4, 0)
        right.setSpacing(6)

        self.quick_actions = QuickActionsGroup(
            manager, settings, state, autofocus_service)
        right.addWidget(self.quick_actions)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        # a right gutter inside the viewport so the group outlines never
        # stick to the vertical scrollbar when it appears
        scroll.setViewportMargins(0, 0, 6, 0)
        groups_col = QWidget()
        groups = QVBoxLayout(groups_col)
        groups.setContentsMargins(0, 0, 0, 8)  # gutters come from `right`
        groups.setSpacing(6)
        self.capture_group = CaptureGroup(settings)
        groups.addWidget(CollapsibleGroup(
            "Capture", self.capture_group, settings=settings,
            state_key="nav"))
        self.camera_group = CameraGroup(manager, settings, autogain)
        groups.addWidget(CollapsibleGroup(
            "Camera", self.camera_group, settings=settings, state_key="nav"))
        self.af_group = AFGroup(settings, state)
        groups.addWidget(CollapsibleGroup(
            "Autofocus", self.af_group, settings=settings, state_key="nav"))
        self.temp_group = TemperatureGroup(manager, settings)
        groups.addWidget(CollapsibleGroup(
            "Temperature", self.temp_group, settings=settings,
            state_key="nav"))
        groups.addStretch(1)
        scroll.setWidget(groups_col)
        right.addWidget(scroll, stretch=1)

        right_col.setMinimumWidth(300)
        right_col.setMaximumWidth(420)
        splitter.addWidget(right_col)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1080, 320])
        layout.addWidget(splitter)

    def update_telem(self, device_key: str, payload: dict) -> None:
        if device_key == "yudian":
            self.temp_group.update_telem(payload)
