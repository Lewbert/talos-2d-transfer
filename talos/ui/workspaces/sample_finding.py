"""Sample Finding workspace: the live view, and a pointer to the Scan
window.

The scan and the identification chain used to live here. They moved to
`Windows ▸ Scan` while they are being proven (the same call the autofocus
panel got with "AF Detail"), for two reasons: the new functions needed
room — a scan map, a filter chain, live processed output — that the
workspace's right column cannot give them, and a tab redesign is the LAST
thing to do to a function that has never run on hardware.

What stays is what this tab is genuinely good at: the largest live view in
the app and the scan camera profile (manual exposure and white balance,
because identification needs a stable image, not an auto-adjusted one).
The plan is to fold the Scan window's panel back in here once it has bench
miles on it.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from talos.ui.widgets.collapsible import CollapsibleGroup
from talos.ui.widgets.control_groups import CameraGroup
from talos.ui.widgets.live_view import LiveViewWidget


class SampleFindingWorkspace(QWidget):
    def __init__(self, manager, settings, state, parent: QWidget | None = None,
                 autofocus_service=None, autogain=None,
                 calibration_context=None, input_system=None):
        super().__init__(parent)
        self._manager = manager
        self._settings = settings
        self._state = state
        self._calibration = calibration_context
        self._last_frame = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: the live view (frames arrive via MainWindow).
        self.live_view = LiveViewWidget()
        splitter.addWidget(self.live_view)

        right_col = QWidget()
        right = QVBoxLayout(right_col)
        right.setContentsMargins(4, 0, 4, 0)
        right.setSpacing(6)

        # The manual-only profile: identification needs a stable image, so
        # auto gain and auto white balance must not chase the scene.
        self.camera_group = CameraGroup(manager, settings, autogain,
                                        profile_kind="scan")
        right.addWidget(CollapsibleGroup("Camera", self.camera_group,
                                         settings=settings, state_key="scan"))

        pointer = QLabel(
            "Scanning and sample identification now run in their own "
            "window — see <b>Windows ▸ Scan</b>. This tab is kept as the "
            "large live view until that panel has bench miles on it.")
        pointer.setWordWrap(True)
        pointer.setObjectName("dim")
        right.addWidget(pointer)
        right.addStretch(1)

        right_col.setMinimumWidth(300)
        right_col.setMaximumWidth(420)
        splitter.addWidget(right_col)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1080, 320])
        layout.addWidget(splitter)

    # ------------------------------------------------------------------

    def on_frame(self, frame) -> None:
        """Keep the newest frame so other parts of the app can read the
        pixel data the operator is looking at."""
        self._last_frame = frame

    def update_telem(self, key: str, payload: dict) -> None:
        """Telemetry is not this workspace's business any more; the Scan
        window has its own handler."""
