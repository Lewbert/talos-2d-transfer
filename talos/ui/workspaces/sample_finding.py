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

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QLabel,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from talos.ui.widgets.collapsible import CollapsibleGroup
from talos.ui.widgets.control_groups import CameraGroup
from talos.ui.widgets.live_view import LiveViewModeBar, LiveViewWidget


class SampleFindingWorkspace(QWidget):
    #: The operator asked for the processed overlay (or turned it off): the
    #: Scan window listens, because it is the one that computes it.
    sig_processed_view = Signal(bool)

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

        # Left: the live view (frames arrive via MainWindow), with the
        # Live/Processed switch floating over it. ONE live view for the
        # whole application: the scan console drives this one rather than
        # showing a second copy of the same stream.
        view = QWidget()
        view_layout = QVBoxLayout(view)
        view_layout.setContentsMargins(0, 0, 0, 0)
        self.live_view = LiveViewWidget()
        view_layout.addWidget(self.live_view)
        self.mode_bar = LiveViewModeBar(self.live_view)
        self.mode_bar.live_btn.toggled.connect(
            lambda on: on and self._set_view_mode("live"))
        self.mode_bar.processed_btn.toggled.connect(
            lambda on: on and self._set_view_mode("processed"))
        self.live_view.installEventFilter(self)
        splitter.addWidget(view)

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
        self.mode_bar.place(self.live_view)

    # ------------------------------------------------------------------

    def on_frame(self, frame) -> None:
        """Keep the newest frame so other parts of the app can read the
        pixel data the operator is looking at."""
        self._last_frame = frame

    def set_processed_frame(self, frame) -> None:
        """The identification overlay, handed over by the Scan window."""
        self.live_view.set_processed_frame(frame)

    def _set_view_mode(self, mode: str) -> None:
        self.live_view.set_view_mode(mode)
        self.sig_processed_view.emit(mode == "processed")

    def eventFilter(self, obj, event):  # noqa: N802
        if obj is self.live_view and event.type() == event.Type.Resize:
            self.mode_bar.place(self.live_view)
        return super().eventFilter(obj, event)

    def update_telem(self, key: str, payload: dict) -> None:
        """Telemetry is not this workspace's business any more; the Scan
        window has its own handler."""
