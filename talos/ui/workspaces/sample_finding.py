"""Sample Finding: the camera, the filters, and the scan, in one tab.

Three columns, and the split is by *what you are doing*, not by which
subsystem owns the code:

- **left** — the camera and the computer vision. The sample colour is
  pinned at the top because it is the one control that is touched
  constantly (it is both the mask's target and the curve's centre); below
  it, in a scroll, the camera's manual profile, the pre-processing chain
  and the identification gates.
- **centre** — the live view, as large as the other two columns leave it,
  with the Original / Pre-processed / Samples switch floating over it.
- **right** — the scan: the map, the run card, the settings worth changing
  at the microscope, and the samples that were found.

This widget is also the **hub**. It owns the detection engine, because the
engine serves two consumers with one queue: the live preview (whose input
is the pre-processing chain and the colour, both edited in the left column)
and the scan's tiles (captured in the right one). It builds both configs
per job, so the worker never reads a widget and never sees a half-edited
configuration; it routes frames to the live view and tiles to the panel;
and it sends results back as markers and list rows.

Two rules it inherits and keeps:

- **Nothing blocks the capture.** Pre-processing and identification run on
  the detection worker. The two processed views are therefore that
  worker's output and lag the stream slightly — the stream itself, the
  autofocus and a running scan never wait on either.
- **The frames identification sees are the frames the operator sees** —
  pre-processed, never composited: no overlay, no scale bar, nothing that
  draws. See ``cv/identify.py`` for why that rule is load-bearing.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from talos.cv.identify import IdentifyConfig, sample_hex
from talos.ui.detect_engine import DetectionEngine
from talos.ui.widgets.collapsible import CollapsibleGroup
from talos.ui.widgets.control_groups import CameraGroup
from talos.ui.widgets.hardware_strip import parse_focus, parse_zolix
from talos.ui.widgets.identify_panel import (ColourGroup, IdentifyGroup,
                                             PreprocessGroup)
from talos.ui.widgets.live_view import LiveViewModeBar, LiveViewWidget
from talos.ui.widgets.scan_panel import ScanPanel

#: The live preview runs on a downscaled copy of the frame. Half is what
#: the console used and the operator never moved it; the full-resolution
#: path is the scan's, where a tile takes a second to capture anyway and
#: nothing is dropped.
PREVIEW_SCALE = 0.5

#: How long the processed views stay held back after the last sign of
#: motion. The telemetry says an axis has stopped before the frame that
#: still shows it settling has been delivered, and a preview that flashes
#: a blurry processed frame on every jog release is worse than one that
#: waits a moment.
MOTION_HOLD_MS = 300


class SampleFindingWorkspace(QWidget):
    #: A line for the log (a scan's folder, an export, a failed run). The
    #: main window owns the log panel; this tab just reports.
    sig_log = Signal(str)

    def __init__(self, manager, settings, state, parent: QWidget | None = None,
                 autofocus_service=None, autogain=None,
                 calibration_context=None, input_system=None):
        super().__init__(parent)
        self._manager = manager
        self._settings = settings
        self._state = state
        self._calibration = calibration_context
        self._input = input_system
        self._autofocus = autofocus_service
        self._last_frame = None
        #: The width of the frame the operator tunes on (see on_frame).
        self._reference_width = 0
        #: True while an axis the camera can see is moving.
        self._stage_moving = False
        self._motion_hold = QTimer(self)
        self._motion_hold.setSingleShot(True)
        self._motion_hold.setInterval(MOTION_HOLD_MS)
        self._motion_hold.timeout.connect(self._on_motion_hold_expired)

        self._engine = DetectionEngine(parent=self)
        self._engine.sig_result.connect(self._on_detected)
        self._engine.sig_log.connect(self._log)
        self._engine.set_source(self._detect_source)

        self._build_ui(manager, settings, autogain)
        self.scan_panel.reload_preferences()
        self._sync_curve_centre()
        self.set_view_mode(self._stored_view_mode())
        self._engine.set_live(True)
        # A scan or an autofocus run holds the processed views back for its
        # whole duration (see _refresh_processed_views).
        self._state.sig_mode_changed.connect(self._on_mode_changed)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_ui(self, manager, settings, autogain) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left(manager, settings, autogain))
        splitter.addWidget(self._build_centre())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([330, 820, 420])
        layout.addWidget(splitter)

    def _build_left(self, manager, settings, autogain) -> QWidget:
        panel = QWidget()
        column = QVBoxLayout(panel)
        column.setContentsMargins(4, 0, 4, 0)
        column.setSpacing(6)

        # Quick access, pinned: the one colour that drives two things.
        self.colour_group = ColourGroup(settings)
        self.colour_group.sig_changed.connect(self._on_colour_changed)
        self.colour_group.sig_dropper.connect(self.arm_colour_pick)
        column.addWidget(self.colour_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setViewportMargins(0, 0, 6, 0)
        inner = QWidget()
        groups = QVBoxLayout(inner)
        groups.setContentsMargins(0, 0, 0, 8)
        groups.setSpacing(6)

        # The manual-only profile: identification needs a stable image, so
        # auto gain and auto white balance must not chase the scene.
        self.camera_group = CameraGroup(manager, settings, autogain,
                                        profile_kind="scan")
        groups.addWidget(CollapsibleGroup("Camera", self.camera_group,
                                          settings=settings,
                                          state_key="scan"))
        self.preprocess_group = PreprocessGroup(settings)
        self.preprocess_group.sig_changed.connect(self._on_preprocess_changed)
        groups.addWidget(CollapsibleGroup("Pre-processing",
                                          self.preprocess_group,
                                          settings=settings,
                                          state_key="scan"))
        self.identify_group = IdentifyGroup(settings)
        self.identify_group.sig_changed.connect(self._on_identify_changed)
        groups.addWidget(CollapsibleGroup("Identification",
                                          self.identify_group,
                                          settings=settings,
                                          state_key="scan"))
        groups.addStretch(1)
        scroll.setWidget(inner)
        column.addWidget(scroll, 1)

        panel.setMinimumWidth(300)
        panel.setMaximumWidth(440)
        return panel

    def _build_centre(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.live_view = LiveViewWidget()
        layout.addWidget(self.live_view)
        self.mode_bar = LiveViewModeBar(self.live_view)
        for mode, button in self.mode_bar.buttons.items():
            button.toggled.connect(
                lambda on, mode=mode: on and self.set_view_mode(mode))
        self.mode_bar.place(self.live_view)
        self.live_view.installEventFilter(self)
        # The dropper is armed here and lands here: one click on the image,
        # sampled from the layer this tab computed.
        self.live_view.sig_frame_clicked.connect(self.on_pick)
        return panel

    def _build_right(self) -> QWidget:
        panel = QWidget()
        column = QVBoxLayout(panel)
        column.setContentsMargins(4, 0, 4, 0)
        column.setSpacing(6)
        self.scan_panel = ScanPanel(self._manager, self._settings, self._state,
                                    calibration_context=self._calibration,
                                    input_system=self._input)
        self.scan_panel.sig_tile_captured.connect(self._on_tile_captured)
        self.scan_panel.sig_log.connect(self._log)
        self.scan_panel.sig_plan_changed.connect(self.refresh_settings)
        self.scan_panel.pending_tiles_fn = lambda: self._engine.pending_tiles
        self.scan_panel.set_flip(self.camera_flip())
        column.addWidget(self.scan_panel, 1)
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(480)
        return panel

    # ------------------------------------------------------------------
    # the view mode
    # ------------------------------------------------------------------

    def _stored_view_mode(self) -> str:
        return str(self._settings.section("ui").get("sample_view_mode")
                   or "original")

    def set_view_mode(self, mode: str) -> None:
        """Original / Pre-processed / Samples. Display only — but the last
        two are computed by the worker, so switching to them is also a
        request for the preview feed to keep running."""
        self.live_view.set_view_mode(mode)
        button = self.mode_bar.buttons.get(self.live_view.view_mode)
        if button is not None and not button.isChecked():
            button.setChecked(True)
        self._settings.section("ui")["sample_view_mode"] = \
            self.live_view.view_mode
        self._settings.save()
        self._engine.set_live(True)
        self._refresh_processed_views()

    def eventFilter(self, obj, event):  # noqa: N802
        if obj is self.live_view and event.type() == event.Type.Resize:
            self.mode_bar.place(self.live_view)
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # the dropper
    # ------------------------------------------------------------------

    def arm_colour_pick(self) -> None:
        """The next click on the live view samples a colour."""
        self.live_view.set_pick_mode(True)

    def on_pick(self, x_px: int, y_px: int) -> None:
        """A click on the live view, in frame pixels.

        The pixel is read from the PRE-PROCESSED layer — the array the
        identification ran on — so the colour the operator picks is a
        colour the mask will look for. In samples mode the screen is
        darkened and outlined, and sampling that would return a colour the
        sample does not have.
        """
        frame = self.live_view.pick_frame()
        if frame is None:
            return
        colour = sample_hex(frame, int(x_px), int(y_px))
        if colour:
            self.colour_group.set_hex(colour)

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------

    def identify_config(self) -> IdentifyConfig:
        """A FRESH config per job: the colour stage from the quick-access
        group, the gates from the chain. The worker never reads a widget."""
        return IdentifyConfig(
            stages=[self.colour_group.stage()] + self.identify_group.stages())

    def preprocess_config(self):
        return self.preprocess_group.config()

    def colour_rgb(self):
        return self.colour_group.rgb()

    def camera_flip(self) -> bool:
        return bool(self._settings.device("camera").get("flip", True))

    def calibration_for(self, frame):
        """The active calibration in THIS frame's pixels.

        The stored value is µm per 4K-SENSOR pixel (``cv/calibration.py``),
        so a frame sampled at another resolution needs the ratio: the same
        field of view over half the pixels means twice the µm per pixel.
        The width ratio is applied to both axes — right for the 16:9 frames
        this bench produces, and the same assumption the scale bar and the
        snapshot burn already make.

        It MUST be the frame's own width, not the live stream's. A scan can
        capture at a resolution the live view is not using
        (Preferences → Scan), and measuring those tiles with the live
        frame's scale gets every µm² wrong by the square of the ratio and
        every go-to-sample offset wrong in proportion.
        """
        from dataclasses import replace

        from talos.cv.calibration import SENSOR_WIDTH_PX

        calib = (self._calibration.calibration() if self._calibration
                 else self.scan_panel.canonical_calibration())
        if frame is None or not frame.shape[1]:
            return calib
        factor = SENSOR_WIDTH_PX / float(frame.shape[1])
        return replace(calib,
                       um_per_px_x=(calib.um_per_px_x or 0.0) * factor,
                       um_per_px_y=(calib.um_per_px_y or 0.0) * factor)

    def live_calibration(self):
        """The active calibration in LIVE-frame pixels."""
        return self.calibration_for(self._last_frame)

    def _sync_curve_centre(self) -> None:
        self.preprocess_group.set_centre(self.colour_group.rgb())

    def _on_colour_changed(self) -> None:
        self._sync_curve_centre()
        self._persist_identify()
        self._engine.set_live(True)
        self._refresh_processed_views()

    def _on_preprocess_changed(self) -> None:
        self._persist_preprocess()
        self._engine.set_live(True)
        self._refresh_processed_views()

    def _on_identify_changed(self) -> None:
        self._persist_identify()
        self._engine.set_live(True)
        self._refresh_processed_views()

    def _persist_identify(self) -> None:
        self._settings.update("identify", self.identify_config().to_dict())
        self._settings.save()

    def _persist_preprocess(self) -> None:
        self._settings.update("preprocess", self.preprocess_config().to_dict())
        self._settings.save()

    # ------------------------------------------------------------------
    # the frames
    # ------------------------------------------------------------------

    def on_frame(self, frame) -> None:
        """The newest streamed frame: the detection source hands it on and
        the picker falls back to it before the worker has run once."""
        self._last_frame = frame
        # The frame the operator is tuning the pixel-unit parameters ON —
        # remembered from before a scan, because a run can switch the live
        # stream to the scan's resolution (Preferences → Scan) and the
        # tiles that arrive then are not what the parameters were judged
        # against. See ``frame_scale_for``.
        if self._state.mode != "SCAN":
            self._reference_width = int(frame.shape[1]) if frame is not None \
                and len(frame.shape) > 1 else 0

    def frame_scale_for(self, frame) -> float:
        """How finely THIS frame samples the field of view, relative to the
        frame the parameters were tuned on.

        Every pixel-unit parameter (the frame-edge margin, the merge gap,
        the sharpness threshold) means pixels of what the operator was
        looking at when they set it. A scan that captures the same field of
        view at twice the resolution needs those numbers doubled (and the
        gradient threshold halved) or the gates silently change meaning —
        which is what "does this work at both scan resolutions?" turns on.
        """
        reference = int(self._reference_width or 0)
        if reference <= 0 or frame is None or len(frame.shape) < 2:
            return 1.0
        return float(frame.shape[1]) / float(reference)

    def _detect_source(self):
        """What the live detection feed works from, one tick at a time."""
        frame = self._last_frame
        if frame is None:
            return None
        return (frame, self.live_calibration(),
                self.scan_panel.stage_position(), self.identify_config(),
                PREVIEW_SCALE, self.camera_flip(), self.preprocess_config(),
                self.colour_rgb())

    def _on_tile_captured(self, index: int, x_um: float, y_um: float,
                          frame) -> None:
        """A scan tile → the detection queue. Full resolution, and the same
        pre-processing the operator tuned on the preview."""
        from talos.models import StagePosition

        self._engine.submit_tile(
            index, frame, self.calibration_for(frame),
            StagePosition(x_um=x_um, y_um=y_um, r_deg=0.0),
            self.identify_config(), scale=1.0, flip=self.camera_flip(),
            preprocess=self.preprocess_config(), colour=self.colour_rgb(),
            frame_scale=self.frame_scale_for(frame))

    def _on_detected(self, index: int, result, preprocessed, overlay) -> None:
        if result is None:
            return
        if index < 0:
            # LIVE jobs only. A tile's frame is the same size and shape as
            # a live one but it is a different part of the sample: letting
            # it become the display layer makes the view jump to that
            # tile's region, and the dropper — which samples the
            # pre-processed layer — would return a colour from a tile the
            # operator is not looking at.
            self.live_view.set_preprocessed_frame(preprocessed)
            self.live_view.set_overlay_frame(overlay)
            self.identify_group.set_counts(result.summary)
            self.scan_panel.show_live_candidates(result.candidates)
            return
        self.scan_panel.on_tile_result(index, result.candidates)
        # A tile just drained: the last one is what unpauses the previews
        # when a scan ends (its captures stop before its detections do).
        self._refresh_processed_views()

    # ------------------------------------------------------------------
    # housekeeping
    # ------------------------------------------------------------------

    def refresh_settings(self) -> None:
        """Follow a change made elsewhere — Preferences applied, a
        workspace switch, the camera flip, a new calibration."""
        self.scan_panel.reload_preferences()
        self.scan_panel.set_flip(self.camera_flip())
        self.colour_group.reload()
        self.preprocess_group.reload()
        self._sync_curve_centre()
        self.scan_panel.refresh_plan()

    def on_camera_flip_changed(self) -> None:
        """A flip is a 180° rotation of every delivered frame: the map's
        layout, the mosaic and the px→stage mapping all turn over, so the
        results computed under the old convention are dropped."""
        self.scan_panel.set_flip(self.camera_flip())
        self.live_view.set_overlay_frame(None)
        self.live_view.set_preprocessed_frame(None)
        self.scan_panel.clear_results()

    def update_telem(self, key: str, payload: dict) -> None:
        self.scan_panel.update_telem(key, payload)
        self._note_motion(key, payload)

    # ------------------------------------------------------------------
    # what the processed views may show
    # ------------------------------------------------------------------

    def _refresh_processed_views(self) -> None:
        """Hold the two processed views back while they would be stale,
        expensive, or both.

        Two cases, and the reason they share one switch: a moving stage
        makes the processed frame a picture of where the stage WAS, and a
        running scan makes it the work the tiles are queued behind.
        """
        scanning = (self._state.mode == "SCAN"
                    or self.scan_panel.is_scanning()
                    or self._engine.pending_tiles > 0)
        paused = scanning or self._stage_moving
        if scanning:
            note = "scanning — showing the live frame"
        elif paused:
            note = "stage moving — showing the live frame"
        else:
            note = ""
        self.live_view.set_processed_paused(paused, note)
        self._engine.set_suspended(scanning)

    def _on_mode_changed(self, _mode: str) -> None:
        self._refresh_processed_views()

    def _note_motion(self, key: str, payload: dict) -> None:
        """Track whether an axis the CAMERA can see is moving.

        The XYR stage blurs the image and the focus axis changes what is in
        it, so both pause the previews. The transfer (XYZ) axes do not
        appear in the image at all AND their firmware has no busy flag — a
        motion inferred from a position delta would hold the previews back
        on a noisy sample for no reason, so they are deliberately left out.

        Telemetry only: the scan's own moves are covered by the mode, and
        the zolix poll is starved while a scan queues jobs anyway.
        """
        if not isinstance(payload, dict):
            return
        if key == "zolix":
            moving = bool(parse_zolix(payload).get("moving"))
        elif key == "focus":
            moving = str(parse_focus(payload).get("mode", "IDLE")) != "IDLE"
        else:
            return
        if moving:
            # Every moving sample pushes the hold back; the LAST one starts
            # the countdown that ends the pause. Restarting it on the quiet
            # samples too (the obvious "elif still moving") would keep it
            # alive forever at the 10 Hz telemetry rate — the pause would
            # never lift.
            self._stage_moving = True
            self._motion_hold.start()
            self._refresh_processed_views()

    def _on_motion_hold_expired(self) -> None:
        self._stage_moving = False
        self._refresh_processed_views()

    def _log(self, message: str) -> None:
        self.sig_log.emit(str(message))

    def shutdown(self) -> None:
        """App teardown: stop the detection thread and any scan worker."""
        self._engine.shutdown()
        self.scan_panel.shutdown()


__all__ = ["PREVIEW_SCALE", "SampleFindingWorkspace"]
