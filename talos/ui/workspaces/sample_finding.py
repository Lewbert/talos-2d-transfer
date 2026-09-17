"""Sample Finding workspace: the navigation skeleton (live view + fixed
right column + shared bottom strip) with scan-oriented panels — Scan
Actions, scrollable Scan/Detection settings + a MANUAL-ONLY camera
profile (no auto settings: detection needs stable brightness), and the
flakes results table.

Mode locking: while a scan/autofocus job runs, the workspace sets the
AppState mode to SCAN/AUTOFOCUS so manual moves can be refused.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace

import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from talos.cv.calibration import SENSOR_WIDTH_PX
from talos.cv.flakes import ClassicFlakeDetector, FlakeConfig, flake_to_stage
from talos.cv.scan import GridScanner, grid_shape
from talos.models import ObjectiveCalibration, ScanParams, StagePosition
from talos.ui.widgets.collapsible import CollapsibleGroup
from talos.ui.widgets.control_groups import CameraGroup
from talos.ui.widgets.live_view import LiveViewWidget, ScanPlanOverlay

_SCAN_KEYS = ("fov_x_um", "fov_y_um", "min_flake_area_um2",
              "width_um", "height_um")


def load_scan_params(settings) -> dict:
    """The scan-section values (pure — unit-testable)."""
    scan = settings.section("scan")
    return {key: scan.get(key) for key in _SCAN_KEYS}


def save_scan_params(settings, fields: dict) -> None:
    scan = settings.section("scan")
    for key, value in fields.items():
        scan[key] = value
    settings.save()


class _Worker(QThread):
    """Runs a blocking callable; emits the result (no exceptions escape)."""

    sig_done = Signal(object)
    sig_log = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            self.sig_done.emit(self._fn())
        except Exception as exc:  # noqa: BLE001
            self.sig_log.emit(f"job failed: {exc}")
            self.sig_done.emit(None)


class SampleFindingWorkspace(QWidget):
    def __init__(self, manager, settings, state, parent: QWidget | None = None,
                 autofocus_service=None, autogain=None,
                 calibration_context=None, input_system=None):
        super().__init__(parent)
        self._autofocus = autofocus_service
        if autofocus_service is not None:
            autofocus_service.sig_af_finished.connect(self._on_af_finished)
        self._manager = manager
        self._settings = settings
        self._state = state
        self._input = input_system
        self._calibration = calibration_context
        self._last_frame: np.ndarray | None = None
        self._flakes: list = []
        # One job owner at a time (scan XOR autofocus XOR idle): a bare
        # mode string let a scan and an AF run clobber each other's mode
        # and action buttons (the scan's Abort button was re-disabled by
        # the AF's finished handler, and starting an AF mid-scan re-opened
        # the manual-motion gate).
        self._job: str | None = None
        self._scan_worker: _Worker | None = None
        self._detect_worker: _Worker | None = None
        self._scanner = None
        self._scan_abort = threading.Event()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: the live view (frames arrive via MainWindow).
        self.live_view = LiveViewWidget()
        splitter.addWidget(self.live_view)

        # Right: scan actions (fixed) + scrollable settings + results.
        # The 4px gutters keep the group outlines off the splitter
        # handle (left) and the window edge (right).
        right_col = QWidget()
        right = QVBoxLayout(right_col)
        right.setContentsMargins(4, 0, 4, 0)
        right.setSpacing(6)

        actions = QGroupBox("Scan Actions")
        actions_layout = QVBoxLayout(actions)
        self._detect_btn = QPushButton("Detect flakes (current frame)")
        self._detect_btn.clicked.connect(self._on_detect)
        actions_layout.addWidget(self._detect_btn)
        self._autofocus_btn = QPushButton("Autofocus (bounded)")
        self._autofocus_btn.clicked.connect(self._on_autofocus)
        actions_layout.addWidget(self._autofocus_btn)
        self._scan_btn = QPushButton("Start grid scan")
        self._scan_btn.clicked.connect(self._on_scan)
        actions_layout.addWidget(self._scan_btn)
        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setEnabled(False)
        self._abort_btn.clicked.connect(self._on_abort)
        actions_layout.addWidget(self._abort_btn)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        actions_layout.addWidget(self._progress)
        self._status = QLabel("Idle")
        self._status.setObjectName("dim")
        actions_layout.addWidget(self._status)
        # foldable like the settings groups — collapsing it frees the
        # height the settings column needs to show the whole Camera group
        right.addWidget(CollapsibleGroup(
            "Scan Actions", actions, settings=settings, state_key="scan"))

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
        groups.addWidget(CollapsibleGroup(
            "Scan & Detection", self._build_scan_group(),
            settings=settings, state_key="scan"))
        self.camera_group = CameraGroup(manager, settings, autogain,
                                        profile_kind="scan")
        groups.addWidget(CollapsibleGroup(
            "Camera", self.camera_group, settings=settings,
            state_key="scan"))
        groups.addStretch(1)
        scroll.setWidget(groups_col)
        right.addWidget(scroll, stretch=3)

        # Results: the flakes table in its own stretch area.
        results = QGroupBox("Flakes")
        results_layout = QVBoxLayout(results)
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["#", "X µm", "Y µm", "Area µm²", "Score"])
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        header = self._table.horizontalHeader()
        for col in range(5):
            header.setSectionResizeMode(col, header.ResizeMode.Stretch)
        self._table.setMinimumHeight(110)
        results_layout.addWidget(self._table)
        self._go_to_btn = QPushButton("Go to selected flake")
        self._go_to_btn.clicked.connect(self._on_go_to)
        results_layout.addWidget(self._go_to_btn)
        right.addWidget(results, stretch=1)

        right_col.setMinimumWidth(300)
        right_col.setMaximumWidth(420)
        splitter.addWidget(right_col)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1080, 320])
        layout.addWidget(splitter)

        # The scan-path indicator starts as a PREVIEW of the configured
        # grid and becomes the live progress display during a run.
        self._refresh_scan_plan()

    def _build_scan_group(self) -> QGroupBox:
        params = QGroupBox("Scan & Detection")
        form = QFormLayout(params)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        saved = load_scan_params(self._settings)

        def field(value, lo, hi, step=1.0):
            box = QDoubleSpinBox()
            box.setRange(lo, hi)
            box.setSingleStep(step)
            box.setValue(float(value))
            return box

        self._fov_x = field(saved.get("fov_x_um") or 700.0, 100, 100000)
        form.addRow("FOV X (µm)", self._fov_x)
        self._fov_y = field(saved.get("fov_y_um") or 390.0, 100, 100000)
        form.addRow("FOV Y (µm)", self._fov_y)
        self._overlap = field(self._settings.section("scan")
                              .get("overlap", 0.10), 0.0, 0.9, 0.05)
        form.addRow("Overlap", self._overlap)
        self._min_area = field(saved.get("min_flake_area_um2") or 30.0,
                               0.5, 1000)
        form.addRow("Min flake (µm²)", self._min_area)
        self._grid_w = field(saved.get("width_um") or 2000.0, 100, 100000)
        form.addRow("Grid W (µm)", self._grid_w)
        self._grid_h = field(saved.get("height_um") or 1000.0, 100, 100000)
        form.addRow("Grid H (µm)", self._grid_h)
        self._scan_fields = {
            "fov_x_um": self._fov_x, "fov_y_um": self._fov_y,
            "overlap": self._overlap, "min_flake_area_um2": self._min_area,
            "width_um": self._grid_w, "height_um": self._grid_h,
        }
        for box in self._scan_fields.values():
            box.editingFinished.connect(self._persist_scan)
            box.valueChanged.connect(self._refresh_scan_plan)
        return params

    def _persist_scan(self) -> None:
        save_scan_params(
            self._settings,
            {key: box.value() for key, box in self._scan_fields.items()})

    # --- scan-path indicator ---------------------------------------------

    def _scan_params(self) -> ScanParams:
        return ScanParams(x0_um=0.0, y0_um=0.0,
                          width_um=self._grid_w.value(),
                          height_um=self._grid_h.value(),
                          overlap=self._overlap.value())

    def _refresh_scan_plan(self) -> None:
        """Preview the serpentine path from the CURRENT field values.

        Same maths as the run (cv.scan.grid_shape), so the preview can
        never promise a grid the scan would not walk.
        """
        fov = (self._fov_x.value(), self._fov_y.value())
        cols, rows = grid_shape(self._scan_params(), fov)
        self.live_view.set_scan_plan(ScanPlanOverlay(
            cols=cols, rows=rows,
            detail=f"{cols} × {rows} grid · {self._grid_w.value():.0f} × "
                   f"{self._grid_h.value():.0f} µm"))

    def _on_scan_progress(self, done: int, total: int) -> None:
        """Live row/column highlight (queued from the scan worker)."""
        plan = self.live_view._scan_plan
        if plan is None or plan.cols <= 0:
            return
        index = max(0, min(int(done) - 1, max(0, int(total) - 1)))
        self.live_view.set_scan_plan(replace(
            plan, active_row=index // plan.cols, active_col=index % plan.cols))

    # ------------------------------------------------------------------

    def on_frame(self, frame: np.ndarray) -> None:
        self._last_frame = frame

    def invalidate_detections(self) -> None:
        """Drop the detected flakes — called when the camera flip changes.

        The table holds pixel centroids from the previous frame
        orientation: after a 180° rotation they point at the diagonally
        opposite spot, and "Go to selected flake" would drive the stage to
        a mirrored position.
        """
        if not self._flakes:
            return
        self._flakes = []
        self._fill_table()
        self._status.setText("Flakes cleared (camera orientation changed)")

    def _active_calibration(self) -> ObjectiveCalibration:
        if self._calibration is not None:
            return self._calibration.calibration()
        # Fallback (no context — the Axiocam 2 µm pixels / magnification).
        pos = self._state.objective
        mag = 5.0 * (2.0 ** pos)
        um_per_px = 2.0 / mag
        return ObjectiveCalibration(objective_id=-1, um_per_px_x=um_per_px,
                                    um_per_px_y=um_per_px, source="pixel_pitch")

    def _live_calibration(self) -> ObjectiveCalibration | None:
        """The active calibration expressed in LIVE-frame pixels.

        The stored value is canonical per 4K-sensor pixel and a 1080p
        frame covers 2× the µm per pixel. Every pixel↔µm conversion in
        this workspace must use the SAME scaled value: the detector scaled
        it, the flake→stage mapping did not, so "go to flake" commanded
        half the distance it should have.
        """
        calib = self._active_calibration()
        frame = self._last_frame
        if calib is None or frame is None or not frame.shape[1]:
            return calib
        factor = SENSOR_WIDTH_PX / float(frame.shape[1])
        return replace(
            calib,
            um_per_px_x=(calib.um_per_px_x or 0.0) * factor,
            um_per_px_y=(calib.um_per_px_y or 0.0) * factor)

    # --- flake detection --------------------------------------------------

    def _on_detect(self) -> None:
        if self._last_frame is None:
            QMessageBox.information(self, "Detect", "No camera frame yet.")
            return
        if self._detect_worker is not None:
            return  # a run is already in flight
        calib = self._live_calibration()
        cfg = FlakeConfig(min_area_um2=self._min_area.value())
        frame = self._last_frame
        self._detect_btn.setEnabled(False)
        self._status.setText("Detecting…")
        # Off the GUI thread: the detector is a full cvtColor + Otsu +
        # morphology + findContours pass over a 1080p frame (hundreds of
        # ms), which froze the live view and the status strip while it ran.
        worker = _Worker(lambda: ClassicFlakeDetector().find(frame, calib, cfg),
                         self)
        self._detect_worker = worker
        worker.sig_done.connect(self._on_detect_done)
        worker.start()

    def _on_detect_done(self, flakes) -> None:
        self._detect_worker = None
        self._detect_btn.setEnabled(self._job is None)
        if flakes is None:
            self._status.setText("Detection failed (see log)")
            return
        self._flakes = flakes
        self._fill_table()
        self._status.setText(f"Detected {len(self._flakes)} flakes")

    def _fill_table(self) -> None:
        self._table.setRowCount(len(self._flakes))
        for row, flake in enumerate(self._flakes):
            cells = [str(row + 1), f"{flake.x_um:.1f}", f"{flake.y_um:.1f}",
                     f"{flake.area_um2:.1f}", f"{flake.score:.1f}"]
            for col, text in enumerate(cells):
                self._table.setItem(row, col, QTableWidgetItem(text))

    def _on_go_to(self) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._flakes):
            QMessageBox.information(self, "Go to", "Select a flake row first.")
            return
        if self._state.mode != "MANUAL":
            # This submits straight to the manager, so it must carry the
            # same gate as the resolver-driven jogs — before, a click here
            # moved the stage while a scan owned the axes.
            QMessageBox.information(
                self, "Go to",
                f"The stage is in use ({self._state.mode}) — "
                "flake moves are refused while a job owns the axes.")
            return
        flake = self._flakes[row]
        calib = self._live_calibration()
        # Current stage position comes from the last Zolix telemetry — a
        # plain dict (asdict), so it must be converted before the mapping
        # reads .x_um off it.
        pos = StagePosition.from_telemetry(self._manager.last_position.get("zolix"))
        dx, dy = self._flakes_to_move(flake, calib, pos)
        answer = QMessageBox.question(
            self, "Move to flake",
            f"Move the stage by ΔX={dx:+.1f} µm, ΔY={dy:+.1f} µm to center this flake?",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Ok:
            self._manager.submit("zolix", "move_rel_um", dx, dy)
            self._status.setText(f"Moved to flake #{row + 1}")

    def _flakes_to_move(self, flake, calib, pos: StagePosition) -> tuple[float, float]:
        if self._last_frame is not None:
            h, w = self._last_frame.shape[:2]
            stage = flake_to_stage(flake.x_px, flake.y_px, (h, w, 3), calib, pos)
            return stage[0] - pos.x_um, stage[1] - pos.y_um
        return 0.0, 0.0

    # --- autofocus ----------------------------------------------------------

    def _on_autofocus(self) -> None:
        """Full autofocus through the service (focus proxy job + exposure
        dance). The Navigation workspace hosts the detailed AF panel."""
        if self._autofocus is None:
            QMessageBox.information(self, "Autofocus",
                                    "Autofocus service unavailable.")
            return
        self._set_job("autofocus")
        self._status.setText("Autofocus running — Esc aborts")
        self._autofocus.start_af_s()

    def _on_af_finished(self, result) -> None:
        if self._job == "autofocus":
            self._set_job(None)
        if result is not None:
            self._status.setText(
                "Autofocus " + ("focused ✔" if result.success
                                else f"{result.message}"))

    def _set_job(self, job: str | None) -> None:
        """Single owner for the workspace's action buttons.

        Scan and autofocus both drive the same axes, so only one may be
        startable at a time; the mode string alone could not express that
        (each job set it and reset it unconditionally)."""
        self._job = job
        busy = job is not None
        self._scan_btn.setEnabled(not busy)
        self._autofocus_btn.setEnabled(not busy)
        self._detect_btn.setEnabled(not busy)
        self._go_to_btn.setEnabled(not busy)
        self._abort_btn.setEnabled(busy)
        if job == "scan":
            self._progress.setVisible(True)
        elif job is None:
            self._progress.setVisible(False)

    # --- scan ----------------------------------------------------------------

    def _on_scan(self) -> None:
        # The scanner refuses to start on a moving axis (the controller
        # rejects opcodes to a moving axis). Cancel held jogs and stop
        # first: the stop job is queued ahead of the first move on the
        # same worker, so it is guaranteed to land before the scan moves.
        if self._input is not None:
            self._input.cancel_all_holds("scan start")
        self._manager.stop_all()
        self._scan_abort.clear()
        self._set_job("scan")
        self._state.set_mode("SCAN")
        self._status.setText("Scan running — Esc aborts")
        self._scan_worker = _Worker(self._run_scan, self)
        self._scan_worker.sig_done.connect(self._on_scan_done)
        self._scan_worker.start()

    def _run_scan(self):
        from talos.cv.scan import GridScanner, scale_scan_speed_config
        from talos.hal.proxies.stage_adapter import ManagerStageAdapter
        from talos.paths import get_scan_dir

        # The active objective's stage multiplier scales the scan speeds
        # (a COPY of the config — the live settings stay untouched).
        rows = self._settings.get("objectives") or []
        row = rows[min(int(self._state.objective), len(rows) - 1)] if rows else {}
        stage_cfg = scale_scan_speed_config(
            self._settings.device("zolix"),
            float(row.get("stage_speed_multiplier") or 1.0))
        # The scan drives the manager's OWN zolix proxy (the adapter
        # submits jobs): a second driver cannot open the same COM port,
        # and a private device was invisible to STOP ALL.
        adapter = ManagerStageAdapter(self._manager, stage_cfg,
                                      abort_check=self._scan_abort.is_set)
        params = self._scan_params()
        scanner = GridScanner(adapter)
        # Live row/column highlight: a queued connection from the worker
        # thread (the slot runs on the GUI thread).
        scanner.sig_progress.connect(self._on_scan_progress)
        self._scanner = scanner
        out_dir = get_scan_dir() / time.strftime("scan_%Y%m%d_%H%M%S")
        try:
            return scanner.run(
                params, out_dir,
                meta={"fov_um": (self._fov_x.value(), self._fov_y.value()),
                      "objective_id": self._state.objective})
        finally:
            self._scanner = None
            adapter.close()

    def _on_scan_done(self, result) -> None:
        # Only the scan clears the SCAN mode: a concurrent job (or a
        # restored mode) must not be clobbered by a late completion.
        if self._state.mode == "SCAN":
            self._state.set_mode("MANUAL")
        if self._job == "scan":
            self._set_job(None)
        # back to a preview (no active row) once the run is over
        self._refresh_scan_plan()
        aborted = self._scan_abort.is_set()
        if result is not None:
            what = "Scan aborted" if aborted else "Scan done"
            missing = getattr(result, "missing", 0)
            note = (f" — {missing} waypoint(s) had no frame (no camera)"
                    if missing else "")
            self._status.setText(f"{what}: {len(result.frames)} frames"
                                 f"{note} → {result.manifest_path}")
        else:
            self._status.setText("Scan failed (see log)")

    def _on_abort(self) -> None:
        """Cooperative abort — never QThread.terminate().

        terminate() killed the scan thread mid-serial-write: its
        `finally` never ran (the port stayed open) and the controller
        could be left mid-move. The scanner stops itself instead — the
        flag breaks the waypoint loop and a priority stop is queued for
        whatever motion is in flight; the worker then unwinds normally.
        """
        scanner = self._scanner
        if scanner is not None:
            self._scan_abort.set()
            scanner.request_abort()
        if self._autofocus is not None and getattr(self._autofocus, "busy", False):
            self._autofocus.abort()
        if self._input is not None:
            self._input.cancel_all_holds("abort")
        self._manager.stop_all()
        self._status.setText("Aborting…")
