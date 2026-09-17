"""The scan half of the Sample Finding tab.

Top to bottom, and in the order the work happens:

- **the map**, which is the only view of a run; double-click opens it big
  (``map_window.py``);
- **the run card** — *Scan from here*, Abort, a progress bar and a
  measured time estimate. Pinned, never scrolled away: it is the one
  thing that must be reachable while a run is going;
- **the scan settings** — a *copy of the Capture group's shape* (an output
  folder with a browse and an open button) holding only what is worth
  changing at the microscope: the area, where the origin sits, the
  directions, the path order, the start axis, whether to come back.
  Overlap, settle, speed, backlash and which extra files a run writes are
  set once in Preferences and not touched again;
- **the found samples** — the list, with *go to* and an export.

Two rules this panel keeps, both inherited from the console it replaces:

- **A scan owns the axes.** Manual input is gated by ``AppState.mode``
  ("SCAN"), and STOP ALL aborts the *run*, not just the motion in flight.
- **Nothing here blocks.** Capture runs on the scan thread; identification
  runs on the detection thread. The live stream, the autofocus and the map
  never wait on either.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from talos.cv.calibration import SENSOR_HEIGHT_PX, SENSOR_WIDTH_PX
from talos.cv.frame_source import LatestFrameSource
from talos.cv.scan import (ORIGIN_LABELS, ORIGINS, PATHS, GridScanner,
                           plan_path, scan_speed_config)
from talos.cv.scan_output import candidate_rows, write_outputs
from talos.models import (FlakeCandidate, ObjectiveCalibration, ScanParams,
                          StagePosition)
from talos.scan_settings import (load_scan_settings, save_scan_settings,
                                 scan_directory)
from talos.ui.widgets.map_window import MapWindow
from talos.ui.widgets.scan_map import (ScanMapMarker, ScanMapPlan,
                                       ScanMapTile, ScanMapWidget)
from talos.ui.widgets.segmented import SegmentedToggle

#: The settings this panel owns. Everything else in the scan section
#: belongs to Preferences — writing it here would overwrite those values
#: with whatever these widgets happen to say.
FIELD_KEYS = ("dir", "width_um", "height_um", "origin", "x_dir", "y_dir",
              "path", "serpentine", "start_axis", "return_to_start")

#: Where a corner origin is more than a nicety: it covers the area with
#: the fewest frames and lands on the far edge exactly.
ORIGIN_OPTIONS = [
    (ORIGINS[0], ORIGIN_LABELS[ORIGINS[0]],
     "The start point is the middle of the first tile. Half a frame of\n"
     "coverage sits behind you."),
    (ORIGINS[1], ORIGIN_LABELS[ORIGINS[1]],
     "The start point is a corner. The first tile is inset half a frame\n"
     "and the last one lands exactly on the far edge — fewest frames."),
    (ORIGINS[2], ORIGIN_LABELS[ORIGINS[2]],
     "The start point is a corner, and every tile sits at exactly the\n"
     "overlap you asked for; the last one overhangs by up to one step."),
]

PATH_OPTIONS = [
    (PATHS[0], "Serpentine",
     "Rows end to end, alternating direction"),
    (PATHS[1], "Spiral", "Rings, working inward (experimental)"),
    (PATHS[2], "Hilbert", "The curve, clipped to the area (experimental)"),
]


class _Worker(QThread):
    """Runs a blocking callable; emits the result (no exception escapes)."""

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


class ScanPanel(QWidget):
    """The map, the run card, the scan settings and the sample list."""

    #: A tile was captured (index, x_um, y_um, full-resolution frame) —
    #: the workspace hands it to the detection engine.
    sig_tile_captured = Signal(int, float, float, object)
    sig_log = Signal(str)
    #: Something that affects the plan changed (the workspace persists).
    sig_plan_changed = Signal()

    def __init__(self, manager, settings, state, calibration_context=None,
                 input_system=None, parent: QWidget | None = None):
        super().__init__(parent)
        self._manager = manager
        self._settings = settings
        self._state = state
        self._calibration = calibration_context
        self._input = input_system

        self._prefs = load_scan_settings(settings)
        self._job: str | None = None
        self._scan_abort = threading.Event()
        self._scanner = None
        self._scan_worker: _Worker | None = None
        self._export_worker: _Worker | None = None
        self._pending_export = None
        self._scan_tiles: dict[int, tuple[float, float]] = {}
        self._scan_hits: dict[int, list[FlakeCandidate]] = {}
        self._live_candidates: list[FlakeCandidate] = []
        self._scan_started_at = 0.0
        self._latest_frame = None
        self._flip = False
        #: Set by the workspace: how many captured tiles have not been
        #: examined yet. The panel owns the files, so it has to know when
        #: the results are complete — but the queue belongs to the engine.
        self.pending_tiles_fn = None

        self._build_ui()
        self._load_settings()
        self._refresh_fov_label()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        # The map gets LESS room than the settings below it, deliberately:
        # an area is usually wider than it is tall, so a fit-to-view plan
        # leaves the bottom half of a tall card empty — and the settings
        # and the sample list are what the operator is actually reading
        # while a run goes. The map is one double-click from being bigger.
        layout.addWidget(self._build_map_card(), 1)
        layout.addWidget(self._build_run_card())
        layout.addWidget(self._build_scroll(), 2)

    def _build_map_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("card")
        outer = QVBoxLayout(card)
        outer.setContentsMargins(6, 4, 6, 6)
        outer.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(4)
        title = QLabel("Scan map")
        bold = title.font()
        bold.setBold(True)
        title.setFont(bold)
        header.addWidget(title)
        header.addStretch(1)
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setObjectName("compact")
        self.fit_btn.setToolTip("Reset the zoom and centre the plan")
        self.fit_btn.clicked.connect(lambda: self.map.fit())
        header.addWidget(self.fit_btn)
        self.enlarge_btn = QPushButton("⤢")
        self.enlarge_btn.setObjectName("compact")
        self.enlarge_btn.setToolTip("Open the map in its own window "
                                    "(double-click the map does the same)")
        self.enlarge_btn.clicked.connect(self.enlarge_map)
        header.addWidget(self.enlarge_btn)
        outer.addLayout(header)

        self.map = ScanMapWidget()
        self.map.sig_enlarge_requested.connect(self.enlarge_map)
        self.map.sig_marker_selected.connect(self._on_marker_selected)

        self._map_page = QWidget()
        page_layout = QVBoxLayout(self._map_page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(self.map)

        self._map_away = QWidget()
        away_layout = QVBoxLayout(self._map_away)
        away_layout.addStretch(1)
        away = QLabel("The map is open in its own window.")
        away.setObjectName("dim")
        away.setAlignment(Qt.AlignmentFlag.AlignCenter)
        away_layout.addWidget(away)
        bring_back = QPushButton("Bring it back")
        bring_back.setObjectName("compact")
        bring_back.clicked.connect(self.retrieve_map)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(bring_back)
        row.addStretch(1)
        away_layout.addLayout(row)
        away_layout.addStretch(1)

        self._map_stack = QStackedWidget()
        self._map_stack.addWidget(self._map_page)
        self._map_stack.addWidget(self._map_away)
        outer.addWidget(self._map_stack, 1)

        self._map_window = MapWindow(input_system=self._input, parent=self)
        self._map_window.sig_dismissed.connect(self.retrieve_map)
        return card

    def _build_run_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.scan_btn = QPushButton("Scan from here")
        self.scan_btn.setObjectName("qa_primary")
        self.scan_btn.setToolTip(
            "Start the run at the current stage position (Esc or Abort\n"
            "stops it; manual control is blocked while it runs)")
        self.scan_btn.clicked.connect(self._on_scan)
        row.addWidget(self.scan_btn, 2)
        self.abort_btn = QPushButton("■ Abort")
        self.abort_btn.setObjectName("danger")
        self.abort_btn.setEnabled(False)
        self.abort_btn.setToolTip("Stop the scan and the stage "
                                  "(Esc does the same from anywhere)")
        self.abort_btn.clicked.connect(lambda: self._on_abort("abort"))
        row.addWidget(self.abort_btn, 1)
        layout.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.status = QLabel("Idle")
        self.status.setObjectName("dim")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        return card

    def _build_scroll(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(0, 0, 4, 4)
        layout.setSpacing(6)
        layout.addWidget(self._build_scan_group())
        layout.addWidget(self._build_samples_group())
        layout.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    def _build_scan_group(self) -> QWidget:
        box = QFrame()
        box.setObjectName("card")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # --- where it goes (the Capture group's shape) -------------------
        where = QLabel("Save to")
        layout.addWidget(where)
        path_row = QHBoxLayout()
        path_row.setSpacing(4)
        self._dir = QLineEdit()
        browse = QPushButton("…")
        browse.setObjectName("compact")
        browse.setToolTip("Choose the scan folder")
        browse.clicked.connect(self._on_browse_dir)
        self.open_btn = QPushButton("Open")
        self.open_btn.setObjectName("compact")
        self.open_btn.setToolTip("Open the scan folder")
        self.open_btn.clicked.connect(self._on_open_dir)
        path_row.addWidget(self._dir, 1)
        path_row.addWidget(browse)
        path_row.addWidget(self.open_btn)
        layout.addLayout(path_row)
        self._dir.editingFinished.connect(self._on_dir_edited)

        # --- the area ----------------------------------------------------
        self.origin = SegmentedToggle(ORIGIN_OPTIONS)
        self.origin.setToolTip("Where the stage is now, relative to the "
                               "area you type")
        layout.addWidget(_row("Origin", self.origin))

        area_row = QHBoxLayout()
        area_row.setSpacing(4)
        self.width = _spin(100.0, 100000.0, 2000.0, " µm")
        self.height = _spin(100.0, 100000.0, 1000.0, " µm")
        area_row.addWidget(QLabel("X"))
        area_row.addWidget(self.width, 1)
        area_row.addWidget(QLabel("Y"))
        area_row.addWidget(self.height, 1)
        layout.addLayout(area_row)

        self.x_dir = SegmentedToggle([(1, "+X", "The area grows to the right"),
                                      (-1, "−X", "The area grows to the left")])
        self.y_dir = SegmentedToggle([(1, "+Y", "The area grows upward in the "
                                             "frame"),
                                      (-1, "−Y", "The area grows downward")])
        dir_row = QHBoxLayout()
        dir_row.setSpacing(4)
        dir_row.addWidget(self.x_dir, 1)
        dir_row.addWidget(self.y_dir, 1)
        layout.addLayout(dir_row)

        self.path = SegmentedToggle(PATH_OPTIONS)
        layout.addWidget(_row("Path", self.path))
        self.serpentine = SegmentedToggle([
            (True, "Serpentine", "Each row continues where the last one "
                                 "ended — shortest moves"),
            (False, "One-way", "Every row starts from the same side, with a "
                               "return between rows")])
        layout.addWidget(_row("Order", self.serpentine))
        self.start_axis = SegmentedToggle([
            ("x", "X first", "Fill a row, then step in Y"),
            ("y", "Y first", "Fill a column, then step in X")])
        layout.addWidget(_row("Start", self.start_axis))
        self.return_home = SegmentedToggle([
            (True, "Return", "Come back to the start point when the run ends"),
            (False, "Stay", "Stop wherever the last tile was")])
        layout.addWidget(_row("Finish", self.return_home))

        # --- what it is (read-only) --------------------------------------
        self._fov_value = QLabel("")
        bold = self._fov_value.font()
        bold.setBold(True)
        self._fov_value.setFont(bold)
        layout.addWidget(self._fov_value)
        self._fov_note = QLabel("")
        self._fov_note.setObjectName("dim")
        self._fov_note.setWordWrap(True)
        layout.addWidget(self._fov_note)

        for toggle in (self.origin, self.x_dir, self.y_dir, self.path,
                       self.serpentine, self.start_axis, self.return_home):
            toggle.sig_changed.connect(self._on_field_changed)
        for widget in (self.width, self.height):
            widget.valueChanged.connect(self._on_field_changed)
            widget.editingFinished.connect(self._persist)
        return box

    def _build_samples_group(self) -> QWidget:
        box = QFrame()
        box.setObjectName("card")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        title = QLabel("Found samples")
        bold = title.font()
        bold.setBold(True)
        title.setFont(bold)
        layout.addWidget(title)
        self._samples_note = QLabel("Nothing matched yet.")
        self._samples_note.setObjectName("dim")
        self._samples_note.setWordWrap(True)
        layout.addWidget(self._samples_note)
        # A scrollable list of labels rather than a table: the panel is
        # narrow, and four numbers per row do not fit without eliding the
        # one that matters (where it is).
        self._sample_list = QVBoxLayout()
        self._sample_list.setSpacing(0)
        holder = QWidget()
        holder.setLayout(self._sample_list)
        list_scroll = QScrollArea()
        list_scroll.setWidgetResizable(True)
        list_scroll.setFrameShape(QFrame.Shape.NoFrame)
        list_scroll.setMinimumHeight(90)
        list_scroll.setWidget(holder)
        layout.addWidget(list_scroll, 1)
        self._buttons: list[QPushButton] = []
        row = QHBoxLayout()
        row.setSpacing(4)
        self._go_to_btn = QPushButton("Go to")
        self._go_to_btn.setObjectName("compact")
        self._go_to_btn.setEnabled(False)
        self._go_to_btn.setToolTip("Move the stage so the selected sample is "
                                   "under the crosshair, at the scan's speed")
        self._go_to_btn.clicked.connect(self._on_go_to)
        export = QPushButton("Export CSV")
        export.setObjectName("compact")
        export.clicked.connect(self._on_export_list)
        clear = QPushButton("Clear")
        clear.setObjectName("compact")
        clear.clicked.connect(self.clear_results)
        row.addWidget(self._go_to_btn)
        row.addWidget(export)
        row.addWidget(clear)
        layout.addLayout(row)
        self._candidates: list[FlakeCandidate] = []
        self._selected = -1
        self._rows = QButtonGroup(self)
        return box

    def set_flip(self, flip: bool) -> None:
        """Follow the camera flip — the map layout mirrors with it, and a
        run's mosaic is assembled with the same convention."""
        self._flip = bool(flip)
        self.map.set_flip(self._flip)

    # ------------------------------------------------------------------
    # settings
    # ------------------------------------------------------------------

    def _load_settings(self) -> None:
        saved = load_scan_settings(self._settings)
        self._dir.setText(str(self._settings.section("scan").get("dir", "")
                              or scan_directory(self._settings)))
        self.width.setValue(float(saved["width_um"]))
        self.height.setValue(float(saved["height_um"]))
        self.origin.set_value(str(saved["origin"]))
        self.x_dir.set_value(int(saved["x_dir"]))
        self.y_dir.set_value(int(saved["y_dir"]))
        self.path.set_value(str(saved["path"]))
        self.serpentine.set_value(bool(saved["serpentine"]))
        self.start_axis.set_value(str(saved["start_axis"]))
        self.return_home.set_value(bool(saved["return_to_start"]))

    def reload_preferences(self) -> None:
        """Re-read the values that live in Preferences (the ones this panel
        deliberately does not show)."""
        self._prefs = load_scan_settings(self._settings)
        self._refresh_fov_label()

    def _persist(self) -> None:
        save_scan_settings(self._settings, {
            "dir": self._dir.text().strip(),
            "width_um": self.width.value(),
            "height_um": self.height.value(),
            "origin": self.origin.value(),
            "x_dir": int(self.x_dir.value() or 1),
            "y_dir": int(self.y_dir.value() or 1),
            "path": self.path.value(),
            "serpentine": bool(self.serpentine.value()),
            "start_axis": self.start_axis.value(),
            "return_to_start": bool(self.return_home.value()),
        })

    def _on_field_changed(self, *_args) -> None:
        self.refresh_plan()
        self._persist()
        self.sig_plan_changed.emit()

    def _on_dir_edited(self) -> None:
        text = self._dir.text().strip()
        if text:
            self._persist()
            self.sig_plan_changed.emit()

    def _on_browse_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Scan folder", self._dir.text().strip())
        if chosen:
            self._dir.setText(chosen)
            self._persist()
            self.sig_plan_changed.emit()

    def _on_open_dir(self) -> None:
        import logging
        import os

        path = Path(self._dir.text().strip() or str(scan_directory(
            self._settings)))
        try:
            path.mkdir(parents=True, exist_ok=True)
            if hasattr(os, "startfile"):
                os.startfile(str(path))          # noqa: S606 - Windows
            else:
                from PySide6.QtCore import QUrl
                from PySide6.QtGui import QDesktopServices

                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        except OSError as exc:
            logging.getLogger(__name__).warning(
                "Could not open the scan folder %s: %s", path, exc)

    # ------------------------------------------------------------------
    # planning
    # ------------------------------------------------------------------

    def objective_row(self) -> dict:
        rows = self._settings.get("objectives") or []
        if not rows:
            return {}
        return rows[min(int(self._state.objective), len(rows) - 1)]

    def stage_position(self) -> StagePosition:
        return StagePosition.from_telemetry(
            self._manager.last_position.get("zolix"))

    def canonical_calibration(self) -> ObjectiveCalibration:
        if self._calibration is not None:
            return self._calibration.calibration()
        position = self._state.objective
        magnitude = 5.0 * (2.0 ** position)
        um_per_px = 2.0 / magnitude
        return ObjectiveCalibration(objective_id=-1, um_per_px_x=um_per_px,
                                    um_per_px_y=um_per_px,
                                    source="pixel_pitch")

    def fov(self) -> tuple[float, float]:
        """The field of view in µm, from the ACTIVE OBJECTIVE's calibration.

        Resolution-independent by construction: the stored value is µm per
        4K-sensor pixel, so a 1080p live frame and a 4K snapshot tile the
        same. Always derived, never entered — two sources for one number
        is a way for them to disagree, and the calibration is the one the
        rest of the application measures with.
        """
        calib = self.canonical_calibration()
        um_x = calib.um_per_px_x or 0.0
        um_y = calib.um_per_px_y or 0.0
        if um_x > 0 and um_y > 0:
            return (um_x * SENSOR_WIDTH_PX, um_y * SENSOR_HEIGHT_PX)
        magnitude = 5.0 * (2.0 ** int(self._state.objective))
        um_per_px = 2.0 / magnitude
        return (um_per_px * SENSOR_WIDTH_PX, um_per_px * SENSOR_HEIGHT_PX)

    def params_for(self, origin: StagePosition | None = None) -> ScanParams:
        origin = origin or self.stage_position()
        return ScanParams(
            x0_um=origin.x_um, y0_um=origin.y_um,
            width_um=self.width.value(), height_um=self.height.value(),
            overlap=float(self._prefs["overlap"]),
            serpentine=bool(self.serpentine.value()),
            speed_pps=float(self._prefs["speed_pps"]),
            path=str(self.path.value()),
            origin=str(self.origin.value()),
            start_axis=str(self.start_axis.value()),
            x_dir=int(self.x_dir.value() or 1),
            y_dir=int(self.y_dir.value() or 1),
            settle_ms=int(self._prefs["settle_ms"]),
            backlash_um=float(self._prefs["backlash_um"]),
            backlash_approach=int(self._prefs["backlash_approach"]),
            return_to_start=bool(self.return_home.value()))

    def refresh_plan(self) -> None:
        """The map always shows the plan the current fields would run."""
        origin = self.stage_position()
        params = self.params_for(origin)
        fov = self.fov()
        waypoints = plan_path(params, fov)
        self.map.set_plan(ScanMapPlan(
            x0_um=params.x0_um, y0_um=params.y0_um,
            width_um=params.width_um, height_um=params.height_um,
            x_dir=params.x_dir, y_dir=params.y_dir,
            fov_x_um=fov[0], fov_y_um=fov[1],
            waypoints=[(w.x_um, w.y_um) for w in waypoints]))
        if not self._scan_tiles:
            self.map.set_caption(f"{len(waypoints)} tiles planned")
        if self._job is None:
            self.set_status(
                f"{len(waypoints)} tiles · area {params.width_um:.0f} × "
                f"{params.height_um:.0f} µm")

    def _refresh_fov_label(self) -> None:
        calib = self.canonical_calibration()
        source = getattr(calib, "source", "none")
        fov_x, fov_y = self.fov()
        self._fov_value.setText(f"Field of view {fov_x:.0f} × {fov_y:.0f} µm")
        if source in ("talos_measured", "labscope"):
            where = ("measured in TALOS" if source == "talos_measured"
                     else "imported from Labscope")
            self._fov_note.setText(f"From the objective ({where}), "
                                   f"{calib.um_per_px_x:.4f} µm/px.")
        else:
            self._fov_note.setText(
                "ESTIMATED from the sensor pitch — set the real value in "
                "Preferences → Objectives & Calibration, where it also fixes "
                "the scale bar and the measured areas.")
        self.refresh_plan()

    # ------------------------------------------------------------------
    # the map, big
    # ------------------------------------------------------------------

    def enlarge_map(self) -> None:
        self._map_window.adopt(self.map)
        self._map_stack.setCurrentIndex(1)
        self._map_window.show()
        self._map_window.raise_()

    def retrieve_map(self) -> None:
        widget = self._map_window.release()
        if widget is not None:
            self._map_page.layout().addWidget(widget)
        self._map_stack.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # frames and telemetry in
    # ------------------------------------------------------------------

    def on_frame(self, frame: np.ndarray) -> None:
        """The newest streamed frame — kept for the live map position and
        for the frame shape recorded in a run's metadata."""
        self._latest_frame = frame

    def update_telem(self, key: str, payload: dict) -> None:
        """Follow the stage on the map.

        The position lives under ``"position"`` in the device payload; the
        whole payload was passed to ``from_telemetry`` before, which reads
        a FLAT dict — so it found nothing, returned (0, 0, 0), and the
        "you are here" box sat at stage origin for every idle frame.
        """
        if key != "zolix" or not payload:
            return
        position = payload.get("position") if isinstance(payload, dict) else None
        if not position:
            return
        pos = StagePosition.from_telemetry(position)
        if self._job is None:
            self.map.set_footprint(pos.x_um, pos.y_um)

    # ------------------------------------------------------------------
    # results in
    # ------------------------------------------------------------------

    def on_tile_result(self, index: int, candidates) -> None:
        """Detection finished for a scan tile."""
        self._scan_hits[index] = list(candidates)
        if self._pending_export is not None:
            self._maybe_export()
        self._show_candidates(self._flatten_scan_hits(),
                              f"scan · {len(self._scan_hits)} tiles")
        self.refresh_markers()

    def show_live_candidates(self, candidates) -> None:
        """Detection finished for the live frame."""
        self._live_candidates = list(candidates)
        if self._job is None and not self._scan_hits:
            self._show_candidates(self._live_candidates, "live view")

    def _flatten_scan_hits(self) -> list[FlakeCandidate]:
        out: list[FlakeCandidate] = []
        for index in sorted(self._scan_hits):
            out.extend(self._scan_hits[index])
        return out

    def _show_candidates(self, candidates, source: str) -> None:
        """Rebuild the list. The rows come from the same formatter the CSV
        export uses, so what is exported is what was shown."""
        self._candidates = list(candidates)
        self._selected = -1
        self._rows = QButtonGroup(self)
        self._rows.setExclusive(True)
        while self._sample_list.count():
            item = self._sample_list.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for index, cand in enumerate(self._candidates):
            row = QPushButton(
                f"{index + 1}.  {cand.x_um:.1f}, {cand.y_um:.1f} µm · "
                f"{cand.area_um2:.1f} µm²")
            row.setObjectName("sample_row")
            row.setCheckable(True)
            row.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            row.setToolTip("Select, then Go to, to bring it under the "
                           "crosshair")
            row.clicked.connect(
                lambda _checked=False, index=index: self._select(index))
            self._rows.addButton(row)
            self._sample_list.addWidget(row)
        count = len(self._candidates)
        self._samples_note.setText(
            f"{count} sample(s) from the {source}." if count
            else f"Nothing matched in the {source}.")
        # an enabled button that only ever answers "select a sample first"
        # is a button that lies
        self._go_to_btn.setEnabled(count > 0 and self._job is None)

    def _select(self, index: int) -> None:
        if not 0 <= index < len(self._candidates):
            return
        self._selected = index
        self._rows.button(index).setChecked(True)
        self._go_to_btn.setEnabled(self._job is None)

    def _on_marker_selected(self, index: int) -> None:
        self._select(index)

    def refresh_markers(self) -> None:
        candidates = self._flatten_scan_hits()
        self.map.set_markers(
            [ScanMapMarker(x_um=c.x_um, y_um=c.y_um, label=str(index + 1))
             for index, c in enumerate(candidates)])
        self.map.set_caption(
            f"{len(self._scan_tiles)} tiles · {len(candidates)} sample(s)")

    def clear_results(self) -> None:
        self._scan_hits.clear()
        self._scan_tiles.clear()
        self._live_candidates = []
        self._candidates = []
        self._selected = -1
        while self._sample_list.count():
            item = self._sample_list.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.map.set_markers([])
        self.map.set_caption("")
        self._samples_note.setText("Results cleared.")
        self._go_to_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # the run
    # ------------------------------------------------------------------

    def is_scanning(self) -> bool:
        return self._job is not None

    def _on_scan(self) -> None:
        if self._job is not None:
            return
        if self._state.mode != "MANUAL":
            QMessageBox.information(
                self, "Scan",
                f"The stage is in use ({self._state.mode}) — a scan cannot "
                "start while another job owns the axes.")
            return
        position = self.stage_position()
        if not self._manager.last_position.get("zolix"):
            QMessageBox.information(
                self, "Scan", "No stage position yet — the Zolix controller "
                "has not reported. Check the connection.")
            return
        # The scanner refuses to start on a moving axis, and the controller
        # rejects opcodes to one: cancel held jogs and queue a stop ahead of
        # the first move (same worker, so it lands first).
        #
        # A ZOLIX stop, deliberately — NOT manager.stop_all(). That would
        # fire sig_stop_all_done, which this panel treats as the operator's
        # STOP ALL, and the scan would abort itself the moment it started.
        if self._input is not None:
            self._input.cancel_all_holds("scan start")
        self._manager.submit("zolix", "stop", priority=1)
        self._scan_abort.clear()
        self._scan_hits.clear()
        self._scan_tiles.clear()
        self._pending_export = None
        self.refresh_plan()
        self.map.clear_tiles()                 # the previous run's tiles
        self._set_job("scan")
        self._state.set_mode("SCAN")
        self._scan_started_at = time.monotonic()
        self.progress.setValue(0)
        self.set_status("Scanning — Esc or Abort stops it")
        self._scan_worker = _Worker(lambda: self._run_scan(position), self)
        self._scan_worker.sig_log.connect(self._log)
        self._scan_worker.sig_done.connect(self._on_scan_done)
        # deleteLater AND drop the reference: keeping a Python handle to a
        # deleted C++ object makes `isRunning()` raise at teardown, which
        # is exactly when shutdown() wants to ask.
        self._scan_worker.finished.connect(self._scan_worker.deleteLater)
        self._scan_worker.finished.connect(self._forget_scan_worker)
        self._scan_worker.start()

    def _forget_scan_worker(self) -> None:
        self._scan_worker = None

    def _run_scan(self, origin: StagePosition):
        """The scan thread. Owns the adapter and the scanner; every UI
        update comes back as a queued signal."""
        from talos.hal.proxies.stage_adapter import ManagerStageAdapter

        stage_cfg = scan_speed_config(self._settings.device("zolix"),
                                      float(self._prefs["speed_pps"]))
        adapter = ManagerStageAdapter(self._manager, stage_cfg,
                                      abort_check=self._scan_abort.is_set)
        params = self.params_for(origin)
        fov = self.fov()
        slot = getattr(self._manager, "frame_slot", None)
        source = LatestFrameSource(slot) if slot is not None else None
        if source is None:
            self._log("no frame slot — the scan will record positions only")
        scanner = GridScanner(adapter, source, thumb_width=192)
        scanner.sig_progress.connect(self._on_progress)
        scanner.sig_tile.connect(self._on_tile)
        scanner.sig_frame.connect(self._on_scan_frame)
        scanner.sig_log.connect(self._log)
        self._scanner = scanner
        out_dir = scan_directory(self._settings) / time.strftime(
            "scan_%Y%m%d_%H%M%S")
        try:
            result = scanner.run(
                params, out_dir,
                meta={"fov_um": fov, "objective_id": self._state.objective,
                      "frame_shape": (self._latest_frame.shape
                                      if self._latest_frame is not None
                                      else None)})
        finally:
            self._scanner = None
            adapter.close()
        return {"result": result, "params": params, "fov": fov,
                "out_dir": out_dir}

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(done)
        elapsed = time.monotonic() - self._scan_started_at
        eta = ""
        if done >= 2 and elapsed > 0:
            remaining = elapsed / done * (total - done)
            eta = f" · {_format_seconds(remaining)} left"
        self.set_status(f"Tile {done}/{total}{eta}")

    def _on_tile(self, index: int, x_um: float, y_um: float, thumb) -> None:
        self._scan_tiles[index] = (x_um, y_um)
        self.map.add_tile(ScanMapTile(index=index, x_um=x_um, y_um=y_um,
                                      thumb=thumb))
        self.refresh_markers()

    def _on_scan_frame(self, index: int, x_um: float, y_um: float,
                       frame) -> None:
        self._scan_tiles[index] = (x_um, y_um)
        self.sig_tile_captured.emit(index, float(x_um), float(y_um), frame)

    def _on_scan_done(self, payload) -> None:
        if self._state.mode == "SCAN":
            self._state.set_mode("MANUAL")
        self._set_job(None)
        self.progress.setVisible(False)
        if not payload:
            self.set_status("Scan failed (see the log)")
            return
        result = payload["result"]
        aborted = result.aborted or self._scan_abort.is_set()
        what = "Scan aborted" if aborted else "Scan done"
        missing = getattr(result, "missing", 0)
        note = f" · {missing} waypoint(s) had no frame" if missing else ""
        self.set_status(f"{what}: {len(result.frames)} frame(s){note}")
        self._log(f"{what} → {payload['out_dir']}")
        self._pending_export = payload
        self._maybe_export()

    def _on_abort(self, reason: str = "abort") -> None:
        """Cooperative abort — never QThread.terminate() (it killed the
        scan thread mid-serial-write once, leaving the port open)."""
        scanner = self._scanner
        if scanner is not None:
            self._scan_abort.set()
            scanner.request_abort()
        if self._input is not None:
            self._input.cancel_all_holds(reason)
        self._manager.stop_all()
        self.set_status("Aborting…")

    def on_stop_all_done(self) -> None:
        """STOP ALL must abort the RUN, not just the motion in flight.

        Without this the scan's abort flag stayed clear, the controller
        stopped, and the run cheerfully continued at the next waypoint —
        Esc looked like it had not worked at all.
        """
        if self._job == "scan":
            self._on_abort("stop all")

    def _set_job(self, job: str | None) -> None:
        self._job = job
        busy = job is not None
        self.scan_btn.setEnabled(not busy)
        self.abort_btn.setEnabled(busy)
        self.progress.setVisible(busy)
        self._go_to_btn.setEnabled(not busy and bool(self._candidates))
        for widget in (self.origin, self.x_dir, self.y_dir, self.path,
                       self.serpentine, self.start_axis, self.return_home,
                       self.width, self.height, self._dir):
            widget.setEnabled(not busy)
        if not busy:
            self._refresh_fov_label()

    # ------------------------------------------------------------------
    # what a finished scan writes
    # ------------------------------------------------------------------

    def _maybe_export(self) -> None:
        """Wait for the detection queue to drain, then write the extras.

        Detection outlives the capture by design (it must not slow the
        stage down), so "the scan finished" is not the same moment as "the
        results are in" — and the extras are computed here rather than at
        the end of the run."""
        if self._pending_export is None or self._export_worker is not None:
            return
        if self._hits_outstanding():
            return
        payload = dict(self._pending_export)
        self._pending_export = None
        payload["hits"] = {index: list(items)
                           for index, items in self._scan_hits.items()}
        payload["tiles"] = dict(self._scan_tiles)
        payload["flip"] = self._flip
        payload["exports"] = {
            "mosaic": bool(self._prefs["export_mosaic"]),
            "candidates": bool(self._prefs["export_candidates"]),
            "overview": bool(self._prefs["export_overview"])}
        self._export_worker = _Worker(
            lambda: self._finish_exports(payload), self)
        self._export_worker.sig_done.connect(self._on_export_done)
        self._export_worker.sig_log.connect(self._log)
        self._export_worker.finished.connect(self._export_worker.deleteLater)
        self._export_worker.start()

    def _hits_outstanding(self) -> int:
        if self.pending_tiles_fn is None:
            return 0
        try:
            return int(self.pending_tiles_fn())
        except Exception:  # noqa: BLE001
            return 0

    @staticmethod
    def _finish_exports(payload) -> dict:
        """Hand the summaries to cv.scan_output, which owns the format."""
        return write_outputs(
            Path(payload["out_dir"]), Path(payload["result"].manifest_path),
            tiles=payload["tiles"], hits=payload["hits"],
            fov_um=payload["fov"], exports=payload["exports"],
            flip=bool(payload.get("flip", False)))

    def _on_export_done(self, summary) -> None:
        self._export_worker = None
        if not summary:
            return
        detail = ", ".join(summary["written"]) or "nothing extra"
        self.set_status(f"Scan done → {summary['dir']} ({detail})")
        self._log(f"wrote {detail} to {summary['dir']}")

    # ------------------------------------------------------------------
    # the sample list
    # ------------------------------------------------------------------

    def _on_export_list(self) -> None:
        """Write the CURRENT list — the live-view candidates when no scan
        has run, the scan's samples otherwise — wherever the user says."""
        if not self._candidates:
            QMessageBox.information(self, "Export", "Nothing to export yet.")
            return
        default = str(scan_directory(self._settings) / "samples.csv")
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export samples", default, "CSV (*.csv)")
        if not path:
            return
        import csv

        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["#", "x_um", "y_um", "area_um2", "edge"])
            writer.writerows(candidate_rows(self._candidates))
        self.set_status(f"Exported {len(self._candidates)} sample(s)")

    def _on_go_to(self) -> None:
        if self._selected < 0 or self._selected >= len(self._candidates):
            QMessageBox.information(self, "Go to", "Select a sample first.")
            return
        if self._state.mode != "MANUAL":
            QMessageBox.information(
                self, "Go to",
                f"The stage is in use ({self._state.mode}) — sample moves "
                "are refused while a job owns the axes.")
            return
        cand = self._candidates[self._selected]
        position = self.stage_position()
        dx = cand.x_um - position.x_um
        dy = cand.y_um - position.y_um
        answer = QMessageBox.question(
            self, "Move to sample",
            f"Move the stage by ΔX={dx:+.1f} µm, ΔY={dy:+.1f} µm to bring "
            "this sample to the crosshair?",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        if answer != QMessageBox.StandardButton.Ok:
            return
        # The same speed the scan runs at, for the same reason: this is a
        # scan move, not a jog.
        self._manager.submit("zolix", "move_rel_um", dx, dy,
                             speed_pps=int(self._prefs["speed_pps"]))
        self.set_status(f"Moving to sample #{self._selected + 1}")

    # ------------------------------------------------------------------
    # small things
    # ------------------------------------------------------------------

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def _log(self, message: str) -> None:
        self.sig_log.emit(str(message))

    def shutdown(self) -> None:
        """App teardown: let the threads finish before the process exits.

        Guarded against a worker whose C++ object is already gone — a
        finished QThread deletes itself, and asking a deleted one whether
        it is running raises rather than answering.
        """
        for worker in (self._scan_worker, self._export_worker):
            if worker is None:
                continue
            try:
                if worker.isRunning():
                    worker.wait(4000)
            except RuntimeError:      # already deleted by deleteLater
                continue


def _row(label: str, widget: QWidget) -> QWidget:
    """A dim label and a control on one line, the label fixed-width so the
    controls line up down the group."""
    holder = QWidget()
    layout = QHBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    text = QLabel(label)
    text.setObjectName("dim")
    text.setFixedWidth(44)
    layout.addWidget(text)
    layout.addWidget(widget, 1)
    return holder


def _spin(low: float, high: float, value: float, suffix: str = ""):
    """A µm field. One decimal: sub-micron precision in a spin box the
    operator nudges with a mouse is noise, and the stage does not resolve
    it anyway."""
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(1)
    box.setSingleStep(10.0)
    box.setSuffix(suffix)
    box.setValue(value)
    return box


def _format_seconds(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes} min {rest:02d} s"


__all__ = ["FIELD_KEYS", "ScanPanel"]
