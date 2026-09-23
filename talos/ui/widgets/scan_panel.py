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
- **the found samples** — the list, with *go to*, an export, and a *View*
  column that opens the frame the sample was found in.

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

import cv2
import numpy as np
from PySide6.QtCore import QThread, Qt, QTimer, Signal
from PySide6.QtWidgets import (
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
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from talos.cv.calibration import SENSOR_HEIGHT_PX, SENSOR_WIDTH_PX
from talos.hal.base import StageSpeed
from talos.cv.frame_source import LatestFrameSource
from talos.cv.scan import (HILBERT, MAX_PENDING_TILES, ONE_WAY,
                           ORIGIN_LABELS, ORIGINS, PATHS, SERPENTINE, SPIRAL,
                           GridScanner, plan_path, scan_speed_config)
from talos.cv.scan_output import (candidate_rows, frames_from_manifest,
                                  write_outputs)
from talos.ui.widgets.sample_review import SampleReviewWindow, SampleShot
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

#: How the stage walks the area — ONE question, four answers. "Serpentine"
#: and "One-way" are the same cells with a different direction rule (the
#: CV layer spells that as ``path='serpentine'`` plus a ``serpentine``
#: flag, and both keys are still written so a stored configuration and the
#: settings schema are unchanged); spiral and Hilbert ship as experimental.
PATH_OPTIONS = [
    (SERPENTINE, "Serpentine",
     "Rows end to end, alternating direction — the shortest moves"),
    (ONE_WAY, "One-way",
     "Every row starts from the same side, with a return between rows"),
    (SPIRAL, "Spiral", "Rings, working inward (experimental)"),
    (HILBERT, "Hilbert", "The curve, clipped to the area (experimental)"),
]

#: The 4K sensor mode's width, which is also how a streamed frame says
#: which mode the camera is in (``capture.resolution`` uses the same 0/1).
_UHD_WIDTH_PX = 3840

#: The sample list's "View" column — the frame the sample was found in. A
#: table COLUMN rather than a button per row: the table is rebuilt on every
#: tile result, so a widget per row would be built and thrown away hundreds
#: of times in a run.
_VIEW_COLUMN = 5

#: How long a run waits for the camera's sensor-mode switch before giving up
#: on it. The switch re-inits the stream and normally lands in well under a
#: second; a camera that never answers would otherwise leave the panel saying
#: "Switching…" with the lock held and no run — forever, silently.
_CAMERA_SWITCH_TIMEOUT_MS = 8000


def _path_kind(toggle_value: str) -> str:
    """The stored ``scan.path`` for the one path toggle.

    "One-way" is the serpentine cells with the direction rule turned off,
    so the CV layer still spells it as ``path='serpentine'``.
    """
    value = str(toggle_value or SERPENTINE).lower()
    return SERPENTINE if value == ONE_WAY else value


def _path_toggle_value(path: str, serpentine: bool) -> str:
    """The toggle's value from the two stored keys — the inverse, and the
    reason a stored configuration needs no migration."""
    kind = str(path or SERPENTINE).lower()
    if kind == SERPENTINE and not serpentine:
        return ONE_WAY
    return kind if kind in PATHS else SERPENTINE


def _same_candidates(a, b) -> bool:
    """Do two candidate lists describe the same samples?

    Position and size only: those are what the operator sees in the table
    (and what the live feed recomputes), so a match means the table would be
    rebuilt into exactly what it already says.
    """
    if len(a) != len(b):
        return False
    return all((x.x_px, x.y_px, x.area_px2, x.area_um2)
               == (y.x_px, y.y_px, y.area_px2, y.area_um2)
               for x, y in zip(a, b))


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
    #: "Store here as the origin" — the workspace passes it on to the main
    #: window, which owns the shared origin and its settings write.
    sig_set_origin = Signal()
    #: Something that affects the plan changed (the workspace persists).
    sig_plan_changed = Signal()
    #: A run started or ended. Everything that must not be touched while it
    #: owns the hardware follows this one signal — the camera and CV groups,
    #: the frame a live sample is reviewed against, the device enable boxes.
    sig_busy_changed = Signal(bool)

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
        #: Where the run whose results are on the map started (see
        #: _plan_origin). None = nothing scanned, the plan follows the stage.
        self._origin: StagePosition | None = None
        #: The last verdict's text and tone, so the export summary can
        #: append to it rather than overwrite it with a success message.
        self._status_base = ""
        self._status_tone: str | None = None
        #: Set by the workspace: how many captured tiles have not been
        #: examined yet. The panel owns the files, so it has to know when
        #: the results are complete — but the queue belongs to the engine.
        self.pending_tiles_fn = None
        #: Set while a run is waiting for the camera to change sensor mode:
        #: (job_id, origin) — the run starts when that job lands.
        self._pending_scan_start: tuple[int, tuple[float, float]] | None = None
        #: The live mode to put back after a run that switched it.
        self._camera_mode_before_scan: int | None = None
        #: Guards the camera switch: see _CAMERA_SWITCH_TIMEOUT_MS.
        self._start_watchdog = QTimer(self)
        self._start_watchdog.setSingleShot(True)
        self._start_watchdog.setInterval(_CAMERA_SWITCH_TIMEOUT_MS)
        self._start_watchdog.timeout.connect(self._on_start_timeout)
        #: The run whose frames the View column reads. Decided when the run
        #: starts (a row can be reviewed while the table is still filling)
        #: and dropped with the results.
        self._run_dir: Path | None = None
        self._frame_paths: dict[int, Path] | None = None
        self._frame_cache: dict[int, object] = {}
        #: row -> the tile the row came from (-1 = the live feed), parallel
        #: to ``_candidates`` and in the same order.
        self._row_tiles: list[int] = []
        self._review: SampleReviewWindow | None = None
        #: Set by the workspace: how to get the LIVE frame a live row came
        #: from, and what to pre-process a reviewed frame with (None when
        #: pre-processing is off, which disables the review's toggle).
        self.live_frame_fn = None
        self.review_context_fn = None
        # The camera's completion relay (the same signals the snapshot
        # busy-gate uses) — the run is sequenced on them.
        manager.sig_job_done.connect(self._on_camera_job_done)
        manager.sig_job_failed.connect(self._on_camera_job_failed)

        self._build_ui()
        self._load_settings()
        self._refresh_fov_label()
        # The origin is shared with the Navigation tab: it can be set there
        # and used here, so the readout follows the state rather than the
        # buttons that happen to be on this panel.
        self._state.sig_stage_origin_changed.connect(
            lambda _p: self.refresh_origin())
        self.refresh_origin()

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
        # The run card and the samples are PINNED, top and bottom, the way
        # Quick Actions is pinned on the Navigation tab: one is what you
        # press, the other is what you read, and neither should be
        # somewhere you have to scroll to find. Only the settings between
        # them scroll.
        layout.addWidget(self._build_run_card())
        layout.addWidget(self._build_scroll(), 2)
        layout.addWidget(self._build_samples_group())

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
        """Where a run starts, and where the origin it can start from lives.

        The origin is here rather than in a card of its own because it is a
        scan setting like the area is: it answers "where does the next run
        begin", which is the question this row already asks twice (from the
        stage, from the origin). *Set origin* sits with the readout, not with
        the two actions that USE the origin — marking it is housekeeping, and
        putting it third in a row of motion buttons made it look like one.
        """
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

        row = QHBoxLayout()
        row.setSpacing(6)
        self.scan_origin_btn = QPushButton("Scan from origin")
        self.scan_origin_btn.setObjectName("qa_primary")
        self.scan_origin_btn.setToolTip(
            "Run the same scan anchored at the origin instead of at the\n"
            "stage. The first move goes to the first tile's centre — which\n"
            "in the corner origin modes is inset half a field of view, not\n"
            "the origin itself.")
        self.scan_origin_btn.clicked.connect(self.scan_from_origin)
        row.addWidget(self.scan_origin_btn, 2)
        self.go_origin_btn = QPushButton("Go to origin")
        self.go_origin_btn.setObjectName("qa")
        self.go_origin_btn.setToolTip("Move the stage to the origin, at the "
                                      "scan's speed")
        self.go_origin_btn.clicked.connect(self.go_to_origin)
        row.addWidget(self.go_origin_btn, 1)
        layout.addLayout(row)

        # The readout and *Set origin* share the rows' 2:1 split, so the
        # three buttons stack in one column: same width, same height, same
        # weight. The button was a "compact" one — the size meant for a
        # bare "…" — which made the smallest control in the card the one
        # that WRITES something, and left it out of line with *Abort* and
        # *Go to origin* above it.
        row = QHBoxLayout()
        row.setSpacing(6)
        self.origin_label = QLabel("origin: not set")
        self.origin_label.setObjectName("dim")
        self.origin_label.setWordWrap(True)
        row.addWidget(self.origin_label, 2)
        self.set_origin_btn = QPushButton("Set origin")
        self.set_origin_btn.setObjectName("qa")
        self.set_origin_btn.setToolTip(
            "Store the current XYR position as the origin. The Navigation\n"
            "tab sets and shows the same one.")
        self.set_origin_btn.clicked.connect(self.sig_set_origin.emit)
        row.addWidget(self.set_origin_btn, 1)
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
        # The samples are NOT in here: `_build_ui` pins them below the
        # scroll (a duplicate built here was never shown, and `self._table`
        # rebinding to the pinned one left it permanently stale).
        layout.addWidget(self._build_scan_group())
        layout.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    def _build_scan_group(self) -> QWidget:
        box = QFrame()
        box.setObjectName("card")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        # Six, not four: these rows are the ones the operator hits with a
        # gloved hand between scans, and the segmented rows are 20 px tall.
        layout.setSpacing(6)

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
                       self.start_axis, self.return_home):
            toggle.sig_changed.connect(self._on_field_changed)
        for widget in (self.width, self.height):
            widget.valueChanged.connect(self._on_field_changed)
            widget.editingFinished.connect(self._persist)
        return box

    def _build_samples_group(self) -> QWidget:
        """The found samples: a table, pinned below the settings scroll.

        A table rather than a stack of buttons. The buttons read like a
        menu of actions and only one of them was ever "selected", which is
        a state the eye could not see; a table row *is* the selection, the
        columns line up so coordinates can be compared at a glance, and
        the same rows are what the CSV export writes.
        """
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

        self._table = QTableWidget(0, _VIEW_COLUMN + 1)
        self._table.setHorizontalHeaderLabels(
            ["#", "X µm", "Y µm", "Area µm²", "Edge", "View"])
        self._table.setToolTip("Double-click a row to move the stage to that "
                               "sample; the View column opens the frame it "
                               "was found in")
        self._table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection)
        self._table.verticalHeader().setVisible(False)
        # Tall enough for about six samples, and no taller: this card is
        # pinned, so every pixel it takes comes off the settings above it.
        # More than six scroll inside the table.
        self._table.setMinimumHeight(110)
        self._table.setMaximumHeight(180)
        header = self._table.horizontalHeader()
        for column in range(_VIEW_COLUMN):
            header.setSectionResizeMode(column, header.ResizeMode.Stretch)
        header.setSectionResizeMode(_VIEW_COLUMN, header.ResizeMode.Fixed)
        self._table.setColumnWidth(_VIEW_COLUMN, 46)
        item = self._table.horizontalHeaderItem(_VIEW_COLUMN)
        if item is not None:
            item.setToolTip("Open the frame this sample was found in")
        self._table.itemSelectionChanged.connect(self._on_row_changed)
        self._table.cellClicked.connect(self._on_cell_clicked)
        self._table.doubleClicked.connect(self._on_double_clicked)
        layout.addWidget(self._table, 1)

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
        # Kept as an attribute: a run disables it (see _set_job) — Clear
        # while the table is filling would drop the samples already found.
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setObjectName("compact")
        self._clear_btn.setToolTip(
            "Forget this run: the sample list, the markers on the map and "
            "its mosaic. The next scan clears them too.")
        self._clear_btn.clicked.connect(self.clear_results)
        row.addWidget(self._go_to_btn)
        row.addWidget(export)
        row.addWidget(self._clear_btn)
        layout.addLayout(row)
        self._candidates: list[FlakeCandidate] = []
        self._selected = -1
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
        self.path.set_value(_path_toggle_value(str(saved["path"]),
                                               bool(saved["serpentine"])))
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
            "path": _path_kind(self.path.value()),
            "serpentine": str(self.path.value()) != ONE_WAY,
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
            serpentine=str(self.path.value()) != ONE_WAY,
            speed_pps=float(self._prefs["speed_pps"]),
            path=_path_kind(self.path.value()),
            origin=str(self.origin.value()),
            start_axis=str(self.start_axis.value()),
            x_dir=int(self.x_dir.value() or 1),
            y_dir=int(self.y_dir.value() or 1),
            settle_ms=int(self._prefs["settle_ms"]),
            backlash_um=float(self._prefs["backlash_um"]),
            backlash_approach=int(self._prefs["backlash_approach"]),
            return_to_start=bool(self.return_home.value()))

    def _plan_origin(self) -> StagePosition:
        """Where the plan is anchored: the run's origin while there is one,
        the stage otherwise.

        Normally the stage — the map answers "what would a scan from here
        cover?". Once a run exists, the answer is where that run was
        ANCHORED, and that covers three moments with one rule:

        - **while it runs**: the plan is the one being walked, which for a
          *scan from origin* is the origin and not the stage (the stage is
          wherever the last waypoint left it);
        - **after it ends**: the stage has returned to the start or stayed
          at the last tile, and letting the plan follow it re-anchors the
          outline under a mosaic that is still being read — with a pulse of
          readback noise in the return position, that also changed the
          plan's identity and took the mosaic with it;
        - **before any run**: no origin has been latched, so the stage.

        The anchor is released when the results are cleared and re-latched
        by the next run. It used to be gated on there being TILES, which is
        false at the one moment a scan-from-origin needs it most: the start
        of the run, before the first capture.
        """
        if self._origin is not None:
            return self._origin
        return self.stage_position()

    def refresh_plan(self) -> None:
        """The map shows the plan the current fields would run — around the
        run's origin while its results are on screen (:meth:`_plan_origin`),
        and around the stage otherwise."""
        params = self.params_for(self._plan_origin())
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
        # Recorded unconditionally; whether it is DRAWN is the map's
        # decision (a run owns the box while it is capturing, and an idle
        # map with nothing scanned shows only the start dot) — and keeping
        # it fed during a run means the box is already current when the run
        # hands it back.
        self.map.set_footprint(pos.x_um, pos.y_um)

    # ------------------------------------------------------------------
    # results in
    # ------------------------------------------------------------------

    def on_tile_result(self, index: int, candidates) -> None:
        """Detection finished for a scan tile."""
        self._scan_hits[index] = list(candidates)
        if self._pending_export is not None:
            self._maybe_export()
        note = f"scan · {len(self._scan_hits)} tiles"
        if self._can_append(index):
            self._append_candidates(index, list(candidates), note)
        else:
            self._show_candidates(self._flatten_scan_hits(), note)
        self.refresh_markers()

    def _can_append(self, index: int) -> bool:
        """Is this tile's result an APPEND to what the table already shows?

        Tiles are examined in capture order (the detection queue is FIFO),
        so the usual case is: the table already lists every earlier tile and
        nothing else. Appending one tile's rows is what keeps the table's
        cost linear — rebuilding it is O(rows), once per tile, on the GUI
        thread, which is quadratic over a run.
        """
        if not self._row_tiles:
            return False                     # nothing shown yet (live rows)
        return all(tile >= 0 for tile in self._row_tiles) \
            and index > self._row_tiles[-1]

    def _append_candidates(self, index: int, candidates, note: str) -> None:
        """Add one tile's rows without touching the ones already there."""
        start = len(self._candidates)
        self._row_tiles.extend([int(index)] * len(candidates))
        self._candidates.extend(candidates)
        rows = candidate_rows(candidates)
        self._table.blockSignals(True)
        self._table.setRowCount(len(self._candidates))
        for offset, cells in enumerate(rows):
            row = start + offset
            cells = list(cells)
            cells[0] = str(row + 1)          # the row's own number
            for column, text in enumerate(cells):
                self._table.setItem(row, column, QTableWidgetItem(text))
            view = QTableWidgetItem("View")
            view.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            view.setToolTip("Show the frame this sample was found in")
            self._table.setItem(row, _VIEW_COLUMN, view)
        self._table.blockSignals(False)
        count = len(self._candidates)
        self._samples_note.setText(
            f"{count} sample(s) from the {note}." if count
            else f"Nothing matched in the {note}.")
        self._go_to_btn.setEnabled(self._selected >= 0 and self._job is None)

    def show_live_candidates(self, candidates) -> None:
        """Detection finished for the live frame.

        Nothing is rebuilt when the list says what the table already says:
        this arrives at the preview's cadence (~7 Hz), and on a static scene
        the answer does not change between frames.
        """
        fresh = list(candidates)
        if self._job is None and not self._scan_hits:
            if fresh and self._candidates and _same_candidates(
                    fresh, self._candidates):
                return
            self._live_candidates = fresh
            self._show_candidates([(-1, cand) for cand in fresh], "live view")

    def _flatten_scan_hits(self) -> list[tuple[int, FlakeCandidate]]:
        """(tile index, candidate) for every found sample, in table order.

        The tile index travels WITH the candidate rather than living on it:
        identification is deliberately frame-agnostic (it never sees which
        frame it ran on), and the panic index is a property of the feed, not
        of the flake. Row N of the table is element N here — which is also
        the numbering the annotated mosaic rings by."""
        out: list[tuple[int, FlakeCandidate]] = []
        for index in sorted(self._scan_hits):
            out.extend((index, cand) for cand in self._scan_hits[index])
        return out

    def _show_candidates(self, pairs, source: str) -> None:
        """Rebuild the table. The rows come from the same formatter the CSV
        export uses, so what is exported is what was shown.

        The operator's chosen row SURVIVES the rebuild when it still exists:
        the table used to clear its selection on every result (and the live
        feed sends one ~7 times a second), so a row picked to inspect — or to
        *go to* — silently disappeared under the cursor.
        """
        keep = self._selected
        self._row_tiles = [int(tile) for tile, _cand in pairs]
        self._candidates = [cand for _tile, cand in pairs]
        rows = candidate_rows(self._candidates)
        self._table.blockSignals(True)
        self._table.setRowCount(len(rows))
        for row, cells in enumerate(rows):
            for column, text in enumerate(cells):
                self._table.setItem(row, column, QTableWidgetItem(text))
            view = QTableWidgetItem("View")
            view.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            view.setToolTip("Show the frame this sample was found in")
            self._table.setItem(row, _VIEW_COLUMN, view)
        self._table.clearSelection()
        if 0 <= keep < len(self._candidates):
            self._selected = keep
            self._table.selectRow(keep)
        else:
            self._selected = -1
        self._table.blockSignals(False)
        count = len(self._candidates)
        self._samples_note.setText(
            f"{count} sample(s) from the {source}." if count
            else f"Nothing matched in the {source}.")
        # an enabled button that only ever answers "select a sample first"
        # is a button that lies — so it follows the SELECTION, not the count
        self._go_to_btn.setEnabled(self._selected >= 0 and self._job is None)

    def _on_cell_clicked(self, row: int, column: int) -> None:
        """The View column opens the sample's frame."""
        if column == _VIEW_COLUMN:
            self._review_row(int(row))

    def _on_double_clicked(self, index) -> None:
        """Double-click keeps meaning *go to* — except on the View column,
        where it means the same as a single click there (an operator who
        double-clicks a button expects it to do the one thing it says)."""
        if index.column() == _VIEW_COLUMN:
            self._review_row(int(index.row()))
        else:
            self._on_go_to()

    # ------------------------------------------------------------------
    # the frame behind a row
    # ------------------------------------------------------------------

    def _review_row(self, row: int, retry: bool = True) -> None:
        """Show the frame this row's sample was found in."""
        if not 0 <= row < len(self._candidates):
            return
        window = self._review_window()
        window.set_source(lambda r: self._sample_shot(r, retry=retry),
                          len(self._candidates), self._review_apply())
        window.show_sample(row)
        self._show_review(window)

    def _review_window(self) -> SampleReviewWindow:
        if self._review is None:
            self._review = SampleReviewWindow(self._input, self.window())
        return self._review

    def _show_review(self, window: SampleReviewWindow) -> None:
        window.show()
        window.raise_()
        window.activateWindow()

    def _sample_shot(self, row: int, retry: bool = True) -> SampleShot:
        """What the review window shows for one row.

        The pixels are read at their own resolution: the tile's PNG for a
        scan row, the live frame for a live one. No resampling anywhere —
        the candidate's ``x_px/y_px`` are pixels of exactly this frame, so
        the ring lands on the flake identification measured.
        """
        if not 0 <= row < len(self._candidates):
            return SampleShot(error="That row is gone.")
        cand = self._candidates[row]
        tile = self._row_tiles[row] if row < len(self._row_tiles) else -1
        shot = SampleShot(
            number=row + 1, tile=tile,
            source=(self._run_dir.name if tile >= 0 and self._run_dir
                    else "live view"),
            x_um=float(cand.x_um), y_um=float(cand.y_um),
            area_um2=float(cand.area_um2), score=float(cand.score),
            x_px=float(cand.x_px), y_px=float(cand.y_px),
            area_px2=float(cand.area_px2))
        if tile < 0:
            frame = self.live_frame_fn() if self.live_frame_fn else None
            shot.frame = frame
            if frame is None:
                shot.error = ("The live frame is not available yet — the "
                              "view-mode layer it lives in has not been "
                              "computed.")
            return shot
        frame, why = self._tile_frame(tile)
        shot.frame = frame
        if frame is None:
            shot.error = why
            if self._job == "scan" and retry:
                # The writer thread can be a tile behind the detector, so a
                # row can exist a moment before its frame is on disk. One
                # re-read, then the operator decides.
                QTimer.singleShot(250, lambda: self._review_retry(row))
        return shot

    def _review_retry(self, row: int) -> None:
        """One re-read of a frame the writer had not flushed yet."""
        if self._review is None or not self._review.isVisible():
            return
        if row != self._review.current_row() or not 0 <= row < len(self._candidates):
            return
        self._review.show_sample(row)

    def _tile_frame(self, tile: int):
        """``(frame, why not)`` — the tile's captured frame as RGB.

        The two ways to have no frame are told apart, because they mean
        different things to the operator: *the run has not written this tile
        yet* (wait) and *it never will* (this row's frame is not coming).
        """
        if tile in self._frame_cache:
            return self._frame_cache[tile], ""
        if self._run_dir is None:
            return None, "No run to read this frame from."
        path = self._frame_paths_for_run().get(int(tile))
        if path is None:
            return None, ("This frame is still being written — trying again…"
                          if self._job == "scan" else
                          "The manifest has no frame for this tile — it was "
                          "recorded as a miss.")
        image = cv2.imread(str(path))
        if image is None:
            return None, f"The frame file could not be read ({path.name})."
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Three frames is what stepping through the list needs; a full-plan
        # cache would be gigabytes at 4K.
        self._frame_cache[tile] = frame
        while len(self._frame_cache) > 3:
            self._frame_cache.pop(next(iter(self._frame_cache)))
        return frame, ""

    def _frame_paths_for_run(self) -> dict[int, Path]:
        """waypoint index -> frame path, from the run's own manifest.

        Re-read while a run is filling it (the manifest is appended and
        flushed per tile, so a reader sees exactly the tiles written so far)
        and cached once the run is over."""
        if self._frame_paths is not None and self._job is None:
            return self._frame_paths
        manifest = (self._run_dir / "manifest.csv") if self._run_dir else None
        paths: dict[int, Path] = {}
        if manifest is not None and manifest.exists():
            try:
                paths = dict(frames_from_manifest(manifest, self._run_dir))
            except (OSError, ValueError):
                paths = {}
        if self._job is None:
            self._frame_paths = paths
        return paths

    def _review_apply(self):
        """The pre-processing function for the review's toggle, or None.

        None (which disables the toggle) whenever the chain is off: there is
        nothing to compare against then, and the checkbox would be a control
        that cannot change anything."""
        context = self.review_context_fn() if self.review_context_fn else None
        if context is None:
            return None
        from talos.cv import preprocess as pre

        cfg, colour = context
        return lambda frame: pre.apply(frame, cfg, colour)

    def _forget_run(self) -> None:
        """Drop the run the View column reads from."""
        self._run_dir = None
        self._frame_paths = None
        self._frame_cache.clear()
        self._row_tiles = []
        if self._review is not None:
            self._review.hide()

    def _on_row_changed(self) -> None:
        """Follow the SELECTION, not the current index.

        ``currentRow()`` outlives a ``clearSelection()`` — the current
        index stays where it was — so reading it here left "Go to" enabled
        after the table had visibly emptied, pointing at a row nothing was
        highlighting.
        """
        rows = self._table.selectionModel().selectedRows()
        self._selected = rows[0].row() if rows else -1
        self._go_to_btn.setEnabled(self._selected >= 0 and self._job is None)

    def _select(self, index: int) -> None:
        """Select a row from elsewhere (the map's markers)."""
        if not 0 <= index < len(self._candidates):
            return
        self._table.selectRow(index)

    def _on_marker_selected(self, index: int) -> None:
        self._select(index)

    def refresh_markers(self) -> None:
        candidates = [cand for _tile, cand in self._flatten_scan_hits()]
        self.map.set_markers(
            [ScanMapMarker(x_um=c.x_um, y_um=c.y_um, label=str(index + 1))
             for index, c in enumerate(candidates)])
        # nothing to say when there is nothing: "0 tiles · 0 sample(s)" is
        # a caption about an empty map
        self.map.set_caption(
            f"{len(self._scan_tiles)} tiles · {len(candidates)} sample(s)"
            if self._scan_tiles or candidates else "")

    def clear_results(self) -> None:
        """Forget the run: the table, the markers AND the mosaic.

        The mosaic is the operator's own act here, deliberately. A finished
        run's tiles stay on the map until this or the next run, because
        after a scan the map is what they read samples off — and that is
        also why this button, which used to leave the tiles behind, clears
        them: it is the one place that means "I am done with this run".
        """
        if self._job is not None:
            # Also reachable from a camera flip and from the shortcut above;
            # a refusal here covers every caller. Detection never revisits a
            # tile, so a sample dropped this way is gone for good.
            self.set_status("A run is filling this list — Clear would drop "
                            "the samples it has already found.", tone="warn")
            return
        self._scan_hits.clear()
        self._scan_tiles.clear()
        self._origin = None
        self._forget_run()
        self._live_candidates = []
        self._candidates = []
        self._selected = -1
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        self._table.clearSelection()
        self._table.blockSignals(False)
        self.map.clear_tiles()
        self.refresh_markers()          # drops the markers and the caption
        self._samples_note.setText("Results cleared.")
        self._go_to_btn.setEnabled(False)
        # the plan follows the stage again now that nothing is anchored
        self.refresh_plan()

    # ------------------------------------------------------------------
    # the run
    # ------------------------------------------------------------------

    def is_scanning(self) -> bool:
        return self._job is not None

    def _on_scan(self) -> None:
        """Scan from where the stage is standing."""
        self._start_scan(self.stage_position())

    def scan_from_origin(self) -> None:
        """Run the same scan, anchored at the stored origin.

        The anchor is where the AREA starts, not where the stage goes: the
        run's first move is to the first tile's centre, which in the corner
        origin modes is inset by half a field of view from the corner the
        operator marked. That is the point of the mode — the origin is the
        corner of the area, not the middle of the first frame.
        """
        origin = getattr(self._state, "stage_origin", None)
        if origin is None:
            self.set_status("No stage origin set — mark one first.",
                            tone="warn")
            return
        if self._start_scan(origin):
            self.set_status(f"Scanning from the origin "
                            f"({origin.x_um:.1f}, {origin.y_um:.1f} µm) — "
                            f"Esc or Abort stops it")

    def go_to_origin(self) -> None:
        origin = getattr(self._state, "stage_origin", None)
        if origin is None:
            self.set_status("No stage origin set — mark one first.",
                            tone="warn")
            return
        self._go_to(origin.x_um, origin.y_um, "the origin")

    def refresh_origin(self) -> None:
        """The origin readout, and which of the origin actions can run."""
        origin = getattr(self._state, "stage_origin", None)
        if origin is None:
            self.origin_label.setText("origin: not set")
        else:
            self.origin_label.setText(f"origin: {origin.x_um:.1f}, "
                                      f"{origin.y_um:.1f} µm")
        usable = origin is not None and self._job is None
        self.go_origin_btn.setEnabled(usable)
        self.scan_origin_btn.setEnabled(usable)

    def _start_scan(self, position: StagePosition) -> bool:
        """Begin a run anchored at ``position``. False when it was refused
        (the caller says which anchor it was for)."""
        if self._job is not None:
            self.set_status("A run is already going — stop it first.",
                            tone="warn")
            return False
        if self._state.mode != "MANUAL":
            QMessageBox.information(
                self, "Scan",
                f"The stage is in use ({self._state.mode}) — a scan cannot "
                "start while another job owns the axes.")
            return False
        if not self._manager.last_position.get("zolix"):
            QMessageBox.information(
                self, "Scan", "No stage position yet — the Zolix controller "
                "has not reported. Check the connection.")
            return False
        # The tiles are captured at the live resolution unless Preferences
        # → Scan asks for another one; switching stops the stream for a
        # pipeline re-init, so it happens ONCE for the run, before the
        # first move, and is put back when the run ends.
        mode = self._camera_mode_to_switch_to()
        if mode is None:
            self._begin_scan(position)
            return True
        # The lock and the mode are taken HERE, not in _begin_scan: waiting
        # for the camera is part of the run (no second scan, no manual
        # control, and STOP ALL must cancel the start rather than be
        # forgotten), and the camera job is already on its way.
        self._set_job("scan")
        self._state.set_mode("SCAN")
        self.set_status("Switching the camera for the scan…")
        job = self._manager.submit_camera("set_property", "resolution", mode)
        if int(job) < 0:
            self._log("the camera cannot change resolution — scanning at "
                      "the live one")
            self._begin_scan(position)
            return True
        self._camera_mode_before_scan = self._current_camera_mode()
        self._pending_scan_start = (int(job), position)
        self._start_watchdog.start()
        return True

    def _begin_scan(self, position: StagePosition) -> None:
        """Everything a run does once the camera is in the right mode."""
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
        # The new run's origin, latched BEFORE the plan is rebuilt: this is
        # the moment the results are let go, so the plan re-anchors here.
        self._origin = position
        # A pending export is NOT dropped: its payload carries its own copy
        # of the results, and a run started before the detection queue
        # drained would otherwise never have its mosaic or CSV written.
        self._forget_run()                     # the PREVIOUS run's frames
        # ... and the new run's folder, decided HERE rather than on the scan
        # thread: a sample can be reviewed while the run is still filling the
        # table, and the reader of that frame needs to know where frames go.
        self._run_dir = scan_directory(self._settings) / time.strftime(
            "scan_%Y%m%d_%H%M%S")
        self.refresh_plan()
        self.map.clear_tiles()                 # the PREVIOUS run's tiles
        self.map.clear_footprint()
        self._set_job("scan")
        self._state.set_mode("SCAN")
        self._scan_started_at = time.monotonic()
        self.progress.setValue(0)
        self.set_status("Scanning — Esc or Abort stops it")
        out_dir = self._run_dir
        # Read from the widgets HERE, on the GUI thread, and hand the values
        # to the worker thread as arguments.
        params = self.params_for(position)
        fov = self.fov()
        frame_shape = (tuple(self._latest_frame.shape)
                       if self._latest_frame is not None else None)
        self._scan_worker = _Worker(
            lambda: self._run_scan(params, fov, out_dir, frame_shape), self)
        self._scan_worker.sig_log.connect(self._log)
        self._scan_worker.sig_done.connect(self._on_scan_done)
        # deleteLater AND drop the reference: keeping a Python handle to a
        # deleted C++ object makes `isRunning()` raise at teardown, which
        # is exactly when shutdown() wants to ask.
        self._scan_worker.finished.connect(self._scan_worker.deleteLater)
        self._scan_worker.finished.connect(self._forget_scan_worker)
        self._scan_worker.start()

    # --- the camera's sensor mode -------------------------------------

    def _current_camera_mode(self) -> int | None:
        """0 = 4K, 1 = 1080p, None when nothing has streamed yet."""
        frame = self._latest_frame
        if frame is None or len(frame.shape) < 2:
            return None
        return 0 if int(frame.shape[1]) >= _UHD_WIDTH_PX else 1

    def _camera_mode_to_switch_to(self) -> int | None:
        """The mode this run needs, or None when it needs no switch."""
        current = self._current_camera_mode()
        if current is None:
            return None
        try:
            want = 0 if int(self._prefs.get("resolution", 1)) == 0 else 1
        except (TypeError, ValueError):
            return None
        return None if want == current else want

    def _on_camera_job_done(self, job_id: int, _result) -> None:
        pending = self._pending_scan_start
        if pending is None or int(job_id) != pending[0]:
            return
        self._pending_scan_start = None
        self._start_watchdog.stop()
        if self._scan_abort.is_set():
            # Esc or STOP ALL landed during the switch: the operator asked
            # for everything to stop, and starting a run that crosses the
            # whole area would be the opposite of that.
            self._cancel_pending_start("aborted while the camera switched")
            return
        self._begin_scan(pending[1])

    def _on_camera_job_failed(self, job_id: int, exc_type: str,
                              message: str) -> None:
        pending = self._pending_scan_start
        if pending is None or int(job_id) != pending[0]:
            return
        self._pending_scan_start = None
        self._start_watchdog.stop()
        self._camera_mode_before_scan = None
        if self._scan_abort.is_set():
            self._cancel_pending_start("aborted while the camera switched")
            return
        self._log(f"camera resolution switch refused ({exc_type}: {message}) "
                  f"— the run will capture at the live resolution")
        self._begin_scan(pending[1])

    def _on_start_timeout(self) -> None:
        """The camera never answered the switch: do not start the run.

        Starting anyway would be worse than not starting — the switch job is
        still out there, so the camera could change mode *during* the run and
        the tiles would silently stop matching each other.
        """
        if self._pending_scan_start is None:
            return
        self._log("the camera did not answer the resolution switch")
        self._cancel_pending_start("the camera did not answer")

    def _cancel_pending_start(self, reason: str) -> None:
        """Give up on a run whose camera switch never landed."""
        self._pending_scan_start = None
        self._start_watchdog.stop()
        if self._state.mode == "SCAN":
            self._state.set_mode("MANUAL")
        self._set_job(None)
        # The switch is queued ahead of this one, so the live mode wins —
        # and a camera left in the scan's mode would outlive the session.
        self._restore_camera_mode()
        self.set_status(f"Scan not started — {reason}.", tone="warn")
        self._log(f"scan not started: {reason}")

    def _restore_camera_mode(self) -> None:
        """Put the live stream back the way the operator had it. Best
        effort: a camera that will not switch back leaves the live view at
        the scan's resolution for this session, and the next connect
        re-applies the configured live resolution anyway."""
        mode = self._camera_mode_before_scan
        self._camera_mode_before_scan = None
        if mode is None:
            return
        job = self._manager.submit_camera("set_property", "resolution", mode)
        if int(job) < 0:
            self._log("the camera did not switch back after the scan — "
                      "reconnect it to restore the live resolution")

    def _forget_scan_worker(self) -> None:
        self._scan_worker = None

    def _run_scan(self, params: ScanParams, fov, out_dir: Path,
                  frame_shape):
        """The scan thread. Owns the adapter and the scanner; every UI
        update comes back as a queued signal.

        Everything a WIDGET holds is read before this thread starts (see
        ``_begin_scan``) and arrives as arguments: ``params_for`` reads eight
        spin boxes and a combo, and Qt widgets belong to the GUI thread.
        """
        from talos.hal.proxies.stage_adapter import ManagerStageAdapter

        stage_cfg = scan_speed_config(self._settings.device("zolix"),
                                      float(self._prefs["speed_pps"]))
        adapter = ManagerStageAdapter(self._manager, stage_cfg,
                                      abort_check=self._scan_abort.is_set)
        slot = getattr(self._manager, "frame_slot", None)
        source = LatestFrameSource(slot) if slot is not None else None
        if source is None:
            self._log("no frame slot — the scan will record positions only")
        # pending_tiles_fn is the pacing valve: tiles are never dropped, so
        # without it a scan whose tiles are identified more slowly than they
        # are captured queues whole frames (6 MB at 1080p, 25 MB at 4K)
        # until the process runs out of memory. It has to be passed HERE —
        # the scanner's default is "no pacing", which is right for the CLI
        # benches and wrong for the app.
        scanner = GridScanner(adapter, source, thumb_width=192,
                              pending_tiles_fn=self.pending_tiles_fn,
                              max_pending_tiles=MAX_PENDING_TILES)
        scanner.sig_progress.connect(self._on_progress)
        scanner.sig_tile.connect(self._on_tile)
        scanner.sig_frame.connect(self._on_scan_frame)
        scanner.sig_log.connect(self._log)
        self._scanner = scanner
        if self._scan_abort.is_set():
            # STOP ALL landed while this thread was still starting up, so
            # nothing has told the SCANNER yet. The adapter already refuses
            # to move (its abort check is the same Event), but the run would
            # then end as "stopped early — scan aborted", which is the
            # operator's own abort reported as a fault.
            scanner.request_abort()
        try:
            result = scanner.run(
                params, out_dir,
                meta={"fov_um": fov, "objective_id": self._state.objective,
                      "frame_shape": frame_shape})
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
        # The box follows the tile being captured: the newest frame in the
        # mosaic is what "where the scan is now" means, and it is a fact
        # already in hand rather than a position read from telemetry that
        # the scan's own jobs are crowding out.
        self.map.set_active_tile(x_um, y_um)
        self.refresh_markers()

    def _on_scan_frame(self, index: int, x_um: float, y_um: float,
                       frame) -> None:
        self._scan_tiles[index] = (x_um, y_um)
        self.sig_tile_captured.emit(index, float(x_um), float(y_um), frame)

    def _on_scan_done(self, payload) -> None:
        """Report what actually happened — three outcomes, not two.

        A run can end because the operator aborted it, because it finished,
        or because something FAILED mid-way (a move that timed out, a soft
        limit, a serial glitch). The third used to be reported as "Scan
        done": the fault was carried in ``result.message`` and the message
        was never shown, so a scan that died at tile 3 of 100 looked like a
        complete one, and the operator had no way to tell. It now says what
        stopped it, where, and why.
        """
        if self._state.mode == "SCAN":
            self._state.set_mode("MANUAL")
        self._set_job(None)
        self.progress.setVisible(False)
        # The run's claim on the map's box ends here; the live position
        # takes it back on the next telemetry sample.
        self.map.set_active_tile(None)
        # The single path for finish, abort AND fault: whatever the run
        # did to the camera's sensor mode, the live view goes back to the
        # operator's resolution here.
        self._restore_camera_mode()
        if not payload:
            self.set_status("Scan failed before it started (see the log)")
            return
        result = payload["result"]
        # Where the run's wall-clock actually went — the number that says
        # whether the next speed-up is a motion parameter or a camera one.
        timing = getattr(getattr(result, "timing", None), "summary", "")
        if timing:
            self._log(f"scan timing: {timing}")
        aborted = result.aborted or self._scan_abort.is_set()
        missing = getattr(result, "missing", 0)
        planned = getattr(result, "planned", len(result.frames))
        visited = getattr(result, "visited", len(result.frames))
        note = f" · {missing} tile(s) had no frame" if missing else ""
        if aborted:
            text = (f"Scan aborted at tile {visited} of {planned} · "
                    f"{len(result.frames)} frame(s){note}")
            self.set_status(text, tone="warn")
            self._log(f"scan aborted → {payload['out_dir']}")
        elif getattr(result, "stopped_early", False):
            reason = str(result.message or "the stage stopped responding")
            text = (f"Scan STOPPED at tile {visited} of {planned} — {reason} · "
                    f"{len(result.frames)} frame(s) are on disk{note}")
            self.set_status(text, tone="error")
            self._log(f"scan stopped early at tile {visited} of {planned}: "
                      f"{reason} → {payload['out_dir']}")
        elif missing >= planned and planned > 0:
            # The stage walked the whole area and captured nothing: the
            # camera was not streaming (ZEN open, camera unplugged, the
            # frame slot never filled). The geometry is recorded and the
            # scan did finish — but "done" alone would send the operator
            # looking for images that were never taken.
            self.set_status(
                f"Scan finished but captured NO frames — is the camera "
                f"streaming? ({planned} tiles recorded in the manifest)",
                tone="error")
            self._log("scan captured no frames: the frame slot was empty "
                      "for every tile")
        else:
            text = f"Scan done: {len(result.frames)} frame(s){note}"
            self.set_status(text, tone="warn" if missing else None)
            self._log(f"scan done → {payload['out_dir']}")
        # The results are copied HERE, while they are still this run's:
        # detection drains behind the capture, and a new run clears the
        # dictionaries — an export that read them later wrote an empty
        # candidates.csv over a run that had found samples.
        payload["hits"] = {index: list(items)
                           for index, items in self._scan_hits.items()}
        payload["tiles"] = dict(self._scan_tiles)
        payload["flip"] = self._flip
        payload["exports"] = {
            "mosaic": bool(self._prefs["export_mosaic"]),
            "candidates": bool(self._prefs["export_candidates"]),
            "annotated": bool(self._prefs["export_annotated"])}
        self._pending_export = payload
        self._maybe_export()

    def _abort_run(self, reason: str) -> None:
        """Flag the run, stop the axis it owns, drop the holds.

        The stop is the ADAPTER's (one priority Zolix job), deliberately not
        ``manager.stop_all()``: that is the caller's business, and doing it
        here is what made stop-all recurse (see ``on_stop_all_done``).
        Once per run — a second abort contributes nothing but traffic on a
        link that is already the flaky part of this bench.
        """
        first = not self._scan_abort.is_set()
        self._scan_abort.set()
        scanner = self._scanner
        if scanner is not None and first:
            scanner.request_abort()
        if self._input is not None:
            self._input.cancel_all_holds(reason)

    def _on_abort(self, reason: str = "abort") -> None:
        """Cooperative abort — never QThread.terminate() (it killed the
        scan thread mid-serial-write once, leaving the port open). The
        operator asked for a stop, so the whole stage gets one."""
        self._abort_run(reason)
        self._manager.stop_all()
        self.set_status("Aborting…")

    def on_stop_all_done(self) -> None:
        """STOP ALL must abort the RUN, not just the motion in flight.

        Without this the scan's abort flag stayed clear, the controller
        stopped, and the run cheerfully continued at the next waypoint —
        Esc looked like it had not worked at all.

        It aborts **without asking for another stop**, and that is not a
        detail: this slot runs when a stop_all COMPLETES, so an abort that
        stopped everything again fired this signal again — a loop that
        re-stopped the already-stopped stage several times a second for as
        long as the run took to unwind. On the bench it was reported as a
        scan that "aborted halfway and then looped forever": the loop was
        the visible symptom, and its fuel was the run still unwinding (a
        move in flight, a capture window) while every iteration fed it
        another round of stop jobs.
        """
        if self._job == "scan":
            self._abort_run("stop all")
            self.set_status("Aborting…")

    def _set_job(self, job: str | None) -> None:
        self._job = job
        busy = job is not None
        self.scan_btn.setEnabled(not busy)
        self.abort_btn.setEnabled(busy)
        self.progress.setVisible(busy)
        self._go_to_btn.setEnabled(not busy and bool(self._candidates))
        # Everything a run would fight over is disabled — except *Set
        # origin*, which only reads the position and is worth having
        # mid-run (marking the spot the run started from), and the two
        # origin actions, whose enable state refresh_origin owns.
        for widget in (self.origin, self.x_dir, self.y_dir, self.path,
                       self.start_axis, self.return_home,
                       self.width, self.height, self._dir, self._clear_btn):
            widget.setEnabled(not busy)
        self.refresh_origin()
        if not busy:
            self._refresh_fov_label()
        self.sig_busy_changed.emit(busy)

    # ------------------------------------------------------------------
    # what a finished scan writes
    # ------------------------------------------------------------------

    def _maybe_export(self) -> None:
        """Wait for the detection queue to drain, then write the extras.

        Detection outlives the capture by design (it must not slow the
        stage down), so "the scan finished" is not the same moment as "the
        results are in" — and the extras are computed here rather than at
        the end of the run.

        The payload's own copy of the results is made when the run ends
        (``_on_scan_done``), not here: a run that is started while this one
        is still draining clears the live dictionaries, and reading them at
        export time then wrote an EMPTY candidates.csv over a run that
        found samples.
        """
        if self._pending_export is None or self._export_worker is not None:
            return
        if self._hits_outstanding():
            return
        payload = self._pending_export
        self._pending_export = None
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
        """Append what was written — never replace the verdict.

        This used to say "Scan done → …" whatever had happened, a second
        after the run ended, so it *overwrote* the honest report with a
        success message. A scan that stopped at tile 5 of 12 said "Scan
        STOPPED — Frame too short" for about a second and then quietly
        became "Scan done". The exports are a footnote to the outcome, not
        a second opinion about it.

        And a second run can start while this one is writing (it reads every
        frame from disk, so it takes seconds): the status line then belongs
        to the NEW run, and the footnote goes to the log instead.
        """
        self._export_worker = None
        if summary:
            detail = ", ".join(summary["written"]) or "nothing extra"
            if self._job is not None:
                self._log(f"wrote {detail} to {summary['dir']}")
            else:
                base = self._status_base or "Scan finished"
                self.set_status(f"{base} · wrote {detail}", self._status_tone)
                self._log(f"wrote {detail} to {summary['dir']}")
        # The run whose export arrived while this one was writing.
        if self._pending_export is not None:
            self._maybe_export()

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
            self.set_status("Select a sample first.", tone="warn")
            return
        cand = self._candidates[self._selected]
        self._go_to(cand.x_um, cand.y_um, f"sample #{self._selected + 1}")

    def _go_to(self, x_um: float, y_um: float, what: str) -> None:
        """Move the stage so a point lands under the crosshair.

        No confirmation dialog: the move is short, the scan's own speed, and
        it is the whole point of the button — asking "shall I?" between the
        click and the motion is a click the operator pays every time for a
        mistake they can see on the map. Refusals say why in the status
        line, which is also where the move reports itself.
        """
        if self._job is not None:
            self.set_status("The stage is busy with a run — stop it first.",
                            tone="warn")
            return
        if self._state.mode != "MANUAL":
            self.set_status(
                f"The stage is in use ({self._state.mode}) — moves are "
                f"refused while a job owns the axes.", tone="warn")
            return
        position = self.stage_position()
        dx = x_um - position.x_um
        dy = y_um - position.y_um
        # The same speed the scan runs at, for the same reason: this is a
        # scan move, not a jog.
        #
        # POSITIONAL, all five of them: ``InstrumentManager.submit`` carries
        # its arguments as a tuple and accepts no keyword arguments, so
        # ``speed_pps=…`` raised a TypeError inside this slot and the move
        # never happened — the button looked dead. The signature is
        # ``move_rel_um(dx, dy, dr_deg, speed, speed_pps)``; the stage
        # adapter fills the same parameter the same way.
        self._manager.submit("zolix", "move_rel_um", dx, dy, None,
                             StageSpeed.SLOW, int(self._prefs["speed_pps"]))
        self.set_status(f"Moving to {what} (ΔX={dx:+.1f}, ΔY={dy:+.1f} µm)")

    # ------------------------------------------------------------------
    # small things
    # ------------------------------------------------------------------

    def set_status(self, text: str, tone: str | None = None) -> None:
        """The run card's one line. ``tone`` is ``"warn"`` or ``"error"``
        for a run that did not finish — a stopped scan must not read in the
        same grey as a finished one.

        The text and tone are remembered so a later line (the export
        summary) can append to the verdict instead of replacing it.
        """
        self.status.setText(text)
        self._status_base = text
        self._status_tone = tone
        wanted = tone or "dim"
        if self.status.objectName() != wanted:
            self.status.setObjectName(wanted)
            # Qt only re-evaluates the stylesheet when it is told the
            # widget changed; without this the colour never moves.
            style = self.status.style()
            style.unpolish(self.status)
            style.polish(self.status)

    def _log(self, message: str) -> None:
        self.sig_log.emit(str(message))

    def shutdown(self) -> None:
        """App teardown: END the run, then let the threads finish.

        The abort flag goes up FIRST. Without it the scan thread keeps
        walking the plan while the process tears down around it — the waits
        below expire, the QThread is destroyed while still running, and Qt
        aborts the process (with a truncated manifest as the parting gift).
        Guarded against a worker whose C++ object is already gone — a
        finished QThread deletes itself, and asking a deleted one whether
        it is running raises rather than answering.
        """
        self._start_watchdog.stop()
        self._scan_abort.set()
        scanner = self._scanner
        if scanner is not None:
            try:
                scanner.request_abort()
            except Exception:  # noqa: BLE001 - teardown never raises
                pass
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
