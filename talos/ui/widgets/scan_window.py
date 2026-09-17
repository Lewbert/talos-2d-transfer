"""ScanWindow: the grid scan and the sample-identification chain, in one
place, until they are proven enough to fold back into the workspace.

The window has a MAP, not a live view. The camera stream belongs to the
Sample Finding tab, which shows it once with a Live/Processed switch over
it — driving that one view beats a second copy of the same pixels, and it
keeps the picture the operator judges in the same place whatever the scan
console is doing. This window computes the processed frame and hands it
over (`sig_processed_frame`), and asks the tab to arm the colour dropper
when the operator wants to pick (`sig_pick_requested`).

Contracts it inherits from the other detail windows: Esc hides it AND
still issues the global STOP ALL (the MainWindow's shortcut cannot fire
while this window has focus), close HIDES rather than destroys, and the
toolbar/menu action stays in sync.

Two rules this window exists to keep:

- **A scan owns the axes.** Manual input is gated by ``AppState.mode``
  ("SCAN"), which every input source already funnels through, and STOP ALL
  aborts the run — not just the motion in flight (``sig_stop_all_done``
  is wired to the same abort path, which the workspace version never was).
- **Nothing here blocks.** Capture runs on the scan thread; identification
  runs on the detection thread and reads frames that were delivered
  anyway. The live stream, the autofocus and the map are never waiting on
  either.
"""

from __future__ import annotations

import threading
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from talos.cv.calibration import SENSOR_HEIGHT_PX, SENSOR_WIDTH_PX
from talos.cv.frame_source import LatestFrameSource
from talos.cv.identify import (IdentifyConfig, Stage, sample_hex, valid_hex)
from talos.cv.scan import (PATH_LABELS, PATHS, GridScanner, grid_shape,
                           plan_path, scan_speed_config)
from talos.cv.scan_output import candidate_rows, write_outputs
from talos.scan_settings import (SCAN_KEYS, load_scan_settings,
                                 save_scan_settings, scan_directory)
from talos.models import (FlakeCandidate, ObjectiveCalibration, ScanParams,
                          StagePosition)
from talos.ui.detect_engine import DetectionEngine
from talos.ui.widgets.scan_map import (ScanMapMarker, ScanMapPlan,
                                       ScanMapTile, ScanMapWidget)

_DIR_CHOICES = ((1, "+"), (-1, "−"))
_AXIS_CHOICES = (("x", "X first"), ("y", "Y first"))
_PREVIEW_CHOICES = ((1.0, "Full resolution"), (0.75, "75 %"),
                    (0.5, "50 %"), (0.35, "35 %"))

#: Parameter names as the operator reads them. Anything not listed falls
#: back to the field name with underscores turned into spaces.
_PARAM_LABELS = {
    "min_area_um2": "Min area (µm²)",
    "max_area_um2": "Max area (µm²)",
    "margin_px": "Margin (px)",
    "gap_px": "Join within (px)",
    "min_saturation": "Min saturation",
    "min_value": "Min brightness",
    "min_edge_strength": "Min edge",
}


def load_scan_settings(settings) -> dict:
    """The scan section, with the dataclass defaults for anything missing
    (pure — unit-testable)."""
    defaults = ScanParams(x0_um=0.0, y0_um=0.0, width_um=2000.0,
                          height_um=1000.0)
    section = settings.section("scan")
    out = {key: section.get(key) for key in SCAN_KEYS}
    out["width_um"] = float(out.get("width_um") or defaults.width_um)
    out["height_um"] = float(out.get("height_um") or defaults.height_um)
    out["overlap"] = float(out.get("overlap") if out.get("overlap") is not None
                           else defaults.overlap)
    return out


def save_scan_settings(settings, values: dict) -> None:
    section = settings.section("scan")
    for key, value in values.items():
        section[key] = value
    settings.save()


def _param_label(name: str) -> str:
    """A stage parameter's label, as the operator reads it."""
    return _PARAM_LABELS.get(name, name.replace("_", " ").capitalize())


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


class ScanWindow(QDialog):
    """The scan console: plan, run, watch, and what it found."""

    sig_settings_changed = Signal()
    #: The processed frame (or None) for the Sample Finding live view.
    sig_processed_frame = Signal(object)
    #: The operator wants the colour dropper armed on the live view.
    sig_pick_requested = Signal()

    def __init__(self, manager, settings, state, calibration_context=None,
                 input_system=None, autofocus_service=None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Scan")
        self._manager = manager
        self._settings = settings
        self._state = state
        self._calibration = calibration_context
        self._input = input_system
        self._autofocus = autofocus_service
        self._toggle_action = None

        self._latest_frame: np.ndarray | None = None
        self._job: str | None = None                 # "scan" | None
        self._scan_abort = threading.Event()
        self._scanner = None
        self._scan_worker: _Worker | None = None
        self._scan_started_at = 0.0
        self._scan_hits: dict[int, list[FlakeCandidate]] = {}
        self._scan_tiles: dict[int, tuple[float, float]] = {}
        self._pending_export: dict | None = None
        self._export_worker: _Worker | None = None
        self._live_candidates: list[FlakeCandidate] = []
        self._preview_wanted = False

        self._engine = DetectionEngine(parent=self)
        self._engine.sig_result.connect(self._on_detected)
        self._engine.sig_log.connect(self._log)
        self._engine.set_source(self._detect_source)

        self._build_ui()
        self._load_settings()
        self._refresh_fov_label()
        self._refresh_plan()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        outer = QSplitter(Qt.Orientation.Vertical)
        top = QSplitter(Qt.Orientation.Horizontal)

        # The MAP takes the whole top row: this window has no live view of
        # its own. It drives the Sample Finding tab's — one stream, one
        # place to look, and the Live/Processed switch lives there.
        self.map = ScanMapWidget()
        top.addWidget(self.map)
        outer.addWidget(top)

        bottom = QSplitter(Qt.Orientation.Horizontal)
        # The run controls are PINNED below the scrolling settings: an
        # Abort button that can scroll out of view is not an abort button.
        scan_column = QWidget()
        scan_layout = QVBoxLayout(scan_column)
        scan_layout.setContentsMargins(0, 0, 0, 0)
        scan_layout.setSpacing(6)
        scan_layout.addWidget(
            self._scroll_column([self._build_scan_group(),
                                 self._build_outputs_group()]), stretch=1)
        scan_layout.addWidget(self._build_run_card())
        bottom.addWidget(scan_column)
        bottom.addWidget(self._scroll_column([self._build_identify_group()]))
        bottom.addWidget(self._build_samples_group())
        bottom.setStretchFactor(0, 3)
        bottom.setStretchFactor(1, 4)
        bottom.setStretchFactor(2, 3)
        outer.addWidget(bottom)
        outer.setStretchFactor(0, 3)
        outer.setStretchFactor(1, 4)
        root.addWidget(outer)
        self.resize(1280, 900)

    @staticmethod
    def _scroll_column(widgets: list[QWidget]) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(6)
        for widget in widgets:
            layout.addWidget(widget)
        layout.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    # --- the scan column ------------------------------------------------

    def _build_scan_group(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        area = QFrame()
        area.setObjectName("card")
        form = QFormLayout(area)
        self._width = self._spin(100.0, 100000.0, 2000.0, " µm")
        self._height = self._spin(100.0, 100000.0, 1000.0, " µm")
        form.addRow("Area X", self._width)
        form.addRow("Area Y", self._height)
        self._x_dir = self._combo(_DIR_CHOICES)
        self._y_dir = self._combo(_DIR_CHOICES)
        form.addRow("X direction", self._x_dir)
        form.addRow("Y direction", self._y_dir)
        self._path = self._combo([(p, PATH_LABELS[p]) for p in PATHS])
        form.addRow("Path", self._path)
        self._serpentine = self._combo([(True, "Bi-directional"),
                                        (False, "Uni-directional")])
        form.addRow("Serpentine", self._serpentine)
        self._start_axis = self._combo(_AXIS_CHOICES)
        form.addRow("Start axis", self._start_axis)
        self._overlap = self._spin(0.0, 90.0, 10.0, " %")
        form.addRow("Overlap", self._overlap)
        self._speed = QSpinBox()
        self._speed.setRange(10, 20000)
        self._speed.setSingleStep(50)
        self._speed.setSuffix(" pps")
        self._speed.setToolTip(
            "The scan's own speed, in pulses per second. The controller\n"
            "ramps it in fixed-steps mode, so the manual jog speeds — and\n"
            "the objective's speed multiplier — do not apply to a scan.")
        form.addRow("Speed", self._speed)
        self._settle = QSpinBox()
        self._settle.setRange(0, 5000)
        self._settle.setSingleStep(50)
        self._settle.setSuffix(" ms")
        form.addRow("Settle", self._settle)
        self._backlash = self._spin(0.0, 100.0, 0.0, " µm")
        form.addRow("Backlash", self._backlash)
        self._approach = self._combo([(1, "from −"), (-1, "from +")])
        form.addRow("Approach", self._approach)
        self._return_home = QCheckBox("Return to the start when finished")
        form.addRow(self._return_home)
        layout.addWidget(area)

        # The field of view is READ from the objective's calibration, with no
        # manual override: two sources for the same number is a way for them
        # to disagree, and the calibration is the one the rest of the
        # application measures with (µm², "go to sample", the scale bar).
        # Correcting it belongs in Preferences → Objectives & Calibration,
        # where it also fixes everything else.
        fov = QFrame()
        fov.setObjectName("card")
        fov_layout = QVBoxLayout(fov)
        fov_layout.setContentsMargins(6, 6, 6, 6)
        fov_layout.setSpacing(2)
        self._fov_value = QLabel("")
        bold = self._fov_value.font()
        bold.setBold(True)
        self._fov_value.setFont(bold)
        fov_layout.addWidget(self._fov_value)
        self._fov_note = QLabel("")
        self._fov_note.setObjectName("dim")
        self._fov_note.setWordWrap(True)
        fov_layout.addWidget(self._fov_note)
        layout.addWidget(fov)

        self._preview = self._combo(_PREVIEW_CHOICES)
        preview_form = QFormLayout()
        preview_form.addRow("Preview", self._preview)
        layout.addLayout(preview_form)

        # any change re-plans the map and persists. Speed is not in the
        # first group: it changes how long a run takes, never where the
        # tiles are.
        for widget in (self._width, self._height, self._overlap,
                       self._settle, self._backlash):
            widget.valueChanged.connect(self._refresh_plan)
            widget.editingFinished.connect(self._persist)
        for widget in (self._x_dir, self._y_dir, self._path, self._serpentine,
                       self._start_axis, self._approach):
            widget.currentIndexChanged.connect(self._refresh_plan)
            widget.currentIndexChanged.connect(self._persist)
        self._speed.valueChanged.connect(self._persist)
        self._return_home.toggled.connect(self._persist)
        self._preview.currentIndexChanged.connect(self._persist)
        return box

    def _build_run_card(self) -> QWidget:
        """Scan / Abort / progress / status — always on screen."""
        run = QFrame()
        run.setObjectName("card")
        run_layout = QVBoxLayout(run)
        run_layout.setContentsMargins(6, 6, 6, 6)
        run_layout.setSpacing(4)
        row = QHBoxLayout()
        self._scan_btn = QPushButton("Scan from here")
        self._scan_btn.setObjectName("primary")
        self._scan_btn.clicked.connect(self._on_scan)
        row.addWidget(self._scan_btn, 2)
        self._abort_btn = QPushButton("■ Abort")
        self._abort_btn.setObjectName("danger")
        self._abort_btn.setEnabled(False)
        self._abort_btn.setToolTip("Stop the scan and the stage (Esc does the "
                                   "same from anywhere)")
        self._abort_btn.clicked.connect(lambda: self._on_abort("abort"))
        row.addWidget(self._abort_btn, 1)
        run_layout.addLayout(row)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        run_layout.addWidget(self._progress)
        self._status = QLabel("Idle")
        self._status.setObjectName("dim")
        self._status.setWordWrap(True)
        run_layout.addWidget(self._status)
        return run

    def _build_outputs_group(self) -> QWidget:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.addWidget(QLabel("Write with every scan"))
        self._export_mosaic = QCheckBox("Stitched mosaic (mosaic.png)")
        self._export_candidates = QCheckBox("Samples (candidates.csv)")
        self._export_overview = QCheckBox("Tile overview (overview.png)")
        for check in (self._export_mosaic, self._export_candidates,
                      self._export_overview):
            check.setChecked(True)
            check.toggled.connect(self._persist)
            layout.addWidget(check)
        note = QLabel("Raw frames and manifest.csv are always written.")
        note.setObjectName("dim")
        note.setWordWrap(True)
        layout.addWidget(note)
        return card

    # --- the identification column --------------------------------------

    def _build_identify_group(self) -> QWidget:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setSpacing(6)
        self._identify_host = QWidget()
        self._identify_host_layout = QVBoxLayout(self._identify_host)
        self._identify_host_layout.setContentsMargins(0, 0, 0, 0)
        self._identify_host_layout.setSpacing(6)
        layout.addWidget(self._identify_host)

        self._counts = QLabel("")
        self._counts.setObjectName("dim")
        self._counts.setWordWrap(True)
        layout.addWidget(self._counts)
        reset = QPushButton("Reset the chain")
        reset.setObjectName("compact")
        reset.clicked.connect(self._reset_identify)
        layout.addWidget(reset)
        self._build_stage_editors(IdentifyConfig())
        return card

    def _build_stage_editors(self, config: IdentifyConfig) -> None:
        while self._identify_host_layout.count():
            item = self._identify_host_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._stage_editors: dict[str, tuple[QWidget, object]] = {}
        for stage in config.stages:
            editor, reader = self._stage_editor(stage)
            self._stage_editors[stage.NAME] = (editor, reader)
            self._identify_host_layout.addWidget(editor)

    def _stage_editor(self, stage: Stage):
        """A checkbox plus one widget per parameter — data-driven from the
        stage itself, so a new stage needs no UI code."""
        card = QFrame()
        card.setObjectName("card")
        outer = QVBoxLayout(card)
        outer.setContentsMargins(6, 4, 6, 6)
        outer.setSpacing(2)
        enable = QCheckBox(stage.LABEL)
        enable.setChecked(bool(stage.enabled))
        font = enable.font()
        font.setBold(True)      # the stage name outranks its parameters
        enable.setFont(font)
        outer.addWidget(enable)
        form = QFormLayout()
        form.setContentsMargins(14, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        outer.addLayout(form)

        editors: dict[str, tuple] = {}
        for field_info in fields(stage):
            name = field_info.name
            if name == "enabled":
                continue
            value = getattr(stage, name)
            if name == "hex_color":
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(4)
                edit = QLineEdit(str(value))
                edit.setMaxLength(7)
                swatch = QLabel()
                swatch.setFixedSize(18, 18)
                pick = QPushButton("Dropper")
                pick.setObjectName("compact")
                pick.setToolTip("Click the live view to sample a colour "
                                "from the camera frame")
                eyedrop = QPushButton("…")
                eyedrop.setObjectName("compact")
                eyedrop.setFixedWidth(24)
                eyedrop.setToolTip("Choose the colour from a dialog")
                row_layout.addWidget(edit, 1)
                row_layout.addWidget(swatch)
                row_layout.addWidget(pick)
                row_layout.addWidget(eyedrop)
                editors[name] = ("hex", edit, swatch, pick, eyedrop)
                form.addRow("Colour", row)
            elif isinstance(value, bool):
                check = QCheckBox()
                check.setChecked(bool(value))
                editors[name] = ("bool", check)
                form.addRow(_param_label(name), check)
            else:
                lo, hi, step = getattr(stage, "RANGES", {}).get(name, (0, 255, 1))
                if isinstance(value, float):
                    spin = QDoubleSpinBox()
                    spin.setDecimals(2)
                else:
                    spin = QSpinBox()
                spin.setRange(lo, hi)
                spin.setSingleStep(step)
                spin.setValue(value)
                editors[name] = ("num", spin)
                form.addRow(_param_label(name), spin)

        def read():
            kwargs = {"enabled": enable.isChecked()}
            for name, spec in editors.items():
                if spec[0] == "hex":
                    kwargs[name] = valid_hex(spec[1].text(),
                                             getattr(stage, name))
                elif spec[0] == "bool":
                    kwargs[name] = spec[1].isChecked()
                else:
                    kwargs[name] = spec[1].value()
            return replace(stage, **kwargs)

        def refresh_swatch():
            colour = QColor(valid_hex(editors["hex_color"][1].text(),
                                      getattr(stage, "hex_color", "#c8a2c8")))
            editors["hex_color"][2].setStyleSheet(
                f"background: {colour.name()}; border: 1px solid #3f3f46;")

        def on_changed(*_args) -> None:
            refresh_swatch()
            self._on_identify_changed()

        if "hex_color" in editors:
            _kind, edit, _swatch, pick, eyedrop = editors["hex_color"]
            refresh_swatch()
            edit.editingFinished.connect(on_changed)
            pick.clicked.connect(self._arm_pick)
            eyedrop.clicked.connect(
                lambda: self._pick_colour_from_dialog(stage))
        enable.toggled.connect(on_changed)
        for name, spec in editors.items():
            if spec[0] == "num":
                spec[1].valueChanged.connect(on_changed)
                spec[1].editingFinished.connect(self._persist)
            elif spec[0] == "bool":
                spec[1].toggled.connect(on_changed)
        return card, read

    def _identify_config(self) -> IdentifyConfig:
        """A FRESH config from the widgets. Built per detection job so the
        worker never reads a widget and never sees a half-edited object."""
        stages = []
        for name, (_editor, reader) in self._stage_editors.items():
            try:
                stages.append(reader())
            except Exception:  # noqa: BLE001 - a bad edit must not stop a run
                continue
        order = IdentifyConfig().stages
        ranked = sorted(stages, key=lambda s: [c.NAME for c in order]
                        .index(s.NAME))
        return IdentifyConfig(stages=ranked)

    def _on_identify_changed(self) -> None:
        # A parameter change is a request to see it: run the preview now if
        # the window is on screen (a hidden window burns no CPU).
        self._engine.set_live(self.isVisible())
        self._persist()

    def _reset_identify(self) -> None:
        self._build_stage_editors(IdentifyConfig())
        self._persist()
        self._set_status("Identification chain reset to defaults")

    def _pick_colour_from_dialog(self, stage: Stage) -> None:
        current = QColor(valid_hex(getattr(stage, "hex_color", "#c8a2c8")))
        chosen = QColorDialog.getColor(current, self, "Sample colour")
        if chosen.isValid():
            widget = self._hex_editor(stage.NAME)
            if widget is not None:
                widget.setText(chosen.name())
                widget.editingFinished.emit()

    def _hex_editor(self, stage_name: str) -> QLineEdit | None:
        entry = self._stage_editors.get(stage_name)
        if entry is None:
            return None
        editor = entry[0]
        for child in editor.findChildren(QLineEdit):
            return child
        return None

    # --- the samples column ---------------------------------------------

    def _build_samples_group(self) -> QWidget:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        self._samples_note = QLabel("No samples yet.")
        self._samples_note.setObjectName("dim")
        self._samples_note.setWordWrap(True)
        layout.addWidget(self._samples_note)
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["#", "X µm", "Y µm", "Area µm²", "Edge"])
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection)
        header = self._table.horizontalHeader()
        for col in range(5):
            header.setSectionResizeMode(col, header.ResizeMode.Stretch)
        layout.addWidget(self._table, stretch=1)
        self._go_to_btn = QPushButton("Go to the selected sample")
        self._go_to_btn.clicked.connect(self._on_go_to)
        layout.addWidget(self._go_to_btn)
        self._clear_btn = QPushButton("Clear results")
        self._clear_btn.setObjectName("compact")
        self._clear_btn.clicked.connect(self._clear_results)
        layout.addWidget(self._clear_btn)
        return card

    # ------------------------------------------------------------------
    # settings
    # ------------------------------------------------------------------

    @staticmethod
    def _spin(lo: float, hi: float, value: float, suffix: str = ""):
        """A µm/percent field. One decimal: sub-micron precision in a spin
        box the operator nudges with a mouse is noise, and the stage does
        not resolve it anyway."""
        box = QDoubleSpinBox()
        box.setRange(lo, hi)
        box.setDecimals(1)
        box.setSingleStep(10.0)
        box.setSuffix(suffix)
        box.setValue(value)
        return box

    @staticmethod
    def _combo(choices) -> QComboBox:
        combo = QComboBox()
        for data, label in choices:
            combo.addItem(label, data)
        return combo

    @staticmethod
    def _select(combo: QComboBox, data) -> None:
        index = combo.findData(data)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _load_settings(self) -> None:
        saved = load_scan_settings(self._settings)
        self._width.setValue(saved["width_um"])
        self._height.setValue(saved["height_um"])
        self._overlap.setValue(float(saved["overlap"]) * 100.0)
        self._select(self._x_dir, int(saved.get("x_dir") or 1))
        self._select(self._y_dir, int(saved.get("y_dir") or 1))
        self._select(self._path, saved.get("path") or "serpentine")
        self._select(self._serpentine, bool(saved.get("serpentine", True)))
        self._select(self._start_axis, saved.get("start_axis") or "x")
        self._speed.setValue(int(saved.get("speed_pps") or 500))
        self._settle.setValue(int(saved.get("settle_ms") or 0))
        self._backlash.setValue(float(saved.get("backlash_um") or 0.0))
        self._select(self._approach, int(saved.get("backlash_approach") or 1))
        self._return_home.setChecked(bool(saved.get("return_to_start", True)))
        # The objective can be changed while this window is open, and the
        # field of view and the plan are derived from it — so they follow.
        self._state.sig_objective_changed.connect(
            lambda _index: self._refresh_fov_label())
        self._export_mosaic.setChecked(bool(saved.get("export_mosaic", True)))
        self._export_candidates.setChecked(
            bool(saved.get("export_candidates", True)))
        self._export_overview.setChecked(
            bool(saved.get("export_overview", True)))
        self._preview.setCurrentIndex(2)          # 50 % — the live preview
        self._stage_editors = {}
        self._apply_identify_from_settings()

    def _apply_identify_from_settings(self) -> None:
        stored = self._settings.section("identify")
        config = IdentifyConfig.from_dict(stored) if stored \
            else IdentifyConfig()
        self._build_stage_editors(config)

    def _persist(self) -> None:
        save_scan_settings(self._settings, {
            "width_um": self._width.value(),
            "height_um": self._height.value(),
            "overlap": self._overlap.value() / 100.0,
            "x_dir": self._x_dir.currentData(),
            "y_dir": self._y_dir.currentData(),
            "path": self._path.currentData(),
            "serpentine": self._serpentine.currentData(),
            "start_axis": self._start_axis.currentData(),
            "speed_pps": self._speed.value(),
            "settle_ms": self._settle.value(),
            "backlash_um": self._backlash.value(),
            "backlash_approach": self._approach.currentData(),
            "return_to_start": self._return_home.isChecked(),
            "export_mosaic": self._export_mosaic.isChecked(),
            "export_candidates": self._export_candidates.isChecked(),
            "export_overview": self._export_overview.isChecked(),
        })
        self._settings.update("identify", self._identify_config().to_dict())
        self._settings.save()
        self.sig_settings_changed.emit()

    def _camera_flip(self) -> bool:
        """The camera flip, which decides how the image axes sit relative
        to the stage — the map, the mosaic and the px→µm mapping all follow
        it (cv/orientation.py)."""
        return bool(self._settings.device("camera").get("flip", True))

    def refresh_settings(self) -> None:
        """Re-read what the panel derives from settings and re-plan."""
        self.map.set_flip(self._camera_flip())
        self._refresh_fov_label()

    def on_camera_flip_changed(self) -> None:
        """The flip rotates every frame 180°: tiles already placed on the
        map, and any sample centroid computed from them, belong to the old
        orientation — "go to" would command a mirrored move."""
        self._live_candidates = []
        self.sig_processed_frame.emit(None)
        if self._scan_tiles:
            self.map.clear_tiles()
            self._scan_tiles.clear()
            self._scan_hits.clear()
            self._show_candidates([], "scan")

    # ------------------------------------------------------------------
    # planning
    # ------------------------------------------------------------------

    def _objective_row(self) -> dict:
        rows = self._settings.get("objectives") or []
        if not rows:
            return {}
        return rows[min(int(self._state.objective), len(rows) - 1)]

    def _stage_position(self) -> StagePosition:
        return StagePosition.from_telemetry(
            self._manager.last_position.get("zolix"))

    def _canonical_calibration(self) -> ObjectiveCalibration:
        if self._calibration is not None:
            return self._calibration.calibration()
        pos = self._state.objective
        mag = 5.0 * (2.0 ** pos)
        um_per_px = 2.0 / mag
        return ObjectiveCalibration(objective_id=-1, um_per_px_x=um_per_px,
                                    um_per_px_y=um_per_px, source="pixel_pitch")

    def _fov(self) -> tuple[float, float]:
        """The field of view in µm, from the ACTIVE OBJECTIVE's calibration.

        Resolution-independent by construction: the stored value is µm per
        4K-sensor pixel, so a 1080p live frame and a 4K snapshot tile the
        same. Always derived, never entered — see the FOV card's comment."""
        calib = self._canonical_calibration()
        um_x = calib.um_per_px_x or 0.0
        um_y = calib.um_per_px_y or 0.0
        if um_x > 0 and um_y > 0:
            return (um_x * SENSOR_WIDTH_PX, um_y * SENSOR_HEIGHT_PX)
        # No calibration at all: a nominal field for the nosepiece, so the
        # preview still draws something and the note can say what it is.
        mag = 5.0 * (2.0 ** int(self._state.objective))
        um_per_px = 2.0 / mag
        return (um_per_px * SENSOR_WIDTH_PX, um_per_px * SENSOR_HEIGHT_PX)

    def _refresh_fov_label(self) -> None:
        """Show the field of view and, more importantly, WHERE IT CAME
        FROM: an estimate must never pass for a measurement."""
        calib = self._canonical_calibration()
        source = getattr(calib, "source", "none")
        fov_x, fov_y = self._fov()
        self._fov_value.setText(f"{fov_x:.0f} × {fov_y:.0f} µm")
        if source in ("talos_measured", "labscope"):
            where = ("measured in TALOS" if source == "talos_measured"
                     else "imported from Labscope")
            self._fov_note.setText(
                f"From the objective ({where}), {calib.um_per_px_x:.4f} "
                f"µm/px.")
        else:
            self._fov_note.setText(
                "ESTIMATED from the sensor pitch — set the real value in "
                "Preferences → Objectives & Calibration, where it also fixes "
                "the scale bar and the measured areas.")
        self._refresh_plan()

    def _params_for(self, origin: StagePosition | None) -> ScanParams:
        origin = origin or self._stage_position()
        return ScanParams(
            x0_um=origin.x_um, y0_um=origin.y_um,
            width_um=self._width.value(), height_um=self._height.value(),
            overlap=self._overlap.value() / 100.0,
            serpentine=bool(self._serpentine.currentData()),
            speed_pps=float(self._speed.value()),
            path=str(self._path.currentData()),
            start_axis=str(self._start_axis.currentData()),
            x_dir=int(self._x_dir.currentData() or 1),
            y_dir=int(self._y_dir.currentData() or 1),
            settle_ms=int(self._settle.value()),
            backlash_um=float(self._backlash.value()),
            backlash_approach=int(self._approach.currentData() or 1),
            return_to_start=self._return_home.isChecked())

    def _refresh_plan(self) -> None:
        """The map always shows the plan the CURRENT fields would run."""
        origin = self._stage_position()
        params = self._params_for(origin)
        fov = self._fov()
        waypoints = plan_path(params, fov)
        self.map.set_plan(ScanMapPlan(
            x0_um=params.x0_um, y0_um=params.y0_um,
            width_um=params.width_um, height_um=params.height_um,
            x_dir=params.x_dir, y_dir=params.y_dir,
            fov_x_um=fov[0], fov_y_um=fov[1],
            waypoints=[(w.x_um, w.y_um) for w in waypoints]))
        self.map.set_footprint(origin.x_um, origin.y_um)
        cols, rows = grid_shape(params, fov)
        if self._job is None:
            self._set_status(
                f"{len(waypoints)} waypoints ({cols} × {rows} tiles) · "
                f"area {params.width_um:.0f} × {params.height_um:.0f} µm · "
                f"move to the first tile and press Scan from here")

    # ------------------------------------------------------------------
    # frames in
    # ------------------------------------------------------------------

    def on_frame(self, frame: np.ndarray) -> None:
        """The newest streamed frame (from the main window). The window
        keeps its own reference: the picker and the live detection both
        need the pixel data, and neither may block the stream."""
        self._latest_frame = frame

    def update_telem(self, key: str, payload: dict) -> None:
        if key != "zolix" or not payload:
            return
        pos = StagePosition.from_telemetry(payload)
        if self._job is None:
            self.map.set_footprint(pos.x_um, pos.y_um)

    # ------------------------------------------------------------------
    # identification
    # ------------------------------------------------------------------

    def _live_calibration(self) -> ObjectiveCalibration:
        """The active calibration expressed in LIVE-frame pixels (the
        stored value is per 4K-sensor pixel)."""
        calib = self._canonical_calibration()
        frame = self._latest_frame
        if frame is None or not frame.shape[1]:
            return calib
        factor = SENSOR_WIDTH_PX / float(frame.shape[1])
        return replace(calib,
                       um_per_px_x=(calib.um_per_px_x or 0.0) * factor,
                       um_per_px_y=(calib.um_per_px_y or 0.0) * factor)

    def _detect_source(self):
        frame = self._latest_frame
        if frame is None:
            return None
        return (frame, self._live_calibration(), self._stage_position(),
                self._identify_config(), float(self._preview.currentData()),
                self._camera_flip())

    def _on_detected(self, index: int, result, preview) -> None:
        if result is None:
            return
        if index < 0:                       # the live feed
            self._live_candidates = list(result.candidates)
            if preview is not None:
                self.sig_processed_frame.emit(preview)
            self._counts.setText(result.summary)
            if self._job is None and not self._scan_hits:
                self._show_candidates(self._live_candidates, "live view")
            return
        # a scan tile
        self._scan_hits[index] = list(result.candidates)
        if self._pending_export is not None:
            self._maybe_export()
        self._show_candidates(self._flatten_scan_hits(),
                              f"scan · {len(self._scan_hits)} tiles")
        self._refresh_markers()

    def _flatten_scan_hits(self) -> list[FlakeCandidate]:
        out: list[FlakeCandidate] = []
        for index in sorted(self._scan_hits):
            out.extend(self._scan_hits[index])
        return out

    def _show_candidates(self, candidates, source: str) -> None:
        rows = candidate_rows(candidates)
        self._table.setRowCount(len(rows))
        for row, cells in enumerate(rows):
            for col, text in enumerate(cells):
                self._table.setItem(row, col, QTableWidgetItem(text))
        self._samples_note.setText(
            f"{len(rows)} sample(s) from the {source}." if rows
            else f"Nothing matched in the {source}.")
        # an enabled button that only ever answers "select a sample first"
        # is a button that lies
        self._go_to_btn.setEnabled(bool(rows) and self._job is None)

    def _refresh_markers(self) -> None:
        candidates = self._flatten_scan_hits()
        markers = [ScanMapMarker(x_um=c.x_um, y_um=c.y_um,
                                 label=str(index + 1))
                   for index, c in enumerate(candidates)]
        self.map.set_markers(markers)
        self.map.set_caption(
            f"{len(self._scan_tiles)} tiles · {len(candidates)} sample(s)")

    def _clear_results(self) -> None:
        self._scan_hits.clear()
        self._scan_tiles.clear()
        self._live_candidates = []
        self._table.setRowCount(0)
        self.map.set_markers([])
        self.map.set_caption("")
        self._samples_note.setText("Results cleared.")

    # --- the dropper -----------------------------------------------------

    def _arm_pick(self) -> None:
        """Ask for the dropper on the SAMPLE FINDING live view — the only
        one there is."""
        self.sig_pick_requested.emit()
        self._set_status("Click the live view to sample a colour")

    def on_pick(self, x_px: int, y_px: int) -> None:
        """A click on the live view, in FRAME pixels.

        The colour comes from the ORIGINAL frame, never from what is on
        screen: in processed mode the display is darkened outside the match
        and outlined, so sampling it would return a colour the sample does
        not have.
        """
        frame = self._latest_frame
        if frame is None:
            return
        colour = sample_hex(frame, x_px, y_px)
        if colour is None:
            return
        editor = self._hex_editor("colour")
        if editor is None:
            return
        editor.setText(colour)
        editor.editingFinished.emit()
        self._set_status(f"Colour sampled: {colour}")

    # ------------------------------------------------------------------
    # the scan
    # ------------------------------------------------------------------

    def _on_scan(self) -> None:
        if self._state.mode != "MANUAL":
            QMessageBox.information(
                self, "Scan",
                f"The stage is in use ({self._state.mode}) — a scan cannot "
                "start while another job owns the axes.")
            return
        position = self._stage_position()
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
        # fire sig_stop_all_done, which this window treats as the operator's
        # STOP ALL, and the scan would abort itself the moment it started.
        # The scan drives the XYR stage; the focus and transfer axes are
        # left exactly as the operator had them.
        if self._input is not None:
            self._input.cancel_all_holds("scan start")
        self._manager.submit("zolix", "stop", priority=1)
        self._scan_abort.clear()
        self._scan_hits.clear()
        self._scan_tiles.clear()
        self._pending_export = None
        self._refresh_plan()
        self.map.clear_tiles()                 # the previous run's tiles
        self._set_job("scan")
        self._state.set_mode("SCAN")
        self._scan_started_at = time.monotonic()
        self._progress.setValue(0)
        self._set_status("Scanning — Esc or Abort stops it")
        self._scan_worker = _Worker(lambda: self._run_scan(position), self)
        self._scan_worker.sig_log.connect(self._log)
        self._scan_worker.sig_done.connect(self._on_scan_done)
        # a finished QThread deletes itself; the window keeps no graveyard
        # of one object per scan
        self._scan_worker.finished.connect(self._scan_worker.deleteLater)
        self._scan_worker.start()

    def _run_scan(self, origin: StagePosition):
        """The scan thread. Owns the adapter and the scanner; every UI
        update comes back as a queued signal."""
        from talos.hal.proxies.stage_adapter import ManagerStageAdapter

        row = self._objective_row()
        stage_cfg = scan_speed_config(
            self._settings.device("zolix"),
            float(row.get("stage_speed_multiplier") or 1.0))
        adapter = ManagerStageAdapter(self._manager, stage_cfg,
                                      abort_check=self._scan_abort.is_set)
        params = self._params_for(origin)
        fov = self._fov()
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
        self._progress.setMaximum(max(1, total))
        self._progress.setValue(done)
        elapsed = time.monotonic() - self._scan_started_at
        eta = ""
        if done >= 2 and elapsed > 0:
            remaining = elapsed / done * (total - done)
            eta = f" · ~{remaining:.0f} s left"
        self._set_status(f"Tile {done}/{total}{eta}")

    def _on_tile(self, index: int, x_um: float, y_um: float,
                 thumb) -> None:
        self._scan_tiles[index] = (x_um, y_um)
        self.map.add_tile(ScanMapTile(index=index, x_um=x_um, y_um=y_um,
                                      thumb=thumb))
        self._refresh_markers()

    def _on_scan_frame(self, index: int, x_um: float, y_um: float,
                       frame) -> None:
        """A captured tile → the detection queue. The pipeline needs the
        FULL-resolution frame; the map holds the thumbnail."""
        self._scan_tiles[index] = (x_um, y_um)
        self._engine.submit_tile(
            index, frame, self._live_calibration(),
            StagePosition(x_um=x_um, y_um=y_um, r_deg=0.0),
            self._identify_config(), scale=1.0, flip=self._camera_flip())

    def _on_scan_done(self, payload) -> None:
        if self._state.mode == "SCAN":
            self._state.set_mode("MANUAL")
        self._set_job(None)
        self._abort_btn.setEnabled(False)
        self._progress.setVisible(False)
        if not payload:
            self._set_status("Scan failed (see the log)")
            return
        result = payload["result"]
        aborted = result.aborted or self._scan_abort.is_set()
        what = "Scan aborted" if aborted else "Scan done"
        missing = getattr(result, "missing", 0)
        note = f" · {missing} waypoint(s) had no frame" if missing else ""
        self._set_status(f"{what}: {len(result.frames)} frame(s){note} → "
                         f"{result.manifest_path}")
        self._pending_export = payload
        self._maybe_export()

    def _on_abort(self, reason: str = "abort") -> None:
        """Cooperative abort — never QThread.terminate() (it killed the
        scan thread mid-serial-write once, leaving the port open)."""
        scanner = self._scanner
        if scanner is not None:
            self._scan_abort.set()
            scanner.request_abort()
        if self._autofocus is not None and getattr(self._autofocus, "busy",
                                                   False):
            self._autofocus.abort()
        if self._input is not None:
            self._input.cancel_all_holds(reason)
        self._manager.stop_all()
        self._set_status("Aborting…")

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
        self._scan_btn.setEnabled(not busy)
        # enabled only when there is something to go to (and no job owns
        # the axes) — an enabled button that can only refuse is a lie
        self._go_to_btn.setEnabled(not busy and self._table.rowCount() > 0)
        self._abort_btn.setEnabled(busy)
        self._progress.setVisible(busy)
        for widget in (self._width, self._height, self._overlap, self._path,
                       self._serpentine, self._start_axis, self._x_dir,
                       self._y_dir, self._speed, self._settle,
                       self._backlash, self._approach, self._return_home):
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
        results are in"."""
        if self._pending_export is None or self._export_worker is not None:
            return
        if self._engine.pending_tiles > 0:
            return
        payload = dict(self._pending_export)
        self._pending_export = None
        # everything the worker needs is copied OUT of the widgets here:
        # reading a checkbox from another thread is not allowed, and the
        # results keep arriving while the write runs.
        payload["hits"] = {index: list(items)
                           for index, items in self._scan_hits.items()}
        payload["tiles"] = dict(self._scan_tiles)
        payload["flip"] = self._camera_flip()
        payload["exports"] = {
            "mosaic": self._export_mosaic.isChecked(),
            "candidates": self._export_candidates.isChecked(),
            "overview": self._export_overview.isChecked()}
        self._export_worker = _Worker(lambda: self._finish_exports(payload),
                                      self)
        self._export_worker.sig_done.connect(self._on_export_done)
        self._export_worker.sig_log.connect(self._log)
        self._export_worker.finished.connect(self._export_worker.deleteLater)
        self._export_worker.start()

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
        self._set_status(
            f"{summary['samples']} sample(s) in {summary['dir'].name} · "
            f"wrote {detail}")

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        self._set_status(str(message))

    def _set_status(self, text: str) -> None:
        self._status.setText(text)

    def _on_go_to(self) -> None:
        row = self._table.currentRow()
        candidates = (self._flatten_scan_hits() if self._scan_hits
                      else self._live_candidates)
        if row < 0 or row >= len(candidates):
            QMessageBox.information(self, "Go to", "Select a sample first.")
            return
        if self._state.mode != "MANUAL":
            QMessageBox.information(
                self, "Go to",
                f"The stage is in use ({self._state.mode}) — sample moves "
                "are refused while a job owns the axes.")
            return
        cand = candidates[row]
        position = self._stage_position()
        dx = cand.x_um - position.x_um
        dy = cand.y_um - position.y_um
        answer = QMessageBox.question(
            self, "Move to sample",
            f"Move the stage by ΔX={dx:+.1f} µm, ΔY={dy:+.1f} µm to bring "
            "this sample to the crosshair?",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Ok:
            self._manager.submit("zolix", "move_rel_um", dx, dy)
            self._set_status(f"Moving to sample #{row + 1}")

    # --- window contract -------------------------------------------------

    def set_toggle_action(self, action) -> None:
        self._toggle_action = action

    def set_preview_wanted(self, wanted: bool) -> None:
        """The Sample Finding tab is showing the processed view: it wants
        the overlay even while this window is hidden. The live preview runs
        while EITHER consumer is interested."""
        self._preview_wanted = bool(wanted)
        self._update_preview()

    def _update_preview(self) -> None:
        self._engine.set_live(self._preview_wanted or self.isVisible())

    def _hide_and_uncheck(self) -> None:
        self.hide()
        if self._toggle_action is not None:
            self._toggle_action.setChecked(False)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.refresh_settings()
        self._refresh_plan()
        self._update_preview()

    def hideEvent(self, event) -> None:  # noqa: N802
        super().hideEvent(event)
        # a hidden window burns no CPU — unless the tab still shows the
        # processed overlay it is fed from here
        self._update_preview()

    def reject(self) -> None:  # Esc
        # The window hides, but Esc stays the global STOP ALL — the main
        # window's shortcut is WindowShortcut-scoped and cannot fire here.
        if self._input is not None:
            self._input.on_escape()
        else:
            self._on_abort("escape")
        self._hide_and_uncheck()

    def closeEvent(self, event) -> None:  # noqa: N802
        event.ignore()                     # never destroyed
        self._hide_and_uncheck()

    def shutdown(self) -> None:
        """App teardown: stop the detection thread."""
        self._engine.shutdown()


__all__ = ["ScanWindow"]
