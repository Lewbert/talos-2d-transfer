"""Navigation right-panel groups: quick actions (fixed top), then the
scrollable Capture / Camera / AF / Temperature settings groups.

Camera edits persist to the settings (the CLI/tooling sessions and the
next app start read the same values); the auto-gain checkbox drives the
AutoGainController. The gain controls disable while auto-gain is on —
manual and auto must never fight.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from talos.cv.autofocus_service import planned_kwargs

_EXPOSURE_MIN, _EXPOSURE_MAX = 61.0, 1_000_000.0


def exposure_to_slider(exposure_us: float) -> int:
    """Log mapping 61 µs .. 1 s → 0..1000."""
    exp = max(_EXPOSURE_MIN, min(exposure_us, _EXPOSURE_MAX))
    return int(round(1000.0 * math.log(exp / _EXPOSURE_MIN)
                     / math.log(_EXPOSURE_MAX / _EXPOSURE_MIN)))


def slider_to_exposure(value: int) -> float:
    return round(_EXPOSURE_MIN * (_EXPOSURE_MAX / _EXPOSURE_MIN)
                 ** (max(0, min(value, 1000)) / 1000.0), 1)


class QuickActionsGroup(QGroupBox):
    """Commonly used buttons: Snapshot, Focus once (AF-S), and the
    software-origin setters (stored for the future scanning function)."""

    sig_snapshot_requested = Signal()
    sig_af_requested = Signal()
    sig_set_stage_origin = Signal()
    sig_set_focus_origin = Signal()

    def __init__(self, manager, settings, state, autofocus_service,
                 parent=None):
        super().__init__("Quick Actions", parent)
        self._state = state
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        row1 = QHBoxLayout()
        self._snap = QPushButton("Snapshot")
        self._snap.setObjectName("qa_primary")
        self._snap.clicked.connect(self.sig_snapshot_requested.emit)
        row1.addWidget(self._snap)
        self._af = QPushButton("AF-S")
        self._af.setObjectName("qa")
        self._af.setToolTip("Focus once (autofocus AF-S)")
        self._af.clicked.connect(self.sig_af_requested.emit)
        row1.addWidget(self._af)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self._stage_origin_btn = QPushButton("Set stage origin")
        self._stage_origin_btn.setObjectName("qa")
        self._stage_origin_btn.setToolTip(
            "Store the current XYR position as the software origin "
            "(used by the future scanning function)")
        self._stage_origin_btn.clicked.connect(
            self.sig_set_stage_origin.emit)
        row2.addWidget(self._stage_origin_btn)
        self._focus_origin_btn = QPushButton("Set focus origin")
        self._focus_origin_btn.setObjectName("qa")
        self._focus_origin_btn.setToolTip(
            "Store the current focus position as the software origin")
        self._focus_origin_btn.clicked.connect(
            self.sig_set_focus_origin.emit)
        row2.addWidget(self._focus_origin_btn)
        layout.addLayout(row2)

        self._origin_label = QLabel("origins: not set")
        self._origin_label.setObjectName("dim")
        self._origin_label.setWordWrap(True)
        layout.addWidget(self._origin_label)
        if state is not None:
            state.sig_stage_origin_changed.connect(
                lambda _p: self._refresh_origin())
            state.sig_focus_origin_changed.connect(
                lambda _s: self._refresh_origin())
            self._refresh_origin()

    def _refresh_origin(self) -> None:
        parts = []
        stage = self._state.stage_origin
        if stage is not None:
            parts.append(f"XYR origin: {stage.x_um:.1f}, {stage.y_um:.1f} µm")
        focus = self._state.focus_origin
        if focus is not None:
            parts.append(f"focus origin: {focus} st")
        self._origin_label.setText(
            " · ".join(parts) if parts else "origins: not set")


class CaptureGroup(QGroupBox):
    """Capture settings: save directory, filename pattern, format and
    capture resolution (the live view stays 1080p either way)."""

    def __init__(self, settings, parent=None):
        super().__init__("Capture", parent)
        self._settings = settings
        cfg = settings.section("capture")
        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setVerticalSpacing(6)

        from talos.capture import default_snapshot_dir

        self._dir = QLineEdit(str(cfg.get("dir", "") or default_snapshot_dir()))
        browse = QPushButton("…")
        browse.setObjectName("compact")
        browse.clicked.connect(self._on_browse)
        open_btn = QPushButton("Open")
        open_btn.setObjectName("compact")
        open_btn.setToolTip("Open the capture folder")
        open_btn.clicked.connect(self._on_open_dir)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self._dir, stretch=1)
        dir_row.addWidget(browse)
        dir_row.addWidget(open_btn)
        form.addRow("Save to", dir_row)
        # A typed/pasted path must persist too — only Browse used to save
        # it, so an edited path silently reverted on the next restart.
        self._dir.editingFinished.connect(self._on_dir_edited)

        self._prefix = QLineEdit(str(cfg.get("prefix", "")))
        self._prefix.setPlaceholderText("snap")
        self._prefix.editingFinished.connect(
            lambda: self._persist("prefix", self._prefix.text()))
        form.addRow("File name", self._prefix)

        self._pattern = QComboBox()
        self._pattern.addItem("Timestamp", "timestamp")
        self._pattern.addItem("Number", "number")
        idx = self._pattern.findData(cfg.get("pattern", "timestamp"))
        self._pattern.setCurrentIndex(max(0, idx))
        self._pattern.currentIndexChanged.connect(
            lambda: self._persist("pattern", self._pattern.currentData()))
        form.addRow("Suffix", self._pattern)

        self._format = QComboBox()
        self._format.addItems(["png", "jpg"])
        self._format.setCurrentText(str(cfg.get("format", "png")))
        self._format.currentTextChanged.connect(
            lambda t: self._persist("format", t))
        form.addRow("Format", self._format)

        self._resolution = QComboBox()
        self._resolution.addItem("4K (3840×2160)", 0)
        self._resolution.addItem("1080p (1920×1080)", 1)
        idx = self._resolution.findData(int(cfg.get("resolution", 0)))
        self._resolution.setCurrentIndex(max(0, idx))
        self._resolution.currentIndexChanged.connect(
            lambda: self._persist("resolution",
                                  self._resolution.currentData()))
        form.addRow("Capture res", self._resolution)

        hint = QLabel("4K capture pauses the live view for a few seconds.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        form.addRow(hint)

    def _on_dir_edited(self) -> None:
        text = self._dir.text().strip()
        if text and text != self._settings.section("capture").get("dir"):
            self._persist("dir", text)

    def _on_browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Snapshot directory", self._dir.text())
        if chosen:
            self._dir.setText(chosen)
            self._persist("dir", chosen)

    def _on_open_dir(self) -> None:
        import logging
        import os
        from pathlib import Path

        from talos.capture import default_snapshot_dir

        path = Path(self._dir.text().strip() or str(default_snapshot_dir()))
        try:
            path.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                os.startfile(str(path))  # noqa: S606 - user-chosen folder
            else:
                from PySide6.QtCore import QUrl
                from PySide6.QtGui import QDesktopServices

                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        except OSError as exc:
            logging.getLogger(__name__).warning(
                "Could not open the capture folder %s: %s", path, exc)

    def _persist(self, key: str, value) -> None:
        self._settings.section("capture")[key] = value
        self._settings.save()


class CameraGroup(QGroupBox):
    """ZEN-simplified camera settings: fixed exposure + software
    auto-gain + white balance. ``profile_kind`` selects the settings
    section edits persist to ("nav" → devices.camera.*, "scan" →
    devices.camera.scan.* — the scan profile has NO auto controls)."""

    def __init__(self, manager, settings, autogain, profile_kind: str = "nav",
                 parent=None):
        super().__init__("Camera", parent)
        from talos.ui.camera_profiles import update_profile_setting

        self._manager = manager
        self._settings = settings
        self._autogain = autogain
        self._profile_kind = profile_kind
        self._update_setting = update_profile_setting
        cfg = settings.device("camera")
        if profile_kind == "scan":
            cfg = cfg.get("scan") or {}
        self._allow_auto = profile_kind != "scan"
        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setVerticalSpacing(6)

        # The stored value is µs (hardware range 61 µs – 1 s); the UI
        # shows ms. The slider keeps its µs-log mapping.
        exp_value = float(cfg.get("exposure_us", 40000))
        self._exposure = QDoubleSpinBox()
        self._exposure.setRange(0.061, 1000.0)
        self._exposure.setDecimals(3)
        self._exposure.setSingleStep(1.0)
        self._exposure.setValue(round(exp_value / 1000.0, 3))
        self._exposure.editingFinished.connect(self._on_exposure)
        self._exp_slider = QSlider(Qt.Orientation.Horizontal)
        self._exp_slider.setRange(0, 1000)
        self._exp_slider.setValue(exposure_to_slider(exp_value))
        self._exp_slider.valueChanged.connect(
            lambda v: self._exposure.setValue(
                round(slider_to_exposure(v) / 1000.0, 3)))
        self._exp_slider.sliderReleased.connect(
            lambda: self._apply_exposure(self._exposure.value()))
        exp_row = QVBoxLayout()
        exp_row.addWidget(self._exposure)
        exp_row.addWidget(self._exp_slider)
        form.addRow("Exposure (ms)", exp_row)

        auto_value = bool(cfg.get("auto_gain", True))
        self._auto_gain = QCheckBox("Auto gain")
        self._auto_gain.setChecked(auto_value)
        self._auto_gain.toggled.connect(self._on_auto_gain)
        self._gain_once = QPushButton("Gain once")
        self._gain_once.setObjectName("compact")
        self._gain_once.setToolTip(
            "Measure the current view and make ONE gain step toward the "
            "target luma — the auto-gain setting itself is untouched")
        self._gain_once.clicked.connect(self._on_gain_once)
        self._gain_target = QSpinBox()
        self._gain_target.setRange(0, 255)
        self._gain_target.setValue(int(cfg.get("auto_gain_target", 120)))
        self._gain_target.editingFinished.connect(self._on_gain_target)
        auto_row = QHBoxLayout()
        auto_row.addWidget(self._auto_gain)
        auto_row.addStretch(1)
        auto_row.addWidget(self._gain_once)
        if self._allow_auto:
            form.addRow("", auto_row)
            form.addRow("Target luma", self._gain_target)
        else:
            # the scan profile is manual-only: no CONTINUOUS auto
            # anything — the one-shot Gain once button stays for the
            # pre-scan adjustment
            self._auto_gain.hide()
            self._gain_target.hide()
            once_row = QHBoxLayout()
            once_row.addStretch(1)
            once_row.addWidget(self._gain_once)
            form.addRow("", once_row)

        gain_value = float(cfg.get("gain", 4.0))
        self._gain = QDoubleSpinBox()
        self._gain.setRange(1.0, 22.0)
        self._gain.setSingleStep(1.0)
        self._gain.setValue(gain_value)
        self._gain.editingFinished.connect(self._on_gain)
        self._gain_slider = QSlider(Qt.Orientation.Horizontal)
        self._gain_slider.setRange(10, 220)  # ×0.1
        self._gain_slider.setValue(int(gain_value * 10))
        self._gain_slider.valueChanged.connect(
            lambda v: self._gain.setValue(v / 10.0))
        self._gain_slider.sliderReleased.connect(
            lambda: self._apply_gain(self._gain.value()))
        gain_row = QVBoxLayout()
        gain_row.addWidget(self._gain)
        gain_row.addWidget(self._gain_slider)
        form.addRow("Gain ×", gain_row)
        self._set_gain_enabled(not auto_value or not self._allow_auto)

        # The stored WB state is ON/OFF only — "Once" is a transient
        # button action (a stored "Once" would re-run a WB pass on every
        # profile application and break a pre-adjusted color).
        self._wb_auto = QCheckBox("Auto WB")
        self._wb_auto.setChecked(
            str(cfg.get("white_balance", "Off")) == "Continuous")
        self._wb_auto.setToolTip(
            "Continuous auto white balance — the camera re-balances "
            "automatically while this is on")
        self._wb_auto.toggled.connect(self._on_wb_auto)
        self._wb_once = QPushButton("Balance once")
        self._wb_once.setObjectName("compact")
        self._wb_once.setToolTip(
            "Run ONE white-balance pass now, then keep it fixed (the "
            "camera stays off afterwards)")
        self._wb_once.clicked.connect(self._on_wb_once)
        wb_row = QHBoxLayout()
        if self._allow_auto:
            wb_row.addWidget(self._wb_auto)
        wb_row.addStretch(1)
        wb_row.addWidget(self._wb_once)
        form.addRow("", wb_row)
        if not self._allow_auto:
            # the scan profile FORCES WB off (detection needs a stable,
            # reproducible color) — the continuous-AWB checkbox does not
            # exist here; only the one-shot Balance once stays
            self._wb_auto.setChecked(False)
            self._wb_auto.hide()

        # The 208c has no hardware manual WB gain pair (WhiteBalance is
        # read-only) — fixed WB = AWB Off + this temperature.
        self._wb_temp = QSpinBox()
        self._wb_temp.setRange(1500, 10000)
        self._wb_temp.setSingleStep(100)
        self._wb_temp.setSuffix(" K")
        self._wb_temp.setValue(int(float(cfg.get("color_temperature", 5500.0))))
        self._wb_temp.setToolTip(
            "AWB target temperature while Auto; the fixed white balance "
            "when AWB is Off")
        self._wb_temp.editingFinished.connect(self._on_wb_temp)
        form.addRow("WB temperature", self._wb_temp)

        if autogain is not None:
            autogain.sig_gain_changed.connect(self._on_autogain_changed)

    # ------------------------------------------------------------------

    def _persist(self, key: str, value) -> None:
        self._update_setting(self._settings, self._profile_kind, key, value)

    def apply_profile(self, profile) -> None:
        """Push a profile into the controls without triggering edits."""
        self._exposure.blockSignals(True)
        self._exposure.setValue(
            round(float(profile.exposure_us) / 1000.0, 3))
        self._exposure.blockSignals(False)
        self._exp_slider.blockSignals(True)
        self._exp_slider.setValue(exposure_to_slider(profile.exposure_us))
        self._exp_slider.blockSignals(False)
        self._gain.blockSignals(True)
        self._gain.setValue(float(profile.gain))
        self._gain.blockSignals(False)
        self._gain_slider.blockSignals(True)
        self._gain_slider.setValue(int(float(profile.gain) * 10))
        self._gain_slider.blockSignals(False)
        self._auto_gain.blockSignals(True)
        self._auto_gain.setChecked(bool(profile.auto_gain)
                                   and self._allow_auto)
        self._auto_gain.blockSignals(False)
        self._gain_target.blockSignals(True)
        self._gain_target.setValue(int(profile.auto_gain_target))
        self._gain_target.blockSignals(False)
        self._wb_auto.blockSignals(True)
        self._wb_auto.setChecked(
            str(profile.white_balance) == "Continuous" and self._allow_auto)
        self._wb_auto.blockSignals(False)
        self._wb_temp.blockSignals(True)
        self._wb_temp.setValue(int(profile.color_temperature))
        self._wb_temp.blockSignals(False)
        self._set_gain_enabled(not profile.auto_gain
                               or not self._allow_auto)

    def _apply_exposure(self, value_ms: float) -> None:
        us = round(value_ms * 1000.0, 1)  # the hardware value stays µs
        self._manager.submit_camera("set_property", "exposure_us", us)
        self._persist("exposure_us", us)

    def _on_exposure(self) -> None:
        us = round(self._exposure.value() * 1000.0, 1)
        self._exp_slider.blockSignals(True)
        self._exp_slider.setValue(exposure_to_slider(us))
        self._exp_slider.blockSignals(False)
        self._apply_exposure(self._exposure.value())

    def _apply_gain(self, value: float) -> None:
        self._manager.submit_camera("set_property", "gain", value)
        self._persist("gain", value)
        if self._autogain is not None:
            self._autogain.note_manual_gain(value)  # keep the loop in sync

    def _on_gain(self) -> None:
        self._apply_gain(self._gain.value())

    def _on_auto_gain(self, on: bool) -> None:
        self._persist("auto_gain", on)
        if self._autogain is not None:
            self._autogain.set_enabled(on)
        self._set_gain_enabled(not on)

    def _on_gain_target(self) -> None:
        value = self._gain_target.value()
        self._persist("auto_gain_target", value)
        if self._autogain is not None:
            self._autogain.set_target(float(value))

    def _on_gain_once(self) -> None:
        if self._autogain is not None:
            self._autogain.once()

    def _set_gain_enabled(self, on: bool) -> None:
        self._gain.setEnabled(on)
        self._gain_slider.setEnabled(on)

    def _on_autogain_changed(self, gain: float) -> None:
        """The controller moved the gain — follow it without feedback."""
        self._gain.blockSignals(True)
        self._gain.setValue(gain)
        self._gain.blockSignals(False)
        self._gain_slider.blockSignals(True)
        self._gain_slider.setValue(int(gain * 10))
        self._gain_slider.blockSignals(False)

    def _on_wb_auto(self, on: bool) -> None:
        value = "Continuous" if on else "Off"
        self._manager.submit_camera("set_property", "white_balance", value)
        self._persist("white_balance", value)

    def _on_wb_once(self) -> None:
        self._manager.submit_camera("set_property", "white_balance", "Once")
        if self._allow_auto:
            # the camera is Off after the one-shot — keep the checkbox
            # and the stored state honest (no re-trigger on re-entry)
            self._wb_auto.blockSignals(True)
            self._wb_auto.setChecked(False)
            self._wb_auto.blockSignals(False)
            self._persist("white_balance", "Off")

    def _on_wb_temp(self) -> None:
        value = self._wb_temp.value()
        self._manager.submit_camera("set_property", "color_temperature", value)
        self._persist("color_temperature", value)


class AfSettingsWidget(QWidget):
    """The autofocus settings block — measurement region, window bounds,
    rough-scan speed, ±µm override and the per-objective readout.

    Lives in TWO places (the right-panel Autofocus group and the AF detail
    window), so the region is driven by the shared ``AfRegionController``
    and every other field persists immediately and announces itself via
    ``sig_settings_changed`` — the second instance refreshes from settings
    instead of drifting.
    """

    sig_settings_changed = Signal()
    sig_roi_arm_requested = Signal()   # "Select ROI" -> rubber band on the live view

    def __init__(self, settings, state, roi, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._state = state
        self._roi = roi
        af = settings.section("autofocus")

        # --- measurement region ------------------------------------------
        self._area = QComboBox()
        self._area.addItem("Full frame", False)
        self._area.addItem("ROI", True)
        self._area.currentIndexChanged.connect(self._on_area_changed)

        self._select_btn = QPushButton("Select ROI")
        self._select_btn.setObjectName("compact")
        self._select_btn.setToolTip("Drag a rectangle on the live view")
        self._select_btn.clicked.connect(self.sig_roi_arm_requested.emit)
        self._reset_btn = QPushButton("Reset ROI")
        self._reset_btn.setObjectName("compact")
        self._reset_btn.setToolTip("Back to the centre two-thirds of the frame")
        self._reset_btn.clicked.connect(self._roi.reset_to_default)

        # One 2-column grid holds the buttons AND the region editor, so the
        # buttons, the X/Y row and the W/H row all share the same two
        # column edges (percentages of the frame, label inside each cell).
        self._roi_spins: dict[str, QDoubleSpinBox] = {}
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        grid.addWidget(self._select_btn, 0, 0)
        grid.addWidget(self._reset_btn, 0, 1)
        for index, (key, text) in enumerate((("x", "X"), ("y", "Y"),
                                             ("w", "W"), ("h", "H"))):
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 100.0)
            spin.setDecimals(1)
            spin.setSingleStep(1.0)
            spin.setSuffix(" %")
            spin.setToolTip(f"Region {text} position/size, as a percentage "
                            "of the frame")
            spin.editingFinished.connect(lambda k=key: self._on_roi_spin(k))
            self._roi_spins[key] = spin
            cell = QWidget()
            cell_row = QHBoxLayout(cell)
            cell_row.setContentsMargins(0, 0, 0, 0)
            cell_row.setSpacing(4)
            label = QLabel(f"{text} %")
            label.setObjectName("dim")
            cell_row.addWidget(label)
            cell_row.addWidget(spin, stretch=1)
            grid.addWidget(cell, 1 + index // 2, index % 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        # --- window / speed knobs -----------------------------------------
        self._minus = QDoubleSpinBox()
        self._minus.setRange(0.0, 10000.0)
        self._minus.setDecimals(1)
        self._minus.setValue(float(af.get("window_minus_um", 500.0)))
        self._minus.editingFinished.connect(
            lambda: self._persist("window_minus_um", self._minus.value()))

        self._plus = QDoubleSpinBox()
        self._plus.setRange(0.0, 10000.0)
        self._plus.setDecimals(1)
        self._plus.setValue(float(af.get("window_plus_um", 500.0)))
        self._plus.editingFinished.connect(
            lambda: self._persist("window_plus_um", self._plus.value()))

        self._base = QDoubleSpinBox()
        self._base.setRange(1.0, 1000.0)
        self._base.setDecimals(1)
        self._base.setValue(float(af.get("coarse_speed_base_um_s", 100.0)))
        self._base.editingFinished.connect(
            lambda: self._persist("coarse_speed_base_um_s",
                                  self._base.value()))

        self._bounds = QDoubleSpinBox()
        self._bounds.setRange(0.0, 10000.0)
        self._bounds.setDecimals(1)
        self._bounds.setValue(float(af.get("manual_bounds_um", 0.0)))
        self._bounds.editingFinished.connect(
            lambda: self._persist("manual_bounds_um", self._bounds.value()))

        # --- layout: a form for the labelled rows, the grid for the ROI ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        area_row = QHBoxLayout()
        area_row.setSpacing(6)
        area_row.addWidget(QLabel("Measure"))
        area_row.addWidget(self._area, stretch=1)
        layout.addLayout(area_row)
        layout.addLayout(grid)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setVerticalSpacing(6)
        form.addRow("Max window − (µm)", self._minus)
        form.addRow("Max window + (µm)", self._plus)
        form.addRow("Max rough scan speed (µm/s)", self._base)
        form.addRow("Search ± µm", self._bounds)
        hint = QLabel("0 = the full objective window")
        hint.setObjectName("hint")
        form.addRow(hint)
        self._readout = QLabel("")
        self._readout.setObjectName("readout")
        self._readout.setWordWrap(True)
        form.addRow(self._readout)
        layout.addLayout(form)

        self._roi.sig_changed.connect(self._on_roi_changed)
        self._state.sig_objective_changed.connect(
            lambda _idx: self._refresh_readout())
        self._sync_roi_widgets()
        self._refresh_readout()

    # -- ROI ---------------------------------------------------------------

    def _on_area_changed(self, _index: int) -> None:
        if self._area.currentData():          # "ROI"
            if self._roi.is_full_frame():
                self._roi.reset_to_default()
            else:
                self._roi.set_roi(self._roi.roi(), persist=False)
        else:
            self._roi.use_full_frame()

    def _on_roi_spin(self, key: str) -> None:
        value = self._roi_spins[key].value() / 100.0
        self._roi.move_region(**{key: value})

    def _on_roi_changed(self, _roi) -> None:
        self._sync_roi_widgets()
        self.sig_settings_changed.emit()

    def _sync_roi_widgets(self) -> None:
        roi = self._roi.roi()
        full = roi is None
        self._area.blockSignals(True)
        self._area.setCurrentIndex(0 if full else 1)
        self._area.blockSignals(False)
        x, y, w, h = self._roi.as_percent(roi)
        for spin, value in zip((self._roi_spins["x"], self._roi_spins["y"],
                                self._roi_spins["w"], self._roi_spins["h"]),
                               (x, y, w, h)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
            # The numbers describe the region; with the whole frame
            # selected they are showing the default the ROI would start
            # from, so they stay visible but switch off.
            spin.setEnabled(not full)
        self._reset_btn.setEnabled(True)

    # -- window / speed knobs ---------------------------------------------

    def _persist(self, key: str, value: float) -> None:
        self._settings.section("autofocus")[key] = value
        self._settings.save()
        self._refresh_readout()
        self.sig_settings_changed.emit()

    def refresh_from_settings(self) -> None:
        """Re-read every knob (the other instance may have edited it)."""
        af = self._settings.section("autofocus")
        pairs = ((self._minus, "window_minus_um", 500.0),
                 (self._plus, "window_plus_um", 500.0),
                 (self._base, "coarse_speed_base_um_s", 100.0),
                 (self._bounds, "manual_bounds_um", 0.0))
        for spin, key, fallback in pairs:
            spin.blockSignals(True)
            spin.setValue(float(af.get(key, fallback)))
            spin.blockSignals(False)
        self._refresh_readout()

    def _refresh_readout(self) -> None:
        rows = self._settings.get("objectives") or []
        if not rows:
            self._readout.setText("no objectives configured")
            return
        idx = int(getattr(self._state, "objective", 0))
        row = rows[min(idx, len(rows) - 1)]
        # The PLANNER's numbers, not a re-derivation: the readout used to
        # scale by "af_speed_multiplier or 1.0" while build_config falls back
        # to (na_min/na)², and it hardcoded a 50 st/s fine floor — so a row
        # without an explicit multiplier was advertised 9× faster and 9×
        # wider than the run.
        kwargs, _warnings = planned_kwargs(self._settings, row)
        um = float(self._settings.device("focus").get("um_per_step", 0.2))
        minus_um = kwargs["window_minus_steps"] * um
        plus_um = kwargs["window_plus_steps"] * um
        self._readout.setText(
            f"{row.get('name', '?')}: search −{minus_um:.0f} / "
            f"+{plus_um:.0f} µm · coarse {kwargs['coarse_speed']} st/s · "
            f"fine {kwargs['fine_speed']} st/s")

    def bounds_um(self) -> float:
        return self._bounds.value()

    def bounds_steps(self, center_steps: int, um_per_step: float) \
            -> tuple[int, int] | None:
        """±µm → (lo, hi) steps around the center; None = unbounded."""
        um = self._bounds.value()
        if um <= 0 or um_per_step <= 0:
            return None
        half = int(round(um / um_per_step))
        if half <= 0:
            return None
        return (int(center_steps) - half, int(center_steps) + half)


class AFGroup(QGroupBox):
    """Right-panel wrapper around the shared AF settings block."""

    def __init__(self, settings, state, roi=None, parent=None):
        super().__init__("Autofocus", parent)
        from talos.ui.af_region import AfRegionController

        self._roi = roi or AfRegionController(settings)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.widget = AfSettingsWidget(settings, state, self._roi)
        layout.addWidget(self.widget)

    # Kept as the group's public surface (MainWindow and the tests use
    # these; the implementation lives in the settings widget).
    def bounds_um(self) -> float:
        return self.widget.bounds_um()

    def bounds_steps(self, center_steps: int, um_per_step: float):
        return self.widget.bounds_steps(center_steps, um_per_step)


class TemperatureGroup(QGroupBox):
    """Temperature setpoint + presets (the strip shows PV/SV live)."""

    def __init__(self, manager, settings, parent=None):
        super().__init__("Temperature", parent)
        self._manager = manager
        cfg = settings.device("yudian")
        self._safety_lo = float(cfg.get("safety_lo_c", -100.0))
        self._safety_hi = float(cfg.get("safety_hi_c", 400.0))

        # A grid (not a form) with a FIXED-width right-aligned label
        # column (the widest label) so all three rows line up on one
        # edge, and identical Set buttons.
        grid = QGridLayout(self)
        grid.setVerticalSpacing(6)
        grid.setHorizontalSpacing(8)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)

        presets = cfg.get("presets", []) or []
        labels: list[QLabel] = []

        def label(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight
                             | Qt.AlignmentFlag.AlignVCenter)
            labels.append(lbl)
            return lbl

        self._sv_seeded = False   # see update_telem (seeds from the device)
        self._sv_box = QDoubleSpinBox()
        self._sv_box.setRange(self._safety_lo, self._safety_hi)
        self._sv_box.setDecimals(1)
        self._sv_box.setSuffix(" °C")
        self._sv_box.setValue(float(presets[0].get("temp_c", 25.0))
                              if presets else 25.0)
        self._set_btn = QPushButton("Set")
        self._set_btn.setFixedWidth(64)
        self._set_btn.clicked.connect(lambda: self._set_temp(self._sv_box.value()))
        set_row = QHBoxLayout()
        set_row.addWidget(self._sv_box, stretch=1)
        set_row.addWidget(self._set_btn)
        grid.addWidget(label("Setpoint"), 0, 0)
        grid.addLayout(set_row, 0, 1)

        self._preset_combo = QComboBox()
        for preset in presets:
            self._preset_combo.addItem(str(preset["name"]),
                                       float(preset["temp_c"]))
        self._preset_set = QPushButton("Set")
        self._preset_set.setFixedWidth(64)  # identical to the setpoint Set
        self._preset_set.clicked.connect(self._on_preset)
        self._preset_set.setEnabled(self._preset_combo.count() > 0)
        preset_row = QHBoxLayout()
        preset_row.addWidget(self._preset_combo, stretch=1)
        preset_row.addWidget(self._preset_set)
        grid.addWidget(label("Presets"), 1, 0)
        grid.addLayout(preset_row, 1, 1)

        self._sv_readout = QLabel("—")
        self._sv_readout.setObjectName("readout")
        grid.addWidget(label("Current"), 2, 0)
        grid.addWidget(self._sv_readout, 2, 1)

        # the label column = the widest label, so the right alignment
        # actually shows (a QLabel hugs its text in a natural cell)
        label_w = max(lbl.sizeHint().width() for lbl in labels)
        for lbl in labels:
            lbl.setFixedWidth(label_w)

    def _set_temp(self, temp_c: float) -> None:
        if not (self._safety_lo <= temp_c <= self._safety_hi):
            QMessageBox.warning(self, "Setpoint rejected",
                                f"{temp_c} °C is outside the safety range "
                                f"[{self._safety_lo}, {self._safety_hi}] °C.")
            return
        self._manager.submit("yudian", "set_sv", float(temp_c))

    def _on_preset(self) -> None:
        value = self._preset_combo.currentData()
        if value is not None:
            self._sv_box.setValue(float(value))
            self._set_temp(float(value))

    def update_telem(self, payload: dict) -> None:
        sv = payload.get("sv")
        if sv is not None:
            self._sv_readout.setText(f"SV {sv:.1f} °C")
            if not self._sv_seeded:
                # Seed the Setpoint box from the device ONCE, when the
                # first telemetry arrives: it used to start at preset[0]
                # (25 °C), so pressing Set without thinking clobbered the
                # controller's real setpoint. Seeded once so it never
                # fights what the user is typing.
                self._sv_seeded = True
                self._sv_box.blockSignals(True)
                self._sv_box.setValue(float(sv))
                self._sv_box.blockSignals(False)
