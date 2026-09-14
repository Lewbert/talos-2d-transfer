"""AutofocusPanel: AF-S controls (the app's only autofocus strategy —
the v3 adaptive), measure area (full / ROI / rubber-band selection),
live progress + sharpness curve, backlash calibration, and the
objectives editor."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from talos.ui.widgets.overlay import PHASE_NAMES
from talos.ui.widgets.sharpness_curve import SharpnessCurveWidget


class AutofocusPanel(QGroupBox):
    def __init__(self, manager, settings, state, service, live_view,
                 parent: QWidget | None = None):
        super().__init__("Autofocus", parent)
        self._manager = manager
        self._settings = settings
        self._state = state
        self._service = service
        self._live_view = live_view
        saved_roi = settings.section("autofocus").get("default_roi_norm")
        self._roi_norm: tuple | None = tuple(saved_roi) if saved_roi else None

        root = QVBoxLayout(self)
        root.setSpacing(6)

        # --- objective / range readout -------------------------------------
        self._objective_label = QLabel()
        self._objective_label.setObjectName("dim")
        root.addWidget(self._objective_label)
        state.sig_objective_changed.connect(lambda _i: self._refresh_objective())

        # --- measure area ---------------------------------------------------
        area_row = QHBoxLayout()
        area_row.addWidget(QLabel("Measure:"))
        self._area = QComboBox()
        self._area.addItems(["Full frame", "ROI", "Select ROI…"])
        self._area.currentIndexChanged.connect(self._on_area_changed)
        area_row.addWidget(self._area, stretch=1)
        root.addLayout(area_row)
        live_view.sig_roi_selected.connect(self._on_roi_selected)
        if self._roi_norm is not None:
            self._area.setCurrentIndex(1)   # restores the saved measure area

        # --- buttons ---------------------------------------------------------
        btn_row = QHBoxLayout()
        self._focus_once = QPushButton("Focus once (AF-S)")
        self._focus_once.clicked.connect(self._on_focus_once)
        btn_row.addWidget(self._focus_once)
        root.addLayout(btn_row)
        btn_row2 = QHBoxLayout()
        self._abort = QPushButton("Abort")
        self._abort.setObjectName("danger")
        # NOT `connect(self._service.abort)`: clicked carries a bool, so it
        # landed in the `reason` parameter and the result message read
        # "aborted: False".
        self._abort.clicked.connect(lambda: self._service.abort())
        self._abort.setEnabled(False)
        btn_row2.addWidget(self._abort)
        self._calibrate = QPushButton("Calibrate backlash")
        self._calibrate.clicked.connect(self._service.calibrate_backlash)
        btn_row2.addWidget(self._calibrate)
        root.addLayout(btn_row2)

        # --- progress + curve ------------------------------------------------
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        root.addWidget(self._progress)
        self._curve = SharpnessCurveWidget()
        self._curve.set_um_per_step(
            float(settings.device("focus").get("um_per_step", 0.2)))
        root.addWidget(self._curve)

        # --- status lines ------------------------------------------------------
        self._phase_label = QLabel("idle")
        self._phase_label.setObjectName("dim")
        root.addWidget(self._phase_label)
        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        self._result_label.setObjectName("dim")
        root.addWidget(self._result_label)

        # --- wiring ------------------------------------------------------------
        service.sig_af_progress.connect(self._on_progress)
        service.sig_af_curve_secondary.connect(self._on_curve_secondary)
        service.sig_af_log.connect(self._on_log)
        service.sig_af_finished.connect(self._on_finished)
        service.sig_cal_finished.connect(self._on_cal_finished)
        self._refresh_objective()

    # ------------------------------------------------------------------

    def _current_roi(self) -> tuple | None:
        return self._roi_norm if self._area.currentIndex() >= 1 else None

    def _on_area_changed(self, index: int) -> None:
        if index == 2:  # "Select ROI…" arms the rubber band
            self._live_view.set_roi_selection_mode(True)
            self._result_label.setText("drag a rectangle on the live view")
        elif index == 0:
            self._live_view.set_roi(None)
        else:
            self._live_view.set_roi(self._roi_norm)

    def _on_roi_selected(self, roi_norm: tuple | None) -> None:
        self._roi_norm = roi_norm
        # Persist the selection: the measure area silently reset to "Full
        # frame" on every launch (the settings key existed and nothing
        # read it).
        self._settings.section("autofocus")["default_roi_norm"] = (
            list(roi_norm) if roi_norm else None)
        self._settings.save()
        if roi_norm is None:
            self._area.setCurrentIndex(0)
            self._result_label.setText("selection too small — using full frame")
        else:
            self._area.setCurrentIndex(1)

    def set_live_view(self, live_view) -> None:
        """Retarget the ROI rubber band at a workspace's live view.

        The panel kept the Navigation view forever, so with the Sample
        Finding tab active the band was armed on the HIDDEN view and
        dragging on the visible one did nothing."""
        if live_view is self._live_view:
            return
        try:
            self._live_view.sig_roi_selected.disconnect(self._on_roi_selected)
        except (RuntimeError, TypeError):
            pass
        self._live_view.set_roi_selection_mode(False)
        self._live_view = live_view
        live_view.sig_roi_selected.connect(self._on_roi_selected)
        if self._area.currentIndex() >= 1:
            live_view.set_roi(self._roi_norm)

    def _on_focus_once(self) -> None:
        self._curve.clear()
        self._progress.setValue(0)
        # Enable Abort immediately: the run can be cancelled even during
        # the 350 ms arm window (_on_finished disables it on every path).
        self._abort.setEnabled(True)
        self._service.start_af_s(roi_norm=self._current_roi())

    # ------------------------------------------------------------------

    def _on_progress(self, fraction: float, phase: int, score: float,
                     pos_steps: float) -> None:
        self._progress.setValue(int(fraction * 1000))
        self._abort.setEnabled(True)
        self._calibrate.setEnabled(False)
        self._phase_label.setText(
            f"{PHASE_NAMES.get(phase, 'phase')}: {score:.0f}")
        self._curve.add_point(pos_steps, score)

    def _on_curve_secondary(self, pos_steps: float, score: float) -> None:
        self._curve.add_secondary(pos_steps, score)

    def _on_log(self, message: str) -> None:
        self._result_label.setText(message)

    def _on_finished(self, result) -> None:
        self._abort.setEnabled(False)
        self._calibrate.setEnabled(True)
        self._progress.setValue(1000 if result is not None and result.success else 0)
        if result.curve:
            self._curve.clear()
            for pos, score in result.curve:
                self._curve.add_point(pos, score)
            for pos, score in getattr(result, "coarse_curve", []):
                self._curve.add_secondary(pos, score)
            self._curve.set_peak(result.best_position if result.success else None)
        um = result.best_position * self._curve.um_per_step
        status = "focused" if result.success else (
            "aborted" if result.aborted else "failed")
        self._result_label.setText(
            f"{status}: {result.message}  "
            f"(best {um:.1f} µm, score {result.best_score:.0f})")
        self._phase_label.setText("idle")

    def _on_cal_finished(self, result) -> None:
        self._calibrate.setEnabled(True)
        if result.success:
            self._result_label.setText(
                f"backlash calibrated: {result.backlash_um} µm "
                f"({result.backlash_steps} steps) — stored")
        else:
            self._result_label.setText(
                f"backlash calibration {result.message}")

    def _refresh_objective(self) -> None:
        rows = self._settings.get("objectives") or []
        index = int(self._state.objective)
        if not rows:
            return
        row = rows[min(index, len(rows) - 1)]
        # the search window = the editable max bounds × the AF speed
        # multiplier (mirrors build_config — window_um is legacy data)
        mult = row.get("af_speed_multiplier") or row.get("speed_multiplier") \
            or 1.0
        af_cfg = self._settings.section("autofocus")
        minus_um = float(af_cfg.get("window_minus_um", 500.0)) * float(mult)
        plus_um = float(af_cfg.get("window_plus_um", 500.0)) * float(mult)
        backlash_um = float(self._settings.device("focus").get("backlash_um", 0.0))
        text = (f"objective: {row.get('name', '?')}  {row.get('mag', '?')}× "
                f"NA {row.get('na', '?')}  DOF ~{row.get('dof_um', '?')} µm  "
                f"search −{minus_um:.0f}/+{plus_um:.0f} µm  "
                f"backlash {backlash_um} µm (mechanism)")
        self._objective_label.setText(text)
