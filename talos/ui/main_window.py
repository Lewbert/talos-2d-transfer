"""Main window: menu bar, workspace bar (with the objective selector as
the corner widget), the two workspaces, the shared bottom hardware strip
and the status bar (STOP ALL far left, device LEDs, controller status,
mode badge, brief app messages docked right).

Window-level keyboard focus feeds the global jog keys; Esc is the global
STOP ALL.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import talos
from talos.capture import load_capture_config, next_snapshot_path, snapshot_dir
from talos.cv.scale_bar import burn_spec
from talos.hal.registry import DEVICE_KEYS
from talos.models import StagePosition
from talos.objective_offsets import compute_offset_move
from talos.ui.auto_gain import AutoGainController
from talos.ui.af_region import AfRegionController, mirror_roi_norm
from talos.ui.calibration_context import CalibrationContext
from talos.ui.camera_profiles import (
    nav_profile,
    profile_as_props,
    profile_diff,
    scan_profile,
)
from talos.ui.theme import DANGER, WARN
from talos.ui.widgets.overlay import PHASE_NAMES
from talos.ui.widgets.focus_window import FocusWindow
from talos.ui.widgets.gamepad_indicator import GamepadIndicator
from talos.ui.widgets.hardware_strip import HardwareStrip
from talos.ui.widgets.log_window import LogWindow
from talos.ui.widgets.stage_control_window import StageControlWindow
from talos.ui.widgets.status_led import CONNECTING, ERROR, OFF, ON, StatusLED
from talos.ui.workspaces import NavigationWorkspace, SampleFindingWorkspace

_OBJECTIVES = ["5x", "10x", "20x", "50x", "100x"]

_QT_KEYSYM = {
    Qt.Key.Key_Up: "Up", Qt.Key.Key_Down: "Down",
    Qt.Key.Key_Left: "Left", Qt.Key.Key_Right: "Right",
    Qt.Key.Key_Equal: "equal", Qt.Key.Key_Plus: "plus",
    Qt.Key.Key_Minus: "minus", Qt.Key.Key_Underscore: "underscore",
    Qt.Key.Key_Shift: "Shift_L",
}


class MainWindow(QMainWindow):
    def __init__(self, manager, settings, state, parent=None, input_system=None,
                 autofocus_service=None):
        super().__init__(parent)
        self._manager = manager
        self._settings = settings
        self._state = state
        self._input = input_system
        self._autofocus = autofocus_service
        self._focus_connected = False
        self.setWindowTitle("TALOS — Transfer and Alignment Laboratory Operating System")
        self.setMinimumSize(1024, 640)
        # Keyboard jogs are global (not tied to a focused widget), so the
        # WINDOW holds focus by default: the first focusable child would
        # otherwise swallow +/−/arrow keys (user-found: input dead at
        # launch until STOP ALL was pressed once).
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # --- auto-gain (software loop; the camera profile drives it) -----
        self._autogain = AutoGainController(manager, settings, state)
        self._applied_profile = None
        self._calibration = CalibrationContext(state)
        # ONE shared AF measurement region: the right-panel AF settings,
        # the AF detail window and the live-view overlay all reflect it,
        # and autofocus itself reads the same persisted value.
        self._af_roi = AfRegionController(settings)
        # The camera flip rotates every frame 180° — remember the value the
        # ROI and the flake table were built against (see _sync_camera_flip).
        self._camera_flip = bool(settings.device("camera").get("flip", True))

        # --- workspaces -------------------------------------------------
        self._tabs = QTabWidget()
        self._navigation = NavigationWorkspace(
            manager, settings, input_system, state=state,
            autofocus_service=autofocus_service, autogain=self._autogain,
            af_roi=self._af_roi)
        self._sample_finding = SampleFindingWorkspace(
            manager, settings, state, autofocus_service=autofocus_service,
            autogain=self._autogain, calibration_context=self._calibration,
            input_system=self._input)
        self._tabs.addTab(self._navigation, "Navigation")
        self._tabs.addTab(self._sample_finding, "Sample Finding")
        # The objective selector rides the workspace bar's right corner
        # (the combo is the operator's source of truth — the nosepiece is
        # manual) and is shared by every workspace.
        self._tabs.setCornerWidget(self._build_objective_corner(),
                                   Qt.Corner.TopRightCorner)
        self._tabs.tabBar().setExpanding(False)
        self._tabs.tabBar().setElideMode(Qt.TextElideMode.ElideRight)
        root.addWidget(self._tabs, stretch=1)

        # --- shared bottom hardware strip --------------------------------
        self._strip = HardwareStrip(manager, settings, input_system, state=state)
        root.addWidget(self._strip)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

        self._build_menu_bar()
        self._build_status_bar()

        # --- wiring ------------------------------------------------------
        manager.sig_device_state.connect(self._on_device_state)
        manager.sig_event.connect(self._on_device_event)
        manager.sig_log.connect(self._log.panel.append)
        manager.sig_log.connect(self._on_log_message)
        manager.sig_stop_all_done.connect(self._on_stop_all_done)
        manager.camera.sig_frame.connect(self._navigation.live_view.show_frame)
        manager.camera.sig_frame.connect(self._sample_finding.on_frame)
        manager.camera.sig_frame.connect(
            self._sample_finding.live_view.show_frame)
        manager.camera.sig_frame.connect(self._autogain.on_frame)
        # STOP ALL must abort a running SCAN, not just the motion in flight.
        manager.sig_stop_all_done.connect(
            self._sample_finding.scan_panel.on_stop_all_done)
        manager.sig_job_done.connect(self._on_job_done)
        manager.sig_job_failed.connect(self._on_job_failed)
        self._snapshot_job: int | None = None
        self._navigation.quick_actions.sig_snapshot_requested.connect(
            self._on_snapshot)
        self._navigation.quick_actions.sig_af_requested.connect(
            self._on_quick_af)
        self._navigation.quick_actions.sig_set_stage_origin.connect(
            self._on_set_stage_origin)
        # The same origin, from the Sample Finding tab's quick access card:
        # one handler, so the value, the settings write and the label in the
        # other tab cannot disagree.
        self._sample_finding.sig_set_stage_origin.connect(
            self._on_set_stage_origin)
        self._navigation.quick_actions.sig_set_focus_origin.connect(
            self._on_set_focus_origin)
        # Workspace switching applies the workspace's camera profile.
        self._tabs.currentChanged.connect(self._on_workspace_changed)
        if input_system is not None:
            input_system.gamepad.sig_connected.connect(
                self._gamepad_indicator.set_connected)
            input_system.gamepad.sig_connected.connect(
                self._strip.trigger_bar().set_connected)
            input_system.gamepad.sig_state.connect(self._on_gamepad_state)
            input_system.sig_dpad_stage.connect(
                self._gamepad_indicator.set_dpad_stage)
            # LT+RT = autofocus once (the same path as the AF-S quick action).
            input_system.sig_af_requested.connect(self._on_quick_af)
            # The input layer's own messages (gamepad gestures, enable
            # toggles, cancelled holds) were emitted into the void: the
            # operator pressed a gesture and NOTHING anywhere said whether
            # it had been recognised.
            input_system.sig_log.connect(
                lambda message: self._log.panel.append("info", message))
            input_system.sig_log.connect(
                lambda message: self._on_log_message("info", message))
        manager.camera.sig_connected.connect(self._on_camera_connected)
        state.sig_mode_changed.connect(self._on_mode_changed)
        self._calibration.sig_changed.connect(
            lambda calib: self._for_each_live_view(
                lambda v: v.set_live_calibration(
                    self._calibration.um_per_px())))
        # The context emits in its own __init__ (before this connect) —
        # push the initial calibration once manually.
        self._for_each_live_view(
            lambda v: v.set_live_calibration(
                self._calibration.um_per_px()))
        if autofocus_service is not None:
            autofocus_service.sig_af_progress.connect(self._on_af_progress)
            autofocus_service.sig_af_finished.connect(self._on_af_finished)

        # Esc = global STOP ALL (hard-wired, not rebindable) + input latch.
        shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        shortcut.activated.connect(self._on_escape)

        self._autogain.start()
        self._restore_origins()
        self._restore_geometry()

    # ------------------------------------------------------------------
    # Chrome builders
    # ------------------------------------------------------------------

    def _build_objective_corner(self) -> QWidget:
        corner = QWidget()
        layout = QHBoxLayout(corner)
        layout.setContentsMargins(0, 2, 4, 2)
        layout.setSpacing(4)
        layout.addWidget(QLabel("Objective:"))
        self._objective = QComboBox()
        self._objective.setObjectName("objective")
        # Objective names come from the user-editable settings table; the
        # selection persists across restarts.
        rows = self._settings.get("objectives") or []
        names = [row.get("name", f"{row.get('mag', '?')}x") for row in rows] \
            or _OBJECTIVES
        self._objective.addItems(names)
        idx = int(self._settings.get("selected_objective", 0) or 0)
        idx = max(0, min(idx, len(names) - 1))
        self._objective.setCurrentIndex(idx)  # BEFORE the connect: no init save
        self._objective.currentIndexChanged.connect(self._on_objective_changed)
        self._state.set_objective(idx)  # AppState defaults to 0 and emits
        # only on change — set the persisted index
        layout.addWidget(self._objective)
        return corner

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        close_action = file_menu.addAction("&Close")
        close_action.setShortcut(QKeySequence.StandardKey.Close)
        close_action.triggered.connect(self.close)

        edit_menu = menu_bar.addMenu("&Edit")
        prefs_action = edit_menu.addAction("&Preferences…")
        prefs_action.setShortcut(QKeySequence.StandardKey.Preferences)
        prefs_action.triggered.connect(self._on_preferences)

        self._display_menu = menu_bar.addMenu("&Display")
        self._build_display_menu(self._display_menu)

        self._windows_menu = menu_bar.addMenu("&Windows")
        self._focus_window = FocusWindow(
            self._manager, self._settings, self._state, self._autofocus,
            input_system=self._input, af_roi=self._af_roi, parent=self)
        # The AF detail window carries the same settings block as the right
        # panel: either instance edits, both announce, both refresh.
        self._focus_window.sig_settings_changed.connect(
            self._navigation.af_group.widget.refresh_from_settings)
        self._navigation.af_group.widget.sig_settings_changed.connect(
            self._focus_window.refresh_settings)
        for widget in (self._navigation.af_group.widget,
                       self._focus_window.settings_widget):
            widget.sig_roi_arm_requested.connect(self._on_af_roi_armed)
        self._af_roi.sig_changed.connect(self._on_af_roi_changed)
        self._route_af_roi_to(self._navigation.live_view)
        self._focus_action = self._windows_menu.addAction("AF Detail")
        self._focus_action.setCheckable(True)
        self._focus_action.toggled.connect(self._focus_window.setVisible)
        self._focus_window.set_toggle_action(self._focus_action)

        self._stage_window = StageControlWindow(
            self._manager, self._settings, self._input,
            state=self._state, parent=self)
        self._stage_action = self._windows_menu.addAction("Stage Control")
        self._stage_action.setCheckable(True)
        self._stage_action.toggled.connect(self._stage_window.setVisible)
        self._stage_window.set_toggle_action(self._stage_action)

        self._log = LogWindow(parent=self)
        self._log_action = self._windows_menu.addAction("Log")
        self._log_action.setCheckable(True)
        self._log_action.toggled.connect(self._log.setVisible)
        self._log.set_toggle_action(self._log_action)

        # The scan and the identification chain live IN the Sample Finding
        # tab now — its own columns, its own live view. The dropper is armed
        # from the colour group there and the click is handled there too, so
        # the tab owns every step: the PRE-PROCESSED pixel is what is
        # sampled, and the panel that computed it is the one that reads it.
        self._sample_finding.sig_log.connect(
            lambda message: self._on_log_message("info", message))

        help_menu = menu_bar.addMenu("&Help")
        about_action = help_menu.addAction("&About TALOS")
        about_action.triggered.connect(self._on_about)

    def _build_display_menu(self, menu) -> None:
        """Display → live-view overlays (persisted to display.*).

        The grouped overlays live in submenus whose FIRST item is the
        on/off toggle ("Show scale bar" / "Show crosshair"), with the
        option that depends on it directly below (enabled only while the
        toggle is on).

        The toggle is an ITEM INSIDE the submenu, not the submenu's own
        action: Qt opens a submenu when its parent action is clicked and
        never triggers it, so a checkable submenu parent is permanently
        stuck (verified with QTest — that is how the scale bar and the
        crosshair used to be impossible to switch off).
        """
        display = self._settings.section("display")

        def _apply_scale_bar(on: bool) -> None:
            self._for_each_live_view(lambda v: v.set_scale_bar_enabled(on))

        scale_menu = menu.addMenu("Scale Bar")
        self._scale_bar_action = scale_menu.addAction("Show scale bar")
        self._scale_bar_action.setCheckable(True)
        self._scale_bar_action.setToolTip(
            "Calibrated bar in the frame's bottom-right corner")
        self._burn_action = scale_menu.addAction("Burn into snapshots")
        self._burn_action.setCheckable(True)
        self._burn_action.setToolTip(
            "Draw the same bar into every saved snapshot")
        self._scale_bar_action.toggled.connect(self._on_scale_bar_toggled)
        self._burn_action.toggled.connect(
            lambda on: self._on_display_toggle("burn_scale_bar",
                                               lambda _o: None, on))

        def _apply_crosshair(on: bool) -> None:
            self._for_each_live_view(lambda v: v.set_crosshair_enabled(on))

        def _apply_crosshair_ticks(on: bool) -> None:
            self._for_each_live_view(
                lambda v: v.set_crosshair_ticks_enabled(on))

        cross_menu = menu.addMenu("Crosshair")
        self._crosshair_action = cross_menu.addAction("Show crosshair")
        self._crosshair_action.setCheckable(True)
        self._crosshair_action.setToolTip(
            "Solid inverse-video crosshair through the frame centre")
        self._crosshair_ticks_action = cross_menu.addAction("Crosshair ticks")
        self._crosshair_ticks_action.setCheckable(True)
        self._crosshair_ticks_action.setToolTip(
            "Calibrated major/minor ticks along the crosshair lines")
        self._crosshair_action.toggled.connect(self._on_crosshair_toggled)
        self._crosshair_ticks_action.toggled.connect(
            lambda on: self._on_display_toggle("crosshair_ticks",
                                               _apply_crosshair_ticks, on))

        scale_on = bool(display.get("scale_bar", True))
        burn_on = bool(display.get("burn_scale_bar", False))
        _apply_scale_bar(scale_on)
        self._scale_bar_action.blockSignals(True)
        self._scale_bar_action.setChecked(scale_on)
        self._scale_bar_action.blockSignals(False)
        self._burn_action.setEnabled(scale_on)
        self._burn_action.blockSignals(True)
        self._burn_action.setChecked(burn_on and scale_on)
        self._burn_action.blockSignals(False)

        cross_on = bool(display.get("crosshair", False))
        ticks_on = bool(display.get("crosshair_ticks", False))
        # BOTH must be pushed to the views: blockSignals above suppresses
        # the toggled handlers, and relying on them meant a stored
        # crosshair-ticks=on came back CHECKED in the menu but absent from
        # the live view until the user toggled it twice.
        _apply_crosshair(cross_on)
        _apply_crosshair_ticks(ticks_on and cross_on)
        self._crosshair_action.blockSignals(True)
        self._crosshair_action.setChecked(cross_on)
        self._crosshair_action.blockSignals(False)
        self._crosshair_ticks_action.setEnabled(cross_on)
        self._crosshair_ticks_action.blockSignals(True)
        self._crosshair_ticks_action.setChecked(ticks_on and cross_on)
        self._crosshair_ticks_action.blockSignals(False)

        specs = [
            ("af_indicator", "AF Indicator", True,
             lambda on: self._for_each_live_view(
                 lambda v: v.set_af_indicator_enabled(on))),
            # Calibrated major/minor ticks on all four frame edges, µm from
            # the frame centre; needs the objective calibration.
            ("ruler", "Tick ruler", False,
             lambda on: self._for_each_live_view(
                 lambda v: v.set_ruler_enabled(on))),
        ]
        for key, label, default, apply in specs:
            action = menu.addAction(label)
            action.setCheckable(True)
            action.toggled.connect(
                lambda on, k=key, a=apply: self._on_display_toggle(k, a, on))
            initial = bool(display.get(key, default))
            apply(initial)  # the initial state must reach the live views
            action.blockSignals(True)
            action.setChecked(initial)
            action.blockSignals(False)

    def _on_crosshair_toggled(self, on: bool) -> None:
        """Crosshair on/off; the ticks option follows it (a reticle without
        a crosshair makes no sense), exactly like the scale bar's burn."""
        self._for_each_live_view(lambda v: v.set_crosshair_enabled(on))
        self._settings.section("display")["crosshair"] = bool(on)
        if not on:
            self._settings.section("display")["crosshair_ticks"] = False
            self._crosshair_ticks_action.blockSignals(True)
            self._crosshair_ticks_action.setChecked(False)
            self._crosshair_ticks_action.blockSignals(False)
            self._for_each_live_view(
                lambda v: v.set_crosshair_ticks_enabled(False))
        self._settings.save()
        self._crosshair_ticks_action.setEnabled(bool(on))

    def _on_scale_bar_toggled(self, on: bool) -> None:
        self._for_each_live_view(lambda v: v.set_scale_bar_enabled(on))
        self._settings.section("display")["scale_bar"] = bool(on)
        self._settings.save()
        self._burn_action.setEnabled(on)
        if not on:
            # the burn-in is a sub-option: unchecking the parent also
            # turns it off (and persists the off state)
            self._burn_action.blockSignals(True)
            self._burn_action.setChecked(False)
            self._burn_action.blockSignals(False)
            self._settings.section("display")["burn_scale_bar"] = False
            self._settings.save()

    def _on_display_toggle(self, key: str, apply, on: bool) -> None:
        self._settings.section("display")[key] = bool(on)
        self._settings.save()
        apply(on)

    def _for_each_live_view(self, fn) -> None:
        for view in self._live_views():
            fn(view)

    def _live_views(self) -> list:
        views = [self._navigation.live_view]
        second = getattr(self._sample_finding, "live_view", None)
        if second is not None:
            views.append(second)
        return views

    # ------------------------------------------------------------------
    # AF measurement region (drawn on the ACTIVE workspace's live view)
    # ------------------------------------------------------------------

    def _active_live_view(self):
        return (self._navigation.live_view if self._tabs.currentIndex() == 0
                else self._sample_finding.live_view)

    def _on_af_roi_armed(self) -> None:
        """'Select ROI' (either AF settings instance): arm the rubber band
        on the view the operator is actually looking at."""
        for view in self._live_views():
            view.set_roi_selection_mode(False)   # never leave one armed
        self._active_live_view().set_roi_selection_mode(True)
        self._on_log_message("info", "Drag a rectangle on the live view "
                                     "to set the AF ROI")

    def _on_af_roi_changed(self, roi) -> None:
        self._active_live_view().set_roi(roi)

    def _on_af_roi_selected(self, roi_norm) -> None:
        self._af_roi.set_roi(roi_norm)

    def _route_af_roi_to(self, view) -> None:
        """Point the ROI overlay + rubber band at a workspace's view.

        The previous view is tracked explicitly: disconnecting blind raises
        a PySide warning for every view that was never connected.
        """
        previous = getattr(self, "_roi_view", None)
        if previous is not None and previous is not view:
            previous.set_roi_selection_mode(False)
            try:
                previous.sig_roi_selected.disconnect(self._on_af_roi_selected)
            except (RuntimeError, TypeError):
                pass
        if previous is not view:
            view.sig_roi_selected.connect(self._on_af_roi_selected)
        self._roi_view = view
        view.set_roi(self._af_roi.roi())

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        self._stop_all = QPushButton("■ STOP ALL")
        self._stop_all.setObjectName("danger")
        self._stop_all.clicked.connect(self._manager.stop_all)
        bar.addWidget(self._stop_all)
        bar.addWidget(self._make_separator())

        # the live framerate sits right after the camera LED (it moved
        # here from the right-panel camera settings group)
        self._fps_label = QLabel("")
        self._fps_label.setObjectName("readout")
        self._leds: dict[str, StatusLED] = {}
        # Abbreviated on the bar (it is a glanceable row), named in full in
        # the tooltip — the same abbreviations the strip's section titles
        # use, so the two bars read as one system.
        for key, label, tip in (
                ("camera", "CAM", "Camera — Zeiss Axiocam 208 (live view)"),
                ("zolix", "XYR", "Zolix XYR sample stage"),
                ("sigmakoki", "XYZ", "SigmaKoki XYZ transfer stage"),
                ("focus", "FOCUS", "Focus stage (no limit sensor)"),
                ("yudian", "TEMP", "Yudian AI-828 temperature controller")):
            led = StatusLED(label)
            led.set_state(OFF)
            led.setToolTip(tip)
            self._leds[key] = led
            bar.addWidget(led)
            if key == "camera":
                bar.addWidget(self._fps_label)
        bar.addWidget(self._make_separator())

        self._gamepad_indicator = GamepadIndicator()
        bar.addWidget(self._gamepad_indicator)
        bar.addWidget(self._make_separator())

        # The mode badge is NOT a permanent "MANUAL" sticker: it appears
        # only while a job owns the axes (SCAN) — see _on_mode_changed.
        self._mode_badge = QLabel("")
        self._mode_badge.setObjectName("mode_badge")
        self._mode_badge.hide()
        bar.addWidget(self._mode_badge)

        # Brief app messages, docked right (the full log lives in the
        # Windows-menu Log window, hidden by default).
        self._msg_label = QLabel("")
        self._msg_label.setObjectName("dim")
        self._msg_label.setAlignment(Qt.AlignmentFlag.AlignRight
                                     | Qt.AlignmentFlag.AlignVCenter)
        bar.addPermanentWidget(self._msg_label, 1)

    @staticmethod
    def _make_separator() -> QLabel:
        sep = QLabel("|")
        sep.setObjectName("dim")
        sep.setFixedWidth(10)
        sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return sep

    # ------------------------------------------------------------------
    # Menu handlers
    # ------------------------------------------------------------------

    def _on_preferences(self) -> None:
        from PySide6.QtWidgets import QApplication

        from talos.ui.dialogs.preferences import PreferencesDialog

        dialog = PreferencesDialog(self._manager, self._settings,
                                   QApplication.instance(), self._autofocus,
                                   parent=self)
        # Apply keeps the dialog open: the calibration cache (scale bar,
        # snapshot burn) must pick up new µm/px values immediately, not
        # when the dialog is finally closed.
        applied: list[bool] = []
        dialog.sig_applied.connect(lambda: applied.append(True))
        dialog.sig_applied.connect(self._on_settings_applied)
        dialog.exec()
        if not applied:
            # Closed without Apply (Cancel or the window button) — some pages
            # write their values live, so refresh once anyway. When Apply DID
            # run this second pass was pure repetition: it re-ran the
            # reconnect sweep, which for a device whose reconnect failed means
            # another 5 s GUI freeze and a replayed warning.
            self._on_settings_applied()

    def _on_settings_applied(self) -> None:
        """Refresh everything that caches settings-derived state."""
        self._calibration.refresh()  # objectives / µm-per-px may have changed
        strip = getattr(self, "_strip", None)
        if strip is not None:
            strip.reload_settings()
        # The AF detail window's panel shows µm-per-step and the planned
        # window: both are settings-derived, and it caches neither.
        panel = getattr(getattr(self, "_focus_window", None), "panel", None)
        if panel is not None:
            panel.refresh_from_settings()
        if self._input is not None:
            # axis inversion / flip X↔Y, jog speeds, focus trigger curve
            self._input.reload_settings()
        # The Sample Finding tab caches the scan's own section (overlap,
        # settle, speed, backlash, exports) — the values it deliberately
        # does not show. Without this the panel keeps running the old ones
        # until the app restarts, which is the worst kind of stale: the
        # operator changed a number and the scan ignored it.
        refresh = getattr(self._sample_finding, "refresh_settings", None)
        if callable(refresh):
            refresh()
        self._sync_camera_flip()
        self._reconnect_changed_devices()

    def _reconnect_changed_devices(self) -> None:
        """Rebuild the connection of every device whose port / baudrate /
        slave address / timeout changed (Preferences → Hardware).

        Refused while a job owns the axes: swapping the zolix driver
        mid-scan breaks the scan's blocking job waits (60-120 s each), and
        swapping the focus driver mid-autofocus strands the service. The
        edit stays in the settings — re-apply after the job.
        """
        changed = getattr(self._manager, "connection_config_changed", None)
        reconnect = getattr(self._manager, "reconnect", None)
        if changed is None or reconnect is None:
            return  # a manager without the reconnect API (stubs/tests)
        mode = getattr(self._state, "mode", "MANUAL")
        busy_axes = mode != "MANUAL" or bool(
            self._autofocus is not None and self._autofocus.busy)
        for key in DEVICE_KEYS:
            if not changed(key):
                continue
            if busy_axes:
                self._on_log_message(
                    "warning",
                    f"{key}: connection change deferred — "
                    f"{mode.lower()} owns the axes; re-apply after the job")
                continue
            if self._input is not None:
                self._input.cancel_all_holds("device reconnect")
            reconnect(key)

    def _sync_camera_flip(self) -> None:
        """React to a camera-flip change (Preferences → Hardware → Camera).

        The flip is pushed to the running camera by the page itself; here
        the frame-space state that was built against the OLD orientation is
        carried over: the AF region is mirrored, and the detected-flake
        table is dropped (its pixel centroids are stale, and
        "go to flake" would command a mirrored physical move).
        """
        flip = bool(self._settings.device("camera").get("flip", True))
        if flip == self._camera_flip:
            return
        self._camera_flip = flip
        roi = self._af_roi.roi()
        if roi is not None:
            self._af_roi.set_roi(mirror_roi_norm(roi))
        self._sample_finding.on_camera_flip_changed()
        self._on_log_message(
            "warning",
            "Camera flip changed — the scan map and the live sample "
            "positions were reset; re-check the µm/px calibration if saved "
            "images are used for measurements")

    def _on_about(self) -> None:
        QMessageBox.about(
            self, "About TALOS",
            f"TALOS — Transfer and Alignment Laboratory Operating System\n"
            f"version {talos.__version__}\n\n"
            "Microscope control + computer vision for 2D-material work.")

    # ------------------------------------------------------------------
    # Camera profiles (per-workspace)
    # ------------------------------------------------------------------

    def _on_camera_connected(self, ok: bool) -> None:
        if not ok:
            return
        self._manager.camera.set_streaming(True)
        # Re-apply the active workspace's profile after a (re)connect.
        # The baseline MUST be dropped first: _apply_camera_profile diffs
        # against the last profile WE applied, which after a reconnect is
        # still the current profile — the diff came out empty, nothing was
        # written, and the backend's connect-time _apply_defaults values
        # silently won (exposure/gain/WB drifting on every camera replug).
        self._applied_profile = None
        profile = (nav_profile(self._settings)
                   if self._tabs.currentIndex() == 0
                   else scan_profile(self._settings))
        self._apply_camera_profile(profile)

    def _on_workspace_changed(self, index: int) -> None:
        profile = nav_profile(self._settings) if index == 0 \
            else scan_profile(self._settings)
        self._apply_camera_profile(profile)
        # The AF ROI overlay + rubber band follow the visible workspace.
        self._route_af_roi_to(self._active_live_view())

    def _apply_camera_profile(self, profile) -> None:
        # Diff against the last profile WE applied — manager.camera_props
        # is a connect-time snapshot (get_properties runs once) and would
        # re-write every key on every switch after any live edit.
        baseline = (profile_as_props(self._applied_profile)
                    if self._applied_profile is not None
                    else (self._manager.camera_props or {}))
        for name, value in profile_diff(baseline, profile):
            self._manager.submit_camera("set_property", name, value)
        self._applied_profile = profile
        self._autogain.set_enabled(profile.auto_gain)
        self._autogain.set_target(profile.auto_gain_target)
        self._autogain.note_manual_gain(profile.gain)
        group = self._active_camera_group()
        if group is not None:
            group.apply_profile(profile)

    def _active_camera_group(self):
        if self._tabs.currentIndex() == 1:
            return self._sample_finding.camera_group
        return self._navigation.camera_group

    # ------------------------------------------------------------------
    # Quick actions (snapshot / AF-S / origins)
    # ------------------------------------------------------------------

    def _on_set_stage_origin(self) -> None:
        """Store where the stage is now as the software XYR origin.

        Shared by both tabs — the Navigation quick actions and the Sample
        Finding origin card — so it lives here, with the settings write
        that keeps it across a restart.

        It used to reference a local ``position`` that was never bound, so
        the state was set, the NameError ended the slot, and the origin was
        never saved: it worked until the app was restarted, and the log
        line that says so never appeared.
        """
        payload = self._manager.last_position.get("zolix")
        if not payload:
            self._on_log_message(
                "warning", "No stage position yet — the origin was not set")
            return
        position = StagePosition.from_telemetry(payload)
        self._state.set_stage_origin(position)
        self._settings.section("origin")["xyr"] = {
            "x_pulses": position.x_pulses, "y_pulses": position.y_pulses,
            "r_pulses": position.r_pulses, "x_um": position.x_um,
            "y_um": position.y_um, "r_deg": position.r_deg}
        self._settings.save()
        self._on_log_message(
            "info", f"Stage origin stored at {position.x_um:.1f}, "
                    f"{position.y_um:.1f} µm")

    def _on_set_focus_origin(self) -> None:
        steps = int(self._manager.focus_position)
        self._state.set_focus_origin(steps)
        self._settings.section("origin")["focus"] = steps
        self._settings.save()
        self._on_log_message("info", "Focus origin stored")

    def _restore_origins(self) -> None:
        """The stored stage/focus origin. Legacy ``{"x","y","r"}`` keys are
        renamed by the settings migration (config._normalize), so the reader
        needs no fallback."""
        origin = self._settings.section("origin")
        xyr = origin.get("xyr") or {}
        if xyr:
            self._state.set_stage_origin(StagePosition(
                x_pulses=int(xyr.get("x_pulses", 0)),
                y_pulses=int(xyr.get("y_pulses", 0)),
                r_pulses=int(xyr.get("r_pulses", 0)),
                x_um=float(xyr.get("x_um", 0.0)),
                y_um=float(xyr.get("y_um", 0.0)),
                r_deg=float(xyr.get("r_deg", 0.0))))
        if origin.get("focus"):
            self._state.set_focus_origin(int(origin["focus"]))

    def _on_snapshot(self) -> None:
        cfg = load_capture_config(self._settings)
        path = next_snapshot_path(cfg)
        snapshot_dir(cfg)
        timeout_s = 25.0 if cfg.resolution == 0 else 15.0
        # the calibration is canonical per 4K-sensor pixel; a 1080p
        # capture covers 2× the µm per pixel (the bar doubles)
        capture_w = 3840 if cfg.resolution == 0 else 1920
        burn = burn_spec(self._calibration.um_per_px_at(capture_w),
                         cfg.burn_scale_bar)
        self._autogain.notify_capture_busy(True)
        self._snapshot_job = self._manager.submit_camera(
            "snapshot", path, timeout_s, cfg.resolution, burn)
        self._on_log_message("info", f"Snapshot: {path.name}…")

    # ------------------------------------------------------------------
    # AF indicator overlays
    # ------------------------------------------------------------------

    def _on_af_progress(self, fraction: float, phase: int, score: float,
                        pos_steps: float) -> None:
        label = PHASE_NAMES.get(int(phase), "running")
        self._for_each_live_view(lambda v: v.set_af_phase(int(phase), label))

    def _on_af_finished(self, result) -> None:
        self._for_each_live_view(
            lambda v: v.set_af_success(bool(result.success),
                                       bool(result.aborted)))

    def _on_quick_af(self) -> None:
        if self._autofocus is None:
            return
        if getattr(self._autofocus, "busy", False):
            self._on_log_message("warning", "Autofocus already running")
            return
        self._on_log_message("info", "Autofocus requested")
        bounds = self._navigation.af_group.bounds_steps(
            self._manager.focus_position,
            float(self._settings.device("focus").get("um_per_step", 0.2)))
        self._autofocus.start_af_s(bounds=bounds)

    def _on_job_done(self, job_id: int, result) -> None:
        if job_id == self._snapshot_job:
            self._snapshot_job = None
            self._autogain.notify_capture_busy(False)
            self._on_log_message("info", f"Snapshot saved: {result}")

    def _on_job_failed(self, job_id: int, method: str, error: str) -> None:
        if job_id == self._snapshot_job:
            self._snapshot_job = None
            self._autogain.notify_capture_busy(False)
            self._on_log_message("error", f"Snapshot failed: {error}")

    # ------------------------------------------------------------------
    # Window geometry
    # ------------------------------------------------------------------

    def _restore_geometry(self) -> None:
        qs = QSettings("TALOS", "TALOS")
        geo = qs.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
            if qs.value("maximized", False) in (True, "true"):
                self.showMaximized()
        else:
            # First launch: the layout targets 1080p 16:9 — go maximized.
            self.showMaximized()

    def stop_workers(self) -> None:
        """Stop the threads the window owns, without closing anything.

        The tab owns the detection thread and any scan worker, and neither
        stops itself. Calling this from ``closeEvent`` alone was not
        enough: the app has shutdown paths that never close the window
        (the headless screenshot rig, the autoquit smoke hook), and there
        Qt destroys a *running* QThread on the way out — which aborts the
        process after a clean exit line. Idempotent.
        """
        self._sample_finding.shutdown()

    def closeEvent(self, event) -> None:  # noqa: N802
        qs = QSettings("TALOS", "TALOS")
        qs.setValue("geometry", self.saveGeometry())
        qs.setValue("maximized", self.isMaximized())
        self.stop_workers()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Keyboard feeding (reference key map lives in ActionResolver)
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.setFocus()

    def _on_objective_changed(self, idx) -> None:
        prev = self._state.objective
        self._state.set_objective(idx)
        self._settings.update("selected_objective", idx)
        self._settings.save()
        self._apply_objective_focus_offset(prev, idx)
        self.setFocus()  # combo keeps focus after a click — hand it back

    def _apply_objective_focus_offset(self, prev_idx: int, new_idx: int) -> None:
        """Compensate the objectives' focus-length differences: move the
        focus by the Z-offset delta (µm → steps). Positive offset = the
        displayed position increases; the firmware soft limits (SLIM)
        remain the safety net."""
        if not self._settings.get("objective_offsets_enabled", True):
            return
        if getattr(self._autofocus, "busy", False) or not self._focus_connected:
            return
        if self._manager.device("focus") is None:
            return
        rows = self._settings.get("objectives") or []
        old_row = rows[prev_idx] if 0 <= prev_idx < len(rows) else None
        new_row = rows[new_idx] if 0 <= new_idx < len(rows) else None
        focus_cfg = self._settings.device("focus")
        um_per_step = float(focus_cfg.get("um_per_step", 0.2))
        move = compute_offset_move(old_row, new_row, um_per_step, focus_cfg)
        if move is None:
            return
        steps, speed = move
        self._manager.submit("focus", "move_rel", steps, speed)
        self._on_log_message(
            "info", f"Objective focus offset: {steps:+d} steps "
            f"({steps * um_per_step:+.1f} µm)")

    def _on_escape(self) -> None:
        """Esc = the global STOP ALL. The input system owns the whole
        sequence (latch + drop the on-screen holds + stop the axes): calling
        ``manager.stop_all()`` here as well issued a second round of stop
        jobs per press."""
        if self._input is not None:
            self._input.on_escape()
        else:
            self._manager.stop_all()   # no input layer (tests/stubs)

    def _keysym(self, event) -> str | None:
        key = event.key()
        if key in _QT_KEYSYM:
            return _QT_KEYSYM[key]
        text = event.text()
        if text and text.isalpha():
            return text.lower()
        return None

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.isAutoRepeat():
            return  # repeats must not re-enter the key map (press-time churn)
        keysym = self._keysym(event)
        if keysym is not None and self._input is not None:
            self._input.key_down(keysym)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if event.isAutoRepeat():
            return
        keysym = self._keysym(event)
        if keysym is not None and self._input is not None:
            self._input.key_up(keysym)
        super().keyReleaseEvent(event)

    # ------------------------------------------------------------------
    # Telemetry / messages
    # ------------------------------------------------------------------

    def _on_mode_changed(self, mode: str) -> None:
        """Show the mode badge only when a job owns the axes.

        It used to be a permanent "MANUAL" sticker — noise that said
        nothing (MANUAL is the resting state, and every manual input is
        obvious from the motion itself). A non-MANUAL mode means the jog
        inputs are gated, which is worth a badge.
        """
        busy = mode != "MANUAL"
        self._mode_badge.setText(mode.upper() if busy else "")
        self._mode_badge.setToolTip(
            f"{mode} owns the axes — manual jog input is gated"
            if busy else "")
        self._mode_badge.setVisible(busy)
        # The nosepiece is manual and unsensed: changing the objective
        # mid-scan would silently invalidate the FOV the running scan is
        # tiling at, and the objective id the manifest records.
        self._objective.setEnabled(not busy)

    def _on_gamepad_state(self, state) -> None:
        self._strip.trigger_bar().set_state(
            state.left_trigger, state.right_trigger)

    def _on_log_message(self, level: str, message: str) -> None:
        """Brief app message docked right (full text in the tooltip)."""
        elided = self._msg_label.fontMetrics().elidedText(
            message, Qt.TextElideMode.ElideRight, 640)
        self._msg_label.setText(elided)
        self._msg_label.setToolTip(message)
        if level in ("warning",):
            self._msg_label.setStyleSheet(f"color: {WARN};")
        elif level in ("error", "critical"):
            self._msg_label.setStyleSheet(f"color: {DANGER};")
        else:
            self._msg_label.setStyleSheet("")

    def _on_stop_all_done(self) -> None:
        self._on_log_message("info", "All stages stopped ✔")

    def _on_device_state(self, key: str, payload: dict) -> None:
        led = self._leds.get(key)
        if led is not None:
            if payload.get("connecting"):
                # A reconnect is swapping the driver (Preferences → Apply).
                led.set_state(CONNECTING)
            elif payload.get("connected") is True:
                led.set_state(ON)
            elif payload.get("connected") is False:
                led.set_state(ERROR)
            elif payload.get("device"):
                led.set_state(ON)  # first telemetry implies alive
        if key == "camera":
            if payload.get("connected") is False:
                self._fps_label.setText("")
            elif payload.get("fps") is not None:
                self._fps_label.setText(f"{payload['fps']:.1f} fps")
        if key == "focus" and payload.get("connected") is not None:
            self._focus_connected = bool(payload["connected"])
        if key in ("zolix", "sigmakoki", "focus", "yudian"):
            self._strip.update_telem(key, payload)
            self._navigation.update_telem(key, payload)
            self._stage_window.update_telem(key, payload)
            self._sample_finding.update_telem(key, payload)

    def _on_device_event(self, key: str, event: str, payload: dict) -> None:
        led = self._leds.get(key)
        if led is not None and "failed" in event:
            led.set_state(ERROR)
        # Substring by design, but over the WORDS we mean (stop / estop /
        # halt): the test used to be `"es" in event.lower() or ...`, so any
        # future event containing "es" — reset, resume, message — would have
        # started reporting itself for no reason.
        if any(word in event.lower() for word in ("stop", "estop", "halt")):
            self._on_log_message("info", f"{key}: {event}")
