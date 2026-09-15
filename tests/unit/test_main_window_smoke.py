"""MainWindow chrome smoke: menu bar, workspace corner widget, status
bar, hidden LogWindow — with a stub manager (no hardware, no services).
"""

import pytest
from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtWidgets import QApplication

from talos.app import AppState
from talos.ui.main_window import MainWindow


class StubCamera(QObject):
    sig_frame = Signal(object)
    sig_connected = Signal(bool)


class StubManager(QObject):
    sig_device_state = Signal(str, dict)
    sig_event = Signal(str, str, dict)
    sig_log = Signal(str, str)
    sig_stop_all_done = Signal()
    sig_job_done = Signal(int, object)
    sig_job_failed = Signal(int, str, str)

    def __init__(self):
        super().__init__()
        self.camera = StubCamera()
        self.camera_props = {}
        self.focus_position = 0
        self._enabled: dict[str, bool] = {}
        self.stopped = 0
        self.camera_submits: list = []
        self.submits: list = []
        self._job = 100
        self._focus_dev = object()

    def stop_all(self):
        self.stopped += 1

    def device(self, key):
        return self._focus_dev if key == "focus" else None

    def submit(self, device_key, method, *args, **kwargs):
        self.submits.append((device_key, method, *args))
        return self._job

    def submit_camera(self, *args):
        self.camera_submits.append(args)
        self._job += 1
        return self._job

    def is_enabled(self, key):
        return self._enabled.get(key, True)

    def set_enabled(self, key, on):
        # Mirrors InstrumentManager: the gate is broadcast so every enable
        # control (strip checkbox, Stage Control panel) follows.
        self._enabled[key] = on
        self.sig_device_state.emit(key, {"enabled": on})


class StubAutofocusService(QObject):
    sig_af_progress = Signal(float, int, float, float)
    sig_af_curve_secondary = Signal(float, float)
    sig_af_log = Signal(str)
    sig_af_finished = Signal(object)
    sig_cal_finished = Signal(object)

    def abort(self):
        pass

    def calibrate_backlash(self):
        pass


class FakeSettings:
    def __init__(self):
        self.data = {
            "selected_objective": 0,
            "objectives": [
                {"name": "5x", "mag": 5, "na": 0.15, "dof_um": 28.0,
                 "window_um": 1000.0, "backlash_um": 0.0,
                 "coarse_step_um": 5.0, "fine_step_um": 1.0,
                 "af_speed_multiplier": 1.0, "focus_manual_multiplier": 1.0,
                 "stage_speed_multiplier": 1.0, "px_um": 0.0},
            ],
            "devices": {
                "camera": {"exposure_us": 40000.0, "gain": 20.0,
                           "white_balance": "Once", "color_temperature": 5500},
                "focus": {"min_speed": 50, "max_speed": 2000, "gamma": 2.2,
                          "deadzone": 0.05, "invert": False,
                          "um_per_step": 0.2},
                "zolix": {"um_per_pulse_xy": 0.625, "um_per_pulse_r": 0.00125},
                "sigmakoki": {"um_per_step_xy": 0.5, "um_per_step_z": 0.25},
                "yudian": {"safety_lo_c": -100.0, "safety_hi_c": 400.0,
                           "presets": []},
            },
            "autofocus": {"coarse_speed_base_um_s": 100.0},
            "scan": {"overlap": 0.1},
            "display": {"scale_bar": True, "burn_scale_bar": False,
                        "af_indicator": True, "crosshair": False},
        }
        self.saved = 0

    def device(self, key):
        return self.data["devices"].setdefault(key, {})

    def section(self, key):
        return self.data.setdefault(key, {})

    def get(self, key, default=None):
        return self.data.get(key, default)

    def update(self, key, value):
        self.data[key] = value

    def save(self):
        self.saved += 1


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def window(qapp, tmp_path, monkeypatch):
    # Keep QSettings away from the real registry.
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      str(tmp_path))
    # The CalibrationContext must not touch the real appdata DB.
    from talos.ui import calibration_context as cc_mod
    monkeypatch.setattr(cc_mod, "get_calibration_db_path",
                        lambda: tmp_path / "cal.db")
    manager = StubManager()
    settings = FakeSettings()
    state = AppState()
    w = MainWindow(manager, settings, state, input_system=None,
                   autofocus_service=StubAutofocusService())
    yield w
    w._log.close()  # destroy the hide-not-destroy dialog
    w._focus_window.close()
    w._stage_window.close()


def test_menu_bar_structure(window):
    menu_bar = window.menuBar()
    titles = [a.text() for a in menu_bar.actions()]
    assert titles == ["&File", "&Edit", "&Display", "&Windows", "&Help"]
    windows = [m for m in (a.menu() for a in menu_bar.actions())
               if m is not None and m.title() == "&Windows"][0]
    win_actions = [a.text() for a in windows.actions()]
    assert "AF Detail" in win_actions
    assert "Stage Control" in win_actions
    assert "Log" in win_actions
    assert all(a.isCheckable() for a in windows.actions())


def test_objective_combo_in_workspace_corner(window):
    corner = window._tabs.cornerWidget()
    assert corner is not None
    combo = corner.findChild(type(window._objective))
    assert combo is window._objective
    assert combo.count() == 1  # the single fake objective row


def test_status_bar_chrome(window):
    bar = window.statusBar()
    assert window._stop_all.parent() is bar
    assert all(led.parent() is bar for led in window._leds.values())
    assert window._gamepad_indicator.parent() is bar
    assert window._mode_badge.parent() is bar
    assert window._msg_label.parent() is bar
    assert window._fps_label.parent() is bar
    # abbreviated LEDs, full names in the tooltips
    labels = [led.text for led in window._leds.values()]
    assert labels == ["CAM", "XYR", "XYZ", "FOCUS", "TEMP"]
    assert all(led.toolTip() for led in window._leds.values())


def test_mode_badge_only_shows_when_a_job_owns_the_axes(window):
    """It used to be a permanent "MANUAL" sticker — noise that said
    nothing. A non-MANUAL mode gates the jog inputs, so that is worth a
    badge."""
    badge = window._mode_badge
    assert badge.isVisible() is False and badge.text() == ""
    window._state.set_mode("SCAN")
    assert badge.isVisible() and badge.text() == "SCAN"
    assert "gated" in badge.toolTip()
    window._state.set_mode("MANUAL")
    assert badge.isVisible() is False


def test_camera_fps_readout(window):
    window._on_device_state("camera", {"fps": 19.93})
    assert window._fps_label.text() == "19.9 fps"
    # disconnect clears it; a props payload (no fps key) leaves it alone
    window._on_device_state("camera", {"connected": False})
    assert window._fps_label.text() == ""
    window._on_device_state("camera", {"fps": 7.5})
    assert window._fps_label.text() == "7.5 fps"
    window._on_device_state("camera", {"backend": "smartcam"})
    assert window._fps_label.text() == "7.5 fps"


def test_log_window_hidden_by_default(window):
    assert not window._log.isVisible()
    assert not window._log_action.isChecked()


def test_stop_all_button_wired(window):
    window._stop_all.click()
    assert window._manager.stopped == 1


def test_status_message_elided(window):
    window._on_log_message("info", "a short message")
    assert window._msg_label.text() == "a short message"
    long = "x" * 4000
    window._on_log_message("error", long)
    assert len(window._msg_label.text()) < 4000
    assert window._msg_label.toolTip() == long


def _second_row(z_offset_um=20.0):
    return {"name": "10x", "mag": 10, "na": 0.30,
            "af_speed_multiplier": 0.25, "focus_manual_multiplier": 0.25,
            "stage_speed_multiplier": 0.5, "px_um": 0.0,
            "z_offset_um": z_offset_um}


def test_objective_switch_applies_focus_offset(window):
    window._settings.data["objectives"].append(_second_row())
    window._on_device_state("focus", {"connected": True})
    assert window._focus_connected is True
    window._on_objective_changed(1)
    # +20 µm ÷ 0.2 µm/step = +100 steps at 2000 × 0.25 = 500 steps/s
    assert window._manager.submits == [("focus", "move_rel", 100, 500)]
    assert "Objective focus offset" in window._msg_label.text()


def test_offset_disabled_does_not_move(window):
    window._settings.data["objective_offsets_enabled"] = False
    window._settings.data["objectives"].append(_second_row())
    window._on_device_state("focus", {"connected": True})
    window._on_objective_changed(1)
    assert window._manager.submits == []


def test_offset_skipped_when_focus_disconnected(window):
    window._settings.data["objectives"].append(_second_row())
    # no focus connected payload → _focus_connected stays False
    window._on_objective_changed(1)
    assert window._manager.submits == []


def test_offset_zero_delta_moves_nothing(window):
    window._settings.data["objectives"].append(_second_row(z_offset_um=0.0))
    window._on_device_state("focus", {"connected": True})
    window._on_objective_changed(1)
    assert window._manager.submits == []


def test_wb_checkbox_reflects_state_and_persists(window):
    group = window._navigation.camera_group
    # a stored "Once" is NOT Continuous → the checkbox reads unchecked
    assert group._wb_auto.isChecked() is False
    group._wb_auto.setChecked(True)
    submits = window._manager.camera_submits
    assert ("set_property", "white_balance", "Continuous") in submits
    assert window._settings.data["devices"]["camera"]["white_balance"] \
        == "Continuous"


def test_wb_once_button_submits_and_unchecks(window):
    group = window._navigation.camera_group
    group._wb_auto.setChecked(True)
    window._manager.camera_submits.clear()
    group._wb_once.click()
    submits = window._manager.camera_submits
    assert ("set_property", "white_balance", "Once") in submits
    # the camera is Off after the one-shot — the stored state follows
    assert group._wb_auto.isChecked() is False
    assert window._settings.data["devices"]["camera"]["white_balance"] == "Off"


def test_scan_camera_group_locks_wb_off(window):
    group = window._sample_finding.camera_group
    # WB is forced off on scans (detection needs stable color): the
    # continuous-AWB checkbox does not EXIST there…
    assert group._wb_auto.isChecked() is False
    assert group._wb_auto.isHidden() is True
    # …but the one-shot buttons stay for the pre-scan adjustment
    assert group._wb_once.isEnabled() is True
    assert group._gain_once.isHidden() is False


def test_gain_once_button_wired(window, monkeypatch):
    calls = []
    monkeypatch.setattr(window._autogain, "once", lambda: calls.append(1))
    window._navigation.camera_group._gain_once.click()
    assert calls == [1]


def test_scan_camera_group_hides_continuous_auto_gain(window):
    group = window._sample_finding.camera_group
    # the scan profile has no CONTINUOUS auto anything: the checkbox and
    # the target stay hidden, only the one-shot button remains
    assert group._auto_gain.isHidden() is True
    assert group._gain_target.isHidden() is True
    assert group._gain_once.isHidden() is False


def test_right_panel_groups_are_collapsible(window):
    from talos.ui.widgets.collapsible import CollapsibleGroup

    nav = window._navigation
    sections = nav.findChildren(CollapsibleGroup)
    assert len(sections) >= 4  # Capture / Camera / Autofocus / Temperature
    titles = {s._header.text()[2:] for s in sections}
    assert {"Capture", "Camera", "Autofocus", "Temperature"} <= titles
    scan = window._sample_finding
    scan_sections = scan.findChildren(CollapsibleGroup)
    scan_titles = {s._header.text()[2:] for s in scan_sections}
    assert {"Camera", "Scan & Detection"} <= scan_titles


def test_telemetry_updates_strip(window):
    from dataclasses import asdict

    from talos.models import StagePosition, StageStatus

    window._on_device_state("zolix", {
        "connected": True,
        "status": asdict(StageStatus()),
        "position": asdict(StagePosition(x_um=1.25, y_um=2.5, r_deg=0.0)),
    })
    assert window._leds["zolix"].state() == "on"
    # fixed-width fields (the numbers must not jitter as digits change)
    text = window._strip._xyr._pos.text()
    assert "1.2" in text and "2.5" in text and text.endswith("°")
    assert window._strip._xyr._moving.text() == ""      # slot reserved
    assert window._strip._xyz._moving.text() == ""


def test_enable_gate_reaches_every_checkbox(window):
    """The gamepad's Start toggles a stage's enable gate via the manager,
    which broadcasts it: the strip's checkbox and the Stage Control panel's
    must both follow, without re-emitting the toggle back."""
    strip_box = window._strip._xyr._enable
    panel_box = window._stage_window._panels["zolix"]._enable
    emitted = []
    strip_box.toggled.connect(emitted.append)

    window._manager.set_enabled("zolix", False)
    assert strip_box.isChecked() is False
    assert panel_box.isChecked() is False
    assert emitted == [], "a programmatic sync must not re-emit toggled"
    window._manager.set_enabled("zolix", True)
    assert strip_box.isChecked() is True
    assert window._manager.is_enabled("zolix") is True


def test_scan_plan_preview_and_progress(window):
    """The Sample Finding live view previews the configured grid and
    highlights the row the scan is on (the indicator is a schematic, so
    only its geometry and the row maths matter)."""
    scan = window._sample_finding
    plan = scan.live_view._scan_plan
    assert plan is not None
    width = scan._grid_w.value()
    height = scan._grid_h.value()
    from talos.cv.scan import grid_shape
    cols, rows = grid_shape(scan._scan_params(),
                            (scan._fov_x.value(), scan._fov_y.value()))
    assert (plan.cols, plan.rows) == (cols, rows)
    assert f"{width:.0f}" in plan.detail
    assert plan.active_row == -1                    # preview, not running

    scan._on_scan_progress(1, cols * rows)          # first waypoint
    assert scan.live_view._scan_plan.active_row == 0
    assert scan.live_view._scan_plan.active_col == 0
    scan._on_scan_progress(cols + 2, cols * rows)   # second row
    assert scan.live_view._scan_plan.active_row == 1
    assert scan.live_view._scan_plan.active_col == 1

    # editing the grid re-derives the preview
    scan._grid_w.setValue(width * 2)
    scan._refresh_scan_plan()
    assert scan.live_view._scan_plan.active_row == -1
    assert scan.live_view._scan_plan.cols >= cols


def test_settings_applied_refreshes_every_cached_consumer(window):
    """Preferences Apply must push the new values into everything that
    caches settings-derived state: the calibration context, the strip's
    scale factors, the input maps and the camera-flip bookkeeping."""
    window._on_settings_applied()   # must not raise with a stub manager
    window._settings.device("camera")["flip"] = False
    window._on_settings_applied()
    assert window._camera_flip is False


def test_snapshot_flow_submits_4k_and_clears_busy(window, tmp_path):
    # Redirect the snapshot dir into tmp (the capture settings default to
    # the appdata snapshots folder otherwise).
    window._settings.data.setdefault("capture", {})["dir"] = str(tmp_path)
    window._settings.data["capture"]["resolution"] = 0
    window._on_snapshot()
    submits = window._manager.camera_submits
    assert submits and submits[-1][0] == "snapshot"
    path, timeout_s, resolution, burn = submits[-1][1:]
    assert path.parent == tmp_path
    assert timeout_s == 25.0  # the 4K timeout
    assert resolution == 0
    assert burn is None  # wired in Commit 3
    assert window._autogain._capture_busy is True
    window._manager.sig_job_done.emit(window._snapshot_job, path)
    assert window._autogain._capture_busy is False
    assert "Snapshot saved" in window._msg_label.text()


def test_snapshot_burn_uses_canonical_sensor_scale(window, tmp_path):
    # The empty-DB fallback = 2 µm sensor pixels / 5x = 0.4 µm/px
    # (CANONICAL: per 4K-sensor pixel). A 4K capture burns with it
    # as-is; a 1080p capture doubles it (1080p pixels cover 2× the µm).
    window._settings.data.setdefault("capture", {})["dir"] = str(tmp_path)
    window._settings.data.setdefault("display", {})["burn_scale_bar"] = True
    window._settings.data["capture"]["resolution"] = 0
    window._on_snapshot()
    _path, _t, res, burn = window._manager.camera_submits[-1][1:]
    assert res == 0
    assert burn["um_per_px"] == pytest.approx(0.4)
    window._settings.data["capture"]["resolution"] = 1
    window._on_snapshot()
    _path, _t, res, burn = window._manager.camera_submits[-1][1:]
    assert res == 1
    assert burn["um_per_px"] == pytest.approx(0.8)


def test_snapshot_failure_clears_busy(window, tmp_path):
    window._settings.data.setdefault("capture", {})["dir"] = str(tmp_path)
    window._on_snapshot()
    job_id = window._snapshot_job
    window._manager.sig_job_failed.emit(job_id, "snapshot", "boom")
    assert window._autogain._capture_busy is False
    assert "Snapshot failed" in window._msg_label.text()


def test_af_progress_reaches_indicator_and_panel(window):
    """Regression: the shared PHASE_NAMES import must keep BOTH consumers
    working (the overlay indicator and the AF panel's phase label)."""
    window._autofocus.sig_af_progress.emit(0.5, 2, 100.0, 0.0)
    view = window._navigation.live_view
    assert view._af_phase == 2
    assert view._af_label == "fine sweep"
    panel = window._focus_window.panel
    assert "fine sweep" in panel._phase_label.text()


def test_af_finished_sets_indicator(window):
    from talos.cv.autofocus import AutofocusResult

    window._autofocus.sig_af_finished.emit(AutofocusResult(
        best_position=0, best_score=1.0, success=True, aborted=False,
        message="focused", phase="done"))
    view = window._navigation.live_view
    assert view._af_success is True
    assert view._af_label == "focused"
    window._autofocus.sig_af_finished.emit(AutofocusResult(
        best_position=0, best_score=0.0, success=False, aborted=True,
        message="aborted", phase="coarse"))
    assert view._af_success is False
