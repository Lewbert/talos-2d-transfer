"""MainWindow chrome smoke: menu bar, workspace corner widget, status
bar, hidden LogWindow — with a stub manager (no hardware, no services).
"""

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtWidgets import QApplication

from talos.app import AppState
from talos.cv.identify import valid_hex
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
        #: The camera's current canonical properties — the real manager keeps
        #: this live (get_properties, then every set_property: see
        #: CameraProxy._note_property), and the per-workspace profiles diff
        #: against it.
        self.camera_props: dict = {}
        self.focus_position = 0
        # The real manager publishes the last telemetry per device; the
        # scan window reads the zolix position to place the map footprint.
        self.last_position: dict[str, dict] = {}
        self.frame_slot = None
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
        if len(args) >= 3 and args[0] == "set_property":
            # mirrors the proxy: a write is a change to what the camera has
            self.camera_props[str(args[1])] = args[2]
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

    def __init__(self):
        super().__init__()
        self.starts: list = []
        self.busy = False

    def start_af_s(self, roi_norm=None, bounds=None):
        if self.busy:
            return False
        self.starts.append({"roi": roi_norm, "bounds": bounds})
        return True

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
    # The tab owns a detection thread; without this it outlives the test
    # and Qt destroys it mid-run at interpreter teardown.
    w.stop_workers()


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


def _display_actions(window) -> dict:
    """Flat items of the Display menu plus every submenu ITEM, by label."""
    display = [a.menu() for a in window.menuBar().actions()
               if a.text() == "&Display"][0]
    found: dict = {}
    for action in display.actions():
        if action.isSeparator():
            continue
        found[action.text()] = action
        if action.menu() is not None:
            for child in action.menu().actions():
                if not child.isSeparator():
                    found[child.text()] = child
    return found


def test_overlay_toggles_work_and_are_never_submenu_parents(window):
    """Every overlay toggle must be CLICKABLE.

    Regression: the scale bar and the crosshair were checkable actions that
    OWNED A SUBMENU. Qt opens a submenu instead of triggering its parent
    action (verified with QTest), so neither could ever be switched off.
    The toggles now live INSIDE the submenu ("Show …").
    """
    actions = _display_actions(window)
    assert set(actions) == {"Scale Bar", "Show scale bar",
                            "Burn into snapshots", "Crosshair",
                            "Show crosshair", "Crosshair ticks",
                            "AF Indicator", "Tick ruler"}
    for label in ("Show scale bar", "Burn into snapshots", "Show crosshair",
                  "Crosshair ticks", "AF Indicator", "Tick ruler"):
        action = actions[label]
        assert action.isCheckable(), label
        assert action.menu() is None, f"{label} owns a submenu — untoggleable"
    # the submenus themselves are plain openers
    for label in ("Scale Bar", "Crosshair"):
        assert actions[label].menu() is not None
        assert actions[label].isCheckable() is False

    # …and each toggle really drives its overlay
    actions["Show scale bar"].trigger()
    assert window._navigation.live_view._scale_bar_enabled is False
    actions["Show scale bar"].trigger()
    assert window._navigation.live_view._scale_bar_enabled is True

    actions["Show crosshair"].trigger()
    assert window._navigation.live_view._crosshair_display is True
    actions["Crosshair ticks"].trigger()
    assert window._navigation.live_view._crosshair_ticks is True
    actions["Tick ruler"].trigger()
    assert window._navigation.live_view._ruler_enabled is True
    actions["Show crosshair"].trigger()
    assert window._navigation.live_view._crosshair_display is False
    # switching the crosshair off takes its ticks with it
    assert window._navigation.live_view._crosshair_ticks is False
    assert actions["Crosshair ticks"].isChecked() is False
    assert actions["Crosshair ticks"].isEnabled() is False


def test_dependent_overlay_options_follow_their_parent(window):
    """Burn needs a scale bar, ticks need a crosshair: both are disabled
    while the parent is off, and the state persists."""
    actions = _display_actions(window)
    assert actions["Burn into snapshots"].isEnabled() is True
    actions["Show scale bar"].trigger()          # off
    assert actions["Burn into snapshots"].isEnabled() is False
    assert actions["Burn into snapshots"].isChecked() is False
    assert window._settings.section("display")["burn_scale_bar"] is False
    actions["Show scale bar"].trigger()          # back on
    assert actions["Burn into snapshots"].isEnabled() is True


def test_stored_overlays_are_applied_at_launch(qapp, tmp_path, monkeypatch):
    """Regression: a stored crosshair_ticks came back CHECKED in the menu
    but absent from the live view — the build path set the check state with
    signals blocked and never pushed the value, so only a manual
    uncheck/recheck made it appear."""
    from talos.ui import calibration_context as cc_mod

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      str(tmp_path))
    monkeypatch.setattr(cc_mod, "get_calibration_db_path",
                        lambda: tmp_path / "cal.db")
    settings = FakeSettings()
    settings.section("display").update({"scale_bar": False,
                                        "crosshair": True,
                                        "crosshair_ticks": True,
                                        "burn_scale_bar": True})
    window = MainWindow(StubManager(), settings, AppState(),
                        input_system=None,
                        autofocus_service=StubAutofocusService())
    view = window._navigation.live_view
    assert view._scale_bar_enabled is False
    assert view._crosshair_display is True
    assert view._crosshair_ticks is True


def test_ticks_cannot_be_stored_without_the_crosshair(qapp, tmp_path,
                                                      monkeypatch):
    """The dependent option is meaningless on its own — a stored
    ticks=True with crosshair=False must not light up."""
    from talos.ui import calibration_context as cc_mod

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      str(tmp_path))
    monkeypatch.setattr(cc_mod, "get_calibration_db_path",
                        lambda: tmp_path / "cal.db")
    settings = FakeSettings()
    settings.section("display").update({"crosshair": False,
                                        "crosshair_ticks": True})
    window = MainWindow(StubManager(), settings, AppState(),
                        input_system=None,
                        autofocus_service=StubAutofocusService())
    assert window._navigation.live_view._crosshair_ticks is False


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
    # The Sample Finding tab carries its own camera/CV column now: the scan
    # camera profile, the pre-processing chain and the identification gates,
    # all collapsible and all under the tab's own state key.
    tab = window._sample_finding
    tab_titles = {s._header.text()[2:]
                  for s in tab.findChildren(CollapsibleGroup)}
    assert tab_titles == {"Camera", "Pre-processing", "Identification"}
    for section in tab.findChildren(CollapsibleGroup):
        assert section._settings is window._settings
        assert section._state_key == "scan"


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
    assert "1.2" in text and "2.5" in text and "°" in text
    # ...and the ink is CENTRED in the elastic box: the fixed-width numeric
    # fields pad from the left, and QLabel centres the spaces too, so the
    # padding is re-balanced (otherwise the numbers sat ~6 px right of the
    # box's middle — measured).
    lead = len(text) - len(text.lstrip())
    trail = len(text) - len(text.rstrip())
    assert abs(lead - trail) <= 1
    # The status word is always visible and only changes colour (grey →
    # green), in the same style as the focus section's IDLE/CONT.
    for section in (window._strip._xyr, window._strip._xyz):
        assert section._moving.text() == "IDLE"
        assert section._moving.objectName() == "strip_mov_idle"


def _strip_panels(window):
    strip = window._strip
    focus = strip._focus_state.parentWidget()
    temp = strip._temp_pv.parentWidget()
    return strip, [strip._xyr, strip._xyz, focus, temp]


def test_strip_panels_share_one_fixed_gap_and_one_elastic_field(window, qapp):
    """The strip's layout rule (the user's): components separated by ONE
    fixed gap, and exactly ONE elastic component per panel that takes the
    leftover. Anything else re-opens the holes the layout keeps growing
    back."""
    from talos.ui.widgets.hardware_strip import FIELD_GAP

    window.resize(1920, 1080)
    window.show()
    qapp.processEvents()
    strip, panels = _strip_panels(window)
    try:
        assert strip.width() > 1000
        for panel in panels:
            body = panel.layout().itemAt(1).layout()
            assert body.spacing() == FIELD_GAP
        stage, xyz, focus, temp = panels
        for panel in (stage, xyz, focus):
            stretched = [i for i in range(panel.layout().itemAt(1)
                                          .layout().count())
                         if panel.layout().itemAt(1).layout().stretch(i)]
            assert len(stretched) == 1, "one elastic field per panel"
        # the elastic fields: the readout / the trigger bar / all three
        # TEMP values, which share the row (the user asked for all three)
        assert stage._pos.width() > 200
        assert window._strip._triggers.width() > 100
        temp_body = temp.layout().itemAt(1).layout()
        assert [temp_body.stretch(i) for i in range(temp_body.count())] \
            == [1, 1, 1]
    finally:
        window.hide()


def test_strip_panels_are_the_same_width_and_temp_fields_match(window, qapp):
    """Four equal panels at every window width, and TEMP's three value
    fields equal to each other (both were checked with a Qt geometry dump;
    this pins them so a layout tweak cannot silently break equality)."""
    window.show()
    try:
        for width in (1024, 1280, 1920):
            window.resize(width, 800)
            qapp.processEvents()
            _, panels = _strip_panels(window)
            widths = [p.width() for p in panels]
            assert max(widths) - min(widths) <= 1, (width, widths)
            fields = [window._strip._temp_pv, window._strip._temp_sv,
                      window._strip._temp_out]
            field_w = [f.width() for f in fields]
            assert max(field_w) - min(field_w) <= 1, (width, field_w)
    finally:
        window.hide()


def test_strip_rows_fit_inside_their_panel_at_the_window_floor(window, qapp):
    """At the 1024 px minimum the strip is tight: every row must still ADD
    UP inside its own panel, or a box overflows the frame and loses its
    border (the dots drop out to buy the room — see set_compact)."""
    window.resize(1024, 640)
    window.show()
    qapp.processEvents()
    try:
        _, panels = _strip_panels(window)
        for panel in panels:
            body = panel.layout().itemAt(1).layout()
            right = 0
            for i in range(body.count()):
                widget = body.itemAt(i).widget()
                if widget is None or widget.isHidden():
                    continue
                right = max(right, widget.x() + widget.width())
            assert right <= panel.width(), (panel.width(), right)
        assert window._strip._xyr._dots_box.isHidden()
    finally:
        window.hide()


def test_state_word_length_does_not_move_its_neighbours(window, qapp):
    """IDLE/MOVE/CONT/BLOCKED all render in ONE slot width — the whole point
    of sizing it for the longest word — so the elastic readout never jumps
    sideways when a stage starts moving or a direction blocks."""
    window.resize(1920, 1080)
    window.show()
    qapp.processEvents()
    try:
        section = window._strip._xyr
        state = window._strip._focus_state
        before = (section._moving.width(), section._pos.x(), state.width())
        section._moving.set_state("MOVE", "strip_mov")
        state.set_state("BLOCKED", "strip_warn")
        qapp.processEvents()
        assert (section._moving.width(), section._pos.x(),
                state.width()) == before
        assert section._moving.width() == state.width()  # one shared slot
    finally:
        window.hide()


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


def test_gamepad_autofocus_request_reaches_the_service(window, qapp):
    """LT+RT arrives as input_system.sig_af_requested and must run the same
    path as the AF-S button — with on-screen feedback, since the operator
    otherwise has no way to tell whether the gesture was recognised."""
    calls = []
    window._input = None                       # not used by this path
    window._on_quick_af()
    assert len(window._autofocus.starts) == 1
    assert window._msg_label.text() == "Autofocus requested"

    class FakeInput(QObject):
        sig_af_requested = Signal()

        def __init__(self):
            super().__init__()
            self.af_connected = None

    fake = FakeInput()
    fake.sig_af_requested.connect(window._on_quick_af)
    fake.sig_af_requested.emit()
    assert len(window._autofocus.starts) == 2

    # a run already in flight is refused with a visible message
    window._autofocus.busy = True
    fake.sig_af_requested.emit()
    assert len(window._autofocus.starts) == 2
    assert "already running" in window._msg_label.text()


def test_input_log_reaches_the_operator(qapp, tmp_path, monkeypatch):
    """The input layer's messages (gestures, enable toggles, cancelled
    holds) were emitted into the void: the operator pressed LT+RT and
    nothing anywhere said whether it had been recognised."""
    from PySide6.QtCore import QObject, Signal

    class StubGamepad(QObject):
        sig_connected = Signal(bool)
        sig_state = Signal(object)

    class StubInput(QObject):
        sig_log = Signal(str)
        sig_dpad_stage = Signal(str)
        sig_af_requested = Signal()

        def __init__(self):
            super().__init__()
            self.gamepad = StubGamepad()

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      str(tmp_path))
    from talos.ui import calibration_context as cc_mod
    monkeypatch.setattr(cc_mod, "get_calibration_db_path",
                        lambda: tmp_path / "cal.db")

    stub = StubInput()
    window = MainWindow(StubManager(), FakeSettings(), AppState(),
                        input_system=stub,
                        autofocus_service=StubAutofocusService())
    stub.sig_log.emit("gamepad LT+RT: autofocus once")
    assert "LT+RT" in "".join(window._log.panel._lines)
    assert window._msg_label.text() == "gamepad LT+RT: autofocus once"


def test_the_scan_panel_previews_the_plan_it_would_run(window):
    """The map shows the route the CURRENT fields would walk — the same
    plan_path() the scanner uses, so a preview cannot promise a grid the
    run would not visit."""
    from talos.cv.scan import grid_shape, plan_path

    scan = window._sample_finding.scan_panel
    params = scan.params_for()
    fov = scan.fov()
    assert scan.map._plan.waypoints == [(w.x_um, w.y_um)
                                        for w in plan_path(params, fov)]
    assert (scan.map._plan.fov_x_um, scan.map._plan.fov_y_um) == fov
    cols, rows = grid_shape(params, fov)
    assert len(scan.map._plan.waypoints) == cols * rows

    # editing the area re-derives the preview
    scan.width.setValue(scan.width.value() * 2)
    scan.refresh_plan()
    assert len(scan.map._plan.waypoints) > cols * rows


def test_the_panel_owns_only_the_settings_it_shows(window):
    """The preferences-owned values (overlap, settle, speed, backlash,
    exports) must survive a panel edit: if the panel wrote them from
    widgets it does not have, every edit would silently reset them."""
    from talos.scan_settings import SCAN_KEYS, load_scan_settings

    scan = window._sample_finding.scan_panel
    section = window._settings.section("scan")
    section["overlap"] = 0.42
    section["settle_ms"] = 777
    section["speed_pps"] = 1234
    scan.reload_preferences()
    scan.width.setValue(scan.width.value() + 100)
    scan._persist()
    saved = load_scan_settings(window._settings)
    assert saved["overlap"] == pytest.approx(0.42)
    assert saved["settle_ms"] == 777
    assert saved["speed_pps"] == 1234
    assert saved["width_um"] == pytest.approx(scan.width.value())
    assert set(scan._prefs) == set(SCAN_KEYS)


def test_scan_progress_and_tiles_reach_the_map(window):
    """What the scan thread emits must land on the map: the progress bar,
    and one tile per captured frame at its readback position."""
    import numpy as np

    scan = window._sample_finding.scan_panel
    scan._scan_tiles.clear()
    scan._on_progress(3, 12)
    assert scan.progress.maximum() == 12
    assert scan.progress.value() == 3
    assert "3/12" in scan.status.text()

    thumb = np.full((54, 96, 3), 90, np.uint8)
    scan._on_tile(0, 1000.0, 2000.0, thumb)
    scan._on_tile(1, 1500.0, 2000.0, thumb)
    assert len(scan.map._tiles) == 2
    assert scan.map._tiles[1].x_um == 1500.0
    assert scan._scan_tiles[0] == (1000.0, 2000.0)
    assert "2 tiles" in scan.map._caption
    # The mosaic survives everything except a new run or an explicit clear:
    # after a scan it is what the operator reads samples off, and the plan
    # around it is only a preview of the next one.
    scan.map.set_plan(scan.map._plan)
    assert len(scan.map._tiles) == 2
    scan.width.setValue(scan.width.value() * 2)          # a new area
    assert len(scan.map._tiles) == 2, "the finished run's mosaic was wiped"
    scan.refresh_plan()
    assert len(scan.map._tiles) == 2
    scan.clear_results()                                  # the operator's act
    assert not scan.map._tiles


def test_a_finished_run_keeps_its_mosaic_when_the_stage_has_moved(window):
    """Regression, reported from the bench: after a scan the map came back
    empty. The plan's origin is the live stage position, a run ends with
    the stage back at the start (or at the last tile), and a pulse of
    readback noise in where it ended was enough to change the plan's
    identity — which dropped the tiles with it. The plan is now anchored to
    the run's origin while its results are on the map."""
    import numpy as np

    from talos.models import StagePosition

    scan = window._sample_finding.scan_panel
    start = StagePosition(x_um=-1000.0, y_um=-500.0, r_deg=0.0)
    window._manager.last_position["zolix"] = {
        "x_um": start.x_um, "y_um": start.y_um, "r_deg": 0.0,
        "x_pulses": 0, "y_pulses": 0, "r_pulses": 0}
    scan._scan_tiles.clear()
    scan._scan_hits.clear()
    scan._origin = start
    scan.refresh_plan()
    scan._on_tile(0, -1000.0, -500.0, np.zeros((8, 16, 3), np.uint8))
    scan._on_tile(1, -500.0, -500.0, np.zeros((8, 16, 3), np.uint8))
    assert len(scan.map._tiles) == 2
    anchored = scan.map._plan.x0_um

    # the run ends and the stage reports itself 0.625 µm (a pulse) away
    window._manager.last_position["zolix"] = {
        "x_um": -999.375, "y_um": -500.0, "r_deg": 0.0,
        "x_pulses": 1, "y_pulses": 0, "r_pulses": 0}
    scan._set_job("scan")
    scan._set_job(None)                     # what _on_scan_done does
    scan.refresh_plan()
    assert len(scan.map._tiles) == 2, "the finished run's mosaic was wiped"
    assert scan.map._plan.x0_um == pytest.approx(anchored)
    assert scan.map._highlight() is not None

    # ... and the anchor is released when the operator clears the results
    scan.clear_results()
    assert scan._origin is None
    scan.refresh_plan()
    assert scan.map._plan.x0_um == pytest.approx(-999.375)


def test_a_captured_tile_reaches_the_detection_queue(window):
    """The panel captures; the workspace detects. The seam between them is
    one signal, and it carries the FULL-resolution frame — the map only
    ever sees the thumbnail."""
    import numpy as np

    finding = window._sample_finding
    seen: list = []
    finding._engine.submit_tile = lambda *a, **kw: seen.append((a, kw))
    frame = np.full((64, 96, 3), 120, np.uint8)
    finding.scan_panel._on_scan_frame(7, 1234.5, -42.0, frame)
    assert len(seen) == 1
    index, delivered = seen[0][0][0], seen[0][0][1]
    assert index == 7
    assert delivered is frame
    assert seen[0][1]["scale"] == 1.0            # tiles run full resolution
    # the map's own position for the tile is recorded too
    assert finding.scan_panel._scan_tiles[7] == (1234.5, -42.0)


def test_the_footprint_follows_the_nested_telemetry(window):
    """Regression: the whole device payload was passed to
    StagePosition.from_telemetry, which reads a FLAT dict — so the position
    came back as (0, 0, 0) and the "you are here" box sat at stage origin
    on every idle frame."""
    scan = window._sample_finding.scan_panel
    scan.update_telem("zolix", {"position": {"x_um": 4321.0, "y_um": -765.0,
                                             "x_pulses": 1, "y_pulses": 2,
                                             "r_pulses": 0, "r_deg": 0.0},
                                "status": {"x_moving": False}})
    assert scan.map._footprint == (4321.0, -765.0)
    # a payload with no position at all must leave it alone, not zero it
    scan.update_telem("zolix", {"connected": True})
    assert scan.map._footprint == (4321.0, -765.0)
    # a different device is not this panel's business
    scan.update_telem("focus", {"position": {"x_um": 1.0, "y_um": 2.0}})
    assert scan.map._footprint == (4321.0, -765.0)


def _ready_to_scan(window, monkeypatch, *, resolution, live_width=1920):
    """A panel that passes every gate in `_on_scan`, with `_begin_scan`
    stubbed out (the run itself has its own tests) and the camera's job
    submissions recorded."""
    import numpy as np

    scan = window._sample_finding.scan_panel
    window._manager.last_position["zolix"] = {"x_um": 0.0, "y_um": 0.0,
                                              "r_deg": 0.0}
    scan._latest_frame = np.zeros((live_width * 9 // 16, live_width, 3),
                                  np.uint8)
    scan._prefs["resolution"] = resolution
    calls: list = []
    started: list = []
    monkeypatch.setattr(scan._manager, "submit_camera",
                        lambda method, *args: (calls.append((method, args)),
                                               4242)[1])
    monkeypatch.setattr(scan, "_begin_scan", started.append)
    return scan, calls, started


def test_the_map_marks_the_tile_being_captured_and_then_the_live_position(
        window):
    """The panel's half of the map's three states: nothing until a run
    starts, the newest tile while it captures, the live position after."""
    import numpy as np

    scan = window._sample_finding.scan_panel
    scan.map.set_footprint(7000.0, -3000.0)
    assert scan.map._highlight() is None            # nothing scanned yet

    scan._on_tile(0, 10.0, 20.0, np.zeros((8, 16, 3), np.uint8))
    scan._on_tile(1, 30.0, 40.0, np.zeros((8, 16, 3), np.uint8))
    assert scan.map._highlight() == (30.0, 40.0, "tile")

    scan._on_scan_done(None)
    assert scan.map._highlight() == (7000.0, -3000.0, "here")
    scan.map.clear_tiles()


def test_the_path_toggle_round_trips_through_the_two_stored_keys(window):
    """Path and Order were one question. The toggle writes the same two
    keys it always did — so a configuration stored as
    ``path=serpentine, serpentine=false`` comes back as "One-way", and
    nothing needs migrating."""
    from talos.cv.scan import ONE_WAY, SERPENTINE
    from talos.scan_settings import load_scan_settings
    from talos.ui.widgets.scan_panel import _path_kind, _path_toggle_value

    assert _path_toggle_value(SERPENTINE, False) == ONE_WAY
    assert _path_toggle_value(SERPENTINE, True) == SERPENTINE
    assert _path_kind(ONE_WAY) == SERPENTINE

    scan = window._sample_finding.scan_panel
    scan.path.set_value(ONE_WAY)
    scan._persist()
    saved = load_scan_settings(window._settings)
    assert saved["path"] == SERPENTINE and saved["serpentine"] is False
    assert scan.params_for(scan.stage_position()).serpentine is False

    scan.path.set_value(SERPENTINE)
    scan._persist()
    saved = load_scan_settings(window._settings)
    assert saved["path"] == SERPENTINE and saved["serpentine"] is True

    # ... and a hand-edited file round-trips too
    window._settings.section("scan")["serpentine"] = False
    scan._load_settings()
    assert scan.path.value() == ONE_WAY


def test_a_scan_at_the_live_resolution_switches_nothing(window, monkeypatch):
    scan, calls, started = _ready_to_scan(window, monkeypatch, resolution=1)
    scan._on_scan()
    assert calls == []
    assert len(started) == 1, "the run should have started at once"


def test_a_scan_at_another_resolution_switches_once_and_puts_it_back(
        window, monkeypatch):
    """The switch is sequenced ahead of the first move: a run that started
    before the camera changed mode would file 1080p frames as 4K tiles."""
    scan, calls, started = _ready_to_scan(window, monkeypatch, resolution=0)
    scan._on_scan()
    assert calls == [("set_property", ("resolution", 0))]
    assert started == [], "the run started before the camera switched"
    assert scan._camera_mode_before_scan == 1

    job = scan._pending_scan_start[0]
    scan._on_camera_job_done(job, None)
    assert len(started) == 1 and scan._pending_scan_start is None

    # ... and the live stream goes back when the run reports in
    scan._on_scan_done(None)
    assert calls[-1] == ("set_property", ("resolution", 1))
    assert scan._camera_mode_before_scan is None


def test_a_refused_switch_still_runs_the_scan(window, monkeypatch):
    """A backend with no live resolution switch (or a camera that errors)
    must not cost the operator the run: it is logged and the scan goes
    ahead at the live resolution."""
    scan, calls, started = _ready_to_scan(window, monkeypatch, resolution=0)
    monkeypatch.setattr(scan._manager, "submit_camera",
                        lambda method, *args: -1)
    logs: list = []
    scan.sig_log.connect(logs.append)
    scan._on_scan()
    assert len(started) == 1
    assert scan._camera_mode_before_scan is None
    assert any("live one" in line for line in logs), logs


def test_a_scan_holds_the_processed_views_back(window):
    """For the whole run: the processed layers would be work the tiles are
    queued behind, and the live feed is what the operator watches the stage
    travel on."""
    finding = window._sample_finding
    finding.set_view_mode("samples")
    assert not finding.live_view.processed_paused

    finding._state.set_mode("SCAN")
    try:
        assert finding.live_view.processed_paused
        assert "scanning" in finding.live_view._pause_note
        assert finding._engine.suspended
    finally:
        finding._state.set_mode("MANUAL")
    assert not finding.live_view.processed_paused
    assert not finding._engine.suspended


def test_the_motion_hold_ends_under_a_steady_stream_of_telemetry(window):
    """Regression, found by the screenshot rig: the hold was restarted on
    every sample, so at the 10 Hz telemetry rate it never expired and the
    processed views never came back. Only a MOVING sample may push it
    back."""
    import time

    from PySide6.QtWidgets import QApplication

    finding = window._sample_finding
    quiet = {"position": {"x_um": 0.0, "y_um": 0.0},
             "status": {"x_moving": False}}
    finding._note_motion("zolix", {"position": {"x_um": 0.0, "y_um": 0.0},
                                   "status": {"x_moving": True}})
    assert finding.live_view.processed_paused
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and finding.live_view.processed_paused:
        finding._note_motion("zolix", quiet)     # telemetry keeps arriving
        QApplication.processEvents()             # let the hold timer fire
        time.sleep(0.05)
    assert not finding.live_view.processed_paused, \
        "the pause never lifted while telemetry kept flowing"


def test_a_moving_axis_holds_the_processed_views_back(window):
    """A jog: the processed frame is a picture of where the stage WAS. The
    live feed keeps running (a jog is exactly when the dropper is used)."""
    finding = window._sample_finding
    moving = {"position": {"x_um": 0.0, "y_um": 0.0},
              "status": {"x_moving": True}}
    finding._note_motion("zolix", moving)
    assert finding.live_view.processed_paused
    assert not finding._engine.suspended

    finding._note_motion("zolix", {"position": {"x_um": 0.0, "y_um": 0.0},
                                   "status": {"x_moving": False}})
    assert finding.live_view.processed_paused, "held until the frame settles"
    finding._on_motion_hold_expired()
    assert not finding.live_view.processed_paused


def test_focus_motion_pauses_the_previews_too(window):
    finding = window._sample_finding
    finding._note_motion("focus", {"status": {"mode": "TRAP"}})
    assert finding.live_view.processed_paused
    finding._note_motion("focus", {"status": {"mode": "IDLE"}})
    finding._on_motion_hold_expired()
    assert not finding.live_view.processed_paused


def test_a_tile_never_becomes_the_live_display_layer(window):
    """A tile is a different part of the sample: showing it (or letting the
    dropper sample it) is how the view jumps to a region nobody is looking
    at."""
    finding = window._sample_finding
    finding.live_view.set_preprocessed_frame(None)
    live_layer = finding.live_view._preprocessed_frame
    finding._on_detected(3, _result_stub(), _tile_frame(), None)
    assert finding.live_view._preprocessed_frame is live_layer


def test_the_dropper_says_when_its_patch_is_not_uniform(window):
    """The "sometimes" in the bench report.

    The dropper averages a 9-px disc, so a click near a flake's edge returns
    the mean of the flake and its substrate — a colour NEITHER of them has,
    which then becomes the mask's target AND the curve's centre. It is said
    out loud now, where the operator is already looking.
    """
    finding = window._sample_finding
    finding._note_pick("#1c3484", 3.0)
    assert "#1c3484" in finding.colour_group._hint.text()
    assert "not uniform" not in finding.colour_group._hint.text()
    assert finding.colour_group._hint.objectName() == "dim"

    finding._note_pick("#5672b2", 34.0)
    assert "not uniform" in finding.colour_group._hint.text()
    assert finding.colour_group._hint.objectName() == "warn"


def test_picking_off_the_live_view_sets_the_colour_and_judges_the_patch(window):
    finding = window._sample_finding
    frame = np.zeros((40, 60, 3), np.uint8)
    frame[:, :30] = (28, 52, 132)          # the dark layer
    frame[:, 30:] = (170, 205, 245)        # the light one
    finding.live_view.set_preprocessed_frame(frame)

    finding.on_pick(30, 20)                # on the boundary between them
    assert "not uniform" in finding.colour_group._hint.text()

    finding.on_pick(8, 20)                 # well inside the dark one
    assert finding.colour_group.hex_color() == "#1c3484"
    assert "not uniform" not in finding.colour_group._hint.text()


def test_a_pick_paler_than_the_floors_says_the_floor_no_longer_applies(window):
    """The clamp in :func:`colour_mask` keeps the picked colour inside its
    own mask by lowering the floor to it. That changes what Min saturation
    means for that colour, so it is said rather than silently done."""
    finding = window._sample_finding
    _chain_colour_editor(window)._editors["min_saturation"][1].setValue(200)
    finding._note_pick("#d0d0d0", 2.0)
    assert "floor" in finding.colour_group._hint.text()
    assert finding.colour_group._hint.objectName() == "warn"


def _result_stub():
    class _Result:
        candidates: list = []
        summary = "colour 1"

    return _Result()


def _tile_frame():
    import numpy as np

    return np.full((32, 48, 3), 200, np.uint8)


def test_stop_all_aborts_the_run_not_just_the_motion(window, monkeypatch):
    """Esc must end a scan. Without this wiring the abort flag stayed
    clear, the controller stopped, and the run continued at the next
    waypoint — which looked exactly like Esc doing nothing.

    And it must end it WITHOUT stopping everything again: this slot runs
    when a stop_all completes, so re-issuing one fired the signal again and
    looped (the bench saw it as a scan that "aborted halfway and then
    looped forever").
    """
    scan = window._sample_finding.scan_panel
    stops: list = []
    monkeypatch.setattr(window._manager, "stop_all",
                        lambda: stops.append("stop"))
    aborted: list = []
    monkeypatch.setattr(scan, "_abort_run",
                        lambda reason: aborted.append(reason))
    scan._scan_abort.clear()
    scan._set_job("scan")
    window._manager.sig_stop_all_done.emit()
    assert aborted == ["stop all"]
    assert stops == [], "the abort re-stopped a stage that had stopped"
    scan._set_job(None)
    window._manager.sig_stop_all_done.emit()
    assert aborted == ["stop all"]              # idle: nothing to abort


def test_the_panel_abort_button_does_stop_the_stage(window, monkeypatch):
    """The operator's own Abort has to stop the motion — nothing else has."""
    scan = window._sample_finding.scan_panel
    stops: list = []
    monkeypatch.setattr(window._manager, "stop_all",
                        lambda: stops.append("stop"))
    scanner_stops: list = []
    monkeypatch.setattr(scan, "_scanner", type("_S", (), {
        "request_abort": lambda self: scanner_stops.append("abort")})())
    scan._scan_abort.clear()
    scan._on_abort("abort")
    assert stops == ["stop"]
    assert scanner_stops == ["abort"]
    assert scan._scan_abort.is_set()

    # a second abort of the same run does not re-command the axis
    scan._on_abort("stop all")
    assert scanner_stops == ["abort"]
    assert len(stops) == 2                      # only the operator's own
    scan._scan_abort.clear()
    monkeypatch.setattr(scan, "_scanner", None)


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


def _window_with_input(tmp_path, monkeypatch, manager=None, **input_attrs):
    """A MainWindow whose input system is a stub that records calls."""
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      str(tmp_path))
    from talos.ui import calibration_context as cc_mod
    monkeypatch.setattr(cc_mod, "get_calibration_db_path",
                        lambda: tmp_path / "cal.db")

    class StubGamepad(QObject):
        sig_connected = Signal(bool)
        sig_state = Signal(object)

    class StubInput(QObject):
        sig_log = Signal(str)
        sig_dpad_stage = Signal(str)
        sig_af_requested = Signal()

        def __init__(self):
            super().__init__()
            self.gamepad = StubGamepad()
            self.escapes = 0
            for name, value in input_attrs.items():
                setattr(self, name, value)

        def on_escape(self):
            self.escapes += 1

        def cancel_all_holds(self, reason="stop requested"):
            pass

        def reload_settings(self):
            pass

    stub = StubInput()
    window = MainWindow(manager or StubManager(), FakeSettings(), AppState(),
                        input_system=stub,
                        autofocus_service=StubAutofocusService())
    return window, stub


def test_escape_issues_one_stop_all(tmp_path, monkeypatch):
    """Regression (2026-09-16): Esc ran STOP ALL twice — MainWindow called
    manager.stop_all() and then InputSystem.on_escape() did it again, so one
    press enqueued two rounds of stop jobs (and two ack reports). The input
    system owns the whole sequence: latch + drop the on-screen holds + stop."""
    manager = StubManager()
    window, stub = _window_with_input(tmp_path, monkeypatch, manager=manager)
    try:
        window._on_escape()
        assert stub.escapes == 1
        assert manager.stopped == 0, "on_escape already stops the axes"
    finally:
        window._log.close()
        window._focus_window.close()
        window._stage_window.close()


def test_trigger_bar_readout_matches_the_commanded_speed(window, qapp):
    """The strip's jog-speed readout must show the number the dispatcher
    sends. Regression: the bar called focus_trigger_to_speed with the RAW
    device bounds while the dispatcher scaled them by the current objective's
    focus_manual_multiplier — with the shipped 20x row it printed ~9x the
    commanded speed, and nothing repainted it when the objective changed."""
    from talos.input.action_resolver import (focus_manual_bounds,
                                             focus_trigger_to_speed)

    settings = window._settings
    settings.data["objectives"].append(
        {"name": "20x", "mag": 20, "na": 0.45, "dof_um": 1.0,
         "coarse_step_um": 1.0, "fine_step_um": 0.5,
         "af_speed_multiplier": 0.1063, "focus_manual_multiplier": 0.1063,
         "stage_speed_multiplier": 0.25, "px_um": 0.0})
    window._state.set_objective(1)
    bar = window._strip.trigger_bar()
    lo, hi = focus_manual_bounds(settings, window._state)
    assert (lo, hi) == pytest.approx((50 * 0.1063, 2000 * 0.1063))
    assert (bar._min_speed, bar._max_speed) == pytest.approx((lo, hi))
    shown = focus_trigger_to_speed(0.0, 1.0, min_speed=bar._min_speed,
                                   max_speed=bar._max_speed, gamma=bar._gamma,
                                   deadzone=bar._deadzone, invert=bar._invert)
    assert shown == 213, "a full press at 20x commands 213 sps, not 2000"


def test_on_screen_hold_buttons_use_the_configured_threshold(window, qapp):
    """The Preferences tap-vs-hold threshold governs the keys and the D-pad;
    the on-screen hold buttons must use the same number (they decide
    click-vs-hold before the resolver ever sees the claim)."""
    window._settings.section("input")["long_press_threshold_ms"] = 700
    buttons = window._stage_window._panels["zolix"]._buttons
    assert buttons["x"]._long_press_ms() == 700
    window._settings.section("input")["long_press_threshold_ms"] = 300
    assert buttons["x"]._long_press_ms() == 300


def test_af_panel_reads_um_per_step_live(window, qapp):
    """Regression (2026-09-16): the AF panel cached µm/step at construction
    while the strip and the settings readout re-read it — after editing it in
    Preferences the same focus position was shown in µm on two scales."""
    panel = window._focus_window.panel
    window._settings.device("focus")["um_per_step"] = 0.5
    window._on_settings_applied()
    assert panel._curve.um_per_step == 0.5


def test_focus_once_refused_keeps_the_curve_and_abort(window, qapp):
    """Regression (2026-09-16): "Focus once" cleared the displayed curve and
    armed Abort BEFORE the service could refuse (busy) — the Abort then killed
    the job that was already running."""
    panel = window._focus_window.panel
    panel._curve.add_point(10.0, 5.0)
    window._autofocus.busy = True
    panel._on_focus_once()
    assert panel._curve._points, "a refused start must not wipe the curve"
    assert panel._abort.isEnabled() is False
    # ...and an accepted start still clears the curve and arms Abort
    window._autofocus.busy = False
    panel._on_focus_once()
    assert panel._curve._points == []
    assert panel._abort.isEnabled() is True


def test_preferences_reach_the_running_scan(window):
    """The scan's out-of-the-way settings (overlap, settle, speed,
    backlash, exports) live in Preferences. Apply must push them into the
    panel, which caches the section at startup — without the refresh the
    panel keeps running the old numbers until the app restarts, and the
    operator has no way to tell that the number they changed was ignored.
    """
    from talos.scan_settings import load_scan_settings
    from talos.ui.dialogs.preferences import _scan_page

    panel = window._sample_finding.scan_panel
    assert panel._prefs["speed_pps"] == 500.0            # the default

    section = window._settings.section("scan")
    section["speed_pps"] = 1234
    section["overlap"] = 0.33
    page = _scan_page(window._settings, window)
    page._apply()
    window._on_settings_applied()

    params = panel.params_for()
    assert params.speed_pps == 1234
    assert params.overlap == pytest.approx(0.33)
    assert load_scan_settings(window._settings)["speed_pps"] == 1234


def _found_sample(window, x_um=1500.0, y_um=-250.0):
    """Put one sample in the panel's table and select it."""
    from talos.models import FlakeCandidate

    scan = window._sample_finding.scan_panel
    scan._show_candidates([(-1, FlakeCandidate(x_px=1.0, y_px=2.0,
                                               area_px2=10.0, area_um2=42.0,
                                               x_um=x_um, y_um=y_um,
                                               score=7.0))], "live view")
    scan._table.selectRow(0)
    return scan


def test_go_to_sample_submits_arguments_the_driver_accepts(window):
    """Regression: 'go to sample' did NOTHING.

    The move was submitted with ``speed_pps=…`` as a KEYWORD, but
    ``InstrumentManager.submit`` carries its arguments as a tuple and
    takes no keyword arguments — so the call raised TypeError inside the
    click handler and the stage never moved. Nothing caught it because
    every test's manager stub accepted **kwargs.

    The check that generalises: whatever the panel hands to ``submit``
    must bind against the real driver's signature.
    """
    import inspect

    from talos.hal.devices.zolix import ZolixXYRStage

    scan = _found_sample(window)
    assert scan._selected == 0

    # answer the confirmation dialog with Ok
    from PySide6.QtWidgets import QMessageBox

    original = QMessageBox.question
    QMessageBox.question = staticmethod(
        lambda *a, **kw: QMessageBox.StandardButton.Ok)
    try:
        scan._on_go_to()
    finally:
        QMessageBox.question = original

    device, method, *args = window._manager.submits[-1]
    assert device == "zolix" and method == "move_rel_um"
    # binds, or raises TypeError — the bug above
    inspect.signature(getattr(ZolixXYRStage, method)).bind(None, *args)
    dx, dy = args[0], args[1]
    assert dx == pytest.approx(1500.0)      # from the stage origin
    assert dy == pytest.approx(-250.0)
    # the scan's speed, not the jog speed
    assert args[4] == 500
    assert "Moving to sample #1" in scan.status.text()


def test_the_sample_move_refuses_while_a_job_owns_the_axes(window):
    """It must SAY why — in the status line, not in a dialog: the operator
    is looking at the sample, and a modal box between the click and the
    motion is a click paid every time."""
    scan = _found_sample(window)
    window._state.set_mode("SCAN")
    try:
        scan._on_go_to()
        assert "in use" in scan.status.text()
    finally:
        window._state.set_mode("MANUAL")
    assert not any(s[1] == "move_rel_um" for s in window._manager.submits)


def test_the_sample_move_does_not_ask_first(window):
    """The confirmation dialog is gone: the move is the button's whole
    purpose, and the distance is on the map."""
    scan = _found_sample(window)
    monkeypatch_target = window._manager
    monkeypatch_target.submits.clear()
    scan._on_go_to()
    assert any(s[1] == "move_rel_um" for s in monkeypatch_target.submits)
    assert "Moving to sample #" in scan.status.text()


def test_selecting_a_table_row_enables_go_to(window):
    scan = _found_sample(window)
    assert scan._go_to_btn.isEnabled()
    scan._table.clearSelection()
    assert not scan._go_to_btn.isEnabled()


def test_the_samples_are_a_table_outside_the_settings_scroll(window):
    """The samples are pinned below the settings, not buried inside them —
    and they are a table, because the row IS the selection."""
    from PySide6.QtWidgets import QScrollArea

    scan = window._sample_finding.scan_panel
    assert scan._table.rowCount() >= 0
    assert [scan._table.horizontalHeaderItem(i).text() for i in range(6)] == \
        ["#", "X µm", "Y µm", "Area µm²", "Edge", "View"]
    # not inside any scroll area
    parent = scan._table.parentWidget()
    while parent is not None and parent is not scan:
        assert not isinstance(parent, QScrollArea), \
            "the samples must not scroll with the scan settings"
        parent = parent.parentWidget()


# --- the previews compute what is on screen, and nothing else -------------

def test_the_live_feed_runs_only_for_the_view_that_needs_it(window):
    """The bench finding: switching to Original — or leaving the tab —
    kept a pre-process, a full identification and an overlay running per
    tick behind an image nobody was looking at."""
    from talos.ui.detect_engine import LIVE_FULL, LIVE_NONE, LIVE_PREPROCESS

    finding = window._sample_finding
    engine = finding._engine
    window._on_workspace_changed(1)          # the operator opens the tab
    try:
        finding.set_view_mode("original")
        assert engine.live_level == LIVE_NONE
        assert not engine.live, "the feed runs for the raw view"

        finding.set_view_mode("preprocessed")
        assert engine.live_level == LIVE_PREPROCESS
        assert engine.live

        finding.set_view_mode("samples")
        assert engine.live_level == LIVE_FULL
        assert engine.live
    finally:
        finding.set_view_mode("original")


def test_the_feed_stops_while_another_workspace_is_on_screen(window):
    finding = window._sample_finding
    finding.set_view_mode("samples")
    window._on_workspace_changed(1)
    assert finding._engine.live, "the tab's own page must feed"
    window._on_workspace_changed(0)              # Navigation: raw view only
    assert not finding._engine.live, "work behind a view that cannot show it"
    window._on_workspace_changed(1)
    assert finding._engine.live, "and it comes back"
    finding.set_view_mode("original")


def test_the_counts_are_cleared_when_the_pipeline_stops(window):
    """Numbers on screen that describe neither the frame being shown nor
    the chain as it is now are worse than no numbers."""
    finding = window._sample_finding
    window._on_workspace_changed(1)
    try:
        finding.set_view_mode("samples")
        finding.identify_group.set_counts("colour 812 → size 12")
        assert finding.identify_group.counts.text()
        finding.set_view_mode("original")
        assert finding.identify_group.counts.text() == ""
    finally:
        finding.set_view_mode("original")


def test_a_pick_is_refused_while_the_stage_moves(window):
    """The click is mapped with the frame on screen; the colour would come
    from a layer computed before the move. Refusing is the honest answer."""
    finding = window._sample_finding
    before = finding.colour_group.hex_color()
    finding.live_view.set_preprocessed_frame(np.full((20, 30, 3), 7, np.uint8))
    finding._stage_moving = True
    try:
        finding.on_pick(5, 5)
    finally:
        finding._stage_moving = False
    assert "moving" in finding.colour_group._hint.text()
    assert finding.colour_group.hex_color() == before


# --- the samples table: what it costs and what it keeps -------------------

def test_a_selected_row_survives_the_next_result(window):
    """The table is rebuilt on every result — and the live feed sends one at
    the preview's cadence (~7 Hz), so a row the operator picked to inspect
    (or to *go to*) used to vanish under the cursor with the button left
    enabled over it."""
    scan = window._sample_finding.scan_panel
    scan.on_tile_result(3, [_candidate(area_um2=42.0)])
    scan._table.selectRow(0)
    assert scan._selected == 0 and scan._go_to_btn.isEnabled()

    scan.on_tile_result(4, [_candidate(area_um2=9.0)])
    assert scan._selected == 0, "the rebuild cleared the selection"
    assert scan._go_to_btn.isEnabled()


def test_go_to_is_not_offered_without_a_selection(window):
    scan = window._sample_finding.scan_panel
    scan.on_tile_result(3, [_candidate()])
    assert scan._selected == -1
    assert not scan._go_to_btn.isEnabled(), "a button that only answers 'select one'"


def test_the_run_paces_itself_to_the_detector(window, monkeypatch, tmp_path):
    """Tiles are never dropped, so a scan that outruns the identification
    queue holds whole frames in memory (25 MB each at 4K) until the process
    dies. The scanner has a valve for that; the APP has to wire it — its
    default is no pacing, which is right for the CLI benches and wrong here.
    """
    from talos.cv.scan import MAX_PENDING_TILES
    from talos.models import StagePosition
    from talos.ui.widgets import scan_panel as sp

    captured: dict = {}

    class Inert:
        """Signals (``.connect``) and ``run()``, all inert."""

        def connect(self, *_args, **_kwargs):
            pass

        def __call__(self, *_args, **_kwargs):
            return None

    class FakeScanner:
        def __init__(self, *_args, **kwargs):
            captured.update(kwargs)

        def __getattr__(self, _name):
            return Inert()

    monkeypatch.setattr(sp, "GridScanner", FakeScanner)
    scan = window._sample_finding.scan_panel
    scan.pending_tiles_fn = lambda: 7
    scan._run_scan(scan.params_for(StagePosition()), (100.0, 100.0),
                   tmp_path / "scan_pacing", (20, 30, 3))

    assert captured.get("pending_tiles_fn") is not None, \
        "the scan runs with no pacing valve"
    assert captured.get("max_pending_tiles") == MAX_PENDING_TILES


def test_a_tile_result_appends_instead_of_rebuilding(window):
    """A full rebuild per tile is O(rows) on the GUI thread, Σ over a run is
    quadratic — and it threw away every QTableWidgetItem each time."""
    scan = window._sample_finding.scan_panel
    scan.on_tile_result(0, [_candidate()])
    first = scan._table.item(0, 0)
    scan.on_tile_result(1, [_candidate()])
    assert scan._table.rowCount() == 2
    assert scan._table.item(0, 0) is first, "the first row was rebuilt"
    assert [scan._table.item(row, 0).text() for row in range(2)] == ["1", "2"]
    assert scan._row_tiles == [0, 1]


def test_live_results_that_say_the_same_thing_do_not_rebuild(window):
    scan = window._sample_finding.scan_panel
    scan.show_live_candidates([_candidate()])
    item = scan._table.item(0, 0)
    scan.show_live_candidates([_candidate()])      # the same sample again
    assert scan._table.item(0, 0) is item, "rebuilt for nothing"
    scan.show_live_candidates([_candidate(area_um2=99.0)])
    assert scan._table.item(0, 3).text() == "99.0"      # the area column


def test_the_export_carries_its_own_copy_of_the_results(window):
    """The extras read every frame from disk and take seconds; the operator
    can start the next run in that window, which clears the live
    dictionaries. Reading them at export time then wrote an EMPTY
    candidates.csv over a run that had found samples."""
    from pathlib import Path

    from talos.cv.scan import ScanResult

    scan = window._sample_finding.scan_panel
    # the detector is still draining: this is the window the bug was in
    saved_pending, scan.pending_tiles_fn = scan.pending_tiles_fn, lambda: 1
    scan._scan_hits.clear()
    scan.on_tile_result(3, [_candidate(area_um2=42.0)])
    scan._scan_tiles[3] = (10.0, 20.0)
    result = ScanResult(manifest_path=Path("C:/tmp/x.csv"), planned=4,
                        visited=4)
    scan._pending_export = None
    scan._on_scan_done({"result": result, "out_dir": Path("C:/tmp"),
                        "fov": (100.0, 100.0)})

    assert scan._pending_export is not None
    assert scan._pending_export.get("hits"), "the results were not copied"
    # the next run clears the live dictionaries...
    scan._scan_hits.clear()
    scan._scan_tiles.clear()
    # ...and the export still has them
    assert scan._pending_export["hits"][3][0].area_um2 == pytest.approx(42.0)
    assert scan._pending_export["tiles"][3] == (10.0, 20.0)
    scan.pending_tiles_fn = saved_pending


# --- a run owns the hardware (the scan lock) ------------------------------

def test_a_run_disables_the_cv_and_camera_controls(window):
    """The bench finding: a mid-run edit of the chain changes how the tiles
    captured after it are detected, so the run stops being one experiment —
    and the dropper would sample a frame the stage has already left."""
    finding = window._sample_finding
    scan = finding.scan_panel
    for widget in (finding.colour_group, finding.camera_group,
                   finding.preprocess_group, finding.identify_group):
        assert widget.isEnabled(), "a disabled control at rest"

    scan._set_job("scan")
    for widget in (finding.colour_group, finding.camera_group,
                   finding.preprocess_group, finding.identify_group):
        assert not widget.isEnabled()
    assert not scan._clear_btn.isEnabled()
    assert not scan.scan_btn.isEnabled()
    assert scan.abort_btn.isEnabled()

    scan._set_job(None)
    for widget in (finding.colour_group, finding.camera_group,
                   finding.preprocess_group, finding.identify_group):
        assert widget.isEnabled()


def test_the_strip_and_the_snapshot_are_locked_by_a_run(window):
    strip = window._strip
    strip.set_enable_locked(False)
    assert strip._xyr._enable.isEnabled()
    strip.set_enable_locked(True)
    assert not strip._xyr._enable.isEnabled()
    assert "owns this stage" in strip._xyr._enable.toolTip()
    strip.set_enable_locked(False)

    window._manager.submits.clear()
    window._manager.camera_submits.clear()
    window._sample_finding.scan_panel._set_job("scan")
    window._on_snapshot()
    assert not window._manager.camera_submits, "a snapshot mid-run"
    window._sample_finding.scan_panel._set_job(None)


def test_a_live_edit_does_not_stop_the_profile_being_applied(window):
    """The per-workspace profiles diff against what the camera HAS.

    Diffing against the last profile WE applied came out empty after any
    live edit — an exposure slider, the auto-gain loop, "Balance once" — so
    the profile was silently not applied: the operator switches to Sample
    Finding to scan and the run captures at the Navigation tab's exposure.
    """
    from talos.ui.camera_profiles import scan_profile

    from talos.ui.camera_profiles import nav_profile

    # the camera reports its properties (a connect, or a get_properties)
    window._manager.camera_props = dict(
        (key, getattr(nav_profile(window._settings), key))
        for key in ("exposure_us", "gain", "white_balance"))
    window._on_workspace_changed(0)            # the Navigation tab is applied
    # the operator drags exposure there — a write outside the profile path
    window._manager.submit_camera("set_property", "exposure_us", 5000.0)
    window._manager.camera_submits.clear()

    window._on_workspace_changed(1)            # ...to the Sample Finding tab
    written = [call for call in window._manager.camera_submits
               if call[0] == "set_property" and call[1] == "exposure_us"]
    assert written, "the profile was not applied after a live edit"
    assert written[-1][2] == pytest.approx(
        scan_profile(window._settings).exposure_us)


def test_a_camera_profile_is_held_for_the_run_not_dropped(window, monkeypatch):
    """A tab click mid-run must not change the camera's exposure — but the
    operator's setting must not be lost either."""
    from talos.ui.camera_profiles import nav_profile

    window._manager.camera_submits.clear()
    window._sample_finding.scan_panel._set_job("scan")
    window._state.set_mode("SCAN")
    try:
        window._apply_camera_profile(nav_profile(window._settings))
        assert not window._manager.camera_submits, "wrote the camera mid-run"
        assert window._profile_pending
    finally:
        window._state.set_mode("MANUAL")
        window._sample_finding.scan_panel._set_job(None)
    # ending the run applies it
    window._flush_deferred()
    assert not window._profile_pending


def test_autofocus_is_refused_while_a_scan_owns_the_axes(window):
    """An autofocus run rewrites the mode to AUTOFOCUS, which unfreezes the
    manual inputs — mid-scan that means a moving focus axis under a run and
    a gate that no longer holds."""
    window._manager.submits.clear()
    window._sample_finding.scan_panel._set_job("scan")
    window._state.set_mode("SCAN")
    try:
        window._on_quick_af()
        assert not window._autofocus.starts, "autofocus started mid-scan"
        assert window._state.mode == "SCAN", "and it stole the mode"
    finally:
        window._state.set_mode("MANUAL")
        window._sample_finding.scan_panel._set_job(None)


def test_closing_the_window_asks_and_stops_the_run(window, monkeypatch):
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import QMessageBox

    scan = window._sample_finding.scan_panel
    scan._set_job("scan")
    window._state.set_mode("SCAN")
    asked: list = []

    def refuse(*args, **kwargs):
        asked.append(args)
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", staticmethod(refuse))
    event = QCloseEvent()
    window.closeEvent(event)
    assert asked, "closing mid-run did not ask"
    assert not event.isAccepted(), "No must leave the window open"

    scan._set_job(None)
    window._state.set_mode("MANUAL")


def test_the_camera_switch_is_watched_and_can_be_cancelled(window):
    """A camera job that never reports used to leave the panel saying
    "Switching…" with the lock held and no run — forever."""
    from talos.models import StagePosition

    scan = window._sample_finding.scan_panel
    monkeypatch_target = window._manager
    monkeypatch_target.camera_submits.clear()
    scan._pending_scan_start = (4242, StagePosition(x_um=1.0, y_um=2.0))
    scan._camera_mode_before_scan = None
    scan._set_job("scan")
    window._state.set_mode("SCAN")

    # the watchdog fires: no run, no lock, and it says why
    scan._on_start_timeout()
    assert scan._pending_scan_start is None
    assert not scan.is_scanning()
    assert window._state.mode == "MANUAL"
    assert "did not answer" in scan.status.text()


# --- the View column: the frame a sample was found in ---------------------

def _candidate(x_um=100.0, y_um=0.0, area_um2=42.0, x_px=20.0, y_px=15.0):
    from talos.models import FlakeCandidate

    return FlakeCandidate(x_px=x_px, y_px=y_px, area_px2=120.0,
                          area_um2=area_um2, x_um=x_um, y_um=y_um,
                          score=7.0, bbox=(10, 10, 20, 10))


def _run_dir_with_frames(tmp_path, tiles=(3, 5), count=None):
    """A run folder: a PNG per index in ``tiles`` and the manifest that
    indexes them — one row per waypoint, as the writer leaves it (a miss
    keeps its row with an empty frame cell)."""
    import csv

    out = tmp_path / "scan_20260923_101500"
    (out / "frames").mkdir(parents=True)
    for index in tiles:
        cv2.imwrite(str(out / "frames" / f"frame_{index:05d}.png"),
                    np.full((40, 60, 3), 10 * index, np.uint8))
    rows = max(tiles) + 1 if count is None else int(count)
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "x_um", "y_um", "r_deg", "t_unix",
                         "objective_id", "focus_pos"])
        for index in range(rows):
            name = f"frame_{index:05d}.png" if index in tiles else ""
            writer.writerow([name, 100.0 * index, 0.0, 0.0, 0.0, 0, 0])
    return out


def test_a_row_knows_which_tile_it_came_from(window, tmp_path):
    """The table is flattened from per-tile results, so the tile index has
    to travel WITH each row — the mosaic's ring numbering, the CSV's tile
    column and this all have to agree."""
    scan = window._sample_finding.scan_panel
    scan.on_tile_result(3, [_candidate(), _candidate(area_um2=10.0)])
    scan.on_tile_result(5, [_candidate(area_um2=99.0)])
    assert scan._row_tiles == [3, 3, 5]
    assert [c.area_um2 for c in scan._candidates] == [42.0, 10.0, 99.0]


def test_the_view_column_opens_the_frame_the_row_came_from(window, tmp_path):
    scan = window._sample_finding.scan_panel
    scan._run_dir = _run_dir_with_frames(tmp_path)
    scan.on_tile_result(3, [_candidate(area_um2=42.0)])
    scan.on_tile_result(5, [_candidate(area_um2=99.0)])
    scan._review_row(0)
    assert scan._review.isVisible()
    shot = scan._review._shot
    assert shot.tile == 3
    assert shot.frame is not None
    assert int(shot.frame[0, 0, 0]) == 30, "tile 3's own pixels"
    # ...and stepping to the next row reads the OTHER frame
    scan._review.step(1)
    assert scan._review._shot.tile == 5
    assert int(scan._review._shot.frame[0, 0, 0]) == 50


def test_the_ring_is_drawn_on_the_sample_not_the_frames_corner(window,
                                                              tmp_path):
    """The ring uses the candidate's own pixels — the window draws in the
    frame's coordinates, so nothing can put it at the wrong scale."""
    scan = window._sample_finding.scan_panel
    scan._run_dir = _run_dir_with_frames(tmp_path)
    scan.on_tile_result(3, [_candidate(x_px=20.0, y_px=15.0)])
    scan._review_row(0)
    marked = scan._review._marked
    plain = scan._review._shot.frame
    assert marked.shape == plain.shape
    assert not np.array_equal(marked, plain), "something was drawn"
    assert np.array_equal(plain, scan._review._shot.frame), \
        "the cached frame is never drawn on"
    # the drawn pixels are around (20, 15) — where the sample is
    diff = (marked != plain).any(axis=2)
    ys, xs = np.nonzero(diff)
    assert abs(int(xs.mean()) - 20) <= 8 and abs(int(ys.mean()) - 15) <= 8


def test_a_frame_the_writer_has_not_flushed_yet_is_not_a_blank_window(
        window, tmp_path):
    """Detection outruns the writer, so a row can exist a moment before its
    frame is on disk. The window says so — and re-reads once the writer
    catches up, rather than leaving the operator to guess."""
    import csv

    scan = window._sample_finding.scan_panel
    out = _run_dir_with_frames(tmp_path, tiles=(3,), count=7)   # rows 0..6
    scan._run_dir = out
    scan._job = "scan"
    try:
        scan.on_tile_result(7, [_candidate()])       # row 7 not written yet
        scan._review_row(0)
        assert "still being written" in scan._review._shot.error
        # the writer catches up: PNG and manifest row together
        cv2.imwrite(str(out / "frames" / "frame_00007.png"),
                    np.full((40, 60, 3), 70, np.uint8))
        with open(out / "manifest.csv", "a", newline="",
                  encoding="utf-8") as fh:
            csv.writer(fh).writerow(["frame_00007.png", 700.0, 0.0, 0.0, 0.0,
                                     0, 0])
        scan._review.show_sample(0)
        assert scan._review._shot.frame is not None
        assert int(scan._review._shot.frame[0, 0, 0]) == 70
    finally:
        scan._job = None


def test_a_missing_frame_says_why_instead_of_showing_nothing(window,
                                                            tmp_path):
    """A tile the run recorded as a MISS has no frame and never will: that
    is a different message from the one above, not a permanent 'trying'."""
    scan = window._sample_finding.scan_panel
    scan._run_dir = _run_dir_with_frames(tmp_path)
    scan.on_tile_result(9, [_candidate()])       # tile 9 has no row
    scan._review_row(0)
    assert "no frame for this tile" in scan._review._shot.error
    assert scan._review._shot.frame is None


def test_a_live_row_is_read_from_the_live_frame(window):
    """Live rows have no file: their frame is the one the detector saw."""
    scan = window._sample_finding.scan_panel
    scan.live_frame_fn = lambda: np.full((20, 30, 3), 5, np.uint8)
    scan.show_live_candidates([_candidate()])
    assert scan._row_tiles == [-1]
    scan._review_row(0)
    assert scan._review._shot.tile == -1
    assert int(scan._review._shot.frame[0, 0, 0]) == 5


def test_clearing_the_results_drops_the_run_and_closes_the_review(window,
                                                                 tmp_path):
    scan = window._sample_finding.scan_panel
    scan._run_dir = _run_dir_with_frames(tmp_path)
    scan.on_tile_result(3, [_candidate()])
    scan._review_row(0)
    assert scan._review.isVisible()
    scan.clear_results()
    assert scan._run_dir is None and scan._frame_paths is None
    assert not scan._review.isVisible()
    assert scan._row_tiles == []


def test_a_new_run_forgets_the_previous_runs_frames(window, tmp_path):
    scan = window._sample_finding.scan_panel
    scan._run_dir = _run_dir_with_frames(tmp_path)
    scan.on_tile_result(3, [_candidate()])
    scan._review_row(0)
    assert scan._review.isVisible()
    scan._forget_run()            # what _begin_scan calls
    assert scan._run_dir is None
    assert not scan._review.isVisible()


def test_the_view_column_is_the_only_one_that_opens_the_review(window):
    """A click on the numbers must not open a window, and a double-click
    keeps meaning go-to everywhere else."""
    scan = window._sample_finding.scan_panel
    scan.on_tile_result(3, [_candidate()])
    scan._on_cell_clicked(0, 1)
    assert scan._review is None or not scan._review.isVisible()
    scan._on_cell_clicked(0, 5)
    assert scan._review is not None and scan._review.isVisible()
    scan._review.hide()
    scan._on_double_clicked(scan._table.model().index(0, 5))
    assert scan._review.isVisible()


def test_the_pre_processed_toggle_is_offered_only_when_the_chain_is_on(
        window, tmp_path):
    """A toggle that cannot change anything is a control that lies."""
    from talos.cv.preprocess import PreprocessConfig

    scan = window._sample_finding.scan_panel
    scan._run_dir = _run_dir_with_frames(tmp_path)
    scan.on_tile_result(3, [_candidate()])
    scan.review_context_fn = lambda: None
    assert scan._review_apply() is None
    scan._review_row(0)
    assert not scan._review.pre_btn.isEnabled()
    # ...and switching the chain on offers it, on the next sample opened
    scan.review_context_fn = lambda: (PreprocessConfig(), (200, 100, 50))
    assert callable(scan._review_apply())
    scan._review_row(0)
    assert scan._review.pre_btn.isEnabled()


def test_a_scan_that_stops_early_is_not_reported_as_done(window):
    """The bench bug of 2026-09-18, at the point the operator saw it.

    A transient Modbus frame error stopped a scan at tile 5 of 12. The
    panel called it "Scan done" — the reason was carried in
    ``result.message`` and never shown — so a third of a dataset looked
    like a complete one. Three outcomes, three messages.
    """
    from pathlib import Path

    from talos.cv.scan import ScanResult

    scan = window._sample_finding.scan_panel
    stopped = ScanResult(manifest_path=Path("C:/tmp/nothing.csv"),
                         message="Read input reg 30016: Frame too short: "
                                 "b'\x01\x04'",
                         missing=0, planned=12, visited=5)
    stopped.frames = [Path(f"C:/tmp/f{i}.png") for i in range(5)]
    scan._pending_export = None
    scan._on_scan_done({"result": stopped, "out_dir": Path("C:/tmp"),
                        "fov": (100.0, 100.0)})

    text = scan.status.text()
    assert "STOPPED" in text, text
    assert "tile 5 of 12" in text, text
    assert "Frame too short" in text, "the reason has to be visible"
    assert "done" not in text.lower()
    # and the styling says so too, not the same grey as a finished run
    assert scan.status.objectName() == "error"

    # a normal finish still reads as one, and clears the tone
    finished = ScanResult(manifest_path=Path("C:/tmp/x.csv"), message="ok",
                          planned=6, visited=6)
    finished.frames = [Path(f"C:/tmp/g{i}.png") for i in range(6)]
    scan._pending_export = None
    scan._on_scan_done({"result": finished, "out_dir": Path("C:/tmp"),
                        "fov": (100.0, 100.0)})
    assert scan.status.text().startswith("Scan done")
    assert scan.status.objectName() == "dim"

    # and an operator abort reads as an abort
    aborted = ScanResult(aborted=True, message="aborted by user",
                         planned=12, visited=3)
    scan._pending_export = None
    scan._on_scan_done({"result": aborted, "out_dir": Path("C:/tmp"),
                        "fov": (100.0, 100.0)})
    assert "aborted" in scan.status.text().lower()
    assert scan.status.objectName() == "warn"


def test_the_export_summary_does_not_overwrite_the_verdict(window):
    """The bug the operator actually saw, one layer further out.

    The panel reported "Scan STOPPED at tile 5 of 12 — Frame too short"
    correctly, and then the export finished a second later and replaced it
    with "Scan done → …". The verdict must survive its own footnote.
    """
    from pathlib import Path

    from talos.cv.scan import ScanResult

    scan = window._sample_finding.scan_panel
    stopped = ScanResult(manifest_path=Path("C:/tmp/n.csv"),
                         message="Frame too short", planned=12, visited=5)
    stopped.frames = [Path(f"C:/tmp/f{i}.png") for i in range(5)]
    scan._pending_export = None
    scan._on_scan_done({"result": stopped, "out_dir": Path("C:/tmp"),
                        "fov": (100.0, 100.0)})
    verdict = scan.status.text()

    scan._export_worker = None
    scan._on_export_done({"dir": Path("C:/tmp"),
                          "written": ["candidates.csv", "mosaic.png"],
                          "samples": 0})
    text = scan.status.text()
    assert text.startswith(verdict), text
    assert "candidates.csv" in text
    assert "done" not in text.lower()
    assert scan.status.objectName() == "error"




def test_the_colour_rows_have_sliders_and_the_gates_do_not(window):
    """The tolerance rows are hunted for by eye while watching the mask, so
    they get the camera's number-plus-slider treatment; a gate's value is
    typed once and left, so its rows stay compact."""
    from talos.cv.identify import SizeStage
    from talos.ui.widgets.identify_panel import StageEditor

    colour = _chain_colour_editor(window)
    for name in ("tolerance", "spread", "min_saturation", "min_value"):
        kind, box, slider = colour._editors[name]
        assert kind == "num"
        assert slider.minimum() == 0
        assert slider.maximum() == (100 if name in ("tolerance", "spread")
                                    else 255)
        assert slider.value() == pytest.approx(round(box.value()))

        # the slider moves the number, and the number moves the slider
        slider.setValue(slider.minimum())
        assert box.value() == slider.minimum()
        box.setValue(box.maximum())
        assert slider.value() == int(round(box.value()))

    # The row is REGISTERED: ``stage()`` reads the parameters back out of
    # ``_editors``, so a slider row that skipped it would be a value the
    # operator sets and the pipeline never sees.
    colour._editors["tolerance"][1].setValue(42.0)
    colour._editors["spread"][1].setValue(12.0)
    stage = colour.stage()
    assert stage.tolerance == pytest.approx(42.0)
    assert stage.spread == pytest.approx(12.0)

    # a gate keeps the compact row
    plain = StageEditor(SizeStage())
    assert all(len(spec) == 2 for spec in plain._editors.values())


# --- the stage origin, in the run card (shared with the Navigation tab) ---

def test_the_origin_row_follows_the_state(window):
    """The two origin ACTIONS start disabled: an origin is something the
    operator MARKS, and a seeded (0, 0) would be the machine origin wearing
    that label. *Set origin* stays available — it is how you mark one."""
    scan = window._sample_finding.scan_panel
    assert not scan.go_origin_btn.isEnabled()
    assert not scan.scan_origin_btn.isEnabled()
    assert scan.set_origin_btn.isEnabled()
    assert "not set" in scan.origin_label.text()

    from talos.models import StagePosition

    window._state.set_stage_origin(StagePosition(x_um=12.5, y_um=-3.0))
    assert scan.go_origin_btn.isEnabled() and scan.scan_origin_btn.isEnabled()
    assert "12.5" in scan.origin_label.text()
    assert "-3.0" in scan.origin_label.text()

    # a run owns the axes: the two actions stand down with everything else
    scan._set_job("scan")
    assert not scan.go_origin_btn.isEnabled()
    assert not scan.scan_origin_btn.isEnabled()
    assert scan.set_origin_btn.isEnabled(), "marking a spot is still useful"
    scan._set_job(None)
    assert scan.go_origin_btn.isEnabled()


def test_the_run_card_sets_the_shared_origin(window):
    """ONE origin for both tabs: the run card's Set button runs the same
    handler as the Navigation quick action — so the value, the settings
    write and the other tab's label cannot disagree.

    The handler also has to WRITE the settings: it used to raise NameError
    halfway through, which left the origin in memory, never saved, and no
    log line — a bug that only shows up after a restart.
    """
    window._manager.last_position["zolix"] = {
        "x_um": 111.0, "y_um": -222.0, "r_deg": 0.0,
        "x_pulses": 1, "y_pulses": 2, "r_pulses": 0}
    window._sample_finding.scan_panel.set_origin_btn.click()
    assert window._state.stage_origin.x_um == pytest.approx(111.0)
    assert window._settings.section("origin")["xyr"]["x_um"] \
        == pytest.approx(111.0)
    # the Navigation tab's label reads the same state object
    assert "111.0" in window._navigation.quick_actions._origin_label.text()

    # with no position to store, it says so instead of storing (0, 0)
    window._state._stage_origin = None
    window._manager.last_position.clear()
    window._sample_finding.scan_panel.set_origin_btn.click()
    assert window._state.stage_origin is None


def test_go_to_origin_moves_there_at_the_scan_speed(window):
    from talos.models import StagePosition

    scan = window._sample_finding.scan_panel
    window._manager.submits.clear()
    window._manager.last_position["zolix"] = {
        "x_um": 100.0, "y_um": 200.0, "r_deg": 0.0,
        "x_pulses": 0, "y_pulses": 0, "r_pulses": 0}
    window._state.set_stage_origin(StagePosition(x_um=50.0, y_um=100.0))
    scan.go_to_origin()
    move = [s for s in window._manager.submits if s[1] == "move_rel_um"]
    assert len(move) == 1
    assert move[0][2] == pytest.approx(-50.0)      # dx
    assert move[0][3] == pytest.approx(-100.0)     # dy
    assert "the origin" in scan.status.text()


def test_scan_from_origin_anchors_the_plan_there_not_at_the_stage(
        window, monkeypatch):
    """The requested behaviour, in one test: the origin is the start of the
    AREA, and the run's first move goes to the first tile's CENTRE — which
    in the corner modes is inset by half a field of view. Moving to the
    origin first and scanning from there would put the first frame half a
    field away from where the operator marked the corner."""
    from talos.cv.scan import plan_path
    from talos.models import StagePosition

    scan = window._sample_finding.scan_panel
    window._manager.last_position["zolix"] = {
        "x_um": 5000.0, "y_um": 5000.0, "r_deg": 0.0,
        "x_pulses": 0, "y_pulses": 0, "r_pulses": 0}
    origin = StagePosition(x_um=-1000.0, y_um=-500.0)
    window._state.set_stage_origin(origin)

    scan.origin.set_value("corner_fit")
    fov = scan.fov()
    scan.scan_from_origin()

    # THE MAP's plan, not a helper's — the first cut of this test called
    # params_for() itself, so it passed while the map drew the plan at the
    # live position: the panel latched the origin for the RUN but the map's
    # preview still followed the stage.
    assert scan.map._plan.x0_um == pytest.approx(-1000.0)
    assert scan.map._plan.y0_um == pytest.approx(-500.0)
    first = scan.map._plan.waypoints[0]
    assert first[0] == pytest.approx(-1000.0 + fov[0] / 2)
    assert first[1] == pytest.approx(-500.0 + fov[1] / 2)

    # ... and the run really was anchored there
    assert scan._origin is not None
    assert (scan._origin.x_um, scan._origin.y_um) == (-1000.0, -500.0)

    # the centre mode keeps waypoint 0 == the origin (that is what centre
    # means), so the two modes differ in exactly the way described
    scan.origin.set_value("centre")
    scan.refresh_plan()
    assert scan.map._plan.waypoints[0][0] == pytest.approx(-1000.0)

    # A run owns its results: Clear is refused while it is filling them
    # (detection never revisits a tile, so what it drops is gone for good).
    scan.clear_results()
    assert scan._origin is not None, "Clear dropped a running run's anchor"
    assert "filling this list" in scan.status.text(), "and it did not say so"

    # ...and once the run is over, clearing releases the anchor: the plan
    # previews from the stage again.
    scan._set_job(None)
    window._state.set_mode("MANUAL")
    scan.clear_results()
    scan.refresh_plan()
    assert scan.map._plan.x0_um == pytest.approx(5000.0)


def test_scan_from_here_still_anchors_the_plan_at_the_stage(window):
    """The control for the test above: the ordinary button must preview from
    where the stage is, both before any run and after the origin latch."""
    window._manager.last_position["zolix"] = {
        "x_um": 250.0, "y_um": 350.0, "r_deg": 0.0,
        "x_pulses": 0, "y_pulses": 0, "r_pulses": 0}
    from talos.models import StagePosition

    window._state.set_stage_origin(StagePosition(x_um=-1000.0, y_um=-500.0))
    scan = window._sample_finding.scan_panel
    scan._origin = None
    scan.refresh_plan()
    assert scan.map._plan.x0_um == pytest.approx(250.0)

    scan._on_scan()                      # the real button's handler
    assert scan.map._plan.x0_um == pytest.approx(250.0)
    assert scan._origin.x_um == pytest.approx(250.0)


def test_scan_from_origin_says_so_when_nothing_is_marked(window):
    scan = window._sample_finding.scan_panel
    window._state._stage_origin = None
    started: list = []
    scan._begin_scan = started.append
    scan.scan_from_origin()
    assert started == []
    assert "No stage origin" in scan.status.text()


def test_a_worker_that_outlives_its_wait_is_reported(window):
    """The app's shutdown takes the ragged exit when a thread is still
    running: detection outlives a scan by design, so a run that just
    finished can still be draining tiles (or writing a mosaic) when the
    operator closes the window — and Qt destroys a RUNNING QThread on the
    way out, which aborts the process after a clean exit line."""
    assert window._threads_alive() is True, \
        "the detection worker runs for the session"
    window.stop_workers()
    window.threads_still_running = window._threads_alive()
    # after a real stop the worker is gone, so a clean exit is safe
    assert window.threads_still_running is False


def test_a_typo_in_the_colour_field_does_not_leave_it_disagreeing(window):
    """The field is free text and the swatch shows what valid_hex made of
    it, so a typo left the colour the mask searches on the swatch and the
    hex the operator typed in the box."""
    finding = window._sample_finding
    editor = finding.colour_group.editor
    edit = editor._editors["hex_color"][1]

    edit.setText("#zzzzzz")
    edit.editingFinished.emit()
    assert "z" not in edit.text(), "the field kept a colour the mask cannot use"
    assert valid_hex(edit.text(), None) == edit.text(), "and it is a colour"
    assert editor.hex_color() == edit.text(), \
        "the field and the colour the mask searches disagree"


# --- matching method, patch size, and the circle ---------------------------

def _chain_colour_editor(window):
    """The colour stage's FULL editor — in the Identification card, first in
    the pipeline order. The pinned card above the scroll shows only the
    colour and the patch size."""
    return window._sample_finding.identify_group.colour_editor()


def _click_segment(toggle, value) -> None:
    """Press one option's button — what the operator does.

    ``set_value`` alone does not emit: a programmatic load must not re-enter
    the persist path (the same rule the scan panel's rows follow).
    """
    toggle._buttons[value].click()


def test_the_chain_card_switches_the_matching_method(window):
    """Three methods, one selector, like the scan panel's path row — and the
    choice has to reach the pipeline, not just the settings file. The
    selector lives in the Identification card, first, because that is where
    the pipeline order starts."""
    from talos.cv.identify import METHOD_RGB, METHOD_WINDOW

    finding = window._sample_finding
    editor = _chain_colour_editor(window)
    assert finding.identify_group._editors[0] is editor, \
        "the colour stage is not first in the chain card"
    kind, toggle = editor._editors["method"]
    assert kind == "choice"
    assert toggle.value() == METHOD_WINDOW, "the default must not change masks"
    assert editor.stage().method == METHOD_WINDOW

    _click_segment(toggle, METHOD_RGB)
    assert editor.stage().method == METHOD_RGB
    # ...and it reaches the config the jobs are built from
    stage = next(s for s in finding.identify_config().stages
                 if s.NAME == "colour")
    assert stage.method == METHOD_RGB

    # the rows the method does NOT have are gone, not greyed
    spread_label, spread_row = editor._rows["spread"]
    # isHidden(), not isVisible(): a widget of a window that was never shown
    # reports isVisible() False for everything.
    assert spread_row.isHidden()
    assert spread_label.isHidden()
    assert editor._rows["tolerance"][0].text() == "Tolerance"
    assert not editor._method_note.isHidden()
    assert "not used" in editor._method_note.text()

    _click_segment(toggle, METHOD_WINDOW)
    assert not spread_row.isHidden() and not spread_label.isHidden()
    assert editor._rows["tolerance"][0].text() == "Hue tolerance"
    assert "thicknesses" in editor._method_note.text()


def test_switching_the_method_does_not_lose_the_spread(window):
    """Spread is hidden under a distance method, not cleared: coming back to
    the window must find the value the operator tuned."""
    from talos.cv.identify import METHOD_RGB

    finding = window._sample_finding
    editor = _chain_colour_editor(window)
    editor._editors["spread"][1].setValue(12.0)
    _click_segment(editor._editors["method"][1], METHOD_RGB)
    _click_segment(editor._editors["method"][1], "window")
    assert editor.stage().spread == pytest.approx(12.0)


def test_reloading_the_chain_card_restores_the_method_and_its_rows(window):
    """`reload()` runs on every workspace switch and Preferences apply, and
    its kind dispatch used to end in `setValue` — which a segmented row does
    not have. It also has to re-apply the rows that depend on the method, or
    a reloaded card would show the window's rows over a distance mask."""
    from talos.cv.identify import METHOD_HSV

    finding = window._sample_finding
    editor = _chain_colour_editor(window)
    settings = window._settings
    stages = settings.section("identify").setdefault("stages", [])
    entry = next((s for s in stages if s.get("name") == "colour"), None)
    if entry is None:
        entry = {"name": "colour"}
        stages.append(entry)
    entry["method"] = METHOD_HSV
    entry["hex_color"] = "#1c3484"

    finding.identify_group.reload()        # what refresh_settings() calls
    assert editor._editors["method"][1].value() == METHOD_HSV
    assert editor._rows["spread"][1].isHidden()
    assert editor._rows["tolerance"][0].text() == "Tolerance"


def test_the_pinned_card_is_only_the_picker(window):
    """The quick-access card shows the colour, the buttons and the patch size
    — nothing that shapes the match, which lives with the pipeline it
    shapes. A second copy of the parameters would be two places for one
    value to disagree with itself."""
    finding = window._sample_finding
    quick = finding.colour_group.editor
    assert list(quick._editors) == ["hex_color"]
    assert "Method" not in quick._rows
    assert not hasattr(finding.colour_group, "_method_rows")
    # ...and the things that ARE the picker stay
    assert finding.colour_group._patch_slider is not None
    assert finding.colour_group._hint is not None


def test_the_two_colour_editors_stay_in_step(window):
    """One stage, two views: the pinned picker and the chain card's first
    editor. Editing either must show up in the other, and neither may report
    the other's refresh as an edit (that would re-persist and re-run the mask
    on every workspace switch)."""
    finding = window._sample_finding
    quick = finding.colour_group
    chain = _chain_colour_editor(window)

    # the picker → the chain (what the dropper does)
    quick.set_hex("#1c3484")
    assert chain.hex_color() == "#1c3484"
    assert finding.colour_stage().hex_color == "#1c3484"
    assert finding.colour_rgb() == (28, 52, 132)

    # the chain → the picker
    chain.set_hex("#aacdf5")
    assert quick.hex_color() == "#aacdf5"
    assert finding.colour_rgb() == (170, 205, 245)

    # a silent reload does not look like an edit
    edits: list = []
    quick.sig_changed.connect(lambda: edits.append(1))
    chain.load(chain._stage)
    quick.reload()
    assert edits == [], "a reload reported itself as an edit"


def test_the_patch_slider_is_what_the_dropper_averages(window, monkeypatch):
    """The radius is a sampling aid, not a mask parameter: it must reach the
    sampler and must NOT re-run the pipeline when it changes."""
    from talos.ui.workspaces import sample_finding as sf

    finding = window._sample_finding
    seen: list = []
    monkeypatch.setattr(
        sf, "sample_hex_stats",
        lambda frame, x, y, radius=4: (seen.append((x, y, radius)),
                                       ("#1c3484", 3.0))[1])
    finding.live_view.set_preprocessed_frame(
        np.full((40, 60, 3), 30, np.uint8))

    finding.colour_group.set_pick_radius(9)
    assert finding.colour_group.pick_radius() == 9
    # the row shows the DIAMETER (what the disc spans), and the slider and
    # the number box agree on it
    assert finding.colour_group._patch_box.value() == 19
    assert finding.colour_group._patch_slider.value() == 19
    finding.on_pick(10, 10)
    assert seen[-1] == (10, 10, 9)

    # arming the dropper hands the same radius to the circle's drawing
    finding.arm_colour_pick()
    assert finding.live_view._pick_radius_px == 9
    finding.live_view.set_pick_mode(False)


def test_the_patch_radius_survives_a_restart(window):
    """Persisted beside the view mode: an operator who sized the patch for
    their objective should not have to size it again tomorrow."""
    finding = window._sample_finding
    finding.colour_group.set_pick_radius(7)
    finding.colour_group._on_patch_released()          # the save path
    assert window._settings.section("ui")["pick_radius_px"] == 7
    assert finding.colour_group._stored_patch_radius() == 7
