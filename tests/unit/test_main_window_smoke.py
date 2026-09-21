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
    # re-stating the SAME plan keeps them (a finished run must not erase
    # what it captured), but a genuinely new area drops them
    scan.map.set_plan(scan.map._plan)
    assert len(scan.map._tiles) == 2
    scan.width.setValue(scan.width.value() * 2)
    assert not scan.map._tiles


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


def _result_stub():
    class _Result:
        candidates: list = []
        summary = "colour 1"

    return _Result()


def _tile_frame():
    import numpy as np

    return np.full((32, 48, 3), 200, np.uint8)


def test_stop_all_aborts_the_run_not_just_the_motion(window):
    """Esc must end a scan. Without this wiring the abort flag stayed
    clear, the controller stopped, and the run continued at the next
    waypoint — which looked exactly like Esc doing nothing."""
    scan = window._sample_finding.scan_panel
    aborted: list = []
    scan._on_abort = lambda reason="abort": aborted.append(reason)
    scan._set_job("scan")
    window._manager.sig_stop_all_done.emit()
    assert aborted == ["stop all"]
    scan._set_job(None)
    window._manager.sig_stop_all_done.emit()
    assert aborted == ["stop all"]              # idle: nothing to abort


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
    scan._show_candidates([FlakeCandidate(x_px=1.0, y_px=2.0, area_px2=10.0,
                                          area_um2=42.0, x_um=x_um, y_um=y_um,
                                          score=7.0)], "live view")
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
    scan = _found_sample(window)
    window._state.set_mode("SCAN")
    from PySide6.QtWidgets import QMessageBox

    seen: list = []
    original = QMessageBox.information
    QMessageBox.information = staticmethod(lambda *a, **kw: seen.append(a))
    try:
        scan._on_go_to()
    finally:
        QMessageBox.information = original
    assert seen, "it must say why, not silently do nothing"
    assert not any(s[1] == "move_rel_um" for s in window._manager.submits)


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
    assert [scan._table.horizontalHeaderItem(i).text() for i in range(5)] == \
        ["#", "X µm", "Y µm", "Area µm²", "Edge"]
    # not inside any scroll area
    parent = scan._table.parentWidget()
    while parent is not None and parent is not scan:
        assert not isinstance(parent, QScrollArea), \
            "the samples must not scroll with the scan settings"
        parent = parent.parentWidget()


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


