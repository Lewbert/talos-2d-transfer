"""Input mapping tests (pure functions — no hardware)."""

from types import SimpleNamespace

import pytest

from talos.input.gamepad import map_button_event, normalize_axis


def event(code, state, ev_type="Absolute"):
    return SimpleNamespace(code=code, state=state, ev_type=ev_type)


def test_normalize_axis_deadzone():
    # XInput axes are signed 16-bit: center 0, full left -32768.
    assert normalize_axis(0, 0.2, 2.2) == 0.0            # centered
    assert normalize_axis(2000, 0.2, 2.2) == 0.0         # inside deadzone
    assert normalize_axis(-32768, 0.2, 2.2) == pytest.approx(-1.0, abs=1e-3)
    assert normalize_axis(32767, 0.2, 2.2) == pytest.approx(1.0, abs=1e-3)


def test_normalize_axis_gamma_shapes_midrange():
    # At 50% deflection with gamma 1.0 the output is 0.5; with gamma 2.2 it
    # is smaller (finer control near center).
    half = 16384
    linear = normalize_axis(half, 0.0, 1.0)
    curved = normalize_axis(half, 0.0, 2.2)
    assert linear == pytest.approx(0.5)
    assert 0 < curved < linear


def test_stick_inversion_is_applied_after_normalization():
    """`input.gamepad.invert_left_x/y` + `invert_right_x/y` are live (the
    older invert_x/invert_y keys were shipped but read by nothing)."""
    from talos.input.gamepad import normalize_stick

    assert normalize_stick(-32768, 0.2, 2.2) == pytest.approx(-1.0, abs=1e-3)
    assert normalize_stick(-32768, 0.2, 2.2, invert=True) == \
        pytest.approx(1.0, abs=1e-3)
    assert normalize_stick(2000, 0.2, 2.2, invert=True) == 0.0  # deadzone
    assert normalize_stick(16384, 0.0, 2.2, invert=True) == \
        -normalize_stick(16384, 0.0, 2.2)


def test_gamepad_reads_per_stick_inversion_settings(tmp_path):
    from talos.config import Settings
    from talos.input.gamepad import GamepadController

    settings = Settings.load(tmp_path / "s.json")
    settings.section("input")["gamepad"].update(
        {"invert_left_y": True, "invert_right_x": True})
    pad = GamepadController(settings)
    assert pad._invert == {"left_x": False, "left_y": True,
                           "right_x": True, "right_y": False}
    # a Preferences Apply must reach the running controller
    settings.section("input")["gamepad"]["invert_left_x"] = True
    pad.reload_settings()
    assert pad._invert["left_x"] is True


def test_start_select_map_to_stop_all():
    for code in ("BTN_START", "BTN_SELECT"):
        command = map_button_event(event(code, 1))
        assert command == ("stop_all", {})


def test_face_buttons_map_to_presets():
    assert map_button_event(event("BTN_SOUTH", 1)) == ("preset", {"index": 0})
    assert map_button_event(event("BTN_EAST", 1)) == ("preset", {"index": 1})
    assert map_button_event(event("BTN_WEST", 1)) == ("preset", {"index": 2})
    assert map_button_event(event("BTN_NORTH", 1)) == ("preset", {"index": 3})


def test_dpad_maps_to_focus_step():
    assert map_button_event(event("ABS_HAT0Y", -1)) == ("focus_step", {"direction": 1})
    assert map_button_event(event("ABS_HAT0Y", 1)) == ("focus_step", {"direction": -1})
    assert map_button_event(event("ABS_HAT0Y", 0)) is None


def test_mode_gate_autofocus_passes_input_scan_blocks_but_never_stops(tmp_path):
    """During AUTOFOCUS the input passes through — the autofocus
    service's sig_job_submitted hook aborts the run and the input wins.
    During SCAN motion stays gated. RELEASES must always pass in every
    mode — a gated release left the zolix unstoppable on hardware."""
    from talos.config import Settings
    from talos.input.input_system import InputSystem

    class FakeState:
        mode = "AUTOFOCUS"

    class FakeManager:
        def __init__(self):
            self.submits: list[tuple] = []

        def submit(self, device, method, *args, priority=0):
            self.submits.append((device, method, priority))

    manager = FakeManager()
    settings = Settings.load(tmp_path / "s.json")
    system = InputSystem(manager, settings, state=FakeState())

    # during AUTOFOCUS the manual start goes through (input wins)
    system._dispatch(("zolix", "x", "continuous_start", 1, 100, "keyboard"))
    assert ("zolix", "move_continuous", 0) in manager.submits
    # a RELEASE passes too (the axis must be stoppable)
    system._dispatch(("zolix", "x", "continuous_stop", 0, 0.0, "keyboard"))
    assert ("zolix", "stop_axis", 1) in manager.submits

    # during SCAN motion is gated but releases still pass
    FakeState.mode = "SCAN"
    n = len(manager.submits)
    system._dispatch(("zolix", "x", "continuous_start", 1, 100, "keyboard"))
    assert len(manager.submits) == n  # dropped
    system._last_continuous["zolix:x"] = (1, 100)
    system._dispatch(("zolix", "x", "continuous_stop", 0, 0.0, "keyboard"))
    assert ("zolix", "stop_axis", 1) in manager.submits

    # in MANUAL mode everything passes
    FakeState.mode = "MANUAL"
    system._dispatch(("zolix", "x", "continuous_start", -1, 100, "keyboard"))
    assert sum(1 for s in manager.submits if s[1] == "move_continuous") == 2


def test_analog_jitter_does_not_resend_the_same_speed_level(tmp_path):
    """The left stick emits a fresh FLOAT speed every 16 ms tick. Deduping
    on the raw value never fired, so the same firmware speed level was
    re-submitted at tick rate (the hardware log showed level 0 re-sent
    dozens of times per second — the queue then acted on commands the
    operator had already left)."""
    from talos.config import Settings
    from talos.input.input_system import InputSystem

    class FakeState:
        mode = "MANUAL"

    class FakeManager:
        def __init__(self):
            self.submits: list[tuple] = []

        def submit(self, device, method, *args, priority=0):
            self.submits.append((device, method, args, priority))

    manager = FakeManager()
    system = InputSystem(manager, Settings.load(tmp_path / "s.json"),
                         state=FakeState())

    def moves():
        return [s for s in manager.submits if s[1] == "move"]

    # 100 Hz and 104 Hz both quantise to level 2 — one command
    system._dispatch(("sigmakoki", "x", "continuous_start", 1, 100.0, "gamepad_stick"))
    system._dispatch(("sigmakoki", "x", "continuous_start", 1, 104.0, "gamepad_stick"))
    assert len(moves()) == 1
    # 300 Hz is a different level — it must go through
    system._dispatch(("sigmakoki", "x", "continuous_start", 1, 300.0, "gamepad_stick"))
    assert len(moves()) == 2
    # ... and a direction reversal even at the same level
    system._dispatch(("sigmakoki", "x", "continuous_start", -1, 300.0, "gamepad_stick"))
    assert len(moves()) == 3


def test_focus_re_press_after_release_is_not_throttled(tmp_path):
    """The 10 Hz focus throttle exists for trigger-curve sweeps; the FIRST
    command after a release must pass, or re-pressing a dialbox hold feels
    up to 100 ms late."""
    from talos.config import Settings
    from talos.input.input_system import InputSystem

    class FakeState:
        mode = "MANUAL"

    class FakeManager:
        def __init__(self):
            self.submits: list[tuple] = []

        def submit(self, device, method, *args, priority=0):
            self.submits.append((device, method, args, priority))

    manager = FakeManager()
    system = InputSystem(manager, Settings.load(tmp_path / "s.json"),
                         state=FakeState())

    system._dispatch(("focus", "z", "continuous_start", 1, 500, "ui_hold"))
    system._dispatch(("focus", "z", "continuous_stop", 0, 0.0, "ui_hold"))
    system._dispatch(("focus", "z", "continuous_start", 1, 500, "ui_hold"))
    speeds = [s for s in manager.submits if s[1] == "set_speed"]
    assert len(speeds) == 2  # release → re-press is not swallowed
    # a sweep (no release in between) stays throttled
    system._dispatch(("focus", "z", "continuous_start", 1, 400, "ui_hold"))
    assert len([s for s in manager.submits if s[1] == "set_speed"]) == 2


def _system(tmp_path, mode="MANUAL"):
    from talos.config import Settings
    from talos.input.input_system import InputSystem

    class FakeState:
        pass

    class FakeManager:
        def __init__(self):
            self.submits: list[tuple] = []
            self.stops = 0

        def submit(self, device, method, *args, priority=0):
            self.submits.append((device, method, args, priority))
            return 1

        def stop_all(self):
            self.stops += 1

    FakeState.mode = mode
    manager = FakeManager()
    system = InputSystem(manager, Settings.load(tmp_path / "s.json"),
                         state=FakeState())
    return system, manager


def test_esc_latch_holds_while_anything_is_still_held(tmp_path):
    """Regression: the predicate was `(not keys and not gamepad.connected)
    or sources_released`. `and` binds tighter than `or`, so with a gamepad
    connected and its sticks centred — the normal case — the latch
    cleared on the very NEXT tick regardless of held keys, and a held
    source that changed its command restarted motion after STOP ALL."""
    system, manager = _system(tmp_path)
    system._gamepad_state.connected = True      # gamepad present, at rest
    system.key_down("Right")                    # ... but this key is held
    system.on_escape()
    assert system._esc_latch
    system._tick()
    assert system._esc_latch, "latch cleared with a key still held"
    system.key_up("Right")
    system._tick()
    assert not system._esc_latch
    assert manager.stops == 1


def test_esc_latch_waits_for_on_screen_holds(tmp_path):
    """ESC drops the holds that existed when it fired (they would
    otherwise re-command their axis the moment the latch clears), but a
    hold pressed WHILE latched still holds the latch open — and cannot
    move anything through it."""
    system, manager = _system(tmp_path)
    system._gamepad_state.connected = True
    system.ui_hold("zolix:x", 1)
    system.on_escape()
    assert system._ui_state == {}, "ESC must release held claims"
    system.ui_hold("zolix:x", 1)          # pressed again while latched
    system._tick()
    assert system._esc_latch, "latch cleared with a UI hold still active"
    system.ui_release("zolix:x")
    system._tick()
    assert not system._esc_latch
    assert manager.submits == [], "nothing may move while latched"


def test_ui_click_respects_the_mode_gate(tmp_path):
    """A short click used to call _do_single_step directly, bypassing the
    gate — a jog button could step the stage mid-scan."""
    system, manager = _system(tmp_path, mode="SCAN")
    system.ui_click("zolix", "x", 1)
    assert manager.submits == []
    system._state.mode = "MANUAL"
    system.ui_click("zolix", "x", 1)
    assert [s[1] for s in manager.submits] == ["move_rel_um"]


def test_ui_click_respects_the_esc_latch(tmp_path):
    system, manager = _system(tmp_path)
    system.on_escape()
    system.ui_click("zolix", "x", 1)
    assert manager.submits == []


def test_cancel_all_holds_then_a_fresh_hold_works(tmp_path):
    system, _ = _system(tmp_path)
    system.ui_hold("zolix:x", 1)
    system.cancel_all_holds("test")
    assert system._ui_state == {}
    system.ui_hold("zolix:x", -1)
    assert "zolix:x" in system._ui_state


def test_stale_hold_expires(tmp_path, monkeypatch):
    """Dead-man switch: a hold whose release signal never arrives must not
    jog the axis forever."""
    import talos.input.input_system as mod

    system, _ = _system(tmp_path)
    system.ui_hold("zolix:x", 1)
    stamp = system._ui_state["zolix:x"][0]
    monkeypatch.setattr(mod.time, "monotonic",
                        lambda: stamp + mod.MAX_UI_HOLD_S + 1.0)
    system._tick()
    assert system._ui_state == {}


def test_released_buttons_do_not_fire():
    assert map_button_event(event("BTN_START", 0)) is None
    assert map_button_event(event("BTN_SOUTH", 0)) is None


# ---------------------------------------------------------------------------
# Gamepad Start / Back (reference mapping: Back cycles the D-pad stage,
# Start toggles THAT stage's enable gate — Esc is the global stop)
# ---------------------------------------------------------------------------

def _gamepad_system(tmp_path):
    from talos.config import Settings
    from talos.input.gamepad import GamepadState
    from talos.input.input_system import InputSystem

    class FakeState:
        mode = "MANUAL"

    class FakeManager:
        def __init__(self):
            self.enabled = {}
            self.broadcasts = []
            self.stops = 0

        def is_enabled(self, key):
            return self.enabled.get(key, True)

        def set_enabled(self, key, on):
            self.enabled[key] = on
            self.broadcasts.append((key, on))

        def stop_all(self):
            self.stops += 1

        def submit(self, *a, **k):
            return 1

    manager = FakeManager()
    system = InputSystem(manager, Settings.load(tmp_path / "s.json"),
                         state=FakeState())
    return system, manager, GamepadState


def test_gamepad_start_toggles_the_selected_stage_enable(tmp_path):
    system, manager, GamepadState = _gamepad_system(tmp_path)
    # the D-pad starts on the transfer stage
    assert system._resolver.dpad_stage == "sigmakoki"
    state = GamepadState(connected=True)
    state.edges = {"start": True}
    system._on_state(state)
    assert manager.enabled == {"sigmakoki": False}
    assert manager.stops == 0, "Start must not be a global stop"

    state = GamepadState(connected=True)
    state.edges = {"start": True}
    system._on_state(state)
    assert manager.enabled == {"sigmakoki": True}


def test_gamepad_start_follows_the_back_button(tmp_path):
    system, manager, GamepadState = _gamepad_system(tmp_path)
    back = GamepadState(connected=True)
    back.edges = {"back": True}
    system._on_state(back)
    assert system._resolver.dpad_stage == "zolix"
    start = GamepadState(connected=True)
    start.edges = {"start": True}
    system._on_state(start)
    assert manager.enabled == {"zolix": False}


def test_gamepad_start_does_not_change_the_latch(tmp_path):
    """Esc owns the global stop + latch; the pad's Start is a per-stage
    gate and must not arm (or clear) the latch."""
    system, _manager, GamepadState = _gamepad_system(tmp_path)
    state = GamepadState(connected=True)
    state.edges = {"start": True}
    system._on_state(state)
    assert system._esc_latch is False


def test_back_reports_the_stage_and_its_enable_state(tmp_path):
    """The status-bar indicator says which stage the D-pad drives, and the
    log line says whether that stage is currently enabled."""
    system, manager, GamepadState = _gamepad_system(tmp_path)
    manager.enabled["zolix"] = False
    stages, logs = [], []
    system.sig_dpad_stage.connect(stages.append)
    system.sig_log.connect(logs.append)
    back = GamepadState(connected=True)
    back.edges = {"back": True}
    system._on_state(back)
    assert stages == ["zolix"]
    assert "DISABLED" in logs[-1]


# ---------------------------------------------------------------------------
# Combo gestures: LT+RT = autofocus once, LB+RB = STOP ALL
# ---------------------------------------------------------------------------

def _combo_system(tmp_path, monkeypatch, mode="MANUAL"):
    import talos.input.input_system as mod

    system, manager, GamepadState = _gamepad_system(tmp_path)
    system._state.mode = mode
    clock = {"t": 500.0}
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock["t"])
    return system, manager, GamepadState, clock, mod


def test_lt_rt_runs_autofocus_once_after_the_hold(tmp_path, monkeypatch):
    system, _manager, GamepadState, clock, mod = _combo_system(tmp_path,
                                                               monkeypatch)
    fired = []
    system.sig_af_requested.connect(lambda: fired.append(1))
    gesture = GamepadState(connected=True, left_trigger=0.8, right_trigger=0.9)
    gesture.edges = {}
    system._on_state(gesture)
    assert fired == [], "a brush must not fire the gesture"
    clock["t"] += mod.COMBO_HOLD_S + 0.01
    system._on_state(gesture)
    assert fired == [1]
    clock["t"] += 1.0
    system._on_state(gesture)
    assert fired == [1], "one autofocus per press"
    # release, then press again → fires again
    released = GamepadState(connected=True)
    released.edges = {}
    system._on_state(released)
    clock["t"] += 0.05
    system._on_state(gesture)
    clock["t"] += mod.COMBO_HOLD_S + 0.01
    system._on_state(gesture)
    assert fired == [1, 1]


def test_lt_rt_is_refused_outside_manual(tmp_path, monkeypatch):
    system, _manager, GamepadState, clock, mod = _combo_system(
        tmp_path, monkeypatch, mode="SCAN")
    fired, logs = [], []
    system.sig_af_requested.connect(lambda: fired.append(1))
    system.sig_log.connect(logs.append)
    gesture = GamepadState(connected=True, left_trigger=0.9, right_trigger=0.9)
    gesture.edges = {}
    system._on_state(gesture)
    clock["t"] += mod.COMBO_HOLD_S + 0.01
    system._on_state(gesture)
    assert fired == [], "a scan owns the axes — no autofocus"
    assert "scan" in logs[-1].lower()


def test_lb_rb_stops_all_and_latches(tmp_path, monkeypatch):
    system, manager, GamepadState, clock, mod = _combo_system(tmp_path,
                                                              monkeypatch)
    gesture = GamepadState(connected=True, button_left_shoulder=True,
                           button_right_shoulder=True)
    gesture.edges = {}
    system._on_state(gesture)
    assert manager.stops == 0, "the hold time gates the stop"
    clock["t"] += mod.COMBO_HOLD_S + 0.01
    system._on_state(gesture)
    assert manager.stops == 1
    assert system._esc_latch, "the stop gesture latches like Esc"
    clock["t"] += 1.0
    system._on_state(gesture)
    assert manager.stops == 1, "one stop per press"


def test_single_bumper_stays_a_fast_modifier(tmp_path, monkeypatch):
    """Only the PAIR is a gesture — one bumper must keep meaning 'fast'."""
    system, manager, GamepadState, clock, mod = _combo_system(tmp_path,
                                                              monkeypatch)
    one = GamepadState(connected=True, button_left_shoulder=True)
    one.edges = {}
    system._on_state(one)
    clock["t"] += mod.COMBO_HOLD_S + 0.5
    system._on_state(one)
    assert manager.stops == 0
    assert system._esc_latch is False


def test_single_trigger_stays_a_focus_jog(tmp_path, monkeypatch):
    system, _manager, GamepadState, clock, mod = _combo_system(tmp_path,
                                                               monkeypatch)
    fired = []
    system.sig_af_requested.connect(lambda: fired.append(1))
    one = GamepadState(connected=True, right_trigger=1.0)
    one.edges = {}
    system._on_state(one)
    clock["t"] += mod.COMBO_HOLD_S + 0.5
    system._on_state(one)
    assert fired == []


def test_fired_gesture_owns_the_triggers_until_they_are_released(
        tmp_path, monkeypatch):
    """Hardware symptom: LT+RT fired, the autofocus was requested, and
    nothing appeared on screen.

    Releasing LT and RT a few milliseconds apart leaves the residual
    trigger commanding a focus jog. Any focus command aborts a running
    (or merely ARMED) autofocus, so the run died inside its 350 ms arm
    window: no sweep, no indicator. Suppression therefore has to outlive
    the hold, until both triggers are actually released.
    """
    system, manager, GamepadState, clock, mod = _combo_system(tmp_path,
                                                              monkeypatch)
    submits: list[tuple] = []
    manager.submit = lambda *a, **k: submits.append(a) or 1
    fired = []
    system.sig_af_requested.connect(lambda: fired.append(1))

    gesture = GamepadState(connected=True, left_trigger=0.9, right_trigger=0.9)
    gesture.edges = {}
    system._on_state(gesture)               # arms
    clock["t"] += mod.COMBO_HOLD_S + 0.01
    system._on_state(gesture)               # fires
    assert fired == [1]

    # LT released first, RT still pulled — the classic release skew
    half = GamepadState(connected=True, left_trigger=0.0, right_trigger=0.9)
    half.edges = {}
    system._gamepad_state = half
    system._on_state(half)
    system._tick()
    assert [s for s in submits if s[0] == "focus"] == [], \
        "the residual trigger jog would abort the autofocus it just started"

    # both released → the latch clears and a single trigger jogs again
    rest = GamepadState(connected=True)
    rest.edges = {}
    system._on_state(rest)
    system._gamepad_state = rest
    system._tick()
    assert "af" not in system._combo_latch
    one = GamepadState(connected=True, right_trigger=1.0)
    one.edges = {}
    system._on_state(one)
    system._gamepad_state = one
    system._tick()
    assert [s for s in submits if s[0] == "focus"], \
        "a deliberate single-trigger jog must still work"


def test_stop_gesture_also_holds_its_inputs_until_release(tmp_path,
                                                          monkeypatch):
    system, manager, GamepadState, clock, mod = _combo_system(tmp_path,
                                                              monkeypatch)
    submits: list[tuple] = []
    manager.submit = lambda *a, **k: submits.append(a) or 1
    gesture = GamepadState(connected=True, button_left_shoulder=True,
                           button_right_shoulder=True, left_x=1.0)
    gesture.edges = {}
    system._on_state(gesture)               # arms
    clock["t"] += mod.COMBO_HOLD_S + 0.01
    system._on_state(gesture)               # fires
    assert manager.stops == 1
    # RB released, LB still held: the stick must not resume jogging yet
    half = GamepadState(connected=True, button_left_shoulder=True, left_x=1.0)
    half.edges = {}
    system._gamepad_state = half
    system._on_state(half)
    system._tick()
    assert [s for s in submits if s[0] == "sigmakoki"] == []
    assert "stop" in system._combo_latch


def test_keyboard_works_without_gamepad_state_emission(tmp_path):
    """Regression: _tick bailed out until the gamepad controller emitted
    its first state — and it never emits without a gamepad connected, so
    keyboard input was dead from launch (user-found: nothing responded
    until STOP ALL was pressed once). The seeded disconnected state must
    let the tick loop run immediately."""
    from talos.config import Settings
    from talos.input.input_system import InputSystem

    class FakeState:
        mode = "MANUAL"

    class FakeManager:
        def __init__(self):
            self.submits: list[tuple] = []

        def submit(self, device, method, *args, priority=0):
            self.submits.append((device, method, args, priority))

    manager = FakeManager()
    settings = Settings.load(tmp_path / "s.json")
    system = InputSystem(manager, settings, state=FakeState())
    # no gamepad state ever emitted (no gamepad connected)
    system.key_down("equal")
    system._tick()
    methods = [(s[0], s[1]) for s in manager.submits]
    assert ("focus", "set_speed") in methods
    signed = manager.submits[0][2][0]
    assert signed > 0  # '+' key = positive = distance increase


def test_autorepeat_churn_does_not_fire_phantom_single_step(tmp_path, monkeypatch):
    """Regression: some IMEs deliver auto-repeat as release+press pairs,
    which used to refresh the key's press timestamp — the release of a
    LONG jog then looked like a short press and fired a single step after
    the continuous stop (user-found: zolix always added one step when a
    jog stopped, making controls laggy)."""
    from talos.config import Settings
    from talos.input.input_system import InputSystem

    clock = {"t": 1000.0}
    monkeypatch.setattr("time.monotonic", lambda: clock["t"])

    class FakeState:
        mode = "MANUAL"

    class FakeManager:
        def __init__(self):
            self.submits: list[tuple] = []

        def submit(self, device, method, *args, priority=0):
            self.submits.append((device, method, args, priority))

    manager = FakeManager()
    settings = Settings.load(tmp_path / "s.json")
    system = InputSystem(manager, settings, state=FakeState())

    system.key_down("d")
    # auto-repeat churn during the hold (release+press pairs, < 50 ms gaps)
    clock["t"] += 0.02
    system.key_up("d")
    system.key_down("d")
    clock["t"] += 0.02
    system.key_up("d")
    system.key_down("d")
    # hold continues past the long-press threshold → continuous starts
    clock["t"] += 0.30
    system._tick()
    methods = [(s[0], s[1]) for s in manager.submits]
    assert ("zolix", "move_continuous") in methods
    # real release → stop, and NO trailing single step
    clock["t"] += 0.20
    system._tick()  # still held
    system.key_up("d")
    clock["t"] += 0.01
    system._tick()
    methods = [(s[0], s[1]) for s in manager.submits]
    assert ("zolix", "stop_axis") in methods
    assert not any(m == ("zolix", "move_rel_um") for m in methods)
