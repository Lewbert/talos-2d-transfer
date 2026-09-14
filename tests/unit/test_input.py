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
