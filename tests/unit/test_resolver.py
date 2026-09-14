"""ActionResolver port tests: reference behaviors on synthetic inputs."""

import time

import pytest

from talos.input.action_resolver import (
    ActionResolver,
    focus_trigger_to_speed,
)

T0 = 1000.0


class FakeGamepad:
    connected = True
    left_x = left_y = right_x = right_y = 0.0
    left_trigger = right_trigger = 0.0
    dpad_up = dpad_down = dpad_left = dpad_right = False
    button_start = button_back = False
    button_a = button_b = button_x = button_y = False
    button_left_shoulder = button_right_shoulder = False


class FakeSettings:
    def __init__(self):
        self._z = {"slow_speed_pps": 500, "fast_speed_pps": 2000,
                   "slow_speed_r": 1000, "fast_speed_r": 10000,
                   "single_step": 5, "um_per_pulse_xy": 0.625,
                   "single_step_r": 80, "um_per_pulse_r": 0.00125}
        self._sk = {"slow_speed_hz": 100, "fast_speed_hz": 1000,
                    "slow_speed_z": 99, "fast_speed_z": 500, "single_step": 3}
        self._fc = {"min_speed": 50, "max_speed": 2000, "gamma": 2.2,
                    "deadzone": 0.05, "invert": False, "step_size": 10}

    def device(self, key):
        return {"zolix": self._z, "sigmakoki": self._sk, "focus": self._fc}[key]

    def section(self, key):
        return {}

    def get(self, key, default=None):
        return default  # no objectives → the multiplier defaults to 1.0


@pytest.fixture()
def resolver():
    return ActionResolver(FakeSettings())


def tick(resolver, key_state, prev, gamepad, now):
    commands = []
    resolver.resolve(key_state, prev, gamepad, now=now,
                     on_command=commands.append)
    return commands


def test_focus_trigger_to_speed_reference_values():
    assert focus_trigger_to_speed(0.0, 0.0, min_speed=50, max_speed=2000) == 0
    assert focus_trigger_to_speed(0.0, 0.02, min_speed=50, max_speed=2000) == 0  # inside deadzone
    s = focus_trigger_to_speed(0.0, 1.0, min_speed=50, max_speed=2000)
    assert s == 2000  # full RT = max speed
    s = focus_trigger_to_speed(1.0, 0.0, min_speed=50, max_speed=2000)
    assert s == -2000  # full LT = negative
    # Just above the deadzone → min speed.
    s = focus_trigger_to_speed(0.0, 0.1, min_speed=50, max_speed=2000)
    assert 50 <= s < 500


def test_keyboard_short_press_single_step(resolver):
    pad = FakeGamepad()
    now = T0
    down = {"d": now}
    commands = tick(resolver, down, {}, pad, now)
    assert commands == []  # press alone does nothing before threshold
    # Release at +0.1 s (< 300 ms) → single step.
    now += 0.1
    commands = tick(resolver, {}, down, pad, now)
    modes = [c[2] for c in commands]
    assert "single_step" in modes
    step = [c for c in commands if c[2] == "single_step"][0]
    assert step[0] == "zolix" and step[1] == "x" and step[3] == 1


def test_keyboard_long_press_continuous_then_stop(resolver):
    pad = FakeGamepad()
    now = T0
    down = {"w": now}
    commands = tick(resolver, down, {}, pad, now)
    assert commands == []
    now += 0.4  # past the 300 ms threshold
    commands = tick(resolver, down, down, pad, now)
    starts = [c for c in commands if c[2] == "continuous_start"]
    assert any(c[0] == "zolix" and c[1] == "y" and c[3] == 1 for c in starts)
    # Release → stop.
    commands = tick(resolver, {}, down, pad, now + 0.01)
    stops = [c for c in commands if c[2] == "continuous_stop"]
    assert any(c[0] == "zolix" and c[1] == "y" for c in stops)


def test_focus_key_starts_continuous_immediately(resolver):
    pad = FakeGamepad()
    now = T0
    down = {"equal": now}
    commands = tick(resolver, down, {}, pad, now)
    starts = [c for c in commands if c[2] == "continuous_start"]
    assert any(c[0] == "focus" for c in starts)
    focus = [c for c in starts if c[0] == "focus"][0]
    assert focus[4] == 500.0  # slow = max(10, 50, 2000//4)
    # Shift held → fast.
    down["Shift_L"] = now
    commands = tick(resolver, down, down, pad, now + 0.01)
    starts = [c for c in commands if c[2] == "continuous_start" and c[0] == "focus"]
    assert starts and starts[0][4] == 2000.0


def test_focus_keys_drive_opposite_directions(resolver):
    """Regression: the direction was derived from the (always-positive)
    focus key speed, so '+' and '-' jogged the SAME direction (user-found
    on hardware). The key map's direction must win."""
    pad = FakeGamepad()
    commands = tick(resolver, {"minus": T0}, {}, pad, T0)
    focus = [c for c in commands
             if c[0] == "focus" and c[2] == "continuous_start"][0]
    assert focus[3] == -1
    assert focus[4] == 500.0
    commands = tick(resolver, {"plus": T0 + 1}, {"minus": T0}, pad, T0 + 1)
    focus = [c for c in commands
             if c[0] == "focus" and c[2] == "continuous_start"][0]
    assert focus[3] == 1


def test_shift_fast_for_zolix_keyboard(resolver):
    pad = FakeGamepad()
    now = T0
    down = {"d": now}
    tick(resolver, down, {}, pad, now)
    now += 0.4
    commands = tick(resolver, down, down, pad, now)
    slow = [c for c in commands if c[2] == "continuous_start" and c[0] == "zolix"][0]
    assert slow[4] == 500.0
    tick(resolver, {}, down, pad, now)
    down2 = {"d": now, "Shift_L": now}
    tick(resolver, down2, {}, pad, now + 0.01)
    commands = tick(resolver, down2, down2, pad, now + 0.4)
    fast = [c for c in commands if c[2] == "continuous_start" and c[0] == "zolix"][0]
    assert fast[4] == 2000.0


def test_focus_ui_hold_uses_focus_speed_and_stops_on_release(resolver):
    """StageControlWindow focus holds route through ui_hold('focus:z'):
    the speed must come from the FOCUS key speed (min/max × objective
    multiplier), not the stage-z table, and the release must stop."""
    pad = FakeGamepad()
    now = T0
    commands = []
    resolver.resolve({}, {}, pad, now=now, on_command=commands.append,
                     ui_state={"focus:z": (now, 1)})
    starts = [c for c in commands
              if c[2] == "continuous_start" and c[0] == "focus"]
    assert starts and starts[0][4] == 500.0  # focus slow key speed
    assert starts[0][3] == 1
    assert starts[0][5] == "ui_button"
    # Release (ui_state cleared) → continuous_stop.
    commands = []
    resolver.resolve({}, {}, pad, now=now + 0.01, on_command=commands.append,
                     ui_state={})
    stops = [c for c in commands
             if c[2] == "continuous_stop" and c[0] == "focus"]
    assert stops and stops[0][5] == "ui_button"


def test_focus_ui_hold_fast_flag():
    """The dialbox's double-arrow hold (3-tuple with fast=True) must
    drive the FAST focus speed."""
    resolver = ActionResolver(FakeSettings())
    pad = FakeGamepad()
    now = T0
    commands = []
    resolver.resolve({}, {}, pad, now=now, on_command=commands.append,
                     ui_state={"focus:z": (now, 1, True)})
    starts = [c for c in commands
              if c[2] == "continuous_start" and c[0] == "focus"]
    assert starts and starts[0][4] == 2000.0  # fast = max speed
    assert starts[0][3] == 1
    # slow 3-tuple stays at the slow key speed
    commands = []
    resolver.resolve({}, {}, pad, now=now + 0.01, on_command=commands.append,
                     ui_state={"focus:z": (now + 0.01, -1, False)})
    starts = [c for c in commands
              if c[2] == "continuous_start" and c[0] == "focus"]
    assert starts and starts[0][4] == 500.0
    assert starts[0][3] == -1


def test_stage_ui_hold_fast_flag_uses_fast_table():
    resolver = ActionResolver(FakeSettings())
    pad = FakeGamepad()
    now = T0
    commands = []
    resolver.resolve({}, {}, pad, now=now, on_command=commands.append,
                     ui_state={"zolix:y": (now, 1, True)})
    starts = [c for c in commands if c[2] == "continuous_start"]
    assert starts and starts[0][4] == 2000.0  # zolix fast 2000 pps


def test_focus_ui_hold_suppresses_triggers(resolver):
    """A held UI focus button claims focus:z — the triggers back off."""
    pad = FakeGamepad()
    pad.right_trigger = 1.0
    now = T0
    commands = []
    resolver.resolve({}, {}, pad, now=now, on_command=commands.append,
                     ui_state={"focus:z": (now, -1)})
    starts = [c for c in commands
              if c[2] == "continuous_start" and c[0] == "focus"]
    assert starts and starts[0][5] == "ui_button"
    assert not [c for c in commands
                if c[2] == "continuous_start" and c[0] == "focus"
                and c[5] == "gamepad_trigger"]


def test_gamepad_triggers_drive_focus(resolver):
    pad = FakeGamepad()
    pad.right_trigger = 1.0
    commands = tick(resolver, {}, {}, pad, T0)
    starts = [c for c in commands if c[2] == "continuous_start" and c[0] == "focus"]
    assert starts and starts[0][4] == 2000.0
    pad.right_trigger = 0.0
    commands = tick(resolver, {}, {}, pad, T0 + 0.05)
    stops = [c for c in commands if c[2] == "continuous_stop" and c[0] == "focus"]
    assert stops


def test_gamepad_left_stick_analog_sigmakoki(resolver):
    pad = FakeGamepad()
    pad.left_x = 0.5
    commands = tick(resolver, {}, {}, pad, T0)
    starts = [c for c in commands if c[2] == "continuous_start"]
    assert any(c[0] == "sigmakoki" and c[1] == "x" and c[4] == pytest.approx(50.0)
               for c in starts)
    pad.left_x = 0.0
    commands = tick(resolver, {}, {}, pad, T0 + 0.05)
    stops = [c for c in commands if c[2] == "continuous_stop"]
    assert any(c[0] == "sigmakoki" and c[1] == "x" for c in stops)


def test_gamepad_right_stick_8dir_zolix(resolver):
    pad = FakeGamepad()
    pad.right_x = 1.0  # full right → cardinal East
    # First tick out of the deadzone commits in 1 tick (reference fast path).
    commands = tick(resolver, {}, {}, pad, T0)
    starts = [c for c in commands if c[2] == "continuous_start"]
    assert any(c[0] == "zolix" and c[1] == "x" and c[3] == 1 for c in starts)
    # Recenter → stops only the active axis.
    pad.right_x = 0.0
    commands = tick(resolver, {}, {}, pad, T0 + 0.05)
    stops = [c for c in commands if c[2] == "continuous_stop"]
    assert any(c[0] == "zolix" and c[1] == "x" for c in stops)


def test_dpad_short_press_step_with_cooldown(resolver):
    pad = FakeGamepad()
    # D-pad drives the SELECTED stage (default sigmakoki).
    pad.dpad_right = True
    tick(resolver, {}, {}, pad, T0)  # press registered
    pad.dpad_right = False
    commands = tick(resolver, {}, {}, pad, T0 + 0.1)
    steps = [c for c in commands if c[2] == "single_step"]
    assert any(c[0] == "sigmakoki" and c[1] == "x" and c[3] == 1 for c in steps)
    # Immediate re-press within 0.2 s → cooldown blocks the step.
    pad.dpad_right = True
    tick(resolver, {}, {}, pad, T0 + 0.12)
    pad.dpad_right = False
    commands = tick(resolver, {}, {}, pad, T0 + 0.15)
    assert not [c for c in commands if c[2] == "single_step"]
    # After the cooldown, the step fires again.
    pad.dpad_right = True
    tick(resolver, {}, {}, pad, T0 + 0.4)
    pad.dpad_right = False
    commands = tick(resolver, {}, {}, pad, T0 + 0.45)
    assert [c for c in commands if c[2] == "single_step"]


def test_dpad_stage_toggle_switches_target(resolver):
    assert resolver.dpad_stage == "sigmakoki"
    assert resolver.toggle_dpad_stage() == "zolix"
    pad = FakeGamepad()
    pad.dpad_right = True
    tick(resolver, {}, {}, pad, T0)
    pad.dpad_right = False
    commands = tick(resolver, {}, {}, pad, T0 + 0.1)
    steps = [c for c in commands if c[2] == "single_step"]
    assert steps and steps[0][0] == "zolix"


def test_face_buttons_map_x_y_to_rotation(resolver):
    pad = FakeGamepad()
    pad.button_x = True
    tick(resolver, {}, {}, pad, T0)
    pad.button_x = False
    commands = tick(resolver, {}, {}, pad, T0 + 0.1)
    steps = [c for c in commands if c[2] == "single_step"]
    assert any(c[0] == "zolix" and c[1] == "r" and c[3] == -1 for c in steps)


def test_keyboard_priority_over_gamepad(resolver):
    pad = FakeGamepad()
    pad.left_x = 0.5  # gamepad wants sigmakoki X
    down = {"d": T0}  # keyboard drives zolix X
    commands = tick(resolver, down, {}, pad, T0)
    # The zolix axis claimed by keyboard; sigmakoki stick still works
    # (different stage) — the reference suppresses per stage:axis.
    starts = [c for c in commands if c[2] == "continuous_start"]
    assert any(c[0] == "sigmakoki" for c in starts)


# ---------------------------------------------------------------------------
# Per-objective manual speed multipliers
# ---------------------------------------------------------------------------

class ObjectiveSettings(FakeSettings):
    """Settings with an objectives table (v5 rows)."""

    def __init__(self, rows):
        super().__init__()
        self._rows = rows

    def get(self, key, default=None):
        if key == "objectives":
            return self._rows
        return default


class FakeState:
    def __init__(self, objective=0):
        self.objective = objective


_OBJECTIVE_ROWS = [
    {"name": "5x", "mag": 5, "na": 0.15, "stage_speed_multiplier": 1.0,
     "focus_manual_multiplier": 1.0},
    {"name": "10x", "mag": 10, "na": 0.30, "stage_speed_multiplier": 0.5,
     "focus_manual_multiplier": 0.25},
]


def test_stage_multiplier_scales_keyboard_jog():
    r = ActionResolver(ObjectiveSettings(_OBJECTIVE_ROWS),
                       state=FakeState(objective=1))
    pad = FakeGamepad()
    commands = tick(r, {"w": T0}, {}, pad, T0)
    assert commands == []
    commands = tick(r, {"w": T0}, {"w": T0}, pad, T0 + 0.4)
    starts = [c for c in commands if c[2] == "continuous_start"]
    jog = [c for c in starts if c[0] == "zolix" and c[1] == "y"][0]
    assert jog[4] == 500 * 0.5  # slow 500 pps × stage mult 0.5


def test_focus_manual_multiplier_scales_key_and_trigger():
    r = ActionResolver(ObjectiveSettings(_OBJECTIVE_ROWS),
                       state=FakeState(objective=1))
    pad = FakeGamepad()
    # keyboard slow = max(10, 50, 2000//4) = 500 × 0.25
    commands = tick(r, {"equal": T0}, {}, pad, T0)
    focus = [c for c in commands if c[2] == "continuous_start"
             and c[0] == "focus"][0]
    assert focus[4] == 500 * 0.25
    # keyboard fast = 2000 × 0.25
    commands = tick(r, {"equal": T0, "Shift_L": T0}, {"equal": T0}, pad, T0 + 0.01)
    focus = [c for c in commands if c[2] == "continuous_start"
             and c[0] == "focus"][-1]
    assert focus[4] == 2000 * 0.25
    # trigger full range = 2000 × 0.25
    pad2 = FakeGamepad()
    pad2.right_trigger = 1.0
    commands = tick(r, {}, {}, pad2, T0 + 0.02)
    trig = [c for c in commands if c[2] == "continuous_start"
            and c[0] == "focus"][0]
    assert trig[4] == 2000 * 0.25


def test_stage_multiplier_scales_gamepad_paths():
    r = ActionResolver(ObjectiveSettings(_OBJECTIVE_ROWS),
                       state=FakeState(objective=1))
    pad = FakeGamepad()
    pad.left_x = 1.0  # sigmakoki stick, slow 100 hz × 0.5
    commands = tick(r, {}, {}, pad, T0)
    stick = [c for c in commands if c[2] == "continuous_start"
             and c[0] == "sigmakoki" and c[1] == "x"][0]
    assert stick[4] == 100 * 0.5
    # dpad (sigmakoki) long press: 100 hz × 0.5
    pad2 = FakeGamepad()
    pad2.dpad_up = True
    tick(r, {}, {}, pad2, T0)
    commands = tick(r, {}, {}, pad2, T0 + 0.4)
    dpad = [c for c in commands if c[2] == "continuous_start"
            and c[0] == "sigmakoki"][0]
    assert dpad[4] == 100 * 0.5


def test_state_none_or_objectiveless_defaults_to_1():
    r = ActionResolver(ObjectiveSettings(_OBJECTIVE_ROWS))  # state=None
    assert r._objective_multiplier("stage") == 1.0
    assert r._objective_multiplier("focus") == 1.0

    class NoAttrState:
        pass

    r2 = ActionResolver(ObjectiveSettings(_OBJECTIVE_ROWS),
                        state=NoAttrState())
    assert r2._objective_multiplier("stage") == 1.0


def test_objective_multiplier_zero_means_unset():
    rows = [{"stage_speed_multiplier": 0.0, "focus_manual_multiplier": 0.0}]
    r = ActionResolver(ObjectiveSettings(rows), state=FakeState(objective=0))
    assert r._objective_multiplier("stage") == 1.0
    assert r._objective_multiplier("focus") == 1.0
