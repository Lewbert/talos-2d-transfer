"""Manual-control axis mapping: per-axis inversion + the X↔Y axis swap.

The mapping is applied in ``InputSystem._dispatch`` — the single choke
point every manual source funnels through. These tests pin the reference
ORDER (flip the axis identity, then invert the swapped axis) and that the
keyboard, the gamepad, the on-screen holds and the single-click path all
end up commanding the same physical axis and direction.
"""

import pytest

from talos.config import Settings
from talos.hal.base import Axis, Direction
from talos.hal.devices.zolix import DIR_NEG, DIR_POS
from talos.input.axis_map import AxisMap, IDENTITY, axis_map_for, axis_maps
from talos.input.gamepad import GamepadState
from talos.input.input_system import InputSystem


class FakeState:
    mode = "MANUAL"


class FakeManager:
    def __init__(self):
        self.submits: list[tuple] = []

    def submit(self, device, method, *args, priority=0):
        self.submits.append((device, method, args))
        return len(self.submits)

    def stop_all(self):
        pass

    def is_enabled(self, key):
        return True

    def set_enabled(self, key, on):
        pass


@pytest.fixture()
def system(tmp_path):
    settings = Settings.load(tmp_path / "s.json")
    manager = FakeManager()
    return InputSystem(manager, settings, state=FakeState()), manager, settings


# --- the pure map ---------------------------------------------------------

def test_identity_map_changes_nothing():
    assert IDENTITY.apply("x", 1) == ("x", 1)
    assert IDENTITY.apply("r", -1) == ("r", -1)


def test_flip_xy_swaps_the_axis_identity():
    mapping = AxisMap(flip_xy=True)
    assert mapping.apply("x", 1) == ("y", 1)
    assert mapping.apply("y", -1) == ("x", -1)
    # the rotation axis is never swapped
    assert mapping.apply("r", 1) == ("r", 1)


def test_invert_follows_the_swapped_axis():
    """Reference order: flip X↔Y FIRST, then invert the swapped name —
    invert_y + flip_xy must invert what is now called y (the physical Y,
    which the "x" command drives)."""
    mapping = AxisMap(invert={"y": True}, flip_xy=True)
    assert mapping.apply("x", 1) == ("y", -1)
    assert mapping.apply("y", 1) == ("x", 1)


def test_stop_direction_is_never_negated():
    mapping = AxisMap(invert={"x": True}, flip_xy=True)
    assert mapping.apply("x", 0) == ("y", 0)


def test_axis_map_for_reads_the_device_keys(tmp_path):
    settings = Settings.load(tmp_path / "s.json")
    settings.device("zolix").update({"invert_x": True, "invert_r": True,
                                     "flip_xy": True})
    settings.device("focus")["invert"] = True
    maps = axis_maps(settings)
    assert maps["zolix"].invert == {"x": True, "y": False, "r": True}
    assert maps["zolix"].flip_xy is True
    # an all-false invert dict IS the neutral map, by behaviour
    assert maps["sigmakoki"].apply("x", 1) == IDENTITY.apply("x", 1)
    assert maps["focus"].invert == {"z": True}
    assert axis_map_for(settings, "unknown") == IDENTITY


# --- the dispatch choke point --------------------------------------------

def test_stage_jog_direction_is_inverted(system):
    system, manager, settings = system
    settings.device("zolix")["invert_x"] = True
    system.reload_settings()
    system._dispatch(("zolix", "x", "continuous_start", 1, 500, "keyboard"))
    assert manager.submits[-1] == ("zolix", "move_continuous",
                                   ("x", DIR_NEG, 500))


def test_every_manual_entry_point_agrees(system):
    """Keyboard, on-screen hold, single click and the gamepad stick must
    all be mapped — a source that bypasses the choke point would jog the
    opposite way from the others."""
    system, manager, settings = system
    settings.device("sigmakoki").update({"invert_x": True})
    settings.device("zolix").update({"invert_x": True})
    settings.device("focus")["invert"] = True
    system.reload_settings()
    manager.submits.clear()
    system._last_continuous.clear()   # each source starts from rest

    # 1) keyboard: 'd' = zolix x +1 (aged past the long-press threshold so
    # the resolver emits the continuous jog rather than a single step)
    system.key_down("d")
    system._key_state["d"] -= 1.0
    system._tick()
    system.key_up("d")
    assert ("zolix", "move_continuous", ("x", DIR_NEG, 500)) in manager.submits

    # 2) on-screen hold on the zolix X button
    manager.submits.clear()
    system._last_continuous.clear()   # each source starts from rest
    system.ui_hold("zolix:x", 1)
    system._tick()
    system.ui_release("zolix:x")
    assert any(s[1] == "move_continuous" and s[2][1] == DIR_NEG
               for s in manager.submits)

    # 3) on-screen single click (sigmakoki X → NEGATIVE steps)
    manager.submits.clear()
    system._last_continuous.clear()   # each source starts from rest
    system.ui_click("sigmakoki", "x", 1)
    assert manager.submits == [("sigmakoki", "step",
                                (Axis.X, Direction.NEGATIVE, 3))]

    # 4) gamepad left stick (sigmakoki X)
    manager.submits.clear()
    system._last_continuous.clear()   # each source starts from rest
    system._gamepad_state = GamepadState(connected=True, left_x=1.0)
    system._tick()
    assert any(s[0] == "sigmakoki" and s[1] == "move"
               and s[2][1] == Direction.NEGATIVE for s in manager.submits)


def test_focus_trigger_inversion_matches_the_readout(system):
    """`devices.focus.invert` is applied ONCE (in the dispatcher): the
    focus triggers, the +/- keys and the dialbox holds must agree, and the
    trigger bar's displayed speed must be the commanded one."""
    from talos.input.action_resolver import focus_trigger_to_speed

    system, manager, settings = system
    settings.device("focus")["invert"] = True
    system.reload_settings()
    system._gamepad_state = GamepadState(connected=True, right_trigger=1.0)
    system._tick()
    speeds = [s[2][0] for s in manager.submits if s[1] == "set_speed"]
    assert speeds == [-2000]
    # the bar (which still computes with invert=True) shows the same value
    shown = focus_trigger_to_speed(0.0, 1.0, min_speed=50, max_speed=2000,
                                   gamma=2.2, deadzone=0.05, invert=True)
    assert shown == speeds[-1]

    # the +/- keys take the same path (they did not honour the key before)
    system._dispatch(("focus", "z", "continuous_stop", 0, 0.0, "keyboard"))
    manager.submits.clear()
    system._last_continuous.clear()   # each source starts from rest
    system._dispatch(("focus", "z", "continuous_start", 1, 500, "keyboard"))
    assert manager.submits[-1] == ("focus", "set_speed", (-500,))


def test_reload_settings_is_live(system):
    """A Preferences Apply must not need an app restart (the speeds and
    the mapping used to be cached for the process lifetime)."""
    system, manager, settings = system
    system._dispatch(("sigmakoki", "y", "continuous_start", 1, 100, "keyboard"))
    assert manager.submits[-1][2] == (Axis.Y, Direction.POSITIVE, 2)

    settings.device("sigmakoki")["invert_y"] = True
    settings.device("sigmakoki")["fast_speed_hz"] = 1500
    system.reload_settings()
    system._dispatch(("sigmakoki", "y", "continuous_start", 1, 1500, "keyboard"))
    assert manager.submits[-1][2][1] == Direction.NEGATIVE


def test_stop_follows_the_flipped_axis(system):
    """A stop must address the axis the start used: the flip applies to
    stops too (their direction 0 is not negated)."""
    system, manager, settings = system
    settings.device("zolix")["flip_xy"] = True
    system.reload_settings()
    system._dispatch(("zolix", "x", "continuous_start", 1, 500, "keyboard"))
    assert manager.submits[-1][2][0] == "y"
    system._dispatch(("zolix", "x", "continuous_stop", 0, 0.0, "keyboard"))
    assert manager.submits[-1][0] == "zolix"
    assert manager.submits[-1][1] == "stop_axis"
    assert manager.submits[-1][2] == ("y",)
