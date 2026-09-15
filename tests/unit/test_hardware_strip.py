"""HardwareStrip telemetry parsers: the stale-key regressions.

The old stage panels read keys that never exist in the payloads (zolix
position "x"/"y" vs x_pulses/...; limit_x+ vs limit_x_pos; sigmakoki
lowercase vs Axis-enum keys; the any_moving property that asdict() never
serializes). The strip parsers must read the REAL shapes.
"""

from dataclasses import asdict

from talos.hal.base import Axis
from talos.hal.devices.sigmakoki import SPEED_LEVEL_TO_HZ
from talos.models import FocusStatus, StagePosition, StageStatus
from talos.ui.widgets.hardware_strip import (
    format_focus_pos,
    parse_focus,
    parse_sigmakoki,
    parse_yudian,
    parse_zolix,
)


def test_format_focus_pos():
    assert format_focus_pos(123, 0.2) == "24.6 µm · 123 st"
    assert format_focus_pos(0, 0.2) == "0.0 µm · 0 st"
    assert format_focus_pos(-50, 0.2) == "-10.0 µm · -50 st"


def test_parse_zolix_real_payload_shape():
    payload = {
        "device": "zolix",
        "status": asdict(StageStatus(x_moving=True, limit_x_pos=True,
                                     limit_y_neg=True)),
        "position": asdict(StagePosition(x_pulses=10, y_pulses=20,
                                         r_pulses=30, x_um=6.25,
                                         y_um=12.5, r_deg=0.0375)),
    }
    parsed = parse_zolix(payload)
    assert parsed["x_um"] == 6.25
    assert parsed["y_um"] == 12.5
    assert parsed["r_deg"] == 0.0375
    assert parsed["moving"] is True  # derived — any_moving is a property
    assert parsed["limits"] == {"x+": True, "x-": False,
                                "y+": False, "y-": True}
    assert parsed["estop"] is False


def test_parse_zolix_idle_and_estop():
    payload = {
        "device": "zolix",
        "status": asdict(StageStatus(estop=True)),
        "position": asdict(StagePosition()),
    }
    parsed = parse_zolix(payload)
    assert parsed["moving"] is False
    assert parsed["estop"] is True


def test_parse_sigmakoki_enum_keys_and_speed_levels():
    payload = {
        "device": "sigmakoki",
        "status": {"x": "100", "y": "200", "z": "300", "xspd": "0",
                   "yspd": "2", "zspd": "3"},
        "position": {Axis.X: 100, Axis.Y: 200, Axis.Z: 300},
    }
    parsed = parse_sigmakoki(payload)
    assert parsed["x"] == 100
    assert parsed["y"] == 200
    assert parsed["z"] == 300
    assert parsed["speed_hz"] == SPEED_LEVEL_TO_HZ[3]
    assert parsed["levels"] == {"x": 0, "y": 2, "z": 3}


def test_sigmakoki_moving_comes_from_the_position_not_the_level():
    """The firmware's speed LEVEL persists after a stop (stopAxis clears
    `moving` only — verified in transfer_stage_controller.ino) and starts
    at level 2, so a level-based MOV lamp would lie from boot onwards."""
    from talos.ui.widgets.hardware_strip import _StageSection

    assert "moving" not in parse_sigmakoki(
        {"status": {"zspd": "3"}, "position": {Axis.Z: 10}})

    section = _StageSection.__new__(_StageSection)
    section._last_pos = None
    moving = section._is_moving
    assert moving({"x": 0, "y": 0, "z": 0}) is False        # first poll
    assert moving({"x": 0, "y": 0, "z": 0}) is False        # settled
    assert moving({"x": 0, "y": 0, "z": 25}) is True        # turning
    assert moving({"x": 0, "y": 0, "z": 25}) is False       # stopped again


def test_zolix_moving_still_uses_the_hardware_flags():
    from talos.ui.widgets.hardware_strip import _StageSection

    section = _StageSection.__new__(_StageSection)
    section._last_pos = None
    assert section._is_moving({"moving": True}) is True
    assert section._is_moving({"moving": False}) is False


def test_parse_sigmakoki_falls_back_to_status_strings():
    payload = {
        "device": "sigmakoki",
        "status": {"x": "42", "y": "43", "z": "44", "zspd": "0"},
        "position": {},
    }
    parsed = parse_sigmakoki(payload)
    assert parsed["x"] == 42 and parsed["z"] == 44
    # speed level 0 maps to 25 Hz (the table has no "stopped" entry)
    assert parsed["speed_hz"] == SPEED_LEVEL_TO_HZ[0]


def test_parse_focus_and_yudian():
    focus = parse_focus({
        "device": "focus",
        "status": asdict(FocusStatus(pos=123, mode="IDLE", blocked_dir="0")),
        "slim_bounds": (-1000, 2000),
    })
    assert focus["pos"] == 123
    assert focus["mode"] == "IDLE"
    assert focus["slim_bounds"] == (-1000, 2000)
    assert focus["blocked"] == "0"

    yudian = parse_yudian(
        {"device": "yudian", "pv": 25.3, "sv": 25.0, "output_percent": 12.5})
    assert yudian["pv"] == 25.3
    assert yudian["sv"] == 25.0
    assert yudian["out"] == 12.5
