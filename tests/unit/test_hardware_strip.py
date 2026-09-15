"""HardwareStrip telemetry parsers: the stale-key regressions.

The old stage panels read keys that never exist in the payloads (zolix
position "x"/"y" vs x_pulses/...; limit_x+ vs limit_x_pos; sigmakoki
lowercase vs Axis-enum keys; the any_moving property that asdict() never
serializes). The strip parsers must read the REAL shapes.
"""

from dataclasses import asdict

import pytest

from talos.hal.base import Axis
from talos.models import FocusStatus, StagePosition, StageStatus
from talos.ui.widgets.hardware_strip import (
    format_focus_pos,
    parse_focus,
    parse_sigmakoki,
    parse_yudian,
    parse_zolix,
)


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    """The state-word and readout tests build QFont/QFontMetrics, which need
    a QGuiApplication: running this file ALONE (no earlier test creating one)
    did not fail the test — it killed the interpreter with a Windows access
    violation, which reads like a code bug rather than a missing fixture."""
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


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


def test_parse_sigmakoki_enum_keys():
    # the payload still carries the firmware's per-axis speed
    # levels; nothing reads them (the strip shows positions and
    # limits, the Stage Control panel has its own table).
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


def test_both_stages_report_the_same_limit_shape():
    """The strip draws both stages through ONE code path, so the parsers
    must hand it the same {'x+': bool, ...} shape — the XYZ dots used to be
    forced off because nothing read its LIMITS? at all."""
    zolix = parse_zolix({
        "status": asdict(StageStatus(limit_x_pos=True, limit_y_neg=True)),
        "position": asdict(StagePosition()),
    })
    sigmakoki = parse_sigmakoki({
        "status": {"xspd": "0", "yspd": "0", "zspd": "0"},
        "position": {Axis.X: 0, Axis.Y: 0, Axis.Z: 0},
        "limits": {"x+": True, "x-": False, "y+": False, "y-": True,
                   "z+": False, "z-": False},
    })
    for parsed in (zolix, sigmakoki):
        assert parsed["limits"] == {"x+": True, "x-": False,
                                    "y+": False, "y-": True}


def test_sigmakoki_limits_are_absent_when_the_payload_has_none():
    parsed = parse_sigmakoki({"status": {}, "position": {}})
    assert parsed["limits"] is None      # the dots stay unlit, not stale


def test_parse_sigmakoki_falls_back_to_status_strings():
    payload = {
        "device": "sigmakoki",
        "status": {"x": "42", "y": "43", "z": "44", "zspd": "0"},
        "position": {},
    }
    parsed = parse_sigmakoki(payload)
    assert parsed["x"] == 42 and parsed["z"] == 44
    # speed level 0 maps to 25 Hz (the table has no "stopped" entry)


def test_temp_power_colour_ladder():
    """The reference project's heat-status ladder, applied to the PWR
    readout: full power red → orange → amber, then green when settled on
    the setpoint, light green when close, blue otherwise."""
    from talos.ui.widgets.hardware_strip import temp_power_color

    assert temp_power_color(25.0, 25.0, 95.0) == "#e53935"   # full power
    assert temp_power_color(25.0, 200.0, 50.0) == "#fb8c00"  # moderate
    assert temp_power_color(25.0, 200.0, 20.0) == "#fdd835"  # gentle
    assert temp_power_color(25.0, 25.0, 0.0) == "#43a047"    # settled
    assert temp_power_color(25.0, 26.0, 0.0) == "#66bb6a"    # near
    assert temp_power_color(25.0, 40.0, 0.0) == "#1e88e5"    # far off
    # no reading → no colour (the label falls back to the theme's dim)
    assert temp_power_color(None, 25.0, 0.0) is None
    assert temp_power_color(25.0, None, 0.0) is None
    assert temp_power_color(25.0, 25.0, None) is None


def test_power_is_ordered_by_power_before_delta():
    """A heater at full power reads RED even while sitting on the setpoint
    — the ladder checks the output first (as the reference does)."""
    from talos.ui.widgets.hardware_strip import temp_power_color

    assert temp_power_color(25.0, 25.0, 100.0) == "#e53935"


def test_centred_readout_keeps_the_width_and_centres_the_ink():
    """The readout's numeric fields pad from the left to stop the text
    jittering; a centred QLabel aligns those spaces too, which put the
    numbers ~6 px right of the middle of the elastic box (measured).
    centred_readout() re-pads the SAME width symmetrically."""
    from talos.ui.widgets.hardware_strip import centred_readout

    text = f"{12.5:6.1f} · {-3.0:6.1f} µm · {1.25:5.2f}°"
    fixed = centred_readout(text)
    assert len(fixed) == len(text)          # width still constant
    lead = len(fixed) - len(fixed.lstrip())
    trail = len(fixed) - len(fixed.rstrip())
    assert abs(lead - trail) <= 1
    assert fixed.strip() == text.strip()
    # a longer value is passed through, not truncated
    long = f"{123456.7:6.1f} · {-3.0:6.1f} µm · {1.25:5.2f}°"
    assert centred_readout(long).strip() == long.strip()


def test_state_slot_fits_the_longest_word_in_the_bold_font():
    """One fixed slot for IDLE/MOVE/CONT/TRAP/BLOCKED, so the components
    beside it never shift when the word changes — and the slot must fit
    the word it will actually render (it was 46 px for a 48 px word)."""
    from PySide6.QtGui import QFont, QFontMetrics

    from talos.ui.widgets.hardware_strip import STATE_WORDS, state_slot_width

    font = QFont("Segoe UI", 12)
    width = state_slot_width(font)
    bold = QFont(font)
    bold.setBold(True)          # the lit states are bold
    metrics = QFontMetrics(bold)
    assert "BLOCKED" in STATE_WORDS
    for word in STATE_WORDS:
        assert metrics.horizontalAdvance(word) <= width


def test_parse_focus_and_yudian():
    focus = parse_focus({
        "device": "focus",
        "status": asdict(FocusStatus(pos=123, mode="IDLE", blocked_dir="0")),
    })
    assert focus["pos"] == 123
    assert focus["mode"] == "IDLE"
    assert focus["blocked"] == "0"

    yudian = parse_yudian(
        {"device": "yudian", "pv": 25.3, "sv": 25.0, "output_percent": 12.5})
    assert yudian["pv"] == 25.3
    assert yudian["sv"] == 25.0
    assert yudian["out"] == 12.5
