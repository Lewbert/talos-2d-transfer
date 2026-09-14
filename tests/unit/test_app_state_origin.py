"""AppState software-origin fields + signals."""

import pytest
from PySide6.QtWidgets import QApplication

from talos.app import AppState
from talos.models import StagePosition


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_origins_default_none():
    state = AppState()
    assert state.stage_origin is None
    assert state.focus_origin is None


def test_set_stage_origin_emits(qapp):
    state = AppState()
    seen = []
    state.sig_stage_origin_changed.connect(seen.append)
    pos = StagePosition(x_pulses=10, y_pulses=20, r_pulses=30,
                        x_um=6.25, y_um=12.5, r_deg=0.0375)
    state.set_stage_origin(pos)
    assert seen == [pos]
    assert state.stage_origin.x_pulses == 10


def test_stage_position_from_telemetry():
    """The proxies publish dataclasses.asdict — a plain DICT. Consumers
    that expected an object (the flake→stage mapping, the stage origin)
    raised AttributeError reading .x_um off it."""
    pos = StagePosition.from_telemetry(
        {"x_pulses": 16, "y_pulses": -8, "r_pulses": 3,
         "x_um": 10.0, "y_um": -5.0, "r_deg": 0.00375})
    assert (pos.x_pulses, pos.y_pulses, pos.r_pulses) == (16, -8, 3)
    assert (pos.x_um, pos.y_um) == (10.0, -5.0)
    # missing / empty payloads degrade to zeros, never to an exception
    assert StagePosition.from_telemetry(None) == StagePosition()
    assert StagePosition.from_telemetry({}) == StagePosition()


def test_set_focus_origin_emits_int(qapp):
    state = AppState()
    seen = []
    state.sig_focus_origin_changed.connect(seen.append)
    state.set_focus_origin(123)
    assert seen == [123]
    assert state.focus_origin == 123
