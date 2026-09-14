"""Right-panel AFGroup: editable max-window bounds + rough-scan base,
the manual override, and the current-objective readout."""

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from talos.ui.widgets.control_groups import AFGroup

ROWS = [
    {"name": "5x", "mag": 5, "na": 0.15, "af_speed_multiplier": 1.0,
     "focus_manual_multiplier": 1.0, "stage_speed_multiplier": 1.0},
    {"name": "20x", "mag": 20, "na": 0.46, "af_speed_multiplier": 0.25,
     "focus_manual_multiplier": 0.25, "stage_speed_multiplier": 0.25},
]


class FakeSettings:
    def __init__(self):
        self.data = {
            "objectives": [dict(r) for r in ROWS],
            "autofocus": {"window_minus_um": 500.0, "window_plus_um": 500.0,
                          "coarse_speed_base_um_s": 100.0,
                          "manual_bounds_um": 0.0},
            "devices": {"focus": {"um_per_step": 0.2}},
        }
        self.saved = 0

    def get(self, key, default=None):
        return self.data.get(key, default)

    def section(self, key):
        return self.data.setdefault(key, {})

    def device(self, key):
        return self.data["devices"].setdefault(key, {})

    def save(self):
        self.saved += 1


class FakeState(QObject):
    sig_objective_changed = Signal(int)

    def __init__(self):
        super().__init__()
        self.objective = 0


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def group(qapp):
    settings = FakeSettings()
    state = FakeState()
    return AFGroup(settings, state), settings, state


def test_initial_values(group):
    group_box, _settings, _state = group
    assert group_box._minus.value() == 500.0
    assert group_box._plus.value() == 500.0
    assert group_box._base.value() == 100.0
    assert group_box._bounds.value() == 0.0


def test_readout_for_current_objective(group):
    group_box, _settings, _state = group
    # 5x: mult 1.0 → ±500 µm, coarse 100 ÷ 0.2 = 500 st/s
    text = group_box._readout.text()
    assert "5x" in text
    assert "−500 / +500 µm" in text
    assert "coarse 500 st/s" in text
    assert "fine 500 st/s" in text


def test_edit_refreshes_and_persists(group):
    group_box, settings, _state = group
    group_box._minus.setValue(200.0)
    group_box._minus.editingFinished.emit()
    assert settings.section("autofocus")["window_minus_um"] == 200.0
    assert "−200 / +500 µm" in group_box._readout.text()
    group_box._base.setValue(150.0)
    group_box._base.editingFinished.emit()
    assert settings.section("autofocus")["coarse_speed_base_um_s"] == 150.0
    assert "coarse 750 st/s" in group_box._readout.text()


def test_objective_switch_refreshes_readout(group):
    group_box, _settings, state = group
    state.objective = 1
    state.sig_objective_changed.emit(1)
    text = group_box._readout.text()
    # 20x: mult 0.25 → ±125 µm, coarse 125 st/s, fine = max(50, 125)
    assert "20x" in text
    assert "−125 / +125 µm" in text
    assert "coarse 125 st/s" in text
    assert "fine 125 st/s" in text


def test_bounds_steps_override_behavior(group):
    group_box, _settings, _state = group
    assert group_box.bounds_steps(1000, 0.2) is None  # 0 = unbounded
    group_box._bounds.setValue(100.0)
    assert group_box.bounds_steps(1000, 0.2) == (500, 1500)
