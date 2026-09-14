"""Right-panel AFGroup: the AF settings block (measurement region,
max-window bounds, rough-scan base, manual override and the
current-objective readout) wrapped in a group."""

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
    assert group_box.widget._minus.value() == 500.0
    assert group_box.widget._plus.value() == 500.0
    assert group_box.widget._base.value() == 100.0
    assert group_box.widget._bounds.value() == 0.0


def test_readout_for_current_objective(group):
    group_box, _settings, _state = group
    # 5x: mult 1.0 → ±500 µm, coarse 100 ÷ 0.2 = 500 st/s
    text = group_box.widget._readout.text()
    assert "5x" in text
    assert "−500 / +500 µm" in text
    assert "coarse 500 st/s" in text
    assert "fine 500 st/s" in text


def test_edit_refreshes_and_persists(group):
    group_box, settings, _state = group
    group_box.widget._minus.setValue(200.0)
    group_box.widget._minus.editingFinished.emit()
    assert settings.section("autofocus")["window_minus_um"] == 200.0
    assert "−200 / +500 µm" in group_box.widget._readout.text()
    group_box.widget._base.setValue(150.0)
    group_box.widget._base.editingFinished.emit()
    assert settings.section("autofocus")["coarse_speed_base_um_s"] == 150.0
    assert "coarse 750 st/s" in group_box.widget._readout.text()


def test_objective_switch_refreshes_readout(group):
    group_box, _settings, state = group
    state.objective = 1
    state.sig_objective_changed.emit(1)
    text = group_box.widget._readout.text()
    # 20x: mult 0.25 → ±125 µm, coarse 125 st/s, fine = max(50, 125)
    assert "20x" in text
    assert "−125 / +125 µm" in text
    assert "coarse 125 st/s" in text
    assert "fine 125 st/s" in text


def test_bounds_steps_override_behavior(group):
    group_box, _settings, _state = group
    assert group_box.bounds_steps(1000, 0.2) is None  # 0 = unbounded
    group_box.widget._bounds.setValue(100.0)
    assert group_box.bounds_steps(1000, 0.2) == (500, 1500)


# ---------------------------------------------------------------------------
# AF measurement region (ROI)
# ---------------------------------------------------------------------------

def test_region_defaults_to_the_whole_frame(group):
    group_box, _settings, _state = group
    widget = group_box.widget
    assert group_box._roi.is_full_frame()
    assert widget._area.currentIndex() == 0          # "Full frame"
    assert not widget._roi_spins["x"].isEnabled()    # numbers are inert


def test_choosing_roi_starts_from_the_centre_default(group):
    from talos.ui.af_region import DEFAULT_ROI_NORM

    group_box, settings, _state = group
    widget = group_box.widget
    widget._area.setCurrentIndex(1)                  # "ROI"
    assert group_box._roi.roi() == pytest.approx(DEFAULT_ROI_NORM)
    assert settings.section("autofocus")["default_roi_norm"] is not None
    assert widget._roi_spins["w"].isEnabled()
    assert widget._roi_spins["w"].value() == pytest.approx(66.7, abs=0.1)


def test_reset_roi_restores_the_centre_default(group):
    from talos.ui.af_region import DEFAULT_ROI_NORM

    group_box, _settings, _state = group
    widget = group_box.widget
    widget._roi_spins["x"].setValue(5.0)
    widget._roi_spins["x"].editingFinished.emit()
    assert group_box._roi.roi()[0] == pytest.approx(0.05)
    widget._reset_btn.click()
    assert group_box._roi.roi() == pytest.approx(DEFAULT_ROI_NORM)


def test_numeric_edit_persists_and_switches_to_roi(group):
    group_box, settings, _state = group
    widget = group_box.widget
    widget._roi_spins["y"].setValue(25.0)
    widget._roi_spins["y"].editingFinished.emit()
    roi = settings.section("autofocus")["default_roi_norm"]
    assert roi is not None and roi[1] == pytest.approx(0.25)
    assert widget._area.currentIndex() == 1          # ROI mode follows
    assert group_box._roi.roi()[1] == pytest.approx(0.25)


def test_full_frame_clears_the_stored_region(group):
    group_box, settings, _state = group
    widget = group_box.widget
    widget._area.setCurrentIndex(1)
    assert settings.section("autofocus")["default_roi_norm"] is not None
    widget._area.setCurrentIndex(0)                  # back to Full frame
    assert settings.section("autofocus")["default_roi_norm"] is None
    assert group_box._roi.is_full_frame()


def test_select_roi_requests_the_rubber_band(group):
    group_box, _settings, _state = group
    seen = []
    group_box.widget.sig_roi_arm_requested.connect(lambda: seen.append(True))
    group_box.widget._select_btn.click()
    assert seen == [True]


def test_spin_edits_are_clamped_into_the_frame(group):
    group_box, _settings, _state = group
    widget = group_box.widget
    widget._roi_spins["x"].setValue(99.0)            # leaves no room
    widget._roi_spins["x"].editingFinished.emit()
    x, _y, w, _h = group_box._roi.roi()
    assert x + w <= 1.0


def test_settings_change_signal_fires_for_knobs_and_roi(group):
    """The signal is what keeps the second instance (the AF detail window)
    in step, so it must fire for BOTH kinds of edit."""
    group_box, _settings, _state = group
    seen = []
    group_box.widget.sig_settings_changed.connect(lambda: seen.append(1))
    group_box.widget._plus.setValue(300.0)
    group_box.widget._plus.editingFinished.emit()
    assert seen, "a knob edit must announce itself"
    seen.clear()
    group_box.widget._area.setCurrentIndex(1)
    assert seen, "an ROI change must announce itself"


def test_refresh_from_settings_picks_up_external_edits(group):
    """The right panel and the AF detail window hold separate instances of
    this block: an edit in one must be visible in the other."""
    group_box, settings, _state = group
    settings.section("autofocus")["window_minus_um"] = 42.0
    assert group_box.widget._minus.value() == 500.0      # not yet refreshed
    group_box.widget.refresh_from_settings()
    assert group_box.widget._minus.value() == 42.0
