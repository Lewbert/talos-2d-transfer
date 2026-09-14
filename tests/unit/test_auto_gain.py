"""Auto-gain math: mean luma sampling, the next-gain step function and
the standby gating (adjust only after a settle interval, on a full and
stable sample set) + the one-shot "Gain once" action."""

import numpy as np
import pytest

from talos.ui.auto_gain import mean_luma, next_gain, should_adjust


def test_mean_luma():
    frame = np.full((1080, 1920, 3), 100, dtype=np.uint8)
    assert mean_luma(frame) == 100.0
    half = np.concatenate(
        [np.full((540, 1920, 3), 0, np.uint8),
         np.full((540, 1920, 3), 200, np.uint8)])
    # Strided sampling: the 8-px grid straddles the 540/1080 boundary
    # slightly — allow a 2% tolerance (a brightness estimate, not a sum).
    assert abs(mean_luma(half) - 100.0) < 2.0
    assert mean_luma(None) == 0.0
    assert mean_luma(np.zeros((0, 0, 3), np.uint8)) == 0.0


def test_next_gain_deadband():
    assert next_gain(120, 120, 4.0) is None
    assert next_gain(125, 120, 4.0) is None  # inside the 10-luma deadband
    assert next_gain(131, 120, 4.0) is not None


def test_next_gain_brightens_dark_scene():
    # luma 60 vs target 120 → ratio 2 → sqrt = 1.41 → step ~+1.66
    value = next_gain(60, 120, 4.0)
    assert value is not None
    assert 4.0 < value <= 5.7


def test_next_gain_darkens_bright_scene():
    # luma 240 vs target 120 → ratio 0.5 → sqrt = 0.707 → step ~-1.17
    value = next_gain(240, 120, 4.0)
    assert value is not None
    assert 2.8 <= value < 4.0


def test_next_gain_clamped():
    assert next_gain(1, 200, 21.5) == 22.0  # hits the ceiling
    assert next_gain(255, 10, 1.2) == 1.0   # hits the floor
    assert next_gain(200, 120, 1.0) is None  # already at the floor, no change


def test_next_gain_max_step():
    # a huge error must not exceed the ±2.0 step bound
    value = next_gain(250, 10, 10.0, max_step=2.0)
    assert value == 8.0


def test_next_gain_rounds_to_tenths():
    value = next_gain(60, 120, 4.0)
    assert value is not None
    assert value == round(value, 1)


# ---------------------------------------------------------------------------
# One-shot "Gain once" (the controller's transient action)
# ---------------------------------------------------------------------------

class _StubManager:
    def __init__(self):
        self.camera_props = {"gain": 4.0}
        self.submits: list = []

    def submit_camera(self, *args):
        self.submits.append(args)


class _StubSettings:
    def __init__(self):
        self.data = {"devices": {"camera": {"auto_gain_target": 120.0,
                                            "auto_gain": False,
                                            "gain": 4.0}}}

    def device(self, key):
        return self.data["devices"].setdefault(key, {})


def _controller():
    from talos.ui.auto_gain import AutoGainController

    manager = _StubManager()
    ctl = AutoGainController(manager, _StubSettings(), None)
    ctl._timer.stop()  # the standby timer must never fire in tests
    return ctl, manager


def test_once_brightens_dark_frame():
    ctl, manager = _controller()
    ctl.on_frame(np.full((64, 64, 3), 60, dtype=np.uint8))
    assert ctl.once() is True
    assert len(manager.submits) == 1
    method, prop, value = manager.submits[0]
    assert method == "set_property"
    assert prop == "gain"
    assert value == pytest.approx(5.7, abs=0.05)  # one step toward 120


def test_once_deadband_does_nothing():
    ctl, manager = _controller()
    ctl.on_frame(np.full((64, 64, 3), 120, dtype=np.uint8))
    assert ctl.once() is False
    assert manager.submits == []


def test_once_without_frames_does_nothing():
    ctl, manager = _controller()
    assert ctl.once() is False
    assert manager.submits == []


# --- standby gating ---------------------------------------------------------

def test_should_adjust_requires_a_full_buffer():
    assert should_adjust(None, 100.0, 3.0, [], 8.0) is False
    assert should_adjust(None, 100.0, 3.0, [110.0], 8.0) is False
    assert should_adjust(None, 100.0, 3.0, [110.0, 112.0], 8.0) is False
    # full + stable + first adjustment → allowed (no blind step: the
    # buffer was filled by three real samples first)
    assert should_adjust(None, 100.0, 3.0, [110.0, 112.0, 111.0], 8.0)


def test_should_adjust_respects_the_settle_interval():
    samples = [110.0, 111.0, 112.0]
    assert should_adjust(100.0, 102.0, 3.0, samples, 8.0) is False
    assert should_adjust(100.0, 103.0, 3.0, samples, 8.0) is True


def test_should_adjust_waits_out_unstable_scenes():
    stable = [110.0, 111.0, 112.0]
    churn = [100.0, 118.0, 105.0]  # spread 18 > 8
    assert should_adjust(None, 100.0, 3.0, churn, 8.0) is False
    assert should_adjust(None, 100.0, 3.0, stable, 8.0) is True
