"""Backlash calibration bounds (fast: no camera, no real motion).

The calibrator drives the axis toward the ends of its travel twice — to
``center ± (sweep + settle)`` — and used to have no bound at all, while the
autofocus sweeps it mirrors always clamp to the focus soft limits and warn
when the firmware is not enforcing them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from talos.cv.backlash_cal import BacklashCalConfig, BacklashCalibrator


class FakeFocus:
    """The slice of the focus driver the calibrator touches."""

    def __init__(self, soft=(-1000, 1000), pos=0, slim=True):
        self.soft = soft
        self.pos = int(pos)
        self.slim = slim
        self.targets: list[int] = []
        self.stopped = 0
        self.logs: list[str] = []

    # driver surface
    def get_soft_limits(self):
        return self.soft

    def get_slim_state(self):
        return self.slim

    def move_abs(self, pos, speed=None):
        self.targets.append(int(pos))
        self.pos = int(pos)

    def get_status(self):
        return SimpleNamespace(pos=self.pos, is_idle=True)

    def wait_idle(self, timeout_s=120.0, poll_s=0.05):
        return None

    def set_speed(self, speed):
        pass

    def stop(self):
        self.stopped += 1

    def drain_events(self):
        return []


def _calibrator(focus) -> BacklashCalibrator:
    cal = BacklashCalibrator(focus, frame_reader=None)
    cal.sig_log.connect(focus.logs.append)
    return cal


def test_sweep_refuses_to_run_outside_the_soft_limits():
    """Regression (2026-09-16): the sweep was unbounded — a ±250-step travel
    with no soft-limit read, no margin and no SLIM check."""
    focus = FakeFocus(soft=(-50, 50))
    result = _calibrator(focus).run(center=0, cfg=BacklashCalConfig(
        sweep_steps=150, settle_steps=100))
    assert result.success is False
    assert "soft limits" in result.message
    assert focus.targets == [], "the axis must not move at all"


def test_every_commanded_target_stays_inside_the_soft_limits():
    focus = FakeFocus(soft=(-200, 200))
    result = _calibrator(focus).run(center=0, cfg=BacklashCalConfig(
        sweep_steps=150, settle_steps=100, timeout_s=1.0, poll_s=0.01,
        speed=2000))
    assert result.success is False          # no frames at all in this rig
    assert focus.targets, "the sweep should have started"
    assert min(focus.targets) >= focus.soft[0]
    assert max(focus.targets) <= focus.soft[1]


def test_a_wide_window_is_left_alone():
    """With room to spare the configured sweep is used unchanged (the
    clamp must not shrink a sweep that already fits)."""
    focus = FakeFocus(soft=(-2_000_000, 2_000_000))
    _calibrator(focus).run(center=0, cfg=BacklashCalConfig(
        sweep_steps=150, settle_steps=100, timeout_s=1.0, poll_s=0.01,
        speed=2000))
    # bottom − settle = −250, top + settle = +250: the full configured travel
    assert min(focus.targets) == -250
    assert max(focus.targets) == 250


def test_slim_off_is_reported():
    """The autofocus path warns when the firmware is not enforcing SLIM;
    the calibration sweep now says the same thing."""
    focus = FakeFocus(soft=(-200, 200), slim=False)
    _calibrator(focus).run(center=0, cfg=BacklashCalConfig(
        sweep_steps=150, settle_steps=100, timeout_s=1.0, poll_s=0.01,
        speed=2000))
    assert any("SLIM" in line for line in focus.logs)


def test_an_unreadable_limit_falls_back_to_the_config_span():
    class NoLimits(FakeFocus):
        def get_soft_limits(self):
            from talos.hal.base import DeviceError

            raise DeviceError("no limits")

    focus = NoLimits(soft=None)
    result = _calibrator(focus).run(center=0, cfg=BacklashCalConfig(
        sweep_steps=150, settle_steps=100, timeout_s=1.0, poll_s=0.01,
        speed=2000))
    assert result.success is False           # no frames
    assert -250 in focus.targets, "an unreadable limit must not stop the run"
