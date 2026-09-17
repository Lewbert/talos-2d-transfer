"""The scan's speed: one number, and nothing else borrowing it.

Scanning and *go to sample* drive the stage in fixed-steps mode, where the
controller generates its own acceleration and deceleration ramp. That is
why there is no slow/fast choice to make and no stability argument for
scaling by the objective — and why these tests check that neither the
manual jog speeds nor the objective multiplier leaks back in.
"""

import pytest

from talos.cv.scan import SCAN_SPEED_MIN_PPS, scan_speed_config
from talos.models import ScanParams

BASE = {"slow_speed_pps": 500, "fast_speed_pps": 2000, "port": "COM3"}


def test_one_speed_fills_both_keys_the_adapter_can_read():
    """Which key the adapter happens to pick must not matter."""
    cfg = scan_speed_config(BASE, 750)
    assert cfg["slow_speed_pps"] == 750
    assert cfg["fast_speed_pps"] == 750


def test_the_manual_speeds_are_never_touched():
    """The scan gets a copy. A run that quietly rewrote the jog speeds
    would change how the stage feels afterwards, with nothing to point at.
    """
    original = dict(BASE)
    scan_speed_config(BASE, 750)
    assert BASE == original


def test_a_speed_of_zero_leaves_the_config_alone():
    """The CLI benches pass their own config and predate this: 0 means
    'use whatever the stage is configured with'."""
    assert scan_speed_config(BASE, 0) == BASE
    assert scan_speed_config(BASE, 0.0) == BASE


def test_the_speed_is_floored():
    cfg = scan_speed_config(BASE, 3)
    assert cfg["slow_speed_pps"] == SCAN_SPEED_MIN_PPS
    assert cfg["fast_speed_pps"] == SCAN_SPEED_MIN_PPS


def test_a_speed_above_the_manual_ones_is_allowed():
    """There is no ceiling borrowed from the jog speeds — a scan is not a
    jog, and the operator's number is the operator's number."""
    cfg = scan_speed_config(BASE, 9000)
    assert cfg["slow_speed_pps"] == 9000


def test_no_multiplier_anywhere_in_the_signature():
    """The objective's stage_speed_multiplier is a manual-jog preference.
    If it is ever reintroduced here, this test is the tripwire."""
    import inspect

    assert list(inspect.signature(scan_speed_config).parameters) == [
        "stage_cfg", "speed_pps"]


def test_scan_params_carry_one_speed():
    assert ScanParams(x0_um=0.0, y0_um=0.0, width_um=1.0,
                      height_um=1.0).speed_pps == 0.0
    assert ScanParams(x0_um=0.0, y0_um=0.0, width_um=1.0, height_um=1.0,
                      speed_pps=600.0).speed_pps == 600.0
    assert not hasattr(ScanParams(x0_um=0.0, y0_um=0.0, width_um=1.0,
                                  height_um=1.0), "slow_speed")


@pytest.mark.parametrize("speed", (250, 500, 1000, 4000))
def test_the_adapter_is_handed_exactly_the_scan_speed(speed):
    """End of the chain: what the stage is actually commanded."""
    from talos.hal.base import StageSpeed
    from talos.hal.proxies.stage_adapter import ManagerStageAdapter

    adapter = ManagerStageAdapter.__new__(ManagerStageAdapter)
    adapter._stage_cfg = scan_speed_config(BASE, speed)
    assert adapter._pps(StageSpeed.SLOW) == speed
    assert adapter._pps(StageSpeed.FAST) == speed
