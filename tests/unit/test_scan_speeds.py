"""Scan speed scaling: the active objective's stage multiplier."""

from talos.cv.scan import scale_scan_speed_config

BASE = {"slow_speed_pps": 500, "fast_speed_pps": 2000, "port": "COM3"}


def test_multiplier_halves_both_speeds():
    cfg = scale_scan_speed_config(BASE, 0.5)
    assert cfg["slow_speed_pps"] == 250
    assert cfg["fast_speed_pps"] == 1000


def test_absent_or_zero_multiplier_keeps_base():
    assert scale_scan_speed_config(BASE, 0.0) == BASE
    assert scale_scan_speed_config(BASE, None) == BASE


def test_multiplier_clamped_to_1():
    cfg = scale_scan_speed_config(BASE, 3.0)
    assert cfg == BASE


def test_floor_10_pps():
    cfg = scale_scan_speed_config({"slow_speed_pps": 100, "fast_speed_pps": 10},
                                  0.05)
    assert cfg["slow_speed_pps"] == 10  # 5 → floored
    assert cfg["fast_speed_pps"] == 10


def test_does_not_mutate_the_input():
    original = dict(BASE)
    scale_scan_speed_config(BASE, 0.25)
    assert BASE == original
