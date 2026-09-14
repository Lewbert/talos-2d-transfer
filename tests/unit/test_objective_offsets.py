"""Objective focus-offset math (pure)."""

from talos.objective_offsets import (
    compute_offset_move,
    offset_delta_um,
    offset_speed_steps_s,
    offset_steps,
)

FOCUS_CFG = {"max_speed": 2000}


def test_offset_delta_missing_rows_is_zero():
    assert offset_delta_um(None, None) == 0.0
    assert offset_delta_um({"z_offset_um": 5.0}, None) == -5.0
    assert offset_delta_um(None, {"z_offset_um": 5.0}) == 5.0
    assert offset_delta_um({"z_offset_um": 2.0}, {"z_offset_um": 5.0}) == 3.0
    # rows without the key read 0
    assert offset_delta_um({}, {"z_offset_um": 5.0}) == 5.0


def test_offset_steps_rounding_and_sign():
    assert offset_steps(20.0, 0.2) == 100
    assert offset_steps(-20.0, 0.2) == -100
    assert offset_steps(0.0, 0.2) == 0
    assert offset_steps(0.05, 0.2) == 0
    assert offset_steps(20.0, 0.0) == 0   # bad um_per_step → no motion
    assert offset_steps(20.0, -0.2) == 0


def test_offset_speed_clamped():
    assert offset_speed_steps_s({"focus_manual_multiplier": 0.25},
                                FOCUS_CFG) == 500
    assert offset_speed_steps_s({"focus_manual_multiplier": 5.0},
                                FOCUS_CFG) == 5000
    # 0 = unset → 1.0 (the resolver's convention); a tiny mult hits the floor
    assert offset_speed_steps_s({"focus_manual_multiplier": 0.0},
                                FOCUS_CFG) == 2000
    assert offset_speed_steps_s({"focus_manual_multiplier": 0.001},
                                FOCUS_CFG) == 10
    assert offset_speed_steps_s(None, FOCUS_CFG) == 2000


def test_compute_offset_move_none_when_no_delta():
    row = {"z_offset_um": 0.0, "focus_manual_multiplier": 1.0}
    assert compute_offset_move(row, row, 0.2, FOCUS_CFG) is None
    assert compute_offset_move(None, None, 0.2, FOCUS_CFG) is None


def test_compute_offset_move_returns_steps_and_speed():
    old = {"z_offset_um": 0.0}
    new = {"z_offset_um": 20.0, "focus_manual_multiplier": 0.5}
    assert compute_offset_move(old, new, 0.2, FOCUS_CFG) == (100, 1000)
    assert compute_offset_move(new, old, 0.2, FOCUS_CFG) == (-100, 2000)
