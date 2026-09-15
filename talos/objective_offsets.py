"""Objective focus-offset math (pure).

On an objective switch the focus moves by the difference of the two
objectives' Z offsets (µm, user-calibrated per objective — different
objectives have slightly different focus lengths), converted to steps via
devices.focus.um_per_step. Sign convention: a POSITIVE offset means the
DISPLAYED focus position increases after the switch — the focus driver
negates raw steps internally (focus.py move_rel), so the caller submits
the raw delta steps and the display moves the same way.
"""

from __future__ import annotations

from talos.cv.af_math import driver_speed_clamp, um_to_steps


def offset_delta_um(old_row: dict | None, new_row: dict | None) -> float:
    """new.z_offset_um − old.z_offset_um; a missing row reads as 0."""
    old = float((old_row or {}).get("z_offset_um") or 0.0)
    new = float((new_row or {}).get("z_offset_um") or 0.0)
    return new - old


def offset_steps(delta_um: float, um_per_step: float) -> int:
    """µm → steps for a DELTA (floor=0: zero means "no motion", and a
    non-positive µm/step means the same). Uses the one conversion helper —
    this used to be a second implementation with different edge behaviour."""
    return um_to_steps(delta_um, um_per_step, floor=0)


def offset_speed_steps_s(row: dict | None, focus_cfg: dict) -> int:
    """The switch-move speed: the focus max speed × the NEW objective's
    manual-focus multiplier, clamped to the DRIVER's window (the same
    helper the autofocus planner uses — this used to carry its own copy of
    the 10/5000 constants, a third home for one rule)."""
    focus_cfg = focus_cfg or {}
    max_speed = float(focus_cfg.get("max_speed", 2000) or 2000)
    mult = float((row or {}).get("focus_manual_multiplier") or 1.0)
    lo, hi = driver_speed_clamp(focus_cfg)
    return int(max(lo, min(hi, round(max_speed * mult))))


def compute_offset_move(old_row: dict | None, new_row: dict | None,
                        um_per_step: float, focus_cfg: dict) \
        -> tuple[int, int] | None:
    """(steps, speed) for the switch compensation, or None when there is
    no motion to make (no delta, a bad um_per_step)."""
    delta = offset_delta_um(old_row, new_row)
    steps = offset_steps(delta, um_per_step)
    if steps == 0:
        return None
    return (steps, offset_speed_steps_s(new_row, focus_cfg))
