"""af_math: interpolation, peak picking, parabolic fit, landing plan,
µm→steps conversion, and the v3 derivative-hybrid math (audit-verified
Gaussian fixtures)."""

import math

import numpy as np
import pytest

from talos.cv.af_math import (
    build_config,
    coarse_curv_stop_ladder,
    curvature_from_three,
    direction_guard_wrong_way,
    fit_slope_with_se,
    guard_reversal_test,
    hill_speed_by_pos,
    hill_v_min,
    interpolate_position,
    landing_plan,
    parabolic_fit,
    peak_is_complete,
    pick_peak,
    pooled_noise_from_probe,
    predictive_stop_steps,
    probe_classify,
    recommended_multipliers,
    probe_direction,
    probe_near_focus,
    sg_curvature_at,
    speed_multiplier,
    um_to_steps,
    window_curvature,
    window_parabola,
)


# ---------------------------------------------------------------------------
# interpolate_position
# ---------------------------------------------------------------------------

def test_interpolation_midpoint_is_linear():
    hist = [(0.0, 0), (0.05, 50), (0.10, 100)]
    assert interpolate_position(0.025, hist) == pytest.approx(25.0)
    assert interpolate_position(0.075, hist) == pytest.approx(75.0)


def test_interpolation_out_of_span_is_none():
    hist = [(0.0, 0), (0.05, 100)]
    assert interpolate_position(0.1, hist) is None
    assert interpolate_position(-0.01, hist) is None


def test_interpolation_exact_sample_time():
    hist = [(0.0, 0), (0.05, 100)]
    assert interpolate_position(0.0, hist) == 0.0
    assert interpolate_position(0.05, hist) == 100.0  # last sample: exact value
    assert interpolate_position(0.06, hist) is None   # beyond the span


def test_interpolation_gap_rejection():
    # Bracket spanning 500 ms — a camera stall made the position history
    # untrustworthy there; the 200 ms bracket stays usable.
    hist = [(0.0, 0), (0.2, 20), (0.7, 70)]
    assert interpolate_position(0.45, hist, gap_max_ms=300.0) is None
    assert interpolate_position(0.1, hist, gap_max_ms=300.0) == pytest.approx(10.0)


def test_interpolation_empty_history():
    assert interpolate_position(0.5, []) is None


# ---------------------------------------------------------------------------
# pick_peak
# ---------------------------------------------------------------------------

def _parabola_curve(center=0.0, peak=100.0, width=10.0, n=21):
    xs = [center - width + 2 * width * i / (n - 1) for i in range(n)]
    ys = [peak - ((x - center) / width) ** 2 * peak for x in xs]
    return list(zip(xs, ys))


def test_pick_peak_single_peak():
    curve = _parabola_curve(center=50.0)
    peak = pick_peak(curve, center=40.0)
    assert peak is not None
    assert peak.pos == pytest.approx(50.0, abs=1.5)
    assert not peak.at_edge


def test_pick_peak_nearest_center_beats_global_max():
    # Two INTERIOR peaks: a tall one at 0 and a weaker one at 20. Arming
    # near the weak one (the wafer surface scenario: the tall peak is the
    # mount) must select it.
    xs = [i * 2.0 - 40.0 for i in range(41)]  # -40 .. +40
    ys = [120.0 - (x / 15.0) ** 2 * 120.0 for x in xs]           # tall peak at 0
    ys = [max(y, 60.0 - ((x - 20.0) / 8.0) ** 2 * 60.0) for x, y in zip(xs, ys)]  # weak at 20
    curve = list(zip(xs, ys))
    peak = pick_peak(curve, center=22.0, prominence=0.1)
    assert peak is not None
    assert peak.pos == pytest.approx(20.0, abs=3.0)
    # arming at the tall peak picks the tall one
    peak2 = pick_peak(curve, center=0.0, prominence=0.1)
    assert peak2.pos == pytest.approx(0.0, abs=3.0)


def test_pick_peak_edge_flag():
    # descending half-parabola: the sharpest point is the window's first
    # sample (the true peak sits beyond the window edge) — must be
    # reported AS an edge peak, not as "no peak".
    xs = [i * 0.5 for i in range(21)]  # 0 .. 10
    ys = [100.0 - (x / 10.0) ** 2 * 90.0 for x in xs]
    curve = list(zip(xs, ys))
    peak = pick_peak(curve, center=0.0, edge_margin_steps=2.0)
    assert peak is not None
    assert peak.at_edge
    assert peak.pos == pytest.approx(0.0)


def test_pick_peak_flat_curve_is_none():
    curve = [(float(i), 1.0) for i in range(20)]
    assert pick_peak(curve, center=10.0) is None


def test_pick_peak_noise_wiggle_below_prominence_is_ignored():
    xs = [float(i) for i in range(30)]
    ys = [50.0 - (x - 15.0) ** 2 for x in xs]
    ys[5] = 40.0  # small bump
    ys[6] = 41.0
    ys[7] = 40.0
    curve = list(zip(xs, ys))
    peak = pick_peak(curve, center=15.0, prominence=0.3)
    assert peak is not None
    assert peak.pos == pytest.approx(15.0, abs=1.0)


def test_pick_peak_too_short_curve():
    assert pick_peak([(0.0, 1.0), (1.0, 2.0)], center=0.0) is None


# ---------------------------------------------------------------------------
# parabolic_fit
# ---------------------------------------------------------------------------

def test_parabolic_fit_exact_quadratic():
    # y = -(x-7)^2 + 50 → peak at x = 7
    curve = [(x, -(x - 7.0) ** 2 + 50.0) for x in (5.0, 6.0, 7.0, 8.0, 9.0)]
    assert parabolic_fit(curve, peak_pos=7.0) == pytest.approx(7.0)


def test_parabolic_fit_upward_opening_is_none():
    curve = [(x, (x - 7.0) ** 2) for x in (5.0, 6.0, 7.0, 8.0, 9.0)]
    assert parabolic_fit(curve, peak_pos=7.0) is None


def test_parabolic_fit_degenerate_x():
    curve = [(5.0, 1.0), (5.0, 2.0), (6.0, 1.5)]
    assert parabolic_fit(curve, peak_pos=5.0) is not None  # falls back to best y x


# ---------------------------------------------------------------------------
# landing_plan
# ---------------------------------------------------------------------------

def test_landing_plan_default_approach_from_below():
    overshoot, direction, degraded = landing_plan(
        target=100.0, backlash_steps=5, margin_steps=10, bounds=(0.0, 200.0))
    assert overshoot == 85          # 100 - (5 + 10)
    assert direction == 1           # final approach upward
    assert degraded is False


def test_landing_plan_flips_when_overshoot_leaves_bounds():
    overshoot, direction, _ = landing_plan(
        target=5.0, backlash_steps=5, margin_steps=10, bounds=(0.0, 200.0))
    assert overshoot == 20          # 5 + 15
    assert direction == -1


def test_landing_plan_clamps_to_bounds():
    overshoot, direction, _ = landing_plan(
        target=50.0, backlash_steps=100, margin_steps=100, bounds=(0.0, 200.0))
    assert overshoot == 200         # clamped to the upper bound
    assert direction == -1


def test_landing_plan_reports_a_degraded_take_up():
    """A bounds clamp can collapse the overshoot onto the target: the two
    landing moves then travel less than one backlash take-up and the load
    can land off by up to backlash_steps. That must be REPORTED, not
    silently treated as a compensated landing."""
    # target at the upper edge and a take-up that fits on NEITHER side:
    # both candidate overshoots leave the bounds, so the clamp leaves just
    # 5 steps of travel for a 200-step take-up
    overshoot, _direction, degraded = landing_plan(
        target=195.0, backlash_steps=200, margin_steps=50, bounds=(0.0, 200.0))
    assert overshoot == 200
    assert degraded is True
    # a take-up that fits is not degraded
    _, _, ok = landing_plan(target=100.0, backlash_steps=5, margin_steps=10,
                            bounds=(0.0, 200.0))
    assert ok is False


# ---------------------------------------------------------------------------
# build_config / um_to_steps
# ---------------------------------------------------------------------------

def test_um_to_steps_rounding():
    assert um_to_steps(0.2, 0.2) == 1
    assert um_to_steps(1.0, 0.2) == 5
    assert um_to_steps(0.5, 0.2) == 2   # round(2.5) → 2 (banker's)
    assert um_to_steps(0.05, 0.2) == 1  # never below one step


def test_build_config_converts_units():
    row = {"name": "LMPlanFL 5x", "mag": 5, "na": 0.15, "dof_um": 28.0,
           "window_um": 150.0, "coarse_step_um": 5.0, "fine_step_um": 1.0,
           "speed_multiplier": 1.0, "backlash_um": 0.0}
    cfg, warnings = build_config(row, um_per_step=0.2,
                                 af_cfg={"af_exposure_us": 20000,
                                         "landing_speed": 50,
                                         "overshoot_margin_um": 3.0})
    assert cfg["coarse_step"] == 25
    assert cfg["fine_step"] == 5
    assert cfg["span_steps"] == 5000        # 1000 µm × 1.0 ÷ 0.2 (v6 window)
    assert cfg["max_speed"] == 500          # 1.0 × 100 µm/s ÷ 0.2 (aggressive base)
    assert cfg["coarse_speed"] == 500
    assert cfg["landing_speed"] == 50
    assert cfg["fine_speed"] == 500         # max(landing, coarse) — classic lever
    assert cfg["hill_v_cap"] == 500         # far zone = coarse speed
    assert cfg["hill_v_min"] == 83          # 5 steps / (3 × 0.02 s)
    assert cfg["backlash_steps"] == 0
    assert cfg["overshoot_margin_steps"] == 15
    # the fine window covers the coarse sampling spacing: 500 sps ÷
    # 15 fps = 33 steps/frame → 3× = 100 (hardware-verified: a small
    # window + sparse coarse sampling misses the peak at the aggressive
    # speed)
    assert cfg["fine_window_steps"] == 100
    # the aggressive base exceeds the sampling-density bound by design
    assert any("sampling" in w for w in warnings)


def test_build_config_sampling_density_warning():
    # base 150 µm/s exceeds coarse_step × fps = 75 — the warning must fire
    row = {"dof_um": 28.0, "coarse_step_um": 5.0, "fine_step_um": 1.0,
           "window_um": 150.0, "speed_multiplier": 1.0, "backlash_um": 0.0}
    cfg, warnings = build_config(row, um_per_step=0.2,
                                 af_cfg={"coarse_speed_base_um_s": 150.0,
                                         "af_exposure_us": 20000})
    assert cfg["max_speed"] == 750
    assert any("sampling" in w for w in warnings)


def test_build_config_blur_warning():
    # multiplier 3.0 × base 150 → 450 µm/s → blur 9 µm > DOF/4 = 7 µm
    row = {"dof_um": 28.0, "coarse_step_um": 5.0, "fine_step_um": 1.0,
           "window_um": 150.0, "speed_multiplier": 3.0, "backlash_um": 0.0}
    _, warnings = build_config(row, um_per_step=0.2,
                               af_cfg={"coarse_speed_base_um_s": 150.0,
                                       "af_exposure_us": 20000})
    assert any("DOF/4" in w and "blur" in w for w in warnings)


def test_build_config_speed_floor_and_step_warnings():
    row = {"dof_um": 2.0, "coarse_step_um": 5.0, "fine_step_um": 1.0,
           "window_um": 60.0, "speed_multiplier": 0.01, "backlash_um": 0.0}
    cfg, warnings = build_config(row, um_per_step=0.2, af_cfg={})
    assert cfg["max_speed"] == 10            # driver floor
    assert cfg["hill_v_cap"] == 10           # floor too
    assert any("clamped" in w for w in warnings)
    assert any("exceeds DOF" in w for w in warnings)
    assert any("DOF/4" in w for w in warnings)


def test_build_config_fine_window_never_collapses():
    # 100x-style row: coarse == fine == 1 step
    row = {"dof_um": 0.8, "coarse_step_um": 0.2, "fine_step_um": 0.2,
           "window_um": 12.0, "speed_multiplier": 0.0311, "backlash_um": 0.0}
    cfg, _ = build_config(row, um_per_step=0.2,
                          af_cfg={"af_exposure_us": 20000, "landing_speed": 50})
    assert cfg["coarse_step"] == 1
    assert cfg["fine_step"] == 1
    assert cfg["fine_window_steps"] == 3    # max(1, 3*1) — never collapses
    assert cfg["max_speed"] == 16           # 3.1 µm/s = 16 steps/s


# ---------------------------------------------------------------------------
# Adaptive-strategy speed math
# ---------------------------------------------------------------------------

def test_speed_multiplier_table():
    assert speed_multiplier(0.15) == pytest.approx(1.0)
    assert speed_multiplier(0.30) == pytest.approx(0.25)
    assert speed_multiplier(0.46) == pytest.approx(0.1063, abs=1e-3)
    assert speed_multiplier(0.55) == pytest.approx(0.0744, abs=1e-3)
    assert speed_multiplier(0.85) == pytest.approx(0.0311, abs=1e-3)


def test_speed_multiplier_defensive_na():
    assert speed_multiplier(0.0) == 1.0
    assert speed_multiplier(-1.0) == 1.0


def test_build_config_derives_multiplier_from_na():
    row = {"na": 0.30, "dof_um": 7.0, "coarse_step_um": 2.0, "fine_step_um": 0.4,
           "window_um": 60.0, "backlash_um": 0.0}
    cfg, _ = build_config(row, um_per_step=0.2, af_cfg={})
    assert cfg["max_speed"] == 125           # 0.25 × 100 µm/s ÷ 0.2
    assert cfg["hill_v_cap"] == 125          # far zone = coarse speed


def test_build_config_migrates_legacy_speed_column():
    # build_config is the defensive layer (config.py normalizes first):
    # a legacy speed column is preserved as an equivalent multiplier.
    row = {"dof_um": 28.0, "coarse_step_um": 5.0, "fine_step_um": 1.0,
           "window_um": 150.0, "coarse_speed_um_s": 15.0, "backlash_um": 0.0}
    cfg, warnings = build_config(row, um_per_step=0.2,
                                 af_cfg={"af_exposure_us": 20000})
    assert cfg["max_speed"] == 75            # 15 µm/s preserved exactly
    assert any("legacy" in w for w in warnings)


def test_build_config_speed_table_referenced_to_the_lowest_power_objective():
    """The scalability model: every objective's auto speed = the global
    base × (na_min/na)² with na_min = the LOWEST-POWER objective's NA
    from the table (its auto multiplier = 1.0 → exactly the base —
    the table's max). Explicit per-row multipliers win."""
    table = [
        {"mag": 2.5, "na": 0.08, "dof_um": 60.0, "window_um": 200.0,
         "coarse_step_um": 10.0, "fine_step_um": 2.0, "backlash_um": 0.0},
        {"mag": 5, "na": 0.15, "dof_um": 28.0, "window_um": 150.0,
         "coarse_step_um": 5.0, "fine_step_um": 1.0, "backlash_um": 0.0},
        {"mag": 50, "na": 0.55, "dof_um": 2.0, "window_um": 20.0,
         "coarse_step_um": 0.4, "fine_step_um": 0.2, "backlash_um": 0.0},
    ]
    na_min = 0.08
    base = 100.0
    speeds = {}
    for row in table:
        cfg, _ = build_config(row, 0.2, {"camera_fps_estimate": 15}, na_min=na_min)
        speeds[row["mag"]] = cfg["max_speed"]
        expected = int(round(base * (na_min / row["na"]) ** 2 / 0.2))
        assert cfg["max_speed"] == expected, \
            f"{row['mag']}x speed off the NA model"
    # the lowest-power objective gets exactly the base speed (the max)
    assert speeds[2.5] == pytest.approx(base / 0.2)
    assert max(speeds.values()) == speeds[2.5]
    assert speeds[50] < speeds[5] < speeds[2.5]
    # an explicit per-row multiplier wins over the NA model
    row5 = dict(table[1], af_speed_multiplier=0.5)
    cfg, _ = build_config(row5, 0.2, {"camera_fps_estimate": 15}, na_min=na_min)
    assert cfg["max_speed"] == pytest.approx(0.5 * base / 0.2)


def test_build_config_multiplier_scales_span_and_stage2_window():
    """v6: BOTH the total search span (1000 µm × the multiplier) and the
    stage-2 window scale with the AF speed multiplier — a higher power
    searches a smaller window, proportionally to the same lever."""
    row5 = {"na": 0.15, "dof_um": 28.0, "coarse_step_um": 5.0,
            "fine_step_um": 1.0, "af_speed_multiplier": 1.0,
            "backlash_um": 0.0}
    row50 = {"na": 0.55, "dof_um": 2.0, "coarse_step_um": 0.4,
             "fine_step_um": 0.2, "af_speed_multiplier": 0.0744,
             "backlash_um": 0.0}
    cfg5, _ = build_config(row5, 0.2, {"camera_fps_estimate": 15},
                           na_min=0.08)
    cfg50, _ = build_config(row50, 0.2, {"camera_fps_estimate": 15},
                            na_min=0.08)
    # 5x: 1000 × 1.0 / 0.2 = 5000 steps; 50x: 1000 × 0.0744 / 0.2 = 372
    assert cfg5["span_steps"] == 5000
    assert cfg50["span_steps"] == 372
    assert cfg50["fine_window_steps"] <= cfg5["fine_window_steps"]
    assert cfg50["max_speed"] < cfg5["max_speed"]
    # a slower 5x (explicit multiplier) shrinks both span and window
    row5_slow = dict(row5, af_speed_multiplier=0.25)
    cfg5_slow, _ = build_config(row5_slow, 0.2, {"camera_fps_estimate": 15},
                                na_min=0.08)
    assert cfg5_slow["span_steps"] == 1250
    assert cfg5_slow["fine_window_steps"] <= cfg5["fine_window_steps"]


def test_build_config_asymmetric_window_bounds():
    # v7: the search window = the editable max bounds × the multiplier,
    # carried separately so the preflight clamp stays asymmetric.
    row = {"na": 0.15, "dof_um": 28.0, "coarse_step_um": 5.0,
           "fine_step_um": 1.0, "af_speed_multiplier": 1.0,
           "backlash_um": 0.0}
    cfg, _ = build_config(row, 0.2, {"window_plus_um": 300.0,
                                     "window_minus_um": 200.0})
    assert cfg["span_steps"] == 2500           # (300 + 200) ÷ 0.2
    assert cfg["window_plus_steps"] == 1500
    assert cfg["window_minus_steps"] == 1000


def test_build_config_window_defaults_are_symmetric():
    # no window keys → ±500 µm × mult, identical to the old 1000 µm span
    row = {"na": 0.15, "dof_um": 28.0, "coarse_step_um": 5.0,
           "fine_step_um": 1.0, "af_speed_multiplier": 1.0,
           "backlash_um": 0.0}
    cfg, _ = build_config(row, 0.2, {})
    assert cfg["span_steps"] == 5000
    assert cfg["window_plus_steps"] == 2500
    assert cfg["window_minus_steps"] == 2500


def test_build_config_window_bounds_scale_with_multiplier():
    row = {"na": 0.30, "dof_um": 7.0, "coarse_step_um": 2.0,
           "fine_step_um": 0.4, "af_speed_multiplier": 0.25,
           "backlash_um": 0.0}
    cfg, _ = build_config(row, 0.2, {})
    assert cfg["span_steps"] == 1250           # 1000 × 0.25 ÷ 0.2
    assert cfg["window_plus_steps"] == 625
    assert cfg["window_minus_steps"] == 625


def test_recommended_multipliers_table():
    # af/focus = (min NA / NA)², stage = min mag / mag — the shipped
    # default table's model
    assert recommended_multipliers(0.15, 5.0) == (1.0, 1.0, 1.0)
    assert recommended_multipliers(0.30, 10.0) == (0.25, 0.25, 0.5)
    assert recommended_multipliers(0.46, 20.0) == \
        (pytest.approx((0.15 / 0.46) ** 2, rel=1e-6),
         pytest.approx((0.15 / 0.46) ** 2, rel=1e-6),
         pytest.approx(0.25))
    assert recommended_multipliers(0.85, 100.0) == \
        (pytest.approx((0.15 / 0.85) ** 2, rel=1e-6),
         pytest.approx((0.15 / 0.85) ** 2, rel=1e-6),
         pytest.approx(0.05))


def test_recommended_multipliers_defensive():
    assert recommended_multipliers(0.0, 5.0) == (1.0, 1.0, 1.0)
    assert recommended_multipliers(0.15, 0.0) == (1.0, 1.0, 1.0)
    assert recommended_multipliers(None, 5.0) == (1.0, 1.0, 1.0)
    assert recommended_multipliers("x", 5.0) == (1.0, 1.0, 1.0)
    # clamps: NA above min → ≤ 1.0; mag below min → clamped to 1.0
    assert recommended_multipliers(0.05, 2.0) == (1.0, 1.0, 1.0)
    # extreme low NA / high mag → floor 0.01
    af, focus, stage = recommended_multipliers(15.0, 1000.0)
    assert af == focus == 0.01
    assert stage == 0.01


def test_build_config_near_window_steps():
    # the near-entry stage-2 window: max(fw, 1.5σ). With the DOF/3 model
    # (σ ≈ 47 steps) the default fw (100) already dominates; with the
    # measured σ (110) the window widens to cover the probe's ±0.7σ
    # near-tolerance zone.
    row = {"dof_um": 28.0, "coarse_step_um": 5.0, "fine_step_um": 1.0,
           "window_um": 150.0, "speed_multiplier": 1.0, "backlash_um": 0.0}
    cfg_auto, _ = build_config(row, 0.2, {"camera_fps_estimate": 15,
                                          "af_exposure_us": 20000})
    assert cfg_auto["near_window_steps"] == cfg_auto["fine_window_steps"]
    cfg_measured, _ = build_config(row, 0.2, {"camera_fps_estimate": 15,
                                              "sigma_steps": 110.0})
    assert cfg_measured["near_window_steps"] == 165
    # explicit override wins
    cfg_override, _ = build_config(row, 0.2, {"sigma_steps": 110.0,
                                              "near_window_steps": 200})
    assert cfg_override["near_window_steps"] == 200


def test_build_config_sigma_steps_override_rescales_thresholds():
    # the bench's Step-0 measured σ (107.8 steps at 5×) replaces the
    # DOF/3 model (46.7) — the derivative thresholds rescale with it
    row = {"dof_um": 28.0, "coarse_step_um": 5.0, "fine_step_um": 1.0,
           "window_um": 150.0, "speed_multiplier": 1.0, "backlash_um": 0.0}
    cfg_auto, _ = build_config(row, 0.2, {"camera_fps_estimate": 15,
                                          "af_exposure_us": 20000})
    cfg_measured, _ = build_config(row, 0.2, {"camera_fps_estimate": 15,
                                              "af_exposure_us": 20000,
                                              "sigma_steps": 107.8})
    assert cfg_auto["coarse_curv_stop"] == pytest.approx(0.2401, abs=1e-3)
    # measured σ widens t = h/σ from 0.71 to 0.31 → the ladder shrinks
    assert cfg_measured["coarse_curv_stop"] == pytest.approx(0.0488, abs=1e-3)
    assert cfg_measured["coarse_curv_stop"] < cfg_auto["coarse_curv_stop"]
    assert cfg_measured["coarse_curv_vertex"] < 0
    # probe thresholds scale as 1/σ²
    assert cfg_measured["probe_curv_in"] == pytest.approx(0.3 / 107.8 ** 2,
                                                          rel=1e-6)
    # explicit threshold overrides still win over the measured σ
    cfg_override, _ = build_config(row, 0.2, {"sigma_steps": 107.8,
                                              "coarse_curv_stop": 0.4})
    assert cfg_override["coarse_curv_stop"] == 0.4


def test_build_config_prefers_af_speed_multiplier():
    # v5 canonical key wins over the pre-v5 speed_multiplier column.
    row = {"na": 0.30, "dof_um": 7.0, "coarse_step_um": 2.0, "fine_step_um": 0.4,
           "window_um": 60.0, "af_speed_multiplier": 0.5,
           "speed_multiplier": 0.25}
    cfg, _ = build_config(row, um_per_step=0.2, af_cfg={})
    assert cfg["max_speed"] == 250           # 0.5 × 100 µm/s ÷ 0.2


def test_build_config_falls_back_to_speed_multiplier():
    # pre-v5 rows without the af_ key still resolve through the old name.
    row = {"na": 0.30, "dof_um": 7.0, "coarse_step_um": 2.0, "fine_step_um": 0.4,
           "window_um": 60.0, "speed_multiplier": 0.25}
    cfg, warnings = build_config(row, um_per_step=0.2, af_cfg={})
    assert cfg["max_speed"] == 125
    assert not any("legacy" in w for w in warnings)


def test_hill_speed_by_pos():
    # high at the bottom (far from the anchor), low at the peak
    assert hill_speed_by_pos(0.0, anchor=0.0, v_min=40, v_cap=200,
                             near_steps=10.0, ramp_end_steps=60.0) == 40
    assert hill_speed_by_pos(10.0, anchor=0.0, v_min=40, v_cap=200,
                             near_steps=10.0, ramp_end_steps=60.0) == 40
    assert hill_speed_by_pos(100.0, anchor=0.0, v_min=40, v_cap=200,
                             near_steps=10.0, ramp_end_steps=60.0) == 200
    # linear ramp between: halfway = (40+200)/2
    assert hill_speed_by_pos(35.0, anchor=0.0, v_min=40, v_cap=200,
                             near_steps=10.0, ramp_end_steps=60.0) == 120
    # v_min is clamped down to v_cap
    assert hill_speed_by_pos(0.0, anchor=0.0, v_min=300, v_cap=200,
                             near_steps=10.0, ramp_end_steps=60.0) == 200


def test_predictive_stop_steps():
    # v²/2A + v×(poll+latency) + safety  (firmware v²−=2A integration)
    assert predictive_stop_steps(750.0, 20000.0, 0.02, 0.05, 10) == \
        pytest.approx(14.0625 + 52.5 + 10.0)
    assert predictive_stop_steps(750.0, 0.0, 0.02, 0.05, 10) == 10.0


def test_hill_v_min_floor_and_budget():
    # fine_step 5 steps / (3 × 0.02 s) = 83.3 sps
    assert hill_v_min(5, 0.02) == pytest.approx(83.33, abs=0.1)
    # 1 step at 40 ms exposure → 8.3 sps → firmware floor
    assert hill_v_min(1, 0.04) == 10.0
    assert hill_v_min(1, 0.0) == 10.0       # degenerate exposure → floor


# ---------------------------------------------------------------------------
# Probe (adaptive v2 start analysis)
# ---------------------------------------------------------------------------

def test_probe_near_focus_clear_local_max():
    assert probe_near_focus(100.0, [80.0, 80.0], peak_ratio=0.15) is True
    # the check is against the WEAKER side: 100 vs min(90, 80) = 80 → 1.25 ≥ 1.15
    assert probe_near_focus(100.0, [90.0, 80.0], peak_ratio=0.15) is True
    # 100 vs min(90, 89) = 89 → 1.12 < 1.15
    assert probe_near_focus(100.0, [90.0, 89.0], peak_ratio=0.15) is False
    # 100 vs min(85, 84) = 84 → 1.19 < 1.25 (the equality case 100 = 1.25×80
    # is accepted by the >= bound)
    assert probe_near_focus(100.0, [85.0, 84.0], peak_ratio=0.25) is False
    assert probe_near_focus(100.0, [75.0, 75.0], peak_ratio=0.25) is True


def test_probe_near_focus_not_a_max():
    assert probe_near_focus(50.0, [80.0, 40.0]) is False    # one side higher
    assert probe_near_focus(50.0, [50.0, 40.0]) is False    # tie with a side
    assert probe_near_focus(50.0, []) is False              # degenerate


def test_probe_direction_rising():
    # low-freq rises toward + (both slopes significant) → +1
    assert probe_direction(10.0, 20.0, 30.0, min_slope=0.10) == 1
    # one significant slope (the other below the noise floor) still gives
    # the call: d_lo = 2.5 ≥ 0.1×22.5, d_hi = 0.1 below it
    assert probe_direction(20.0, 22.5, 22.6, min_slope=0.10) == 1
    # falling toward − → −1
    assert probe_direction(30.0, 20.0, 10.0, min_slope=0.10) == -1


def test_probe_direction_blind():
    # both slopes below the noise threshold → None
    assert probe_direction(100.0, 101.0, 102.0, min_slope=0.10) is None
    # conflicting significant slopes (center inside the hill) → None
    assert probe_direction(30.0, 50.0, 30.0, min_slope=0.10) is None
    # a tiny conflicting side below the threshold does not veto the call
    assert probe_direction(45.0, 50.0, 50.5, min_slope=0.10) == 1
    # zero scores are noise-proof (threshold floors at 1e-9)
    assert probe_direction(0.0, 0.0, 0.0, min_slope=0.10) is None


def test_direction_guard_wrong_way():
    # low-freq falling, sharp never rose → wrong way
    assert direction_guard_wrong_way([10, 9, 9], [20, 16, 14]) is True
    # sharp IS climbing (peak-seen gate met) → the guard must not fire
    assert direction_guard_wrong_way([10, 13, 15], [20, 16, 14]) is False
    # low-freq not falling → fine
    assert direction_guard_wrong_way([10, 9, 9], [20, 22, 21]) is False
    # fewer than 3 samples → no call
    assert direction_guard_wrong_way([10, 9], [20, 16]) is False
    # drop below the ratio, but sharp rose exactly to the gate bound
    assert direction_guard_wrong_way([10, 12, 12], [20, 16, 14]) is False


def test_peak_is_complete():
    # samples on BOTH sides of the peak + ≥4 total points → complete
    curve = [(-20, 60), (-10, 88), (0, 100), (10, 70), (20, 40)]
    assert peak_is_complete(curve, 0.0, 100.0) is True
    # both sides but too few total points → not enough to fit
    sparse = [(-10, 80), (0, 100), (10, 70)]
    assert peak_is_complete(sparse, 0.0, 100.0) is False
    # no samples left of the peak (ended while climbing) → incomplete
    one_sided = [(0, 100), (10, 70), (20, 40), (30, 30)]
    assert peak_is_complete(one_sided, 0.0, 100.0) is False
    one_sided_r = [(-30, 30), (-20, 40), (-10, 70), (0, 100)]
    assert peak_is_complete(one_sided_r, 0.0, 100.0) is False
    # boundary case: exactly min_samples total points
    exact = [(-20, 86), (-10, 90), (0, 100), (10, 88)]
    assert peak_is_complete(exact, 0.0, 100.0) is True
    # min_samples is configurable
    assert peak_is_complete(sparse, 0.0, 100.0, min_samples=3) is True


# ---------------------------------------------------------------------------
# Probe v3 (derivative-hybrid) — audit-verified Gaussian fixtures
#
# Model: S = A·exp(−d²/2σ²), σ = 10, A = 100. The pinned values are the
# audit's table (plan project-talos-cheerful-stearns.md §1).
# ---------------------------------------------------------------------------

def _gauss(d, sigma=10.0, a=100.0):
    return a * math.exp(-(d / sigma) ** 2 / 2.0)


def test_curvature_from_three_at_peak():
    # C(0) with δ = 1.4σ: (2·e^(−0.98) − 2)/δ² = −0.6374/σ²
    c = curvature_from_three(_gauss(-14.0), _gauss(0.0), _gauss(14.0), 14.0)
    assert c == pytest.approx(-0.6374 / 100.0, abs=1e-6)


def test_curvature_from_three_zero_crossing_matches_the_formula():
    # d* = arcosh(e^(δ²/2σ²))·σ²/δ — the audit's zero crossing (δ-dependent)
    for delta, sigma in ((14.0, 10.0), (11.0, 10.0), (16.0, 10.0)):
        d_star = math.acosh(math.exp((delta / sigma) ** 2 / 2.0)) \
            * sigma * sigma / delta
        c = curvature_from_three(_gauss(d_star - delta, sigma),
                                 _gauss(d_star, sigma),
                                 _gauss(d_star + delta, sigma), delta)
        assert c == pytest.approx(0.0, abs=1e-9)


def test_curvature_from_three_cancels_linear_background():
    # a linear background (m·d) contributes NOTHING to C — the (1, −2, 1)
    # kernel has zero sum and zero first moment
    delta, m = 14.0, 3.0
    bare = curvature_from_three(_gauss(-delta), _gauss(0.0), _gauss(delta),
                                delta)
    bg = curvature_from_three(_gauss(-delta) + m * -delta,
                              _gauss(0.0), _gauss(delta) + m * delta, delta)
    assert bg == pytest.approx(bare, abs=1e-9)


def test_curvature_from_three_degenerate_center_is_none():
    assert curvature_from_three(10.0, 0.0, 10.0, 5.0) is None
    assert curvature_from_three(10.0, -1.0, 10.0, 5.0) is None


def test_window_parabola_matches_the_audit_raw_table():
    # 5 samples at h = 0.5σ, window center d = 0.4σ, A = 1: the audit's
    # raw 2a (x in σ units) = −0.6211/σ² and b = −0.2496/σ (kernel
    # (2,−1,−2,−1,2)/7h² and (−2,−1,0,1,2)/10h)
    xs = [0.4 + 0.5 * i for i in (-2, -1, 0, 1, 2)]
    ys = [math.exp(-x * x / 2.0) for x in xs]
    a, b = window_parabola(xs, ys)
    assert 2.0 * a == pytest.approx(-0.6211, abs=1e-3)
    assert b == pytest.approx(-0.2496, abs=1e-3)
    # d = 0.8σ: 2a = −0.2752/σ² (the audit's −0.28)
    xs = [0.8 + 0.5 * i for i in (-2, -1, 0, 1, 2)]
    ys = [math.exp(-x * x / 2.0) for x in xs]
    a, b = window_parabola(xs, ys)
    assert 2.0 * a == pytest.approx(-0.2752, abs=1e-3)


def test_window_curvature_metric_and_zero_crossing():
    # the normalized metric 2a·h²/S_max (scale-free — the stop criterion);
    # S_max is the WINDOW MAX (99.5 at x = −1 step, not the 92.3 center):
    # at d = 0.4σ, h = 0.5σ → −0.6211·0.25/0.9950 = −0.1561
    xs = [(0.4 + 0.5 * i) * 10.0 for i in (-2, -1, 0, 1, 2)]
    ys = [_gauss(x) for x in xs]
    curv, b = window_curvature(xs, ys)
    assert curv == pytest.approx(-0.1561, abs=1e-3)
    # the metric crosses zero at ≈ 1.10σ (the audit's zero) — and is
    # POSITIVE beyond it
    for d, sign in ((1.02, -1.0), (1.10, 0.0), (1.18, 1.0)):
        xs = [(d + 0.5 * i) * 10.0 for i in (-2, -1, 0, 1, 2)]
        ys = [_gauss(x) for x in xs]
        curv, _b = window_curvature(xs, ys)
        if sign == 0.0:
            assert abs(curv) < 0.01
        else:
            assert curv * sign > 0


def test_sg_curvature_at_matches_the_audit_table():
    # the closed form reproduces the metric table: −0.156 @ 0.4σ, −0.070
    # @ 0.8σ, zero ≈ 1.10σ at t = 0.5 (S_max = the window max)
    assert sg_curvature_at(0.4, 0.5) == pytest.approx(-0.1561, abs=1e-3)
    assert sg_curvature_at(0.8, 0.5) == pytest.approx(-0.0702, abs=1e-3)
    assert abs(sg_curvature_at(1.10, 0.5)) < 0.01


def test_window_parabola_noise_se_matches_the_audit_table():
    # SE(2a) = 2.138·σₙ and SE(b) = 0.633·σₙ at h = 0.5σ (x in σ units) —
    # the audit's kernel-noise figures, Monte-Carlo-verified
    rng = np.random.default_rng(42)
    xs = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    two_a, b = [], []
    for _ in range(2000):
        ys = rng.normal(0.0, 1.0, 5)
        a, bb = window_parabola(xs, ys)
        two_a.append(2.0 * a)
        b.append(bb)
    assert float(np.std(two_a)) == pytest.approx(2.138, abs=0.15)
    assert float(np.std(b)) == pytest.approx(0.633, abs=0.05)


def test_window_vertex_error_regimes():
    # vertex = window_center − b/(2a): ≤ 0.5σ error only for d ≤ 0.7σ
    # (the trusted-vertex floor's rationale); beyond ~0.9σ the window
    # parabola opens UPWARD (a > 0) and there is no vertex at all
    for d in (0.4, 0.7):
        xs = [(d + 0.5 * i) for i in (-2, -1, 0, 1, 2)]  # σ units
        ys = [math.exp(-x * x / 2.0) for x in xs]
        a, b = window_parabola(xs, ys)
        assert a < 0
        vertex = d - b / (2.0 * a)
        assert abs(vertex) <= 0.5 + 1e-6
    xs = [(1.3 + 0.5 * i) for i in (-2, -1, 0, 1, 2)]
    ys = [math.exp(-x * x / 2.0) for x in xs]
    a, _b = window_parabola(xs, ys)
    assert a > 0  # no valid vertex on the far flank


def test_coarse_curv_stop_ladder_matches_the_audit_mags():
    # the audit's three points: 5× (t≈0.7) → 0.25, 10× (t≈1.1) → 0.10,
    # 20× (t≈2.1) → 0.06 (clamped)
    assert coarse_curv_stop_ladder(0.7) == pytest.approx(0.25)
    assert coarse_curv_stop_ladder(1.1) == pytest.approx(0.1012, abs=1e-3)
    assert coarse_curv_stop_ladder(2.1) == pytest.approx(0.06)
    # dense sampling (sim, t = 0.25): the metric shrinks ∝ t² — the
    # threshold shrinks with it (fire distance stays ~constant)
    assert coarse_curv_stop_ladder(0.25) == pytest.approx(0.0319, abs=1e-3)


def test_coarse_curv_stop_ladder_fire_distance_regimes():
    # dense sampling (t ≤ 0.7): the stop fires at a curvature BELOW the
    # 0.7σ vertex floor → the vertex IS trusted (fire inside the valid
    # regime); wide sampling (t > 0.7): the stop fires before the floor
    # is reached → the stop position is used instead (the b-gate is the
    # load-bearing discriminator there)
    for t in (0.25, 0.5, 0.7):
        thr = coarse_curv_stop_ladder(t)
        assert sg_curvature_at(0.7, t) >= -thr
    for t in (1.1, 2.1):
        thr = coarse_curv_stop_ladder(t)
        assert sg_curvature_at(0.7, t) < -thr


def test_fit_slope_with_se_line_and_noise():
    rng = np.random.default_rng(7)
    xs = np.arange(20.0)
    ys = 0.5 * xs + rng.normal(0.0, 2.0, 20)
    b, se = fit_slope_with_se(xs, ys)
    assert b == pytest.approx(0.5, abs=0.3)
    sxx = float(((xs - xs.mean()) ** 2).sum())
    assert se == pytest.approx(2.0 / math.sqrt(sxx), rel=0.3)
    # degenerate inputs
    assert fit_slope_with_se([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert fit_slope_with_se([1.0], [1.0]) is None


def test_pooled_noise_from_probe_detrends_the_hill():
    # three stationary points ON A HILL + σₙ = 2 noise: the detrended
    # residual recovers ~σₙ, not the hill's spread
    rng = np.random.default_rng(11)
    xs = [-14.0, 0.0, 14.0]
    slope = 0.5
    sigma = pooled_noise_from_probe(
        [slope * x + 50.0 + rng.normal(0.0, 2.0) for x in xs], xs)
    assert 0.5 < sigma < 4.0
    # three points EXACTLY on a line → floored at 1% of the mean score
    floored = pooled_noise_from_probe([40.0, 50.0, 60.0], xs)
    assert floored == pytest.approx(0.01 * 50.0)
    # degenerate
    assert pooled_noise_from_probe([1.0, 2.0]) is None


def test_guard_reversal_test_2sigma_confidence():
    # a consistent wrong-way slope: b̂ = −0.6 with σₙ = 0.5 over a wide
    # position range → the 2σ interval lies below zero → reverse
    xs = [float(i) for i in range(6)]
    ys = [10.0 - 0.6 * x for x in xs]
    assert guard_reversal_test(xs, ys, sigma_noise=0.5) is True
    # the same slope with a large σₙ → not significant → no reverse
    assert guard_reversal_test(xs, ys, sigma_noise=1.5) is False
    # rising → never reverse
    ys_up = [10.0 + 0.6 * x for x in xs]
    assert guard_reversal_test(xs, ys_up, sigma_noise=0.5) is False
    # noisy flat (fixed seed) → no reverse
    rng = np.random.default_rng(3)
    ys_flat = [50.0 + rng.normal(0.0, 1.0) for _ in xs]
    assert guard_reversal_test(xs, ys_flat, sigma_noise=0.8) is False
    # degenerate positions → no call
    assert guard_reversal_test([1.0, 1.0, 1.0], [1.0, 2.0, 3.0],
                               sigma_noise=0.5) is False


def _classify(sharp, low, delta=14.0, floor=20.0,
              curv_in=0.3 / 100.0, curv_out=0.1 / 100.0,
              cluster_center_ratio=0.5, cluster_side_ratio=0.4,
              cluster_min_score=10.0):
    # min_score 10 enables the branch for every fixture's scale (the
    # real gate cases pass their own explicit values)
    return probe_classify(sharp[0], sharp[1], sharp[2], delta,
                          low[0], low[1], low[2], floor, curv_in, curv_out,
                          cluster_center_ratio=cluster_center_ratio,
                          cluster_side_ratio=cluster_side_ratio,
                          cluster_min_score=cluster_min_score)


def test_probe_classify_near_curvature():
    # Gaussian peak dead-on (δ = 1.4σ): C = −0.637/σ² < −0.3/σ²
    v = _classify([_gauss(-14.0), 100.0, _gauss(14.0)],
                  [50.0, 100.0, 50.0])
    assert v.near and v.branch == "near-curvature"
    assert v.direction is None


def test_probe_classify_curvature_catches_what_weaker_side_misses():
    # small δ/σ (0.5): the sides are 88.25 — the v2 weaker-side test
    # needs 100 ≥ 1.15·88.25 = 101.5 and FAILS; the curvature branch
    # (C = −0.0094 < −0.003) still says near
    sides = _gauss(5.0)
    v = _classify([sides, 100.0, sides], [50.0, 100.0, 50.0], delta=5.0)
    assert v.near and v.branch == "near-curvature"


def test_probe_classify_near_plateau():
    # flat-topped plateau: C too shallow for the curvature branch but the
    # v2 weaker-side test passes — the plateau OR-branch
    v = _classify([90.0, 100.0, 85.0], [50.0, 100.0, 50.0])
    assert v.near and v.branch == "near-plateau"
    assert v.direction is None


def test_probe_classify_valley_is_not_near():
    # audit #8/#9: the between-planes valley — center BELOW a side. The
    # center-max gate must block the near call (C would be huge-positive
    # through the small S_c) and the case must fall to the flank/direction
    # (the relative floor passes: armed IN the valley the preflight
    # baseline ≈ the valley score, so 0.3×baseline < S_c). The low-freq
    # metric is broad — monotone toward the wafer side, no valley dip.
    v = _classify([40.0, 10.0, 55.0], [20.0, 40.0, 90.0], floor=5.0)
    assert not v.near
    assert v.branch == "flank"
    assert v.direction == 1  # the low-freq slope owns the direction


def test_probe_classify_valley_below_floor_is_blind():
    # the same valley with a strict floor: the floor short-circuit fires
    # first — still never a near call
    v = _classify([40.0, 10.0, 55.0], [20.0, 40.0, 90.0], floor=20.0)
    assert not v.near and v.branch == "slope" and v.direction == 1


def test_probe_classify_flank_direction_from_slope():
    # flank beyond the inflection (C > +curv_out) but NO reliable low-freq
    # slope → still a flank, direction None (blind) — the caller runs the
    # blind sweep
    v = _classify([40.0, 10.0, 55.0], [40.0, 40.0, 40.0], floor=5.0)
    assert not v.near and v.branch == "flank"
    assert v.direction is None


def test_probe_classify_slope_and_blind():
    # no near (tie center), no flank, rising low-freq → slope fallback
    v = _classify([40.0, 41.0, 40.0], [10.0, 20.0, 30.0])
    assert not v.near and v.branch == "slope"
    assert v.direction == 1
    # flat everything → blind
    v = _classify([6.0, 6.0, 6.0], [10.0, 10.0, 10.0], floor=3.0)
    assert not v.near and v.branch == "blind"
    assert v.direction is None


def test_probe_classify_near_cluster():
    # the arm sits on the multi-peak plateau's shoulder: a neighbor
    # sub-peak beats the center (center-max fails) but all three points
    # are HIGH — the cluster branch enters stage 2 at the best point
    # instead of the slow blind sweep (hardware-found: the 5× field's
    # ~190-step-wide bump of comparable sub-peaks)
    v = _classify([1500.0, 1800.0, 2400.0], [40.0, 50.0, 45.0])
    assert v.near and v.branch == "near-cluster"
    assert v.direction is None
    # a weaker center between two comparable, well-spread planes
    v = _classify([1800.0, 1700.0, 2400.0], [50.0, 55.0, 50.0])
    assert v.near and v.branch == "near-cluster"


def test_probe_classify_cluster_excludes_the_flat_noise_triple():
    # a far-away FLAT region: a noise triple has ≤5% spread (below the
    # 1.2 spread gate) — the cluster branch must NOT fire (stage 2
    # would fail on the flat where the blind sweep finds the peak)
    v = _classify([41.0, 40.0, 41.5], [10.0, 20.0, 30.0])
    assert not v.near
    assert v.branch == "slope" and v.direction == 1
    # the apex case (center IS the max, weaker-side fails) stays on the
    # v2 path — the cluster is only the center-max-fails shoulder
    v = _classify([40.0, 41.0, 40.0], [10.0, 20.0, 30.0])
    assert not v.near
    assert v.branch == "slope"


def test_probe_classify_cluster_excludes_the_valley():
    # audit #8/#9's valley: the center is a small fraction of the sides
    # (0.24 vs 2.93 → below the 0.5 center ratio) — the cluster branch
    # must NOT fire (the mount must not win a direct stage-2 entry)
    v = _classify([166.7, 24.2, 292.8], [20.0, 40.0, 90.0], floor=5.0)
    assert not v.near
    assert v.branch == "flank"


def test_probe_classify_cluster_min_score_gate():
    # the far tail's points are mutually comparable (the ratios pass)
    # but ABSOLUTELY low — only the min-score gate can tell a flat far
    # tail from a plateau shoulder (sim-found: the blind-ahead test
    # landed a far wiggle at 112 instead of the truth at 600)
    v = _classify([4.7, 5.6, 6.6], [10.0, 20.0, 30.0], cluster_min_score=500.0)
    assert not v.near
    # the real shoulder passes the same gate
    v = _classify([1500.0, 1800.0, 2400.0], [40.0, 50.0, 45.0],
                  cluster_min_score=500.0)
    assert v.near and v.branch == "near-cluster"
    # 0 disables the branch entirely
    v = _classify([1500.0, 1800.0, 2400.0], [40.0, 50.0, 45.0],
                  cluster_min_score=0.0)
    assert not v.near


def test_probe_classify_cluster_excludes_the_flank():
    # on a real flank the FAR side is low (15% of the center — below the
    # 0.4 side ratio): the case stays on the directed/blind path
    v = _classify([2500.0, 1200.0, 300.0], [10.0, 20.0, 30.0])
    assert not v.near
    # a zero side ratio DISABLES the branch (hybrid-OR safety rule)
    v = _classify([1500.0, 1800.0, 2400.0], [40.0, 50.0, 45.0],
                  cluster_side_ratio=0.0)
    assert not v.near


def test_probe_classify_below_floor():
    # S_c below the relative floor → classification short-circuits to the
    # low-freq call (a noise triple must never read "near")
    v = _classify([4.0, 5.0, 4.0], [10.0, 20.0, 30.0], floor=10.0)
    assert not v.near and v.branch == "slope" and v.direction == 1
    v = _classify([4.0, 5.0, 4.0], [10.0, 10.0, 10.0], floor=10.0)
    assert not v.near and v.branch == "blind" and v.direction is None
