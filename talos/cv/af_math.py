"""Pure autofocus math: position interpolation for measure-while-moving
sweeps, multi-peak selection, parabolic fit, backlash landing planning,
and µm→steps config conversion.

No Qt, no cv2 — numpy-only, so everything here is trivially unit-testable
and reusable from the GUI service and the CLI tools alike.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import numpy as np

# The firmware's absolute speed window (steps/s). FocusStageDriver clamps to
# [max(clamp_speed_lo, _SPEED_LO), min(clamp_speed_hi, _SPEED_HI)], so these
# are the widest bounds any planner may assume.
_SPEED_LO, _SPEED_HI = 10, 5000


def driver_speed_clamp(focus_cfg: dict | None = None) -> tuple[int, int]:
    """The (lo, hi) steps/s the focus DRIVER will accept, from the same
    settings keys it reads.

    Planning a speed outside this window does not get clamped by anyone
    downstream — the driver raises CommandRejectedError mid-sweep, which the
    autofocus reports as "internal error". The planner therefore clamps to
    the driver's window rather than to a constant that merely matches the
    shipped defaults (lowering ``devices.focus.clamp_speed_hi`` used to make
    the two disagree).
    """
    cfg = focus_cfg or {}
    try:
        lo = max(int(cfg.get("clamp_speed_lo", _SPEED_LO)), _SPEED_LO)
        hi = min(int(cfg.get("clamp_speed_hi", _SPEED_HI)), _SPEED_HI)
    except (TypeError, ValueError):
        return _SPEED_LO, _SPEED_HI
    if hi < lo:            # a hand-edited, self-contradicting pair
        return _SPEED_LO, _SPEED_HI
    return lo, hi


# ---------------------------------------------------------------------------
# Position interpolation
# ---------------------------------------------------------------------------

def _time_of(item: tuple) -> float:
    return item[0]


def interpolate_position(t: float, history: list[tuple[float, int]],
                         gap_max_ms: float = 300.0) -> float | None:
    """Position at monotonic time ``t`` from a (t, pos) history (sorted by
    time). Linear interpolation between the bracketing samples; None when
    ``t`` lies outside the history span or the bracket gap exceeds
    ``gap_max_ms`` (a camera stall means no trustworthy position)."""
    if not history:
        return None
    # bisect with a key: this is called once per SCORED FRAME while the
    # history grows for the whole pass, so materializing the time list
    # here was O(n) per call (O(n²) per pass).
    idx = bisect.bisect_left(history, t, key=_time_of)
    if idx == 0:
        return float(history[0][1]) if history[0][0] == t else None
    if idx == len(history):
        return None
    t_l, p_l = history[idx - 1]
    t_r, p_r = history[idx]
    if t_r <= t_l:
        return None
    if (t_r - t_l) * 1000.0 > gap_max_ms:
        return None
    frac = (t - t_l) / (t_r - t_l)
    return p_l + (p_r - p_l) * frac


# ---------------------------------------------------------------------------
# Peak picking
# ---------------------------------------------------------------------------

@dataclass
class PeakInfo:
    pos: float
    score: float
    at_edge: bool


def pick_peak(curve: list[tuple[float, float]], center: float,
              prominence: float = 0.15,
              edge_margin_steps: float = 1.0) -> PeakInfo | None:
    """Best focus peak in a (pos, score) curve.

    - Scores are normalized to 0..1; a local maximum must rise at least
      ``prominence`` (fraction of the score range above the curve floor)
      to be a candidate — noise wiggles do not count.
    - When several candidates exist (e.g. wafer surface AND copper mount
      both inside the window), the one NEAREST ``center`` (the arm
      position) wins — never the global maximum.
    - ``at_edge`` flags peaks sitting within ``edge_margin_steps`` of the
      window bounds (the true peak is probably outside).
    - A flat curve (range ~ 0) or no candidate → None."""
    if len(curve) < 3:
        return None
    xs = [p for p, _ in curve]
    ys = [s for _, s in curve]
    lo, hi = min(ys), max(ys)
    if hi - lo <= 1e-12:
        return None
    ys_norm = [(y - lo) / (hi - lo) for y in ys]
    # 3-point moving average (endpoints averaged over 2) — keeps the
    # position axis unchanged while suppressing single-sample spikes.
    smoothed = [(ys_norm[0] + ys_norm[1]) / 2.0]
    smoothed += [(ys_norm[i - 1] + ys_norm[i] + ys_norm[i + 1]) / 3.0
                 for i in range(1, len(ys_norm) - 1)]
    smoothed.append((ys_norm[-2] + ys_norm[-1]) / 2.0)
    # Endpoints ARE candidates (>= their single neighbor): a curve that
    # climbs monotonically into the window edge reports an edge peak —
    # the diagnostic the controller turns into "widen the search window".
    # Interior peaks accept a score PLATEAU (>= on the right): symmetric
    # scenes give the two points straddling the true peak identical
    # scores, and a strict > would reject both.
    candidates = []
    if smoothed[0] >= smoothed[1] and smoothed[0] >= prominence:
        candidates.append(0)
    candidates += [i for i in range(1, len(smoothed) - 1)
                   if smoothed[i] > smoothed[i - 1]
                   and smoothed[i] >= smoothed[i + 1]
                   and smoothed[i] >= prominence]
    if smoothed[-1] >= smoothed[-2] and smoothed[-1] >= prominence:
        candidates.append(len(smoothed) - 1)
    if not candidates:
        return None
    best_i = min(candidates, key=lambda i: abs(xs[i] - center))
    x_min, x_max = min(xs), max(xs)
    at_edge = (xs[best_i] - x_min <= edge_margin_steps
               or x_max - xs[best_i] <= edge_margin_steps)
    return PeakInfo(pos=float(xs[best_i]), score=float(ys[best_i]),
                    at_edge=at_edge)


# ---------------------------------------------------------------------------
# Probe (adaptive v2 start analysis)
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    near_focus: bool
    direction: int | None            # +1 / -1 / None (blind — no info)
    samples: list[tuple[float, float, float]]  # (pos, sharp_score, low_score)


def probe_near_focus(sharp_center: float, sharp_sides: list[float],
                     peak_ratio: float = 0.15) -> bool:
    """Is the center a clear local maximum on the SHARP metric? The center
    must beat both sides AND clear the weaker side by ``peak_ratio`` (a
    noise flat would otherwise read as a peak)."""
    if not sharp_sides:
        return False
    if sharp_center <= max(sharp_sides):
        return False
    return sharp_center >= (1.0 + peak_ratio) * min(sharp_sides)


def probe_direction(low_lo: float, low_center: float, low_hi: float,
                    min_slope: float = 0.10) -> int | None:
    """Hill direction from three LOW-FREQUENCY scores at [center−δ, center,
    center+δ]. The low-freq metric's broad response gives a usable slope
    far from focus, where the sharp metric is flat (the reason the probe
    uses it for direction but the sharp metric for near-focus).

    None (blind) when both slopes sit below the noise threshold, or when
    they point in OPPOSITE directions with both significant (the center is
    inside the hill — but probe_near_focus should have caught that).
    """
    threshold = min_slope * max(low_center, 1e-9)
    d_lo = low_center - low_lo      # rising toward + when positive
    d_hi = low_hi - low_center
    if abs(d_lo) < threshold and abs(d_hi) < threshold:
        return None
    if d_lo * d_hi < 0 and min(abs(d_lo), abs(d_hi)) >= threshold:
        return None                  # conflicting significant slopes
    return 1 if (d_lo + d_hi) >= 0 else -1


def direction_guard_wrong_way(sharp_scores: list[float],
                              low_scores: list[float],
                              drop_ratio: float = 0.15,
                              rise_ratio: float = 0.2) -> bool:
    """Early-scan wrong-direction check: the low-frequency series is
    FALLING (moving away from the hill) while the sharp series has NOT
    yet risen ``rise_ratio`` above its start (the cont_pass peak-seen
    gate — a real climb must not trigger the guard)."""
    if len(low_scores) < 3 or len(sharp_scores) < 3:
        return False
    if low_scores[-1] >= low_scores[0] * (1.0 - drop_ratio):
        return False
    if max(sharp_scores) >= sharp_scores[0] * (1.0 + rise_ratio):
        return False
    return True


def peak_is_complete(curve: list[tuple[float, float]], peak_pos: float,
                     peak_score: float, min_samples: int = 4) -> bool:
    """Does the curve contain a COMPLETE peak at ``peak_pos`` — samples on
    BOTH sides of it (the pass climbed over the peak, it did not end
    while still climbing) and at least ``min_samples`` total points (the
    peak is a picked prominence, not a 2-sample wiggle)? The adaptive v2
    boundary-salvage gate.

    Deliberately permissive on density: the sharp metric's peak is so
    narrow that a near-peak score fraction (e.g. ≥85% of the peak) is
    only true within a fraction of a fine step — a fly-by curve would
    almost never qualify, defeating the salvage. The downstream
    fit-vs-anchor trust check and the stationary re-measure's end-max
    discipline are the real guards; this gate only rejects curves that
    ended while still climbing (peak at the curve end) or are too small
    to fit at all."""
    xs = [p for p, _ in curve]
    if not (any(x < peak_pos for x in xs) and any(x > peak_pos for x in xs)):
        return False
    return len(curve) >= min_samples


# ---------------------------------------------------------------------------
# Parabolic fit
# ---------------------------------------------------------------------------

def parabolic_fit(curve: list[tuple[float, float]], peak_pos: float,
                  n_parabolic_points: int = 3) -> float | None:
    """Parabola through the n points nearest the peak (least squares for
    n > 3 — an asymmetric sample set around a narrow peak biases an exact
    3-point vertex). None when the fit is invalid (flat or
    upward-opening)."""
    near = sorted(curve, key=lambda item: abs(item[0] - peak_pos))[:max(n_parabolic_points, 3)]
    if len(near) < 3:
        return None
    near.sort(key=lambda item: item[0])
    xs = [p for p, _ in near]
    ys = [s for _, s in near]
    if len(set(xs)) < 3:
        return float(xs[ys.index(max(ys))])
    # Flat-topped peak (DOF plateau — real optics AND symmetric sampling
    # both produce it): a parabola through a plateau is unstable, so when
    # most near points tie at the top, take the plateau's centroid.
    ties = [x for x, y in near if y >= max(ys) * 0.98]
    if len(ties) >= 3:
        return float(sum(ties) / len(ties))
    # Least-squares fit of y = a x^2 + b x + c (exact for 3 points).
    matrix = np.array([[x * x, x, 1.0] for x in xs])
    a, b, _c = np.linalg.lstsq(matrix, ys, rcond=None)[0]
    if a >= 0:  # flat or valley — the parabola opens upward
        return None
    return float(-b / (2 * a))


# ---------------------------------------------------------------------------
# Backlash landing
# ---------------------------------------------------------------------------

def landing_plan(target: float, backlash_steps: int, margin_steps: int,
                 bounds: tuple[float, float],
                 preferred_dir: int = 1) -> tuple[int, int, bool]:
    """Overshoot-and-return landing: approach the target from ONE side so
    gear backlash is taken up identically every time.

    Returns (overshoot_target, approach_direction, degraded). The approach
    direction defaults to ``preferred_dir`` (the fine sweep's direction —
    the fitted peak position is only valid when measured and landed from
    the same side of the backlash); it flips when the overshoot would
    leave the search bounds.

    ``degraded`` is True when the bounds clamp left the overshoot closer
    to the target than one backlash take-up: the two landing moves then
    travel too little to absorb the backlash, so the load can land off by
    up to ``backlash_steps`` (the caller reports it instead of silently
    pretending the landing was compensated)."""
    lo, hi = bounds
    travel = int(backlash_steps + margin_steps)
    direction = 1 if preferred_dir >= 0 else -1
    overshoot = target - direction * travel
    if overshoot < lo:
        direction = -1
        overshoot = target + travel
    elif overshoot > hi:
        direction = 1
        overshoot = target - travel
    overshoot = max(lo, min(hi, overshoot))
    degraded = abs(overshoot - target) < int(backlash_steps)
    return int(overshoot), direction, degraded


# ---------------------------------------------------------------------------
# Adaptive-strategy speed math
# ---------------------------------------------------------------------------

def speed_multiplier(na: float, na_min: float = 0.15) -> float:
    """Per-objective speed scale, default proportional to 1/NA² (the DoF
    scales as λ/NA², so the tolerable motion-blur budget — and with it the
    usable scan speed — scales the same way). Normalized so the
    lowest-NA objective gets 1.0."""
    na = float(na)
    if na <= 0:
        return 1.0
    return (na_min / na) ** 2


def recommended_multipliers(na: float, mag: float, na_min: float = 0.15,
                            mag_min: float = 5.0) -> tuple[float, float, float]:
    """(af, focus, stage) auto-calc targets from mag & NA: the AF and
    manual-focus speeds scale with 1/NA² (the DoF blur budget, the same
    model as ``speed_multiplier``); the stage speed scales with 1/mag
    (XY motion carries no focus budget). Normalized to the lowest-power
    objective and clamped to [0.01, 1.0]; invalid input falls back to
    (1.0, 1.0, 1.0)."""
    try:
        na = float(na)
        mag = float(mag)
    except (TypeError, ValueError):
        return (1.0, 1.0, 1.0)
    if na <= 0 or mag <= 0:
        return (1.0, 1.0, 1.0)

    def _clamp(x: float) -> float:
        return max(0.01, min(1.0, x))

    af = _clamp((na_min / na) ** 2)
    stage = _clamp(mag_min / mag)
    return (af, af, stage)


def hill_speed_by_pos(pos: float, anchor: float, v_min: int, v_cap: int,
                      near_steps: float = 10.0,
                      ramp_end_steps: float = 60.0) -> int:
    """Position-based climb speed: full ``v_cap`` far from the anchor
    (the pass edges — "high at the bottom"), ``v_min`` close to it (the
    peak — "low at the peak"), linear ramp between. Deterministic: the
    score-based schedule collapsed to v_min everywhere on a DOF plateau
    (the plateau's scores sit above the threshold at every position),
    so the user never saw the variable speed — hardware-verified."""
    v_min = min(int(v_min), int(v_cap))
    d = abs(float(pos) - float(anchor))
    if d <= near_steps:
        return int(v_min)
    if d >= ramp_end_steps:
        return int(v_cap)
    frac = (d - near_steps) / max(ramp_end_steps - near_steps, 1.0)
    return int(round(v_min + frac * (v_cap - v_min)))


def predictive_stop_steps(v: float, accel: float, poll_s: float,
                          latency_s: float, safety_steps: float) -> float:
    """How many steps before the pass edge to begin the ramp stop: the
    firmware deceleration distance (v²/2A, exact discrete integration) +
    the worst-case position slip while the stop command is in flight +
    a safety margin. Mirrors contKinematics' v² −= 2A per step."""
    if accel <= 0:
        return float(safety_steps)
    return (v * v) / (2.0 * accel) + v * (poll_s + latency_s) + safety_steps


def hill_v_min(fine_step_steps: int, exposure_s: float,
               floor_sps: int = 10) -> float:
    """Near-peak speed: motion blur (v × exposure) must stay ≤ fine_step/3
    so the samples the parabolic lock-on fits are sharp. Never below the
    firmware floor."""
    if exposure_s <= 0:
        return float(floor_sps)
    return max(float(floor_sps), fine_step_steps / (3.0 * exposure_s))


# ---------------------------------------------------------------------------
# µm → steps config conversion
# ---------------------------------------------------------------------------

def um_to_steps(um: float, um_per_step: float, *, floor: int = 1) -> int:
    """µm → whole steps, rounded. ``floor`` is the smallest magnitude the
    caller will accept:

    - ``floor=1`` (default) for a step SIZE — a commanded step must move;
    - ``floor=0`` for a DELTA — "no motion" is a real answer, which is why
      ``objective_offsets`` used to carry its own copy of this conversion.

    A non-positive ``um_per_step`` (hand-edited settings) returns 0 instead
    of raising: the span then collapses and the autofocus reports an empty
    search window rather than dying with a ZeroDivisionError.
    """
    if um_per_step <= 0:
        return 0
    steps = int(round(um / um_per_step))
    if floor <= 0 or abs(steps) >= floor:
        return steps
    return floor if steps >= 0 else -floor


def _clamp_speed(steps_per_s: float,
                 clamp: tuple[int, int] = (_SPEED_LO, _SPEED_HI)) \
        -> tuple[int, str | None]:
    lo, hi = clamp
    if steps_per_s < lo:
        return lo, f"speed clamped to the driver floor {lo} steps/s"
    if steps_per_s > hi:
        return hi, f"speed clamped to the driver ceiling {hi} steps/s"
    return int(round(steps_per_s)), None


def build_config(objective_row: dict, um_per_step: float, af_cfg: dict,
                 na_min: float | None = None, focus_cfg: dict | None = None) \
        -> tuple[dict, list[str]]:
    """Convert a user-facing objective row (µm units) into step-based
    AutofocusConfig kwargs, with sanity warnings.

    Speeds: every objective's effective speed = the global base × its
    ``af_speed_multiplier`` (default ∝ 1/NA² referenced to ``na_min`` —
    the LOWEST-POWER objective's NA from the table, passed by the
    caller; that objective gets exactly the base speed = the max of the
    table). Legacy ``speed_multiplier`` and ``coarse_speed_um_s``
    columns still accepted; explicit per-row multipliers win. The
    coarse budget is blur ≤ DOF/4 — the coarse metric is low-frequency
    and only needs to localize the peak, so it tolerates far more blur
    than the fine metric. The multiplier flows into the stage-2 window
    through the coarse sampling spacing (speed/fps); the TOTAL search
    span stays the user's per-objective window (the args own the
    bounds).

    Returns (kwargs, warnings). kwargs keys: coarse_step, fine_step,
    span_steps, window_plus_steps, window_minus_steps, max_speed,
    fine_speed, landing_speed, backlash_steps, overshoot_margin_steps,
    fine_window_steps, coarse_speed, hill_v_cap, hill_v_min (+ the v3
    derivative fields)."""
    warnings: list[str] = []
    dof_um = float(objective_row.get("dof_um", 0.0) or 0.0)
    coarse_step_um = float(objective_row.get("coarse_step_um", 5.0))
    fine_step_um = float(objective_row.get("fine_step_um", 1.0))
    backlash_um = float(objective_row.get("backlash_um", 0.0) or 0.0)
    na = float(objective_row.get("na", 0.0) or 0.0)
    base_coarse_um_s = float(af_cfg.get("coarse_speed_base_um_s", 100.0))

    # v5: af_speed_multiplier is canonical; the pre-v5 speed_multiplier and
    # the v3 coarse_speed_um_s column still resolve for older settings.
    mult = objective_row.get("af_speed_multiplier")
    if not mult or float(mult) <= 0:
        mult = objective_row.get("speed_multiplier")
    if not mult or float(mult) <= 0:
        legacy = float(objective_row.get("coarse_speed_um_s", 0.0) or 0.0)
        if legacy > 0:
            mult = legacy / base_coarse_um_s
            warnings.append(f"legacy coarse_speed_um_s {legacy:.2f} — derived "
                            f"af_speed_multiplier {mult:.4f}")
        elif na > 0:
            mult = speed_multiplier(na, na_min if na_min else 0.15)
        else:
            mult = 1.0
            warnings.append("no NA and no speed_multiplier — multiplier 1.0")
    mult = float(mult)
    # v7: the search window = the user-editable max-window bounds
    # (autofocus.window_plus_um / window_minus_um, ±500 µm by default)
    # scaled by the AF speed multiplier — every objective derives its
    # default search span from the same lever (the per-objective
    # window_um column is legacy display data). The bounds stay
    # separately representable so asymmetric windows survive to the
    # preflight clamp.
    plus_um = float(af_cfg.get("window_plus_um", 500.0))
    minus_um = float(af_cfg.get("window_minus_um", 500.0))
    if plus_um <= 0 or minus_um <= 0:
        # A 0 µm bound is not "unbounded": µm→steps floors a step size at 1,
        # so it used to become a ±1-step window and the run died claiming
        # "the arm is outside the window" — a message that points at the
        # wrong thing entirely. Keep the search alive with one fine step and
        # say what happened.
        warnings.append(
            f"search window is 0 µm on the "
            f"{'plus' if plus_um <= 0 else 'minus'} side — using one fine "
            f"step ({fine_step_um} µm); set the window in Autofocus settings")
        plus_um = max(plus_um, fine_step_um)
        minus_um = max(minus_um, fine_step_um)
    span_um = (plus_um + minus_um) * mult

    coarse_step = um_to_steps(coarse_step_um, um_per_step)
    fine_step = um_to_steps(fine_step_um, um_per_step)
    span_steps = um_to_steps(span_um, um_per_step)
    window_plus_steps = um_to_steps(plus_um * mult, um_per_step)
    window_minus_steps = um_to_steps(minus_um * mult, um_per_step)
    eff_coarse_um_s = mult * base_coarse_um_s
    # The DRIVER's window, not a constant (see driver_speed_clamp): every
    # speed below must land inside it or the sweep dies on CommandRejected.
    speed_lo, speed_hi = driver_speed_clamp(focus_cfg)
    coarse_speed, warn = _clamp_speed(eff_coarse_um_s / um_per_step,
                                      (speed_lo, speed_hi))
    if warn:
        warnings.append(f"coarse speed: {warn}")
    max_speed = coarse_speed
    landing_speed = int(af_cfg.get("landing_speed", 50))
    landing_speed = max(speed_lo, min(speed_hi, landing_speed))
    # Classic step-and-shoot transit at full speed: the moves themselves
    # are blur-free (only stationary frames are scored), so the fine
    # sweep travels as fast as the coarse scan.
    fine_speed = max(landing_speed, max_speed)
    # Adaptive hill-climb cap: the far-from-peak zone runs at the coarse
    # speed (the schedule drops to v_min at 0.6× the coarse peak — the
    # near-peak zone is where the blur budget matters).
    hill_v_cap = coarse_speed
    exposure_s = float(af_cfg.get("af_exposure_us", 20000)) / 1e6
    if exposure_s > 0 and fine_step / (3.0 * exposure_s) < speed_lo:
        warnings.append(f"hill v_min below the driver's {speed_lo} sps floor — "
                        f"near-peak blur budget exceeded (warning)")
    hill_v_min_steps_s = hill_v_min(fine_step, exposure_s, floor_sps=speed_lo)
    hill_v_min_steps_s = int(min(hill_v_min_steps_s,
                                 max(hill_v_cap, speed_lo)))
    backlash_steps = um_to_steps(backlash_um, um_per_step) if backlash_um > 0 else 0
    overshoot_margin_steps = um_to_steps(
        float(af_cfg.get("overshoot_margin_um", 3.0)), um_per_step)
    # The fine sweep window must never collapse at high magnification
    # (coarse_step == fine_step == 1), AND it must cover the coarse
    # localization uncertainty: the coarse sampling spacing (speed ÷ camera
    # fps) bounds the peak error, and the low-frequency metric's broad
    # plateau widens it further. Hardware-verified at 5×: at 500 sps
    # (~33 steps/frame) the coarse peak missed the plane by up to ~2
    # spacings and the hill pass ran straight to the window edge.
    fps = float(af_cfg.get("camera_fps_estimate", 15.0))
    spacing_steps = (eff_coarse_um_s / um_per_step) / fps if fps > 0 else 0.0
    fine_window_steps = max(coarse_step, 3 * fine_step,
                            int(3.0 * spacing_steps))

    # v3 derivative thresholds (audit L1): per-objective defaults from
    # DOF (σ ≈ DOF/3) and the coarse sampling spacing (h = speed/fps);
    # per-row and per-config overrides win (0 = auto). The audit's
    # ladder values (5× ≈ 0.25, 10× ≈ 0.10, 20× ≈ 0.06) fall out of
    # coarse_curv_stop_ladder at the hardware t = h/σ; everything is
    # re-tuned on hardware after the bench σ measurement. The bench's
    # Step-0 measured σ goes into af_cfg["sigma_steps"] — the measured
    # value REPLACES the DOF model (hardware: 107.8 steps at 5× vs the
    # DOF/3 estimate of 46.7 — the model underestimates by 2.3×).
    sigma_steps = max(dof_um / 3.0 / um_per_step, float(fine_step)) \
        if dof_um > 0 else float(span_steps) / 10.0
    measured_sigma = af_cfg.get("sigma_steps")
    if measured_sigma:
        sigma_steps = float(measured_sigma)
    sigma_steps = max(sigma_steps, 1.0)
    h_steps = max(spacing_steps, 1.0)
    t = h_steps / sigma_steps

    def _curv_override(key: str, auto: float) -> float:
        for source in (objective_row, af_cfg):
            value = source.get(key)
            if value:
                return float(value)
        return auto

    coarse_curv_stop = _curv_override(
        "coarse_curv_stop", coarse_curv_stop_ladder(t))
    coarse_curv_vertex = _curv_override(
        "coarse_curv_vertex", sg_curvature_at(0.7, t))
    probe_curv_in = _curv_override(
        "probe_curv_in", 0.3 / sigma_steps ** 2)
    probe_curv_out = _curv_override(
        "probe_curv_out", 0.1 / sigma_steps ** 2)

    # The near-entry stage-2 window: the probe's near verdict tolerates
    # the arm up to ~±0.7σ off the true peak (the weaker-side test's
    # zone) — the hill window must cover that, and the fit-distrust
    # tolerance (fw//2) scales with it. 1.5σ covers the zone + the
    # broad-peak plateau's sub-structure.
    near_window_steps = _curv_override(
        "near_window_steps", max(fine_window_steps,
                                 int(1.5 * sigma_steps)))

    # Sanity checks (warnings only — the operator tunes on hardware).
    if dof_um > 0:
        if coarse_step_um > dof_um:
            warnings.append(f"coarse_step {coarse_step_um} µm exceeds DOF {dof_um} µm "
                            f"— peak may be skipped")
        if fine_step_um > dof_um / 4:
            warnings.append(f"fine_step {fine_step_um} µm exceeds DOF/4 "
                            f"({dof_um / 4:.2f} µm)")
        blur_um = eff_coarse_um_s * exposure_s
        if blur_um > dof_um / 4:
            warnings.append(f"coarse motion blur {blur_um:.2f} µm exceeds "
                            f"DOF/4 ({dof_um / 4:.2f} µm) — lower speed_multiplier "
                            f"or af_exposure_us")
    # Sampling density: at ~1 sample per coarse_step the peak localization
    # guarantee breaks (hardware-verified at 5× — 150 µm/s gave sparse
    # curves and false edge/weak-peak rejects). The camera's effective
    # scored-frame rate (fps minus the newest-wins loop skips) bounds the
    # usable speed below the blur budget.
    fps = float(af_cfg.get("camera_fps_estimate", 15.0))
    if fps > 0 and eff_coarse_um_s > coarse_step_um * fps:
        warnings.append(f"coarse speed {eff_coarse_um_s:.1f} µm/s exceeds the "
                        f"sampling-density bound {coarse_step_um * fps:.1f} µm/s "
                        f"(coarse_step {coarse_step_um} µm × ~{fps} fps) — peak "
                        f"localization may fail")

    return {
        "coarse_step": coarse_step,
        "fine_step": fine_step,
        "span_steps": span_steps,
        "window_plus_steps": window_plus_steps,
        "window_minus_steps": window_minus_steps,
        "max_speed": max_speed,
        "fine_speed": fine_speed,
        "landing_speed": landing_speed,
        "backlash_steps": backlash_steps,
        "overshoot_margin_steps": overshoot_margin_steps,
        "fine_window_steps": fine_window_steps,
        "coarse_speed": coarse_speed,
        "hill_v_cap": hill_v_cap,
        "hill_v_min": hill_v_min_steps_s,
        "probe_curv_in": probe_curv_in,
        "probe_curv_out": probe_curv_out,
        "coarse_curv_stop": coarse_curv_stop,
        "coarse_curv_vertex": coarse_curv_vertex,
        "near_window_steps": near_window_steps,
    }, warnings


# ---------------------------------------------------------------------------
# Probe v3 (derivative-hybrid) — the audit-verified Gaussian model
#
# Sharpness vs defocus is near-Gaussian (S = A·exp(−d²/2σ²), σ ≈ DOF/3):
# its second derivative is strongly negative near the peak, positive past
# the inflection — the signal the v3 probe and coarse stop exploit. All
# quantities here are pure math on score samples; the numbers pinned by
# the unit tests are the audit-verified table (plan
# project-talos-cheerful-stearns.md §1).
# ---------------------------------------------------------------------------

@dataclass
class ProbeVerdict:
    near: bool
    branch: str              # near-curvature | near-plateau | flank | slope | blind
    direction: int | None    # +1 / -1 / None (blind)
    curvature: float | None  # the raw C (logging/tests)


def curvature_from_three(s_lo: float, s_c: float, s_hi: float,
                         delta: float) -> float | None:
    """Normalized 3-point probe curvature
    C = (S₋δ − 2·S_c + S₊δ)/(δ²·S_c).

    On a Gaussian curve C(0) ≈ −0.64/σ² (δ=1.4σ), zero crossing at
    d* = arcosh(e^(δ²/2σ²))·σ²/δ (≈ 1.1–1.2σ for δ/σ in 1.1–1.6),
    positive beyond. LINEAR BACKGROUNDS CANCEL EXACTLY in C (the
    (1, −2, 1) kernel has zero sum and zero first moment) — unlike the
    fitted slope, which carries the background 1:1. Beyond ~1.5σ the
    values are noise-dominated (the S_c normalization diverges in the
    tail). None when S_c ≤ 0 (degenerate)."""
    if s_c <= 0:
        return None
    return (s_lo - 2.0 * s_c + s_hi) / (delta * delta * s_c)


def fit_slope_with_se(xs, ys) -> tuple[float, float] | None:
    """Least-squares line y = b·x + c. Returns (b, SE(b)) with SE from
    the residual variance (N−2 DOF). None when the samples are
    degenerate (< 2 distinct x). NOTE: for a 3-parameter PARABOLA the
    residual DOF is N−3 — a z=2 test on it is only a 9.2% one-sided
    t-test; the v3 guard uses the pooled stationary σₙ instead and
    treats this SE as a sanity check (audit A10)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2 or np.ptp(xs) <= 0:
        return None
    x_bar = float(xs.mean())
    sxx = float(((xs - x_bar) ** 2).sum())
    if sxx <= 0:
        return None
    b = float(((xs - x_bar) * ys).sum() / sxx)
    c = float(ys.mean() - b * x_bar)
    resid = ys - (b * xs + c)
    dof = xs.size - 2
    se = float(np.sqrt((resid ** 2).sum() / dof / sxx)) if dof > 0 \
        else float("inf")
    return b, se


def window_parabola(xs, ys) -> tuple[float, float] | None:
    """Least-squares parabola fitted on the CENTERED axis:
    y = a·(x−x̄)² + b·(x−x̄) + c. Returns (a, b) in RAW units — the
    local curvature and slope at the window center (the vertex is
    x̄ − b/(2a)). On 5 equally spaced samples 2a equals the audit's
    Savitzky-Golay curvature kernel (2, −1, −2, −1, 2)/7h² and b its
    slope kernel (−2, −1, 0, 1, 2)/10h — the unit tests pin the audit's
    exact table. None when the window is degenerate (< 3 distinct x)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 3 or np.ptp(xs) <= 0:
        return None
    u = xs - float(xs.mean())
    matrix = np.column_stack([u * u, u, np.ones_like(xs)])
    coefs, *_ = np.linalg.lstsq(matrix, ys, rcond=None)
    return float(coefs[0]), float(coefs[1])


def window_curvature(xs, ys, fit: tuple[float, float] | None = None) \
        -> tuple[float, float] | None:
    """The coarse curvature-stop metric over a window: (curv, b) with
    curv = 2a·h²/S_max — the dimensionless normalized curvature (h = the
    window's median spacing, S_max = the window's max score; scale-free
    in both axes, so thresholds transfer across objectives) and b = the
    fitted slope in raw score/step units (the sweep-direction gate's
    sign). On a Gaussian this is ≈ −0.19 at the peak, −0.16@0.4σ,
    −0.07@0.8σ (h = 0.5σ) and crosses zero ≈ 1.1σ out; the raw audit
    table (−0.62@0.4σ, −0.28@0.8σ, zero@1.10σ) is the same kernel at
    h = 0.5σ expressed as 2a·σ². None when degenerate. ``fit`` lets a
    caller that already fitted this window pass the result in — the v3
    coarse rung fits the same window for its b gate, and refitting here
    duplicated an lstsq on every sample of the live sweep."""
    if fit is None:
        fit = window_parabola(xs, ys)
    if fit is None:
        return None
    a, b = fit
    s_max = float(np.asarray(ys, dtype=float).max())
    if s_max <= 0:
        return None
    gaps = np.diff(np.sort(np.asarray(xs, dtype=float)))
    h = float(np.median(gaps)) if gaps.size else 0.0
    if h <= 0:
        return None
    return 2.0 * a * h * h / s_max, b


def sg_curvature_at(d_sigma: float, t: float) -> float:
    """The normalized window curvature ON a Gaussian S = A·exp(−d²/2σ²)
    at distance d_sigma·σ with sampling spacing t = h/σ — the audit's
    table in closed form, divided by the WINDOW'S MAX score (the
    criterion's S_max denominator — the window max, not the center, is
    what the controller sees). Used to derive the per-objective
    trusted-vertex floor (the curvature value AT 0.7σ — the vertex
    estimate is only valid within ≤ 0.7σ; beyond ~0.9σ the window
    parabola opens upward and there is no vertex at all)."""
    kernel = (2.0, -1.0, -2.0, -1.0, 2.0)
    xs = [d_sigma + t * i for i in (-2, -1, 0, 1, 2)]
    ys = [math.exp(-x * x / 2.0) for x in xs]
    numer = sum(k * y for k, y in zip(kernel, ys))
    return numer / (7.0 * max(ys))


def coarse_curv_stop_ladder(t: float) -> float:
    """The per-objective coarse curvature-stop threshold from the
    audit's ladder (5× ≈ 0.25, 10× ≈ 0.10, 20× ≈ 0.06) with the t = h/σ
    scaling extended to the dense-sampling regime: at t ≤ 0.7 the metric
    magnitudes shrink ∝ t², so the threshold shrinks with them (keeps the
    fire distance ~constant); at t > 0.7 the audit's decreasing ladder
    applies (a wide window contains the peak until far out, and the
    sweep-aware b-gate — not the curvature — is the load-bearing
    discriminator). Hardware-tuned after the bench σ measurement."""
    if t <= 0.7:
        return 0.25 * (t / 0.7) ** 2
    return max(0.06, min(0.25, 0.25 * (0.7 / t) ** 2))


def pooled_noise_from_probe(low_scores, positions=None,
                            floor_frac: float = 0.01) -> float | None:
    """Stationary noise σₙ pooled over the probe's three low-frequency
    points: the residual std of a LINEAR DETREND (the hill's slope must
    not pollute the noise estimate; N−2 DOF). Positions default to the
    equally spaced [−1, 0, 1]. Floored at floor_frac × the mean score —
    three points exactly on a line are not evidence of zero noise (the
    known stationary level is ~2–5%)."""
    ys = [float(y) for y in low_scores]
    if len(ys) < 3:
        return None
    if positions is None:
        xs = [-1.0, 0.0, 1.0][:len(ys)]
    else:
        xs = [float(x) for x in positions][:len(ys)]
    fit = fit_slope_with_se(xs, ys)
    if fit is None:
        return None
    b, _se = fit
    # residuals around the full line (intercept included): centered form
    # avoids needing the intercept back out of the fit
    x_bar = sum(xs) / len(xs)
    y_bar = sum(ys) / len(ys)
    resid = [(y - y_bar) - b * (x - x_bar) for x, y in zip(xs, ys)]
    dof = max(len(ys) - 2, 1)
    sigma = math.sqrt(sum(r * r for r in resid) / dof)
    floor = floor_frac * max(abs(float(np.mean(ys))), 1e-9)
    return max(sigma, floor)


def guard_reversal_test(xs, ys, sigma_noise: float,
                        z: float = 2.0) -> bool:
    """The v3 2σ cascade-guard trigger: fit the pass's low-freq series
    vs POSITION (sequence-index fits are meaningless on non-monotonic
    samples — audit #6) and reverse when the fitted slope's zσ
    confidence interval lies entirely below zero:
    b̂ + z·σₙ/√Sxx < 0, with σₙ the POOLED STATIONARY noise from the
    probe points (audit A10 — the residual SE of the fit itself has
    N−3 = 2 DOF, so z=2 on it is a 9.2% one-sided test and is used as
    a sanity check only). The sharp-rise gate is applied by the caller
    (audit #7: this trigger must never swallow the start-on-peak
    walk-away case)."""
    fit = fit_slope_with_se(xs, ys)
    if fit is None or sigma_noise <= 0:
        return False
    b, _residual_se = fit
    xs_arr = np.asarray(xs, dtype=float)
    sxx = float(((xs_arr - xs_arr.mean()) ** 2).sum())
    if sxx <= 0:
        return False
    return b + z * sigma_noise / math.sqrt(sxx) < 0


def probe_classify(sharp_lo: float, sharp_c: float, sharp_hi: float,
                   delta: float, low_lo: float, low_c: float, low_hi: float,
                   floor: float, curv_in: float, curv_out: float,
                   peak_ratio: float = 0.15, min_slope: float = 0.10,
                   cluster_center_ratio: float = 0.5,
                   cluster_side_ratio: float = 0.4,
                   cluster_spread_ratio: float = 1.2,
                   cluster_min_score: float = 0.0) -> ProbeVerdict:
    """Hybrid v3 probe classification (audit-revised):

        floor      = S_c ≥ probe_score_floor (RELATIVE — × the
                     far-defocus baseline measured in preflight)
        center_max = S_c > max(S_lo, S_hi)   # audit #8/#9: the VALLEY
                     gate — without it the between-planes valley reads
                     "curvature-near" and the mount wins
        near       = floor and center_max and
                     (C < −curv_in        # curvature branch: a true
                                          #   Gaussian peak (works where
                                          #   the weaker-side test fails,
                                          #   e.g. small δ/σ)
                      or weaker-side test)  # plateau branch: the v2 test
                     (C ≈ 0 on flat tops)
        near-cluster = floor and (NOT center_max) and the three points
                     are ALL HIGH and genuinely SPREAD — S_c ≥
                     cluster_center_ratio × the best side AND the worst
                     side ≥ cluster_side_ratio × S_c AND the best side ≥
                     cluster_spread_ratio × the worst AND the best point
                     clears cluster_min_score ABSOLUTELY. The arm sits
                     on a multi-peak plateau's shoulder: a neighbor
                     sub-peak within δ beats the center. Excluded: the
                     valley (the center is a small fraction of its
                     sides — the center ratio), the far-away FLAT region
                     (a noise triple has ≤5% spread — the spread gate;
                     and the far tail's points are mutually comparable
                     yet ABSOLUTELY LOW — ratio gates cannot tell a flat
                     tail from a plateau shoulder, only the score scale
                     can: cluster_min_score, 0 = the branch disabled;
                     a per-field knob the user tunes to their peak's
                     magnitude). Stage 2 enters directly, anchored at
                     the BEST probe point.
        flank      = floor and (not near, not cluster) and C > +curv_out
                     — past the inflection; the direction call is the
                     low-freq slope's job either way
        else       → direction = low-freq slope; None → blind

    The "curvature-wins-where-weaker-side-fails" disagreement branch is
    deliberately DROPPED (audit #9: it loses on the mount-within-δ
    hardware geometry — that case falls through to the v2 directed
    coarse + nearest-center discipline)."""
    direction = probe_direction(low_lo, low_c, low_hi, min_slope)
    curv = curvature_from_three(sharp_lo, sharp_c, sharp_hi, delta)
    if sharp_c < floor:
        branch = "blind" if direction is None else "slope"
        return ProbeVerdict(False, branch, direction, curv)
    center_max = sharp_c > max(sharp_lo, sharp_hi)
    weaker = probe_near_focus(sharp_c, [sharp_lo, sharp_hi], peak_ratio)
    # a zero threshold DISABLES the branch (it must never degrade to
    # "any negative curvature" — the audit's hybrid-OR safety rule)
    curv_near = curv_in > 0 and curv is not None and curv < -curv_in
    if center_max and (curv_near or weaker):
        branch = "near-curvature" if curv_near else "near-plateau"
        return ProbeVerdict(True, branch, None, curv)
    cluster = (cluster_min_score > 0  # 0 disables the branch entirely
               and not center_max
               and cluster_center_ratio > 0 and cluster_side_ratio > 0
               and sharp_c >= cluster_center_ratio * max(sharp_lo, sharp_hi)
               and min(sharp_lo, sharp_hi) >= cluster_side_ratio * sharp_c
               and max(sharp_lo, sharp_hi)
               >= cluster_spread_ratio * min(sharp_lo, sharp_hi)
               and max(sharp_lo, sharp_hi) >= cluster_min_score)
    if cluster:
        return ProbeVerdict(True, "near-cluster", None, curv)
    if curv_out > 0 and curv is not None and curv > curv_out:
        return ProbeVerdict(False, "flank", direction, curv)
    branch = "blind" if direction is None else "slope"
    return ProbeVerdict(False, branch, direction, curv)
