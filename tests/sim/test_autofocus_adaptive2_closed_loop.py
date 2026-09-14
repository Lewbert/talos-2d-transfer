"""Adaptive v2 closed-loop tests: the probe-driven dispatcher — probe
near-focus direct entry, direction-guided coarse with the first-peak
stop, the direction-guard reversal, boundary-hit salvage (paths A/B),
AF_REFINE near/far, and the shared failure disciplines. The V1 suite
(tests/sim/test_autofocus_adaptive_closed_loop.py) stays the frozen
baseline's preservation proof."""
from __future__ import annotations

import threading
import time

import pytest
from PySide6.QtWidgets import QApplication

from talos.cv.af_adaptive import (
    DIRECTION_GUARD,
    PHASE_COARSE_PASS,
    AdaptiveAutofocusControllerV2,
)
from talos.cv.autofocus import AutofocusConfig, _AfExit
from talos.cv.frame_slot import LatestFrameSlot
from tests.testing.sim_images import (
    split_plane_source,
    synthetic_flake_image,
    two_plane_source,
)
from talos.hal.sim import SimFocusStage

from tests.sim.test_autofocus_adaptive_closed_loop import ADAPTIVE_CFG
from tests.sim.test_autofocus_closed_loop import (

    FrameProducer,
    make_rig,
    retry_once,
    running,
)

# Real-time closed-loop simulation: the whole file is marked slow
# (deselect with `-m "not slow"` for a fast edit loop).
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


ADAPTIVE2_CFG = {**ADAPTIVE_CFG,
                 "probe_step_steps": 0,       # → 3×coarse_step = 120
                 "probe_peak_ratio": 0.15,
                 "probe_min_slope": 0.10,
                 "guard_samples": 4,
                 "guard_drop_ratio": 0.15,
                 "coarse_early_stop_samples": 2}


def run_adaptive2(focus, slot, center, **cfg_kwargs):
    cfg = AutofocusConfig(**{**ADAPTIVE2_CFG, **cfg_kwargs})
    ctrl = AdaptiveAutofocusControllerV2(focus, slot)
    return ctrl.run(center=center, cfg=cfg)


def prime(ctrl, cfg, center):
    """Direct-stage-call harness: hand the controller the state run()
    establishes (probe-free — the stage under test sees a clean axis)."""
    ctrl._cfg = cfg
    ctrl._t_start = time.monotonic()
    ctrl._deadline = ctrl._t_start + cfg.timeout_s
    ctrl._current_pos = int(center)
    ctrl._arm_center = int(center)
    ctrl._low_curve = []
    ctrl._curve = []
    ctrl._stage1_curve = []
    ctrl._probe_samples = []
    ctrl.abort_requested = False


def make_cfg(**kwargs):
    return AutofocusConfig(**{**ADAPTIVE2_CFG, **kwargs})


# ---------------------------------------------------------------------------
# Probe-driven dispatch
# ---------------------------------------------------------------------------

def test_probe_near_focus_enters_stage2_directly(qapp):
    """Armed at the peak: the probe detects near-focus and the run skips
    the coarse pass entirely (no phase-4 progress), landing via the hill
    climb + lock-on."""
    def case():
        focus, slot, producer = make_rig(truth=0.0)
        with running(producer):
            cfg = make_cfg()
            ctrl = AdaptiveAutofocusControllerV2(focus, slot)
            phases: list[int] = []
            ctrl.sig_progress.connect(
                lambda f, p, s, pos: phases.append(p))
            result = ctrl.run(center=0, cfg=cfg)
        assert result.success, result.message
        assert abs(result.best_position - 0.0) <= ADAPTIVE2_CFG["fine_step"]
        assert PHASE_COARSE_PASS not in phases
        assert 7 in phases  # the probe ran
    retry_once(case)


def test_probe_far_direction_guided_sweep(qapp):
    """Truth 120 steps away: the probe estimates the direction and the
    coarse pass sweeps only the rising side — the far side is never
    walked (the probe points at −δ are the leftmost curve data)."""
    def case():
        focus, slot, producer = make_rig(truth=120.0)
        with running(producer):
            result = run_adaptive2(focus, slot, center=0.0)
        assert result.success, result.message
        assert abs(result.best_position - 120.0) <= ADAPTIVE2_CFG["fine_step"]
        xs = [p for p, _ in result.curve]
        assert min(xs) >= -(120 + 10), f"swept the wrong side: min {min(xs)}"
    retry_once(case)


def test_probe_blind_fallback_sweeps_both_sides(qapp):
    """Far truth (no near-focus, no direction info with a huge min_slope)
    → the blind V1 sweep (nearest edge, full window) runs and recovers
    the peak from across the window."""
    def case():
        focus, slot, producer = make_rig(truth=600.0)
        with running(producer):
            result = run_adaptive2(focus, slot, center=0.0,
                                   span_steps=1600, probe_min_slope=10.0)
        assert result.success, result.message
        assert abs(result.best_position - 600.0) <= ADAPTIVE2_CFG["fine_step"]
        xs = [p for p, _ in result.curve]
        assert min(xs) < -150, f"blind sweep never crossed the window: {min(xs)}"
    retry_once(case)


# ---------------------------------------------------------------------------
# Direction guard (direct stage call — deterministic)
# ---------------------------------------------------------------------------

def test_direction_guard_reverses_the_coarse_pass(qapp):
    """Wrong-direction call on purpose (truth at −120, sweep +1): the
    guard fires inside the first samples and the reversed pass recovers
    the peak on the other side."""
    def case():
        focus, slot, producer = make_rig(truth=-120.0)
        with running(producer):
            cfg = make_cfg()
            ctrl = AdaptiveAutofocusControllerV2(focus, slot)
            prime(ctrl, cfg, 0)
            logs: list[str] = []
            ctrl.sig_log.connect(logs.append)
            peak = ctrl._coarse_pass(0, (-200, 200), cfg, direction=+1)
        assert abs(peak.pos - (-120.0)) <= 40, f"reversed pass missed: {peak.pos}"
        assert any("direction guard" in m for m in logs)
        assert focus.get_status().is_idle
    retry_once(case)


def test_first_peak_stop_never_walks_the_window(qapp):
    """Two-plane scene with the peak AT the pass start (mount plane):
    the pass starts ON the peak, walks away, and the start-on-peak
    walk-away guard stops it shortly after — the far window edge is
    never walked. The probe seeds (as the real flow leaves them) give
    the peak interior curve extent so it is not an edge peak."""
    source = two_plane_source(shape=(240, 320), seed=7, z_top=0.0,
                              z_bottom=150.0, top_weight=0.4)

    def builder(focus):
        return lambda: source(focus.load_pos)

    def case():
        focus, slot, producer = make_rig(source_builder=builder)
        with running(producer):
            cfg = make_cfg()
            ctrl = AdaptiveAutofocusControllerV2(focus, slot)
            prime(ctrl, cfg, 150)
            # the probe's sharp samples as the real flow leaves them:
            # mount at 150 sharp, wafer side weak, far side blurred. The
            # −100 seed sits BELOW any pass endpoint so the baseline check
            # (curve ends) stays pinned regardless of pass timing.
            ctrl._probe_samples = [(150.0, 900.0, 80.0),
                                   (30.0, 40.0, 60.0),
                                   (270.0, 30.0, 40.0),
                                   (-100.0, 30.0, 20.0)]
            logs: list[str] = []
            ctrl.sig_log.connect(logs.append)
            peak = ctrl._coarse_pass(150, (-50, 350), cfg, direction=-1)
        assert abs(peak.pos - 150.0) <= 2.0 * ADAPTIVE2_CFG["fine_step"]
        # either early-stop mechanism may win: the direction guard
        # reverses at the first samples (len ≤ guard_samples) or the
        # start-on-peak walk-away stop fires at len ≥ 5 — both halt the
        # pass shortly after the mount peak
        assert any("direction guard" in m or "walked away" in m for m in logs)
        xs = [p for p, _ in ctrl._stage1_curve]
        # the pass halts shortly after the mount peak (150) — the far
        # window edge (350) is never walked
        assert max(xs) < 300, f"pass ran to the far edge: {max(xs)}"
    retry_once(case)


# ---------------------------------------------------------------------------
# Boundary-hit salvage
# ---------------------------------------------------------------------------

def test_boundary_salvage_path_b(qapp):
    """The v1-failing case: the true peak sits just inside the search
    bound (fine window clipped AT the bound). The hill pass climbs over
    the peak and stops near the edge — v1 fails with the hit-window
    message; v2 accepts the complete curve and lands on the peak."""
    def case():
        focus, slot, producer = make_rig(truth=268.0)
        with running(producer):
            result = run_adaptive2(focus, slot, center=180,
                                   span_steps=200, fine_window_steps=80)
        assert result.success, result.message
        assert abs(result.best_position - 268.0) <= ADAPTIVE2_CFG["fine_step"]
    retry_once(case)


def test_hill_pass_salvages_complete_curve_on_budget_cut(qapp):
    """Path A: the hill pass is cut by a budget reason AFTER crossing the
    peak — the collected curve is salvaged instead of discarded."""
    def case():
        focus, slot, producer = make_rig(truth=0.0)
        with running(producer):
            cfg = make_cfg(fine_window_steps=40)
            ctrl = AdaptiveAutofocusControllerV2(focus, slot)
            prime(ctrl, cfg, 0)
            ctrl._current_pos = 40  # hill starts at fine_end=40, sweeps −
            logs: list[str] = []
            ctrl.sig_log.connect(logs.append)

            def check(phase_dl=None):
                if focus.get_status().pos < -10:  # past the peak at 0
                    return "phase budget exceeded"
                return None

            ctrl._check = check
            hill_curve = ctrl._hill_pass(0.0, (-200, 200), cfg)
        assert hill_curve, "salvage must keep the curve"
        assert any("cut short" in m for m in logs)
        peak_x = max((p for p, _ in hill_curve),
                     key=lambda p: dict(hill_curve)[p])
        assert abs(peak_x - 0.0) <= ADAPTIVE2_CFG["fine_step"]
    retry_once(case)


def test_hill_pass_no_salvage_before_the_peak(qapp):
    """The budget cut fires BEFORE the peak is crossed — no complete peak
    in the data, the fail path runs exactly as in v1."""
    def case():
        focus, slot, producer = make_rig(truth=0.0)
        with running(producer):
            cfg = make_cfg(fine_window_steps=40)
            ctrl = AdaptiveAutofocusControllerV2(focus, slot)
            prime(ctrl, cfg, 0)
            ctrl._current_pos = 40

            def check(phase_dl=None):
                if focus.get_status().pos < 35:  # immediately
                    return "phase budget exceeded"
                return None

            ctrl._check = check
            with pytest.raises(_AfExit):
                ctrl._hill_pass(0.0, (-200, 200), cfg)
    retry_once(case)


def test_stop_stage_restore_updates_current_pos(qapp):
    """Regression (user-found): _stop_stage restores the axis to the arm
    position but left _current_pos at the failed pass's end — the stage-2
    retry then planned its coarse pass from the stale position while the
    axis sat at the arm, sweeping far away. The restore must track the
    tracked position."""
    focus, slot, producer = make_rig(truth=0.0)
    with running(producer):
        cfg = make_cfg()
        ctrl = AdaptiveAutofocusControllerV2(focus, slot)
        prime(ctrl, cfg, 100)
        ctrl._current_pos = 37  # stale: a pass that ended far away
        ctrl._stop_stage("phase budget exceeded", "fine")
    assert ctrl._current_pos == 100
    assert focus.get_status().pos == 100
    assert focus.get_status().is_idle


def test_probe_measures_center_first_and_returns_to_it(qapp):
    """The probe samples [center, center−δ, center+δ] (the arm position
    first — the near-focus decision is anchored on the arm's own score)
    and RETURNS to center: the directed coarse pass must start there, so
    a peak between the probe points is never skipped and the pass crosses
    the peak mid-way. (A monotonic probe ordering broke the near-focus
    logic when armed at the peak and saved no time — user-reverted.)"""
    focus, slot, producer = make_rig(truth=0.0)
    with running(producer):
        cfg = make_cfg()
        ctrl = AdaptiveAutofocusControllerV2(focus, slot)
        moves: list[tuple[int, int]] = []
        orig_move = ctrl._move_to

        def recording_move(pos, speed):
            moves.append((int(pos), int(speed)))
            orig_move(pos, speed)

        ctrl._move_to = recording_move
        result = ctrl.run(center=0, cfg=cfg)
    assert result.success, result.message
    delta = 3 * cfg.coarse_step
    assert moves[:4] == [(0, cfg.stage_speed),
                         (-delta, cfg.stage_speed),
                         (delta, cfg.stage_speed),
                         (0, cfg.stage_speed)], \
        f"probe move sequence wrong: {moves[:5]}"


def test_stage2_interior_edge_widens_the_fine_window(qapp):
    """The coarse anchor misses the true peak (sampling-error
    simulation): the hill peak sits at an INTERIOR fine-window edge, the
    stage-2 loop widens the window (x2, x4) and re-runs the hill —
    the run recovers the truth without a stage-1 fallback."""
    from types import SimpleNamespace

    def case():
        focus, slot, producer = make_rig(truth=0.0)
        with running(producer):
            cfg = make_cfg(fine_window_steps=40, stage2_retries=1,
                           probe_peak_ratio=10.0)  # force the far path
            ctrl = AdaptiveAutofocusControllerV2(focus, slot)
            logs: list[str] = []
            ctrl.sig_log.connect(logs.append)
            orig_coarse = ctrl._coarse_pass

            def offset_coarse(center, bounds, cfg, direction=None):
                # fake the sampling-limited localization: the anchor
                # lands 100 steps past the planted truth (whichever way
                # the probe would have gone)
                ctrl._stage1_curve = [(float(center + 100), 500.0),
                                      (float(center + 110), 400.0),
                                      (float(center + 90), 400.0)]
                return SimpleNamespace(pos=float(center + 100))

            ctrl._coarse_pass = offset_coarse
            result = ctrl.run(center=40, cfg=cfg)
        assert result.success, result.message
        assert abs(result.best_position - 0.0) <= ADAPTIVE2_CFG["fine_step"]
        assert any("widening the fine window" in m for m in logs)
    retry_once(case)


def test_stage2_retry_succeeds_around_the_current_position(qapp):
    """A failed first hill pass retries stage 2 around the CURRENT
    position (no arm restore, no stage-1 fallback, no far-away sweep) —
    the second attempt recovers the planted truth."""
    def case():
        focus, slot, producer = make_rig(truth=-120.0)
        with running(producer):
            cfg = make_cfg()
            ctrl = AdaptiveAutofocusControllerV2(focus, slot)
            positions: list[float] = []
            ctrl.sig_progress.connect(
                lambda f, p, s, pos: positions.append(float(pos)))
            orig_hill = ctrl._hill_pass
            calls: list[float] = []

            def flaky_hill(anchor, bounds, cfg, fw_scale=1.0):
                calls.append(anchor)
                if len(calls) == 1:
                    raise _AfExit(ctrl._stop_stage(
                        "phase budget exceeded", "fine", restore=False))
                return orig_hill(anchor, bounds, cfg, fw_scale=fw_scale)

            ctrl._hill_pass = flaky_hill
            logs: list[str] = []
            ctrl.sig_log.connect(logs.append)
            result = ctrl.run(center=0, cfg=cfg)
        assert result.success, result.message
        assert abs(result.best_position - (-120.0)) <= ADAPTIVE2_CFG["fine_step"]
        assert any("retrying the hill climb (same anchor)" in m for m in logs)
        # never a full-window excursion: every scored position stays
        # inside the search window ± allowance
        span = cfg.span_steps
        assert all(abs(p) <= span / 2 + 100 for p in positions), \
            f"run wandered outside the window: {positions}"
    retry_once(case)


def test_stage2_retries_exhausted_stops_at_current_position(qapp):
    """Stage 2 keeps failing: after stage2_retries attempts the run
    stops AT the current position — no arm restore, no stage-1
    fallback."""
    def case():
        focus, slot, producer = make_rig(truth=-120.0)
        with running(producer):
            cfg = make_cfg(stage2_retries=2)
            ctrl = AdaptiveAutofocusControllerV2(focus, slot)
            logs: list[str] = []
            ctrl.sig_log.connect(logs.append)

            def always_fail(anchor, bounds, cfg, fw_scale=1.0):
                raise _AfExit(ctrl._stop_stage(
                    "phase budget exceeded", "fine", restore=False))

            ctrl._hill_pass = always_fail
            result = ctrl.run(center=0, cfg=cfg)
        assert not result.success
        assert "stage 2 failed after 3 attempt(s)" in result.message
        assert result.phase == "fine"
        # the axis stays where the last attempt stopped — NOT the arm
        assert focus.get_status().is_idle
        assert focus.get_status().pos != 0
        assert result.best_position == ctrl._current_pos
        assert any("retrying the hill climb (same anchor)" in m for m in logs)
    retry_once(case)


# ---------------------------------------------------------------------------
# AF_REFINE (AF-C re-peaks run the same probe-driven pipeline)
# ---------------------------------------------------------------------------

def test_refine_near_focus_stays_in_the_refine_window(qapp):
    """Small drift: the probe says near-focus → stage 2 directly; the
    coarse pass never runs (no phase-4 progress) and the curve stays
    inside the probe + refine-window extent. (δ = 100 keeps both probe
    sides farther from the truth than the arm position — near-focus
    needs the center to be the local max, i.e. δ > 2×drift.)"""
    def case():
        focus, slot, producer = make_rig(truth=0.0)
        with running(producer):
            cfg = make_cfg(mode="AF_REFINE", fine_window_steps=60,
                           probe_step_steps=100)
            ctrl = AdaptiveAutofocusControllerV2(focus, slot)
            phases: list[int] = []
            ctrl.sig_progress.connect(
                lambda f, p, s, pos: phases.append(p))
            result = ctrl.run(center=40, cfg=cfg)
        assert result.success, result.message
        assert abs(result.best_position - 0.0) <= ADAPTIVE2_CFG["fine_step"]
        assert PHASE_COARSE_PASS not in phases
        span = max(p for p, _ in result.curve) - min(p for p, _ in result.curve)
        assert span <= 2 * 60 + 2 * 100 + 20, f"refine curve span {span} too wide"
    retry_once(case)


def test_refine_far_from_focus_runs_the_full_pipeline(qapp):
    """Heavy drift after a pause: the probe says far → full stage 1→3."""
    def case():
        focus, slot, producer = make_rig(truth=0.0)
        with running(producer):
            result = run_adaptive2(focus, slot, center=300, mode="AF_REFINE",
                                   span_steps=800, fine_window_steps=60)
        assert result.success, result.message
        assert abs(result.best_position - 0.0) <= ADAPTIVE2_CFG["fine_step"]
    retry_once(case)


# ---------------------------------------------------------------------------
# Shared disciplines
# ---------------------------------------------------------------------------

def test_adaptive2_abort_mid_probe_stops_stage(qapp):
    focus, slot, producer = make_rig(truth=0.0)
    with running(producer):
        cfg = make_cfg()
        ctrl = AdaptiveAutofocusControllerV2(focus, slot)
        result_box: dict = {}

        def worker():
            result_box["result"] = ctrl.run(center=0, cfg=cfg)

        thread = threading.Thread(target=worker)
        thread.start()
        time.sleep(0.5)  # mid-probe (the probe takes a few seconds)
        ctrl.request_abort()
        thread.join(timeout=30.0)
        result = result_box["result"]
    assert result.aborted, result.message
    assert focus.get_status().is_idle


def test_adaptive2_two_plane_nearest_center(qapp):
    """The probe near-focus decision + nearest-center peak preference:
    armed at either plane, the run lands on THAT plane."""
    source = two_plane_source(shape=(240, 320), seed=7, z_top=0.0,
                              z_bottom=150.0, top_weight=0.4)

    def builder(focus):
        return lambda: source(focus.load_pos)

    def case():
        focus, slot, producer = make_rig(source_builder=builder)
        with running(producer):
            result = run_adaptive2(focus, slot, center=0.0)
        assert result.success, result.message
        assert abs(result.best_position - 0.0) <= 1.5 * ADAPTIVE2_CFG["fine_step"]

        focus2, slot2, producer2 = make_rig(source_builder=builder)
        with running(producer2):
            result2 = run_adaptive2(focus2, slot2, center=150.0)
        assert result2.success
        assert abs(result2.best_position - 150.0) <= 2.5 * ADAPTIVE2_CFG["fine_step"]
    retry_once(case)


def test_adaptive2_roi_selects_the_plane(qapp):
    """Same ROI contract: a left-half ROI pulls the metric (and the
    probe) to the wafer plane while armed at the mount."""
    source = split_plane_source(shape=(240, 320), seed=7, z_top=0.0,
                                z_bottom=150.0, top_flakes=4, bottom_flakes=14)

    def builder(focus):
        return lambda: source(focus.load_pos)

    def case():
        focus, slot, producer = make_rig(source_builder=builder)
        roi = (0.02, 0.05, 0.45, 0.9)  # left half only (the wafer flakes)
        with running(producer):
            result = run_adaptive2(focus, slot, center=150.0, roi_norm=roi)
        assert result.success, result.message
        assert abs(result.best_position - 0.0) <= 1.5 * ADAPTIVE2_CFG["fine_step"]
    retry_once(case)


def test_adaptive2_producer_stall_completes_cleanly(qapp):
    """Camera stall mid-run: no hang, clean failure or recovery, idle."""
    def case():
        focus, slot, producer = make_rig(truth=30.0, k=0.05)

        class StallingProducer(FrameProducer):
            def __init__(self, slot, source_fn, stall_at):
                super().__init__(slot, source_fn, fps=25.0)
                self._stall_at = stall_at
                self._t0 = time.monotonic()

            def run(self):
                interval = 1.0 / self._fps
                next_t = time.monotonic() + interval
                while not self._stop_event.is_set():
                    if time.monotonic() - self._t0 > self._stall_at:
                        time.sleep(0.2)  # near-total stall: ~5 fps trickle
                        next_t = time.monotonic()
                        continue
                    t_cap = time.monotonic()
                    frame = self._source()
                    if frame is not None:
                        self._slot.write(frame, t_cap, self._seq)
                        self._seq += 1
                    next_t += interval
                    delay = next_t - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_t = time.monotonic()

        stalled = StallingProducer(producer._slot, producer._source,
                                   stall_at=1.0)
        t0 = time.monotonic()
        with running(stalled):
            result = run_adaptive2(focus, slot, center=0.0,
                                   fine_wait_timeout_s=0.5)
        assert time.monotonic() - t0 < 60.0  # no hang
        assert result.success or "peak" in result.message \
            or "stationary" in result.message
        assert focus.get_status().is_idle
    retry_once(case)


def test_adaptive2_dry_run_restores_arm_center(qapp):
    """No camera: the probe degrades to blind, the passes are walked
    motion-only and the run fails without a blind move."""
    focus = SimFocusStage({"max_speed": 2000, "latency_s": 0.001})
    focus.connect()
    result = run_adaptive2(focus, None, center=0.0)
    assert not result.success
    assert "peak" in result.message
    assert focus.get_status().pos == 0
    assert focus.get_status().is_idle
