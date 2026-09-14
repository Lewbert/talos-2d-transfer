"""Adaptive v3 closed-loop tests: the derivative-hybrid additions —
the probe curvature classification (incl. the valley gate), the
before-peak curvature stop + bypass, the 2σ cascade guard, and the
shared disciplines re-run against v3. The Gaussian sources
(talos/cv/sim_images.py) give the v3 model its own shape — the v2
rig's score curve is a zero-width cusp (audit #11)."""

from __future__ import annotations

import math
import threading
import time

import pytest
from PySide6.QtWidgets import QApplication

from talos.cv.af_math import ProbeResult, sg_curvature_at
from talos.cv.af_v3 import AdaptiveAutofocusController
from talos.cv.autofocus import AutofocusConfig, _AfExit
from tests.testing.sim_images import (
    gaussian_amplitude_source,
    synthetic_flake_image,
    two_plane_gaussian_source,
)
from talos.hal.sim import SimFocusStage

from tests.sim.test_autofocus_adaptive_closed_loop import ADAPTIVE_CFG
from tests.sim.test_autofocus_adaptive2_closed_loop import (
    ADAPTIVE2_CFG,
    prime,
)
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


# The Gaussian source's tenengrad is QUADRATIC in the amplitude →
# the score curve's effective σ = sigma_z/√2 ≈ 28.3 steps; the coarse
# sampling spacing = max_speed 200 / ~20 fps ≈ 10 steps → t = h/σ ≈ 0.35.
# The stop threshold 0.08 fires at ≈ 0.8σ before the peak (metric −0.08,
# b-gate open — verified against the SG kernel numerically).
_V3_SIGMA = 40.0 / math.sqrt(2.0)
_V3_T = 10.0 / _V3_SIGMA


def _v3_fields(**overrides):
    return {
        "probe_curv_in": 0.3 / _V3_SIGMA ** 2,
        "probe_curv_out": 0.1 / _V3_SIGMA ** 2,
        "coarse_curv_stop": 0.08,
        "coarse_curv_vertex": sg_curvature_at(0.7, _V3_T),
        **overrides,
    }


ADAPTIVE3_CFG = {**ADAPTIVE2_CFG, **_v3_fields()}


def make_cfg3(**kwargs):
    return AutofocusConfig(**{**ADAPTIVE3_CFG, **kwargs})


def _slow_fields():
    """A slow-pass variant for the curvature-stop tests: spacing ~5-8
    steps (t ≈ 0.18-0.28) puts 3-4× more samples in the fire zone than
    the default 200 sps pass — the 2-window debounce completes reliably
    under the full-suite load (the suite's producer starvation was
    flaking these tests at ~50%). The threshold is set low enough to
    fire across the whole t range (the metric scales with t²; 0.012
    fires at ~0.7σ for t=0.18, earlier for wider spacing)."""
    return {"max_speed": 100, "coarse_speed": 100,
            "coarse_curv_stop": 0.012,
            "coarse_curv_vertex": sg_curvature_at(0.7, 0.2)}


def run_adaptive3(focus, slot, center, **cfg_kwargs):
    cfg = make_cfg3(**cfg_kwargs)
    ctrl = AdaptiveAutofocusController(focus, slot)
    return ctrl.run(center=center, cfg=cfg)


def gaussian_rig(truth=0.0, sigma_z=40.0, noise=0.0, shape=(240, 320)):
    """A rig whose score curve is Gaussian in z (σ_eff = sigma_z/√2)."""
    sharp = synthetic_flake_image(seed=7, shape=shape, noise=0.0)
    source = gaussian_amplitude_source(sharp, focus_pos=truth,
                                       sigma_z=sigma_z, noise=noise)

    def builder(focus):
        return lambda: source(focus.load_pos)

    return make_rig(source_builder=builder)


def arm_rig_at(rig, pos: int) -> None:
    """The sim axis starts at 0; the real system sits AT the arm when
    run() is called. Move the axis there before the run — the preflight
    baseline (and the probe's relative floor) must read the arm's own
    score."""
    focus, _slot, _producer = rig
    focus.move_abs(pos, speed=2000)
    focus.wait_idle(timeout_s=10.0)


# ---------------------------------------------------------------------------
# Probe classification (curvature branch + the valley gate)
# ---------------------------------------------------------------------------

def test_v3_probe_near_via_curvature_enters_stage2(qapp):
    """Armed at the peak with a small probe δ (0.42σ — the v2 weaker-side
    test FAILS there: the sides are at 91.5% and 100 < 1.15×91.5): the
    curvature branch catches it and the coarse pass is skipped."""
    def case():
        focus, slot, producer = gaussian_rig(truth=0.0)
        with running(producer):
            cfg = make_cfg3(fine_step=4, probe_step_steps=12)
            ctrl = AdaptiveAutofocusController(focus, slot)
            phases: list[int] = []
            logs: list[str] = []
            ctrl.sig_progress.connect(
                lambda f, p, s, pos: phases.append(p))
            ctrl.sig_log.connect(logs.append)
            result = ctrl.run(center=0, cfg=cfg)
        assert result.success, result.message
        assert abs(result.best_position - 0.0) <= cfg.fine_step
        assert 4 not in phases  # no coarse pass
        assert any("near (curvature+max)" in m for m in logs)
        # audit B3: _probe_samples populated on the near path too
        assert len(ctrl._probe_samples) == 3
    retry_once(case)


def test_v3_bounds_override_asymmetric_window(qapp):
    """The bounds ARG: an asymmetric absolute-stage window replaces the
    symmetric center ± span/2 one — the truth outside the symmetric
    window but inside the override is found, and the pass never leaves
    the override."""
    def case():
        focus, slot, producer = make_rig(truth=400.0)
        arm_rig_at((focus, slot, producer), 0)
        with running(producer):
            cfg = make_cfg3(span_steps=400, probe_min_slope=10.0)
            ctrl = AdaptiveAutofocusController(focus, slot)
            result = ctrl.run(center=0, cfg=cfg, bounds=(100, 500))
        assert result.success, result.message
        assert abs(result.best_position - 400.0) <= ADAPTIVE2_CFG["fine_step"]
        xs = [p for p, _ in result.curve]
        # the pass STAGES from the arm (0) and sweeps into the override —
        # the arm-side samples are legitimate; the search never leaves
        # the override's far side
        assert max(xs) <= 500.0 + 10, f"left the override: {max(xs)}"
    retry_once(case)


def test_v3_coarse_direction_forced(qapp):
    """The coarse_direction ARG: the blind fallback sweeps the FORCED
    direction (−1, toward the sample) from the arm — no wrong-way leg,
    no reversal, the peak behind is reached directly."""
    def case():
        focus, slot, producer = make_rig(truth=-150.0)
        arm_rig_at((focus, slot, producer), 0)
        with running(producer):
            cfg = make_cfg3(span_steps=800, probe_min_slope=10.0,
                            coarse_direction=-1)
            ctrl = AdaptiveAutofocusController(focus, slot)
            logs: list[str] = []
            ctrl.sig_log.connect(logs.append)
            result = ctrl.run(center=0, cfg=cfg)
        assert result.success, result.message
        assert abs(result.best_position - (-150.0)) <= ADAPTIVE2_CFG["fine_step"]
        assert not any("reversing" in m for m in logs), \
            f"the forced direction still reversed: {logs}"
        xs = [p for p, _ in result.curve]
        assert min(xs) < -150.0, f"never swept the forced side: {min(xs)}"
    retry_once(case)


def test_v3_near_entry_widens_the_stage2_window(qapp):
    """The near verdict tolerates the arm up to ~±0.7σ off the true
    peak — the stage-2 window (and the fit-distrust tolerance fw//2)
    must cover the probe's uncertainty, not just the coarse
    localization error the default fw is sized for (hardware-found:
    the hill fit −74 was distrusted against the anchor 0 with fw 100,
    and the re-measure around the anchor failed on the flat shoulder)."""
    def case():
        focus, slot, producer = gaussian_rig(truth=0.0)
        with running(producer):
            cfg = make_cfg3(near_window_steps=170)
            ctrl = AdaptiveAutofocusController(focus, slot)
            prime(ctrl, cfg, 0)
            probe = ctrl._probe(0, (-200, 200), cfg)
        assert probe.near_focus
        assert cfg.fine_window_steps == 170, \
            f"near entry left the window at {cfg.fine_window_steps}"
    retry_once(case)


def test_v3_near_cluster_probe_enters_stage2_at_the_best_point(qapp):
    """The multi-peak shoulder case (hardware-found): the center-max
    gate fails (a neighbor sub-peak beats the center) but all three
    probe points are HIGH — the cluster branch enters stage 2 directly,
    anchored at the BEST probe point, and lands the truth. (The probe
    itself is faked — the branch math is unit-pinned; this test
    exercises the _run integration through _probe_anchor.)"""
    def case():
        focus, slot, producer = gaussian_rig(truth=120.0)
        arm_rig_at((focus, slot, producer), 0)
        with running(producer):
            cfg = make_cfg3()
            ctrl = AdaptiveAutofocusController(focus, slot)
            phases: list[int] = []
            ctrl.sig_progress.connect(
                lambda f, p, s, pos: phases.append(p))

            def fake_probe(center, bounds, cfg_in):
                ctrl._probe_samples = [(-120.0, 1500.0, 500.0),
                                       (0.0, 1800.0, 600.0),
                                       (120.0, 2400.0, 700.0)]
                ctrl._cluster_anchor = 120.0
                return ProbeResult(True, None, ctrl._probe_samples)

            ctrl._probe = fake_probe
            result = ctrl.run(center=0, cfg=cfg)
        assert result.success, result.message
        assert abs(result.best_position - 120.0) <= cfg.fine_step
        assert 4 not in phases  # no coarse pass — direct stage 2
    retry_once(case)


def test_v3_two_plane_valley_gate_blocks_the_near_call(qapp):
    """Audit #8/#9 geometry: the wafer (weak plane, z=0) and the mount
    (strong, z=240) with the arm in the valley at 80 — the probe center
    reads the valley floor (28 vs the mount's 293). The requirement the
    audit pins is the GATE: the probe must NOT read near (a false near
    would chase the mount from the arm). The landing side in a valley
    is genuinely ambiguous for the arm-start blind pass (it sweeps the
    mount side; mount landings are acceptable per the user's feedback —
    the physical restrain protects the objective)."""
    source = two_plane_gaussian_source(shape=(240, 320), seed=7,
                                       z_top=0.0, z_bottom=240.0,
                                       top_weight=0.4)

    def builder(focus):
        return lambda: source(focus.load_pos)

    def case():
        focus, slot, producer = make_rig(source_builder=builder)
        arm_rig_at((focus, slot, producer), 80)
        with running(producer):
            cfg = make_cfg3(coarse_curv_stop=0.015)
            ctrl = AdaptiveAutofocusController(focus, slot)
            logs: list[str] = []
            ctrl.sig_log.connect(logs.append)
            result = ctrl.run(center=80, cfg=cfg)
        assert result.success, result.message
        assert not any("probe: near" in m for m in logs), \
            "the valley must never read near"
        assert any("flank" in m for m in logs)
        # either plane is a valid landing (the valley's ambiguity)
        assert min(abs(result.best_position - 0.0),
                   abs(result.best_position - 240.0)) <= 2 * cfg.fine_step, \
            f"landed on neither plane: {result.best_position}"
    retry_once(case)


def test_v3_blind_from_arm_ahead_no_reversal(qapp):
    """The blind fallback starts AT the arm sweeping +1: with the truth
    ahead, the early samples rise, no reversal fires, and the pass
    reaches the peak in a few hundred steps instead of walking from the
    far window edge — the − side is never touched."""
    def case():
        focus, slot, producer = make_rig(truth=600.0)
        arm_rig_at((focus, slot, producer), 0)
        with running(producer):
            cfg = make_cfg3(span_steps=1600, probe_min_slope=10.0)
            ctrl = AdaptiveAutofocusController(focus, slot)
            logs: list[str] = []
            ctrl.sig_log.connect(logs.append)
            result = ctrl.run(center=0, cfg=cfg)
        assert result.success, result.message
        assert abs(result.best_position - 600.0) <= ADAPTIVE2_CFG["fine_step"]
        # the probe's own seed at −120 is the curve's leftmost point by
        # design — the PASS itself never went below the arm (0)
        xs = [p for p, _ in result.curve]
        assert min(xs) >= -120.5, f"swept the wrong side: {min(xs)}"
        assert not any("reversing" in m for m in logs)
    retry_once(case)


def test_v3_blind_from_arm_reverses_when_the_peak_is_behind(qapp):
    """The blind fallback with the truth BEHIND: the early samples fall
    (both metrics), the 2σ trigger reverses the pass, and the reversed
    sweep recovers the peak — the stage-1 direction decided by the
    pass's own samples, not the probe."""
    def case():
        focus, slot, producer = make_rig(truth=-150.0)
        arm_rig_at((focus, slot, producer), 0)
        with running(producer):
            cfg = make_cfg3(span_steps=800, probe_min_slope=10.0)
            ctrl = AdaptiveAutofocusController(focus, slot)
            logs: list[str] = []
            ctrl.sig_log.connect(logs.append)
            result = ctrl.run(center=0, cfg=cfg)
        assert result.success, result.message
        assert abs(result.best_position - (-150.0)) <= ADAPTIVE2_CFG["fine_step"]
        assert any("reversing" in m for m in logs), \
            f"the early direction check never reversed: {logs}"
        xs = [p for p, _ in result.curve]
        assert min(xs) < -150.0, f"never swept toward the peak: {min(xs)}"
    retry_once(case)


# ---------------------------------------------------------------------------
# Curvature stop (both sweep directions) + the bypass
# ---------------------------------------------------------------------------

def test_v3_curvature_stop_fires_before_the_peak_closed_loop(qapp):
    """Far from focus: the probe gives the direction and the coarse pass
    stops ~0.8σ BEFORE the peak (the v2 early-stop needs 2 post-peak
    samples by design). The bypass then skips pick_peak/edge checks and
    the trusted vertex anchors stage 2 — the run lands the truth."""
    def case():
        focus, slot, producer = gaussian_rig(truth=0.0)
        arm_rig_at((focus, slot, producer), -120)
        with running(producer):
            cfg = make_cfg3(**_slow_fields())
            ctrl = AdaptiveAutofocusController(focus, slot)
            logs: list[str] = []
            progress: list[tuple[int, float]] = []
            ctrl.sig_log.connect(logs.append)
            ctrl.sig_progress.connect(
                lambda f, p, s, pos: progress.append((p, float(pos))))
            result = ctrl.run(center=-120, cfg=cfg)
        assert result.success, result.message
        assert abs(result.best_position - 0.0) <= cfg.fine_step
        stop_lines = [m for m in logs if "curvature stop" in m]
        assert stop_lines, "the curvature stop never fired"
        assert any("trusted vertex" in m for m in logs)
        # the pass halted far short of the v2 early-stop's post-peak
        # distance (~70+): the stop fired at the peak's shoulder and the
        # frame-capture-to-halt latency coasted a few steps past the
        # crest (200 sps × ~50-100 ms) — the trusted vertex is the
        # anchor, not the halt position. (The HARDWARE bench asserts the
        # true before-peak stop at its slower, real latencies.)
        ph4 = [pos for p, pos in progress if p == 4]
        assert ph4 and ph4[-1] < 50.0, f"pass ran far past the peak: {ph4[-1]}"
    retry_once(case)


def test_v3_curvature_stop_both_sweep_directions(qapp):
    """Direct-stage: the sweep-aware b gate passes on BOTH directions
    (approaching from below has b > 0, from above b < 0 — the gate
    multiplies by the sweep direction). The bypass PeakInfo carries
    the trusted vertex (within ~2 sampling spacings of the truth),
    never an edge flag."""
    def one(direction, arm):
        def case():
            focus, slot, producer = gaussian_rig(truth=0.0)
            # the AXIS must sit at the arm too — prime() only sets the
            # controller's tracked position; the pass starts from the
            # axis's resting place
            focus.move_abs(arm, speed=2000)
            focus.wait_idle(timeout_s=10.0)
            with running(producer):
                cfg = make_cfg3(**_slow_fields())
                ctrl = AdaptiveAutofocusController(focus, slot)
                prime(ctrl, cfg, arm)
                logs: list[str] = []
                ctrl.sig_log.connect(logs.append)
                peak = ctrl._coarse_pass(arm, (-200, 200), cfg,
                                         direction=direction)
            assert peak is not None
            assert not peak.at_edge  # the bypass never flags an edge
            assert getattr(peak, "curvature_stop", False)
            assert abs(peak.pos - 0.0) <= 2 * 10.0, \
                f"vertex anchor off: {peak.pos} (arm {arm}, dir {direction})"
            assert any("curvature stop" in m for m in logs)
            assert any("trusted vertex" in m for m in logs)
            # the stage halted around the peak's crest, far short of
            # the v2 early-stop's post-peak stop (~+70): the stop fired
            # at the shoulder and the capture-to-halt coast carried a
            # few steps past the crest
            assert focus.get_status().pos * direction < 20.0
            assert focus.get_status().is_idle
        retry_once(case)

    # arm at the window EDGE: the pass's first 6 samples sit on the
    # source's flat far tail, so the detrend's b_early reads the true
    # background (≈0) and cannot cancel the fire window's slope
    one(+1, -200)
    one(-1, +200)


def test_v3_debounce_single_negative_window_does_not_stop(qapp):
    """A single deep sample dip (one noise-negative window) must NOT
    fire the stop — the 2-consecutive-window debounce. The pass
    continues past the dip and the real stop fires near the peak.
    (The dip sits PAST the guard window — inside it the v2 drop test
    would legitimately fire first: the brenner is quadratic in the
    amplitude, so a 50% amplitude dip reads as a 75% low-freq fall.)"""
    sharp = synthetic_flake_image(seed=7, shape=(240, 320), noise=0.0)
    base_source = gaussian_amplitude_source(sharp, focus_pos=0.0,
                                            sigma_z=40.0)

    def notched(pos):
        # a ONE-SAMPLE dip (width 1.5 steps): even at the slow pass's
        # 5-7-step spacing the dip lands in a single window — the
        # debounce's premise (one isolated negative window). It sits
        # past the guard's fit window (~−85 under moderate load)
        dip = 0.5 * math.exp(-(pos + 75.0) ** 2 / (2.0 * 2.25))
        return base_source(pos).astype("float32") * (1.0 - dip)

    def case():
        focus, slot, producer = make_rig(
            source_builder=lambda f: (lambda: notched(f.load_pos)))
        arm_rig_at((focus, slot, producer), -120)
        with running(producer):
            # the drop test silenced (the brenner is quadratic — a 50%
            # amplitude dip reads as a 75% low-freq fall); the debounce
            # under test is the curvature rung's
            cfg = make_cfg3(**_slow_fields(), guard_drop_ratio=0.9)
            ctrl = AdaptiveAutofocusController(focus, slot)
            logs: list[str] = []
            ctrl.sig_log.connect(logs.append)
            result = ctrl.run(center=-120, cfg=cfg)
        assert result.success, result.message
        stop_lines = [m for m in logs if "curvature stop" in m]
        assert stop_lines
        for m in stop_lines:
            pos = float(m.split(" at ")[1].split(" ")[0])
            assert pos > -40, f"stopped at the notch: {m}"
    retry_once(case)


# ---------------------------------------------------------------------------
# Cascade direction guard (2σ pooled-noise trigger)
# ---------------------------------------------------------------------------

def test_v3_guard_pooled_2sigma_reversal(qapp):
    """A consistent but GENTLE wrong-way slope (the v2 15% drop test
    stays silent with guard_drop_ratio 0.35 — the fit trigger's domain):
    the 2σ pooled-noise slope test reverses the pass and the reversed
    pass recovers the peak."""
    def case():
        focus, slot, producer = gaussian_rig(truth=-150.0, sigma_z=200.0)
        with running(producer):
            cfg = make_cfg3(guard_drop_ratio=0.35)
            ctrl = AdaptiveAutofocusController(focus, slot)
            prime(ctrl, cfg, 0)
            # run the REAL probe first (fills _probe_samples → the
            # pooled σₙ), then force the WRONG direction
            probe = ctrl._probe(0, (-200, 200), cfg)
            assert probe.direction == -1  # the truth IS at −150
            logs: list[str] = []
            ctrl.sig_log.connect(logs.append)
            peak = ctrl._coarse_pass(0, (-200, 200), cfg, direction=+1)
        assert abs(peak.pos - (-150.0)) <= 40, f"reversed pass missed: {peak.pos}"
        assert any("2σ" in m for m in logs), logs
        assert focus.get_status().is_idle
    retry_once(case)


def test_v3_guard_no_fire_on_noisy_flat(qapp):
    """A flat noisy field: the fit trigger must stay silent (b̂ ≈ 0).
    The pass outcome is not the point (a noise wiggle may or may not
    pass the peak checks) — the trigger must not reverse on it."""
    sharp = synthetic_flake_image(seed=7, shape=(240, 320), noise=0.0)
    source = gaussian_amplitude_source(sharp, focus_pos=0.0,
                                       sigma_z=40.0, floor=1.0, noise=0.05)

    def builder(focus):
        return lambda: source(focus.load_pos)

    def case():
        focus, slot, producer = make_rig(source_builder=builder)
        with running(producer):
            cfg = make_cfg3()
            ctrl = AdaptiveAutofocusController(focus, slot)
            prime(ctrl, cfg, 0)
            ctrl._probe_samples = [(-120.0, 900.0, 250.0),
                                   (0.0, 910.0, 255.0),
                                   (120.0, 890.0, 245.0)]
            logs: list[str] = []
            ctrl.sig_log.connect(logs.append)
            try:
                ctrl._coarse_pass(0, (-200, 200), cfg, direction=+1)
            except _AfExit:
                pass
        assert not any("2σ" in m for m in logs), logs
        assert focus.get_status().is_idle
    retry_once(case)


def test_v3_guard_v2_drop_test_fallback_retained(qapp):
    """A STEEP wrong-way slope still fires the v2 15% drop test FIRST
    (the guard's first rung) — the 2σ trigger never gets the chance."""
    def case():
        focus, slot, producer = gaussian_rig(truth=-60.0, sigma_z=40.0)
        with running(producer):
            cfg = make_cfg3()
            ctrl = AdaptiveAutofocusController(focus, slot)
            prime(ctrl, cfg, 0)
            ctrl._probe_samples = [(-120.0, 700.0, 220.0),
                                   (0.0, 155.0, 90.0),
                                   (120.0, 62.0, 23.0)]
            logs: list[str] = []
            ctrl.sig_log.connect(logs.append)
            peak = ctrl._coarse_pass(0, (-200, 200), cfg, direction=+1)
        assert abs(peak.pos - (-60.0)) <= 40, f"reversed pass missed: {peak.pos}"
        assert any("direction guard fired" in m for m in logs)
        assert not any("2σ" in m for m in logs), logs
    retry_once(case)


# ---------------------------------------------------------------------------
# Shared disciplines against v3 (thresholds at 0 → pure v2 behavior —
# the inherited machinery must stay intact through the overrides)
# ---------------------------------------------------------------------------

def test_v3_zero_thresholds_behaves_like_v2(qapp):
    """With the derivative thresholds disabled (0), the v3 controller
    degrades to exactly the v2 pipeline — the v2 far-probe scenario
    runs identically through the v3 class."""
    def case():
        focus, slot, producer = make_rig(truth=120.0)
        with running(producer):
            cfg = AutofocusConfig(
                **{**ADAPTIVE2_CFG,
                   "probe_curv_in": 0.0, "probe_curv_out": 0.0,
                   "coarse_curv_stop": 0.0, "coarse_curv_vertex": 0.0})
            ctrl = AdaptiveAutofocusController(focus, slot)
            result = ctrl.run(center=0, cfg=cfg)
        assert result.success, result.message
        assert abs(result.best_position - 120.0) <= ADAPTIVE2_CFG["fine_step"]
        xs = [p for p, _ in result.curve]
        assert min(xs) >= -(120 + 10), f"swept the wrong side: min {min(xs)}"
    retry_once(case)


def test_v3_abort_mid_probe_stops_stage(qapp):
    focus, slot, producer = gaussian_rig(truth=0.0)
    with running(producer):
        cfg = make_cfg3()
        ctrl = AdaptiveAutofocusController(focus, slot)
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


def test_v3_producer_stall_completes_cleanly(qapp):
    """Camera stall mid-run: no hang, clean failure or recovery, idle."""
    def case():
        focus, slot, producer = gaussian_rig(truth=0.0)

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
            result = run_adaptive3(focus, slot, center=0.0,
                                   fine_wait_timeout_s=0.5)
        assert time.monotonic() - t0 < 60.0  # no hang
        assert result.success or "peak" in result.message \
            or "stationary" in result.message
        assert focus.get_status().is_idle
    retry_once(case)


def test_v3_dry_run_failure_stays_at_current_position(qapp):
    """No camera: the probe degrades to blind, the passes are walked
    motion-only and the run fails. v3's failure policy: NO arm restore —
    the axis stays where the pass stopped, and the reported position is
    that stop position (not the staging position)."""
    focus = SimFocusStage({"max_speed": 2000, "latency_s": 0.001})
    focus.connect()
    result = run_adaptive3(focus, None, center=0.0)
    assert not result.success
    assert "peak" in result.message
    assert "stopped at current position" in result.message
    assert focus.get_status().pos > 0  # the blind sweep moved off the arm
    assert result.best_position == focus.get_status().pos  # truthful
    assert focus.get_status().is_idle


def test_v3_refine_near_stays_in_the_refine_window(qapp):
    """AF_REFINE with a small drift: the probe says near (plateau) →
    stage 2 directly, no coarse pass."""
    def case():
        focus, slot, producer = gaussian_rig(truth=0.0)
        arm_rig_at((focus, slot, producer), 40)
        with running(producer):
            cfg = make_cfg3(mode="AF_REFINE", fine_window_steps=60,
                            probe_step_steps=100)
            ctrl = AdaptiveAutofocusController(focus, slot)
            phases: list[int] = []
            ctrl.sig_progress.connect(
                lambda f, p, s, pos: phases.append(p))
            result = ctrl.run(center=40, cfg=cfg)
        assert result.success, result.message
        assert abs(result.best_position - 0.0) <= cfg.fine_step
        assert 4 not in phases
    retry_once(case)


def test_v3_refine_far_runs_the_full_pipeline(qapp):
    """AF_REFINE with heavy drift: the full pipeline (v2 rig — the
    derivative thresholds at 0 through the refine window geometry)."""
    def case():
        focus, slot, producer = make_rig(truth=0.0)
        arm_rig_at((focus, slot, producer), 300)
        with running(producer):
            result = run_adaptive3(focus, slot, center=300,
                                   mode="AF_REFINE", span_steps=800,
                                   fine_window_steps=60,
                                   probe_curv_in=0.0, probe_curv_out=0.0,
                                   coarse_curv_stop=0.0,
                                   coarse_curv_vertex=0.0)
        assert result.success, result.message
        assert abs(result.best_position - 0.0) <= ADAPTIVE2_CFG["fine_step"]
    retry_once(case)
