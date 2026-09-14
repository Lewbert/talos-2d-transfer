"""Adaptive 3-stage closed-loop tests: SimFocusStage (CONT fly-by passes,
live position, backlash) + the same real-time producer harness as the
classic suite. The classic suite is untouched — these tests pin the NEW
strategy's contract: planted-truth recovery, early-stop, the speed
schedule, abort, refine, backlash landing, the stationary fallback, and
the no-camera/preflight failure paths."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from talos.cv.af_adaptive import AdaptiveAutofocusControllerV1
from talos.cv.autofocus import AutofocusConfig
from talos.cv.frame_slot import LatestFrameSlot
from tests.testing.sim_images import (
    defocus_blur,
    split_plane_source,
    synthetic_flake_image,
    two_plane_source,
)
from talos.hal.sim import SimFocusStage

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


class RecordingFocus(SimFocusStage):
    """Records set_speed (CONT) and move_abs (TRAP staging/landing)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.speeds: list[int] = []
        self.moves: list[tuple] = []

    def set_speed(self, steps_per_s):
        self.speeds.append(int(steps_per_s))
        super().set_speed(steps_per_s)

    def move_abs(self, position, speed=None):
        self.moves.append((int(position), speed))
        super().move_abs(position, speed)


# max_speed 200 × span 400 = 2 s coarse pass at ~20 fps (same margin
# reasoning as the classic DEFAULT_CFG — see that file's comment).
ADAPTIVE_CFG = dict(coarse_step=40, fine_step=8, span_steps=400, max_speed=200,
                    fine_speed=200, landing_speed=50, stage_speed=200,
                    quality_threshold=0.05,
                    timeout_s=90.0, freshness_ms=800.0, coarse_poll_s=0.04,
                    settle_frames=1, fail_on_edge_peak=True, metric="tenengrad",
                    strategy="adaptive", coarse_speed=200, hill_v_cap=100,
                    hill_v_min=50, fine_window_steps=40, hill_ratio=0.6,
                    hill_early_stop_samples=2, lock_samples_required=4,
                    lock_score_frac=0.85, stationary_points=5,
                    stop_accel_sps2=20000, stop_latency_s=0.05,
                    stop_safety_steps=10, early_stop_ratio=0.6,
                    early_stop_samples=3, coarse_metric="brenner_k",
                    coarse_metric_k=8, coarse_bin=2)


def run_adaptive(focus, slot, center, **cfg_kwargs):
    cfg = AutofocusConfig(**{**ADAPTIVE_CFG, **cfg_kwargs})
    ctrl = AdaptiveAutofocusControllerV1(focus, slot)
    return ctrl.run(center=center, cfg=cfg)


# ---------------------------------------------------------------------------
# Peak recovery
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("truth,center,seed", [
    (0, 0, 1),
    (120, -40, 2),
    (-160, -120, 3),
    (40, 60, 4),
])
def test_adaptive_recovers_planted_truth(qapp, truth, center, seed):
    def case():
        focus, slot, producer = make_rig(truth=truth, seed=seed)
        with running(producer):
            result = run_adaptive(focus, slot, center)
        assert result.success, result.message
        assert abs(result.best_position - truth) <= ADAPTIVE_CFG["fine_step"]
        assert focus.get_status().pos == result.best_position  # landed & verified
    retry_once(case)


def test_adaptive_coarse_early_stops(qapp):
    """Both passes stop shortly after the peak — the far window edge is
    never walked (the user-requested agility behavior). The low-frequency
    coarse metric is broader than Tenengrad, so the 0.6 early-stop fires
    later than the classic's — the bound proves the stop happened well
    inside the 400-step span."""
    focus = RecordingFocus({"max_speed": 2000, "latency_s": 0.001})
    focus, slot, producer = make_rig(focus=focus, truth=0.0)
    with running(producer):
        result = run_adaptive(focus, slot, center=0.0)
    assert result.success, result.message
    furthest = max(p for p, _ in result.curve)
    assert furthest < 200, f"passes ran too far: {furthest}"


def test_adaptive_speed_sequence(qapp):
    """The CONT passes drive the stage through the speed ladder: coarse at
    coarse_speed, the hill climb capped at hill_v_cap with the near-peak
    hill_v_min, and every pass ends with the ramp stop (SPD:0)."""
    def case():
        focus = RecordingFocus({"max_speed": 2000, "latency_s": 0.001})
        focus, slot, producer = make_rig(focus=focus, truth=0.0)
        with running(producer):
            result = run_adaptive(focus, slot, 0)
        assert result.success, result.message
        mags = [abs(s) for s in focus.speeds]
        assert ADAPTIVE_CFG["coarse_speed"] in mags
        assert ADAPTIVE_CFG["hill_v_cap"] in mags
        assert ADAPTIVE_CFG["hill_v_min"] in mags
        assert 0 in focus.speeds                 # ramp stops
        # a pass never exceeds its configured speed
        assert max(mags) <= ADAPTIVE_CFG["coarse_speed"]
    retry_once(case)


def test_adaptive_abort_mid_pass_stops_stage(qapp):
    focus, slot, producer = make_rig(truth=0.0)
    with running(producer):
        cfg = AutofocusConfig(**ADAPTIVE_CFG)
        ctrl = AdaptiveAutofocusControllerV1(focus, slot)
        result_box: dict = {}

        def worker():
            result_box["result"] = ctrl.run(center=0, cfg=cfg)

        thread = threading.Thread(target=worker)
        thread.start()
        time.sleep(0.5)  # mid-coarse (the pass takes ~2 s)
        ctrl.request_abort()
        thread.join(timeout=30.0)
        result = result_box["result"]
    assert result.aborted, result.message
    assert focus.get_status().is_idle  # ramp-halted


def test_adaptive_refine_repeaks_without_coarse(qapp):
    """AF_REFINE: no full-window pass — the curve stays inside the refine
    window around the armed position, and the load returns to the peak."""
    def case():
        focus, slot, producer = make_rig(truth=0.0)
        with running(producer):
            focus.move_abs(0, speed=100)
            focus.wait_idle()
            time.sleep(0.15)
            focus.move_abs(40, speed=100)  # drift (thermal/settling)
            focus.wait_idle()
            time.sleep(0.15)
            result = run_adaptive(focus, slot, center=40, mode="AF_REFINE",
                                  fine_window_steps=60)
        assert result.success, result.message
        assert abs(focus.load_pos - 0.0) <= ADAPTIVE_CFG["fine_step"]
        # no coarse pass: every scored position stays inside the refine window
        span = max(p for p, _ in result.curve) - min(p for p, _ in result.curve)
        assert span <= 2 * 60 + 20, f"refine curve span {span} too wide"
    retry_once(case)


def test_adaptive_backlash_landing_puts_the_load_on_truth(qapp):
    """B=8: the direction-matched landing must place the LOAD (what the
    camera sees) on the planted truth, same contract as the classic."""
    def case():
        focus = SimFocusStage({"max_speed": 2000, "latency_s": 0.001,
                               "backlash_steps": 8})
        focus.connect()
        slot = LatestFrameSlot()
        truth = 50.0
        producer = FrameProducer(
            slot, lambda: defocus_blur(
                synthetic_flake_image(seed=7, shape=(240, 320)),
                0.4 + 0.05 * abs(focus.load_pos - truth)))
        with running(producer):
            result = run_adaptive(focus, slot, center=0.0)
        assert result.success, result.message
        assert abs(focus.load_pos - truth) <= ADAPTIVE_CFG["fine_step"] + 1
        assert focus.get_status().pos == result.best_position
    retry_once(case)


def test_adaptive_producer_stall_completes_cleanly(qapp):
    """The camera stalls mid-run: freshness gates drop points (holes,
    never fake zeros); the stationary re-measure either recovers or the
    run fails cleanly — never a hang, never a blind move."""
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
            result = run_adaptive(focus, slot, center=0.0,
                                  fine_wait_timeout_s=0.5)
        assert time.monotonic() - t0 < 60.0  # no hang
        assert result.success or "peak" in result.message \
            or "stationary" in result.message
        assert focus.get_status().is_idle
    retry_once(case)


def test_adaptive_two_plane_nearest_center(qapp):
    """Wafer surface (weak, near the arm position) vs mount below: the
    hill window around the nearest-center coarse peak must not wander to
    the other plane."""
    source = two_plane_source(shape=(240, 320), seed=7, z_top=0.0,
                              z_bottom=150.0, top_weight=0.4)

    def builder(focus):
        return lambda: source(focus.load_pos)

    def case():
        focus, slot, producer = make_rig(source_builder=builder)
        with running(producer):
            result = run_adaptive(focus, slot, center=0.0)
        assert result.success, result.message
        assert abs(result.best_position - 0.0) <= 1.5 * ADAPTIVE_CFG["fine_step"]

        focus2, slot2, producer2 = make_rig(source_builder=builder)
        with running(producer2):
            result2 = run_adaptive(focus2, slot2, center=150.0)
        assert result2.success
        assert abs(result2.best_position - 150.0) <= 2.5 * ADAPTIVE_CFG["fine_step"]
    retry_once(case)


def test_adaptive_roi_selects_the_plane(qapp):
    """Same ROI contract as the classic: a left-half ROI pulls the metric
    (and the adaptive stages) to the wafer plane while armed at the
    mount."""
    source = split_plane_source(shape=(240, 320), seed=7, z_top=0.0,
                                z_bottom=150.0, top_flakes=4, bottom_flakes=14)

    def builder(focus):
        return lambda: source(focus.load_pos)

    def case():
        focus, slot, producer = make_rig(source_builder=builder)
        roi = (0.02, 0.05, 0.45, 0.9)  # left half only (the wafer flakes)
        with running(producer):
            result = run_adaptive(focus, slot, center=150.0, roi_norm=roi)
        assert result.success, result.message
        assert abs(result.best_position - 0.0) <= 1.5 * ADAPTIVE_CFG["fine_step"]
    retry_once(case)


def test_adaptive_dry_run_restores_arm_center(qapp):
    """No camera: the passes are walked motion-only and the run fails
    without a blind move — the axis returns to the arm position."""
    focus = SimFocusStage({"max_speed": 2000, "latency_s": 0.001})
    focus.connect()
    result = run_adaptive(focus, None, center=0.0)
    assert not result.success  # no frames → no peak
    assert "peak" in result.message
    assert focus.get_status().pos == 0  # restored to the arm position
    assert focus.get_status().is_idle


def test_adaptive_flat_scene_preflight_rejects(qapp):
    focus = SimFocusStage({"max_speed": 2000, "latency_s": 0.001})
    focus.connect()
    slot = LatestFrameSlot()
    slot.write(np.full((240, 320, 3), 128, np.uint8), 1.0, 1)
    result = run_adaptive(focus, slot, center=0.0)
    assert not result.success
    assert "contrast" in result.message
    assert focus.get_status().pos == 0  # never moved
    assert result.phase == "preflight"
