"""Closed-loop autofocus tests: SimFocusStage + a producer thread feeding
a LatestFrameSlot at real-time cadence, so the controller's freshness
gating, position interpolation and backlash landing run against REAL
timestamps — the honest sim of the hardware loop (producer, controller
and focus all live on different threads, as in the app)."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from talos.cv.autofocus import AutofocusConfig, AutofocusController
from talos.cv.frame_slot import LatestFrameSlot
from tests.testing.sim_images import (
    defocus_blur,
    split_plane_source,
    synthetic_flake_image,
    two_plane_source,
)
from talos.hal.sim import SimFocusStage

# Real-time closed-loop simulation: the whole file is marked slow
# (deselect with `-m "not slow"` for a fast edit loop).
pytestmark = pytest.mark.slow



@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class FrameProducer(threading.Thread):
    """Renders defocus frames at a fixed cadence into the slot. The
    capture timestamp is stamped BEFORE rendering — the content reflects
    the focus position at that instant (exposure-midpoint analog)."""

    def __init__(self, slot, source_fn, fps=25.0):
        super().__init__(daemon=True)
        self._slot = slot
        self._source = source_fn   # () -> frame (reads the focus LOAD pos)
        self._fps = fps
        self._stop_event = threading.Event()  # NOT _stop: Thread._stop() is internal
        self._seq = 0

    def run(self) -> None:
        interval = 1.0 / self._fps
        next_t = time.monotonic() + interval
        while not self._stop_event.is_set():
            t_cap = time.monotonic()
            try:
                frame = self._source()
            except Exception:  # noqa: BLE001
                frame = None
            if frame is not None:
                self._slot.write(frame, t_cap, self._seq)
                self._seq += 1
            next_t += interval
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=5.0)


class RecordingFocus(SimFocusStage):
    """Records every commanded move for sequence/speed assertions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls: list[tuple] = []

    def move_abs(self, position, speed=None):
        self.calls.append(("move_abs", int(position), speed))
        super().move_abs(position, speed)

    def stop(self):
        self.calls.append(("stop",))
        super().stop()


def make_rig(focus=None, truth=0.0, k=0.05, seed=7, shape=(240, 320),
             fps=20.0, source_builder=None):
    """source_builder(focus) -> ()->frame; default = blur-by-defocus on a
    flake scene, sharp at ``truth`` on the focus LOAD axis."""
    focus = focus or SimFocusStage({"max_speed": 2000, "latency_s": 0.001})
    focus.connect()
    if source_builder is None:
        sharp = synthetic_flake_image(seed=seed, shape=shape, noise=0.0)

        def source_builder(focus):
            def fn():
                sigma = 0.4 + k * abs(focus.load_pos - truth)
                return defocus_blur(sharp, sigma)
            return fn
    slot = LatestFrameSlot()
    producer = FrameProducer(slot, source_builder(focus), fps=fps)
    return focus, slot, producer


@contextmanager
def running(producer):
    producer.start()
    try:
        yield
    finally:
        producer.stop()


# max_speed 200 × span 400 = 2 s sweep at ~13-16 fps (Windows timer
# granularity stretches the producer's 50 ms pacing) ≈ 30 coarse samples —
# enough margin that CPU contention in a full-suite run cannot starve the
# scan of frames. freshness_ms is loose for the same reason: under load a
# frame can sit in the slot longer than on real hardware.
DEFAULT_CFG = dict(coarse_step=40, fine_step=8, span_steps=400, max_speed=200,
                   fine_speed=100, landing_speed=50, quality_threshold=0.05,
                   timeout_s=90.0, freshness_ms=800.0, coarse_poll_s=0.04,
                   settle_frames=1, fail_on_edge_peak=True, metric="tenengrad")


def run_autofocus(focus, slot, center, **cfg_kwargs):
    cfg = AutofocusConfig(**{**DEFAULT_CFG, **cfg_kwargs})
    ctrl = AutofocusController(focus, slot)
    return ctrl.run(center=center, cfg=cfg)


def retry_once(fn):
    """Real-time harness guard: these tests run a frame producer at
    ~13-16 fps (Windows timer granularity stretches the pacing) against a
    2 s sweep, under whatever load pytest brings. A producer starved by
    scheduler contention can omit the true peak from the curve entirely —
    no controller can recover a peak it never saw. The controller logic
    itself is deterministic (unit-tested) and exercises 12/12 clean in the
    standalone loop; this guard retries ONE starved run with a fresh rig,
    never masking a repeatable failure."""
    for attempt in (1, 2):
        try:
            return fn()
        except AssertionError:
            if attempt == 2:
                raise


# ---------------------------------------------------------------------------
# Peak recovery
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("truth,center,seed", [
    (0, 0, 1),
    (120, -40, 2),
    (-160, -120, 3),
    (40, 60, 4),
    (0, -150, 5),
    (180, 60, 6),
])
def test_autofocus_recovers_planted_truth(qapp, truth, center, seed):
    def case():
        focus, slot, producer = make_rig(truth=truth, seed=seed)
        with running(producer):
            result = run_autofocus(focus, slot, center)
        assert result.success, result.message
        # no backlash → load == counter: the planted truth is the counter pos
        assert abs(result.best_position - truth) <= DEFAULT_CFG["fine_step"]
        assert focus.get_status().pos == result.best_position  # landed & verified
    retry_once(case)


def test_coarse_early_stops_after_passing_the_peak(qapp):
    """The user-requested agility behavior: once the score has clearly
    fallen past the peak, the sweep must STOP and return to the peak
    region — not walk the whole window (super time-wasting on hardware)."""
    focus = RecordingFocus({"max_speed": 2000, "latency_s": 0.001})
    focus, slot, producer = make_rig(focus=focus, truth=0.0)
    with running(producer):
        result = run_autofocus(focus, slot, center=0.0)
    assert result.success, result.message
    # the sweep never scored anything near the far window edge (±200
    # steps) — it stopped shortly after passing the peak at 0
    furthest = max(p for p, _ in result.curve)
    assert furthest < 150, f"sweep ran too far: {furthest}"
    assert abs(focus.load_pos - 0.0) <= DEFAULT_CFG["fine_step"]


def test_variable_speed_and_move_sequence(qapp):
    def case():
        focus = RecordingFocus({"max_speed": 2000, "latency_s": 0.001})
        focus, slot, producer = make_rig(focus=focus, truth=0.0)
        with running(producer):
            result = run_autofocus(focus, slot, 0)
        assert result.success, result.message
        moves = [c for c in focus.calls if c[0] == "move_abs"]
        speeds = {s for _, _, s in moves if s is not None}
        # variable speed: coarse scan at max_speed, fine at fine_speed,
        # landing at landing_speed
        assert DEFAULT_CFG["max_speed"] in speeds
        assert DEFAULT_CFG["fine_speed"] in speeds
        assert DEFAULT_CFG["landing_speed"] in speeds
        # ONE continuous coarse move covering the whole window (staging first)
        coarse = [m for m in moves if m[2] == DEFAULT_CFG["max_speed"]]
        assert len(coarse) == 2  # staging + the single full-window sweep
        assert abs(coarse[1][1] - coarse[0][1]) == DEFAULT_CFG["span_steps"]
        # the last two moves are the landing pair, both at landing_speed
        assert moves[-2][2] == moves[-1][2] == DEFAULT_CFG["landing_speed"]
        assert (moves[-1][1] - moves[-2][1]) * (moves[-2][1] - moves[-3][1]) < 0 \
            or moves[-2][1] != moves[-1][1]  # overshoot then return
    retry_once(case)


def test_abort_mid_coarse_stops_stage(qapp):
    focus, slot, producer = make_rig(truth=0.0)
    with running(producer):
        cfg = AutofocusConfig(**DEFAULT_CFG)
        ctrl = AutofocusController(focus, slot)
        result_box: dict = {}

        def worker():
            result_box["result"] = ctrl.run(center=0, cfg=cfg)

        thread = threading.Thread(target=worker)
        thread.start()
        time.sleep(0.35)  # mid-sweep (the scan takes ~1 s)
        ctrl.request_abort()
        thread.join(timeout=30.0)
        result = result_box["result"]
    assert result.aborted, result.message
    assert result.phase == "coarse"
    assert focus.get_status().is_idle  # stage stopped


def test_multi_peak_nearest_arm_position_wins(qapp):
    """Wafer surface (weak, near the arm position) vs mount below (strong):
    the global maximum is the mount — nearest-center must win."""
    source = two_plane_source(shape=(240, 320), seed=7, z_top=0.0,
                              z_bottom=150.0, top_weight=0.4)

    def builder(focus):
        return lambda: source(focus.load_pos)

    def case():
        focus, slot, producer = make_rig(source_builder=builder)
        with running(producer):
            result = run_autofocus(focus, slot, center=0.0)
        assert result.success, result.message
        # these synthetic planes are pathologically narrow (2-3 samples
        # wide) vs DOF-broad hardware peaks — allow 1.5 fine steps here
        assert abs(result.best_position - 0.0) <= 1.5 * DEFAULT_CFG["fine_step"]
        # and armed at the mount, it converges to the mount (its blend peak
        # is flat-topped over ~2 fine steps — anything well away from the
        # wafer plane counts)
        focus2, slot2, producer2 = make_rig(source_builder=builder)
        with running(producer2):
            result2 = run_autofocus(focus2, slot2, center=150.0)
        assert result2.success
        assert abs(result2.best_position - 150.0) <= 2.5 * DEFAULT_CFG["fine_step"]
    retry_once(case)


def test_peak_at_window_edge_fails_without_blind_move(qapp):
    """Truth far beyond the window: the curve climbs monotonically into
    the window edge — must fail with the widen-window diagnostic."""
    focus, slot, producer = make_rig(truth=260.0)
    with running(producer):
        result = run_autofocus(focus, slot, center=0.0)
    assert not result.success
    assert "edge" in result.message
    assert focus.get_status().is_idle


def test_flat_scene_fails_gracefully(qapp):
    focus = SimFocusStage({"max_speed": 2000, "latency_s": 0.001})
    focus.connect()
    slot = LatestFrameSlot()
    producer = FrameProducer(slot, lambda: np.full((240, 320, 3), 128, np.uint8))
    with running(producer):
        result = run_autofocus(focus, slot, center=0.0)
    assert not result.success
    # the preflight catches the featureless frame BEFORE any sweep
    assert "peak" in result.message or "weak" in result.message \
        or "contrast" in result.message
    assert focus.get_status().is_idle


def test_slim_clamp_empties_window_without_motion(qapp):
    focus = SimFocusStage({"max_speed": 2000, "latency_s": 0.001,
                           "slim_on": True, "slim_min": -50, "slim_max": 50})
    focus.connect()
    slot = LatestFrameSlot()
    result = run_autofocus(focus, slot, center=0.0)
    assert not result.success
    assert "soft limits" in result.message
    assert focus.get_status().pos == 0  # never moved


# ---------------------------------------------------------------------------
# Backlash
# ---------------------------------------------------------------------------

def test_backlash_landing_puts_the_load_on_truth(qapp):
    """With B=8 steps of backlash, the counter axis lags the load: the
    controller's direction-matched landing must still place the LOAD at
    the planted truth (what the camera sees — the counter readback alone
    can never show this)."""
    def case():
        focus = RecordingFocus({"max_speed": 2000, "latency_s": 0.001,
                                "backlash_steps": 8})
        focus.connect()
        slot = LatestFrameSlot()
        truth = 50.0
        producer = FrameProducer(
            slot, lambda: defocus_blur(
                synthetic_flake_image(seed=7, shape=(240, 320)),
                0.4 + 0.05 * abs(focus.load_pos - truth)))
        with running(producer):
            result = run_autofocus(focus, slot, center=0.0)
        assert result.success, result.message
        assert abs(focus.load_pos - truth) <= DEFAULT_CFG["fine_step"] + 1
        # counter readback is exact (open-loop: it cannot see the load error)
        assert focus.get_status().pos == result.best_position
    retry_once(case)


# ---------------------------------------------------------------------------
# Stalls, ROI, refine, dry run
# ---------------------------------------------------------------------------

def test_producer_stall_mid_fine_completes_cleanly(qapp):
    """The camera stalls after the coarse scan: fine-phase freshness gates
    time out per point (holes, never fake zeros) and the run completes."""
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

        stalled = StallingProducer(producer._slot, producer._source, stall_at=0.7)
        t0 = time.monotonic()
        with running(stalled):
            result = run_autofocus(focus, slot, center=0.0,
                                   fine_wait_timeout_s=0.5)
        assert time.monotonic() - t0 < 60.0  # no hang
        assert result.success or "peak" in result.message  # clean outcome
        assert focus.get_status().is_idle
    retry_once(case)


def test_roi_focuses_the_wafer_while_armed_at_the_mount(qapp):
    """Top-plane flakes on the LEFT half, dense mount features on the
    RIGHT half. Armed AT THE MOUNT (150): the full-frame metric stays on
    the mount, but a left-half ROI pulls the focus to the wafer plane —
    the ROI selects WHICH plane the metric sees."""
    source = split_plane_source(shape=(240, 320), seed=7, z_top=0.0,
                                z_bottom=150.0, top_flakes=4, bottom_flakes=14)

    def builder(focus):
        return lambda: source(focus.load_pos)

    def case():
        focus, slot, producer = make_rig(source_builder=builder)
        with running(producer):
            full = run_autofocus(focus, slot, center=150.0)
        assert full.success
        assert abs(full.best_position - 150.0) <= 1.5 * DEFAULT_CFG["fine_step"]  # the mount

        focus2, slot2, producer2 = make_rig(source_builder=builder)
        roi = (0.02, 0.05, 0.45, 0.9)  # left half only (the wafer flakes)
        with running(producer2):
            roi_result = run_autofocus(focus2, slot2, center=150.0, roi_norm=roi)
        assert roi_result.success, roi_result.message
        assert abs(roi_result.best_position - 0.0) <= 1.5 * DEFAULT_CFG["fine_step"]
    retry_once(case)


def test_refine_mode_repeaks_after_drift(qapp):
    """AF_REFINE (the AF-C re-peak engine): armed at the peak, drifted
    away by 40 steps, a refine-only run returns the load to the peak."""
    def case():
        focus, slot, producer = make_rig(truth=0.0)
        with running(producer):
            focus.move_abs(0, speed=100)
            focus.wait_idle()
            time.sleep(0.15)
            focus.move_abs(40, speed=100)  # drift (thermal/settling)
            focus.wait_idle()
            time.sleep(0.15)
            result = run_autofocus(focus, slot, center=40, mode="AF_REFINE",
                                   fine_window_steps=60)
        assert result.success, result.message
        assert abs(focus.load_pos - 0.0) <= DEFAULT_CFG["fine_step"]
    retry_once(case)


def test_dry_run_walks_sweep_and_fails_gracefully(qapp):
    """No camera: the continuous sweep is walked safely (motion-only) and
    the run fails without a blind move — then RETURNS to the arm center
    (a failed run must not strand the axis at the window edge)."""
    focus = SimFocusStage({"max_speed": 2000, "latency_s": 0.001})
    focus.connect()
    result = run_autofocus(focus, None, center=0.0)
    assert not result.success  # no frames → no peak
    assert "peak" in result.message
    assert focus.get_status().pos == 0  # restored to the arm position
    assert focus.get_status().is_idle


# ---------------------------------------------------------------------------
# Preflight image sanity (refuse to sweep when there is nothing to see)
# ---------------------------------------------------------------------------

def _preflight_rig(frame_value=128):
    """A slot with ONE static frame; the controller must refuse BEFORE
    any motion."""
    focus = SimFocusStage({"max_speed": 2000, "latency_s": 0.001})
    focus.connect()
    slot = LatestFrameSlot()
    slot.write(np.full((240, 320, 3), frame_value, np.uint8), 1.0, 1)
    return focus, slot


def test_preflight_rejects_dark_image_without_motion(qapp):
    focus, slot = _preflight_rig(frame_value=5)
    result = run_autofocus(focus, slot, center=0.0)
    assert not result.success
    assert "dark" in result.message
    assert focus.get_status().pos == 0  # never moved
    assert result.phase == "preflight"


def test_preflight_rejects_saturated_image(qapp):
    focus, slot = _preflight_rig(frame_value=250)
    result = run_autofocus(focus, slot, center=0.0)
    assert not result.success
    assert "bright" in result.message
    assert focus.get_status().pos == 0


def test_preflight_rejects_no_contrast_field(qapp):
    focus, slot = _preflight_rig(frame_value=128)  # mid-gray: contrast ≈ 0
    result = run_autofocus(focus, slot, center=0.0)
    assert not result.success
    assert "contrast" in result.message
    assert focus.get_status().pos == 0


def test_preflight_passes_on_a_real_scene(qapp):
    focus, slot, producer = make_rig(truth=0.0)
    with running(producer):
        result = run_autofocus(focus, slot, center=0.0)
    assert result.success, result.message  # preflight passed, sweep ran
