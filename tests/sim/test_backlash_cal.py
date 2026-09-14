"""Backlash auto-calibration on the sim's backlash model: planted B must
be recovered from the up/down peak offset (real timestamps, real thread
topology — producer / calibrator / focus)."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from talos.cv.backlash_cal import BacklashCalConfig, BacklashCalibrator
from talos.cv.frame_slot import LatestFrameSlot
from tests.testing.sim_images import defocus_blur, synthetic_flake_image
from talos.hal.sim import SimFocusStage

# Real-time closed-loop simulation: the whole file is marked slow
# (deselect with `-m "not slow"` for a fast edit loop).
pytestmark = pytest.mark.slow



@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class Producer(threading.Thread):
    def __init__(self, slot, source_fn, fps=20.0):
        super().__init__(daemon=True)
        self._slot = slot
        self._source = source_fn
        self._fps = fps
        self._stop_event = threading.Event()
        self._seq = 0

    def run(self):
        interval = 1.0 / self._fps
        next_t = time.monotonic() + interval
        while not self._stop_event.is_set():
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

    def stop(self):
        self._stop_event.set()
        self.join(timeout=5.0)


@contextmanager
def running(producer):
    producer.start()
    time.sleep(0.25)
    try:
        yield
    finally:
        producer.stop()


CFG = dict(sweep_steps=150, speed=50, settle_steps=25, poll_s=0.04,
           freshness_ms=800.0, timeout_s=150.0)


def retry_once(fn):
    """Real-time harness guard (see the closed-loop suite): a starved
    producer can empty the curves — retry ONCE with a fresh rig."""
    for attempt in (1, 2):
        try:
            return fn()
        except AssertionError:
            if attempt == 2:
                raise


def make_rig(backlash=8, truth=0.0, source_fn=None, seed=7):
    focus = SimFocusStage({"max_speed": 2000, "latency_s": 0.001,
                           "backlash_steps": backlash})
    focus.connect()
    if source_fn is None:
        sharp = synthetic_flake_image(seed=seed, shape=(240, 320), noise=0.0)

        def source_fn():
            sigma = 0.4 + 0.05 * abs(focus.load_pos - truth)
            return defocus_blur(sharp, sigma)
    slot = LatestFrameSlot()
    return focus, slot, Producer(slot, source_fn)


def run_cal(focus, slot, center=0, **cfg_overrides):
    cal = BacklashCalibrator(focus, slot)
    return cal.run(center=center, cfg=BacklashCalConfig(**{**CFG, **cfg_overrides}))


def test_recovers_planted_backlash(qapp):
    def case():
        focus, slot, producer = make_rig(backlash=8, truth=0.0)
        with running(producer):
            result = run_cal(focus, slot, center=0)
        assert result.success, result.message
        assert abs(result.backlash_steps - 8) <= 1
        assert result.backlash_um == pytest.approx(8 * 0.2, abs=0.3)
        assert len(result.up_curve) >= 5 and len(result.down_curve) >= 5
    retry_once(case)


def test_zero_backlash_when_none_planted(qapp):
    def case():
        focus, slot, producer = make_rig(backlash=0, truth=30.0)
        with running(producer):
            result = run_cal(focus, slot, center=30)
        assert result.success, result.message
        assert result.backlash_steps <= 1
    retry_once(case)


def test_flat_scene_fails_gracefully(qapp):
    def case():
        focus, slot, producer = make_rig(
            source_fn=lambda: np.full((240, 320, 3), 128, np.uint8))
        with running(producer):
            result = run_cal(focus, slot, center=0)
        assert not result.success
        assert "flat" in result.message
        assert focus.get_status().is_idle
    retry_once(case)


def test_abort_mid_sweep_stops_stage(qapp):
    focus, slot, producer = make_rig(backlash=8, truth=0.0)
    with running(producer):
        cal = BacklashCalibrator(focus, slot)
        result_box: dict = {}

        def worker():
            result_box["result"] = cal.run(
                center=0, cfg=BacklashCalConfig(**CFG))

        thread = threading.Thread(target=worker)
        thread.start()
        time.sleep(0.5)  # mid first sweep
        cal.request_abort()
        thread.join(timeout=30.0)
        result = result_box["result"]
    assert result.aborted, result.message
    assert focus.get_status().is_idle
