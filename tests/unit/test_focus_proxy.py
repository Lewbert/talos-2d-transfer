"""FocusProxy: the exclusive autofocus job — closed-loop through the
proxy's job queue (SimFocusStage + a producer thread feeding the frame
slot), abort paths, and real-thread teardown."""

from __future__ import annotations

import threading
import time

import pytest
from PySide6.QtCore import QEventLoop, QMetaObject, QObject, Qt, QTimer, Slot
from PySide6.QtWidgets import QApplication

from talos.cv.autofocus import AutofocusConfig, AutofocusRequest
from talos.cv.frame_slot import LatestFrameSlot
from tests.testing.sim_images import defocus_blur, synthetic_flake_image
from talos.hal.proxies.device_proxy import PRIORITY_NORMAL
from talos.hal.proxies.focus_proxy import FocusProxy
from talos.hal.sim import SimFocusStage


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class Collector(QObject):
    def __init__(self):
        super().__init__()
        self.af_done: list[object] = []
        self.af_log: list[str] = []
        self.done: list[tuple[int, object]] = []

    @Slot(object)
    def on_af_done(self, result) -> None:
        self.af_done.append(result)

    @Slot(str)
    def on_af_log(self, message: str) -> None:
        self.af_log.append(message)

    @Slot(int, object)
    def on_done(self, job_id: int, result) -> None:
        self.done.append((job_id, result))


def flush(ms: int = 50) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


class RecordingFocus(SimFocusStage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls: list[tuple] = []

    def move_abs(self, position, speed=None):
        self.calls.append(("move_abs", int(position), speed))
        super().move_abs(position, speed)

    def set_speed(self, speed):
        self.calls.append(("set_speed", int(speed)))
        super().set_speed(speed)

    def stop(self):
        self.calls.append(("stop",))
        super().stop()


class Producer(threading.Thread):
    def __init__(self, slot, focus, truth, seed=7):
        super().__init__(daemon=True)
        self._slot = slot
        self._focus = focus
        self._truth = truth
        self._sharp = synthetic_flake_image(seed=seed, shape=(240, 320), noise=0.0)
        self._stop_event = threading.Event()
        self._seq = 0

    def run(self):
        interval = 0.05
        next_t = time.monotonic() + interval
        while not self._stop_event.is_set():
            t_cap = time.monotonic()
            sigma = 0.4 + 0.05 * abs(self._focus.load_pos - self._truth)
            self._slot.write(defocus_blur(self._sharp, sigma), t_cap, self._seq)
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


# Fast enough for unit tests (span 200, sweep ~0.7 s), still >10 frames.
CFG = dict(coarse_step=30, fine_step=8, span_steps=200, max_speed=300,
           fine_speed=100, landing_speed=50, quality_threshold=0.05,
           timeout_s=60.0, freshness_ms=800.0, coarse_poll_s=0.04,
           settle_frames=1, fail_on_edge_peak=True, metric="tenengrad")


def make_proxy(focus, slot):
    proxy = FocusProxy("focus", lambda: focus, poll_interval_ms=100)
    proxy._driver = focus  # skip real thread start
    proxy._frame_slot = slot
    collector = Collector()
    proxy.sig_af_done.connect(collector.on_af_done)
    proxy.sig_af_log.connect(collector.on_af_log)
    proxy.sig_command_done.connect(collector.on_done)
    proxy._events = collector
    return proxy, collector


def submit_af(proxy, center=0, **cfg_overrides):
    cfg = AutofocusConfig(**{**CFG, **cfg_overrides})
    request = AutofocusRequest(center_steps=center, config=cfg)
    proxy.enqueue(1, "autofocus", (request,), PRIORITY_NORMAL)
    return request


def test_autofocus_job_runs_v3_closed_loop_and_lands(qapp):
    """The proxy's exclusive job now runs the v3 adaptive controller
    directly (the strategy registry is gone — the app has ONE strategy)."""
    focus = RecordingFocus({"max_speed": 2000, "latency_s": 0.001})
    focus.connect()
    slot = LatestFrameSlot()
    producer = Producer(slot, focus, truth=40.0)
    producer.start()
    time.sleep(0.25)
    proxy, collector = make_proxy(focus, slot)
    try:
        # v3 thresholds pinned well above this rig's probe curvature
        # (≈ 1.9e-4) so the probe says "far" and the full v2 pipeline
        # runs through the v3 class (coarse → hill → lock → land).
        submit_af(proxy, center=0, probe_curv_in=0.5, probe_curv_out=0.5)
        proxy._drain()  # blocking run, test thread
        assert collector.af_done, "no sig_af_done"
        result = collector.af_done[-1]
        assert result.success, result.message
        assert abs(result.best_position - 40.0) <= CFG["fine_step"]
        # the job result is reported through the normal job accounting too
        assert (1, result) in collector.done
        # call sequence: probe staging at stage_speed, the adaptive CONT
        # passes via set_speed, landing pair at landing_speed
        moves = [c for c in focus.calls if c[0] == "move_abs"]
        assert any(m[2] == 2000 for m in moves)  # staging (stage_speed default)
        assert moves[-2][2] == moves[-1][2] == CFG["landing_speed"]
        assert any(c[0] == "set_speed" for c in focus.calls)  # coarse/hill CONT
    finally:
        producer.stop()


def test_request_abort_before_job_start_latches_and_aborts(qapp):
    """Regression (audit-found): an abort landing between the job's
    enqueue and the worker's controller construction used to be a silent
    no-op — the AF ran to completion. The proxy now latches it and the
    controller's _check polls the latch (run() resets the local flag at
    entry, so a pre-run flag set would be lost)."""
    focus = SimFocusStage({"max_speed": 2000, "latency_s": 0.001})
    focus.connect()
    slot = LatestFrameSlot()
    producer = Producer(slot, focus, truth=0.0)
    producer.start()
    time.sleep(0.25)
    proxy, collector = make_proxy(focus, slot)
    try:
        submit_af(proxy, center=0)
        # the job is queued but the worker has not drained — the abort
        # can only latch
        proxy.request_abort("aborted by stage motion input")
        proxy._drain()
        assert collector.af_done, "no sig_af_done"
        result = collector.af_done[-1]
        assert result.aborted, result.message
        assert result.message == "aborted by stage motion input"
        assert focus.get_status().is_idle
    finally:
        producer.stop()


def test_autofocus_job_crash_still_emits_done(qapp, monkeypatch):
    """Regression (audit-found): an exception escaping run()'s handlers
    used to leave the service stuck in AUTOFOCUS forever (sig_af_done
    never fired; nothing consumes sig_command_failed)."""
    focus = SimFocusStage({"max_speed": 2000, "latency_s": 0.001})
    focus.connect()
    slot = LatestFrameSlot()
    producer = Producer(slot, focus, truth=0.0)
    producer.start()
    time.sleep(0.25)
    proxy, collector = make_proxy(focus, slot)

    def boom(self, center=None, cfg=None, bounds=None):
        raise RuntimeError("simulated crash inside run()")

    monkeypatch.setattr(
        "talos.hal.proxies.focus_proxy.AdaptiveAutofocusController.run", boom)
    try:
        submit_af(proxy, center=0)
        proxy._drain()
        assert collector.af_done, "no sig_af_done after the crash"
        result = collector.af_done[-1]
        assert not result.success
        assert not result.aborted
        assert "autofocus crashed" in result.message
    finally:
        producer.stop()


def test_request_abort_from_other_thread(qapp):
    focus = SimFocusStage({"max_speed": 2000, "latency_s": 0.001})
    focus.connect()
    slot = LatestFrameSlot()
    producer = Producer(slot, focus, truth=0.0)
    producer.start()
    time.sleep(0.25)
    proxy, collector = make_proxy(focus, slot)
    try:
        submit_af(proxy, center=0)
        thread = threading.Thread(target=proxy._drain)
        thread.start()
        time.sleep(0.4)  # mid-coarse (sweep ~0.7 s)
        proxy.request_abort()  # plain method, any thread
        thread.join(timeout=30.0)
        flush()  # af_done was emitted from the helper thread → queued
        assert collector.af_done
        result = collector.af_done[-1]
        assert result.aborted, result.message
        assert focus.get_status().is_idle
    finally:
        producer.stop()


def test_enqueue_stop_aborts_running_job(qapp):
    focus = SimFocusStage({"max_speed": 2000, "latency_s": 0.001})
    focus.connect()
    slot = LatestFrameSlot()
    producer = Producer(slot, focus, truth=0.0)
    producer.start()
    time.sleep(0.25)
    proxy, collector = make_proxy(focus, slot)
    try:
        submit_af(proxy, center=0)
        thread = threading.Thread(target=proxy._drain)
        thread.start()
        time.sleep(0.4)
        proxy.enqueue_stop()  # STOP ALL path: queue stop + abort the job
        thread.join(timeout=30.0)
        flush()
        assert collector.af_done
        assert collector.af_done[-1].aborted
        assert focus.get_status().is_idle
    finally:
        producer.stop()


def test_real_thread_lifecycle_and_shutdown(qapp):
    """Full thread path: start → connected → AF job in the worker thread →
    queued request_shutdown → clean quit."""
    focus = SimFocusStage({"max_speed": 2000, "latency_s": 0.001})
    focus.connect()
    slot = LatestFrameSlot()
    producer = Producer(slot, focus, truth=0.0)
    producer.start()
    time.sleep(0.25)
    proxy = FocusProxy("focus", lambda: focus, poll_interval_ms=100)
    proxy._frame_slot = slot
    collector = Collector()
    proxy.sig_af_done.connect(collector.on_af_done)
    try:
        proxy.start()
        flush(200)  # thread starts, _run connects the driver
        assert proxy._thread.isRunning()
        submit_af(proxy, center=0)
        deadline = time.monotonic() + 30.0
        while not collector.af_done and time.monotonic() < deadline:
            flush(100)
        assert collector.af_done, "AF job never finished on the worker thread"
        assert collector.af_done[-1].success
        QMetaObject.invokeMethod(proxy, "request_shutdown",
                                 Qt.ConnectionType.QueuedConnection)
        assert proxy.wait(3000), "shutdown wait timed out"
        assert not proxy._thread.isRunning()
    finally:
        producer.stop()
