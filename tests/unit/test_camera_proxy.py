"""CameraProxy tests with stub cameras (no real threads — _run/_tick are
called directly; QTimer creation is inert without an event loop).

Signals are collected with a QObject receiver: with the proxy's thread
affinity moved, lambda connections get posted to the (never-started)
worker thread's queue and are lost — bound slots queue to the receiver's
(main) thread instead, and flush() spins a loop to deliver them.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEventLoop, QObject, QTimer, Slot

from talos.hal.proxies.camera_proxy import CameraProxy


@pytest.fixture(scope="session", autouse=True)
def _qapp():
    """An event-loop host is required for event-loop delivery (queued
    signal emissions to main-thread receivers). A full QApplication is
    created — NOT QCoreApplication — because when this file runs before
    the QWidget test files, a bare core app would abort them."""
    from PySide6.QtWidgets import QApplication
    app = QCoreApplication.instance() or QApplication([])
    yield app


class Collector(QObject):
    """Signal receiver (main thread): records emissions into public lists."""

    def __init__(self):
        super().__init__()
        self.connected: list[bool] = []
        self.frames: list[object] = []
        self.props: list[dict] = []
        self.events: list[tuple] = []
        self.done: list[tuple] = []
        self.failed: list[tuple] = []

    @Slot(bool)
    def on_connected(self, ok: bool) -> None:
        self.connected.append(ok)

    @Slot(object)
    def on_frame(self, frame) -> None:
        self.frames.append(frame)

    @Slot(dict)
    def on_props(self, props: dict) -> None:
        self.props.append(props)

    @Slot(str, dict)
    def on_event(self, event: str, payload: dict) -> None:
        self.events.append((event, payload))

    @Slot(int, object)
    def on_command_done(self, job_id: int, result) -> None:
        self.done.append((job_id, result))

    @Slot(int, str, str)
    def on_command_failed(self, job_id: int, kind: str, message: str) -> None:
        self.failed.append((job_id, kind, message))


def flush(ms: int = 50) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


class StubCamera:
    """Minimal Camera stand-in with scriptable behaviors."""

    def __init__(self, name="stub", connect_fail=None, start_fail=None):
        self._name = name
        self.connect_fail = connect_fail
        self.start_fail = start_fail
        self.connected = False
        self.started = False
        self.stopped = 0
        self.disconnected = 0
        self.frames = 0
        self.props = {"backend": name, "exposure_us": 20000.0}
        self.set_calls: list[tuple] = []

    @property
    def device_id(self):
        return f"camera@{self._name}"

    def connect(self):
        if self.connect_fail:
            raise self.connect_fail
        self.connected = True

    def disconnect(self):
        self.connected = False
        self.disconnected += 1

    def start(self):
        if self.start_fail:
            raise self.start_fail
        self.started = True

    def stop(self):
        self.started = False
        self.stopped += 1

    def fetch(self, timeout_ms=2000.0):
        self.frames += 1
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def get_properties(self):
        return dict(self.props)

    def set_property(self, name, value):
        self.set_calls.append((name, value))

    def snapshot(self, path, timeout_s=15.0, resolution=None, burn=None):
        return getattr(self, "snapshot_result", None)


def make_proxy(factory) -> CameraProxy:
    proxy = CameraProxy(factory)
    collector = Collector()
    proxy.sig_connected.connect(collector.on_connected)
    proxy.sig_frame.connect(collector.on_frame)
    proxy.sig_properties.connect(collector.on_props)
    proxy.sig_event.connect(collector.on_event)
    proxy.sig_command_done.connect(collector.on_command_done)
    proxy.sig_command_failed.connect(collector.on_command_failed)
    proxy._events = collector
    return proxy


def _drain(proxy: CameraProxy) -> None:
    """Deliver queued signal emissions."""
    flush()


def test_chain_falls_back_to_second_candidate():
    from talos.hal.base import DeviceConnectionError
    first = StubCamera("a", connect_fail=DeviceConnectionError("nope"))
    second = StubCamera("b")
    proxy = make_proxy(lambda: [first, second])
    proxy._run()
    _drain(proxy)
    assert proxy._camera is second
    assert proxy._events.connected == [True]
    assert first.disconnected == 1  # never abandoned half-connected
    proxy.request_shutdown()


def test_all_candidates_fail_emits_connect_failed():
    from talos.hal.base import DeviceConnectionError
    proxy = make_proxy(lambda: [StubCamera(connect_fail=DeviceConnectionError("x"))])
    proxy._run()
    _drain(proxy)
    assert proxy._events.connected == [False]
    assert proxy._events.events[0][0] == "connect_failed"


def test_factory_error_emits_connect_failed():
    def boom():
        raise KeyError("Unknown camera backend: 'nope'")
    proxy = make_proxy(boom)
    proxy._run()
    _drain(proxy)
    assert proxy._events.connected == [False]
    assert proxy._events.events[0][0] == "connect_failed"


def test_start_failure_disconnects_candidate():
    from talos.hal.base import DeviceConnectionError
    bad = StubCamera(start_fail=DeviceConnectionError("no stream"))
    proxy = make_proxy(lambda: [bad])
    proxy._run()
    _drain(proxy)
    assert proxy._events.connected == [False]
    assert bad.disconnected == 1


def test_shutdown_flag_during_connect_window():
    proxy = make_proxy(lambda: [StubCamera()])
    proxy.request_shutdown()
    proxy._run()  # worker loop would check the flag after connect
    _drain(proxy)
    assert proxy._events.connected == [False]
    assert proxy._camera is None


def test_get_properties_normalized_and_emitted():
    cam = StubCamera("smartcam")
    cam.props = {"backend": "smartcam", "exposure_us": 20000.0,
                 "white_balance": 1, "resolution": [1920, 1080]}
    proxy = make_proxy(lambda: [cam])
    proxy._run()
    proxy.enqueue(7, "get_properties", ())
    proxy._tick()
    _drain(proxy)
    assert proxy._events.props
    props = proxy._events.props[-1]
    assert props["white_balance"] == "Continuous"  # canonical string
    assert props["exposure_us"] == 20000.0
    proxy.request_shutdown()


def test_set_property_mapped_to_native_names():
    cam = StubCamera("mmcore")
    proxy = make_proxy(lambda: [cam])
    proxy._run()
    proxy.enqueue(1, "set_property", ("exposure_us", 50000.0))
    proxy._tick()
    _drain(proxy)
    assert ("exposure_ms", 50.0) in cam.set_calls  # canonical us -> native ms
    proxy.enqueue(2, "set_property", ("gain", 4.0))
    proxy._tick()
    _drain(proxy)
    assert ("gain", 4.0) in cam.set_calls
    proxy.request_shutdown()


def test_streaming_state_machine_starts_and_stops_backend():
    cam = StubCamera("smartcam")
    proxy = make_proxy(lambda: [cam])
    proxy._run()
    proxy.set_streaming(True)
    proxy._tick()
    _drain(proxy)
    assert cam.started
    proxy._tick()
    _drain(proxy)
    assert len(proxy._events.frames) == 2  # start+fetch tick 1, fetch tick 2
    proxy.set_streaming(False)
    proxy._tick()
    _drain(proxy)
    assert not cam.started
    assert cam.stopped == 1
    proxy.request_shutdown()


def test_command_errors_are_captured():
    from talos.hal.base import CommandRejectedError

    def reject(name, value):
        raise CommandRejectedError("rejected")
    cam = StubCamera("smartcam")
    cam.set_property = reject
    proxy = make_proxy(lambda: [cam])
    proxy._run()
    proxy.enqueue(1, "set_property", ("gain", 99.0))
    proxy._tick()
    _drain(proxy)
    assert any(e == "command_error" for e, _ in proxy._events.events)
    proxy.request_shutdown()


def test_snapshot_job_emits_command_done_with_result():
    """The UI's snapshot busy-gating depends on the camera job-completion
    contract (sig_command_done relays to the manager's sig_job_done)."""
    cam = StubCamera("smartcam")
    cam.snapshot_result = "C:\\snaps\\x.png"
    proxy = make_proxy(lambda: [cam])
    proxy._run()
    proxy.enqueue(42, "snapshot", ("C:\\snaps\\x.png",))
    proxy._tick()
    _drain(proxy)
    assert proxy._events.done == [(42, "C:\\snaps\\x.png")]
    assert proxy._events.failed == []
    proxy.request_shutdown()


def test_snapshot_failure_emits_command_failed():
    cam = StubCamera("smartcam")

    def boom(*args):
        raise RuntimeError("snap failed")
    cam.snapshot = boom
    proxy = make_proxy(lambda: [cam])
    proxy._run()
    proxy.enqueue(7, "snapshot", ("x.png",))
    proxy._tick()
    _drain(proxy)
    assert proxy._events.failed
    assert proxy._events.failed[0][0] == 7
    assert proxy._events.failed[0][1] == "RuntimeError"
    proxy.request_shutdown()


def test_fetch_exception_is_captured():
    def boom(timeout_ms=1000.0):
        raise RuntimeError("usb gone")
    cam = StubCamera("smartcam")
    cam.fetch = boom
    proxy = make_proxy(lambda: [cam])
    proxy._run()
    proxy.set_streaming(True)
    proxy._tick()  # start backend
    proxy._tick()  # fetch fails -> frame_error
    _drain(proxy)
    assert any(e == "frame_error" for e, _ in proxy._events.events)
    proxy.request_shutdown()


def test_frame_slot_receives_timestamps_and_seq():
    from talos.cv.frame_slot import LatestFrameSlot
    cam = StubCamera("smartcam")
    cam.capture_time = lambda: 1000.0  # fixed capture-side timestamps
    proxy = make_proxy(lambda: [cam])
    slot = LatestFrameSlot()
    proxy.set_frame_slot(slot)
    proxy._run()
    proxy.set_streaming(True)
    proxy._tick()  # start backend + fetch (seq 1)
    proxy._tick()  # fetch (seq 2)
    proxy._tick()  # fetch (seq 3)
    _drain(proxy)
    frame, meta = slot.read()
    assert meta.seq == 3
    assert meta.t_capture == 1000.0
    assert meta.shape == (8, 8, 3)
    # sig_frame payload unchanged for the GUI consumers
    assert len(proxy._events.frames) == 3
    proxy.request_shutdown()


def test_missing_capture_time_falls_back_to_monotonic():
    from talos.cv.frame_slot import LatestFrameSlot
    cam = StubCamera("smartcam")  # no capture_time attribute at all
    proxy = make_proxy(lambda: [cam])
    slot = LatestFrameSlot()
    proxy.set_frame_slot(slot)
    proxy._run()
    proxy.set_streaming(True)
    proxy._tick()  # start backend + fetch
    proxy._tick()  # fetch
    _drain(proxy)
    import time
    t_after = time.monotonic()
    _, meta = slot.read()
    assert meta.seq == 2
    assert meta.t_capture <= t_after  # produced before now
    assert len(proxy._events.frames) == 2
    proxy.request_shutdown()
