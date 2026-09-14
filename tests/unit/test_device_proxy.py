"""DeviceProxy tests: command coalescing, poll-vs-command exclusivity,
poll backoff. No real threads — _drain/_poll are called directly."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QApplication

from talos.hal.base import DeviceError
from talos.hal.proxies.device_proxy import (
    PRIORITY_NORMAL,
    PRIORITY_STOP,
    DeviceProxy,
)


@pytest.fixture(scope="session", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class Collector(QObject):
    def __init__(self):
        super().__init__()
        self.done: list[tuple[int, object]] = []
        self.failed: list[tuple] = []
        self.events: list[tuple] = []
        self.telems: list[dict] = []

    @Slot(int, object)
    def on_done(self, job_id: int, result) -> None:
        self.done.append((job_id, result))

    @Slot(int, str, str)
    def on_failed(self, job_id: int, exc_type: str, message: str) -> None:
        self.failed.append((job_id, exc_type, message))

    @Slot(str, dict)
    def on_event(self, event: str, payload: dict) -> None:
        self.events.append((event, payload))

    @Slot(dict)
    def on_telem(self, payload: dict) -> None:
        self.telems.append(payload)


def _flush(ms: int = 50) -> None:
    """Deliver queued signal emissions to main-thread receivers."""
    from PySide6.QtCore import QEventLoop, QTimer
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


class FakeDriver:
    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []
        self.status_fail = 0

    def connect(self):
        self.calls.append(("connect", ()))

    def disconnect(self):
        self.calls.append(("disconnect", ()))

    def stop(self):
        self.calls.append(("stop", ()))

    def set_speed(self, *args):
        self.calls.append(("set_speed", args))
        return args[-1] if args else None

    def move_continuous(self, *args):
        self.calls.append(("move_continuous", args))
        return args[-1] if args else None

    def move_rel(self, steps):
        self.calls.append(("move_rel", (steps,)))
        return steps

    def move(self, *args):
        self.calls.append(("move", args))
        return args[-1] if args else None

    def step(self, *args):
        self.calls.append(("step", args))
        return args[-1] if args else None

    def drain_events(self):
        return []

    def get_status(self):
        if self.status_fail:
            self.status_fail -= 1
            raise DeviceError("garbled")
        self.calls.append(("get_status", ()))
        return "ok"

    def get_position(self):
        self.calls.append(("get_position", ()))
        return "pos"

    def get_soft_limits(self):
        return None

    def read_pv(self):
        return None

    def read_sv(self):
        return None

    def read_output_percent(self):
        return None

    def calls_of(self, name: str) -> list[tuple]:
        return [args for n, args in self.calls if n == name]


@pytest.fixture
def rig():
    driver = FakeDriver()
    proxy = DeviceProxy("focus", lambda: driver, poll_interval_ms=100)
    collector = Collector()
    proxy.sig_command_done.connect(collector.on_done)
    proxy.sig_command_failed.connect(collector.on_failed)
    proxy.sig_event.connect(collector.on_event)
    proxy._driver = driver  # skip real thread start
    from PySide6.QtCore import QTimer
    proxy._poll_timer = QTimer(proxy)
    proxy._poll_timer.setInterval(100)
    return proxy, driver, collector


def test_coalesce_queued_speed_commands(rig):
    proxy, driver, collector = rig
    proxy.enqueue(1, "set_speed", (100,), PRIORITY_NORMAL)
    proxy.enqueue(2, "set_speed", (200,), PRIORITY_NORMAL)
    proxy.enqueue(3, "set_speed", (300,), PRIORITY_NORMAL)
    proxy._drain()
    assert driver.calls_of("set_speed") == [(300,)]
    # superseded jobs complete with None so job accounting stays consistent
    assert (1, None) in collector.done and (2, None) in collector.done
    assert (3, 300) in collector.done


def test_non_coalesced_moves_keep_fifo(rig):
    proxy, driver, _ = rig
    proxy.enqueue(1, "move_rel", (10,), PRIORITY_NORMAL)
    proxy.enqueue(2, "move_rel", (-20,), PRIORITY_NORMAL)
    proxy._drain()
    assert driver.calls_of("move_rel") == [(10,), (-20,)]


def _sigmakoki_rig():
    """A proxy wired to the sigmakoki coalesce table (the shared `rig`
    fixture is keyed to focus)."""
    from talos.hal.base import Axis, Direction

    driver = FakeDriver()
    proxy = DeviceProxy("sigmakoki", lambda: driver, poll_interval_ms=100)
    collector = Collector()
    proxy.sig_command_done.connect(collector.on_done)
    proxy._driver = driver
    return proxy, driver, collector, Axis, Direction


def test_sigmakoki_continuous_moves_coalesce_per_axis():
    """An analog-stick hold re-emits MV every 16 ms tick; only the newest
    move per axis is worth executing — a queue of stale moves made the
    stage act on positions the operator had already left."""
    proxy, driver, collector, Axis, Direction = _sigmakoki_rig()
    proxy.enqueue(1, "move", (Axis.X, Direction.POSITIVE, 0), PRIORITY_NORMAL)
    proxy.enqueue(2, "move", (Axis.X, Direction.POSITIVE, 3), PRIORITY_NORMAL)
    proxy._drain()
    assert driver.calls_of("move") == [(Axis.X, Direction.POSITIVE, 3)]
    assert (1, None) in collector.done  # superseded job still completes
    assert (2, 3) in collector.done


def test_sigmakoki_coalescing_keeps_axes_independent():
    proxy, driver, _, Axis, Direction = _sigmakoki_rig()
    proxy.enqueue(1, "move", (Axis.X, Direction.POSITIVE, 2), PRIORITY_NORMAL)
    proxy.enqueue(2, "move", (Axis.Y, Direction.NEGATIVE, 4), PRIORITY_NORMAL)
    proxy._drain()
    assert driver.calls_of("move") == [(Axis.X, Direction.POSITIVE, 2),
                                       (Axis.Y, Direction.NEGATIVE, 4)]


def test_sigmakoki_single_steps_are_not_coalesced():
    """Discrete steps are intentional commands — every one must execute."""
    proxy, driver, _, Axis, Direction = _sigmakoki_rig()
    proxy.enqueue(1, "step", (Axis.X, Direction.POSITIVE, 3), PRIORITY_NORMAL)
    proxy.enqueue(2, "step", (Axis.X, Direction.POSITIVE, 3), PRIORITY_NORMAL)
    proxy._drain()
    assert len(driver.calls_of("step")) == 2


def test_poll_uses_one_combined_telemetry_read():
    """A driver that exposes get_telemetry must be polled ONCE per cycle
    (sigmakoki's STATUS? already carries the position)."""

    class TelemetryDriver(FakeDriver):
        def __init__(self):
            super().__init__()
            self.telemetry_calls = 0

        def get_telemetry(self):
            self.telemetry_calls += 1
            return {"status": "ok", "position": "pos"}

    driver = TelemetryDriver()
    proxy = DeviceProxy("sigmakoki", lambda: driver, poll_interval_ms=100)
    collector = Collector()
    proxy.sig_telem.connect(collector.on_telem)
    proxy._driver = driver
    proxy._poll()
    assert driver.telemetry_calls == 1
    assert driver.calls_of("get_status") == []
    assert driver.calls_of("get_position") == []
    assert collector.telems[-1]["status"] == "ok"
    assert collector.telems[-1]["position"] == "pos"


def test_stop_purges_stale_motion_commands(rig):
    """A stop must not be followed by queued motion commands — they would
    re-start the axis right after the stop (hardware-verified: released
    the trigger but the focus kept jogging). Stale commands are purged
    and their jobs complete as superseded."""
    proxy, driver, collector = rig
    proxy.enqueue(1, "set_speed", (100,), PRIORITY_NORMAL)
    proxy.enqueue(2, "move_rel", (10,), PRIORITY_NORMAL)
    proxy.enqueue_stop()
    proxy._drain()
    assert [n for n, _ in driver.calls] == ["stop"]  # nothing else ran
    assert (1, None) in collector.done and (2, None) in collector.done


def test_plain_stop_enqueue_purges_stale_continuous_motion(rig):
    """Regression (the AF input-abort hazard): a quick tap queues
    set_speed + the release stop behind the blocking AF job; the stop
    sorts FIRST and the stale set_speed would re-start the axis after
    it. A plain release-stop enqueue must purge queued CONTINUOUS
    motion (discrete moves like single steps are intentional)."""
    proxy, driver, collector = rig
    proxy.enqueue(1, "set_speed", (200,), PRIORITY_NORMAL)
    proxy.enqueue(2, "move_rel", (10,), PRIORITY_NORMAL)  # a single step
    proxy.enqueue(3, "stop", (), PRIORITY_STOP)
    proxy._drain()
    # the stop ran FIRST (PRIORITY_STOP), the continuous set_speed was
    # purged, and the discrete move survives (it is intentional, not a
    # stale jog)
    assert [n for n, _ in driver.calls] == ["stop", "move_rel"]
    assert (1, None) in collector.done  # the purged jog: superseded
    assert (2, 10) in collector.done


def test_stop_axis_purges_queued_continuous_on_other_axes_too(rig):
    """zolix stop_axis is also a release-stop: queued move_continuous
    commands must not restart the axis after it."""
    from talos.hal.proxies.device_proxy import DeviceProxy
    driver = FakeDriver()
    driver.stop_axis = lambda *a: driver.calls.append(("stop_axis", a))
    proxy = DeviceProxy("zolix", lambda: driver, poll_interval_ms=100)
    proxy._driver = driver
    proxy.enqueue(1, "move_continuous", ("x", 1), PRIORITY_NORMAL)
    proxy.enqueue(2, "stop_axis", ("x",), PRIORITY_STOP)
    proxy._drain()
    assert driver.calls_of("move_continuous") == []
    assert [n for n, _ in driver.calls] == ["stop_axis"]


def test_zolix_diagonal_axes_are_independent(rig):
    # zolix diagonal = two move_continuous commands (X and Y) — coalescing
    # must NOT drop one axis (regression: user-reported loss of diagonal).
    from talos.hal.proxies.device_proxy import DeviceProxy
    driver = FakeDriver()
    proxy = DeviceProxy("zolix", lambda: driver, poll_interval_ms=100)
    proxy._driver = driver
    proxy.enqueue(1, "move_continuous", ("x", 1), PRIORITY_NORMAL)
    proxy.enqueue(2, "move_continuous", ("y", 1), PRIORITY_NORMAL)
    proxy.enqueue(3, "move_continuous", ("x", -1), PRIORITY_NORMAL)
    proxy._drain()
    # X coalesced to the newest; Y preserved
    assert driver.calls_of("move_continuous") == [("y", 1), ("x", -1)]


def test_sigmakoki_speeds_are_per_axis(rig):
    from talos.hal.proxies.device_proxy import DeviceProxy
    driver = FakeDriver()
    proxy = DeviceProxy("sigmakoki", lambda: driver, poll_interval_ms=100)
    proxy._driver = driver
    proxy.enqueue(1, "set_speed", ("x", 3), PRIORITY_NORMAL)
    proxy.enqueue(2, "set_speed", ("y", 4), PRIORITY_NORMAL)
    proxy.enqueue(3, "set_speed", ("x", 5), PRIORITY_NORMAL)
    proxy._drain()
    assert driver.calls_of("set_speed") == [("y", 4), ("x", 5)]


def test_poll_skips_while_commands_pending(rig):
    proxy, driver, _ = rig
    proxy.enqueue(1, "set_speed", (100,), PRIORITY_NORMAL)  # not drained yet
    proxy._poll()
    assert driver.calls_of("get_status") == []
    assert driver.calls_of("get_position") == []
    proxy._drain()
    proxy._poll()
    assert driver.calls_of("get_status") == [()]


def test_poll_backoff_on_serial_failures(rig):
    proxy, driver, _ = rig
    driver.status_fail = 3
    for _ in range(3):
        proxy._poll()
    assert proxy._poll_timer.interval() == 200  # doubled after 3 failures
    driver.status_fail = 0
    proxy._poll()
    assert proxy._poll_timer.interval() == 100  # restored on success
    assert driver.calls_of("get_status") == [()]


def test_missing_optional_capabilities_do_not_fail_poll():
    # Regression: focus has no get_position, only the yudian has
    # read_pv/read_sv/read_output_percent — each AttributeError used to mark
    # the whole poll failed and drive backoff to the 2000 ms cap, freezing
    # telemetry at one update per 2 s.
    from PySide6.QtCore import QTimer
    from talos.hal.proxies.device_proxy import DeviceProxy

    class BareFocusDriver:
        def connect(self):
            pass

        def stop(self):
            pass

        def disconnect(self):
            pass

        def drain_events(self):
            return []

        def get_status(self):
            return "ok"
        # deliberately NO get_position / read_pv / read_sv /
        # read_output_percent / get_soft_limits

    driver = BareFocusDriver()
    proxy = DeviceProxy("focus", lambda: driver, poll_interval_ms=100)
    proxy._driver = driver
    proxy._poll_timer = QTimer(proxy)
    proxy._poll_timer.setInterval(100)
    collector = Collector()
    proxy.sig_telem.connect(collector.on_telem)
    for _ in range(5):
        proxy._poll()
    _flush()
    payloads = collector.telems
    assert proxy._poll_timer.interval() == 100  # NO backoff
    assert payloads[-1]["status"] == "ok"
    assert payloads[-1]["device"] == "focus"
    # absent capabilities are reported as None, never as failures
    assert payloads[-1]["pv"] is None
    assert payloads[-1]["sv"] is None
    assert payloads[-1]["output_percent"] is None
    assert "position" not in payloads[-1]
    assert "slim_bounds" not in payloads[-1]
