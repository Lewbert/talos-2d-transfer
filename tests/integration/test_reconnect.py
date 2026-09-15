"""Device reconnect: connection edits rebuild the driver safely.

Preferences → Hardware → Apply reconnects every device whose port /
baudrate / slave address / timeout changed. The hard parts these tests
pin: the retired proxy must not speak for the live one, queued jobs must
FAIL (not silently vanish) so no waiter blocks on a timeout, and the
shared frame slot / enable gate must survive the swap.
"""

import gc

import pytest
from PySide6.QtCore import QEventLoop, QObject, QTimer
from PySide6.QtWidgets import QApplication

from talos.hal.proxies import FocusProxy
from talos.hal.registry import DEVICE_KEYS
from talos.instruments import InstrumentManager

_PORTS = {"zolix": "COM3", "sigmakoki": "COM6", "focus": "COM10",
          "yudian": "COM5"}


class StubSettings:
    def __init__(self):
        self.data = {key: {"port": port, "baudrate": 115200, "enabled": True,
                           "url": "keep-me"} if key == "yudian" else
                     {"port": port, "baudrate": 115200, "enabled": True}
                     for key, port in _PORTS.items()}
        self.data["camera"] = {"backend": "manual", "manual_folder": "."}

    def device(self, key):
        return self.data.setdefault(key, {})

    def section(self, key):
        return self.data.setdefault(key, {})


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def manager(qapp):
    mgr = InstrumentManager(StubSettings(), sim=True)
    yield mgr
    mgr.shutdown()
    # Destroy the proxy/thread graph deterministically HERE. Left to
    # Python's collector it was destroyed at an arbitrary moment — in
    # practice inside a later test's nested event loop, which crashed the
    # interpreter (Windows access violation) once three managers had
    # accumulated. Same class of hazard the app dodges by keeping its
    # manager alive for the whole session.
    del mgr
    gc.collect()
    flush(50)


def flush(ms: int = 200) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def collect_state(manager) -> list[tuple[str, dict]]:
    seen: list[tuple[str, dict]] = []
    manager.sig_device_state.connect(lambda k, p: seen.append((k, p)))
    return seen


class _Collector(QObject):
    """A receiver that LIVES IN THE MAIN THREAD.

    A bare lambda connected to a proxy signal is context-less: PySide ties
    it to the sender's thread affinity (the worker QThread), so an emit
    from the GUI thread is queued into a worker loop that may not run.
    Bound slots on a main-thread QObject are delivered directly.
    """

    def __init__(self):
        super().__init__()
        self.failed: list[tuple] = []

    def on_failed(self, job_id, kind, message):
        self.failed.append((job_id, kind, message))


# --- change detection -----------------------------------------------------

def test_connection_config_changed_tracks_the_built_keys(manager):
    assert not manager.connection_config_changed("zolix")
    manager.settings.device("zolix")["port"] = "COM9"
    assert manager.connection_config_changed("zolix")
    manager.settings.device("zolix")["port"] = "COM3"
    assert not manager.connection_config_changed("zolix")
    # a non-connection key is not a reason to rebuild the driver
    manager.settings.device("zolix")["um_per_pulse_xy"] = 0.5
    assert not manager.connection_config_changed("zolix")


def test_unknown_device_is_not_changed(manager):
    assert not manager.connection_config_changed("nope")
    assert manager.reconnect("nope") is False


# --- the swap -------------------------------------------------------------

def test_reconnect_swaps_the_proxy_and_reconnects(manager):
    old = manager.device("zolix")
    manager.settings.device("zolix")["port"] = "COM9"
    manager.connect_all()
    flush(100)

    replaced: list = []
    manager.sig_proxy_replaced.connect(lambda k, p: replaced.append((k, p)))
    states = collect_state(manager)
    assert manager.reconnect("zolix") is True

    new = manager.device("zolix")
    assert new is not old
    assert replaced and replaced[0][0] == "zolix"
    assert replaced[0][1] is new
    # the new factory reads the NEW config (drivers copy scalars into
    # attributes at construction, so a live driver never sees an edit)
    assert new._factory().config["port"] == "COM9"
    # the retired proxy is kept (its queued completions must still land)
    assert manager._retired["zolix"] is old
    # the LED is told the swap is happening…
    assert ("zolix", {"connecting": True}) in states
    flush(200)
    # …and the new driver reports its own connection
    assert any(k == "zolix" and p.get("connected") is True
               for k, p in states)


def test_retired_proxy_signals_are_ignored(manager):
    """A retired proxy carries the SAME device_key as its replacement: its
    late telemetry/connection signals must not speak for the live device.
    (Markers, because the live proxy polls real telemetry of its own.)"""
    old = manager.device("sigmakoki")
    manager.settings.device("sigmakoki")["port"] = "COM8"
    manager.reconnect("sigmakoki")
    states = collect_state(manager)

    old.sig_telem.emit({"device": "sigmakoki", "marker": "retired"})
    old.sig_connected.emit(False)
    new = manager.device("sigmakoki")
    new.sig_telem.emit({"device": "sigmakoki", "marker": "live"})
    flush(80)

    markers = [p.get("marker") for k, p in states if k == "sigmakoki"]
    assert "live" in markers
    assert "retired" not in markers
    assert not any(k == "sigmakoki" and p.get("connected") is False
                   for k, p in states)


def test_queued_jobs_fail_instead_of_vanishing(manager):
    """request_shutdown clears the queue silently — a waiter (the grid
    scan's StageAdapter) would block for its full 60-120 s timeout."""
    proxy = manager.device("zolix")
    collector = _Collector()
    proxy.sig_command_failed.connect(collector.on_failed)
    proxy.enqueue(101, "move_rel_um", (1.0, 0.0), 0)
    proxy.enqueue(102, "move_rel_um", (2.0, 0.0), 0)
    proxy.cancel_pending()
    assert [job for job, _t, _m in collector.failed] == [101, 102]
    assert all(reason == "Reconnecting"
               for _j, reason, _m in collector.failed)
    assert not proxy._queue


def test_frame_slot_follows_a_focus_reconnect(manager):
    class Slot:
        pass

    slot = Slot()
    manager.set_frame_slot(slot)
    assert manager.device("focus")._frame_slot is slot
    manager.reconnect("focus")
    new = manager.device("focus")
    assert isinstance(new, FocusProxy)
    assert new is not manager._retired["focus"]
    assert new._frame_slot is slot


def test_reconnect_keeps_the_enable_gate(manager):
    manager.set_enabled("zolix", False)
    assert manager.is_enabled("zolix") is False
    manager.settings.device("zolix")["port"] = "COM9"
    states = collect_state(manager)
    assert manager.reconnect("zolix") is True
    assert manager.is_enabled("zolix") is False          # untouched
    assert ("zolix", {"enabled": False}) in states       # re-broadcast


def test_focus_reconnect_refused_while_a_job_owns_the_worker(manager):
    """An autofocus/calibration job's completion would die with its proxy
    and strand the service in AUTOFOCUS forever."""
    focus = manager.device("focus")
    focus.enqueue(1, "autofocus", (object(),), 0)  # queued, never drained
    assert focus.busy is True
    manager.settings.device("focus")["port"] = "COM11"
    assert manager.reconnect("focus") is False
    assert manager.device("focus") is focus  # unchanged
    assert not manager._retired.get("focus")


def test_reconnect_refused_during_shutdown(manager):
    manager._shutting_down = True
    assert manager.reconnect("zolix") is False


def test_stop_all_still_reaches_every_stage_after_a_reconnect(manager):
    manager.connect_all()
    flush(100)
    manager.settings.device("zolix")["port"] = "COM9"
    assert manager.reconnect("zolix") is True
    done: list = []
    manager.sig_stop_all_done.connect(lambda: done.append(True))
    manager.stop_all()
    loop = QEventLoop()
    QTimer.singleShot(2000, loop.quit)
    loop.exec()
    assert done


def test_all_devices_can_be_reconnected(qapp):
    settings = StubSettings()
    manager = InstrumentManager(settings, sim=True)
    try:
        manager.connect_all()
        flush(100)
        for key in DEVICE_KEYS:
            settings.device(key)["port"] = "COM42"
            assert manager.reconnect(key) is True, key
        assert len(manager._retired) == len(DEVICE_KEYS)
        flush(150)
        assert all(manager.device(k) is not None for k in DEVICE_KEYS)
    finally:
        manager.shutdown()
