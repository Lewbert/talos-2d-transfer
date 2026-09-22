"""ManagerStageAdapter: the grid scan's blocking view of the manager's
zolix proxy (the scan must never open its own COM handle — a second
driver on the same port cannot open, and a private device is invisible
to STOP ALL)."""

import time

import pytest
from PySide6.QtCore import QObject, Signal

from talos.hal.base import DeviceError, DeviceTimeoutError, StageSpeed
from talos.hal.proxies import stage_adapter as sa
from talos.hal.proxies.stage_adapter import ManagerStageAdapter
from talos.models import StagePosition, StageStatus


class StubManager(QObject):
    """Completes jobs synchronously — the hardest case for the wait graph
    (the completion arrives before submit() has even returned)."""

    sig_job_done = Signal(int, object)
    sig_job_failed = Signal(int, str, str)

    def __init__(self, results=None, fail=None, disabled=False, sequence=None):
        super().__init__()
        self.jobs: list[tuple] = []
        self.last_position: dict = {}
        self._results = results or {}
        #: per-method lists whose entries are consumed one per call, for a
        #: value that changes between reads (moving → stopped, say)
        self._sequence = {k: list(v) for k, v in (sequence or {}).items()}
        self._fail = fail or {}
        self._disabled = disabled
        self._next = 0

    def submit(self, device, method, *args, priority=0):
        if self._disabled:
            return -1
        self.jobs.append((device, method, args, priority))
        self._next += 1
        job_id = self._next
        if method in self._fail:
            self.sig_job_failed.emit(job_id, "DeviceError", self._fail[method])
        else:
            feed = self._sequence.get(method)
            result = feed.pop(0) if feed else self._results.get(method)
            self.sig_job_done.emit(job_id, result)
        return job_id


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch):
    monkeypatch.setattr(sa, "_WAIT_POLL_S", 0.001)


def _adapter(**kwargs):
    manager = StubManager(**kwargs)
    cfg = {"slow_speed_pps": 500, "fast_speed_pps": 2000}
    return ManagerStageAdapter(manager, cfg), manager


def test_move_abs_um_passes_the_scaled_speed():
    adapter, manager = _adapter()
    adapter.move_abs_um(100.0, 50.0, speed=StageSpeed.SLOW)
    device, method, args, _priority = manager.jobs[-1]
    assert device == "zolix" and method == "move_abs_um"
    assert args == (100.0, 50.0, None, StageSpeed.SLOW, 500)


def test_move_abs_um_uses_the_fast_speed_when_asked():
    adapter, manager = _adapter()
    adapter.move_abs_um(0.0, 0.0, None, StageSpeed.FAST)
    assert manager.jobs[-1][2][-1] == 2000


def test_scan_speed_config_reaches_the_adapter():
    """The scan hands the adapter a config carrying its own single speed,
    and the adapter reads it — the last link in the chain from the
    operator's number to the commanded pulses."""
    from talos.cv.scan import scan_speed_config

    cfg = scan_speed_config({"slow_speed_pps": 500, "fast_speed_pps": 2000},
                            900)
    adapter = ManagerStageAdapter(StubManager(), cfg)
    assert adapter._pps(StageSpeed.SLOW) == 900
    assert adapter._pps(StageSpeed.FAST) == 900


def test_job_failure_surfaces_as_device_error():
    adapter, _ = _adapter(fail={"move_abs_um": "axes are moving"})
    with pytest.raises(DeviceError, match="axes are moving"):
        adapter.move_abs_um(1.0, 1.0)


def test_timeout_when_a_job_never_completes():
    manager = StubManager()
    adapter = ManagerStageAdapter(manager, {})

    def _never(*args, **kwargs):
        return 99  # accepted, but no completion is ever emitted

    manager.submit = _never  # type: ignore[method-assign]
    with pytest.raises(DeviceTimeoutError):
        adapter._call("get_position", timeout_s=0.05)


def test_disabled_device_is_refused():
    adapter, _ = _adapter(disabled=True)
    with pytest.raises(DeviceError):
        adapter.move_abs_um(1.0, 1.0)


def test_abort_short_circuits_before_submitting():
    manager = StubManager()
    adapter = ManagerStageAdapter(manager, {}, abort_check=lambda: True)
    with pytest.raises(DeviceError, match="abort"):
        adapter.move_abs_um(1.0, 1.0)
    assert manager.jobs == []


def test_get_position_returns_the_readback():
    pos = StagePosition(x_pulses=10, y_pulses=20, r_pulses=30)
    adapter, _ = _adapter(results={"get_position": pos})
    got = adapter.get_position()
    assert (got.x_pulses, got.y_pulses, got.r_pulses) == (10, 20, 30)


def test_get_position_never_returns_the_commanded_value():
    """A job that completes with something unexpected degrades to an empty
    readback rather than crashing the scan."""
    adapter, _ = _adapter(results={"get_position": "not-a-position"})
    assert isinstance(adapter.get_position(), StagePosition)


def test_wait_idle_settles_on_the_first_stopped_status():
    """One status read is the whole wait: the controller's own motion bits
    answer the question directly, so there is nothing to average."""
    adapter, manager = _adapter(results={"get_status": StageStatus()})
    adapter.wait_idle(timeout_s=1.0)
    assert [method for _d, method, _a, _p in manager.jobs] == ["get_status"]


def test_wait_idle_polls_while_the_axes_are_moving():
    adapter, manager = _adapter(sequence={
        "get_status": [StageStatus(x_moving=True), StageStatus(y_moving=True),
                       StageStatus()],
    })
    adapter.wait_idle(timeout_s=5.0)
    assert len(manager.jobs) == 3


def test_wait_idle_never_reads_the_telemetry_it_used_to():
    """The old rule watched `manager.last_position`, which the proxy stops
    publishing while a job is queued — so the check was blind for exactly
    the window it was meant to measure, and each sample cost 0.2 s."""
    adapter, manager = _adapter(results={"get_status": StageStatus()})
    manager.last_position["zolix"] = {"x_pulses": 0, "y_pulses": 0, "r_pulses": 0}
    adapter.wait_idle(timeout_s=1.0)
    assert all(method == "get_status" for _d, method, _a, _p in manager.jobs)


def test_wait_idle_times_out_on_a_stage_that_never_stops():
    adapter, _ = _adapter(results={"get_status": StageStatus(x_moving=True)})
    with pytest.raises(DeviceTimeoutError):
        adapter.wait_idle(timeout_s=0.05)


def test_wait_idle_refuses_to_read_an_unreadable_answer_as_stopped():
    """A job that completes with something unexpected cannot say "the axes
    stopped" — it is a failed poll, three of which end the wait. Reading a
    missing answer as a yes is the failure this method's predecessor was
    written about."""
    adapter, _ = _adapter(results={"get_status": "not-a-status"})
    with pytest.raises(DeviceError, match="did not answer"):
        adapter.wait_idle(timeout_s=5.0)


def test_wait_idle_gives_up_on_a_link_that_stops_answering():
    adapter, _ = _adapter(fail={"get_status": "CRC mismatch"})
    with pytest.raises(DeviceError, match="CRC mismatch"):
        adapter.wait_idle(timeout_s=5.0)


def test_wait_idle_returns_early_on_abort():
    manager = StubManager()
    adapter = ManagerStageAdapter(manager, {}, abort_check=lambda: True)
    adapter.wait_idle(timeout_s=5.0)  # must not raise nor wait
    assert manager.jobs == []


def test_wait_idle_returns_quietly_when_the_abort_lands_mid_read():
    """The abort can arrive while a status job is in flight; `_call` then
    raises, and the scan must see a clean stop rather than a fault."""
    flag = {"abort": False}

    class _Aborting(StubManager):
        def submit(self, device, method, *args, priority=0):
            flag["abort"] = True          # the operator hits Esc mid-read
            return -1

    adapter = ManagerStageAdapter(_Aborting(), {},
                                 abort_check=lambda: flag["abort"])
    adapter.wait_idle(timeout_s=5.0)


def test_stop_uses_the_priority_path():
    adapter, manager = _adapter()
    adapter.stop()
    device, method, _args, priority = manager.jobs[-1]
    assert (device, method) == ("zolix", "stop")
    assert priority == 1


def test_close_detaches_from_the_manager():
    adapter, manager = _adapter()
    adapter.close()
    adapter.close()  # idempotent
    assert adapter._done == {}


def test_an_abort_does_not_wait_out_a_job_that_never_reports():
    """An abort mid-run can arrive while a move is in flight, and a stop
    that purges the queue or a link that drops the reply leaves a job that
    never completes. The flag is what the operator asked for, so it wins
    over a completion that may not be coming — otherwise an aborted scan
    sits on the full job timeout (120 s for a move) with the stage already
    stopped and the panel looping on stop-all."""
    manager = StubManager()
    adapter = ManagerStageAdapter(manager, {})
    flag = {"abort": False}

    def _never(*args, **kwargs):
        flag["abort"] = True      # the operator hits Esc a moment later
        return 99                 # accepted, but nothing will ever report

    manager.submit = _never  # type: ignore[method-assign]
    adapter._abort_check = lambda: flag["abort"]
    started = time.monotonic()
    with pytest.raises(DeviceError, match="abort"):
        adapter._call("move_abs_um", 0.0, 0.0, timeout_s=120.0)
    assert time.monotonic() - started < 1.0


def test_wait_idle_still_returns_quietly_when_the_abort_lands_mid_read():
    """The same check in the wait loop must stay a quiet return: the scan
    reports "aborted", which is a different thing from a fault."""
    manager = StubManager()
    adapter = ManagerStageAdapter(manager, {})
    flag = {"abort": False}

    def _never(*args, **kwargs):
        flag["abort"] = True
        return 99

    manager.submit = _never  # type: ignore[method-assign]
    adapter._abort_check = lambda: flag["abort"]
    adapter.wait_idle(timeout_s=120.0)      # returns, does not raise
