"""ManagerStageAdapter: the grid scan's blocking view of the manager's
zolix proxy (the scan must never open its own COM handle — a second
driver on the same port cannot open, and a private device is invisible
to STOP ALL)."""

import pytest
from PySide6.QtCore import QObject, Signal

from talos.hal.base import DeviceError, DeviceTimeoutError, StageSpeed
from talos.hal.proxies import stage_adapter as sa
from talos.hal.proxies.stage_adapter import ManagerStageAdapter
from talos.models import StagePosition


class StubManager(QObject):
    """Completes jobs synchronously — the hardest case for the wait graph
    (the completion arrives before submit() has even returned)."""

    sig_job_done = Signal(int, object)
    sig_job_failed = Signal(int, str, str)

    def __init__(self, results=None, fail=None, disabled=False):
        super().__init__()
        self.jobs: list[tuple] = []
        self.last_position: dict = {}
        self._results = results or {}
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
            self.sig_job_done.emit(job_id, self._results.get(method))
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


def test_scan_speed_config_scales_the_pps():
    """The workspace hands the adapter an already-scaled config (the
    objective's stage multiplier), exactly as the old private-device path
    did."""
    from talos.cv.scan import scale_scan_speed_config

    cfg = scale_scan_speed_config({"slow_speed_pps": 500, "fast_speed_pps": 2000}, 0.5)
    adapter = ManagerStageAdapter(StubManager(), cfg)
    assert adapter._pps(StageSpeed.SLOW) == 250
    assert adapter._pps(StageSpeed.FAST) == 1000


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


def test_wait_idle_settles_on_stable_telemetry():
    # the proxy publishes the position as a plain DICT (asdict), not a
    # StagePosition — reading it with getattr made every sample (0,0,0)
    # and the scan moved on before the stage had stopped
    adapter, manager = _adapter()
    manager.last_position["zolix"] = {"x_pulses": 5, "y_pulses": 5, "r_pulses": 0}
    adapter.wait_idle(timeout_s=1.0)  # returns immediately (stable)


def test_wait_idle_moves_on_only_after_three_stable_samples():
    adapter, manager = _adapter()
    samples = [{"x_pulses": 1}, {"x_pulses": 1}, {"x_pulses": 9},
               {"x_pulses": 9}, {"x_pulses": 9}, {"x_pulses": 9}]

    class _Feed:
        def get(self, key, default=None):
            if key != "zolix":
                return default
            return samples.pop(0) if samples else {"x_pulses": 9}

    manager.last_position = _Feed()
    adapter.wait_idle(timeout_s=1.0)
    # the single stable sample before the jump must NOT have settled it
    assert len(samples) <= 3


def test_wait_idle_times_out_on_a_moving_stage():
    adapter, manager = _adapter()
    positions = [StagePosition(x_pulses=i, y_pulses=i) for i in range(1000)]

    class _Feed:
        def get(self, key, default=None):
            return positions.pop(0) if positions else None

    manager.last_position = _Feed()
    with pytest.raises(DeviceTimeoutError):
        adapter.wait_idle(timeout_s=0.05)


def test_wait_idle_returns_early_on_abort():
    manager = StubManager()
    adapter = ManagerStageAdapter(manager, {}, abort_check=lambda: True)
    adapter.wait_idle(timeout_s=5.0)  # must not raise nor wait


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
