"""The stop phase: what a scan does NOT do between two waypoints.

A grid scan used to spend ≥0.6 s of every waypoint waiting for the
manager's telemetry position to look stable across three samples taken
0.2 s apart — and the telemetry it watched stops updating while a job is
queued, i.e. exactly while a move is in flight. On top of that, the PNG
encode, the thumbnail and the manifest row were written inline, on the one
thread that could have been commanding the next move.

Both are pinned here, through the manager's own proxy — the path the app
takes. (Every other sim scan test hands ``GridScanner`` a raw
``SimZolixXYRStage`` and so bypasses ``ManagerStageAdapter`` entirely,
which is why the 0.6 s never showed up in the suite.)

Real-time, so the whole file is slow (`-m "not slow"` skips it).
"""

import csv
import threading
import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from talos.cv.frame_slot import FrameMeta
from talos.cv.scan import GridScanner
from talos.hal.proxies.stage_adapter import ManagerStageAdapter
from talos.instruments import InstrumentManager
from talos.models import ScanParams

pytestmark = pytest.mark.slow

#: The old telemetry rule alone cost this much per tile, so anything under
#: it cannot be that rule still running. Generous on purpose: the point is
#: to catch a regression, not to measure the machine.
OLD_RULE_S_PER_TILE = 0.6
MAX_STOPPED_S_PER_TILE = 0.25


class StubSettings:
    """Sim devices on paper ports — the manager builds the simulators."""

    def __init__(self):
        self.data = {key: {"port": f"COM{index + 3}", "baudrate": 115200,
                           "enabled": True}
                     for index, key in enumerate(("zolix", "sigmakoki",
                                                  "focus", "yudian"))}
        self.data["camera"] = {"backend": "manual", "manual_folder": "."}

    def device(self, key):
        return self.data.setdefault(key, {})

    def section(self, key):
        return self.data.setdefault(key, {})


class InstantFrameSource:
    """A frame is always ready: this file is about the stage, not the
    camera (the capture gate has its own tests in test_scan_capture.py)."""

    def __init__(self):
        self.calls = 0

    def grab(self, settle_s: float = 0.0, timeout_s: float = 0.0):
        self.calls += 1
        frame = np.zeros((8, 16, 3), dtype=np.uint8)
        meta = FrameMeta(t_capture=time.monotonic(), seq=self.calls,
                         shape=frame.shape)
        return frame, meta


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def manager(qapp):
    mgr = InstrumentManager(StubSettings(), sim=True)
    mgr.connect_all()
    yield mgr
    mgr.shutdown()


def _run_with_the_gui_pumping(scanner, params, out_dir, qapp, timeout_s=60.0):
    """The scan blocks on the manager's jobs, and those completions are
    delivered on the GUI thread — so the test must pump it, exactly as the
    app does."""
    box: dict = {}

    def worker():
        try:
            box["result"] = scanner.run(params, out_dir,
                                        meta={"fov_um": (100.0, 100.0)})
        except Exception as exc:  # noqa: BLE001
            box["error"] = repr(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout_s
    while thread.is_alive() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    thread.join(timeout=5.0)
    assert "error" not in box, box.get("error")
    assert "result" in box, "the scan did not finish"
    return box["result"]


def _adapter(manager, pps=4000):
    # ~100 um tiles at 0.625 um/pulse = 160 pulses -> a 40 ms move
    return ManagerStageAdapter(manager, {"slow_speed_pps": pps,
                                         "fast_speed_pps": pps})


def test_a_waypoint_does_not_wait_for_a_stable_position(manager, qapp,
                                                        tmp_path):
    """The regression test for the 0.6 s: with no settle, the stopped phase
    is a status read and a frame, and nothing else."""
    adapter = _adapter(manager)
    frames = InstantFrameSource()
    scanner = GridScanner(adapter, frames)
    params = ScanParams(x0_um=0.0, y0_um=0.0, width_um=200.0, height_um=100.0,
                        overlap=0.0, settle_ms=0, return_to_start=False)
    result = _run_with_the_gui_pumping(scanner, params, tmp_path, qapp)
    adapter.close()

    assert not result.aborted and not result.failed, result.message
    assert result.visited == result.planned == frames.calls
    assert result.timing.tiles == result.planned
    per_tile = result.timing.stopped_s / result.timing.tiles
    assert per_tile < MAX_STOPPED_S_PER_TILE, (
        f"{per_tile:.2f} s stopped per tile — the telemetry-sample settle "
        f"rule (>= {OLD_RULE_S_PER_TILE} s) is back")


def test_the_run_paces_itself_to_the_detector(manager, qapp, tmp_path,
                                              monkeypatch):
    """Tiles are never dropped, so the scan must not race ahead of the
    identification queue — a 4K frame is 25 MB and the queue is otherwise
    unbounded. Past the cap the stage waits (and the safety valve in
    ``_wait_for_the_detector`` is what ends the wait, rather than a scan
    that hangs with the stage parked mid-plan)."""
    from talos.cv import scan as scan_mod

    monkeypatch.setattr(scan_mod, "_PENDING_TIMEOUT_S", 0.5)
    adapter = _adapter(manager)
    scanner = GridScanner(adapter, InstantFrameSource(),
                          pending_tiles_fn=lambda: 99)   # a wedged detector
    logged: list[str] = []
    scanner.sig_log.connect(logged.append)
    params = ScanParams(x0_um=0.0, y0_um=0.0, width_um=100.0, height_um=100.0,
                        overlap=0.0, settle_ms=0, return_to_start=False)
    t0 = time.monotonic()
    result = _run_with_the_gui_pumping(scanner, params, tmp_path, qapp,
                                       timeout_s=10.0)
    adapter.close()
    elapsed = time.monotonic() - t0

    assert result.visited == result.planned == 4   # 2 x 2 tiles at 100 um
    assert elapsed < 60.0        # the valve opened, the run did not hang
    assert any("waiting to be identified" in line for line in logged), logged


def test_the_manifest_is_complete_when_run_returns(manager, qapp, tmp_path):
    """The writer is joined before the run returns: scan_output re-reads
    the manifest from disk the moment it does."""
    adapter = _adapter(manager)
    scanner = GridScanner(adapter, InstantFrameSource(), thumb_width=8)
    params = ScanParams(x0_um=0.0, y0_um=0.0, width_um=100.0, height_um=100.0,
                        overlap=0.0, settle_ms=0, return_to_start=False)
    result = _run_with_the_gui_pumping(scanner, params, tmp_path, qapp)
    adapter.close()

    with open(result.manifest_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == result.planned
    assert all(row["frame"] for row in rows)
    for row in rows:
        assert (tmp_path / "frames" / row["frame"]).exists()
    assert [p.name for p in result.frames] == [row["frame"] for row in rows]
