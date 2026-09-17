"""Closed-loop scan capture through the frame slot — the application's path.

The sim camera is pumped into a ``LatestFrameSlot`` exactly as the app's
camera worker does, and the scanner reads it through ``LatestFrameSource``.
This is the wiring that has never once run on the bench, so the contract it
must satisfy is spelled out here: every waypoint gets a frame captured
AFTER the stage arrived, and a stream that stalls leaves waypoints MISSING
rather than filing the previous frame under the new position.
"""

import csv
import json
import threading
import time

import pytest
from PySide6.QtWidgets import QApplication

from talos.cv.frame_slot import LatestFrameSlot
from talos.cv.frame_source import LatestFrameSource
from talos.cv.scan import GridScanner
from talos.hal.sim import SimCamera, SimZolixXYRStage
from talos.models import ScanParams

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _CameraWorker(threading.Thread):
    """The app's camera proxy in miniature: fetch → slot, forever."""

    def __init__(self, camera, slot):
        super().__init__(daemon=True)
        self._camera = camera
        self._slot = slot
        self._stop_event = threading.Event()
        self._seq = 0
        self.frames = 0

    def run(self):
        while not self._stop_event.is_set():
            frame = self._camera.fetch(timeout_ms=100.0)
            if frame is None:
                continue
            t = self._camera.capture_time() or time.monotonic()
            self._seq += 1
            self.frames += 1
            self._slot.write(frame, t, self._seq)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=5.0)


@pytest.fixture
def rig():
    stage = SimZolixXYRStage({"latency_s": 0.001, "slow_speed_pps": 5000})
    stage.connect()
    camera = SimCamera({"width": 320, "height": 240, "fps": 60})
    camera.connect()
    camera.start()
    slot = LatestFrameSlot()
    worker = _CameraWorker(camera, slot)
    worker.start()
    yield stage, camera, slot, worker
    worker.stop()
    camera.stop()
    camera.disconnect()
    stage.disconnect()


def _params(**kw) -> ScanParams:
    base = dict(x0_um=0.0, y0_um=0.0, width_um=200.0, height_um=100.0,
                overlap=0.0, settle_ms=60, return_to_start=False)
    base.update(kw)
    return ScanParams(**base)


def test_a_scan_captures_a_frame_at_every_waypoint(qapp, rig, tmp_path):
    stage, _camera, slot, _worker = rig
    scanner = GridScanner(stage, LatestFrameSource(slot))
    tiles: list = []
    frames: list = []
    scanner.sig_tile.connect(
        lambda i, x, y, thumb: tiles.append((i, x, y, thumb)))
    scanner.sig_frame.connect(lambda i, x, y, frame: frames.append((i, frame)))

    result = scanner.run(_params(width_um=300.0, height_um=200.0), tmp_path,
                         meta={"fov_um": (100.0, 100.0), "objective_id": 2})
    qapp.processEvents()      # drain the queued signal, if it was queued

    assert not result.aborted, result.message
    assert result.missing == 0
    assert len(result.frames) == 6          # 2 cols × 3 rows
    for path in result.frames:
        assert path.exists() and path.stat().st_size > 0

    with open(result.manifest_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["frame"] for row in rows] == [p.name for p in result.frames]
    assert rows[0]["objective_id"] == "2"

    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_frames"] == 6 and meta["n_missing"] == 0
    assert meta["frame_shape"] == [240, 320, 3]

    # the map gets a tile per waypoint, in order, at its readback position
    assert [i for i, _x, _y, _t in tiles] == list(range(6))
    assert tiles[0][3].shape == (120, 160, 3)
    assert (tiles[0][1], tiles[0][2]) == pytest.approx((0.0, 0.0))
    assert (tiles[1][1], tiles[1][2]) == pytest.approx((100.0, 0.0))
    # and the detection feed gets the FULL frame with the same position
    assert [i for i, _f in frames] == list(range(6))
    assert frames[0][1].shape == (240, 320, 3)


def test_a_stalled_stream_leaves_waypoints_missing(qapp, rig, tmp_path):
    """The camera dies mid-scan. The frames already captured stay on disk
    and in the manifest; the rest are reported MISSING — a scan never files
    the frame from before the move under the new position."""
    stage, camera, slot, worker = rig
    scanner = GridScanner(stage, LatestFrameSource(slot))
    box: dict = {}
    # 6 waypoints at a 200 ms settle cannot finish inside 0.5 s, and the
    # first one is long since captured — so the kill lands mid-run.
    thread = threading.Thread(
        target=lambda: box.update(result=scanner.run(
            _params(width_um=300.0, height_um=200.0, settle_ms=200),
            tmp_path, meta={"fov_um": (100.0, 100.0)})))
    thread.start()
    time.sleep(0.5)
    worker.stop()
    camera.stop()
    thread.join(timeout=60.0)
    result = box["result"]

    assert 0 < len(result.frames) < 6, "the kill landed at the wrong moment"
    assert result.missing + len(result.frames) == 6
    with open(result.manifest_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6
    empties = [row for row in rows if row["frame"] == ""]
    assert len(empties) == result.missing


def test_the_stage_returns_to_the_start_when_asked(qapp, rig, tmp_path):
    stage, _camera, slot, _worker = rig
    stage.move_abs_um(500.0, 400.0)
    stage.wait_idle(timeout_s=5.0)
    params = _params(x0_um=500.0, y0_um=400.0, return_to_start=True)
    scanner = GridScanner(stage, LatestFrameSource(slot))
    scanner.run(params, tmp_path, meta={"fov_um": (100.0, 100.0)})
    pos = stage.get_position()
    assert (pos.x_um, pos.y_um) == pytest.approx((500.0, 400.0), abs=1.0)


def test_no_return_to_start_on_an_abort(qapp, rig, tmp_path):
    """An abort means "stop moving" — driving back to the origin would be
    the opposite of what was asked."""
    stage, _camera, slot, _worker = rig
    stage.slow_speed_pps = 200
    params = _params(width_um=500.0, height_um=500.0, return_to_start=True)
    scanner = GridScanner(stage, LatestFrameSource(slot))
    box: dict = {}
    thread = threading.Thread(
        target=lambda: box.update(result=scanner.run(
            params, tmp_path, meta={"fov_um": (100.0, 100.0)})))
    thread.start()
    time.sleep(0.5)
    scanner.request_abort()
    thread.join(timeout=30.0)
    assert box["result"].aborted
    assert not stage.get_status().any_moving
