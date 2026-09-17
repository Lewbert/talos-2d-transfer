"""Closed-loop grid-scan tests: SimZolixXYRStage + SimCamera."""

import csv
import json

import pytest
from PySide6.QtWidgets import QApplication

from talos.cv.frame_source import CameraFrameSource
from talos.cv.scan import GridScanner
from talos.hal.sim import SimCamera, SimZolixXYRStage
from talos.models import ScanParams

# Real-time closed-loop simulation: the whole file is marked slow
# (deselect with `-m "not slow"` for a fast edit loop).
pytestmark = pytest.mark.slow



@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def make_rig():
    stage = SimZolixXYRStage({"latency_s": 0.001, "slow_speed_pps": 5000})
    stage.connect()
    camera = SimCamera({"width": 320, "height": 240, "fps": 1000})
    camera.connect()
    camera.start()
    return stage, camera


def test_plan_serpentine_geometry(qapp):
    """The visit order, and — just as important — how many tiles the default
    origin needs to cover the area.

    A 300 × 200 µm area with a 100 µm frame needs FOUR columns, not three:
    the operator's position is the centre of the first tile, so half a
    frame of the covered span lies behind it and the last column has to
    reach 350 µm to cover 300.
    """
    params = ScanParams(x0_um=0, y0_um=0, width_um=300, height_um=200,
                        overlap=0.0, serpentine=True)
    scanner = GridScanner(SimZolixXYRStage())
    waypoints = scanner.plan(params, fov_um=(100.0, 100.0))
    assert len(waypoints) == 12                      # 4 cols x 3 rows
    assert [w.x_um for w in waypoints[:4]] == [0.0, 100.0, 200.0, 300.0]
    assert [w.x_um for w in waypoints[4:8]] == [300.0, 200.0, 100.0, 0.0]
    assert [w.y_um for w in waypoints[:4]] == [0.0] * 4
    assert [w.y_um for w in waypoints[4:8]] == [100.0] * 4
    assert [w.y_um for w in waypoints[8:]] == [200.0] * 4


def test_the_corner_origin_is_how_you_cover_an_area_from_its_edge(qapp):
    """The same area from a CORNER: three columns instead of four, and the
    last tile's outer edge lands exactly on the far edge."""
    params = ScanParams(x0_um=0, y0_um=0, width_um=300, height_um=200,
                        overlap=0.0, serpentine=True, origin="corner_fit")
    scanner = GridScanner(SimZolixXYRStage())
    waypoints = scanner.plan(params, fov_um=(100.0, 100.0))
    assert len(waypoints) == 6                       # 3 cols x 2 rows
    assert [w.x_um for w in waypoints[:3]] == [50.0, 150.0, 250.0]
    assert [w.y_um for w in waypoints[:3]] == [50.0] * 3
    assert waypoints[2].x_um + 50.0 == pytest.approx(300.0)
    assert waypoints[-1].y_um + 50.0 == pytest.approx(200.0)


def test_plan_overlap_spacing(qapp):
    params = ScanParams(x0_um=0, y0_um=0, width_um=100, height_um=100,
                        overlap=0.5, serpentine=False)
    scanner = GridScanner(SimZolixXYRStage())
    waypoints = scanner.plan(params, fov_um=(100.0, 100.0))
    assert len(waypoints) == 4  # 2x2 grid
    assert [w.x_um for w in waypoints[:2]] == [0.0, 50.0]  # first row
    assert [w.y_um for w in waypoints[:2]] == [0.0, 0.0]
    assert [w.y_um for w in waypoints[2:]] == [50.0, 50.0]


def test_run_scan_manifest_and_frames(qapp, tmp_path):
    stage, camera = make_rig()
    params = ScanParams(x0_um=0, y0_um=0, width_um=300, height_um=200,
                        overlap=0.0, serpentine=True)
    scanner = GridScanner(stage, CameraFrameSource(camera))
    waypoints = scanner.plan(params, (100.0, 100.0))
    result = scanner.run(params, tmp_path, meta={"objective_id": 0, "focus_pos": 42,
                                                 "fov_um": (100.0, 100.0)})
    assert not result.aborted, result.message
    # the count comes from the plan, not from this test: what is being
    # checked here is that every waypoint produced a frame and a manifest
    # row, and that they line up
    assert len(result.frames) == len(waypoints)
    with open(result.manifest_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(waypoints)
    # Readback positions match the plan (sim stage lands exactly).
    for row, waypoint in zip(rows, waypoints):
        assert float(row["x_um"]) == pytest.approx(waypoint.x_um)
        assert float(row["y_um"]) == pytest.approx(waypoint.y_um)
    # serpentine: the even rows reverse
    assert float(rows[0]["x_um"]) == pytest.approx(0.0)
    assert float(rows[4]["x_um"]) == pytest.approx(300.0)
    assert float(rows[5]["x_um"]) == pytest.approx(200.0)
    assert rows[0]["focus_pos"] == "42"
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_frames"] == len(waypoints)


def test_scan_without_a_camera_reports_no_frames(qapp, tmp_path):
    """A camera-less scan used to touch() an empty file per waypoint and
    count it: the UI reported "Scan done: N frames" over a dataset with
    no images at all."""
    stage, _camera = make_rig()
    params = ScanParams(x0_um=0, y0_um=0, width_um=300, height_um=200,
                        overlap=0.0)
    scanner = GridScanner(stage)
    waypoints = scanner.plan(params, (100.0, 100.0))
    result = scanner.run(params, tmp_path, meta={"fov_um": (100.0, 100.0)})
    assert len(result.frames) == 0
    assert result.missing == len(waypoints)
    # the geometry is still recorded — with an empty frame column
    with open(result.manifest_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(waypoints)
    assert {row["frame"] for row in rows} == {""}
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_frames"] == 0 and meta["n_missing"] == len(waypoints)


def test_run_scan_abort_stops_cleanly(qapp, tmp_path):
    stage, camera = make_rig()
    stage.slow_speed_pps = 200  # slow moves so the abort lands mid-scan
    params = ScanParams(x0_um=0, y0_um=0, width_um=500, height_um=500,
                        overlap=0.0)
    scanner = GridScanner(stage, CameraFrameSource(camera))

    import threading

    box: dict = {}

    def worker():
        box["result"] = scanner.run(params, tmp_path,
                                    meta={"fov_um": (100.0, 100.0)})

    thread = threading.Thread(target=worker)
    thread.start()
    import time

    time.sleep(0.6)
    scanner.request_abort()
    thread.join(timeout=30.0)
    result = box["result"]
    assert result.aborted
    assert not stage.get_status().any_moving  # stage stopped
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["aborted"] is True
