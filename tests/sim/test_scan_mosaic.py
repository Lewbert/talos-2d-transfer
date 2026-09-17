"""A scan over a moving wafer must stitch into a CONTINUOUS mosaic.

The bug this exists to catch: a static test scene renders the same picture
at every waypoint, so a mosaic assembled from it looks plausible even when
the tiles are placed backwards or rotated — which is exactly how a scan
shipped with every tile rotated 180°, leaving the mosaic doubled at every
overlap and the scan map upside down against the live view.

Here the simulated camera images a wafer that the simulated stage carries
(`SimCamera(wafer=True)` + `hal/sim/bench.py`), so a tile's content depends
on where it was taken. Two properties then hold only if the geometry is
right: the wafer's beacon lands where the readback says it should, and the
mosaic's illumination — a smooth ramp in wafer coordinates — has no step at
a tile boundary.
"""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from talos.cv.frame_source import CameraFrameSource
from talos.cv.orientation import mosaic_offset
from talos.cv.scan import GridScanner
from talos.cv.stitch import build_mosaic, mosaic_geometry
from talos.hal.sim import SimCamera, SimZolixXYRStage, bench
from talos.models import ScanParams

pytestmark = pytest.mark.slow

WIDTH, HEIGHT = 640, 480
UM_PER_PX = 2.0
FOV_UM = (WIDTH * UM_PER_PX, HEIGHT * UM_PER_PX)     # 1280 × 960 µm


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def rig():
    bench.reset()
    stage = SimZolixXYRStage({"latency_s": 0.0, "slow_speed_pps": 20000})
    stage.connect()
    camera = SimCamera({"width": WIDTH, "height": HEIGHT, "fps": 1000,
                        "noise": 0.0, "wafer": True,
                        "wafer_um_per_px": UM_PER_PX})
    camera.connect()
    camera.start()
    yield stage, camera
    camera.stop()
    camera.disconnect()
    stage.disconnect()
    bench.reset()


def _scan(stage, camera, params, tmp_path, settle_ms=0):
    params.settle_ms = settle_ms
    scanner = GridScanner(stage, CameraFrameSource(camera))
    tiles: list = []
    scanner.sig_frame.connect(lambda i, x, y, f: tiles.append((x, y, f)))
    result = scanner.run(params, tmp_path, meta={"fov_um": FOV_UM})
    assert not result.aborted, result.message
    assert result.missing == 0
    return tiles, result


def _beacon_centre(image: np.ndarray):
    """The wafer's magenta beacon, found by colour."""
    mask = ((image[:, :, 0] > 200) & (image[:, :, 1] < 80)
            & (image[:, :, 2] > 200))
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _mosaic_pixel(tiles, fov_um, x_um, y_um, flip):
    """Where a sample point lands in the mosaic — the builder's own layout,
    not a re-derivation of it."""
    px_per_um, x0, y0, shrink = mosaic_geometry(tiles, fov_um, flip=flip)
    x_eff, y_eff = mosaic_offset(x_um, y_um, flip)
    return ((x_eff - x0) * px_per_um * shrink,
            (y_eff - y0) * px_per_um * shrink)


@pytest.mark.parametrize("flip", [False, True])
def test_a_scan_stitches_into_one_continuous_image(qapp, rig, tmp_path, flip):
    """Both flip settings must stitch — the flip mirrors the layout, and a
    layout that does not follow it puts every feature on the mosaic twice
    (which is what the first implementation did)."""
    stage, camera = rig
    camera.set_flip(flip)
    # 50 % overlap: every interior tile shares half its area with its
    # neighbour, so a placement error shows up immediately
    params = ScanParams(x0_um=-1000.0, y0_um=-500.0, width_um=2000.0,
                        height_um=1000.0, overlap=0.5, serpentine=True,
                        return_to_start=False)
    tiles, _result = _scan(stage, camera, params, tmp_path)
    # a half-FOV pitch over 2000 × 1000 µm at a 1280 × 960 µm field
    assert len(tiles) == 4 * 3

    mosaic = build_mosaic([(x, y, frame) for x, y, frame in tiles], FOV_UM,
                          flip=flip)
    assert mosaic is not None

    # 1. ONE beacon, where the readback says it is (a doubled beacon is the
    #    signature of a mis-placed tile: its two copies fall in different
    #    places and neither is where the geometry says)
    found = _beacon_centre(mosaic)
    assert found is not None, "the beacon fell outside the mosaic"
    expected = _mosaic_pixel([(x, y, f) for x, y, f in tiles], FOV_UM,
                             0.0, 0.0, flip)
    assert found[0] == pytest.approx(expected[0], abs=12.0)
    assert found[1] == pytest.approx(expected[1], abs=12.0)
    loose = ((mosaic[:, :, 0].astype(int) - mosaic[:, :, 1]) > 40) & \
            ((mosaic[:, :, 2].astype(int) - mosaic[:, :, 1]) > 40)
    assert loose.sum() < 25000, "the beacon appears more than once"

    # 2. no seam. Sample a row that runs BETWEEN the wafer's feature rows
    #    (they sit on a 500 µm grid, the beacon on y = 0): along that row
    #    the only thing that can step is a tile boundary, because the
    #    illumination ramps smoothly in wafer coordinates.
    px_per_um, _x0, y0, shrink = mosaic_geometry(tile_list :=
                                                 [(x, y, f)
                                                  for x, y, f in tiles],
                                                 FOV_UM, flip=flip)
    _x_eff, y_eff = mosaic_offset(0.0, 250.0, flip)
    row_index = int(round((y_eff - y0) * px_per_um * shrink))
    assert 0 <= row_index < mosaic.shape[0]
    row = mosaic[row_index, :, 0].astype(np.int16)
    steps = np.abs(np.diff(row))
    assert steps.max() < 8, (
        f"a step of {steps.max()} counts at column "
        f"{int(np.argmax(steps))} of row {row_index} — the tiles do not "
        "line up")


def test_the_sim_models_the_measured_mounting(qapp, rig):
    """The bench statement, pinned so the mosaic tests mean something.

    Measured on the bench 2026-09-17, flip in its default state: jogging the
    stage +X moves a feature RIGHT in the frame, and +Y moves it UP. If the
    simulation ever models a different mounting, the stitching tests below
    would happily certify a mirror-image mosaic — which is exactly how the
    first version shipped wrong.
    """
    stage, camera = rig
    camera.set_flip(True)
    start = _beacon_centre(camera.fetch())
    assert start is not None
    stage.move_abs_um(200.0, 0.0)
    stage.wait_idle(timeout_s=10.0)
    moved_x = _beacon_centre(camera.fetch())
    assert moved_x[0] > start[0] + 50.0, "+X must move a feature right"

    stage.move_abs_um(200.0, 200.0)
    stage.wait_idle(timeout_s=10.0)
    moved_y = _beacon_centre(camera.fetch())
    assert moved_y[1] < moved_x[1] - 25.0, "+Y must move a feature up"


def test_the_mosaic_grows_in_the_scan_direction(qapp, rig, tmp_path):
    """A tile taken further along +X must land further along +X — the sign
    the map and the mosaic share with the readback."""
    stage, camera = rig
    # two tiles side by side along +X, one row
    params = ScanParams(x0_um=-400.0, y0_um=0.0, width_um=1200.0,
                        height_um=100.0, overlap=0.4, serpentine=False,
                        return_to_start=False)
    tiles, _result = _scan(stage, camera, params, tmp_path)
    assert len(tiles) == 2
    first, second = tiles
    assert second[0] > first[0]                       # readback advanced
    mosaic = build_mosaic([(x, y, f) for x, y, f in tiles], FOV_UM,
                          flip=False)
    left = _beacon_centre(first[2])
    right = _beacon_centre(second[2])
    # in the DISPLAYED frame the wafer slides the same way the stage moves
    # (that is what the camera flip means), so the beacon moved with it
    assert right[0] > left[0]
