"""Scan-path planning: every order must cover the rectangle exactly once,
the area must actually be covered, and the preview must be the grid the
scan actually walks."""

import pytest

from talos.cv.scan import (ONE_WAY, ORIGINS, PATHS, GridScanner, grid_shape,
                           plan_cells, plan_geometry, plan_path, plan_steps)
from talos.models import ScanParams

FOV = (100.0, 100.0)

#: Rectangles worth proving: the degenerate ones (a single row / column)
#: are where a spiral or a clipped curve loses a cell.
GRIDS = [(1, 1), (1, 5), (5, 1), (2, 2), (3, 2), (4, 3), (7, 5), (9, 1)]


def _params(nx=1, ny=1, **kw) -> ScanParams:
    """Params whose grid is exactly nx × ny at a 100 µm pitch, in the
    default (centre) origin.

    A tile centred on the origin covers half a frame behind it, so n
    tiles reach ``(n − ½) · pitch`` past the start: the area that needs
    exactly n is just inside that.
    """
    kw.setdefault("x0_um", 0.0)
    kw.setdefault("y0_um", 0.0)
    kw.setdefault("overlap", 0.0)
    return ScanParams(width_um=(nx - 0.5) * 100 - 1,
                      height_um=(ny - 0.5) * 100 - 1, **kw)


def test_grid_shape_matches_the_plan_the_scanner_walks():
    """A preview that disagrees with the run is worse than no preview."""
    scanner = GridScanner(stage=None)
    params = _params(nx=20, ny=10)
    for fov in ((700.0, 390.0), (1000.0, 1000.0), (250.0, 120.0)):
        waypoints = scanner.plan(params, fov)
        cols, rows = grid_shape(params, fov)
        assert len(waypoints) == cols * rows
        assert waypoints[-1].index == cols * rows - 1


def test_grid_shape_is_never_empty():
    # a grid larger than the scan area still has one waypoint
    assert grid_shape(_params(nx=1, ny=1, overlap=0.0), (700.0, 390.0)) == (1, 1)


def test_pitch_is_the_fov_less_the_overlap():
    params = _params(nx=4, ny=4, overlap=0.25)
    assert plan_steps(params, FOV) == pytest.approx((75.0, 75.0))
    assert grid_shape(params, FOV) == (5, 5)   # overlap tiles the area more
                                               # densely, it never shrinks
                                               # the grid


@pytest.mark.parametrize("path", PATHS)
def test_every_path_covers_the_grid_exactly_once(path):
    """The tile set is the contract: same cells whatever the order."""
    for nx, ny in GRIDS:
        cells = plan_cells(_params(nx, ny, path=path), FOV)
        assert len(cells) == nx * ny, f"{path} {nx}x{ny}: wrong tile count"
        # the length check above is what catches a repeated cell
        assert set(cells) == {(c, r) for r in range(ny) for c in range(nx)}, \
            f"{path} {nx}x{ny}: cells are not a permutation of the grid"


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("x_dir,y_dir", [(1, 1), (-1, 1), (1, -1), (-1, -1)])
def test_waypoints_stay_inside_the_rectangle(path, x_dir, y_dir):
    """A direction sign must mirror the area, never step outside it."""
    nx, ny = 4, 3
    params = _params(nx, ny, path=path, x_dir=x_dir, y_dir=y_dir)
    waypoints = plan_path(params, FOV)
    xs = [w.x_um for w in waypoints]
    ys = [w.y_um for w in waypoints]
    assert min(xs) == pytest.approx(min(0.0, x_dir * (nx - 1) * 100.0))
    assert max(xs) == pytest.approx(max(0.0, x_dir * (nx - 1) * 100.0))
    assert min(ys) == pytest.approx(min(0.0, y_dir * (ny - 1) * 100.0))
    assert max(ys) == pytest.approx(max(0.0, y_dir * (ny - 1) * 100.0))


@pytest.mark.parametrize("path", PATHS)
def test_the_first_waypoint_is_the_start_point_when_centred(path):
    """'Scan from here' in the default origin: the operator is already on
    the first tile."""
    params = _params(4, 3, path=path, x0_um=-120.5, y0_um=44.0,
                     x_dir=-1, y_dir=-1)
    first = plan_path(params, FOV)[0]
    assert (first.x_um, first.y_um) == pytest.approx((-120.5, 44.0))


# --- the origin modes ------------------------------------------------------

#: Areas to prove coverage on, including the ones that expose the
#: half-frame overhang: an area just past a whole number of pitches, and
#: one that fits in a single frame.
AREAS = [(2000.0, 1000.0), (399.0, 299.0), (900.0, 900.0), (60.0, 40.0),
         (1234.0, 567.0), (5000.0, 3000.0)]
BIG_FOV = (1000.0, 700.0)


@pytest.mark.parametrize("origin", ORIGINS)
@pytest.mark.parametrize("width,height", AREAS)
@pytest.mark.parametrize("overlap", (0.0, 0.10, 0.5))
def test_every_origin_mode_covers_the_area_it_was_given(origin, width, height,
                                                        overlap):
    """The union of the tile footprints must cover the rectangle the
    operator typed — that is the whole promise of a scan.

    This is the test that was missing. The centre origin used to plan
    ``ceil(area / pitch)`` tiles, and because half a frame of the covered
    span lies BEHIND the start point, the far edge fell short whenever
    the remainder landed in that last half-frame: on the bench 5× a
    2000 µm area was covered only to 1912 µm.
    """
    params = ScanParams(x0_um=137.5, y0_um=-42.0, width_um=width,
                        height_um=height, overlap=overlap, origin=origin)
    for x_dir, y_dir in ((1, 1), (-1, -1)):
        params.x_dir, params.y_dir = x_dir, y_dir
        waypoints = plan_path(params, BIG_FOV)
        xs = [w.x_um for w in waypoints]
        ys = [w.y_um for w in waypoints]
        near_x, far_x = sorted((params.x0_um, params.x0_um + x_dir * width))
        near_y, far_y = sorted((params.y0_um, params.y0_um + y_dir * height))
        assert min(xs) - BIG_FOV[0] / 2 <= near_x + 1e-6
        assert max(xs) + BIG_FOV[0] / 2 >= far_x - 1e-6
        assert min(ys) - BIG_FOV[1] / 2 <= near_y + 1e-6
        assert max(ys) + BIG_FOV[1] / 2 >= far_y - 1e-6


@pytest.mark.parametrize("origin", ORIGINS)
@pytest.mark.parametrize("width,height", AREAS)
def test_the_preview_and_the_run_agree_on_the_tile_count(origin, width,
                                                         height):
    params = ScanParams(x0_um=0.0, y0_um=0.0, width_um=width,
                        height_um=height, origin=origin)
    geometry = plan_geometry(params, BIG_FOV)
    assert grid_shape(params, BIG_FOV) == (geometry.cols, geometry.rows)
    assert len(plan_path(params, BIG_FOV)) == geometry.count


def test_the_corner_modes_put_the_first_tile_edge_on_the_corner():
    """The operator stands on the corner: half a frame of the first tile
    is REGAINED rather than spent behind them."""
    params = ScanParams(x0_um=0.0, y0_um=0.0, width_um=2000.0,
                        height_um=1000.0, origin="corner_fit")
    first = plan_path(params, BIG_FOV)[0]
    assert first.x_um == pytest.approx(BIG_FOV[0] / 2.0)
    assert first.y_um == pytest.approx(BIG_FOV[1] / 2.0)


def test_corner_fit_lands_exactly_on_the_far_edge_and_uses_fewest_tiles():
    """The far edge is covered to the micron — no overhang, and no strip
    left unimaged."""
    params = ScanParams(x0_um=0.0, y0_um=0.0, width_um=2000.0,
                        height_um=1000.0, origin="corner_fit")
    waypoints = plan_path(params, BIG_FOV)
    assert max(w.x_um for w in waypoints) + BIG_FOV[0] / 2 == \
        pytest.approx(2000.0)
    # and it is the cheapest of the three ways to get there
    counts = {origin: plan_geometry(
        ScanParams(x0_um=0.0, y0_um=0.0, width_um=2000.0, height_um=1000.0,
                   origin=origin), BIG_FOV).count
        for origin in ORIGINS}
    assert counts["corner_fit"] < counts["corner_pitch"]
    assert counts["corner_fit"] <= counts["centre"]


def test_corner_pitch_keeps_the_requested_step():
    params = ScanParams(x0_um=0.0, y0_um=0.0, width_um=2000.0,
                        height_um=1000.0, origin="corner_pitch")
    geometry = plan_geometry(params, BIG_FOV)
    assert geometry.step_x == pytest.approx(
        plan_steps(params, BIG_FOV)[0])
    assert geometry.step_y == pytest.approx(
        plan_steps(params, BIG_FOV)[1])


def test_an_area_smaller_than_one_frame_is_one_tile_in_the_middle():
    """A corner origin with an area the frame swallows: the tile is put
    over the middle of the area, not offset by half a frame."""
    params = ScanParams(x0_um=10.0, y0_um=20.0, width_um=300.0,
                        height_um=200.0, origin="corner_fit")
    waypoints = plan_path(params, BIG_FOV)
    assert len(waypoints) == 1
    assert waypoints[0].x_um == pytest.approx(10.0 + 150.0)
    assert waypoints[0].y_um == pytest.approx(20.0 + 100.0)


def test_an_unknown_origin_falls_back_to_centre():
    """A settings file from the future must not produce a random walk."""
    odd = ScanParams(x0_um=0.0, y0_um=0.0, width_um=2000.0, height_um=1000.0,
                     origin="wat")
    plain = ScanParams(x0_um=0.0, y0_um=0.0, width_um=2000.0,
                       height_um=1000.0)
    assert plan_geometry(odd, BIG_FOV) == plan_geometry(plain, BIG_FOV)


def test_serpentine_rows_alternate_and_uni_does_not():
    bi = plan_cells(_params(4, 3, serpentine=True), FOV)
    assert [c for c, _ in bi[:4]] == [0, 1, 2, 3]
    assert [c for c, _ in bi[4:8]] == [3, 2, 1, 0]
    assert [c for c, _ in bi[8:]] == [0, 1, 2, 3]

    uni = plan_cells(_params(4, 3, serpentine=False), FOV)
    assert [c for c, _ in uni] == [0, 1, 2, 3] * 3


def test_one_way_is_a_path_name_for_the_same_cells():
    """The panel asks the walk question ONCE now, so ``one_way`` is a path
    the CV layer understands — and it has to be exactly what the old
    ``serpentine=False`` meant, because a stored configuration still says
    that and must not change meaning."""
    one_way = plan_cells(_params(4, 3, path=ONE_WAY), FOV)
    flag = plan_cells(_params(4, 3, serpentine=False), FOV)
    assert one_way == flag
    assert one_way != plan_cells(_params(4, 3, serpentine=True), FOV)
    assert ONE_WAY in PATHS


def test_start_axis_y_runs_the_columns_first():
    cells = plan_cells(_params(3, 2, start_axis="y"), FOV)
    assert cells[:2] == [(0, 0), (0, 1)]
    assert cells[2:4] == [(1, 1), (1, 0)]      # bi-directional column
    assert cells[4:] == [(2, 0), (2, 1)]


def test_unknown_path_falls_back_to_the_serpentine():
    """A settings file from the future must not produce a random walk."""
    params = _params(4, 3, path="wat")
    assert plan_cells(params, FOV) == plan_cells(_params(4, 3), FOV)


def test_spiral_and_hilbert_start_at_a_corner_and_stay_local():
    """Sanity on the orders themselves — the coverage test above is the
    real proof, this one catches a silently-degenerate implementation."""
    spiral = plan_cells(_params(5, 4, path="spiral"), FOV)
    assert spiral[0] == (0, 0)
    last_x, last_y = spiral[-1]
    assert 1 <= last_x <= 2 and 1 <= last_y <= 2   # works inward to the middle

    hilbert = plan_cells(_params(4, 4, path="hilbert"), FOV)
    assert hilbert[0] == (0, 0)
    # adjacent cells in the order are neighbours in the grid (that is the
    # whole point of the curve)
    for (x0, y0), (x1, y1) in zip(hilbert, hilbert[1:]):
        assert abs(x0 - x1) + abs(y0 - y1) == 1
