"""Scan-path planning: every order must cover the rectangle exactly once,
and the preview must be the grid the scan actually walks."""

import pytest

from talos.cv.scan import (PATHS, GridScanner, grid_shape, plan_cells,
                           plan_path, plan_steps)
from talos.models import ScanParams

FOV = (100.0, 100.0)

#: Rectangles worth proving: the degenerate ones (a single row / column)
#: are where a spiral or a clipped curve loses a cell.
GRIDS = [(1, 1), (1, 5), (5, 1), (2, 2), (3, 2), (4, 3), (7, 5), (9, 1)]


def _params(nx=1, ny=1, **kw) -> ScanParams:
    """Params whose grid is exactly nx × ny at a 100 µm pitch."""
    kw.setdefault("x0_um", 0.0)
    kw.setdefault("y0_um", 0.0)
    kw.setdefault("overlap", 0.0)
    return ScanParams(width_um=nx * 100 - 1, height_um=ny * 100 - 1, **kw)


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
    assert grid_shape(params, FOV) == (6, 6)   # ceil(399 / 75) — overlap
                                               # tiles the area, it never
                                               # shrinks the grid


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
def test_the_first_waypoint_is_the_start_point(path):
    """'Scan from here': the operator is already on the first tile."""
    params = _params(4, 3, path=path, x0_um=-120.5, y0_um=44.0,
                     x_dir=-1, y_dir=-1)
    first = plan_path(params, FOV)[0]
    assert (first.x_um, first.y_um) == pytest.approx((-120.5, 44.0))


def test_serpentine_rows_alternate_and_uni_does_not():
    bi = plan_cells(_params(4, 3, serpentine=True), FOV)
    assert [c for c, _ in bi[:4]] == [0, 1, 2, 3]
    assert [c for c, _ in bi[4:8]] == [3, 2, 1, 0]
    assert [c for c, _ in bi[8:]] == [0, 1, 2, 3]

    uni = plan_cells(_params(4, 3, serpentine=False), FOV)
    assert [c for c, _ in uni] == [0, 1, 2, 3] * 3


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
