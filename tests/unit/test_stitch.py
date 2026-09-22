"""Mosaic assembly for a finished scan, and the rings drawn on it."""

import math

import numpy as np
import pytest

from talos.cv.stitch import (build_mosaic, draw_sample_rings,
                             mosaic_geometry)
from talos.models import FlakeCandidate


def tile(value: int, shape=(80, 100)) -> np.ndarray:
    return np.full((shape[0], shape[1], 3), value, np.uint8)


FOV = (1000.0, 800.0)      # 100 px / 1000 µm → 0.1 px per µm


def test_tiles_land_where_the_stage_was():
    """Two tiles 1000 µm apart, one FOV wide each: they end up side by side
    with no gap and no overlap."""
    mosaic = build_mosaic([(0.0, 0.0, tile(50)), (1000.0, 0.0, tile(200))],
                          FOV)
    assert mosaic.shape == (80, 200, 3)
    assert int(mosaic[40, 50, 0]) == 50      # first tile's own pixels
    assert int(mosaic[40, 150, 0]) == 200


def test_the_y_axis_follows_the_mounting():
    """Stage +Y is drawn UPWARD — the bench's vertical axis runs the other
    way from the assumption the first version made (see
    cv/orientation.py). The flip negates BOTH axes, so it is downward
    again with the flip on."""
    up = build_mosaic([(0.0, 0.0, tile(50)), (0.0, 800.0, tile(200))], FOV)
    assert up.shape == (160, 100, 3)
    assert int(up[120, 50, 0]) == 50         # y = 0 is the LOWER tile
    assert int(up[40, 50, 0]) == 200

    down = build_mosaic([(0.0, 0.0, tile(50)), (0.0, 800.0, tile(200))], FOV,
                        flip=True)
    assert down.shape == (160, 100, 3)
    assert int(down[40, 50, 0]) == 50
    assert int(down[120, 50, 0]) == 200


def test_overlapping_tiles_are_averaged_not_stamped():
    """A seam is what you get from overwriting; averaging is what makes an
    overlap look like one image."""
    mosaic = build_mosaic([(0.0, 0.0, tile(0)), (500.0, 0.0, tile(200))], FOV)
    assert mosaic.shape == (80, 150, 3)
    assert int(mosaic[40, 20, 0]) == 0            # first tile alone
    assert int(mosaic[40, 120, 0]) == 200         # second tile alone
    assert int(mosaic[40, 75, 0]) == pytest.approx(100, abs=1)   # the seam


def test_a_single_tile_is_just_that_tile():
    mosaic = build_mosaic([(0.0, 0.0, tile(77))], FOV)
    assert mosaic.shape == (80, 100, 3)
    assert np.all(mosaic == 77)


def test_nothing_to_draw_is_not_an_error():
    assert build_mosaic([], FOV) is None
    assert build_mosaic([(0.0, 0.0, None)], FOV) is None
    assert build_mosaic([(0.0, 0.0, tile(1))], (0.0, 0.0)) is None


def test_the_long_edge_is_capped():
    """A wide scan is summarised, not stored: the frames on disk are the
    data, and the mosaic has to stay openable."""
    tiles = [(i * 1000.0, 0.0, tile(120)) for i in range(40)]
    mosaic = build_mosaic(tiles, FOV, max_px=512)
    assert max(mosaic.shape[:2]) <= 512
    assert mosaic.shape[1] > mosaic.shape[0]      # still the right shape


def test_differently_sized_tiles_are_resized_to_match():
    big = np.full((160, 200, 3), 90, np.uint8)
    mosaic = build_mosaic([(0.0, 0.0, big), (1000.0, 0.0, tile(200))], FOV)
    assert mosaic.shape == (80, 200, 3)


# --- the sample rings on the mosaic ---------------------------------------

def _sample(x_um: float, y_um: float, area_um2: float) -> FlakeCandidate:
    return FlakeCandidate(x_px=50.0, y_px=40.0, area_px2=area_um2,
                          area_um2=area_um2, x_um=x_um, y_um=y_um,
                          bbox=(20, 20, 40, 20))


TILES = [(0.0, 0.0, None)]          # the tile is made inside the helpers
FOV_ONE = (100.0, 100.0)


def _one_tile_mosaic():
    tiles = [(0.0, 0.0, tile(40))]
    return build_mosaic(tiles, FOV_ONE), tiles


def _ring(mosaic, tiles, samples, **kwargs):
    return draw_sample_rings(mosaic, samples, tiles, FOV_ONE, **kwargs)


def test_a_ring_lands_on_the_sample_it_names():
    """A sample at the tile's centre is ringed at the image's centre: the
    ring uses the mosaic's own layout, not a re-derivation of it."""
    mosaic, tiles = _one_tile_mosaic()
    height, width = mosaic.shape[:2]
    _ring(mosaic, tiles, [_sample(0.0, 0.0, 400.0)])
    px_per_um, _x0, _y0, shrink = mosaic_geometry(tiles, FOV_ONE)
    radius = int(round(math.sqrt(400.0 / math.pi) * px_per_um * shrink))
    cx, cy = width // 2, height // 2
    # the ring's pixels are on the circle, not inside it
    assert tuple(int(v) for v in mosaic[cy, cx + radius]) == (0, 255, 90)
    assert tuple(int(v) for v in mosaic[cy, cx]) == (40, 40, 40)


def test_the_rings_are_numbered_like_the_sample_list():
    mosaic, tiles = _one_tile_mosaic()
    before = mosaic.copy()
    _ring(mosaic, tiles, [_sample(0.0, 0.0, 400.0)])
    assert not np.array_equal(before, mosaic)
    plain = before.copy()
    _ring(plain, tiles, [_sample(0.0, 0.0, 400.0)], label=False)
    # the label is the only difference between the two runs
    assert not np.array_equal(plain, mosaic)


def test_rings_survive_having_nothing_to_draw():
    mosaic, tiles = _one_tile_mosaic()
    untouched = mosaic.copy()
    assert _ring(mosaic, tiles, []) is mosaic
    assert np.array_equal(untouched, mosaic)
    assert draw_sample_rings(None, [_sample(0, 0, 100)], tiles,
                             FOV_ONE) is None
    assert _ring(mosaic, tiles, [_sample(500.0, 500.0, 100.0)]) is mosaic
    assert np.array_equal(untouched, mosaic), "off-canvas samples are skipped"
