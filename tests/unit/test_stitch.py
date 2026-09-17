"""Mosaic and tile-overview assembly for a finished scan."""

import numpy as np
import pytest

from talos.cv.stitch import build_mosaic, build_overview
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


def test_y_increases_downwards_like_an_image():
    mosaic = build_mosaic([(0.0, 0.0, tile(50)), (0.0, 800.0, tile(200))], FOV)
    assert mosaic.shape == (160, 100, 3)
    assert int(mosaic[40, 50, 0]) == 50
    assert int(mosaic[120, 50, 0]) == 200


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


# --- the tile overview -----------------------------------------------------

def test_overview_puts_a_thumbnail_per_tile_in_scan_order():
    entries = [(tile(40 + 10 * i), []) for i in range(4)]
    sheet = build_overview(entries, thumb_w=100)
    # 2 × 2 thumbnails at 100 × 80, 4 px padding around and between
    assert sheet.shape == (2 * 84 + 4, 2 * 104 + 4, 3)
    assert int(sheet[10, 10, 0]) == 40            # the first tile
    assert int(sheet[94, 10, 0]) == 60            # the third (row 2, col 1)


def test_overview_draws_the_detection_boxes():
    cand = FlakeCandidate(x_px=50.0, y_px=40.0, area_px2=100.0,
                          bbox=(20, 20, 40, 20))
    plain = build_overview([(tile(30), [])], thumb_w=100)
    boxed = build_overview([(tile(30), [cand])], thumb_w=100)
    assert np.array_equal(plain, boxed) is False
    # the box is drawn in the tint colour: bbox (20,20,40,20) scaled by the
    # thumbnail factor (1.0) and offset by the sheet's 4 px padding
    assert tuple(int(v) for v in boxed[30, 24]) == (0, 200, 255)
    assert tuple(int(v) for v in plain[30, 24]) == (30, 30, 30)


def test_overview_with_nothing_captured_is_not_an_error():
    assert build_overview([]) is None
    assert build_overview([(None, [])]) is None
