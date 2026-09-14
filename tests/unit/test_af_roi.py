"""af_roi: letterbox widget↔frame mapping, normalized ROI round-trips,
degeneracy fallback."""

import pytest

from talos.cv.af_roi import (
    is_degenerate,
    letterbox_map,
    letterbox_rect,
    normalized_roi,
    roi_for_resolution,
)


# 1920×1080 into an 800×600 widget: scale = 800/1920 = 5/12 ≈ 0.41667,
# scaled image 800×450, letterbox bands top/bottom of 75 px each.
WIDGET = (800, 600)
FRAME = (1080, 1920, 3)
SCALE = 800 / 1920


def test_letterbox_map_center_is_frame_center():
    fx, fy = letterbox_map(WIDGET, FRAME, (400, 300))
    assert fx == pytest.approx(960.0)
    assert fy == pytest.approx(540.0)


def test_letterbox_map_corners():
    # top-left of the IMAGE sits at widget (0, 75)
    assert letterbox_map(WIDGET, FRAME, (0, 75)) == pytest.approx((0.0, 0.0))
    assert letterbox_map(WIDGET, FRAME, (800, 525)) == pytest.approx((1920.0, 1080.0))


def test_letterbox_map_into_letterbox_band_goes_out_of_frame():
    # a point inside the black band maps above the image
    fx, fy = letterbox_map(WIDGET, FRAME, (400, 10))
    assert fy < 0


def test_letterbox_rect_full_image_area():
    rect = letterbox_rect(WIDGET, FRAME, (0, 75, 800, 450))
    assert rect == pytest.approx((0.0, 0.0, 1920.0, 1080.0))


def test_letterbox_rect_clamps_overshoot():
    # drag extends into the letterbox band → clamped to the image
    rect = letterbox_rect(WIDGET, FRAME, (400, 0, 400, 600))
    fx0, fy0, w, h = rect
    assert fy0 == pytest.approx(0.0)
    assert fy0 + h == pytest.approx(1080.0)


def test_letterbox_rect_degenerate_is_none():
    assert letterbox_rect(WIDGET, FRAME, (400, 300, 0, 0)) is None
    assert letterbox_rect(WIDGET, FRAME, (400, 300, -5, 10)) is None


def test_letterbox_rect_square_widget():
    # 1080×1080 widget: scale limited by height, bands left/right
    rect = letterbox_rect((1080, 1080), FRAME, (0, 0, 1080, 1080))
    assert rect == pytest.approx((0.0, 0.0, 1920.0, 1080.0))


def test_normalized_roi_round_trip_across_resolutions():
    norm = normalized_roi((192, 108, 960, 540), FRAME)  # center half
    assert norm == pytest.approx((0.1, 0.1, 0.5, 0.5))
    # same selection at 1280×720
    roi = roi_for_resolution(norm, (720, 1280, 3))
    assert roi == (128, 72, 640, 360)
    # and back at 4K
    roi4k = roi_for_resolution(norm, (2160, 3840, 3))
    assert roi4k == (384, 216, 1920, 1080)


def test_normalized_roi_clamps_partial_out_of_frame():
    norm = normalized_roi((-100, -50, 500, 400), FRAME)
    assert norm[0] == 0.0
    assert norm[1] == 0.0
    assert norm[2] == pytest.approx(400 / 1920)
    assert norm[3] == pytest.approx(350 / 1080)


def test_normalized_roi_degenerate_is_none():
    assert normalized_roi((10, 10, 0, 10), FRAME) is None
    assert normalized_roi((10, 10, 10, 0), FRAME) is None


def test_roi_for_resolution_none_passthrough():
    assert roi_for_resolution(None, FRAME) is None


def test_is_degenerate_thresholds():
    full = (0, 0, 1920, 1080)
    assert not is_degenerate(full, FRAME)
    tiny_area = (0, 0, 100, 100)      # 0.5% of the frame
    assert is_degenerate(tiny_area, FRAME)
    thin = (0, 0, 1920, 4)            # side < 8 px
    assert is_degenerate(thin, FRAME)
    assert is_degenerate((0, 0, 0, 0), FRAME)
