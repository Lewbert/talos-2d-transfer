"""Scale-bar geometry: the ≤-ladder, the shared layout spec, and the
cv2 burn (label under the bar, equal right/bottom margins)."""

import numpy as np

from talos.cv.scale_bar import (
    ScaleBarSpec,
    burn_spec,
    draw_scale_bar_cv,
    format_length_um,
    nice_length_um_at_most,
    scale_bar_layout,
    scale_bar_px,
)



def test_nice_length_at_most_never_exceeds_the_fraction():
    # max 240 µm at 0.5 µm/px / 1920 px → largest ladder ≤ 240 → 200
    assert nice_length_um_at_most(0.5, 1920) == 200.0
    # 1 µm/px: max 480 → largest ladder ≤ 480 → 200 (500 would exceed)
    assert nice_length_um_at_most(1.0, 1920) == 200.0
    # the overshoot trap: a target just above 200 must NOT give 500
    assert nice_length_um_at_most(0.45, 1920) == 200.0  # max 216
    # fractional max: 0.1 µm/px → max 48 → 20
    assert nice_length_um_at_most(0.1, 1920) == 20.0
    # the ≤ 1/4 guarantee across a sweep of calibrations
    for um_per_px in (0.01, 0.05, 0.1, 0.3, 0.7, 1.42, 2.0, 5.0):
        length = nice_length_um_at_most(um_per_px, 1920)
        assert scale_bar_px(length, um_per_px) <= 1920 * 0.25 + 1
    assert nice_length_um_at_most(0.0, 1920) == 0.0


def test_scale_bar_layout_equal_margins_and_text_under_bar():
    spec = scale_bar_layout(0.5, (1080, 1920, 3), margin_px=14)
    assert isinstance(spec, ScaleBarSpec)
    # EQUAL spacing to the image right and bottom edges
    x1, y1, x2, y2 = spec.box
    assert 1920 - x2 == 14
    assert 1080 - y2 == 14
    # bar ≤ 1/4 of the image width
    assert spec.bar_px <= 1920 * 0.25
    # the text band sits strictly UNDER the bar (no overlap)
    bx, by, bw, bh = spec.bar_rect
    tx, ty, tw, th_band = spec.text_rect
    assert ty >= by + bh
    assert spec.label == "200 µm"
    # the box hugs both
    assert x1 <= bx and tx <= x2 and y2 >= ty + th_band - 1


def test_scale_bar_layout_invalid():
    assert scale_bar_layout(0.0, (1080, 1920, 3)) is None
    assert scale_bar_layout(None, (1080, 1920, 3)) is None
    assert scale_bar_layout(0.5, (0, 0, 3)) is None


def test_scale_bar_px_and_format():
    assert scale_bar_px(200.0, 0.5) == 400
    assert scale_bar_px(200.0, 0.0) == 0
    assert format_length_um(500.0) == "500 µm"
    assert format_length_um(1500.0) == "1.5 mm"
    assert format_length_um(0.5) == "500 nm"


def test_burn_spec():
    assert burn_spec(0.5, True) == {"um_per_px": 0.5}
    assert burn_spec(None, True) is None
    assert burn_spec(0.5, False) is None
    assert burn_spec(0.0, True) is None


def test_draw_scale_bar_cv():
    frame = np.full((1080, 1920, 3), 80, dtype=np.uint8)
    out = draw_scale_bar_cv(frame, 0.5)
    assert out.shape == frame.shape
    # The bar/box region must have changed (see the layout test for the
    # exact box: right edge 1906, bottom 1066).
    region_before = frame[1000:1066, 1450:1906].sum()
    region_after = out[1000:1066, 1450:1906].sum()
    assert region_after != region_before


def test_draw_scale_bar_cv_invalid_returns_identity():
    frame = np.full((100, 100, 3), 80, dtype=np.uint8)
    assert draw_scale_bar_cv(frame, 0.0) is frame
