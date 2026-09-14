"""Overlay helpers: AF phase colors + the scale-bar overlay geometry
(pure parts) — including the resize-accuracy regression."""

import pytest
from PySide6.QtGui import QColor

from talos.cv.af_roi import fit_transform
from talos.cv.scale_bar import scale_bar_layout, scale_bar_px
from talos.ui.theme import DANGER, TEXT_DIM, WARN
from talos.ui.widgets.overlay import af_phase_color, scale_bar_rect


def test_af_phase_colors_match_the_contract():
    # stage 1 (coarse family) = red
    for phase in (1, 4, 7):
        assert af_phase_color(phase).name() == QColor(DANGER).name()
    # stage 2 (fine family incl. landing) = orange
    for phase in (2, 3, 5, 6):
        assert af_phase_color(phase).name() == QColor(WARN).name()
    # unknown = dim
    assert af_phase_color(99).name() == QColor(TEXT_DIM).name()


def test_scale_bar_rect_anchors_to_the_frame_not_the_widget():
    # 16:9 frame (1920×1080) in a 4:3 widget (1600×1200): the frame
    # letterboxes to 1600×900 with 150px bars top and bottom.
    bar, label = scale_bar_rect(0.5, (1080, 1920, 3), (1600, 1200))
    assert not bar.isEmpty()
    assert label == "200 µm"
    # the bar's right edge sits inside the frame's right edge, NOT the
    # widget's (which would put it into the letterbox pillar)
    assert bar.right() <= 1600 - 14
    assert bar.top() > 150  # below the top letterbox bar


def test_scale_bar_rect_scales_with_the_letterbox():
    """The resize-accuracy regression: the drawn bar must span exactly
    spec.bar_px × the letterbox scale in widget pixels — before the fix
    the frame-pixel length was drawn 1:1 in widget coordinates and the
    bar's µm label lied at every non-frame-sized window."""
    um_per_px = 0.5
    frame_shape = (1080, 1920, 3)
    spec = scale_bar_layout(um_per_px, frame_shape)
    for widget_size in ((1920, 1080), (1600, 1200), (1024, 768),
                        (500, 400)):
        scale, _ox, _oy = fit_transform(widget_size, frame_shape)
        bar, _label = scale_bar_rect(um_per_px, frame_shape, widget_size)
        expected = spec.bar_px * scale
        assert bar.width() == pytest.approx(expected, abs=1e-6)
        # sanity: the spec ladder value must agree with the label width
        assert spec.bar_px == scale_bar_px(spec.length_um, um_per_px)


def test_scale_bar_rect_empty_without_calibration():
    bar, label = scale_bar_rect(0.0, (1080, 1920, 3), (1600, 1200))
    assert bar.isEmpty() and label == ""
