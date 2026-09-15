"""LiveViewWidget: frame bookkeeping for the ROI overlay.

The overlay surface paints AFTER the parent's paintEvent has run
_render_pending (which consumes and nulls the pending frame), so anything
that reads "_pending" from the overlay path sees nothing on every
streamed frame.
"""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from talos.ui.widgets.live_view import LiveViewWidget


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _frame(h: int = 240, w: int = 320) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_roi_rect_survives_the_render_that_consumes_the_frame(qapp):
    """Regression: the ROI rectangle was computed from the pending frame,
    which the parent paint had just consumed — the rubber band vanished
    (and flickered at frame rate) instead of staying on screen."""
    view = LiveViewWidget()
    view.resize(640, 480)
    view.show_frame(_frame())
    view.set_roi((0.25, 0.25, 0.5, 0.5))
    view._render_pending()          # the paint that eats the pending frame
    rect = view._roi_rect()
    assert not rect.isEmpty()
    # and it is the requested quarter of the letterboxed frame
    assert rect.width() == pytest.approx(0.5 * 320 * 480 / 240, rel=0.02)


def test_roi_rect_empty_without_a_roi(qapp):
    view = LiveViewWidget()
    view.resize(640, 480)
    view.show_frame(_frame())
    view._render_pending()
    assert view._roi_rect().isEmpty()


def test_roi_rect_empty_before_the_first_frame(qapp):
    view = LiveViewWidget()
    view.resize(640, 480)
    view.set_roi((0.25, 0.25, 0.5, 0.5))
    assert view._roi_rect().isEmpty()


# --- baked overlays (inverse-video crosshair + calibrated ruler) ----------

def _grey_frame(value: int = 90, h: int = 240, w: int = 320) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _rendered(view) -> "QImage":
    """The pixmap the label currently shows, as an image we can sample."""
    return view._label.pixmap().toImage()


def test_crosshair_is_inverse_video_and_solid(qapp):
    """Minecraft-style: the line inverts the pixels under it, so it is
    visible on any image (and it is CLIPPED to the frame, not drawn across
    the letterbox bars)."""
    view = LiveViewWidget()
    view.resize(640, 480)
    view.show_frame(_grey_frame(90))
    view._render_pending()
    plain = _rendered(view)
    view.set_crosshair_enabled(True)
    on = _rendered(view)

    assert plain.pixelColor(10, 10).red() == 90     # away from the lines
    cx, cy = on.width() // 2, on.height() // 2
    # |90 - 255| = 165 — an inversion, not a fixed colour
    assert on.pixelColor(cx, 40).red() == pytest.approx(165, abs=2)
    assert on.pixelColor(40, cy).red() == pytest.approx(165, abs=2)
    assert on.pixelColor(10, 10).red() == 90        # nothing else changed
    # solid: the neighbouring pixel along the line is inverted too
    assert on.pixelColor(cx, 41).red() == pytest.approx(165, abs=2)

    view.set_crosshair_enabled(False)
    assert _rendered(view).pixelColor(cx, 40).red() == 90


def test_crosshair_does_not_paint_the_cached_frame(qapp):
    """_compose copies the frame pixmap: the cached frame must stay clean,
    or the next compose would invert the inversion (flicker)."""
    view = LiveViewWidget()
    view.resize(640, 480)
    view.show_frame(_grey_frame(90))
    view._render_pending()
    view.set_crosshair_enabled(True)
    view.set_crosshair_enabled(False)
    view.set_crosshair_enabled(True)
    cx = _rendered(view).width() // 2
    assert _rendered(view).pixelColor(cx, 40).red() == pytest.approx(165, abs=2)


def test_crosshair_needs_a_frame(qapp):
    """Before the first frame there is no pixmap to bake into — the label
    must keep its placeholder instead of being blanked."""
    view = LiveViewWidget()
    view.resize(640, 480)
    view.set_crosshair_enabled(True)
    assert view._frame_pixmap is None
    assert not view._label.pixmap().isNull() or view._label.text() == "No camera"


def test_ruler_draws_ticks_along_the_frame_edges(qapp):
    view = LiveViewWidget()
    view.resize(640, 480)
    view.show_frame(_grey_frame(90))
    view._render_pending()
    view.set_live_calibration(1.0)          # 1 µm per 4K-sensor pixel
    view.set_ruler_enabled(True)
    img = _rendered(view)
    top = [img.pixelColor(x, 2).red() for x in range(img.width())]
    # at least a handful of ticks (inverted → 165) along the top edge
    assert sum(1 for value in top if abs(value - 165) <= 2) >= 8
    left = [img.pixelColor(2, y).red() for y in range(img.height())]
    assert sum(1 for value in left if abs(value - 165) <= 2) >= 8

    view.set_ruler_enabled(False)
    assert _rendered(view).pixelColor(2, 2).red() == 90


def test_ruler_needs_calibration(qapp):
    view = LiveViewWidget()
    view.resize(640, 480)
    view.show_frame(_grey_frame(90))
    view._render_pending()
    view.set_ruler_enabled(True)            # no calibration yet
    assert _rendered(view).pixelColor(2, 2).red() == 90
