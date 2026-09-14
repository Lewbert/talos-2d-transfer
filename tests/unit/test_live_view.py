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
