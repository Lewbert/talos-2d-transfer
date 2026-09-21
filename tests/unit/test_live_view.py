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


def _inverted_columns(img, y0: int, y1: int) -> list[int]:
    """X positions holding an inverted (≈165) pixel in a horizontal band."""
    hits = []
    for x in range(img.width()):
        column = [img.pixelColor(x, y).red() for y in range(y0, y1)]
        if any(abs(v - 165) <= 3 for v in column):
            hits.append(x)
    return hits


def test_ruler_does_not_label_the_centre(qapp):
    """The crosshair marks the optical axis; a "0" numeral in the same few
    pixels only fights with it."""
    view = LiveViewWidget()
    view.resize(640, 480)
    view.show_frame(_grey_frame(90))
    view._render_pending()
    view.set_live_calibration(1.0)
    view.set_ruler_enabled(True)          # crosshair deliberately OFF
    img = _rendered(view)
    # band just under the top ticks: label glyphs only
    hits = _inverted_columns(img, 11, 22)
    assert hits, "the ruler drew no labels at all"
    centre = img.width() // 2
    assert all(abs(x - centre) > 10 for x in hits), \
        "a numeral is drawn on the optical axis"


def test_crosshair_ticks_draw_a_calibrated_reticle(qapp):
    view = LiveViewWidget()
    view.resize(640, 480)
    view.show_frame(_grey_frame(90))
    view._render_pending()
    view.set_live_calibration(1.0)
    view.set_crosshair_enabled(True)
    plain = _rendered(view)
    view.set_crosshair_ticks_enabled(True)
    ticked = _rendered(view)

    cx, cy = ticked.width() // 2, ticked.height() // 2
    # off the crossed lines: nothing in either image
    assert plain.pixelColor(cx + 40, cy + 40).red() == 90
    new = [(x, y) for x in range(ticked.width())
           for y in (cy - 3, cy + 3)
           if abs(ticked.pixelColor(x, y).red() - 165) <= 3
           and plain.pixelColor(x, y).red() == 90]
    assert new, "no ticks along the horizontal crosshair line"
    assert all(abs(x - cx) > 4 for x, _ in new), "a tick sits on the cross"


def test_crosshair_ticks_need_the_crosshair_and_a_calibration(qapp):
    view = LiveViewWidget()
    view.resize(640, 480)
    view.show_frame(_grey_frame(90))
    view._render_pending()
    view.set_crosshair_ticks_enabled(True)     # no calibration, no crosshair
    assert _rendered(view).pixelColor(200, 120).red() == 90


def test_ruler_needs_calibration(qapp):
    view = LiveViewWidget()
    view.resize(640, 480)
    view.show_frame(_grey_frame(90))
    view._render_pending()
    view.set_ruler_enabled(True)            # no calibration yet
    assert _rendered(view).pixelColor(2, 2).red() == 90


# --- three views, three buffers --------------------------------------------

def test_the_mode_bar_offers_exactly_the_three_views(qapp):
    from talos.ui.widgets.live_view import (VIEW_MODES, VIEW_MODE_BUTTONS,
                                            LiveViewModeBar)
    bar = LiveViewModeBar(LiveViewWidget())
    assert tuple(mode for mode, _ in VIEW_MODE_BUTTONS) == VIEW_MODES
    assert tuple(bar.buttons) == VIEW_MODES
    for mode in VIEW_MODES:
        assert bar.buttons[mode].text()
        assert bar.buttons[mode].toolTip()
    # exclusive: checking one unchecks the last
    bar.buttons["samples"].setChecked(True)
    assert not bar.buttons["original"].isChecked()


def test_each_mode_shows_its_own_buffer(qapp):
    view = LiveViewWidget()
    live = _frame()
    filtered = np.full((240, 320, 3), 90, np.uint8)
    overlay = np.full((240, 320, 3), 200, np.uint8)
    view.show_frame(live)
    view.set_preprocessed_frame(filtered)
    view.set_overlay_frame(overlay)

    assert view._shown_frame() is live
    view.set_view_mode("preprocessed")
    assert view._shown_frame() is filtered
    view.set_view_mode("samples")
    assert view._shown_frame() is overlay
    # and back: the live stream was never disturbed
    view.set_view_mode("original")
    assert view._shown_frame() is live


def test_a_processed_view_falls_back_to_the_stream_until_it_arrives(qapp):
    """A mode switched on before the worker has produced anything must not
    blank the view — the operator is watching a live microscope."""
    view = LiveViewWidget()
    live = _frame()
    view.show_frame(live)
    view.set_view_mode("preprocessed")
    assert view._shown_frame() is live
    view.set_view_mode("samples")
    assert view._shown_frame() is live


def test_an_unknown_mode_falls_back_to_the_stream(qapp):
    view = LiveViewWidget()
    view.set_view_mode("live")             # the pre-third-button name
    assert view.view_mode == "original"
    view.set_view_mode("nonsense")
    assert view.view_mode == "original"


def test_the_dropper_samples_the_preprocessed_layer_never_the_display(qapp):
    """In samples mode the screen is darkened and outlined: sampling it
    would return a colour the sample does not have."""
    view = LiveViewWidget()
    live = np.full((240, 320, 3), 30, np.uint8)
    filtered = np.full((240, 320, 3), 130, np.uint8)
    overlay = np.full((240, 320, 3), 240, np.uint8)
    view.show_frame(live)
    assert view.pick_frame() is live          # nothing filtered yet
    view.set_preprocessed_frame(filtered)
    view.set_overlay_frame(overlay)
    view.set_view_mode("samples")
    assert view.pick_frame() is filtered


# --- held-back processed views -------------------------------------------

def test_a_paused_processed_view_shows_the_live_frame(qapp):
    """While the stage moves or a scan runs, the two processed buffers are
    stale and expensive: the display falls back to the stream, whatever the
    mode bar says."""
    view = LiveViewWidget()
    view.resize(640, 480)
    live, processed = _grey_frame(50), _grey_frame(200)
    view.set_view_mode("preprocessed")
    view.show_frame(live)
    view.set_preprocessed_frame(processed)
    assert view._shown_frame() is processed

    view.set_processed_paused(True, "stage moving — showing the live frame")
    assert view._shown_frame() is live
    assert view.processed_paused

    view.set_processed_paused(False)
    assert view._shown_frame() is processed
    assert not view.processed_paused


def test_the_dropper_keeps_sampling_the_processed_layer_while_paused(qapp):
    """The pause is a DISPLAY decision. The mask searches the pre-processed
    pixels, so a colour picked off a paused screen must still be the colour
    the chain produced."""
    view = LiveViewWidget()
    live, processed = _grey_frame(50), _grey_frame(200)
    view.show_frame(live)
    view.set_preprocessed_frame(processed)
    view.set_processed_paused(True, "scanning — showing the live frame")
    assert view.pick_frame() is processed


def test_a_paused_view_repaints_when_it_is_released(qapp):
    """Otherwise the stale processed frame stays on screen until the next
    streamed frame happens to arrive."""
    view = LiveViewWidget()
    view.resize(640, 480)
    view.set_view_mode("samples")
    view.show_frame(_grey_frame(50))
    view.set_overlay_frame(_grey_frame(200))
    view.set_processed_paused(True, "scanning")
    view.show_frame(_grey_frame(60))
    view._render_pending()

    view.set_processed_paused(False)
    assert view._pending is view._overlay_frame


def test_the_pause_note_is_only_kept_while_paused(qapp):
    view = LiveViewWidget()
    view.set_processed_paused(True, "scanning — showing the live frame")
    assert view._pause_note == "scanning — showing the live frame"
    view.set_processed_paused(False, "ignored")
    assert view._pause_note == ""
