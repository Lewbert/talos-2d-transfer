"""The scan map's geometry: the sample frame, and what panning does to it.

The map is drawn in STAGE coordinates (what the readback reports), which is
also the sample's frame — the camera walks across a scene that stays put.
The transforms are pure functions so they can be checked without a screen.
"""

import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from talos.ui.widgets.scan_map import (ScanMapMarker, ScanMapPlan,
                                       ScanMapTile, ScanMapWidget, fit_view,
                                       plan_bounds)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _plan(**kw) -> ScanMapPlan:
    base = dict(x0_um=0.0, y0_um=0.0, width_um=4000.0, height_um=2000.0,
                fov_x_um=1000.0, fov_y_um=500.0,
                waypoints=[(0.0, 0.0), (900.0, 0.0), (900.0, 500.0)])
    base.update(kw)
    return ScanMapPlan(**base)


def test_bounds_cover_the_area_plus_half_a_footprint():
    """The corner footprints must fit inside the view, so the bounds pad
    the area by half a FOV on every side."""
    assert plan_bounds(_plan()) == pytest.approx((-500.0, -250.0, 4500.0,
                                                  2250.0))


def test_bounds_follow_the_direction_signs():
    """A scan that grows to −X/−Y from the start point is drawn to the
    LEFT/UP of it, not mirrored."""
    plan = _plan(x_dir=-1, y_dir=-1)
    x0, y0, x1, y1 = plan_bounds(plan)
    assert (x0, y0) == pytest.approx((-4500.0, -2250.0))
    assert (x1, y1) == pytest.approx((500.0, 250.0))


def test_an_empty_plan_has_no_bounds():
    assert plan_bounds(ScanMapPlan()) == (0.0, 0.0, 0.0, 0.0)


def test_fit_view_centres_and_scales_the_area():
    bounds = (0.0, 0.0, 1000.0, 500.0)
    scale, offset = fit_view(bounds, (1000, 500), margin_px=0)
    assert scale == pytest.approx(1.0)
    assert (offset.x(), offset.y()) == pytest.approx((0.0, 0.0))
    # a wider widget leaves the content centred, not stretched
    scale, offset = fit_view(bounds, (2000, 500), margin_px=0)
    assert scale == pytest.approx(1.0)
    assert offset.x() == pytest.approx(500.0)


def test_fit_view_keeps_the_aspect_ratio():
    scale, _offset = fit_view((0.0, 0.0, 1000.0, 1000.0), (400, 200),
                              margin_px=0)
    assert scale == pytest.approx(200.0 / 1000.0)     # the tighter axis wins


def test_y_grows_downwards():
    """The map shows the same way up as the mosaic and the live view."""
    scale, offset = fit_view((0.0, 0.0, 100.0, 100.0), (100, 100),
                             margin_px=0)
    y_small = offset.y() + 10.0 * scale
    y_large = offset.y() + 90.0 * scale
    assert y_small < y_large


def test_a_new_area_drops_the_tiles_but_the_same_area_keeps_them(qapp):
    """A finished run must not erase what it captured — the window
    re-states the plan when the job ends."""
    widget = ScanMapWidget()
    widget.set_plan(_plan())
    widget.add_tile(ScanMapTile(index=0, x_um=0.0, y_um=0.0))
    widget.set_markers([ScanMapMarker(x_um=0.0, y_um=0.0, label="1")])
    assert len(widget._tiles) == 1 and widget.marker_count == 1

    widget.set_plan(_plan())                      # identical: kept
    assert len(widget._tiles) == 1 and widget.marker_count == 1

    widget.set_plan(_plan(width_um=8000.0))       # a new area: dropped
    assert not widget._tiles and widget.marker_count == 0


def test_clear_tiles_keeps_the_plan(qapp):
    widget = ScanMapWidget()
    widget.set_plan(_plan())
    widget.add_tile(ScanMapTile(index=0, x_um=0.0, y_um=0.0))
    widget.clear_tiles()
    assert not widget._tiles
    assert len(widget._plan.waypoints) == 3


def test_tiles_are_placed_at_their_readback_position(qapp):
    widget = ScanMapWidget()
    widget.resize(800, 400)
    widget.set_plan(_plan())
    thumb = np.full((27, 48, 3), 120, np.uint8)
    widget.add_tile(ScanMapTile(index=0, x_um=0.0, y_um=0.0, thumb=thumb))
    widget.add_tile(ScanMapTile(index=1, x_um=3000.0, y_um=0.0, thumb=thumb))
    left = widget._to_widget(0.0, 0.0)
    right = widget._to_widget(3000.0, 0.0)
    assert right.x() > left.x()
    assert right.y() == pytest.approx(left.y())
    # the thumbnail was converted once, for painting
    assert set(widget._images) == {0, 1}


def test_the_map_follows_the_camera_flip(qapp):
    """The map is drawn in the sample frame AS THE FRAMES SHOW IT: with the
    flip on the whole layout mirrors, so a tile taken further along +X is
    drawn further along −x. Without that the operator sees a map whose
    orientation disagrees with the live view beside it (and a mosaic whose
    tiles land twice — the bug this mirrors).
    """
    widget = ScanMapWidget()
    widget.resize(800, 400)
    widget.set_plan(_plan())

    def drawn(x_um, y_um):
        """Where the map draws a stage coordinate (the drawing rule)."""
        return widget._to_widget(*widget._at(x_um, y_um))

    widget.set_flip(False)
    right_of_start = drawn(3000.0, 0.0).x() > drawn(0.0, 0.0).x()
    widget.set_flip(True)
    left_of_start = drawn(3000.0, 0.0).x() < drawn(0.0, 0.0).x()
    assert right_of_start and left_of_start, \
        "a stage +X offset must change sides with the flip"

    # the tile, the marker and the footprint all mirror together
    widget.set_markers([ScanMapMarker(x_um=3000.0, y_um=0.0, label="1")])
    assert widget._marker_at(drawn(3000.0, 0.0)) == 0

    # and the planned area keeps its size, only its side of the origin
    x0, y0, x1, y1 = plan_bounds(widget._effective_plan())
    assert (x1 - x0, y1 - y0) == pytest.approx((5000.0, 2500.0))


def test_a_marker_can_be_hit_and_selected(qapp):
    widget = ScanMapWidget()
    widget.resize(800, 400)
    widget.set_plan(_plan())
    widget.set_markers([ScanMapMarker(x_um=0.0, y_um=0.0, label="1"),
                        ScanMapMarker(x_um=3000.0, y_um=0.0, label="2")])
    point = widget._to_widget(3000.0, 0.0)
    assert widget._marker_at(point) == 1
    assert widget._marker_at(QPointF(point.x() + 200.0, point.y())) == -1


def test_zoom_keeps_the_centre_fixed(qapp):
    widget = ScanMapWidget()
    widget.resize(800, 400)
    widget.set_plan(_plan())
    before = widget._to_widget(0.0, 0.0)
    widget._zoom = 2.0
    after = widget._to_widget(0.0, 0.0)
    centre = QPointF(widget.width() / 2.0, widget.height() / 2.0)
    # the content moves AWAY from the centre by the zoom factor
    assert (after - centre).manhattanLength() > (before - centre).manhattanLength()
    widget.fit()
    assert widget._zoom == 1.0


# --- the "you are here" footprint ------------------------------------------

def test_the_footprint_is_drawn_at_the_stage_position(qapp):
    from talos.cv.orientation import mosaic_offset

    widget = ScanMapWidget()
    widget.resize(800, 400)
    widget.set_plan(_plan())
    widget.set_footprint(900.0, 500.0)
    assert widget._footprint == (900.0, 500.0)
    # through the SAME transform as the tiles and the markers — which is
    # the mosaic layout's, so the mounting's Y inversion applies here too
    assert (widget._at(*widget._footprint)
            == pytest.approx(mosaic_offset(900.0, 500.0, False)))
    assert widget._at(*widget._footprint)[1] == pytest.approx(-500.0)


def test_an_unknown_position_draws_nothing(qapp):
    """Regression: the footprint used to be fed zeroes whenever the
    position was unknown, and stage (0, 0) is a real place — the box sat
    there looking like a measurement, unrelated to the plan, on most
    frames."""
    widget = ScanMapWidget()
    widget.resize(800, 400)
    widget.set_plan(_plan())
    widget.set_footprint(100.0, 200.0)
    widget.set_footprint(None)
    assert widget._footprint is None
    widget.set_footprint(10.0, None)
    assert widget._footprint is None


def test_the_footprint_can_be_cleared(qapp):
    widget = ScanMapWidget()
    widget.resize(800, 400)
    widget.set_plan(_plan())
    widget.set_footprint(0.0, 0.0)
    widget.clear_footprint()
    assert widget._footprint is None


def test_the_footprint_is_a_survivable_kind_of_wrong_after_a_flip(qapp):
    """The flip mirrors the layout, so the footprint has to be mirrored
    with it — otherwise the box reports the camera on the wrong side of
    the sample."""
    widget = ScanMapWidget()
    widget.resize(800, 400)
    widget.set_plan(_plan())
    widget.set_footprint(900.0, 500.0)
    straight = widget._at(*widget._footprint)
    widget.set_flip(True)
    flipped = widget._at(*widget._footprint)
    assert flipped == pytest.approx((-straight[0], -straight[1]))


def test_the_footprint_uses_the_effective_plans_field_of_view(qapp):
    """It read the raw plan while everything else read the oriented one —
    harmless only while oriented() happened not to scale the field of view."""
    widget = ScanMapWidget()
    widget.resize(800, 400)
    widget.set_plan(_plan(fov_x_um=1000.0, fov_y_um=500.0))
    widget.set_flip(True)
    plan = widget._effective_plan()
    assert (plan.fov_x_um, plan.fov_y_um) == (1000.0, 500.0)


def test_a_double_click_asks_to_be_enlarged_rather_than_fitting(qapp):
    """Double-click used to call fit(); the map is the smallest panel and
    the one worth seeing big, so the gesture opens it instead. Fit moved
    to a button."""
    widget = ScanMapWidget()
    widget.resize(800, 400)
    widget.set_plan(_plan())
    widget._zoom = 3.0
    asked: list = []
    widget.sig_enlarge_requested.connect(lambda: asked.append(True))
    widget.mouseDoubleClickEvent(None)
    assert asked == [True]
    assert widget._zoom == 3.0          # the view was not reset
    widget.fit()
    assert widget._zoom == 1.0


def test_the_box_is_hidden_until_something_has_been_scanned(qapp):
    """One rule, three states. Idle with nothing scanned, the map shows
    the plan and its start dot and NO box: an empty rectangle drawn from a
    position nobody asked about is what made this control look broken."""
    widget = ScanMapWidget()
    widget.resize(800, 400)
    widget.set_plan(_plan())
    widget.set_footprint(900.0, 500.0)
    assert widget._highlight() is None


def test_the_box_follows_the_tile_a_run_is_capturing(qapp):
    """During a run it marks the newest frame in the mosaic; afterwards it
    goes back to being the live position, and it stops altogether when the
    tiles are cleared (a new run of the same area)."""
    widget = ScanMapWidget()
    widget.resize(800, 400)
    widget.set_plan(_plan())
    widget.set_footprint(900.0, 500.0)

    widget.add_tile(ScanMapTile(0, 100.0, 200.0, None))
    assert widget._highlight() == (900.0, 500.0, "here")

    widget.set_active_tile(1500.0, 700.0)
    assert widget._highlight() == (1500.0, 700.0, "tile")

    widget.set_active_tile(None)
    assert widget._highlight() == (900.0, 500.0, "here")

    widget.clear_tiles()
    assert widget._highlight() is None
    assert widget._active_tile is None


def test_a_new_plan_drops_the_tile_the_box_was_marking(qapp):
    widget = ScanMapWidget()
    widget.resize(800, 400)
    widget.set_plan(_plan())
    widget.set_active_tile(1500.0, 700.0)
    widget.set_plan(_plan(width_um=3000.0))
    assert widget._highlight() is None
