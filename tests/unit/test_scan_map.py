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
