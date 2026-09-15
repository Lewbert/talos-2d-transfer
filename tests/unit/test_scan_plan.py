"""Scan-path indicator: the grid preview must be the grid the scan walks."""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from talos.cv.scan import GridScanner, grid_shape
from talos.models import ScanParams
from talos.ui.widgets.live_view import LiveViewWidget, ScanPlanOverlay


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _params(width=2000.0, height=1000.0, overlap=0.10) -> ScanParams:
    return ScanParams(x0_um=0.0, y0_um=0.0, width_um=width,
                      height_um=height, overlap=overlap)


def test_grid_shape_matches_the_plan_the_scanner_walks():
    """A preview that disagrees with the run is worse than no preview."""
    scanner = GridScanner(stage=None)
    params = _params()
    for fov in ((700.0, 390.0), (1000.0, 1000.0), (250.0, 120.0)):
        waypoints = scanner.plan(params, fov)
        cols, rows = grid_shape(params, fov)
        assert len(waypoints) == cols * rows
        assert waypoints[-1].index == cols * rows - 1


def test_grid_shape_is_never_empty():
    # a grid larger than the scan area still has one waypoint
    assert grid_shape(_params(width=100.0, height=100.0, overlap=0.0),
                      (700.0, 390.0)) == (1, 1)


def test_status_text_previews_and_reports_the_active_row():
    preview = ScanPlanOverlay(cols=4, rows=3, detail="4 × 3 grid")
    assert preview.status_text() == "12 waypoints · ready"

    first = ScanPlanOverlay(cols=4, rows=3, active_row=0, active_col=0)
    assert first.status_text() == "Row 1/3 · col 1/4 →"
    # serpentine: odd rows run backwards
    second = ScanPlanOverlay(cols=4, rows=3, active_row=1, active_col=2)
    assert second.status_text() == "Row 2/3 · col 3/4 ←"
    straight = ScanPlanOverlay(cols=4, rows=3, active_row=1, active_col=2,
                               serpentine=False)
    assert straight.status_text() == "Row 2/3 · col 3/4 →"


def test_live_view_draws_the_plan_only_when_enabled(qapp):
    view = LiveViewWidget()
    view.resize(640, 480)
    view.show_frame(np.full((240, 320, 3), 90, np.uint8))
    view._render_pending()
    view.set_scan_plan(ScanPlanOverlay(cols=3, rows=2, detail="3 × 2 grid"))
    view._overlay.repaint()
    assert view._scan_plan is not None

    view.set_scan_path_enabled(False)
    view.set_scan_plan(None)
    assert view._scan_plan is None
