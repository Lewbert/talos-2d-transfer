"""The compact exclusive toggle, and the map's pop-out window."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from talos.ui.widgets.map_window import MapWindow
from talos.ui.widgets.scan_map import ScanMapPlan, ScanMapWidget
from talos.ui.widgets.segmented import SegmentedToggle

OPTIONS = [("centre", "Centre", "The start point is the middle"),
           ("corner_fit", "Corner", "The start point is a corner"),
           ("corner_pitch", "Corner (pitch)", "Corners, at the set pitch")]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# --- SegmentedToggle -------------------------------------------------------

def test_it_starts_on_the_first_option(qapp):
    toggle = SegmentedToggle(OPTIONS)
    assert toggle.value() == "centre"
    assert toggle._buttons["centre"].isChecked()


def test_it_starts_on_the_value_it_was_given(qapp):
    assert SegmentedToggle(OPTIONS, "corner_fit").value() == "corner_fit"


def test_clicking_emits_the_value_not_an_index(qapp):
    """An index is a thing that quietly means something else after the
    list is reordered; a settings file holds values."""
    toggle = SegmentedToggle(OPTIONS)
    seen: list = []
    toggle.sig_changed.connect(seen.append)
    toggle._buttons["corner_pitch"].click()
    assert seen == ["corner_pitch"]
    assert toggle.value() == "corner_pitch"
    assert toggle._buttons["centre"].isChecked() is False


def test_an_unknown_value_selects_the_first_option(qapp):
    """A hand-edited settings file must not leave the row blank."""
    assert SegmentedToggle(OPTIONS, "nonsense").value() == "centre"


def test_options_can_be_replaced_keeping_the_value(qapp):
    toggle = SegmentedToggle(OPTIONS, "corner_fit")
    toggle.set_options([("a", "A"), ("corner_fit", "Still here")])
    assert toggle.value() == "corner_fit"
    assert tuple(toggle._buttons) == ("a", "corner_fit")


def test_replacing_options_drops_a_value_that_no_longer_exists(qapp):
    toggle = SegmentedToggle(OPTIONS, "corner_pitch")
    toggle.set_options([("a", "A"), ("b", "B")])
    assert toggle.value() == "a"


def test_a_plain_value_gets_a_label_from_itself(qapp):
    toggle = SegmentedToggle([1, -1])
    assert tuple(toggle._buttons) == (1, -1)
    assert toggle._buttons[1].text() == "1"


def test_disabling_reaches_every_button(qapp):
    toggle = SegmentedToggle(OPTIONS)
    toggle.setEnabled(False)
    assert all(not button.isEnabled() for button in toggle._buttons.values())


# --- MapWindow -------------------------------------------------------------

def _map() -> ScanMapWidget:
    widget = ScanMapWidget()
    widget.set_plan(ScanMapPlan(x0_um=0.0, y0_um=0.0, width_um=1000.0,
                                height_um=500.0, fov_x_um=100.0,
                                fov_y_um=100.0, waypoints=[(0.0, 0.0)]))
    return widget


def test_the_window_adopts_and_returns_the_one_map(qapp):
    """One widget, one set of tiles, one view transform — a second copy
    is two chances to disagree about what the operator is looking at."""
    source = _map()
    window = MapWindow()
    window.adopt(source)
    assert source.parent() is window
    assert window._map is source
    returned = window.release()
    assert returned is source
    assert source.parent() is None
    assert window._map is None
    assert window.release() is None          # idempotent


def test_closing_hides_and_hands_the_map_back(qapp):
    window = MapWindow()
    window.adopt(_map())
    dismissed: list = []
    window.sig_dismissed.connect(lambda: dismissed.append(True))
    window.show()
    window.close()
    assert not window.isVisible()
    assert dismissed == [True]


def test_escape_stops_everything_then_hides(qapp):
    """Esc means STOP ALL in every other window here. A map that quietly
    swallowed it would be the one place the operator cannot panic in."""
    class _Input:
        def __init__(self):
            self.escapes = 0

        def on_escape(self):
            self.escapes += 1

    source = _map()
    window = MapWindow(input_system=_Input())
    window.adopt(source)
    window.show()
    window.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress,
                                   Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
    assert window._input.escapes == 1
    assert not window.isVisible()


def test_the_fit_button_reaches_the_map(qapp):
    source = _map()
    window = MapWindow()
    window.adopt(source)
    source._zoom = 4.0
    window.fit_btn.click()
    assert source._zoom == 1.0
