"""AfRegionController: the shared autofocus measurement region.

One source of truth for three consumers (right-panel AF settings, AF
detail window, live-view overlay) — and for autofocus itself, which reads
the same persisted settings key.
"""

import pytest
from PySide6.QtWidgets import QApplication

from talos.ui.af_region import (
    DEFAULT_ROI_NORM,
    AfRegionController,
    sanitize_roi,
)


class FakeSettings:
    def __init__(self, roi=None):
        self.data = {"autofocus": {"default_roi_norm": roi}}
        self.saved = 0

    def section(self, key):
        return self.data.setdefault(key, {})

    def save(self):
        self.saved += 1


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def ctl(qapp):
    settings = FakeSettings()
    return AfRegionController(settings), settings


def test_default_is_the_whole_frame(ctl):
    controller, settings = ctl
    assert controller.is_full_frame()
    assert controller.roi() is None
    assert controller.describe() == "whole frame"
    assert settings.saved == 0          # constructing persists nothing


def test_stored_region_is_loaded(qapp):
    settings = FakeSettings([0.1, 0.2, 0.3, 0.4])
    controller = AfRegionController(settings)
    assert controller.roi() == pytest.approx((0.1, 0.2, 0.3, 0.4))
    assert "10 %" in controller.describe()


def test_set_persists_and_announces(ctl):
    controller, settings = ctl
    seen = []
    controller.sig_changed.connect(seen.append)
    controller.set_roi((0.25, 0.25, 0.5, 0.5))
    assert settings.section("autofocus")["default_roi_norm"] == pytest.approx(
        [0.25, 0.25, 0.5, 0.5])
    assert settings.saved == 1
    assert seen == [pytest.approx((0.25, 0.25, 0.5, 0.5))]


def test_full_frame_clears_the_stored_value(ctl):
    controller, settings = ctl
    controller.reset_to_default()
    assert settings.section("autofocus")["default_roi_norm"] is not None
    controller.use_full_frame()
    assert settings.section("autofocus")["default_roi_norm"] is None
    assert controller.is_full_frame()


def test_reset_restores_the_centre_default(ctl):
    controller, _settings = ctl
    controller.use_full_frame()
    controller.reset_to_default()
    assert controller.roi() == pytest.approx(DEFAULT_ROI_NORM)


def test_move_region_edits_one_field(ctl):
    controller, _settings = ctl
    controller.reset_to_default()
    controller.move_region(w=0.3)
    x, y, w, h = controller.roi()
    assert w == pytest.approx(0.3)
    # the other three fields are untouched
    assert (x, y, h) == pytest.approx((DEFAULT_ROI_NORM[0], DEFAULT_ROI_NORM[1],
                                       DEFAULT_ROI_NORM[3]))


def test_move_region_from_full_frame_starts_at_the_default(ctl):
    """Editing a number while 'Full frame' is selected must mean something:
    it starts from the default region rather than 0-sized nonsense."""
    controller, _settings = ctl
    controller.move_region(y=0.4)
    x, y, w, h = controller.roi()
    assert y == pytest.approx(0.4)
    assert w == pytest.approx(DEFAULT_ROI_NORM[2])


def test_no_change_no_save_no_signal(ctl):
    controller, settings = ctl
    seen = []
    controller.sig_changed.connect(seen.append)
    controller.use_full_frame()          # already full frame
    assert settings.saved == 0 and seen == []


def test_sanitize_clamps_into_the_frame():
    assert sanitize_roi((0.9, 0.9, 0.5, 0.5)) == pytest.approx((0.9, 0.9, 0.1, 0.1))
    assert sanitize_roi((-1, -1, 2, 2)) == pytest.approx((0.0, 0.0, 1.0, 1.0))
    assert sanitize_roi((0.0, 0.0, 0.0, 0.0)) == pytest.approx(
        (0.0, 0.0, 0.02, 0.02))          # a zero ROI is unusable
    assert sanitize_roi("nonsense") is None
    assert sanitize_roi(None) is None


def test_writes_the_key_autofocus_reads(ctl):
    """The service resolves its region from autofocus.default_roi_norm —
    the very key this controller persists (the service side of the contract
    is pinned in test_autofocus_service)."""
    controller, settings = ctl
    controller.set_roi((0.2, 0.2, 0.6, 0.6))
    assert settings.section("autofocus")["default_roi_norm"] == pytest.approx(
        [0.2, 0.2, 0.6, 0.6])
    controller.use_full_frame()
    assert settings.section("autofocus")["default_roi_norm"] is None
