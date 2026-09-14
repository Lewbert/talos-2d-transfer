"""CalibrationContext: canonical 4K-sensor µm/px + frame-width scaling."""

import pytest
from PySide6.QtWidgets import QApplication

from talos.ui.calibration_context import CalibrationContext


class FakeState:
    class _Sig:
        def connect(self, *a):
            pass

    sig_objective_changed = _Sig()
    objective = 0


class FakeStore:
    def list_objectives(self):
        return []

    def get_active_calibration(self, _oid):
        return None


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def ctx(qapp):
    # an empty store → the pixel-pitch fallback: 2.0 µm / 5x = 0.4
    return CalibrationContext(FakeState(), store=FakeStore())


def test_um_per_px_is_canonical_sensor_value(ctx):
    assert ctx.um_per_px() == pytest.approx(0.4)


def test_um_per_px_at_scales_to_frame_width(ctx):
    assert ctx.um_per_px_at(3840) == pytest.approx(0.4)   # 4K sensor
    assert ctx.um_per_px_at(1920) == pytest.approx(0.8)   # 1080p live/capture
    assert ctx.um_per_px_at(0) is None
    assert ctx.um_per_px_at(-5) is None


def test_a_damaged_store_falls_back_instead_of_raising(qapp, monkeypatch):
    """A corrupt calibration.db used to brick startup (sqlite3 raises from
    the constructor, before any UI exists)."""
    import talos.ui.calibration_context as cc

    def _boom(*_a, **_k):
        raise RuntimeError("file is not a database")

    monkeypatch.setattr(cc, "CalibrationStore", _boom)
    monkeypatch.setattr(cc, "get_calibration_db_path", lambda: "x.db")
    ctx = CalibrationContext(FakeState())
    assert ctx.um_per_px() == pytest.approx(0.4)   # pixel-pitch fallback


def test_a_failing_lookup_falls_back(qapp):
    """A DB that opens but errors later must degrade the same way."""

    class BrokenStore:
        def list_objectives(self):
            raise RuntimeError("db gone")

        def get_active_calibration(self, _oid):
            raise RuntimeError("db gone")

    ctx = CalibrationContext(FakeState(), store=BrokenStore())
    assert ctx.um_per_px() == pytest.approx(0.4)
