"""ObjectivesDialog: settings save + one-way CalibrationStore sync —
objective registry mirror (labscope columns preserved) and the manual
px_um calibration entry (deduped on repeat saves)."""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from talos.calibration_store import CalibrationStore
from talos.ui.dialogs import objectives_page
from talos.ui.dialogs.objectives_dialog import ObjectivesDialog


def _redirect_db(tmp_path, monkeypatch):
    """The page body resolves the DB path itself — patch its namespace."""
    monkeypatch.setattr(objectives_page, "get_calibration_db_path",
                        lambda: tmp_path / "calibration.db")


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class FakeSettings:
    def __init__(self, rows):
        self.data = {"objectives": rows, "autofocus": {}}
        self.saved = 0

    def get(self, key, default=None):
        return self.data.get(key, default)

    def section(self, key):
        return self.data.get(key, {})

    def save(self):
        self.saved += 1


def _row(**overrides):
    row = {"name": "My 5x", "mag": 5, "na": 0.15, "dof_um": 28.0,
           "window_um": 1000.0, "coarse_step_um": 5.0, "fine_step_um": 1.0,
           "af_speed_multiplier": 1.0, "focus_manual_multiplier": 1.0,
           "stage_speed_multiplier": 1.0, "px_um": 0.0, "z_offset_um": 0.0,
           "backlash_um": 0.0}
    row.update(overrides)
    return row


def test_save_syncs_objectives_and_manual_px_into_store(qapp, tmp_path,
                                                        monkeypatch):
    _redirect_db(tmp_path, monkeypatch)
    settings = FakeSettings([_row(name="My 5x", px_um=0.65)])
    dialog = ObjectivesDialog(settings)
    dialog._on_save()
    assert settings.saved == 1
    store = CalibrationStore(tmp_path / "calibration.db")
    objs = store.list_objectives()
    assert len(objs) == 1
    assert objs[0]["nosepiece_position"] == 0
    assert objs[0]["name"] == "My 5x"
    assert objs[0]["magnification"] == 5.0
    # the manual px_um lands as a measured-style entry — the composition
    # prefers it over any Labscope value
    calib = store.get_active_calibration(objs[0]["id"])
    assert calib.source == "talos_measured"
    assert calib.um_per_px_x == pytest.approx(0.65)
    assert calib.um_per_px_y == pytest.approx(0.65)
    store.close()


def test_save_preserves_labscope_and_dedupes_px_entries(qapp, tmp_path,
                                                        monkeypatch):
    _redirect_db(tmp_path, monkeypatch)
    store = CalibrationStore(tmp_path / "calibration.db")
    store.upsert_objective("Lab 5x", 0, labscope_px_per_unit=1422.77,
                           labscope_units="px_per_mm", magnification=5.0,
                           source="labscope_import")
    store.close()
    rows = [_row(name="My 5x", px_um=0.65)]
    ObjectivesDialog(FakeSettings(rows))._on_save()
    ObjectivesDialog(FakeSettings(rows))._on_save()  # identical second save
    store = CalibrationStore(tmp_path / "calibration.db")
    obj = store.get_objective(store.list_objectives()[0]["id"])
    # the upsert passes the labscope columns through (a naive UPDATE
    # overwrite would erase the import)
    assert obj["labscope_px_per_unit"] == pytest.approx(1422.77)
    assert obj["labscope_px_per_unit_units"] == "px_per_mm"
    entries = store.list_entries(obj["id"])
    px_entries = [e for e in entries if e["kind"] == "px_um"]
    assert len(px_entries) == 1  # deduped — no timestamp churn
    assert px_entries[0]["provenance"] == '{"app": "settings_manual"}'
    store.close()


def test_save_without_px_writes_no_entry(qapp, tmp_path, monkeypatch):
    _redirect_db(tmp_path, monkeypatch)
    ObjectivesDialog(FakeSettings([_row()]))._on_save()
    store = CalibrationStore(tmp_path / "calibration.db")
    obj = store.list_objectives()[0]
    assert store.list_entries(obj["id"]) == []
    store.close()


def test_auto_calculate_fills_recommended_multipliers(qapp, tmp_path,
                                                      monkeypatch):
    _redirect_db(tmp_path, monkeypatch)
    rows = [
        _row(name="5x", mag=5, na=0.15),
        _row(name="10x", mag=10, na=0.30),
        _row(name="20x", mag=20, na=0.46),
    ]
    dialog = ObjectivesDialog(FakeSettings(rows))
    # the button fills ALL rows from the table's CURRENT mag/NA cells
    dialog.page._auto_calculate()
    table = dialog.page._advanced
    # 20x row: af/focus = (0.15/0.46)² ≈ 0.1063, stage = 5/20 = 0.25
    assert table.item(2, 0).text() == str(round((0.15 / 0.46) ** 2, 4))
    assert table.item(2, 1).text() == str(round((0.15 / 0.46) ** 2, 4))
    assert table.item(2, 2).text() == "0.25"
    # the lowest-power row normalizes to 1.0
    assert table.item(0, 0).text() == "1.0"
    assert table.item(0, 2).text() == "1.0"
    # 10x: af 0.25, stage 0.5
    assert table.item(1, 0).text() == "0.25"
    assert table.item(1, 2).text() == "0.5"


def test_offsets_checkbox_persists(qapp, tmp_path, monkeypatch):
    _redirect_db(tmp_path, monkeypatch)
    settings = FakeSettings([_row()])
    dialog = ObjectivesDialog(settings)
    dialog.page._offsets_check.setChecked(False)
    dialog._on_save()
    assert settings.data["objective_offsets_enabled"] is False


def test_z_offset_saved_to_settings_not_db(qapp, tmp_path, monkeypatch):
    _redirect_db(tmp_path, monkeypatch)
    settings = FakeSettings([_row(z_offset_um=12.5)])
    ObjectivesDialog(settings)._on_save()
    assert settings.data["objectives"][0]["z_offset_um"] == 12.5
    store = CalibrationStore(tmp_path / "calibration.db")
    obj = store.list_objectives()[0]
    kinds = [e["kind"] for e in store.list_entries(obj["id"])]
    assert "focus_offset" not in kinds  # focus_offset entries stay wizard-owned
    store.close()
