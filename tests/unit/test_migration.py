"""App-bootstrap Labscope materialization (talos.migration)."""

import json

import pytest

from talos.calibration_store import CalibrationStore
from talos.config import Settings
from talos.migration import materialize_labscope_calibration


def _seed_db(db_path):
    store = CalibrationStore(db_path)
    for pos, (name, value) in enumerate([
            ("Pos0 5x (id 38)", 1422.7672), ("Pos1 10x (id 1)", 709.4143)]):
        store.upsert_objective(name, pos, labscope_px_per_unit=value,
                               labscope_units="nm_per_px",
                               magnification=5.0 * (2.0 ** pos))
    store.close()


def test_materialize_fills_store_and_settings(tmp_path, monkeypatch):
    from talos import paths

    db = tmp_path / "calibration.db"
    monkeypatch.setattr(paths, "get_calibration_db_path", lambda: db)
    _seed_db(db)
    settings = Settings.load(tmp_path / "settings.json")

    counts = materialize_labscope_calibration(settings)
    assert counts["entries"] == 2
    assert counts["renamed"] == 2  # Pos0… → the settings-row names

    saved = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    rows = saved["objectives"]
    assert rows[0]["px_um"] == pytest.approx(1422.7672 / 4000.0, abs=1e-6)
    assert rows[1]["px_um"] == pytest.approx(709.4143 / 4000.0, abs=1e-6)
    # names adopted from the settings rows
    store = CalibrationStore(db)
    by_pos = {o["nosepiece_position"]: o for o in store.list_objectives()}
    assert by_pos[0]["name"] == "UMPlanFl 5x"
    assert by_pos[1]["name"] == "UMPlanFl 10x"
    # magnifications adopted too (the Zeiss ladder mislabels 50x/100x)
    assert by_pos[0]["magnification"] == 5.0
    assert by_pos[1]["magnification"] == 10.0
    store.close()


def test_materialize_second_run_is_a_noop(tmp_path, monkeypatch):
    from talos import paths

    db = tmp_path / "calibration.db"
    monkeypatch.setattr(paths, "get_calibration_db_path", lambda: db)
    _seed_db(db)
    settings = Settings.load(tmp_path / "settings.json")
    assert materialize_labscope_calibration(settings)["entries"] == 2

    saves = []
    settings.save = lambda: saves.append(1)
    counts = materialize_labscope_calibration(settings)
    assert counts["entries"] == 0
    assert saves == []  # nothing changed → no save


def test_materialize_never_raises(tmp_path, monkeypatch):
    import talos.calibration_store as store_mod
    from talos import paths

    monkeypatch.setattr(paths, "get_calibration_db_path",
                        lambda: tmp_path / "calibration.db")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("db broken")

    monkeypatch.setattr(store_mod, "CalibrationStore", _boom)
    settings = Settings.load(tmp_path / "settings.json")
    assert materialize_labscope_calibration(settings) == {}
