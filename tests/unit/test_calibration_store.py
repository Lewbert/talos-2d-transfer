"""CalibrationStore CRUD, Labscope import (real file), composition tests."""

import sys

import pytest

from talos.calibration_store import CalibrationStore
from talos.cv.calibration import estimate_jacobian, phase_correlation_shift
from talos.models import CalibrationEntry


@pytest.fixture()
def store(tmp_path):
    s = CalibrationStore(tmp_path / "calibration.db")
    yield s
    s.close()


def test_upsert_update_is_committed(store, tmp_path):
    obj_id = store.upsert_objective("A", 0, labscope_px_per_unit=1.0,
                                    labscope_units="unknown", magnification=5.0)
    store.upsert_objective("B", 0, labscope_px_per_unit=2.0,
                           labscope_units="nm_per_px", magnification=5.0)
    store.close()
    s2 = CalibrationStore(tmp_path / "calibration.db")
    obj = s2.get_objective(obj_id)
    assert obj["name"] == "B"
    assert obj["labscope_px_per_unit"] == pytest.approx(2.0)
    assert obj["labscope_px_per_unit_units"] == "nm_per_px"
    s2.close()


def test_store_roundtrip_and_latest(store):
    obj_id = store.upsert_objective("Test 20x", 2, magnification=20.0)
    assert store.get_objective(obj_id)["nosepiece_position"] == 2
    store.insert_entry(CalibrationEntry(
        objective_id=obj_id, kind="px_um", um_per_px_x=0.1, um_per_px_y=0.1,
        jacobian=[[0.0, 0.0], [0.0, 0.0]], residual_px=0.5, move_um=50.0,
        provenance={"app": "test"}))
    latest = store.get_latest(obj_id, "px_um")
    assert latest is not None
    assert latest.um_per_px_x == pytest.approx(0.1)
    assert latest.provenance == {"app": "test"}


def test_store_composition_talos_wins_over_labscope(store):
    obj_id = store.upsert_objective(
        "EC Epiplan 50x", 3, labscope_px_per_unit=141.889,
        labscope_units="unknown", magnification=50.0, source="labscope_import")
    # Labscope-only: units unknown → no µm/px.
    calib = store.get_active_calibration(obj_id)
    assert calib.source == "none"
    # Resolve units → fallback works (the bench correction: the table
    # reads 2x coarse at the live view → /4 in canonical 4K terms).
    store.upsert_objective("EC Epiplan 50x", 3, labscope_px_per_unit=141.889,
                           labscope_units="px_per_mm", magnification=50.0)
    calib = store.get_active_calibration(obj_id)
    assert calib.source == "labscope"
    assert calib.um_per_px_x == pytest.approx(1000 / 141.889 / 4.0,
                                              rel=1e-6)
    # TALOS measurement wins.
    store.insert_entry(CalibrationEntry(
        objective_id=obj_id, kind="px_um", um_per_px_x=0.02, um_per_px_y=0.02,
        provenance={"app": "wizard"}))
    calib = store.get_active_calibration(obj_id)
    assert calib.source == "talos_measured"
    assert calib.um_per_px_x == pytest.approx(0.02)


def test_store_composition_labscope_nm_per_px(store):
    obj_id = store.upsert_objective(
        "Primostar3 10x", 1, labscope_px_per_unit=709.414,
        labscope_units="nm_per_px", magnification=10.0,
        source="labscope_import")
    calib = store.get_active_calibration(obj_id)
    assert calib.source == "labscope"
    # bench correction: the table is 2x coarse at the live view (the
    # binned-pitch mixup) → /4 in canonical 4K-frame terms
    assert calib.um_per_px_x == pytest.approx(0.709414 / 4.0, rel=1e-6)
    assert calib.um_per_px_y == pytest.approx(0.709414 / 4.0, rel=1e-6)


def test_store_focus_offset_entry(store):
    obj_id = store.upsert_objective("Ref", 0)
    store.insert_entry(CalibrationEntry(
        objective_id=obj_id, kind="focus_offset", focus_offset_steps=123,
        ref_objective_id=obj_id, provenance={"app": "wizard"}))
    calib = store.get_active_calibration(obj_id)
    assert calib.focus_offset_steps == 123


def test_import_labscope_microscope_json(store):
    """The Labscope import contract, pinned on a synthetic payload.

    This used to read the machine's real microscope.json (a Zeiss path
    containing the instrument's serial number) and skip when absent —
    environment-dependent, and private data in a public repo. The
    px-per-unit values below are the ones this bench's instrument
    reports, kept as a plain table.
    """
    data = {
        "DefaultCalibratedValues": {"0": 1422.7672, "1": 711.3836,
                                    "2": 355.6918, "3": 142.2767,
                                    "4": 71.0431},
        "DefaultNosepiece": {"0": 38, "1": 39, "2": 40, "3": 41, "4": 221},
    }
    count = store.import_labscope(data)
    assert count == 5
    objectives = store.list_objectives()
    assert len(objectives) == 5
    by_pos = {o["nosepiece_position"]: o for o in objectives}
    assert by_pos[0]["labscope_px_per_unit"] == pytest.approx(1422.7672, rel=1e-5)
    assert by_pos[0]["magnification"] == pytest.approx(5.0)
    assert by_pos[4]["labscope_px_per_unit"] == pytest.approx(71.0431, rel=1e-5)
    assert by_pos[4]["magnification"] == pytest.approx(80.0)  # 5 * 2^4


# --- Labscope materialization (one-shot unwire) ------------------------------

def test_canonical_um_per_px():
    assert CalibrationStore.canonical_um_per_px(
        709.414, "nm_per_px") == pytest.approx(709.414 / 4000.0)
    assert CalibrationStore.canonical_um_per_px(
        141.889, "px_per_mm") == pytest.approx(1000 / 141.889 / 4.0)
    assert CalibrationStore.canonical_um_per_px(1.0, "units") is None
    assert CalibrationStore.canonical_um_per_px(0.0, "px_per_mm") is None
    assert CalibrationStore.canonical_um_per_px(None, "nm_per_px") is None


def test_materialize_labscope_creates_entries_once(store):
    obj_id = store.upsert_objective(
        "Pos1 10x (id 1)", 1, labscope_px_per_unit=709.414,
        labscope_units="nm_per_px", magnification=10.0)
    counts = store.materialize_labscope({1: "UMPlanFl 10x"}, {1: 10.0})
    assert counts == {"entries": 1, "renamed": 1, "skipped": 0}
    entry = store.get_latest(obj_id, "px_um")
    assert entry.um_per_px_x == pytest.approx(709.414 / 4000.0, rel=1e-6)
    assert entry.um_per_px_y == pytest.approx(709.414 / 4000.0, rel=1e-6)
    assert entry.provenance == {"app": "labscope_materialized"}
    assert store.get_objective(obj_id)["name"] == "UMPlanFl 10x"
    # second run: the entry exists → skipped, nothing duplicated
    counts2 = store.materialize_labscope()
    assert counts2 == {"entries": 0, "renamed": 0, "skipped": 1}
    assert len(store.list_entries(obj_id)) == 1


def test_materialize_adopts_magnifications(store):
    # the Zeiss import derived a faulty 40x/80x ladder — the settings-row
    # mags (50x/100x) must win
    obj_id = store.upsert_objective(
        "Pos3 40x (id 18)", 3, labscope_px_per_unit=141.889,
        labscope_units="nm_per_px", magnification=40.0)
    counts = store.materialize_labscope({}, {3: 50.0})
    assert counts["renamed"] == 1
    assert store.get_objective(obj_id)["magnification"] == 50.0
    assert store.get_objective(obj_id)["name"] == "Pos3 40x (id 18)"


def test_materialize_labscope_px_per_mm_branch(store):
    obj_id = store.upsert_objective(
        "EC Epiplan 50x", 3, labscope_px_per_unit=141.889,
        labscope_units="px_per_mm", magnification=50.0)
    store.materialize_labscope()
    entry = store.get_latest(obj_id, "px_um")
    assert entry.um_per_px_x == pytest.approx(1000 / 141.889 / 4.0, rel=1e-6)


def test_materialize_skips_existing_px_entry(store):
    obj_id = store.upsert_objective(
        "A", 0, labscope_px_per_unit=709.414, labscope_units="nm_per_px",
        magnification=10.0)
    store.insert_entry(CalibrationEntry(
        objective_id=obj_id, kind="px_um", um_per_px_x=0.02, um_per_px_y=0.02,
        provenance={"app": "wizard"}))
    counts = store.materialize_labscope()
    assert counts["entries"] == 0
    assert store.get_latest(obj_id, "px_um").um_per_px_x == pytest.approx(0.02)


def test_materialize_ignores_unknown_units(store):
    obj_id = store.upsert_objective(
        "A", 0, labscope_px_per_unit=709.414, labscope_units="unknown",
        magnification=10.0)
    counts = store.materialize_labscope()
    assert counts["entries"] == 0
    assert store.get_latest(obj_id, "px_um") is None


# --- jacobian fitting --------------------------------------------------------

def test_estimate_jacobian_pure_scale():
    # 0.2 px per µm along X, 0.1 along Y, no shear.
    moves = [(50.0, 0.0), (0.0, 50.0), (-50.0, 0.0), (0.0, -50.0)]
    shifts = [(10.0, 0.0), (0.0, 5.0), (-10.0, 0.0), (0.0, -5.0)]
    fit = estimate_jacobian(moves, shifts)
    assert fit.jacobian[0][0] == pytest.approx(0.2, rel=1e-6)
    assert fit.jacobian[1][1] == pytest.approx(0.1, rel=1e-6)
    assert fit.residual_px < 1e-6
    assert fit.um_per_px_x == pytest.approx(5.0, rel=1e-6)
    assert fit.um_per_px_y == pytest.approx(10.0, rel=1e-6)


def test_estimate_jacobian_requires_two_moves():
    with pytest.raises(ValueError):
        estimate_jacobian([(1.0, 0.0)], [(1.0, 0.0)])


def test_phase_correlation_shift_recovers_translation():
    import numpy as np

    rng = np.random.default_rng(0)
    scene = (rng.random((400, 600)) * 255).astype(np.uint8)
    dx, dy = 17, -9
    shifted = np.roll(np.roll(scene, dx, axis=1), dy, axis=0)
    a = np.dstack([scene] * 3)
    b = np.dstack([shifted] * 3)
    sx, sy = phase_correlation_shift(a, b)
    # Convention: positive shift = scene content moved right/down in the
    # after-image (b = shift(a)).
    assert sx == pytest.approx(dx, abs=1.5)
    assert sy == pytest.approx(dy, abs=1.5)
