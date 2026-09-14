"""SQLite calibration store: objectives + per-objective calibration entries.

%APPDATA%\\TALOS\\calibration.db, WAL mode, single writer (GUI thread).
Composition rule: a TALOS-measured entry wins over the Labscope import;
jacobian-identity fallback uses the Labscope value when units are known.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from talos.models import CalibrationEntry, ObjectiveCalibration

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS objectives (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  nosepiece_position INTEGER NOT NULL,
  labscope_px_per_unit REAL,
  labscope_px_per_unit_units TEXT,
  magnification REAL,
  source TEXT NOT NULL DEFAULT 'labscope_import',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS calibration_entries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  objective_id INTEGER NOT NULL REFERENCES objectives(id) ON DELETE CASCADE,
  kind TEXT NOT NULL CHECK (kind IN ('px_um','focus_offset')),
  um_per_px_x REAL,
  um_per_px_y REAL,
  jacobian TEXT,
  residual_px REAL,
  move_um REAL,
  focus_offset_steps INTEGER,
  ref_objective_id INTEGER,
  provenance TEXT NOT NULL,
  notes TEXT,
  measured_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entries_obj
  ON calibration_entries(objective_id, kind, measured_at);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class CalibrationStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version < _SCHEMA_VERSION:
            # v1 is the first version; future migrations append here.
            self._conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Objectives
    # ------------------------------------------------------------------

    def list_objectives(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM objectives ORDER BY nosepiece_position").fetchall()
        return [dict(r) for r in rows]

    def get_objective(self, objective_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM objectives WHERE id = ?", (objective_id,)).fetchone()
        return dict(row) if row else None

    def upsert_objective(self, name: str, nosepiece_position: int,
                         labscope_px_per_unit: float | None = None,
                         labscope_units: str | None = None,
                         magnification: float | None = None,
                         source: str = "talos_manual") -> int:
        row = self._conn.execute(
            "SELECT id FROM objectives WHERE nosepiece_position = ?",
            (int(nosepiece_position),)).fetchone()
        now = _now()
        if row:
            self._conn.execute(
                "UPDATE objectives SET name=?, labscope_px_per_unit=?, "
                "labscope_px_per_unit_units=?, magnification=?, source=?, "
                "updated_at=? WHERE id=?",
                (name, labscope_px_per_unit, labscope_units, magnification,
                 source, now, row["id"]))
            self._conn.commit()
            return int(row["id"])
        cursor = self._conn.execute(
            "INSERT INTO objectives (name, nosepiece_position, "
            "labscope_px_per_unit, labscope_px_per_unit_units, magnification, "
            "source, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (name, int(nosepiece_position), labscope_px_per_unit, labscope_units,
             magnification, source, now, now))
        self._conn.commit()
        return int(cursor.lastrowid)

    def delete_objective(self, objective_id: int) -> None:
        self._conn.execute("DELETE FROM objectives WHERE id = ?", (objective_id,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Labscope import
    # ------------------------------------------------------------------

    def import_labscope(self, microscope_json: dict,
                        objective_names: dict[int, str] | None = None) -> int:
        """Seed objectives from Labscope's microscope.json.

        microscope_json keys: DefaultCalibratedValues {pos: value},
        DefaultNosepiece {pos: objective_id}. Names resolve via the
        optional objective_names map (from Objective.json), else
        "Objective <id>".
        """
        calibrated = microscope_json.get("DefaultCalibratedValues", {})
        nosepiece = microscope_json.get("DefaultNosepiece", {})
        count = 0
        for pos_str, value in calibrated.items():
            pos = int(pos_str)
            obj_id = nosepiece.get(pos_str) or nosepiece.get(pos)
            name = objective_names.get(obj_id, f"Objective {obj_id}") \
                if objective_names else f"Objective {obj_id or '?'}"
            # The 2× ladder starts at 5x on position 0.
            magnification = 5.0 * (2.0 ** pos)
            self.upsert_objective(
                name=name, nosepiece_position=pos,
                labscope_px_per_unit=float(value),
                labscope_units="unknown",  # px/mm vs nm/px resolved by the wizard
                magnification=magnification, source="labscope_import")
            count += 1
        return count

    # ------------------------------------------------------------------
    # Materialization (one-shot Labscope unwire)
    # ------------------------------------------------------------------

    _LABSCOPE_UNIT_CONVERSIONS = {
        "nm_per_px": lambda v: float(v) / 4000.0,
        "px_per_mm": lambda v: 1000.0 / float(v) / 4.0,
    }

    @staticmethod
    def canonical_um_per_px(labscope_value: float, units: str) -> float | None:
        """Bench-verified canonical 4K-frame µm/px for a Labscope reading
        (the table reads 2× coarse at the live view — the binned-pitch
        mixup — so it divides by 4 in canonical terms: live = /2, 4K = /1).
        Unknown units or a zero reading → None."""
        fn = CalibrationStore._LABSCOPE_UNIT_CONVERSIONS.get(str(units))
        if fn is None:
            return None
        try:
            return fn(labscope_value)
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def materialize_labscope(self, objective_names: dict[int, str] | None = None,
                             objective_mags: dict[int, float] | None = None) \
            -> dict[str, int]:
        """One-shot idempotent migration: for each objective with a known
        Labscope value and NO px_um entry, insert one with provenance
        {"app": "labscope_materialized"}; adopt the settings-row names AND
        magnifications by nosepiece position (the Zeiss import derived
        faulty 40x/80x mags from the position ladder). Existing px_um
        entries (wizard/manual) always win. Returns {"entries": n,
        "renamed": m, "skipped": k}."""
        names = objective_names or {}
        mags = objective_mags or {}
        created = renamed = skipped = 0
        for obj in self.list_objectives():
            value = obj.get("labscope_px_per_unit")
            units = obj.get("labscope_px_per_unit_units")
            if value and units and self.get_latest(obj["id"], "px_um") is None:
                um = self.canonical_um_per_px(float(value), str(units))
                if um and um > 0:
                    self.insert_entry(CalibrationEntry(
                        objective_id=obj["id"], kind="px_um",
                        um_per_px_x=um, um_per_px_y=um,
                        provenance={"app": "labscope_materialized"},
                        notes=f"materialized from Labscope {units} {value}"))
                    created += 1
                else:
                    skipped += 1
            else:
                skipped += 1
            pos = obj["nosepiece_position"]
            new_name = (names.get(pos) or "").strip()
            new_mag = mags.get(pos)
            if new_mag and float(new_mag) > 0 \
                    and float(new_mag) != obj.get("magnification"):
                self._conn.execute(
                    "UPDATE objectives SET magnification=?, updated_at=? "
                    "WHERE id=?", (float(new_mag), _now(), obj["id"]))
                renamed += 1
            if new_name and new_name != obj.get("name"):
                self._conn.execute(
                    "UPDATE objectives SET name=?, updated_at=? WHERE id=?",
                    (new_name, _now(), obj["id"]))
                renamed += 1
        if renamed:
            self._conn.commit()
        return {"entries": created, "renamed": renamed, "skipped": skipped}

    # ------------------------------------------------------------------
    # Entries
    # ------------------------------------------------------------------

    def insert_entry(self, entry: CalibrationEntry) -> int:
        cursor = self._conn.execute(
            "INSERT INTO calibration_entries (objective_id, kind, um_per_px_x, "
            "um_per_px_y, jacobian, residual_px, move_um, focus_offset_steps, "
            "ref_objective_id, provenance, notes, measured_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (entry.objective_id, entry.kind, entry.um_per_px_x, entry.um_per_px_y,
             json.dumps(entry.jacobian) if entry.jacobian else None,
             entry.residual_px, entry.move_um, entry.focus_offset_steps,
             entry.ref_objective_id, json.dumps(entry.provenance), entry.notes,
             entry.measured_at))
        self._conn.commit()
        return int(cursor.lastrowid)

    def get_latest(self, objective_id: int, kind: str) -> CalibrationEntry | None:
        row = self._conn.execute(
            "SELECT * FROM calibration_entries WHERE objective_id=? AND kind=? "
            "ORDER BY measured_at DESC, id DESC LIMIT 1",
            (objective_id, kind)).fetchone()
        if row is None:
            return None
        return CalibrationEntry(
            objective_id=row["objective_id"], kind=row["kind"],
            um_per_px_x=row["um_per_px_x"], um_per_px_y=row["um_per_px_y"],
            jacobian=json.loads(row["jacobian"]) if row["jacobian"] else None,
            residual_px=row["residual_px"], move_um=row["move_um"],
            focus_offset_steps=row["focus_offset_steps"],
            ref_objective_id=row["ref_objective_id"],
            provenance=json.loads(row["provenance"] or "{}"),
            notes=row["notes"] or "", measured_at=row["measured_at"])

    def list_entries(self, objective_id: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM calibration_entries WHERE objective_id=? "
            "ORDER BY measured_at DESC", (objective_id,)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def get_active_calibration(self, objective_id: int) -> ObjectiveCalibration:
        """Compose the effective calibration: TALOS-measured wins over the
        Labscope value; jacobian-identity fallback otherwise."""
        objective = self.get_objective(objective_id) or {}
        px_entry = self.get_latest(objective_id, "px_um")
        focus_entry = self.get_latest(objective_id, "focus_offset")
        calib = ObjectiveCalibration(
            objective_id=objective_id,
            name=objective.get("name", ""),
            nosepiece_position=objective.get("nosepiece_position", -1),
            focus_offset_steps=(focus_entry.focus_offset_steps
                                if focus_entry else None))
        if px_entry is not None:
            calib.um_per_px_x = px_entry.um_per_px_x
            calib.um_per_px_y = px_entry.um_per_px_y
            if px_entry.jacobian:
                j = px_entry.jacobian
                calib.jacobian_px_per_um = ((j[0][0], j[0][1]), (j[1][0], j[1][1]))
            calib.source = "talos_measured"
            return calib
        labscope_value = objective.get("labscope_px_per_unit")
        units = objective.get("labscope_px_per_unit_units")
        if labscope_value and units == "px_per_mm":
            # same bench correction as the nm_per_px branch below
            um_per_px = 1000.0 / labscope_value / 4.0
            calib.um_per_px_x = calib.um_per_px_y = um_per_px
            calib.source = "labscope"
        elif labscope_value and units == "nm_per_px":
            # Labscope's DefaultCalibratedValues on the Axiocam 208c are
            # nm/pixel. BENCH-VERIFIED correction (2026-09-10): the table
            # reads 2x coarse at the 1080p live view — a known 50 um
            # reference object spans the "100 um" bar when the raw value
            # is applied (and "200 um" after the previous 2x live-view
            # scaling). The import was originally verified against the
            # 3.70 um pixel size reported by ZEN — that is the BINNED
            # pitch; the 208c's native pixels are ~1.85 um. The canonical
            # value is expressed per 4K-frame pixel (the 4K capture is a
            # 2x upsampled frame of the native 1920-wide sensor), so the
            # table divides by 4: live = /2, 4K = /1.
            um_per_px = labscope_value / 4000.0
            calib.um_per_px_x = calib.um_per_px_y = um_per_px
            calib.source = "labscope"
        else:
            calib.source = "none"
        return calib
