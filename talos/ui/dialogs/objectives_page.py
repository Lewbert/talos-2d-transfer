"""ObjectivesPage: the per-objective table — Basic (name, optics,
µm/px, Z offset) + Advanced (the speed multipliers, with an
auto-calculate button) — the body shared by the standalone
ObjectivesDialog and the Preferences "Objectives & Calibration" page.

Everything else autocalculates from the global max bases × these
multipliers, so dof/window/coarse/fine stay file-level legacy columns
(no UI). The Z offsets drive the focus compensation on objective
switches (talos.objective_offsets).

NOTE: backlash is NOT a per-objective field — it is a mechanism
property of the focus axis (devices.focus.backlash_um).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from talos.calibration_store import CalibrationStore
from talos.cv.af_math import recommended_multipliers
from talos.models import CalibrationEntry
from talos.paths import get_calibration_db_path

# Basic = the user-facing identity + calibration; Advanced = the speed
# levers. The multipliers scale the GLOBAL speed bases (AF window bounds,
# AF rough-scan speed, focus max speed, stage slow/fast speeds).
_BASIC_COLUMNS = [("name", "Name", str), ("mag", "Mag", float),
                  ("na", "NA", float), ("px_um", "µm/px", float),
                  ("z_offset_um", "Z offset µm", float)]
_ADVANCED_COLUMNS = [("af_speed_multiplier", "AF speed ×", float),
                     ("focus_manual_multiplier", "Focus speed ×", float),
                     ("stage_speed_multiplier", "Stage speed ×", float)]

_Z_OFFSET_TOOLTIP = (
    "Focus compensation on objective switch (µm). Positive offset = the "
    "focus position display increases by this many µm after switching. "
    "Motion is clamped by the focus soft limits (SLIM).")


def _float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class ObjectivesPage(QWidget):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._offsets_check = QCheckBox(
            "Compensate focus offsets on objective switch")
        self._offsets_check.setChecked(
            bool(settings.get("objective_offsets_enabled", True)))
        self._offsets_check.setToolTip(
            "When switching objectives, move the focus by the difference "
            "of the two objectives' Z offsets.")
        layout.addWidget(self._offsets_check)

        rows = settings.get("objectives") or []
        basic_box = QGroupBox("Basic")
        basic_layout = QVBoxLayout(basic_box)
        self._basic = self._make_table(rows, _BASIC_COLUMNS)
        basic_layout.addWidget(self._basic)

        adv_box = QGroupBox("Advanced")
        adv_layout = QVBoxLayout(adv_box)
        self._advanced = self._make_table(rows, _ADVANCED_COLUMNS)
        adv_layout.addWidget(self._advanced)
        calc_row = QHBoxLayout()
        self._calc_btn = QPushButton("Auto-calculate recommended")
        self._calc_btn.clicked.connect(self._auto_calculate)
        calc_row.addWidget(self._calc_btn)
        hint = QLabel("from mag & NA: AF/focus = (min NA ÷ NA)², "
                      "stage = min mag ÷ mag")
        hint.setObjectName("hint")
        calc_row.addWidget(hint)
        calc_row.addStretch(1)
        adv_layout.addLayout(calc_row)

        layout.addWidget(basic_box)
        layout.addWidget(adv_box)
        layout.addStretch(1)

    @staticmethod
    def _make_table(rows, columns) -> QTableWidget:
        table = QTableWidget(len(rows), len(columns))
        table.setHorizontalHeaderLabels([c[1] for c in columns])
        for r, row in enumerate(rows):
            for c, (key, _label, _kind) in enumerate(columns):
                value = row.get(key, "")
                if value is None:
                    value = ""
                table.setItem(r, c, QTableWidgetItem(str(value)))
        for c, (key, _label, _kind) in enumerate(columns):
            if key == "z_offset_um":
                table.horizontalHeaderItem(c).setToolTip(_Z_OFFSET_TOOLTIP)
        return table

    def _auto_calculate(self) -> None:
        """Fill the Advanced multipliers from the Basic table's current
        mag/NA cells (unsaved edits count): af/focus = (min NA ÷ NA)²,
        stage = min mag ÷ mag, referenced to the lowest-power row."""
        rows = []
        for r in range(self._basic.rowCount()):
            mag_item = self._basic.item(r, 1)
            na_item = self._basic.item(r, 2)
            rows.append((_float(mag_item.text().strip()) if mag_item else None,
                         _float(na_item.text().strip()) if na_item else None))
        valid = [(m, n) for m, n in rows if m and m > 0 and n and n > 0]
        na_min = min(n for _m, n in valid) if valid else 0.15
        mag_min = min(m for m, _n in valid) if valid else 5.0
        for r, (mag, na) in enumerate(rows):
            if not mag or not na:
                continue
            af, focus, stage = recommended_multipliers(na, mag, na_min, mag_min)
            self._advanced.item(r, 0).setText(str(round(af, 4)))
            self._advanced.item(r, 1).setText(str(round(focus, 4)))
            self._advanced.item(r, 2).setText(str(round(stage, 4)))

    def save(self) -> None:
        rows = self._settings.get("objectives") or []
        self._save_table(rows, self._basic, _BASIC_COLUMNS)
        self._save_table(rows, self._advanced, _ADVANCED_COLUMNS)
        self._settings.data["objectives"] = rows
        self._settings.data["objective_offsets_enabled"] = \
            self._offsets_check.isChecked()
        self._settings.save()
        self._sync_calibration_store(rows)

    def _apply(self) -> None:
        # the Preferences dialog calls _apply() on every page — without
        # this the table edits were silently discarded on Apply/OK
        self.save()

    @staticmethod
    def _save_table(rows, table, columns) -> None:
        for r in range(table.rowCount()):
            if r >= len(rows):
                rows.append({})
            for c, (key, _label, kind) in enumerate(columns):
                item = table.item(r, c)
                text = item.text().strip() if item else ""
                if not text:
                    continue
                if kind is str:
                    rows[r][key] = text
                else:
                    try:
                        rows[r][key] = kind(text)
                    except ValueError:
                        pass  # keep the old value on bad input

    def _sync_calibration_store(self, rows) -> None:
        """One-way settings → store: the settings rows are the global
        objective registry; the store holds the measured calibration
        history. Name/mag mirror into the store's objectives table —
        passing the existing labscope columns through (upsert_objective's
        UPDATE overwrites them with the passed values, which would
        silently erase a Labscope import) — and a manual px_um lands as a
        calibration entry so get_active_calibration prefers it over the
        Labscope value."""
        store = CalibrationStore(get_calibration_db_path())
        try:
            existing = {o["nosepiece_position"]: o
                        for o in store.list_objectives()}
            for pos, row in enumerate(rows):
                prev = existing.get(pos, {})
                obj_id = store.upsert_objective(
                    str(row.get("name") or f"Objective {pos + 1}"), pos,
                    labscope_px_per_unit=prev.get("labscope_px_per_unit"),
                    labscope_units=prev.get("labscope_px_per_unit_units"),
                    magnification=_float(row.get("mag")),
                    source="talos_manual")
                px = _float(row.get("px_um")) or 0.0
                if px <= 0:
                    continue
                latest = store.get_latest(obj_id, "px_um")
                if latest is not None \
                        and latest.um_per_px_x == px \
                        and latest.um_per_px_y == px:
                    # unchanged — no timestamp churn (the provenance may
                    # differ: labscope_materialized after the migration)
                    continue
                store.insert_entry(CalibrationEntry(
                    objective_id=obj_id, kind="px_um",
                    um_per_px_x=px, um_per_px_y=px,
                    provenance={"app": "settings_manual"},
                    notes="manual entry from the Objectives page"))
        finally:
            store.close()
