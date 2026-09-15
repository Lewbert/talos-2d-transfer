"""Preferences dialog (Edit → Preferences): General (UI font, debug
console/logging), Objectives & Calibration (the merged per-objective
table), AutoFocus (curated knobs + backlash calibration), and the
per-device Hardware pages (camera / focus / zolix XYR / sigmakoki XYZ /
temperature). Workspace-dependent camera settings (exposure/gain/WB)
live in the right panels only — not here.

Connection parameters are annotated "applies after reconnect", speed
parameters "applies after restart" (the ActionResolver caches speeds at
construction). OK/Apply persist via the shared Settings instance.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QFrame,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from talos import debug_console
from talos.ui import theme
from talos.ui.dialogs.objectives_page import ObjectivesPage

logger = logging.getLogger(__name__)


class _Field:
    """One settings value → widget binding (path: "devices.camera.gain").

    ``scale`` maps the DISPLAY value to the stored value
    (stored = displayed / scale) — e.g. exposure stored in µs but shown
    in ms uses scale 0.001. Default 1.0 = no conversion."""

    def __init__(self, page: "_FormPage", path: str, label: str,
                 widget, setter, getter, annotation: str | None = None,
                 scale: float = 1.0):
        self.path = path
        self.widget = widget
        self._setter = setter
        self._getter = getter
        self.annotation = annotation
        self._scale = scale
        page._fields.append(self)

    def value(self):
        value = self._getter()
        return value / self._scale if self._scale != 1.0 else value

    def set_value(self, value) -> None:
        self._setter(value)  # display units


def _decimals_for(*values: float) -> int:
    """Decimals needed to represent these numbers exactly (max 6).

    A QDoubleSpinBox defaults to 2 decimals and the float fields used to
    force 3: `um_per_pulse_r` (0.00125) was silently stored as 0.00 and
    `um_per_pulse_xy` (0.625) as 0.63 on ANY Apply — real corruption of
    the live settings file, in the field.
    """
    decimals = 0
    for value in values:
        try:
            text = f"{abs(float(value)):.10f}".rstrip("0")
        except (TypeError, ValueError):
            continue
        if "." in text:
            decimals = max(decimals, len(text.split(".")[1]))
    return max(0, min(6, decimals))


def list_serial_ports() -> list[tuple[str, str]]:
    """(port, description) pairs currently present, [] when unavailable.

    ``comports()`` can raise (or hang) with a buggy driver — the reference
    project cached the list for exactly that reason — so enumeration must
    never break the dialog: failures degrade to an empty list and the port
    field falls back to the configured value.
    """
    try:
        from serial.tools import list_ports

        return [(p.device, p.description or "") for p in list_ports.comports()]
    except Exception:  # noqa: BLE001
        logger.debug("serial port enumeration failed", exc_info=True)
        return []


# Standard ladder for the Modbus/ASCII devices on this bench.
BAUDRATES = (1200, 2400, 4800, 9600, 14400, 19200, 38400, 57600, 115200,
             230400, 250000, 460800, 500000, 921600)

#: Width of the numeric fields — a QFormLayout with AllNonFixedFieldsGrow
#: stretches a lone spinbox across the whole page, which reads as an empty
#: form (the port/baud combos and text rows still grow).
_NUM_FIELD_W = 170


def _port_label(device: str, description: str) -> str:
    """'COM3' + 'USB Serial Port (COM3)' → 'COM3 — USB Serial Port'.

    pyserial's description usually repeats the port name in parentheses;
    showing the raw pair reads as "COM3 — USB Serial Port (COM3)".
    """
    text = (description or "").strip()
    if text.endswith(f"({device})"):
        text = text[: -len(device) - 2].strip()
    return f"{device} — {text}" if text else device


class _FormPage(QWidget):
    """A form over a settings dict: fields declared via add_* and
    applied back on _apply(). ``cfg`` is the dict itself (settings
    sections are top-level only — device dicts come from
    settings.device(key)).

    Fields added before any ``add_group()`` land in an ungrouped form at
    the top; each ``add_group(title)`` opens a titled box that collects
    everything declared after it.
    """

    def __init__(self, settings, cfg: dict, annotation: str | None = None,
                 parent=None):
        super().__init__(parent)
        self._settings = settings
        self._cfg = cfg
        self._fields: list[_Field] = []
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(9, 9, 9, 9)
        self._layout.setSpacing(10)
        # Kept last so groups sit at the top of the page instead of being
        # stretched to fill it.
        self._layout.addStretch(1)
        self._form: QFormLayout | None = None
        if annotation:
            label = QLabel(annotation)
            label.setObjectName("hint")
            label.setWordWrap(True)
            self._layout.insertWidget(self._layout.count() - 1, label)

    # --- layout -----------------------------------------------------------

    def _new_form(self, target: QVBoxLayout) -> QFormLayout:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setVerticalSpacing(6)
        form.setHorizontalSpacing(12)
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        if target is self._layout:
            # before the trailing stretch that keeps groups top-aligned
            target.insertLayout(target.count() - 1, form)
        else:
            target.addLayout(form)
        self._form = form
        return form

    @property
    def _target(self) -> QFormLayout:
        """The form new fields go into (created on first use)."""
        if self._form is None:
            self._new_form(self._layout)
        return self._form

    def add_group(self, title: str) -> QGroupBox:
        """Open a titled group; subsequent add_* calls land inside it."""
        box = QGroupBox(title)
        inner = QVBoxLayout(box)
        inner.setContentsMargins(9, 6, 9, 9)
        self._layout.insertWidget(self._layout.count() - 1, box)
        self._new_form(inner)
        return box

    def add_hint(self, text: str) -> QLabel:
        """A wrapped dim note inside the CURRENT group (or the page)."""
        label = QLabel(text)
        label.setObjectName("hint")
        label.setWordWrap(True)
        if self._form is None:
            self._layout.insertWidget(self._layout.count() - 1, label)
        else:
            self._form.addRow(label)
        return label

    # --- field builders ---------------------------------------------------

    def _label(self, text: str, annotation: str | None) -> str:
        return f"{text} *" if annotation else text

    def add_float(self, key: str, label: str, lo: float, hi: float,
                  step: float = 1.0, scale: float = 1.0,
                  annotation: str | None = None) -> None:
        box = QDoubleSpinBox()
        box.setRange(lo, hi)
        box.setSingleStep(step)
        box.setMaximumWidth(_NUM_FIELD_W)   # a full-width spinbox reads as
        box.setAlignment(Qt.AlignmentFlag.AlignRight)  # an empty form
        value = float(self._cfg.get(key, (lo + hi) / 2))
        # the decimals must fit the configured VALUE as well as the step,
        # or the spinbox rounds the stored number on the way in
        decimals = _decimals_for(step, lo, hi, value)
        box.setDecimals(decimals)
        box.setValue(round(value * scale, decimals))
        box.setToolTip(f"{label} {annotation}" if annotation else label)
        self._target.addRow(self._label(label, annotation), box)
        _Field(self, key, label, box, box.setValue, box.value, annotation,
               scale)

    def add_int(self, key: str, label: str, lo: int, hi: int,
                annotation: str | None = None) -> None:
        box = QSpinBox()
        box.setRange(lo, hi)
        box.setValue(int(self._cfg.get(key, lo)))
        box.setMaximumWidth(_NUM_FIELD_W)
        box.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._target.addRow(self._label(label, annotation), box)
        _Field(self, key, label, box, box.setValue, box.value, annotation)

    def add_text(self, key: str, label: str,
                 annotation: str | None = None) -> None:
        edit = QLineEdit(str(self._cfg.get(key, "")))
        self._target.addRow(self._label(label, annotation), edit)
        _Field(self, key, label, edit, edit.setText, edit.text, annotation)

    def add_bool(self, key: str, label: str, default: bool = False,
                 annotation: str | None = None) -> None:
        check = QCheckBox(label)
        check.setChecked(bool(self._cfg.get(key, default)))
        if annotation:
            check.setToolTip(f"{label} {annotation}")
        self._target.addRow("", check)
        _Field(self, key, label, check, check.setChecked, check.isChecked,
               annotation)

    def add_combo(self, key: str, label: str, choices: list[str],
                  annotation: str | None = None) -> None:
        combo = QComboBox()
        combo.addItems(choices)
        current = str(self._cfg.get(key, choices[0]))
        if current in choices:
            combo.setCurrentText(current)
        self._target.addRow(self._label(label, annotation), combo)
        _Field(self, key, label, combo, combo.setCurrentText,
               combo.currentText, annotation)

    def add_port(self, key: str, label: str,
                 annotation: str | None = None) -> None:
        """COM port picker: the ports actually present, plus the
        configured one when it is not detected (a device that is simply
        unplugged must not lose its setting), plus a Refresh button.

        The value stored is the port NAME — the description is display
        only.
        """
        current = str(self._cfg.get(key, ""))
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        combo = QComboBox()
        combo.setMinimumWidth(160)
        refresh = QPushButton("Refresh")
        refresh.setObjectName("compact")
        refresh.setToolTip("Re-scan the serial ports")
        layout.addWidget(combo, stretch=1)
        layout.addWidget(refresh)

        def fill() -> None:
            keep = str(combo.currentData() or current)
            ports = list_serial_ports()
            combo.blockSignals(True)
            combo.clear()
            for device, description in ports:
                combo.addItem(_port_label(device, description), device)
            if keep and keep not in [p for p, _ in ports]:
                combo.insertItem(0, f"{keep} (not detected)", keep)
            if not keep:
                # Nothing configured: offer an explicit "unset" entry
                # instead of preselecting an enumerated port — an Apply
                # must never silently bind a device to a random COM port.
                combo.insertItem(0, "(not set)", "")
                combo.setCurrentIndex(0)
            else:
                index = combo.findData(keep)
                combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)

        fill()
        refresh.clicked.connect(fill)
        self._target.addRow(self._label(label, annotation), combo)
        _Field(self, key, label, combo,
               lambda value: combo.setCurrentIndex(
                   max(0, combo.findData(str(value)))),
               lambda: combo.currentData() or "", annotation)

    def add_baud(self, key: str, label: str,
                 annotation: str | None = None) -> None:
        """Baudrate picker over the standard ladder (the configured value
        is inserted when it is not on it)."""
        combo = QComboBox()
        values = [int(v) for v in BAUDRATES]
        current = int(self._cfg.get(key, BAUDRATES[0]) or BAUDRATES[0])
        if current not in values:
            values.append(current)
            values.sort()
        for value in values:
            combo.addItem(str(value), value)
        combo.setCurrentIndex(max(0, combo.findData(current)))
        self._target.addRow(self._label(label, annotation), combo)
        _Field(self, key, label, combo,
               lambda value: combo.setCurrentIndex(
                   max(0, combo.findData(int(value)))),
               lambda: int(combo.currentData()), annotation)

    def add_dir(self, key: str, label: str, title: str = "Choose folder") -> None:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit(str(self._cfg.get(key, "")))
        browse = QPushButton("…")
        browse.setObjectName("compact")
        browse.clicked.connect(
            lambda: self._browse_dir(edit, title))
        layout.addWidget(edit, stretch=1)
        layout.addWidget(browse)
        self._target.addRow(label, row)
        _Field(self, key, label, edit, edit.setText, edit.text)

    @staticmethod
    def _browse_dir(edit: QLineEdit, title: str) -> None:
        chosen = QFileDialog.getExistingDirectory(edit, title, edit.text())
        if chosen:
            edit.setText(chosen)

    def add_custom(self, widget: QWidget, label: str) -> None:
        self._target.addRow(label, widget)

    def add_custom_row(self, widget: QWidget) -> None:
        self._target.addRow(widget)

    # --- apply --------------------------------------------------------------

    def _apply(self) -> None:
        for field in self._fields:
            self._cfg[field.path] = field.value()
        self._settings.save()


class GeneralPage(_FormPage):
    def __init__(self, settings, qapp, parent=None):
        super().__init__(settings, settings.section("ui"), parent=parent)
        self._qapp = qapp
        self._debug_cfg = settings.section("debug")

        self.add_group("Appearance")
        size = QSpinBox()
        size.setRange(8, 14)
        size.setValue(int(self._cfg.get("font_size", 12)))
        size.valueChanged.connect(self._on_font_size)
        self._target.addRow("Font size (px)", size)
        self._font_size = size

        self.add_custom_row(QLabel("Theme: dark (fixed)"))

        # Accent: grouped preset combo (disabled headers) + custom hex
        # + color-dialog picker.
        accent_row = QWidget()
        row = QHBoxLayout(accent_row)
        row.setContentsMargins(0, 0, 0, 0)
        self._accent_combo = QComboBox()
        model = QStandardItemModel(self._accent_combo)
        self._accent_combo.setModel(model)
        by_name = {name: hex_ for name, hex_, _dark in theme.ACCENT_PRESETS}
        header_font = QFont()
        header_font.setBold(True)
        for group, names in theme.ACCENT_GROUPS:
            header = QStandardItem(group)
            header.setEnabled(False)
            header.setFont(header_font)
            model.appendRow(header)
            for name in names:
                item = QStandardItem(f"{name} ({by_name[name]})")
                item.setData(by_name[name], Qt.ItemDataRole.UserRole)
                model.appendRow(item)
        model.appendRow(QStandardItem("Custom…"))
        self._accent_edit = QLineEdit()
        self._accent_edit.setPlaceholderText("#rrggbb")
        self._accent_pick = QPushButton("Pick…")
        self._accent_pick.setObjectName("compact")
        self._accent_pick.clicked.connect(self._on_accent_pick)
        self._last_valid_accent_idx = 0
        self._accent_combo.currentIndexChanged.connect(self._on_accent_changed)
        self._accent_edit.editingFinished.connect(self._apply_accent_from_edit)
        row.addWidget(self._accent_combo, stretch=1)
        row.addWidget(self._accent_edit, stretch=1)
        row.addWidget(self._accent_pick)
        self._target.addRow("Accent", accent_row)
        current = str(self._cfg.get("accent", "#00BCBC"))
        matching = [i for i in range(self._accent_combo.count())
                    if self._accent_combo.itemData(i)
                    and self._accent_combo.itemData(i).lower()
                    == current.lower()]
        # signal-blocked init: a live set_accent + setFocus during
        # construction would steal focus every dialog open when the
        # saved accent is unmatched — sync the widget states manually.
        self._accent_combo.blockSignals(True)
        if matching:
            self._accent_combo.setCurrentIndex(matching[0])
            self._accent_edit.setText(current)
        else:
            self._accent_combo.setCurrentIndex(
                self._accent_combo.count() - 1)  # Custom…
            self._accent_edit.setText(current)
        self._accent_combo.blockSignals(False)
        custom = self._accent_combo.currentData() is None
        self._accent_edit.setEnabled(custom)
        self._accent_pick.setEnabled(custom)
        self._last_valid_accent_idx = self._accent_combo.currentIndex()

        self.add_group("Diagnostics")
        self._console = QCheckBox("Debug console (separate system window)")
        self._console.setChecked(bool(self._debug_cfg.get("console_enabled", True)))
        self._console.toggled.connect(self._on_console)
        self._target.addRow("", self._console)

        self._verbose = QCheckBox("Verbose logging (applies on restart)")
        self._verbose.setChecked(bool(self._debug_cfg.get("verbose_logging", True)))
        self._target.addRow("", self._verbose)

    # --- accent -------------------------------------------------------------

    def _on_accent_changed(self, index: int) -> None:
        item = self._accent_combo.model().item(index)
        if item is not None and not item.isEnabled():
            # keyboard navigation can land on a disabled group header
            self._accent_combo.setCurrentIndex(self._last_valid_accent_idx)
            return
        self._last_valid_accent_idx = index
        hex_ = self._accent_combo.itemData(index)
        custom = hex_ is None
        self._accent_edit.setEnabled(custom)
        self._accent_pick.setEnabled(custom)
        if custom:
            self._accent_edit.setFocus()
            return
        self._accent_edit.setText(str(hex_))
        theme.set_accent(self._qapp, str(hex_))  # the paired dark applies

    def _apply_accent_from_edit(self) -> None:
        text = self._accent_edit.text().strip()
        if self._accent_combo.currentData() is not None and text:
            return  # a preset is selected — the combo owns the value
        applied = theme.set_accent(self._qapp, text or theme.ACCENT)
        self._accent_edit.setText(applied)

    def _on_accent_pick(self) -> None:
        from PySide6.QtGui import QColor

        initial = QColor(self._accent_edit.text())
        chosen = QColorDialog.getColor(
            initial if initial.isValid() else QColor(theme.ACCENT), self,
            "Accent color")
        if chosen.isValid():
            self._accent_edit.setText(chosen.name())
            self._apply_accent_from_edit()

    def _resolved_accent(self) -> str:
        hex_ = self._accent_combo.currentData()
        if hex_ is not None:
            return str(hex_)
        applied = theme.set_accent(self._qapp,
                                   self._accent_edit.text().strip()
                                   or theme.ACCENT)
        return applied

    def _on_font_size(self, px: int) -> None:
        theme.set_font_size(self._qapp, px)

    def _on_console(self, on: bool) -> None:
        # applies immediately (Blender-style toggle)
        if on:
            debug_console.enable()
        else:
            debug_console.disable()

    def _apply(self) -> None:
        self._cfg["font_size"] = self._font_size.value()
        self._cfg["accent"] = self._resolved_accent()
        self._debug_cfg["console_enabled"] = self._console.isChecked()
        self._debug_cfg["verbose_logging"] = self._verbose.isChecked()
        self._settings.save()


class AutoFocusPage(QWidget):
    """Curated AF knobs + the backlash calibration. The window bounds
    and the rough-scan speed base live in the right-panel AF group
    (per-objective multipliers apply at AF start)."""

    _RECONNECT = "applies after restart"

    def __init__(self, settings, autofocus_service, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._service = autofocus_service
        self._form_page = _FormPage(settings, settings.section("autofocus"),
                                    parent=self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._form_page)

        fp = self._form_page
        fp.add_float("timeout_s", "Timeout (s)", 10, 600, 10)
        fp.add_float("quality_threshold", "Quality threshold", 0.0, 1.0, 0.05)
        fp.add_int("stage2_retries", "Stage-2 retries", 0, 5)
        fp.add_float("near_window_steps", "Near window (steps)", 0, 2000, 10)
        fp.add_float("sigma_steps", "σ steps (0 = auto)", 0, 1000, 5)

        cal_row = QWidget()
        cal_layout = QHBoxLayout(cal_row)
        cal_layout.setContentsMargins(0, 0, 0, 0)
        self._cal_btn = QPushButton("Calibrate backlash")
        self._cal_btn.clicked.connect(self._on_calibrate)
        cal_layout.addWidget(self._cal_btn)
        self._backlash_label = QLabel("—")
        self._backlash_label.setObjectName("dim")
        cal_layout.addWidget(self._backlash_label)
        cal_layout.addStretch(1)
        fp.add_custom_row(cal_row)
        self._refresh_backlash()

        if autofocus_service is not None:
            autofocus_service.sig_cal_finished.connect(
                lambda r: self._refresh_backlash())
            autofocus_service.sig_cal_finished.connect(
                lambda r: self._cal_btn.setEnabled(True))

    def _refresh_backlash(self) -> None:
        cfg = self._settings.device("focus")
        value = cfg.get("backlash_um", 0.0)
        measured = cfg.get("backlash_measured_at", "")
        self._backlash_label.setText(
            f"measured: {float(value):.2f} µm"
            + (f" ({str(measured)[:16]})" if measured else ""))

    def _on_calibrate(self) -> None:
        if self._service is None:
            return
        if self._service.busy:
            return  # a run is already in flight (re-enables on its finish)
        # calibrate_backlash only SUBMITS a job and returns, so the old
        # disable/enable pair re-enabled the button immediately: repeat
        # clicks queued several calibrations. It stays disabled until
        # sig_cal_finished arrives.
        self._cal_btn.setEnabled(False)
        self._service.calibrate_backlash()

    def _apply(self) -> None:
        self._form_page._apply()


class TemperaturePage(QWidget):
    """Connection + safety limits + the user-editable setpoint presets
    (the right-panel Temperature group's dropdown)."""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._form_page = _FormPage(settings, settings.device("yudian"),
                                    parent=self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._form_page)

        fp = self._form_page
        # Only its sizeHint height — the page shares the panel with the
        # presets table, and an expanding form would open a gap.
        fp.setSizePolicy(QSizePolicy.Policy.Preferred,
                         QSizePolicy.Policy.Maximum)
        fp.add_group("Connection")
        fp.add_port("port", "Port", annotation=_RECONNECT)
        fp.add_baud("baudrate", "Baudrate", annotation=_RECONNECT)
        fp.add_int("slave_address", "Modbus slave address", 1, 247,
                   _RECONNECT)
        fp.add_group("Safety limits")
        fp.add_float("safety_lo_c", "Safety low (°C)", -100, 400, 1)
        fp.add_float("safety_hi_c", "Safety high (°C)", -100, 400, 1)
        fp.add_hint("Setpoint writes are refused outside this range "
                    "(a hardware-protection gate, not a display limit).")

        presets = QGroupBox("Presets")
        preset_layout = QVBoxLayout(presets)
        self._presets = QTableWidget(0, 2)
        self._presets.setHorizontalHeaderLabels(["Name", "°C"])
        self._presets.setMaximumHeight(210)  # ~6 rows: the shipped preset
        # list fits without a half-cut row; beyond this the table's own
        # scrollbar takes over (the page itself scrolls in the dialog).
        # beyond this the table's own scrollbar takes over
        preset_layout.addWidget(self._presets)
        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_preset)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_preset)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch(1)
        preset_layout.addLayout(btn_row)
        layout.addWidget(presets)
        layout.addStretch(1)
        self._load_presets()

    def _load_presets(self) -> None:
        self._presets.setRowCount(0)
        for preset in self._settings.device("yudian").get("presets") or []:
            r = self._presets.rowCount()
            self._presets.insertRow(r)
            self._presets.setItem(
                r, 0, QTableWidgetItem(str(preset.get("name", ""))))
            self._presets.setItem(
                r, 1, QTableWidgetItem(str(float(preset.get("temp_c", 25.0)))))

    def _add_preset(self) -> None:
        r = self._presets.rowCount()
        self._presets.insertRow(r)
        self._presets.setItem(r, 0, QTableWidgetItem("New preset"))
        self._presets.setItem(r, 1, QTableWidgetItem("25.0"))

    def _remove_preset(self) -> None:
        row = self._presets.currentRow()
        if row >= 0:
            self._presets.removeRow(row)

    def _apply(self) -> None:
        self._form_page._apply()
        presets = []
        for r in range(self._presets.rowCount()):
            name_item = self._presets.item(r, 0)
            temp_item = self._presets.item(r, 1)
            name = name_item.text().strip() if name_item else ""
            if not name:
                continue
            try:
                temp_c = float(temp_item.text().strip()) if temp_item else 25.0
            except ValueError:
                temp_c = 25.0
            presets.append({"name": name, "temp_c": temp_c})
        self._settings.device("yudian")["presets"] = presets
        self._settings.save()


def _device_page(settings, section: str, fields: list, annotation: str) \
        -> _FormPage:
    """fields: (kind, key, label, *args) tuples; section names a DEVICE
    (settings.device). "group"/"hint" specs carry no key — they open a
    titled box / add a note for the fields that follow."""
    page = _FormPage(settings, settings.device(section), annotation)
    for spec in fields:
        kind = spec[0]
        if kind == "group":
            page.add_group(spec[1])
        elif kind == "hint":
            page.add_hint(spec[1])
        elif kind == "float":
            page.add_float(*spec[1:])
        elif kind == "int":
            page.add_int(*spec[1:])
        elif kind == "text":
            page.add_text(*spec[1:])
        elif kind == "bool":
            page.add_bool(*spec[1:])
        elif kind == "combo":
            page.add_combo(*spec[1:])
        elif kind == "dir":
            page.add_dir(*spec[1:])
        elif kind == "port":
            page.add_port(*spec[1:])
        elif kind == "baud":
            page.add_baud(*spec[1:])
        else:
            raise ValueError(f"Unknown preference field kind: {kind!r}")
    return page


# Hints. Every group that only affects MANUAL motion says so — the
# distinction between "what I drive by hand" and "what the automation
# does" is the one that matters when tuning a scan.
_MANUAL_HINT = ("Manual control only — never affects the position readout, "
                "autofocus or the grid scan. Flip X↔Y swaps the two axes: "
                "the Stage Control buttons X+/Y+ then drive the other "
                "physical axis.")
_SCALE_HINT = ("Used by EVERY move — manual jogs, autofocus and the grid "
               "scan — so a wrong value here shows up everywhere. The "
               "objective's µm/px (Objectives & Calibration) is what the "
               "scale bar and measurements use.")


class CameraPage(_FormPage):
    """Camera device page: connection-level knobs only. Exposure / gain /
    white balance are workspace-dependent and live in the right panels.

    The image flip is pushed to a RUNNING camera on Apply — no reconnect:
    the backend applies it while decoding, so every consumer (live view,
    autofocus, flake detection, snapshots) sees the same orientation.
    """

    def __init__(self, settings, manager=None, parent=None):
        super().__init__(settings, settings.device("camera"), parent=parent)
        self._manager = manager
        self.add_group("Live capture")
        self.add_int("resolution", "Live resolution (0=4K, 1=1080p)", 0, 1,
                     _NEXT_CONNECT)
        self.add_group("Image orientation")
        self.add_bool("flip", "Rotate 180° (undo the optics' inversion)",
                      True)
        self.add_hint(
            "The bench optics present the specimen rotated 180°; this "
            "restores the real-world orientation for the live view, "
            "autofocus, flake detection and the saved snapshots. It is "
            "INDEPENDENT of the stage axis inversion (Hardware → Focus / "
            "Zolix XYR / SigmaKoki XYZ → Axis direction) — flipping the "
            "camera never inverts a stage and never changes the scan "
            "direction.")

    def _apply(self) -> None:
        before = bool(self._cfg.get("flip", True))
        super()._apply()
        after = bool(self._cfg.get("flip", True))
        if after != before and self._manager is not None:
            self._manager.submit_camera("set_property", "flip", after)


# Annotations (*-suffixed labels + tooltips). Manual-control values are
# applied live (InputSystem.reload_settings on Apply); a connection change
# reconnects that device on Apply, and the live resolution still needs the
# next connect.
_RECONNECT = "reconnects on Apply"
_NEXT_CONNECT = "applies on the next connect"


def _build_pages(settings, qapp, manager, autofocus_service, parent):
    """The full page list as (parent, label, page) triples — the nav
    tree groups the device pages under a "Hardware" parent."""
    pages: list[tuple[str | None, str, QWidget]] = []

    pages.append((None, "General", GeneralPage(settings, qapp, parent)))

    pages.append((None, "Objectives & Calibration",
                  ObjectivesPage(settings, parent)))
    pages.append((None, "AutoFocus",
                  AutoFocusPage(settings, autofocus_service, parent)))
    pages.append((None, "Input & Gamepad", InputPage(settings, parent)))

    # Workspace-dependent camera settings (exposure/gain/WB/auto-gain)
    # live in the right panels only — the device page keeps the
    # connection-level knobs and the sensor orientation.
    pages.append(("Hardware", "Camera",
                  CameraPage(settings, manager, parent)))
    # NOTE on the speed ranges: a QSpinBox CLAMPS its range on
    # construction, so a range narrower than the stored value silently
    # rewrites the setting on any Apply (um_per_pulse_r was lost to this
    # before). Ranges here always include the shipped defaults.
    # Page order is CONNECTION → SCALE → MANUAL CONTROL → the rest: the
    # step↔µm conversion affects every move (manual, autofocus AND the grid
    # scan), so it outranks the purely-manual jog settings, and every
    # manual-only group says so in its title.
    focus_fields = [
        ("group", "Connection"),
        ("port", "port", "Port", _RECONNECT),
        ("baud", "baudrate", "Baudrate", _RECONNECT),
        ("group", "Scale — µm per step"),
        ("float", "um_per_step", "µm per step", 0.01, 10, 0.01),
        ("float", "backlash_um", "Backlash (µm, mechanism)", 0.0, 50, 0.1),
        ("hint", _SCALE_HINT),
        ("group", "Manual control — jog speeds"),
        ("int", "min_speed", "Min speed (steps/s)", 10, 1000),
        ("int", "max_speed", "Max speed (steps/s)", 50, 5000),
        ("group", "Manual control — direction"),
        ("bool", "invert", "Invert jog direction (triggers, keys, buttons)"),
        ("hint", _MANUAL_HINT),
        ("group", "Soft limits"),
        ("bool", "slim_on", "Soft limits on (firmware SLIM)"),
        ("int", "slim_min", "Soft limit min (steps)", -2000000, 2000000),
        ("int", "slim_max", "Soft limit max (steps)", -2000000, 2000000),
    ]
    pages.append(("Hardware", "Focus",
                  _device_page(settings, "focus", focus_fields, "")))
    zolix_fields = [
        ("group", "Connection"),
        ("port", "port", "Port", _RECONNECT),
        ("baud", "baudrate", "Baudrate", _RECONNECT),
        ("int", "slave_address", "Modbus slave address", 1, 247, _RECONNECT),
        ("group", "Scale — µm per pulse"),
        ("float", "um_per_pulse_xy", "µm per pulse XY", 0.01, 10, 0.01),
        ("float", "um_per_pulse_r", "µm per pulse R", 0.0001, 0.1, 0.0001),
        ("hint", _SCALE_HINT),
        ("group", "Manual control — jog speeds & steps"),
        ("int", "slow_speed_pps", "Slow speed (pps)", 10, 10000),
        ("int", "fast_speed_pps", "Fast speed (pps)", 10, 100000),
        ("int", "slow_speed_r", "Slow R speed (pps)", 10, 100000),
        ("int", "fast_speed_r", "Fast R speed (pps)", 10, 200000),
        ("int", "single_step", "Single step XY (pulses)", 1, 100000),
        ("int", "single_step_r", "Single step R (pulses)", 1, 100000),
        ("group", "Manual control — direction"),
        ("bool", "invert_x", "Invert X"),
        ("bool", "invert_y", "Invert Y"),
        ("bool", "invert_r", "Invert R"),
        ("bool", "flip_xy", "Flip X↔Y (swap the two axes)"),
        ("hint", _MANUAL_HINT),
        ("group", "Options"),
        ("bool", "rotation_enabled", "Rotation enabled"),
    ]
    pages.append(("Hardware", "Zolix XYR",
                  _device_page(settings, "zolix", zolix_fields, "")))
    sigm_fields = [
        ("group", "Connection"),
        ("port", "port", "Port", _RECONNECT),
        ("baud", "baudrate", "Baudrate", _RECONNECT),
        ("group", "Scale — µm per step"),
        ("float", "um_per_step_xy", "µm per step XY", 0.01, 10, 0.01),
        ("float", "um_per_step_z", "µm per step Z", 0.01, 10, 0.01),
        ("hint", _SCALE_HINT),
        ("group", "Manual control — jog speeds & steps"),
        ("int", "slow_speed_hz", "Slow speed XY (Hz)", 25, 2000),
        ("int", "fast_speed_hz", "Fast speed XY (Hz)", 25, 2000),
        ("int", "slow_speed_z", "Slow Z speed (Hz)", 25, 2000),
        ("int", "fast_speed_z", "Fast Z speed (Hz)", 25, 2000),
        ("int", "single_step", "Single step XY (steps)", 1, 100000),
        ("int", "single_step_z", "Single step Z (steps)", 1, 100000),
        ("group", "Manual control — direction"),
        ("bool", "invert_x", "Invert X"),
        ("bool", "invert_y", "Invert Y"),
        ("bool", "invert_z", "Invert Z"),
        ("bool", "flip_xy", "Flip X↔Y (swap the two axes)"),
        ("hint", _MANUAL_HINT),
    ]
    pages.append(("Hardware", "SigmaKoki XYZ",
                  _device_page(settings, "sigmakoki", sigm_fields, "")))
    pages.append(("Hardware", "Temperature",
                  TemperaturePage(settings, parent)))
    return pages


class InputPage(QWidget):
    """Manual-input QoL: the input loop, and the gamepad's response curve
    and per-stick inversion. Everything here applies to MANUAL motion
    only — autofocus and the grid scan are unaffected."""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        page = _FormPage(settings, settings.section("input"), parent=self)
        page.add_group("Manual control — keyboard & mouse")
        page.add_int("long_press_threshold_ms",
                     "Long-press threshold (ms)", 100, 1000)
        page.add_int("loop_rate_hz", "Input loop rate (Hz)", 20, 120)
        page.add_hint("A hold shorter than the threshold becomes a single "
                      "step, a longer one a continuous jog.")
        layout.addWidget(page)

        gamepad = _FormPage(settings,
                            settings.section("input").setdefault("gamepad", {}),
                            parent=self)
        gamepad.add_group("Manual control — gamepad")
        gamepad.add_float("deadzone", "Stick deadzone", 0.0, 0.9, 0.05)
        gamepad.add_float("gamma", "Stick response gamma", 1.0, 4.0, 0.1)
        gamepad.add_float("trigger_threshold", "Trigger threshold",
                          0.0, 1.0, 0.05)
        gamepad.add_group("Manual control — stick direction")
        gamepad.add_bool("invert_left_x", "Invert left stick X")
        gamepad.add_bool("invert_left_y", "Invert left stick Y")
        gamepad.add_bool("invert_right_x", "Invert right stick X")
        gamepad.add_bool("invert_right_y", "Invert right stick Y")
        gamepad.add_hint("The left stick jogs the transfer (SigmaKoki) "
                         "stage, the right stick the XYR stage. Axis "
                         "inversion for the keyboard, D-pad and on-screen "
                         "buttons lives on each device's page "
                         "(Hardware → … → Manual control — direction).")
        layout.addWidget(gamepad)
        layout.addStretch(1)

        # Each form page must take only its sizeHint height, or the two
        # pages split the panel and the trailing stretch inside each one
        # opens a gap between the groups.
        for form in (page, gamepad):
            form.setSizePolicy(QSizePolicy.Policy.Preferred,
                               QSizePolicy.Policy.Maximum)

        self._pages = [page, gamepad]

    def _apply(self) -> None:
        for page in self._pages:
            page._apply()


class PreferencesDialog(QDialog):
    # Emitted after Apply/OK persists the pages: consumers that cache
    # settings-derived state (the calibration context, the hardware strip)
    # must refresh then, not only when the dialog closes.
    sig_applied = Signal()

    def __init__(self, manager, settings, qapp, autofocus_service,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.resize(920, 620)
        root = QVBoxLayout(self)
        body = QHBoxLayout()

        # Tree nav: General / Hardware (Camera, Focus, Zolix XYR,
        # SigmaKoki XYZ, Temperature) / Objectives / AutoFocus /
        # Calibration.
        self._nav = QTreeWidget()
        self._nav.setHeaderHidden(True)
        self._nav.setObjectName("prefs_nav")
        self._nav.setFixedWidth(200)
        self._stack = QStackedWidget()
        self._pages: list[QWidget] = []
        self._page_items: list[QTreeWidgetItem] = []
        parents: dict[str, QTreeWidgetItem] = {}
        for parent, title, page in _build_pages(settings, qapp, manager,
                                                autofocus_service, self):
            self._pages.append(page)
            # Every page scrolls: a long device page (or a group that
            # gained fields) must not push the buttons off a 620 px dialog.
            self._stack.addWidget(self._scroll_area(page))
            if parent is not None:
                # NOTE: not setdefault — the default QTreeWidgetItem(...)
                # would be constructed eagerly every iteration and each
                # construction attaches an orphan top-level item.
                top = parents.get(parent)
                if top is None:
                    top = QTreeWidgetItem(self._nav, [parent])
                    top.setFlags(top.flags()
                                 & ~Qt.ItemFlag.ItemIsSelectable)
                    parents[parent] = top
                item = QTreeWidgetItem(top, [title])
            else:
                item = QTreeWidgetItem(self._nav, [title])
            self._page_items.append(item)
        self._nav.currentItemChanged.connect(self._on_nav_changed)
        self._nav.setCurrentItem(self._page_items[0])
        self._nav.expandAll()
        body.addWidget(self._nav)
        body.addWidget(self._stack, stretch=1)
        root.addLayout(body, stretch=1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).clicked.connect(
            self._on_ok)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
            self._on_apply)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    @staticmethod
    def _scroll_area(page: QWidget) -> QScrollArea:
        """Wrap a page so tall content scrolls instead of stretching the
        dialog (widgetResizable keeps the page as wide as the viewport —
        without it the forms collapse to their minimum width)."""
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        area.setWidget(page)
        return area

    def _on_nav_changed(self, current: QTreeWidgetItem,
                        _previous) -> None:
        if current is None:
            return
        try:
            index = self._page_items.index(current)
        except ValueError:
            return  # a parent node — no page
        self._stack.setCurrentIndex(index)

    def _on_apply(self) -> None:
        for page in self._pages:
            apply = getattr(page, "_apply", None)
            if apply is not None:
                apply()
        self.sig_applied.emit()

    def _on_ok(self) -> None:
        self._on_apply()
        self.accept()
