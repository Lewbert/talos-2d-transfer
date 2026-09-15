"""PreferencesDialog: page construction + _apply persistence."""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from talos.ui.dialogs.preferences import PreferencesDialog


class FakeSettings:
    def __init__(self):
        self.data = {
            "objectives": [
                {"name": "5x", "mag": 5, "na": 0.15, "dof_um": 28.0,
                 "window_um": 1000.0, "coarse_step_um": 5.0,
                 "fine_step_um": 1.0, "af_speed_multiplier": 1.0,
                 "focus_manual_multiplier": 1.0,
                 "stage_speed_multiplier": 1.0, "px_um": 0.0},
            ],
            "devices": {
                "camera": {"exposure_us": 40000.0, "gain": 20.0,
                           "white_balance": "Once"},
                "focus": {"max_speed": 2000, "min_speed": 50,
                          "um_per_step": 0.2, "backlash_um": 0.0},
                "zolix": {"port": "COM3"},
                "sigmakoki": {"port": "COM6"},
                "yudian": {"port": "COM5"},
            },
            "autofocus": {"stage_speed": 2000.0, "stage2_retries": 1,
                          "manual_bounds_um": 0.0},
            "ui": {"font_size": 12, "accent": "#00BCBC"},
            "debug": {"console_enabled": True, "verbose_logging": True},
            "scan": {"overlap": 0.1},
        }
        self.saved = 0

    def device(self, key):
        return self.data["devices"].setdefault(key, {})

    def section(self, key):
        return self.data.setdefault(key, {})

    def get(self, key, default=None):
        return self.data.get(key, default)

    def save(self):
        self.saved += 1


class StubManager:
    pass


class StubService:
    class _Sig:
        def connect(self, *a):
            pass

    sig_cal_finished = _Sig()

    def calibrate_backlash(self):
        pass


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def dialog(qapp, tmp_path, monkeypatch):
    # The objectives page syncs the calibration DB — redirect it.
    from talos.ui.dialogs import objectives_page
    monkeypatch.setattr(objectives_page, "get_calibration_db_path",
                        lambda: tmp_path / "cal.db")
    return PreferencesDialog(StubManager(), FakeSettings(), qapp,
                             StubService())


def _nav_leaf_labels(dialog) -> list[str]:
    """The selectable leaf items in nav order (the tree's parents are
    non-selectable group headers)."""
    labels: list[str] = []

    def walk(item):
        for i in range(item.childCount()):
            child = item.child(i)
            if child.childCount() == 0:
                labels.append(child.text(0))
            else:
                walk(child)

    walk(dialog._nav.invisibleRootItem())
    return labels


def _page(dialog, label: str):
    """Look a page up by its nav label (page ORDER is a UI decision)."""
    return dialog._pages[_nav_leaf_labels(dialog).index(label)]


def test_pages_exist(dialog):
    assert _nav_leaf_labels(dialog) == [
        "General", "Objectives & Calibration", "AutoFocus",
        "Input & Gamepad", "Camera", "Focus", "Zolix XYR",
        "SigmaKoki XYZ", "Temperature"]
    assert dialog._stack.count() == 9
    # the Hardware parent is a non-selectable group
    top = dialog._nav.topLevelItem(4)
    assert top.text(0) == "Hardware"
    assert not (top.flags() & Qt.ItemFlag.ItemIsSelectable)


def test_every_page_scrolls_and_applies(dialog):
    """A page must (a) live inside a scroll area — a long device page used
    to push the buttons off the dialog — and (b) implement _apply: the
    dialog silently SKIPS a page without one, discarding every edit."""
    from PySide6.QtWidgets import QScrollArea

    for index, page in enumerate(dialog._pages):
        assert isinstance(dialog._stack.widget(index), QScrollArea)
        assert dialog._stack.widget(index).widget() is page
        assert callable(getattr(page, "_apply", None)), \
            f"page {type(page).__name__} has no _apply — its edits would be lost"


def test_nav_selection_switches_pages(dialog):
    # pick the AutoFocus leaf (index 2) and the stack must follow
    focus_item = dialog._page_items[2]
    dialog._nav.setCurrentItem(focus_item)
    assert dialog._stack.currentIndex() == 2


def test_apply_persists_edits(dialog):
    camera_page = _page(dialog, "Camera")
    settings = camera_page._settings
    # Flip the live resolution via the form field and apply.
    for field in camera_page._fields:
        if field.path == "resolution":
            field.set_value(0)
    before = settings.saved
    dialog._on_apply()
    assert settings.device("camera")["resolution"] == 0
    assert settings.saved >= before + 1


def test_float_fields_keep_their_precision(dialog):
    """Regression (data loss in the field): QDoubleSpinBox defaults to 2
    decimals and the float fields forced 3, so a Preferences OK silently
    stored zolix um_per_pulse_r = 0.00125 as 0.00 and um_per_pulse_xy =
    0.625 as 0.63 — the axis scale is unusable at 0."""
    zolix = next(p for p in dialog._pages
                 if any(f.path == "um_per_pulse_r"
                        for f in getattr(p, "_fields", [])))
    fields = {f.path: f for f in zolix._fields}
    assert fields["um_per_pulse_r"].widget.decimals() >= 5
    assert fields["um_per_pulse_xy"].widget.decimals() >= 3
    # a round-trip must return the same numbers
    for path, value in (("um_per_pulse_r", 0.00125), ("um_per_pulse_xy", 0.625)):
        fields[path].set_value(value)
        assert float(fields[path].widget.value()) == pytest.approx(value)
    dialog._on_apply()
    zolix_cfg = zolix._settings.device("zolix")
    assert zolix_cfg["um_per_pulse_r"] == pytest.approx(0.00125)
    assert zolix_cfg["um_per_pulse_xy"] == pytest.approx(0.625)


def test_camera_page_keeps_device_level_fields_only(dialog):
    # exposure/gain/WB/auto-gain are workspace-dependent → right panels
    # only; the device page keeps the resolution and the sensor
    # orientation (the flip is a decode-time software rotation).
    camera_page = _page(dialog, "Camera")
    assert [f.path for f in camera_page._fields] == ["resolution", "flip"]


def test_device_page_group_order_and_manual_labelling(dialog):
    """Connection → Scale (affects every move) → Manual control → rest.

    The step↔µm conversion outranks the jog settings because it applies to
    the manual jogs, autofocus AND the grid scan; the purely-manual groups
    say so in their titles so they cannot be mistaken for scan settings.
    """
    from PySide6.QtWidgets import QGroupBox

    for label in ("Zolix XYR", "SigmaKoki XYZ", "Focus"):
        page = _page(dialog, label)
        titles = [box.title() for box in page.findChildren(QGroupBox)]
        assert titles[0] == "Connection", (label, titles)
        assert titles[1].startswith("Scale —"), (label, titles)
        manual = [t for t in titles if t.startswith("Manual control —")]
        assert manual, (label, titles)
        # every manual group sits after the scale group
        assert titles.index(manual[0]) > 1, (label, titles)

    input_page = _page(dialog, "Input & Gamepad")
    input_titles = [box.title() for box in input_page.findChildren(QGroupBox)]
    assert input_titles[0].startswith("Manual control —")


def test_objectives_merged_page_present(dialog):
    page = _page(dialog, "Objectives & Calibration")
    assert page._offsets_check.isChecked() is True
    assert page._basic.columnCount() == 5
    assert page._basic.horizontalHeaderItem(0).text() == "Name"
    assert page._basic.horizontalHeaderItem(3).text() == "µm/px"
    assert page._basic.horizontalHeaderItem(4).text() == "Z offset µm"
    assert page._advanced.columnCount() == 3
    assert page._advanced.horizontalHeaderItem(0).text() == "AF speed ×"
    assert page._calc_btn.text() == "Auto-calculate recommended"


def test_objectives_page_apply_persists_edits(dialog, qapp):
    # regression: the Preferences Apply path must save the merged table
    # (the page has no _apply() in the first revision — edits were
    # silently discarded and reset to 0.0 on the next open)
    from PySide6.QtWidgets import QTableWidgetItem

    page = _page(dialog, "Objectives & Calibration")
    settings = page._settings
    page._basic.setItem(0, 4, QTableWidgetItem("12.5"))
    dialog._on_apply()
    assert settings.get("objectives")[0]["z_offset_um"] == 12.5
    # reopening shows the persisted value
    d2 = PreferencesDialog(StubManager(), settings, qapp, StubService())
    assert _page(d2, "Objectives & Calibration")._basic.item(0, 4).text() == "12.5"


def test_temperature_presets_editor_roundtrip(dialog):
    from PySide6.QtWidgets import QTableWidgetItem

    page = _page(dialog, "Temperature")
    page._presets.setRowCount(1)
    page._presets.setItem(0, 0, QTableWidgetItem("Melt 200"))
    page._presets.setItem(0, 1, QTableWidgetItem("200.0"))
    dialog._on_apply()
    assert page._settings.device("yudian")["presets"] == [
        {"name": "Melt 200", "temp_c": 200.0}]


def test_general_page_has_debug_toggles(dialog):
    general = _page(dialog, "General")
    assert general._console.isChecked() is True
    assert general._verbose.isChecked() is True


def test_accent_preset_selection_persists(dialog):
    general = _page(dialog, "General")
    combo = general._accent_combo
    # the Wuling default is a preset and preselects; the old hexes are gone
    assert combo.currentData() == "#00BCBC"
    presets = [combo.itemData(i) for i in range(combo.count())]
    assert "#4a9eff" not in presets
    assert "#3fb950" not in presets
    assert "#d29922" not in presets
    idx = [i for i in range(combo.count())
           if combo.itemData(i) == "#e5484d"][0]
    combo.setCurrentIndex(idx)
    dialog._on_apply()
    assert general._settings.section("ui")["accent"] == "#e5484d"


def test_accent_combo_group_headers_non_selectable(dialog):
    combo = _page(dialog, "General")._accent_combo
    model = combo.model()
    # 2 group headers + 7 presets + Custom…
    assert model.rowCount() == 10
    headers = [i for i in range(model.rowCount())
               if not model.item(i).isEnabled()]
    assert headers == [0, 4]
    assert model.item(0).text() == "Endfield"
    assert model.item(4).text() == "Warhammer"
    assert model.item(0).font().bold() is True
    assert model.item(9).text() == "Custom…"
    assert model.item(9).isEnabled() is True
    # selecting a preset while a header is current reverts to it
    combo.setCurrentIndex(0)
    combo.setCurrentIndex(2)
    assert combo.currentIndex() == 2


def test_accent_custom_hex_applies_live(dialog, qapp):
    from talos.ui import theme

    general = _page(dialog, "General")
    general._accent_combo.setCurrentIndex(general._accent_combo.count() - 1)
    general._accent_edit.setText("#5f8a3c")
    general._apply_accent_from_edit()
    assert theme.ACCENT == "#5f8a3c"
    # restore the Wuling default for the other tests
    theme.set_accent(qapp, "#00BCBC")
    assert theme.ACCENT == "#00bcbc"  # QColor.name() lowercases


def test_accent_dark_for_preset_and_custom():
    from talos.ui import theme

    for _name, accent, dark in theme.ACCENT_PRESETS:
        assert theme.accent_dark_for(accent) == dark
        assert theme.accent_dark_for(accent.upper()) == dark  # case-insensitive
    custom = theme.accent_dark_for("#5f8a3c")
    assert custom != "#008686"  # custom hex → the darken factor, not a preset


def test_button_text_luminance_rule():
    from talos.ui import theme

    # bright yellow: dark text (lightness() alone would pick white)
    assert "color: #17181d;" in theme.build_qss(accent="#FFFA00")
    # blood red: white text
    assert "color: #17181d;" not in theme.build_qss(accent="#e5484d")
    # the Wuling default: dark text
    assert "color: #17181d;" in theme.build_qss(accent="#00BCBC")
