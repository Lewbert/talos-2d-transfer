"""CollapsibleGroup: the right-panel accordion sections."""

import pytest
from PySide6.QtWidgets import QApplication, QGroupBox, QLabel, QVBoxLayout

from talos.ui.widgets.collapsible import CollapsibleGroup


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _box():
    box = QGroupBox("Camera")
    layout = QVBoxLayout(box)
    layout.addWidget(QLabel("content"))
    return box


def _section(qapp, collapsed=False):
    box = _box()
    return CollapsibleGroup("Camera", box, collapsed=collapsed), box


def test_initial_state_expanded(qapp):
    section, box = _section(qapp)
    assert section.expanded is True
    assert box.isVisibleTo(section) is True
    # the inner title is cleared — the header IS the title
    assert box.title() == ""
    assert section._header.text().startswith("▾")


def test_click_collapses_and_expands(qapp):
    section, box = _section(qapp)
    section._header.click()
    assert section.expanded is False
    assert box.isVisibleTo(section) is False
    assert section._header.text().startswith("▸")
    section._header.click()
    assert section.expanded is True
    assert box.isVisibleTo(section) is True
    assert section._header.text().startswith("▾")


def test_collapsed_at_construction(qapp):
    section, box = _section(qapp, collapsed=True)
    assert section.expanded is False
    assert box.isVisibleTo(section) is False
    assert section._header.text().startswith("▸")


def test_header_never_takes_keyboard_focus(qapp):
    from PySide6.QtCore import Qt

    section, _box = _section(qapp)
    assert section._header.focusPolicy() == Qt.FocusPolicy.NoFocus


def test_set_expanded_programmatically(qapp):
    section, box = _section(qapp)
    section.set_expanded(False)
    assert section.expanded is False
    assert box.isVisibleTo(section) is False
    section.set_expanded(True)
    assert box.isVisibleTo(section) is True


# ---------------------------------------------------------------------------
# Persistence (ui.collapsed_sections[state_key])
# ---------------------------------------------------------------------------

class _StubSettings:
    def __init__(self, collapsed=None):
        self.data = {"ui": {"collapsed_sections": {"nav": list(collapsed or [])}}}
        self.saved = 0

    def section(self, key):
        return self.data.setdefault(key, {})

    def save(self):
        self.saved += 1


def test_collapse_state_restored_from_settings(qapp):
    settings = _StubSettings(collapsed=["Camera"])
    section = CollapsibleGroup("Camera", _box(), settings=settings,
                               state_key="nav")
    assert section.expanded is False


def test_toggle_persists_collapse_state(qapp):
    settings = _StubSettings()
    section = CollapsibleGroup("Camera", _box(), settings=settings,
                               state_key="nav")
    assert settings.saved == 0  # construction never writes
    section._header.click()
    assert settings.data["ui"]["collapsed_sections"]["nav"] == ["Camera"]
    assert settings.saved == 1
    section._header.click()
    assert settings.data["ui"]["collapsed_sections"]["nav"] == []
    assert settings.saved == 2


def test_persistence_keeps_other_sections(qapp):
    settings = _StubSettings(collapsed=["Temperature"])
    section = CollapsibleGroup("Camera", _box(), settings=settings,
                               state_key="nav")
    section._header.click()
    assert settings.data["ui"]["collapsed_sections"]["nav"] == \
        ["Temperature", "Camera"]
