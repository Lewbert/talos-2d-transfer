"""Scan settings and where a scan goes: pure, and hard to break by hand.

A settings file is edited by people and by older versions of this app, so
the loader has to survive a missing key, a string where a number belongs
and a value from a schema that no longer exists.
"""

from __future__ import annotations

from pathlib import Path

from talos.scan_settings import (DEFAULTS, SCAN_KEYS, default_scan_dir,
                                 load_scan_settings, save_scan_settings,
                                 scan_directory)
from talos.config import Settings


class _Settings:
    """The smallest thing that quacks like Settings for these functions."""

    def __init__(self, data=None):
        self._data = data or {}

    def section(self, key):
        return self._data.setdefault(key, {})

    def save(self):
        pass


def test_the_keys_are_exactly_the_defaults_table():
    """SCAN_KEYS and DEFAULTS are two halves of one list, and a key added
    to one and not the other is a value that silently never loads."""
    assert set(SCAN_KEYS) == set(DEFAULTS)


def test_missing_keys_come_back_at_their_defaults():
    loaded = load_scan_settings(_Settings())
    assert loaded == DEFAULTS
    assert loaded["origin"] == "centre"
    assert loaded["speed_pps"] == 500.0


def test_stored_values_are_coerced_to_the_right_type():
    loaded = load_scan_settings(_Settings({"scan": {
        "width_um": "1500", "x_dir": "-1", "serpentine": False,
        "return_to_start": False, "origin": "corner_fit"}}))
    assert loaded["width_um"] == 1500.0 and isinstance(loaded["width_um"], float)
    assert loaded["x_dir"] == -1 and isinstance(loaded["x_dir"], int)
    assert loaded["serpentine"] is False
    assert loaded["return_to_start"] is False
    assert loaded["origin"] == "corner_fit"


def test_a_hand_edited_value_falls_back_instead_of_raising():
    """A settings file must never be able to stop a scan from being
    planned — the same rule the identification config follows."""
    loaded = load_scan_settings(_Settings({"scan": {
        "width_um": "wide", "overlap": None, "settle_ms": "soon"}}))
    assert loaded["width_um"] == DEFAULTS["width_um"]
    assert loaded["overlap"] == DEFAULTS["overlap"]
    assert loaded["settle_ms"] == DEFAULTS["settle_ms"]


def test_saving_writes_only_the_scan_section():
    settings = _Settings({"scan": {}, "capture": {"dir": "keep me"}})
    save_scan_settings(settings, {"width_um": 1200.0, "origin": "corner_pitch"})
    assert settings.section("scan")["width_um"] == 1200.0
    assert settings.section("scan")["origin"] == "corner_pitch"
    assert settings.section("capture")["dir"] == "keep me"


def test_the_default_folder_sits_beside_the_snapshots():
    scans = default_scan_dir()
    assert scans.parent == Path.home() / "Pictures" / "TALOS"
    assert scans.parent.name != scans.name


def test_an_empty_setting_means_the_default_folder():
    assert scan_directory(_Settings({"scan": {"dir": ""}})) == default_scan_dir()
    assert scan_directory(_Settings()) == default_scan_dir()


def test_a_configured_folder_is_used_as_given():
    settings = _Settings({"scan": {"dir": "  D:/data/scans  "}})
    assert scan_directory(settings) == Path("D:/data/scans")


def test_the_settings_round_trip_through_a_real_settings_object(tmp_path):
    """The one that matters for a user upgrading: write with the real
    Settings class, read back, and nothing is lost or invented."""
    path = tmp_path / "settings.json"
    settings = Settings.load(path)
    loaded = load_scan_settings(settings)
    loaded["origin"] = "corner_pitch"
    loaded["speed_pps"] = 800.0
    loaded["dir"] = str(tmp_path / "scans")
    save_scan_settings(settings, loaded)
    settings.save()
    again = load_scan_settings(Settings.load(path))
    assert again["origin"] == "corner_pitch"
    assert again["speed_pps"] == 800.0
    assert scan_directory(Settings.load(path)) == tmp_path / "scans"
