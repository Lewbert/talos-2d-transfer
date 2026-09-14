"""Camera profiles: nav vs scan extraction, diffing, persistence routing."""

from talos.ui.camera_profiles import (
    nav_profile,
    profile_diff,
    scan_profile,
    update_profile_setting,
)


class FakeSettings:
    def __init__(self, camera):
        self.data = {"devices": {"camera": camera}}
        self.saved = 0

    def device(self, key):
        return self.data["devices"].setdefault(key, {})

    def save(self):
        self.saved += 1


def test_nav_profile_reads_live_keys():
    s = FakeSettings({"exposure_us": 40000, "gain": 20.0,
                      "white_balance": "Continuous", "auto_gain": True,
                      "auto_gain_target": 120.0,
                      "color_temperature": 3200})
    p = nav_profile(s)
    assert p.exposure_us == 40000
    assert p.gain == 20.0
    assert p.white_balance == "Continuous"
    assert p.color_temperature == 3200.0
    assert p.auto_gain is True
    assert p.allow_auto is True


def test_nav_profile_normalizes_non_continuous_to_off():
    # "Once" is a transient UI action now — a stored "Once" must never
    # re-enter the profile (it would re-run a WB pass on every apply)
    s = FakeSettings({"white_balance": "Once"})
    assert nav_profile(s).white_balance == "Off"
    s = FakeSettings({"white_balance": "Blinking"})
    assert nav_profile(s).white_balance == "Off"
    s = FakeSettings({})
    assert nav_profile(s).white_balance == "Off"


def test_nav_profile_defaults_color_temperature():
    p = nav_profile(FakeSettings({}))
    assert p.color_temperature == 5500.0


def test_scan_profile_is_manual_only():
    s = FakeSettings({"gain": 20.0, "auto_gain": True,
                      "scan": {"exposure_us": 30000, "gain": 4.0,
                               "white_balance": "Off",
                               "color_temperature": 3200}})
    p = scan_profile(s)
    assert p.exposure_us == 30000
    assert p.gain == 4.0
    assert p.white_balance == "Off"
    assert p.color_temperature == 3200.0
    assert p.auto_gain is False  # no auto anything on scans
    assert p.allow_auto is False


def test_scan_profile_forces_wb_off():
    # the scan workspace locks WB off — detection needs stable color
    # (the pre-scan adjustment is the Once button, not a stored state)
    s = FakeSettings({"scan": {"white_balance": "Continuous"}})
    assert scan_profile(s).white_balance == "Off"


def test_scan_profile_defaults_when_absent():
    s = FakeSettings({})
    p = scan_profile(s)
    assert p.gain == 4.0
    assert p.white_balance == "Off"
    assert p.color_temperature == 5500.0


def test_profile_diff_skips_unchanged():
    p = scan_profile(FakeSettings({}))
    current = {"exposure_us": 40000, "gain": 4.0, "white_balance": "Off",
               "color_temperature": 5500, "color_mode": 1}
    assert profile_diff(current, p) == []
    current["gain"] = 9.0
    assert profile_diff(current, p) == [("gain", 4.0)]
    # a color-temperature drift is detected like any other float key
    current["gain"] = 4.0
    current["color_temperature"] = 3200
    assert profile_diff(current, p) == [("color_temperature", 5500.0)]


def test_profile_diff_ignores_missing_canonical_keys():
    p = nav_profile(FakeSettings({"exposure_us": 12345}))
    assert ("exposure_us", 12345.0) in profile_diff({}, p)
    # unknown camera state → all four keys pushed
    assert len(profile_diff({}, p)) == 4


def test_update_profile_setting_routes_to_the_right_section():
    s = FakeSettings({})
    update_profile_setting(s, "nav", "exposure_us", 123.0)
    assert s.device("camera")["exposure_us"] == 123.0
    update_profile_setting(s, "scan", "gain", 7.0)
    assert s.device("camera")["scan"]["gain"] == 7.0
    assert "exposure_us" not in s.device("camera")["scan"]
    assert s.saved == 2
