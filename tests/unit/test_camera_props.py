"""Canonical camera-property mapping across backends."""

from __future__ import annotations

import pytest

from talos.hal.camera_props import map_property, normalize_props


def test_normalize_mmcore_exposure_ms_to_us():
    props = normalize_props("mmcore", {"exposure_ms": 50.0, "gain": 2.0})
    assert props["exposure_us"] == 50000.0
    assert props["gain"] == 2.0


def test_normalize_directshow_exposure():
    props = normalize_props("directshow", {"exposure": 12.5})
    assert props["exposure_us"] == 12500.0


def test_normalize_harvesters_exposure_time_passthrough():
    props = normalize_props("harvesters", {"ExposureTime": 25000.0})
    assert props["exposure_us"] == 25000.0


def test_normalize_passes_unknown_keys():
    props = normalize_props("smartcam", {"backend": "smartcam",
                                         "resolution": [1920, 1080]})
    assert props["backend"] == "smartcam"
    assert props["resolution"] == [1920, 1080]


def test_normalize_fps_from_measured_fps():
    assert normalize_props("smartcam", {"measured_fps": 20.2})["fps"] == 20.2


def test_map_property_returns_native_name_and_converts_units():
    # canonical us -> native ms for mmcore/directshow
    assert map_property("mmcore", "exposure_us", 50000.0) == ("exposure_ms", 50.0)
    assert map_property("directshow", "exposure_us", 12500.0) == ("exposure", 12.5)
    # GenTL/smartcam speak canonical us already
    assert map_property("harvesters", "exposure_us", 20.0) == ("ExposureTime", 20.0)
    assert map_property("smartcam", "gain", 4.0) == ("gain", 4.0)


def test_map_property_unknown_name_raises():
    with pytest.raises(KeyError):
        map_property("smartcam", "warp_drive", 9)


def test_map_property_unsupported_backend_raises():
    with pytest.raises(KeyError):
        map_property("mcam", "color_mode", 1)


def test_map_color_temperature():
    assert map_property("smartcam", "color_temperature", 3200) == \
        ("color_temperature", 3200)
    assert map_property("sim", "color_temperature", 3200) == \
        ("color_temperature", 3200)
    with pytest.raises(KeyError):
        map_property("harvesters", "color_temperature", 3200)


def test_resolution_is_a_live_mode_for_the_backends_that_have_one():
    """A scan whose capture resolution differs from the live view's has to
    change the sensor mode WITHOUT a reconnect, so this one is routed like
    any other property — and a backend with no live mode refuses it, which
    is how the panel hears "no" instead of a switch that quietly did
    nothing."""
    assert map_property("smartcam", "resolution", 0) == ("resolution", 0)
    assert map_property("sim", "resolution", 1) == ("resolution", 1)
    with pytest.raises(KeyError):
        map_property("harvesters", "resolution", 0)
    with pytest.raises(KeyError):
        map_property("manual", "resolution", 0)
