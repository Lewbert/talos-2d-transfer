"""Invariants of the SmartCamApi parameter tables (docs/SMARTCAM_API.md)."""

from __future__ import annotations

import pytest

from talos.hal.devices.camera.smartcam_params import (
    PARAM_TYPE_NAMES,
    PROPERTIES,
    ZEN_DEFAULTS,
    ApiError,
    ApiEvent,
    ParamKey,
    clamp,
)


def test_param_key_ordinals_match_zen_log_lines():
    # ZEN's own log statements hardcode these ordinals — they pin the enum.
    assert int(ParamKey.WhiteBalance) == 52
    assert int(ParamKey.LedWaveLength) == 72
    # The critical ones for TALOS:
    assert int(ParamKey.ColorMode) == 26
    assert int(ParamKey.ExposureTime) == 32
    assert int(ParamKey.ExposureGain) == 31
    assert int(ParamKey.AutoExposure) == 3
    assert int(ParamKey.AutoWhiteBalance) == 5
    assert int(ParamKey.ColorTemperature) == 29
    assert int(ParamKey.Resolution) == 42
    assert int(ParamKey.TransferFormat) == 48


def test_error_and_event_ordinals():
    assert int(ApiError.ImageNotReady) == 20
    assert int(ApiError.NoError) == 0
    assert int(ApiError.InvalidParameterType) == 17
    assert int(ApiEvent.ImageAcquired) == 5


def test_property_ids_unique():
    ids = [spec["key"] for spec in PROPERTIES.values()]
    assert len(ids) == len(set(ids))


def test_property_ranges_match_xml():
    assert PROPERTIES["exposure_us"]["range"] == (61.0, 1_000_000.0)
    assert PROPERTIES["exposure_us"]["scale"] == 1000.0  # camera unit is ms
    assert PROPERTIES["gain"]["range"] == (1.0, 22.0)
    assert PROPERTIES["white_balance"]["range"] == (0, 2)
    assert PROPERTIES["color_temperature"]["range"] == (1500, 10000)
    assert PROPERTIES["gamma"]["range"] == (0.1, 3.0)
    assert PROPERTIES["resolution"]["range"] == (0, 1)
    assert PROPERTIES["transfer_format"]["range"] == (0, 1)


def test_clamp():
    assert clamp("exposure_us", 10.0) == 61.0
    assert clamp("exposure_us", 5_000_000.0) == 1_000_000.0
    assert clamp("exposure_us", 20_000.0) == 20_000.0
    assert clamp("gain", 100.0) == 22.0


def test_zen_defaults_are_properties():
    assert set(ZEN_DEFAULTS) <= set(PROPERTIES)
    assert ZEN_DEFAULTS["exposure_us"] == 20_000.0  # ZEN: 20 ms
    assert ZEN_DEFAULTS["gain"] == 4.0
    assert ZEN_DEFAULTS["color_mode"] == 1


def test_param_type_names_complete():
    assert set(PARAM_TYPE_NAMES) == set(range(9))


# ---------------------------------------------------------------------------
# Capture-latency cache (the 2 s TTL regression: a periodic DLL query
# stalled one frame's delivery every 2 s — the bench "EMI windows")
# ---------------------------------------------------------------------------

def test_capture_latency_cache_never_refreshes_on_a_timer():
    from talos.hal.devices.camera.smartcam_backend import SmartCamCamera

    cam = SmartCamCamera({"smartcam": {"capture_latency_ms": 15.0}})
    calls = []

    def fake_get_param(key, kind):
        calls.append(key)
        return 30.0  # 30 ms exposure (camera unit; scale ×1000) → midpoint 15 ms

    cam._get_param_raw = fake_get_param
    first = cam._capture_latency_s()
    assert first == pytest.approx(0.015 + 0.015)
    # called once per delivered frame — but the query must happen ONLY
    # once per connect/exposure-change, never on a timer
    for _ in range(10):
        assert cam._capture_latency_s() == first
    assert len(calls) == 1


def test_exposure_write_invalidates_latency_cache():
    from talos.hal.devices.camera.smartcam_backend import SmartCamCamera

    cam = SmartCamCamera({"smartcam": {"capture_latency_ms": 15.0}})
    exposure_values = [30.0, 20.0]  # camera unit (ms)
    calls = []

    def fake_get_param(key, kind):
        calls.append(key)
        return exposure_values[min(len(calls) - 1, len(exposure_values) - 1)]

    class FakeDll:
        def ApiCam_SetParameterValue(self, handle, key, arg):
            return 0

    cam._get_param_raw = fake_get_param
    cam._dll = FakeDll()
    first = cam._capture_latency_s()
    assert first == pytest.approx(0.030)
    # an exposure write through the low-level writer (set_property AND
    # _apply_defaults both land here) invalidates → one fresh query
    cam._set_param("exposure_us", 20000.0)  # +1 call: the write's readback
    second = cam._capture_latency_s()       # +1 call: the fresh query
    assert second == pytest.approx(0.025)
    assert len(calls) == 3
    assert cam._capture_latency_s() == second
    assert len(calls) == 3                  # cached — no further queries
