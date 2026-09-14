"""Camera registry / fallback chain order."""

from __future__ import annotations

import pytest

from talos.hal.devices.camera import _BACKENDS, camera_chain, make_camera


def test_auto_chain_is_smartcam_only():
    cameras = camera_chain({"backend": "auto"})
    names = [type(c).__name__ for c in cameras]
    assert names == ["SmartCamCamera"]


def test_dead_backends_unwired_from_the_chain():
    from talos.hal.devices.camera import _AUTO_CHAIN

    assert _AUTO_CHAIN == ("smartcam",)
    # the dead backends (2.5 fps / B-W, or no device for the 208) must
    # never re-enter the chain — the code stays for dev tools only
    assert set(_AUTO_CHAIN).isdisjoint(
        {"harvesters", "mmcore", "mcam", "directshow"})


def test_named_backend_single_candidate():
    cameras = camera_chain({"backend": "manual"})
    assert len(cameras) == 1
    assert type(cameras[0]).__name__ == "ManualCamera"


def test_unknown_backend_raises():
    with pytest.raises(KeyError):
        make_camera({"backend": "warpcam"})


def test_registry_keys_match_backend_ids():
    # some backends append a variant suffix (camera@mmcore:?); the proxy
    # strips it, so the prefix must match the registry key exactly
    for name, cls in _BACKENDS.items():
        assert cls({}).device_id.startswith(f"camera@{name}")


def test_every_backend_accepts_the_ui_snapshot_call():
    """Regression: the UI calls snapshot(path, timeout_s, resolution,
    burn) and the ABC declares all four, but five backends took only
    (path, timeout_s) — every snapshot on the live manual backend died
    with TypeError, surfaced as a generic command error."""
    import inspect
    from pathlib import Path

    for name, cls in _BACKENDS.items():
        sig = inspect.signature(cls.snapshot)
        sig.bind(cls.__new__(cls), Path("x.png"), 15.0, 0, {"um_per_px": 1.0})
