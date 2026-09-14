"""Camera backend registry and fallback chain.

SmartCamApi is the ONLY backend that finds the Axiocam 202/208 (libusb0
binding; hardware-verified color 1080p @ ~15-19 fps). The other backends
(harvesters/mmcore/mcam/directshow) are DEAD for the 208 — GenTL finds no
device, MM adapters never load, the MCam SDK is the 503/506 family, and
DirectShow cannot see libusb0 cameras — so they are unwired from the app:
they register LAZILY (only when their optional dependencies import) and
are absent from the auto chain, but the code stays for dev tools and
other cameras. A named backend that failed to register raises KeyError.
"""

from __future__ import annotations

import logging
from typing import Any

from talos.hal.base import Camera
from talos.hal.devices.camera.manual_backend import ManualCamera
from talos.hal.devices.camera.smartcam_backend import SmartCamCamera

logger = logging.getLogger(__name__)

_BACKENDS: dict[str, type[Camera]] = {
    "smartcam": SmartCamCamera,
    "manual": ManualCamera,
}


def _try_register(name: str, module: str, cls: str) -> None:
    try:
        mod = __import__(f"talos.hal.devices.camera.{module}", fromlist=[cls])
        _BACKENDS[name] = getattr(mod, cls)
    except ImportError as exc:
        logger.debug("camera backend %s unavailable: %s", name, exc)


for _name, _module, _cls in (
    ("harvesters", "harvesters_backend", "HarvestersCamera"),
    ("mmcore", "mmcore_backend", "MMCoreCamera"),
    ("mcam", "mcam_backend", "McamCamera"),
    ("directshow", "directshow_backend", "DirectShowCamera"),
):
    _try_register(_name, _module, _cls)

# SmartCamApi only — the dead backends above are deliberately unwired
# from the app (2.5 fps / B-W or simply no device for the 208).
_AUTO_CHAIN = ("smartcam",)


def make_camera(config: dict[str, Any]) -> Camera:
    name = config.get("backend", "auto")
    if name == "auto":
        name = _AUTO_CHAIN[0]
    cls = _BACKENDS.get(name)
    if cls is None:
        raise KeyError(f"Unknown camera backend: {name!r}")
    return cls(config)


def camera_chain(config: dict[str, Any]) -> list[Camera]:
    """Ordered candidate backends: the proxy connects the first that works.

    "auto" → smartcam, harvesters, mmcore, mcam, directshow. A named
    backend → only that one.
    """
    name = config.get("backend", "auto")
    if name == "auto":
        return [_BACKENDS[n](config) for n in _AUTO_CHAIN]
    return [make_camera(config)]
