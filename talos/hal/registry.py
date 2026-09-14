"""Device factory: settings key -> driver class (real or simulated)."""

from __future__ import annotations

from typing import Any

from talos.hal.base import AbstractDevice
from talos.hal.devices import (
    FocusStageDriver,
    SigmaKokiXYZStage,
    YudianTempController,
    ZolixXYRStage,
)
from talos.hal.sim import (
    SimFocusStage,
    SimSigmaKokiXYZStage,
    SimYudianTempController,
    SimZolixXYRStage,
)

_REAL = {
    "zolix": ZolixXYRStage,
    "sigmakoki": SigmaKokiXYZStage,
    "focus": FocusStageDriver,
    "yudian": YudianTempController,
}

_SIM = {
    "zolix": SimZolixXYRStage,
    "sigmakoki": SimSigmaKokiXYZStage,
    "focus": SimFocusStage,
    "yudian": SimYudianTempController,
}

DEVICE_KEYS = ("zolix", "sigmakoki", "focus", "yudian")
MOTION_KEYS = ("zolix", "sigmakoki", "focus")  # stop_all order: focus first


def make_device(key: str, config: dict[str, Any], sim: bool) -> AbstractDevice:
    table = _SIM if sim else _REAL
    cls = table.get(key)
    if cls is None:
        raise KeyError(f"Unknown device key: {key!r}")
    return cls(config)


def make_camera(config: dict[str, Any], sim: bool):
    """Camera factory: a single SimCamera in sim mode, otherwise the ordered
    fallback chain (smartcam → harvesters → mmcore → mcam → directshow)."""
    if sim:
        from talos.hal.sim.sim_camera import SimCamera

        return [SimCamera(config)]
    from talos.hal.devices.camera import camera_chain

    return camera_chain(config)
