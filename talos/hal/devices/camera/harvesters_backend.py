"""GenICam GenTL camera backend via harvesters (GenTL/U3V path — NOT usable for the Axiocam 208, which is not
a U3V camera; kept in the chain for other cameras).

The Zeiss producer DLL (zeiss_u3vgentlk.cti) is discovered via the
validated chain in talos.paths.find_cti — see that function for the
frozen-safe search order. The camera requires exclusive USB access:
LabscopeService.exe must be closed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from talos.hal.base import Camera, DeviceConnectionError
from talos.paths import find_cti
from talos.protocols.pixel_unpack import convert_to_rgb

logger = logging.getLogger(__name__)

# Suppress harvesters/genicam debug spam on stdout.
logging.getLogger("harvesters").setLevel(logging.WARNING)
logging.getLogger("genicam").setLevel(logging.WARNING)


class HarvestersCamera(Camera):
    APPLIES_FLIP = True

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.cti_path = config.get("gentl_cti_path") or ""
        self.expected_size = (int(config.get("height", 2160)),
                              int(config.get("width", 3840)))
        self._h = None
        self._ia = None
        self._node_map = None
        self._last_format = "BayerRG8"

    @property
    def device_id(self) -> str:
        # matches the _BACKENDS registry key — CameraProxy derives the
        # canonical-property backend name from this
        return "camera@harvesters"

    # ------------------------------------------------------------------

    def connect(self) -> None:
        from harvesters.core import Harvester

        if not self.cti_path or not Path(self.cti_path).is_file():
            self.cti_path = find_cti()
        if not self.cti_path:
            raise DeviceConnectionError(
                "No GenTL producer (.cti) found — is ZeissVisionSuite installed?")
        logger.info("Using GenTL producer: %s", self.cti_path)
        self._h = Harvester()
        self._h.add_file(self.cti_path)
        self._h.update()
        devices = self._h.device_info_list
        if not devices:
            self._h.reset()
            self._h = None
            raise DeviceConnectionError(
                "No GenTL devices enumerated — LabscopeService.exe likely holds "
                "the camera (close Labscope for exclusive access)")
        self._ia = self._h.create_image_acquirer(0)
        self._node_map = self._ia.remote_device.node_map
        logger.info("GenTL camera: %s", getattr(devices[0], "model", "unknown"))
        self._connected = True

    def disconnect(self) -> None:
        try:
            if self._ia is not None:
                self._ia.stop_acquisition()
                self._ia.destroy()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._h is not None:
                self._h.reset()
        except Exception:  # noqa: BLE001
            pass
        self._ia = None
        self._h = None
        self._node_map = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ia is not None

    def start(self) -> None:
        self._ia.start_acquisition()

    def stop(self) -> None:
        try:
            self._ia.stop_acquisition()
        except Exception:  # noqa: BLE001
            pass

    def fetch(self, timeout_ms: float = 2000.0) -> np.ndarray | None:
        try:
            with self._ia.fetch_buffer(timeout=timeout_ms) as buffer:
                component = buffer.payload.components[0]
                frame = convert_to_rgb(component.data, component.width,
                                       component.height, component.data_format)
                self._last_format = str(component.data_format).split(".")[-1]
                return self.apply_flip(frame)
        except Exception as exc:  # noqa: BLE001
            logger.debug("fetch failed: %s", exc)
            return None

    def get_properties(self) -> dict[str, Any]:
        props: dict[str, Any] = {
            "width": self.expected_size[1],
            "height": self.expected_size[0],
            "pixel_format": self._last_format,
            "exposure_us": None,
            "gain": None,
            "white_balance": None,
        }
        if self._node_map is None:
            return props
        try:
            props["exposure_us"] = float(self._node_map.ExposureTime.value)
        except Exception:  # noqa: BLE001
            pass
        try:
            props["gain"] = float(self._node_map.Gain.value)
        except Exception:  # noqa: BLE001
            pass
        try:
            props["white_balance"] = str(self._node_map.BalanceWhiteAuto.value)
        except Exception:  # noqa: BLE001
            pass
        return props

    def set_property(self, name: str, value: Any) -> None:
        if self.try_set_flip(name, value):
            return
        if self._node_map is None:
            raise DeviceConnectionError("Camera not connected")
        node_name = {
            "exposure_us": "ExposureTime",
            "gain": "Gain",
            "white_balance": "BalanceWhiteAuto",
        }.get(name)
        if node_name is None:
            raise KeyError(f"Unknown camera property: {name}")
        node = getattr(self._node_map, node_name)
        if name == "white_balance":
            node.value = str(value)  # e.g. "Continuous" | "Off"
        else:
            node.value = float(value)

    def snapshot(self, path: Path, timeout_s: float = 15.0,
                 resolution: int | None = None,
                 burn: dict | None = None) -> Path:
        # resolution/burn accepted and ignored — kept only so the
        # signature matches the ABC (the UI passes all four args,
        # and a mismatch raised TypeError on every snapshot).
        frame = self.fetch(timeout_ms=timeout_s * 1000)
        if frame is None:
            raise DeviceConnectionError("GenTL camera: no frame")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return path
