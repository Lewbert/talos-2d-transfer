"""Micro-Manager camera backend via pymmcore-plus.

Device adapters to try: "AxioCam" (Zeiss), then "Usb3CamHS". The adapter
runtime is discovered by pymmcore-plus itself (bundled in the frozen app —
verified by the M1 packaging spike: 265 adapters loaded frozen).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from talos.hal.base import Camera, DeviceConnectionError

logger = logging.getLogger(__name__)

# (adapter, device names to try — the Zeiss adapter exposes "Zeiss AxioCam").
_ADAPTERS = (("AxioCam", ("Zeiss AxioCam", "AxioCam")), ("Usb3CamHS", ("Camera",)))


class MMCoreCamera(Camera):
    APPLIES_FLIP = True

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.adapter_dir = config.get("mmcore_adapter_dir") or ""
        self._mmc = None
        self._adapter = ""

    @property
    def device_id(self) -> str:
        return f"camera@mmcore:{self._adapter or '?'}"

    def connect(self) -> None:
        from pymmcore_plus import CMMCorePlus

        self._mmc = CMMCorePlus.instance()
        paths = [self.adapter_dir] if self.adapter_dir and Path(self.adapter_dir).is_dir() else []
        paths += self._bundled_adapter_dirs()
        if paths:
            try:
                self._mmc.setDeviceAdapterSearchPaths(paths)
                logger.debug("Adapter search paths: %s", paths)
            except Exception as exc:  # noqa: BLE001
                logger.debug("setDeviceAdapterSearchPaths failed: %s", exc)
        loaded = False
        for adapter, device_names in _ADAPTERS:
            for device_name in device_names:
                try:
                    self._mmc.loadDevice("Camera", adapter, device_name)
                    self._adapter = adapter
                    self._device_name = device_name
                    loaded = True
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.info("Adapter %s/%s unavailable: %s", adapter, device_name, exc)
            if loaded:
                break
        if not loaded:
            self._mmc = None
            raise DeviceConnectionError(
                "No Micro-Manager camera adapter could load (tried "
                f"{', '.join(a for a, _ in _ADAPTERS)}) — is the camera free (Labscope closed)?")
        try:
            self._mmc.setCameraDevice("Camera")  # the loadDevice label
            self._mmc.initializeDevice("Camera")
        except Exception as exc:  # noqa: BLE001
            self._mmc = None
            raise DeviceConnectionError(f"Micro-Manager camera init failed: {exc}") from exc
        self._connected = True
        logger.info("Micro-Manager camera via adapter %s (%s)", self._adapter, self._device_name)

    @staticmethod
    def _bundled_adapter_dirs() -> list[str]:
        """Prefer the pymmcore-plus bundled MM runtime over stale user
        installs (e.g. an old 2.0.3 in %LOCALAPPDATA%)."""
        import glob
        import os

        import pymmcore_plus

        dirs: list[str] = []
        base = Path(pymmcore_plus.__file__).resolve().parent
        for candidate in sorted(base.glob("**/mmgr_dal_*.dll"), reverse=True):
            parent = str(candidate.parent)
            if parent not in dirs:
                dirs.append(parent)
        return dirs[:3]

    def disconnect(self) -> None:
        try:
            if self._mmc is not None:
                self._mmc.reset()
        except Exception:  # noqa: BLE001
            pass
        self._mmc = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._mmc is not None

    def start(self) -> None:
        try:
            self._mmc.startContinuousSequenceAcquisition(0)
        except Exception:  # noqa: BLE001
            pass

    def stop(self) -> None:
        try:
            self._mmc.stopSequenceAcquisition()
        except Exception:  # noqa: BLE001
            pass

    def fetch(self, timeout_ms: float = 2000.0) -> np.ndarray | None:
        import time

        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            try:
                self._mmc.snapImage()
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.05)
        else:
            return None
        img = self._mmc.getImage()
        if img is None:
            return None
        if img.ndim == 2:
            return self.apply_flip(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
        if img.shape[2] == 4:  # RGBA -> RGB
            return self.apply_flip(img[:, :, :3].copy())
        return self.apply_flip(img.copy())

    def get_properties(self) -> dict[str, Any]:
        props: dict[str, Any] = {"adapter": self._adapter,
                                 "exposure_ms": None, "gain": None}
        if self._mmc is None:
            return props
        try:
            props["exposure_ms"] = float(self._mmc.getExposure())
        except Exception:  # noqa: BLE001
            pass
        try:
            props["gain"] = float(self._mmc.getProperty("Camera", "Gain"))
        except Exception:  # noqa: BLE001
            pass
        return props

    def set_property(self, name: str, value: Any) -> None:
        if self.try_set_flip(name, value):
            return
        if self._mmc is None:
            raise DeviceConnectionError("Camera not connected")
        if name == "exposure_ms":
            self._mmc.setExposure(float(value))
        elif name == "gain":
            self._mmc.setProperty("Camera", "Gain", str(value))
        else:
            raise KeyError(f"Unknown camera property: {name}")

    def snapshot(self, path: Path, timeout_s: float = 15.0,
                 resolution: int | None = None,
                 burn: dict | None = None) -> Path:
        # resolution/burn accepted and ignored — kept only so the
        # signature matches the ABC (the UI passes all four args,
        # and a mismatch raised TypeError on every snapshot).
        frame = self.fetch(timeout_ms=timeout_s * 1000)
        if frame is None:
            raise DeviceConnectionError("Micro-Manager camera: no frame")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return path
