"""DirectShow/OpenCV camera backend.

Note: the Zeiss Axiocam 208 is bound to libusb0 by Labscope and is
generally NOT visible to DirectShow; this backend is for auxiliary
webcams and as a last-resort fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from talos.hal.base import Camera, DeviceConnectionError


class DirectShowCamera(Camera):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.index = int(config.get("camera_index", 0))
        self._cap: cv2.VideoCapture | None = None

    @property
    def device_id(self) -> str:
        return f"camera@directshow:{self.index}"

    def connect(self) -> None:
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            raise DeviceConnectionError(
                f"DirectShow camera {self.index} could not be opened "
                "(the Axiocam is libusb0-bound and invisible to DirectShow)")
        self._cap = cap
        self._connected = True

    def disconnect(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                pass
            self._cap = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._cap is not None

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def fetch(self, timeout_ms: float = 2000.0) -> np.ndarray | None:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_properties(self) -> dict[str, Any]:
        cap = self._cap
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "exposure": float(cap.get(cv2.CAP_PROP_EXPOSURE)),
            "gain": float(cap.get(cv2.CAP_PROP_GAIN)),
        }

    def set_property(self, name: str, value: Any) -> None:
        mapping = {
            "exposure": cv2.CAP_PROP_EXPOSURE,
            "gain": cv2.CAP_PROP_GAIN,
            "brightness": cv2.CAP_PROP_BRIGHTNESS,
        }
        if name not in mapping:
            raise KeyError(f"Unknown camera property: {name}")
        self._cap.set(mapping[name], float(value))

    def snapshot(self, path: Path, timeout_s: float = 15.0,
                 resolution: int | None = None,
                 burn: dict | None = None) -> Path:
        # resolution/burn accepted and ignored — kept only so the
        # signature matches the ABC (the UI passes all four args,
        # and a mismatch raised TypeError on every snapshot).
        frame = self.fetch()
        if frame is None:
            raise DeviceConnectionError("DirectShow camera: no frame")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return path
