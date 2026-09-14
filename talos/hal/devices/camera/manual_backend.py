"""Manual camera backend: cycles image files from a folder.

Used for development and for CV validation on recorded datasets.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from talos.hal.base import Camera, DeviceConnectionError

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class ManualCamera(Camera):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.folder = Path(config.get("manual_folder", "."))
        self.fps = float(config.get("fps", 5.0))
        self._files: list[Path] = []
        self._index = 0
        self._last_frame_t = 0.0

    @property
    def device_id(self) -> str:
        return f"camera@manual:{self.folder}"

    def connect(self) -> None:
        if not self.folder.is_dir():
            raise DeviceConnectionError(f"Manual camera folder not found: {self.folder}")
        self._files = sorted(p for p in self.folder.iterdir()
                             if p.suffix.lower() in _IMAGE_EXTS)
        if not self._files:
            raise DeviceConnectionError(f"No images in manual camera folder: {self.folder}")
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def fetch(self, timeout_ms: float = 2000.0) -> np.ndarray | None:
        interval = 1.0 / self.fps if self.fps > 0 else 0.0
        elapsed = time.monotonic() - self._last_frame_t
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_frame_t = time.monotonic()
        path = self._files[self._index % len(self._files)]
        self._index += 1
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_properties(self) -> dict[str, Any]:
        return {"folder": str(self.folder), "fps": self.fps, "n_files": len(self._files)}

    def set_property(self, name: str, value: Any) -> None:
        if name == "fps":
            self.fps = float(value)
        else:
            raise KeyError(f"Unknown camera property: {name}")

    def snapshot(self, path: Path, timeout_s: float = 15.0,
                 resolution: int | None = None,
                 burn: dict | None = None) -> Path:
        # resolution/burn are accepted and ignored: this backend streams
        # still images, so there is no mode to switch and the frame is
        # whatever size the folder holds (the ABC signature must still
        # match — the UI passes all four arguments).
        frame = self.fetch()
        if frame is None:
            raise DeviceConnectionError("Manual camera: no frame available")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return path
