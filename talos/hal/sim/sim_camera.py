"""Simulated camera: synthesizes a scene with a bright flake-like blob on
a textured background, optional defocus blur, and a moving target.

Used for closed-loop autofocus/scan tests and UI development.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from talos.hal.base import Camera


class SimCamera(Camera):
    APPLIES_FLIP = True

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config or {})
        config = self.config
        self.width = int(config.get("width", 1280))
        self.height = int(config.get("height", 960))
        self.fps = float(config.get("fps", 10.0))
        self.blur_sigma = float(config.get("blur_sigma", 0.0))
        self.noise = float(config.get("noise", 0.02))
        self.target_x = float(config.get("target_x", self.width / 2))
        self.target_y = float(config.get("target_y", self.height / 2))
        self.target_radius = float(config.get("target_radius", 30.0))
        self._props: dict[str, Any] = {
            "exposure_us": int(config.get("exposure_us", 5000)),
            "gain": float(config.get("gain", 1.0)),
            "white_balance": "auto",
            "color_temperature": int(config.get("color_temperature", 5500)),
            "width": self.width,
            "height": self.height,
            "framerate": self.fps,
        }
        self._frame_no = 0
        self._last_frame_t = 0.0
        self._rng = np.random.default_rng(42)

    @property
    def device_id(self) -> str:
        return "camera@sim"

    def connect(self) -> None:
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

    # ------------------------------------------------------------------

    def _render(self) -> np.ndarray:
        """Single frame egress: live fetch AND the snapshot render both come
        through here, so the orientation flip lands before any scale-bar
        burn (mirroring the SmartCam backend's contract)."""
        return self.apply_flip(self._render_scene())

    def _render_scene(self) -> np.ndarray:
        h, w = self.height, self.width
        # Textured background (substrate-like) with smooth illumination.
        x = np.linspace(0, 4 * np.pi, w)
        y = np.linspace(0, 3 * np.pi, h)
        bg = 90 + 25 * np.sin(x)[None, :] + 15 * np.sin(y)[:, None]
        bg = np.repeat(bg[:, :, None], 3, axis=2)
        # A few static small features (flake-like) so phase correlation works.
        img = np.ascontiguousarray(bg.astype(np.uint8))
        for k, (fx, fy, fr) in enumerate([
            (w * 0.3, h * 0.4, 14), (w * 0.6, h * 0.7, 22), (w * 0.75, h * 0.2, 10),
        ]):
            color = (140 + 40 * k, 90, 60 + 30 * k)
            cv2.circle(img, (int(fx), int(fy)), fr, color, -1)
        # Moving target: bright disc, sharpest when blur_sigma == 0.
        cv2.circle(img, (int(self.target_x), int(self.target_y)),
                   int(self.target_radius), (255, 255, 255), -1)
        cv2.circle(img, (int(self.target_x), int(self.target_y)),
                   max(2, int(self.target_radius * 0.3)), (30, 30, 30), -1)
        if self.blur_sigma > 0:
            k = max(1, int(self.blur_sigma) * 2 + 1)
            img = cv2.GaussianBlur(img, (k, k), self.blur_sigma)
        if self.noise > 0:
            noise = self._rng.normal(0, self.noise * 255, img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img

    def fetch(self, timeout_ms: float = 2000.0) -> np.ndarray | None:
        # Pace to the configured fps.
        interval = 1.0 / self.fps if self.fps > 0 else 0.0
        elapsed = time.monotonic() - self._last_frame_t
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_frame_t = time.monotonic()
        self._frame_no += 1
        return self._render()

    def capture_time(self) -> float | None:
        return self._last_frame_t

    def get_properties(self) -> dict[str, Any]:
        return {**self._props, "flip": self.flip_enabled}

    def set_property(self, name: str, value: Any) -> None:
        if self.try_set_flip(name, value):
            return
        if name not in self._props:
            raise KeyError(f"Unknown camera property: {name}")
        self._props[name] = value

    def snapshot(self, path: Path, timeout_s: float = 15.0,
                 resolution: int | None = None,
                 burn: dict | None = None) -> Path:
        saved_w, saved_h = self.width, self.height
        if resolution is not None and int(resolution) == 0:
            self.width, self.height = 3840, 2160  # the 4K sim upscale
        try:
            frame = self._render()
            if burn and burn.get("um_per_px"):
                from talos.cv.scale_bar import draw_scale_bar_cv

                frame = draw_scale_bar_cv(frame, float(burn["um_per_px"]))
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), frame)
            return path
        finally:
            self.width, self.height = saved_w, saved_h
