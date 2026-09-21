"""Simulated camera: synthesizes a scene with a bright flake-like blob on
a textured background, optional defocus blur, and a moving target.

With ``wafer: True`` it images a WAFER instead: a large synthetic sample
that the stage carries, so the scene genuinely moves with the stage. That
is what makes a scan's stitching checkable — a static scene renders the
same picture at every waypoint, and a mosaic built from it looks correct
even when the tiles are placed backwards or mirrored. The wafer's
illumination ramps and its feature field are functions of WAFER
coordinates, so a correct mosaic is continuous and a wrong one is visibly
doubled.

Used for closed-loop autofocus/scan tests and UI development.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from talos.hal.base import Camera
from talos.hal.sim import bench

#: The wafer's features, in µm from the wafer origin: (x, y, radius, rgb).
#: The magenta beacon is unique — a test can find it in a mosaic and check
#: where it landed. The rest is a field dense enough that any tile the scan
#: visits contains several features.
_WAFER_BEACON = (0.0, 0.0, 150.0, (255, 0, 255))
_WAFER_PITCH_UM = 500.0
_WAFER_FIELD_UM = 3000.0

#: The sensor modes the real camera has, as the frame size they produce:
#: the same encoding ``capture.resolution`` and the smartcam backend use.
_RESOLUTION_MODES = {0: (3840, 2160), 1: (1920, 1080)}


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
        # The wafer mode: a sample that moves with the stage (see the module
        # docstring). Off unless asked for, so every closed-loop suite keeps
        # the scene it was calibrated against.
        self.wafer = bool(config.get("wafer", False))
        self._wafer_fixed_um_per_px = float(config.get("wafer_um_per_px", 0.0))
        self._wafer_fov_um = float(config.get("wafer_fov_um", 1400.0))
        self.wafer_um_per_px = self._wafer_fixed_um_per_px \
            or (self._wafer_fov_um / max(1, self.width))
        self._wafer_features = self._build_wafer_features()
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
        # Per-frame scratch, reused instead of reallocated (see _render_scene
        # and the profile note there). Keyed by shape because the 4K snapshot
        # path temporarily changes it.
        self._base_scenes: dict[tuple[int, int], np.ndarray] = {}
        self._noise_buf: np.ndarray | None = None

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

    def _base_scene(self, h: int, w: int) -> np.ndarray:
        """The STATIC part of the scene: the textured illumination plus the
        flake-like features phase correlation locks onto.

        Cached per frame size. Rebuilding it every frame was 41 of the 185 ms
        a 1080p frame cost in sim mode (a linspace/sin/repeat/astype chain
        whose result cannot change while the size does not); the 4K snapshot
        path changes the size temporarily, which is why the key is the size.
        """
        cached = self._base_scenes.get((h, w))
        if cached is None:
            x = np.linspace(0, 4 * np.pi, w)
            y = np.linspace(0, 3 * np.pi, h)
            bg = 90 + 25 * np.sin(x)[None, :] + 15 * np.sin(y)[:, None]
            bg = np.repeat(bg[:, :, None], 3, axis=2)
            cached = np.ascontiguousarray(bg.astype(np.uint8))
            for k, (fx, fy, fr) in enumerate([
                (w * 0.3, h * 0.4, 14), (w * 0.6, h * 0.7, 22),
                (w * 0.75, h * 0.2, 10),
            ]):
                color = (140 + 40 * k, 90, 60 + 30 * k)
                cv2.circle(cached, (int(fx), int(fy)), fr, color, -1)
            if len(self._base_scenes) > 1:
                self._base_scenes.clear()   # the size went back and forth
            self._base_scenes[(h, w)] = cached
        return cached

    @staticmethod
    def _build_wafer_features() -> list[tuple[float, float, float, tuple]]:
        """The wafer's features in µm: a beacon plus a colour field."""
        features = [_WAFER_BEACON]
        palette = ((70, 90, 210), (90, 200, 120), (210, 150, 60),
                   (150, 110, 200), (60, 170, 200))
        steps = int(_WAFER_FIELD_UM / _WAFER_PITCH_UM) + 1
        for ix in range(-steps, steps + 1):
            for iy in range(-steps, steps + 1):
                if (ix, iy) == (0, 0):
                    continue
                colour = palette[(ix * 7 + iy * 3) % len(palette)]
                features.append((ix * _WAFER_PITCH_UM, iy * _WAFER_PITCH_UM,
                                 70.0, colour))
        return features

    def _wafer_scene(self, h: int, w: int) -> np.ndarray:
        """The frame the wafer shows at the CURRENT stage position.

        Everything is a function of wafer coordinates: the illumination
        ramps and the feature field both pan with the stage, so two tiles
        that overlap agree in the overlap — which is what makes a wrong
        mosaic visible (and testable) instead of looking plausible.
        """
        x_um, y_um = bench.get_xy()
        spp = self.wafer_um_per_px
        left = x_um - (w * spp) / 2.0
        # The bench's mounting runs the vertical axis the other way — image
        # +Y is the wafer's −Y (measured 2026-09-17, see cv/orientation.py).
        # The sim mirrors the vertical axis so it models THAT convention:
        # a simulation that asserts a mounting the hardware does not have
        # would happily certify a mirror-image mosaic.
        top = y_um + (h * spp) / 2.0
        xs = left + np.arange(w) * spp
        ys = top - np.arange(h) * spp
        base = (128.0
                + 26.0 * np.sin(xs / 2200.0 * 2.0 * np.pi)[None, :]
                + 20.0 * np.cos(ys / 1700.0 * 2.0 * np.pi)[:, None])
        img = np.empty((h, w, 3), np.uint8)
        img[:, :, 0] = np.clip(base, 0, 255).astype(np.uint8)
        img[:, :, 1] = np.clip(base * 0.96, 0, 255).astype(np.uint8)
        img[:, :, 2] = np.clip(base * 1.04, 0, 255).astype(np.uint8)
        for fx, fy, radius_um, colour in self._wafer_features:
            cx = int(round((fx - left) / spp))
            cy = int(round((top - fy) / spp))
            r = max(1, int(round(radius_um / spp)))
            if -r <= cx < w + r and -r <= cy < h + r:
                cv2.circle(img, (cx, cy), r, colour, -1)
        return img

    def _render_scene(self) -> np.ndarray:
        h, w = self.height, self.width
        if self.wafer:
            img = self._wafer_scene(h, w)
            if self.blur_sigma > 0:
                k = max(1, int(self.blur_sigma) * 2 + 1)
                img = cv2.GaussianBlur(img, (k, k), self.blur_sigma)
            if self.noise > 0:
                img = self._add_noise(img)
            return img
        # The caller owns the returned array, so hand out a fresh copy.
        img = self._base_scene(h, w).copy()
        # Moving target: bright disc, sharpest when blur_sigma == 0.
        cv2.circle(img, (int(self.target_x), int(self.target_y)),
                   int(self.target_radius), (255, 255, 255), -1)
        cv2.circle(img, (int(self.target_x), int(self.target_y)),
                   max(2, int(self.target_radius * 0.3)), (30, 30, 30), -1)
        if self.blur_sigma > 0:
            k = max(1, int(self.blur_sigma) * 2 + 1)
            img = cv2.GaussianBlur(img, (k, k), self.blur_sigma)
        if self.noise > 0:
            img = self._add_noise(img)
        return img

    def _add_noise(self, img: np.ndarray) -> np.ndarray:
        """img + N(0, noise·255), clipped to uint8.

        Identical PIXELS to the obvious `np.clip(img.astype(np.float32) +
        rng.normal(...))`, but the float64 temporaries are reused instead of
        reallocated — this path was 141 ms of a 1080p frame (three 48 MB
        arrays per frame) and is now 86 ms. Two facts make it exact:
        ``scale * standard_normal`` IS ``normal(0, scale)`` (same stream,
        same Ziggurat draws), and uint8 → float64 is exact, so adding the
        uint8 image directly gives the same float64 sum as the float32 upcast.
        """
        if self._noise_buf is None or self._noise_buf.shape != img.shape:
            self._noise_buf = np.empty(img.shape, dtype=np.float64)
        buf = self._noise_buf
        self._rng.standard_normal(img.shape, out=buf)
        np.multiply(buf, self.noise * 255, out=buf)
        np.add(img, buf, out=buf, casting="unsafe")
        np.clip(buf, 0, 255, out=buf)
        return buf.astype(np.uint8)

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
        if name == "resolution":
            self.set_resolution(int(value))
            return
        if name not in self._props:
            raise KeyError(f"Unknown camera property: {name}")
        self._props[name] = value

    def set_resolution(self, mode: int) -> None:
        """Switch the sensor mode (0 = 4K, 1 = 1080p) and keep streaming.

        The real backend stops the stream, writes the parameter and resumes
        with a pipeline re-init; the sim only has to change its frame size.
        A scan whose capture resolution differs from the live view's does
        this once for the run and once again at the end.
        """
        size = _RESOLUTION_MODES.get(int(mode))
        if size is None:
            raise ValueError(f"unknown resolution mode: {mode!r}")
        self.width, self.height = size
        self._props["width"], self._props["height"] = size
        # The wafer moves under the same optics, so its µm per pixel halves
        # when the same field of view is sampled by twice the pixels.
        if not self._wafer_fixed_um_per_px:
            self.wafer_um_per_px = self._wafer_fov_um / max(1, self.width)

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
