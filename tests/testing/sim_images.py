"""Synthetic images: flake-like scenes and defocus blur stacks for
closed-loop autofocus tests and CV development."""

from __future__ import annotations

import math

import cv2
import numpy as np


def synthetic_flake_image(shape=(960, 1280), n_flakes=8, seed=None,
                          substrate=(90, 110, 130), noise=0.02) -> np.ndarray:
    """A textured SiO2-like substrate with a few bright, sharp flakes."""
    rng = np.random.default_rng(seed)
    h, w = shape
    img = np.full((h, w, 3), substrate, dtype=np.float32)
    # Smooth illumination gradient.
    xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    img += (xx * 18 + yy * 10)[:, :, None]
    # Texture (substrate roughness).
    texture = rng.normal(0, 4, (h, w, 1))
    texture = cv2.GaussianBlur(texture, (5, 5), 1.0).reshape(h, w, 1)
    img += texture
    for _ in range(n_flakes):
        fx = int(rng.uniform(0.1 * w, 0.9 * w))
        fy = int(rng.uniform(0.1 * h, 0.9 * h))
        radius = int(rng.uniform(8, 28))
        color = tuple(int(c) for c in rng.choice([
            (180, 120, 150),   # purple-ish
            (150, 160, 120),   # green-ish
            (200, 180, 120),   # gold-ish
        ]))
        cv2.circle(img, (fx, fy), radius, color, -1)
        # A darker nucleus for texture.
        cv2.circle(img, (fx, fy), max(2, radius // 3),
                   tuple(int(c * 0.5) for c in color), -1)
    if noise:
        img += rng.normal(0, noise * 255, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def defocus_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-blur proxy for defocus.

    The kernel must stay odd and >= 3 taps even for tiny sigma — a 1×1
    kernel makes every sigma below ~1.5 render IDENTICALLY sharp (a dead
    zone that fakes a flat-topped focus plateau)."""
    if sigma <= 0:
        return img
    k = max(3, int(sigma * 2 + 3) | 1)  # odd, monotonic in sigma
    return cv2.GaussianBlur(img, (k, k), sigma)


def focus_stack(positions: np.ndarray, focus_pos: float, img_gen,
                k_per_step: float = 0.02, base_sigma: float = 0.4,
                noise: float = 0.02) -> list[np.ndarray]:
    """Blur a scene by |z - focus_pos| — sigma = base + k*|dz|.

    ``img_gen`` is a callable returning the sharp scene (e.g.
    ``functools.partial(synthetic_flake_image, seed=7)``).
    """
    frames = []
    sharp = img_gen()
    for pos in positions:
        sigma = base_sigma + k_per_step * abs(float(pos) - focus_pos)
        frame = defocus_blur(sharp, sigma)
        if noise:
            seed = int(pos) & 0x7FFFFFFF  # rng seeds must be non-negative
            frame = np.clip(frame.astype(np.float32)
                            + np.random.default_rng(seed).normal(0, noise * 255, frame.shape),
                            0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


def two_plane_source(shape=(480, 640), seed=7, z_top=0.0, z_bottom=150.0,
                     k_per_step=0.03, base_sigma=0.4, top_weight=0.4) -> callable:
    """Focus source with TWO sharp planes at different z (wafer surface at
    z_top, mount at z_bottom): two independent flake images blurred by
    their own distance, blended. The TOP plane is deliberately the WEAKER
    one — a naive global-max pick would land on the mount."""
    img_top = synthetic_flake_image(shape=shape, seed=seed, noise=0.0)
    img_bottom = synthetic_flake_image(shape=shape, seed=seed + 1, noise=0.0)

    def source(pos: float) -> np.ndarray:
        a = defocus_blur(img_top, base_sigma + k_per_step * abs(pos - z_top))
        b = defocus_blur(img_bottom, base_sigma + k_per_step * abs(pos - z_bottom))
        blended = (top_weight * a.astype(np.float32)
                   + (1.0 - top_weight) * b.astype(np.float32))
        return np.clip(blended, 0, 255).astype(np.uint8)

    return source


def split_plane_source(shape=(480, 640), seed=7, z_top=0.0, z_bottom=150.0,
                       k_per_step=0.03, base_sigma=0.4, split=0.5,
                       top_flakes=4, bottom_flakes=14) -> callable:
    """Spatially separated planes: top-plane flakes occupy the LEFT half,
    bottom-plane flakes the RIGHT half (and are much denser → the
    full-frame metric peaks at z_bottom while a left-half ROI follows
    z_top)."""
    img_top = synthetic_flake_image(shape=shape, seed=seed, n_flakes=top_flakes,
                                    noise=0.0)
    img_bottom = synthetic_flake_image(shape=shape, seed=seed + 1,
                                       n_flakes=bottom_flakes, noise=0.0)
    split_x = int(split * shape[1])
    left = np.zeros(shape[:2], dtype=bool)
    left[:, :split_x] = True

    def source(pos: float) -> np.ndarray:
        s_top = base_sigma + k_per_step * abs(pos - z_top)
        s_bot = base_sigma + k_per_step * abs(pos - z_bottom)
        out = defocus_blur(img_top, s_top).copy()
        out[~left] = defocus_blur(img_bottom, s_bot)[~left]
        return out

    return source


# ---------------------------------------------------------------------------
# Gaussian sharpness-curve sources (adaptive v3 — audit #11)
#
# The v2 rig's score curve (σ = 0.4 + k·|d| blur) is a zero-width cusp:
# convex everywhere except a measure-zero apex, ~30× narrower than the
# hardware's, and its sharpness-vs-z is NOT Gaussian. The v3 suite needs
# the model's own shape: the high-frequency AMPLITUDE follows a Gaussian
# in z, σ_z ≈ 1×coarse_step.
# ---------------------------------------------------------------------------

def gaussian_amplitude_source(sharp, focus_pos: float = 0.0,
                              sigma_z: float = 40.0, blur_sigma: float = 8.0,
                              floor: float = 0.0, noise: float = 0.0) -> callable:
    """A scene whose sharpness-vs-z curve is GAUSSIAN (S ≈ A·exp(−d²/2σ_z²)
    — the v3 model's shape): the sharp scene's high-frequency detail is
    amplitude-modulated by the Gaussian while a smoothed base stays
    visible at any defocus. `floor` adds a constant fraction of the
    detail back (floor=1 → a flat curve); `noise` = per-position
    deterministic frame noise (fraction of 255 — the debounce/2σ tests)."""
    sharp_arr = np.asarray(sharp, dtype=np.float32)
    base = defocus_blur(sharp, blur_sigma).astype(np.float32)
    detail = sharp_arr - base

    def source(pos: float) -> np.ndarray:
        g = math.exp(-(pos - focus_pos) ** 2 / (2.0 * sigma_z * sigma_z))
        amp = floor + (1.0 - floor) * g
        frame = base + amp * detail
        if noise:
            seed = (int(pos) & 0x7FFFFFFF)  # rng seeds must be non-negative
            frame = frame + np.random.default_rng(seed).normal(
                0, noise * 255, frame.shape)
        return np.clip(frame, 0, 255).astype(np.uint8)

    return source


def two_plane_gaussian_source(shape=(480, 640), seed=7, z_top=0.0,
                              z_bottom=240.0, sigma_z=40.0,
                              blur_sigma=8.0, top_weight=0.4,
                              noise=0.0) -> callable:
    """Two GAUSSIAN sharpness planes (the audit #8/#9 geometry): the
    wafer at z_top is deliberately the WEAKER plane (top_weight) — a
    naive global-max pick would land on the mount at z_bottom. Arm
    between the planes: the probe's center reads the valley, the sides
    the flanks — the center-max gate must block the near call and the
    run must fall through to the coarse machinery, landing the wafer
    (the nearest plane)."""
    img_top = synthetic_flake_image(shape=shape, seed=seed, noise=0.0)
    img_bottom = synthetic_flake_image(shape=shape, seed=seed + 1, noise=0.0)
    base_top = defocus_blur(img_top, blur_sigma).astype(np.float32)
    base_bottom = defocus_blur(img_bottom, blur_sigma).astype(np.float32)
    detail_top = img_top.astype(np.float32) - base_top
    detail_bottom = img_bottom.astype(np.float32) - base_bottom

    def source(pos: float) -> np.ndarray:
        g_top = math.exp(-(pos - z_top) ** 2 / (2.0 * sigma_z * sigma_z))
        g_bot = math.exp(-(pos - z_bottom) ** 2 / (2.0 * sigma_z * sigma_z))
        a = base_top + g_top * detail_top
        b = base_bottom + g_bot * detail_bottom
        frame = top_weight * a + (1.0 - top_weight) * b
        if noise:
            seed = (int(pos) & 0x7FFFFFFF)
            frame = frame + np.random.default_rng(seed).normal(
                0, noise * 255, frame.shape)
        return np.clip(frame, 0, 255).astype(np.uint8)

    return source
