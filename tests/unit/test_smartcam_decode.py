"""Decode tests for the SmartCamApi pixel formats (pure numpy/cv2)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from talos.hal.devices.camera.smartcam_decode import (
    apply_wb_lut,
    bayer_pattern_stats,
    decode_yuv420,
    decode_yuv420_planar,
    demosaic,
    detect_bayer_pattern,
    score_against_reference,
    trim_data_extent,
)

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "smartcam"


def rgb_to_nv12(rgb: np.ndarray) -> np.ndarray:
    """Reference encoder in the CAMERA's format: Y + interleaved (V, U) pairs."""
    h, w = rgb.shape[:2]
    yuv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2YUV)
    y = yuv[:, :, 0]
    u = cv2.resize(yuv[:, :, 1], (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    v = cv2.resize(yuv[:, :, 2], (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    vu = np.empty((h // 2, w // 2, 2), dtype=np.uint8)
    vu[:, :, 0] = v
    vu[:, :, 1] = u
    return np.concatenate([y.ravel(), vu.ravel()])


def test_trim_data_extent_drops_trailing_zeros():
    raw = np.zeros(100, dtype=np.uint8)
    raw[:20] = 7
    assert trim_data_extent(raw).size == 20


def test_nv12_solid_colors():
    w, h = 64, 64
    for name, rgb in (("red", (200, 20, 20)), ("green", (20, 200, 20)),
                      ("blue", (20, 20, 200)), ("gray", (128, 128, 128))):
        img = np.full((h, w, 3), rgb, dtype=np.uint8)
        out = decode_yuv420(rgb_to_nv12(img), w, h)
        assert out.shape == (h, w, 3)
        # subsampling error tolerance; hue must survive
        if name == "red":
            assert out[:, :, 0].mean() > out[:, :, 2].mean()
        elif name == "blue":
            assert out[:, :, 2].mean() > out[:, :, 0].mean()
        elif name == "green":
            assert out[:, :, 1].mean() > out[:, :, 0].mean()
        elif name == "gray":
            stds = [out[:, :, i].std() for i in range(3)]
            assert max(stds) - min(stds) < 8


def test_nv12_textured_roundtrip():
    rng = np.random.default_rng(7)
    noisy = rng.integers(0, 255, (180, 320, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(noisy, (7, 7), 0)  # smooth: subsampling is accurate
    out = decode_yuv420(rgb_to_nv12(img), 320, 180)
    err = np.abs(out.astype(float) - img.astype(float)).mean()
    assert err < 10  # chroma subsampling + interpolation loss


def test_nv12_requires_full_size():
    with pytest.raises(ValueError):
        decode_yuv420(np.zeros(100, dtype=np.uint8), 64, 64)


def test_apply_wb_lut_boosts_channels():
    # ZEN's 5500K factors (0.62657, 1.0, 0.55448): R x1.596, B x1.8035
    img = np.full((8, 8, 3), (100, 100, 100), dtype=np.uint8)
    out = apply_wb_lut(img, (0.62657004317365, 1.0, 0.55448148303058))
    assert out.shape == img.shape
    assert float(out[:, :, 2].mean()) == pytest.approx(180.35, abs=1.0)
    assert float(out[:, :, 0].mean()) == pytest.approx(159.6, abs=1.0)
    assert float(out[:, :, 1].mean()) == pytest.approx(100.0, abs=1.0)
    # clipping at 255
    assert apply_wb_lut(np.full((4, 4, 3), 200, dtype=np.uint8),
                        (0.5, 1.0, 0.5)).max() == 255


def test_planar_decode_still_roundtrips_planar_data():
    w, h = 64, 64
    y = np.full((h, w), 100, dtype=np.uint8)
    u = np.full((h // 2, w // 2), 128, dtype=np.uint8)
    v = np.full((h // 2, w // 2), 128, dtype=np.uint8)
    planar = np.concatenate([y.ravel(), u.ravel(), v.ravel()])
    out = decode_yuv420_planar(planar, w, h)
    assert out.shape == (h, w, 3)
    assert abs(float(out[:, :, 0].mean()) - 100.0) < 5


def _make_mosaic(rgb: np.ndarray, pattern: str) -> np.ndarray:
    """Downsample an RGB image into a Bayer mosaic of the given pattern."""
    h, w = rgb.shape[:2]
    mosaic = np.zeros((h, w), dtype=np.uint8)
    for pat, (y0, x0, ch) in (("rggb", (0, 0, 0)), ("rggb", (0, 1, 1)),
                              ("rggb", (1, 0, 1)), ("rggb", (1, 1, 2)),
                              ("bggr", (0, 0, 2)), ("bggr", (0, 1, 1)),
                              ("bggr", (1, 0, 1)), ("bggr", (1, 1, 0)),
                              ("grbg", (0, 0, 1)), ("grbg", (0, 1, 0)),
                              ("grbg", (1, 0, 2)), ("grbg", (1, 1, 1)),
                              ("gbrg", (0, 0, 1)), ("gbrg", (0, 1, 2)),
                              ("gbrg", (1, 0, 0)), ("gbrg", (1, 1, 1))):
        if pat != pattern:
            continue
        mosaic[y0::2, x0::2] = rgb[y0::2, x0::2, ch]
    return mosaic


def test_detect_bayer_on_textured_mosaics():
    rng = np.random.default_rng(3)
    img = cv2.GaussianBlur(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8),
                           (5, 5), 0)
    # red-dominant texture so the red phase is detectable
    img[:, :, 0] = np.clip(img[:, :, 0].astype(int) + 60, 0, 255).astype(np.uint8)
    for pattern in ("rggb", "bggr", "grbg", "gbrg"):
        mosaic = _make_mosaic(img, pattern)
        stats = bayer_pattern_stats(mosaic)
        assert stats["ratio_h"] > 1.5 and stats["ratio_v"] > 1.5
        detected = detect_bayer_pattern(mosaic)
        assert detected != "gray"
        demosaic(mosaic, detected)  # must not raise


def test_detect_bayer_smooth_gray_is_gray():
    img = cv2.GaussianBlur(np.full((64, 64), 100, dtype=np.uint8), (3, 3), 0)
    assert detect_bayer_pattern(img) == "gray"


def test_fixture_nv12_decode():
    raw = np.load(FIXTURES / "raw_post_zen.npy")
    trimmed = trim_data_extent(raw)
    out = decode_yuv420(trimmed)
    assert out.shape == (1080, 1920, 3)
    assert out.dtype == np.uint8
    # real captured buffer: luma sane, chroma present
    assert 60 < out.mean() < 130
    y = cv2.cvtColor(out, cv2.COLOR_RGB2YUV)[:, :, 0]
    assert abs(float(y.mean()) - 91.0) < 10  # known Y mean of this capture


def test_score_against_reference_self_match():
    rng = np.random.default_rng(11)
    img = cv2.GaussianBlur(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8),
                           (5, 5), 0)
    score = score_against_reference(img, img)
    assert abs(score["luma_ncc"] - 1.0) < 1e-6
