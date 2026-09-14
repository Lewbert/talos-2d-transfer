"""Pixel-format conversion tests."""

import numpy as np
import pytest

from talos.protocols.pixel_unpack import convert_to_rgb, unpack_mono12p


def test_mono12p_unpack():
    # 2 pixels: p0=0xABC, p1=0x123
    # bytes: b0 = p0 & 0xFF = 0xBC,
    #        b1 = (p1 & 0xF) << 4 | (p0 >> 8) = 0x3A,
    #        b2 = p1 >> 4 = 0x12
    raw = np.array([0xBC, 0x3A, 0x12], dtype=np.uint8)
    out = unpack_mono12p(raw, width=2, height=1)
    assert out[0, 0] == 0xABC
    assert out[0, 1] == 0x123


def test_convert_mono8_to_rgb():
    gray = np.arange(24, dtype=np.uint8).reshape(4, 6)
    rgb = convert_to_rgb(gray.tobytes(), 6, 4, "Mono8")
    assert rgb.shape == (4, 6, 3)
    assert (rgb[:, :, 0] == gray).all()


def test_convert_bayerrg8_to_rgb():
    h, w = 4, 6
    bayer = np.zeros((h, w), dtype=np.uint8)
    bayer[1, 1] = 255  # a bright R-ish pixel
    rgb = convert_to_rgb(bayer.tobytes(), w, h, "BayerRG8")
    assert rgb.shape == (h, w, 3)
    assert rgb.max() > 0


def test_convert_unsupported_format_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        convert_to_rgb(b"\x00" * 12, 2, 2, "WeirdFormat42")


def test_convert_format_name_with_enum_suffix():
    gray = np.full((2, 2), 100, dtype=np.uint8)
    rgb = convert_to_rgb(gray.tobytes(), 2, 2, "genicam.genapi.PFNC.Mono8")
    assert rgb.shape == (2, 2, 3)
    assert (rgb[:, :, 0] == 100).all()
