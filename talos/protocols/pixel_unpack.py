"""Pixel-format conversions for camera payloads (GenTL/GenICam formats)."""

from __future__ import annotations

import numpy as np


def unpack_mono12p(data: np.ndarray, width: int, height: int) -> np.ndarray:
    """Unpack packed 12-bit (3 bytes = 2 pixels) into uint16."""
    flat = data.reshape(-1).astype(np.uint16)
    n = (len(flat) // 3) * 2
    out = np.zeros(n, dtype=np.uint16)
    # Little-endian packing: [p0 low | p0 high<<4 in b1 low nibble | p1 low<<4 in b1 high | p1 high]
    b0 = flat[0::3]
    b1 = flat[1::3]
    b2 = flat[2::3]
    out[0::2] = (b1 & 0x0F) << 8 | b0
    out[1::2] = b2 << 4 | (b1 >> 4)
    return out.reshape(height, width)


def convert_to_rgb(data, width: int, height: int, data_format: str) -> np.ndarray:
    """Convert a raw GenTL payload to RGB uint8 HxWx3.

    ``data`` is a bytes-like (memoryview ok); ``data_format`` may be a
    GenICam enum — ``str()`` normalization accepts both.
    """
    import cv2

    fmt = str(data_format).split(".")[-1].lower()
    raw = np.frombuffer(data, dtype=np.uint8)

    if fmt in ("mono8", "mono8s"):
        gray = raw.reshape(height, width)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if fmt == "mono12p":
        gray16 = unpack_mono12p(raw, width, height)
        return cv2.cvtColor((gray16 >> 4).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if fmt in ("mono12", "mono16"):
        gray16 = raw.reshape(-1, 2).view(np.uint16).reshape(height, width)
        return cv2.cvtColor((gray16 >> 8).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if fmt == "rgb8":
        return raw.reshape(height, width, 3).copy()
    if fmt == "bgr8":
        return cv2.cvtColor(raw.reshape(height, width, 3), cv2.COLOR_BGR2RGB)
    if fmt == "yuyv":
        return cv2.cvtColor(raw.reshape(height, width, 2), cv2.COLOR_YUV2RGB_YUYV)
    if fmt.startswith("bayer"):
        pattern = fmt[5:7].upper()  # "BayerRG8" -> "RG"
        code = getattr(cv2, f"COLOR_Bayer{pattern}2RGB", None)
        if code is None:
            raise ValueError(f"Unsupported Bayer pattern: {data_format}")
        if "12p" in fmt:
            gray16 = unpack_mono12p(raw, width, height)
            return cv2.cvtColor((gray16 >> 4).astype(np.uint8), code)
        gray = raw.reshape(height, width)
        return cv2.cvtColor(gray, code)
    raise ValueError(f"Unsupported pixel format: {data_format}")
