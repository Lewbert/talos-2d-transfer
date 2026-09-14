"""Pixel decoding for the SmartCamApi camera (Axiocam 202/208).

Pure numpy/cv2 — no Qt, no driver imports — so it is unit-testable with
saved raw buffers (fixtures under tests/data/smartcam/).

Facts extracted from ZEN's own SmartCam wrapper (see docs/SMARTCAM_API.md):
the camera's ``TransferFormat 0`` "YUV420" is really **semi-planar NV12**
(Y plane + ONE interleaved U/V plane). Decoding it as planar I420 — as the
old backend did — corrupts the chroma and makes the stream look gray.
"""

from __future__ import annotations

import numpy as np
import cv2

LIVE_W, LIVE_H = 1920, 1080


def trim_data_extent(raw: np.ndarray) -> np.ndarray:
    """Drop trailing zero padding of the fixed-size acquisition buffer."""
    nz = np.nonzero(raw)[0]
    if nz.size == 0:
        return raw
    return raw[: int(nz[-1]) + 1]


def decode_yuv420(raw: np.ndarray, w: int = LIVE_W, h: int = LIVE_H,
                  wb: tuple[float, float, float] | None = None) -> np.ndarray:
    """Decode the camera's semi-planar YUV420 transfer to RGB uint8.

    Layout: Y plane w*h, then ONE interleaved chroma plane (w*h/2 bytes,
    pairs at half resolution in both axes). HARDWARE-VERIFIED order: each
    pair is (V, U) — i.e. the first byte is the red axis, second the blue
    axis. (ZEN's docs call this transfer "YUV420"; a standard U-first
    decode yields a cold blue cast — swapped gives the correct purple
    wafer / orange copper, chroma correlation 0.84/0.98 vs ZEN.)

    That V-first semi-planar layout is NV21 — decoded natively by OpenCV
    (validated against the previous resize/stack pipeline: mean abs diff
    3.7/255 chroma-siting only, R correlation vs the ZEN reference
    IMPROVES 0.874 -> 0.934, and it costs 1.6 ms vs 17.4 ms per 1080p
    frame).

    ``wb``: optional ZEN software white-balance factors (R, G, B as
    relative gains, e.g. the 5500K preset (0.62657, 1.0, 0.55448)).
    Applied the same way ZEN does (channel LUT, 1/ratio).
    """
    raw = np.asarray(raw, dtype=np.uint8)
    y_size = w * h
    if raw.size < y_size + y_size // 2:
        raise ValueError(f"YUV420 needs {y_size + y_size // 2} bytes, got {raw.size}")
    nv21 = raw[:y_size + y_size // 2].reshape(h + h // 2, w)
    rgb = cv2.cvtColor(nv21, cv2.COLOR_YUV2RGB_NV21)
    if wb is not None:
        rgb = apply_wb_lut(rgb, wb)
    return rgb


_WB_LUT_CACHE: dict[tuple, np.ndarray] = {}
_WB_LUT_CACHE_MAX = 8


def _wb_lut(factors: tuple[float, float, float]) -> np.ndarray:
    """The 1×256×3 byte LUT for these factors, built once.

    The factors come from a fixed preset table, so rebuilding the table
    (two arange/clip passes plus the allocation) on EVERY frame was pure
    waste in the live path.
    """
    key = tuple(round(float(f), 6) for f in factors)
    lut = _WB_LUT_CACHE.get(key)
    if lut is None:
        lut = np.empty((1, 256, 3), dtype=np.uint8)
        for i, factor in enumerate(key):
            lut[0, :, i] = np.clip(np.arange(256, dtype=np.float64)
                                   / max(factor, 1e-9), 0, 255)
        if len(_WB_LUT_CACHE) >= _WB_LUT_CACHE_MAX:
            _WB_LUT_CACHE.pop(next(iter(_WB_LUT_CACHE)))
        _WB_LUT_CACHE[key] = lut
    return lut


def apply_wb_lut(rgb: np.ndarray, factors: tuple[float, float, float]) -> np.ndarray:
    """ZEN's software white-balance LUT: multiply each channel by
    1/relative-gain (green normalized to 1), via a single multi-channel
    byte LUT (bit-identical to per-channel LUTs, ~3x faster)."""
    return cv2.LUT(rgb, _wb_lut(factors))


def decode_yuv420_planar(raw: np.ndarray, w: int = LIVE_W, h: int = LIVE_H) -> np.ndarray:
    """Legacy planar-I420 decode — kept for comparison studies only."""
    raw = np.asarray(raw, dtype=np.uint8)
    y_size = w * h
    uv_size = (w // 2) * (h // 2)
    y = raw[:y_size].reshape(h, w)
    u = cv2.resize(raw[y_size:y_size + uv_size].reshape(h // 2, w // 2), (w, h),
                   interpolation=cv2.INTER_LINEAR)
    v = cv2.resize(raw[y_size + uv_size:y_size + 2 * uv_size].reshape(h // 2, w // 2),
                   (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(np.stack([y, u, v], axis=2), cv2.COLOR_YUV2RGB)


def bayer_pattern_stats(mono: np.ndarray) -> dict[str, float]:
    """Diagnostics to distinguish a Bayer mosaic from plain gray.

    For a true Bayer mosaic the adjacent-pixel difference energy dominates
    the 2-apart difference energy in BOTH axes (checkerboard). A smooth or
    normally-textured gray image shows comparable energies.
    """
    a = mono.astype(np.float64)
    adj_h = float(np.abs(np.diff(a, axis=1)).mean())
    adj_v = float(np.abs(np.diff(a, axis=0)).mean())
    two_h = float(np.abs(a[:, 2:] - a[:, :-2]).mean())
    two_v = float(np.abs(a[2:, :] - a[:-2, :]).mean())
    phases = {}
    for (pi, pj), name in zip(((0, 0), (0, 1), (1, 0), (1, 1)),
                              ("p00", "p01", "p10", "p11")):
        phases[name] = float(a[pi::2, pj::2].mean())
    return {
        "adj_h": adj_h, "adj_v": adj_v, "two_h": two_h, "two_v": two_v,
        "ratio_h": adj_h / two_h if two_h > 0 else 0.0,
        "ratio_v": adj_v / two_v if two_v > 0 else 0.0,
        **phases,
    }


def detect_bayer_pattern(mono: np.ndarray) -> str:
    """Classify a monochrome array: "gray" or a Bayer pattern name.

    Bayer pattern discrimination: if the image is a mosaic, both
    adj/2-apart ratios are clearly above ~1.6 (checkerboard); the pattern is
    then picked by which 2x2 phase has the strongest high-frequency content
    mismatch — a simple approach: compare per-phase means and locate the two
    green phases (the phases whose neighbor-difference is largest).
    """
    stats = bayer_pattern_stats(mono)
    if stats["ratio_h"] < 1.6 or stats["ratio_v"] < 1.6:
        return "gray"
    a = mono.astype(np.float64)
    # Green phases carry the most local variance in a Bayer mosaic; the two
    # green phases are diagonal from each other. The red/blue phases are the
    # remaining pair, with the red phase usually brighter than blue.
    var00 = float(a[0::2, 0::2].var())
    var11 = float(a[1::2, 1::2].var())
    var01 = float(a[0::2, 1::2].var())
    var10 = float(a[1::2, 0::2].var())
    diag_a = var00 + var11
    diag_b = var01 + var10
    if diag_a > diag_b:
        greens = {(0, 0), (1, 1)}
        others = {(0, 1), (1, 0)}
    else:
        greens = {(0, 1), (1, 0)}
        others = {(0, 0), (1, 1)}
    # In BGGR/RGGB the red phase is row 0 of `others`; in GBRG/GRBG it is
    # column-dependent. Pick the brighter of the two non-green phases as red.
    (r_pos, b_pos) = sorted(others, key=lambda p: float(a[p[0]::2, p[1]::2].mean()),
                            reverse=True)
    if greens == {(0, 0), (1, 1)}:
        return "bggr" if r_pos == (0, 0) else "rggb"
    return "gbrg" if r_pos == (0, 1) else "grbg"


def demosaic(mono: np.ndarray, pattern: str) -> np.ndarray:
    """cv2 Bayer demosaic; pattern is the OpenCV-ish name (first row letters)."""
    codes = {"rggb": cv2.COLOR_BayerBG2RGB, "bggr": cv2.COLOR_BayerRG2RGB,
             "grbg": cv2.COLOR_BayerGB2RGB, "gbrg": cv2.COLOR_BayerGR2RGB}
    if pattern not in codes:
        raise ValueError(f"unknown bayer pattern {pattern!r}")
    return cv2.cvtColor(mono, codes[pattern])


def score_against_reference(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Compare a candidate decode against a color ground-truth image.

    Returns normalized cross-correlation on luma plus per-channel histogram
    correlation, after area-resizing the candidate to the reference size.
    """
    cand = cv2.resize(candidate, (reference.shape[1], reference.shape[0]),
                      interpolation=cv2.INTER_AREA)
    cl = cv2.cvtColor(cand, cv2.COLOR_RGB2GRAY).astype(np.float64)
    rl = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float64)
    cl = (cl - cl.mean()) / (cl.std() + 1e-9)
    rl = (rl - rl.mean()) / (rl.std() + 1e-9)
    ncc = float((cl * rl).mean())
    hist = {}
    for i, ch in enumerate("rgb"):
        c = cand[:, :, i].astype(np.float64)
        r = reference[:, :, i].astype(np.float64)
        c = (c - c.mean()) / (c.std() + 1e-9)
        r = (r - r.mean()) / (r.std() + 1e-9)
        hist[ch] = float((c * r).mean())
    return {"luma_ncc": ncc, **hist}
