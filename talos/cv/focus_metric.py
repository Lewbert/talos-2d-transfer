"""Focus-quality metrics. Primary: Laplacian variance (robust, standard,
fast on 1080p crops). Alternates kept as config-selectable cross-checks."""

from __future__ import annotations

import cv2
import numpy as np


def _gray(img: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    # Crop FIRST, then convert: the default ROI is the centre 2/3 of the
    # frame, so converting the whole 1080p image for a 44%-area crop was
    # ~2.3× wasted work per metric call — twice per scored frame in the
    # live AF loop.
    if roi is not None:
        x, y, w, h = roi
        img = img[y:y + h, x:x + w]
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def laplacian_variance(img: np.ndarray,
                       roi: tuple[int, int, int, int] | None = None) -> float:
    """Variance of the Laplacian — the primary focus metric."""
    gray = _gray(img, roi)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad(img: np.ndarray,
              roi: tuple[int, int, int, int] | None = None) -> float:
    """Sobel magnitude variance (backup metric)."""
    gray = _gray(img, roi)
    # CV_32F, not CV_64F: this runs over up to ~900k pixels per frame and
    # the metric is relative — 32-bit precision is far beyond what the
    # peak fitting needs, at half the memory traffic.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return float((gx * gx + gy * gy).mean())


def _row_delta(gray: np.ndarray, k: int) -> np.ndarray:
    """Row differences at offset k, or an EMPTY array when the ROI is too
    short. An empty difference array made mean() return NaN with a numpy
    warning, and nothing downstream rejects NaN: pick_peak's `hi - lo <=
    1e-12` test is False for NaN, so a NaN-poisoned curve failed later
    with a misleading message instead of scoring 0."""
    if gray.shape[0] <= k:
        return np.empty((0,), dtype=np.float64)
    return gray[k:, :] - gray[:-k, :]


def brenner(img: np.ndarray,
            roi: tuple[int, int, int, int] | None = None) -> float:
    """Cheap neighbor-difference-squared metric (fast fallback)."""
    gray = _gray(img, roi).astype(np.float64)
    diff = _row_delta(gray, 2)
    return float((diff ** 2).mean()) if diff.size else 0.0


def brenner_k(img: np.ndarray,
              roi: tuple[int, int, int, int] | None = None,
              k: int = 8) -> float:
    """Large-kernel Brenner: row differences at offset ``k``. The low
    spatial frequency keeps the peak broad and blur-tolerant — the coarse
    sweep scores with this while moving fast (motion blur smears fine
    detail far more than it does k-pixel-scale structure)."""
    gray = _gray(img, roi).astype(np.float64)
    diff = _row_delta(gray, k)
    return float((diff ** 2).mean()) if diff.size else 0.0


def abs_diff(img: np.ndarray,
             roi: tuple[int, int, int, int] | None = None,
             k: int = 8) -> float:
    """Absolute-difference (SMD) at offset ``k`` — even cheaper and more
    linear than squared differences; same low-frequency rationale."""
    gray = _gray(img, roi).astype(np.float64)
    diff = _row_delta(gray, k)
    return float(np.abs(diff).mean()) if diff.size else 0.0


def bin2(img: np.ndarray, roi: tuple[int, int, int, int] | None = None) \
        -> np.ndarray:
    """2×2 area-average binning of the (optionally cropped) gray image.
    INTER_AREA averaging is noise-robust — coarse-sweep metrics run on the
    binned frame, where 2×-scale structure is all that matters."""
    gray = _gray(img, roi)
    return cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2),
                      interpolation=cv2.INTER_AREA)


METRICS = {
    "laplacian": laplacian_variance,
    "tenengrad": tenengrad,
    "brenner": brenner,
    "brenner_k": brenner_k,
    "abs_diff": abs_diff,
}


def default_roi(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Center 50% crop — excludes edges and speeds up computation."""
    h, w = shape[:2]
    return (w // 4, h // 4, w // 2, h // 2)


def sharpness_profile(positions: np.ndarray, frames: list[np.ndarray],
                      metric=laplacian_variance,
                      roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Metric curve over a focus stack (one value per frame)."""
    roi = roi if roi is not None else default_roi(frames[0].shape)
    return np.array([metric(frame, roi) for frame in frames], dtype=float)
