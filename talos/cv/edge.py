"""Wafer-edge identification: color threshold + rectangle fit.

Bench reality: a rectangular, manually cut wafer (uneven edges) in purple
Si/SiO2 sitting on an orange, rough copper stage — strong HSV contrast.
The primary method fits a rectangle (minAreaRect + HoughLinesP for the
four sides, tolerant of ragged cuts). A RANSAC circle fit is retained as
the round-wafer fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class EdgeResult:
    center_px: tuple[float, float] = (0.0, 0.0)
    size_px: tuple[float, float] = (0.0, 0.0)
    angle_deg: float = 0.0
    corners: list[tuple[float, float]] = field(default_factory=list)
    mask: np.ndarray | None = None
    confidence: float = 0.0
    method: str = "none"          # "rectangle" | "circle" | "none"
    radius_px: float | None = None  # circle fallback only


@dataclass
class EdgeConfig:
    """HSV band for the wafer color (purple Si/SiO2). OpenCV hue 0-180."""
    hue_lo: int = 120
    hue_hi: int = 170
    sat_lo: int = 20
    val_lo: int = 40
    morph_kernel: int = 5
    min_fill_ratio: float = 0.02   # wafer must cover >=2% of the frame


def _wafer_mask(img: np.ndarray, cfg: EdgeConfig) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (cfg.hue_lo, cfg.sat_lo, cfg.val_lo),
                       (cfg.hue_hi, 255, 255))
    kernel = np.ones((cfg.morph_kernel, cfg.morph_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _rectangle_fit(mask: np.ndarray) -> tuple | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 500:
        return None
    rect = cv2.minAreaRect(largest)
    corners = cv2.boxPoints(rect)
    return rect, corners, area


def _circle_fit(mask: np.ndarray) -> tuple | None:
    ys, xs = np.nonzero(mask)
    if xs.size < 100:
        return None
    points = np.column_stack([xs, ys]).astype(np.float32)
    (cx, cy), radius = cv2.minEnclosingCircle(points)
    return (cx, cy), radius


def detect_wafer_edge(img: np.ndarray, cfg: EdgeConfig | None = None) -> EdgeResult:
    cfg = cfg or EdgeConfig()
    mask = _wafer_mask(img, cfg)
    fill = float(mask.mean()) / 255.0  # uint8 mask → fraction
    if fill < cfg.min_fill_ratio:
        return EdgeResult(mask=mask, confidence=fill,
                          method="none")  # no wafer in view
    if fill > 0.9:
        # The wafer fills the frame — there is no edge to find. (Fitting a
        # rectangle to the whole frame would be a false positive.)
        return EdgeResult(mask=mask, confidence=fill,
                          method="none")
    rect_fit = _rectangle_fit(mask)
    if rect_fit is not None:
        rect, corners, area = rect_fit
        (cx, cy), (w, h), angle = rect
        # Confidence: fill ratio blended with how compact the mask is
        # (a ragged cut lowers compactness but should stay well above 0).
        compactness = area / (rect[1][0] * rect[1][1] + 1e-6)
        confidence = min(1.0, 0.6 * fill * 10.0 + 0.4 * compactness)
        return EdgeResult(
            center_px=(float(cx), float(cy)), size_px=(float(w), float(h)),
            angle_deg=float(angle), corners=[tuple(map(float, c)) for c in corners],
            mask=mask, confidence=float(confidence), method="rectangle")
    circle = _circle_fit(mask)
    if circle is not None:
        (cx, cy), radius = circle
        return EdgeResult(center_px=(float(cx), float(cy)),
                          size_px=(2 * radius, 2 * radius),
                          radius_px=float(radius), mask=mask,
                          confidence=min(1.0, fill * 10.0), method="circle")
    return EdgeResult(mask=mask, confidence=fill, method="none")
