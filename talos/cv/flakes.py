"""Classic flake detection (v1): illumination flatten → contrast threshold →
contour filtering → size filter (µm² via calibration) → HSV color gate.

Upgrade path: per-material HSV models, then ML (candidate crops exported
for later training). ``FlakeDetector`` is a Protocol so the pipeline can
swap implementations without touching callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from talos.models import FlakeCandidate, ObjectiveCalibration, StagePosition


@dataclass
class FlakeConfig:
    min_area_um2: float = 30.0
    max_area_um2: float = 100000.0
    blur_sigma: float =15.0           # illumination-flatten kernel
    morph_kernel: int = 3
    min_score: float = 6.0            # mean-contrast floor (0-255 scale)
    min_edge_strength: float = 8.0    # mean boundary Sobel magnitude —
                                      # crystals have sharp edges, tape
                                      # residue and smudges are diffuse
    border_margin_px: int = 4         # reject candidates touching the FOV edge
    merge_gap_px: int = 25            # merge fragmented boxes within this gap
    # Reject pure-red annotation overlays (scale bars, markers): mean hue
    # in the red extremes AND strongly saturated.
    reject_annotations: bool = True
    annotation_sat_min: int = 120
    # Color gate is OFF by default: flake interference colors span the
    # rainbow, so hue gating needs per-material tuning.
    color_gate: bool = False
    reject_hue_lo: int = 18           # yellow-green organic residue
    reject_hue_hi: int = 48
    reject_sat_max: int = 90          # only gate strongly saturated hues


class FlakeDetector(Protocol):
    def find(self, img: np.ndarray, calib: ObjectiveCalibration,
             cfg: FlakeConfig) -> list[FlakeCandidate]: ...


# ----------------------------------------------------------------------
# Shared pieces, used by this detector AND by the stage pipeline in
# ``talos.cv.identify`` — one implementation of each, so a fix (or a
# behaviour change) cannot land in only one of the two paths.
# ----------------------------------------------------------------------

def flatten_contrast(img: np.ndarray,
                     blur_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """(contrast_u8, binary mask): the illumination-flattened contrast image
    and its Otsu threshold. One blur pass serves both — the classic detector
    scores candidates on the contrast, the stage pipeline only needs the mask.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    background = cv2.GaussianBlur(gray, (0, 0), float(blur_sigma))
    flat = gray / (background + 1e-6)
    contrast = np.abs(flat - 1.0)
    contrast_u8 = np.clip(contrast * 255.0, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(contrast_u8, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return contrast_u8, binary


def merge_fragments(candidates: list[FlakeCandidate], gap_px: float,
                    um2_per_px2: float) -> list[FlakeCandidate]:
    """Union boxes within ``gap_px`` — a single flake often fragments into
    several adjacent contrast blobs."""
    if len(candidates) <= 1:
        return list(candidates)
    merged: list[FlakeCandidate] = []
    rest = sorted(candidates, key=lambda c: c.area_px2, reverse=True)
    while rest:
        base = rest.pop(0)
        bx, by, bw, bh = base.bbox
        absorbed = [base]
        i = 0
        while i < len(rest):
            other = rest[i]
            ox, oy, ow, oh = other.bbox
            gap_x = max(bx, ox) - min(bx + bw, ox + ow)
            gap_y = max(by, oy) - min(by + bh, oy + oh)
            if gap_x <= gap_px and gap_y <= gap_px:
                absorbed.append(other)
                rest.pop(i)
                # Grow the union box.
                nx0, ny0 = min(bx, ox), min(by, oy)
                nx1, ny1 = max(bx + bw, ox + ow), max(by + bh, oy + oh)
                bx, by, bw, bh = nx0, ny0, nx1 - nx0, ny1 - ny0
            else:
                i += 1
        if len(absorbed) == 1:
            merged.append(base)
        else:
            total_area = sum(c.area_px2 for c in absorbed)
            merged.append(FlakeCandidate(
                x_px=bx + bw / 2.0, y_px=by + bh / 2.0,
                area_px2=total_area,
                area_um2=total_area * um2_per_px2,
                score=max(c.score for c in absorbed),
                bbox=(bx, by, bw, bh)))
    return merged


def region_mean_hsv(hsv: np.ndarray, contour) -> tuple[float, float] | None:
    """Mean (hue, saturation) inside a contour, or None when it is empty."""
    mask = np.zeros(hsv.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    hue = hsv[:, :, 0][mask > 0]
    sat = hsv[:, :, 1][mask > 0]
    if hue.size == 0:
        return None
    return float(np.mean(hue)), float(np.mean(sat))


def is_red_annotation(hsv: np.ndarray, contour, sat_min: float) -> bool:
    """Pure-red overlays (scale bar text/bar) — mean hue at the red
    extremes with strong saturation. Wrap-around band: <10 or >168."""
    mean = region_mean_hsv(hsv, contour)
    if mean is None:
        return False
    mean_hue, mean_sat = mean
    if mean_sat < sat_min:
        return False
    return mean_hue < 10 or mean_hue > 168


def is_rejected_colour(hsv: np.ndarray, contour, cfg: FlakeConfig) -> bool:
    """Reject strongly yellow-green regions (organic residue) while keeping
    the purple/blue interference fringes of real flakes."""
    mean = region_mean_hsv(hsv, contour)
    if mean is None:
        return False
    mean_hue, mean_sat = mean
    if mean_sat > cfg.reject_sat_max:
        return False
    return cfg.reject_hue_lo <= mean_hue <= cfg.reject_hue_hi


class ClassicFlakeDetector:
    """Threshold + contour + size + color gate (see module docstring)."""

    def find(self, img: np.ndarray, calib: ObjectiveCalibration,
             cfg: FlakeConfig) -> list[FlakeCandidate]:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        # Illumination flatten: contrast against a heavily blurred copy.
        contrast_u8, binary = flatten_contrast(img, cfg.blur_sigma)
        kernel = np.ones((cfg.morph_kernel, cfg.morph_kernel), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        um_per_px = (calib.um_per_px_x or 1.0, calib.um_per_px_y or 1.0)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # Boundary-gradient magnitude for the per-candidate edge gate.
        grad = np.sqrt(
            cv2.Sobel(gray, cv2.CV_64F, 1, 0) ** 2
            + cv2.Sobel(gray, cv2.CV_64F, 0, 1) ** 2)

        candidates: list[FlakeCandidate] = []
        for contour in contours:
            area_px2 = float(cv2.contourArea(contour))
            if area_px2 < 4:
                continue
            area_um2 = area_px2 * um_per_px[0] * um_per_px[1]
            if not (cfg.min_area_um2 <= area_um2 <= cfg.max_area_um2):
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            x_px = moments["m10"] / moments["m00"]
            y_px = moments["m01"] / moments["m00"]
            if cfg.color_gate and self._rejected_color(hsv, contour, cfg):
                continue
            bbox = cv2.boundingRect(contour)
            # Border rejection: FOV-edge artifacts are not flakes.
            x, y, bw, bh = bbox
            if (x <= cfg.border_margin_px or y <= cfg.border_margin_px
                    or x + bw >= w - cfg.border_margin_px
                    or y + bh >= h - cfg.border_margin_px):
                continue
            score = float(np.mean(contrast_u8[y:y + bh, x:x + bw]))
            if score < cfg.min_score:
                continue
            # Sharp crystalline boundary? Smudges/tape residue are diffuse.
            # The ring is drawn on a BBOX-sized canvas: zeroing a
            # full-frame buffer per surviving candidate cost 2 MB of
            # allocation each at 1080p.
            pad = 2
            rx = max(0, x - pad)
            ry = max(0, y - pad)
            rw = min(w - rx, bw + 2 * pad)
            rh = min(h - ry, bh + 2 * pad)
            ring = np.zeros((rh, rw), dtype=np.uint8)
            cv2.drawContours(ring, [contour - (rx, ry)], -1, 255, 2)
            ring_values = grad[ry:ry + rh, rx:rx + rw][ring > 0]
            edge_strength = (float(ring_values.mean())
                             if ring_values.size else 0.0)
            if edge_strength < cfg.min_edge_strength:
                continue
            if cfg.reject_annotations and self._is_red_annotation(hsv, contour, cfg):
                continue
            candidates.append(FlakeCandidate(
                x_px=float(x_px), y_px=float(y_px), area_px2=area_px2,
                area_um2=area_um2, score=score, bbox=bbox,
                thumbnail=None))
        merged = self._merge_fragments(candidates, cfg,
                                       um_per_px[0] * um_per_px[1])
        merged.sort(key=lambda c: c.score, reverse=True)
        return merged

    # The shared implementations live at module level (see above) so the
    # stage pipeline in talos.cv.identify cannot drift from this detector.

    @staticmethod
    def _merge_fragments(candidates, cfg, um2_per_px2):
        return merge_fragments(candidates, cfg.merge_gap_px, um2_per_px2)

    @staticmethod
    def _is_red_annotation(hsv, contour, cfg):
        return is_red_annotation(hsv, contour, cfg.annotation_sat_min)

    @staticmethod
    def _rejected_color(hsv, contour, cfg):
        return is_rejected_colour(hsv, contour, cfg)


def flake_to_stage(x_px: float, y_px: float, img_shape: tuple[int, ...],
                   calib: ObjectiveCalibration,
                   stage_pos: StagePosition) -> tuple[float, float]:
    """Map a pixel position to stage coordinates (µm).

    v1: orthotropic pixel scale (um_per_px_x/y) with the image center at
    the stage position. M7 upgrades to the measured 2×2 jacobian.
    """
    h, w = img_shape[:2]
    um_per_px_x = calib.um_per_px_x or 1.0
    um_per_px_y = calib.um_per_px_y or 1.0
    dx_um = (x_px - w / 2.0) * um_per_px_x
    dy_um = (y_px - h / 2.0) * um_per_px_y
    return stage_pos.x_um + dx_um, stage_pos.y_um + dy_um


__all__ = ["FlakeConfig", "FlakeDetector", "ClassicFlakeDetector",
           "find_flakes", "flake_to_stage", "flatten_contrast",
           "merge_fragments", "region_mean_hsv", "is_red_annotation",
           "is_rejected_colour"]


def find_flakes(img: np.ndarray, calib: ObjectiveCalibration,
                cfg: FlakeConfig | None = None) -> list[FlakeCandidate]:
    """Convenience wrapper (classic detector)."""
    return ClassicFlakeDetector().find(img, calib, cfg or FlakeConfig())
