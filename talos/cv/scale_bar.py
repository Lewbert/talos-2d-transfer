"""Scale-bar geometry + drawing — ONE shared layout spec consumed by
both the live-view overlay (Qt, via the letterbox transform) and the
snapshot burn-in (cv2, at the captured frame's native resolution), so
both renderers show IDENTICAL box/bar/margins.

Layout rules (scientific accuracy first):
- The bar length is the LARGEST 1/2/5×10^k ("good number") whose span
  covers at most ``max_fraction`` (1/4) of the image width —
  ``nice_length_um_at_most``. The historical ``nice_length_um`` (nearest
  ladder value, up to 2.5× overshoot) was removed in the 2026-09-16 audit:
  nothing called it, and only its own tests kept it alive.
- The backing box has EQUAL spacing to the image's right and bottom
  edges; the label sits strictly UNDER the bar (no overlap); bar and
  label are centered horizontally in the box.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

_LADDER = (1.0, 2.0, 5.0)



def nice_length_um_at_most(um_per_px: float, view_width_px: int,
                           max_fraction: float = 0.25) -> float:
    """LARGEST 1/2/5×10^k whose pixel span is ≤ ``max_fraction`` of the
    view width — the scale-bar ladder (the bar may never exceed the
    fraction; the ≥-ladder can overshoot it by up to 2.5×)."""
    if um_per_px <= 0 or view_width_px <= 0:
        return 0.0
    max_um = um_per_px * view_width_px * max_fraction
    exponent = math.floor(math.log10(max_um)) if max_um > 0 else -3
    for k in range(exponent, exponent - 5, -1):
        for step in reversed(_LADDER):
            candidate = step * (10.0 ** k)
            if candidate <= max_um:
                return candidate
    # degenerate: nothing in the window fits — use the smallest candidate
    return _LADDER[0] * (10.0 ** (exponent - 5))


def scale_bar_px(length_um: float, um_per_px: float) -> int:
    if um_per_px <= 0:
        return 0
    return int(round(length_um / um_per_px))


def format_length_um(length_um: float) -> str:
    if length_um >= 1000.0:
        return f"{length_um / 1000.0:g} mm"
    if length_um >= 1.0:
        return f"{length_um:g} µm"
    return f"{length_um * 1000.0:g} nm"


def burn_spec(um_per_px: float | None, enabled: bool) -> dict | None:
    """The snapshot burn request: None when disabled or uncalibrated."""
    if not enabled or not um_per_px or um_per_px <= 0:
        return None
    return {"um_per_px": float(um_per_px)}


@dataclass(frozen=True)
class ScaleBarSpec:
    """The scale-bar geometry in FRAME pixels, bottom-right with equal
    right/bottom margins. The Qt overlay maps it through the letterbox
    transform; the cv2 burn uses it directly."""

    length_um: float
    bar_px: int
    bar_h_px: int
    box: tuple[int, int, int, int]        # (x1, y1, x2, y2)
    bar_rect: tuple[int, int, int, int]   # (x, y, w, h); y = TOP edge
    text_rect: tuple[int, int, int, int]  # the label band, under bar_rect
    label: str
    font_scale: float
    thickness: int
    margin_px: int


@lru_cache(maxsize=32)
def _scale_bar_layout_cached(um_per_px: float, frame_shape: tuple,
                             max_fraction: float, margin_px: int) \
        -> ScaleBarSpec | None:
    return _scale_bar_layout(um_per_px, frame_shape, max_fraction, margin_px)


def scale_bar_layout(um_per_px: float, frame_shape: tuple,
                     max_fraction: float = 0.25,
                     margin_px: int = 14) -> ScaleBarSpec | None:
    """The shared layout spec. None when uncalibrated/degenerate.

    Cached: both renderers call this on every repaint/redraw and the inputs
    only change on a calibration edit, a resolution switch or a resize.
    The spec is frozen, so sharing it is safe.
    """
    if um_per_px is None or not isinstance(um_per_px, (int, float)):
        return None
    shape = tuple(frame_shape[:2]) if frame_shape is not None else ()
    return _scale_bar_layout_cached(float(um_per_px), shape,
                                    float(max_fraction), int(margin_px))


def _scale_bar_layout(um_per_px: float, frame_shape: tuple,
                      max_fraction: float = 0.25,
                      margin_px: int = 14) -> ScaleBarSpec | None:
    """The layout itself (uncached — see scale_bar_layout)."""
    if um_per_px is None or um_per_px <= 0:
        return None
    if len(frame_shape) < 2:
        return None
    h, w = int(frame_shape[0]), int(frame_shape[1])
    if w <= 0 or h <= 0:
        return None
    length_um = nice_length_um_at_most(um_per_px, w, max_fraction)
    bar_px = scale_bar_px(length_um, um_per_px)
    if bar_px <= 0:
        return None
    label = format_length_um(length_um)
    font_scale = max(0.7, w / 2400.0)
    thickness = max(1, int(round(w / 1000.0)))
    bar_h = max(2, thickness + 1)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                         font_scale, thickness)
    pad = margin_px // 2
    gap = max(4, thickness)
    box_w = max(bar_px, tw) + 2 * pad
    box_h = pad + bar_h + gap + th + baseline + pad
    # EQUAL spacing to the image right and bottom edges.
    x2 = w - margin_px
    y2 = h - margin_px
    x1 = x2 - box_w
    y1 = y2 - box_h
    bar_x = x1 + (box_w - bar_px) // 2
    bar_y = y1 + pad
    bar_rect = (bar_x, bar_y, bar_px, bar_h)
    text_y = bar_y + bar_h + gap
    text_rect = (x1, text_y, box_w, th + baseline)
    return ScaleBarSpec(length_um=length_um, bar_px=bar_px, bar_h_px=bar_h,
                        box=(x1, y1, x2, y2), bar_rect=bar_rect,
                        text_rect=text_rect, label=label,
                        font_scale=font_scale, thickness=thickness,
                        margin_px=margin_px)


def draw_scale_bar_cv(frame: np.ndarray, um_per_px: float,
                      max_fraction: float = 0.25, margin_px: int = 14,
                      color: tuple = (255, 255, 255),
                      bg_alpha: float = 0.45) -> np.ndarray:
    """Burn the scale bar into the bottom-right corner of ``frame`` from
    the shared layout spec (label UNDER the bar, equal right/bottom
    margins). Returns the input array unchanged when no bar applies."""
    spec = scale_bar_layout(um_per_px, frame.shape, max_fraction, margin_px)
    if spec is None:
        return frame
    x1, y1, x2, y2 = spec.box
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, bg_alpha, frame, 1.0 - bg_alpha, 0)
    bx, by, bw, bh = spec.bar_rect
    # cv2 fills rects inclusive of BOTH edges — subtract 1 so the drawn
    # bar spans exactly bar_px pixels (the spec's width, not +1).
    cv2.rectangle(frame, (bx, by), (bx + bw - 1, by + bh - 1), color, -1)
    tx, ty, tw, th_band = spec.text_rect
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw2, th2), baseline = cv2.getTextSize(spec.label, font,
                                           spec.font_scale, spec.thickness)
    text_x = tx + (tw - tw2) // 2
    text_y = ty + th_band - baseline
    cv2.putText(frame, spec.label, (text_x, text_y), font,
                spec.font_scale, color, spec.thickness, cv2.LINE_AA)
    return frame
