"""ROI math for the live view: widget↔frame coordinate mapping for the
KeepAspectRatio letterboxed fit, plus normalized-ROI scaling across
camera resolution changes.

The transform inverted here is exactly the one applied in
LiveViewWidget._render_pending: image.scaled(label.size(),
KeepAspectRatio, SmoothTransformation) into a label that fills the widget
(no margins) and centers the scaled image.

No Qt — the widget layer converts QRectF to the plain tuples used here.
"""

from __future__ import annotations

#: A ROI smaller than 2 % of a side is not useful.
MIN_SIDE = 0.02


def sanitize_roi_norm(norm) -> tuple[float, float, float, float] | None:
    """Clamp an arbitrary normalized (x, y, w, h) into the frame; None when
    unusable.

    Lives here (not in the widget layer) because BOTH sides need it: the
    rubber band's result and the value read back from settings at run time.
    The autofocus used to pass the stored tuple through unsanitized, so a
    hand-edited ``default_roi_norm`` of (0.9999, 0.9999, 0.0001, 0.0001)
    produced a 1-pixel crop at the frame edge and cv2 raised inside the
    metric — surfaced to the operator as "internal error".
    """
    if norm is None:
        return None
    try:
        x, y, w, h = (float(v) for v in norm)
    except (TypeError, ValueError):
        return None
    x = min(max(x, 0.0), 1.0 - MIN_SIDE)
    y = min(max(y, 0.0), 1.0 - MIN_SIDE)
    w = min(max(w, MIN_SIDE), 1.0 - x)
    h = min(max(h, MIN_SIDE), 1.0 - y)
    return (round(x, 4), round(y, 4), round(w, 4), round(h, 4))


def fit_transform(widget_size: tuple[int, int], frame_shape: tuple) \
        -> tuple[float, float, float]:
    """(scale, offset_x, offset_y) of the letterboxed fit."""
    w_w, w_h = widget_size
    h, w = frame_shape[:2]
    if w_w <= 0 or w_h <= 0 or w <= 0 or h <= 0:
        return 1.0, 0.0, 0.0
    scale = min(w_w / w, w_h / h)
    off_x = (w_w - w * scale) / 2.0
    off_y = (w_h - h * scale) / 2.0
    return scale, off_x, off_y


# the historical private name — kept for the stored-knowledge callers
_fit = fit_transform


def letterbox_map(widget_size: tuple[int, int], frame_shape: tuple,
                  widget_point: tuple[float, float]) -> tuple[float, float]:
    """Map a WIDGET point to FRAME pixel coordinates (float, unclamped —
    points outside the image area map outside [0, w]×[0, h])."""
    scale, off_x, off_y = _fit(widget_size, frame_shape)
    wx, wy = widget_point
    return (wx - off_x) / scale, (wy - off_y) / scale


def letterbox_rect(widget_size: tuple[int, int], frame_shape: tuple,
                   widget_rect: tuple[float, float, float, float]) \
        -> tuple[float, float, float, float] | None:
    """Widget (x, y, w, h) rect → frame-pixel rect (ordered corners,
    clamped to the image). None when the rect is empty or degenerate."""
    x, y, w, h = widget_rect
    if w <= 0 or h <= 0:
        return None
    (x0, y0) = letterbox_map(widget_size, frame_shape, (x, y))
    (x1, y1) = letterbox_map(widget_size, frame_shape, (x + w, y + h))
    fw, fh = frame_shape[1], frame_shape[0]
    fx0 = max(0.0, min(fw, min(x0, x1)))
    fy0 = max(0.0, min(fh, min(y0, y1)))
    fx1 = max(0.0, min(fw, max(x0, x1)))
    fy1 = max(0.0, min(fh, max(y0, y1)))
    if fx1 - fx0 < 1.0 or fy1 - fy0 < 1.0:
        return None
    return (fx0, fy0, fx1 - fx0, fy1 - fy0)


def normalized_roi(roi_px: tuple[float, float, float, float],
                   frame_shape: tuple) -> tuple[float, float, float, float] | None:
    """Pixel (x, y, w, h) → normalized (xn, yn, wn, hn) in 0..1 — the
    INTERSECTION with the frame (a drag beyond the edge contributes only
    its visible part). None when degenerate."""
    x, y, w, h = roi_px
    fw, fh = frame_shape[1], frame_shape[0]
    if w <= 0 or h <= 0 or fw <= 0 or fh <= 0:
        return None
    xn = max(0.0, min(1.0, x / fw))
    yn = max(0.0, min(1.0, y / fh))
    xn2 = max(0.0, min(1.0, (x + w) / fw))
    yn2 = max(0.0, min(1.0, (y + h) / fh))
    wn, hn = xn2 - xn, yn2 - yn
    if wn <= 0.0 or hn <= 0.0:
        return None
    return (xn, yn, wn, hn)


def roi_for_resolution(roi_norm: tuple[float, float, float, float] | None,
                       frame_shape: tuple) -> tuple[int, int, int, int] | None:
    """Normalized ROI → pixel ROI for a frame of the given shape (used
    when the camera resolution changes: the selection follows the same
    scene area).

    The result is guaranteed to be a NON-EMPTY slice of the frame: the old
    version floored w/h at 1 pixel but let x run to the frame edge, so a
    bad stored ROI produced an empty array and cv2 raised inside the metric.
    """
    if roi_norm is None:
        return None
    xn, yn, wn, hn = roi_norm
    fw, fh = frame_shape[1], frame_shape[0]
    if fw <= 0 or fh <= 0:
        return None
    x = max(0, min(int(round(xn * fw)), fw - 1))
    y = max(0, min(int(round(yn * fh)), fh - 1))
    w = max(1, min(int(round(wn * fw)), fw - x))
    h = max(1, min(int(round(hn * fh)), fh - y))
    # NOTE: the crop is guaranteed non-empty (a 1-pixel ROI is still a valid
    # — if unhelpful — region; `is_degenerate` is the separate "too small to
    # score reliably" judgement its callers apply, and it is much stricter:
    # applying it here would silently replace a small user ROI with the
    # whole frame).
    return (x, y, w, h)


def is_degenerate(roi_px: tuple[float, float, float, float],
                  frame_shape: tuple, min_area_fraction: float = 0.02,
                  min_side: int = 8) -> bool:
    """True when the pixel ROI is too small to score reliably — callers
    fall back to the full frame."""
    x, y, w, h = roi_px
    fw, fh = frame_shape[1], frame_shape[0]
    if fw <= 0 or fh <= 0 or w <= 0 or h <= 0:
        return True
    area_fraction = (w * h) / (fw * fh)
    return area_fraction < min_area_fraction or w < min_side or h < min_side
