"""Live-view overlay drawing: the AF-status indicator (top-right), the
scale bar (bottom-right, frame-accurate via the letterbox transform), the
calibrated tick ruler and the display crosshair. Pure QPainter helpers
shared by both workspaces' live views; the color mapping follows the AF
phase contract:

    red   = stage 1 (coarse family: coarse scan/pass, probe)
    orange = stage 2 (fine family: fine sweep, hill climb, lock-on)
    green  = success (fades out)

The phase-name map is shared with the AF panel (moved here so the
indicator and the panel can never drift apart).
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen

from talos.cv.af_roi import fit_transform
from talos.cv.ruler import ruler_spec, tick_label, tick_offsets
from talos.cv.scale_bar import scale_bar_layout
from talos.ui import theme
from talos.ui.theme import DANGER, OK, TEXT, TEXT_DIM, WARN

PHASE_NAMES = {1: "coarse scan", 2: "fine sweep", 3: "landing",
               4: "coarse pass", 5: "hill climb", 6: "lock-on",
               7: "probe"}

# stage-1 (coarse family) phases: 1 coarse scan, 4 coarse pass, 7 probe
_STAGE1 = (1, 4, 7)
# stage-2 (fine family) phases: 2 fine sweep, 5 hill climb, 6 lock-on,
# 3 landing (the last approach)
_STAGE2 = (2, 3, 5, 6)


def af_phase_color(phase: int) -> QColor:
    """Red = stage 1, orange = stage 2, dim otherwise."""
    if phase in _STAGE1:
        return QColor(DANGER)
    if phase in _STAGE2:
        return QColor(WARN)
    return QColor(TEXT_DIM)


def draw_af_indicator(painter: QPainter, label: str, color: QColor,
                      frame_shape: tuple, widget_size: tuple,
                      margin_px: int = 10, alpha: int = 235) -> None:
    """One pill (dot + label, vertically centered on the same centerline)
    anchored to the FRAME's top-right corner — the letterbox transform
    keeps it on the image at any window size."""
    scale, off_x, off_y = fit_transform(widget_size, frame_shape)
    if scale <= 0:
        return
    h, w = frame_shape[:2]
    pill_h = 22.0  # widget px — fixed so it stays readable at any size
    painter.save()
    painter.setOpacity(alpha / 255.0)
    text = f"AF: {label}"
    font = painter.font()
    font.setBold(True)
    painter.setFont(font)
    text_w = painter.fontMetrics().horizontalAdvance(text)
    dot_r = 5.0
    pill_w = 6.0 + dot_r + 6.0 + text_w + 8.0
    pill_right = off_x + w * scale - margin_px
    pill_top = off_y + margin_px
    rect = QRectF(pill_right - pill_w, pill_top, pill_w, pill_h)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(0, 0, 0, 165))
    painter.drawRoundedRect(rect, pill_h / 2, pill_h / 2)
    # dot + text share the pill's vertical centerline
    cy = rect.center().y()
    painter.setPen(QPen(QColor("#000000"), 1))
    painter.setBrush(color)
    dot_x = rect.left() + 6.0 + dot_r
    painter.drawEllipse(QRectF(dot_x - dot_r, cy - dot_r,
                               2 * dot_r, 2 * dot_r))
    painter.setPen(QColor(TEXT))
    painter.drawText(QRectF(dot_x + dot_r + 6.0, rect.top(), text_w,
                            pill_h),
                     Qt.AlignmentFlag.AlignLeft
                     | Qt.AlignmentFlag.AlignVCenter, text)
    painter.restore()


def scale_bar_rect(um_per_px: float, frame_shape: tuple,
                   widget_size: tuple, margin_px: int = 14) \
        -> tuple[QRectF, str]:
    """The scale-bar geometry from the shared spec, mapped through the
    letterbox transform. ACCURACY: every spec dimension (frame pixels)
    is multiplied by the letterbox scale — the drawn bar must span
    exactly `bar_px × scale` widget pixels or its µm label lies."""
    spec = scale_bar_layout(um_per_px, frame_shape, margin_px=margin_px)
    if spec is None:
        return QRectF(), ""
    scale, off_x, off_y = fit_transform(widget_size, frame_shape)
    if scale <= 0:
        return QRectF(), ""
    bx, by, bw, bh = spec.bar_rect
    return (QRectF(off_x + bx * scale, off_y + by * scale,
                   bw * scale, max(2.0, bh * scale)),
            spec.label)


def draw_scale_bar_q(painter: QPainter, um_per_px: float,
                     frame_shape: tuple, widget_size: tuple) -> None:
    """Draw the scale bar from the shared spec: translucent box, white
    bar, label UNDER the bar (no overlap), equal right/bottom margins —
    identical geometry to the snapshot burn."""
    spec = scale_bar_layout(um_per_px, frame_shape)
    if spec is None:
        return
    scale, off_x, off_y = fit_transform(widget_size, frame_shape)
    if scale <= 0:
        return
    x1, y1, x2, y2 = spec.box

    def _map(rect) -> QRectF:
        rx, ry, rw, rh = rect
        return QRectF(off_x + rx * scale, off_y + ry * scale,
                      rw * scale, max(2.0, rh * scale))

    painter.save()
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(0, 0, 0, 175))
    box_x1, box_y1, box_x2, box_y2 = spec.box  # corner coords, not x/y/w/h
    painter.drawRoundedRect(
        QRectF(off_x + box_x1 * scale, off_y + box_y1 * scale,
               (box_x2 - box_x1) * scale, (box_y2 - box_y1) * scale), 3, 3)
    painter.setBrush(QColor(TEXT))
    painter.drawRect(_map(spec.bar_rect))
    font = painter.font()
    # Hershey cap height ≈ 22×font_scale — the Qt band matches the cv2
    # band so both renderers show the same box proportions.
    font.setPixelSize(max(8, int(round(spec.font_scale * 22 * scale))))
    painter.setFont(font)
    painter.setPen(QColor(TEXT))
    painter.drawText(_map(spec.text_rect),
                     Qt.AlignmentFlag.AlignHCenter
                     | Qt.AlignmentFlag.AlignVCenter, spec.label)
    painter.restore()


def draw_ruler_q(painter: QPainter, um_per_px: float, frame_shape: tuple,
                 size: tuple, *, tick_minor_px: int = 4,
                 tick_major_px: int = 9, label_min_px: int = 54) -> None:
    """Calibrated ticks along all four edges of the FRAME.

    Drawn in the painter's coordinates over the frame rect — the live view
    calls this on the frame PIXMAP (where the frame fills the pixmap, so
    frame pixels scale by ``size / frame``), which is also what makes the
    inverse-video effect possible: the ticks invert the image underneath
    instead of relying on a colour that may match it.

    Major ticks are longer and labelled with their µm offset from the
    frame centre (0 = the optical axis, where the crosshair sits); the
    labels are dropped when the majors get too close together to read.
    """
    spec = ruler_spec(um_per_px, frame_shape)
    if spec is None:
        return
    w, h = int(size[0]), int(size[1])
    fh, fw = int(frame_shape[0]), int(frame_shape[1])
    if fw <= 0 or fh <= 0 or w <= 0 or h <= 0:
        return
    scale = w / float(fw)
    majors_x, minors_x = tick_offsets(spec, um_per_px, fw)
    majors_y, minors_y = tick_offsets(spec, um_per_px, fh)
    cx, cy = w / 2.0, h / 2.0

    painter.save()
    font = painter.font()
    font.setPixelSize(10)
    painter.setFont(font)
    show_labels = spec.major_um / um_per_px * scale >= label_min_px

    def _ticks(offsets: list[float], centre: float, length: int,
               horizontal: bool, from_start: bool) -> None:
        painter.setPen(QPen(painter.pen().color(), 1))
        for offset in offsets:
            pos = int(round(centre + offset * scale))
            if pos < 0 or pos >= (w if horizontal else h):
                continue
            if horizontal:
                painter.drawLine(pos, 0 if from_start else h,
                                 pos, length if from_start else h - length)
            else:
                painter.drawLine(0 if from_start else w, pos,
                                 length if from_start else w - length, pos)

    _ticks(minors_x, cx, tick_minor_px, True, True)
    _ticks(minors_x, cx, tick_minor_px, True, False)
    _ticks(minors_y, cy, tick_minor_px, False, True)
    _ticks(minors_y, cy, tick_minor_px, False, False)
    _ticks(majors_x, cx, tick_major_px, True, True)
    _ticks(majors_x, cx, tick_major_px, True, False)
    _ticks(majors_y, cy, tick_major_px, False, True)
    _ticks(majors_y, cy, tick_major_px, False, False)

    if not show_labels:
        painter.restore()
        return
    # Labels: along the TOP edge (X offsets) and the LEFT edge (Y offsets).
    for offset in majors_x:
        pos = int(round(cx + offset * scale))
        if pos < 4 or pos > w - 4:
            continue
        text = tick_label(offset * um_per_px)
        width = painter.fontMetrics().horizontalAdvance(text)
        painter.drawText(QRectF(pos - width / 2.0 - 2, tick_major_px + 1,
                                width + 4, 12),
                         Qt.AlignmentFlag.AlignHCenter
                         | Qt.AlignmentFlag.AlignVCenter, text)
    for offset in majors_y:
        pos = int(round(cy + offset * scale))
        if pos < 8 or pos > h - 8:
            continue
        painter.drawText(QRectF(tick_major_px + 3, pos - 6, 60, 12),
                         Qt.AlignmentFlag.AlignLeft
                         | Qt.AlignmentFlag.AlignVCenter,
                         tick_label(offset * um_per_px))
    painter.restore()


# NOTE: the DISPLAY crosshair is not drawn here — it is painted into the
# frame pixmap with an inverse-video composition (LiveViewWidget._compose):
# the overlay surface is a translucent child repainted on every frame, so
# a Difference-mode pen there would blend against its own previous output
# instead of the image. The ROI-arming crosshair (a mode affordance that
# must show before the first frame) still lives in live_view.
