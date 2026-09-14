"""Live camera view: displays the newest frame, drops intermediate ones.

M4: ROI selection (rubber band) + crosshair overlay. The widget→frame
mapping inverts the KeepAspectRatio letterbox transform of the render
path (scale-first, centered) — see talos.cv.af_roi.

Overlays (Display menu): scale bar (bottom-right, frame-anchored),
crosshair, AF-status indicator (top-right).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from talos.cv.af_roi import letterbox_rect, normalized_roi
from talos.cv.calibration import SENSOR_WIDTH_PX
from talos.ui.theme import OK
from talos.ui.widgets.overlay import (
    af_phase_color,
    draw_af_indicator,
    draw_crosshair,
    draw_scale_bar_q,
)

_ROI_COLOR = QColor(0, 200, 255, 160)
_ROI_PEN = QPen(QColor(0, 200, 255, 220), 2, Qt.PenStyle.DashLine)

_AF_FADE_MS = 3000


class _OverlaySurface(QWidget):
    """Transparent child stacked ABOVE the frame label. Qt paints
    children after the parent, so overlays drawn in the parent's
    paintEvent would sit UNDER the frame pixmap — this surface flips
    the layering. Mouse-transparent: events keep flowing to the parent
    (the ROI rubber band lives there)."""

    def __init__(self, owner: "LiveViewWidget"):
        super().__init__(owner)
        self._owner = owner
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        self._owner._draw_overlays(painter,
                                   QRectF(0.0, 0.0, float(self.width()),
                                          float(self.height())))
        painter.end()


class LiveViewWidget(QWidget):
    sig_roi_selected = Signal(object)   # normalized (x, y, w, h) | None

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._label = QLabel("No camera", self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setObjectName("liveview")
        self._label.setMinimumSize(480, 320)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)
        # The overlay is NOT layout-managed: it is stacked on top of the
        # label (the last child paints above it) and tracks the widget
        # size in resizeEvent.
        self._overlay = _OverlaySurface(self)
        self._overlay.setGeometry(self.rect())
        self._overlay.raise_()
        self._pending: np.ndarray | None = None
        self._last_shape: tuple | None = None
        # ROI state (normalized to the frame, so resolution changes keep it)
        self._roi_norm: tuple | None = None
        self._selecting = False
        self._drag_start = None
        self._drag_rect: QRectF | None = None
        self._crosshair = False
        # Display-menu overlays
        self._scale_bar_enabled = False
        self._um_per_px: float | None = None
        self._crosshair_display = False
        self._af_indicator_enabled = False
        self._af_phase: int | None = None
        self._af_label = ""
        self._af_success = False
        self._af_fade = QTimer(self)
        self._af_fade.setSingleShot(True)
        self._af_fade.timeout.connect(self._clear_af_indicator)

    # ------------------------------------------------------------------

    def show_frame(self, frame: np.ndarray) -> None:
        """Queue the newest frame; actual paint happens in paintEvent."""
        self._pending = frame
        self._last_shape = frame.shape
        self.update()
        self._overlay.update()  # re-draw the overlays over the new frame

    # --- Display overlays ------------------------------------------------

    def set_scale_bar_enabled(self, on: bool) -> None:
        self._scale_bar_enabled = bool(on)
        self._overlay.update()

    def set_scale_bar_calibration(self, um_per_px: float | None) -> None:
        """µm/px for the scale bar; None hides it (uncalibrated)."""
        self._um_per_px = um_per_px
        self._overlay.update()

    def set_crosshair_enabled(self, on: bool) -> None:
        """The Display-menu crosshair (separate from the ROI-selection
        crosshair shown while arming a rubber band)."""
        self._crosshair_display = bool(on)
        self._overlay.update()

    def set_af_indicator_enabled(self, on: bool) -> None:
        self._af_indicator_enabled = bool(on)
        self._overlay.update()

    def set_af_phase(self, phase: int, label: str) -> None:
        """AF running: show the phase color + name (red stage-1, orange
        stage-2)."""
        self._af_fade.stop()
        self._af_phase = int(phase)
        self._af_label = label
        self._af_success = False
        self._overlay.update()

    def set_af_success(self, success: bool, aborted: bool) -> None:
        """AF finished: green pulse on success (fades), cleared otherwise."""
        if not success or aborted:
            self._clear_af_indicator()
            return
        self._af_fade.stop()
        self._af_phase = None
        self._af_label = "focused"
        self._af_success = True
        self._overlay.update()
        self._af_fade.start(_AF_FADE_MS)

    def _clear_af_indicator(self) -> None:
        self._af_phase = None
        self._af_label = ""
        self._af_success = False
        self._overlay.update()

    # --- ROI ------------------------------------------------------------

    def set_roi_selection_mode(self, on: bool) -> None:
        """Crosshair cursor + rubber-band selection. The selection is
        emitted NORMALIZED to the current frame."""
        self._selecting = on
        self._crosshair = on
        self._drag_start = None
        self._drag_rect = None
        self.setCursor(Qt.CursorShape.CrossCursor if on
                       else Qt.CursorShape.ArrowCursor)
        self._overlay.update()

    def set_roi(self, roi_norm: tuple | None) -> None:
        self._roi_norm = roi_norm
        self._overlay.update()

    def clear_roi(self) -> None:
        self._roi_norm = None
        self._overlay.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._selecting and event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position()
            self._drag_rect = QRectF(self._drag_start, self._drag_start)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._selecting and self._drag_start is not None:
            self._drag_rect = QRectF(self._drag_start,
                                     event.position()).normalized()
            self._overlay.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if not (self._selecting and self._drag_start is not None):
            return
        rect = self._drag_rect
        self._drag_start = None
        self._drag_rect = None
        self._selecting = False
        self._crosshair = False
        self.setCursor(Qt.CursorShape.ArrowCursor)
        # _last_shape, NOT _pending: the pending frame is consumed by the
        # next paint (which runs before every overlay paint), so reading it
        # here silently dropped the selection whenever the renderer had
        # just run.
        shape = self._last_shape
        if rect is None or shape is None:
            return
        size = (self.width(), self.height())
        pixel_rect = letterbox_rect(
            size, shape, (rect.x(), rect.y(), rect.width(), rect.height()))
        norm = normalized_roi(pixel_rect, shape) if pixel_rect else None
        self._roi_norm = norm
        self.sig_roi_selected.emit(norm)
        self._overlay.update()

    # ------------------------------------------------------------------

    def _render_pending(self) -> None:
        if self._pending is None:
            return
        frame = self._pending
        self._pending = None
        h, w = frame.shape[:2]
        # Scale first, convert second: converting the full 1080p frame to a
        # pixmap (~8 MB ARGB32) before scaling is pure waste. NOTE: the
        # QImage wraps frame's memory — safe only because QPixmap.fromImage
        # consumes it synchronously inside this scope.
        image = QImage(frame.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        scaled = image.scaled(self._label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
        self._label.setPixmap(QPixmap.fromImage(scaled))

    def paintEvent(self, event) -> None:  # noqa: N802
        self._render_pending()
        super().paintEvent(event)
        # All overlays live on the _OverlaySurface child (stacked ABOVE
        # the frame label — parent-painted content would sit UNDER the
        # frame pixmap).

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._overlay.setGeometry(self.rect())

    def _draw_overlays(self, painter: QPainter, rect: QRectF) -> None:
        """Called from the overlay surface's paintEvent (topmost layer)."""
        if self._crosshair or self._crosshair_display:
            painter.setPen(QPen(QColor(0, 200, 255, 120), 1,
                                Qt.PenStyle.DashLine))
            cx, cy = self.width() / 2, self.height() / 2
            painter.drawLine(int(cx), 0, int(cx), self.height())
            painter.drawLine(0, int(cy), self.width(), int(cy))
        if self._roi_norm is not None:
            painter.setPen(_ROI_PEN)
            painter.setBrush(_ROI_COLOR)
            painter.drawRect(self._roi_rect())
        if self._drag_rect is not None:
            painter.setPen(_ROI_PEN)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self._drag_rect)
        if self._scale_bar_enabled and self._um_per_px \
                and self._last_shape is not None:
            # the length choice lives in the shared spec now (same ladder
            # as the snapshot burn); the ×letterbox-scale mapping inside
            # keeps the drawn bar accurate at ANY window size. The
            # calibration is canonical per 4K-sensor pixel — a 1080p
            # live frame covers 2× the µm per pixel.
            live_um_per_px = self._um_per_px * (
                SENSOR_WIDTH_PX / self._last_shape[1])
            draw_scale_bar_q(painter, live_um_per_px, self._last_shape,
                             (self.width(), self.height()))
        if self._af_indicator_enabled and self._last_shape is not None:
            # frame-anchored: skipped until the first frame arrives (the
            # letterbox math needs the frame shape)
            if self._af_success:
                draw_af_indicator(painter, self._af_label, QColor(OK),
                                  self._last_shape,
                                  (self.width(), self.height()))
            elif self._af_phase is not None:
                draw_af_indicator(painter, self._af_label,
                                  af_phase_color(self._af_phase),
                                  self._last_shape,
                                  (self.width(), self.height()))

    def _roi_rect(self) -> QRectF:
        """Normalized ROI → widget coordinates (inverse letterbox).

        Uses the LAST frame shape rather than the pending frame: the
        overlay paints after the parent's paintEvent consumed the pending
        frame, so this returned an empty rect on every streamed frame and
        the ROI rubber band flickered at frame rate.
        """
        from talos.cv.af_roi import roi_for_resolution

        shape = self._last_shape
        if self._roi_norm is None or shape is None:
            return QRectF()
        pixel_rect = roi_for_resolution(self._roi_norm, shape)
        if pixel_rect is None:
            return QRectF()
        x, y, w, h = pixel_rect
        fh, fw = shape[0], shape[1]
        widget_w, widget_h = self.width(), self.height()
        scale = min(widget_w / fw, widget_h / fh) if fw and fh else 1.0
        off_x = (widget_w - fw * scale) / 2
        off_y = (widget_h - fh * scale) / 2
        return QRectF(off_x + x * scale, off_y + y * scale,
                      w * scale, h * scale)
