"""Live camera view: displays the newest frame, drops intermediate ones.

M4: ROI selection (rubber band) + crosshair overlay. The widget→frame
mapping inverts the KeepAspectRatio letterbox transform of the render
path (scale-first, centered) — see talos.cv.af_roi.

Overlays (Display menu): scale bar (bottom-right, frame-anchored),
crosshair, AF-status indicator (top-right).
"""

from __future__ import annotations

from dataclasses import dataclass

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
    draw_ruler_q,
    draw_scale_bar_q,
    draw_scan_plan_q,
)


@dataclass(frozen=True)
class ScanPlanOverlay:
    """The grid-scan path shown near the top of the live view.

    A schematic (one arrow per row, serpentine order) — see
    draw_scan_plan_q for why it is not registered to the image.
    """

    cols: int
    rows: int
    detail: str = ""              # "8 × 5 grid · 700 × 390 µm"
    active_row: int = -1          # -1 = preview, nothing running
    active_col: int = -1
    serpentine: bool = True

    def status_text(self) -> str:
        if self.active_row < 0:
            return f"{self.cols * self.rows} waypoints · ready"
        forward = not (self.serpentine and self.active_row % 2 == 1)
        arrow = "→" if forward else "←"
        col = self.active_col if self.active_col >= 0 else 0
        return (f"Row {self.active_row + 1}/{self.rows} "
                f"· col {col + 1}/{self.cols} {arrow}")

# The AF region is drawn in the CROSSHAIR's style (thin dashed cyan) and
# completely unfilled, so it never hides the image it is measuring; the
# "AF ROI" tag next to it says what the outline means.
_CROSSHAIR_PEN = QPen(QColor(0, 200, 255, 120), 1, Qt.PenStyle.DashLine)
_ROI_PEN = _CROSSHAIR_PEN
_ROI_LABEL = "AF ROI"

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
        # The CLEAN frame pixmap (no baked overlays) — _compose() copies it
        # and paints the inverse-video overlays on the copy.
        self._frame_pixmap: QPixmap | None = None
        # ROI state (normalized to the frame, so resolution changes keep it)
        self._roi_norm: tuple | None = None
        self._selecting = False
        self._drag_start = None
        self._drag_rect: QRectF | None = None
        self._crosshair = False
        # Display-menu overlays
        self._scale_bar_enabled = False
        self._ruler_enabled = False
        self._um_per_px: float | None = None
        self._crosshair_display = False
        self._af_indicator_enabled = False
        self._scan_plan: ScanPlanOverlay | None = None
        self._scan_path_enabled = True
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

    def set_live_calibration(self, um_per_px: float | None) -> None:
        """µm/px for the scale bar AND the tick ruler; None hides both
        (uncalibrated)."""
        self._um_per_px = um_per_px
        self._overlay.update()
        self._refresh_pixmap()   # the ruler lives on the frame pixmap

    def set_crosshair_enabled(self, on: bool) -> None:
        """The Display-menu crosshair: solid, inverse-video, baked into the
        frame pixmap (separate from the ROI-selection crosshair, which is a
        dashed cyan affordance on the overlay)."""
        self._crosshair_display = bool(on)
        # Repaint immediately: the crosshair must appear/disappear even
        # while the live view is paused (no new frame to trigger it).
        self._refresh_pixmap()
        self._overlay.update()

    def set_ruler_enabled(self, on: bool) -> None:
        """The Display-menu tick ruler (calibrated, all four edges)."""
        self._ruler_enabled = bool(on)
        self._refresh_pixmap()
        self._overlay.update()

    def set_af_indicator_enabled(self, on: bool) -> None:
        self._af_indicator_enabled = bool(on)
        self._overlay.update()

    def set_scan_path_enabled(self, on: bool) -> None:
        """Display-menu toggle for the grid-scan path indicator."""
        self._scan_path_enabled = bool(on)
        self._overlay.update()

    def set_scan_plan(self, plan: ScanPlanOverlay | None) -> None:
        """Show (or clear) the scan-path indicator. ``None`` hides it."""
        self._scan_plan = plan
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

    def _live_um_per_px(self) -> float | None:
        """Calibration in LIVE-frame pixels. The stored value is canonical
        per 4K-sensor pixel; a 1080p frame covers 2× the µm per pixel."""
        if self._um_per_px is None or self._last_shape is None:
            return None
        return self._um_per_px * (SENSOR_WIDTH_PX / self._last_shape[1])

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
        self._frame_pixmap = QPixmap.fromImage(scaled)
        self._label.setPixmap(self._compose())

    def _refresh_pixmap(self) -> None:
        """Re-bake the baked overlays into the label's pixmap. No-op before
        the first frame — a null pixmap would wipe the "No camera" text."""
        if self._frame_pixmap is not None:
            self._label.setPixmap(self._compose())

    def _compose(self) -> QPixmap:
        """The frame plus the INVERSE-VIDEO overlays (crosshair, ruler).

        Painted on the frame pixmap, not on the overlay surface: that
        surface is a translucent child repainted with every frame, so a
        Difference-mode pen there would blend against its own previous
        output (flicker) and against an unspecified destination. On the
        pixmap, Difference-against-white IS the Minecraft-style inversion
        of the image underneath — and the lines are clipped to the frame
        instead of running across the letterbox bars.
        """
        base = self._frame_pixmap
        if base is None:
            return QPixmap()
        crosshair = self._crosshair_display and not self._selecting
        ruler = (self._ruler_enabled and self._last_shape is not None
                 and self._live_um_per_px())
        if not crosshair and not ruler:
            return base
        pixmap = QPixmap(base)          # detach: never paint the cache
        painter = QPainter(pixmap)
        painter.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_Difference)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        if crosshair:
            cx, cy = pixmap.width() // 2, pixmap.height() // 2
            painter.drawLine(cx, 0, cx, pixmap.height() - 1)
            painter.drawLine(0, cy, pixmap.width() - 1, cy)
        if ruler:
            draw_ruler_q(painter, self._live_um_per_px(), self._last_shape,
                         (pixmap.width(), pixmap.height()))
        painter.end()
        return pixmap

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
        if self._crosshair:
            # The ROI-ARMING crosshair only: a mode affordance, drawn on
            # the overlay so it is visible before the first frame arrives.
            # The Display-menu crosshair is inverse-video on the pixmap.
            painter.setPen(_CROSSHAIR_PEN)
            cx, cy = self.width() / 2, self.height() / 2
            painter.drawLine(int(cx), 0, int(cx), self.height())
            painter.drawLine(0, int(cy), self.width(), int(cy))
        if self._roi_norm is not None:
            self._draw_roi(painter, self._roi_rect())
        if self._drag_rect is not None:
            painter.setPen(_ROI_PEN)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self._drag_rect)
        if self._scan_path_enabled and self._scan_plan is not None \
                and self._last_shape is not None:
            draw_scan_plan_q(painter, self._scan_plan, self._last_shape,
                             (self.width(), self.height()))
        live_um_per_px = self._live_um_per_px()
        if self._scale_bar_enabled and live_um_per_px is not None:
            # the length choice lives in the shared spec now (same ladder
            # as the snapshot burn); the ×letterbox-scale mapping inside
            # keeps the drawn bar accurate at ANY window size.
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

    def _draw_roi(self, painter: QPainter, rect: QRectF) -> None:
        """Unfilled dashed outline + a small "AF ROI" tag (the tag is drawn
        with a dark shadow so it stays legible on bright images — no filled
        box, which would colour the region it is labelling)."""
        if rect.isEmpty():
            return
        painter.save()
        painter.setPen(_ROI_PEN)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)
        font = painter.font()
        font.setPixelSize(11)
        painter.setFont(font)
        text_rect = QRectF(rect.left() + 4, max(0.0, rect.top() - 15),
                           max(80.0, rect.width() - 8), 14)
        for offset, colour in ((1, QColor(0, 0, 0, 160)),
                               (0, QColor(0, 200, 255, 220))):
            painter.setPen(QPen(colour))
            painter.drawText(text_rect.translated(offset, offset),
                             Qt.AlignmentFlag.AlignLeft
                             | Qt.AlignmentFlag.AlignVCenter, _ROI_LABEL)
        painter.restore()

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
