"""SharpnessCurveWidget: live plot of the autofocus sweep — score vs
position (µm), with the peak marker and an optional second series (the
low-frequency metric curve, adaptive v2). QPainter only, dark theme."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from talos.ui.theme import TEXT_DIM

_LINE = QPen(QColor(80, 180, 255), 2)       # sharp metric (main series)
_LINE2 = QPen(QColor(210, 153, 34), 2)      # low-freq metric (secondary)
_PEAK = QPen(QColor(255, 200, 60), 2)
_GRID = QPen(QColor(70, 74, 84), 1, Qt.PenStyle.DashLine)


class SharpnessCurveWidget(QWidget):
    """Score vs position; points arrive from sig_af_progress (main) and
    sig_af_curve_secondary (low-freq). Position is converted steps→µm by
    set_um_per_step before plotting. The two series share the position
    axis but each normalizes to its OWN score range — the low-frequency
    metric's magnitudes differ from tenengrad by orders of magnitude."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(96)
        self._points: list[tuple[float, float]] = []
        self._points2: list[tuple[float, float]] = []
        self._peak_pos: float | None = None
        self._um_per_step = 0.2

    def set_um_per_step(self, um_per_step: float) -> None:
        self._um_per_step = um_per_step

    @property
    def um_per_step(self) -> float:
        return self._um_per_step

    def clear(self) -> None:
        self._points = []
        self._points2 = []
        self._peak_pos = None
        self.update()

    def add_point(self, pos_steps: float, score: float) -> None:
        self._points.append((pos_steps * self._um_per_step, score))
        self.update()

    def add_secondary(self, pos_steps: float, score: float) -> None:
        self._points2.append((pos_steps * self._um_per_step, score))
        self.update()

    def set_peak(self, pos_steps: int | None) -> None:
        self._peak_pos = (pos_steps * self._um_per_step
                          if pos_steps is not None else None)
        self.update()

    def _paint_series(self, painter, points, x_map, h, margin, pen):
        if len(points) < 2:
            return
        ys = [s for _, s in points]
        y_min, y_max = min(ys), max(ys)
        y_span = (y_max - y_min) or 1.0

        def py(y):
            return h - margin - (y - y_min) / y_span * (h - 2 * margin)

        painter.setPen(pen)
        for i, (x, y) in enumerate(points[1:], start=1):
            x0, y0 = points[i - 1]
            painter.drawLine(int(x_map(x0)), int(py(y0)),
                             int(x_map(x)), int(py(y)))

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        w, h = self.width(), self.height()
        margin = 8
        painter.fillRect(self.rect(), QColor(24, 26, 32))
        painter.setPen(_GRID)
        painter.drawRect(margin, margin, w - 2 * margin, h - 2 * margin)
        all_points = self._points + self._points2
        if len(all_points) >= 2:
            xs = [p for p, _ in all_points]
            x_min, x_max = min(xs), max(xs)
            x_span = (x_max - x_min) or 1.0

            def px(x):
                return margin + (x - x_min) / x_span * (w - 2 * margin)

            self._paint_series(painter, self._points, px, h, margin, _LINE)
            self._paint_series(painter, self._points2, px, h, margin, _LINE2)
            if self._peak_pos is not None:
                painter.setPen(_PEAK)
                painter.drawLine(int(px(self._peak_pos)), margin,
                                 int(px(self._peak_pos)), h - margin)
            # axis labels + legend (top-left, one line per series present)
            painter.setPen(QColor(TEXT_DIM))
            painter.drawText(margin, h - 1,
                             f"{x_min:.0f} µm  —  {x_max:.0f} µm")
            legend_y = margin + 12
            if self._points:
                painter.setPen(_LINE)
                painter.drawText(margin + 2, legend_y, "sharp")
                legend_y += 12
            if self._points2:
                painter.setPen(_LINE2)
                painter.drawText(margin + 2, legend_y, "low-freq")
        else:
            painter.setPen(QColor(TEXT_DIM))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "focus curve")
        painter.end()
