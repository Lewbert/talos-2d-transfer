"""ScanMapWidget: the scan drawn in the SAMPLE's frame.

The stage carries the sample, so a scan does not move a camera across a
fixed scene — it drags the sample past a fixed objective. Drawn in stage
coordinates, though, the picture the operator wants falls out for free:
the tiles sit where the stage read back, and the camera's footprint walks
across them. Sample stable, camera scanning.

**Tiles are placed as the operator saw them.** Each tile is a frame from
the same source the live view uses — the backend's own egress, camera flip
already applied — so it is laid down at its readback position with NO
content rotation. Rotating a tile to "correct" it would put its content
180° from its neighbour's at every overlap, which is exactly what a
glitched mosaic looks like: the same features twice, offset the wrong way.
The map therefore follows the live view's orientation for free, whatever
the camera flip and the axis settings are set to.

The tile POSITIONS are the manifest readback, so no stage↔image sign
convention is assumed anywhere here either.

Wheel zooms, drag pans, Fit resets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from talos.cv.orientation import axis_signs, mosaic_offset
from talos.ui import theme

#: Cap on the QImage conversions kept alive for painting. A 200-tile scan
#: holds 200 small thumbnails; the oldest are dropped, not the newest.
_MAX_TILES = 600


@dataclass
class ScanMapTile:
    index: int
    x_um: float
    y_um: float
    thumb: np.ndarray | None = None      # RGB uint8


@dataclass
class ScanMapMarker:
    x_um: float
    y_um: float
    label: str = ""
    tile: int = -1


@dataclass
class ScanMapPlan:
    """What the map draws when a scan is planned or running."""

    x0_um: float = 0.0
    y0_um: float = 0.0
    width_um: float = 0.0
    height_um: float = 0.0
    x_dir: int = 1
    y_dir: int = 1
    fov_x_um: float = 0.0
    fov_y_um: float = 0.0
    waypoints: list = field(default_factory=list)   # [(x_um, y_um), ...]

    def key(self) -> tuple:
        """What makes two plans the SAME plan. The map keeps the captured
        tiles when the key is unchanged and drops them when it is not:
        re-stating a plan (a refresh, a finished run) must not erase what
        the run already captured."""
        return (round(self.x0_um, 6), round(self.y0_um, 6),
                round(self.width_um, 6), round(self.height_um, 6),
                int(self.x_dir), int(self.y_dir),
                round(self.fov_x_um, 6), round(self.fov_y_um, 6),
                tuple((round(x, 6), round(y, 6)) for x, y in self.waypoints))

    def oriented(self, flip: bool) -> "ScanMapPlan":
        """The same plan drawn in the frames' coordinates.

        The map is laid out in the sample frame AS THE FRAMES SHOW IT, so
        every stage coordinate — and every direction — is scaled by the
        axis signs in cv/orientation.py: the same rule the mosaic uses, and
        the reason a scan's tiles line up instead of appearing twice, with
        the map pointing the same way as the live view beside it.
        """
        sx, sy = axis_signs(flip)
        return ScanMapPlan(
            x0_um=sx * self.x0_um, y0_um=sy * self.y0_um,
            width_um=self.width_um, height_um=self.height_um,
            x_dir=sx * self.x_dir, y_dir=sy * self.y_dir,
            fov_x_um=self.fov_x_um, fov_y_um=self.fov_y_um,
            waypoints=[(x * sx, y * sy) for x, y in self.waypoints])


def plan_bounds(plan: ScanMapPlan) -> tuple[float, float, float, float]:
    """(x_min, y_min, x_max, y_max) in µm, padded by half a FOV so the
    footprints at the corners fit inside the view."""
    if plan is None or plan.width_um <= 0 or plan.height_um <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    x0, x1 = sorted((plan.x0_um, plan.x0_um + plan.x_dir * plan.width_um))
    y0, y1 = sorted((plan.y0_um, plan.y0_um + plan.y_dir * plan.height_um))
    pad_x, pad_y = plan.fov_x_um / 2.0, plan.fov_y_um / 2.0
    return (x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y)


def fit_view(bounds: tuple[float, float, float, float],
             size: tuple[int, int], margin_px: int = 12) -> tuple[float, QPointF]:
    """(scale px/µm, offset) that fits ``bounds`` into a widget.

    Y grows DOWNWARD, matching the mosaic and an image: the map and the
    stitched overview then show the same thing the same way up.
    """
    x0, y0, x1, y1 = bounds
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    avail_w = max(1, size[0] - 2 * margin_px)
    avail_h = max(1, size[1] - 2 * margin_px)
    scale = min(avail_w / w, avail_h / h)
    off_x = margin_px + (avail_w - w * scale) / 2.0 - x0 * scale
    off_y = margin_px + (avail_h - h * scale) / 2.0 - y0 * scale
    return scale, QPointF(off_x, off_y)


class ScanMapWidget(QWidget):
    sig_marker_selected = Signal(int)      # index into markers

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.setMouseTracking(True)
        self.setObjectName("scanmap")
        self._plan = ScanMapPlan()
        self._tiles: dict[int, ScanMapTile] = {}
        self._images: dict[int, QImage] = {}
        self._markers: list[ScanMapMarker] = []
        self._footprint: tuple[float, float] | None = None
        self._flip = False
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._drag_from = None
        self._selected = -1
        self._caption = ""

    # --- data in ------------------------------------------------------

    def set_flip(self, flip: bool) -> None:
        """Follow the camera flip. The map is drawn in the sample frame as
        the FRAMES show it — the layout mirrors with the flip, exactly as
        the mosaic does, so a flipped scan's tiles line up instead of
        landing twice (cv/orientation.py)."""
        flip = bool(flip)
        if flip == self._flip:
            return
        self._flip = flip
        self.update()

    def _effective_plan(self) -> ScanMapPlan:
        """The plan in the coordinates this map draws in."""
        return self._plan.oriented(self._flip)

    def _at(self, x_um: float, y_um: float) -> tuple[float, float]:
        """A stage coordinate in the coordinates this map draws in — the
        SAME transform the mosaic uses, so the two cannot disagree about
        which way an axis runs."""
        return mosaic_offset(x_um, y_um, self._flip)

    def set_plan(self, plan: ScanMapPlan) -> None:
        """A NEW plan: the tiles and markers belong to the old one and are
        dropped. Re-stating the same plan changes nothing (see key())."""
        plan = plan or ScanMapPlan()
        if plan.key() == self._plan.key():
            self._plan = plan
            self.update()
            return
        self._plan = plan
        self._tiles.clear()
        self._images.clear()
        self._markers.clear()
        self._selected = -1
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    def add_tile(self, tile: ScanMapTile) -> None:
        self._tiles[tile.index] = tile
        if tile.thumb is not None:
            self._images[tile.index] = self._qimage(tile.thumb)
        if len(self._tiles) > _MAX_TILES:
            for key in sorted(self._images)[:len(self._images) - _MAX_TILES]:
                self._images.pop(key, None)
        self.update()

    def clear_tiles(self) -> None:
        """Drop the captured tiles and markers but KEEP the plan — what a
        new run of the same area needs."""
        self._tiles.clear()
        self._images.clear()
        self._markers.clear()
        self._selected = -1
        self.update()

    def set_markers(self, markers: list[ScanMapMarker]) -> None:
        self._markers = list(markers)
        self.update()

    def set_footprint(self, x_um: float, y_um: float) -> None:
        """Where the camera is looking right now (stage readback)."""
        self._footprint = (float(x_um), float(y_um))
        self.update()

    def set_caption(self, text: str) -> None:
        self._caption = str(text)
        self.update()

    def fit(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    @property
    def marker_count(self) -> int:
        return len(self._markers)

    # --- interaction --------------------------------------------------

    def wheelEvent(self, event) -> None:  # noqa: N802
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        factor = 1.25 ** steps
        self._zoom = max(0.2, min(12.0, self._zoom * factor))
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            hit = self._marker_at(event.position())
            if hit >= 0:
                self._selected = hit
                self.sig_marker_selected.emit(hit)
                self.update()
                return
            self._drag_from = event.position()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_from is not None:
            delta = event.position() - self._drag_from
            self._pan += delta
            self._drag_from = event.position()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_from = None

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.fit()

    # --- painting -----------------------------------------------------

    def _transform(self) -> tuple[float, QPointF]:
        scale, offset = fit_view(plan_bounds(self._effective_plan()),
                                 (self.width(), self.height()))
        base = QPointF(self.width() / 2.0, self.height() / 2.0)
        offset = base + (offset - base) * self._zoom + self._pan
        return scale * self._zoom, offset

    def _to_widget(self, x_um: float, y_um: float) -> QPointF:
        scale, offset = self._transform()
        return QPointF(offset.x() + x_um * scale, offset.y() + y_um * scale)

    def _marker_at(self, pos) -> int:
        for index, marker in enumerate(self._markers):
            point = self._to_widget(*self._at(marker.x_um, marker.y_um))
            if (point - pos).manhattanLength() <= 12:
                return index
        return -1

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.PANEL))
        plan = self._effective_plan()
        if plan.width_um <= 0 or plan.height_um <= 0:
            painter.setPen(QColor(theme.TEXT_DIM))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No scan planned — set an area and use "
                             "\"Scan from here\"")
            painter.end()
            return

        scale, _offset = self._transform()
        self._draw_area(painter)
        self._draw_tiles(painter, scale)
        self._draw_path(painter)
        self._draw_footprint(painter, scale)
        self._draw_markers(painter)
        self._draw_caption(painter)
        painter.end()

    # --- pieces -------------------------------------------------------

    def _draw_area(self, painter: QPainter) -> None:
        plan = self._effective_plan()
        # the rectangle grows from the start point along the direction
        # signs, so its corners are NOT necessarily (x0, y0) → (+w, +h)
        corner = self._to_widget(plan.x0_um, plan.y0_um)
        opposite = self._to_widget(plan.x0_um + plan.x_dir * plan.width_um,
                                   plan.y0_um + plan.y_dir * plan.height_um)
        rect = QRectF(
            QPointF(min(corner.x(), opposite.x()),
                    min(corner.y(), opposite.y())),
            QPointF(max(corner.x(), opposite.x()),
                    max(corner.y(), opposite.y())))
        painter.setPen(QPen(QColor(theme.TEXT_DIM), 1, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)
        # the start corner, where the operator was standing
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.ACCENT))
        painter.drawEllipse(self._to_widget(plan.x0_um, plan.y0_um), 3.0, 3.0)

    def _draw_tiles(self, painter: QPainter, scale: float) -> None:
        plan = self._effective_plan()
        for tile in self._tiles.values():
            rect = QRectF(0.0, 0.0, plan.fov_x_um * scale,
                          plan.fov_y_um * scale)
            centre = self._to_widget(*self._at(tile.x_um, tile.y_um))
            rect.moveCenter(centre)
            image = self._images.get(tile.index)
            if image is None:
                painter.setPen(QPen(QColor(theme.TEXT_DIM), 1))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(rect)
                continue
            # as captured, as displayed in the live view — see the module
            # docstring for why this is deliberately unrotated
            painter.drawImage(rect, image)

    def _draw_path(self, painter: QPainter) -> None:
        waypoints = self._effective_plan().waypoints
        if len(waypoints) < 2:
            return
        pen = QPen(QColor(0, 200, 255, 90), 1)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        previous = self._to_widget(*waypoints[0])
        for x_um, y_um in waypoints[1:]:
            point = self._to_widget(x_um, y_um)
            painter.drawLine(previous, point)
            previous = point

    def _draw_footprint(self, painter: QPainter, scale: float) -> None:
        if self._footprint is None:
            return
        rect = QRectF(0.0, 0.0, self._plan.fov_x_um * scale,
                      self._plan.fov_y_um * scale)
        rect.moveCenter(self._to_widget(*self._at(*self._footprint)))
        painter.setPen(QPen(QColor(theme.ACCENT), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

    def _draw_markers(self, painter: QPainter) -> None:
        font = QFont(painter.font())
        font.setPixelSize(9)
        painter.setFont(font)
        for index, marker in enumerate(self._markers):
            point = self._to_widget(*self._at(marker.x_um, marker.y_um))
            selected = index == self._selected
            colour = QColor(theme.ACCENT) if selected else QColor(255, 90, 90)
            painter.setPen(QPen(colour, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            radius = 6.0 if selected else 4.5
            painter.drawEllipse(point, radius, radius)
            if marker.label:
                painter.setPen(QColor(theme.TEXT))
                painter.drawText(point + QPointF(7.0, 3.0), marker.label)

    def _draw_caption(self, painter: QPainter) -> None:
        if not self._caption:
            return
        painter.setPen(QColor(theme.TEXT_DIM))
        painter.drawText(QRectF(6.0, self.height() - 18.0,
                                self.width() - 12.0, 14.0),
                         Qt.AlignmentFlag.AlignLeft
                         | Qt.AlignmentFlag.AlignVCenter, self._caption)

    @staticmethod
    def _qimage(thumb: np.ndarray) -> QImage:
        """numpy RGB → QImage. Copied: the array belongs to the worker
        thread that produced it."""
        array = np.ascontiguousarray(thumb)
        image = QImage(array.data, array.shape[1], array.shape[0],
                       3 * array.shape[1], QImage.Format.Format_RGB888)
        return image.copy()


__all__ = ["ScanMapMarker", "ScanMapPlan", "ScanMapTile", "ScanMapWidget",
           "fit_view", "plan_bounds"]
