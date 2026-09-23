"""SampleReviewWindow: one found sample, on the frame it was found in.

The sample list can already MOVE the stage to a sample. This is the other
half of "is this one worth going to": the frame the detector found it in,
with the sample ringed, so the operator can judge the flake — its shape, its
edge, whether the colour match caught the material or the tape residue next
to it — before spending a stage move on it.

Two things make the ring trustworthy. The frame is the one that was
CAPTURED (the PNG in the run's ``frames/``, or the live frame for a live
row), and the ring is drawn at the candidate's own ``x_px/y_px`` — the same
pixels identification measured, at full resolution, with no resampling
anywhere on the tile path. The window therefore needs no calibration and no
convention: it draws in the frame's own coordinates.

The window is a dumb viewer on purpose. It knows nothing about run
directories or manifests; the panel hands it a ``fetch(row)`` callback and a
pre-processing function, so the one place that knows where the pixels live
stays the one place that owns the run.

Esc behaves as it does in every other window here — STOP ALL first, then
hide — because a window that quietly swallowed the global stop would be the
one place the operator cannot panic in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from talos.cv.stitch import mark_sample
from talos.ui import theme

#: How much of the frame the 1:1 view shows, as a multiple of the ring's
#: radius. Tight enough that the flake fills the window, loose enough to
#: see what is around it (which is how "it caught the residue" is spotted).
_ZOOM_MARGIN = 4.0
_ZOOM_MIN_HALF_PX = 48


@dataclass
class SampleShot:
    """What the review window shows for one row.

    ``frame`` is the captured frame (or the live one) as RGB; ``error`` says
    why there is none, and is shown instead of a blank. The pixel fields are
    the candidate's OWN: the ring is drawn from them, never re-derived from
    µm, so it lands on the pixels identification measured.
    """

    frame: np.ndarray | None = None
    error: str = ""
    number: int = 0
    tile: int = -1
    source: str = ""
    x_um: float = 0.0
    y_um: float = 0.0
    area_um2: float = 0.0
    score: float = 0.0
    x_px: float = 0.0
    y_px: float = 0.0
    area_px2: float = 0.0
    extra: dict = field(default_factory=dict)


class _PreprocessWorker(QThread):
    """One off-thread ``pre.apply`` for the toggle.

    Off-thread because denoise at 4K is hundreds of milliseconds, and this
    window sits in front of a live microscope: a click that freezes the app
    for a fifth of a second is a click the operator will not make twice.
    """

    sig_done = Signal(object)         # the processed frame, or None

    def __init__(self, frame, apply_fn, parent=None):
        super().__init__(parent)
        self._frame = frame
        self._apply = apply_fn

    def run(self) -> None:  # noqa: D102
        try:
            self.sig_done.emit(self._apply(self._frame))
        except Exception:  # noqa: BLE001 - a failed filter shows the raw frame
            self.sig_done.emit(None)


def _compact_button(text: str, tooltip: str) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("compact")
    button.setToolTip(tooltip)
    button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return button


class SampleReviewWindow(QDialog):
    """The frame behind one row of the sample list, with the sample ringed."""

    def __init__(self, input_system=None, parent: QWidget | None = None):
        super().__init__(parent)
        self._input = input_system
        #: row -> SampleShot. Set by the panel (see :meth:`set_source`).
        self._fetch = None
        self._preprocess = None
        self._count = 0
        self._row = -1
        self._shot: SampleShot | None = None
        self._marked: np.ndarray | None = None      # ringed, as fetched
        self._shown: np.ndarray | None = None       # ringed + pre-processed
        self._worker: _PreprocessWorker | None = None
        self._zoom = False

        self.setWindowTitle("Sample")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setModal(False)
        self._build_ui()
        self.resize(900, 640)

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(6)
        self.title = QLabel("No sample")
        bold = self.title.font()
        bold.setBold(True)
        self.title.setFont(bold)
        top.addWidget(self.title, 1)
        self.prev_btn = _compact_button("◀", "Previous sample (←)")
        self.prev_btn.clicked.connect(lambda: self.step(-1))
        self.next_btn = _compact_button("▶", "Next sample (→)")
        self.next_btn.clicked.connect(lambda: self.step(1))
        top.addWidget(self.prev_btn)
        top.addWidget(self.next_btn)
        layout.addLayout(top)

        self.detail = QLabel("")
        self.detail.setObjectName("dim")
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)

        self.view = QLabel("")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(360, 240)
        self.view.setStyleSheet(f"background: {theme.BG}; "
                                f"border: 1px solid {theme.BORDER};")
        layout.addWidget(self.view, 1)

        bottom = QHBoxLayout()
        bottom.setSpacing(6)
        self.zoom_btn = QPushButton("1:1")
        self.zoom_btn.setObjectName("compact")
        self.zoom_btn.setCheckable(True)
        self.zoom_btn.setToolTip("Show the sample at one frame pixel per "
                                 "screen pixel instead of the whole frame")
        self.zoom_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.zoom_btn.toggled.connect(self._set_zoom)
        bottom.addWidget(self.zoom_btn)
        self.pre_btn = QCheckBox("Pre-processed")
        self.pre_btn.setToolTip("Apply the pre-processing chain to this frame "
                                "— the pixels identification actually saw")
        self.pre_btn.setEnabled(False)
        self.pre_btn.toggled.connect(self._set_preprocessed)
        bottom.addWidget(self.pre_btn)
        bottom.addStretch(1)
        self.note = QLabel("")
        self.note.setObjectName("dim")
        bottom.addWidget(self.note, 2)
        layout.addLayout(bottom)

    # ------------------------------------------------------------------

    def set_source(self, fetch, count: int, apply_fn=None) -> None:
        """``fetch(row) -> SampleShot``, ``count`` rows, and the optional
        pre-processing function for the toggle."""
        self._fetch = fetch
        self._preprocess = apply_fn
        self._count = max(0, int(count))
        self.pre_btn.setEnabled(apply_fn is not None)

    def show_sample(self, row: int) -> None:
        """Display row ``row`` (0-based, the list's own order)."""
        if self._fetch is None or not 0 <= row < self._count:
            return
        self._row = row
        self._clear_worker()
        self.pre_btn.blockSignals(True)
        self.pre_btn.setChecked(False)
        self.pre_btn.blockSignals(False)
        self._shown = None
        shot = self._fetch(row)
        self._shot = shot if isinstance(shot, SampleShot) else None
        if self._shot is None:
            self._marked = None
        else:
            self._marked = (mark_sample(
                self._shot.frame, self._shot.x_px, self._shot.y_px,
                self._shot.area_px2, label=self._shot.number or None)
                if self._shot.frame is not None else None)
        self._refresh()

    def step(self, delta: int) -> None:
        if self._count <= 0:
            return
        self.show_sample((self._row + delta) % self._count)

    def current_row(self) -> int:
        """Which row is on screen (-1 before the first show)."""
        return self._row

    def showEvent(self, event) -> None:  # noqa: N802, D102
        super().showEvent(event)
        self._render()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._render()

    def closeEvent(self, event) -> None:  # noqa: N802
        """Hide rather than destroy: the next review reuses this window."""
        event.ignore()
        self.hide()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key.Key_Escape:
            # Esc is the global STOP ALL, as in every other window here.
            if self._input is not None:
                self._input.on_escape()
            self.hide()
            event.accept()
            return
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Up):
            self.step(-1)
            event.accept()
            return
        if key in (Qt.Key.Key_Right, Qt.Key.Key_Down):
            self.step(1)
            event.accept()
            return
        if key == Qt.Key.Key_Space:
            self.zoom_btn.toggle()
            event.accept()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------

    def _set_zoom(self, on: bool) -> None:
        self._zoom = bool(on)
        self.zoom_btn.setText("Whole frame" if self._zoom else "1:1")
        self._render()

    def _set_preprocessed(self, on: bool) -> None:
        self._clear_worker()
        if not on or self._marked is None or self._preprocess is None:
            self._shown = None
            self._render()
            return
        self.note.setText("Pre-processing…")
        self._worker = _PreprocessWorker(self._marked, self._preprocess, self)
        self._worker.sig_done.connect(self._on_preprocessed)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_preprocessed(self, frame) -> None:
        self._worker = None
        if frame is None:
            self.note.setText("Pre-processing failed — showing the frame "
                              "as captured")
            self._shown = None
        else:
            self._shown = np.asarray(frame)
        self._render()

    def _clear_worker(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None and worker.isRunning():
            worker.wait(2000)

    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        """Header, details, buttons, note — everything except the image."""
        shot = self._shot
        self.prev_btn.setEnabled(self._count > 1)
        self.next_btn.setEnabled(self._count > 1)
        if shot is None:
            self.title.setText("No sample")
            self.detail.setText("")
            self.note.setText("")
            self.view.setPixmap(QPixmap())
            return
        number = shot.number or (self._row + 1)
        self.title.setText(
            f"Sample {number} of {self._count}"
            + (f"  ·  tile {shot.tile}" if shot.tile >= 0 else "  ·  live view"))
        bits = [f"X {shot.x_um:.1f} µm", f"Y {shot.y_um:.1f} µm",
                f"area {shot.area_um2:.1f} µm²", f"edge {shot.score:.1f}"]
        if shot.frame is not None:
            height, width = shot.frame.shape[:2]
            bits.append(f"frame {width}×{height}")
        if shot.source:
            bits.insert(0, shot.source)
        self.detail.setText("  ·  ".join(bits))
        if shot.error:
            self.note.setText(shot.error)
        elif not self.pre_btn.isChecked():
            self.note.setText("")
        self._render()

    def _render(self) -> None:
        frame = self._shown if self._shown is not None else self._marked
        if frame is None or not getattr(frame, "size", 0):
            self.view.setPixmap(QPixmap())
            return
        if self._zoom and self._shot is not None:
            frame = self._crop(frame, self._shot)
        height, width = frame.shape[:2]
        image = QImage(np.ascontiguousarray(frame).data, width, height,
                       3 * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image.copy())
        area = self.view.size()
        if (pixmap.width() > area.width()
                or pixmap.height() > area.height()):
            pixmap = pixmap.scaled(area, Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
        self.view.setPixmap(pixmap)

    def _crop(self, frame: np.ndarray, shot: SampleShot) -> np.ndarray:
        """A window around the sample at one frame pixel per screen pixel."""
        radius = max(4.0, float(np.sqrt(max(float(shot.area_px2), 1.0)
                                         / np.pi)))
        half = max(_ZOOM_MIN_HALF_PX, int(round(radius * _ZOOM_MARGIN)))
        height, width = frame.shape[:2]
        cx, cy = int(round(shot.x_px)), int(round(shot.y_px))
        x0, x1 = max(0, cx - half), min(width, cx + half)
        y0, y1 = max(0, cy - half), min(height, cy + half)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return frame
        return frame[y0:y1, x0:x1]


__all__ = ["SampleReviewWindow", "SampleShot"]
