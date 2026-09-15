"""Generate the bundled QSS image assets (resources/qss/).

Qt's stylesheet loader cannot take data: URIs, so QSS glyphs ship as
real PNG files. Run from the repo root:

    pwsh -Command "conda activate talos; python tools/make_qss_assets.py"

Generates checkbox_dot.png and combo_arrow.png (the arrows_up/down.png
predate the script and stay as-is).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen

OUT_DIR = Path(__file__).resolve().parent.parent / "resources" / "qss"

#: The theme's TEXT_DIM — the chevron's colour (kept in sync by hand: this
#: script must not import talos.ui, which would pull in QtWidgets).
_CHEVRON = "#9b9b9b"


def make_checkbox_dot(size: int = 7) -> None:
    """A solid white circle on transparency — the checkbox's checked dot.
    Drawn at the FINAL size: an odd-size canvas centers the circle on a
    real pixel (a downscale of a larger canvas lands on a fractional
    offset, which reads as a visibly off-center dot)."""
    canvas = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    canvas.fill(QColor(0, 0, 0, 0))
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QColor(0, 0, 0, 0))  # no outline
    painter.setBrush(QColor(255, 255, 255))
    painter.drawEllipse(0, 0, size, size)
    painter.end()
    out = OUT_DIR / "checkbox_dot.png"
    canvas.save(str(out))
    print(f"wrote {out}")


def make_combo_arrow(width: int = 9, height: int = 6) -> None:
    """A downward chevron for QComboBox::down-arrow.

    Without it every combo in the app renders as a plain box: Qt switches
    to stylesheet-drawn subcontrols as soon as ``QComboBox::drop-down`` is
    styled, and has no default arrow image to fall back on.
    """
    canvas = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    canvas.fill(QColor(0, 0, 0, 0))
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(_CHEVRON))
    pen.setWidthF(1.4)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.drawLine(QPointF(0.7, 1.2), QPointF(width / 2.0, height - 1.2))
    painter.drawLine(QPointF(width / 2.0, height - 1.2),
                     QPointF(width - 0.7, 1.2))
    painter.end()
    out = OUT_DIR / "combo_arrow.png"
    canvas.save(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    make_checkbox_dot()
    make_combo_arrow()
