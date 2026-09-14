"""Generate the bundled QSS image assets (resources/qss/).

Qt's stylesheet loader cannot take data: URIs, so QSS glyphs ship as
real PNG files. Run from the repo root:

    pwsh -Command "conda activate talos; python tools/make_qss_assets.py"

Only checkbox_dot.png is generated (the arrows_up/down.png predate the
script and stay as-is).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter

OUT_DIR = Path(__file__).resolve().parent.parent / "resources" / "qss"


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


if __name__ == "__main__":
    make_checkbox_dot()
