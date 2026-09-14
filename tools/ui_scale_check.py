"""End-to-end scale-bar pixel checker (sim, no hardware).

Verifies all three render paths against the CANONICAL 4K-sensor
calibration (0.4 µm/px = the 2 µm Axiocam sensor pixels / 5x):

  1. the live-view overlay at 1080p (calibration × 3840/1920 = 0.8)
  2. the 1080p snapshot burn (0.8)
  3. the 4K snapshot burn (0.4)

Each check measures the bar's actual pixel run in the rendered image
and compares it with the expected length (bar_um / um_per_px). Run on
the real display (the live view grab needs real widgets):

    pwsh -Command "conda activate talos; python tools/ui_scale_check.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QEventLoop, QTimer, Qt  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402

from talos.cv.calibration import SENSOR_WIDTH_PX  # noqa: E402
from talos.cv.scale_bar import scale_bar_layout  # noqa: E402

CANON = 0.4  # µm per 4K-sensor pixel (the 5x pixel-pitch fallback)


def _measure_bar_px(img: np.ndarray, spec, thresh: int = 200) -> int:
    """The bar's pixel run in the bottom-right (bar fill ≈ white)."""
    x, y, w, h = spec.bar_rect
    row = img[y + h // 2, max(0, x - 10):x + w + 10]
    bright = np.where(row >= thresh)[0]
    if len(bright) == 0:
        return 0
    return int(bright[-1] - bright[0]) + 1


def flush(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def check_burns(out_dir: Path) -> list[str]:
    """Snapshots at 4K (0.4) and 1080p (0.8) via the sim camera."""
    from talos.hal.sim.sim_camera import SimCamera

    results = []
    cam = SimCamera({"width": 1920, "height": 1080, "fps": 60})
    for tag, resolution, um_per_px, width in (
        ("4k", 0, CANON, 3840),
        ("1080p", 1, CANON * (SENSOR_WIDTH_PX / 1920.0), 1920),
    ):
        path = out_dir / f"scale_{tag}.png"
        cam.snapshot(path, resolution=resolution,
                     burn={"um_per_px": um_per_px})
        img = cv2.imread(str(path))[:, :, ::-1]
        spec = scale_bar_layout(um_per_px, img.shape[:2])
        measured = _measure_bar_px(img, spec)
        expected = spec.bar_px  # the spec IS the expected geometry
        ok = abs(measured - expected) <= 1
        results.append(
            f"burn {tag}: label={spec.label} um_per_px={um_per_px} "
            f"measured={measured}px expected={expected}px "
            f"{'OK' if ok else 'FAIL'}")
    return results


def check_live_view(out_dir: Path) -> list[str]:
    """The live view overlay at a 1920x1080 frame (calibration ×2)."""
    from talos.app import TALOSApplication
    from talos.bootstrap import bootstrap

    qapp = bootstrap()
    app = TALOSApplication(qapp, sim=True)
    window = app.window
    window.setWindowState(Qt.WindowState.WindowNoState)
    window.resize(1920, 1080)
    window.show()
    app.manager.connect_all()
    app.manager.camera.set_frame_slot(app.frame_slot)
    flush(600)
    view = window._navigation.live_view
    view.set_scale_bar_calibration(CANON)  # canonical 4K value
    view.set_scale_bar_enabled(True)
    # the sim camera streams its own 1280x960 frames — the grab uses the
    # live view's ACTUAL frame shape (the draw path does the same)
    flush(300)
    frame_h, frame_w = view._last_shape[:2]
    pix = view.grab()
    img = pix.toImage().convertToFormat(QImage.Format.Format_ARGB32)
    w, h = img.width(), img.height()
    bpl = img.bytesPerLine()
    arr = np.frombuffer(img.constBits(), dtype=np.uint8) \
        .reshape(h, bpl // 4, 4)[:, :w, :3]
    path = out_dir / "scale_live.png"
    pix.save(str(path))

    from talos.cv.af_roi import fit_transform
    from talos.cv.scale_bar import scale_bar_layout

    live_um = CANON * (SENSOR_WIDTH_PX / float(frame_w))
    spec = scale_bar_layout(live_um, (frame_h, frame_w))
    scale, off_x, off_y = fit_transform((w, h), (frame_h, frame_w))
    expected_widget = spec.bar_px * scale
    # measure in WIDGET space: map the frame-space bar rect through the
    # letterbox transform
    x, y, bw, bh = spec.bar_rect
    wy = int(off_y + (y + bh / 2.0) * scale)
    wx = int(off_x + x * scale)
    row = arr[wy, max(0, wx - 10):int(wx + bw * scale) + 10]
    bright = np.where(row >= 200)[0]
    measured = int(bright[-1] - bright[0]) + 1 if len(bright) else 0
    ok = abs(measured - expected_widget) <= 2.0
    app.shutdown()
    return [f"live {frame_w}px stream: label={spec.label} "
            f"live_um={live_um:.3f} measured={measured}px "
            f"expected={expected_widget:.1f}px {'OK' if ok else 'FAIL'}"]


def main() -> int:
    out_dir = Path("docs/bench/ui")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = check_burns(out_dir) + check_live_view(out_dir)
    for line in results:
        print(line)
    return 0 if all("OK" in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
