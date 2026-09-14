"""Headless UI screenshot rig for vision-MCP audits.

Builds the REAL app object graph in sim mode (no hardware), shows the
window, lets the sim camera stream a few frames, then grabs each
workspace + the auxiliary windows at 1920×1080 and 1024×640.

    python tools/ui_shot.py [--out DIR] [--shots nav,scan,stage,log,prefs]

Runs on the real desktop (sim only — no hardware is touched; the window
briefly appears). Set QT_QPA_PLATFORM=offscreen to run headless — note
that offscreen renders text as placeholder boxes and caps the grab size,
so the real display is preferred for visual QA.

Note: with the debug console enabled (default), a separate console
window tails the log file (the app itself never detaches from the
launching terminal).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

from talos.app import TALOSApplication  # noqa: E402
from talos.bootstrap import bootstrap  # noqa: E402


def flush(ms: int) -> None:
    from PySide6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def synthetic_frame(app: TALOSApplication) -> None:
    """Emit one 16:9 synthetic frame (the sim camera's own pattern is
    square — this keeps the live view letterboxed like real 1080p)."""
    rng = np.random.default_rng(7)
    frame = (rng.integers(10, 90, (1080, 1920, 3))).astype(np.uint8)
    app.manager.camera.sig_frame.emit(frame)


def main() -> int:
    parser = argparse.ArgumentParser(description="TALOS UI screenshot rig")
    parser.add_argument("--out", type=Path, default=Path("docs/bench/ui"))
    parser.add_argument("--size", nargs=2, type=int, default=None,
                        metavar=("W", "H"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    qapp = bootstrap()
    app = TALOSApplication(qapp, sim=True)
    window = app.window

    # The restored window state may be maximized (persisted QSettings) —
    # a resize is ignored while the state flag is set.
    window.setWindowState(Qt.WindowState.WindowNoState)
    if args.size:
        window.resize(*args.size)
    else:
        window.resize(1920, 1080)  # the design target (offscreen has no
        # real screen — showMaximized does not give 1080p there)
    window.show()
    app.manager.connect_all()
    app.manager.camera.set_frame_slot(app.frame_slot)
    flush(800)
    synthetic_frame(app)
    flush(300)

    tag = f"_{args.size[0]}x{args.size[1]}" if args.size else "_1080p"
    shots = {
        f"nav{tag}": window,
        f"stage{tag}": window._stage_window,
        f"log{tag}": window._log,
        f"focus{tag}": window._focus_window,
        # embedded close-ups for the control-panel QA (checkbox dot,
        # slider handle, ms exposure, temperature grid)
        f"camgroup{tag}": window._navigation.camera_group,
        f"temp{tag}": window._navigation.temp_group,
    }

    for name, widget in shots.items():
        widget.show()
        flush(150)
        synthetic_frame(app)  # grab() paints the pending frame immediately
        widget.repaint()  # flush child paint events (the overlay surface)
        pix = widget.grab()
        path = args.out / f"{name}.png"
        pix.save(str(path))
        print(f"saved {path}")
        if widget is not window and widget.isWindow():
            # top-level dialogs get hidden again; embedded groups must
            # stay visible inside the main window for the later shots
            widget.hide()

    # The Sample Finding workspace (tab 1), grabbed while it is active.
    window._tabs.setCurrentIndex(1)
    flush(600)
    synthetic_frame(app)
    window.repaint()
    path = args.out / f"scan{tag}.png"
    window.grab().save(str(path))
    print(f"saved {path}")
    window._tabs.setCurrentIndex(0)
    flush(200)

    # The Preferences dialog for the QA.
    from talos.ui.dialogs.preferences import PreferencesDialog

    prefs = PreferencesDialog(app.manager, app.settings, qapp, app.autofocus)
    prefs.show()
    flush(300)
    prefs.repaint()
    path = args.out / f"prefs{tag}.png"
    prefs.grab().save(str(path))
    print(f"saved {path}")

    # The accent-combo popup (group-header QA).
    combo = prefs._pages[0]._accent_combo
    combo.showPopup()
    flush(200)
    path = args.out / f"prefs_accent_popup{tag}.png"
    combo.view().grab().save(str(path))
    print(f"saved {path}")
    combo.hidePopup()

    # The merged Objectives & Calibration page (Basic/Advanced tables).
    prefs._nav.setCurrentItem(prefs._page_items[1])
    flush(200)
    prefs.repaint()
    path = args.out / f"prefs_objectives{tag}.png"
    prefs.grab().save(str(path))
    print(f"saved {path}")

    # The Temperature page with the preset editor.
    prefs._nav.setCurrentItem(prefs._page_items[7])
    flush(200)
    prefs.repaint()
    path = args.out / f"prefs_temperature{tag}.png"
    prefs.grab().save(str(path))
    print(f"saved {path}")
    prefs.hide()

    # AF-indicator states (orange stage-2 + green success) for the QA.
    from talos.cv.autofocus import AutofocusResult

    app.autofocus.sig_af_progress.emit(0.5, 2, 100.0, 0.0)
    flush(150)
    synthetic_frame(app)
    window.repaint()  # flush child paint events (the overlay surface)
    path = args.out / f"nav_af_stage2{tag}.png"
    window.grab().save(str(path))
    print(f"saved {path}")
    app.autofocus.sig_af_finished.emit(AutofocusResult(
        best_position=0, best_score=100.0, success=True, aborted=False,
        message="focused", phase="done"))
    flush(150)
    synthetic_frame(app)
    window.repaint()
    path = args.out / f"nav_af_done{tag}.png"
    window.grab().save(str(path))
    print(f"saved {path}")

    app.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
