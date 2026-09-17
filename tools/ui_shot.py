"""Headless UI screenshot rig for vision-MCP audits.

Builds the REAL app object graph in sim mode (no hardware), shows the
window, lets the sim camera stream a few frames, then grabs each
workspace + the auxiliary windows at 1920×1080 and 1024×640.

    python tools/ui_shot.py [--out DIR] [--shots nav,sample,stage,log]

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
from talos.hal.base import Axis  # noqa: E402


def flush(ms: int) -> None:
    from PySide6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def synthetic_frame(app: TALOSApplication, flecks: bool = False) -> None:
    """Emit one 16:9 synthetic frame (the sim camera's own pattern is
    square — this keeps the live view letterboxed like real 1080p).

    ``flecks`` plants a few blobs of the colour the Sample Finding tab is
    currently looking for, so the processed and sample views have
    something to find. Without it the chain correctly finds nothing and
    the doc images show an empty overlay, which reads as a broken tab.
    """
    rng = np.random.default_rng(7)
    frame = (rng.integers(10, 90, (1080, 1920, 3))).astype(np.uint8)
    if flecks:
        try:
            from talos.cv.identify import hex_to_rgb

            colour = hex_to_rgb(app.window._sample_finding.colour_group
                                .hex_color())
        except Exception:  # noqa: BLE001 - a shot must never fail on this
            colour = (200, 162, 200)
        for x, y, w, h in ((520, 300, 260, 170), (1180, 620, 190, 130),
                           (900, 220, 120, 90), (330, 700, 150, 110)):
            frame[y:y + h, x:x + w] = colour
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

    # The AF measurement region: a deliberately off-centre box so the
    # overlay style (dashed outline, "AF ROI" tag, no fill) is visible and
    # the ROI numbers in both settings instances are non-default.
    window._af_roi.set_roi((0.25, 0.2, 0.4, 0.45))
    flush(150)

    # The inverse-video crosshair + the calibrated tick ruler (the frame
    # is the synthetic 1920×1080 pattern, so the canonical 4K calibration
    # runs at ×2 — see set_live_calibration).
    window._for_each_live_view(lambda v: v.set_crosshair_enabled(True))
    window._for_each_live_view(lambda v: v.set_crosshair_ticks_enabled(True))
    window._for_each_live_view(lambda v: v.set_ruler_enabled(True))
    flush(150)
    synthetic_frame(app)
    window.repaint()
    path = args.out / f"nav_overlays{tag}.png"
    window.grab().save(str(path))
    print(f"saved {path}")
    window._for_each_live_view(lambda v: v.set_ruler_enabled(False))
    window._for_each_live_view(lambda v: v.set_crosshair_ticks_enabled(False))
    window._for_each_live_view(lambda v: v.set_crosshair_enabled(False))
    flush(100)

    shots = {
        f"nav{tag}": window,
        f"stage{tag}": window._stage_window,
        f"log{tag}": window._log,
        f"focus{tag}": window._focus_window,
        # the Sample Finding tab, both halves of it: the CV column and
        # the scan column are judged separately (see --shots below)
        f"scanmap{tag}": window._sample_finding.scan_panel.map,
        f"identify{tag}": window._sample_finding.colour_group,
        # embedded close-ups for the control-panel QA (checkbox dot,
        # slider handle, ms exposure, temperature grid, the AF ROI block)
        f"camgroup{tag}": window._navigation.camera_group,
        f"temp{tag}": window._navigation.temp_group,
        f"afgroup{tag}": window._navigation.af_group,
        f"roi_live{tag}": window._navigation.live_view,
        # the bottom instrument strip on its own — the layout QA crop
        f"strip{tag}": window._strip,
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

    # The strip in its live states: a stage turning (state word lit, a
    # limit switch on), the focus mid-jog, the heater under power. The
    # idle shot above is the same widget a moment earlier.
    from dataclasses import asdict

    from talos.models import FocusStatus, StagePosition, StageStatus

    window._on_device_state("zolix", {
        "connected": True,
        "status": asdict(StageStatus(x_moving=True, limit_x_pos=True)),
        "position": asdict(StagePosition(x_um=12.5, y_um=-3.0, r_deg=1.25)),
    })
    window._on_device_state("sigmakoki", {
        "connected": True,
        "status": {"xspd": "0", "yspd": "0", "zspd": "0"},
        "position": {Axis.X: 100, Axis.Y: -40, Axis.Z: 12},
        "limits": {"x+": False, "x-": True, "y+": False, "y-": False},
    })
    window._on_device_state("focus", {
        "status": asdict(FocusStatus(pos=123, mode="CONT", blocked_dir="0")),
        "slim_bounds": (-1000, 2000),
    })
    window._on_device_state("yudian", {"pv": 24.8, "sv": 25.0,
                                       "output_percent": 12.0})
    # NO flush(): the sim devices are still polling and would overwrite
    # these values. repaint() + grab() are synchronous, so the states
    # above are exactly what lands in the image.
    window._strip.repaint()
    path = args.out / f"strip_active{tag}.png"
    window._strip.grab().save(str(path))
    print(f"saved {path}")

    # ...and with the right trigger held: the bar's middle then shows the
    # computed jog speed, which is the only time that text exists (at rest
    # it would just repeat the state word beside the bar).
    window._strip.trigger_bar().set_connected(True)
    window._strip.trigger_bar().set_state(0.0, 0.75)
    window._strip.repaint()
    path = args.out / f"strip_jog{tag}.png"
    window._strip.grab().save(str(path))
    print(f"saved {path}")

    # The Sample Finding workspace (tab 1), grabbed while it is active and
    # with its filters on and its chain finding something — an empty
    # overlay would read as a broken tab rather than as a quiet one.
    window._tabs.setCurrentIndex(1)
    flush(400)
    # Silence the sim camera's own stream first: it would overwrite every
    # synthetic frame between the emit and the grab, and the tab would be
    # showing the simulator's scene rather than the one being posed.
    window._sample_finding._engine.set_live(False)
    try:
        app.manager.camera.set_streaming(False)
    except Exception:  # noqa: BLE001 - the shot is worth more than the API being there
        pass
    flush(200)
    finding = window._sample_finding
    finding._engine.set_live(True)
    finding.preprocess_group.enable.setChecked(True)
    finding.preprocess_group.local.enable.setChecked(True)
    finding.preprocess_group.local.gain.setValue(4.0)
    finding.preprocess_group.local.width.setValue(24.0)
    flush(300)
    for _ in range(6):
        synthetic_frame(app, flecks=True)
        flush(260)                      # let the detection worker catch up
    path = args.out / f"scan{tag}.png"
    window.grab().save(str(path))
    print(f"saved {path}")

    # ...and the same tab in each of the other two view modes, for the
    # documentation: what the filters make of the frame, and what the
    # chain found in it.
    for mode in ("preprocessed", "samples"):
        finding.set_view_mode(mode)
        for _ in range(3):
            synthetic_frame(app, flecks=True)
            flush(260)
        path = args.out / f"scan_{mode}{tag}.png"
        window.grab().save(str(path))
        print(f"saved {path}")
    finding.set_view_mode("original")
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

    def prefs_page(label: str) -> None:
        """Select a page by its nav label (page ORDER is a UI decision —
        index-based lookups broke every time a page was added)."""
        labels = [item.text(0) for item in prefs._page_items]
        prefs._nav.setCurrentItem(prefs._page_items[labels.index(label)])
        flush(200)
        prefs.repaint()

    # The accent-combo popup (group-header QA).
    combo = prefs._pages[0]._accent_combo
    combo.showPopup()
    flush(200)
    path = args.out / f"prefs_accent_popup{tag}.png"
    combo.view().grab().save(str(path))
    print(f"saved {path}")
    combo.hidePopup()

    # The merged Objectives & Calibration page (Basic/Advanced tables).
    prefs_page("Objectives & Calibration")
    path = args.out / f"prefs_objectives{tag}.png"
    prefs.grab().save(str(path))
    print(f"saved {path}")

    # The Temperature page with the preset editor.
    prefs_page("Temperature")
    path = args.out / f"prefs_temperature{tag}.png"
    prefs.grab().save(str(path))
    print(f"saved {path}")

    # Grouped device page (Connection / Manual controls / Axis direction).
    prefs_page("Zolix XYR")
    path = args.out / f"prefs_zolix{tag}.png"
    prefs.grab().save(str(path))
    print(f"saved {path}")

    # The new input page + the camera page (flip).
    prefs_page("Input & Gamepad")
    path = args.out / f"prefs_input{tag}.png"
    prefs.grab().save(str(path))
    print(f"saved {path}")
    prefs_page("Camera")
    path = args.out / f"prefs_camera{tag}.png"
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
