"""UI-rework hardware bench (real hardware; the app must NOT be running).

Drives the NEW capture/auto-gain paths headless (manager + service +
proxies, no window) and leaves the camera settings as found:

  1. 4K snapshot through the proxy (the new resolution-switch path) —
     file lands, 3840x2160, the live stream resumes afterwards
  2. 1080p snapshot — 1920x1080
  3. snapshot with the scale-bar burn — the bottom-right differs
  4. auto-gain convergence: gain forced low, the controller runs for a
     few seconds, the frame mean moves toward the target
  5. AF-S regression: a normal run from focus (the UI now passes the
     AF-group bounds; unbounded by default) — retried like af_refocus
  6. teardown: restore exposure/gain/WB, leave the stage focused

Usage:
    python tools/ui_hw_bench.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from talos.app import AppState  # noqa: E402
from talos.config import Settings  # noqa: E402
from talos.cv.autofocus_service import AutofocusService  # noqa: E402
from talos.cv.frame_slot import LatestFrameSlot  # noqa: E402
from talos.instruments import InstrumentManager  # noqa: E402
from talos.logging_setup import setup_logging  # noqa: E402
from talos.ui.auto_gain import AutoGainController, mean_luma  # noqa: E402

BENCH_DIR = Path(__file__).resolve().parent.parent / "docs" / "bench"
FAILURES: list[str] = []


def _safe(msg: str) -> None:
    print(msg.encode(sys.stdout.encoding or "ascii", "replace")
          .decode(sys.stdout.encoding or "ascii"), flush=True)


def flush(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def wait_for(pred, timeout_s: float, label: str) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        flush(100)
    _safe(f"  WAIT TIMEOUT: {label}")
    return False


class Harness:
    def __init__(self, qapp):
        self.settings = Settings.load()
        self.state = AppState()
        self.manager = InstrumentManager(self.settings)
        self.frame_slot = LatestFrameSlot()
        self.service = AutofocusService(self.manager, self.settings,
                                        self.state, self.frame_slot)
        self.finished: list = []
        self.service.sig_af_finished.connect(self.finished.append)

    def connect(self) -> bool:
        self.manager.connect_all()
        self.manager.camera.set_streaming(True)
        self.manager.camera.set_frame_slot(self.frame_slot)
        focus = self.manager.device("focus")
        from talos.hal.proxies.focus_proxy import FocusProxy
        if isinstance(focus, FocusProxy):
            focus.set_frame_slot(self.frame_slot)
        if not wait_for(lambda: self.frame_slot.read() is not None, 30.0,
                        "camera frames"):
            _safe("FATAL: no camera frames (app/ZEN running? cable?)")
            return False
        return True

    def restore_camera(self) -> None:
        for key, value in self.original_props.items():
            if key in ("exposure_us", "gain", "white_balance"):
                self.manager.submit_camera("set_property", key, value)
                flush(600)

    def shutdown(self) -> None:
        self.service.shutdown()
        self.manager.shutdown()
        flush(500)


def snapshot(h: Harness, path: Path, resolution: int,
             burn: dict | None = None) -> bool:
    """Submit a snapshot and wait for the JOB to finish (polling
    path.exists() races the cv2.imwrite inside the worker)."""
    done: list = []
    h.manager.sig_job_done.connect(lambda jid, res: done.append(jid))
    job = h.manager.submit_camera("snapshot", str(path), 25.0, resolution,
                                  burn)
    ok = wait_for(lambda: job in done, 90.0, f"snapshot job ({path.name})")
    flush(300)  # let the worker's post-job housekeeping settle
    return ok and path.exists() and path.stat().st_size > 0


def main() -> int:
    setup_logging(verbose=False)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    qapp = QApplication.instance() or QApplication([])
    h = Harness(qapp)
    _safe("connecting devices (the app must NOT be running)…")
    if not h.connect():
        return 1
    h.original_props = dict(h.manager.camera_props)
    _safe(f"connected — props: {h.original_props}")

    # ---- 1. 4K snapshot ---------------------------------------------------
    _safe("\n=== 1. 4K snapshot (resolution switch) ===")
    path4k = BENCH_DIR / "ui_hw_4k.png"
    ok = snapshot(h, path4k, 0)
    img = cv2.imread(str(path4k)) if ok else None
    shape = img.shape if img is not None else None
    _safe(f"  4K: ok={ok} shape={shape} size={path4k.stat().st_size if ok else '?'}")
    if not (ok and shape == (2160, 3840, 3)):
        FAILURES.append("1. 4K snapshot")
    else:
        # the live stream must have resumed (frames keep flowing)
        resumed = wait_for(lambda: h.frame_slot.read() is not None, 15.0,
                           "stream resumed after 4K")
        _safe(f"  stream resumed: {resumed}")
        if not resumed:
            FAILURES.append("1. stream resume after 4K")

    # ---- 2. 1080p snapshot --------------------------------------------------
    _safe("\n=== 2. 1080p snapshot ===")
    path1080 = BENCH_DIR / "ui_hw_1080p.png"
    ok = snapshot(h, path1080, 1)
    img = cv2.imread(str(path1080)) if ok else None
    shape = img.shape if img is not None else None
    _safe(f"  1080p: ok={ok} shape={shape}")
    if not (ok and shape == (1080, 1920, 3)):
        FAILURES.append("2. 1080p snapshot")

    # ---- 3. snapshot with the scale-bar burn -------------------------------
    _safe("\n=== 3. snapshot with scale-bar burn ===")
    path_burn = BENCH_DIR / "ui_hw_burn.png"
    ok = snapshot(h, path_burn, 1, burn={"um_per_px": 1.42})
    if not ok:
        FAILURES.append("3. burn snapshot")
    else:
        plain = cv2.imread(str(path1080))
        burned = cv2.imread(str(path_burn))
        diff = cv2.absdiff(plain, burned)
        changed = int((diff.sum(axis=2) > 30).sum())
        _safe(f"  burn: ok={ok} changed_px={changed}")
        if changed < 500:  # the bar + label cover several thousand px
            FAILURES.append("3. burn visibly absent")

    # ---- 4. auto-gain: standby convergence + framerate impact ------------
    _safe("\n=== 4. auto-gain standby + framerate impact ===")

    # 4a. BASELINE fps at the current gain (10 s, ~2 samples/s).
    fps_baseline: list[float] = []
    h.manager.camera.sig_fps.connect(fps_baseline.append)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        flush(500)
    baseline_median = sorted(fps_baseline)[len(fps_baseline) // 2] \
        if fps_baseline else 0.0

    # 4b. force the scene dark and let the loop converge, counting every
    # gain WRITE (each write stalls the camera's frame delivery — the
    # standby gating must keep the count low and the fps alive).
    h.manager.submit_camera("set_property", "gain", 1.0)
    flush(1500)
    autogain = AutoGainController(h.manager, h.settings, h.state)
    h.manager.camera.sig_frame.connect(autogain.on_frame)  # feed the loop
    autogain.set_enabled(True)
    autogain.note_manual_gain(1.0)  # after set_enabled (it resets _applied)
    autogain.set_target(120.0)
    autogain.start()
    adjustments: list = []
    autogain.sig_gain_changed.connect(adjustments.append)
    fps_window: list[float] = []
    h.manager.camera.sig_fps.connect(fps_window.append)
    item = h.frame_slot.read()
    luma0 = mean_luma(item[0]) if item else 0.0
    _safe(f"  gain forced 1.0, luma0={luma0:.1f}, "
          f"baseline fps median={baseline_median:.1f}")
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        flush(500)
    autogain.stop()
    flush(1000)
    item = h.frame_slot.read()
    luma1 = mean_luma(item[0]) if item else 0.0
    gain_after = autogain._current_gain()
    min_fps = min(fps_window) if fps_window else 0.0
    _safe(f"  gain: 1.0 → {gain_after:.1f} in {len(adjustments)} writes "
          f"(luma {luma0:.1f} → {luma1:.1f}, "
          f"fps baseline {baseline_median:.1f} / window min {min_fps:.1f})")
    moved = (gain_after > 1.5) and (abs(luma1 - 120.0) < abs(luma0 - 120.0))
    if not moved:
        FAILURES.append("4a. auto-gain did not converge")
    if len(adjustments) > 8:  # one write per ~1 s settle, stability-gated
        FAILURES.append(f"4b. too many gain writes ({len(adjustments)})")
    if baseline_median > 0 and min_fps < 0.5 * baseline_median:
        FAILURES.append("4c. framerate impact during adjustment")

    # ---- 5. AF-S regression -------------------------------------------------
    _safe("\n=== 5. AF-S regression (UI bounds path, unbounded default) ===")
    result = None
    for attempt in range(1, 4):
        h.finished.clear()
        h.service.start_af_s(bounds=None)
        ok = wait_for(lambda: h.finished, 150.0, f"AF finish ({attempt})")
        result = h.finished[-1] if h.finished else None
        if ok and result is not None and result.success:
            break
        _safe(f"  attempt {attempt}: {result.message if result else 'no result'}")
    if not (result is not None and result.success):
        FAILURES.append("5. AF-S regression")

    # ---- teardown -----------------------------------------------------------
    _safe("\n=== teardown ===")
    h.restore_camera()
    flush(1000)
    h.shutdown()
    _safe("\n" + ("ALL DRILLS PASSED" if not FAILURES
                  else "FAILURES: " + "; ".join(FAILURES)))
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
