"""Refocus the stage: repeat normal AF-S runs until one succeeds (the
multi-peak field's stage-2 edge family fails occasionally — retries
from the failure position usually recover).

    python tools/af_refocus.py [--attempts 5] [--snapshot out.png]
"""

from __future__ import annotations

import argparse
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
from tools.af_bench import _safe  # noqa: E402


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Refocus the stage (AF-S retries)")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--snapshot", type=str, default=None)
    args = parser.parse_args()

    setup_logging(verbose=False)
    qapp = QApplication.instance() or QApplication([])
    settings = Settings.load()
    state = AppState()
    manager = InstrumentManager(settings)
    slot = LatestFrameSlot()
    service = AutofocusService(manager, settings, state, slot)
    finished: list = []
    service.sig_af_finished.connect(finished.append)
    service.sig_af_log.connect(lambda m: _safe(f"  [service] {m}"))

    manager.connect_all()
    manager.camera.set_streaming(True)  # the MainWindow does this in the app
    manager.camera.set_frame_slot(slot)
    from talos.hal.proxies.focus_proxy import FocusProxy
    focus_proxy = manager.device("focus")
    if isinstance(focus_proxy, FocusProxy):
        focus_proxy.set_frame_slot(slot)
    if not wait_for(lambda: slot.read() is not None, 30.0, "camera frames"):
        _safe("FATAL: no camera frames (app/ZEN running? cable?)")
        return 1

    result = None
    for attempt in range(1, args.attempts + 1):
        finished.clear()
        service.start_af_s()
        ok = wait_for(lambda: finished, 150.0, f"AF finish (attempt {attempt})")
        result = finished[-1] if finished else None
        if ok and result is not None and result.success:
            _safe(f"FOCUSED on attempt {attempt}: best {result.best_position} "
                  f"steps, score {result.best_score:.0f}")
            break
        _safe(f"  attempt {attempt}: "
              f"{result.message if result is not None else 'no result'}")

    ok = result is not None and result.success
    if ok and args.snapshot:
        item = slot.read()
        if item is not None:
            frame, _meta = item
            cv2.imwrite(args.snapshot,
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            _safe(f"landed frame saved: {args.snapshot}")

    service.shutdown()
    manager.shutdown()
    flush(500)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
