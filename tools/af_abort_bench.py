"""AF-S input-abort + stay-on-failure hardware bench (full-stack, real
hardware, user approved full-auto — the post-unwire verification).

Drives the REAL app wiring headless: InstrumentManager + AppState +
AutofocusService + the real proxies/workers (no window) — so the
sig_job_submitted hook, the arm-window cancel and the failure policy are
exercised exactly as the app exercises them.

Protocol (all deliberate defocus moves AWAY from the sample only):
  1. real AF-S run from focus → success (the baseline sanity)
  2. focus-jog submit mid-run → ABORT + the jog executes after (input wins)
  3. XY (go-to) submit mid-run → ABORT (the stage moves)
  4. snapshot submit mid-run → ABORT + the file lands
  5. exposure tweak mid-run → NO abort (the run continues, not aborted)
  6. input during the 350 ms arm window → the run never starts
  7. failure (bounded window, the peak outside) → the axis STAYS at the
     failure position (no arm restore — the new v3 policy)
  8. final normal AF-S → refocus from +400 away (the known-good
     deep-defocus family) + a landed frame for the vision MCP

Usage:
    python tools/af_abort_bench.py
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
from tools.af_bench import BENCH_DIR, _safe  # noqa: E402

FAILURES: list[str] = []


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
    """The headless mini-app: manager + state + service + frame slot."""

    def __init__(self, qapp):
        self.settings = Settings.load()
        self.state = AppState()
        self.manager = InstrumentManager(self.settings)
        self.frame_slot = LatestFrameSlot()
        self.service = AutofocusService(self.manager, self.settings,
                                        self.state, self.frame_slot)
        self.finished: list = []
        self.service.sig_af_finished.connect(self.finished.append)
        self.service.sig_af_log.connect(lambda m: _safe(f"  [service] {m}"))
        self.focus_status: dict = {}
        self.manager.sig_device_state.connect(self._on_device_state)

    def _on_device_state(self, key: str, payload: dict) -> None:
        if key == "focus" and isinstance(payload.get("status"), dict):
            self.focus_status = payload["status"]

    def focus_idle(self) -> bool:
        """True when the focus axis is idle. The telemetry dict carries
        the `mode` field only — `is_idle` is a property on the status
        object and never reaches asdict()."""
        return self.focus_status.get("mode", "IDLE") == "IDLE"

    def connect(self) -> bool:
        self.manager.connect_all()
        # The app's MainWindow turns streaming on via the connected
        # signal — a headless harness must do it itself.
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
        # The position counter is session-relative (0 on reconnect) — wait
        # for the STATUS payload itself, not a nonzero position.
        if not wait_for(lambda: "pos" in self.focus_status, 15.0,
                        "focus telemetry"):
            _safe("FATAL: focus telemetry never arrived")
            return False
        return True

    def stop_focus_now(self) -> None:
        """Belt-and-braces: kill any focus motion this bench started."""
        self.manager.submit("focus", "stop", priority=1)
        time.sleep(0.3)
        self.manager.submit("focus", "stop", priority=1)

    def shutdown(self) -> None:
        self.service.shutdown()
        self.manager.shutdown()
        flush(500)


def drill_real_af(h: Harness, tag: str, bounds=None) -> tuple:
    h.finished.clear()
    h.service.start_af_s(bounds=bounds)
    ok = wait_for(lambda: h.finished, 150.0, f"AF finish ({tag})")
    flush(500)
    result = h.finished[-1] if h.finished else None
    return ok, result


def wait_armed(h: Harness) -> bool:
    """True once the arm delay elapsed and the job is running."""
    return h.service._job_kind == "af_s" and h.service._pending_request is None


def main() -> int:
    parser = argparse.ArgumentParser(description="AF-S input-abort bench")
    args = parser.parse_args()

    setup_logging(verbose=False)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    qapp = QApplication.instance() or QApplication([])
    h = Harness(qapp)
    _safe("connecting devices (the app must NOT be running)…")
    if not h.connect():
        return 1
    arm0 = h.manager.focus_position
    _safe(f"connected — focus at {arm0} steps")

    # ---- 1. baseline real AF-S --------------------------------------------
    _safe("\n=== 1. real AF-S from focus ===")
    ok, result = drill_real_af(h, "baseline")
    status = "OK" if ok and result.success else f"FAIL: {result.message if result else 'no result'}"
    _safe(f"  baseline: {status}")
    if not (ok and result.success):
        FAILURES.append("1. baseline AF-S")

    # ---- 2. focus jog mid-run → abort + input wins -------------------------
    _safe("\n=== 2. focus jog mid-run ===")
    h.finished.clear()
    h.service.start_af_s()
    if not wait_for(lambda: wait_armed(h), 5.0, "armed"):
        h.stop_focus_now()
        FAILURES.append("2. focus jog (arm never fired)")
    else:
        flush(800)  # mid-probe/coarse
        h.manager.submit("focus", "set_speed", 200)  # the UI jog path
        ok = wait_for(lambda: h.finished, 30.0, "abort after jog")
        flush(500)
        result = h.finished[-1] if h.finished else None
        aborted = ok and result is not None and result.aborted
        # input wins: the queued jog executes after the job returns
        moved = wait_for(
            lambda: not h.focus_idle(), 5.0, "jog executed")
        h.stop_focus_now()
        if not wait_for(lambda: h.focus_idle(), 5.0,
                        "axis idle after stop"):
            FAILURES.append("2. focus jog (axis not idle)")
            h.stop_focus_now()
        _safe(f"  aborted={aborted} msg={result.message if result else '?'} "
              f"jog_executed={moved}")
        if not (aborted and moved):
            FAILURES.append("2. focus jog (abort/input-wins)")

    # ---- 3. XY go-to mid-run → abort ---------------------------------------
    zolix = h.manager._proxies.get("zolix")
    if zolix is None:
        _safe("\n=== 3. XY mid-run: zolix absent — skipped ===")
    else:
        _safe("\n=== 3. XY go-to mid-run ===")
        h.finished.clear()
        h.service.start_af_s()
        if not wait_for(lambda: wait_armed(h), 5.0, "armed"):
            h.stop_focus_now()
            FAILURES.append("3. XY (arm never fired)")
        else:
            flush(800)
            h.manager.submit("zolix", "move_rel_um", 3.0, 0.0)
            ok = wait_for(lambda: h.finished, 30.0, "abort after XY")
            flush(500)
            result = h.finished[-1] if h.finished else None
            aborted = ok and result is not None and result.aborted
            _safe(f"  aborted={aborted} msg={result.message if result else '?'}")
            if not aborted:
                FAILURES.append("3. XY (no abort)")
            time.sleep(1.0)
            h.manager.submit("zolix", "move_rel_um", -3.0, 0.0)  # back
            flush(1500)

    # ---- 4. snapshot mid-run → abort + file -------------------------------
    _safe("\n=== 4. snapshot mid-run ===")
    snap_path = BENCH_DIR / "abort_drill_snapshot.png"
    h.finished.clear()
    h.service.start_af_s()
    if not wait_for(lambda: wait_armed(h), 5.0, "armed"):
        h.stop_focus_now()
        FAILURES.append("4. snapshot (arm never fired)")
    else:
        flush(800)
        h.manager.submit_camera("snapshot", str(snap_path))
        ok = wait_for(lambda: h.finished, 30.0, "abort after snapshot")
        flush(500)
        result = h.finished[-1] if h.finished else None
        aborted = ok and result is not None and result.aborted
        exists = snap_path.exists() and snap_path.stat().st_size > 0
        _safe(f"  aborted={aborted} snapshot_exists={exists} "
              f"msg={result.message if result else '?'}")
        if not (aborted and exists):
            FAILURES.append("4. snapshot")

    # ---- 5. exposure tweak mid-run → NO abort ------------------------------
    _safe("\n=== 5. exposure tweak mid-run (must NOT abort) ===")
    props = dict(h.manager.camera_props)
    exp_before = props.get("exposure_us") or props.get("exposure")
    h.finished.clear()
    h.service.start_af_s()
    if not wait_for(lambda: wait_armed(h), 5.0, "armed"):
        h.stop_focus_now()
        FAILURES.append("5. exposure (arm never fired)")
    else:
        flush(800)
        if exp_before:
            h.manager.submit_camera("set_property", "exposure_us",
                                    float(exp_before) * 1.05)
        ok = wait_for(lambda: h.finished, 90.0, "run finish")
        flush(500)
        result = h.finished[-1] if h.finished else None
        not_aborted = ok and result is not None and not result.aborted
        _safe(f"  finished={ok} aborted={result.aborted if result else '?'} "
              f"success={result.success if result else '?'} "
              f"msg={result.message if result else '?'}")
        if not not_aborted:
            FAILURES.append("5. exposure (aborted despite the exclusion)")
        if exp_before:
            h.manager.submit_camera("set_property", "exposure_us",
                                    float(exp_before))
            flush(1000)

    # ---- 6. input during the arm window → run never starts ----------------
    _safe("\n=== 6. input during the 350 ms arm window ===")
    h.finished.clear()
    h.service.start_af_s()
    h.manager.submit("focus", "set_speed", 150)  # no flush — inside the window
    h.manager.submit("focus", "stop", priority=1)
    flush(100)
    cancelled = (h.service._job_kind == "" and h.state.mode == "MANUAL"
                 and h.finished and h.finished[-1].aborted)
    flush(600)  # the arm timer must not fire
    jobs = 0  # the manager's focus job counter: no autofocus job ran
    if not wait_for(lambda: h.focus_idle(), 5.0,
                    "axis idle"):
        h.stop_focus_now()
        FAILURES.append("6. arm window (axis not idle)")
    _safe(f"  cancelled={cancelled} phase={getattr(h.finished[-1], 'phase', '?') if h.finished else '?'} "
          f"msg={h.finished[-1].message if h.finished else '?'}")
    if not cancelled:
        FAILURES.append("6. arm window (not cancelled)")

    # ---- 7. failure stays put (bounded window, peak outside) ---------------
    _safe("\n=== 7. bounded-window failure → NO arm restore ===")
    arm_before = h.manager.focus_position
    h.manager.submit("focus", "move_rel", 400)  # defocus AWAY (the rule)
    wait_for(lambda: h.manager.focus_position >= arm_before + 380, 30.0,
             "defocus move")
    flush(1000)
    arm = h.manager.focus_position
    h.finished.clear()
    h.service.start_af_s(bounds=(arm - 100, arm + 100))
    ok = wait_for(lambda: h.finished, 120.0, "bounded AF finish")
    flush(500)
    result = h.finished[-1] if h.finished else None
    if ok and result is not None:
        final = h.manager.focus_position
        stayed = (not result.success
                  and abs(final - arm) > 20
                  and "stopped at current position" in (result.message or ""))
        _safe(f"  success={result.success} arm={arm} final={final} "
              f"msg={result.message}")
        if not stayed:
            FAILURES.append("7. stay-on-failure (axis restored?)")
    else:
        FAILURES.append("7. stay-on-failure (no result)")

    # ---- 8. final normal AF-S → refocus from +400 --------------------------
    _safe("\n=== 8. final normal AF-S (deep-defocus recovery) ===")
    ok, result = drill_real_af(h, "final")
    for attempt in range(2, 4):  # the known multi-peak stage-2 family retries
        if ok and result is not None and result.success:
            break
        _safe(f"  final attempt {attempt - 1} failed — retrying")
        ok, result = drill_real_af(h, f"final_retry{attempt}")
    status = "OK" if ok and result.success else f"FAIL: {result.message if result else 'no result'}"
    _safe(f"  final: {status}")
    if not (ok and result.success):
        FAILURES.append("8. final AF-S")
    else:
        item = h.frame_slot.read()
        if item is not None:
            frame, _meta = item
            out = BENCH_DIR / "af_abort_bench_landed.png"
            cv2.imwrite(str(out), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            _safe(f"  landed frame saved: {out} (for the vision MCP)")

    # ---- teardown ----------------------------------------------------------
    if not wait_for(lambda: h.focus_idle(), 5.0, "idle"):
        h.stop_focus_now()
    h.shutdown()
    _safe("\n" + ("ALL DRILLS PASSED" if not FAILURES
                  else "FAILURES: " + "; ".join(FAILURES)))
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
