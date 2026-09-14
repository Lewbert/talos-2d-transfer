"""Correlate camera frame stalls (1-2 s lags) with stage motion.

Hypotheses being tested:
  H1: motor EMI couples into the long USB cable -> USB3 link retrain ->
      1-2 s delivery gaps (would also hit Labscope/ZEN — user confirmed).
  H2: the camera's own AWB/ISP stalls when the scene changes during motion
      (tested by repeating the focus jog with AWB off + fixed color temp).

Phases (each ~20 s), tiny bounded motions only:
  idle -> focus jog -> zolix jog -> idle -> [AWB off] focus jog -> idle

Usage (ZEN/Labscope closed, user present at the bench):
    python tools/frame_stall_probe.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from talos.config import Settings  # noqa: E402
from talos.hal.devices.camera.smartcam_backend import SmartCamCamera  # noqa: E402
from talos.hal.devices.focus import FocusStageDriver  # noqa: E402
from talos.hal.devices.zolix import ZolixXYRStage  # noqa: E402

PHASE_S = 20.0
STALL_MS = 300.0


class PhasedCam(SmartCamCamera):
    def __init__(self, config):
        super().__init__(config)
        self.phase = "setup"
        self.events: list[tuple[str, float]] = []

    def _on_event(self, event_index, event_data_ptr):
        super()._on_event(event_index, event_data_ptr)
        if event_index == 5:
            self.events.append((self.phase, time.monotonic()))


def report(events: list[tuple[str, float]], phase: str) -> None:
    t = np.asarray([ts for ph, ts in events if ph == phase])
    if t.size < 2:
        print(f"  [{phase:16s}] no frames")
        return
    iv = np.diff(t) * 1000
    stalls = iv[iv > STALL_MS]
    print(f"  [{phase:16s}] frames={t.size:4d} mean={iv.mean():6.1f}ms "
          f"max={iv.max():6.1f}ms stalls>{STALL_MS:.0f}ms: {stalls.size}"
          + (f"  worst={stalls.max():.0f}ms" if stalls.size else ""))


def jog_focus(focus: FocusStageDriver, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    direction = 1
    while time.monotonic() < deadline:
        try:
            focus.move_rel(30 * direction, speed=200)
            direction = -direction
            time.sleep(0.25)
        except Exception as exc:  # noqa: BLE001
            print(f"  focus jog error: {exc}")
            break
    try:
        focus.stop()
    except Exception:  # noqa: BLE001
        pass


def jog_zolix(zolix: ZolixXYRStage, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    direction = 1
    while time.monotonic() < deadline:
        try:
            zolix.wait_idle(timeout_s=5.0)
            zolix.move_rel_um(150.0 * direction, 0.0)
            direction = -direction
        except Exception as exc:  # noqa: BLE001
            print(f"  zolix jog error: {exc}")
            break
    try:
        zolix.wait_idle(timeout_s=5.0)
        zolix.stop()
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    settings = Settings.load()
    cam = PhasedCam({"smartcam": {"apply_defaults": True, "settle_s": 2.0}})
    cam.connect()
    cam.start()

    # stages (best-effort — the probe still works without them)
    focus = zolix = None
    try:
        focus = FocusStageDriver(settings.device("focus"))
        focus.connect()
        print("focus stage: connected")
    except Exception as exc:  # noqa: BLE001
        print(f"focus stage: unavailable ({exc})")
    try:
        zolix = ZolixXYRStage(settings.device("zolix"))
        zolix.connect()
        print("zolix stage: connected")
    except Exception as exc:  # noqa: BLE001
        print(f"zolix stage: unavailable ({exc})")

    def phase(name: str, fn=None) -> None:
        print(f"== {name} ({PHASE_S:.0f}s) ==", flush=True)
        cam.phase = name
        if fn is not None:
            fn()
        time.sleep(PHASE_S)

    try:
        phase("idle_1")
        if focus is not None:
            phase("focus_jog", lambda: jog_focus(focus, PHASE_S))
        if zolix is not None:
            phase("zolix_jog", lambda: jog_zolix(zolix, PHASE_S))
        phase("idle_2")
        # H2 test: AWB off + fixed color temperature, then jog again
        cam.stop()
        cam.set_property("white_balance", 0)
        cam.set_property("color_temperature", 5500)
        cam.start()
        time.sleep(0.5)
        if focus is not None:
            phase("focus_jog_awb_off", lambda: jog_focus(focus, PHASE_S))
        phase("idle_3")
    finally:
        cam.stop()
        # restore the ZEN-default state
        for name, value in (("white_balance", 1), ("exposure_us", 20000.0),
                            ("gain", 4.0)):
            try:
                cam.set_property(name, value)
            except Exception:  # noqa: BLE001
                pass
        cam.disconnect()
        for dev in (focus, zolix):
            if dev is not None:
                try:
                    dev.disconnect()
                except Exception:  # noqa: BLE001
                    pass

    print("\n--- per-phase summary ---")
    for ph in ("idle_1", "focus_jog", "zolix_jog", "idle_2",
               "focus_jog_awb_off", "idle_3"):
        report(cam.events, ph)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
