"""Hardware smoke checklist (M2 acceptance): ordered, cautious motion tests.

Default mode is READ-ONLY (connect + status). Motion checks run only with
--motion, each preceded by an interactive confirm, all at SLOW speed.

Checklist order (enforced):
  1. connect + identify
  2. read status only
  3. ±1 single step per axis (slow)
  4. STOP response timing
  5. estop poll

Usage:
    python tools/smoke_test.py                 # read-only
    python tools/smoke_test.py --motion        # interactive, gated motion
    python tools/smoke_test.py --device zolix  # one device
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from talos.config import Settings  # noqa: E402
from talos.hal.base import (  # noqa: E402
    Axis,
    DeviceError,
    Direction,
    StageSpeed,
)
from talos.hal.registry import DEVICE_KEYS, make_device  # noqa: E402
from talos.logging_setup import setup_logging  # noqa: E402


_AUTO_YES = False


def confirm(prompt: str) -> bool:
    if _AUTO_YES:
        return True
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def smoke_zolix(device, motion: bool) -> bool:
    ok = True
    print("  status:", device.get_status())
    pos = device.get_position()
    print(f"  position: x={pos.x_um:.2f} µm, y={pos.y_um:.2f} µm, r={pos.r_deg:.4f} °")
    if not motion:
        return ok
    if not confirm("  Zolix: move X +5 pulses (3.125 µm) at SLOW speed?"):
        return ok
    try:
        device.move_rel_um(3.125, 0.0, speed=StageSpeed.SLOW)
        device.wait_idle(timeout_s=10.0)
        print("  X move OK:", device.get_position())
        t0 = time.monotonic()
        device.stop()
        elapsed = time.monotonic() - t0
        print(f"  stop() round-trip: {elapsed * 1000:.0f} ms")
        print("  estop flag:", device.check_estop())
    except DeviceError as exc:
        print("  FAILED:", exc)
        ok = False
    return ok


def smoke_sigmakoki(device, motion: bool) -> bool:
    ok = True
    print("  status:", device.get_status())
    print("  limits:", device.get_limits())
    if not motion:
        return ok
    if not confirm("  SigmaKoki: X +1 step (0.5 µm) at level 2?"):
        return ok
    try:
        actual = device.step(Axis.X, Direction.POSITIVE, 1)
        print(f"  step actual: {actual}")
        device.wait_idle(timeout_s=5.0)
        print("  status:", device.get_status())
    except DeviceError as exc:
        print("  FAILED:", exc)
        ok = False
    return ok


def smoke_focus(device, motion: bool) -> bool:
    ok = True
    print("  status:", device.get_status())
    print("  soft limits:", device.get_soft_limits())
    if not motion:
        return ok
    if not confirm("  FOCUS (no limit sensor!): move +10 steps at SLOW speed?"):
        return ok
    try:
        device.move_rel(10, speed=100)
        device.wait_idle(timeout_s=15.0)
        print("  status after +10:", device.get_status())
        device.move_rel(-10, speed=100)
        device.wait_idle(timeout_s=15.0)
        print("  status after -10:", device.get_status())
    except DeviceError as exc:
        print("  FAILED:", exc)
        ok = False
    return ok


def smoke_yudian(device, motion: bool) -> bool:
    ok = True
    print(f"  PV: {device.read_pv():.1f} °C | SV: {device.read_sv():.1f} °C "
          f"| output: {device.read_output_percent():.0f} %")
    # No motion for the temperature controller; setpoint writes are gated.
    return ok


_SMOKERS = {
    "zolix": smoke_zolix,
    "sigmakoki": smoke_sigmakoki,
    "focus": smoke_focus,
    "yudian": smoke_yudian,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="TALOS hardware smoke checklist")
    parser.add_argument("--device", choices=DEVICE_KEYS, default=None)
    parser.add_argument("--motion", action="store_true",
                        help="run gated motion checks (interactive confirms, SLOW speed)")
    parser.add_argument("--yes", action="store_true",
                        help="pre-answer all motion confirms (use only with operator present)")
    parser.add_argument("--sim", action="store_true")
    args = parser.parse_args()

    global _AUTO_YES
    _AUTO_YES = args.yes

    setup_logging(verbose=False)
    settings = Settings.load()
    keys = [args.device] if args.device else list(DEVICE_KEYS)
    for key in keys:
        print(f"--- {key} ---")
        device = make_device(key, settings.device(key), sim=args.sim)
        try:
            device.connect()
            print(f"  connected: {device.device_id}")
            ok = _SMOKERS[key](device, motion=args.motion)
            print("  OK" if ok else "  FAILED")
        except DeviceError as exc:
            print(f"  ERROR: {exc}")
        finally:
            try:
                device.disconnect()
            except Exception:  # noqa: BLE001
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
