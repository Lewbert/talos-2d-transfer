"""Read-only device console (M1 acceptance): connect + status reads, NO motion.

Usage (from the repo root):
    pwsh -Command "conda activate talos; python tools/dev_console.py"
    pwsh -Command "conda activate talos; python tools/dev_console.py --device zolix"
    pwsh -Command "conda activate talos; python tools/dev_console.py --sim"

Only read commands are issued: get_status / get_position / read_pv /
get_limits / get_soft_limits. No move, no home, no set, no zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from talos.config import Settings  # noqa: E402
from talos.hal.base import DeviceError  # noqa: E402
from talos.hal.registry import DEVICE_KEYS, make_device  # noqa: E402
from talos.logging_setup import setup_logging  # noqa: E402


def probe(key: str, cfg: dict, sim: bool) -> None:
    device = make_device(key, cfg, sim)
    print(f"--- {key} ---")
    try:
        device.connect()
        print(f"  connected: {device.device_id}")
        if hasattr(device, "get_status"):
            print(f"  status: {device.get_status()}")
        if hasattr(device, "get_position"):
            print(f"  position: {device.get_position()}")
        if hasattr(device, "get_limits"):
            print(f"  limits: {device.get_limits()}")
        if hasattr(device, "get_soft_limits"):
            print(f"  soft limits: {device.get_soft_limits()}")
        if hasattr(device, "read_pv"):
            print(f"  PV: {device.read_pv():.1f} °C | SV: {device.read_sv():.1f} °C "
                  f"| output: {device.read_output_percent():.0f} %")
    except DeviceError as exc:
        print(f"  ERROR: {exc}")
    finally:
        try:
            device.disconnect()
        except Exception:  # noqa: BLE001
            pass
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only device probe (no motion)")
    parser.add_argument("--device", choices=DEVICE_KEYS, default=None,
                        help="probe one device only (default: all)")
    parser.add_argument("--sim", action="store_true", help="use simulated devices")
    args = parser.parse_args()

    setup_logging(verbose=False)
    settings = Settings.load()
    keys = [args.device] if args.device else list(DEVICE_KEYS)
    for key in keys:
        probe(key, settings.device(key), sim=args.sim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
