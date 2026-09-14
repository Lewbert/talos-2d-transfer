"""SigmaKoki jog-latency drill — proves the MV ack round trip on hardware.

Measures what one continuous-jog command costs, and prints the RAW first
reply line so the firmware's ack format is verified on this bench.

Why: the driver waited for a bare ``"MV"`` prefix while the firmware
answers ``OK:MV:<axis>:<dir>:<level>``. The ack never matched, so it was
skipped as a stray event line and EVERY jog blocked the worker for the
full serial timeout (0.3 s). STOP was unaffected (it matches any line),
which is why stopping felt snappy and jogging did not.

Motion safety: level 0 (25 steps/s) only, alternating direction, with a
STOP after every command — net displacement ~0. The transfer stage moves
in the sample plane; no defocus motion is involved.

Usage:
    pwsh -Command "conda activate talos; python tools/sigmakoki_latency.py --yes"
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from talos.config import Settings  # noqa: E402
from talos.hal.base import Axis, Direction  # noqa: E402
from talos.hal.registry import make_device  # noqa: E402
from talos.logging_setup import setup_logging  # noqa: E402


def _stats(label: str, samples: list[float]) -> None:
    print(f"  {label:<18} min {min(samples):6.1f} ms | "
          f"median {statistics.median(samples):6.1f} ms | "
          f"max {max(samples):6.1f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5,
                        help="timed move+stop pairs (default 5)")
    parser.add_argument("--yes", action="store_true",
                        help="confirm the motion drill")
    parser.add_argument("--sim", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=False)
    settings = Settings.load()
    cfg = settings.device("sigmakoki")
    print(f"SigmaKoki on {cfg.get('port')} (timeout_s={cfg.get('timeout_s')})")
    if not args.sim and not args.yes:
        print("motion drill: re-run with --yes (level 0, alternating, STOP after each)")
        return 2

    device = make_device("sigmakoki", cfg, args.sim)
    device.connect()
    try:
        print("limits:", device.get_limits())
        print("status:", device.get_status())

        # 1) the raw ack this firmware actually sends
        device._io.send("MV:X:+1:0")
        t0 = time.perf_counter()
        line = device._io.read_line()
        raw_ms = (time.perf_counter() - t0) * 1000
        print(f"raw MV ack: {line!r} after {raw_ms:.1f} ms")
        device.stop()
        device.wait_idle(timeout_s=5.0)

        # 2) the driver's own path (what the input system calls per jog)
        moves: list[float] = []
        stops: list[float] = []
        for i in range(args.runs):
            direction = Direction.POSITIVE if i % 2 == 0 else Direction.NEGATIVE
            t0 = time.perf_counter()
            device.move(Axis.X, direction, 0)
            moves.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            device.stop()
            stops.append((time.perf_counter() - t0) * 1000)
            device.wait_idle(timeout_s=5.0)

        # 3) a second axis, to be sure the ack is not axis-specific
        t0 = time.perf_counter()
        device.move(Axis.Y, Direction.POSITIVE, 0)
        y_ms = (time.perf_counter() - t0) * 1000
        device.stop()
        device.wait_idle(timeout_s=5.0)

        print(f"driver move() cost over {args.runs} runs (level 0, alternating):")
        _stats("move()", moves)
        _stats("stop()", stops)
        print(f"  Y axis move():     {y_ms:6.1f} ms (single sample)")
        print(f"  final position:    {device.get_position()}")
        slow = [m for m in moves if m > 100.0]
        if slow:
            print(f"  SLOW: {len(slow)} move(s) exceeded 100 ms — the ack is not "
                  "being consumed (check the firmware ack format)")
            return 1
        print("  OK: every jog returned well inside the serial timeout")
        return 0
    finally:
        device.stop()
        device.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
