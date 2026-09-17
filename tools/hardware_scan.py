"""Hardware grid scan (M5 acceptance): serpentine waypoints on the real
Zolix XYR stage with live camera frames.

Usage:
    python tools/hardware_scan.py                      # 3x3, 10% overlap
    python tools/hardware_scan.py --width 1500 --height 800 --overlap 0.1
    Ctrl+C aborts mid-scan (stage stops, dataset closes cleanly).

FOV defaults are the Axiocam-208-at-5x estimates (2 µm pixel / 5x =
0.4 µm/px → 768 x 432 µm) unless --fov-x/--fov-y are given.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from talos.config import Settings  # noqa: E402
from talos.cv.frame_source import CameraFrameSource  # noqa: E402
from talos.cv.scan import GridScanner  # noqa: E402
from talos.hal.base import DeviceError  # noqa: E402
from talos.hal.devices.camera import camera_chain  # noqa: E402
from talos.hal.registry import make_device  # noqa: E402
from talos.logging_setup import setup_logging  # noqa: E402
from talos.models import ScanParams  # noqa: E402
from talos.paths import get_scan_dir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Hardware grid scan")
    parser.add_argument("--width", type=float, default=1400.0)
    parser.add_argument("--height", type=float, default=800.0)
    parser.add_argument("--overlap", type=float, default=0.10)
    parser.add_argument("--fov-x", type=float, default=768.0)
    parser.add_argument("--fov-y", type=float, default=432.0)
    parser.add_argument("--settle-ms", type=int, default=200,
                        help="quiet time between the move and the capture")
    parser.add_argument("--backlash-um", type=float, default=0.0,
                        help="play to take up on every move (0 = off)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    setup_logging(verbose=False)
    settings = Settings.load()

    camera = None
    for candidate in camera_chain(settings.device("camera")):
        try:
            candidate.connect()
            candidate.start()
            camera = candidate
            print(f"camera: {candidate.device_id}")
            break
        except DeviceError as exc:
            print(f"camera backend {candidate.__class__.__name__} failed: {exc}")
    if camera is None:
        print("no camera available — aborting")
        return 1
    # This tool OWNS the camera (no app, no camera worker running), so it
    # sets the mode itself: real-time ops run at 1080p — 4K is laggy with
    # an unstable framerate on the Axiocam 208.
    try:
        camera.set_property("resolution", 1)
        print("camera resolution: 1080p")
    except Exception as exc:  # noqa: BLE001
        print(f"resolution default (set failed: {exc})")

    stage = make_device("zolix", settings.device("zolix"), sim=False)
    stage.connect()
    pos = stage.get_position()
    print(f"stage start: x={pos.x_um:.1f} µm, y={pos.y_um:.1f} µm")

    params = ScanParams(x0_um=pos.x_um, y0_um=pos.y_um,
                        width_um=args.width, height_um=args.height,
                        overlap=args.overlap, serpentine=True, slow_speed=True,
                        settle_ms=args.settle_ms,
                        backlash_um=args.backlash_um)
    out_dir = args.out or (get_scan_dir() / "hardware_scan")
    scanner = GridScanner(stage, CameraFrameSource(camera))
    box: dict = {}

    def worker():
        box["result"] = scanner.run(params, out_dir,
                                    meta={"fov_um": (args.fov_x, args.fov_y),
                                          "objective_id": 0})

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    print("scan running — Ctrl+C aborts")
    try:
        while thread.is_alive():
            thread.join(0.2)
    except KeyboardInterrupt:
        print("\naborting…")
        scanner.request_abort()
        thread.join(timeout=30.0)
    result = box.get("result")
    if result is None:
        return 1
    print(f"scan: {len(result.frames)} frames, aborted={result.aborted}, "
          f"msg={result.message!r}")
    print(f"manifest: {result.manifest_path}")
    if not result.aborted:
        # Return the stage to the scan start (composed fixed-length moves).
        print("returning stage to start position…")
        stage.move_abs_um(pos.x_um, pos.y_um)
        stage.wait_idle(timeout_s=120.0)
        back = stage.get_position()
        print(f"returned to x={back.x_um:.1f} µm, y={back.y_um:.1f} µm")
    camera.stop()
    camera.disconnect()
    stage.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
