"""Camera backend benchmark: connect each candidate backend, measure fps and
capabilities, and write the decision matrix to %APPDATA%\\TALOS\\camera_benchmark.json.

Usage:
    python tools/camera_benchmark.py            # full chain (auto)
    python tools/camera_benchmark.py --backend harvesters
    python tools/camera_benchmark.py --frames 10

Note: the Axiocam needs exclusive access — close Labscope first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from talos.config import Settings  # noqa: E402
from talos.hal.base import DeviceError  # noqa: E402
from talos.hal.devices.camera import camera_chain  # noqa: E402
from talos.logging_setup import setup_logging  # noqa: E402
from talos.paths import get_appdata_dir  # noqa: E402


def bench(camera, n_frames: int) -> dict:
    result = {"backend": camera.__class__.__name__, "device_id": camera.device_id}
    try:
        camera.connect()
        camera.start()
        result["connected"] = True
        result["properties"] = camera.get_properties()
        t0 = time.monotonic()
        frames = 0
        for _ in range(n_frames):
            frame = camera.fetch(timeout_ms=3000.0)
            if frame is not None:
                frames += 1
                shape = frame.shape
            else:
                time.sleep(0.05)
        elapsed = time.monotonic() - t0
        result["frames"] = frames
        result["fps"] = round(frames / elapsed, 2) if elapsed > 0 else 0.0
        result["shape"] = list(shape) if frames else None
        path = Path(get_appdata_dir()) / "benchmark" / f"{result['backend']}_snap.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            camera.snapshot(path, timeout_s=10.0)
            result["snapshot"] = str(path)
        except DeviceError as exc:
            result["snapshot_error"] = str(exc)
    except DeviceError as exc:
        result["connected"] = False
        result["error"] = str(exc)
    finally:
        try:
            camera.stop()
            camera.disconnect()
        except Exception:  # noqa: BLE001
            pass
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Camera backend benchmark")
    parser.add_argument("--backend",
                        choices=["harvesters", "mmcore", "mcam", "smartcam",
                                 "directshow", "manual"],
                        default=None, help="bench one backend (default: auto chain)")
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()

    setup_logging(verbose=False)
    settings = Settings.load()
    cfg = dict(settings.device("camera"))
    if args.backend:
        cfg["backend"] = args.backend

    results = [bench(cam, args.frames) for cam in camera_chain(cfg)]

    out_path = get_appdata_dir() / "camera_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "results": results}, indent=2), encoding="utf-8")

    for r in results:
        state = "OK" if r.get("connected") else "FAIL"
        print(f"[{state}] {r['backend']}: "
              + (f"{r.get('shape')} @ {r.get('fps')} fps, {r.get('frames')}/{args.frames} frames"
                 if r.get("connected") else r.get("error")))
    print(f"report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
