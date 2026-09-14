"""SmartCamApi live verification ladder — the Phase 2 hardware steps.

Each --step opens the camera (no defaults applied), runs a measurement,
saves frames to %APPDATA%\\TALOS\\benchmark\\ and prints fps + chroma
metrics. Run IN ORDER; the ladder's make-or-break is step `exposure`.

Usage (ZEN/Labscope must be closed, camera USB3 connected):
    python tools/smartcam_live.py --step baseline
    python tools/smartcam_live.py --step exposure
    python tools/smartcam_live.py --step color
    python tools/smartcam_live.py --step exposure_sweep
    python tools/smartcam_live.py --step gain_sweep
    python tools/smartcam_live.py --step wb
    python tools/smartcam_live.py --step fullres
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from talos.hal.devices.camera.smartcam_backend import SmartCamCamera  # noqa: E402
from talos.paths import get_appdata_dir  # noqa: E402

OUT = Path(get_appdata_dir()) / "benchmark"
MEASURE_S = 5.0


class RawCaptureCam(SmartCamCamera):
    """Backend subclass that also keeps the raw transfer buffer."""

    def __init__(self, config):
        super().__init__(config)
        self.last_raw: bytes | None = None

    def _decode(self, raw):
        self.last_raw = bytes(raw)
        return super()._decode(raw)


def make_cam() -> RawCaptureCam:
    return RawCaptureCam({"smartcam": {"apply_defaults": False, "settle_s": 2.0}})


def measure(cam: RawCaptureCam, seconds: float = MEASURE_S) -> dict:
    """Collect frames for `seconds`, return fps + luma/chroma stats."""
    cam.start()
    times: list[float] = []
    means: list[float] = []
    chroma: list[tuple[float, float, float]] = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        frame = cam.fetch(timeout_ms=1000.0)
        if frame is None:
            continue
        times.append(time.monotonic())
        means.append(float(frame.mean()))
        chroma.append(tuple(float(frame[:, :, i].std()) for i in range(3)))
    cam.stop()
    intervals = [b - a for a, b in zip(times, times[1:])]
    result = {
        "frames": len(times),
        "fps_mean": round(len(intervals) / sum(intervals), 2) if intervals else 0.0,
        "fps_min": round(1 / max(intervals), 2) if intervals else 0.0,
        "fps_max": round(1 / min(intervals), 2) if intervals else 0.0,
        "luma_mean": round(float(np.mean(means)), 1) if means else None,
        "chroma_std_rgb": [round(float(np.mean([c[i] for c in chroma])), 1)
                           for i in range(3)] if chroma else None,
    }
    return result


def save_frame(cam: RawCaptureCam, name: str) -> Path:
    cam.start()
    frame = None
    deadline = time.monotonic() + 8.0
    while frame is None and time.monotonic() < deadline:
        frame = cam.fetch(timeout_ms=1000.0)
    cam.stop()
    if frame is None:
        raise RuntimeError(f"{name}: no frame")
    import cv2
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{name}.png"
    cv2.imwrite(str(png), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if cam.last_raw is not None:
        np.save(OUT / f"{name}.npy", np.frombuffer(cam.last_raw, dtype=np.uint8))
    return png


def step_baseline(cam: RawCaptureCam) -> None:
    cam.connect()
    print("baseline (camera state as-is — did ZEN's settings persist?)")
    stats = measure(cam)
    print(json.dumps(stats, indent=2))
    save_frame(cam, "live_baseline")


def step_exposure(cam: RawCaptureCam) -> None:
    cam.connect()
    print("before:", json.dumps(measure(cam)))
    print("setting exposure_us = 20000 (ZEN live value)...")
    cam.set_property("exposure_us", 20000.0)
    time.sleep(0.3)
    stats = measure(cam)
    print("after:", json.dumps(stats))
    save_frame(cam, "live_exposure20ms")
    # make-or-break: fps/brightness must have changed vs baseline (~2.5 fps)


def step_color(cam: RawCaptureCam) -> None:
    cam.connect()
    print("applying ZEN AfterInitialize: color_mode=1, exposure 20 ms, gain 4, "
          "white_balance=1 (AWB on)...")
    for name, value in (("color_mode", 1), ("exposure_us", 20000.0),
                        ("gain", 4.0), ("white_balance", 1)):
        try:
            cam.set_property(name, value)
            print(f"  {name} = {value} ok")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name} = {value} FAILED: {exc}")
    time.sleep(0.5)
    stats = measure(cam)
    print(json.dumps(stats, indent=2))
    png = save_frame(cam, "live_color")
    print(f"check {png} — expect COLOR (chroma std >> 10)")
    if stats["chroma_std_rgb"]:
        spread = max(stats["chroma_std_rgb"]) - min(stats["chroma_std_rgb"])
        print(f"channel-std spread: {spread:.1f} (gray stream ~0, color >> 5)")


def sweep(cam: RawCaptureCam, name: str, values: list) -> dict:
    results = []
    for value in values:
        cam.set_property(name, value)
        time.sleep(0.4)
        stats = measure(cam, seconds=3.0)
        results.append({"value": value, **stats})
        print(f"  {name}={value}: luma {stats['luma_mean']} "
              f"fps {stats['fps_mean']}")
    return results


def restore_defaults(cam: RawCaptureCam) -> None:
    """Camera settings PERSIST — leave the camera in the ZEN-default state
    (20 ms, gain 4, AWB on) so later steps/app sessions start clean."""
    for name, value in (("exposure_us", 20000.0), ("gain", 4.0),
                        ("white_balance", 1), ("color_temperature", 5500)):
        try:
            cam.set_property(name, value)
        except Exception as exc:  # noqa: BLE001
            print(f"  restore {name} failed: {exc}")
    print("restored ZEN defaults")


def step_exposure_sweep(cam: RawCaptureCam) -> None:
    cam.connect()
    cam.set_property("gain", 4.0)
    try:
        results = sweep(cam, "exposure_us",
                        [61.0, 100.0, 1000.0, 10000.0, 50000.0, 200000.0, 1000000.0])
        (OUT / "exposure_sweep.json").write_text(json.dumps(results, indent=2))
        lumas = [r["luma_mean"] for r in results]
        print(f"luma monotonic: {lumas == sorted(lumas)}")
    finally:
        restore_defaults(cam)


def step_gain_sweep(cam: RawCaptureCam) -> None:
    cam.connect()
    cam.set_property("exposure_us", 20000.0)
    try:
        results = sweep(cam, "gain", [1.0, 4.0, 8.0, 16.0, 22.0])
        (OUT / "gain_sweep.json").write_text(json.dumps(results, indent=2))
        lumas = [r["luma_mean"] for r in results]
        print(f"luma monotonic: {lumas == sorted(lumas)}")
    finally:
        restore_defaults(cam)


def step_wb(cam: RawCaptureCam) -> None:
    cam.connect()
    cam.set_property("white_balance", 0)  # AWB off -> manual color temp
    try:
        for temp in (3000, 5500, 10000):
            cam.set_property("color_temperature", temp)
            time.sleep(0.4)
            frame = None
            cam.start()
            deadline = time.monotonic() + 8.0
            while frame is None and time.monotonic() < deadline:
                frame = cam.fetch(timeout_ms=1000.0)
            cam.stop()
            if frame is None:
                print(f"  {temp}K: no frame")
                continue
            rb = float(frame[:, :, 0].mean()) / max(float(frame[:, :, 2].mean()), 1e-9)
            print(f"  {temp}K: R/B ratio {rb:.3f}")
    finally:
        restore_defaults(cam)


def step_fullres(cam: RawCaptureCam) -> None:
    """EXPERIMENTAL: request a 2-frame buffer via GetAcquisitionBufferSize
    (the second argument is an IMAGE COUNT, not a resolution mode — the
    earlier 'full-res index 2' theory was a guess; see docs/SMARTCAM_API.md).
    The 208's real maximum is the 4K Resolution=0 mode."""
    import ctypes
    cam.connect()
    dll = cam._dll
    size = ctypes.c_ulonglong(0)
    rc = dll.ApiCam_GetAcquisitionBufferSize(cam._handle, 2, ctypes.byref(size))
    print(f"2-frame buffer size: {size.value} (rc={rc})")
    if rc != 0 or size.value == 0:
        print("2-frame acquisition unavailable")
        return
    buf = (ctypes.c_char * size.value)()
    rc = dll.ApiCam_AcquireSingleImage(cam._handle, size.value, buf)
    if rc != 0:
        print(f"AcquireSingleImage rc={rc} ({cam._error_string(rc)}) — trying sequence")
        rc = dll.ApiCam_StartSequenceAcquisition(cam._handle, 1, size.value, buf)
        if rc != 0:
            print(f"StartSequenceAcquisition rc={rc} ({cam._error_string(rc)})")
            return
        deadline = time.monotonic() + 15.0
        while True:
            rc = dll.ApiCam_GetSequenceImage(cam._handle)
            if rc == 0:
                break
            if rc != 20 or time.monotonic() > deadline:
                print(f"GetSequenceImage rc={rc}")
                dll.ApiCam_AbortAcquisition(cam._handle)
                return
            time.sleep(0.01)
        dll.ApiCam_AbortAcquisition(cam._handle)
    raw = np.frombuffer(buf, dtype=np.uint8, count=size.value).copy()
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "fullres_2frame.npy", raw)
    print(f"saved fullres_2frame.npy ({raw.size} bytes) — run the decode study on it")


STEPS = {
    "baseline": step_baseline,
    "exposure": step_exposure,
    "color": step_color,
    "exposure_sweep": step_exposure_sweep,
    "gain_sweep": step_gain_sweep,
    "wb": step_wb,
    "fullres": step_fullres,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="SmartCamApi live ladder")
    parser.add_argument("--step", choices=STEPS, required=True)
    args = parser.parse_args()
    cam = make_cam()
    try:
        STEPS[args.step](cam)
    finally:
        cam.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
