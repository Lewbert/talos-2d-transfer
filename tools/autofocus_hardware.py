"""Hardware autofocus + backlash calibration CLI (tuning companion to the
in-app Autofocus panel; same controller, same camera/focus drivers).

Usage:
    python tools/autofocus_hardware.py                          # step-based defaults
    python tools/autofocus_hardware.py --objective 2            # µm settings from the objectives table
    python tools/autofocus_hardware.py --objective 0 --defocus 200 --mode AF_REFINE
    python tools/autofocus_hardware.py --calibrate-backlash [--store]
    Ctrl+C during a search aborts and stops the stage.

Safety: bounded window (soft-limit clamped), readback verification, and
the physical restrain of the stage itself.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from talos.config import Settings  # noqa: E402
from talos.cv.af_math import build_config  # noqa: E402
from talos.cv.autofocus import (  # noqa: E402
    AutofocusConfig,
    AutofocusController,
)
from talos.cv.backlash_cal import BacklashCalConfig, BacklashCalibrator  # noqa: E402
from talos.cv.frame_slot import LatestFrameSlot  # noqa: E402
from talos.hal.devices.camera import camera_chain  # noqa: E402
from talos.hal.devices.focus import FocusStageDriver  # noqa: E402
from talos.hal.base import DeviceError  # noqa: E402
from talos.logging_setup import setup_logging  # noqa: E402
from talos.paths import get_appdata_dir  # noqa: E402


class _FetchFrameReader:
    """Adapter: camera pump thread → LatestFrameSlot → the controller's
    frame_reader protocol. The on-demand fetch pattern (blocking inside
    the AF loop) starves the SmartCam delivery to ~3 fps and empties the
    interpolation history — the pump keeps the pipeline flowing at ~21 fps
    (hardware-verified)."""

    def __init__(self, camera):
        self._camera = camera
        self._slot = LatestFrameSlot()
        self._seq = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._pump, daemon=True)

    def _pump(self):
        while not self._stop_event.is_set():
            frame = self._camera.fetch(timeout_ms=500.0)
            if frame is None:
                continue
            capture_time = getattr(self._camera, "capture_time", None)
            t = capture_time() if capture_time is not None else time.monotonic()
            self._slot.write(frame, t, self._seq)
            self._seq += 1

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5.0)

    def read_since(self, min_t: float | None = None, min_seq: int = -1):
        return self._slot.read_since(min_t=min_t, min_seq=min_seq)


def _open_camera(settings, no_camera: bool):
    camera = None
    if not no_camera:
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
            print("WARNING: no camera available — running motion-only dry run")
    if camera is not None:
        # Real-time ops run at 1080p — 4K is laggy with unstable framerate
        # on the Axiocam 208 (backend handles the mid-stream switch).
        try:
            camera.set_property("resolution", 1)
            print("camera resolution: 1080p")
        except Exception as exc:  # noqa: BLE001
            print(f"camera resolution default (set failed: {exc})")
    return camera


def _build_config(args, settings) -> AutofocusConfig:
    af_cfg = settings.section("autofocus")
    if args.objective is not None:
        rows = settings.get("objectives") or []
        row = rows[min(args.objective, len(rows) - 1)] if rows else {}
        um_per_step = float(settings.device("focus").get("um_per_step", 0.2))
        kwargs, warnings = build_config(row, um_per_step, af_cfg)
        for warning in warnings:
            print(f"config warning: {warning}")
        print(f"objective: {row.get('name', '?')} — "
              f"{kwargs['coarse_step']} coarse / {kwargs['fine_step']} fine steps, "
              f"span {kwargs['span_steps']} steps @ {kwargs['max_speed']} steps/s")
    else:
        kwargs = dict(coarse_step=args.coarse, fine_step=args.fine,
                      span_steps=args.span, max_speed=args.max_speed)
    # Backlash is a mechanism property (devices.focus.backlash_um) —
    # --backlash-steps overrides it explicitly. build_config's row-derived
    # value is dropped in favour of the mechanism value.
    kwargs.pop("backlash_steps", None)
    focus_cfg = settings.device("focus")
    from talos.cv.af_math import um_to_steps
    backlash = args.backlash_steps if args.backlash_steps else um_to_steps(
        float(focus_cfg.get("backlash_um", 0.0)),
        float(focus_cfg.get("um_per_step", 0.2)))
    cfg = AutofocusConfig(
        **kwargs,
        metric=args.metric,
        backlash_steps=backlash,
        timeout_s=float(af_cfg.get("timeout_s", 180.0)),
        freshness_ms=float(af_cfg.get("freshness_ms", 400.0)),
        coarse_poll_s=float(af_cfg.get("coarse_poll_s", 0.05)),
        fine_wait_timeout_s=float(af_cfg.get("fine_wait_timeout_s", 2.0)),
        settle_frames=int(af_cfg.get("settle_frames", 2)),
        early_stop_ratio=float(af_cfg.get("early_stop_ratio", 0.5)),
        early_stop_samples=int(af_cfg.get("early_stop_samples", 3)),
        mode=args.mode)
    if args.roi is not None:
        cfg.roi_norm = tuple(float(v) for v in args.roi.split(","))
    return cfg


def _save_curve(result, name="autofocus_curve.csv") -> Path:
    out = get_appdata_dir() / name
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("position,score\n")
        for pos, score in result.curve:
            fh.write(f"{pos},{score:.3f}\n")
    print(f"curve: {out}")
    return out


def run_autofocus(args, settings, camera, frame_reader, focus, center) -> int:
    cfg = _build_config(args, settings)
    ctrl = AutofocusController(focus, frame_reader)
    ctrl.sig_log.connect(lambda m: print(f"  {m}"))
    result_box: dict = {}
    thread = threading.Thread(
        target=lambda: result_box.setdefault("result", ctrl.run(center=center, cfg=cfg)),
        daemon=True)
    thread.start()
    print("search running — Ctrl+C aborts")
    try:
        while thread.is_alive():
            thread.join(0.2)
    except KeyboardInterrupt:
        print("\naborting…")
        ctrl.request_abort()
        thread.join(timeout=30.0)
    result = result_box.get("result")
    if result is None:
        print("no result")
        return 1
    print(f"result: success={result.success} aborted={result.aborted} "
          f"phase={result.phase} best={result.best_position} "
          f"score={result.best_score:.1f} msg={result.message!r}")
    _save_curve(result)
    print(f"focus end position: {focus.get_status().pos}")
    if args.snapshot and camera and result.success:
        import cv2

        frame = camera.fetch(timeout_ms=4000.0)
        if frame is not None:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.snapshot), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"snapshot: {args.snapshot}")
    return 0 if (result.success or args.no_camera) else 1


def run_calibration(args, settings, frame_reader, focus, center) -> int:
    """Backlash auto-calibration: two-direction sweeps on a feature-rich
    field. --store writes the result into the objective row in settings."""
    um_per_step = float(settings.device("focus").get("um_per_step", 0.2))
    cal = BacklashCalibrator(focus, frame_reader)
    cal.sig_progress.connect(lambda frac, label: print(f"  {frac * 100:5.0f}% {label}"))
    cal.sig_log.connect(lambda m: print(f"  {m}"))
    result_box: dict = {}
    thread = threading.Thread(
        target=lambda: result_box.setdefault("result", cal.run(
            center=center, cfg=BacklashCalConfig(um_per_step=um_per_step))),
        daemon=True)
    thread.start()
    print("calibration running (~20 s) — use a feature-rich field; Ctrl+C aborts")
    try:
        while thread.is_alive():
            thread.join(0.2)
    except KeyboardInterrupt:
        print("\naborting…")
        cal.request_abort()
        thread.join(timeout=30.0)
    result = result_box.get("result")
    if result is None:
        return 1
    print(f"calibration: success={result.success} msg={result.message!r}")
    if result.success:
        print(f"backlash: {result.backlash_steps} steps = {result.backlash_um} µm")
        if args.store:
            focus_cfg = settings.data.setdefault(
                "devices", {}).setdefault("focus", {})
            focus_cfg["backlash_um"] = result.backlash_um
            focus_cfg["backlash_measured_at"] = time.strftime(
                "%Y-%m-%dT%H:%M:%S")
            settings.save()
            print("stored as the mechanism backlash "
                  "(devices.focus.backlash_um)")
    return 0 if result.success else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Hardware autofocus (bounded)")
    parser.add_argument("--span", type=int, default=600)
    parser.add_argument("--coarse", type=int, default=100)
    parser.add_argument("--fine", type=int, default=10)
    parser.add_argument("--max-speed", type=int, default=500)
    parser.add_argument("--objective", type=int, default=None,
                        help="0-based index into the settings objectives table "
                             "(µm parameters override the step flags)")
    parser.add_argument("--backlash-steps", type=int, default=0,
                        help="measured backlash used by the landing approach")
    parser.add_argument("--roi", type=str, default=None,
                        help="normalized measure area: x,y,w,h (0..1)")
    parser.add_argument("--mode", choices=["AF_S", "AF_REFINE"],
                        default="AF_S", help="AF_REFINE = small re-peak around "
                                             "the current position (AF-C engine)")
    parser.add_argument("--calibrate-backlash", action="store_true",
                        help="measure the mechanism backlash instead of focusing")
    parser.add_argument("--store", action="store_true",
                        help="with --calibrate-backlash: write the result into "
                             "the objective row (--objective, default 0)")
    parser.add_argument("--no-camera", action="store_true",
                        help="motion-only dry run (no camera frames)")
    parser.add_argument("--defocus", type=int, default=0,
                        help="deliberately defocus by N steps before searching "
                             "(tests whether the sweep range exceeds the DOF)")
    parser.add_argument("--center", type=int, default=None,
                        help="absolute search center in steps (default: start position)")
    parser.add_argument("--metric", choices=["laplacian", "tenengrad", "brenner"],
                        default="tenengrad")
    parser.add_argument("--snapshot", type=Path, default=None,
                        help="save a frame at the landed position to this path")
    parser.add_argument("--stale-frame-drop", type=int, default=1,
                        help=argparse.SUPPRESS)  # deprecated: freshness gating
    args = parser.parse_args()

    setup_logging(verbose=False)
    settings = Settings.load()

    camera = _open_camera(settings, args.no_camera)
    frame_reader = _FetchFrameReader(camera) if camera else None
    if frame_reader is not None:
        frame_reader.start()

    focus = FocusStageDriver(settings.device("focus"))
    focus.connect()
    center = focus.get_status().pos
    print(f"focus start position: {center} steps")
    if args.defocus:
        focus.move_rel(args.defocus, speed=args.max_speed)
        focus.wait_idle(timeout_s=60.0)
        # Search centered on the PRE-defocus position: the sharp plane
        # sits at the middle of the window.
        center = focus.get_status().pos - args.defocus
        print(f"defocused by {args.defocus} steps; search centered at {center}")
    if args.center is not None:
        center = args.center
        print(f"search center overridden to {center}")

    try:
        if args.calibrate_backlash:
            return run_calibration(args, settings, frame_reader, focus, center)
        return run_autofocus(args, settings, camera, frame_reader, focus, center)
    finally:
        if frame_reader is not None:
            frame_reader.stop()
        if camera:
            camera.stop()
            camera.disconnect()
        focus.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
