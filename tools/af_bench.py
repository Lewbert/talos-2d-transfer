"""Adaptive-autofocus hardware verification bench (5× objective, user
approved for unattended runs — see docs/PLAN.md handoff).

Protocol (all motion bounded: window ± span/2 around the arm position;
defocus moves AWAY from the sample only):
  1. settings sanity (v4 migration: speed_multiplier present)
  2. adaptive AF-S × N with per-run timing, defocusing +DEFOCUS steps
     before each run (repeatability = spread of the landed positions)
  3. classic A/B × M with the same protocol
  4. one abort mid-coarse (Esc-equivalent): stage idle < 1.5 s
  5. camera exposure/gain readout before/after — must be bit-identical
  6. firmware CFG? round trip (new driver get_config)
  7. snapshots of the landed field for vision-MCP focus-quality checks

Usage:
    python tools/af_bench.py                 # full protocol
    python tools/af_bench.py --runs 3 --classic-runs 1
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from talos.config import Settings  # noqa: E402
from talos.cv.af_adaptive import AdaptiveAutofocusControllerV1  # noqa: E402
from talos.cv.af_math import build_config, um_to_steps  # noqa: E402
from talos.cv.autofocus import AutofocusConfig, AutofocusController  # noqa: E402
from talos.cv.frame_slot import LatestFrameSlot  # noqa: E402
from talos.hal.base import DeviceError  # noqa: E402
from talos.hal.devices.camera import camera_chain  # noqa: E402
from talos.hal.devices.focus import FocusStageDriver  # noqa: E402
from talos.logging_setup import setup_logging  # noqa: E402

BENCH_DIR = Path(__file__).resolve().parent.parent / "docs" / "bench"


def _safe(msg: str) -> None:
    """ASCII-safe console print (cp1252 consoles choke on →/✓ etc.)."""
    print(msg.encode("ascii", "backslashreplace").decode("ascii"))


class CameraPump(threading.Thread):
    """Camera worker: greedy fetch → LatestFrameSlot, exactly like the
    app's camera proxy. The SmartCam delivery is gated by the MAIN
    thread's wake cadence (message-pump-like DLL coupling — hardware-
    verified: a main thread sleeping ≥1 s drops the pump to ~3 fps; a
    10-20 ms tick keeps ~21 fps). All bench waits must therefore tick the
    main thread with short sleeps."""

    def __init__(self, camera, slot):
        super().__init__(daemon=True)
        self._camera = camera
        self._slot = slot
        self._stop_event = threading.Event()
        self._seq = 0
        self.write_times: list[float] = []

    def run(self):
        while not self._stop_event.is_set():
            frame = self._camera.fetch(timeout_ms=500.0)
            if frame is None:
                continue
            capture_time = getattr(self._camera, "capture_time", None)
            t = capture_time() if capture_time is not None \
                else time.monotonic()
            self._slot.write(frame, t, self._seq)
            self.write_times.append(time.monotonic())
            self._seq += 1

    def stop(self):
        self._stop_event.set()
        self.join(timeout=5.0)

    def rate_since(self, t0: float) -> int:
        writes = [t for t in self.write_times if t >= t0]
        return len(writes)

    def healthy(self, window_s: float = 2.0, max_gap_s: float = 0.2) -> bool:
        """Delivery health: at least ~half the expected 21 fps with no
        long gaps in the last ``window_s`` (a USB re-enumeration — focus
        DTR reset / serial heal — stalls the shared bus and starves the
        camera; runs during such windows fail spuriously)."""
        now = time.monotonic()
        recent = [t for t in self.write_times if now - window_s <= t <= now]
        if len(recent) < 8:
            return False
        gaps = [b - a for a, b in zip(recent, recent[1:])]
        return max(gaps) <= max_gap_s


def tick(wait_s: float) -> None:
    """Main-thread-ticking sleep: the SmartCam delivery needs the main
    thread to wake FREQUENTLY (message-pump-like DLL coupling — a 20 ms
    tick is already too slow; 10 ms keeps ~20 fps; see CameraPump)."""
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        time.sleep(0.01)


def wait_until_healthy(pump, tries: int = 5) -> bool:
    for attempt in range(tries):
        if pump.healthy():
            return True
        now = time.monotonic()
        recent = [t for t in pump.write_times if now - 2.0 <= t <= now]
        gaps = [b - a for a, b in zip(recent, recent[1:])]
        _safe(f"  camera delivery noisy (EMI window) — waiting "
              f"{30 * (attempt + 1)} s [diag: {len(recent)} frames/2 s, "
              f"max gap {max(gaps) * 1000:.0f} ms]" if gaps
              else f"  camera delivery noisy — waiting "
                   f"{30 * (attempt + 1)} s [diag: {len(recent)} frames/2 s]")
        tick(30 * (attempt + 1))
    return pump.healthy()


def make_config(settings, strategy: str, base_speed: float | None = None) \
        -> AutofocusConfig:
    """Mirror AutofocusService._make_config for the active objective."""
    rows = settings.get("objectives") or []
    row = dict(rows[0])  # 5× — the mounted objective (user-approved bench)
    row["backlash_um"] = float(settings.device("focus").get(
        "backlash_um", row.get("backlash_um", 0.0)))
    af_cfg = dict(settings.section("autofocus"))
    af_cfg["af_exposure_us"] = float(settings.device("camera").get(
        "exposure_us", af_cfg.get("af_exposure_us", 20000)))
    if base_speed is not None:
        af_cfg["coarse_speed_base_um_s"] = base_speed
    kwargs, warnings = build_config(row, float(
        settings.device("focus").get("um_per_step", 0.2)), af_cfg)
    for warning in warnings:
        print(f"  config warning: {warning}")
    cfg = AutofocusConfig(
        **kwargs,
        metric=af_cfg.get("metric", "tenengrad"),
        strategy=strategy,
        coarse_metric=af_cfg.get("coarse_metric", "brenner_k"),
        coarse_metric_k=int(af_cfg.get("coarse_metric_k", 8)),
        coarse_bin=int(af_cfg.get("coarse_bin", 2)),
        hill_ratio=float(af_cfg.get("hill_ratio", 0.6)),
        hill_early_stop_samples=int(af_cfg.get("hill_early_stop_samples", 2)),
        lock_samples_required=int(af_cfg.get("lock_samples_required", 4)),
        lock_score_frac=float(af_cfg.get("lock_score_frac", 0.85)),
        stationary_points=int(af_cfg.get("stationary_points", 5)),
        stop_accel_sps2=int(af_cfg.get("stop_accel_sps2", 20000)),
        stop_latency_s=float(af_cfg.get("stop_latency_s", 0.05)),
        stop_safety_steps=int(af_cfg.get("stop_safety_steps", 10)),
        quality_threshold=float(af_cfg.get("quality_threshold", 0.3)),
        timeout_s=float(af_cfg.get("timeout_s", 180.0)),
        freshness_ms=float(af_cfg.get("freshness_ms", 400.0)),
        interp_max_gap_ms=float(af_cfg.get("interp_max_gap_ms", 300.0)),
        coarse_poll_s=float(af_cfg.get("coarse_poll_s", 0.05)),
        stage_speed=int(af_cfg.get("stage_speed", 2000)),
        fine_wait_timeout_s=float(af_cfg.get("fine_wait_timeout_s", 2.0)),
        settle_frames=int(af_cfg.get("settle_frames", 2)),
        peak_prominence=float(af_cfg.get("peak_prominence", 0.15)),
        fail_on_edge_peak=bool(af_cfg.get("fail_on_edge_peak", True)),
        early_stop_ratio=float(af_cfg.get("early_stop_ratio", 0.5)),
        early_stop_samples=int(af_cfg.get("early_stop_samples", 3)),
        early_stop_rise=float(af_cfg.get("early_stop_rise", 0.2)),
        mode="AF_S")
    roi = af_cfg.get("default_roi_norm")
    if isinstance(roi, (list, tuple)) and len(roi) == 4:
        cfg.roi_norm = tuple(float(v) for v in roi)
    return cfg


def run_one(ctrl_cls, focus, frame_reader, cfg, center: int,
            defocus_steps: int, snapshot_dir: Path, tag: str,
            pump=None):
    """Defocus away, run, return (elapsed_s, result, landed_snapshot)."""
    logs: list[str] = []
    progress: list[tuple[float, int, float]] = []
    ctrl = ctrl_cls(focus, frame_reader)
    ctrl.sig_log.connect(lambda m: logs.append(m))
    t0 = time.monotonic()
    ctrl.sig_progress.connect(
        lambda frac, phase, score, pos: progress.append(
            (time.monotonic() - t0, phase, float(pos))))
    focus.move_rel(defocus_steps, speed=cfg.max_speed)
    focus.wait_idle(timeout_s=60.0)
    tick(0.3)  # let the first fresh frames arrive at the defocused pos
    t0 = time.monotonic()
    result = ctrl.run(center=center, cfg=cfg)
    elapsed = time.monotonic() - t0
    if pump is not None:
        _safe(f"      pump: {pump.rate_since(t0)} frames during the run")
    for m in logs:
        _safe(f"      {m}")
    if progress:
        first = {4: None, 5: None, 6: None, 1: None, 2: None, 3: None}
        last = {4: None, 5: None, 6: None, 1: None, 2: None, 3: None}
        for t, phase, pos in progress:
            first.setdefault(phase, (t, pos))
            last[phase] = (t, pos)
        parts = []
        for phase in sorted(first):
            if first[phase] is None:
                continue
            ft, fp = first[phase]
            lt, lp = last[phase]
            parts.append(f"ph{phase} t{ft:.2f}-{lt:.2f}s "
                         f"pos {fp:.0f}->{lp:.0f}")
        _safe("      timeline: " + " | ".join(parts))
    snapshot = None
    if snapshot_dir is not None:
        if result.curve:
            out = snapshot_dir / f"{tag}_curve.csv"
            with open(out, "w", encoding="utf-8") as fh:
                fh.write("position,score\n")
                for pos, score in result.curve:
                    fh.write(f"{pos},{score:.3f}\n")
        if result.success:
            import cv2

            item = frame_reader.read()
            if item is not None:
                frame, _meta = item
                snapshot = snapshot_dir / f"{tag}_landed.png"
                cv2.imwrite(str(snapshot),
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return elapsed, result, snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Adaptive AF bench (5×)")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--classic-runs", type=int, default=3)
    parser.add_argument("--defocus", type=int, default=150,
                        help="defocus steps before each run (+ = away from "
                             "the sample)")
    parser.add_argument("--base-speed", type=float, default=None,
                        help="override coarse_speed_base_um_s (µm/s) — "
                             "bench tuning knob")
    parser.add_argument("--skip-abort", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=False)
    import logging

    class _PrintHandler(logging.Handler):
        def emit(self, record):
            _safe(f"      [focus-log {record.levelname}] {record.getMessage()}")

    logging.getLogger("talos.hal.devices.focus").addHandler(_PrintHandler())
    settings = Settings.load()
    rows = settings.get("objectives") or []
    row = rows[0]
    print(f"objective: {row.get('name')}  NA {row.get('na')}  "
          f"speed_multiplier {row.get('af_speed_multiplier', row.get('speed_multiplier'))}")

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
        print("FATAL: no camera (ZEN running? app running?)")
        return 1
    try:
        camera.set_property("resolution", 1)
    except Exception as exc:  # noqa: BLE001
        print(f"resolution default (set failed: {exc})")
    slot = LatestFrameSlot()
    pump = CameraPump(camera, slot)
    pump.start()
    frame_reader = slot

    def show_rate(label: str) -> None:
        tick(3.0)
        _safe(f"  [diag] {label}: {pump.rate_since(time.monotonic() - 3.0)} "
              f"frames/3s")

    show_rate("after pump start")
    focus = FocusStageDriver(settings.device("focus"))
    focus.connect()
    print(f"focus start position: {focus.get_status().pos} steps")
    print(f"firmware config: {focus.get_config()}")
    show_rate("after focus connect")

    # The focus connect's DTR reset re-enumerates the 16U2 and hiccups the
    # shared USB bus — the camera pipeline goes slow (~3 fps) afterwards
    # and does NOT recover on its own. Re-setting the resolution re-inits
    # the pipeline (documented stop/start path) and restores ~21 fps
    # (hardware-verified: probe connects after a slow bench deliver clean).
    try:
        camera.set_property("resolution", 1)
        tick(0.5)
    except Exception as exc:  # noqa: BLE001
        print(f"camera pipeline re-init failed: {exc}")

    show_rate("after pipeline re-init")
    exposure_before = None
    props = camera.get_properties() if hasattr(camera, "get_properties") else {}
    exposure_before = props.get("exposure_us", props.get("exposure", None))
    gain_before = props.get("gain", None)
    show_rate("after get_properties")

    cfg_adaptive = make_config(settings, "adaptive_v1", args.base_speed)
    cfg_classic = make_config(settings, "classic", args.base_speed)
    um_per_step = float(settings.device("focus").get("um_per_step", 0.2))
    print(f"adaptive: coarse_speed {cfg_adaptive.coarse_speed} sps "
          f"({cfg_adaptive.coarse_speed * um_per_step:.1f} µm/s), "
          f"hill v_cap {cfg_adaptive.hill_v_cap}, v_min {cfg_adaptive.hill_v_min}, "
          f"fine_window {cfg_adaptive.fine_window_steps}, "
          f"span {cfg_adaptive.span_steps}, coarse {cfg_adaptive.coarse_step}, "
          f"fine {cfg_adaptive.fine_step}")
    print(f"classic:   max_speed {cfg_classic.max_speed} sps, "
          f"fine_speed {cfg_classic.fine_speed}, "
          f"fine_window {cfg_classic.fine_window_steps}")

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    arm = focus.get_status().pos

    # ---- adaptive runs --------------------------------------------------
    positions: list[int] = []
    times: list[float] = []
    if not wait_until_healthy(pump):
        print("FATAL: camera delivery stays noisy — EMI hunt needed "
              "(yudian off / driver cutback off)")
        return 1
    print(f"\n=== adaptive AF-S ×{args.runs} (defocus {args.defocus} steps) ===")
    for i in range(args.runs):
        center = focus.get_status().pos
        elapsed, result, snap = run_one(
            AdaptiveAutofocusControllerV1, focus, frame_reader, cfg_adaptive,
            center, args.defocus, BENCH_DIR, f"adaptive_{i + 1}", pump=pump)
        times.append(elapsed)
        status = "OK" if result.success else f"FAIL: {result.message}"
        print(f"  run {i + 1}: {elapsed:5.2f} s  best {result.best_position} "
              f"(load {result.best_position * um_per_step:.1f} µm)  {status}")
        if snap:
            print(f"    snapshot: {snap}")
        positions.append(result.best_position)
        tick(0.5)
    if len(positions) > 1:
        spread = max(positions) - min(positions)
        print(f"adaptive repeatability: spread {spread} steps "
              f"({spread * um_per_step:.2f} µm) over {positions}")

    # ---- classic A/B -----------------------------------------------------
    classic_pos: list[int] = []
    print(f"\n=== classic A/B ×{args.classic_runs} (same protocol) ===")
    for i in range(args.classic_runs):
        center = focus.get_status().pos
        elapsed, result, snap = run_one(
            AutofocusController, focus, frame_reader, cfg_classic,
            center, args.defocus, BENCH_DIR, f"classic_{i + 1}", pump=pump)
        status = "OK" if result.success else f"FAIL: {result.message}"
        print(f"  run {i + 1}: {elapsed:5.2f} s  best {result.best_position}  "
              f"{status}")
        classic_pos.append(result.best_position)
        tick(0.5)
    if positions and classic_pos:
        agree = max(p for p in positions if p) - min(p for p in classic_pos if p)
        print(f"adaptive vs classic: positions agree within "
              f"{max(abs(positions[-1] - classic_pos[-1]), 0)} steps")

    # ---- abort mid-coarse -------------------------------------------------
    if not args.skip_abort:
        print("\n=== abort mid-coarse ===")
        focus.move_rel(args.defocus, speed=cfg_adaptive.max_speed)
        focus.wait_idle(timeout_s=60.0)
        tick(0.3)
        center = focus.get_status().pos - args.defocus
        ctrl = AdaptiveAutofocusControllerV1(focus, frame_reader)
        result_box: dict = {}
        t0 = time.monotonic()
        thread = threading.Thread(
            target=lambda: result_box.setdefault(
                "result", ctrl.run(center=center, cfg=cfg_adaptive)),
            daemon=True)
        thread.start()
        tick(0.4)  # mid-coarse at 750 sps (tick keeps the camera flowing)
        ctrl.request_abort()
        thread.join(timeout=30.0)
        elapsed = time.monotonic() - t0
        result = result_box.get("result")
        print(f"  abort: {elapsed:5.2f} s total  aborted={getattr(result, 'aborted', '?')} "
              f"idle={focus.get_status().is_idle}  "
              f"pos {focus.get_status().pos}")

    # ---- exposure untouched ----------------------------------------------
    props = camera.get_properties() if hasattr(camera, "get_properties") else {}
    exposure_after = props.get("exposure_us", props.get("exposure", None))
    gain_after = props.get("gain", None)
    print(f"\nexposure: before {exposure_before} after {exposure_after} — "
          f"{'UNTOUCHED [OK]' if exposure_before == exposure_after else 'CHANGED [!!]'}")
    print(f"gain:     before {gain_before} after {gain_after} — "
          f"{'UNTOUCHED [OK]' if gain_before == gain_after else 'CHANGED [!!]'}")

    # ---- restore the arm position -----------------------------------------
    focus.move_abs(arm, speed=cfg_adaptive.max_speed)
    focus.wait_idle(timeout_s=60.0)
    print(f"restored arm position: {focus.get_status().pos}")

    camera.stop()
    camera.disconnect()
    focus.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
