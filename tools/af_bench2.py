"""Adaptive v2 (probe-driven) hardware bench (5× objective, user approved
— see docs/PLAN.md handoff #10).

Protocol (all motion bounded: window ± span/2 around the arm position;
defocus moves AWAY from the sample only — the user's established rule):
  1. camera health gate (EMI-quiet window)
  2. vision pre-flight: a stationary frame saved to docs/bench/
     vision_preflight.png + the current exposure/gain/luma — the operator
     feeds it to the vision MCP and re-runs with --exposure/--gain
     overrides until the field is clear (sample-reach area, textured)
  3. probe-accuracy runs at +200/+400/+600 steps defocus: success +
     timing + the phase timeline must show the probe (ph7) then the
     coarse pass (ph4)
  4. near-focus direct entry (no defocus): timeline probe -> hill, NO
     coarse pass, shorter elapsed
  5. boundary-salvage reproduction: a small search window puts the true
     peak just inside the bound (the v1 hit-window failing geometry) —
     v2 must land it (salvage log line or a clean lock-on)
  6. repeatability ×N at the default defocus
  7. classic A/B ×M
  8. one abort mid-run (Esc-equivalent): stage idle < 1.5 s
  9. camera exposure/gain readout before/after — bit-identical

Usage:
    python tools/af_bench2.py --preflight-only
    python tools/af_bench2.py --exposure 40000 --gain 20
    python tools/af_bench2.py --runs 3 --skip-salvage
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from talos.config import Settings  # noqa: E402
from talos.cv.af_adaptive import AdaptiveAutofocusControllerV2  # noqa: E402
from talos.cv.af_math import build_config  # noqa: E402
from talos.cv.autofocus import AutofocusConfig, AutofocusController  # noqa: E402
from talos.cv.frame_slot import LatestFrameSlot  # noqa: E402
from talos.hal.base import DeviceError  # noqa: E402
from talos.hal.devices.camera import camera_chain  # noqa: E402
from talos.hal.devices.focus import FocusStageDriver  # noqa: E402
from talos.logging_setup import setup_logging  # noqa: E402
from tools.af_bench import (  # noqa: E402
    BENCH_DIR,
    CameraPump,
    _safe,
    tick,
    wait_until_healthy,
)

PHASES = {4: None, 5: None, 6: None, 7: None, 1: None, 2: None, 3: None}


def make_config2(settings, strategy: str, base_speed: float | None = None,
                 window_um: float | None = None) -> AutofocusConfig:
    """Mirror AutofocusService._make_config (+ the v2 keys) with a window
    override for the boundary-salvage reproduction."""
    rows = settings.get("objectives") or []
    row = dict(rows[0])  # 5× — the mounted objective (user-approved bench)
    if window_um is not None:
        row["window_um"] = window_um
    row["backlash_um"] = float(settings.device("focus").get(
        "backlash_um", row.get("backlash_um", 0.0)))
    af_cfg = dict(settings.section("autofocus"))
    af_cfg["af_exposure_us"] = float(settings.device("camera").get(
        "exposure_us", af_cfg.get("af_exposure_us", 20000)))
    if base_speed is not None:
        af_cfg["coarse_speed_base_um_s"] = base_speed
    na_min = min((float(r["na"]) for r in rows if r.get("na")),
                 default=None)
    kwargs, warnings = build_config(row, float(
        settings.device("focus").get("um_per_step", 0.2)), af_cfg,
        na_min=na_min)
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
        probe_step_steps=int(af_cfg.get("probe_step_steps", 0)),
        probe_peak_ratio=float(af_cfg.get("probe_peak_ratio", 0.15)),
        probe_min_slope=float(af_cfg.get("probe_min_slope", 0.10)),
        guard_samples=int(af_cfg.get("guard_samples", 4)),
        guard_drop_ratio=float(af_cfg.get("guard_drop_ratio", 0.15)),
        coarse_early_stop_samples=int(af_cfg.get("coarse_early_stop_samples", 2)),
        stage2_retries=int(af_cfg.get("stage2_retries", 1)),
        probe_score_floor_ratio=float(
            af_cfg.get("probe_score_floor_ratio", 0.3)),
        probe_cluster_center_ratio=float(
            af_cfg.get("probe_cluster_center_ratio", 0.5)),
        probe_cluster_side_ratio=float(
            af_cfg.get("probe_cluster_side_ratio", 0.4)),
        probe_cluster_min_score=float(
            af_cfg.get("probe_cluster_min_score", 0.0)),
        coarse_direction=int(af_cfg.get("coarse_direction", 0)),
        guard_fit_samples=int(af_cfg.get("guard_fit_samples", 6)),
        guard_sigma=float(af_cfg.get("guard_sigma", 2.0)),
        coarse_curv_window=int(af_cfg.get("coarse_curv_window", 5)),
        mode="AF_S")
    roi = af_cfg.get("default_roi_norm")
    if isinstance(roi, (list, tuple)) and len(roi) == 4:
        cfg.roi_norm = tuple(float(v) for v in roi)
    return cfg


def run_one2(ctrl_cls, focus, frame_reader, cfg, center: int,
             defocus_steps: int, snapshot_dir: Path, tag: str, pump=None):
    """Defocus away, run the given controller, report the phase timeline
    (incl. ph7 probe) and save both curves + the landed snapshot."""
    logs: list[str] = []
    progress: list[tuple[float, int, float]] = []
    ctrl = ctrl_cls(focus, frame_reader)
    ctrl.sig_log.connect(logs.append)
    t0 = time.monotonic()
    ctrl.sig_progress.connect(
        lambda frac, phase, score, pos: progress.append(
            (time.monotonic() - t0, phase, float(pos))))
    focus.move_rel(defocus_steps, speed=cfg.max_speed)
    focus.wait_idle(timeout_s=60.0)
    tick(0.3)  # first fresh frames at the defocused position
    t0 = time.monotonic()
    result = ctrl.run(center=center, cfg=cfg)
    elapsed = time.monotonic() - t0
    if pump is not None:
        _safe(f"      pump: {pump.rate_since(t0)} frames during the run")
    for m in logs:
        _safe(f"      {m}")
    if progress:
        first: dict[int, tuple] = dict(PHASES)
        last: dict[int, tuple] = dict(PHASES)
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
        if getattr(result, "coarse_curve", []):
            out = snapshot_dir / f"{tag}_lowfreq.csv"
            with open(out, "w", encoding="utf-8") as fh:
                fh.write("position,score\n")
                for pos, score in result.coarse_curve:
                    fh.write(f"{pos},{score:.3f}\n")
        if result.success:
            item = frame_reader.read()
            if item is not None:
                frame, _meta = item
                snapshot = snapshot_dir / f"{tag}_landed.png"
                cv2.imwrite(str(snapshot),
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return elapsed, result, snapshot


def capture_preflight(frame_reader) -> None:
    """Save a stationary frame for the vision-MCP pre-flight and report
    its luma stats + the camera's exposure/gain."""
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    tick(2.0)
    item = frame_reader.read()
    if item is None:
        _safe("FATAL: no camera frame for the pre-flight")
        return
    frame, _meta = item
    out = BENCH_DIR / "vision_preflight.png"
    cv2.imwrite(str(out), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    luma = float(frame.mean())
    _safe(f"  pre-flight frame: {out}  shape {frame.shape} "
          f"luma {luma:.1f}/255")


def main() -> int:
    parser = argparse.ArgumentParser(description="Adaptive v2 AF bench (5x)")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--classic-runs", type=int, default=3)
    parser.add_argument("--defocus", type=int, default=150,
                        help="default defocus steps (+ = away from sample)")
    parser.add_argument("--probe-defocus", type=str, default="200,400,600",
                        help="comma list of away-defocus steps for the "
                             "probe-accuracy runs")
    parser.add_argument("--salvage-window-um", type=float, default=60.0,
                        help="search window for the boundary-salvage "
                             "reproduction (puts the peak near the bound)")
    parser.add_argument("--salvage-defocus", type=int, default=125,
                        help="defocus for the salvage reproduction")
    parser.add_argument("--skip-near-focus", action="store_true")
    parser.add_argument("--skip-salvage", action="store_true")
    parser.add_argument("--skip-abort", action="store_true")
    parser.add_argument("--preflight-only", action="store_true",
                        help="capture the vision pre-flight frame and exit")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="skip the pre-flight capture (retries — the "
                             "first run already saved it)")
    parser.add_argument("--health-tries", type=int, default=5,
                        help="camera health-gate attempts before giving up "
                             "(each waits 30*(attempt+1) s)")
    parser.add_argument("--exposure", type=float, default=None,
                        help="camera exposure override (us) — vision-tuned")
    parser.add_argument("--gain", type=float, default=None,
                        help="camera gain override — vision-tuned")
    parser.add_argument("--base-speed", type=float, default=None)
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
          f"af_speed_multiplier {row.get('af_speed_multiplier', '?')}")

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
    # DTR-reset bus hiccup: re-init the camera pipeline (see af_bench).
    try:
        camera.set_property("resolution", 1)
        tick(0.5)
    except Exception as exc:  # noqa: BLE001
        print(f"camera pipeline re-init failed: {exc}")
    show_rate("after pipeline re-init")

    props = camera.get_properties() if hasattr(camera, "get_properties") else {}
    exposure_before = props.get("exposure_us", props.get("exposure", None))
    gain_before = props.get("gain", None)
    if args.exposure is not None:
        camera.set_property("exposure_us", args.exposure)
        tick(0.5)
    if args.gain is not None:
        camera.set_property("gain", args.gain)
        tick(0.5)
    props = camera.get_properties() if hasattr(camera, "get_properties") else {}
    exposure_active = props.get("exposure_us", props.get("exposure", None))
    gain_active = props.get("gain", None)
    print(f"camera exposure {exposure_active} us  gain {gain_active}")

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    arm = focus.get_status().pos

    if not wait_until_healthy(pump, tries=args.health_tries):
        print("FATAL: camera delivery stays noisy — EMI hunt needed "
              "(yudian off / driver cutback off)")
        return 1
    if not args.skip_preflight:
        capture_preflight(frame_reader)
    if args.preflight_only:
        camera.stop()
        camera.disconnect()
        focus.disconnect()
        return 0

    cfg_adaptive = make_config2(settings, "adaptive", args.base_speed)
    cfg_classic = make_config2(settings, "classic", args.base_speed)
    um_per_step = float(settings.device("focus").get("um_per_step", 0.2))
    print(f"adaptive: coarse_speed {cfg_adaptive.coarse_speed} sps "
          f"({cfg_adaptive.coarse_speed * um_per_step:.1f} µm/s), "
          f"hill v_cap {cfg_adaptive.hill_v_cap}, v_min {cfg_adaptive.hill_v_min}, "
          f"fine_window {cfg_adaptive.fine_window_steps}, "
          f"span {cfg_adaptive.span_steps}, probe_delta "
          f"{cfg_adaptive.probe_step_steps or 3 * cfg_adaptive.coarse_step}")

    # ---- probe accuracy (away defocus, direction −) -----------------------
    print("\n=== probe accuracy (defocus away) ===")
    for d in [int(v) for v in args.probe_defocus.split(",") if v.strip()]:
        center = focus.get_status().pos
        elapsed, result, snap = run_one2(
            AdaptiveAutofocusControllerV2, focus, frame_reader, cfg_adaptive,
            center, d, BENCH_DIR, f"probe_{d}", pump=pump)
        status = "OK" if result.success else f"FAIL: {result.message}"
        print(f"  defocus +{d}: {elapsed:5.2f} s  best {result.best_position}  "
              f"{status}")
        if snap:
            print(f"    snapshot: {snap}")
        tick(0.5)

    # ---- near-focus direct entry ------------------------------------------
    if not args.skip_near_focus:
        print("\n=== near-focus direct entry (no defocus) ===")
        center = focus.get_status().pos
        elapsed, result, snap = run_one2(
            AdaptiveAutofocusControllerV2, focus, frame_reader, cfg_adaptive,
            center, 0, BENCH_DIR, "near_focus", pump=pump)
        status = "OK" if result.success else f"FAIL: {result.message}"
        print(f"  near-focus: {elapsed:5.2f} s  best {result.best_position}  "
              f"{status} (expect probe->hill, NO coarse pass)")
        tick(0.5)

    # ---- boundary-salvage reproduction ------------------------------------
    if not args.skip_salvage:
        print("\n=== boundary-salvage reproduction (peak near the bound) ===")
        cfg_salvage = make_config2(settings, "adaptive", args.base_speed,
                                   window_um=args.salvage_window_um)
        center = focus.get_status().pos
        elapsed, result, snap = run_one2(
            AdaptiveAutofocusControllerV2, focus, frame_reader, cfg_salvage,
            center, args.salvage_defocus, BENCH_DIR, "salvage", pump=pump)
        status = "OK" if result.success else f"FAIL: {result.message}"
        print(f"  salvage: {elapsed:5.2f} s  best {result.best_position}  "
              f"{status} (v1 failed this geometry with the hit-window message)")
        tick(0.5)

    # ---- repeatability ----------------------------------------------------
    positions: list[int] = []
    times: list[float] = []
    print(f"\n=== adaptive v2 AF-S x{args.runs} (defocus {args.defocus}) ===")
    for i in range(args.runs):
        center = focus.get_status().pos
        elapsed, result, snap = run_one2(
            AdaptiveAutofocusControllerV2, focus, frame_reader, cfg_adaptive,
            center, args.defocus, BENCH_DIR, f"v2_{i + 1}", pump=pump)
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
        print(f"v2 repeatability: spread {spread} steps "
              f"({spread * um_per_step:.2f} µm) over {positions}")

    # ---- classic A/B ------------------------------------------------------
    classic_pos: list[int] = []
    print(f"\n=== classic A/B x{args.classic_runs} ===")
    for i in range(args.classic_runs):
        center = focus.get_status().pos
        elapsed, result, snap = run_one2(
            AutofocusController, focus, frame_reader, cfg_classic,
            center, args.defocus, BENCH_DIR, f"classic_{i + 1}", pump=pump)
        status = "OK" if result.success else f"FAIL: {result.message}"
        print(f"  run {i + 1}: {elapsed:5.2f} s  best {result.best_position}  "
              f"{status}")
        classic_pos.append(result.best_position)
        tick(0.5)
    if positions and classic_pos:
        print(f"v2 vs classic: last positions differ by "
              f"{abs(positions[-1] - classic_pos[-1])} steps")

    # ---- abort mid-run ----------------------------------------------------
    if not args.skip_abort:
        print("\n=== abort mid-run ===")
        focus.move_rel(args.defocus, speed=cfg_adaptive.max_speed)
        focus.wait_idle(timeout_s=60.0)
        tick(0.3)
        center = focus.get_status().pos - args.defocus
        ctrl = AdaptiveAutofocusControllerV2(focus, frame_reader)
        result_box: dict = {}
        t0 = time.monotonic()
        thread = threading.Thread(
            target=lambda: result_box.setdefault(
                "result", ctrl.run(center=center, cfg=cfg_adaptive)),
            daemon=True)
        thread.start()
        tick(0.4)  # mid-probe (tick keeps the camera flowing)
        ctrl.request_abort()
        thread.join(timeout=30.0)
        elapsed = time.monotonic() - t0
        result = result_box.get("result")
        print(f"  abort: {elapsed:5.2f} s total  "
              f"aborted={getattr(result, 'aborted', '?')} "
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

    # ---- restore the arm position ----------------------------------------
    focus.move_abs(arm, speed=cfg_adaptive.max_speed)
    focus.wait_idle(timeout_s=60.0)
    print(f"restored arm position: {focus.get_status().pos}")

    camera.stop()
    camera.disconnect()
    focus.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
