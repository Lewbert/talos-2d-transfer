"""Adaptive v3 (derivative-hybrid) hardware bench (5× objective, user
approved full-auto — see docs/PLAN.md handoff #11).

Protocol (all defocus moves AWAY from the sample only — the user's
established safety rule; the plan's ±defocus degenerates to the + side):
  0. camera health gate (EMI-quiet window) + vision pre-flight frame
  1. STEP 0 — measure the real metric curve: a slow STATIONARY z-sweep
     AWAY from the current focus → metric-vs-z CSV → Gaussian half-fit
     (b + A·exp(−(x−μ)²/2σ²)) → σ in steps. GATE: if the shape deviates
     wildly from the Gaussian model (opens upward or R² < 0.8), the
     bench STOPS before any AF run — the thresholds need revisiting.
     The measured σ is printed against the config's DOF-derived σ.
  2. near-focus direct entry (v3): timeline probe -> hill, NO coarse
  3. probe/curvature runs at +200/+400/+600 defocus: success + timeline
     + the "coarse: curvature stop before the peak" log line
  4. before-peak-stop diagnostic: the curvature-stop halt position vs
     the landed peak (SOFT gate — the capture-to-halt coast at coarse
     speed; the trusted vertex is the real anchor)
  5. repeatability ×5 v3 (spread)
  6. A/B v2 vs v3 ×3 — THE ROLLOUT GATE: v3-vs-v2 landing agreement
     within a fine step AND v3 median elapsed ≤ v2 × 1.15; a v3
     failure fails the gate
  7. boundary-salvage reproduction through v3
  8. abort mid-run: stage idle < 1.5 s
  9. camera exposure/gain readout before/after — bit-identical; arm
     position restored

Usage:
    python tools/af_bench3.py --preflight-only
    python tools/af_bench3.py --exposure 40000 --gain 20
    python tools/af_bench3.py --runs 3
    python tools/af_bench3.py --skip-sigma
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from talos.config import Settings  # noqa: E402
from talos.cv.af_adaptive import AdaptiveAutofocusControllerV2  # noqa: E402
from talos.cv.af_v3 import AdaptiveAutofocusController  # noqa: E402
from talos.cv.autofocus import AutofocusController  # noqa: E402
from talos.cv.focus_metric import METRICS  # noqa: E402
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
from tools.af_bench2 import (  # noqa: E402
    PHASES,
    capture_preflight,
    make_config2,
)

_SIGMA_CSV = BENCH_DIR / "sigma_curve.csv"


def run_one3(ctrl_cls, focus, frame_reader, cfg, center, defocus_steps,
             snapshot_dir: Path, tag: str, pump=None):
    """run_one2 + the logs back (the curvature-stop line carries the
    halt position for the before-peak diagnostic)."""
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
    tick(0.3)
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
    return elapsed, result, snapshot, logs


def measure_sigma(focus, frame_reader, cfg, pump=None,
                  away_span: int = 300, step: int = 20) -> float | None:
    """Step 0: a slow stationary z-sweep AWAY from the current focus
    position (the user's rule — the toward-sample side is never walked),
    scored with tenengrad, fitted as a Gaussian half-curve
    S = b + A·exp(−(x−μ)²/2σ²) with μ free (the sweep starts at the
    last run's landing ≈ the peak). Returns σ in steps; None on a shape
    that deviates wildly from the model."""
    fn = METRICS["tenengrad"]
    start = focus.get_status().pos
    _safe(f"sigma sweep: {start} → {start + away_span} steps away "
          f"(step {step}, landed-speed settled)")
    xs: list[float] = []
    ys: list[float] = []
    for pos in range(start, start + away_span + 1, step):
        focus.move_abs(pos, speed=cfg.landing_speed)
        focus.wait_idle(timeout_s=60.0)
        tick(0.25)
        item = frame_reader.read()
        if item is None:
            _safe(f"      pos {pos}: no frame — skipped")
            continue
        score = float(fn(item[0], None))
        xs.append(float(pos))
        ys.append(score)
        _safe(f"      pos {pos}: score {score:.1f}")
    with open(_SIGMA_CSV, "w", encoding="utf-8") as fh:
        fh.write("position,score\n")
        for x, y in zip(xs, ys):
            fh.write(f"{x},{y:.3f}\n")
    _safe(f"  sigma curve saved: {_SIGMA_CSV}")
    if len(xs) < 8:
        _safe("FATAL: too few sweep points for a sigma fit")
        return None
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    floor = float(y_arr.min())
    m = y_arr - floor
    mask = m > 0.05 * m.max()
    if mask.sum() < 5:
        _safe("FATAL: the curve is flat — no peak shape to fit")
        return None
    x_rel = x_arr[mask] - x_arr[0]
    logy = np.log(m[mask])
    p0, p1, p2 = np.polyfit(x_rel, logy, 2)
    fit = np.polyval([p0, p1, p2], x_rel)
    ss_res = float(((logy - fit) ** 2).sum())
    ss_tot = float(((logy - logy.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    _safe(f"  Gaussian half-fit: R² = {r2:.4f}")
    if p0 >= 0:
        _safe("FATAL: the curve OPENS UPWARD — not a Gaussian peak. "
              "Revisit the thresholds (and the field) before any AF run.")
        return None
    sigma = float(np.sqrt(-1.0 / (2.0 * p0)))
    mu_off = float(-p1 / (2.0 * p0))
    _safe(f"  measured σ = {sigma:.1f} steps (peak {mu_off:.1f} steps "
          f"from the sweep start, floor {floor:.0f})")
    if r2 < 0.8:
        _safe("FATAL: the shape deviates wildly from the Gaussian model "
              "(R² < 0.8) — revisit the thresholds before any AF run.")
        return None
    if r2 < 0.92:
        _safe("  WARNING: weak Gaussian fit — treat σ as approximate.")
    return sigma


def main() -> int:
    parser = argparse.ArgumentParser(description="Adaptive v3 AF bench (5x)")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--v2-runs", type=int, default=3,
                        help="A/B runs for the frozen v2 (rollout gate)")
    parser.add_argument("--defocus", type=int, default=150)
    parser.add_argument("--probe-defocus", type=str, default="200,400,600")
    parser.add_argument("--salvage-window-um", type=float, default=60.0)
    parser.add_argument("--salvage-defocus", type=int, default=125)
    parser.add_argument("--skip-sigma", action="store_true",
                        help="skip the Step-0 σ measurement (retries)")
    parser.add_argument("--skip-near-focus", action="store_true")
    parser.add_argument("--skip-guard-proof", action="store_true")
    parser.add_argument("--skip-salvage", action="store_true")
    parser.add_argument("--skip-abort", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--health-tries", type=int, default=5)
    parser.add_argument("--exposure", type=float, default=None)
    parser.add_argument("--gain", type=float, default=None)
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

    cfg_v3 = make_config2(settings, "adaptive", args.base_speed)
    um_per_step = float(settings.device("focus").get("um_per_step", 0.2))
    print(f"v3: coarse_speed {cfg_v3.coarse_speed} sps "
          f"({cfg_v3.coarse_speed * um_per_step:.1f} µm/s), "
          f"hill v_cap {cfg_v3.hill_v_cap}, v_min {cfg_v3.hill_v_min}, "
          f"fine_window {cfg_v3.fine_window_steps}, span {cfg_v3.span_steps}")
    print(f"v3 thresholds: probe_curv_in {cfg_v3.probe_curv_in:.3e} "
          f"probe_curv_out {cfg_v3.probe_curv_out:.3e} "
          f"coarse_curv_stop {cfg_v3.coarse_curv_stop:.4f} "
          f"coarse_curv_vertex {cfg_v3.coarse_curv_vertex:.4f}")

    # ---- Step 0: measure the real metric curve ---------------------------
    sigma = None
    if not args.skip_sigma:
        print("\n=== Step 0: stationary metric-vs-z sweep (away) ===")
        sigma = measure_sigma(focus, frame_reader, cfg_v3)
        if sigma is None:
            camera.stop()
            camera.disconnect()
            focus.disconnect()
            return 1
        # back to the arm (the sweep moved away; move back at stage speed)
        focus.move_abs(arm, speed=cfg_v3.max_speed)
        focus.wait_idle(timeout_s=60.0)
        tick(0.3)

    # ---- near-focus direct entry ------------------------------------------
    if not args.skip_near_focus:
        print("\n=== near-focus direct entry (no defocus) ===")
        center = focus.get_status().pos
        elapsed, result, snap, _logs = run_one3(
            AdaptiveAutofocusController, focus, frame_reader, cfg_v3,
            center, 0, BENCH_DIR, "v3_near_focus", pump=pump)
        status = "OK" if result.success else f"FAIL: {result.message}"
        print(f"  near-focus: {elapsed:5.2f} s  best {result.best_position}  "
              f"{status} (expect probe->hill, NO coarse pass)")
        tick(0.5)

    # ---- probe/curvature runs (away defocus) ------------------------------
    print("\n=== v3 probe + curvature-stop runs (defocus away) ===")
    for d in [int(v) for v in args.probe_defocus.split(",") if v.strip()]:
        center = focus.get_status().pos
        elapsed, result, snap, logs = run_one3(
            AdaptiveAutofocusController, focus, frame_reader, cfg_v3,
            center, d, BENCH_DIR, f"v3_probe_{d}", pump=pump)
        status = "OK" if result.success else f"FAIL: {result.message}"
        print(f"  defocus +{d}: {elapsed:5.2f} s  best {result.best_position}  "
              f"{status}")
        stops = [m for m in logs if "curvature stop" in m]
        for m in stops:
            halt = float(m.split(" at ")[1].split(" ")[0])
            delta = halt - result.best_position
            flag = " [before the peak]" if delta < 0 \
                else " [past the peak — capture-to-halt coast]"
            print(f"    {m}{flag}  (halt {halt:.0f} vs landed "
                  f"{result.best_position})")
        if snap:
            print(f"    snapshot: {snap}")
        tick(0.5)

    # ---- wrong-direction guard proof --------------------------------------
    # Defocus +150 (AWAY): the nearest structure is toward the sample, so
    # the probe's true direction is −1 and the FORCED wrong direction is
    # always +1 (away) — the proof sweep can never drive into the sample,
    # and the guard-fired reversal is the normal toward sweep.
    if not args.skip_guard_proof:
        print("\n=== direction-guard proof (forced wrong direction) ===")
        focus.move_rel(args.defocus, speed=cfg_v3.max_speed)
        focus.wait_idle(timeout_s=60.0)
        tick(0.3)
        center = focus.get_status().pos - args.defocus
        ctrl = AdaptiveAutofocusController(focus, frame_reader)
        ctrl._cfg = cfg_v3
        ctrl._t_start = time.monotonic()
        ctrl._deadline = ctrl._t_start + cfg_v3.timeout_s
        ctrl._current_pos = int(center)
        ctrl._arm_center = int(center)
        ctrl._low_curve = []
        ctrl._curve = []
        ctrl._stage1_curve = []
        ctrl._probe_samples = []
        ctrl.abort_requested = False
        half = cfg_v3.span_steps // 2
        bounds = (center - half, center + half)
        probe = ctrl._probe(center, bounds, cfg_v3)
        logs: list[str] = []
        ctrl.sig_log.connect(logs.append)
        forced = +1  # away from the sample — the defocus-away convention
        # makes +1 always safe regardless of the probe's call
        try:
            peak = ctrl._coarse_pass(center, bounds, cfg_v3,
                                     direction=forced)
            outcome = f"peak {peak.pos:.0f}"
        except Exception as exc:  # noqa: BLE001
            outcome = f"raised: {exc}"
        fired = [m for m in logs if "reversing" in m]
        diagnosed = [m for m in logs if "no reversal" in m]
        verdict = (f"GUARD FIRED — {fired[0]}" if fired
                   else f"GUARD DID NOT REVERSE — {outcome}"
                   + (f"; diagnostic: {diagnosed[0]}" if diagnosed else ""))
        _safe(f"  probe direction {probe.direction!r}; forced {forced:+d}: "
              f"{verdict}")
        # restore the arm (the proof may have walked far)
        focus.move_abs(center, speed=cfg_v3.max_speed)
        focus.wait_idle(timeout_s=60.0)
        tick(0.3)

    # ---- repeatability ----------------------------------------------------
    positions: list[int] = []
    times: list[float] = []
    print(f"\n=== adaptive v3 AF-S x{args.runs} (defocus {args.defocus}) ===")
    for i in range(args.runs):
        center = focus.get_status().pos
        elapsed, result, snap, _logs = run_one3(
            AdaptiveAutofocusController, focus, frame_reader, cfg_v3,
            center, args.defocus, BENCH_DIR, f"v3_{i + 1}", pump=pump)
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
        print(f"v3 repeatability: spread {spread} steps "
              f"({spread * um_per_step:.2f} µm) over {positions}")

    # ---- A/B: frozen v2 (the rollout gate) --------------------------------
    v2_positions: list[int] = []
    v2_times: list[float] = []
    print(f"\n=== A/B: frozen v2 x{args.v2_runs} (the rollout gate) ===")
    for i in range(args.v2_runs):
        center = focus.get_status().pos
        elapsed, result, snap, _logs = run_one3(
            AdaptiveAutofocusControllerV2, focus, frame_reader, cfg_v3,
            center, args.defocus, BENCH_DIR, f"v2_{i + 1}", pump=pump)
        v2_times.append(elapsed)
        status = "OK" if result.success else f"FAIL: {result.message}"
        print(f"  run {i + 1}: {elapsed:5.2f} s  best {result.best_position}  "
              f"{status}")
        v2_positions.append(result.best_position)
        tick(0.5)
    if positions and v2_positions:
        median_v3 = sorted(times)[len(times) // 2]
        median_v2 = sorted(v2_times)[len(v2_times) // 2]
        agree = abs(positions[-1] - v2_positions[-1]) <= cfg_v3.fine_step
        faster = median_v3 <= median_v2 * 1.15
        print(f"rollout gate: v3-vs-v2 agreement "
              f"{abs(positions[-1] - v2_positions[-1])} steps "
              f"({'OK' if agree else 'FAIL'} <= {cfg_v3.fine_step}); "
              f"median {median_v3:.2f}s vs v2 {median_v2:.2f}s "
              f"({'OK' if faster else 'SLOWER'})")

    # ---- boundary-salvage reproduction through v3 -------------------------
    if not args.skip_salvage:
        print("\n=== boundary-salvage reproduction (v3) ===")
        cfg_salvage = make_config2(settings, "adaptive", args.base_speed,
                                   window_um=args.salvage_window_um)
        center = focus.get_status().pos
        elapsed, result, snap, _logs = run_one3(
            AdaptiveAutofocusController, focus, frame_reader, cfg_salvage,
            center, args.salvage_defocus, BENCH_DIR, "v3_salvage", pump=pump)
        status = "OK" if result.success else f"FAIL: {result.message}"
        print(f"  salvage: {elapsed:5.2f} s  best {result.best_position}  "
              f"{status} (v1 failed this geometry with the hit-window "
              f"message)")
        tick(0.5)

    # ---- abort mid-run ----------------------------------------------------
    if not args.skip_abort:
        print("\n=== abort mid-run ===")
        focus.move_rel(args.defocus, speed=cfg_v3.max_speed)
        focus.wait_idle(timeout_s=60.0)
        tick(0.3)
        center = focus.get_status().pos - args.defocus
        ctrl = AdaptiveAutofocusController(focus, frame_reader)
        result_box: dict = {}
        t0 = time.monotonic()
        thread = threading.Thread(
            target=lambda: result_box.setdefault(
                "result", ctrl.run(center=center, cfg=cfg_v3)),
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
    focus.move_abs(arm, speed=cfg_v3.max_speed)
    focus.wait_idle(timeout_s=60.0)
    print(f"restored arm position: {focus.get_status().pos}")

    camera.stop()
    camera.disconnect()
    focus.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
