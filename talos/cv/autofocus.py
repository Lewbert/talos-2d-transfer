"""Agile + accurate autofocus for the DIY focus stage (open-loop).

Safety posture (stage has NO limit sensor; a physical restrain protects the
objective — the real hazard is lost steps):
- The search window is a RELATIVE span around the arm position, clamped
  inside the firmware soft limits (SLIM) with a 10×fine-step margin.
- Coarse scan runs at cfg.max_speed; fine sweep slower; landing slowest.
- Every loop iteration checks abort_requested and the wall-clock budget
  (per-phase fractions); on any trigger the stage is stopped immediately.
- Position readback is verified after every move (±2 steps).
- A failed peak pick NEVER moves blindly — it returns failure.
- The firmware's 5 s serial-inactivity auto-stop is treated as a safety
  feature; EV:TMO is recoverable (one re-issue, then failure).

Classic strategy (this module):
- COARSE: one continuous monotonic pass across the window while scoring
  every fresh camera frame. Position comes from a live (t, POS) history
  (STATUS? reports POS during motion) interpolated at each frame's
  CAPTURE timestamp — no per-point settle, no trapezoid math, and the
  single direction keeps gear backlash constant within the curve.
- PEAK: multi-peak selection prefers the peak NEAREST the arm position
  (wafer surface vs mount scenario); edge peaks fail with "widen the
  window"; flat curves fail without a blind move.
- FINE: slow step-and-shoot around the fitted peak; every point is scored
  with a frame captured AFTER the move ended (freshness gate — camera
  stalls drop points, they never fake valleys).
- LANDING: overshoot-and-return from the SAME direction the fine sweep
  measured (so the fitted position is valid on a backlash-lagged axis);
  slowest reliable speed.
- The counter axis lags the load by backlash (load = counter − dir×B):
  the fine sweep and the final approach share one direction, so the
  fitted counter position IS the right landing command.

Adaptive strategy (talos.cv.af_adaptive): 3-stage fly-by — coarse
low-frequency sweep → velocity-proportional Tenengrad hill climb →
parabolic lock-on — sharing this module's scaffolding via
_BaseAutofocusController.

The controller is a QObject whose run() is blocking — invoke it from the
focus proxy's worker thread; abort via request_abort() from any thread.
The frame source is a ``frame_reader`` protocol:
``read_since(min_t=None, min_seq=-1) -> (frame, FrameMeta) | None`` —
LatestFrameSlot implements it; the CLI wraps camera.fetch in an adapter.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, Signal

from talos.cv.af_math import (
    interpolate_position,
    landing_plan,
    parabolic_fit,
    pick_peak,
)
from talos.cv.af_roi import roi_for_resolution
from talos.cv.focus_metric import METRICS
from talos.hal.base import DeviceError, DeviceTimeoutError

logger = logging.getLogger(__name__)

PHASE_COARSE = 1
PHASE_FINE = 2
PHASE_LANDING = 3

_MODE_AF_S = "AF_S"
_MODE_AF_REFINE = "AF_REFINE"


def slim_enforced(focus) -> bool | None:
    """Whether the FIRMWARE enforces the soft limits (SLIM).

    True/False when the driver can report it, None when it cannot. It gates
    every limit check on the controller and ships with the flag OFF, so
    "the bounds read back" was never evidence that a sweep cannot run past
    them.
    """
    getter = getattr(focus, "get_slim_state", None)
    if getter is None:
        return None
    try:
        return bool(getter())
    except DeviceError:
        return None


def clamp_to_soft_limits(focus, lo: int, hi: int, margin: int,
                         on_log=None) -> tuple[int, int] | None:
    """Clamp the travel window [lo, hi] to the focus axis's soft limits,
    ``margin`` steps inside them. None when the window collapses.

    Shared by the autofocus preflight AND the backlash calibration: both
    drive the axis toward the ends of its travel. The calibration used to
    have no bound at all — it swept ``center ± 250`` steps with no limits
    read and no SLIM check, while autofocus clamps and warns. A software
    bound is free and stricter than none, and it is the ONLY bound when the
    firmware flag is off.
    """
    try:
        soft = focus.get_soft_limits()
    except DeviceError:
        soft = None
        if on_log is not None:
            on_log("soft limits unreadable — using the configured span only")
    if slim_enforced(focus) is False and on_log is not None:
        on_log("WARNING: firmware soft limits are OFF (SLIM=0) — the "
               "bounds below are a software clamp only")
    if soft is not None:
        soft_lo, soft_hi = soft
        if soft_lo is not None and soft_hi is not None:
            lo = max(int(lo), int(soft_lo) + int(margin))
            hi = min(int(hi), int(soft_hi) - int(margin))
    return (lo, hi) if lo < hi else None


def move_to_verified(focus, pos: int, speed: int,
                     on_log=None, check=None) -> bool:
    """Move and verify readback (tolerates a single EV:TMO re-issue).
    Shared by the controller and the backlash calibrator.

    Returns True when the move completed, False when ``check()`` reported
    an abort while it was in flight (the stage is stopped in that case and
    the readback is not verified — the caller owns the reason).

    ``check`` exists because ``wait_idle`` blocks for up to 60 s with no
    way out: a manual jog or a STOP ALL that arrived during a move used to
    wait out the whole budget before the run noticed (the abort latch is
    polled between moves otherwise).
    """
    for attempt in (1, 2):
        try:
            focus.move_abs(pos, speed=speed)
            if not _wait_idle_checked(focus, check):
                return False
            break
        except DeviceTimeoutError:
            if attempt == 2:
                raise
            if on_log is not None:
                on_log("inactivity stop during move — re-issuing once")
    readback = focus.get_status().pos
    if abs(readback - pos) > 2:
        raise DeviceTimeoutError(
            f"focus readback mismatch after move to {pos}: got {readback}")
    return True


def _wait_idle_checked(focus, check, timeout_s: float = 60.0) -> bool:
    """wait_idle that polls ``check`` while the axis travels.

    Mirrors the driver's contract (returns as soon as the axis is idle;
    DeviceTimeoutError on EV:TMO so the caller can re-issue once;
    transient poll errors tolerated until the deadline) but returns the
    moment ``check`` reports an abort, after stopping the axis.
    """
    if check is None:
        focus.wait_idle(timeout_s=timeout_s)
        return True
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if check() is not None:
            # Stop NOW — the caller unwinds on its next check, and the
            # axis must not keep travelling until then.
            try:
                focus.stop()
            except Exception:  # noqa: BLE001 - a stop must never raise
                pass
            return False
        try:
            events = focus.drain_events()
        except Exception:  # noqa: BLE001
            events = []
        for event in events:
            if event.startswith("EV:TMO"):
                raise DeviceTimeoutError(
                    f"Focus move aborted by inactivity stop: {event}")
        try:
            if focus.get_status().is_idle:
                return True
        except DeviceError as exc:
            last_error = exc
            logger.warning("Autofocus wait_idle: transient poll error: %s", exc)
            time.sleep(0.2)  # give a reset firmware time to come back
        time.sleep(0.02)
    raise DeviceTimeoutError(
        f"Focus stage not idle after {timeout_s}s (last error: {last_error})")


def continuous_scan(focus, frame_reader, scan_start: int, scan_end: int,
                    speed: int, poll_s: float, freshness_ms: float,
                    interp_max_gap_ms: float, check, on_score=None,
                    on_log=None, early_stop_ratio: float = 0.0,
                    early_stop_samples: int = 3) \
        -> tuple[list[tuple[float, float]], str | None, float]:
    """One monotonic measure-while-moving pass across [scan_start,
    scan_end]. Stages to the start, then ONE move to the far edge while
    scoring every fresh frame at its interpolated position (live STATUS?
    POS history). Returns (curve, abort_reason, end_pos) — the stage is
    STOPPED when a reason is returned or an early stop triggers.

    ``check()`` → reason string or None (polled every iteration).
    ``on_score(pos, frame)`` → score | None: called for each scored
    point; the returned score is appended to the curve (the caller owns
    the metric). When omitted, the sweep is walked motion-only.

    Early stop (``early_stop_ratio`` > 0): once the peak has clearly been
    passed — the score has fallen below ``1 − ratio`` of the running max
    for ``early_stop_samples`` consecutive frames — the sweep aborts and
    the caller proceeds straight to the fine phase. The full-window walk
    was pure waste (user-observed); calibration sweeps pass ratio 0."""
    curve: list[tuple[float, float]] = []
    if not move_to_verified(focus, scan_start, speed, check=check):
        focus.stop()
        return curve, check() or "aborted", scan_start
    focus.move_abs(scan_end, speed=speed)
    history: list[tuple[float, int]] = []
    last_seq = -1
    running_max = 0.0
    below_streak = 0
    while True:
        reason = check()
        if reason:
            focus.stop()
            return curve, reason, float(history[-1][1]) if history else 0.0
        try:
            status = focus.get_status()
        except DeviceError as exc:
            if on_log is not None:
                on_log(f"scan poll error: {exc}")
            time.sleep(poll_s)
            continue
        history.append((time.monotonic(), status.pos))
        for event in focus.drain_events():
            if event.startswith("EV:TMO"):
                focus.stop()
                return curve, f"focus inactivity stop during scan: {event}", \
                    float(status.pos)
            if event.startswith("EV:LIM"):
                focus.stop()
                return curve, f"focus limit during scan: {event}", \
                    float(status.pos)
        if status.is_idle:
            return curve, None, float(status.pos)
        if frame_reader is not None:
            item = frame_reader.read_since(min_seq=last_seq)
            if item is not None:
                frame, meta = item
                last_seq = meta.seq  # consume regardless of freshness
                if time.monotonic() - meta.t_capture <= freshness_ms / 1000.0:
                    pos = interpolate_position(meta.t_capture, history,
                                               gap_max_ms=interp_max_gap_ms)
                    if pos is not None and on_score is not None:
                        score = on_score(pos, frame)
                        if score is not None:
                            curve.append((pos, score))
                            if early_stop_ratio > 0 and len(curve) >= 5:
                                if score > running_max:
                                    running_max = score
                                    below_streak = 0
                                elif score < running_max * (1.0 - early_stop_ratio):
                                    below_streak += 1
                                    if below_streak >= early_stop_samples:
                                        focus.stop()
                                        return curve, None, pos
                                else:
                                    below_streak = 0
        time.sleep(poll_s)
    return curve, None, float(scan_end)


@dataclass
class AutofocusConfig:
    # motion, in focus-stage steps
    coarse_step: int = 100        # coarse localization tolerance
    fine_step: int = 10
    span_steps: int = 6000        # total search window (± span//2)
    window_plus_steps: int = 0    # asymmetric search bounds (0 = symmetric
    window_minus_steps: int = 0   # span_steps // 2)
    max_speed: int = 500          # steps/s during the coarse continuous scan
    fine_speed: int = 200
    landing_speed: int = 50
    stage_speed: int = 2000       # dead TRAP staging moves (no scoring — no
                                  # blur budget; manual-jog-proven speed)
    backlash_steps: int = 0       # measured per objective (auto-calibration)
    overshoot_margin_steps: int = 15
    fine_window_steps: int = 0    # 0 → max(coarse_step, 3×fine_step)
    near_window_steps: int = 0    # the stage-2 window when the probe says
                                  # NEAR/cluster (no coarse pass ran — the
                                  # anchor's uncertainty is the PROBE's, up
                                  # to ~±0.7σ, wider than the coarse
                                  # localization's): 0 → fine_window_steps;
                                  # the fit-distrust tolerance (fw//2)
                                  # scales with it
    # scoring
    quality_threshold: float = 0.3
    metric: str = "tenengrad"     # hardware-verified best on this bench
    roi_norm: tuple | None = None  # normalized (x, y, w, h); None = full frame
    # latency / timing
    timeout_s: float = 180.0
    freshness_ms: float = 400.0   # skip frames older than this at score time
    interp_max_gap_ms: float = 300.0
    coarse_poll_s: float = 0.05
    fine_wait_timeout_s: float = 2.0
    settle_frames: int = 2
    # peak selection
    n_parabolic_points: int = 5
    peak_prominence: float = 0.15
    fail_on_edge_peak: bool = True
    # preflight image check (refuse to sweep when there is nothing to see)
    preflight_check: bool = True
    preflight_luma_lo: float = 15.0    # 0..255 RGB mean
    preflight_luma_hi: float = 245.0
    preflight_min_score: float = 1.0   # metric floor (essentially no edges)
    # coarse-scan early termination (agility: stop after passing the peak)
    early_stop_ratio: float = 0.5      # score below 50% of the running max…
    early_stop_samples: int = 3        # …for 3 consecutive frames → stop
    early_stop_rise: float = 0.2       # peak-seen gate: the running max must
                                       # rise 20% above the curve start before
                                       # the early stop may fire (a noise dip
                                       # during the climb must not stop it)
    # strategy dispatch (classic = this module; adaptive = af_adaptive)
    strategy: str = "classic"
    # --- adaptive strategy ---
    coarse_metric: str = "brenner_k"   # low-frequency, blur-tolerant
    coarse_metric_k: int = 8
    coarse_bin: int = 2                # 2×2 area binning for the coarse metric
    coarse_speed: int = 0              # 0 → use max_speed
    hill_v_cap: int = 0                # far-from-peak climb speed (steps/s)
    hill_v_min: float = 0.0            # near-peak climb speed (steps/s)
    hill_ratio: float = 0.6            # early stop: below 60% of running max…
    hill_early_stop_samples: int = 2   # …for 2 consecutive frames → stop
    lock_samples_required: int = 4     # near-peak samples for a direct fit
    lock_score_frac: float = 0.85      # "near peak" = ≥ 85% of the peak score
    stationary_points: int = 5         # fallback re-measure point count
    stop_accel_sps2: int = 20000       # firmware ACC for the predictive stop
    stop_latency_s: float = 0.05       # serial in-flight allowance
    stop_safety_steps: int = 10
    # --- probe + direction guard (adaptive v2) ---
    probe_step_steps: int = 0          # 0 → 3×coarse_step, clamped ≥3×fine_step
    probe_peak_ratio: float = 0.15     # near-focus: center must clear the sides
    probe_min_slope: float = 0.10      # low-freq slope floor for a direction call
    guard_samples: int = 4             # early-scan window the direction guard watches
    guard_drop_ratio: float = 0.15     # low-freq fall that marks the wrong way
    coarse_early_stop_samples: int = 2 # first-peak stop (v2 coarse only — the
                                       # shared early_stop_samples also serves
                                       # the classic and stays at its default)
    stage2_retries: int = 1           # stage-2 re-attempts around the current
                                       # position (0 = fail immediately; never
                                       # restores the arm, never stage 1)
    # --- derivative-hybrid (adaptive v3) ---
    probe_score_floor_ratio: float = 0.3  # probe floor = ratio × the preflight
                                       # score (the arm's own baseline)
    probe_curv_in: float = 0.0       # near via curvature: C < −in (1/step²);
                                       # 0 → per-objective from DOF (build_config)
    probe_curv_out: float = 0.0      # flank: C > +out (1/step²); 0 → auto
    probe_cluster_center_ratio: float = 0.5  # near-cluster: S_c ≥ ratio ×
                                       # the best side (the valley's center is
                                       # a small fraction — excluded)
    probe_cluster_side_ratio: float = 0.4  # near-cluster: the worst side ≥
                                       # ratio × S_c (a real flank's far side
                                       # is low — excluded)
    probe_cluster_min_score: float = 0.0  # near-cluster ABSOLUTE gate: the
                                       # best point must clear this score
                                       # (0 = branch disabled — ratio gates
                                       # cannot tell a flat far tail from a
                                       # plateau shoulder; a per-field knob)
    coarse_direction: int = 0         # the DEFAULT coarse scan direction when
                                       # the probe has no direction info
                                       # (0 = auto: +1 away from the sample);
                                       # ±1 forces the sweep direction
                                       # (+1 = away/up, −1 = toward/down —
                                       # scenario-specific overrides)
    guard_fit_samples: int = 6       # early pass samples for the 2σ slope fit
    guard_sigma: float = 2.0         # z for the pooled-noise reversal test
    coarse_curv_window: int = 5      # the pass-only curvature window
    coarse_curv_stop: float = 0.0    # normalized stop threshold; 0 → auto
    coarse_curv_vertex: float = 0.0  # trusted-vertex floor (NEGATIVE);
                                       # 0 → never trust the vertex
    # mode
    mode: str = _MODE_AF_S        # AF_S = full search; AF_REFINE = fine+land
                                  # around the current position (AF-C re-peak)


@dataclass
class AutofocusResult:
    best_position: int
    best_score: float
    curve: list[tuple[float, float]] = field(default_factory=list)
    coarse_curve: list[tuple[float, float]] = field(default_factory=list)
    # the low-frequency metric series (probe + coarse pass; adaptive v2 —
    # classic runs leave it empty)
    success: bool = False
    aborted: bool = False
    message: str = ""
    phase: str = ""               # where it stopped: preflight|coarse|fine|landing|done|motion
    peak_at_edge: bool = False
    restore_on_fail: bool = True  # False: keep the axis where it stopped
                                  # (stage-2 failures — already near focus)


@dataclass
class AutofocusRequest:
    """GUI → focus-worker job payload: a complete AF run specification in
    steps (the service builds it from the µm-based settings)."""
    center_steps: int = 0
    config: AutofocusConfig = field(default_factory=AutofocusConfig)
    roi_norm: tuple | None = None   # normalized (x, y, w, h); None = full frame
    bounds: tuple | None = None     # absolute-stage search window override
                                    # (None = the symmetric config window)

    def to_config(self) -> AutofocusConfig:
        return replace(self.config, roi_norm=self.roi_norm)


class _AfExit(Exception):
    """Internal early exit carrying the final result."""

    def __init__(self, result: AutofocusResult):
        super().__init__(result.message)
        self.result = result


class _BaseAutofocusController(QObject):
    """Scaffolding shared by the classic and adaptive strategies: signals,
    the blocking run() wrapper (abort flag, deadline, error handling,
    arm-position restore), preflight, scoring, and the landing/verify
    finish. Subclasses implement _run()."""

    sig_progress = Signal(float, int, float, float)  # fraction, phase, score, pos
    sig_curve_secondary = Signal(float, float)  # low-freq series (pos, score)
    sig_done = Signal(object)                 # AutofocusResult
    sig_log = Signal(str)

    # Failure-restore policy: True = failed runs return the axis to the
    # arm position; False = failures and stops end at the CURRENT position.
    # The v3 controller overrides to False (user policy); the stored-
    # knowledge controllers (classic/V1/V2) keep the arm restore.
    _restore_on_fail: bool = True

    def __init__(self, focus, frame_reader, parent: QObject | None = None,
                 abort_check: Callable[[], str | None] | None = None):
        super().__init__(parent)
        self._focus = focus            # FocusStage driver (worker-thread owned)
        self._frame_reader = frame_reader  # read_since(min_t, min_seq) protocol
        self.abort_requested = False
        self._abort_reason = "aborted by user"
        # An external abort latch consulted by _check: the proxy sets it
        # for aborts that land between the job's enqueue and the
        # controller's construction (request_abort can't reach a
        # controller that doesn't exist yet; run() resets the local
        # flag at entry, so the latch must be polled). None for the
        # stored-knowledge controllers — zero behavior change.
        self._abort_check = abort_check
        self._curve: list[tuple[float, float]] = []
        self._low_curve: list[tuple[float, float]] = []  # adaptive v2 series
        self._metric_fn = METRICS["tenengrad"]
        self._t_start = 0.0
        self._deadline = 0.0
        self._cfg = AutofocusConfig()
        self._current_pos = 0
        self._fine_dir = 1            # direction of the fine sweep (landing matches)
        # (shape, roi) cache: the AF loop scores a frame per delivered
        # image and the shape only changes on a resolution switch.
        self._roi_cache: tuple | None = None

    # ------------------------------------------------------------------

    def request_abort(self, reason: str | None = None) -> None:
        """Plain method — callable from any thread (documented path).
        The reason is normalized to the 'aborted' prefix — the result's
        aborted flag is derived from it."""
        self.abort_requested = True
        if reason:
            if not reason.startswith("aborted"):
                reason = "aborted — " + reason
            self._abort_reason = reason

    def run(self, center: int, cfg: AutofocusConfig | None = None,
            bounds: tuple[int, int] | None = None) -> AutofocusResult:
        """``bounds`` (absolute stage steps) overrides the symmetric
        center ± span/2 search window — asymmetric windows for
        scenario-specific AF (e.g. the known structure sits only on one
        side of the arm). Still clamped inside the soft limits. None =
        the symmetric config window."""
        cfg = cfg or AutofocusConfig()
        self._cfg = cfg
        self.abort_requested = False
        self._abort_reason = "aborted by user"
        self._curve = []
        self._low_curve = []
        self._metric_fn = METRICS.get(cfg.metric, METRICS["tenengrad"])
        self._t_start = time.monotonic()
        self._deadline = self._t_start + cfg.timeout_s
        self._current_pos = int(center)
        self._arm_center = int(center)
        self._bounds_override = bounds
        try:
            return self._run(int(center), cfg)
        except _AfExit as exc:
            # Failed runs return the axis to the arm position — never
            # strand it at a window edge. Aborts move nothing, and
            # restore_on_fail=False results (stage-2 failures — the axis
            # is already near focus) stay where they stopped. The
            # _restore_on_fail class policy (False for v3) disables the
            # restore entirely: failures and stops end at the CURRENT
            # position.
            if not exc.result.aborted and not self.abort_requested \
                    and exc.result.restore_on_fail and self._restore_on_fail:
                try:
                    move_to_verified(self._focus, self._arm_center,
                                     self._cfg.max_speed)
                    self._current_pos = self._arm_center  # tracks the restore
                except Exception as restore_exc:  # noqa: BLE001
                    self.sig_log.emit(f"position restore failed: {restore_exc}")
            elif not exc.result.aborted and not exc.result.success \
                    and not self._restore_on_fail \
                    and exc.result.phase != "preflight":
                # The preflight failed BEFORE any motion — the axis never
                # moved, so the suffix would lie.
                exc.result.message += " — stopped at current position"
                exc.result.restore_on_fail = False  # truthful for v3
            return exc.result
        except DeviceTimeoutError as exc:
            # EV:TMO: the inactivity auto-stop aborted a move — recoverable once.
            logger.warning("Autofocus TMO: %s", exc)
            return self._stop_stage(f"focus timeout: {exc}", "motion")
        except DeviceError as exc:
            logger.warning("Autofocus device error: %s", exc)
            try:
                self._focus.stop()
            except Exception:  # noqa: BLE001
                pass
            return AutofocusResult(self._current_pos, 0.0, self._curve,
                                   coarse_curve=self._low_curve,
                                   message=str(exc), phase="motion",
                                   restore_on_fail=self._restore_on_fail)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Autofocus crashed")
            try:
                self._focus.stop()
            except Exception:  # noqa: BLE001
                pass
            return AutofocusResult(self._current_pos, 0.0, self._curve,
                                   coarse_curve=self._low_curve,
                                   message=f"internal error: {exc!r}", phase="motion",
                                   restore_on_fail=self._restore_on_fail)

    # ------------------------------------------------------------------

    def _run(self, center: int, cfg: AutofocusConfig) -> AutofocusResult:
        raise NotImplementedError

    def _check(self, phase_dl: float | None = None) -> str | None:
        """Abort reason, or None to continue."""
        if self.abort_requested:
            return self._abort_reason
        if self._abort_check is not None:
            # The external abort latch (aborts that landed before this
            # controller existed — see __init__).
            reason = self._abort_check()
            if reason:
                self.abort_requested = True
                self._abort_reason = reason
                return reason
        now_t = time.monotonic()
        if phase_dl is not None and now_t > phase_dl:
            return "phase budget exceeded"
        if now_t > self._deadline:
            return "wall-clock budget exceeded"
        return None

    def _stop_stage(self, reason: str, phase: str,
                    restore: bool | None = None) -> AutofocusResult:
        try:
            self._focus.stop()
        except Exception:  # noqa: BLE001
            pass
        # Failed runs return the axis to the arm position (never strand it
        # at a window edge); aborts stop everything and move nothing.
        # restore=False keeps the axis where it stopped (stage-2 failures
        # — already near focus). None resolves to the _restore_on_fail
        # class policy (False for v3: failures end at the CURRENT
        # position); explicit False callers still override.
        if restore is None:
            restore = self._restore_on_fail
        if restore and not reason.startswith("aborted") \
                and not self.abort_requested:
            try:
                move_to_verified(self._focus, self._arm_center, self._cfg.max_speed)
                # the verified restore MOVED the axis — _current_pos must
                # track reality: a pass that starts from a stale position
                # plans its span/edge-stop from the wrong place.
                self._current_pos = self._arm_center
            except Exception as exc:  # noqa: BLE001
                self.sig_log.emit(f"position restore failed: {exc}")
        return AutofocusResult(
            best_position=self._current_pos, best_score=0.0, curve=self._curve,
            coarse_curve=getattr(self, "_low_curve", []),
            aborted=reason.startswith("aborted"), message=reason, phase=phase,
            restore_on_fail=restore)

    def _slim_enforced(self) -> bool | None:
        """True/False when the driver can report it, None when it cannot."""
        return slim_enforced(self._focus)

    def _preflight(self, center: int, cfg: AutofocusConfig) \
            -> tuple[int, int] | None:
        """Soft-limit clamp + image sanity. Returns (window_lo, window_hi),
        None when the clamped window is empty (the caller reports it —
        the stage has not moved, so no restore is due), raises _AfExit on
        image sanity failures."""
        curve = self._curve
        margin = max(cfg.fine_step * 10, 1)
        override = getattr(self, "_bounds_override", None)
        if override is not None:
            window_lo, window_hi = int(override[0]), int(override[1])
        else:
            minus = cfg.window_minus_steps if cfg.window_minus_steps > 0 \
                else cfg.span_steps // 2
            plus = cfg.window_plus_steps if cfg.window_plus_steps > 0 \
                else cfg.span_steps // 2
            window_lo = center - minus
            window_hi = center + plus
        window = clamp_to_soft_limits(self._focus, window_lo, window_hi,
                                      margin, on_log=self.sig_log.emit)
        if window is None:
            return None
        window_lo, window_hi = window

        # ---- Image sanity: nothing to see → refuse to sweep --------------
        if cfg.preflight_check and self._frame_reader is not None:
            # Poll briefly: the first frame may still be in flight.
            deadline = time.monotonic() + cfg.fine_wait_timeout_s
            item = None
            while item is None and time.monotonic() < deadline:
                item = self._frame_reader.read_since()
                if item is None:
                    time.sleep(cfg.coarse_poll_s)
            if item is None:
                raise _AfExit(AutofocusResult(
                    center, 0.0, curve,
                    message="no camera frames before autofocus — camera "
                            "stalled or not streaming",
                    phase="preflight"))
            frame = item[0]
            luma = float(np.asarray(frame).mean())
            if luma < cfg.preflight_luma_lo:
                raise _AfExit(AutofocusResult(
                    center, 0.0, curve,
                    message=f"image too dark (mean {luma:.0f}/255) — adjust "
                            f"exposure/gain or the illumination",
                    phase="preflight"))
            if luma > cfg.preflight_luma_hi:
                raise _AfExit(AutofocusResult(
                    center, 0.0, curve,
                    message=f"image too bright (mean {luma:.0f}/255) — reduce "
                            f"exposure/gain",
                    phase="preflight"))
            score = self._score_frame(frame)
            self._preflight_score = score  # the v3 probe's relative-floor baseline
            if score < cfg.preflight_min_score:
                raise _AfExit(AutofocusResult(
                    center, 0.0, curve,
                    message="no contrast in view — nothing to focus on "
                            "(empty field or far from the sample)",
                    phase="preflight"))
        return window_lo, window_hi

    def _roi_for(self, shape) -> tuple[int, int, int, int] | None:
        """ROI pixels for this frame shape (cached — the score loop calls
        this per delivered frame)."""
        if not self._cfg.roi_norm:
            return None
        cached = self._roi_cache
        if cached is not None and cached[0] == shape:
            return cached[1]
        roi = roi_for_resolution(self._cfg.roi_norm, shape)
        self._roi_cache = (shape, roi)
        return roi

    def _score_frame(self, frame) -> float:
        return self._metric_fn(frame, self._roi_for(frame.shape))

    def _move_to(self, pos: int, speed: int, phase: str = "motion") -> None:
        """Move and verify readback (tolerates a single EV:TMO re-issue).

        The abort latch is polled WHILE the axis travels (see
        move_to_verified): a jog or STOP ALL that arrives mid-move must
        stop the stage now, not after wait_idle's 60 s budget.
        """
        if not move_to_verified(self._focus, pos, speed,
                                on_log=self.sig_log.emit, check=self._check):
            reason = self._check() or self._abort_reason or "aborted by user"
            raise _AfExit(self._stop_stage(reason, phase))
        self._current_pos = pos

    def _wait_fresh(self, min_t: float, timeout_s: float,
                    poll_s: float) -> tuple | None:
        """Wait for a frame captured at/after min_t (move end). None when
        no camera is attached or the pipeline is stalled."""
        if self._frame_reader is None:
            return None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            item = self._frame_reader.read_since(min_t=min_t)
            if item is not None:
                return item
            time.sleep(poll_s)
        return None

    def _landing(self, final_pos: int, bounds: tuple[int, int],
                 cfg: AutofocusConfig) -> None:
        """Overshoot-and-return, approaching from the SAME direction the
        fine sweep measured (the fitted position is only valid on that
        side of the backlash)."""
        reason = self._check(self._t_start + cfg.timeout_s * 0.97)
        if reason:
            raise _AfExit(self._stop_stage(reason, "landing"))
        overshoot, _direction, degraded = landing_plan(
            final_pos, cfg.backlash_steps, cfg.overshoot_margin_steps, bounds,
            preferred_dir=self._fine_dir)
        if degraded:
            self.sig_log.emit(
                f"landing at the window edge: the overshoot clamps to "
                f"{overshoot} (backlash take-up {cfg.backlash_steps} steps "
                "cannot fit) — the load may land off by up to that much")
        self._move_to(overshoot, cfg.landing_speed)
        self.sig_progress.emit(0.90, PHASE_LANDING, 0.0, float(overshoot))
        reason = self._check(self._t_start + cfg.timeout_s * 0.97)
        if reason:
            raise _AfExit(self._stop_stage(reason, "landing"))
        self._move_to(final_pos, cfg.landing_speed)
        self.sig_progress.emit(0.96, PHASE_LANDING, 0.0, float(final_pos))

    def _land_and_finish(self, final_pos: int, peak, curve,
                         bounds: tuple[int, int],
                         cfg: AutofocusConfig) -> AutofocusResult:
        """Landing + readback verify + the success result (shared finish
        for both strategies)."""
        self._landing(final_pos, bounds, cfg)
        readback = self._focus.get_status().pos
        if abs(readback - final_pos) > 2:
            raise _AfExit(AutofocusResult(
                final_pos, peak.score, curve,
                message=f"landing readback mismatch: {readback} != {final_pos}",
                phase="landing"))
        # The readback proves the axis ARRIVED — not that the peak was
        # real. Without an absolute floor a noise spike on a flat field
        # was reported as "focused ✔" (the quality gate is purely relative
        # to the baseline, and every metric is non-negative, so its
        # `baseline >= 0` guard was always true).
        if cfg.preflight_check and peak.score < cfg.preflight_min_score:
            self.sig_log.emit(
                f"landed score {peak.score:.3g} is below the contrast floor "
                f"{cfg.preflight_min_score:.3g} — no real peak in view")
            raise _AfExit(AutofocusResult(
                final_pos, peak.score, curve,
                message="no usable focus peak (score below the contrast "
                        "floor) — nothing to focus on",
                phase="landing"))
        result = AutofocusResult(final_pos, peak.score, curve,
                                 coarse_curve=getattr(self, "_low_curve", []),
                                 success=True, message="ok", phase="done")
        self.sig_done.emit(result)
        return result


class AutofocusController(_BaseAutofocusController):
    """Classic 6-phase flow (hardware-verified baseline; also the fallback
    strategy when the adaptive algorithm is unavailable or fails)."""

    # ------------------------------------------------------------------
    # Main flow
    # ------------------------------------------------------------------

    def _run(self, center: int, cfg: AutofocusConfig) -> AutofocusResult:
        curve = self._curve

        # ---- 1. Preflight: soft-limit clamp + image sanity ---------------
        bounds = self._preflight(center, cfg)
        if bounds is None:
            return AutofocusResult(center, 0.0, curve,
                                   message="search window empty (soft limits)",
                                   phase="preflight")
        window_lo, window_hi = bounds

        # ---- 2. Coarse continuous scan (AF_S) ------------------------------
        if cfg.mode == _MODE_AF_S:
            self._coarse_scan(center, bounds, cfg)
            peak = pick_peak(curve, float(center),
                             prominence=cfg.peak_prominence,
                             edge_margin_steps=2.0 * cfg.fine_step)
            if peak is None:
                raise _AfExit(AutofocusResult(
                    center, 0.0, curve,
                    message="no clear focus peak (flat curve)", phase="coarse"))
            baseline = (curve[0][1] + curve[-1][1]) / 2.0
            if baseline >= 0 and peak.score < baseline * (1.0 + cfg.quality_threshold):
                raise _AfExit(AutofocusResult(
                    int(peak.pos), peak.score, curve,
                    message="peak too weak vs baseline", phase="coarse"))
            if peak.at_edge and cfg.fail_on_edge_peak:
                raise _AfExit(AutofocusResult(
                    int(peak.pos), peak.score, curve,
                    message="peak at window edge — widen the search window",
                    phase="coarse", peak_at_edge=True))
            fit = parabolic_fit(curve, peak.pos, cfg.n_parabolic_points)
            if fit is None:
                # A valid peak with a non-parabolic shape (narrow spike,
                # two-plane cusp): the coarse phase only LOCALIZES the fine
                # window — fall back to the measured peak, the fine sweep
                # re-measures densely and verifies (or fails safely).
                fit = float(peak.pos)
                self.sig_log.emit(f"coarse peak at {peak.pos:.1f} — fit "
                                  f"degenerate, using peak position")
            else:
                self.sig_log.emit(f"coarse peak at {peak.pos:.1f} → fit {fit:.1f}")
            fw = cfg.fine_window_steps or max(cfg.coarse_step, 3 * cfg.fine_step)
            fine_start = max(window_lo, int(fit) - fw)
            fine_end = min(window_hi, int(fit) + fw)
        else:
            # AF_REFINE: re-peak around the current position (AF-C drift fix)
            fw = cfg.fine_window_steps or max(cfg.coarse_step, 3 * cfg.fine_step)
            fine_start = max(window_lo, center - fw)
            fine_end = min(window_hi, center + fw)
            self.sig_log.emit(f"AF_REFINE: fine sweep [{fine_start}, {fine_end}]")

        # ---- 3. Fine step-and-shoot ----------------------------------------
        self._fine_sweep(fine_start, fine_end, cfg)

        # ---- 4. Final peak selection from the combined curve ---------------
        peak = self._final_pick(curve, center, cfg)
        if peak is None:
            raise _AfExit(AutofocusResult(
                center, 0.0, curve,
                message="no clear focus peak in fine window", phase="fine"))
        if peak.at_edge and cfg.fail_on_edge_peak:
            # Coarse localization may have been poor (sparse frames, low
            # fps): extend the fine window ONCE toward the edge and re-sweep
            # the extension before giving up. Only the search-window bound
            # is terminal.
            fw = cfg.fine_window_steps or max(cfg.coarse_step, 3 * cfg.fine_step)
            if peak.pos <= fine_start:
                extension = (max(window_lo, fine_start - fw), fine_start - cfg.fine_step)
            else:
                extension = (fine_end + cfg.fine_step, min(window_hi, fine_end + fw))
            if extension[0] < extension[1]:
                self.sig_log.emit(f"peak at fine-window edge — extending sweep to "
                                  f"[{extension[0]}, {extension[1]}]")
                self._fine_sweep(extension[0], extension[1], cfg)
                peak = self._final_pick(curve, center, cfg)
            if peak is None:
                raise _AfExit(AutofocusResult(
                    center, 0.0, curve,
                    message="no clear focus peak after window extension",
                    phase="fine"))
            if peak.at_edge:
                raise _AfExit(AutofocusResult(
                    int(peak.pos), peak.score, curve,
                    message="peak at window edge — widen the search window",
                    phase="fine", peak_at_edge=True))
        best_fit = parabolic_fit(curve, peak.pos, cfg.n_parabolic_points)
        final_pos = int(round(best_fit)) if best_fit is not None else int(peak.pos)
        final_pos = max(window_lo, min(window_hi, final_pos))

        # ---- 5+6. Landing + verify ----------------------------------------
        return self._land_and_finish(final_pos, peak, curve, bounds, cfg)

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    def _coarse_scan(self, center: int, bounds: tuple[int, int],
                     cfg: AutofocusConfig) -> None:
        """One monotonic pass across the window, scoring fresh frames at
        interpolated positions. Raises _AfExit on abort/timeout/events."""
        window_lo, window_hi = bounds
        # Nearest-edge-first: the arm position (≈ the peak) is crossed early.
        if abs(center - window_lo) <= abs(window_hi - center):
            scan_start, scan_end = window_lo, window_hi
        else:
            scan_start, scan_end = window_hi, window_lo
        span = abs(scan_end - scan_start) or 1

        def on_score(pos: float, frame) -> float:
            score = self._score_frame(frame)
            frac = 0.03 + 0.55 * min(1.0, abs(pos - scan_start) / span)
            self.sig_progress.emit(frac, PHASE_COARSE, score, pos)
            return score

        curve, reason, end_pos = continuous_scan(
            self._focus, self._frame_reader, scan_start, scan_end,
            cfg.max_speed, poll_s=cfg.coarse_poll_s,
            freshness_ms=cfg.freshness_ms,
            interp_max_gap_ms=cfg.interp_max_gap_ms,
            check=lambda: self._check(self._t_start + cfg.timeout_s * 0.58),
            on_score=on_score, on_log=self.sig_log.emit,
            early_stop_ratio=cfg.early_stop_ratio,
            early_stop_samples=cfg.early_stop_samples)
        if reason:
            raise _AfExit(self._stop_stage(reason, "coarse"))
        self._curve.extend(curve)
        self._current_pos = int(end_pos)
        # Settle frames at the stop position: stationary scores,
        # freshness-gated.
        move_end_t = time.monotonic()
        for _ in range(cfg.settle_frames):
            item = self._wait_fresh(move_end_t, cfg.fine_wait_timeout_s,
                                    cfg.coarse_poll_s)
            if item is None:
                break  # no camera / stalled — don't burn the budget
            score = self._score_frame(item[0])
            self._curve.append((float(end_pos), score))

    def _fine_sweep(self, fine_start: int, fine_end: int,
                    cfg: AutofocusConfig) -> None:
        """Slow step-and-shoot; only frames captured AFTER each move ended
        are scored (freshness gate). Direction follows the entry move so
        every measured point shares one backlash state."""
        if fine_start >= fine_end:
            return
        positions = list(range(fine_start, fine_end + 1, cfg.fine_step))
        if fine_start < self._current_pos:
            positions = list(reversed(positions))  # descend: same dir as entry
        self._fine_dir = 1 if positions[-1] > positions[0] else -1
        n = len(positions)
        for i, pos in enumerate(positions):
            reason = self._check(self._t_start + cfg.timeout_s * 0.88)
            if reason:
                raise _AfExit(self._stop_stage(reason, "fine"))
            self._move_to(pos, min(cfg.max_speed, cfg.fine_speed))
            t_end = time.monotonic()
            item = self._wait_fresh(t_end, cfg.fine_wait_timeout_s,
                                    cfg.coarse_poll_s)
            if item is None:
                self.sig_log.emit(f"no fresh frame at {pos} — skipped")
                continue
            score = self._score_frame(item[0])
            self._curve.append((float(pos), score))
            self.sig_progress.emit(0.60 + 0.28 * (i + 1) / n, PHASE_FINE,
                                   score, float(pos))

    def _final_pick(self, curve, center, cfg):
        return pick_peak(curve, float(center), prominence=cfg.peak_prominence,
                         edge_margin_steps=2.0 * cfg.fine_step)
