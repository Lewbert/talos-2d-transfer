"""Adaptive 3-stage autofocus: coarse low-frequency fly-by sweep →
velocity-proportional Tenengrad hill climb → parabolic lock-on.

Motion architecture: both passes run in firmware CONT mode (signed SPD =
direction). Stops are RAMP stops — ``SPD:0`` decelerates at the firmware
ACC and drops to MODE_IDLE (axis_engine contKinematics) — never the
instant STOP, because this axis is open-loop (no encoder): an instant stop
at aggressive speed loses steps silently. Verified firmware behaviors:

- SPD issued mid-CONT changes speed/direction live (target-only update;
  v² ±= 2A per step).
- SPD:0 ramps to IDLE with NO event pushed — the pass must watch
  status.is_idle, never wait for an EV after a ramp command.
- MOVE/GOTO during CONT returns ERR:BUSY — every pass is staged with a
  TRAP move to its start edge first.
- EV:LIM fires during CONT; EV:TMO only if the serial link actually dies
  (20 ms STATUS? polling keeps RX alive).
- set_speed updates are hard-capped at 10 Hz (the input system proved SPD
  flooding wedges the firmware RX).

Stage 1 — guided coarse sweep: one CONT pass across the clamped window at
the per-objective speed, scored with Tenengrad (the sharp metric — the
low-frequency metric's broad plateau cannot localize the peak on a
DOF-wide scene; hardware-verified) + a stationary settle anchor at the
stop position (stationary scores ~5× the blur-penalized moving ones).
Early-stop after passing the peak, gated so a noise dip during the climb
cannot stop the pass early.

Stage 2 — velocity-proportional hill climb: a CONT pass across the fine
window around the coarse peak, scored with Tenengrad at full resolution.
The speed profile is POSITION-based: v_cap at the pass edges ("high at
the bottom"), v_min (blur ≤ fine_step/3) within 2×fine_step of the
coarse anchor ("low at the peak"), linear ramp between — deterministic,
where the score-based schedule collapsed to v_min everywhere on a DOF
plateau (hardware-verified). Early-stop with the same peak-seen gate.

Stage 3 — parabolic lock-on: if ≥ 4 hill samples sit within 85% of the
peak and the parabolic fit succeeds, land directly; otherwise a 5-point
stationary step-and-shoot re-measure (classic machinery) supplies the fit.
Then the shared overshoot-and-return backlash landing + readback verify.

Failure paths mirror the classic controller: abort stops the stage and
moves nothing; failures restore the arm position; a failed peak pick
never moves blindly.

AdaptiveAutofocusControllerV1 = the hardware-verified 2026-09-07
baseline, frozen as the preservation reference (see docs/PLAN.md
handoff #9). AdaptiveAutofocusControllerV2 = the probe-driven v2,
hardware-verified and FROZEN as the v3 rollout backup (see
docs/PLAN.md handoff #11); the derivative-hybrid v3 lives in
talos/cv/af_v3.py.
"""

from __future__ import annotations

import time
from dataclasses import replace

from talos.cv.af_math import (
    PeakInfo,
    ProbeResult,
    direction_guard_wrong_way,
    hill_speed_by_pos,
    interpolate_position,
    parabolic_fit,
    peak_is_complete,
    pick_peak,
    predictive_stop_steps,
    probe_direction,
    probe_near_focus,
)
from talos.cv.af_roi import roi_for_resolution
from talos.cv.autofocus import (
    _AfExit,
    _BaseAutofocusController,
    _MODE_AF_S,
    AutofocusConfig,
    AutofocusResult,
    move_to_verified,
)
from talos.cv.focus_metric import METRICS, bin2
from talos.hal.base import DeviceError, DeviceTimeoutError

PHASE_COARSE_PASS = 4
PHASE_HILL = 5
PHASE_LOCK = 6
PHASE_PROBE = 7

DIRECTION_GUARD = "direction guard: peak on the other side"
START_PEAK_WALKAWAY = "coarse pass started on the peak and walked away"

_SPD_SCHEDULE_S = 0.1     # set_speed updates at most every 100 ms
_RAMP_IDLE_TIMEOUT_S = 1.5


def cont_pass(focus, frame_reader, start: int, direction: int, span: int,
              v0: int, speed_schedule, check, on_score, on_log,
              poll_s: float, freshness_ms: float, interp_max_gap_ms: float,
              early_stop_ratio: float, early_stop_samples: int,
              stop_accel: float, stop_latency_s: float,
              stop_safety_steps: int,
              edge_allowance_steps: float = 0.0,
              early_stop_rise: float = 0.2,
              guard=None) \
        -> tuple[list[tuple[float, float]], str | None, float]:
    """One CONT-mode measure-while-moving pass from ``start`` toward
    ``start + direction*span``. The caller stages to ``start`` with a TRAP
    move first (CONT cannot accept MOVE/GOTO and the staging position must
    be exact). Returns (curve, reason, end_pos) — the stage is at rest
    when a reason is returned or a stop triggers.

    ``speed_schedule(pos, frame, score, running_max)`` → new speed | None:
    polled at most every 100 ms (and only after a scored frame); a
    ``set_speed`` is issued only when the value changes. ``check()`` →
    reason string or None (polled every iteration).

    ``guard(curve)`` → reason string | None: polled after each scored
    frame — an early-abort hook for callers that want to re-decide the
    pass (e.g. the adaptive v2 direction guard). A non-None return halts
    the pass exactly like a ``check()`` reason (ramp stop, stage at rest,
    curve preserved) — the caller owns the reason's meaning.

    The early stop has a "peak-seen" gate: it can only fire once the
    running max has risen ``early_stop_rise`` above the curve's first
    score — a noise dip during the climb (before any peak) must not stop
    the pass (user-observed: the climb stopped too early on noisy
    fields).

    Stops are ramp stops (SPD:0 + wait_idle, belt-and-braces stop() on
    timeout/error) except for EV:TMO / EV:LIM, which hard-stop. The
    predictive edge stop begins the ramp when the remaining distance
    reaches the decel distance + worst-case poll/command slip + safety.
    ``edge_allowance_steps`` lets the pass overshoot the pass edge (the
    search window is clamped 10×fine-step inside the firmware SLIM, so
    the edge itself is not sacred — a legitimate peak can sit inside the
    stop zone; truncating there fakes an edge peak)."""
    curve: list[tuple[float, float]] = []
    edge_pos = start + direction * span
    v = max(int(v0), 1)
    last_issued_v = v
    last_sched_t = 0.0
    running_max = 0.0
    first_score: float | None = None
    below_streak = 0
    last_seq = -1

    def halt() -> float:
        """Ramp stop + belt-and-braces; returns the resting position."""
        try:
            focus.set_speed(0)
            focus.wait_idle(timeout_s=_RAMP_IDLE_TIMEOUT_S, poll_s=0.02)
        except (DeviceError, DeviceTimeoutError):
            try:
                focus.stop()
            except Exception:  # noqa: BLE001
                pass
        try:
            return float(focus.get_status().pos)
        except DeviceError:
            return float(edge_pos)

    focus.set_speed(direction * v)
    history: list[tuple[float, int]] = []
    while True:
        reason = check()
        if reason:
            return curve, reason, halt()
        try:
            status = focus.get_status()
        except DeviceError as exc:
            if on_log is not None:
                on_log(f"cont pass poll error: {exc}")
            time.sleep(poll_s)
            continue
        history.append((time.monotonic(), status.pos))
        for event in focus.drain_events():
            if event.startswith("EV:TMO"):
                focus.stop()
                return curve, f"focus inactivity stop during pass: {event}", \
                    float(status.pos)
            if event.startswith("EV:LIM"):
                focus.stop()
                return curve, f"focus limit during pass: {event}", \
                    float(status.pos)
        if status.is_idle:
            return curve, None, float(status.pos)
        # Predictive edge stop: begin the ramp while the decel + slip
        # budget still fits inside the pass span (minus the allowance —
        # the pass may overshoot the edge by that much and still stay
        # inside the SLIM margin). SIGNED distance: with an allowance
        # larger than the stop budget the trigger lies beyond the edge,
        # and abs() would make it unreachable.
        remaining = (edge_pos - status.pos) * direction
        if remaining <= predictive_stop_steps(
                v, stop_accel, poll_s, stop_latency_s, stop_safety_steps) \
                - edge_allowance_steps:
            return curve, None, halt()
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
                            if first_score is None:
                                first_score = score
                            if early_stop_ratio > 0 and len(curve) >= 5:
                                if score > running_max:
                                    running_max = score
                                    below_streak = 0
                                elif score < running_max * (1.0 - early_stop_ratio) \
                                        and first_score is not None \
                                        and running_max >= first_score * (1.0 + early_stop_rise):
                                    # the peak-seen gate: only stop AFTER a
                                    # clear rise above the curve start
                                    below_streak += 1
                                    if below_streak >= early_stop_samples:
                                        return curve, None, halt()
                                else:
                                    below_streak = 0
                            if guard is not None:
                                reason = guard(curve)
                                if reason:
                                    return curve, reason, halt()
                            if speed_schedule is not None:
                                now_t = time.monotonic()
                                if now_t - last_sched_t >= _SPD_SCHEDULE_S:
                                    last_sched_t = now_t
                                    new_v = speed_schedule(
                                        pos, frame, score, running_max)
                                    if new_v is not None and int(new_v) != last_issued_v:
                                        try:
                                            focus.set_speed(direction * int(new_v))
                                        except DeviceError as exc:
                                            if on_log is not None:
                                                on_log(f"speed update failed: {exc}")
                                        else:
                                            last_issued_v = int(new_v)
                                            v = int(new_v)
        time.sleep(poll_s)


class AdaptiveAutofocusControllerV1(_BaseAutofocusController):
    """3-stage fly-by strategy (see the module docstring) — the frozen
    2026-09-07 hardware-verified baseline."""

    def __init__(self, focus, frame_reader, parent=None):
        super().__init__(focus, frame_reader, parent)

    # ------------------------------------------------------------------
    # Main flow
    # ------------------------------------------------------------------

    def _run(self, center: int, cfg: AutofocusConfig) -> AutofocusResult:
        curve = self._curve

        # ---- 1. Preflight ------------------------------------------------
        bounds = self._preflight(center, cfg)
        if bounds is None:
            return AutofocusResult(center, 0.0, curve,
                                   message="search window empty (soft limits)",
                                   phase="preflight")

        # ---- 2. Guided coarse pass (AF_S) ---------------------------------
        if cfg.mode == _MODE_AF_S:
            coarse_peak = self._coarse_pass(center, bounds, cfg)
            anchor = parabolic_fit(self._stage1_curve, coarse_peak.pos,
                                   cfg.n_parabolic_points)
            if anchor is None:
                anchor = coarse_peak.pos
        else:
            # AF_REFINE: re-peak around the current position (AF-C drift fix)
            coarse_peak = None
            anchor = float(center)
            self.sig_log.emit("AF_REFINE: adaptive hill climb around "
                              f"{int(anchor)}")

        # ---- 3. Velocity-proportional hill climb --------------------------
        hill_curve = self._hill_pass(anchor, bounds, cfg)

        # ---- 4. Parabolic lock-on -----------------------------------------
        final_pos, peak = self._lock_on(hill_curve, anchor, bounds, cfg)

        # ---- 5+6. Landing + verify ----------------------------------------
        return self._land_and_finish(final_pos, peak, curve, bounds, cfg)

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _coarse_pass(self, center: int, bounds: tuple[int, int],
                     cfg: AutofocusConfig):
        """Stage 1: one CONT pass across the clamped window, scored with
        the binned low-frequency metric. Raises _AfExit on abort/events;
        returns the picked PeakInfo."""
        window_lo, window_hi = bounds
        # Nearest-edge-first: the arm position (≈ the peak) is crossed early.
        if abs(center - window_lo) <= abs(window_hi - center):
            start, direction = window_lo, 1
        else:
            start, direction = window_hi, -1
        span = abs(window_hi - window_lo)
        v0 = cfg.coarse_speed or cfg.max_speed
        # Stage to the pass start with a TRAP move (CONT cannot accept
        # MOVE/GOTO, and the start position must be exact + blur-free).
        # The staging is UNSCORED — no blur budget — so it runs at the
        # fast manual-jog speed (a ±500 µm window costs 5 s at the pass
        # speed vs 1.25 s at 2000 sps).
        self._move_to(start, cfg.stage_speed)

        def on_score(pos: float, frame) -> float:
            # The stage-1 CURVE scores with the sharp metric (tenengrad):
            # the low-frequency metric's broad plateau cannot localize the
            # peak on a DOF-wide scene (hardware-verified at 5×: the
            # brenner curve was flat 22-57 with a noise bump picked 50
            # steps off the wafer, while the tenengrad curve peaked 6×
            # over baseline exactly on it).
            score = self._score_frame(frame)
            frac = 0.03 + 0.50 * min(1.0, abs(pos - start) / span)
            self.sig_progress.emit(frac, PHASE_COARSE_PASS, score, pos)
            return score

        stage1_curve, reason, end_pos = cont_pass(
            self._focus, self._frame_reader, start, direction, span, v0,
            speed_schedule=None,
            check=lambda: self._check(self._t_start + cfg.timeout_s * 0.55),
            on_score=on_score, on_log=self.sig_log.emit,
            poll_s=cfg.coarse_poll_s, freshness_ms=cfg.freshness_ms,
            interp_max_gap_ms=cfg.interp_max_gap_ms,
            early_stop_ratio=cfg.early_stop_ratio,
            early_stop_samples=cfg.early_stop_samples,
            stop_accel=cfg.stop_accel_sps2, stop_latency_s=cfg.stop_latency_s,
            stop_safety_steps=cfg.stop_safety_steps,
            early_stop_rise=cfg.early_stop_rise,
            edge_allowance_steps=8.0 * cfg.fine_step)
        if reason:
            raise _AfExit(self._stop_stage(reason, "coarse"))
        # Settle frames at the stop position: the moving measurements are
        # blur-penalized at these speeds (the classic's sharp coarse peak
        # IS its post-stop settle frame — hardware-verified: stationary
        # scores ~5× the moving ones at 375 sps). The stationary anchor
        # tames the coarse localization on DOF-wide plateaus.
        move_end_t = time.monotonic()
        for _ in range(max(1, cfg.settle_frames)):
            item = self._wait_fresh(move_end_t, cfg.fine_wait_timeout_s,
                                    cfg.coarse_poll_s)
            if item is None:
                break  # no camera / stalled — don't burn the budget
            score = self._score_frame(item[0])
            stage1_curve.append((float(end_pos), score))
            self.sig_progress.emit(0.54, PHASE_COARSE_PASS, score,
                                   float(end_pos))
        self._stage1_curve = stage1_curve
        self._curve.extend(stage1_curve)
        self._current_pos = int(end_pos)
        peak = pick_peak(stage1_curve, float(center),
                         prominence=cfg.peak_prominence,
                         edge_margin_steps=2.0 * cfg.fine_step)
        if peak is None:
            raise _AfExit(AutofocusResult(
                center, 0.0, self._curve,
                message="no clear focus peak (flat curve)", phase="coarse"))
        baseline = (stage1_curve[0][1] + stage1_curve[-1][1]) / 2.0
        if baseline >= 0 and peak.score < baseline * (1.0 + cfg.quality_threshold):
            raise _AfExit(AutofocusResult(
                int(peak.pos), peak.score, self._curve,
                message="peak too weak vs baseline", phase="coarse"))
        if peak.at_edge and cfg.fail_on_edge_peak:
            raise _AfExit(AutofocusResult(
                int(peak.pos), peak.score, self._curve,
                message="peak at window edge — widen the search window",
                phase="coarse", peak_at_edge=True))
        self.sig_log.emit(f"coarse peak at {peak.pos:.1f} "
                          f"(score {peak.score:.0f})")
        return peak

    def _hill_pass(self, anchor: float, bounds: tuple[int, int],
                   cfg: AutofocusConfig) -> list[tuple[float, float]]:
        """Stage 2: CONT pass across the fine window around ``anchor``,
        Tenengrad-scored, with the position-based velocity profile."""
        window_lo, window_hi = bounds
        fw = cfg.fine_window_steps or max(cfg.coarse_step, 3 * cfg.fine_step)
        fine_start = max(window_lo, int(anchor) - fw)
        fine_end = min(window_hi, int(anchor) + fw)
        if fine_start >= fine_end:
            return []
        # Start at the edge nearest the current position: the entry TRAP
        # move continues the coarse pass's direction, so the hill curve is
        # measured in one direction (one backlash state, classic argument).
        if abs(self._current_pos - fine_end) <= abs(self._current_pos - fine_start):
            start, direction = fine_end, -1
        else:
            start, direction = fine_start, 1
        self._fine_dir = direction
        self._move_to(start, cfg.max_speed)
        span = fine_end - fine_start
        v_cap = cfg.hill_v_cap or cfg.fine_speed
        v_min = int(cfg.hill_v_min) if cfg.hill_v_min > 0 else v_cap

        def speed_schedule(pos, frame, score, running_max):
            # POSITION-based velocity profile: fast at the pass edges
            # ("high at the bottom"), slow near the coarse anchor ("low at
            # the peak"), linear ramp between. Deterministic — the
            # score-based schedule collapsed to v_min everywhere on a
            # DOF plateau (hardware-verified; the user saw a constant
            # slow climb).
            return hill_speed_by_pos(pos, anchor, v_min, v_cap,
                                     near_steps=2.0 * cfg.fine_step,
                                     ramp_end_steps=float(fw))

        def on_score(pos: float, frame) -> float:
            score = self._score_frame(frame)
            frac = 0.55 + 0.30 * min(1.0, abs(pos - start) / span)
            self.sig_progress.emit(frac, PHASE_HILL, score, pos)
            return score

        hill_curve, reason, end_pos = cont_pass(
            self._focus, self._frame_reader, start, direction, span, v_cap,
            speed_schedule=speed_schedule,
            check=lambda: self._check(self._t_start + cfg.timeout_s * 0.85),
            on_score=on_score, on_log=self.sig_log.emit,
            poll_s=cfg.coarse_poll_s, freshness_ms=cfg.freshness_ms,
            interp_max_gap_ms=cfg.interp_max_gap_ms,
            early_stop_ratio=cfg.hill_ratio,
            early_stop_samples=cfg.hill_early_stop_samples,
            stop_accel=cfg.stop_accel_sps2, stop_latency_s=cfg.stop_latency_s,
            stop_safety_steps=cfg.stop_safety_steps,
            early_stop_rise=cfg.early_stop_rise,
            edge_allowance_steps=8.0 * cfg.fine_step)
        if reason:
            raise _AfExit(self._stop_stage(reason, "fine"))
        self._curve.extend(hill_curve)
        self._current_pos = int(end_pos)
        return hill_curve

    def _lock_on(self, hill_curve: list[tuple[float, float]], anchor: float,
                 bounds: tuple[int, int],
                 cfg: AutofocusConfig) -> tuple[int, object]:
        """Stage 3: direct parabolic fit when the hill curve is dense
        near the peak, else a stationary 5-point re-measure. Returns
        (final_pos, PeakInfo)."""
        window_lo, window_hi = bounds
        if not hill_curve:
            raise _AfExit(AutofocusResult(
                int(anchor), 0.0, self._curve,
                message="no clear focus peak in hill window", phase="fine"))
        peak = pick_peak(hill_curve, float(anchor),
                         prominence=cfg.peak_prominence,
                         edge_margin_steps=2.0 * cfg.fine_step)
        if peak is None:
            raise _AfExit(AutofocusResult(
                int(anchor), 0.0, self._curve,
                message="no clear focus peak in hill window", phase="fine"))
        # Edge check against the FINE-WINDOW bounds, not the curve extent:
        # the hill pass early-stops shortly after the peak, so the peak is
        # ALWAYS near the curve's end by design (pick_peak's at_edge is
        # meaningless here). A peak near the fine-window edge means the
        # coarse localization was poor or the true peak is outside.
        fw = cfg.fine_window_steps or max(cfg.coarse_step, 3 * cfg.fine_step)
        fine_start = max(window_lo, int(anchor) - fw)
        fine_end = min(window_hi, int(anchor) + fw)
        margin = 2.0 * cfg.fine_step
        near_left = peak.pos - fine_start <= margin
        near_right = fine_end - peak.pos <= margin
        if (near_left or near_right) and cfg.fail_on_edge_peak:
            # Interior fine-window edge → the stationary re-measure around
            # the edge peak is the honest retry (never a blind move). An
            # edge that IS the search-window bound → the true peak is
            # outside — fail like the classic.
            if (near_left and fine_start > window_lo) \
                    or (near_right and fine_end < window_hi):
                self.sig_log.emit("hill peak at interior window edge — "
                                  "stationary re-measure")
            else:
                raise _AfExit(AutofocusResult(
                    int(peak.pos), peak.score, self._curve,
                    message="peak at window edge — widen the search window",
                    phase="fine", peak_at_edge=True))
        near = [p for p, s in hill_curve
                if s >= cfg.lock_score_frac * peak.score]
        fit = parabolic_fit(hill_curve, peak.pos, cfg.n_parabolic_points)
        fw = cfg.fine_window_steps or max(cfg.coarse_step, 3 * cfg.fine_step)
        # Trust the hill only when it agrees with the coarse anchor: a
        # fit far from it chased a competing plane's slope (the mount
        # below the wafer — hardware-verified: the hill descended into
        # the mount's rising scores and the fit landed 55 steps off).
        if fit is not None and abs(fit - anchor) > fw // 2:
            self.sig_log.emit(f"hill fit {fit:.1f} far from the coarse "
                              f"anchor {anchor:.1f} — distrusting it")
            return self._stationary_remeasure(
                PeakInfo(pos=anchor, score=peak.score, at_edge=False),
                bounds, cfg)
        if len(near) >= cfg.lock_samples_required and fit is not None:
            final_pos = int(round(fit))
            final_pos = max(window_lo, min(window_hi, final_pos))
            self.sig_progress.emit(0.92, PHASE_LOCK, peak.score,
                                   float(final_pos))
            self.sig_log.emit(f"lock-on: direct fit {fit:.1f} from "
                              f"{len(near)} near-peak samples")
            return final_pos, peak
        return self._stationary_remeasure(peak, bounds, cfg)

    def _stationary_remeasure(self, peak, bounds: tuple[int, int],
                              cfg: AutofocusConfig) -> tuple[int, object]:
        """Classic 5-point step-and-shoot around the hill peak (the
        precision guarantee when the hill curve is sparse or degenerate),
        with a classic-style one-shot extension when the curve is still
        climbing at its far edge."""
        self.sig_log.emit("hill curve sparse/degenerate — stationary re-measure")
        n = cfg.stationary_points
        offsets = [cfg.fine_step * (i - n // 2) for i in range(n)]

        def shoot(positions: list[int]) -> list[tuple[float, float]]:
            stat_curve: list[tuple[float, float]] = []
            for i, pos in enumerate(positions):
                reason = self._check(self._t_start + cfg.timeout_s * 0.88)
                if reason:
                    raise _AfExit(self._stop_stage(reason, "fine"))
                self._move_to(pos, cfg.max_speed)
                t_end = time.monotonic()
                item = self._wait_fresh(t_end, cfg.fine_wait_timeout_s,
                                        cfg.coarse_poll_s)
                if item is None:
                    self.sig_log.emit(f"no fresh frame at {pos} — skipped")
                    continue
                score = self._score_frame(item[0])
                stat_curve.append((float(pos), score))
                self._curve.append((float(pos), score))
                self.sig_progress.emit(0.86 + 0.06 * (i + 1) / n, PHASE_LOCK,
                                       score, float(pos))
            return stat_curve

        def fit_curve(curve, anchor):
            sp = pick_peak(curve, float(anchor),
                           prominence=cfg.peak_prominence,
                           edge_margin_steps=2.0 * cfg.fine_step)
            a = sp.pos if sp is not None else anchor
            return sp, a, parabolic_fit(curve, a, cfg.n_parabolic_points)

        positions = [int(round(peak.pos + off)) for off in offsets]
        positions = [max(bounds[0], min(bounds[1], p)) for p in positions]
        # Sweep direction = the hill pass direction (one backlash state).
        # The hill early-stop leaves the stage just past the peak — INSIDE
        # this tiny window — so the first point is staged from the far
        # side: one unscored move past it, then every scored point is
        # entered in the sweep direction. Without the staging, the first
        # point carries the OPPOSITE backlash state and its load sits at
        # the sharp plane at the wrong counter — a spurious score next to
        # the peak that corrupts the fit (sim-verified).
        if self._fine_dir < 0:
            positions = list(reversed(positions))
            staging = min(bounds[1], positions[0] + cfg.fine_step)
        else:
            staging = max(bounds[0], positions[0] - cfg.fine_step)
        if staging != positions[0]:
            self._move_to(staging, cfg.max_speed)
        stat_curve = shoot(positions)
        if len(stat_curve) < 3:
            raise _AfExit(AutofocusResult(
                int(peak.pos), peak.score, self._curve,
                message="stationary re-measure failed (camera stalled)",
                phase="fine"))
        stat_peak, anchor, fit = fit_curve(stat_curve, peak.pos)
        # Still climbing at the far edge → the true peak is further out:
        # extend once in the sweep direction (state-preserving — the
        # landing direction discipline is unchanged), never a blind move.
        if stat_curve and stat_curve[-1][1] == max(s for _, s in stat_curve):
            xs = [p for p, _ in stat_curve]
            direction = 1 if xs[-1] > xs[0] else -1
            ext = [int(round(xs[-1] + direction * cfg.fine_step * (i + 1)))
                   for i in range(n)]
            ext = [p for p in ext if bounds[0] <= p <= bounds[1]]
            if ext:
                self.sig_log.emit(f"stationary curve climbing at edge — "
                                  f"extending sweep to {ext[0]}..{ext[-1]}")
                stat_curve += shoot(ext)
                stat_peak, anchor, fit = fit_curve(stat_curve, anchor)
        # Never chase a curve whose maximum sits at its END points: the
        # true peak is outside the window (or a stronger plane below — the
        # mount must never win over the wafer). Fail like the classic
        # instead of fitting the ramp (hardware-verified runaway). The
        # pick's at_edge margin is too wide here — the stationary window
        # only spans ±2×fine_step, so every interior point is "at edge".
        if stat_curve:
            max_idx = max(range(len(stat_curve)),
                          key=lambda i: stat_curve[i][1])
            if max_idx in (0, len(stat_curve) - 1):
                raise _AfExit(AutofocusResult(
                    int(peak.pos), peak.score, self._curve,
                    message="peak at window edge — widen the search window",
                    phase="fine", peak_at_edge=True))
        final_pos = int(round(fit)) if fit is not None else int(anchor)
        final_pos = max(bounds[0], min(bounds[1], final_pos))
        result_peak = stat_peak if stat_peak is not None else peak
        self.sig_log.emit(f"lock-on: stationary fit {final_pos}")
        return final_pos, result_peak


class AdaptiveAutofocusControllerV2(AdaptiveAutofocusControllerV1):
    """Probe-driven adaptive v2 (see docs/AUTOFOCUS.md) — FROZEN
    2026-09-08 as the v3 rollout backup (handoff #11). Zero behavior
    edits from here; the v3 controller subclasses it in
    talos/cv/af_v3.py.

    On top of the V1 stages, a start PROBE decides the entry point:
    three points at [center−δ, center, center+δ] scored with both the
    sharp metric (near-focus check — a clear local max means we are
    already on the hill and stage 1 is skipped entirely) and the
    low-frequency metric (hill-direction estimate far from focus). No
    information → the blind V1 sweep (nearest edge, full window).

    Stage 1 with a direction: the CONT pass starts AT the arm position
    and sweeps toward the rising side, stopping at the FIRST peak
    (``coarse_early_stop_samples``); a direction guard watches the first
    few samples and reverses the sweep early when the low-frequency
    series falls while the sharp series never rose (peak-seen gate not
    met — a real climb cannot trip it). One reversal only.

    Stage 2 robustness: when the hill pass is cut short by a
    limit/budget reason (never an abort) or the picked peak sits near a
    window edge, the collected curve is salvaged when it contains a
    COMPLETE peak (samples on both sides + dense near-peak samples) —
    the data is enough to reconstruct the peak, so the hit-window fail
    only fires when the pass genuinely ended while still climbing.

    A stage-2 failure (phase "fine", not aborted) RETRIES THE STAGE
    LOCALLY up to ``stage2_retries`` times around the current position —
    stage 2 is already near focus, so it never returns to the arm
    position and never falls back to stage 1; after the retries the run
    stops AT the current position (no restore).
    """

    # ------------------------------------------------------------------
    # Main flow
    # ------------------------------------------------------------------

    def _run(self, center: int, cfg: AutofocusConfig) -> AutofocusResult:
        curve = self._curve
        self._stage1_curve = []  # init here — v1 relied on _coarse_pass
        self._probe_samples = []

        # ---- 1. Preflight ------------------------------------------------
        bounds = self._preflight(center, cfg)
        if bounds is None:
            return AutofocusResult(center, 0.0, curve,
                                   coarse_curve=self._low_curve,
                                   message="search window empty (soft limits)",
                                   phase="preflight")

        # ---- 2. Start probe: near-focus? direction? ----------------------
        probe = self._probe(center, bounds, cfg)
        if probe.near_focus:
            anchor = self._probe_anchor(center, probe, cfg)
            self.sig_log.emit(f"probe: near focus — entering hill climb "
                              f"directly at {int(anchor)}")
        else:
            if probe.direction is not None:
                self.sig_log.emit(f"probe: far from focus — coarse sweep "
                                  f"direction {probe.direction:+d}")
            else:
                self.sig_log.emit("probe: no direction information — "
                                  "blind sweep")
            self._probe_samples = probe.samples
            coarse_peak = self._coarse_pass(center, bounds, cfg,
                                            direction=probe.direction)
            anchor = self._coarse_anchor(coarse_peak, cfg)

        # ---- 3+4. Hill climb + lock-on (stage-2 local retries) ----------
        # A stage-2 failure retries the stage around the CURRENT position
        # (stage 2 is already near focus) — never a restore to the arm,
        # never a stage-1 fallback. After the retries the run stops at
        # the current position (user policy). A peak sitting at an
        # INTERIOR fine-window edge (the coarse anchor missed the true
        # peak) widens the fine window instead — the classic's one-shot
        # extension, bounded (×2, ×4) and kept local to stage 2.
        fw_scale = 1.0
        attempt = 0
        while True:
            try:
                hill_curve = self._hill_pass(anchor, bounds, cfg,
                                             fw_scale=fw_scale)
                final_pos, peak = self._lock_on(hill_curve, anchor, bounds,
                                                cfg, fw_scale=fw_scale)
                break
            except _AfExit as exc:
                if exc.result.aborted or exc.result.phase != "fine":
                    raise
                if "interior window edge" in exc.result.message \
                        and fw_scale < 4.0:
                    fw_scale *= 2.0
                    self.sig_log.emit(
                        "hill peak at interior window edge — widening the "
                        f"fine window x{fw_scale:.0f}")
                    continue
                if attempt >= cfg.stage2_retries:
                    raise _AfExit(replace(
                        exc.result,
                        best_position=self._current_pos,
                        message=f"stage 2 failed after "
                                f"{attempt + 1} attempt(s): "
                                f"{exc.result.message}",
                        restore_on_fail=False))
                attempt += 1
                # Keep the SAME anchor (the coarse peak estimate): the
                # axis sits at the last pass's HALT position, deep past
                # the peak — re-anchoring there mis-centers the retried
                # window (sim-verified cascade into the search-bound
                # edge failure). Mis-centered anchors are the widening
                # path's job, not the retry's.
                self.sig_log.emit(f"stage 2 attempt {attempt} failed "
                                  f"({exc.result.message}) — retrying the "
                                  f"hill climb (same anchor)")

        # ---- 5+6. Landing + verify ----------------------------------------
        return self._land_and_finish(final_pos, peak, curve, bounds, cfg)

    # ------------------------------------------------------------------
    # Probe
    # ------------------------------------------------------------------

    def _probe_anchor(self, center: int, probe: ProbeResult,
                      cfg: AutofocusConfig) -> float:
        """The stage-2 anchor when the probe says near: the arm position
        (extracted so v3's near-cluster branch can anchor at the best
        probe point instead)."""
        return float(center)

    def _coarse_anchor(self, coarse_peak, cfg: AutofocusConfig) -> float:
        """Stage-2 anchor from the stage-1 peak: the parabolic re-fit
        over the stage-1 curve (else the picked peak position). Extracted
        so the v3 curvature-stop bypass can skip the re-fit entirely
        (audit #12/#13 — the settle outlier + the climbing curve end
        pull the re-fit off the trusted vertex)."""
        anchor = parabolic_fit(self._stage1_curve, coarse_peak.pos,
                               cfg.n_parabolic_points)
        return coarse_peak.pos if anchor is None else anchor

    def _probe(self, center: int, bounds: tuple[int, int],
               cfg: AutofocusConfig) -> ProbeResult:
        """Three-point start analysis at [center−δ, center, center+δ] with
        δ = probe_step_steps or 3×coarse_step (≥ 3×fine_step). Sharp
        metric → near-focus check; low-frequency metric → direction.
        Clamped/deduped positions collapse to < 3 → blind."""
        window_lo, window_hi = bounds
        delta = cfg.probe_step_steps or 3 * cfg.coarse_step
        delta = max(int(delta), 3 * cfg.fine_step)
        positions: list[int] = []
        for p in (int(center), int(center) - delta, int(center) + delta):
            p = max(window_lo, min(window_hi, p))
            if p not in positions:
                positions.append(p)
        if len(positions) < 3:
            self.sig_log.emit("probe: window too narrow for a 3-point "
                              "probe — blind")
            return ProbeResult(False, None, [])
        samples: list[tuple[float, float, float]] = []
        for i, pos in enumerate(positions):
            reason = self._check(self._t_start + cfg.timeout_s * 0.10)
            if reason:
                raise _AfExit(self._stop_stage(reason, "probe"))
            # Dead TRAP staging speed — the probe moves are unscored.
            self._move_to(pos, cfg.stage_speed)
            t_end = time.monotonic()
            item = self._wait_fresh(t_end, cfg.fine_wait_timeout_s,
                                    cfg.coarse_poll_s)
            if item is None:
                self.sig_log.emit(f"probe: no fresh frame at {pos} — skipped")
                continue
            sharp = self._score_frame(item[0])
            low = self._score_frame_low(item[0])
            samples.append((float(pos), sharp, low))
            self._low_curve.append((float(pos), low))
            self._curve.append((float(pos), sharp))
            self.sig_progress.emit(0.02 + 0.08 * (i + 1) / len(positions),
                                   PHASE_PROBE, sharp, float(pos))
            self.sig_curve_secondary.emit(float(pos), low)
        if len(samples) < 3:
            self.sig_log.emit("probe: incomplete samples — blind")
            return ProbeResult(False, None, samples)
        # Return to the arm position: the directed coarse pass must start
        # from the CENTER (not δ past it) — a peak between the probe
        # points must not be skipped, and the pass must cross the peak
        # mid-way (the early-stop's peak-seen gate needs a rise above the
        # pass start; a pass that STARTS on the peak can never rise).
        if self._current_pos != int(center):
            self._move_to(int(center), cfg.stage_speed)
        _p0, sharp_c, low_c = samples[0]
        _, sharp_lo, low_lo = samples[1]
        _, sharp_hi, low_hi = samples[2]
        near = probe_near_focus(sharp_c, [sharp_lo, sharp_hi],
                                cfg.probe_peak_ratio)
        direction = None if near else probe_direction(low_lo, low_c, low_hi,
                                                      cfg.probe_min_slope)
        return ProbeResult(near, direction, samples)

    def _score_frame_low(self, frame) -> float:
        """The LOW-FREQUENCY metric on the 2×-binned ROI — the wide
        response that direction estimation needs far from focus (the dead
        coarse_metric/coarse_metric_k/coarse_bin config finally lives
        here)."""
        binned = bin2(frame, self._roi_for(frame.shape))
        fn = METRICS.get(self._cfg.coarse_metric, METRICS["brenner_k"])
        if self._cfg.coarse_metric in ("brenner_k", "abs_diff"):
            return fn(binned, None, self._cfg.coarse_metric_k)
        return fn(binned, None)

    # ------------------------------------------------------------------
    # Stage 1 (probe-guided with direction guard)
    # ------------------------------------------------------------------

    def _coarse_pass(self, center: int, bounds: tuple[int, int],
                     cfg: AutofocusConfig, direction: int | None = None):
        """Stage 1 v2: direction given → the pass starts AT the arm
        position and sweeps toward the rising side, stopping at the first
        peak; a direction guard reverses it early when the first samples
        say we are moving away. direction None → the blind V1 sweep
        (nearest edge, full window). Raises _AfExit on abort/events;
        returns the picked PeakInfo."""
        window_lo, window_hi = bounds
        v0 = cfg.coarse_speed or cfg.max_speed
        low_series: list[float] = []
        if direction is None:
            # Blind fallback: today's nearest-edge-first full-window sweep.
            if abs(center - window_lo) <= abs(window_hi - center):
                start, direction = window_lo, 1
            else:
                start, direction = window_hi, -1
            span = abs(window_hi - window_lo)
            self._move_to(start, cfg.stage_speed)
            guard = None
        else:
            start = int(self._current_pos)
            span = (window_hi - start) if direction > 0 else (start - window_lo)
            if span < 3 * cfg.fine_step:
                self.sig_log.emit(f"coarse: direction {direction:+d} has no "
                                  f"room (span {span}) — blind sweep")
                return self._coarse_pass(center, bounds, cfg, direction=None)
            self.sig_log.emit(f"coarse: probe-guided sweep from {start} "
                              f"toward {start + direction * span} "
                              f"(direction {direction:+d})")

            def guard(curve):
                scores = [s for _, s in curve]
                # Start-on-peak walk-away (any sample count): when the
                # pass's FIRST sample IS the running max, cont_pass's
                # peak-seen gate can never fire (running_max ==
                # first_score forever) — the pass would walk the whole
                # span away from a peak it started on. Stop it once two
                # samples fall below the early-stop threshold.
                if len(curve) >= 5 and scores[0] >= max(scores) \
                        and all(s < (1.0 - cfg.early_stop_ratio) * scores[0]
                                for s in scores[-2:]):
                    return START_PEAK_WALKAWAY
                # Only the first few samples decide the direction: the
                # low-freq series falling while the sharp series never
                # rose (the peak-seen gate) means we are walking away.
                if len(curve) > cfg.guard_samples or len(low_series) < 3:
                    return None
                if direction_guard_wrong_way(
                        scores, low_series, cfg.guard_drop_ratio,
                        cfg.early_stop_rise):
                    return DIRECTION_GUARD
                return None

        def on_score(pos: float, frame) -> float:
            score = self._score_frame(frame)
            low = self._score_frame_low(frame)
            low_series.append(low)
            self._low_curve.append((pos, low))
            self.sig_curve_secondary.emit(pos, low)
            frac = 0.03 + 0.50 * min(1.0, abs(pos - start) / span)
            self.sig_progress.emit(frac, PHASE_COARSE_PASS, score, pos)
            return score

        stage1_curve, reason, end_pos = cont_pass(
            self._focus, self._frame_reader, start, direction, span, v0,
            speed_schedule=None,
            check=lambda: self._check(self._t_start + cfg.timeout_s * 0.55),
            on_score=on_score, on_log=self.sig_log.emit,
            poll_s=cfg.coarse_poll_s, freshness_ms=cfg.freshness_ms,
            interp_max_gap_ms=cfg.interp_max_gap_ms,
            early_stop_ratio=cfg.early_stop_ratio,
            early_stop_samples=cfg.coarse_early_stop_samples,
            stop_accel=cfg.stop_accel_sps2, stop_latency_s=cfg.stop_latency_s,
            stop_safety_steps=cfg.stop_safety_steps,
            early_stop_rise=cfg.early_stop_rise,
            edge_allowance_steps=8.0 * cfg.fine_step,
            guard=guard)
        if reason == DIRECTION_GUARD:
            self.sig_log.emit("direction guard fired — reversing coarse sweep")
            self._current_pos = int(end_pos)
            span = (window_hi - int(end_pos)) if direction < 0 \
                else (int(end_pos) - window_lo)
            stage1_curve, reason, end_pos = cont_pass(
                self._focus, self._frame_reader, int(end_pos), -direction,
                span, v0,
                speed_schedule=None,
                check=lambda: self._check(self._t_start + cfg.timeout_s * 0.55),
                on_score=on_score, on_log=self.sig_log.emit,
                poll_s=cfg.coarse_poll_s, freshness_ms=cfg.freshness_ms,
                interp_max_gap_ms=cfg.interp_max_gap_ms,
                early_stop_ratio=cfg.early_stop_ratio,
                early_stop_samples=cfg.coarse_early_stop_samples,
                stop_accel=cfg.stop_accel_sps2, stop_latency_s=cfg.stop_latency_s,
                stop_safety_steps=cfg.stop_safety_steps,
                early_stop_rise=cfg.early_stop_rise,
                edge_allowance_steps=8.0 * cfg.fine_step)
            if reason == DIRECTION_GUARD:
                # One reversal only — accept what we have and let the
                # peak checks adjudicate.
                self.sig_log.emit("second direction guard — accepting the pass")
                reason = None
        if reason == START_PEAK_WALKAWAY:
            # A normal, early first-peak stop: the curve's first sample is
            # the peak (the probe seeds give it interior extent).
            self.sig_log.emit("coarse pass started on the peak and walked "
                              "away — accepting the curve")
            reason = None
        if reason:
            raise _AfExit(self._stop_stage(reason, "coarse"))
        # Settle frames at the stop position (stationary anchor; see v1).
        move_end_t = time.monotonic()
        for _ in range(max(1, cfg.settle_frames)):
            item = self._wait_fresh(move_end_t, cfg.fine_wait_timeout_s,
                                    cfg.coarse_poll_s)
            if item is None:
                break  # no camera / stalled — don't burn the budget
            score = self._score_frame(item[0])
            stage1_curve.append((float(end_pos), score))
            self.sig_progress.emit(0.54, PHASE_COARSE_PASS, score,
                                   float(end_pos))
        # Seed the probe's sharp samples (real stage-1 measurements): a
        # peak sitting AT the pass start (or just beyond a probe point)
        # reads as an edge peak against the pass-only x-extent, and the
        # nearest-center preference needs the arm-position samples.
        prefix = [(p, s) for p, s, _low in self._probe_samples]
        stage1_curve = prefix + stage1_curve
        # Position-sort before analysis: the probe prefix is non-monotonic
        # in x (center, −δ, +δ) and pick_peak's 3-point moving average
        # runs over the SEQUENCE — a sharp probe spike bleeds into its
        # neighbors and fakes a peak at the arm position (sim-verified).
        stage1_curve.sort(key=lambda item: item[0])
        self._stage1_curve = stage1_curve
        self._curve.extend(stage1_curve[len(prefix):])
        self._current_pos = int(end_pos)
        peak = pick_peak(stage1_curve, float(center),
                         prominence=cfg.peak_prominence,
                         edge_margin_steps=2.0 * cfg.fine_step)
        if peak is None:
            raise _AfExit(AutofocusResult(
                center, 0.0, self._curve,
                coarse_curve=self._low_curve,
                message="no clear focus peak (flat curve)", phase="coarse"))
        baseline = (stage1_curve[0][1] + stage1_curve[-1][1]) / 2.0
        if baseline >= 0 and peak.score < baseline * (1.0 + cfg.quality_threshold):
            raise _AfExit(AutofocusResult(
                int(peak.pos), peak.score, self._curve,
                coarse_curve=self._low_curve,
                message="peak too weak vs baseline", phase="coarse"))
        if peak.at_edge and cfg.fail_on_edge_peak:
            raise _AfExit(AutofocusResult(
                int(peak.pos), peak.score, self._curve,
                coarse_curve=self._low_curve,
                message="peak at window edge — widen the search window",
                phase="coarse", peak_at_edge=True))
        self.sig_log.emit(f"coarse peak at {peak.pos:.1f} "
                          f"(score {peak.score:.0f})")
        return peak

    # ------------------------------------------------------------------
    # Stage 2 (+ boundary-hit salvage)
    # ------------------------------------------------------------------

    def _hill_pass(self, anchor: float, bounds: tuple[int, int],
                   cfg: AutofocusConfig, fw_scale: float = 1.0) \
            -> list[tuple[float, float]]:
        """V1 stage 2 plus the salvage path: a pass cut short by a
        limit/budget reason (never an abort) still returns its curve when
        that curve contains a COMPLETE peak — the data is enough to
        reconstruct the focus point, so the hit-window fail only fires
        when the pass ended while still climbing. fw_scale widens the
        fine window (the stage-2 interior-edge extension)."""
        window_lo, window_hi = bounds
        fw = int((cfg.fine_window_steps or max(cfg.coarse_step,
                                               3 * cfg.fine_step))
                 * fw_scale)
        fine_start = max(window_lo, int(anchor) - fw)
        fine_end = min(window_hi, int(anchor) + fw)
        if fine_start >= fine_end:
            return []
        if abs(self._current_pos - fine_end) <= abs(self._current_pos - fine_start):
            start, direction = fine_end, -1
        else:
            start, direction = fine_start, 1
        self._fine_dir = direction
        self._move_to(start, cfg.max_speed)
        span = fine_end - fine_start
        v_cap = cfg.hill_v_cap or cfg.fine_speed
        v_min = int(cfg.hill_v_min) if cfg.hill_v_min > 0 else v_cap

        def speed_schedule(pos, frame, score, running_max):
            return hill_speed_by_pos(pos, anchor, v_min, v_cap,
                                     near_steps=2.0 * cfg.fine_step,
                                     ramp_end_steps=float(fw))

        def on_score(pos: float, frame) -> float:
            score = self._score_frame(frame)
            frac = 0.55 + 0.30 * min(1.0, abs(pos - start) / span)
            self.sig_progress.emit(frac, PHASE_HILL, score, pos)
            return score

        hill_curve, reason, end_pos = cont_pass(
            self._focus, self._frame_reader, start, direction, span, v_cap,
            speed_schedule=speed_schedule,
            check=lambda: self._check(self._t_start + cfg.timeout_s * 0.85),
            on_score=on_score, on_log=self.sig_log.emit,
            poll_s=cfg.coarse_poll_s, freshness_ms=cfg.freshness_ms,
            interp_max_gap_ms=cfg.interp_max_gap_ms,
            early_stop_ratio=cfg.hill_ratio,
            early_stop_samples=cfg.hill_early_stop_samples,
            stop_accel=cfg.stop_accel_sps2, stop_latency_s=cfg.stop_latency_s,
            stop_safety_steps=cfg.stop_safety_steps,
            early_stop_rise=cfg.early_stop_rise,
            edge_allowance_steps=8.0 * cfg.fine_step)
        if reason:
            if hill_curve and not reason.startswith("aborted"):
                peak = pick_peak(hill_curve, float(anchor),
                                 prominence=cfg.peak_prominence,
                                 edge_margin_steps=2.0 * cfg.fine_step)
                if peak is not None and peak_is_complete(
                        hill_curve, peak.pos, peak.score,
                        cfg.lock_samples_required):
                    self.sig_log.emit(
                        f"hill pass cut short ({reason}) but curve contains "
                        f"the peak — locking on collected data")
                    reason = None
            if reason:
                # Stage 2 is already near focus: NO arm restore (the
                # stage-2 retry starts around the CURRENT position).
                self._current_pos = int(end_pos)
                raise _AfExit(self._stop_stage(reason, "fine",
                                               restore=False))
        self._curve.extend(hill_curve)
        self._current_pos = int(end_pos)
        return hill_curve

    # ------------------------------------------------------------------
    # Stage 3 (+ complete-peak edge acceptance)
    # ------------------------------------------------------------------

    def _lock_on(self, hill_curve: list[tuple[float, float]], anchor: float,
                 bounds: tuple[int, int], cfg: AutofocusConfig,
                 fw_scale: float = 1.0) -> tuple[int, object]:
        """V1 stage 3 plus the complete-peak gate: a peak near a window
        edge is accepted when the curve climbed OVER it (samples on both
        sides + dense near-peak samples) instead of failing with the
        hit-window message. The stationary end-max discipline is
        untouched. A peak at an INTERIOR fine-window edge raises a
        dedicated retryable failure (the coarse anchor missed the true
        peak — the stage-2 loop widens the window)."""
        window_lo, window_hi = bounds
        if not hill_curve:
            raise _AfExit(AutofocusResult(
                int(anchor), 0.0, self._curve,
                coarse_curve=self._low_curve,
                message="no clear focus peak in hill window", phase="fine",
                restore_on_fail=False))
        peak = pick_peak(hill_curve, float(anchor),
                         prominence=cfg.peak_prominence,
                         edge_margin_steps=2.0 * cfg.fine_step)
        if peak is None:
            raise _AfExit(AutofocusResult(
                int(anchor), 0.0, self._curve,
                coarse_curve=self._low_curve,
                message="no clear focus peak in hill window", phase="fine",
                restore_on_fail=False))
        # Edge check against the FINE-WINDOW bounds, not the curve extent
        # (the hill pass early-stops shortly after the peak by design).
        fw = int((cfg.fine_window_steps or max(cfg.coarse_step,
                                               3 * cfg.fine_step))
                 * fw_scale)
        fine_start = max(window_lo, int(anchor) - fw)
        fine_end = min(window_hi, int(anchor) + fw)
        margin = 2.0 * cfg.fine_step
        near_left = peak.pos - fine_start <= margin
        near_right = fine_end - peak.pos <= margin
        if (near_left or near_right) and cfg.fail_on_edge_peak:
            if peak_is_complete(hill_curve, peak.pos, peak.score,
                                cfg.lock_samples_required):
                # The pass climbed OVER the peak and only then hit the
                # boundary — the collected data reconstructs the peak
                # precisely, so this is not a hit-window failure.
                self.sig_log.emit("hill peak near the window edge but the "
                                  "curve contains the full peak — accepting")
            elif (near_left and fine_start > window_lo) \
                    or (near_right and fine_end < window_hi):
                # Interior edge: the coarse anchor missed the true peak —
                # the stage-2 loop widens the fine window and re-runs the
                # hill (the classic's one-shot extension). A stationary
                # re-measure around an edge peak usually fails (end-max
                # discipline), so it is no longer attempted here.
                raise _AfExit(AutofocusResult(
                    int(peak.pos), peak.score, self._curve,
                    coarse_curve=self._low_curve,
                    message="hill peak at interior window edge — widen "
                            "the fine window",
                    phase="fine", restore_on_fail=False))
            else:
                raise _AfExit(AutofocusResult(
                    int(peak.pos), peak.score, self._curve,
                    coarse_curve=self._low_curve,
                    message="peak at window edge — widen the search window",
                    phase="fine", peak_at_edge=True,
                    restore_on_fail=False))
        near = [p for p, s in hill_curve
                if s >= cfg.lock_score_frac * peak.score]
        fit = parabolic_fit(hill_curve, peak.pos, cfg.n_parabolic_points)
        fw = int((cfg.fine_window_steps or max(cfg.coarse_step,
                                               3 * cfg.fine_step))
                 * fw_scale)
        if fit is not None and abs(fit - anchor) > fw // 2:
            if fw_scale > 1.0:
                # The window was WIDENED because the coarse anchor is
                # unreliable — the fit is supported by the hill curve
                # and the anchor is the known-bad reference: re-measure
                # around the FIT (the stationary end-max discipline
                # still guards against a fit chasing a competing plane).
                self.sig_log.emit(f"widened window: hill fit {fit:.1f} far "
                                  f"from the coarse anchor {anchor:.1f} — "
                                  f"stationary re-measure around the fit")
                return self._stationary_remeasure(
                    PeakInfo(pos=fit, score=peak.score, at_edge=False),
                    bounds, cfg)
            self.sig_log.emit(f"hill fit {fit:.1f} far from the coarse "
                              f"anchor {anchor:.1f} — distrusting it")
            return self._stationary_remeasure(
                PeakInfo(pos=anchor, score=peak.score, at_edge=False),
                bounds, cfg)
        if len(near) >= cfg.lock_samples_required and fit is not None:
            final_pos = int(round(fit))
            final_pos = max(window_lo, min(window_hi, final_pos))
            self.sig_progress.emit(0.92, PHASE_LOCK, peak.score,
                                   float(final_pos))
            self.sig_log.emit(f"lock-on: direct fit {fit:.1f} from "
                              f"{len(near)} near-peak samples")
            return final_pos, peak
        return self._stationary_remeasure(peak, bounds, cfg)
