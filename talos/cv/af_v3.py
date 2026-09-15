"""Derivative-hybrid adaptive v3 (audit-revised — see
docs/PLAN.md handoff #11 / the plan project-talos-cheerful-stearns.md).

The sharpness-vs-defocus curve is near-Gaussian (S = A·exp(−d²/2σ²),
σ ≈ DOF/3): its second derivative is strongly negative near the peak,
zero at ≈ 1.1σ, positive beyond. v3 exploits it WITHOUT replacing the
v2 machinery — each derivative signal ADDS a hybrid branch:

- PROBE (audit-verified pure math in af_math.probe_classify): the
  curvature C = (S₋δ − 2S_c + S₊δ)/(δ²S_c) classifies the three-point
  probe: C strongly negative + the center-max VALLEY gate (audit
  #8/#9 — the between-planes valley must never read "near") → near via
  curvature (works where the v2 weaker-side test fails, e.g. small
  δ/σ); the v2 weaker-side test stays as the plateau OR-branch
  (C ≈ 0 on flat tops); C > +curv_out → flank (direction is the
  low-freq slope's job); below the RELATIVE floor (× the preflight
  baseline) → the low-freq call only.
- STAGE-1 STOP LADDER: the curvature stop (5-sample pass-only window,
  normalized metric 2a·h²/S_max below the per-objective threshold AND
  the sweep-aware detrended b gate AND a 2-window debounce) fires
  BEFORE the peak — the v2 post-peak early-stop, walk-away and
  direction guard stay as the other rungs. On the curvature-stop
  reason the peak analysis BYPASSES pick_peak + the edge check (the
  curve ends climbing — audit #1), excludes the settle frames, and
  returns the trusted vertex (≤ 0.7σ regime) or the stop position —
  `_coarse_anchor` then uses it directly (no parabolic re-fit —
  audit #12/#13).
- CASCADE DIRECTION GUARD: the v2 15%-drop test + sharp-rise gate stay
  EXACTLY as v2; an ADDITIONAL trigger fits the pass-only low-freq
  series vs POSITION and reverses when the slope's 2σ pooled-noise
  (the probe's stationary points — audit A10) interval lies entirely
  below zero, with the sharp-rise gate unmet (audit #7). One
  reversal, same as v2.

The v2 (AdaptiveAutofocusControllerV2) is frozen as the rollout
backup; every v3 signal that misfires degrades to the v2 behavior.
"""

from __future__ import annotations

import time

from talos.cv.af_adaptive import (
    DIRECTION_GUARD,
    PHASE_COARSE_PASS,
    START_PEAK_WALKAWAY,
    AdaptiveAutofocusControllerV2,
    cont_pass,
)
from talos.cv.af_math import (
    PeakInfo,
    ProbeResult,
    direction_guard_wrong_way,
    fit_slope_with_se,
    guard_reversal_test,
    pick_peak,
    pooled_noise_from_probe,
    probe_classify,
    window_curvature,
    window_parabola,
)
from talos.cv.autofocus import (
    _AfExit,
    AutofocusConfig,
    AutofocusResult,
)

CURVATURE_STOP = "coarse: curvature stop before the peak"


class AdaptiveAutofocusController(AdaptiveAutofocusControllerV2):
    """Derivative-hybrid adaptive v3 — the v2 pipeline plus the
    derivative signals above. The default `adaptive` strategy.

    Failure semantics: failures and stops end at the CURRENT position —
    the base-class arm restore is disabled by the `_restore_on_fail`
    policy (user policy; the stored-knowledge controllers keep it)."""

    def __init__(self, focus, frame_reader, parent=None,
                 abort_check=None):
        # V1's frozen __init__ takes only (focus, frame_reader, parent) —
        # the base-class abort latch is set as an attribute afterwards.
        super().__init__(focus, frame_reader, parent)
        self._restore_on_fail = False
        self._abort_check = abort_check

    # ------------------------------------------------------------------
    # Anchor (the curvature-stop trusted vertex / the cluster probe point)
    # ------------------------------------------------------------------

    def _probe_anchor(self, center: int, probe: ProbeResult,
                      cfg: AutofocusConfig) -> float:
        """The near-cluster branch anchors stage 2 at the BEST probe
        point (the strongest sub-peak the probe touched), not the arm —
        the arm sits on the multi-peak plateau's shoulder."""
        anchor = getattr(self, "_cluster_anchor", None)
        return float(anchor) if anchor is not None else float(center)

    def _coarse_anchor(self, coarse_peak, cfg: AutofocusConfig) -> float:
        """The curvature-stop bypass already computed the anchor (the
        trusted vertex or the stop position) — re-fitting the stage-1
        curve would drag it off with the settle outlier and the
        climbing curve end (audit #12/#13). All other stage-1 reasons
        keep the v2 parabolic re-fit."""
        if getattr(coarse_peak, "curvature_stop", False):
            return coarse_peak.pos
        return super()._coarse_anchor(coarse_peak, cfg)

    # ------------------------------------------------------------------
    # Probe (v2 motion + the hybrid classification)
    # ------------------------------------------------------------------

    def _probe(self, center: int, bounds: tuple[int, int],
               cfg: AutofocusConfig) -> ProbeResult:
        """The v2 three-point motion (center-first, return-to-center)
        with the hybrid classification. The relative floor is
        `probe_score_floor_ratio` × the preflight score (the arm's own
        baseline — far-defocus when armed far, peak-level when armed
        near). Sets `_probe_samples` on BOTH paths (audit B3 — `_run`
        populates it only on the far path)."""
        probe = super()._probe(center, bounds, cfg)
        self._probe_samples = probe.samples
        self._cluster_anchor = None
        if len(probe.samples) < 3:
            return probe  # incomplete probe — blind (v2 log already out)
        # super() samples in move order: [center, center−δ, center+δ].
        (_pos_c, sharp_c, low_c), (_pos_lo, sharp_lo, low_lo), \
            (_pos_hi, sharp_hi, low_hi) = probe.samples
        delta = cfg.probe_step_steps or 3 * cfg.coarse_step
        delta = max(int(delta), 3 * cfg.fine_step)
        floor = cfg.probe_score_floor_ratio * \
            getattr(self, "_preflight_score", 0.0)
        verdict = probe_classify(
            sharp_lo, sharp_c, sharp_hi, float(delta),
            low_lo, low_c, low_hi, floor,
            cfg.probe_curv_in, cfg.probe_curv_out,
            cfg.probe_peak_ratio, cfg.probe_min_slope,
            cfg.probe_cluster_center_ratio, cfg.probe_cluster_side_ratio,
            1.2, cfg.probe_cluster_min_score)
        self._probe_verdict = verdict
        if verdict.near and cfg.near_window_steps > 0 \
                and cfg.fine_window_steps < cfg.near_window_steps:
            # the near verdict tolerates the arm up to ~±0.7σ off the
            # true peak — the hill window (and the fit-distrust
            # tolerance fw//2) must cover that, not just the coarse
            # localization error the default fw is sized for
            cfg.fine_window_steps = cfg.near_window_steps
            self.sig_log.emit(
                f"probe: near entry — stage-2 window widened to "
                f"{cfg.near_window_steps} steps")
        if verdict.branch == "near-cluster":
            # the arm sits on the multi-peak plateau's shoulder: stage 2
            # enters directly, anchored at the BEST probe point
            best = max(probe.samples, key=lambda s: s[1])
            self._cluster_anchor = best[0]
            self.sig_log.emit(
                f"probe: near (cluster) — anchoring at the best probe "
                f"point {best[0]:.0f}")
        elif verdict.near:
            branch = "curvature+max" if verdict.branch == "near-curvature" \
                else "plateau"
            self.sig_log.emit(f"probe: near ({branch})")
        else:
            self.sig_log.emit(
                f"probe: {verdict.branch} — direction "
                f"{verdict.direction if verdict.direction is not None else 'blind'}"
                f" (curv {verdict.curvature:.2e})")
        return ProbeResult(verdict.near, verdict.direction, probe.samples)

    # ------------------------------------------------------------------
    # Stage 1 (stop ladder + curvature bypass + cascade guard)
    # ------------------------------------------------------------------

    def _coarse_pass(self, center: int, bounds: tuple[int, int],
                     cfg: AutofocusConfig, direction: int | None = None):
        """Stage 1 v3: the v2 dispatch plus the derivative ladder. The
        guard callback runs the rungs per cont_pass's per-sample order:
        walk-away (v2) → direction guard (v2 drop test + the v3 2σ fit
        trigger, first pass only) → the curvature stop (every pass,
        pass-only window — the probe seeds and settle frames enter the
        curve only AFTER the pass, so the guard's window is pass-only
        by construction — audit #4). The curvature state lives inside
        each cont_pass invocation (auto-reset on the reversal); the
        blind branch (direction None) starts AT the arm sweeping +1
        with the early direction check armed — the pass decides its
        own direction from its first samples (audit B4: the v2
        self-recursion would dispatch into THIS override, so the branch
        is re-implemented explicitly)."""
        window_lo, window_hi = bounds
        v0 = cfg.coarse_speed or cfg.max_speed
        low_series: list[float] = []
        low_pos: list[float] = []

        sigma_pooled = None
        if len(self._probe_samples) >= 3:
            sigma_pooled = pooled_noise_from_probe(
                [low for _p, _s, low in self._probe_samples],
                [p for p, _s, _low in self._probe_samples])
            # CAP at 5% of the mean low-freq score: on structured fields
            # (multi-peak plateaus) the probe's three points straddle
            # real sub-peaks and the detrended residual reads as noise
            # 10-50× the honest stationary level — the 2σ trigger would
            # be permanently disabled. The known stationary noise is
            # 1-5%; beyond that the estimate is structure, not noise.
            mean_low = sum(l for _p, _s, l in self._probe_samples) \
                / len(self._probe_samples)
            sigma_pooled = min(sigma_pooled, 0.05 * max(mean_low, 1e-9))

        def make_guard(first_pass: bool):
            state = {"prev_neg": False, "b_early": None,
                     "stop_curv": None, "stop_b": None,
                     "stop_vertex": None, "diag_done": False}

            def guard(curve):
                scores = [s for _, s in curve]
                # rung: start-on-peak walk-away (v2, unchanged)
                if len(curve) >= 5 and scores[0] >= max(scores) \
                        and all(s < (1.0 - cfg.early_stop_ratio) * scores[0]
                                for s in scores[-2:]):
                    return START_PEAK_WALKAWAY
                if first_pass:
                    # rung: the v2 direction guard (pass-only series,
                    # unchanged — the fallback must stay intact)
                    if len(curve) <= cfg.guard_samples \
                            and len(low_series) >= 3:
                        if direction_guard_wrong_way(
                                scores, low_series, cfg.guard_drop_ratio,
                                cfg.early_stop_rise):
                            return DIRECTION_GUARD
                    # rung: the v3 cascade trigger — the 2σ pooled-noise
                    # slope test (audit A10) at guard_fit_samples, with
                    # the sharp-rise gate unmet (audit #7)
                    if len(curve) == cfg.guard_fit_samples \
                            and sigma_pooled is not None \
                            and cfg.guard_fit_samples >= 3:
                        if not any(s >= scores[0] * (1.0 + cfg.early_stop_rise)
                                   for s in scores) \
                                and guard_reversal_test(
                                    low_pos[:len(low_series)],
                                    low_series, sigma_pooled,
                                    cfg.guard_sigma):
                            self.sig_log.emit(
                                "direction guard: slope 2σ-negative "
                                f"(pooled σ {sigma_pooled:.1f}) — reversing")
                            return DIRECTION_GUARD
                    # one-shot diagnostic once BOTH guard windows closed:
                    # what the guard saw, and why it did not reverse
                    if not state["diag_done"] \
                            and len(curve) > max(cfg.guard_samples,
                                                 cfg.guard_fit_samples) \
                            and len(low_series) >= 3:
                        state["diag_done"] = True
                        fall = (low_series[0] - low_series[-1]) \
                            / max(low_series[0], 1e-9)
                        rise = max(scores) >= scores[0] \
                            * (1.0 + cfg.early_stop_rise)
                        self.sig_log.emit(
                            f"direction guard: low-freq fall {fall * 100:.0f}% "
                            f"over {len(low_series)} samples, sharp rise "
                            f"gate {'MET' if rise else 'unmet'} — no "
                            f"reversal")
                # rung: the curvature stop (every pass; pass-only window;
                # detrended sweep-aware b gate — audit L3 — and the
                # 2-consecutive-window debounce — audit A9)
                if len(curve) >= cfg.coarse_curv_window:
                    win_x = [p for p, _ in curve[-cfg.coarse_curv_window:]]
                    win_y = scores[-cfg.coarse_curv_window:]
                    fit = window_parabola(win_x, win_y)
                    if fit is not None:
                        a, b = fit
                        # reuse the fit: window_curvature refits the same
                        # window otherwise (an extra lstsq per sample)
                        wc = window_curvature(win_x, win_y, fit=fit)
                        if wc is not None:
                            curv, _b = wc
                            if state["b_early"] is None \
                                    and len(curve) >= cfg.guard_fit_samples:
                                early = fit_slope_with_se(
                                    [p for p, _ in
                                     curve[:cfg.guard_fit_samples]],
                                    scores[:cfg.guard_fit_samples])
                                state["b_early"] = early[0] if early else 0.0
                            b_eff = b - (state["b_early"] or 0.0)
                            neg = curv < 0
                            # the vertex-in-window gate: the metric's
                            # far-tail oscillations pass the threshold and
                            # the b gate, but their parabola vertex lies
                            # many spacings outside the window (a true
                            # ≤0.7σ approach puts it ~0.25σ ahead —
                            # hardware-verified: a tail fire's "vertex"
                            # was 483 steps = 14h outside)
                            sx = sorted(win_x)
                            gaps = [hi - lo for lo, hi in zip(sx, sx[1:])]
                            h_win = gaps[len(gaps) // 2] if gaps else 0.0
                            vertex = None
                            if a < 0 and h_win > 0:
                                candidate = sum(win_x) / len(win_x) \
                                    - b / (2.0 * a)
                                if abs(candidate - sum(win_x) / len(win_x)) \
                                        <= 2.5 * h_win:
                                    vertex = candidate
                            if (neg and state["prev_neg"]
                                    and cfg.coarse_curv_stop > 0
                                    and curv < -cfg.coarse_curv_stop
                                    and b_eff * direction > 0
                                    and vertex is not None):
                                state["stop_curv"] = curv
                                state["stop_b"] = b_eff
                                state["stop_vertex"] = vertex
                                return CURVATURE_STOP
                            state["prev_neg"] = neg
                return None

            return guard, state

        if direction is None:
            # v3 blind fallback: start AT the arm and sweep the DEFAULT
            # direction (cfg.coarse_direction: 0 → +1 away from the
            # sample — the safe default; ±1 forces it — scenario
            # overrides), letting the pass's OWN early samples decide:
            # the drop test and the 2σ fall trigger (rise gate unmet)
            # reverse it toward the peak side, exactly like the
            # probe-directed pass. The probe's 3 points are only
            # trustworthy for the near-peak call on structured fields
            # (user-verified) — the stage-1 direction comes from the
            # pass itself. Strictly better than the v2 nearest-edge
            # walk (frozen in V2): the peak region sits near the arm,
            # so the pass reaches structure in ~100s of steps instead
            # of ~2500, and the walk-away / early-stop / curvature
            # rungs stop at the first structure crossed either way.
            start = int(self._current_pos)
            direction = cfg.coarse_direction if cfg.coarse_direction != 0 \
                else +1
            span = (window_hi - start) if direction > 0 \
                else (start - window_lo)
            if span < 3 * cfg.fine_step:
                # The arm can sit outside the search window (an absolute
                # bounds override does that): the span then measured
                # negative, cont_pass halted on its first sample and the
                # run ended with an EMPTY stage-1 curve reported as "no
                # clear focus peak (flat curve)" — the same trap the
                # directed branch guards against below, but with the
                # direction handed to us there is nothing to flip to.
                self.sig_log.emit(
                    f"coarse: the arm at {start} leaves no room toward "
                    f"{direction:+d} inside the search window "
                    f"[{window_lo}, {window_hi}] — re-arm or widen the window")
                raise _AfExit(AutofocusResult(
                    start, 0.0, self._curve,
                    # Name BOTH causes and the actual numbers: the old text
                    # blamed the arm position, but a 0 µm window in the
                    # settings produces the same dead end.
                    message="search window is empty "
                            f"([{window_lo}, {window_hi}] around the arm at "
                            f"{start}) — widen the window or re-arm the axis",
                    phase="coarse"))
            guard, state = make_guard(first_pass=True)
            self.sig_log.emit(f"coarse: blind sweep from the arm toward "
                              f"{start + direction * span} "
                              f"(direction {direction:+d}, early "
                              f"direction check armed)")
        else:
            start = int(self._current_pos)
            span = (window_hi - start) if direction > 0 \
                else (start - window_lo)
            if span < 3 * cfg.fine_step:
                self.sig_log.emit(f"coarse: direction {direction:+d} has no "
                                  f"room (span {span}) — blind sweep")
                return self._coarse_pass(center, bounds, cfg, direction=None)
            self.sig_log.emit(f"coarse: probe-guided sweep from {start} "
                              f"toward {start + direction * span} "
                              f"(direction {direction:+d}, guard armed)")
            guard, state = make_guard(first_pass=True)

        def on_score(pos: float, frame) -> float:
            score = self._score_frame(frame)
            low = self._score_frame_low(frame)
            low_series.append(low)
            low_pos.append(pos)
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
            guard, state = make_guard(first_pass=False)
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
                edge_allowance_steps=8.0 * cfg.fine_step,
                guard=guard)
            if reason == DIRECTION_GUARD:
                # One reversal only — accept what we have and let the
                # peak checks adjudicate.
                self.sig_log.emit("second direction guard — accepting the pass")
                reason = None
        curvature_stop = reason == CURVATURE_STOP
        if curvature_stop:
            self.sig_log.emit(
                f"coarse: curvature stop before the peak at {end_pos} "
                f"(curv {state['stop_curv']:.3f}, detrended b "
                f"{state['stop_b']:+.1f})")
            reason = None
        elif reason == START_PEAK_WALKAWAY:
            self.sig_log.emit("coarse pass started on the peak and walked "
                              "away — accepting the curve")
            reason = None
        if reason:
            # Track where the axis actually halted BEFORE building the
            # result — with the no-restore policy the reported position
            # must be the stop position, not the staging position.
            self._current_pos = int(end_pos)
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
        # Seed the probe's sharp samples (real stage-1 measurements) and
        # position-sort (pick_peak's moving average runs over the
        # SEQUENCE — a non-monotonic prefix fakes peaks; sim-verified).
        prefix = [(p, s) for p, s, _low in self._probe_samples]
        prefix_set = set(prefix)
        stage1_curve = prefix + stage1_curve
        stage1_curve.sort(key=lambda item: item[0])
        self._stage1_curve = stage1_curve
        # The probe samples were already plotted during the probe — drop
        # exactly those by identity (a positional slice would also drop
        # real pass points when the sweep went below center−δ).
        self._curve.extend((p, s) for p, s in stage1_curve
                           if (p, s) not in prefix_set)
        self._current_pos = int(end_pos)
        if curvature_stop:
            # The curvature-stop BYPASS (audit #1): skip pick_peak and
            # the edge check entirely (the curve ends CLIMBING — a
            # picked peak would be flagged at_edge and aborted). The
            # quality gate and the trusted-vertex discipline stand in,
            # analyzed on the PASS-ONLY series (exclude the settle
            # frames AND the probe prefix — audit #4's principle: the
            # stationary probe points are outliers, and a sharp mount
            # sample in the prefix would inflate the baseline).
            pass_only = [(p, s) for p, s in stage1_curve
                         if p != float(end_pos)
                         and (p, s) not in prefix_set]
            if not pass_only:
                raise _AfExit(AutofocusResult(
                    self._current_pos, 0.0, self._curve,
                    coarse_curve=self._low_curve,
                    message="no clear focus peak (flat curve)",
                    phase="coarse"))
            s_max = max(s for _p, s in pass_only)
            baseline = (pass_only[0][1] + pass_only[-1][1]) / 2.0
            if baseline >= 0 and s_max < baseline * (
                    1.0 + cfg.quality_threshold):
                raise _AfExit(AutofocusResult(
                    int(end_pos), s_max, self._curve,
                    coarse_curve=self._low_curve,
                    message="peak too weak vs baseline", phase="coarse"))
            trusted = cfg.coarse_curv_vertex < 0 \
                and state["stop_curv"] <= cfg.coarse_curv_vertex
            pos = state["stop_vertex"] if trusted else float(end_pos)
            pos = max(window_lo, min(window_hi, pos))
            peak = PeakInfo(pos=float(pos), score=s_max, at_edge=False)
            peak.curvature_stop = True
            self.sig_log.emit(
                f"coarse peak {pos:.1f} (score {s_max:.0f}, "
                f"{'trusted vertex' if trusted else 'stop position'})")
            return peak
        # The v2 normal path (walk-away / early-stop / full pass).
        peak = pick_peak(stage1_curve, float(center),
                         prominence=cfg.peak_prominence,
                         edge_margin_steps=2.0 * cfg.fine_step)
        if peak is None:
            raise _AfExit(AutofocusResult(
                self._current_pos, 0.0, self._curve,
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