"""Backlash auto-calibration: measure the mechanical backlash of the
open-loop focus axis by sweeping a feature-rich field in BOTH directions
and measuring the peak offset between the two curves.

Physics (the sim models this exactly — see SimFocusStage):
- The STATUS? counter is the COMMANDED axis; the load (what the camera
  sees) lags the counter by B in the current direction after a reversal.
- An UP sweep (dir +1, load = counter − B) peaks in counter coordinates
  at truth + B; a DOWN sweep (dir −1, load = counter + B) peaks at
  truth − B. The peak offset is 2B.
- The state at sweep start depends on the staging direction, so the
  sequence below forces a known backlash state with a two-leg approach
  (drop past the start, return: the return ALWAYS reverses → dead zone
  taken up → deterministic load offset).

Run on a feature-rich field (flakes, wafer edge, scratches) at any
magnification — B is a mechanism property, independent of optics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QObject, Signal

from talos.cv.af_math import parabolic_fit
from talos.cv.af_roi import roi_for_resolution
from talos.cv.autofocus import (clamp_to_soft_limits, continuous_scan,
                                move_to_verified)
from talos.cv.focus_metric import METRICS

logger = logging.getLogger(__name__)


@dataclass
class BacklashCalConfig:
    sweep_steps: int = 150          # half-span of each sweep (±30 µm at 0.2
                                    # µm/step), CLAMPED to the focus soft
                                    # limits at run() time — the approach
                                    # legs need the same room again
    speed: int = 30                 # steps/s (6 µm/s): SLOW — sample density
                                    # for the measurement AND minimal motion
                                    # blur (0.24 µm at 40 ms exposure); the
                                    # operator must keep sight of the sample
    settle_steps: int = 100         # the past-the-start/end approach legs —
                                    # MUST exceed the backlash being measured
                                    # or the forced-state dance never takes
                                    # the dead zone up (hardware-verified:
                                    # 25 steps → 15 µm garbage; 100 → sane)
    poll_s: float = 0.05
    freshness_ms: float = 800.0
    interp_max_gap_ms: float = 300.0
    timeout_s: float = 120.0
    max_backlash_steps: int = 250   # sanity ceiling (50 µm)
    um_per_step: float = 0.2
    metric: str = "tenengrad"
    roi_norm: tuple | None = None


@dataclass
class BacklashResult:
    backlash_steps: int = 0
    backlash_um: float = 0.0
    up_curve: list[tuple[float, float]] = field(default_factory=list)
    down_curve: list[tuple[float, float]] = field(default_factory=list)
    success: bool = False
    aborted: bool = False
    message: str = ""


class BacklashCalibrator(QObject):
    sig_progress = Signal(float, str)   # fraction, phase label
    sig_done = Signal(object)           # BacklashResult
    sig_log = Signal(str)

    def __init__(self, focus, frame_reader, parent: QObject | None = None,
                 abort_check=None):
        super().__init__(parent)
        self._focus = focus
        self._frame_reader = frame_reader
        self.abort_requested = False
        self._abort_reason = "aborted by user"
        self._abort_check = abort_check  # the proxy's enqueue-gap latch

    def request_abort(self, reason: str | None = None) -> None:
        self.abort_requested = True
        if reason:
            if not reason.startswith("aborted"):
                reason = "aborted — " + reason
            self._abort_reason = reason

    def run(self, center: int, cfg: BacklashCalConfig | None = None) -> BacklashResult:
        cfg = cfg or BacklashCalConfig()
        self.abort_requested = False
        self._abort_reason = "aborted by user"
        deadline = time.monotonic() + cfg.timeout_s
        bottom, top = center - cfg.sweep_steps, center + cfg.sweep_steps
        # Clamp the WHOLE travel — including the past-the-end approach legs —
        # to the focus axis's soft limits: this sweep used to run
        # center ± (sweep + settle) steps with no limits read and no SLIM
        # check, i.e. with none of the protection every autofocus sweep gets.
        margin = max(cfg.settle_steps, 1)
        window = clamp_to_soft_limits(self._focus, bottom, top, margin,
                                      on_log=self.sig_log.emit)
        if window is None:
            return BacklashResult(
                message="sweep does not fit inside the focus soft limits — "
                        "reduce sweep_steps or re-centre the axis")
        bottom, top = window
        sweep = top - bottom
        if sweep < 2 * cfg.settle_steps:
            return BacklashResult(
                message=f"only {sweep} steps of travel inside the soft limits "
                        f"— the sweep needs more than 2× the "
                        f"{cfg.settle_steps}-step approach leg")
        up_curve: list[tuple[float, float]] = []
        down_curve: list[tuple[float, float]] = []

        def check() -> str | None:
            if self.abort_requested:
                return self._abort_reason
            if self._abort_check is not None:
                reason = self._abort_check()
                if reason:
                    self.abort_requested = True
                    self._abort_reason = reason
                    return reason
            if time.monotonic() > deadline:
                return "wall-clock budget exceeded"
            return None

        def stop_stage(reason: str) -> BacklashResult:
            """Failure path — keep the measured curves for diagnosis, and
            RETURN the axis to the arm position (a run left at the sweep
            bottom walks the arm down 30 µm per run — hardware-verified:
            the plane left the sweep within three runs)."""
            try:
                self._focus.stop()
            except Exception:  # noqa: BLE001
                pass
            if not reason.startswith("aborted"):
                try:
                    move_to_verified(self._focus, center, cfg.speed)
                except Exception as exc:  # noqa: BLE001
                    self.sig_log.emit(f"position restore failed: {exc}")
            return BacklashResult(aborted=reason.startswith("aborted"),
                                  up_curve=up_curve, down_curve=down_curve,
                                  message=reason)

        metric_fn = METRICS.get(cfg.metric, METRICS["tenengrad"])

        def on_score(pos: float, frame) -> float | None:
            roi = roi_for_resolution(cfg.roi_norm, frame.shape) \
                if cfg.roi_norm else None
            return metric_fn(frame, roi)

        try:
            # Force a deterministic backlash state: drop past the bottom,
            # return (the return always reverses → dead zone taken up →
            # load = bottom − B with direction +1).
            move_to_verified(self._focus, bottom - cfg.settle_steps, cfg.speed)
            move_to_verified(self._focus, bottom, cfg.speed)
            move_to_verified(self._focus, bottom - cfg.settle_steps, cfg.speed)
            move_to_verified(self._focus, bottom, cfg.speed)

            self.sig_progress.emit(0.05, "sweep up")
            up_curve, reason, _end = continuous_scan(
                self._focus, self._frame_reader, bottom, top, cfg.speed,
                poll_s=cfg.poll_s, freshness_ms=cfg.freshness_ms,
                interp_max_gap_ms=cfg.interp_max_gap_ms, check=check,
                on_score=on_score, on_log=self.sig_log.emit)
            if reason:
                return stop_stage(reason)

            self.sig_progress.emit(0.55, "sweep down")
            # Continue past the top, then return: the return reverses →
            # load = top + B with direction −1.
            move_to_verified(self._focus, top + cfg.settle_steps, cfg.speed)
            move_to_verified(self._focus, top, cfg.speed)
            down_curve, reason, _end = continuous_scan(
                self._focus, self._frame_reader, top, bottom, cfg.speed,
                poll_s=cfg.poll_s, freshness_ms=cfg.freshness_ms,
                interp_max_gap_ms=cfg.interp_max_gap_ms, check=check,
                on_score=on_score, on_log=self.sig_log.emit)
            if reason:
                return stop_stage(reason)

            # ---- Measure the peak offset (2B) ---------------------------
            if len(up_curve) < 3 or len(down_curve) < 3:
                return stop_stage("not enough scored frames — no camera or stalled")
            for name, curve in (("up", up_curve), ("down", down_curve)):
                if self._flat_check(curve) is None:
                    return stop_stage(
                        f"flat {name} curve — no contrast in the field "
                        f"(use a crisp single-plane feature)")
                pos, _ = max(curve, key=lambda pt: pt[1])
                # Edge check against the SWEEP bounds (not the scored
                # range — camera startup latency skips the first part of
                # the sweep, and an interior peak would look 'at the
                # edge' of the scored span).
                if pos <= bottom + 5 or pos >= top - 5:
                    return stop_stage(
                        f"{name} peak at sweep edge — the sharp plane is "
                        f"outside the sweep; reposition the field")
            up_pos, _ = max(up_curve, key=lambda pt: pt[1])
            down_pos, _ = max(down_curve, key=lambda pt: pt[1])
            up_fit = parabolic_fit(up_curve, up_pos, 5)
            down_fit = parabolic_fit(down_curve, down_pos, 5)
            if up_fit is None or down_fit is None:
                return stop_stage("degenerate peak fit — choose a feature-rich field")
            backlash = abs(up_fit - down_fit) / 2  # offset = 2B
            if backlash > cfg.max_backlash_steps:
                return stop_stage(
                    f"implausible backlash {backlash:.0f} steps — "
                    f"retry on a single-plane feature-rich field")
            # Return the axis to the arm position so the next run measures
            # the SAME plane.
            move_to_verified(self._focus, center, cfg.speed)
            result = BacklashResult(
                backlash_steps=int(round(backlash)),
                backlash_um=round(backlash * cfg.um_per_step, 2),
                up_curve=up_curve, down_curve=down_curve,
                success=True, message="ok")
            self.sig_progress.emit(1.0, "done")
            self.sig_done.emit(result)
            return result
        except Exception as exc:  # noqa: BLE001
            try:
                self._focus.stop()
            except Exception:  # noqa: BLE001
                pass
            return BacklashResult(up_curve=up_curve, down_curve=down_curve,
                                  message=f"calibration error: {exc!r}")

    @staticmethod
    def _flat_check(curve: list[tuple[float, float]]) -> np.ndarray | None:
        """None when the curve is featureless. RELATIVE check: real
        hardware has a Tenengrad noise floor of a few units, so the
        variation must be a meaningful fraction of the peak (hardware-
        verified: a 4-8-unit noise floor passed an absolute threshold
        and produced a garbage 10 µm estimate)."""
        ys = [s for _, s in curve]
        arr = np.asarray(ys, dtype=float)
        peak = arr.max()
        if peak <= 1e-9 or (arr.max() - arr.min()) <= 0.1 * peak:
            return None
        return arr - arr.mean()
