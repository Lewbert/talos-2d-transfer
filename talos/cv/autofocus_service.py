"""AutofocusService: the GUI-thread orchestrator for autofocus.

- AF-S (focus and stop): the exclusive focus-worker job running the v3
  adaptive controller — the app's ONLY autofocus strategy. AF-C and the
  older AF-S algorithms stay in the codebase as stored knowledge
  (talos/cv/af_c.py, talos/cv/af_adaptive.py, talos/cv/autofocus.py) —
  reachable from the sim suites and the bench tools, never from the app.
- Input-abort: any user motion/snapshot submission while a job runs
  (stage jogs, go-to, home/zero, manual focus, snapshot) aborts the job
  and the input then proceeds — input wins. Camera exposure/gain/WB
  changes and the yudian setpoint do NOT abort. Inputs arriving during
  the arm-delay window cancel the pending arm instead of starting.
- Backlash calibration: launches the FocusProxy job, stores the result
  into the active objective row (settings.save()).
"""

from __future__ import annotations

import logging
import time

from PySide6.QtCore import QObject, QTimer, Signal

from talos.cv.af_math import build_config, um_to_steps
from talos.cv.autofocus import (
    AutofocusConfig,
    AutofocusRequest,
    AutofocusResult,
)
from talos.cv.backlash_cal import BacklashCalConfig
from talos.hal.proxies.focus_proxy import FocusProxy

logger = logging.getLogger(__name__)

_MANUAL_FOCUS_METHODS = {"set_speed", "move_rel", "move_abs", "zero"}
_XY_MOTION_METHODS = {"move", "step", "move_rel_um", "move_abs_um",
                      "move_abs_pulses", "move_continuous", "home"}
# Submissions that must NOT abort a running job: the service's own jobs,
# the jog release (its set_speed sibling already aborted), and camera
# property tweaks (exposure/gain/WB — user decision: they do not
# interrupt AF).
_ABORT_EXCLUSIONS = {
    ("focus", "autofocus"), ("focus", "backlash_calibrate"),
    ("focus", "stop"), ("camera", "set_property"),
}


class AutofocusService(QObject):
    sig_af_finished = Signal(object)            # AutofocusResult
    sig_af_progress = Signal(float, int, float, float)  # relayed from the proxy
    sig_af_curve_secondary = Signal(float, float)  # low-freq series (pos, score)
    sig_af_log = Signal(str)
    sig_cal_finished = Signal(object)           # BacklashResult

    def __init__(self, manager, settings, state, frame_slot,
                 parent: QObject | None = None):
        super().__init__(parent)
        self._manager = manager
        self._settings = settings
        self._state = state                     # AppState
        self._frame_slot = frame_slot
        self._focus: FocusProxy = manager.device("focus")
        self._job_kind = ""                     # "" | "af_s" | "cal"
        self._roi_norm: tuple | None = None
        self._pending_request: AutofocusRequest | None = None
        # Optimistic until the first connect signal: a dead focus worker
        # must refuse starts instead of arming into the void (stuck
        # AUTOFOCUS/busy forever).
        self._focus_connected = True

        self._arm_timer = QTimer(self)
        self._arm_timer.setSingleShot(True)
        self._arm_timer.setInterval(350)        # one camera tick for fresh frames
        self._arm_timer.timeout.connect(self._on_arm_timer)

        manager.sig_job_submitted.connect(self._on_job_submitted)
        manager.sig_device_state.connect(self._on_device_state)
        self._focus.sig_af_done.connect(self._on_af_done)
        self._focus.sig_af_progress.connect(self.sig_af_progress)
        self._focus.sig_af_curve_secondary.connect(self.sig_af_curve_secondary)
        self._focus.sig_af_log.connect(self.sig_af_log)
        self._focus.sig_cal_done.connect(self._on_cal_done)
        # STOP ALL reaches the service through the focus proxy so it also
        # cancels a pending arm (the proxy's own abort covers a running job).
        self._focus.sig_stop_requested.connect(self._on_stop_requested)

    # ------------------------------------------------------------------
    # Settings-derived helpers
    # ------------------------------------------------------------------

    @property
    def _af_cfg(self) -> dict:
        """The `autofocus` settings SECTION (the `_af_c` name predates
        the AF-C removal — not the af_c.* keys)."""
        return self._settings.section("autofocus")

    @property
    def _focus_cfg(self) -> dict:
        return self._settings.device("focus")

    def _objective_row(self) -> dict:
        rows = self._settings.get("objectives") or []
        index = int(self._state.objective)
        return rows[min(index, len(rows) - 1)] if rows else {}

    def _um_per_step(self) -> float:
        return float(self._focus_cfg.get("um_per_step", 0.2))

    @property
    def busy(self) -> bool:
        return bool(self._job_kind)

    def _on_device_state(self, key: str, payload: dict) -> None:
        """Track the focus proxy's connectivity — a dead worker must
        refuse starts (an armed job into a dead event loop strands the
        service in AUTOFOCUS/busy forever)."""
        if key == "focus" and "connected" in payload:
            self._focus_connected = bool(payload["connected"])

    # ------------------------------------------------------------------
    # AF-S
    # ------------------------------------------------------------------

    def _default_roi(self) -> tuple | None:
        """Center 1280×720-equivalent crop when no user ROI is drawn —
        metric computation on ~2× fewer pixels than the full 1080p frame."""
        roi = self._af_cfg.get("default_roi_norm")
        if not isinstance(roi, (list, tuple)) or len(roi) != 4:
            return None
        return tuple(float(v) for v in roi)

    def start_af_s(self, roi_norm: tuple | None = None,
                   bounds: tuple[int, int] | None = None) -> None:
        """One-shot focus-and-stop, armed at the current position.
        ``bounds`` (absolute stage steps) overrides the symmetric search
        window — asymmetric scenario-specific AF."""
        if self.busy:
            logger.warning("Autofocus: busy (%s) — ignoring start", self._job_kind)
            return
        if not self._focus_connected:
            logger.warning("Autofocus: focus stage not connected — refusing start")
            self.sig_af_finished.emit(AutofocusResult(
                best_position=self._manager.focus_position, best_score=0.0,
                message="focus stage not connected — autofocus unavailable",
                phase="connect"))
            return
        self._roi_norm = roi_norm if roi_norm is not None else self._default_roi()
        cfg = self._make_config(mode="AF_S")
        self._launch("af_s", AutofocusRequest(
            center_steps=self._manager.focus_position, config=cfg,
            roi_norm=self._roi_norm, bounds=bounds))

    # ------------------------------------------------------------------
    # Backlash calibration
    # ------------------------------------------------------------------

    def calibrate_backlash(self) -> None:
        if self.busy:
            logger.warning("Autofocus: busy (%s) — ignoring calibrate", self._job_kind)
            return
        if not self._focus_connected:
            logger.warning("Autofocus: focus stage not connected — "
                           "ignoring calibrate")
            return
        cfg = BacklashCalConfig(
            sweep_steps=um_to_steps(60.0, self._um_per_step()),
            um_per_step=self._um_per_step())
        self._job_kind = "cal"
        self._manager.submit(
            "focus", "backlash_calibrate",
            (self._manager.focus_position, cfg))
        self._state.set_mode("AUTOFOCUS")

    def _on_cal_done(self, result) -> None:
        if self._job_kind != "cal":
            return
        self._job_kind = ""
        # Only revert OUR mode — a concurrent SCAN owns the axes now and
        # must not be clobbered back to MANUAL mid-scan.
        if self._state.mode == "AUTOFOCUS":
            self._state.set_mode("MANUAL")
        if result.success:
            focus_cfg = self._settings.data.setdefault(
                "devices", {}).setdefault("focus", {})
            focus_cfg["backlash_um"] = result.backlash_um
            focus_cfg["backlash_measured_at"] = time.strftime(
                "%Y-%m-%dT%H:%M:%S")
            self._settings.save()
            self.sig_af_log.emit(
                f"backlash stored (mechanism): {result.backlash_um} µm "
                f"({result.backlash_steps} steps)")
        self.sig_cal_finished.emit(result)

    # ------------------------------------------------------------------
    # Job plumbing
    # ------------------------------------------------------------------

    def _make_config(self, mode: str) -> AutofocusConfig:
        """µm settings → step-based AutofocusConfig for the current
        objective.

        The autofocus NEVER touches camera exposure/gain (user rule) —
        the blur-budget warning is computed against the exposure the
        camera actually has."""
        row = self._objective_row()
        # Backlash is a MECHANISM property of the focus axis — one value
        # serves every objective (measured at the highest magnification,
        # where the sharpness peak is narrowest).
        row = dict(row)
        row["backlash_um"] = float(self._focus_cfg.get(
            "backlash_um", row.get("backlash_um", 0.0)))
        af_cfg = dict(self._af_cfg)
        af_cfg["af_exposure_us"] = float(self._settings.device("camera").get(
            "exposure_us", af_cfg.get("af_exposure_us", 20000)))
        # the speed reference: the LOWEST-POWER objective's NA — its
        # auto multiplier is 1.0 (the table's max speed = the global
        # base); every other objective scales by (na_min/na)²
        rows = self._settings.get("objectives") or []
        na_min = min((float(r["na"]) for r in rows if r.get("na")),
                     default=None)
        kwargs, warnings = build_config(
            row, self._um_per_step(), af_cfg, na_min=na_min)
        for warning in warnings:
            self.sig_af_log.emit(f"config warning: {warning}")
        cfg = AutofocusConfig(
            **kwargs,
            metric=af_cfg.get("metric", "tenengrad"),
            strategy="adaptive",  # hardcoded — the app runs v3 only
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
            coarse_early_stop_samples=int(
                af_cfg.get("coarse_early_stop_samples", 2)),
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
            mode=mode)
        return cfg

    def _launch(self, kind: str, request: AutofocusRequest) -> None:
        self._job_kind = kind
        # Lock manual input NOW — the job is imminent and the mode gate
        # must not leave a jog window open.
        self._state.set_mode("AUTOFOCUS")
        # One camera tick so the first scored frames are fresh. The
        # pending request is cancellable: abort() during this window
        # kills the arm instead of starting.
        self._pending_request = request
        self._arm_timer.start()

    def _on_arm_timer(self) -> None:
        request, self._pending_request = self._pending_request, None
        if request is not None:
            self._arm(request)

    def _arm(self, request: AutofocusRequest) -> None:
        job_id = self._manager.submit("focus", "autofocus", request)
        if job_id < 0:
            # Unknown/disabled device — the job will never run and no
            # done signal will ever arrive: fail cleanly instead of
            # stranding the service in AUTOFOCUS/busy.
            logger.warning("Autofocus: arm rejected (focus disabled?)")
            self._job_kind = ""
            if self._state.mode == "AUTOFOCUS":
                self._state.set_mode("MANUAL")
            self.sig_af_finished.emit(AutofocusResult(
                best_position=self._manager.focus_position, best_score=0.0,
                message="focus stage disabled — autofocus unavailable",
                phase="arm"))

    # ------------------------------------------------------------------
    # AF job completion
    # ------------------------------------------------------------------

    def _on_af_done(self, result: AutofocusResult) -> None:
        logger.info("AF done: success=%s best=%s phase=%s msg=%r",
                    result.success, result.best_position,
                    result.phase, result.message)
        self._job_kind = ""
        # Only revert OUR mode — a concurrent SCAN owns the axes now and
        # must not be clobbered back to MANUAL mid-scan.
        if self._state.mode == "AUTOFOCUS":
            self._state.set_mode("MANUAL")
        self.sig_af_finished.emit(result)

    # ------------------------------------------------------------------
    # Input-abort (user motion/snapshot submissions abort a running job)
    # ------------------------------------------------------------------

    def _on_job_submitted(self, device: str, method: str) -> None:
        """Any user motion/snapshot submission while a job runs aborts
        it — the input then proceeds (input wins). Camera property
        tweaks and the service's own jobs do not."""
        if self._job_kind not in ("af_s", "cal"):
            return
        if (device, method) in _ABORT_EXCLUSIONS:
            return
        if device == "focus" and method in _MANUAL_FOCUS_METHODS:
            source = "manual focus"
        elif device in ("zolix", "sigmakoki") and method in _XY_MOTION_METHODS:
            source = "stage motion"
        elif device == "camera" and method == "snapshot":
            source = "snapshot"
        else:
            return
        self.sig_af_log.emit(f"job aborted: {source} input")
        self.abort(reason=f"aborted by {source} input")

    def _on_stop_requested(self) -> None:
        """STOP ALL (Esc / the STOP buttons) also cancels a pending arm
        — the proxy's own abort already covers a running job."""
        self.abort()

    # ------------------------------------------------------------------
    # Abort / teardown
    # ------------------------------------------------------------------

    def abort(self, reason: str = "aborted by user") -> None:
        """Abort the running job (AF-S / calibration) or cancel the
        pending arm."""
        if self._pending_request is not None:
            self._pending_request = None
            self._arm_timer.stop()
            self._job_kind = ""
            if self._state.mode == "AUTOFOCUS":
                self._state.set_mode("MANUAL")
            self.sig_af_log.emit("AF-S cancelled before the arm move")
            self.sig_af_finished.emit(AutofocusResult(
                best_position=self._manager.focus_position, best_score=0.0,
                aborted=True, message=reason, phase="arm"))
        self._focus.request_abort(reason)

    def shutdown(self) -> None:
        """Called before the manager teardown: abort everything, stop
        timers."""
        if self.busy:
            self.abort()
        self._arm_timer.stop()
        self._pending_request = None
        self._job_kind = ""
