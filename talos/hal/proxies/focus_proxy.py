"""FocusProxy: DeviceProxy + exclusive autofocus jobs.

The autofocus controller is BLOCKING and needs the raw focus driver — it
runs inside the focus worker's _drain as a special job, with exclusive
access to the serial line (commands queue behind it; the poll timer is
skipped by the _draining guard). Frames arrive via the shared
LatestFrameSlot, written by the camera worker — the controller polls it
directly because queued Qt signals cannot reach a blocked worker.

Abort: request_abort() is a plain method (any thread); STOP ALL reaches
it through the enqueue_stop() override (called as a plain method from the
GUI thread by InstrumentManager.stop_all).
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal, Slot

from talos.cv.af_v3 import AdaptiveAutofocusController
from talos.cv.autofocus import AutofocusRequest, AutofocusResult
from talos.cv.backlash_cal import BacklashCalibrator, BacklashResult
from talos.hal.proxies.device_proxy import DeviceProxy

logger = logging.getLogger(__name__)

# The app runs ONE autofocus strategy: the derivative-hybrid adaptive v3.
# The older algorithms (Adaptive V1/V2 in talos.cv.af_adaptive, the classic
# controller in talos.cv.autofocus, AF-C in talos.cv.af_c) stay in the
# codebase as stored knowledge — reachable only from the sim suites and the
# bench tools, never from the app path.


class FocusProxy(DeviceProxy):
    sig_af_progress = Signal(float, int, float, float)  # fraction, phase, score, pos
    sig_af_curve_secondary = Signal(float, float)  # low-freq series (pos, score)
    sig_af_done = Signal(object)                 # AutofocusResult
    sig_af_log = Signal(str)
    sig_stop_requested = Signal()                # STOP ALL (→ the service cancels a pending arm)
    sig_cal_progress = Signal(float, str)        # backlash calibration
    sig_cal_done = Signal(object)                # BacklashResult

    def __init__(self, key: str, driver_factory, poll_interval_ms: int = 100,
                 parent: QObject | None = None):
        super().__init__(key, driver_factory, poll_interval_ms, parent)
        self._special_methods = {"autofocus": self._autofocus_job,
                                 "backlash_calibrate": self._backlash_cal_job}
        self._af_ctrl: AdaptiveAutofocusController | None = None
        self._cal_ctrl: BacklashCalibrator | None = None
        self._frame_slot = None
        # The abort latch: request_abort() can only reach a controller
        # once the worker constructs it — aborts landing between the
        # job's enqueue and the construction go here and are polled by
        # the controller's _check (run() resets the controller-local
        # flag at entry, so a pre-run flag set would be lost).
        self._abort_pending_reason: str | None = None

    @Slot(int, str, tuple, int)
    def enqueue(self, job_id: int, method_name: str, args: tuple,
                priority: int) -> None:
        if method_name in ("autofocus", "backlash_calibrate"):
            # A fresh job supersedes any stale latch (e.g. a STOP ALL
            # that arrived while NO job was queued).
            self._abort_pending_reason = None
        super().enqueue(job_id, method_name, args, priority)

    def _abort_check(self) -> str | None:
        """The latch poll for the running controller's _check (worker
        thread)."""
        reason, self._abort_pending_reason = self._abort_pending_reason, None
        return reason

    # ------------------------------------------------------------------
    # GUI thread
    # ------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        """A special job (autofocus / backlash calibration) is queued or
        running — a reconnect would replace the proxy and its completion
        signals, stranding the service in AUTOFOCUS/busy forever."""
        # NOTE: the abort latch is deliberately NOT consulted — STOP ALL
        # arms it with no job in flight, and a stale latch would refuse
        # every later reconnect.
        if self._af_ctrl is not None or self._cal_ctrl is not None:
            return True
        return any(item[2] in self._special_methods
                   for item in self.pending_jobs())

    def set_frame_slot(self, slot) -> None:
        """Attach the shared LatestFrameSlot (camera worker writes; the AF
        job reads). Plain attribute — called from the GUI thread before
        the first run."""
        self._frame_slot = slot

    def request_abort(self, reason: str | None = None) -> None:
        """Abort the running autofocus/calibration job. Plain method —
        callable from any thread (the documented abort path). Aborts
        landing before the worker constructs the controller latch for
        the job start (the _abort_check poll)."""
        if self._af_ctrl is not None:
            self._af_ctrl.request_abort(reason)
        else:
            self._abort_pending_reason = reason or "aborted by user"
        if self._cal_ctrl is not None:
            self._cal_ctrl.request_abort(reason)

    @Slot()
    def enqueue_stop(self) -> None:
        """STOP ALL must also abort a running autofocus job — a queued
        stop would sit behind the blocking run until it finished."""
        super().enqueue_stop()
        self.request_abort()
        self.sig_stop_requested.emit()

    # ------------------------------------------------------------------
    # Special jobs (run in the WORKER thread via _drain)
    # ------------------------------------------------------------------

    def _autofocus_job(self, request: AutofocusRequest) -> AutofocusResult:
        cfg = request.to_config()
        # The app runs the v3 adaptive controller only (see the module
        # docstring note on the stored-knowledge strategies).
        ctrl = AdaptiveAutofocusController(self._driver, self._frame_slot,
                                           abort_check=self._abort_check)
        ctrl.sig_progress.connect(self.sig_af_progress)
        ctrl.sig_curve_secondary.connect(self.sig_af_curve_secondary)
        ctrl.sig_log.connect(self.sig_af_log)
        self._af_ctrl = ctrl
        try:
            result = ctrl.run(center=request.center_steps, cfg=cfg,
                              bounds=request.bounds)
            self.sig_af_done.emit(result)
            return result
        except Exception as exc:  # noqa: BLE001
            # run() catches everything the algorithm raises — an escape
            # here (e.g. a BaseException or a raise inside a handler)
            # must not leave the service stuck in AUTOFOCUS with no done
            # signal (audit-found: _drain's sig_command_failed has no
            # service consumer).
            logger.exception("Autofocus job crashed")
            result = AutofocusResult(
                best_position=request.center_steps, best_score=0.0,
                message=f"autofocus crashed: {exc!r}", phase="motion")
            self.sig_af_done.emit(result)
            return result
        finally:
            self._af_ctrl = None

    def _backlash_cal_job(self, request: tuple) -> BacklashResult:
        center_steps, cfg = request
        cal = BacklashCalibrator(self._driver, self._frame_slot,
                                 abort_check=self._abort_check)
        cal.sig_progress.connect(self.sig_cal_progress)
        cal.sig_log.connect(self.sig_af_log)
        self._cal_ctrl = cal
        try:
            result = cal.run(center=center_steps, cfg=cfg)
            self.sig_cal_done.emit(result)
            return result
        except Exception as exc:  # noqa: BLE001
            logger.exception("Backlash calibration crashed")
            result = BacklashResult(success=False,
                                    message=f"calibration crashed: {exc!r}")
            self.sig_cal_done.emit(result)
            return result
        finally:
            self._cal_ctrl = None
