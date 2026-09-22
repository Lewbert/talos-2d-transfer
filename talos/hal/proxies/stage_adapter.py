"""ManagerStageAdapter: an XYRStage-shaped view over the manager's zolix proxy.

The Sample Finding grid scan needs blocking, readback-verified moves — but
the manager owns the ONLY serial handle to the controller. A second
driver on the same COM port cannot even open on Windows, so the scan used
to fail on hardware (it only ever ran against the sim), and a private
driver would be invisible to STOP ALL and to status polling.

This adapter keeps the manager the single writer: each move is submitted
as a normal job on the zolix worker — so the priority stop path and
STOP ALL cover it — and the calling (scan) thread blocks on the job's
completion. `wait_idle` never runs a blocking driver call on the worker
either: it polls the controller's motion bits with short `get_status`
jobs, so the worker is free *between* reads and a queued stop is executed
the moment it arrives.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from talos.hal.base import (
    DeviceConnectionError,
    DeviceError,
    DeviceTimeoutError,
    StageSpeed,
)
from talos.models import StagePosition, StageStatus

logger = logging.getLogger(__name__)

# Jobs tracked for the wait-graph (all completions are recorded, not just
# the ones being awaited — a job can finish before the submitter has
# stored its id). Oldest entries are dropped past this cap.
_MAX_TRACKED_JOBS = 64
#: Motion-end poll interval. The read itself costs ~10-25 ms on the wire,
#: so this is a floor between jobs, not a sleep budget.
_WAIT_POLL_S = 0.05
#: Consecutive unanswered polls before `wait_idle` gives up: a blip is
#: retried (the driver retries the read three times already), a dead link
#: must fail the waypoint rather than spin to the timeout.
_MAX_WAIT_FAILURES = 3


class ManagerStageAdapter:
    """Blocking stage façade for the grid scan (XYRStage subset)."""

    def __init__(self, manager, stage_cfg: dict,
                 abort_check: Callable[[], bool] | None = None):
        self._manager = manager
        self._stage_cfg = dict(stage_cfg or {})
        self._abort_check = abort_check or (lambda: False)
        self._cond = threading.Condition()
        self._done: dict[int, tuple[Any, str | None]] = {}
        self._attached = True
        manager.sig_job_done.connect(self._on_job_done)
        manager.sig_job_failed.connect(self._on_job_failed)

    # ------------------------------------------------------------------
    # Job plumbing
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Detach from the manager (the scan is finished with the stage).

        Without this the manager's signals keep the adapter alive for the
        app's lifetime, and every later job would be recorded in a dict
        nobody reads.
        """
        if not self._attached:
            return
        self._attached = False
        for signal_, slot in ((self._manager.sig_job_done, self._on_job_done),
                              (self._manager.sig_job_failed, self._on_job_failed)):
            try:
                signal_.disconnect(slot)
            except (RuntimeError, TypeError):
                pass

    def _record(self, job_id: int, result: Any, error: str | None) -> None:
        with self._cond:
            self._done[job_id] = (result, error)
            while len(self._done) > _MAX_TRACKED_JOBS:
                self._done.pop(next(iter(self._done)))
            self._cond.notify_all()

    def _on_job_done(self, job_id: int, result: Any) -> None:
        self._record(job_id, result, None)

    def _on_job_failed(self, job_id: int, exc_type: str, message: str) -> None:
        self._record(job_id, None, f"{exc_type}: {message}")

    def _call(self, method: str, *args, timeout_s: float = 60.0) -> Any:
        """Submit one job and block until it completes.

        The blocking happens on the CALLING thread (the scan worker), so
        the device worker stays free to run a priority stop.

        The wait watches the abort as well as the deadline. It has to: an
        abort mid-run can arrive while a move is in flight, and a stop that
        purges the queue or a link that drops the reply can leave a job
        that never reports — which would hold an aborted scan for the whole
        timeout (120 s for a move) with the operator watching a stage that
        had already stopped. The flag is what the operator asked for; it
        wins over a completion that may not be coming.
        """
        if self._abort_check():
            raise DeviceError("scan aborted")
        job_id = self._manager.submit("zolix", method, *args)
        if job_id < 0:
            raise DeviceConnectionError("zolix is unavailable — scan refused")
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while job_id not in self._done:
                if self._abort_check():
                    raise DeviceError("scan aborted")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DeviceTimeoutError(
                        f"zolix.{method} did not complete within {timeout_s:.0f} s")
                self._cond.wait(min(remaining, _WAIT_POLL_S))
            result, error = self._done.pop(job_id)
        if error is not None:
            raise DeviceError(f"zolix.{method}: {error}")
        return result

    # ------------------------------------------------------------------
    # XYRStage subset used by GridScanner
    # ------------------------------------------------------------------

    def _pps(self, speed: StageSpeed) -> int | None:
        key = "slow_speed_pps" if speed is StageSpeed.SLOW else "fast_speed_pps"
        value = self._stage_cfg.get(key)
        try:
            return int(value) if value else None
        except (TypeError, ValueError):
            return None

    def move_abs_um(self, x_um: float, y_um: float, r_deg: float | None = None,
                    speed: StageSpeed = StageSpeed.SLOW) -> None:
        """Absolute move, executed on the device worker with the
        objective-scaled speed for this scan."""
        self._call("move_abs_um", float(x_um), float(y_um), r_deg, speed,
                   self._pps(speed), timeout_s=120.0)

    def wait_idle(self, timeout_s: float = 120.0) -> None:
        """Block until the CONTROLLER reports every axis stopped.

        The motion bits are the controller's own answer (regs 30012-30014,
        one register per axis), so this returns as soon as the ramp has
        finished — one poll, ~50 ms — instead of waiting for the position
        to look stable across three telemetry samples 0.2 s apart. That
        rule cost every waypoint at least 0.6 s of stage-idle time, and it
        read the *GUI-delivered* telemetry, which stops updating entirely
        while a job is queued (i.e. exactly while a move is in flight).

        The driver's own ``wait_idle`` would be faster still, but a
        blocking call on the device worker holds a queued STOP ALL behind
        it. A short job per poll keeps the worker free between reads.

        A result that is not a ``StageStatus`` counts as a FAILED poll,
        never as "stopped": the same trap the old telemetry key was
        written about (a value that cannot answer the question must not be
        read as a yes).
        """
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        failures = 0
        while True:
            if self._abort_check():
                return
            if time.monotonic() >= deadline:
                raise DeviceTimeoutError(
                    f"Zolix stage not settled after {timeout_s:.0f} s")
            try:
                status = self._call("get_status", timeout_s=10.0)
            except DeviceError as exc:
                if self._abort_check():
                    return
                failures += 1
                if failures >= _MAX_WAIT_FAILURES:
                    raise DeviceError(f"zolix.wait_idle: {exc}") from exc
            else:
                if isinstance(status, StageStatus):
                    failures = 0
                    if not status.any_moving:
                        return
                else:
                    failures += 1
                    if failures >= _MAX_WAIT_FAILURES:
                        raise DeviceError(
                            "zolix.wait_idle: the controller did not answer "
                            "with a status")
            time.sleep(_WAIT_POLL_S)

    def get_position(self) -> StagePosition:
        """Fresh hardware readback (never the commanded value)."""
        pos = self._call("get_position", timeout_s=30.0)
        return pos if isinstance(pos, StagePosition) else StagePosition()

    def stop(self) -> None:
        """Release-stop on the priority path (purges queued motion)."""
        try:
            self._manager.submit("zolix", "stop", priority=1)
        except Exception:  # noqa: BLE001 - a stop must never raise
            logger.exception("scan stop failed")
