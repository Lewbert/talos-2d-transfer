"""DeviceProxy: owns one HAL driver on its own QThread.

Contract:
- The driver is created, connected, and exclusively used inside the worker
  thread. Commands arrive via a priority deque (STOP beats NORMAL), and
  telemetry is polled with a QTimer living in the worker thread.
- The GUI talks to the proxy only through queued signals/slots.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections import deque
from typing import Any, Callable

from PySide6.QtCore import QMetaObject, QObject, QThread, QTimer, Qt, Signal, Slot

from talos.hal.base import DeviceError

logger = logging.getLogger(__name__)

PRIORITY_NORMAL = 0
PRIORITY_STOP = 1

# Commands where only the NEWEST queued value matters — a burst of stale
# speed updates behind a slow ACK makes the stage feel laggy/jerky.
# The coalesce KEY is per device+method so that independent axes are NOT
# dropped: zolix diagonal movement is two move_continuous commands (X and
# Y); sigmakoki speeds are per-axis too.
_COALESCE_KEYS: dict[str, dict[str, Callable[[tuple], tuple]]] = {
    "focus": {"set_speed": lambda args: ()},                  # newest wins
    # sigmakoki: newest speed AND newest continuous move win, per axis —
    # an analog-stick hold re-emits at input rate, and a queue of stale
    # MV commands behind a slow command made the stage act on positions
    # the operator had already left.
    "sigmakoki": {"set_speed": lambda args: (args[0],),
                  "move": lambda args: (args[0],)},
    "zolix": {"move_continuous": lambda args: (args[0],)},    # per axis
}

# Continuous motion commands dropped when a release-stop is enqueued (see
# enqueue): the stop executes FIRST (PRIORITY_STOP), and a stale
# continuous command behind it would re-start the axis with no input held
# — hardware-verified hazard after the AF input-abort change ("tapped the
# jog during an AF run and the focus kept moving after the release").
# Discrete moves (single steps, go-to) are intentional commands and
# survive.
_CONTINUOUS_MOTION = ("set_speed", "move", "move_continuous")

# Every motion method — a STOP purges these (see enqueue_stop), because a
# stale one behind the stop would move the axis again with nothing held.
_MOTION = ("set_speed", "move", "move_continuous", "move_rel", "move_abs",
           "step")

#: Sample the limit switches (a second serial round trip) every Nth poll —
#: ~0.5 s at the default 100 ms cadence.
LIMITS_POLL_EVERY = 5


class DeviceProxy(QObject):
    sig_connected = Signal(bool)
    sig_telem = Signal(dict)
    sig_event = Signal(str, dict)
    sig_command_done = Signal(int, object)
    sig_command_failed = Signal(int, str, str)
    sig_all_stopped = Signal()

    def __init__(self, key: str, driver_factory: Callable[[], Any],
                 poll_interval_ms: int = 100, parent: QObject | None = None):
        # NOTE: parent must stay None — moveToThread() refuses objects with
        # a parent, and a proxy that never moved would run serial I/O in the
        # GUI thread.
        super().__init__(None)
        self._key = key
        self._factory = driver_factory
        self._poll_interval_ms = poll_interval_ms
        self._queue: deque[tuple[int, int, str, tuple]] = deque()
        # The queue has THREE producers (enqueue/enqueue_stop from the GUI and
        # the scan worker, _drain's consumer on the worker): rebuilding it
        # without a lock lost items appended between the sort and the rebind
        # and could raise "deque mutated during iteration". Held only around
        # the queue itself — NEVER across a driver call.
        self._queue_lock = threading.Lock()
        self._driver = None
        # None = usable; a string = the proxy will never run another command
        # (connect failed, or it was torn down). Read from the GUI thread so a
        # submit fails immediately instead of queueing into a dead worker.
        self._unavailable: str | None = None
        # Set by begin_retire() (GUI thread) so a drain that is ALREADY
        # queued — or running — stands down instead of executing jobs the
        # retire is about to fail (see retire).
        self._retiring = False
        self._draining = False
        self._poll_timer: QTimer | None = None
        self._poll_fail_streak = 0
        # Limit switches (sigmakoki only) are sampled every Nth poll — see
        # _poll. None until the first successful read.
        self._limits_cache: dict | None = None
        self._poll_count = 0
        # Subclass-provided job handlers (e.g. FocusProxy's blocking
        # autofocus run) dispatched by method name instead of the driver.
        self._special_methods: dict[str, Callable] = {}

        self._thread = QThread(self)
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run)

    # ------------------------------------------------------------------
    # Lifecycle (public, GUI thread)
    # ------------------------------------------------------------------

    @Slot()
    def start(self) -> None:
        self._thread.start()

    def mark_unavailable(self, reason: str) -> None:
        """GUI thread: this proxy will run no further command. Used by the
        reconnect timeout, where the queued retire cannot run yet (the worker
        is inside a blocking call) but no submit should be accepted either."""
        self._unavailable = reason

    @Slot()
    def cancel_pending(self) -> None:
        """FAIL every queued job (reconnect path).

        ``request_shutdown`` clears the queue silently, which leaves any
        waiter blocked until its own timeout — the grid scan's
        ``StageAdapter._call`` waits 60-120 s on ``sig_command_done``.
        Failure (not "done") is deliberate: a done-with-None result is read
        as a VALUE by the adapter (``get_position`` would return a
        zeroed StagePosition and the scan would move on numbers that were
        never read).
        """
        self._fail_queued("device is reconnecting")

    def begin_retire(self) -> None:
        """GUI thread: stand the drain down and post the retire.

        Two things had to change for the queued job to be FAILED rather than
        executed. (1) The steps were posted separately, and the drain
        scheduled by ``enqueue_stop`` ran BEFORE ``cancel_pending``, so the
        drain consumed the queue — including the blocking job cancel_pending
        exists to fail — and the waiter it was meant to protect still blocked
        for its full 60-120 s timeout. (2) A drain posted by an earlier
        ``enqueue`` is still ahead of the retire in the worker's queue, so it
        must be told to stand down, which is what ``_retiring`` does.
        """
        self._retiring = True
        QMetaObject.invokeMethod(self, "retire",
                                 Qt.ConnectionType.QueuedConnection)

    @Slot()
    def retire(self) -> None:
        """Worker-thread slot: the whole reconnect teardown, in ONE step —
        stop the axes, fail what is queued, then let the driver go."""
        self._retiring = True
        self._purge_continuous()
        self._fail_queued("device is reconnecting")
        self._stop_now()
        self._teardown("device proxy retired")

    @Slot()
    def request_shutdown(self) -> None:
        """Worker-thread slot: stop, disconnect, and quit the thread."""
        self._stop_now()
        self._teardown("device proxy shut down")

    # -- worker-thread helpers -------------------------------------------

    def _fail_queued(self, reason: str) -> None:
        """FAIL every queued job. Worker thread for the pop; the emits are
        signal-to-signal relays (they cross to the GUI thread themselves)."""
        with self._queue_lock:
            pending = list(self._queue)
            self._queue.clear()
        for _priority, job_id, _method, _args in pending:
            if job_id >= 0:
                self.sig_command_failed.emit(job_id, "Reconnecting", reason)

    def _purge_continuous(self) -> None:
        """Drop every queued MOTION job, COMPLETING it (superseded, not
        failed — a waiter on a discrete move must not read a phantom
        failure). Callable from the GUI thread: the lock protects the queue,
        and the emissions happen outside it."""
        with self._queue_lock:
            kept, superseded = [], []
            for item in self._queue:
                if item[0] == PRIORITY_NORMAL and item[2] in _MOTION:
                    superseded.append(item[1])
                else:
                    kept.append(item)
            self._queue = deque(kept)
        for job_id in superseded:
            self.sig_command_done.emit(job_id, None)  # superseded

    def _stop_now(self) -> None:
        """Stop the driver directly (we are already on the worker thread) —
        a queued stop job would sit behind whatever the worker is doing."""
        if self._driver is None:
            return
        handler = self._special_methods.get("stop")
        try:
            (handler if handler is not None else self._driver.stop)()
        except Exception:  # noqa: BLE001 - a stop must never raise
            logger.warning("Proxy %s: stop failed during teardown",
                           self._key, exc_info=True)

    def _teardown(self, reason: str) -> None:
        """Disconnect, mark unusable, and quit the worker's event loop."""
        self._unavailable = reason
        with self._queue_lock:
            self._queue.clear()
        if self._driver is not None:
            try:
                self._driver.disconnect()
            except Exception:  # noqa: BLE001
                pass
            self._driver = None
        if self._poll_timer is not None:
            self._poll_timer.stop()
        self._thread.quit()

    def wait(self, timeout_ms: int = 3000) -> bool:
        if self._thread.isRunning():
            return self._thread.wait(timeout_ms)
        return True

    @property
    def key(self) -> str:
        return self._key

    # ------------------------------------------------------------------
    # Command intake (GUI thread → queued to worker)
    # ------------------------------------------------------------------

    @Slot(int, str, tuple, int)
    def enqueue(self, job_id: int, method_name: str, args: tuple, priority: int) -> None:
        if self._unavailable is not None:
            # A proxy whose connect FAILED (or that was torn down) has no
            # event loop to run this job and never will: failing it here is
            # the difference between a waiter seeing an error and a waiter
            # hanging until its own 60-120 s timeout. Before the fix the
            # worker had already quit its thread, so the job vanished.
            if job_id >= 0:
                self.sig_command_failed.emit(job_id, "NotConnected",
                                             self._unavailable)
            return
        args = tuple(args)
        superseded: list[int] = []
        key_fn = _COALESCE_KEYS.get(self._key, {}).get(method_name)
        with self._queue_lock:
            if priority == PRIORITY_NORMAL and key_fn is not None:
                # Newest value for this coalesce key wins: drop older queued
                # updates (they would otherwise apply late after a slow ACK
                # and make control laggy). Independent axes keep their own
                # keys.
                new_key = key_fn(args)
                kept = []
                for item in self._queue:
                    if (item[0] == PRIORITY_NORMAL and item[2] == method_name
                            and key_fn(item[3]) == new_key):
                        superseded.append(item[1])
                    else:
                        kept.append(item)
                self._queue = deque(kept)
            elif priority == PRIORITY_STOP and method_name in ("stop", "stop_axis"):
                # A release-stop must never be followed by a stale CONTINUOUS
                # command: the drain sorts STOP first, so a queued
                # set_speed/move/move_continuous behind it would re-start the
                # axis after the release (no input held). Discrete commands
                # (single steps, go-to) are intentional and survive.
                kept = []
                for item in self._queue:
                    if item[0] == PRIORITY_NORMAL \
                            and item[2] in _CONTINUOUS_MOTION:
                        superseded.append(item[1])
                    else:
                        kept.append(item)
                self._queue = deque(kept)
            self._queue.append((priority, job_id, method_name, args))
        # Superseded jobs are COMPLETED (not failed) — they were replaced, and
        # a waiter must not see a phantom failure. Emitted outside the lock:
        # these signals cross to the GUI thread.
        for old_id in superseded:
            self.sig_command_done.emit(old_id, None)
        self._schedule_drain()

    def _schedule_drain(self) -> None:
        """Kick the worker's drain. MUST be a QUEUED invocation: a bare
        QTimer.singleShot(0, self._drain) fires in the CALLING thread,
        which runs blocking serial I/O on the GUI thread — hardware-
        verified as a 9 GB frame-queue GUI freeze when the firmware wedged
        mid-jog."""
        QMetaObject.invokeMethod(self, "_drain",
                                 Qt.ConnectionType.QueuedConnection)

    @Slot()
    def enqueue_stop(self) -> None:
        # A stop must not be followed by stale motion commands: they would
        # re-start the axis right after the stop (hardware-verified:
        # "released the trigger but the focus kept jogging"). Purge them,
        # completing the superseded jobs for accounting consistency.
        self._purge_continuous()
        with self._queue_lock:
            self._queue.appendleft((PRIORITY_STOP, -1, "stop", ()))
        self._schedule_drain()

    # ------------------------------------------------------------------
    # Worker thread internals
    # ------------------------------------------------------------------

    @Slot()
    def _run(self) -> None:
        # NOTE: _run is connected to QThread.started, i.e. it runs BEFORE
        # run() reaches exec(). Calling quit() here (as the failure path used
        # to) makes exec() return at once: the worker never gets an event
        # loop, so it can neither poll nor serve a queued request_shutdown —
        # and every later submit vanished silently. A failed connect now
        # leaves the loop RUNNING and the proxy marked unusable, so the app
        # can still tear it down cleanly and the operator gets an error.
        try:
            self._driver = self._factory()
            self._driver.connect()
        except DeviceError as exc:
            logger.error("Proxy %s: connect failed: %s", self._key, exc)
            self._connect_failed(f"connect failed: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Proxy %s: unexpected connect failure", self._key)
            self._connect_failed(f"connect failed: {exc!r}")
            return
        self.sig_connected.emit(True)
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(self._poll_interval_ms)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start()

    def _connect_failed(self, reason: str) -> None:
        """Worker thread: record the failure, fail what is queued, and leave
        the event loop alive (see _run)."""
        if self._driver is not None:
            try:
                self._driver.disconnect()   # a half-open port must not linger
            except Exception:  # noqa: BLE001
                pass
            self._driver = None
        self._unavailable = reason
        self._fail_queued(reason)
        self.sig_connected.emit(False)
        self.sig_event.emit("connect_failed", {"error": reason})

    @Slot()
    def _drain(self) -> None:
        if self._draining or self._driver is None or self._retiring:
            return
        self._draining = True
        try:
            while True:
                if self._retiring:
                    break   # a retire is queued: it owns the rest of the queue
                job = self._pop_next_job()
                if job is None:
                    break
                priority, job_id, method_name, args = job
                try:
                    handler = self._special_methods.get(method_name)
                    result = (handler(*args) if handler is not None
                              else getattr(self._driver, method_name)(*args))
                    if job_id >= 0:
                        self.sig_command_done.emit(job_id, result)
                    if method_name == "stop":
                        self.sig_all_stopped.emit()
                except DeviceError as exc:
                    logger.warning("Proxy %s: %s failed: %s", self._key, method_name, exc)
                    if job_id >= 0:
                        self.sig_command_failed.emit(job_id, type(exc).__name__, str(exc))
                    else:
                        self.sig_event.emit("command_error", {"error": str(exc)})
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Proxy %s: %s crashed", self._key, method_name)
                    self.sig_command_failed.emit(job_id, type(exc).__name__, str(exc))
        finally:
            self._draining = False

    def _pop_next_job(self) -> tuple | None:
        """Pop the next job (STOP first, then FIFO) under the queue lock."""
        with self._queue_lock:
            if not self._queue:
                return None
            items = sorted(self._queue, key=lambda item: -item[0])
            job = items[0]
            self._queue = deque(items[1:])
            return job

    def pending_jobs(self) -> tuple[tuple, ...]:
        """Snapshot of the queue (for the FocusProxy's busy check)."""
        with self._queue_lock:
            return tuple(self._queue)

    @Slot()
    def _poll(self) -> None:
        if self._driver is None or self._draining:
            return
        # Commands get exclusive serial access: status polling interleaved
        # with motion commands garbles the firmware's replies (ERR:UNKNOWN
        # garbage) and slows ACKs — skip polls while commands are pending.
        with self._queue_lock:
            if self._queue:
                return
        failed = False
        for event in self._driver.drain_events():
            self.sig_event.emit(event, {})
        payload = {"device": self._key}

        def _optional(name: str) -> Any:
            """Call an optional driver capability if it exists. A MISSING
            method is not a failure — focus has no get_position, only the
            yudian has read_pv/... (each AttributeError used to mark the
            whole poll failed and drive backoff to the 2 s cap). A serial
            error from a method that DOES exist is a failure."""
            fn = getattr(self._driver, name, None)
            if fn is None:
                return None
            nonlocal failed
            try:
                return fn()
            except DeviceError:
                failed = True
            except Exception:  # noqa: BLE001
                failed = True
            return None

        def _fold(key: str, value: Any) -> None:
            if value is None:
                return
            payload[key] = (dataclasses.asdict(value)
                            if dataclasses.is_dataclass(value) else value)

        if hasattr(self._driver, "get_telemetry"):
            # Drivers whose status query already carries the position
            # expose a combined read (sigmakoki: STATUS? feeds both), so
            # one poll costs one round trip instead of two on the same
            # serial line.
            telemetry = _optional("get_telemetry") or {}
            _fold("status", telemetry.get("status"))
            _fold("position", telemetry.get("position"))
        else:
            _fold("status", _optional("get_status"))
            _fold("position", _optional("get_position"))
        payload["pv"] = _optional("read_pv")
        payload["sv"] = _optional("read_sv")
        payload["output_percent"] = _optional("read_output_percent")
        # Limit switches (the XYZ stage's LIMITS?): a SECOND round trip on
        # a serial line that drops bytes under motion load, and a switch
        # does not change between two 100 ms polls — sample every Nth poll
        # and keep the last reading. Drivers whose limits ride along with
        # the status registers (zolix) never expose get_limits.
        self._poll_count += 1
        if self._limits_cache is None or self._poll_count % LIMITS_POLL_EVERY == 0:
            read = _optional("get_limits")
            if read is not None:
                self._limits_cache = dict(read)
        if self._limits_cache:
            payload["limits"] = dict(self._limits_cache)
        limits = _optional("get_soft_limits")
        if limits is not None:
            payload["slim_bounds"] = limits
        # Whether the firmware actually ENFORCES those bounds: the focus
        # firmware gates every limit check on SLIM and ships with it off.
        # Copied out of the status payload we already fetched — asking the
        # driver again would double the poll's serial traffic.
        status_payload = payload.get("status")
        if isinstance(status_payload, dict) and "slim_on" in status_payload:
            payload["slim_on"] = bool(status_payload["slim_on"])
        self._apply_poll_backoff(failed)
        self.sig_telem.emit(payload)

    def _apply_poll_backoff(self, failed: bool) -> None:
        """Slow polling down on repeated failures (a struggling serial line
        must not be hammered — each retry adds traffic and worsens it)."""
        if failed:
            self._poll_fail_streak = getattr(self, "_poll_fail_streak", 0) + 1
            if self._poll_fail_streak >= 3 and self._poll_timer is not None:
                interval = min(self._poll_timer.interval() * 2, 2000)
                self._poll_timer.setInterval(interval)
                self._poll_fail_streak = 0
                logger.warning("Proxy %s: serial struggling — poll interval %d ms",
                               self._key, interval)
        else:
            self._poll_fail_streak = 0
            if self._poll_timer is not None and \
                    self._poll_timer.interval() != self._poll_interval_ms:
                self._poll_timer.setInterval(self._poll_interval_ms)
