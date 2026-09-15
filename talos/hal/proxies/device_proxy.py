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
        self._driver = None
        self._draining = False
        self._poll_timer: QTimer | None = None
        self._poll_fail_streak = 0
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

    @Slot()
    def cancel_pending(self) -> None:
        """Worker-thread slot: FAIL every queued job (reconnect path).

        ``request_shutdown`` clears the queue silently, which leaves any
        waiter blocked until its own timeout — the grid scan's
        ``StageAdapter._call`` waits 60-120 s on ``sig_command_done``.
        Failure (not "done") is deliberate: a done-with-None result is read
        as a VALUE by the adapter (``get_position`` would return a
        zeroed StagePosition and the scan would move on numbers that were
        never read).
        """
        while self._queue:
            _priority, job_id, _method, _args = self._queue.popleft()
            if job_id >= 0:
                self.sig_command_failed.emit(job_id, "Reconnecting",
                                             "device is reconnecting")

    @Slot()
    def request_shutdown(self) -> None:
        """Worker-thread slot: stop, disconnect, and quit the thread."""
        self._queue.clear()
        if self._driver is not None:
            try:
                self._driver.stop()
            except Exception:  # noqa: BLE001
                pass
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
        args = tuple(args)
        key_fn = _COALESCE_KEYS.get(self._key, {}).get(method_name)
        if priority == PRIORITY_NORMAL and key_fn is not None:
            # Newest value for this coalesce key wins: drop older queued
            # updates (they would otherwise apply late after a slow ACK and
            # make control laggy). Independent axes keep their own keys.
            new_key = key_fn(args)
            kept = []
            for item in self._queue:
                if (item[0] == PRIORITY_NORMAL and item[2] == method_name
                        and key_fn(item[3]) == new_key):
                    self.sig_command_done.emit(item[1], None)  # superseded
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
                    self.sig_command_done.emit(item[1], None)  # superseded
                else:
                    kept.append(item)
            self._queue = deque(kept)
        self._queue.append((priority, job_id, method_name, args))
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
        _MOTION = ("set_speed", "move", "move_continuous", "move_rel",
                   "move_abs", "step")
        kept = []
        for item in self._queue:
            if item[0] == PRIORITY_NORMAL and item[2] in _MOTION:
                self.sig_command_done.emit(item[1], None)  # superseded
            else:
                kept.append(item)
        self._queue = deque(kept)
        self._queue.appendleft((PRIORITY_STOP, -1, "stop", ()))
        self._schedule_drain()

    # ------------------------------------------------------------------
    # Worker thread internals
    # ------------------------------------------------------------------

    @Slot()
    def _run(self) -> None:
        try:
            self._driver = self._factory()
            self._driver.connect()
        except DeviceError as exc:
            logger.error("Proxy %s: connect failed: %s", self._key, exc)
            self.sig_connected.emit(False)
            self.sig_event.emit("connect_failed", {"error": str(exc)})
            self._thread.quit()
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Proxy %s: unexpected connect failure", self._key)
            self.sig_connected.emit(False)
            self.sig_event.emit("connect_failed", {"error": str(exc)})
            self._thread.quit()
            return
        self.sig_connected.emit(True)
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(self._poll_interval_ms)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start()

    @Slot()
    def _drain(self) -> None:
        if self._draining or self._driver is None:
            return
        self._draining = True
        try:
            while self._queue:
                # Stable sort by -priority: STOP first, NORMAL FIFO.
                self._queue = deque(sorted(self._queue, key=lambda item: -item[0]))
                priority, job_id, method_name, args = self._queue.popleft()
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

    @Slot()
    def _poll(self) -> None:
        if self._driver is None or self._draining:
            return
        # Commands get exclusive serial access: status polling interleaved
        # with motion commands garbles the firmware's replies (ERR:UNKNOWN
        # garbage) and slows ACKs — skip polls while commands are pending.
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
