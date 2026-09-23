"""CameraProxy: owns one Camera on its own QThread.

The acquire loop and property commands share a single QTimer tick in the
worker thread (single-thread ownership — camera backends are not
thread-safe). Frames are passed to the GUI by reference; the Camera
contract guarantees each fetch returns a fresh C-contiguous RGB array.

Threading contract: teardown MUST go through the worker event loop —
InstrumentManager.shutdown() invokes request_shutdown via a QUEUED
invocation so in-flight ticks / blocking snapshots finish first.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot

from talos.hal.base import DeviceError
from talos.hal.camera_props import map_property, normalize_props

logger = logging.getLogger(__name__)


class CameraProxy(QObject):
    sig_connected = Signal(bool)
    sig_frame = Signal(object)          # np.ndarray RGB uint8 (fresh array)
    sig_properties = Signal(dict)
    sig_event = Signal(str, dict)
    sig_fps = Signal(float)
    # Job completion, same contract as DeviceProxy (the manager relays
    # these to sig_job_done/sig_job_failed — the UI's snapshot
    # busy-gating depends on them).
    sig_command_done = Signal(int, object)
    sig_command_failed = Signal(int, str, str)   # (job_id, exc_type, msg)

    def __init__(self, camera_factory: Callable[[], Any],
                 frame_interval_ms: int = 16, parent: QObject | None = None):
        # NOTE: parent must stay None — see DeviceProxy (moveToThread).
        super().__init__(None)
        # The factory returns one Camera or a list of candidates; the proxy
        # connects the first that works.
        self._factory = camera_factory
        self._frame_interval_ms = frame_interval_ms
        self._camera = None
        self._backend_name = ""
        # None = usable; a string = the connect failed for good. Read from the
        # GUI thread so a submit fails immediately: the snapshot busy-gate
        # (auto-gain) is released by sig_command_failed, and before this the
        # command was queued into a worker whose thread had quit.
        self._unavailable: str | None = None
        self._streaming = False          # GUI-requested streaming state
        self._backend_streaming = False  # camera.start() actually active
        self._shutdown_requested = False
        self._commands: deque[tuple[int, str, tuple]] = deque()
        self._tick_timer: QTimer | None = None
        self._fps_times: deque[float] = deque(maxlen=60)
        self._last_fps_emit = 0.0
        #: The canonical properties we believe the camera has: the last
        #: get_properties, plus every set_property since (see _note_property).
        self._props: dict = {}
        self._frame_slot = None          # LatestFrameSlot | None
        self._frame_seq = 0              # written with every published frame

        self._thread = QThread(self)
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run)

    # ------------------------------------------------------------------
    # Lifecycle (GUI thread)
    # ------------------------------------------------------------------

    @Slot()
    def start(self) -> None:
        self._thread.start()

    @Slot()
    def request_shutdown(self) -> None:
        """Runs on the WORKER thread (queued invocation)."""
        self._shutdown_requested = True
        self._commands.clear()
        self._streaming = False
        self._backend_streaming = False
        if self._tick_timer is not None:
            self._tick_timer.stop()
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:  # noqa: BLE001
                pass
            try:
                self._camera.disconnect()
            except Exception:  # noqa: BLE001
                pass
            self._camera = None
        self._thread.quit()

    def wait(self, timeout_ms: int = 3000) -> bool:
        if self._thread.isRunning():
            return self._thread.wait(timeout_ms)
        return True

    # ------------------------------------------------------------------
    # Commands (GUI thread → queued to worker)
    # ------------------------------------------------------------------

    @Slot(bool)
    def set_streaming(self, on: bool) -> None:
        self._streaming = on

    def set_frame_slot(self, slot) -> None:
        """Attach a LatestFrameSlot (GUI thread, before streaming). The
        worker publishes every fetched frame there, with its capture-side
        timestamp and a monotonic sequence number. Plain attribute —
        called from the GUI thread before the first AF run."""
        self._frame_slot = slot

    @Slot(int, str, tuple)
    def enqueue(self, job_id: int, method_name: str, args: tuple) -> None:
        if self._unavailable is not None:
            # No worker loop will run this: fail it so a waiter (the snapshot
            # job's busy-gate) hears back. A snapshot queued into a dead
            # worker used to disable auto-gain for the rest of the session.
            if job_id >= 0:
                self.sig_command_failed.emit(job_id, "NotConnected",
                                             self._unavailable)
            return
        self._commands.append((job_id, method_name, tuple(args)))

    # ------------------------------------------------------------------
    # Worker thread internals
    # ------------------------------------------------------------------

    @Slot()
    def _run(self) -> None:
        # NOTE: _run runs on QThread.started, BEFORE exec(). Quitting here (as
        # the failure paths used to) leaves the worker with no event loop at
        # all: request_shutdown can never run and every later command is
        # swallowed. The loop stays alive on failure and the proxy is marked
        # unusable instead.
        try:
            candidates = self._factory()
        except Exception as exc:  # noqa: BLE001 — e.g. unknown backend in settings
            logger.error("CameraProxy: factory failed: %s", exc)
            self._connect_failed(str(exc))
            return
        if not isinstance(candidates, (list, tuple)):
            candidates = [candidates]
        last_error: Exception | None = None
        for camera in candidates:
            try:
                camera.connect()
                camera.start()
                self._camera = camera
                # "camera@smartcam" / "camera@directshow:0" -> registry key
                self._backend_name = camera.device_id.split("@", 1)[-1].split(":", 1)[0]
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("CameraProxy: backend %s failed: %s",
                               camera.device_id, exc)
                # Never abandon a half-connected camera — disconnect it so
                # the next candidate (and later reconnects) get clean access.
                try:
                    camera.stop()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    camera.disconnect()
                except Exception:  # noqa: BLE001
                    pass
        if self._camera is None:
            self._connect_failed(str(last_error))
            return
        if self._shutdown_requested:
            # Shutdown raced the connect window — tear down immediately
            # instead of streaming into a dead session.
            try:
                self._camera.stop()
            except Exception:  # noqa: BLE001
                pass
            try:
                self._camera.disconnect()
            except Exception:  # noqa: BLE001
                pass
            self._camera = None
            self.sig_connected.emit(False)
            self._thread.quit()
            return
        self.sig_connected.emit(True)
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(self._frame_interval_ms)
        # PreciseTimer: the 50 ms coarse quantization caused a beat against
        # the camera's ~49.5 ms frame period (15-19 fps instead of 20.2).
        self._tick_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._tick_timer.timeout.connect(self._tick)
        self._tick_timer.start()

    def _connect_failed(self, reason: str) -> None:
        """Worker thread: record the failure and fail what is queued — the
        event loop stays alive (see _run)."""
        self._unavailable = f"connect failed: {reason}"
        while self._commands:
            job_id, _method, _args = self._commands.popleft()
            if job_id >= 0:
                self.sig_command_failed.emit(job_id, "NotConnected", reason)
        self.sig_connected.emit(False)
        self.sig_event.emit("connect_failed", {"error": reason})

    def _emit_fps(self) -> None:
        """Rolling 1 s fps, emitted at most twice a second."""
        now = time.monotonic()
        if now - self._last_fps_emit < 0.5:
            return
        while self._fps_times and now - self._fps_times[0] > 1.0:
            self._fps_times.popleft()
        if len(self._fps_times) >= 2:
            span = self._fps_times[-1] - self._fps_times[0]
            if span > 0:
                self._last_fps_emit = now
                self.sig_fps.emit(round((len(self._fps_times) - 1) / span, 1))

    def _note_property(self, key: str, value) -> None:
        """Record a write we have just made, and announce it.

        ``get_properties`` was the only thing that ever updated the manager's
        view of the camera, so that view went stale the moment anything wrote
        a value (a slider, the auto-gain loop, "Balance once", a scan's
        resolution switch). Everything that diffs against it — the
        per-workspace camera profiles — then computed an empty difference and
        did not write what it meant to.
        """
        self._props[key] = value
        self.sig_properties.emit(dict(self._props))

    @Slot()
    def _tick(self) -> None:
        cam = self._camera
        if cam is None:
            return
        # Drain property/snapshot commands first.
        while self._commands:
            job_id, method_name, args = self._commands.popleft()
            try:
                result = None
                if method_name == "set_property" and len(args) >= 2:
                    native_name, native_value = map_property(
                        self._backend_name, args[0], args[1])
                    cam.set_property(native_name, native_value)
                    # Announce the change so the manager's view of the
                    # camera's properties stays live. It was a connect-time
                    # snapshot before, and anything that diffs against it
                    # (the per-workspace profiles) then compared a stale
                    # value with the one it wanted to write: a needed write
                    # came out empty and the profile was silently not
                    # applied — after ANY live edit, including the next
                    # workspace switch following one.
                    self._note_property(str(args[0]), args[1])
                elif method_name == "get_properties":
                    result = cam.get_properties()
                    self._props = normalize_props(self._backend_name,
                                                  result or {})
                    if job_id >= 0:
                        self.sig_properties.emit(dict(self._props))
                else:
                    result = getattr(cam, method_name)(*args)
                self.sig_command_done.emit(job_id, result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("CameraProxy command %s failed: %s", method_name, exc)
                self.sig_event.emit("command_error", {"error": str(exc)})
                self.sig_command_failed.emit(job_id, type(exc).__name__,
                                             str(exc))
        # Streaming state machine: start/stop the backend on the worker
        # thread only (set_streaming from the GUI only flips the flag).
        if not self._streaming:
            if self._backend_streaming:
                try:
                    cam.stop()
                except Exception:  # noqa: BLE001
                    pass
                self._backend_streaming = False
            return
        if not self._backend_streaming:
            try:
                cam.start()
                self._backend_streaming = True
            except Exception as exc:  # noqa: BLE001
                self.sig_event.emit("command_error", {"error": str(exc)})
                return
        try:
            # Short timeout: a stalled USB must not wedge the worker (and
            # every queued command) for a full second.
            frame = cam.fetch(timeout_ms=200.0)
            if frame is not None:
                if self._frame_slot is not None:
                    # getattr fallback: backends without capture_time (older
                    # stubs/adapters) still publish frames, with monotonic
                    # now as the best available timestamp.
                    capture_time = getattr(cam, "capture_time", None)
                    t_cap = capture_time() if capture_time is not None \
                        else time.monotonic()
                    self._frame_seq += 1
                    self._frame_slot.write(frame, t_cap, self._frame_seq)
                self.sig_frame.emit(frame)
                self._fps_times.append(time.monotonic())
                self._emit_fps()
        except Exception as exc:  # noqa: BLE001
            logger.warning("CameraProxy fetch: %s", exc)
            self.sig_event.emit("frame_error", {"error": str(exc)})
