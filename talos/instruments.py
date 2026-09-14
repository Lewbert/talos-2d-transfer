"""InstrumentManager: owns all device/camera proxies, job dispatch, and the
safety-critical stop_all/shutdown state machines.

Single-writer discipline: the manager is the only component that submits
jobs to proxies; the GUI emits commands to the manager.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QMetaObject, QObject, QTimer, Qt, Signal, Slot

from talos.hal.proxies import CameraProxy, DeviceProxy, FocusProxy
from talos.hal.registry import DEVICE_KEYS, MOTION_KEYS, make_camera, make_device
from talos.models import Job

logger = logging.getLogger(__name__)

STOP_ALL_BUDGET_MS = 1500


class InstrumentManager(QObject):
    sig_device_state = Signal(str, dict)     # device key -> state dict
    sig_log = Signal(str, str)               # level, message
    sig_stop_all_done = Signal()
    sig_job_done = Signal(int, object)
    sig_job_failed = Signal(int, str, str)
    sig_event = Signal(str, str, dict)       # device key, event, payload
    sig_job_submitted = Signal(str, str)     # device key, method — the AF-S
                                             # input-abort detector (service)

    def __init__(self, settings, sim: bool = False, parent: QObject | None = None):
        super().__init__(parent)
        self.settings = settings
        self.sim = sim
        self._job_counter = 0
        self._proxies: dict[str, DeviceProxy] = {}
        self._camera: CameraProxy | None = None
        self._enabled: dict[str, bool] = {}
        # True when a worker thread refused to join during shutdown (a
        # wedged device call) — the app must os._exit past Qt teardown
        # (destroying a live QThread aborts the process).
        self.shutdown_ragged = False
        # Latest telemetry snapshots per device (for UI consumers like the
        # flake "Go to" math, which needs the current stage position).
        self.last_position: dict[str, Any] = {}
        # Autofocus needs the current focus position (arm center) and the
        # camera's current exposure (restore after the AF run).
        self._focus_position: int = 0
        self._camera_props: dict[str, Any] = {}
        self._stop_acks = 0
        self._stop_budget_timer = QTimer(self)
        self._stop_budget_timer.setSingleShot(True)
        self._stop_budget_timer.timeout.connect(self._on_stop_budget)

        for key in DEVICE_KEYS:
            device_cfg = settings.device(key)
            if not device_cfg.get("enabled", True):
                continue
            poll_ms = int(device_cfg.get("poll_interval_ms", 100))
            # NOTE: device_cfg must be bound as a default argument — a bare
            # closure captures the loop variable and every device would get
            # the LAST section (hardware-verified bug, all-on-COM5).
            proxy_cls = FocusProxy if key == "focus" else DeviceProxy
            proxy = proxy_cls(key,
                              lambda k=key, cfg=device_cfg: make_device(k, cfg, sim),
                              poll_interval_ms=poll_ms, parent=self)
            # Bound slots (not lambdas): Qt lambdas with no receiver execute
            # in the EMITTING (worker) thread; slots on this manager queue
            # to the GUI thread. The key is recovered via sender().
            proxy.device_key = key
            proxy.sig_connected.connect(self._on_proxy_connected)
            proxy.sig_telem.connect(self._on_proxy_telem)
            proxy.sig_event.connect(self._on_proxy_event)
            proxy.sig_command_done.connect(self.sig_job_done)
            proxy.sig_command_failed.connect(self.sig_job_failed)
            proxy.sig_all_stopped.connect(self._on_proxy_stopped)
            self._proxies[key] = proxy

        camera_cfg = settings.device("camera")
        # The tick interval is a poll floor only — fetch blocks on the
        # camera's own cadence, so the loop self-paces (see CameraProxy).
        self._camera = CameraProxy(lambda: make_camera(camera_cfg, sim), parent=self)
        self._camera.sig_connected.connect(self._on_camera_connected)
        self._camera.sig_event.connect(self._on_camera_event)
        self._camera.sig_fps.connect(self._on_camera_fps)
        self._camera.sig_properties.connect(self._on_camera_props)
        # The same job-completion contract as the device proxies (the UI's
        # snapshot busy-gating depends on sig_job_done reaching it).
        self._camera.sig_command_done.connect(self.sig_job_done)
        self._camera.sig_command_failed.connect(self.sig_job_failed)

    # ------------------------------------------------------------------
    # Public API (GUI thread)
    # ------------------------------------------------------------------

    @Slot()
    def connect_all(self) -> None:
        for proxy in self._proxies.values():
            proxy.start()
        self._camera.start()

    def device(self, key: str) -> DeviceProxy | None:
        return self._proxies.get(key)

    @property
    def camera(self) -> CameraProxy:
        return self._camera

    @property
    def focus_position(self) -> int:
        """Last focus position from telemetry (steps) — the arm center for
        autofocus runs."""
        return self._focus_position

    @property
    def camera_props(self) -> dict[str, Any]:
        """Last canonical camera properties (exposure_us etc.)."""
        return dict(self._camera_props)

    def is_enabled(self, device_key: str) -> bool:
        """The software enable gate (defaults to enabled)."""
        return bool(self._enabled.get(device_key, True))

    def set_enabled(self, device_key: str, enabled: bool) -> None:
        """Software enable gate (reference behavior): disabling stops the
        device and drops further commands until re-enabled.

        The state is broadcast on sig_device_state so every enable control
        (the strip's checkbox, the Stage Control panels) follows when the
        gamepad toggles it — and vice versa.
        """
        self._enabled[device_key] = enabled
        if not enabled:
            proxy = self._proxies.get(device_key)
            if proxy is not None:
                proxy.enqueue_stop()
            self._log("warning", f"{device_key}: DISABLED (commands dropped)")
        else:
            self._log("info", f"{device_key}: enabled")
        self.sig_device_state.emit(device_key, {"enabled": enabled})

    def submit(self, device_key: str, method_name: str, *args, priority: int = 0) -> int:
        """Queue a driver command; returns a job id."""
        if not self._enabled.get(device_key, True):
            return -1
        proxy = self._proxies.get(device_key)
        if proxy is None:
            logger.warning("submit: unknown device %r", device_key)
            return -1
        self._job_counter += 1
        job = Job(self._job_counter, device_key, method_name, tuple(args), priority)
        # Debug-only: a held jog submits at input rate (~60 Hz), and both
        # the log write and the sig_log emit (which crosses to the GUI
        # thread) were measurable churn. Verbose logging is ON by default
        # until release, so these lines are still recorded.
        if logger.isEnabledFor(logging.DEBUG):
            self._log("debug", f"job #{job.job_id}: {device_key}.{method_name}{args}")
        proxy.enqueue(job.job_id, method_name, job.args, job.priority)
        self.sig_job_submitted.emit(device_key, method_name)
        return job.job_id

    def submit_camera(self, method_name: str, *args) -> int:
        """Queue a camera property/snapshot command; returns a job id."""
        if self._camera is None:
            return -1
        self._job_counter += 1
        self._camera.enqueue(self._job_counter, method_name, tuple(args))
        self.sig_job_submitted.emit("camera", method_name)
        return self._job_counter

    @Slot()
    def stop_all(self) -> None:
        """Emergency stop: focus → transfer → XYR, verified within budget."""
        self._stop_acks = 0
        self._log("warning", "STOP ALL requested")
        for key in MOTION_KEYS:
            proxy = self._proxies.get(key)
            if proxy is not None:
                proxy.enqueue_stop()
        if not self._stop_budget_timer.isActive():
            self._stop_budget_timer.start(STOP_ALL_BUDGET_MS)

    @Slot()
    def shutdown(self) -> None:
        """Ordered teardown: hard stop → camera off → proxies disconnect → threads join.

        Teardown is serialized through each worker's event loop (QUEUED
        invocation): stop/disconnect must never run on the GUI thread while
        the worker may be inside a blocking DLL call (snapshot poll etc.).
        """
        self.stop_all()
        # A running autofocus job blocks the focus worker's drain — abort it
        # directly so the queued stop/disconnect can run afterwards.
        focus = self._proxies.get("focus")
        if isinstance(focus, FocusProxy):
            focus.request_abort()
        if self._camera is not None:
            self._camera.set_streaming(False)
        for proxy in list(self._proxies.values()) + [self._camera]:
            if proxy is not None:
                QMetaObject.invokeMethod(proxy, "request_shutdown",
                                         Qt.ConnectionType.QueuedConnection)
        for proxy in list(self._proxies.values()) + [self._camera]:
            if proxy is not None and not proxy.wait(3000):
                # a wedged backend (e.g. Labscope holding the camera)
                # blocks the worker in a DLL call — one grace round
                if not proxy.wait(7000):
                    self.shutdown_ragged = True
                    self._log("error",
                              f"{proxy}: shutdown wait timeout (10 s) — "
                              "the thread is still inside a device call")
        self._log("info", "Shutdown complete")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_connected(self, key: str, ok: bool) -> None:
        self._log("info" if ok else "error",
                  f"{key}: {'connected' if ok else 'CONNECT FAILED'}")
        self.sig_device_state.emit(key, {"connected": ok})
        if ok and key == "camera":
            # Pull the backend properties once so the UI can show the
            # active backend and hardware ranges.
            self.submit_camera("get_properties")

    def _on_telem(self, key: str, payload: dict) -> None:
        if payload.get("position"):
            self.last_position[key] = payload["position"]
        if key == "focus":
            status = payload.get("status")
            if isinstance(status, dict) and "pos" in status:
                self._focus_position = int(status["pos"])
        self.sig_device_state.emit(key, payload)

    def _on_event(self, key: str, event: str, payload: dict) -> None:
        self._log("warning", f"{key}: event {event} {payload}")
        self.sig_event.emit(key, event, payload)

    # ------------------------------------------------------------------
    # Proxy signal slots (bound → queued to the GUI thread; the sender's
    # device_key routes the event — lambdas would run on worker threads).
    # ------------------------------------------------------------------

    def _sender_key(self) -> str:
        sender = self.sender()
        return getattr(sender, "device_key", "?") if sender is not None else "?"

    @Slot(bool)
    def _on_proxy_connected(self, ok: bool) -> None:
        self._on_connected(self._sender_key(), ok)

    @Slot(dict)
    def _on_proxy_telem(self, payload: dict) -> None:
        self._on_telem(self._sender_key(), payload)

    @Slot(str, dict)
    def _on_proxy_event(self, event: str, payload: dict) -> None:
        self._on_event(self._sender_key(), event, payload)

    @Slot(bool)
    def _on_camera_connected(self, ok: bool) -> None:
        self._on_connected("camera", ok)

    @Slot(str, dict)
    def _on_camera_event(self, event: str, payload: dict) -> None:
        self._on_event("camera", event, payload)

    @Slot(float)
    def _on_camera_fps(self, fps: float) -> None:
        self.sig_device_state.emit("camera", {"fps": fps})

    @Slot(dict)
    def _on_camera_props(self, props: dict) -> None:
        self._camera_props = dict(props)
        self.sig_device_state.emit("camera", props)

    def _on_proxy_stopped(self) -> None:
        self._stop_acks += 1
        if self._stop_acks >= len([k for k in MOTION_KEYS if k in self._proxies]):
            self._stop_budget_timer.stop()
            self._log("info", "All stages stopped")
            self.sig_stop_all_done.emit()

    @Slot()
    def _on_stop_budget(self) -> None:
        self._log("warning", "STOP ALL budget elapsed (some acks missing)")
        self.sig_stop_all_done.emit()

    def _log(self, level: str, message: str) -> None:
        getattr(logger, level)(message)
        self.sig_log.emit(level, message)
