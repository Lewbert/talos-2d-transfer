"""SmartCamApi camera backend — the ZEN/Labscope path for the Zeiss
Axiocam 202/208 (libusb0-bound, no ZEN required to run).

Complete rewrite built on the API facts extracted from ZEN's own SmartCam
wrapper (docs/SMARTCAM_API.md):

- SetParameterValue/GetParameterValue take POINTERS to the value (the old
  code passed a double by value — that is why the blind ID sweep no-op'd).
- Live = StartContinuousAcquisition + the ImageAcquired event callback
  (ZEN's flow; the old code restarted the sequence per frame).
- Snap = StartSequenceAcquisition(1) -> poll GetSequenceImage -> Abort.
- The transfer's "YUV420" is semi-planar VU-order (NV21) — decoded
  natively by OpenCV (old decode misread the chroma as U-first).
- After OpenCamera the camera needs a ~2 s settle.

Requirements: SmartCamApi.dll (ships with ZEN blue / Labscope) and
exclusive access — ZEN and Labscope must be closed. Override the DLL via
the SMARTCAM_DLL env var (fallback experiments); override the decode via
SMARTCAM_PIXEL_FORMAT.
"""

from __future__ import annotations

import collections
import ctypes
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from talos.hal.base import Camera, CommandRejectedError, DeviceConnectionError
from talos.hal.devices.camera import smartcam_decode as sd
from talos.hal.devices.camera.smartcam_params import (
    AUTO_OFF,
    AUTO_ON,
    TRANSFER_FORMATS,
    WB_NAMES,
    WB_PRESETS,
    ApiError,
    MetadataType,
    ParamKey,
    PROPERTIES,
    ZEN_DEFAULTS,
    clamp,
)

logger = logging.getLogger(__name__)

# Process-global last-close timestamp for the reopen settle: the DLL
# state is process-wide (ctypes caches the loaded library), so the guard
# must span camera INSTANCES — a fresh instance reconnecting right after
# another instance closed would otherwise skip it.
_LAST_CLOSE_T = 0.0

_SMARTCAM_DLL_PATHS = [
    r"C:\Program Files\Carl Zeiss\Labscope\DriverInstaller"
    r"\Primostar3HD_Axiocam202_208\SmartCamApi.dll",
    r"C:\Program Files\Carl Zeiss\Labscope\SmartCamApi.dll",
    r"C:\Program Files\Carl Zeiss\ZEN 2\ZEN 2 (blue edition)\SmartCamApi.dll",
    r"C:\Program Files (x86)\Carl Zeiss\Labscope\SmartCamApi.dll",
]

_LIVE_BUFFER_COUNT = 5         # ZEN's AcquisitionBufferCount (min 5)
_EVENT_IMAGE_ACQUIRED = 5      # ApiEvent.ImageAcquired


def _find_dll() -> str | None:
    for path in _SMARTCAM_DLL_PATHS:
        if os.path.isfile(path):
            return path
    for root in (r"C:\Program Files\Carl Zeiss", r"C:\Program Files (x86)\Carl Zeiss"):
        if os.path.isdir(root):
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if fn.lower() == "smartcamapi.dll":
                        return os.path.join(dirpath, fn)
    return None


class _ApiOptions(ctypes.Structure):
    _fields_ = [("OptionVersion", ctypes.c_ushort),
                ("OptionSize", ctypes.c_ushort),
                ("OptionFlags", ctypes.c_uint)]


class _ApiInformation(ctypes.Structure):
    _fields_ = [("InfoVersion", ctypes.c_ushort), ("InfoSize", ctypes.c_ushort),
                ("ApiVersion", ctypes.c_ushort), ("MaxStringLength", ctypes.c_ushort),
                ("MaxParameterCount", ctypes.c_ushort),
                ("MaxFunctionCount", ctypes.c_ushort),
                ("MaxEventCount", ctypes.c_ushort), ("MaxError", ctypes.c_ushort),
                ("ImageHeaderSize", ctypes.c_ushort)]


class _EventData(ctypes.Structure):
    _fields_ = [("EventID", ctypes.c_int), ("_pad", ctypes.c_int),
                ("Param1", ctypes.c_longlong), ("Param2", ctypes.c_longlong)]


_EVENT_CB = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_void_p)


class SmartCamCamera(Camera):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._dll = None
        self._handle: ctypes.c_void_p | None = None
        self._lib_initialized = False
        self._info = None
        self._param_types: dict[int, int] = {}
        self._forced_format = os.environ.get("SMARTCAM_PIXEL_FORMAT")
        self._dll_path = os.environ.get("SMARTCAM_DLL") or _find_dll()
        # live acquisition state
        self._streaming = False
        self._cb = None
        self._live_buf = None
        self._retired_buf = None
        self._frame_size = 0
        # (payload, capture timestamp) published as ONE tuple: as separate
        # attributes a fetch could land between the two stores and pair a
        # NEW frame with the PREVIOUS stamp — a one-frame (~47 ms) error,
        # i.e. ~25 steps of stage position inside a 500 sps AF sweep.
        self._pending: tuple[bytes, float] | None = None
        # Guards the callback-written state (pending + event times): the
        # DLL's own thread appends to _event_times, and list(deque) while
        # it appends raises "deque mutated during iteration".
        self._state_lock = threading.Lock()
        self._last_fetched_t: float | None = None
        self._latency_cache_s: float | None = None
        self._latency_cache_t = 0.0
        self._event_times: collections.deque[float] = collections.deque(maxlen=60)
        self._last_event_t: float | None = None
        self._last_dump_t = 0.0  # stall-dump rate limit (see _on_event)
        # Cached DLL parameter readback (see get_properties); invalidated
        # by every write.
        self._props_cache: dict[str, Any] | None = None
        # decode geometry (updated after Resolution changes)
        self._width, self._height = 1920, 1080
        # ZEN software white-balance (applied on decode, like ZEN's LUT)
        smartcam_cfg = self.config.get("smartcam", {})
        if smartcam_cfg.get("apply_wb", True):
            kelvin = int(smartcam_cfg.get("wb_kelvin", 5500))
            if kelvin not in WB_PRESETS:
                # Only 3200/5500 exist: any other setting silently used the
                # 5500K factors, so a "4000 K" image came out with the
                # wrong tint and no explanation.
                logger.warning(
                    "smartcam.wb_kelvin=%d has no software WB preset "
                    "(available: %s) — applying the 5500 K factors",
                    kelvin, sorted(WB_PRESETS))
            self._wb_factors = WB_PRESETS.get(kelvin, WB_PRESETS[5500])
        else:
            self._wb_factors = None

    @property
    def device_id(self) -> str:
        return "camera@smartcam"

    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self.is_connected:
            return  # double-connect would tear down the healthy session
        # The DLL hangs indefinitely when re-opened immediately after a
        # close (bench-verified: a 0 s gap wedged the connect >4 min; a
        # 10 s settle reconnected cleanly in 3.6 s) — settle first. The
        # timestamp is PROCESS-global: the loaded DLL is shared, so a
        # fresh instance after another instance's close still settles.
        global _LAST_CLOSE_T
        reopen_settle = float(self.config.get("smartcam", {})
                              .get("reopen_settle_s", 10.0))
        since_close = time.monotonic() - _LAST_CLOSE_T
        if since_close < reopen_settle:
            time.sleep(reopen_settle - since_close)
        dll_path = self._dll_path
        if dll_path is None:
            raise DeviceConnectionError(
                "SmartCamApi.dll not found (install ZEN blue or Labscope)")
        try:
            os.add_dll_directory(str(Path(dll_path).parent))
        except OSError:
            pass
        try:
            # Exports are Cdecl (docs/SMARTCAM_API.md); CDLL is correct on
            # x64 and 32-bit alike. WinDLL kept as a fallback.
            self._dll = ctypes.CDLL(dll_path)
        except OSError:
            try:
                self._dll = ctypes.WinDLL(dll_path)
            except OSError as exc:
                raise DeviceConnectionError(
                    f"SmartCamApi.dll could not be loaded: {exc}") from exc
        smartcam_cfg = self.config.get("smartcam", {})
        retries = int(smartcam_cfg.get("enum_retries", 30))
        retry_sleep = float(smartcam_cfg.get("retry_sleep_s", 1.0))
        self._setup_prototypes()
        try:
            # ZEN retries up to 30x while the camera enumerates; each
            # attempt re-initializes the library (finalize + init cycle).
            count = ctypes.c_int(0)
            for attempt in range(retries):
                options = _ApiOptions(256, ctypes.sizeof(_ApiOptions), 0)
                self._call(self._dll.ApiLib_InitializeLibrary, "InitializeLibrary",
                           ctypes.byref(options))
                self._lib_initialized = True
                self._call(self._dll.ApiLib_GetCameraCount, "GetCameraCount",
                           ctypes.byref(count))
                if count.value > 0:
                    break
                self._dll.ApiLib_FinalizeLibrary()
                self._lib_initialized = False
                if attempt < retries - 1:
                    time.sleep(retry_sleep)
            if count.value == 0:
                raise DeviceConnectionError(
                    "SmartCamApi found no cameras — close Labscope/ZEN and "
                    "check the USB3 connection and the libusb0 binding")
            info = _ApiInformation(1, ctypes.sizeof(_ApiInformation))
            rc = self._dll.ApiLib_GetLibraryInformation(ctypes.byref(info))
            if rc == 0:
                self._info = info
            handle = ctypes.c_void_p()
            self._call(self._dll.ApiCam_OpenCamera, "OpenCamera", 0,
                       ctypes.byref(handle))
            if not handle.value:
                raise DeviceConnectionError("SmartCamApi OpenCamera returned no handle")
            self._handle = handle
            time.sleep(float(smartcam_cfg.get("settle_s", 2.0)))  # camera settle
            self._enumerate_parameters()
            # Register the event handler once for the camera lifetime (ZEN
            # registers it in InitializeCore and removes it at shutdown).
            self._cb = _EVENT_CB(self._on_event)
            rc = self._dll.ApiCam_AddEventHandler(self._handle, self._cb)
            if rc != 0:
                raise DeviceConnectionError(f"SmartCamApi AddEventHandler failed: "
                                            f"{self._error_string(rc)}")
            if smartcam_cfg.get("apply_defaults", True):
                self._apply_defaults()
        except Exception:
            self._cleanup()
            raise
        self._connected = True
        logger.info("Connected via SmartCamApi (dll=%s, %d camera(s), %d params)",
                    dll_path, count.value, len(self._param_types))

    def disconnect(self) -> None:
        self._cleanup()
        self._connected = False
        global _LAST_CLOSE_T
        _LAST_CLOSE_T = time.monotonic()

    @property
    def is_connected(self) -> bool:
        return self._connected and self._handle is not None and bool(self._handle.value)

    def _query_frame_size(self) -> int:
        """Ask the DLL for the single-frame buffer size; fall back to the
        NV12 geometry of the current mode (the old _LIVE_BYTES constant
        matched no real mode and silently zeroed the stream)."""
        if self._dll is not None and self._handle:
            size = ctypes.c_ulonglong(0)
            rc = self._dll.ApiCam_GetAcquisitionBufferSize(self._handle, 1,
                                                           ctypes.byref(size))
            if rc == 0 and size.value > 0:
                return int(size.value)
        fallback = self._width * self._height * 3 // 2
        logger.warning("SmartCamApi GetAcquisitionBufferSize unavailable — "
                       "using NV12 geometry fallback %d bytes", fallback)
        return fallback

    def start(self) -> None:
        """Begin continuous live acquisition (ZEN's StartLiveCore flow)."""
        if not self.is_connected or self._streaming:
            return
        new_size = self._query_frame_size()
        # Reuse the buffer when the size is unchanged; retire (not free) the
        # old one on size changes — AbortAcquisition is not provably
        # synchronous with the DLL's in-flight DMA, so a freed buffer could
        # still be written. The retired buffer is freed on the NEXT start.
        if self._live_buf is not None and new_size == self._frame_size:
            buffer = self._live_buf
        else:
            buffer = (ctypes.c_char * (new_size * _LIVE_BUFFER_COUNT))()
            self._retired_buf = self._live_buf
            self._live_buf = buffer
        self._frame_size = new_size
        self._pending = None
        rc = self._dll.ApiCam_StartContinuousAcquisition(
            self._handle, new_size * _LIVE_BUFFER_COUNT, buffer)
        if rc != 0:
            raise DeviceConnectionError(f"SmartCamApi StartContinuousAcquisition failed: "
                                        f"{self._error_string(rc)}")
        self._streaming = True
        with self._state_lock:
            self._event_times.clear()
        self._last_event_t = None

    def stop(self) -> None:
        if self._streaming and self._dll is not None and self._handle:
            try:
                self._dll.ApiCam_AbortAcquisition(self._handle)
            except Exception:  # noqa: BLE001
                pass
        self._streaming = False
        with self._state_lock:
            self._pending = None

    # ------------------------------------------------------------------

    def _on_event(self, event_index: int, event_data_ptr: int) -> None:
        """ImageAcquired callback — runs on the DLL's acquisition thread
        (GIL held). Copy the frame out of the ring buffer immediately.

        The frame size is captured in a local: the GIL can switch at the
        string_at call boundary, so reading self._frame_size twice (guard
        + copy) could straddle a resolution change and copy the new size
        from the old (retired) buffer.
        """
        if event_index != _EVENT_IMAGE_ACQUIRED or not self._streaming:
            return
        size = self._frame_size
        try:
            ev = ctypes.cast(event_data_ptr, ctypes.POINTER(_EventData)).contents
            if ev.Param2 == size and ev.Param1:
                now = time.monotonic()
                with self._state_lock:
                    self._event_times.append(now)
                # Stall watchdog: the camera delivers frames at ~47 ms cadence;
                # a gap far above that is either a camera-pipeline stall or a
                # USB link retrain (e.g. EMI from stage motors on long cables).
                # Log it so hardware sessions can correlate lags with motion.
                if self._last_event_t is not None:
                    gap_ms = (now - self._last_event_t) * 1000.0
                    warn_ms = float(self.config.get("smartcam", {}).get(
                        "stall_warn_ms", 300.0))
                    if gap_ms > warn_ms:
                        logger.warning(
                            "SmartCamApi frame gap %.0f ms (camera delivery "
                            "stall) — check USB cabling/shielding and "
                            "nearby motor EMI", gap_ms)
                        # Rate-limited: during an EMI burst EVERY frame
                        # exceeds the threshold, and a stack dump + file
                        # flush per frame is a storm exactly when the
                        # pipeline needs to recover.
                        if now - self._last_dump_t > 5.0:
                            self._last_dump_t = now
                            self._dump_thread_stacks()
                self._last_event_t = now
                # the memcpy happens OUTSIDE the lock (3 MB at 1080p)
                payload = ctypes.string_at(ev.Param1, size)
                with self._state_lock:
                    self._pending = (payload, now)
        except Exception:  # noqa: BLE001
            logger.debug("SmartCamApi event copy failed", exc_info=True)

    def _dump_thread_stacks(self) -> None:
        """Stall forensics: log every Python thread's current stack. The
        callback runs the moment the stall ends — if the gap was GIL
        starvation, the holder's stack shows where it was just released."""
        try:
            import sys
            import traceback
            for tid, frame in sys._current_frames().items():
                stack = traceback.extract_stack(frame)
                brief = " | ".join(
                    f"{s.filename.rsplit(chr(92), 1)[-1]}:{s.lineno}:{s.name}"
                    for s in stack[-6:])
                logger.warning("  stall: thread %s stack: %s", tid, brief)
        except Exception:  # noqa: BLE001
            logger.debug("stall stack dump failed", exc_info=True)

    def fetch(self, timeout_ms: float = 2000.0) -> np.ndarray | None:
        if not self._streaming or self._dll is None:
            return None
        deadline = time.monotonic() + timeout_ms / 1000.0
        while self._streaming and time.monotonic() < deadline:
            with self._state_lock:
                pending = self._pending
                self._pending = None
            if pending is not None:
                payload, t_capture = pending
                self._last_fetched_t = t_capture
                raw = np.frombuffer(payload, dtype=np.uint8)
                return self._decode(raw)
            time.sleep(0.003)
        return None

    def capture_time(self) -> float | None:
        """Content-capture timestamp (monotonic) of the last fetched
        frame. The ImageAcquired event fires when the frame is DELIVERED;
        the scene was captured ~half an exposure + transfer earlier.
        High-speed focus passes interpolate stage positions from this
        stamp — without the correction the measured peak shifts by
        v×latency (hardware-verified: ~25 steps off at 500 sps), and
        freshness gates accept frames whose content still shows the
        PREVIOUS focus position."""
        t = self._last_fetched_t
        if t is None:
            return None
        return t - self._capture_latency_s()

    def _capture_latency_s(self) -> float:
        """Exposure midpoint + transfer/pipeline allowance, cached until
        an exposure write invalidates it (see _set_param).

        This is called once per delivered frame by capture_time(). The
        value depends ONLY on the exposure + a transfer allowance, so a
        time-based refresh is pure waste — and actively harmful: the live
        DLL parameter query stalls one frame's delivery (~330 ms on the
        bench's USB path — hardware-measured), and a 2 s TTL turned that
        into a PERIODIC stall at exactly the TTL cadence, tripping the
        bench health gate every time. The query now happens at most once
        per connect/exposure-change; a per-frame query throttles the
        pipeline to ~3 fps (the original hardware-verified failure)."""
        if self._latency_cache_s is not None:
            return self._latency_cache_s
        try:
            spec = PROPERTIES["exposure_us"]
            raw = self._get_param_raw(int(spec["key"]), spec["kind"])
            exposure_s = (raw * spec.get("scale", 1.0)) / 1e6
        except Exception:  # noqa: BLE001
            exposure_s = 0.02
        transfer_ms = float(self.config.get("smartcam", {}).get(
            "capture_latency_ms", 15.0))
        self._latency_cache_s = exposure_s / 2.0 + transfer_ms / 1000.0
        self._latency_cache_t = time.monotonic()
        return self._latency_cache_s

    def _decode(self, raw: np.ndarray) -> np.ndarray | None:
        if self._forced_format:
            return self._decode_format(raw, self._forced_format)
        w, h = self._width, self._height
        frame_bytes = w * h * 3 // 2  # NV12 payload for the current geometry
        if raw.size < frame_bytes:
            # The acquisition buffer is the camera's MAX transfer size
            # (16.6 MB, fixed at every resolution) and the live frame sits
            # in its PREFIX — a buffer smaller than the payload is a
            # truncated frame, which used to fall through to a luma-only
            # crop with no explanation.
            logger.warning(
                "SmartCamApi buffer %d is smaller than the %dx%d frame "
                "(%d bytes) — decoding the luma plane only; the resolution "
                "change did not refresh the pipeline", raw.size, w, h,
                frame_bytes)
        try:
            return sd.decode_yuv420(raw, w, h, wb=self._wb_factors)
        except ValueError:
            if raw.size >= w * h:
                gray = raw[:w * h].reshape(h, w)
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            logger.warning("Unexpected buffer size %d — cannot decode", raw.size)
            return None

    # ------------------------------------------------------------------

    def set_property(self, name: str, value: Any) -> None:
        if name not in PROPERTIES:
            raise KeyError(f"Unknown camera property: {name}")
        if not self.is_connected:
            raise DeviceConnectionError("Camera not connected")
        if name == "white_balance" and isinstance(value, str):
            requested = value
            value = {"Continuous": AUTO_ON, "Once": 2, "Off": AUTO_OFF}.get(value)
            if value is None:
                raise ValueError(f"Invalid white_balance mode: {requested!r}")
        if name == "resolution":
            # Resolution changes re-initialize the camera pipeline — stop the
            # stream, set + refresh geometry, then resume. 1080p for
            # real-time work; 4K only for deliberate high-res captures.
            was_streaming = self._streaming
            self.stop()
            try:
                self._set_param(name, value)
                time.sleep(0.3)  # pipeline re-init
                self._refresh_geometry()
            finally:
                if was_streaming:
                    self.start()
            return
        self._set_param(name, value)

    def _set_param(self, name: str, value: Any) -> float:
        spec = PROPERTIES[name]
        key = int(spec["key"])
        kind = self._param_kind(key, spec["kind"])
        value = float(clamp(name, value))
        scale = spec.get("scale", 1.0)
        if scale != 1.0:
            value = value / scale
        if kind == "double":
            arg = ctypes.c_double(value)
            # exposure increments by 0.01 ms — tolerate half a step
            tolerance = max(0.006, abs(value) * 1e-4)
        elif kind == "byte":
            arg = ctypes.c_byte(int(round(value)))
            tolerance = 0.5
        else:
            arg = ctypes.c_int(int(round(value)))
            tolerance = 0.5
        rc = self._dll.ApiCam_SetParameterValue(self._handle, key, ctypes.byref(arg))
        if rc != 0:
            raise CommandRejectedError(
                f"SmartCamApi set {name} ({key}) to {value}: {self._error_string(rc)}")
        # The readback query is EXPENSIVE on the live pipeline (one frame
        # delivery stalls ~330 ms per query — hardware-measured). Gain is
        # written repeatedly by the software auto-gain loop, so it skips
        # the readback entirely (write-only diagnostics would stall the
        # stream for no gain in certainty); exposure keeps it.
        readback = None if name == "gain" else self._get_param_raw(key, kind)
        if readback is not None and abs(readback - value) > tolerance:
            logger.warning("SmartCamApi %s write/readback mismatch: set %.4f, read %.4f",
                           name, value, readback)
        if name == "exposure_us":
            # the capture-latency cache depends on it — invalidate on
            # EVERY write path (set_property AND _apply_defaults)
            self._latency_cache_s = None
        # any write invalidates the readback cache (get_properties)
        self._props_cache = None
        return value

    def _param_kind(self, key: int, fallback: str) -> str:
        """The type the camera DECLARES for a parameter (0 byte/1 int/2 double).

        The property table's kind is not authoritative: the DLL writes a
        value of the declared type into the pointer it is given, so a
        double-valued parameter read into c_int is an out-of-bounds
        ctypes write (and a silently wrong readback otherwise).
        """
        if self._param_types:
            return {0: "byte", 1: "int", 2: "double"}.get(
                self._param_types.get(key), fallback)
        return fallback

    def _get_param_raw(self, key: int, kind: str = "int") -> float | None:
        if self._dll is None or not self._handle:
            return None
        kind = self._param_kind(key, kind)
        if kind == "double":
            out = ctypes.c_double(0.0)
        elif kind == "byte":
            out = ctypes.c_byte(0)
        else:
            out = ctypes.c_int(0)
        rc = self._dll.ApiCam_GetParameterValue(self._handle, key, ctypes.byref(out))
        if rc != 0:
            return None
        return float(out.value)

    def get_properties(self) -> dict[str, Any]:
        props: dict[str, Any] = {
            "backend": "smartcam",
            "resolution": [self._width, self._height],
            "format": self._forced_format or "nv12",
            "transfer_formats": TRANSFER_FORMATS,
            "dll": self._dll_path,
        }
        # The five parameter queries below each stall frame delivery
        # ~330 ms (hardware-measured), i.e. ~1.6 s of frozen live view per
        # call — and the UI asks for properties on connect and on every
        # panel refresh. Cache them; every write invalidates the cache
        # (see _set_param), so the values are never older than the last
        # command. fps is deliberately NOT cached: it is derived from
        # event arrival times and must stay live.
        cached = self._props_cache
        if cached is None:
            cached = {}
            for name in ("exposure_us", "gain", "color_mode", "white_balance",
                         "color_temperature"):
                spec = PROPERTIES[name]
                kind = "double" if spec["kind"] == "double" else "int"
                raw = self._get_param_raw(int(spec["key"]), kind)
                if raw is not None:
                    cached[name] = raw * spec.get("scale", 1.0)
            self._props_cache = cached
        props.update(cached)
        # fps from event-arrival timestamps = the TRUE camera rate
        # (fetch timestamps measure the consumer cadence instead).
        # Copied under the lock: the DLL's callback thread appends to this
        # deque, and iterating it concurrently raises RuntimeError.
        with self._state_lock:
            t = list(self._event_times)
        if t:
            if len(t) >= 2:
                span = t[-1] - t[0]
                props["fps"] = round((len(t) - 1) / span, 1) if span > 0 else 0.0
        return props

    def snapshot(self, path: Path, timeout_s: float = 15.0,
                 resolution: int | None = None,
                 burn: dict | None = None) -> Path:
        """Full-quality still via the ZEN snap flow (sequence acquisition).

        ``resolution``: temporary mode switch for the capture (0 = 4K,
        1 = 1080p) — the live stream pauses for the pipeline re-init and
        is restored afterwards. The frame buffer size is re-queried after
        the switch (it is per-mode; the connect-time size would make the
        DLL write a 4K frame into a 1080p buffer).
        """
        if not self.is_connected or self._dll is None:
            raise DeviceConnectionError("Camera not connected")
        current_mode = 0 if self._width >= 3840 else 1
        switched = resolution is not None and int(resolution) != current_mode
        if switched:
            # set_property("resolution") handles stop → set → geometry →
            # resume internally.
            self.set_property("resolution", int(resolution))
            self._frame_size = self._query_frame_size()
        if self._frame_size <= 0:
            self._frame_size = self._query_frame_size()
        was_streaming = self._streaming
        self.stop()
        try:
            buf = (ctypes.c_char * self._frame_size)()
            rc = self._dll.ApiCam_StartSequenceAcquisition(
                self._handle, 1, self._frame_size, buf)
            if rc != 0:
                raise DeviceConnectionError(
                    f"SmartCamApi StartSequenceAcquisition failed: {self._error_string(rc)}")
            deadline = time.monotonic() + timeout_s
            while True:
                rc = self._dll.ApiCam_GetSequenceImage(self._handle)
                if rc == 0:
                    break
                if rc != ApiError.ImageNotReady:
                    self._dll.ApiCam_AbortAcquisition(self._handle)
                    raise DeviceConnectionError(
                        f"SmartCamApi GetSequenceImage failed: {self._error_string(rc)}")
                if time.monotonic() > deadline:
                    self._dll.ApiCam_AbortAcquisition(self._handle)
                    raise DeviceConnectionError("SmartCamApi: snapshot timed out")
                time.sleep(0.005)
            self._dll.ApiCam_AbortAcquisition(self._handle)
            frame = self._decode(np.frombuffer(buf, dtype=np.uint8,
                                               count=self._frame_size))
            if frame is None:
                raise DeviceConnectionError("SmartCamApi: snapshot decode failed")
            if burn and burn.get("um_per_px"):
                from talos.cv.scale_bar import draw_scale_bar_cv

                frame = draw_scale_bar_cv(frame, float(burn["um_per_px"]))
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            return path
        finally:
            if switched:
                try:
                    # restore the live mode (the stream is stopped here,
                    # so set_property's internal resume cannot fire)
                    self.set_property("resolution", current_mode)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("SmartCamApi: snapshot resolution restore "
                                   "failed: %s", exc)
            if was_streaming:
                self.start()

    # ------------------------------------------------------------------

    def _enumerate_parameters(self) -> None:
        max_count = self._info.MaxParameterCount if self._info else 128
        if max_count == 0:
            max_count = 128
        # +1 slot for the 0 terminator — a camera with exactly MaxParameterCount
        # parameters would otherwise write the terminator one int past the array.
        arr = (ctypes.c_int * (max_count + 1))()
        try:
            rc = self._dll.ApiCam_GetParameterList(self._handle, arr)
            if rc != 0:
                logger.warning("GetParameterList rc=%d (%s)", rc, self._error_string(rc))
                return
        except Exception:  # noqa: BLE001
            return
        for i in range(max_count + 1):
            key = arr[i]
            if key == 0:
                break
            typ = ctypes.c_int(0)
            try:
                self._dll.ApiCam_GetParameterMetadata(self._handle, key,
                                                      int(MetadataType.Type),
                                                      ctypes.byref(typ))
                self._param_types[key] = typ.value
            except Exception:  # noqa: BLE001
                pass

    def _apply_defaults(self) -> None:
        """Apply ZEN's AfterInitialize values (color + 20 ms exposure).

        Reads the current value first and skips unchanged parameters —
        writing Resolution for instance re-initializes the camera pipeline.
        The user-editable settings exposure_us/gain/white_balance/
        color_temperature override the ZEN constants so the settings file
        is authoritative — the camera re-initializes some parameters at
        open, so without this every reconnect would flip AWB back ON.
        color_temperature is deliberately NOT in ZEN_DEFAULTS: absent a
        settings value, the camera's stored/ZEN value is left alone.
        """
        defaults = dict(ZEN_DEFAULTS)
        defaults["resolution"] = int(self.config.get("resolution",
                                                     ZEN_DEFAULTS["resolution"]))
        if "exposure_us" in self.config:
            defaults["exposure_us"] = float(self.config["exposure_us"])
        if "gain" in self.config:
            defaults["gain"] = float(self.config["gain"])
        if "white_balance" in self.config:
            defaults["white_balance"] = WB_NAMES.get(
                str(self.config["white_balance"]), defaults["white_balance"])
        if "color_temperature" in self.config:
            defaults["color_temperature"] = int(self.config["color_temperature"])
        for name, value in defaults.items():
            spec = PROPERTIES[name]
            key = int(spec["key"])
            if self._param_types and key not in self._param_types:
                continue
            kind = "double" if spec["kind"] == "double" else "int"
            try:
                current = self._get_param_raw(key, kind)
                if current is not None:
                    current = current * spec.get("scale", 1.0)
                    # camera quantization: exposure steps are 10 us
                    tolerance = max(10.0, abs(value) * 1e-4) \
                        if spec["kind"] == "double" else 0.5
                    if abs(current - float(value)) < tolerance:
                        continue
                self._set_param(name, value)
                logger.info("SmartCamApi default %s = %s", name, value)
            except Exception as exc:  # noqa: BLE001
                logger.warning("SmartCamApi default %s failed: %s", name, exc)
        self._refresh_geometry()

    def _refresh_geometry(self) -> None:
        """Re-read CameraSize (current mode) after resolution changes."""
        if self._dll is None or not self._handle:
            return
        arr = (ctypes.c_int * 2)()
        try:
            rc = self._dll.ApiCam_GetParameterValue(self._handle,
                                                    int(ParamKey.CameraSize), arr)
            if rc == 0 and arr[0] > 0 and arr[1] > 0:
                self._width, self._height = int(arr[0]), int(arr[1])
        except Exception:  # noqa: BLE001
            pass

    def _setup_prototypes(self) -> None:
        dll = self._dll
        dll.ApiLib_InitializeLibrary.restype = ctypes.c_int
        dll.ApiLib_InitializeLibrary.argtypes = [ctypes.c_void_p]
        dll.ApiLib_FinalizeLibrary.restype = ctypes.c_int
        dll.ApiLib_FinalizeLibrary.argtypes = []
        dll.ApiLib_GetLibraryInformation.restype = ctypes.c_int
        dll.ApiLib_GetLibraryInformation.argtypes = [ctypes.c_void_p]
        dll.ApiLib_GetCameraCount.restype = ctypes.c_int
        dll.ApiLib_GetCameraCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        dll.ApiLib_GetErrorDescription.restype = ctypes.c_int
        dll.ApiLib_GetErrorDescription.argtypes = [ctypes.c_int, ctypes.c_char_p,
                                                   ctypes.c_int]
        dll.ApiCam_OpenCamera.restype = ctypes.c_int
        dll.ApiCam_OpenCamera.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        dll.ApiCam_CloseCamera.restype = ctypes.c_int
        dll.ApiCam_CloseCamera.argtypes = [ctypes.c_void_p]
        # parameter API: the value argument is always a POINTER (ref/out)
        dll.ApiCam_GetParameterList.restype = ctypes.c_int
        dll.ApiCam_GetParameterList.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        dll.ApiCam_GetParameterMetadata.restype = ctypes.c_int
        dll.ApiCam_GetParameterMetadata.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                    ctypes.c_int, ctypes.c_void_p]
        dll.ApiCam_GetEnumParameterMetadata.restype = ctypes.c_int
        dll.ApiCam_GetEnumParameterMetadata.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                        ctypes.c_int, ctypes.c_void_p,
                                                        ctypes.c_char_p]
        dll.ApiCam_GetParameterDescription.restype = ctypes.c_int
        dll.ApiCam_GetParameterDescription.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                       ctypes.c_char_p]
        dll.ApiCam_GetParameterValue.restype = ctypes.c_int
        dll.ApiCam_GetParameterValue.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                 ctypes.c_void_p]
        dll.ApiCam_SetParameterValue.restype = ctypes.c_int
        dll.ApiCam_SetParameterValue.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                 ctypes.c_void_p]
        # acquisition API
        dll.ApiCam_GetAcquisitionBufferSize.restype = ctypes.c_int
        dll.ApiCam_GetAcquisitionBufferSize.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        dll.ApiCam_AcquireSingleImage.restype = ctypes.c_int
        dll.ApiCam_AcquireSingleImage.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong,
                                                  ctypes.c_void_p]
        dll.ApiCam_StartSequenceAcquisition.restype = ctypes.c_int
        dll.ApiCam_StartSequenceAcquisition.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong, ctypes.c_void_p]
        dll.ApiCam_GetSequenceImage.restype = ctypes.c_int
        dll.ApiCam_GetSequenceImage.argtypes = [ctypes.c_void_p]
        dll.ApiCam_StartContinuousAcquisition.restype = ctypes.c_int
        dll.ApiCam_StartContinuousAcquisition.argtypes = [ctypes.c_void_p,
                                                          ctypes.c_longlong,
                                                          ctypes.c_void_p]
        dll.ApiCam_RaiseSoftwareTrigger.restype = ctypes.c_int
        dll.ApiCam_RaiseSoftwareTrigger.argtypes = [ctypes.c_void_p]
        dll.ApiCam_AbortAcquisition.restype = ctypes.c_int
        dll.ApiCam_AbortAcquisition.argtypes = [ctypes.c_void_p]
        # events + functions
        dll.ApiCam_GetEventList.restype = ctypes.c_int
        dll.ApiCam_GetEventList.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        dll.ApiCam_AddEventHandler.restype = ctypes.c_int
        dll.ApiCam_AddEventHandler.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        dll.ApiCam_RemoveEventHandler.restype = ctypes.c_int
        dll.ApiCam_RemoveEventHandler.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        dll.ApiCam_GetFunctionList.restype = ctypes.c_int
        dll.ApiCam_GetFunctionList.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        dll.ApiCam_PerformFunction.restype = ctypes.c_int
        dll.ApiCam_PerformFunction.argtypes = [ctypes.c_int]
        dll.ApiCam_GetImageMetadata.restype = ctypes.c_int
        dll.ApiCam_GetImageMetadata.argtypes = [ctypes.c_int, ctypes.c_void_p,
                                                ctypes.c_void_p]

    def _call(self, func, name: str, *args) -> int:
        rc = int(func(*args))
        if rc != 0:
            raise DeviceConnectionError(f"SmartCamApi {name} failed: "
                                        f"{self._error_string(rc)}")
        return rc

    def _error_string(self, code: int) -> str:
        name = ApiError(code).name if code in iter(ApiError) else f"error {code}"
        try:
            buf = ctypes.create_string_buffer(256)
            self._dll.ApiLib_GetErrorDescription(code, buf, 256)
            return f"{name} ({buf.value.decode('ascii', 'replace')})"
        except Exception:  # noqa: BLE001
            return name

    def _cleanup(self) -> None:
        if self._dll is not None and self._handle is not None:
            # The handler was registered for the camera lifetime — it MUST be
            # removed regardless of streaming state. Freeing the CFUNCTYPE
            # thunk while the DLL still holds the pointer is a use-after-free.
            try:
                if self._cb is not None:
                    self._dll.ApiCam_RemoveEventHandler(self._handle, self._cb)
            except Exception:  # noqa: BLE001
                pass
            try:
                if self._handle.value:
                    self._dll.ApiCam_AbortAcquisition(self._handle)
            except Exception:  # noqa: BLE001
                pass
            try:
                if self._handle.value:
                    self._dll.ApiCam_CloseCamera(self._handle)
            except Exception:  # noqa: BLE001
                pass
        self._handle = None
        self._cb = None
        self._live_buf = None
        self._retired_buf = None
        with self._state_lock:
            self._pending = None
        self._streaming = False
        # Stamp the close HERE too, not only in disconnect(): every
        # failed-connect path funnels through _cleanup, and an immediate
        # reopen after a failed open is exactly the DLL wedge the
        # reopen-settle guard exists for (>4 min hang, hardware-seen).
        global _LAST_CLOSE_T
        _LAST_CLOSE_T = time.monotonic()
        if self._lib_initialized and self._dll is not None:
            try:
                self._dll.ApiLib_FinalizeLibrary()
            except Exception:  # noqa: BLE001
                pass
        self._lib_initialized = False

    # ------------------------------------------------------------------
    # Experimental decode overrides (SMARTCAM_PIXEL_FORMAT) — kept from the
    # old backend for full-res/bayer experiments.

    def _decode_format(self, raw: np.ndarray, spec: str) -> np.ndarray | None:
        spec = spec.strip().lower()
        if ":" in spec:
            kind, pattern = spec.split(":", 1)
        else:
            kind, pattern = spec, "rggb"
        if kind == "gray8":
            side = int(np.sqrt(raw.size))
            return cv2.cvtColor(raw[:side * side].reshape(side, side),
                                cv2.COLOR_GRAY2RGB)
        if kind == "nv12":
            w, h = (int(v) for v in pattern.split("x")) if "x" in pattern \
                else (self._width, self._height)
            return sd.decode_yuv420(raw, w, h)
        if kind == "bayer16":
            needed = 3840 * 2160 * 2
            if raw.size < needed:
                logger.warning("bayer16 needs %d bytes, buffer has %d", needed, raw.size)
                return None
            arr16 = raw[:needed].view("<u2").reshape(2160, 3840)
            shift = 4 if int(arr16.max()) <= 4095 else 8
            bayer8 = (arr16 >> shift).astype(np.uint8)
            codes = {"rggb": cv2.COLOR_BayerBG2RGB, "bggr": cv2.COLOR_BayerRG2RGB,
                     "grbg": cv2.COLOR_BayerGB2RGB, "gbrg": cv2.COLOR_BayerGR2RGB}
            return cv2.cvtColor(bayer8, codes.get(pattern, codes["rggb"]))
        if kind == "bayer8":
            needed = 1920 * 1080
            if raw.size < needed:
                logger.warning("bayer8 needs %d bytes, buffer has %d", needed, raw.size)
                return None
            bayer = raw[:needed].reshape(1080, 1920)
            codes = {"rggb": cv2.COLOR_BayerBG2RGB, "bggr": cv2.COLOR_BayerRG2RGB,
                     "grbg": cv2.COLOR_BayerGB2RGB, "gbrg": cv2.COLOR_BayerGR2RGB}
            return cv2.cvtColor(bayer, codes.get(pattern, codes["rggb"]))
        if kind == "yuy2":
            needed = 1920 * 1080 * 2
            if raw.size < needed:
                logger.warning("yuy2 needs %d bytes, buffer has %d", needed, raw.size)
                return None
            arr = raw[:needed].reshape(1080, 1920, 2)
            return cv2.cvtColor(arr, cv2.COLOR_YUV2RGB_YUYV)
        if kind == "packed12":
            w, h = 1920, 1080
            if "@" in pattern:
                pattern, res = pattern.rsplit("@", 1)
                w, h = (int(v) for v in res.split("x"))
            needed = w * h * 3 // 2
            if raw.size < needed:
                logger.warning("packed12 needs %d bytes, buffer has %d", needed, raw.size)
                return None
            bayer16 = self._unpack12(raw[:needed], w, h)
            bayer8 = (bayer16 >> 4).astype(np.uint8)
            codes = {"rggb": cv2.COLOR_BayerBG2RGB, "bggr": cv2.COLOR_BayerRG2RGB,
                     "grbg": cv2.COLOR_BayerGB2RGB, "gbrg": cv2.COLOR_BayerGR2RGB}
            return cv2.cvtColor(bayer8, codes.get(pattern, codes["rggb"]))
        logger.warning("Unsupported SMARTCAM_PIXEL_FORMAT %r", spec)
        return None

    @staticmethod
    def _unpack12(packed: np.ndarray, w: int, h: int) -> np.ndarray:
        """3 bytes -> 2 pixels, GenICam Mono12p convention."""
        b = packed.reshape(-1, 3).astype(np.uint16)
        p0 = b[:, 0] | ((b[:, 1] & 0x0F) << 8)
        p1 = (b[:, 1] >> 4) | (b[:, 2] << 4)
        pixels = np.empty(b.shape[0] * 2, dtype=np.uint16)
        pixels[0::2] = p0
        pixels[1::2] = p1
        return pixels.reshape(h, w)
