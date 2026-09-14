"""SmartCamCamera backend tests with a recording fake DLL (no hardware).

The fake mirrors the REAL DLL contract extracted from ZEN's wrapper:
SetParameterValue/GetParameterValue take POINTERS (byref), GetParameterList
returns a 0-terminated int array, and frames arrive via the ImageAcquired
event callback.
"""

from __future__ import annotations

import ctypes
import time

import cv2
import numpy as np
import pytest

from talos.hal.base import CommandRejectedError, DeviceConnectionError
from talos.hal.devices.camera.smartcam_backend import _EventData, SmartCamCamera


def rgb_to_nv12(rgb: np.ndarray) -> bytes:
    """Encode in the CAMERA's format: Y + interleaved (V, U) pairs."""
    h, w = rgb.shape[:2]
    yuv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2YUV)
    u = cv2.resize(yuv[:, :, 1], (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    v = cv2.resize(yuv[:, :, 2], (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    vu = np.empty((h // 2, w // 2, 2), dtype=np.uint8)
    vu[:, :, 0] = v
    vu[:, :, 1] = u
    return np.concatenate([yuv[:, :, 0].ravel(), vu.ravel()]).tobytes()


def write_ptr(ptr, value):
    """Write into a ctypes.byref() CArgObject or a direct array (scalar or array)."""
    obj = getattr(ptr, "_obj", ptr)
    if isinstance(obj, ctypes.Array):
        value = list(value)
        obj[:] = (value + [0] * len(obj))[: len(obj)]
    else:
        ctypes.cast(ptr, ctypes.POINTER(type(obj)))[0] = value


class FakeDll:
    """Recording stub: unknown functions return 0; handlers script the rest."""

    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []
        self.handlers = {}
        self.params = {32: 200.0, 26: 0, 31: 4.0, 5: 1, 42: 1, 48: 0,
                       44: 1, 39: 1, 29: 5500, 20: [3840, 2160]}  # CameraSize: [w, h]
        self.param_types = {32: 2, 26: 4, 31: 2, 5: 1, 42: 4, 48: 4, 44: 0,
                            39: 0, 29: 1}
        self.set_param_rc = 0
        self.sequence_rcs = [0]
        self._register_default_handlers()

    def __getattr__(self, name):
        def caller(*args):
            self.calls.append((name, args))
            if name in self.handlers:
                return self.handlers[name](*args)
            return 0
        return caller

    def _register_default_handlers(self):
        fake = self

        def get_camera_count(count_ptr):
            write_ptr(count_ptr, 1)
            return 0

        def open_camera(idx, handle_ptr):
            write_ptr(handle_ptr, 0x1234)
            return 0

        def get_parameter_list(handle, arr_ptr):
            keys = list(fake.param_types) + [0]
            write_ptr(arr_ptr, keys + [0] * (128 - len(keys)))
            return 0

        def get_parameter_metadata(handle, key, meta_type, value_ptr):
            write_ptr(value_ptr, fake.param_types.get(key, 0))
            return 0

        def get_parameter_value(handle, key, value_ptr):
            value = fake.params.get(key, 0)
            if isinstance(value, (list, tuple)):
                write_ptr(value_ptr, value)
            else:
                write_ptr(value_ptr, value)
            return 0

        def set_parameter_value(handle, key, value_ptr):
            if fake.set_param_rc:
                return fake.set_param_rc
            fake.params[key] = value_ptr._obj.value
            return 0

        def get_acquisition_buffer_size(handle, count, size_ptr):
            write_ptr(size_ptr, 64 * 64 * 3 // 2)
            return 0

        def get_sequence_image(handle):
            if not fake.sequence_rcs:
                return 20  # ImageNotReady
            rc = fake.sequence_rcs.pop(0)
            return rc

        self.handlers = {
            "ApiLib_GetCameraCount": get_camera_count,
            "ApiCam_OpenCamera": open_camera,
            "ApiCam_GetParameterList": get_parameter_list,
            "ApiCam_GetParameterMetadata": get_parameter_metadata,
            "ApiCam_GetParameterValue": get_parameter_value,
            "ApiCam_SetParameterValue": set_parameter_value,
            "ApiCam_GetAcquisitionBufferSize": get_acquisition_buffer_size,
            "ApiCam_GetSequenceImage": get_sequence_image,
        }

    def set_calls(self, name: str) -> list[tuple]:
        return [args for (n, args) in self.calls if n == name]


@pytest.fixture
def fake():
    return FakeDll()


@pytest.fixture
def cam(monkeypatch, fake):
    monkeypatch.setattr(ctypes, "WinDLL", lambda path: fake)
    monkeypatch.setattr(ctypes, "CDLL", lambda path: fake)
    instance = SmartCamCamera({"smartcam": {"apply_defaults": True,
                                            "apply_wb": False,
                                            "settle_s": 0.0,
                                            "enum_retries": 2}})
    instance._dll_path = "C:\\fake\\SmartCamApi.dll"
    return instance


def test_connect_enumerates_and_applies_defaults(cam, fake):
    cam.connect()
    assert cam.is_connected
    # param types enumerated via GetParameterList + GetParameterMetadata
    assert len(cam._param_types) == 9
    # ZEN defaults applied: exposure 200ms -> 20ms, color_mode 0 -> 1
    expo = [a for a in fake.set_calls("ApiCam_SetParameterValue") if a[1] == 32]
    colo = [a for a in fake.set_calls("ApiCam_SetParameterValue") if a[1] == 26]
    assert expo and abs(expo[0][2]._obj.value - 20.0) < 1e-6  # ms
    assert colo and colo[0][2]._obj.value == 1
    # unchanged defaults skipped (gain 4, white_balance 1, resolution 1...)
    assert not [a for a in fake.set_calls("ApiCam_SetParameterValue") if a[1] == 31]
    # geometry refreshed from CameraSize
    assert (cam._width, cam._height) == (3840, 2160)
    props = cam.get_properties()
    assert props["exposure_us"] == 20000.0
    assert props["color_mode"] == 1


def test_connect_no_camera_raises(cam, fake):
    def no_cameras(ptr):
        write_ptr(ptr, 0)
        return 0
    fake.handlers["ApiLib_GetCameraCount"] = no_cameras
    with pytest.raises(DeviceConnectionError, match="no cameras"):
        cam.connect()
    assert not cam.is_connected


def test_set_property_writes_and_reads_back(cam, fake):
    cam.connect()
    cam.set_property("gain", 8.0)
    assert fake.params[31] == 8.0
    cam.set_property("exposure_us", 5000.0)
    assert abs(fake.params[32] - 5.0) < 1e-6  # stored in ms


def test_set_property_rejected_raises(cam, fake):
    cam.connect()
    fake.set_param_rc = 17  # InvalidParameterType
    with pytest.raises(CommandRejectedError):
        cam.set_property("gain", 8.0)


def test_set_property_unknown_raises(cam, fake):
    cam.connect()
    with pytest.raises(KeyError):
        cam.set_property("warp_drive", 9)


def test_white_balance_strings_map_to_ints(cam, fake):
    cam.connect()
    cam.set_property("white_balance", "Continuous")
    assert fake.params[5] == 1
    cam.set_property("white_balance", "Off")
    assert fake.params[5] == 0


def _make_cam(monkeypatch, fake, extra):
    monkeypatch.setattr(ctypes, "WinDLL", lambda path: fake)
    monkeypatch.setattr(ctypes, "CDLL", lambda path: fake)
    instance = SmartCamCamera({**{"smartcam": {"apply_defaults": True,
                                               "apply_wb": False,
                                               "settle_s": 0.0,
                                               "enum_retries": 2}},
                               **extra})
    instance._dll_path = "C:\\fake\\SmartCamApi.dll"
    return instance


def test_reconnect_settles_before_open(monkeypatch, fake):
    # bench-verified: an immediate reopen wedges the DLL — the connect
    # must wait out the reopen-settle window after a close. The marker is
    # process-global: a FRESH instance after another instance's close
    # still settles (the loaded DLL is shared).
    from talos.hal.devices.camera import smartcam_backend as sb

    monkeypatch.setattr(ctypes, "WinDLL", lambda path: fake)
    monkeypatch.setattr(ctypes, "CDLL", lambda path: fake)
    cam = SmartCamCamera({"smartcam": {"apply_defaults": False,
                                       "settle_s": 0.0, "enum_retries": 2,
                                       "reopen_settle_s": 10.0}})
    cam._dll_path = "C:\\fake\\SmartCamApi.dll"
    now = [50.0]
    sleeps: list[float] = []
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(sb, "_LAST_CLOSE_T", 42.0)
    cam.connect()
    # 8 s of the 10 s window already elapsed → sleep the remaining 2 s
    assert sleeps and sleeps[0] == pytest.approx(2.0)


def test_first_connect_never_settles(monkeypatch, fake):
    monkeypatch.setattr(ctypes, "WinDLL", lambda path: fake)
    monkeypatch.setattr(ctypes, "CDLL", lambda path: fake)
    cam = SmartCamCamera({"smartcam": {"apply_defaults": False,
                                       "settle_s": 0.0, "enum_retries": 2,
                                       "reopen_settle_s": 10.0}})
    cam._dll_path = "C:\\fake\\SmartCamApi.dll"
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    cam.connect()
    # no prior close → no reopen settle (0.0 sleeps come from settle_s)
    assert 10.0 not in sleeps and all(s == 0.0 for s in sleeps)


def test_apply_defaults_respects_settings_white_balance(monkeypatch, fake):
    # the reconnect bug: _apply_defaults used to write AWB ON on EVERY
    # connect — the settings value must win
    cam = _make_cam(monkeypatch, fake, {"white_balance": "Off"})
    cam.connect()
    wb = [a for a in fake.set_calls("ApiCam_SetParameterValue") if a[1] == 5]
    assert wb and wb[0][2]._obj.value == 0  # AutoWhiteBalance = Off


def test_apply_defaults_applies_color_temperature(monkeypatch, fake):
    cam = _make_cam(monkeypatch, fake, {"color_temperature": 3200})
    cam.connect()
    ct = [a for a in fake.set_calls("ApiCam_SetParameterValue") if a[1] == 29]
    assert ct and ct[0][2]._obj.value == 3200
    # param 52 (the READ-ONLY WhiteBalance gain triple) is never written
    assert not [a for a in fake.set_calls("ApiCam_SetParameterValue")
                if a[1] == 52]


def test_resolution_set_stops_stream_and_refreshes(cam, fake):
    cam.connect()
    cam.start()
    assert cam._streaming
    cam.set_property("resolution", 0)  # 4K
    assert fake.params[42] == 0
    assert cam._streaming  # restarted after the switch
    stops = len([n for n, _ in fake.calls if n == "ApiCam_AbortAcquisition"])
    starts = len([n for n, _ in fake.calls if n == "ApiCam_StartContinuousAcquisition"])
    assert stops >= 1 and starts == 2  # initial start + restart after switch


def test_fetch_uses_event_callback_frames(cam, fake):
    cam.connect()
    cam._width, cam._height = 64, 64
    cam.start()
    blob = rgb_to_nv12(np.full((64, 64, 3), (200, 20, 20), dtype=np.uint8))
    arr = (ctypes.c_char * len(blob))()
    arr.raw = blob
    ev = _EventData(0, 0, ctypes.addressof(arr), len(blob))
    cam._on_event(5, ctypes.addressof(ev))  # ImageAcquired
    frame = cam.fetch(timeout_ms=500.0)
    assert frame is not None
    assert frame.shape == (64, 64, 3)
    assert frame[:, :, 0].mean() > frame[:, :, 2].mean()  # red survives
    cam.stop()
    assert fake.set_calls("ApiCam_StartContinuousAcquisition")


def test_snapshot_uses_sequence_and_decodes(cam, fake, tmp_path):
    cam.connect()
    cam._width, cam._height = 64, 64
    cam.start()
    blob = rgb_to_nv12(np.full((64, 64, 3), (20, 20, 200), dtype=np.uint8))
    arr = (ctypes.c_char * len(blob))()
    arr.raw = blob

    def start_sequence(handle, count, size, buf):
        ctypes.memmove(buf, arr, len(blob))
        return 0

    fake.handlers["ApiCam_StartSequenceAcquisition"] = start_sequence
    fake.sequence_rcs = [20, 0]  # not ready, then ready
    path = cam.snapshot(tmp_path / "snap.png")
    assert path.is_file()
    img = cv2.imread(str(path))
    assert img.shape == (64, 64, 3)
    assert img[:, :, 0].mean() > img[:, :, 2].mean()  # blue -> BGR: blue channel high
    assert cam._streaming  # live restarted after snap


def test_disconnect_releases_everything(cam, fake):
    cam.connect()
    cam.disconnect()
    assert not cam.is_connected
    assert any(n == "ApiCam_CloseCamera" for n, _ in fake.calls)
    assert any(n == "ApiLib_FinalizeLibrary" for n, _ in fake.calls)


# ----------------------------------------------------------------------
# Audit-driven additions (2026-09-06): every failure/timeout/edge branch
# ----------------------------------------------------------------------

def test_retry_loop_reinitializes_library(cam, fake):
    calls = {"count": 0}

    def flaky_count(ptr):
        calls["count"] += 1
        write_ptr(ptr, 1 if calls["count"] >= 2 else 0)
        return 0

    fake.handlers["ApiLib_GetCameraCount"] = flaky_count
    cam.connect()
    assert cam.is_connected
    init_calls = len([n for n, _ in fake.calls if n == "ApiLib_InitializeLibrary"])
    fin_calls = len([n for n, _ in fake.calls if n == "ApiLib_FinalizeLibrary"])
    assert init_calls == 2  # re-initialized after the first empty poll
    assert fin_calls == 1


def test_disconnect_removes_event_handler_without_streaming(cam, fake):
    cam.connect()  # handler registered; stream never started
    cam.disconnect()
    assert any(n == "ApiCam_RemoveEventHandler" for n, _ in fake.calls)
    assert any(n == "ApiCam_CloseCamera" for n, _ in fake.calls)


def test_add_event_handler_failure_cleans_up(cam, fake):
    fake.handlers["ApiCam_AddEventHandler"] = lambda h, cb: 1
    with pytest.raises(DeviceConnectionError):
        cam.connect()
    assert not cam.is_connected
    assert any(n == "ApiCam_CloseCamera" for n, _ in fake.calls)
    assert any(n == "ApiLib_FinalizeLibrary" for n, _ in fake.calls)


def test_snapshot_4k_switch_requeries_size_and_restores(cam, fake, tmp_path):
    """snapshot(resolution=0) from 1080p: the mode switches for the
    capture, the buffer size is re-queried at the NEW geometry, and the
    live mode + stream come back afterwards."""
    cam.connect()
    cam._width, cam._height = 1920, 1080  # pretend the live mode is 1080p
    cam.start()
    fake.params[42] = 1
    base_get = fake.handlers["ApiCam_GetParameterValue"]

    def camera_size(handle, key, value_ptr):
        if key == 20:  # CameraSize follows the current resolution mode
            write_ptr(value_ptr, [64, 64] if fake.params.get(42) == 0
                      else [1920, 1080])
            return 0
        return base_get(handle, key, value_ptr)

    fake.handlers["ApiCam_GetParameterValue"] = camera_size
    blob = rgb_to_nv12(np.full((64, 64, 3), (20, 20, 200), dtype=np.uint8))
    arr = (ctypes.c_char * len(blob))()
    arr.raw = blob

    def start_sequence(handle, count, size, buf):
        ctypes.memmove(buf, arr, len(blob))
        return 0

    fake.handlers["ApiCam_StartSequenceAcquisition"] = start_sequence

    path = cam.snapshot(tmp_path / "snap4k.png", resolution=0)
    img = cv2.imread(str(path))
    assert img.shape == (64, 64, 3)
    assert fake.params[42] == 1  # the live mode is restored
    assert cam._streaming
    assert cam._width == 1920 and cam._height == 1080


def test_gain_write_skips_readback_but_exposure_keeps_it(cam, fake):
    """The auto-gain loop writes gain repeatedly — each readback query
    stalls one frame delivery (~330 ms, hardware-measured), so gain
    writes must NOT query back. Exposure keeps the verification."""
    from talos.hal.devices.camera.smartcam_params import PROPERTIES

    cam.connect()
    gain_key = int(PROPERTIES["gain"]["key"])
    exp_key = int(PROPERTIES["exposure_us"]["key"])
    fake.calls.clear()
    cam.set_property("gain", 5.0)
    gains = [a for a in fake.set_calls("ApiCam_GetParameterValue")
             if a[1] == gain_key]
    assert gains == []
    cam.set_property("exposure_us", 30000.0)
    exps = [a for a in fake.set_calls("ApiCam_GetParameterValue")
            if a[1] == exp_key]
    assert exps


def test_snapshot_timeout_aborts_and_restarts(cam, fake, tmp_path):
    cam.connect()
    cam._width, cam._height = 64, 64
    cam.start()
    fake.sequence_rcs = [20]  # never ready
    with pytest.raises(DeviceConnectionError, match="timed out"):
        cam.snapshot(tmp_path / "snap.png", timeout_s=0.3)
    assert cam._streaming  # live restarted
    aborts = len([n for n, _ in fake.calls if n == "ApiCam_AbortAcquisition"])
    assert aborts >= 2  # stop-for-snap + timeout abort


def test_on_event_ignores_foreign_events_and_bad_sizes(cam, fake):
    cam.connect()
    cam._width, cam._height = 64, 64
    cam.start()
    blob = rgb_to_nv12(np.full((64, 64, 3), (200, 20, 20), dtype=np.uint8))
    arr = (ctypes.c_char * len(blob))()
    arr.raw = blob
    # non-ImageAcquired event
    cam._on_event(3, ctypes.addressof(_EventData()))
    assert cam._pending is None
    # wrong size
    cam._on_event(5, ctypes.addressof(_EventData(0, 0, ctypes.addressof(arr),
                                                 len(blob) + 1)))
    assert cam._pending is None
    # correct size delivered
    cam._on_event(5, ctypes.addressof(_EventData(0, 0, ctypes.addressof(arr),
                                                 len(blob))))
    assert cam._pending is not None


def test_fetch_timeout_returns_none(cam, fake):
    cam.connect()
    cam._width, cam._height = 64, 64
    assert cam.fetch(timeout_ms=20.0) is None  # not started
    cam.start()
    assert cam.fetch(timeout_ms=20.0) is None  # no event fires


def test_decode_gray_fallback_and_garbage(cam, fake):
    cam.connect()
    cam._width, cam._height = 64, 64
    gray = np.full(64 * 64, 100, dtype=np.uint8)
    out = cam._decode(gray)
    assert out is not None and out.shape == (64, 64, 3)
    assert cam._decode(np.zeros(10, dtype=np.uint8)) is None


def test_white_balance_once_and_invalid_string(cam, fake):
    cam.connect()
    cam.set_property("white_balance", "Once")
    assert fake.params[5] == 2
    with pytest.raises(ValueError, match="Blinking"):
        cam.set_property("white_balance", "Blinking")


def test_get_properties_fps_from_event_times(cam, fake):
    cam.connect()
    cam._width, cam._height = 64, 64
    cam.start()
    cam._event_times.extend([10.0 + i * 0.05 for i in range(10)])
    props = cam.get_properties()
    assert abs(props["fps"] - 20.0) < 1.0


def test_get_properties_caches_the_parameter_queries(cam, fake):
    """Each parameter query stalls frame delivery ~330 ms (hardware-
    measured), so five of them per call froze the live view for ~1.6 s on
    every panel refresh. The readback is cached until a WRITE invalidates
    it; fps stays live (it comes from event timestamps)."""
    cam.connect()
    cam._width, cam._height = 64, 64
    cam.start()
    calls = []
    original = cam._get_param_raw

    def _counted(key, kind="int"):
        calls.append(key)
        return original(key, kind)

    cam._get_param_raw = _counted  # type: ignore[method-assign]
    first = cam.get_properties()
    n_first = len(calls)
    assert n_first > 0, "the first call must query the camera"
    second = cam.get_properties()
    assert len(calls) == n_first, "the second call must hit the cache"
    assert second["exposure_us"] == first["exposure_us"]

    # a write invalidates the cache
    cam.set_property("exposure_us", 12345.0)
    cam.get_properties()
    assert len(calls) > n_first


def test_packed12_decode_override(cam, fake):
    # GenICam Mono12p: 3 bytes = 2 pixels
    import numpy as np
    w, h = 8, 8
    packed = np.zeros(w * h * 3 // 2, dtype=np.uint8)
    packed[0], packed[1], packed[2] = 0xFF, 0x0F, 0x00  # 0xFFF, 0x000
    out = cam._decode_format(packed, "packed12:rggb@8x8")
    assert out is not None and out.shape == (h, w, 3)
    out2 = cam._decode_format(np.full(64, 50, dtype=np.uint8), "gray8")
    assert out2 is not None and out2.shape == (8, 8, 3)


def test_snapshot_before_connect_raises(cam, tmp_path):
    with pytest.raises(DeviceConnectionError):
        cam.snapshot(tmp_path / "x.png")
