"""MCam (AxCam SDK) camera backend — the path ZEN itself uses.

Loads ``axcam64.dll`` from the ZEN install via ctypes (the Zeiss AxCam
SDK, headers: mcam.h / mcam_zei.h / mcam_zei_ex.h). This is the
FULL-QUALITY path: color processing, white balance, exposure/gain
parameter ranges, up to full sensor resolution.

v1 acquisition model: blocking single-frame captures (McammAcquisitionEx)
into a reusable raw buffer — perfect for autofocus/scan/verification;
the continuous-live path can be layered on later.

The DLL exposes the MCam API; ZEN's image quality implies the full
processing pipeline (color matrix, sharpener, references) is active by
default with color processing enabled.
"""

from __future__ import annotations

import ctypes
import logging
import os
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from talos.hal.base import Camera, DeviceConnectionError

logger = logging.getLogger(__name__)

_DLL_SEARCH = [
    r"C:\Program Files\Carl Zeiss\ZEN 2\ZEN 2 (blue edition)\axcam64.dll",
    r"C:\Program Files\Carl Zeiss\ZEN 2\ZEN 2 (blue edition)",
]


def _find_dll() -> str | None:
    for path in _DLL_SEARCH:
        if os.path.isfile(path):
            return path
    for root in (r"C:\Program Files\Carl Zeiss", r"C:\Program Files (x86)\Carl Zeiss"):
        if os.path.isdir(root):
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if fn.lower() == "axcam64.dll":
                        return os.path.join(dirpath, fn)
    return None


class _SMCAMINFO(ctypes.Structure):
    _fields_ = [("Revision", ctypes.c_long), ("SerienNummer", ctypes.c_long),
                ("Type", ctypes.c_long), ("Features", ctypes.c_long)]


class McamCamera(Camera):
    APPLIES_FLIP = True

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._dll = None
        self._lib_init = False
        self._cam_init = False
        self._buffer = None
        self._buffer_size = 0
        self._exposure_range = None
        self._gain_range = None
        self._width = 0
        self._height = 0
        self._bpp = 0

    @property
    def device_id(self) -> str:
        return "camera@mcam"

    # ------------------------------------------------------------------

    def connect(self) -> None:
        dll_path = _find_dll()
        if dll_path is None:
            raise DeviceConnectionError(
                "axcam64.dll not found (install ZEN blue edition — this is "
                "the full-quality ZEN camera path)")
        # Dependent DLLs live next to axcam64.dll.
        os.add_dll_directory(str(Path(dll_path).parent))
        try:
            self._dll = ctypes.WinDLL(dll_path)
        except OSError as exc:
            raise DeviceConnectionError(f"axcam64.dll load failed: {exc}") from exc
        self._setup_prototypes()
        try:
            self._call(self._dll.McammLibInit, "McammLibInit", ctypes.c_bool(False))
            self._lib_init = True
            count = int(self._dll.McamGetNumberofCameras())
            if count == 0:
                raise DeviceConnectionError("MCam found no cameras — is Labscope/ZEN closed?")
            self._call(self._dll.McammInit, "McammInit", 0)
            self._cam_init = True
            info = _SMCAMINFO()
            self._call(self._dll.McammInfo, "McammInfo", 0, ctypes.byref(info))
            logger.info("MCam camera: type=%d serial=%d features=%d",
                        info.Type, info.SerienNummer, info.Features)
            self._call(self._dll.McammEnableColorProcessing, "EnableColorProcessing",
                       0, ctypes.c_bool(True))
            self._read_sizes()
            self._read_parameter_ranges()
        except Exception:
            self.disconnect()
            raise
        self._connected = True
        logger.info("Connected via MCam (exposure range=%s, gain range=%s)",
                    self._exposure_range, self._gain_range)

    def disconnect(self) -> None:
        try:
            if self._cam_init and self._dll is not None:
                self._dll.McammStopContinuousAcquisition(0)
                self._dll.McammClose(0)
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._lib_init and self._dll is not None:
                self._dll.McammLibTerm()
        except Exception:  # noqa: BLE001
            pass
        self._cam_init = False
        self._lib_init = False
        self._dll = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._dll is not None

    def start(self) -> None:
        pass

    def stop(self) -> None:
        try:
            if self._cam_init and self._dll is not None:
                self._dll.McammStopContinuousAcquisition(0)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------

    def fetch(self, timeout_ms: float = 2000.0) -> np.ndarray | None:
        """Blocking single-frame acquisition (full quality + color)."""
        if self._buffer is None:
            return None
        progress = ctypes.CFUNCTYPE(
            wintypes.BOOL, ctypes.c_long, ctypes.c_long, ctypes.c_long,
            ctypes.c_void_p)(lambda done, total, status, user: True)
        try:
            rc = self._dll.McammAcquisitionEx(
                0, self._buffer, self._buffer_size, progress, None)
            if rc != 0:
                logger.debug("McammAcquisitionEx rc=%d", rc)
                return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCam acquisition failed: %s", exc)
            return None
        raw = np.frombuffer(self._buffer, dtype=np.uint8, count=self._buffer_size)
        return self._decode(raw)

    def _decode(self, raw: np.ndarray) -> np.ndarray:
        """Single frame egress for both the live fetch and the snapshot."""
        return self.apply_flip(self._decode_frame(raw))

    def _decode_frame(self, raw: np.ndarray) -> np.ndarray:
        """Decode the raw buffer. The processed image is typically 8-bit
        RGB (color processing on) at width×height; BGR16/Bayer variants
        are covered defensively."""
        need_rgb = self._width * self._height * 3
        if self._width > 0 and raw.size >= need_rgb:
            rgb = raw[:need_rgb].reshape(self._height, self._width, 3)
            # MCam processed images arrive as RGB (BGR for bpp=24?); the
            # histogram test below picks the correct channel order.
            return self._as_rgb(rgb)
        need_gray = self._width * self._height
        if self._width > 0 and raw.size >= need_gray:
            gray = raw[:need_gray].reshape(self._height, self._width)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        logger.warning("MCam: cannot decode buffer (%d bytes, %dx%d)",
                       raw.size, self._width, self._height)
        return None

    @staticmethod
    def _as_rgb(rgb: np.ndarray) -> np.ndarray:
        # The microscopy scene is not magenta-dominated; if the red/blue
        # means look swapped, flip the channel order.
        r_mean = float(rgb[:, :, 0].mean())
        b_mean = float(rgb[:, :, 2].mean())
        if b_mean > r_mean * 1.3:
            return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb

    def get_properties(self) -> dict[str, Any]:
        return {"backend": "mcam", "width": self._width, "height": self._height,
                "bpp": self._bpp,
                "exposure_range": list(self._exposure_range or ()),
                "gain_range": list(self._gain_range or ()),
                "exposure_us": None, "gain": None}

    def set_property(self, name: str, value: Any) -> None:
        if self.try_set_flip(name, value):
            return
        # The parameter setter export (MCammSet) exists in the DLL but its
        # prototype is not in the available headers — deferred; the MCam
        # defaults give full-quality color frames regardless.
        raise KeyError(f"MCam property {name!r} is not settable yet (defaults active)")

    def snapshot(self, path: Path, timeout_s: float = 15.0,
                 resolution: int | None = None,
                 burn: dict | None = None) -> Path:
        # resolution/burn accepted and ignored — kept only so the
        # signature matches the ABC (the UI passes all four args,
        # and a mismatch raised TypeError on every snapshot).
        frame = self.fetch(timeout_ms=timeout_s * 1000)
        if frame is None:
            raise DeviceConnectionError("MCam: no frame")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return path

    # ------------------------------------------------------------------

    def _read_sizes(self) -> None:
        w, h = ctypes.c_long(0), ctypes.c_long(0)
        try:
            if self._dll.McammGetCurrentDataSize(0, ctypes.byref(w), ctypes.byref(h)) == 0:
                self._width, self._height = int(w.value), int(h.value)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._bpp = int(self._dll.McammGetCurrentBitsPerPixel(0))
        except Exception:  # noqa: BLE001
            self._bpp = 0
        size = ctypes.c_long(0)
        try:
            if self._dll.McammGetMaxRawImageDataSize(0, ctypes.byref(size)) == 0:
                self._buffer_size = int(size.value)
        except Exception:  # noqa: BLE001
            pass
        if self._buffer_size <= 0:
            self._buffer_size = max(1, self._width * self._height * 3)
        self._buffer = ctypes.create_string_buffer(self._buffer_size)
        logger.info("MCam image: %dx%d bpp=%d buffer=%d bytes",
                    self._width, self._height, self._bpp, self._buffer_size)

    def _read_parameter_ranges(self) -> None:
        for attr, param_id in (("_exposure_range", 2), ("_gain_range", 3)):
            pmin, pmax = ctypes.c_long(0), ctypes.c_long(0)
            try:
                if self._dll.MCammGetParameterRange(
                        0, param_id, ctypes.byref(pmin), ctypes.byref(pmax)) == 0:
                    setattr(self, attr, (int(pmin.value), int(pmax.value)))
            except Exception:  # noqa: BLE001
                pass

    def _call(self, func, name: str, *args) -> int:
        rc = int(func(*args))
        if rc != 0:
            raise DeviceConnectionError(f"MCam {name} failed: rc={rc}")
        return rc

    def _setup_prototypes(self) -> None:
        dll = self._dll
        BOOL = ctypes.c_bool  # these headers are C++: bool is 1 byte
        dll.McammLibInit.argtypes = [BOOL]
        dll.McammLibInit.restype = ctypes.c_long
        dll.McammLibTerm.argtypes = []
        dll.McammLibTerm.restype = ctypes.c_long
        dll.McamGetNumberofCameras.argtypes = []
        dll.McamGetNumberofCameras.restype = ctypes.c_long
        dll.McammInit.argtypes = [ctypes.c_long]
        dll.McammInit.restype = ctypes.c_long
        dll.McammClose.argtypes = [ctypes.c_long]
        dll.McammClose.restype = None
        dll.McammInfo.argtypes = [ctypes.c_long, ctypes.POINTER(_SMCAMINFO)]
        dll.McammInfo.restype = ctypes.c_long
        dll.McammEnableColorProcessing.argtypes = [ctypes.c_long, BOOL]
        dll.McammEnableColorProcessing.restype = ctypes.c_long
        dll.McammGetCurrentDataSize.argtypes = [ctypes.c_long,
                                                ctypes.POINTER(ctypes.c_long),
                                                ctypes.POINTER(ctypes.c_long)]
        dll.McammGetCurrentDataSize.restype = ctypes.c_long
        dll.McammGetCurrentBitsPerPixel.argtypes = [ctypes.c_long]
        dll.McammGetCurrentBitsPerPixel.restype = ctypes.c_long
        dll.McammGetMaxRawImageDataSize.argtypes = [ctypes.c_long,
                                                    ctypes.POINTER(ctypes.c_long)]
        dll.McammGetMaxRawImageDataSize.restype = ctypes.c_long
        dll.MCammGetParameterRange.argtypes = [ctypes.c_long, ctypes.c_long,
                                               ctypes.POINTER(ctypes.c_long),
                                               ctypes.POINTER(ctypes.c_long)]
        dll.MCammGetParameterRange.restype = ctypes.c_long
        dll.McammStartContinuousAcquisition.argtypes = [ctypes.c_long, ctypes.c_int,
                                                        ctypes.c_void_p]
        dll.McammStartContinuousAcquisition.restype = ctypes.c_long
        dll.McammStopContinuousAcquisition.argtypes = [ctypes.c_long]
        dll.McammStopContinuousAcquisition.restype = ctypes.c_long
        dll.McammAcquisitionEx.argtypes = [ctypes.c_long, ctypes.c_void_p,
                                           ctypes.c_long, ctypes.c_void_p,
                                           ctypes.c_void_p]
        dll.McammAcquisitionEx.restype = ctypes.c_long
