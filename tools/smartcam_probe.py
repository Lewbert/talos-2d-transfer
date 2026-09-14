"""SmartCamApi parameter probe — the Phase 2 keystone.

Dumps the camera's FULL parameter table (names via description, types,
ranges, enum values, current values) plus the function/event lists to
%APPDATA%\\TALOS\\benchmark\\smartcam_param_dump.json.

This explains why the old blind 0-150 ID sweep no-op'd: parameters are
addressed by the ParameterKey enum and their values are passed BY POINTER.

Usage (ZEN/Labscope must be closed, camera USB3 connected):
    python tools/smartcam_probe.py              # full dump
    python tools/smartcam_probe.py --dll "C:\\...\\SmartCamApi.dll"
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from talos.hal.devices.camera.smartcam_backend import (  # noqa: E402
    _ApiInformation,
    _ApiOptions,
    _find_dll,
)
from talos.hal.devices.camera.smartcam_params import (  # noqa: E402
    PARAM_TYPE_NAMES,
    MetadataType,
    ParamKey,
)
from talos.paths import get_appdata_dir  # noqa: E402

OUT = Path(get_appdata_dir()) / "benchmark"


def load_dll(path: str):
    try:
        return ctypes.WinDLL(path)
    except OSError:
        return ctypes.CDLL(path)


def setup_prototypes(dll) -> None:
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
    dll.ApiCam_GetEventList.restype = ctypes.c_int
    dll.ApiCam_GetEventList.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    dll.ApiCam_GetFunctionList.restype = ctypes.c_int
    dll.ApiCam_GetFunctionList.argtypes = [ctypes.c_void_p, ctypes.c_void_p]


def error_string(dll, code: int) -> str:
    name = ""
    from talos.hal.devices.camera.smartcam_params import ApiError
    try:
        name = ApiError(code).name
    except ValueError:
        pass
    try:
        buf = ctypes.create_string_buffer(256)
        dll.ApiLib_GetErrorDescription(code, buf, 256)
        return f"{name} ({buf.value.decode('ascii', 'replace')})"
    except Exception:
        return name or f"error {code}"


def get_str(dll, handle, fn_name, key) -> str:
    buf = ctypes.create_string_buffer(256)
    fn = getattr(dll, fn_name)
    rc = fn(handle, key, buf)
    if rc != 0:
        return ""
    return buf.value.decode("ascii", "replace")


def get_value(dll, handle, key: int, ptype: int):
    """Return (rc, python value) for the parameter's type."""
    if ptype == 0:  # boolean
        v = ctypes.c_byte(0)
        rc = dll.ApiCam_GetParameterValue(handle, key, ctypes.byref(v))
        return rc, bool(v.value)
    if ptype == 1 or ptype == 4:  # integer / enum
        v = ctypes.c_int(0)
        rc = dll.ApiCam_GetParameterValue(handle, key, ctypes.byref(v))
        return rc, int(v.value)
    if ptype == 2:  # double
        v = ctypes.c_double(0.0)
        rc = dll.ApiCam_GetParameterValue(handle, key, ctypes.byref(v))
        return rc, float(v.value)
    if ptype == 3:  # string
        buf = ctypes.create_string_buffer(256)
        rc = dll.ApiCam_GetParameterValue(handle, key, buf)
        return rc, buf.value.decode("ascii", "replace")
    if ptype == 5:  # IndexAndBoolean
        v = (ctypes.c_int * 2)(0, 0)
        rc = dll.ApiCam_GetParameterValue(handle, key, v)
        return rc, [int(v[0]), bool(v[1])]
    if ptype == 6:  # IndexAndInteger
        v = (ctypes.c_int * 2)(0, 0)
        rc = dll.ApiCam_GetParameterValue(handle, key, v)
        return rc, [int(v[0]), int(v[1])]
    if ptype == 7:  # IndexAndDouble
        v = (ctypes.c_int * 2)(0, 0)
        rc = dll.ApiCam_GetParameterValue(handle, key, v)
        return rc, [int(v[0]), int(v[1])]  # raw int view; doubles printed by caller
    if ptype == 8:  # IndexAndString
        v = (ctypes.c_int * 2)(0, 0)
        rc = dll.ApiCam_GetParameterValue(handle, key, v)
        return rc, [int(v[0]), int(v[1])]
    return -1, None


def get_meta_int(dll, handle, key: int, mtype: int) -> int | None:
    v = ctypes.c_int(0)
    rc = dll.ApiCam_GetParameterMetadata(handle, key, mtype, ctypes.byref(v))
    return int(v.value) if rc == 0 else None


def get_meta_double(dll, handle, key: int, mtype: int) -> float | None:
    v = ctypes.c_double(0.0)
    rc = dll.ApiCam_GetParameterMetadata(handle, key, mtype, ctypes.byref(v))
    return float(v.value) if rc == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description="SmartCamApi parameter dump")
    parser.add_argument("--dll", default=None, help="explicit SmartCamApi.dll path")
    args = parser.parse_args()

    dll_path = args.dll or _find_dll()
    if not dll_path:
        print("SmartCamApi.dll not found")
        return 1
    print(f"dll: {dll_path}")
    dll = load_dll(dll_path)
    setup_prototypes(dll)

    options = _ApiOptions(256, ctypes.sizeof(_ApiOptions), 0)
    rc = dll.ApiLib_InitializeLibrary(ctypes.byref(options))
    if rc != 0:
        print(f"InitializeLibrary: {error_string(dll, rc)}")
        return 1
    try:
        count = ctypes.c_int(0)
        rc = dll.ApiLib_GetCameraCount(ctypes.byref(count))
        print(f"cameras: {count.value} (rc={rc})")
        if rc != 0 or count.value == 0:
            return 1
        info = _ApiInformation()
        if dll.ApiLib_GetLibraryInformation(ctypes.byref(info)) == 0:
            print(f"api: v{info.ApiVersion} maxstr={info.MaxStringLength} "
                  f"maxparams={info.MaxParameterCount} imageheadersize={info.ImageHeaderSize}")
        handle = ctypes.c_void_p(0)
        rc = dll.ApiCam_OpenCamera(0, ctypes.byref(handle))
        if rc != 0:
            print(f"OpenCamera: {error_string(dll, rc)}")
            return 1
        try:
            print("settling 2 s...")
            time.sleep(2.0)

            # parameter list
            arr = (ctypes.c_int * 128)()
            rc = dll.ApiCam_GetParameterList(handle, arr)
            keys = [int(arr[i]) for i in range(128) if arr[i] != 0]
            print(f"parameter list (rc={rc}): {len(keys)} keys")

            dump: dict = {"dll": dll_path, "cameras": count.value, "params": []}
            for key in keys:
                ptype = get_meta_int(dll, handle, key, int(MetadataType.Type))
                pname = PARAM_TYPE_NAMES.get(ptype or -1, f"type{ptype}")
                name = ParamKey(key).name if key in iter(ParamKey) else f"key{key}"
                desc = get_str(dll, handle, "ApiCam_GetParameterDescription", key)
                access = get_meta_int(dll, handle, key, int(MetadataType.Access))
                row = {"key": key, "name": name, "type": ptype,
                       "type_name": pname, "description": desc, "access": access}
                if ptype in (1, 4):
                    row["min"] = get_meta_int(dll, handle, key, int(MetadataType.Minimum))
                    row["max"] = get_meta_int(dll, handle, key, int(MetadataType.Maximum))
                    row["inc"] = get_meta_int(dll, handle, key, int(MetadataType.Increment))
                    row["default"] = get_meta_int(dll, handle, key, int(MetadataType.Default))
                elif ptype == 2:
                    row["min"] = get_meta_double(dll, handle, key, int(MetadataType.Minimum))
                    row["max"] = get_meta_double(dll, handle, key, int(MetadataType.Maximum))
                    row["inc"] = get_meta_double(dll, handle, key, int(MetadataType.Increment))
                    row["default"] = get_meta_double(dll, handle, key, int(MetadataType.Default))
                if ptype == 4:
                    n = get_meta_int(dll, handle, key, int(MetadataType.Count)) or 0
                    row["enum"] = []
                    for i in range(n):
                        ev = ctypes.c_int(0)
                        ebuf = ctypes.create_string_buffer(128)
                        erc = dll.ApiCam_GetEnumParameterMetadata(handle, key, i,
                                                                 ctypes.byref(ev), ebuf)
                        if erc == 0:
                            row["enum"].append([int(ev.value),
                                                ebuf.value.decode("ascii", "replace")])
                rc_v, value = get_value(dll, handle, key, ptype)
                row["value_rc"] = rc_v
                row["value"] = value
                dump["params"].append(row)
                print(f"  {key:3d} {name:24s} {pname:13s} = {value!r}"
                      f"  [{desc[:60]}]")

            # function + event lists
            for label, fn_name in (("functions", "ApiCam_GetFunctionList"),
                                   ("events", "ApiCam_GetEventList")):
                lst = (ctypes.c_int * 32)()
                fn = getattr(dll, fn_name)
                erc = fn(handle, lst)
                vals = [int(lst[i]) for i in range(32) if lst[i] != 0]
                dump[label] = vals
                print(f"{label} (rc={erc}): {vals}")

            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "smartcam_param_dump.json").write_text(json.dumps(dump, indent=2))
            print(f"wrote {OUT / 'smartcam_param_dump.json'}")
        finally:
            dll.ApiCam_CloseCamera(handle)
    finally:
        dll.ApiLib_FinalizeLibrary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
