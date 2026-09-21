"""Canonical camera-property layer.

The Camera ABC promises ``exposure_us``/``gain``/``white_balance`` but the
backends disagree (mmcore uses ``exposure_ms``, directshow ``exposure``,
harvesters ``ExposureTime``). CameraProxy routes BOTH directions through
this module so the UI speaks ONLY canonical names:

- get direction: ``normalize_props(backend, props)``
- set direction: ``map_property(backend, name, value) -> (native, value)``
  with unit conversion (canonical µs -> mmcore/directshow ms).

Canonical keys: exposure_us (float, microseconds), gain (float, x),
white_balance (str "Continuous" | "Once" | "Off"), color_temperature
(int, kelvin — the AWB target while auto, the fixed WB when AWB is Off;
the 208c has no hardware manual gain pair), color_mode (int 0/1),
fps (float).
"""

from __future__ import annotations

from typing import Any

# canonical name -> backend -> (native name, value scale)
_VALUE_MAP: dict[str, dict[str, tuple[str, float]]] = {
    "exposure_us": {"smartcam": ("exposure_us", 1.0), "harvesters": ("ExposureTime", 1.0),
                    "mmcore": ("exposure_ms", 0.001), "mcam": ("exposure_us", 1.0),
                    "directshow": ("exposure", 0.001), "manual": ("exposure_us", 1.0),
                    "sim": ("exposure_us", 1.0)},
    "gain": {"smartcam": ("gain", 1.0), "harvesters": ("Gain", 1.0),
             "mmcore": ("gain", 1.0), "mcam": ("gain", 1.0),
             "directshow": ("gain", 1.0), "manual": ("gain", 1.0),
             "sim": ("gain", 1.0)},
    "white_balance": {"smartcam": ("white_balance", 1.0),
                      "harvesters": ("BalanceWhiteAuto", 1.0),
                      "mmcore": ("white_balance", 1.0), "mcam": ("white_balance", 1.0),
                      "directshow": ("white_balance", 1.0),
                      "manual": ("white_balance", 1.0), "sim": ("white_balance", 1.0)},
    "color_temperature": {"smartcam": ("color_temperature", 1.0),
                          "manual": ("color_temperature", 1.0),
                          "sim": ("color_temperature", 1.0)},
    "color_mode": {"smartcam": ("color_mode", 1.0), "sim": ("color_mode", 1.0)},
    # The live sensor mode (0 = 4K, 1 = 1080p), for the backends that can
    # change it WHILE RUNNING — which is what a scan that captures at a
    # resolution other than the live view's needs. Everywhere else a
    # resolution is a SNAPSHOT argument, so those backends are absent on
    # purpose: the KeyError is how the caller hears "no" instead of a
    # switch that silently did nothing.
    "resolution": {"smartcam": ("resolution", 1.0), "sim": ("resolution", 1.0)},
    # software-level, shared by EVERY backend (the 180° decode-time
    # rotation) — a missing row raises KeyError on the first UI write
    "flip": {"smartcam": ("flip", 1.0), "harvesters": ("flip", 1.0),
             "mmcore": ("flip", 1.0), "mcam": ("flip", 1.0),
             "directshow": ("flip", 1.0), "manual": ("flip", 1.0),
             "sim": ("flip", 1.0)},
}

# native name -> (canonical name, transform to canonical units)
_TRANSFORM = {
    "ExposureTime": ("exposure_us", lambda v: float(v)),  # GenTL node is already us
    "exposure_ms": ("exposure_us", lambda v: float(v) * 1000.0),
    "exposure": ("exposure_us", lambda v: float(v) * 1000.0),  # directshow: ms
}

_CANONICAL_KEYS = ("exposure_us", "gain", "white_balance", "color_temperature",
                   "color_mode", "fps", "flip")

_WB_NAMES = {0: "Off", 1: "Continuous", 2: "Once"}


def normalize_props(backend: str, props: dict[str, Any]) -> dict[str, Any]:
    """Translate a backend's get_properties() dict into canonical keys.

    Unknown keys are passed through untouched; known aliases are renamed and
    unit-converted; smartcam's integer white_balance becomes the canonical
    string; missing canonical keys are omitted.
    """
    out: dict[str, Any] = {}
    for name, value in (props or {}).items():
        if name in _TRANSFORM:
            canon, conv = _TRANSFORM[name]
            try:
                out[canon] = conv(value)
            except (TypeError, ValueError):
                pass
        elif name in _CANONICAL_KEYS:
            if name == "white_balance" and isinstance(value, (int, float)) \
                    and backend == "smartcam":
                out[name] = _WB_NAMES.get(int(value), str(value))
            else:
                out[name] = value
        else:
            out[name] = value
    if "fps" not in out and "measured_fps" in (props or {}):
        out["fps"] = props["measured_fps"]
    return out


def map_property(backend: str, name: str, value: Any) -> tuple[str, Any]:
    """Translate a canonical property set into the backend's native
    (name, value) pair, with unit conversion. Raises KeyError for
    unsupported canonical names or backends."""
    entry = _VALUE_MAP.get(name)
    if entry is None:
        raise KeyError(f"Unknown canonical camera property: {name}")
    spec = entry.get(backend)
    if spec is None:
        raise KeyError(f"Camera property {name!r} is not supported by backend {backend!r}")
    native, scale = spec
    if scale != 1.0 and isinstance(value, (int, float)):
        value = float(value) * scale
    return native, value
