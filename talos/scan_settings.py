"""Scan settings and the scan output folder — pure, no Qt.

The same shape as :mod:`talos.capture` (which owns the snapshot folder and
filename rules) so the two output locations are described the same way and
sit next to each other in the settings file and in Preferences.

The scan's own section is deliberately small. What the operator changes
while looking down the eyepieces stays on the panel (area, origin,
direction, path, start axis, return); everything else — overlap, settle,
speed, backlash, which extra files a run writes — lives in Preferences,
because it is set once and then forgotten. ``SCAN_KEYS`` is the list of
what this module owns, and it is what the defaults file and the
migration both read.
"""

from __future__ import annotations

from pathlib import Path

#: Settings keys under the "scan" section that this module owns.
#: ``dir`` is not here: it is a path, not a number, and it carries the
#: "empty means the default folder" convention — see :func:`scan_directory`.
SCAN_KEYS = ("width_um", "height_um", "origin", "overlap", "path",
             "serpentine", "start_axis", "x_dir", "y_dir", "speed_pps",
             "settle_ms", "resolution", "backlash_um", "backlash_approach",
             "return_to_start", "export_mosaic", "export_candidates",
             "export_overview")

#: What a scan is when nobody has said otherwise.
DEFAULTS = {
    "width_um": 2000.0,
    "height_um": 1000.0,
    "origin": "centre",
    "overlap": 0.10,
    "path": "serpentine",
    "serpentine": True,
    "start_axis": "x",
    "x_dir": 1,
    "y_dir": 1,
    "speed_pps": 500.0,
    #: Mechanical quiet after a stop, before the exposure. Was 200 ms while
    #: the settle window began ~0.6 s after the motion ended (the old
    #: telemetry-sample wait); now it begins when the stage actually stops,
    #: so the same number is a longer real quiet time. Bench-tested at 100.
    "settle_ms": 100,
    #: Which sensor mode the tiles are captured in: 0 = 4K, 1 = 1080p, the
    #: same encoding as ``capture.resolution``. Equal to the live mode by
    #: default, i.e. no switch and today's behaviour.
    "resolution": 1,
    "backlash_um": 0.0,
    "backlash_approach": 1,
    "return_to_start": True,
    "export_mosaic": True,
    "export_candidates": True,
    "export_overview": True,
}


def default_scan_dir() -> Path:
    """Where scans go when no folder has been chosen: the TALOS folder in
    the user's pictures, beside the snapshots.

    It used to be ``~/Documents/TALOS_scans`` — a second convention for
    the same kind of output, in a different tree from the snapshots it
    sits next to in the UI. Both live under ``~/Pictures/TALOS`` now.
    """
    return Path.home() / "Pictures" / "TALOS" / "scans"


def scan_directory(settings) -> Path:
    """The scan output folder, creating nothing."""
    configured = str(settings.section("scan").get("dir", "") or "").strip()
    return Path(configured) if configured else default_scan_dir()


def load_scan_settings(settings) -> dict:
    """The scan section with the defaults filled in for anything missing.

    Every value is coerced, because the dict is fed straight into
    ``ScanParams`` and a string where a float belongs would fail at the
    first waypoint rather than at load.
    """
    section = settings.section("scan")
    out = dict(DEFAULTS)
    for key in SCAN_KEYS:
        stored = section.get(key)
        if stored is None:
            continue
        default = DEFAULTS[key]
        try:
            if isinstance(default, bool):
                out[key] = bool(stored)
            elif isinstance(default, int):
                out[key] = int(stored)
            elif isinstance(default, float):
                out[key] = float(stored)
            else:
                out[key] = str(stored)
        except (TypeError, ValueError):
            continue            # a hand-edited value falls back to the default
    return out


def save_scan_settings(settings, values: dict) -> None:
    section = settings.section("scan")
    for key, value in values.items():
        section[key] = value
    settings.save()


__all__ = ["DEFAULTS", "SCAN_KEYS", "default_scan_dir", "load_scan_settings",
           "save_scan_settings", "scan_directory"]
