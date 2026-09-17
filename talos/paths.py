"""Frozen-safe path helpers.

All user data lives under %APPDATA%\\TALOS (override via TALOS_APPDATA).
Bundled resources resolve via sys._MEIPASS when frozen.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from talos.scan_settings import default_scan_dir

_APP_NAME = "TALOS"


def get_appdata_dir() -> Path:
    override = os.environ.get("TALOS_APPDATA")
    if override:
        return Path(override)
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / _APP_NAME


def get_log_dir() -> Path:
    return get_appdata_dir() / "logs"


def get_settings_path() -> Path:
    return get_appdata_dir() / "settings.json"


def get_calibration_db_path() -> Path:
    return get_appdata_dir() / "calibration.db"


def get_scan_dir() -> Path:
    """Where the CLI benches put a scan.

    The APP's folder comes from ``scan.dir`` (see ``talos/scan_settings.py``)
    and defaults to ``~/Pictures/TALOS/scans``, beside the snapshots. This
    one exists for the tools, which take no settings object — and it now
    returns the SAME place, so a bench scan and an app scan are found in one
    directory instead of two.
    """
    return default_scan_dir()


def resource_path(rel: str) -> Path:
    """Resolve a bundled resource: sys._MEIPASS when frozen, repo root otherwise."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base) / rel
    return Path(__file__).resolve().parent.parent / rel


def app_icon_path() -> Path | None:
    """The app icon file (talos.ico preferred on Windows, else talos.png) —
    None while no icon has been dropped into resources/icons/."""
    icons = resource_path("resources/icons")
    for name in ("talos.ico", "talos.png"):
        candidate = icons / name
        if candidate.is_file():
            return candidate
    return None


def find_cti(search_paths: list[str] | None = None) -> str | None:
    """Validated GenTL producer discovery (frozen-safe).

    Order: TALOS_CTI env var → GENICAM_GENTL64/32_PATH entries (validated
    with isfile — the machine-level placeholder value must be filtered) →
    explicit search paths → bounded glob under the Zeiss install dirs.
    Returns the first candidate that exists as a file.
    """
    import glob

    candidates: list[str] = []
    candidates.append(os.environ.get("TALOS_CTI", ""))
    for var in ("GENICAM_GENTL64_PATH", "GENICAM_GENTL32_PATH"):
        for entry in os.environ.get(var, "").split(os.pathsep):
            if entry and os.path.isfile(entry):
                candidates.append(entry)
    for path in search_paths or []:
        p = Path(path)
        if p.is_file():
            candidates.append(str(p))
        elif p.is_dir():
            candidates += [str(f) for f in sorted(p.glob("*.cti"))]
    for root in (r"C:\Program Files\Zeiss", r"C:\Program Files (x86)\Zeiss"):
        try:
            candidates += sorted(glob.glob(os.path.join(root, "**", "*.cti"),
                                          recursive=True))[:5]
        except Exception:  # noqa: BLE001 - drive may not exist
            pass
    valid = [c for c in candidates if c and os.path.isfile(c)]
    # Prefer 64-bit producers (this is a 64-bit Python); a 32-bit .cti loads
    # but cannot enumerate any devices.
    valid.sort(key=lambda p: -1 if "64" in p.lower() else 0)
    return valid[0] if valid else None
