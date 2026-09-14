"""First-run migration: harvest TransferStageControl / FocusControl AppData
settings into %APPDATA%\\TALOS\\settings.json.

The bundled defaults already carry the same deployed values (schema v2);
this tool fills gaps if the user's AppData files differ (e.g. changed COM
ports or presets). Idempotent: existing TALOS settings are merged, never
overwritten wholesale.

Usage:
    python tools/migrate_settings.py            # apply
    python tools/migrate_settings.py --dry-run  # show what would change
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from talos.config import Settings  # noqa: E402
from talos.logging_setup import setup_logging  # noqa: E402

_SOURCES = [
    ("TransferStageControl", "settings.json"),
    ("FocusControl", "config.json"),
]


def _appdata() -> Path:
    import os

    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base)


def collect() -> dict:
    """Read the legacy config files and map them into TALOS schema v2."""
    mapping: dict = {}
    for app, filename in _SOURCES:
        path = _appdata() / app / filename
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARN: could not read {path}: {exc}")
            continue
        if app == "TransferStageControl":
            if "sigmakoki" in data:
                mapping.setdefault("sigmakoki", {}).update({
                    "port": data["sigmakoki"].get("port"),
                    "um_per_step_xy": data["sigmakoki"].get("um_per_step_xy"),
                    "um_per_step_z": data["sigmakoki"].get("um_per_step_z"),
                    "slow_speed_hz": data["sigmakoki"].get("slow_speed_hz"),
                    "fast_speed_hz": data["sigmakoki"].get("fast_speed_hz"),
                    "slow_speed_z": data["sigmakoki"].get("slow_speed_z"),
                    "fast_speed_z": data["sigmakoki"].get("fast_speed_z"),
                    "single_step": data["sigmakoki"].get("single_step_amount"),
                    "single_step_z": data["sigmakoki"].get("single_step_z"),
                })
            if "zolix" in data:
                mapping.setdefault("zolix", {}).update({
                    "port": data["zolix"].get("port"),
                    "um_per_pulse_xy": data["zolix"].get("um_per_step_xy"),
                    "um_per_pulse_r": data["zolix"].get("um_per_step_r"),
                    "slow_speed_pps": data["zolix"].get("slow_speed_pps"),
                    "fast_speed_pps": data["zolix"].get("fast_speed_pps"),
                    "slow_speed_r": data["zolix"].get("slow_speed_r"),
                    "fast_speed_r": data["zolix"].get("fast_speed_r"),
                    "single_step": data["zolix"].get("single_step_amount"),
                    "single_step_r": data["zolix"].get("single_step_r"),
                    "stop_mode": data["zolix"].get("stop_mode"),
                })
            if "focus" in data:
                mapping.setdefault("focus", {}).update({
                    "port": data["focus"].get("port"),
                    "max_speed": data["focus"].get("max_speed"),
                    "gamma": data["focus"].get("gamma"),
                    "deadzone": data["focus"].get("deadzone"),
                })
            if "yudian" in data:
                mapping.setdefault("yudian", {}).update({
                    "port": data["yudian"].get("port"),
                    "safety_lo_c": data["yudian"].get("safety_temp_lo_c"),
                    "safety_hi_c": data["yudian"].get("safety_temp_hi_c"),
                    "presets": data["yudian"].get("presets"),
                })
            if "gamepad" in data:
                mapping.setdefault("gamepad", {}).update(data["gamepad"])
        elif app == "FocusControl":
            mapping.setdefault("focus", {}).update({
                "port": data.get("port"),
                "max_speed": data.get("max_speed"),
                "jog_speed": data.get("jog_speed"),
                "step_size": data.get("step_size"),
                "slim_on": data.get("slim_on"),
                "slim_min": data.get("slim_min"),
                "slim_max": data.get("slim_max"),
                "awoff_on_exit": data.get("awoff_on_exit"),
            })
    # Drop None values so defaults survive.
    for section in mapping.values():
        for key in list(section):
            if section[key] is None:
                del section[key]
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate legacy AppData settings into TALOS")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=False)
    settings = Settings.load()
    mapping = collect()

    changes = []
    devices = settings.section("devices")
    for key, values in mapping.items():
        target = devices.get(key) or {}
        for field, value in values.items():
            if field not in target or target[field] != value:
                changes.append((key, field, target.get(field), value))
    if not changes:
        print("Nothing to migrate (settings already up to date).")
        return 0

    for key, field, old, new in changes:
        print(f"  devices.{key}.{field}: {old!r} -> {new!r}")
    if args.dry_run:
        print("Dry run — no changes written.")
        return 0

    for key, values in mapping.items():
        devices.setdefault(key, {}).update(values)
    settings.save()
    print("Migration applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
