"""Settings: load %APPDATA%\\TALOS\\settings.json, deep-merge over bundled
defaults, atomic save (tmp + os.replace)."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

from talos.paths import get_settings_path, resource_path

logger = logging.getLogger(__name__)

_DEFAULTS_PATH = "resources/defaults/default_settings.json"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base; override wins."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_defaults() -> dict:
    """Bundled defaults (frozen-safe)."""
    try:
        with open(resource_path(_DEFAULTS_PATH), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not load bundled defaults: %s", exc)
        return {"_version": 3}


# Keys superseded by schema v3 (autofocus is µm-based now; the focus
# jog/step keys were never wired to anything). Dropped from user settings
# on load so stale values cannot shadow the new schema.
_DEAD_KEYS: dict[str, set[str]] = {
    "devices.focus": {"jog_speed", "step_size"},
    "autofocus": {"coarse_step", "fine_step", "span_steps", "max_speed"},
    # input.gamepad.invert_x/invert_y were shipped but read by NOTHING;
    # they are replaced by the per-stick invert_left_*/invert_right_* keys
    # (a single pair cannot express "invert only the Zolix stick").
    "input.gamepad": {"invert_x", "invert_y"},
}


def _normalize(data: dict) -> dict:
    """Detection-based migrations after the deep-merge (never version-
    gated): drop dead keys, run the per-row legacy migrations (v4, v5),
    then back-fill the objectives table from defaults LAST. Idempotent;
    user overrides survive."""
    dropped: list[str] = []
    for path, keys in _DEAD_KEYS.items():
        section: Any = data
        for part in path.split("."):
            section = section.get(part) if isinstance(section, dict) else None
            if section is None:
                break
        if not isinstance(section, dict):
            continue
        for key in keys:
            if key in section:
                del section[key]
                dropped.append(f"{path}.{key}")
    # WB "Once" is no longer a STORED state (it became a transient UI
    # action — a button) — a legacy stored "Once" would re-run a WB pass
    # at every connect / workspace switch, breaking a pre-adjusted color.
    cam = data.get("devices", {}).get("camera")
    if isinstance(cam, dict):
        if cam.get("white_balance") == "Once":
            cam["white_balance"] = "Off"
            dropped.append("devices.camera.white_balance Once→Off")
        scan = cam.get("scan")
        if isinstance(scan, dict) and scan.get("white_balance") == "Once":
            scan["white_balance"] = "Off"
            dropped.append("devices.camera.scan.white_balance Once→Off")
    default_rows = load_defaults().get("objectives", [])
    rows = data.get("objectives")
    if not isinstance(rows, list):
        rows = copy.deepcopy(default_rows)
        dropped.append("objectives (rebuilt from defaults)")
    else:
        for i, default_row in enumerate(default_rows):
            if i >= len(rows):
                rows.append(copy.deepcopy(default_row))
                dropped.append(f"objectives[{i}] back-filled")
    # v4: per-objective speed is now a MULTIPLIER (× the global bases
    # autofocus.coarse_speed_base_um_s / fine_speed_base_um_s). Rows still
    # carrying the legacy coarse_speed_um_s column: unmodified legacy
    # defaults (15/6/3/2/2 µm/s) are superseded by the new aggressive
    # defaults; customized values are preserved proportionally.
    legacy_speed = {5: 15.0, 10: 6.0, 20: 3.0, 50: 2.0, 100: 2.0}
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or "coarse_speed_um_s" not in row:
            continue
        old = float(row.get("coarse_speed_um_s", 0.0) or 0.0)
        if old > 0 and abs(old - legacy_speed.get(row.get("mag"), -1.0)) > 1e-9:
            row["speed_multiplier"] = round(old / 100.0, 4)
            dropped.append(f"objectives[{i}].coarse_speed_um_s → "
                           f"speed_multiplier {row['speed_multiplier']}")
        else:
            dropped.append(f"objectives[{i}].coarse_speed_um_s (superseded)")
        del row["coarse_speed_um_s"]
    # v5: the AF multiplier split — speed_multiplier → af_speed_multiplier.
    # MUST run BEFORE the per-row backfill: the new defaults carry
    # af_speed_multiplier, so a naive backfill-then-rename would shadow
    # the user's customized speed_multiplier and silently delete it.
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or "speed_multiplier" not in row:
            continue
        if "af_speed_multiplier" not in row:
            row["af_speed_multiplier"] = row["speed_multiplier"]
            dropped.append(f"objectives[{i}].speed_multiplier → "
                           f"af_speed_multiplier {row['speed_multiplier']}")
        else:
            dropped.append(f"objectives[{i}].speed_multiplier "
                           "(superseded — af_speed_multiplier present)")
        del row["speed_multiplier"]
    # Per-row backfill LAST: new columns (focus_manual_multiplier,
    # stage_speed_multiplier, px_um, …) land only on rows still missing
    # them, after the migrations have rewritten the legacy keys.
    for i, default_row in enumerate(default_rows):
        if i < len(rows) and isinstance(rows[i], dict):
            for key, value in default_row.items():
                rows[i].setdefault(key, copy.deepcopy(value))
    data["objectives"] = rows
    if dropped:
        logger.info("Settings normalized to v5: %s", ", ".join(dropped))
    return data


class Settings:
    """Typed-ish accessor over the merged settings dict."""

    def __init__(self, data: dict[str, Any], path: Path | None = None):
        self.data = data
        self.path = path or get_settings_path()

    # -- factory ---------------------------------------------------------

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or get_settings_path()
        data = load_defaults()
        if path.exists():
            try:
                with open(path, encoding="utf-8") as fh:
                    user = json.load(fh)
                data = _deep_merge(data, user)
            except (OSError, json.JSONDecodeError) as exc:
                logger.error("Could not read %s: %s — using defaults", path, exc)
        try:
            data = _normalize(data)
        except Exception as exc:  # noqa: BLE001
            # A hand-edited value of the wrong type (e.g. "auto" where a
            # float is expected) used to brick startup — the migrations
            # run on every launch, before any UI exists. Keep the user's
            # data (never silently fall back to defaults, that would
            # discard their whole configuration) and run without the
            # migration; the affected key fails loudly where it is used.
            logger.error("Settings migration failed (%s) — continuing with "
                         "the loaded values unmodified", exc)
        return cls(data, path=path)

    # -- accessors ---------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def section(self, key: str) -> dict:
        value = self.data.get(key)
        return value if isinstance(value, dict) else {}

    def device(self, key: str) -> dict:
        return self.section("devices").get(key, {})

    @property
    def sim(self) -> bool:
        return self.data.get("device_mode") == "sim"

    # -- persistence --------------------------------------------------------

    def update(self, key: str, value: Any) -> None:
        self.data[key] = value

    def save(self, path: Path | None = None) -> None:
        """Atomic save: write tmp in the same directory, then os.replace."""
        path = path or self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2, ensure_ascii=False)
        tmp.replace(path)
