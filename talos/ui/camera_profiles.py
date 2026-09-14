"""Per-workspace camera profiles (pure).

Navigation = the live profile: devices.camera.* (auto-gain allowed, ON
by default — auto-EXPOSURE changes the framerate, auto-gain does not).
Sample Finding = devices.camera.scan.*: MANUAL exposure/gain/WB, no
auto anything (detection needs a stable, reproducible brightness).
Switching workspaces applies the profile to the camera; edits persist
to the active profile's section.
"""

from __future__ import annotations

from dataclasses import dataclass

#: the canonical camera keys a profile applies (exposure/gain/WB only —
#: resolution stays 1080p live and is a capture setting, not a profile)
_PROFILE_KEYS = ("exposure_us", "gain", "white_balance", "color_temperature")


@dataclass(frozen=True)
class CameraProfile:
    exposure_us: float
    gain: float
    white_balance: str
    color_temperature: float = 5500.0
    auto_gain: bool = False
    auto_gain_target: float = 120.0
    allow_auto: bool = True


def nav_profile(settings) -> CameraProfile:
    cfg = settings.device("camera")
    wb = str(cfg.get("white_balance", "Off"))
    return CameraProfile(
        exposure_us=float(cfg.get("exposure_us", 40000)),
        gain=float(cfg.get("gain", 20.0)),
        # the stored state is ON/OFF only — "Once" is a transient UI
        # action, never persisted (a stored "Once" would re-run a WB
        # pass on every profile application)
        white_balance=wb if wb == "Continuous" else "Off",
        color_temperature=float(cfg.get("color_temperature", 5500.0)),
        auto_gain=bool(cfg.get("auto_gain", True)),
        auto_gain_target=float(cfg.get("auto_gain_target", 120.0)),
        allow_auto=True)


def scan_profile(settings) -> CameraProfile:
    cfg = settings.device("camera").get("scan") or {}
    nav_cfg = settings.device("camera")
    return CameraProfile(
        exposure_us=float(cfg.get("exposure_us", 40000)),
        gain=float(cfg.get("gain", 4.0)),
        # the scan workspace forces WB OFF — detection needs a stable,
        # reproducible color (the pre-scan adjustment is the Once button)
        white_balance="Off",
        color_temperature=float(cfg.get("color_temperature", 5500.0)),
        auto_gain=False,      # no auto settings on the scan workspace
        # The target is NOT exposed in the scan panel (auto-gain is off
        # there) but the Gain-once button still uses it — inherit the nav
        # value unless the scan section overrides it, so the button can
        # never silently use a number nobody can see or edit.
        auto_gain_target=float(cfg.get("auto_gain_target",
                                       nav_cfg.get("auto_gain_target", 120.0))),
        allow_auto=False)


def profile_as_props(profile: CameraProfile) -> dict:
    """The profile's canonical prop subset ({exposure_us, gain, wb})."""
    return {key: getattr(profile, key) for key in _PROFILE_KEYS}


def profile_diff(current_props: dict, profile: CameraProfile) \
        -> list[tuple[str, object]]:
    """The profile values that differ from the camera's current
    canonical props (exposure/gain/WB only — unchanged keys are skipped
    so the camera is not re-written on every workspace switch)."""
    changes: list[tuple[str, object]] = []
    for key in _PROFILE_KEYS:
        current = current_props.get(key)
        target = getattr(profile, key)
        if current is None:
            changes.append((key, target))
        elif key == "white_balance":
            if str(current) != str(target):
                changes.append((key, target))
        elif float(current) != float(target):
            changes.append((key, target))
    return changes


def update_profile_setting(settings, workspace: str, key: str, value) -> None:
    """Persist a live edit into the active profile's settings section."""
    if workspace == "scan":
        section = settings.device("camera").setdefault("scan", {})
    else:
        section = settings.device("camera")
    section[key] = value
    settings.save()
