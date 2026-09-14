"""Snapshot capture configuration + filename generation (pure, no Qt).

Naming = FILENAME + SUFFIX: ``timestamp`` = {name}_YYYYmmdd_HHMMSS.png
(with _2/_3 suffixes on collision); ``number`` = {name}_0001.png (a
zero-padded counter scanning the existing files). The name defaults to
"snap"; the save directory defaults to the user folder
``Pictures/TALOS/snapshots`` when ``capture.dir`` is empty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_NUMBER_RE = re.compile(rf"^(.*?)(\d+)$")


def default_snapshot_dir() -> Path:
    """The user-folder default: ~/Pictures/TALOS/snapshots."""
    return Path.home() / "Pictures" / "TALOS" / "snapshots"


@dataclass(frozen=True)
class CaptureConfig:
    directory: Path
    prefix: str
    pattern: str          # "timestamp" | "number"
    format: str           # "png" | "jpg"
    resolution: int       # 0 (4K) | 1 (1080p)
    burn_scale_bar: bool


def load_capture_config(settings,
                        default_dir: Path | None = None) -> CaptureConfig:
    cap = settings.section("capture")
    display = settings.section("display")
    directory = str(cap.get("dir", "") or "").strip()
    return CaptureConfig(
        directory=(Path(directory) if directory
                   else (default_dir or default_snapshot_dir())),
        prefix=str(cap.get("prefix", "") or ""),
        pattern=str(cap.get("pattern", "timestamp")),
        format=str(cap.get("format", "png")),
        resolution=int(cap.get("resolution", 0)),
        burn_scale_bar=bool(display.get("burn_scale_bar", False)),
    )


def snapshot_dir(cfg: CaptureConfig) -> Path:
    cfg.directory.mkdir(parents=True, exist_ok=True)
    return cfg.directory


def _stamp(now: datetime) -> str:
    return now.strftime("%Y%m%d_%H%M%S")


def next_snapshot_path(cfg: CaptureConfig,
                       now: datetime | None = None) -> Path:
    """The next snapshot filename for this config (no file is created):
    {filename}_{suffix}.{fmt} where the suffix is the timestamp or a
    zero-padded counter."""
    now = now or datetime.now()
    fmt = cfg.format if cfg.format in ("png", "jpg") else "png"
    name = (cfg.prefix or "snap").strip().rstrip("_") or "snap"
    if cfg.pattern == "number":
        highest = 0
        for existing in cfg.directory.glob(f"{name}_*.{fmt}"):
            match = _NUMBER_RE.match(existing.stem[len(name) + 1:])
            if match and match.group(2):
                highest = max(highest, int(match.group(2)))
        return cfg.directory / f"{name}_{highest + 1:04d}.{fmt}"
    path = cfg.directory / f"{name}_{_stamp(now)}.{fmt}"
    counter = 2
    while path.exists():
        path = cfg.directory / f"{name}_{_stamp(now)}_{counter}.{fmt}"
        counter += 1
    return path
