"""Capture config + filename generation."""

from datetime import datetime
from pathlib import Path

from talos.capture import (
    CaptureConfig,
    default_snapshot_dir,
    load_capture_config,
    next_snapshot_path,
    snapshot_dir,
)


class FakeSettings:
    def __init__(self, capture=None, display=None):
        self.data = {"capture": capture or {}, "display": display or {}}

    def section(self, key):
        return self.data.setdefault(key, {})


def cfg(directory: Path, **overrides) -> CaptureConfig:
    values = dict(directory=directory, prefix="", pattern="timestamp",
                  format="png", resolution=0, burn_scale_bar=False)
    values.update(overrides)
    return CaptureConfig(**values)


def test_default_snapshot_dir_is_in_the_user_folder(monkeypatch):
    from pathlib import Path

    class _FakeHome:
        pass

    monkeypatch.setattr(Path, "home", lambda: Path("C:/Users/tester"))
    assert default_snapshot_dir() == Path(
        "C:/Users/tester/Pictures/TALOS/snapshots")


def test_load_capture_config_defaults(tmp_path):
    s = FakeSettings()
    loaded = load_capture_config(s, tmp_path / "snaps")
    assert loaded.directory == tmp_path / "snaps"  # explicit dir → used
    assert loaded.pattern == "timestamp"
    assert loaded.format == "png"
    assert loaded.resolution == 0  # 4K default


def test_load_capture_config_empty_dir_falls_back_to_user_folder(
        monkeypatch, tmp_path):
    from pathlib import Path

    monkeypatch.setattr(Path, "home", lambda: Path("C:/Users/tester"))
    loaded = load_capture_config(FakeSettings())
    assert loaded.directory == Path(
        "C:/Users/tester/Pictures/TALOS/snapshots")


def test_load_capture_config_explicit(tmp_path):
    s = FakeSettings(capture={"dir": str(tmp_path / "custom"), "prefix": "img",
                              "pattern": "number", "format": "jpg",
                              "resolution": 1},
                     display={"burn_scale_bar": True})
    loaded = load_capture_config(s)
    assert loaded.directory == tmp_path / "custom"
    assert loaded.prefix == "img"
    assert loaded.pattern == "number"
    assert loaded.format == "jpg"
    assert loaded.resolution == 1
    assert loaded.burn_scale_bar is True


def test_timestamp_pattern(tmp_path):
    c = cfg(tmp_path)
    now = datetime(2026, 9, 9, 14, 30, 5)
    assert next_snapshot_path(c, now) == tmp_path / "snap_20260909_143005.png"


def test_timestamp_collision_suffix(tmp_path):
    c = cfg(tmp_path)
    now = datetime(2026, 9, 9, 14, 30, 5)
    (tmp_path / "snap_20260909_143005.png").touch()
    (tmp_path / "snap_20260909_143005_2.png").touch()
    assert next_snapshot_path(c, now) == tmp_path / "snap_20260909_143005_3.png"


def test_number_pattern_scans_existing(tmp_path):
    c = cfg(tmp_path, pattern="number", prefix="img")
    (tmp_path / "img_0001.png").touch()
    (tmp_path / "img_0007.png").touch()
    (tmp_path / "other_0099.png").touch()  # different name — ignored
    assert next_snapshot_path(c) == tmp_path / "img_0008.png"


def test_filename_suffix_scheme(tmp_path):
    """The user-facing scheme: filename + suffix (timestamp or number)."""
    now = datetime(2026, 9, 9, 14, 30, 5)
    c = cfg(tmp_path, prefix="sample")
    assert next_snapshot_path(c, now) == tmp_path / "sample_20260909_143005.png"
    c2 = cfg(tmp_path, prefix="sample", pattern="number")
    assert next_snapshot_path(c2) == tmp_path / "sample_0001.png"
    # trailing underscores in the name are cleaned up
    c3 = cfg(tmp_path, prefix="sample_")
    assert next_snapshot_path(c3, now) == tmp_path / "sample_20260909_143005.png"


def test_number_pattern_first(tmp_path):
    c = cfg(tmp_path, pattern="number")
    assert next_snapshot_path(c) == tmp_path / "snap_0001.png"


def test_snapshot_dir_creates(tmp_path):
    assert snapshot_dir(cfg(tmp_path / "deep" / "nest")) == tmp_path / "deep" / "nest"


def test_format_jpg(tmp_path):
    c = cfg(tmp_path, format="jpg")
    now = datetime(2026, 9, 9, 14, 30, 5)
    assert next_snapshot_path(c, now).suffix == ".jpg"
