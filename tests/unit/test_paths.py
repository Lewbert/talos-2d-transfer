"""Path helpers: the app-icon discovery (drop-in resources/icons/)."""

import talos.paths as paths


def test_app_icon_path_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "resource_path", lambda rel: tmp_path)
    assert paths.app_icon_path() is None


def test_app_icon_path_prefers_ico(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "resource_path", lambda rel: tmp_path)
    (tmp_path / "talos.png").write_bytes(b"x")
    assert paths.app_icon_path() == tmp_path / "talos.png"
    (tmp_path / "talos.ico").write_bytes(b"x")
    assert paths.app_icon_path() == tmp_path / "talos.ico"
