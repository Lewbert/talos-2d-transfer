"""Settings load/merge/atomic-save and migration mapping tests."""

import copy
import json

from talos.config import Settings, _deep_merge, _normalize, load_defaults


def test_deep_merge_nested():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 20, "z": 30}}
    merged = _deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3}
    # base must be untouched
    assert base == {"a": {"x": 1, "y": 2}, "b": 3}


def test_load_merges_user_over_defaults(tmp_path):
    user = {"devices": {"zolix": {"port": "COM9"}}, "ui": {"font_size": 12}}
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(user), encoding="utf-8")
    settings = Settings.load(path)
    assert settings.device("zolix")["port"] == "COM9"
    # untouched default survives the merge
    assert settings.device("zolix")["um_per_pulse_xy"] == 0.625
    assert settings.section("ui")["font_size"] == 12


def test_load_missing_file_uses_defaults(tmp_path):
    settings = Settings.load(tmp_path / "missing.json")
    assert settings.device("focus")["port"] == "COM10"


def test_save_is_atomic_and_roundtrips(tmp_path):
    settings = Settings.load(tmp_path / "s.json")
    settings.update("ui", {"font_size": 14})
    settings.save()
    saved = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert saved["ui"]["font_size"] == 14
    assert not (tmp_path / "s.json.tmp").exists()  # tmp replaced away


def test_load_corrupt_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    settings = Settings.load(path)
    assert settings.device("zolix")["port"] == "COM3"


def test_load_survives_a_malformed_value(tmp_path):
    """_normalize ran outside the try: ONE hand-edited value of the wrong
    type raised straight out of Settings.load() — before any UI existed —
    and the app could not start. The user's values are now kept as they
    are (never silently replaced by defaults, which would discard the
    whole configuration)."""
    path = tmp_path / "s.json"
    path.write_text(json.dumps({
        "objectives": [{"coarse_speed_um_s": "auto", "mag": 5}],
    }), encoding="utf-8")
    settings = Settings.load(path)          # must not raise
    rows = settings.get("objectives")
    assert rows[0]["coarse_speed_um_s"] == "auto"   # kept, not wiped
    assert settings.device("zolix")["port"] == "COM3"  # defaults intact


# ---------------------------------------------------------------------------
# Schema v3 normalization
# ---------------------------------------------------------------------------

def test_normalize_drops_dead_keys():
    data = copy.deepcopy(json.loads("""{
        "devices": {"focus": {"jog_speed": 200, "step_size": 10, "port": "COM10"}},
        "autofocus": {"coarse_step": 100, "fine_step": 10,
                      "span_steps": 6000, "max_speed": 500}
    }"""))
    out = _normalize(data)
    assert "jog_speed" not in out["devices"]["focus"]
    assert "step_size" not in out["devices"]["focus"]
    assert out["devices"]["focus"]["port"] == "COM10"  # survivors untouched
    assert set(out["autofocus"]) & {"coarse_step", "fine_step",
                                    "span_steps", "max_speed"} == set()


def test_normalize_wb_once_becomes_off():
    # "Once" is no longer a stored WB state (a transient button action) —
    # legacy stored "Once" values must not re-run a WB pass per apply
    data = {"devices": {"camera": {"white_balance": "Once",
                                   "scan": {"white_balance": "Once"}}}}
    out = _normalize(data)
    assert out["devices"]["camera"]["white_balance"] == "Off"
    assert out["devices"]["camera"]["scan"]["white_balance"] == "Off"
    # Continuous and Off pass through untouched
    data2 = {"devices": {"camera": {"white_balance": "Continuous",
                                    "scan": {"white_balance": "Off"}}}}
    out2 = _normalize(data2)
    assert out2["devices"]["camera"]["white_balance"] == "Continuous"
    assert out2["devices"]["camera"]["scan"]["white_balance"] == "Off"


def test_normalize_is_idempotent():
    data = {"devices": {"focus": {"jog_speed": 200}},
            "autofocus": {"coarse_step": 100}, "objectives": []}
    once = _normalize(data)
    twice = _normalize(once)
    assert once == twice


def test_normalize_backfills_missing_objective_rows():
    defaults = load_defaults()
    n_default = len(defaults["objectives"])
    data = {"objectives": [defaults["objectives"][0]]}  # only one row
    out = _normalize(data)
    assert len(out["objectives"]) == n_default
    assert out["objectives"][0] == defaults["objectives"][0]
    # non-list objectives rebuilt entirely
    out2 = _normalize({"objectives": {"oops": 1}})
    assert out2["objectives"] == defaults["objectives"]


def test_normalize_preserves_user_objective_overrides():
    data = {"objectives": [{"name": "My 10x", "window_um": 99.0}]}
    out = _normalize(data)
    assert out["objectives"][0]["name"] == "My 10x"
    assert out["objectives"][0]["window_um"] == 99.0
    # fields the user did not touch are filled from the default row
    assert out["objectives"][0]["mag"] == 5
    assert "na" in out["objectives"][0]


def test_load_applies_normalization(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({
        "devices": {"focus": {"jog_speed": 200}},
        "autofocus": {"coarse_step": 100, "quality_threshold": 0.5},
    }), encoding="utf-8")
    settings = Settings.load(path)
    assert "jog_speed" not in settings.device("focus")
    assert "coarse_step" not in settings.section("autofocus")
    assert settings.section("autofocus")["quality_threshold"] == 0.5
    # the schema marker: nothing gates on it, but it must match the
    # bundled defaults (it used to disagree with them AND the code)
    assert settings.get("_version") == 7
    assert len(settings.get("objectives")) == 5


# ---------------------------------------------------------------------------
# Schema v4 normalization: per-objective speed_multiplier
# ---------------------------------------------------------------------------

def test_normalize_supersedes_unmodified_legacy_speed():
    # The pre-v4 default speeds (15/6/3/2/2 µm/s) are superseded by the
    # new aggressive multipliers — an untouched installation must get the
    # speedup, not silently keep the old conservative table. (The
    # objectives list is index-aligned with the defaults, as on disk.)
    rows = copy.deepcopy(load_defaults()["objectives"])
    rows[1]["coarse_speed_um_s"] = 6.0        # 10× legacy default
    rows[1].pop("af_speed_multiplier", None)  # pre-v4 row shape
    out = _normalize({"objectives": rows})
    row = out["objectives"][1]
    assert "coarse_speed_um_s" not in row
    assert row["af_speed_multiplier"] == 0.25   # new default, not 6/150
    twice = _normalize(out)
    assert twice == out                          # idempotent


def test_normalize_preserves_custom_legacy_speed():
    # A user-customized speed (≠ any legacy default) is preserved
    # proportionally as a multiplier.
    rows = copy.deepcopy(load_defaults()["objectives"])
    rows[1]["coarse_speed_um_s"] = 12.0        # custom 10× speed
    rows[1].pop("af_speed_multiplier", None)
    out = _normalize({"objectives": rows})
    row = out["objectives"][1]
    assert "coarse_speed_um_s" not in row
    assert row["af_speed_multiplier"] == 0.12   # 12 µm/s ÷ 100


# ---------------------------------------------------------------------------
# Schema v5 normalization: af_speed_multiplier rename + new columns
# ---------------------------------------------------------------------------

def test_normalize_renames_speed_multiplier_preserving_custom_value():
    # A user-customized v4 speed_multiplier survives the v5 rename — the
    # rename runs BEFORE the defaults backfill, so the backfilled default
    # af_speed_multiplier cannot shadow it.
    rows = copy.deepcopy(load_defaults()["objectives"])
    rows[2].pop("af_speed_multiplier", None)
    rows[2]["speed_multiplier"] = 0.3333       # custom 20× v4 value
    out = _normalize({"objectives": rows})
    row = out["objectives"][2]
    assert "speed_multiplier" not in row
    assert row["af_speed_multiplier"] == 0.3333


def test_normalize_v5_rename_is_idempotent():
    rows = copy.deepcopy(load_defaults()["objectives"])
    rows[2].pop("af_speed_multiplier", None)
    rows[2]["speed_multiplier"] = 0.3333
    once = _normalize({"objectives": rows})
    twice = _normalize(copy.deepcopy(once))
    assert twice == once


def test_normalize_v5_backfills_new_columns():
    # Old-shaped rows gain the v5 columns from the defaults (rename first,
    # backfill last — already-migrated values untouched).
    data = {"objectives": [{"name": "My 5x"}]}
    out = _normalize(data)
    row = out["objectives"][0]
    assert row["af_speed_multiplier"] == 1.0
    assert row["focus_manual_multiplier"] == 1.0
    assert row["stage_speed_multiplier"] == 1.0
    assert row["px_um"] == 0.0
    assert row["z_offset_um"] == 0.0


def test_normalize_v5_prefers_existing_af_key():
    # Both keys present: af_speed_multiplier (the canonical v5 key) wins;
    # the legacy column is dropped.
    rows = copy.deepcopy(load_defaults()["objectives"])
    rows[1]["speed_multiplier"] = 0.99
    out = _normalize({"objectives": rows})
    row = out["objectives"][1]
    assert row["af_speed_multiplier"] == 0.25
    assert "speed_multiplier" not in row


def test_migrate_mapping_uses_deployed_keys(tmp_path, monkeypatch):
    import os

    from tools import migrate_settings

    appdata = tmp_path
    monkeypatch.setattr(migrate_settings, "_appdata", lambda: appdata)
    legacy = appdata / "TransferStageControl"
    legacy.mkdir(parents=True)
    legacy_settings = {
        "zolix": {"port": "COM3", "um_per_step_xy": 0.625, "um_per_step_r": 0.00125,
                  "slow_speed_pps": 500, "fast_speed_pps": 2000,
                  "slow_speed_r": 1000, "fast_speed_r": 10000,
                  "single_step_amount": 5, "single_step_r": 80,
                  "stop_mode": "immediate"},
        "focus": {"port": "COM10", "max_speed": 2000, "gamma": 2.2, "deadzone": 0.05},
    }
    (legacy / "settings.json").write_text(json.dumps(legacy_settings), encoding="utf-8")
    focus_cfg = {"port": "COM10", "max_speed": 2000, "jog_speed": 200, "step_size": 10}
    (appdata / "FocusControl").mkdir(parents=True)
    (appdata / "FocusControl" / "config.json").write_text(json.dumps(focus_cfg),
                                                          encoding="utf-8")

    mapping = migrate_settings.collect()
    assert mapping["zolix"]["single_step"] == 5
    assert mapping["zolix"]["um_per_pulse_xy"] == 0.625
    assert mapping["focus"]["jog_speed"] == 200
    assert mapping["focus"]["max_speed"] == 2000
