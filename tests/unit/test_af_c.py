"""AfCStateMachine: pure transitions — drift re-peak, pause/resume, and
the scene-change exit (lamp off / nosepiece rotated)."""

from talos.cv.af_c import HOLD, PAUSED, REPEAK, STOPPED, AfCStateMachine


def make(**kwargs):
    return AfCStateMachine(**kwargs)


def test_no_baseline_no_actions():
    m = make()
    assert m.sample(0.0) is None


def test_healthy_scores_hold():
    m = make()
    m.set_baseline(1000.0)
    assert m.sample(980.0) is None
    assert m.sample(900.0) is None  # above hysteresis (750)
    assert m.state == HOLD


def test_scene_change_exits_on_single_collapse():
    """The user-requested guard: during HOLD nothing moves, so a violent
    single-sample collapse means the scene itself changed (lamp off,
    nosepiece rotated, sample removed) — AF-C must EXIT, not re-peak."""
    m = make(scene_change_ratio=0.6)
    m.set_baseline(1000.0)
    action = m.sample(350.0)  # 35% of baseline — below the 40% floor
    assert action == "stop:scene_changed"
    assert m.state == STOPPED
    # terminal: further samples ignored
    assert m.sample(1000.0) is None


def test_slow_degrade_triggers_refine_after_n_samples():
    m = make(hysteresis=0.25, degrade_samples=3)
    m.set_baseline(1000.0)
    assert m.sample(700.0) is None   # below 750 but above scene floor 400
    assert m.sample(700.0) is None
    assert m.state == HOLD
    assert m.sample(700.0) == "refine"
    assert m.state == REPEAK
    # samples ignored while re-peaking
    assert m.sample(700.0) is None


def test_recovery_resets_the_streak():
    m = make(hysteresis=0.25, degrade_samples=3)
    m.set_baseline(1000.0)
    m.sample(700.0)
    m.sample(700.0)
    assert m.sample(950.0) is None  # recovered
    assert m.sample(700.0) is None  # streak restarted: only 1/3
    assert m.sample(700.0) is None  # 2/3
    assert m.sample(700.0) == "refine"


def test_xy_and_manual_activity_pause():
    m = make()
    m.set_baseline(1000.0)
    assert m.activity("xy") == "pause"
    assert m.state == PAUSED
    # samples ignored while paused
    assert m.sample(100.0) is None
    assert m.state == PAUSED

    m2 = make()
    m2.set_baseline(1000.0)
    assert m2.activity("manual") == "pause"

    m3 = make(pause_on_xy_motion=False)
    m3.set_baseline(1000.0)
    assert m3.activity("xy") is None
    assert m3.state == HOLD


def test_activity_while_repeaking_is_ignored():
    m = make(hysteresis=0.25, degrade_samples=1)
    m.set_baseline(1000.0)
    assert m.sample(700.0) == "refine"
    assert m.activity("xy") is None  # REPEAK, not HOLD


def test_idle_after_pause_refocuses_by_default():
    m = make(resume_refocus=True)
    m.set_baseline(1000.0)
    m.activity("xy")
    assert m.idle_after_pause() == "resume_refocus"
    assert m.state == REPEAK
    # the re-peak completes and re-baselines
    assert m.job_done(True) is None
    assert m.state == HOLD


def test_idle_after_pause_hold_when_refocus_disabled():
    m = make(resume_refocus=False)
    m.set_baseline(1000.0)
    m.activity("manual")
    assert m.idle_after_pause() is None
    assert m.state == HOLD


def test_refine_failure_stops():
    m = make(hysteresis=0.25, degrade_samples=1)
    m.set_baseline(1000.0)
    m.sample(700.0)
    assert m.job_done(False) == "stop:refine_failed"
    assert m.state == STOPPED


def test_stop_is_terminal():
    m = make()
    m.set_baseline(1000.0)
    m.stop()
    assert m.state == STOPPED
    assert m.sample(0.0) is None
    assert m.activity("xy") is None
    assert m.idle_after_pause() is None
