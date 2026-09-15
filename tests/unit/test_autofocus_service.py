"""AutofocusService: AF-S job launch/arm-cancel, the input-abort hook
(user motion/snapshot submissions abort a running job), and backlash-
calibration storage."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QEventLoop, QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication

from talos.cv.autofocus import AutofocusResult
from talos.cv.autofocus_service import AutofocusService
from talos.cv.frame_slot import LatestFrameSlot
from tests.testing.sim_images import synthetic_flake_image


@pytest.fixture(scope="session", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def flush(ms: int = 50) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


_AF_DEFAULTS = {
    "metric": "tenengrad", "af_exposure_us": 20000.0,
    "quality_threshold": 0.3, "timeout_s": 180.0,
    "freshness_ms": 400.0, "interp_max_gap_ms": 300.0,
    "coarse_poll_s": 0.02, "fine_wait_timeout_s": 2.0,
    "settle_frames": 1, "overshoot_margin_um": 3.0,
    "peak_prominence": 0.15, "fail_on_edge_peak": True,
    "coarse_speed_base_um_s": 100.0, "camera_fps_estimate": 15,
    "coarse_metric": "brenner_k", "coarse_metric_k": 8, "coarse_bin": 2,
    "default_roi_norm": [0.1667, 0.1667, 0.6667, 0.6667],
    "hill_ratio": 0.6, "hill_early_stop_samples": 2,
    "lock_samples_required": 4, "lock_score_frac": 0.85,
    "stationary_points": 5, "stop_accel_sps2": 20000,
    "stop_latency_s": 0.05, "stop_safety_steps": 10,
}

_OBJECTIVES = [
    {"name": f"LMPlanFL {m}x", "mag": m, "na": na, "dof_um": dof,
     "window_um": win, "coarse_step_um": cs, "fine_step_um": fs,
     "speed_multiplier": mult, "backlash_um": 0.0, "backlash_measured_at": None}
    for m, na, dof, win, cs, fs, mult in (
        (5, 0.15, 28.0, 150.0, 5.0, 1.0, 1.0),
        (10, 0.30, 7.0, 60.0, 2.0, 0.4, 0.25),
        (20, 0.40, 4.0, 30.0, 1.0, 0.2, 0.20),
        (50, 0.55, 2.0, 20.0, 0.4, 0.2, 0.10),
        (100, 0.85, 0.8, 12.0, 0.2, 0.2, 0.05),
    )
]


class FakeSettings:
    def __init__(self):
        self.data = {"autofocus": dict(_AF_DEFAULTS),
                     "objectives": _OBJECTIVES,
                     "devices": {"focus": {"um_per_step": 0.2,
                                           "landing_speed": 50}}}
        self.saved = 0

    def device(self, key):
        return self.data["devices"].get(key, {})

    def section(self, key):
        return self.data.get(key, {})

    def get(self, key, default=None):
        return self.data.get(key, default)

    def save(self):
        self.saved += 1


class FakeState:
    def __init__(self):
        self.objective = 0
        self.mode = "MANUAL"
        self.modes: list[str] = []

    def set_mode(self, mode):
        self.mode = mode
        self.modes.append(mode)


class FakeFocusProxy(QObject):
    sig_af_progress = Signal(float, int, float, float)
    sig_af_curve_secondary = Signal(float, float)
    sig_af_done = Signal(object)
    sig_af_log = Signal(str)
    sig_cal_progress = Signal(float, str)
    sig_cal_done = Signal(object)
    sig_stop_requested = Signal()

    def __init__(self):
        super().__init__()
        self.aborts = 0
        self.abort_reasons: list = []

    def request_abort(self, reason=None):
        self.aborts += 1
        self.abort_reasons.append(reason)


class FakeManager(QObject):
    sig_job_submitted = Signal(str, str)
    sig_device_state = Signal(str, dict)
    sig_proxy_replaced = Signal(str, object)

    def __init__(self):
        super().__init__()
        self.submits: list[tuple] = []
        self.camera_submits: list[tuple] = []
        self._camera_props = {"exposure_us": 40000.0}
        self._focus_position = 1234
        self._focus = FakeFocusProxy()

    @property
    def camera_props(self):
        return dict(self._camera_props)

    @property
    def focus_position(self):
        return self._focus_position

    def device(self, key):
        return self._focus

    def submit(self, device_key, method_name, *args, priority=0):
        self.submits.append((device_key, method_name, args, priority))
        return len(self.submits)

    def submit_camera(self, method_name, *args):
        self.camera_submits.append((method_name, args))
        return len(self.camera_submits)


@pytest.fixture
def rig():
    manager = FakeManager()
    settings = FakeSettings()
    state = FakeState()
    slot = LatestFrameSlot()
    service = AutofocusService(manager, settings, state, slot)
    return manager, settings, state, slot, service


def test_measurement_region_comes_from_settings(rig):
    """The AF measurement region is a single persisted preference
    (autofocus.default_roi_norm) — the UI writes it, autofocus reads it,
    and None means the WHOLE frame. The panel used to pass its own copy,
    which is how it could claim "Full frame" while autofocus measured the
    centre crop."""
    _manager, settings, _state, _slot, service = rig
    settings.section("autofocus")["default_roi_norm"] = [0.2, 0.3, 0.4, 0.5]
    assert service._default_roi() == (0.2, 0.3, 0.4, 0.5)
    settings.section("autofocus")["default_roi_norm"] = None
    assert service._default_roi() is None


def test_reconnect_rebinds_the_focus_proxy(rig):
    """A device reconnect replaces the focus proxy: the service must follow
    it, or a finished run reports into the retired object (dropped) and the
    service stays busy/AUTOFOCUS forever."""
    manager, _settings, state, _slot, service = rig
    service.start_af_s()
    flush(500)  # the arm timer submits the job
    assert service.busy

    new_proxy = FakeFocusProxy()
    manager.sig_proxy_replaced.emit("focus", new_proxy)
    assert service._focus is new_proxy

    # the OLD proxy is disconnected…
    manager._focus.sig_af_done.emit(_result())
    flush()
    assert service.busy and state.mode == "AUTOFOCUS"
    # …and the new one drives the completion
    new_proxy.sig_af_done.emit(_result())
    flush()
    assert not service.busy and state.mode == "MANUAL"


def test_reconnect_aborts_an_inflight_run(rig):
    """The job's completion would die with its proxy — a reconnect must
    abort the run instead of stranding the service."""
    manager, _settings, state, _slot, service = rig
    service.start_af_s()
    flush(500)
    assert service.busy
    manager.sig_device_state.emit("focus", {"connecting": True})
    assert manager._focus.aborts >= 1
    assert "focus device reconnecting" in manager._focus.abort_reasons
    # new starts are refused while the device is reconnecting
    assert service._focus_connected is False
    assert service.busy  # still waiting for the aborted run's completion


def _result(success=True, baseline=1000.0, message="ok"):
    return AutofocusResult(best_position=42, best_score=baseline,
                           success=success, message=message, phase="done")


def test_af_s_launch_never_touches_the_camera(rig):
    """User rule: autofocus consumes the camera output — it must NEVER
    alter exposure/gain. (An earlier build ran an exposure dance; the
    user saw the brightness jump when AF started.)"""
    manager, _settings, state, _slot, service = rig
    finished = []
    service.sig_af_finished.connect(lambda r: finished.append(r))
    service.start_af_s()
    assert state.mode == "AUTOFOCUS"
    flush(500)  # the 350 ms arm timer
    assert manager.camera_submits == []  # no camera commands, ever
    jobs = [s for s in manager.submits if s[1] == "autofocus"]
    assert len(jobs) == 1
    (request,) = jobs[0][2]
    assert request.center_steps == 1234
    assert request.config.mode == "AF_S"
    # job completes → mode back, finished emitted
    manager._focus.sig_af_done.emit(_result())
    flush()
    assert manager.camera_submits == []
    assert state.mode == "MANUAL"
    assert len(finished) == 1 and finished[0].success


def test_af_s_abort_returns_to_manual(rig):
    manager, _settings, state, _slot, service = rig
    finished = []
    service.sig_af_finished.connect(lambda r: finished.append(r))
    service.start_af_s()
    flush(500)
    manager._focus.sig_af_done.emit(_result(success=False, baseline=0.0,
                                            message="aborted by user"))
    flush()
    assert manager.camera_submits == []
    assert state.mode == "MANUAL"
    assert finished and not finished[0].success


def test_calibrate_backlash_stores_result(rig):
    manager, settings, state, _slot, service = rig
    service.calibrate_backlash()
    jobs = [s for s in manager.submits if s[1] == "backlash_calibrate"]
    assert len(jobs) == 1
    (center, cfg), = jobs[0][2]
    assert center == 1234
    from talos.cv.backlash_cal import BacklashResult
    manager._focus.sig_cal_done.emit(BacklashResult(
        backlash_steps=6, backlash_um=1.2, success=True, message="ok"))
    flush()
    assert settings.saved == 1
    # stored as a MECHANISM property of the focus axis, not per objective
    assert settings.data["devices"]["focus"]["backlash_um"] == 1.2
    assert settings.data["devices"]["focus"]["backlash_measured_at"]
    assert state.mode == "MANUAL"


def test_busy_ignores_second_start(rig):
    manager, _settings, _state, _slot, service = rig
    service.start_af_s()
    flush(500)
    n_jobs = len([s for s in manager.submits if s[1] == "autofocus"])
    service.start_af_s()
    flush(500)
    assert len([s for s in manager.submits if s[1] == "autofocus"]) == n_jobs


def test_abort_forwards_to_focus_proxy(rig):
    manager, _settings, _state, _slot, service = rig
    service.start_af_s()
    flush(500)
    service.abort()
    assert manager._focus.aborts == 1
    assert manager._focus.abort_reasons == ["aborted by user"]


def test_af_done_does_not_clobber_scan_mode(rig):
    """A scan started while AF ran must not be knocked back to MANUAL
    when the AF job finishes (audit-found clobber)."""
    manager, _settings, state, _slot, service = rig
    service.start_af_s()
    flush(500)
    state.mode = "SCAN"  # the scan owns the axes now
    manager._focus.sig_af_done.emit(_result(success=False, baseline=0.0,
                                            message="whatever"))
    flush()
    assert state.mode == "SCAN"


def test_arm_cancel_does_not_clobber_scan_mode(rig):
    manager, _settings, state, _slot, service = rig
    service.start_af_s()
    state.mode = "SCAN"
    service.abort()  # the pending-arm cancel path
    assert state.mode == "SCAN"


def test_af_s_refused_when_focus_not_connected(rig):
    """A dead focus worker must refuse starts (an armed job into a dead
    event loop strands the service in AUTOFOCUS/busy forever)."""
    manager, _settings, state, _slot, service = rig
    finished = []
    service.sig_af_finished.connect(lambda r: finished.append(r))
    manager.sig_device_state.emit("focus", {"connected": False})
    service.start_af_s()
    assert finished and not finished[0].success
    assert "not connected" in finished[0].message
    assert manager.submits == []  # nothing armed
    assert state.mode == "MANUAL"


# ---------------------------------------------------------------------------
# The input-abort hook: user motion/snapshot submissions abort a running job
# ---------------------------------------------------------------------------

def test_af_s_aborts_on_manual_focus_submit(rig):
    manager, _settings, _state, _slot, service = rig
    service.start_af_s()
    flush(500)  # the job is armed + submitted
    manager.sig_job_submitted.emit("focus", "set_speed")  # focus jog
    assert manager._focus.aborts == 1
    assert manager._focus.abort_reasons == ["aborted by manual focus input"]


def test_af_s_aborts_on_xy_motion_submit(rig):
    manager, _settings, _state, _slot, service = rig
    service.start_af_s()
    flush(500)
    manager.sig_job_submitted.emit("zolix", "move_rel_um")   # XY jog / go-to
    manager.sig_job_submitted.emit("sigmakoki", "home")      # stage ZERO
    assert manager._focus.aborts == 2
    assert manager._focus.abort_reasons == [
        "aborted by stage motion input", "aborted by stage motion input"]


def test_af_s_aborts_on_snapshot_submit(rig):
    manager, _settings, _state, _slot, service = rig
    service.start_af_s()
    flush(500)
    manager.sig_job_submitted.emit("camera", "snapshot")
    assert manager._focus.aborts == 1
    assert manager._focus.abort_reasons == ["aborted by snapshot input"]


def test_af_s_ignores_non_triggering_submits(rig):
    manager, _settings, _state, _slot, service = rig
    service.start_af_s()
    flush(500)
    # camera property tweaks (exposure/gain/WB) do NOT abort (user decision)
    manager.sig_job_submitted.emit("camera", "set_property")
    # the yudian setpoint does not abort
    manager.sig_job_submitted.emit("yudian", "set_sv")
    # the service's own jobs do not abort themselves
    manager.sig_job_submitted.emit("focus", "autofocus")
    manager.sig_job_submitted.emit("focus", "backlash_calibrate")
    # the jog release does not abort (its set_speed sibling already did)
    manager.sig_job_submitted.emit("focus", "stop")
    assert manager._focus.aborts == 0


def test_af_s_no_abort_when_idle(rig):
    manager, _settings, _state, _slot, service = rig
    manager.sig_job_submitted.emit("focus", "set_speed")
    manager.sig_job_submitted.emit("zolix", "move_rel_um")
    manager.sig_job_submitted.emit("camera", "snapshot")
    assert manager._focus.aborts == 0


def test_input_during_arm_window_cancels_af_s(rig):
    """An input arriving in the 350 ms arm window kills the pending arm —
    the run never starts and the mode returns to MANUAL."""
    manager, _settings, state, _slot, service = rig
    finished = []
    service.sig_af_finished.connect(lambda r: finished.append(r))
    service.start_af_s()
    assert state.mode == "AUTOFOCUS"
    manager.sig_job_submitted.emit("focus", "set_speed")  # no flush first
    assert service._job_kind == ""
    assert state.mode == "MANUAL"
    assert finished and finished[0].aborted
    assert finished[0].message == "aborted by manual focus input"
    flush(600)  # the arm timer would have fired — it must not submit
    jobs = [s for s in manager.submits if s[1] == "autofocus"]
    assert jobs == []


def test_stop_all_during_arm_window_cancels_af_s(rig):
    """Esc / STOP ALL must also cancel a pending arm (via the focus
    proxy's sig_stop_requested)."""
    manager, _settings, state, _slot, service = rig
    finished = []
    service.sig_af_finished.connect(lambda r: finished.append(r))
    service.start_af_s()
    manager._focus.sig_stop_requested.emit()  # stop_all → enqueue_stop
    assert service._job_kind == ""
    assert state.mode == "MANUAL"
    assert finished and finished[0].aborted
    flush(600)
    assert [s for s in manager.submits if s[1] == "autofocus"] == []


def test_abort_during_arm_window_is_idempotent(rig):
    manager, _settings, state, _slot, service = rig
    finished = []
    service.sig_af_finished.connect(lambda r: finished.append(r))
    service.start_af_s()
    service.abort()
    service.abort()  # the second call must not emit a second finished
    assert len(finished) == 1
    assert finished[0].aborted
    assert state.mode == "MANUAL"
    flush(600)
    assert [s for s in manager.submits if s[1] == "autofocus"] == []


# ---------------------------------------------------------------------------
# Config plumbing + the input-abort hook
# ---------------------------------------------------------------------------

def test_af_s_config_speed_table(rig):
    manager, _settings, _state, _slot, service = rig
    service.start_af_s()
    flush(500)
    (request,) = [s for s in manager.submits if s[1] == "autofocus"][0][2]
    cfg = request.config
    assert cfg.strategy == "adaptive"  # hardcoded after the strategy-unwiring
    # 5× row: multiplier 1.0 × 100 µm/s ÷ 0.2 = 500 steps/s (aggressive)
    assert cfg.coarse_speed == 500
    assert cfg.max_speed == 500
    assert cfg.hill_v_cap == 500           # far zone = coarse speed
    assert cfg.hill_v_min == 83
    assert cfg.coarse_metric == "brenner_k"
    assert cfg.coarse_metric_k == 8
    assert cfg.coarse_bin == 2


def test_probe_and_guard_keys_pass_through(rig):
    manager, settings, _state, _slot, service = rig
    settings.data["autofocus"].update({
        "probe_step_steps": 25, "probe_peak_ratio": 0.2,
        "probe_min_slope": 0.05, "guard_samples": 5,
        "guard_drop_ratio": 0.25, "coarse_early_stop_samples": 3,
        "stage2_retries": 4,
        "probe_score_floor_ratio": 0.5, "guard_fit_samples": 8,
        "guard_sigma": 2.5, "coarse_curv_window": 7,
        "probe_curv_in": 0.001, "probe_curv_out": 0.0002,
        "coarse_curv_stop": 0.4, "coarse_curv_vertex": -0.3,
    })
    service.start_af_s()
    flush(500)
    (request,) = [s for s in manager.submits if s[1] == "autofocus"][0][2]
    cfg = request.config
    assert cfg.probe_step_steps == 25
    assert cfg.probe_peak_ratio == 0.2
    assert cfg.probe_min_slope == 0.05
    assert cfg.guard_samples == 5
    assert cfg.guard_drop_ratio == 0.25
    assert cfg.coarse_early_stop_samples == 3
    assert cfg.stage2_retries == 4
    assert cfg.probe_score_floor_ratio == 0.5
    assert cfg.guard_fit_samples == 8
    assert cfg.guard_sigma == 2.5
    assert cfg.coarse_curv_window == 7
    # the derivative thresholds flow through build_config (af_cfg
    # overrides win over the DOF-computed defaults)
    assert cfg.probe_curv_in == 0.001
    assert cfg.probe_curv_out == 0.0002
    assert cfg.coarse_curv_stop == 0.4
    assert cfg.coarse_curv_vertex == -0.3


def test_probe_and_guard_keys_default_when_absent(rig):
    manager, settings, _state, _slot, service = rig
    service.start_af_s()
    flush(500)
    (request,) = [s for s in manager.submits if s[1] == "autofocus"][0][2]
    cfg = request.config
    assert cfg.probe_step_steps == 0
    assert cfg.probe_peak_ratio == 0.15
    assert cfg.probe_min_slope == 0.10
    assert cfg.guard_samples == 4
    assert cfg.guard_drop_ratio == 0.15
    assert cfg.coarse_early_stop_samples == 2
    assert cfg.stage2_retries == 1
    assert cfg.probe_score_floor_ratio == 0.3
    assert cfg.guard_fit_samples == 6
    assert cfg.guard_sigma == 2.0
    assert cfg.coarse_curv_window == 5
    # the derivative thresholds are AUTO-computed from DOF at 5×
    # (σ ≈ DOF/3 = 47 steps, h = 500/15 ≈ 33 steps → t ≈ 0.71 →
    # the ladder ≈ 0.24; the vertex floor and probe thresholds follow)
    assert 0.1 < cfg.coarse_curv_stop < 0.4
    assert cfg.coarse_curv_vertex < 0
    assert cfg.probe_curv_in > 0 and cfg.probe_curv_out > 0


def test_default_roi_applied_when_no_user_roi(rig):
    manager, _settings, _state, _slot, service = rig
    service.start_af_s()
    flush(500)
    (request,) = [s for s in manager.submits if s[1] == "autofocus"][0][2]
    assert request.roi_norm == (0.1667, 0.1667, 0.6667, 0.6667)
    assert request.to_config().roi_norm == request.roi_norm


def test_user_roi_overrides_default(rig):
    manager, _settings, _state, _slot, service = rig
    user_roi = (0.25, 0.25, 0.5, 0.5)
    service.start_af_s(roi_norm=user_roi)
    flush(500)
    (request,) = [s for s in manager.submits if s[1] == "autofocus"][0][2]
    assert request.roi_norm == user_roi


class NoFocusManager(FakeManager):
    """The manager skipped the focus proxy entirely (a hand-edited
    ``devices.focus.enabled = false``): device() returns None and submit()
    refuses with -1, exactly like InstrumentManager."""

    def device(self, key):
        return None

    def submit(self, device_key, method_name, *args, priority=0):
        self.submits.append((device_key, method_name, args, priority))
        return -1


def test_service_survives_a_missing_focus_proxy():
    """Regression (2026-09-16): with the focus device disabled the service
    connected its signals on a None proxy — AttributeError at startup, which
    app.py does not catch, so the whole app died before the window existed."""
    manager = NoFocusManager()
    service = AutofocusService(manager, FakeSettings(), FakeState(),
                               LatestFrameSlot())
    assert service.busy is False
    results = []
    service.sig_af_finished.connect(results.append)
    service.start_af_s()
    assert len(results) == 1 and "not connected" in results[0].message
    assert service.busy is False
    service.calibrate_backlash()          # refused, not armed
    assert service.busy is False
    service.abort()                       # must not raise
    service.shutdown()


def test_calibration_refusal_does_not_strand_the_service():
    """Regression (2026-09-16): calibrate_backlash ignored submit()'s -1
    return while _arm guarded it, so _job_kind stayed "cal": the service was
    busy forever, the mode stayed AUTOFOCUS, and the calibration buttons
    stayed disabled waiting for a completion that could not arrive."""
    manager = NoFocusManager()
    service = AutofocusService(manager, FakeSettings(), FakeState(),
                               LatestFrameSlot())
    service._focus_connected = True       # the device is there, then refuses
    results = []
    service.sig_cal_finished.connect(results.append)
    service.calibrate_backlash()
    assert service.busy is False
    assert results and not results[0].success
    assert "disabled" in results[0].message
    assert service._state.mode == "MANUAL", "the mode must not stay AUTOFOCUS"


def test_planned_kwargs_is_the_one_mapping_the_readouts_show():
    """Regression (2026-09-16): the UI readouts re-derived the multiplier
    with "af_speed_multiplier or speed_multiplier or 1.0" and a hardcoded
    50 st/s fine floor, while build_config falls back to (na_min/na)² — a row
    without an explicit multiplier was advertised as an unscaled
    ±500 µm / 500 st/s search and actually ran ±70 µm / 70 st/s."""
    from talos.cv.autofocus_service import planned_kwargs

    settings = FakeSettings()
    row = {"name": "custom 20x", "mag": 20, "na": 0.40, "dof_um": 4.0,
           "coarse_step_um": 1.0, "fine_step_um": 0.2}
    kwargs, _warnings = planned_kwargs(settings, row)
    # (na_min/na)² = (0.15/0.40)² = 0.1406 — the numbers the run will use
    assert 300 <= kwargs["window_plus_steps"] <= 400      # not 2500
    assert 40 <= kwargs["coarse_speed"] <= 100            # not 500
    assert kwargs["fine_speed"] >= kwargs["coarse_speed"]


def test_default_roi_from_settings_is_sanitized():
    """A hand-edited default_roi_norm used to reach the metric verbatim."""
    settings = FakeSettings()
    settings.data["autofocus"]["default_roi_norm"] = (0.9999, 0.9999,
                                                     0.0001, 0.0001)
    service = AutofocusService(FakeManager(), settings, FakeState(),
                               LatestFrameSlot())
    assert service._default_roi() == (0.98, 0.98, 0.02, 0.02)
    settings.data["autofocus"]["default_roi_norm"] = None
    assert service._default_roi() is None      # the shipped "whole frame"
    settings.data["autofocus"]["default_roi_norm"] = [0.1, 0.1]
    assert service._default_roi() is None
