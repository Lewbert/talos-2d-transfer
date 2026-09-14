"""Simulated device behavior tests (fast, deterministic)."""

import numpy as np
import pytest

from talos.hal.base import (
    CommandRejectedError,
    DeviceBusyError,
    Direction,
    EStopError,
    LimitHitError,
)
from talos.hal.sim import (
    SimCamera,
    SimFocusStage,
    SimSigmaKokiXYZStage,
    SimYudianTempController,
    SimZolixXYRStage,
)
from talos.hal.base import Axis


# --- SimFocusStage ---------------------------------------------------------

def test_sim_focus_move_and_wait_idle():
    focus = SimFocusStage({"max_speed": 2000, "latency_s": 0.001})
    focus.connect()
    focus.move_rel(100)
    assert not focus.get_status().is_idle
    focus.wait_idle(timeout_s=5.0)
    assert focus.get_status().pos == 100
    assert any(e.startswith("EV:DONE") for e in focus.drain_events())


def test_sim_focus_soft_limits_block():
    focus = SimFocusStage()
    focus.connect()
    focus.set_soft_limits(-500, 500)
    with pytest.raises(LimitHitError):
        focus.move_rel(2000)
    assert focus.get_status().mode == "LIMIT"


def test_sim_focus_busy_rejects_second_move():
    focus = SimFocusStage({"max_speed": 100, "latency_s": 0.001})
    focus.connect()
    focus.move_rel(100)  # takes ~1 s at 100 steps/s
    with pytest.raises(DeviceBusyError):
        focus.move_rel(50)
    focus.stop()


def test_sim_focus_speed_clamp():
    focus = SimFocusStage()
    focus.connect()
    with pytest.raises(CommandRejectedError):
        focus.set_speed(99999)


# --- SimZolixXYRStage --------------------------------------------------------

def test_sim_zolix_move_and_idle():
    zolix = SimZolixXYRStage({"latency_s": 0.001})
    zolix.connect()
    zolix.move_abs_um(625.0, 0.0)
    assert zolix.get_status().any_moving
    zolix.wait_idle(timeout_s=5.0)
    pos = zolix.get_position()
    assert pos.x_pulses == 1000
    assert pos.x_um == pytest.approx(625.0)


def test_sim_zolix_busy_rejects_move():
    zolix = SimZolixXYRStage({"slow_speed_pps": 100, "latency_s": 0.001})
    zolix.connect()
    zolix.move_abs_um(625.0, 0.0)  # 1000 pulses / 100 pps = 10 s
    with pytest.raises(DeviceBusyError):
        zolix.move_abs_um(0.0, 100.0)
    zolix.stop()


def test_sim_zolix_estop_latches_until_cleared():
    zolix = SimZolixXYRStage()
    zolix.connect()
    zolix.trigger_estop()
    assert zolix.check_estop()
    with pytest.raises(EStopError):
        zolix.move_abs_um(10.0, 0.0)
    zolix.stop()  # estop is latched
    with pytest.raises(EStopError):
        zolix.move_abs_um(10.0, 0.0)
    zolix.clear_estop()
    zolix.move_abs_um(10.0, 0.0)  # now fine
    zolix.wait_idle(timeout_s=5.0)


def test_sim_zolix_limit_blocks_direction():
    zolix = SimZolixXYRStage()
    zolix.connect()
    zolix.set_limit("x+", True)
    with pytest.raises(LimitHitError):
        zolix.move_abs_um(100.0, 0.0)
    zolix.set_limit("x+", False)
    zolix.move_abs_um(-100.0, 0.0)
    zolix.wait_idle(timeout_s=5.0)


# --- SimSigmaKokiXYZStage -----------------------------------------------------

def test_sim_sigmakoki_step_and_settle():
    stage = SimSigmaKokiXYZStage({"latency_s": 0.001})
    stage.connect()
    actual = stage.step(Axis.X, Direction.POSITIVE, 10)
    assert actual == 10
    stage.wait_idle(timeout_s=5.0)
    assert stage.get_position()[Axis.X] == 10


def test_sim_sigmakoki_limit_blocks():
    stage = SimSigmaKokiXYZStage()
    stage.connect()
    stage._limits["x+"] = True
    with pytest.raises(LimitHitError):
        stage.step(Axis.X, Direction.POSITIVE, 5)
    assert "EV:LIM" in stage.drain_events()[0]


def test_sim_sigmakoki_stop_all_clears_motion():
    stage = SimSigmaKokiXYZStage()
    stage.connect()
    stage.move(Axis.X, Direction.POSITIVE, 5)
    stage.stop()
    assert stage._moving == {}


# --- SimYudianTempController ---------------------------------------------------

def test_sim_yudian_set_sv_and_lag():
    yudian = SimYudianTempController({"thermal_tau_s": 0.5})
    yudian.connect()
    assert yudian.read_pv() == pytest.approx(25.0)
    yudian.set_sv(100.0)
    assert yudian.read_sv() == pytest.approx(100.0)
    import time

    time.sleep(0.6)
    assert 25.0 < yudian.read_pv() < 100.0  # lagging approach


def test_sim_yudian_safety_clamp():
    yudian = SimYudianTempController()
    yudian.connect()
    with pytest.raises(CommandRejectedError):
        yudian.set_sv(500.0)


# --- SimCamera -----------------------------------------------------------------

def test_sim_camera_fetch_shape_and_snapshot(tmp_path):
    cam = SimCamera({"width": 320, "height": 240, "fps": 1000})
    cam.connect()
    frame = cam.fetch()
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8
    path = cam.snapshot(tmp_path / "snap.png")
    assert path.exists()


def test_sim_camera_4k_snapshot_and_burn(tmp_path):
    import cv2

    cam = SimCamera({"width": 320, "height": 240, "fps": 1000})
    cam.connect()
    path = cam.snapshot(tmp_path / "snap4k.png", resolution=0)
    img = cv2.imread(str(path))
    assert img.shape == (2160, 3840, 3)
    # the live geometry is restored afterwards
    assert cam.fetch().shape == (240, 320, 3)
    # burn changes the bottom-right corner vs a plain snapshot
    burned = cam.snapshot(tmp_path / "burn.png", burn={"um_per_px": 0.5})
    plain = cam.snapshot(tmp_path / "plain.png")
    img_b = cv2.imread(str(burned))
    img_p = cv2.imread(str(plain))
    assert not np.array_equal(img_b, img_p)


def test_sim_camera_blur_reduces_sharpness():
    import cv2

    sharp = SimCamera({"width": 320, "height": 240, "blur_sigma": 0.0, "noise": 0.0})
    blurred = SimCamera({"width": 320, "height": 240, "blur_sigma": 6.0, "noise": 0.0})
    f_sharp = sharp.fetch()
    f_blur = blurred.fetch()
    var = lambda img: cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()  # noqa: E731
    assert var(f_sharp) > var(f_blur)
