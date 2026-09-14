"""FocusStageDriver tests against a scripted fake serial."""

import pytest

from talos.hal.base import (
    CommandRejectedError,
    DeviceBusyError,
    DeviceTimeoutError,
    LimitHitError,
)
from talos.hal.devices.focus import FocusStageDriver
from tests.testing.fake_serial import LineFakeSerial

IDLE = b"S:POS:-589,MODE:IDLE,V:0,SPD:0,LIM:0,SLIM:0\n"


def make_driver(**script):
    fake = LineFakeSerial(script)
    driver = FocusStageDriver({"port": "COM10", "max_speed": 2000, "timeout_s": 0.2})
    driver._ser = fake
    from talos.protocols.ascii_line import LineIO

    driver._io = LineIO(fake, timeout_s=0.2)
    driver._connected = True
    return driver, fake


def test_get_status_parses():
    driver, _ = make_driver(**{"STATUS?": IDLE})
    status = driver.get_status()
    assert status.pos == 589  # firmware -589 → driver +589 (sign convention)
    assert status.mode == "IDLE"
    assert status.is_idle


def test_sign_convention_flip():
    """Driver + = distance INCREASE = firmware −. Uniform at the boundary:
    speed, relative/absolute moves, position, blocked direction."""
    driver, fake = make_driver(**{
        "MOVE:100": b"OK:MOVE:100\n",
        "GOTO:-50": b"OK:GOTO:-50\n",
        "SPD:-30": b"OK:SPD:-30\n",
        "STATUS?": b"S:POS:-7,MODE:IDLE,V:0,SPD:0,LIM:+,SLIM:0\n",
    })
    driver.move_rel(-100)
    driver.move_abs(50)
    driver.set_speed(30)
    assert fake.writes == ["MOVE:100", "GOTO:-50", "SPD:-30"]
    status = driver.get_status()
    assert status.pos == 7
    assert status.blocked_dir == "-"  # firmware '+' block → driver '-'
    assert status.lim


def test_move_rel_ok():
    driver, fake = make_driver(**{"MOVE:100": b"OK:MOVE:100\n"})
    driver.move_rel(-100)
    assert fake.writes == ["MOVE:100"]


def test_move_busy():
    driver, _ = make_driver(**{"MOVE:-100": b"ERR:BUSY\n"})
    with pytest.raises(DeviceBusyError):
        driver.move_rel(100)


def test_move_limit():
    driver, _ = make_driver(**{"MOVE:-100": b"ERR:LIMIT\n"})
    with pytest.raises(LimitHitError):
        driver.move_rel(100)


def test_speed_clamp_rejects_before_write():
    driver, fake = make_driver()
    with pytest.raises(CommandRejectedError):
        driver.set_speed(99999)
    with pytest.raises(CommandRejectedError):
        driver.set_speed(-5)
    assert fake.writes == []


def test_speed_zero_is_ramp_stop():
    driver, fake = make_driver(**{"SPD:0": b"OK:SPD:0\n"})
    driver.set_speed(0)
    assert fake.writes == ["SPD:0"]


def test_soft_limits_parse():
    driver, _ = make_driver(**{"SLIM?": b"SLIM:1:-3000000:5000000\n"})
    assert driver.get_soft_limits() == (-5000000, 3000000)  # negated + swapped


def test_soft_limits_set_flips_to_firmware_units():
    driver, fake = make_driver(**{"SLIM:SET:-7000000:-1000000": b"OK:SLIM:1:-7000000:-1000000\n"})
    driver.set_soft_limits(1000000, 7000000)
    assert fake.writes == ["SLIM:SET:-7000000:-1000000"]
    assert driver.get_soft_limits() == (1000000, 7000000)


def test_soft_limits_set_rejects_bad_range():
    driver, fake = make_driver()
    with pytest.raises(CommandRejectedError):
        driver.set_soft_limits(100, 50)
    assert fake.writes == []


def test_wait_idle_until_idle():
    driver, _ = make_driver(**{
        "STATUS?": [b"S:POS:0,MODE:TRAP,V:0,SPD:0,LIM:0,SLIM:0\n", IDLE],
    })
    driver.wait_idle(timeout_s=2.0, poll_s=0.01)


def test_wait_idle_raises_on_tmo_event():
    driver, fake = make_driver(**{"STATUS?": IDLE})
    fake._pending = b"EV:TMO:12\n"
    with pytest.raises(DeviceTimeoutError, match="inactivity"):
        driver.wait_idle(timeout_s=2.0, poll_s=0.01)


def test_wait_idle_tolerates_transient_poll_error():
    """Regression: a stepper-current glitch resets the firmware mid-poll —
    the failing STATUS? must be tolerated until the deadline. (The except
    DeviceError handler in wait_idle referenced an UNIMPORTED name until
    hardware tripped it — this test would have caught the NameError.)"""
    from talos.hal.base import DeviceError

    driver, fake = make_driver(**{"STATUS?": IDLE})
    calls = {"n": 0}
    real_get_status = driver.get_status

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise DeviceError("garbled")
        return real_get_status()

    driver.get_status = flaky
    driver.wait_idle(timeout_s=2.0, poll_s=0.01)
    assert calls["n"] >= 2  # first poll failed, later ones succeeded


def test_self_heals_after_persistent_garbles():
    """Persistent ERR:UNKNOWN = a desynced firmware line discipline. The
    driver must DTR-reset (reopen via the serial factory) + re-handshake,
    and the next command on the fresh link must succeed. (Hardware-
    verified: the desync never recovers without a reset.)"""
    from talos.protocols.ascii_line import LineIO

    class BannerFake(LineFakeSerial):
        def reset_input_buffer(self):
            # the handshake reads the banner — re-seed instead of clearing
            self._pending = b"BOOT:FOCUSCTRL:1.0\nREADY\n"

    factory_calls = {"n": 0}

    def make_fake(**kwargs):
        factory_calls["n"] += 1
        script = {"PING": b"PONG\n",
                  "SLIM?": b"SLIM:0:-2000000:2000000\n"}
        if factory_calls["n"] == 1:
            script["STATUS?"] = b"ERR:UNKNOWN:garbage\n"
        else:
            script["STATUS?"] = IDLE
        return BannerFake(script)

    driver = FocusStageDriver({"port": "COM10", "max_speed": 2000,
                               "timeout_s": 0.2,
                               "serial_factory": make_fake})
    driver.connect()
    assert factory_calls["n"] == 1
    # five failing commands (each retries once) — the fifth triggers the
    # self-heal: the factory is called again (DTR reset) + handshake
    for _ in range(5):
        with pytest.raises(CommandRejectedError):
            driver.get_status()
    assert factory_calls["n"] == 2
    # the fresh link answers cleanly
    assert driver.get_status().mode == "IDLE"


def test_stop_never_raises_when_disconnected():
    driver, _ = make_driver()
    driver._connected = False
    driver._io = None
    driver.stop()  # must not raise
