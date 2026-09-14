"""SigmaKoki XYZ stage driver tests against a scripted fake serial.

Headline regression: the MV ack is ``OK:MV:<axis>:<dir>:<level>``. The
driver used to wait for a bare ``"MV"`` prefix, which never matched — so
the ack was skipped as a stray line and EVERY continuous jog blocked the
worker for the whole serial timeout (0.3 s on the bench). STOP was
unaffected (it goes through ``_send_loose``, which matches any line),
which is exactly why stopping felt snappy and jogging did not.
"""

import time

import pytest

from talos.hal.base import Axis, Direction, LimitHitError
from talos.hal.devices.sigmakoki import SigmaKokiXYZStage
from tests.testing.fake_serial import LineFakeSerial

STATUS = b"S:X:120,Y:-40,Z:7,XSPD:0,YSPD:0,ZSPD:0\n"


def make_driver(**script):
    from talos.protocols.ascii_line import LineIO

    fake = LineFakeSerial(script)
    driver = SigmaKokiXYZStage({"port": "COM6", "timeout_s": 0.2})
    driver._ser = fake
    driver._io = LineIO(fake, timeout_s=0.2)
    driver._connected = True
    return driver, fake


def test_move_consumes_the_real_ack():
    driver, fake = make_driver(**{"MV:X:+1:3": b"OK:MV:X:1:3\n"})
    driver.move(Axis.X, Direction.POSITIVE, 3)
    assert fake.writes == ["MV:X:+1:3"]


def test_move_returns_without_waiting_for_the_timeout():
    """The measured call must be near-instant: with the wrong prefix this
    took the full 0.2 s timeout (0.3 s on the bench)."""
    driver, _ = make_driver(**{"MV:Y:-1:2": b"OK:MV:Y:-1:2\n"})
    t0 = time.monotonic()
    driver.move(Axis.Y, Direction.NEGATIVE, 2)
    assert time.monotonic() - t0 < 0.05


def test_move_limit_error_still_raises():
    driver, _ = make_driver(**{"MV:X:+1:0": b"ERR:X:LIMIT\n"})
    with pytest.raises(LimitHitError):
        driver.move(Axis.X, Direction.POSITIVE, 0)


def test_move_level_is_clamped():
    driver, fake = make_driver(**{"MV:Z:+1:5": b"OK:MV:Z:1:5\n"})
    driver.move(Axis.Z, Direction.POSITIVE, 99)
    assert fake.writes == ["MV:Z:+1:5"]


def test_move_without_ack_is_not_fatal():
    """A deaf firmware must not fail the jog — MV is idempotent and the
    controller has already applied it."""
    driver, _ = make_driver()  # no scripted reply
    driver._io.timeout_s = 0.02
    driver.move(Axis.X, Direction.POSITIVE, 1)  # must not raise


def test_get_telemetry_costs_one_round_trip():
    """STATUS? carries both status and position; the proxy used to ask
    for them separately, doubling the serial traffic every poll."""
    driver, fake = make_driver(**{"STATUS?": STATUS})
    telemetry = driver.get_telemetry()
    assert fake.writes == ["STATUS?"]
    assert telemetry["status"]["x"] == "120"
    assert telemetry["position"] == {Axis.X: 120, Axis.Y: -40, Axis.Z: 7}


def test_get_position_matches_the_telemetry_path():
    driver, _ = make_driver(**{"STATUS?": STATUS})
    assert driver.get_position() == {Axis.X: 120, Axis.Y: -40, Axis.Z: 7}
