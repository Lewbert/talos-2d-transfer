"""Abort latency inside blocking autofocus moves.

`wait_idle` blocks for up to 60 s with no way out, so the abort latch —
which the controller polls between moves — could not take effect until a
move finished: a manual jog or a STOP ALL waited the whole budget while
the axis kept travelling. move_to_verified now polls a check callable
while the move is in flight and stops the axis itself.
"""

import time

import pytest

from talos.cv.autofocus import move_to_verified
from talos.hal.base import DeviceTimeoutError


class _SlowFocus:
    """Never idle; every wait_idle call would burn its whole timeout."""

    def __init__(self, idle: bool = False, pos: int = 0):
        self.idle = idle
        self.pos = pos
        self.stopped = 0
        self.moves: list[tuple[int, int]] = []
        self.waits: list[float] = []

    def move_abs(self, pos, speed=0):
        self.moves.append((pos, speed))

    def get_status(self):
        from types import SimpleNamespace

        return SimpleNamespace(is_idle=self.idle, pos=self.pos)

    def drain_events(self):
        return []

    def stop(self):
        self.stopped += 1

    def wait_idle(self, timeout_s=60.0, poll_s=0.05):
        """Mirrors the driver: returns when idle, else raises."""
        self.waits.append(timeout_s)
        if self.idle:
            return
        raise DeviceTimeoutError(f"Focus stage not idle after {timeout_s}s")


def test_abort_during_a_move_stops_the_axis_at_once():
    focus = _SlowFocus()
    calls = {"n": 0}

    def check():
        calls["n"] += 1
        return "aborted by user" if calls["n"] >= 2 else None

    started = time.monotonic()
    assert move_to_verified(focus, 100, 500, check=check) is False
    assert focus.stopped == 1, "the aborting move must stop the axis"
    assert time.monotonic() - started < 1.0, "abort must not wait out the budget"


def test_checked_move_settles_normally():
    focus = _SlowFocus(idle=True, pos=100)
    assert move_to_verified(focus, 100, 500, check=lambda: None) is True
    assert focus.stopped == 0


def test_readback_mismatch_still_raises():
    focus = _SlowFocus(idle=True, pos=40)
    with pytest.raises(DeviceTimeoutError, match="readback mismatch"):
        move_to_verified(focus, 100, 500, check=lambda: None)


def test_without_a_check_the_driver_wait_is_used():
    """The backlash calibrator (and any other caller) keeps the original
    blocking behavior."""
    focus = _SlowFocus(idle=True, pos=100)
    assert move_to_verified(focus, 100, 500) is True
    # the DRIVER's blocking wait_idle was used (not the polling mirror)
    assert focus.waits == [60.0]
    assert focus.stopped == 0


def test_checked_move_times_out_when_never_idle(monkeypatch):
    """The mirror of the driver's own timeout: a move that never settles
    still raises DeviceTimeoutError (the caller re-issues once)."""
    from talos.cv import autofocus as af

    from talos.cv.autofocus import _wait_idle_checked

    focus = _SlowFocus()
    with pytest.raises(DeviceTimeoutError, match="not idle"):
        _wait_idle_checked(focus, check=lambda: None, timeout_s=0.05)
