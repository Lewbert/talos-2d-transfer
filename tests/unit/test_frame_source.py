"""Frame capture for the scan: a stale frame must never pass as a fresh one.

The manifest records where the stage IS (a readback) next to every frame,
so filing a frame taken before the move would mislabel the image — the
whole reason the capture is gated on a timestamp rather than on "the
newest frame we happen to have".
"""

import threading
import time

import numpy as np

from talos.cv.frame_slot import LatestFrameSlot
from talos.cv.frame_source import CameraFrameSource, LatestFrameSource


def _frame(value: int = 7) -> np.ndarray:
    return np.full((4, 4, 3), value, np.uint8)


# --- the application's path: the shared slot -------------------------------

def test_a_frame_captured_after_the_settle_is_returned():
    slot = LatestFrameSlot()
    slot.write(_frame(11), time.monotonic(), 1)
    got = LatestFrameSource(slot).grab(settle_s=0.0, timeout_s=0.5)
    assert got is not None
    assert int(got[0][0, 0, 0]) == 11
    assert got[1].seq == 1


def test_a_frame_from_before_the_move_is_refused():
    """The camera stalled: the slot still holds the frame from before the
    stage moved. That frame must not be filed under the new position."""
    slot = LatestFrameSlot()
    slot.write(_frame(3), time.monotonic() - 60.0, 1)   # a minute old
    assert LatestFrameSource(slot).grab(settle_s=0.0, timeout_s=0.05) is None


def test_nothing_at_all_is_not_an_error():
    assert LatestFrameSource(LatestFrameSlot()).grab(
        settle_s=0.0, timeout_s=0.05) is None


def test_the_settle_window_is_spent_before_the_freshness_gate():
    """A frame already in flight when the move ended is not a settled one:
    it is written DURING the settle, and the capture after it must win."""
    slot = LatestFrameSlot()
    started = threading.Event()

    def writer():
        started.set()
        # write one immediately (during the settle) and one after it
        slot.write(_frame(1), time.monotonic(), 1)
        time.sleep(0.12)
        slot.write(_frame(2), time.monotonic(), 2)

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    started.wait(timeout=1.0)
    got = LatestFrameSource(slot).grab(settle_s=0.08, timeout_s=1.0)
    thread.join(timeout=2.0)
    assert got is not None
    assert int(got[0][0, 0, 0]) == 2, "the pre-settle frame was accepted"


# --- the CLI / test path: a camera the caller owns -------------------------

class _FakeCamera:
    """A camera that hands out prepared (frame, t_capture) pairs."""

    def __init__(self, items):
        self._items = list(items)
        self._last_t = None
        self.fetches = 0

    def fetch(self, timeout_ms=2000.0):
        self.fetches += 1
        if not self._items:
            return None
        frame, t = self._items.pop(0)
        self._last_t = t
        return frame

    def capture_time(self):
        return self._last_t


def test_camera_source_returns_a_fresh_frame():
    camera = _FakeCamera([(_frame(5), time.monotonic())])
    got = CameraFrameSource(camera).grab(settle_s=0.0, timeout_s=0.5)
    assert got is not None and int(got[0][0, 0, 0]) == 5


def test_camera_source_retries_a_frame_that_predates_the_settle():
    """The backend delivered a frame that was captured before the settle
    ended — ask again rather than filing it."""
    now = time.monotonic()
    camera = _FakeCamera([(_frame(1), now - 30.0),
                          (_frame(2), time.monotonic())])
    got = CameraFrameSource(camera).grab(settle_s=0.0, timeout_s=1.0)
    assert got is not None and int(got[0][0, 0, 0]) == 2
    assert camera.fetches >= 2


def test_camera_source_gives_up_instead_of_returning_a_stale_frame():
    camera = _FakeCamera([(_frame(1), time.monotonic() - 30.0)])
    assert CameraFrameSource(camera).grab(
        settle_s=0.0, timeout_s=0.05) is None


def test_camera_source_without_a_capture_clock_trusts_fetch():
    """A backend that cannot stamp its frames: fetch() waits for the NEXT
    frame, so what comes back is fresh by construction."""

    class _Unstamped:
        def fetch(self, timeout_ms=2000.0):
            return _frame(9)

    got = CameraFrameSource(_Unstamped()).grab(settle_s=0.0, timeout_s=0.2)
    assert got is not None and int(got[0][0, 0, 0]) == 9
    assert got[1].t_capture > 0
