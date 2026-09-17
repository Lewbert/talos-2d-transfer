"""Frame capture for the grid scan: one contract, two sources.

``grab(settle_s, timeout_s) -> (frame, FrameMeta) | None`` returns a frame
whose CONTENT was captured *after* the settle window that follows a move —
or None, which the scanner counts as a missing waypoint.

Two implementations, because the app and the CLI tools own the camera
differently:

- :class:`LatestFrameSource` reads the shared ``LatestFrameSlot``. This is
  the application's path, and the reason the scanner no longer takes a
  camera object: the camera backend belongs to its own worker thread
  (``CameraProxy``) and the SmartCamApi binding is exclusive and not
  thread-safe, so the old ``GridScanner(camera=...)`` fetched straight
  into a backend another thread owned. That is exactly why every real run
  passed ``camera=None`` and captured nothing. The slot is the documented
  handoff for this case — the autofocus controller reads it the same way —
  and it costs the scan nothing, because the stream is already running.
- :class:`CameraFrameSource` drives a camera the caller OWNS (the CLI bench
  tools, the simulated tests). ``fetch()`` blocks until a new frame
  arrives, so the gate is the capture timestamp rather than a mailbox.

Both refuse a stale frame. A scan must never file the frame from before the
move as if it were the frame from after it: the position in the manifest is
a readback of where the stage IS, and a frame taken before the move would
mislabel the image. A refused capture returns None rather than a guess.

No Qt, no cv2 — usable from a worker thread and from the CLI tools.
"""

from __future__ import annotations

import time

from talos.cv.frame_slot import FrameMeta

#: How long to wait for the stream after the settle window before giving
#: up on a waypoint. A 15 fps stream needs ~70 ms per frame; a stalled
#: stream must fail rather than hang the scan.
DEFAULT_TIMEOUT_S = 2.0

_POLL_S = 0.01


class LatestFrameSource:
    """Capture from the shared frame slot (the application's path)."""

    def __init__(self, slot, poll_s: float = _POLL_S):
        self._slot = slot
        self._poll_s = float(poll_s)

    def grab(self, settle_s: float = 0.0,
             timeout_s: float = DEFAULT_TIMEOUT_S):
        """(frame, FrameMeta) captured after ``settle_s``, or None.

        The settle is mechanical (vibration after a stop); it is spent
        BEFORE the freshness gate is taken, so a frame that was already in
        flight when the move ended cannot pass as a settled one.
        """
        settle_end = time.monotonic() + max(0.0, float(settle_s))
        if settle_s > 0:
            time.sleep(float(settle_s))
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            item = self._slot.read_since(min_t=settle_end)
            if item is not None:
                return item
            if time.monotonic() >= deadline:
                return None
            time.sleep(self._poll_s)


class CameraFrameSource:
    """Capture from a camera the caller owns (CLI tools, simulated tests)."""

    def __init__(self, camera):
        self._camera = camera
        self._seq = 0

    def grab(self, settle_s: float = 0.0,
             timeout_s: float = DEFAULT_TIMEOUT_S):
        settle_end = time.monotonic() + max(0.0, float(settle_s))
        if settle_s > 0:
            time.sleep(float(settle_s))
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            remaining_ms = max(50.0, (deadline - time.monotonic()) * 1000.0)
            frame = self._camera.fetch(timeout_ms=remaining_ms)
            if frame is None:
                return None
            t_capture = self._capture_time()
            # A backend that cannot stamp its frames is trusted to have
            # just produced one: fetch() waits for the NEXT frame, it does
            # not hand back the newest buffered one.
            if t_capture is None or t_capture >= settle_end:
                self._seq += 1
                return frame, FrameMeta(
                    t_capture=t_capture if t_capture is not None
                    else time.monotonic(),
                    seq=self._seq, shape=tuple(frame.shape))
            if time.monotonic() >= deadline:
                return None

    def _capture_time(self) -> float | None:
        getter = getattr(self._camera, "capture_time", None)
        if getter is None:
            return None
        try:
            return getter()
        except Exception:  # noqa: BLE001 - a stamp is optional, never fatal
            return None


__all__ = ["CameraFrameSource", "LatestFrameSource", "DEFAULT_TIMEOUT_S"]
