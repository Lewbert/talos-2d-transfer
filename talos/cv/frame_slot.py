"""LatestFrameSlot: thread-safe handoff of the newest camera frame between
the camera worker (writer) and any reader (the autofocus worker, the
GUI thread, tests).

Why a shared slot: the autofocus job runs BLOCKING inside the focus
worker's _drain — queued Qt signals cannot reach it mid-run, so the
controller polls this slot directly. Frames are handed over by reference
(no copies): the Camera contract guarantees a fresh C-contiguous array
per fetch, and the slot keeps the newest one only (newest wins — older
frames are dropped, which is exactly what a focus sweep wants).

No Qt, no cv2 — usable from any thread and from the CLI tools.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class FrameMeta:
    t_capture: float   # monotonic capture-side timestamp
    seq: int           # monotonically increasing per-frame sequence
    shape: tuple       # (h, w, c)


class LatestFrameSlot:
    """Single-writer (camera worker), multi-reader frame mailbox."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame = None          # np.ndarray | None
        self._meta: FrameMeta | None = None

    def write(self, frame, t_capture: float, seq: int) -> None:
        """Publish a fresh frame (by reference). The caller must not reuse
        or mutate the array afterwards (Camera contract)."""
        meta = FrameMeta(t_capture=t_capture, seq=seq, shape=tuple(frame.shape))
        with self._lock:
            self._frame = frame
            self._meta = meta

    def read(self) -> tuple | None:
        """Newest frame as (frame, FrameMeta), or None when empty."""
        with self._lock:
            if self._frame is None or self._meta is None:
                return None
            return self._frame, self._meta

    def read_since(self, min_t: float | None = None,
                   min_seq: int = -1) -> tuple | None:
        """Newest frame newer than BOTH gates (t_capture >= min_t when
        given, seq > min_seq). None when nothing satisfies them — callers
        use this to skip frames they have already scored or that predate
        a move."""
        with self._lock:
            if self._frame is None or self._meta is None:
                return None
            meta = self._meta
            if meta.seq <= min_seq:
                return None
            if min_t is not None and meta.t_capture < min_t:
                return None
            return self._frame, meta
