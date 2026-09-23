"""The scan's frame writer: encode, thumbnail and record on its own thread.

Per waypoint the scan has a frame in hand and a position it was taken at,
and then a pile of work that has nothing to do with the stage: encode a
PNG (40-120 ms at 1080p, several hundred at 4K), resize a thumbnail, write
a manifest row and flush it. All of that used to happen inline, on the one
thread that could instead be commanding the next move — so every tile paid
it twice over, once as the work and once as the stage standing still.

This puts it on a writer thread behind a bounded queue. The scanner hands
over ``(index, position, frame)`` and moves on; the writer encodes, appends
the manifest row and flushes.

The scan's *signals* stay on the scan thread (the scanner emits them itself
before handing the frame over). Emitting them from here would have been one
thread fewer in the way of the encode, but it costs more than it saves: two
emitters for one signal, an ordering question against ``sig_done``, and a
queued connection that a synchronous caller — the sim tests, the CLI benches
— never sees delivered.

Three properties the callers depend on:

- **Order.** One thread, one FIFO, and every waypoint produces exactly one
  row — including the ones with no frame and the ones with no position.
  The manifest is 1:1 with the plan whatever happened.
- **The manifest is the record of what is on disk.** A row names a file
  only once that file exists; a failed encode writes the row with an empty
  frame column and records the error. Nothing downstream re-reads a file
  the manifest promised (see ``cv/scan_output.py``, which rebuilds the
  mosaic from the manifest).
- **Bounded memory.** ``max_queue`` frames are held at most, and a full
  queue blocks the *writer's* caller — the scan thread — which is the
  intended backpressure: a slow disk costs speed, never RAM. The wait is
  bounded too (see ``_ENQUEUE_TIMEOUT_S``): a disk that stops draining
  entirely must fail the run, not wedge the thread that owns the abort flag.

The writer is a plain thread, not a QThread: it has no event loop and no
slots, and the signals it emits belong to the scanner (emitting a signal
from another thread is safe; the receivers' delivery is decided by their
own thread affinity).
"""

from __future__ import annotations

import csv
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from talos.hal.base import DeviceError

logger = logging.getLogger(__name__)

MANIFEST_HEADER = ["frame", "x_um", "y_um", "r_deg", "t_unix",
                   "objective_id", "focus_pos"]

#: How long the scan thread will wait for room in the writer's queue before
#: declaring the disk stalled. Long enough for a genuinely busy disk (a 4K
#: PNG is tens of milliseconds), short enough that a wedged one fails the run
#: while the operator is still watching it.
_ENQUEUE_TIMEOUT_S = 20.0
#: How often the wait re-checks that deadline (a blocking ``put`` gives the
#: abort no chance to be noticed between attempts).
_ENQUEUE_POLL_S = 0.2


class WriterStalledError(DeviceError):
    """The writer's queue did not drain — the run cannot be recorded.

    A ``DeviceError`` so the scanner's own handler reports it as a stopped
    run with a reason (its `except DeviceError` is where three outcomes are
    told apart), rather than letting it escape as an unhandled exception.
    """

#: A sentinel that stops the drain loop. A module-level unique object so no
#: real item can be mistaken for it.
_STOP = object()


@dataclass
class _Item:
    """One manifest row's worth of work."""

    index: int
    #: (x_um, y_um, r_deg) READBACK, or None when the controller did not
    #: answer — in which case the row carries no position and no frame.
    pos: tuple[float, float, float] | None
    frame: object | None
    t_unix: float


class FrameWriter:
    """Owns the scan's `frames/` directory and `manifest.csv` on a thread."""

    def __init__(self, out_dir: Path, meta: dict | None = None, *,
                 max_queue: int = 4):
        self._out_dir = Path(out_dir)
        self._meta = dict(meta or {})
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(max_queue)))
        self._frames: list[Path] = []
        self._error: str | None = None
        self._thread: threading.Thread | None = None
        self._manifest = None
        self._rows = None
        self.frames_dir = self._out_dir / "frames"
        self.manifest_path = self._out_dir / "manifest.csv"

    # ------------------------------------------------------------------
    # the scan thread
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the manifest and start draining."""
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = open(self.manifest_path, "w", newline="",
                              encoding="utf-8")
        self._rows = csv.writer(self._manifest)
        self._rows.writerow(MANIFEST_HEADER)
        self._manifest.flush()
        self._thread = threading.Thread(target=self._drain, name="scan-writer",
                                        daemon=True)
        self._thread.start()

    def submit_frame(self, index: int, pos: tuple[float, float, float],
                     frame, t_unix: float) -> None:
        """Queue a captured frame with the readback position it was taken
        at. Blocks only while the queue is full."""
        self._enqueue(_Item(int(index), (float(pos[0]), float(pos[1]),
                                         float(pos[2])), frame,
                            float(t_unix)))

    def submit_missing(self, index: int, pos: tuple[float, float, float] | None,
                       t_unix: float) -> None:
        """Queue a waypoint that produced no usable frame: the row is
        written with an empty position and/or an empty frame column."""
        self._enqueue(_Item(int(index), pos, None, float(t_unix)))

    def _enqueue(self, item: _Item) -> None:
        """Queue one item, and never block the SCAN thread for ever.

        The queue is the backpressure that keeps a slow disk from eating
        RAM, so a full queue is normal and waiting is the point. Waiting
        without a limit is not: the scan folder is operator-chosen, and one
        on a network share or a spun-down disk blocks inside ``imwrite`` for
        minutes. The scan thread would then be stuck in ``put`` — where the
        abort flag is never read, ``close()`` is never reached and the run
        can never unwind: no ``sig_done``, the job stays "scan", the axes
        stay locked and there is no way out of the UI.

        So the wait is bounded, and giving up is an ERROR of the writer
        rather than a hang: the scan stops with a reason, which is what
        ``close()`` already does for a wedged writer.
        """
        deadline = time.monotonic() + _ENQUEUE_TIMEOUT_S
        while True:
            try:
                self._queue.put(item, timeout=_ENQUEUE_POLL_S)
                return
            except queue.Full:
                if time.monotonic() >= deadline:
                    self._note(
                        f"the frame writer has not drained for "
                        f"{_ENQUEUE_TIMEOUT_S:.0f} s (a stalled disk?) — "
                        f"giving up on this tile")
                    raise WriterStalledError(self._error or "the frame writer "
                                                             "is stalled")

    def close(self, timeout_s: float = 60.0) -> str | None:
        """Finish what is queued and close the manifest.

        MUST be called before anything reads the manifest back from disk
        or reports how many frames a run produced. Returns the first error
        (None if the run recorded everything it was given).
        """
        thread = self._thread
        try:
            # Bounded wait: a writer wedged in an I/O call has a full queue
            # and will never drain, and close() must not hang with it.
            self._queue.put(_STOP, timeout=max(0.0, float(timeout_s)))
        except queue.Full:
            self._note("the frame writer is not draining")
        if thread is not None:
            thread.join(max(0.0, float(timeout_s)))
            if thread.is_alive():
                # A writer wedged in an I/O call owns the open manifest;
                # closing it from here would pull the file out from under
                # it. The rows written so far are on disk and the run is
                # reported as failed rather than quietly short.
                self._note(f"the frame writer did not finish within "
                           f"{timeout_s:.0f} s")
        return self._error

    # ------------------------------------------------------------------
    # the writer thread
    # ------------------------------------------------------------------

    def _note(self, message: str) -> None:
        """Record the FIRST failure — later ones are the same wound."""
        if self._error is None:
            self._error = message
            logger.warning("scan frame writer: %s", message)

    def _drain(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is _STOP:
                    break
                try:
                    self._write(item)
                except Exception as exc:  # noqa: BLE001 - one bad tile, not a crash
                    self._note(f"{type(exc).__name__}: {exc}")
                    # The row is still written — with no frame name, because
                    # there is no file — so the manifest stays 1:1 with the
                    # plan and no row promises an image that does not exist.
                    self._row(item, frame_name="")
        finally:
            try:
                if self._manifest is not None:
                    self._manifest.flush()
                    self._manifest.close()
            except Exception as exc:  # noqa: BLE001
                self._note(f"closing the manifest failed: {exc}")

    def _write(self, item: _Item) -> None:
        if item.frame is None:
            self._row(item, frame_name="")
            return
        frame_path = self.frames_dir / f"frame_{item.index:05d}.png"
        written = cv2.imwrite(str(frame_path),
                              cv2.cvtColor(item.frame, cv2.COLOR_RGB2BGR))
        if not written:
            raise OSError(f"could not write {frame_path.name}")
        self._frames.append(frame_path)
        self._row(item, frame_name=frame_path.name)

    def _row(self, item: _Item, frame_name: str) -> None:
        if self._rows is None:
            return
        if item.pos is None:
            x_um = y_um = r_deg = ""
        else:
            x_um, y_um, r_deg = (f"{item.pos[0]:.3f}", f"{item.pos[1]:.3f}",
                                 f"{item.pos[2]:.4f}")
        self._rows.writerow([frame_name, x_um, y_um, r_deg,
                             f"{item.t_unix:.3f}",
                             self._meta.get("objective_id", ""),
                             self._meta.get("focus_pos", "")])
        self._manifest.flush()

    # ------------------------------------------------------------------
    # results
    # ------------------------------------------------------------------

    @property
    def frames(self) -> list[Path]:
        """The frames actually on disk, in waypoint order."""
        return list(self._frames)

    @property
    def error(self) -> str | None:
        return self._error


__all__ = ["FrameWriter", "MANIFEST_HEADER"]
