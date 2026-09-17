"""DetectionEngine: run the identification pipeline off the GUI thread.

Two feeds, one worker, and they are deliberately NOT treated the same way:

- the **live feed** (a timer sampling the newest frame) — newest wins. A
  frame that arrives while the previous one is still being processed is
  DROPPED, because a preview lagging several hundred milliseconds behind
  the stream is worse than a preview that skips frames;
- the **scan feed** (one job per captured tile) — never dropped. A tile
  that is not examined is a sample that was not found, so tiles queue up
  and drain while the scan runs and after it finishes. The map fills in
  behind the stage.

Nothing here touches the camera, the live view's stream or the autofocus.
It consumes frames that were delivered anyway, and hands back candidates
plus (for the live feed) two display images. A failure in here can never
stop a scan: it is logged and the next job runs.

**Pre-processing happens here, and only here.** Both the pipeline and the
frame the operator is looking at are the output of the same chain
(``cv/preprocess.py``), applied once per job. That is the non-blocking
guarantee: a denoise costing hundreds of milliseconds delays the next
preview and nothing else — not the camera's capture sequence, not the
frame slot, not the autofocus, not a running scan. It is also why the two
processed views lag the live one slightly, which is the honest price of
never making the capture path wait.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

from PySide6.QtCore import QObject, QThread, QTimer, Signal

from talos.cv import preprocess as pre
from talos.cv.identify import IdentifyPipeline, render_overlay

#: Live-feed sample period. The stream runs at ~15-19 fps; asking for a
#: preview every ~7 frames keeps the GUI thread free and the panel usable.
DEFAULT_INTERVAL_MS = 140


@dataclass
class DetectJob:
    index: int            # -1 = the live feed, else the tile's waypoint index
    frame: object
    calib: object
    stage_pos: object
    config: object
    scale: float = 1.0
    render: bool = True
    flip: bool = False    # the camera flip, for the px→stage mapping
    preprocess: object = None    # PreprocessConfig | None
    colour: object = None        # the picked colour, for the curve's centre


class _DetectWorker(QThread):
    """The pipeline, on its own thread, fed by a queue."""

    #: index, IdentifyResult, the pre-processed frame, the overlay
    sig_result = Signal(int, object, object, object)
    sig_log = Signal(str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._queue: queue.Queue = queue.Queue()
        self._stopping = threading.Event()

    def submit(self, job: DetectJob) -> None:
        self._queue.put(job)

    def pending(self) -> int:
        return self._queue.qsize()

    def stop(self) -> None:
        self._stopping.set()
        self._queue.put(None)          # wake the consumer
        self.wait(4000)

    def run(self) -> None:  # noqa: D102
        while not self._stopping.is_set():
            job = self._queue.get()
            if job is None or self._stopping.is_set():
                return
            try:
                # One transform per job, feeding BOTH the pipeline and the
                # two processed views — so the mask can only ever find
                # colours that are on the screen the operator tuned it on.
                work = pre.apply(job.frame, job.preprocess, job.colour)
                result = IdentifyPipeline().run(
                    work, job.calib, config=job.config,
                    stage_pos=job.stage_pos, scale=job.scale,
                    flip=job.flip)
                overlay = (render_overlay(work, result)
                           if job.render else None)
            except Exception as exc:  # noqa: BLE001 - never kill the scan
                self.sig_log.emit(f"detection failed: {exc}")
                continue
            self.sig_result.emit(job.index, result, work, overlay)


class DetectionEngine(QObject):
    """Owns the worker, the live timer, and the "one job in flight" rule."""

    #: index, IdentifyResult, the pre-processed frame, the overlay
    sig_result = Signal(int, object, object, object)
    sig_log = Signal(str)

    def __init__(self, interval_ms: int = DEFAULT_INTERVAL_MS,
                 parent: QObject | None = None):
        super().__init__(parent)
        self._worker = _DetectWorker(self)
        self._worker.sig_result.connect(self._on_result)
        self._worker.sig_log.connect(self.sig_log)
        self._worker.start()
        self._source = None            # () -> the seven-tuple below
        self._busy = False
        self._live_on = False
        self._tiles_outstanding = 0
        self._timer = QTimer(self)
        self._timer.setInterval(max(30, int(interval_ms)))
        self._timer.timeout.connect(self._tick)

    # --- the live feed --------------------------------------------------

    def set_source(self, fn) -> None:
        """``fn()`` returns the newest ``(frame, calib, stage_pos, config,
        scale, flip, preprocess, colour)`` or None. Called on the GUI
        thread, once per tick."""
        self._source = fn

    def set_live(self, on: bool) -> None:
        """Start/stop the preview feed (the window toggles this with its
        own visibility — a hidden window must not burn CPU)."""
        self._live_on = bool(on)
        if self._live_on and not self._timer.isActive():
            self._timer.start()
        elif not self._live_on:
            self._timer.stop()

    @property
    def live(self) -> bool:
        return self._timer.isActive()

    def _tick(self) -> None:
        if self._busy or self._source is None:
            return                 # one job in flight: drop, don't queue
        try:
            item = self._source()
        except Exception as exc:  # noqa: BLE001
            self.sig_log.emit(f"detection source failed: {exc}")
            return
        if not item:
            return
        (frame, calib, stage_pos, config, scale, flip,
         preprocess, colour) = item
        if frame is None or config is None:
            return
        self._busy = True
        self._worker.submit(DetectJob(index=-1, frame=frame, calib=calib,
                                      stage_pos=stage_pos, config=config,
                                      scale=scale, render=True, flip=flip,
                                      preprocess=preprocess, colour=colour))

    # --- the scan feed --------------------------------------------------

    def submit_tile(self, index: int, frame, calib, stage_pos, config,
                    scale: float = 1.0, flip: bool = False,
                    preprocess=None, colour=None) -> None:
        """One captured tile. Queued unconditionally: a scan must not be
        able to outrun the detector and silently lose a sample.

        The tile is pre-processed the same way the live preview is, so a
        parameter tuned while looking at the screen means the same thing
        on the full-resolution tile. The PNG on disk stays raw."""
        self._tiles_outstanding += 1
        self._worker.submit(DetectJob(index=int(index), frame=frame,
                                      calib=calib, stage_pos=stage_pos,
                                      config=config, scale=scale,
                                      render=False, flip=flip,
                                      preprocess=preprocess, colour=colour))

    @property
    def busy(self) -> bool:
        return self._busy

    def pending(self) -> int:
        return self._worker.pending()

    @property
    def pending_tiles(self) -> int:
        """Tiles still to examine (queued or in flight) — the window waits
        for this to reach zero before writing a scan's results."""
        return self._tiles_outstanding

    def _on_result(self, index, result, preprocessed, overlay) -> None:
        if index < 0:
            self._busy = False         # the live slot is free again
        elif self._tiles_outstanding > 0:
            self._tiles_outstanding -= 1
        self.sig_result.emit(index, result, preprocessed, overlay)

    def shutdown(self) -> None:
        self._timer.stop()
        self._worker.stop()


__all__ = ["DEFAULT_INTERVAL_MS", "DetectJob", "DetectionEngine"]
