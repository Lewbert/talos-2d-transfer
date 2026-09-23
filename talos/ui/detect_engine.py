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

#: What a LIVE job computes. Derived from what the operator can actually
#: see (``SampleFindingWorkspace._live_level``), because the alternative —
#: running the chain for a view nobody is looking at — was the bench's
#: "it is doing work in the background" report. Tiles are always FULL: a
#: tile that is not identified is a sample that was not found.
LIVE_NONE = "none"            # nothing to show but the raw frame
LIVE_PREPROCESS = "preprocess"   # the pre-processed view
LIVE_FULL = "full"            # ...and the identification, for the samples view


@dataclass
class DetectJob:
    index: int            # -1 = the live feed, else the tile's waypoint index
    frame: object
    calib: object
    stage_pos: object
    config: object
    scale: float = 1.0
    #: How finely this frame samples the field of view compared with the
    #: frame the operator tunes on — 1.0 for the live feed, and the ratio
    #: for a tile captured at another resolution (Preferences → Scan).
    frame_scale: float = 1.0
    render: bool = True
    flip: bool = False    # the camera flip, for the px→stage mapping
    preprocess: object = None    # PreprocessConfig | None
    colour: object = None        # the picked colour, for the curve's centre
    #: How much of the chain to run for a LIVE job (see the LIVE_* names).
    #: Ignored for a tile: those are always identified.
    level: str = LIVE_FULL
    #: Which scan run this tile belongs to (0 = the live feed). A run's
    #: tiles queue behind the previous run's when the operator scans again
    #: immediately, and without this the stragglers were filed into the NEW
    #: run's sample list under the OLD run's indices.
    token: int = 0


class _DetectWorker(QThread):
    """The pipeline, on its own thread, fed by a queue."""

    #: index, IdentifyResult, the pre-processed frame, the overlay, the run
    #: token the job was submitted with
    sig_result = Signal(int, object, object, object, int)
    #: The index of a job that RAISED. Its slot has to be freed somewhere,
    #: and the result signal cannot carry that.
    sig_failed = Signal(int)
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
                if job.level == LIVE_PREPROCESS:
                    # The pre-processed view is on screen and the samples
                    # view is not: the mask and the overlay are work for
                    # nobody, and the frame is still handed on.
                    result = None
                    overlay = None
                else:
                    result = IdentifyPipeline().run(
                        work, job.calib, config=job.config,
                        stage_pos=job.stage_pos, scale=job.scale,
                        flip=job.flip, frame_scale=job.frame_scale)
                    overlay = (render_overlay(work, result)
                               if job.render else None)
            except Exception as exc:  # noqa: BLE001 - never kill the scan
                self.sig_log.emit(f"detection failed: {exc}")
                # The slot has to be freed even when the job raised: the
                # live feed stops for the session if _busy is left set, and
                # a tile that never decrements keeps the export waiting for
                # a queue that will never drain.
                self.sig_failed.emit(job.index)
                continue
            self.sig_result.emit(job.index, result, work, overlay, job.token)


class DetectionEngine(QObject):
    """Owns the worker, the live timer, and the "one job in flight" rule."""

    #: index, IdentifyResult, the pre-processed frame, the overlay, the token
    sig_result = Signal(int, object, object, object, int)
    sig_log = Signal(str)

    def __init__(self, interval_ms: int = DEFAULT_INTERVAL_MS,
                 parent: QObject | None = None):
        super().__init__(parent)
        self._worker = _DetectWorker(self)
        self._worker.sig_result.connect(self._on_result)
        self._worker.sig_failed.connect(self._on_failed)
        self._worker.sig_log.connect(self.sig_log)
        self._worker.start()
        self._source = None            # () -> the eight-tuple below
        self._busy = False
        self._live_on = False
        self._live_level = LIVE_FULL
        self._suspended = False
        #: Bumped by ``begin_run``. A tile whose token is not the current
        #: one is a straggler from the run before, and is dropped here.
        self._run_token = 0
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

    def set_suspended(self, suspended: bool) -> None:
        """Hold the LIVE feed without turning it off.

        A scan owns this worker while it runs: its tiles are never dropped,
        and every live job they wait behind is a sample found late. So the
        preview feed is suspended for the run — but NOT with
        ``set_live(False)``, which four handlers in the tab re-arm on every
        edit (a colour, a filter, a view-mode change), and which the
        dropper depends on staying fed.
        """
        self._suspended = bool(suspended)

    @property
    def suspended(self) -> bool:
        return self._suspended

    def set_live_level(self, level: str) -> None:
        """How much of the chain the live feed should run (see the LIVE_*
        names). Set from what the operator can see — the view mode and
        whether the tab is the visible page."""
        level = str(level)
        if level not in (LIVE_NONE, LIVE_PREPROCESS, LIVE_FULL):
            level = LIVE_FULL
        self._live_level = level

    @property
    def live_level(self) -> str:
        return self._live_level

    def _tick(self) -> None:
        if (self._busy or self._source is None or self._suspended
                or self._live_level == LIVE_NONE):
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
        if frame is None:
            return
        if config is None and self._live_level == LIVE_FULL:
            return
        self._busy = True
        self._worker.submit(DetectJob(index=-1, frame=frame, calib=calib,
                                      stage_pos=stage_pos, config=config,
                                      scale=scale, render=True, flip=flip,
                                      preprocess=preprocess, colour=colour,
                                      level=self._live_level,
                                      token=self._run_token))

    # --- the scan feed --------------------------------------------------

    def submit_tile(self, index: int, frame, calib, stage_pos, config,
                    scale: float = 1.0, flip: bool = False,
                    preprocess=None, colour=None,
                    frame_scale: float = 1.0) -> None:
        """One captured tile. Queued unconditionally: a scan must not be
        able to outrun the detector and silently lose a sample.

        The tile is pre-processed the same way the live preview is, so a
        parameter tuned while looking at the screen means the same thing
        on the full-resolution tile. The PNG on disk stays raw."""
        self._tiles_outstanding += 1
        self._worker.submit(DetectJob(index=int(index), frame=frame,
                                      calib=calib, stage_pos=stage_pos,
                                      config=config, scale=scale,
                                      frame_scale=frame_scale,
                                      render=False, flip=flip,
                                      preprocess=preprocess, colour=colour,
                                      level=LIVE_FULL, token=self._run_token))

    def begin_run(self) -> None:
        """A new scan is starting: its tiles get their own identity.

        Detection outlives the capture by design, so a run that is aborted
        leaves tiles in this queue. The operator pressing *Scan* again starts
        the next run immediately, and those stragglers used to arrive with
        indices the new run also uses — listed as the new run's samples, at
        the new run's positions, and ringed on its mosaic.
        """
        self._run_token += 1

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

    def _on_result(self, index, result, preprocessed, overlay, token=0) -> None:
        if index < 0:
            self._busy = False         # the live slot is free again
        elif self._tiles_outstanding > 0:
            self._tiles_outstanding -= 1
        if index >= 0 and int(token) != self._run_token:
            # A tile from the run before this one. It still counted as
            # drained above (the job IS finished); its result is not ours.
            self.sig_log.emit(
                f"tile {index} was identified after its run ended — the "
                f"result is not filed (it belongs to the previous scan)")
            return
        self.sig_result.emit(index, result, preprocessed, overlay, token)

    def _on_failed(self, index: int) -> None:
        """Free the slot of a job that raised (it emits no result).

        Without this the live feed stops for the rest of the session
        (``_busy`` is only ever cleared by a live result) and a tile job that
        raised keeps ``pending_tiles`` above zero forever — which also holds
        the run's exports hostage, because they wait for that number to reach
        zero. One bad frame must cost one frame.
        """
        if index < 0:
            self._busy = False
        elif self._tiles_outstanding > 0:
            self._tiles_outstanding -= 1

    def shutdown(self) -> None:
        self._timer.stop()
        self._worker.stop()


__all__ = ["DEFAULT_INTERVAL_MS", "DetectJob", "DetectionEngine"]
