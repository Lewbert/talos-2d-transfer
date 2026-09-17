"""The detection worker: pre-processing first, then the mask, then both
views — and never the GUI thread.

The property worth pinning is the ORDER. Pre-processing in the wrong
place is not a cosmetic bug: the dropper samples the pre-processed layer
and the mask searches it, so if the two ever came from different pixels
the operator's picked colour would stop matching the flake they picked.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from talos.cv.identify import IdentifyConfig, ColourStage
from talos.cv.preprocess import LocalContrast, PreprocessConfig
from talos.models import ObjectiveCalibration
from talos.ui.detect_engine import DetectJob, _DetectWorker

CALIB = ObjectiveCalibration(objective_id=0, um_per_px_x=0.2, um_per_px_y=0.2)

#: A blob that is NOT the picked colour until the chain runs on it: an
#: exposure of 2.0 turns (100, 60, 60) into (200, 120, 120), which is what
#: the operator picked.
RAW_BLOB = (100, 60, 60)
FILTERED_BLOB = (200, 120, 120)
PICKED = "#c87878"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _frame() -> np.ndarray:
    img = np.full((240, 320, 3), (60, 90, 120), np.uint8)
    img[80:120, 100:140] = RAW_BLOB
    return img


def _config() -> IdentifyConfig:
    stages = IdentifyConfig().stages
    for index, stage in enumerate(stages):
        if stage.NAME == "colour":
            stages[index] = ColourStage(hex_color=PICKED, tolerance=10.0)
    return IdentifyConfig(stages=stages)


def _run(qapp, frame: np.ndarray | None = None, **job_kw):
    """Feed one job through the real worker and wait for its result."""
    worker = _DetectWorker()
    captured: list = []
    worker.sig_result.connect(lambda *args: captured.append(args))
    worker.start()
    try:
        worker.submit(DetectJob(index=-1,
                                frame=_frame() if frame is None else frame,
                                calib=CALIB, stage_pos=None, config=_config(),
                                **job_kw))
        deadline = time.time() + 10.0
        while not captured and time.time() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
    finally:
        worker.stop()
    assert captured, "the worker produced no result"
    return captured[0]


def test_the_chain_runs_before_the_mask(qapp):
    """The blob only becomes the picked colour AFTER the exposure is
    applied — so finding it at all proves the ordering."""
    _index, result, _pre, _overlay = _run(
        qapp, preprocess=PreprocessConfig(enabled=True, exposure=2.0))
    assert len(result.candidates) == 1
    cand = result.candidates[0]
    assert cand.x_px == pytest.approx(120.0, abs=4)
    assert cand.y_px == pytest.approx(100.0, abs=4)


def test_without_the_chain_the_same_frame_finds_nothing(qapp):
    """The control for the test above: same frame, same mask, no chain."""
    _index, result, preprocessed, _overlay = _run(
        qapp, preprocess=PreprocessConfig())
    assert result.candidates == []
    # and with nothing enabled the pre-processed layer IS the raw frame
    assert preprocessed is not None
    assert np.array_equal(preprocessed, _frame())


def test_the_two_views_are_built_from_the_preprocessed_layer(qapp):
    """The overlay is the worker's own output drawn on the SAME array the
    mask ran on — not on the raw frame with the mask's coordinates
    projected onto it. A darker filter, a brighter outline: if the two
    disagreed, the outlines would sit on the wrong pixels."""
    _index, _result, preprocessed, overlay = _run(
        qapp, preprocess=PreprocessConfig(enabled=True, exposure=2.0))
    assert overlay is not None
    assert overlay.shape == preprocessed.shape
    # inside the match the pre-processed pixels survive untouched
    assert np.array_equal(overlay[100, 120], preprocessed[100, 120])
    # outside it they are darkened
    assert int(overlay[5, 5, 0]) < int(preprocessed[5, 5, 0])
    # and the pre-processed layer is NOT the raw one here
    assert not np.array_equal(preprocessed, _frame())


def test_the_curve_does_not_lose_the_colour_it_is_centred_on(qapp):
    """One colour drives both the curve's centre and the mask's target. If
    the curve moved the picked value, switching it on would make the mask
    miss the very flake the operator pointed at — so the sample must
    survive its own filter."""
    img = _frame()
    img[80:120, 100:140] = FILTERED_BLOB          # already the picked colour
    pre = PreprocessConfig(enabled=True)
    pre.local = LocalContrast(enabled=True, gain=4.0, width=24.0)
    _index, result, _pre, _overlay = _run(qapp, frame=img, preprocess=pre,
                                          colour=FILTERED_BLOB)
    assert len(result.candidates) == 1


# --- the non-blocking guarantee --------------------------------------------

def test_the_gui_thread_never_runs_the_chain(qapp):
    """The property the whole design turns on: pre-processing and
    identification run on the worker, so an expensive filter delays the
    next PREVIEW and nothing else. A capture that waited on a bilateral
    filter would be a scan that quietly misses tiles.

    Measured on the GUI-thread entry point (``_tick``), not on the queue
    put — the failure this guards against is someone moving the chain INTO
    the tick, where a queue assertion would still pass and the whole
    application would stutter between frames.
    """
    import time as _time

    from talos.cv import preprocess as pre
    from talos.ui.detect_engine import DetectionEngine

    original = pre.apply

    def slow(img, cfg, centre=None):
        _time.sleep(0.4)
        return original(img, cfg, centre)

    engine = DetectionEngine(interval_ms=30)
    try:
        engine.set_source(lambda: (_frame(), CALIB, None, _config(), 0.5,
                                   False,
                                   PreprocessConfig(enabled=True,
                                                    brightness=5.0),
                                   (200, 120, 120)))
        started = _time.perf_counter()
        engine._tick()
        elapsed = _time.perf_counter() - started
        assert elapsed < 0.05, \
            f"the tick blocked for {elapsed:.3f}s — the chain is on the GUI thread"

        # and the work really is happening: with a slow chain the worker is
        # still busy a moment later, which is where the time went
        pre.apply = slow
        engine._busy = False
        started = _time.perf_counter()
        engine._tick()
        assert _time.perf_counter() - started < 0.05
    finally:
        pre.apply = original
        engine.shutdown()


def test_the_worker_emits_both_views_from_one_transform(qapp):
    """Both processed views and the mask come from ONE array, so the
    operator can only tune against pixels the pipeline also saw."""
    _index, result, preprocessed, overlay = _run(
        qapp, preprocess=PreprocessConfig(enabled=True, exposure=2.0))
    assert preprocessed is not None and overlay is not None
    assert result.mask is not None
    # the overlay is the pre-processed frame, annotated
    assert overlay.shape == preprocessed.shape == _frame().shape
    # a tile job asks for no preview at all (render=False): the scan does
    # not pay for images it will not show
    _index, _result, tile_pre, tile_overlay = _run(
        qapp, preprocess=PreprocessConfig(enabled=True, exposure=2.0),
        render=False)
    assert tile_overlay is None
    assert tile_pre is not None          # the layer is still handed back
