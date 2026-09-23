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
from talos.cv.preprocess import LocalContrast, PreprocessConfig, build_lut
from talos.models import ObjectiveCalibration
from talos.ui.detect_engine import DetectJob, _DetectWorker

CALIB = ObjectiveCalibration(objective_id=0, um_per_px_x=0.2, um_per_px_y=0.2)

#: The frame: a 40 x 40 blob on a flat substrate. The blob is NOT the
#: picked colour until the chain runs on it — the curve is centred on the
#: substrate, which leaves the substrate pinned and pushes the blob 45 DN
#: away from what it was — so finding it at all proves the chain ran
#: before the mask, on the same pixels the mask then searched.
SUBSTRATE = (60, 90, 120)
RAW_BLOB = (100, 60, 60)

#: The curve, and the colour the blob comes out as. Read from the LUT the
#: worker itself builds, so the test cannot drift from the implementation.
CURVE_GAIN, CURVE_WIDTH = 4.0, 64.0


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _frame() -> np.ndarray:
    img = np.full((240, 320, 3), SUBSTRATE, np.uint8)
    img[80:120, 100:140] = RAW_BLOB
    return img


def _preprocess() -> PreprocessConfig:
    cfg = PreprocessConfig(enabled=True)
    cfg.local = LocalContrast(enabled=True, gain=CURVE_GAIN,
                              width=CURVE_WIDTH)
    return cfg


def _filtered_blob() -> tuple:
    lut = build_lut(_preprocess(), SUBSTRATE)
    return tuple(int(lut[channel][RAW_BLOB[channel]]) for channel in range(3))


def _picked() -> str:
    return "#%02x%02x%02x" % _filtered_blob()


def _config() -> IdentifyConfig:
    stages = IdentifyConfig().stages
    for index, stage in enumerate(stages):
        if stage.NAME == "colour":
            stages[index] = ColourStage(hex_color=_picked(), tolerance=10.0)
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
    """The blob only reaches the picked colour AFTER the curve is applied —
    so finding it at all proves the ordering."""
    _index, result, _pre, _overlay, _token = _run(
        qapp, preprocess=_preprocess(), colour=SUBSTRATE)
    assert len(result.candidates) == 1
    cand = result.candidates[0]
    assert cand.x_px == pytest.approx(120.0, abs=4)
    assert cand.y_px == pytest.approx(100.0, abs=4)


def test_without_the_chain_the_same_frame_finds_nothing(qapp):
    """The control for the test above: same frame, same mask, no chain."""
    _index, result, preprocessed, _overlay, _token = _run(
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
    _index, _result, preprocessed, overlay, _token = _run(
        qapp, preprocess=_preprocess(), colour=SUBSTRATE)
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
    picked = _filtered_blob()
    img = _frame()
    img[80:120, 100:140] = picked                 # already the picked colour
    pre = _preprocess()
    _index, result, _pre, _overlay, _token = _run(qapp, frame=img, preprocess=pre,
                                          colour=picked)
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

    The "and it really ran" half is not decoration: this test used to build
    a ``PreprocessConfig(brightness=…)``, a field that stopped existing when
    the tone operations were removed, so the source raised, the tick caught
    it and returned, and both assertions passed without any chain work at
    all — including if the chain had moved onto the GUI thread.
    """
    import time as _time

    from talos.cv import preprocess as pre
    from talos.ui.detect_engine import DetectionEngine, LIVE_FULL

    original = pre.apply
    ran: list = []

    def slow(img, cfg, centre=None):
        ran.append(1)               # this runs on the WORKER, not the tick
        _time.sleep(0.3)
        return original(img, cfg, centre)

    engine = DetectionEngine(interval_ms=30)
    try:
        engine.set_live_level(LIVE_FULL)
        engine.set_source(lambda: (_frame(), CALIB, None, _config(), 0.5,
                                   False, _preprocess(), SUBSTRATE))
        pre.apply = slow
        started = _time.perf_counter()
        engine._tick()
        elapsed = _time.perf_counter() - started
        assert elapsed < 0.05, \
            f"the tick blocked for {elapsed:.3f}s — the chain is on the GUI thread"

        deadline = _time.monotonic() + 3.0
        while _time.monotonic() < deadline and not ran:
            qapp.processEvents()
            _time.sleep(0.01)
        assert ran, "the chain never ran — this test passed vacuously"
    finally:
        pre.apply = original
        engine.shutdown()


def test_the_worker_emits_both_views_from_one_transform(qapp):
    """Both processed views and the mask come from ONE array, so the
    operator can only tune against pixels the pipeline also saw."""
    _index, result, preprocessed, overlay, _token = _run(
        qapp, preprocess=_preprocess(), colour=SUBSTRATE)
    assert preprocessed is not None and overlay is not None
    assert result.mask is not None
    # the overlay is the pre-processed frame, annotated
    assert overlay.shape == preprocessed.shape == _frame().shape
    # a tile job asks for no preview at all (render=False): the scan does
    # not pay for images it will not show
    _index, _result, tile_pre, tile_overlay, _token = _run(
        qapp, preprocess=_preprocess(), colour=SUBSTRATE, render=False)
    assert tile_overlay is None
    assert tile_pre is not None          # the layer is still handed back


def test_a_suspended_engine_stops_sampling_but_finishes_its_tiles(qapp):
    """A scan suspends the live feed so its tiles are not queued behind
    previews — and the tiles already submitted still drain."""
    from talos.ui.detect_engine import DetectionEngine

    engine = DetectionEngine(interval_ms=30)
    try:
        calls: list = []
        engine.set_source(lambda: calls.append(1) or None)
        engine.set_live(True)
        engine.set_suspended(True)
        deadline = time.monotonic() + 0.4
        while time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert calls == [], "the live feed sampled while suspended"

        engine.set_suspended(False)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not calls:
            qapp.processEvents()
            time.sleep(0.01)
        assert calls, "the live feed did not come back"
    finally:
        engine.shutdown()


def test_a_preprocess_only_job_skips_the_pipeline(qapp):
    """The Pre-processed view shows the chain's output and nothing of the
    mask: running the identification for it is work for nobody."""
    from talos.ui.detect_engine import LIVE_PREPROCESS

    index, result, preprocessed, overlay, _token = _run(
        qapp, preprocess=_preprocess(), colour=SUBSTRATE,
        level=LIVE_PREPROCESS)
    assert index < 0
    assert result is None and overlay is None
    assert preprocessed is not None
    assert preprocessed.shape == _frame().shape
    assert preprocessed is not None and not np.array_equal(preprocessed,
                                                           _frame())


def test_a_job_that_raises_still_frees_its_slot(qapp):
    """A live job that raised used to leave ``_busy`` set for the rest of
    the session — the previews never came back — and a tile job that raised
    left ``pending_tiles`` above zero for ever, which also held the run's
    exports hostage: they wait for that number to reach zero, so a run whose
    last tile raised wrote no mosaic, no annotated mosaic and no CSV."""
    from talos.cv import preprocess as pre
    from talos.ui.detect_engine import DetectionEngine

    original = pre.apply

    def boom(*args, **kwargs):
        raise RuntimeError("no frame for you")

    engine = DetectionEngine(interval_ms=30)
    try:
        pre.apply = boom
        engine.set_source(lambda: (_frame(), CALIB, None, _config(), 0.5,
                                   False, _preprocess(), SUBSTRATE))
        engine.set_live(True)
        engine._tick()
        engine.submit_tile(3, _frame(), CALIB, None, _config(),
                           preprocess=_preprocess(), colour=SUBSTRATE)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and (engine.busy
                                               or engine.pending_tiles):
            qapp.processEvents()
            time.sleep(0.01)
        assert not engine.busy, "the live slot latched busy for the session"
        assert engine.pending_tiles == 0, "the tile slot never drained"
    finally:
        pre.apply = original
        engine.shutdown()


def test_a_previous_run_s_tile_is_not_filed_as_the_next_run_s(qapp,
                                                              monkeypatch):
    """Detection outlives the capture by design: a run that was aborted
    leaves tiles in the queue, and the operator pressing Scan again starts
    the next run immediately. Those stragglers arrive with indices the new
    run also uses — listed as its samples, at its positions, and ringed on
    its annotated mosaic."""
    from talos.cv import preprocess as pre
    from talos.ui.detect_engine import DetectionEngine

    original = pre.apply

    def slow(img, cfg, centre=None):
        time.sleep(0.3)
        return original(img, cfg, centre)

    engine = DetectionEngine(interval_ms=1000)
    delivered: list = []
    engine.sig_result.connect(lambda *args: delivered.append(args))
    logs: list = []
    engine.sig_log.connect(logs.append)
    monkeypatch.setattr(pre, "apply", slow)
    try:
        engine.begin_run()                     # run 1
        engine.submit_tile(3, _frame(), CALIB, None, _config(),
                           preprocess=_preprocess(), colour=SUBSTRATE)
        engine.begin_run()                     # run 2 starts at once
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and engine.pending_tiles:
            qapp.processEvents()
            time.sleep(0.02)
        assert engine.pending_tiles == 0, "the job never drained"
        assert not [args for args in delivered if args[0] >= 0], \
            "the previous run's tile was delivered as this run's"
        assert any("after its run ended" in line for line in logs), logs
    finally:
        engine.shutdown()


def test_a_tile_from_the_current_run_is_delivered(qapp):
    """The control for the test above: the filter must not eat live results."""
    from talos.ui.detect_engine import DetectionEngine

    engine = DetectionEngine(interval_ms=1000)
    delivered: list = []
    engine.sig_result.connect(lambda *args: delivered.append(args))
    try:
        engine.begin_run()
        engine.submit_tile(3, _frame(), CALIB, None, _config(),
                           preprocess=_preprocess(), colour=SUBSTRATE)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not delivered:
            qapp.processEvents()
            time.sleep(0.02)
        assert [args[0] for args in delivered] == [3]
    finally:
        engine.shutdown()
