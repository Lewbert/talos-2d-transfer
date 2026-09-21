"""Identification at two resolutions of the same field of view.

A scan can capture at 1080p or at 4K (Preferences → Scan). Both cover the
same µm, so a parameter the operator set as a length has to mean the same
distance on the sample in either — otherwise every threshold silently
changes meaning when a scan is switched to 4K, which is the class of bug
that a single-resolution build cannot show you.

The reference is the frame the operator tunes on (the live preview), and
the pipeline is TOLD how much more finely a tile samples it. It is never
inferred from the frame width: a 640×480 camera and a 4K one are not the
same field of view, and guessing would move the thresholds of every setup
that is not the one on this bench.
"""

import cv2
import numpy as np
import pytest

from talos.cv.identify import (BorderStage, ColourStage, IdentifyConfig,
                               IdentifyPipeline, MergeStage, SharpnessStage,
                               SizeStage, _Ctx)
from talos.models import ObjectiveCalibration


def _ctx(width: int, height: int, scale: float = 1.0, um_per_px=(1.0, 1.0),
         frame_scale: float = 1.0):
    work = np.zeros((int(height * scale), int(width * scale), 3), np.uint8)
    return _Ctx(work, (um_per_px[0] / scale, um_per_px[1] / scale), scale,
                frame_scale)


def test_the_default_frame_scale_changes_nothing():
    """Every path that does not capture at two resolutions gets exactly the
    arithmetic it had before there was a frame_scale at all."""
    ctx = _ctx(960, 540, scale=0.5)
    assert ctx.frame_scale == 1.0
    assert ctx.ref_px(4) == pytest.approx(4 * 0.5)     # the old value*scale
    assert ctx.ref_grad(4.0) == pytest.approx(4 * 0.5)


def test_a_length_covers_the_same_distance_at_either_resolution():
    """4 px at 1080p and 8 px at 4K are the same distance on the sample:
    the tile's µm/px is half, so the pixel count doubles."""
    ref_1080 = _ctx(1920, 1080, um_per_px=(1.0, 1.0), frame_scale=1.0)
    tile_4k = _ctx(3840, 2160, um_per_px=(0.5, 0.5), frame_scale=2.0)
    assert ref_1080.ref_px(4) == pytest.approx(4.0)
    assert tile_4k.ref_px(4) == pytest.approx(8.0)
    assert ref_1080.ref_px(4) * 1.0 == pytest.approx(tile_4k.ref_px(4) * 0.5)


def test_a_gradient_threshold_scales_the_other_way():
    """Twice the pixels across one edge means half the per-pixel gradient
    for the same physical edge, so the threshold divides."""
    assert _ctx(1920, 1080).ref_grad(4.0) == pytest.approx(4.0)
    assert _ctx(3840, 2160, frame_scale=2.0).ref_grad(4.0) == pytest.approx(2.0)


# ----------------------------------------------------------------------
# the same scene, two resolutions, the same verdict
# ----------------------------------------------------------------------

#: A square in a saturated colour the colour stage can match.
_FLAKE_RGB = (200, 90, 150)


def _scene(width: int, height: int, x0: int, size: int) -> np.ndarray:
    img = np.zeros((height, width, 3), np.uint8)
    img[size:2 * size, x0:x0 + size] = _FLAKE_RGB    # a band, not a speck
    return img


def _calib(um_per_px: float) -> ObjectiveCalibration:
    return ObjectiveCalibration(objective_id=0, um_per_px_x=um_per_px,
                                um_per_px_y=um_per_px)


def _config(*extra):
    """Colour source, a neutral gate, then whatever the test is about.

    The neutral gate is not decoration: the pipeline only turns the mask
    into candidates when it reaches a gate or merge stage, so a
    sources-only config returns none whatever the mask says.
    """
    return IdentifyConfig(stages=[
        ColourStage(hex_color="#c85a96", tolerance=60.0, min_saturation=40.0),
        SizeStage(min_area_um2=0.0, max_area_um2=1_000_000.0),
        *extra,
    ])


def _run(img, calib, config, frame_scale=1.0):
    return IdentifyPipeline().run(img, calib, config=config, scale=1.0,
                                  frame_scale=frame_scale)


def _upscale(img):
    """The same scene as a 4K frame — same field of view, twice the pixels."""
    return cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2),
                      interpolation=cv2.INTER_NEAREST)


def test_a_flake_touching_the_frame_edge_is_judged_the_same_at_4k():
    """The discriminating case: near the edge at 1080p (rejected) is near
    the edge at 4K (rejected). A margin left in raw pixels would accept it
    at 4K, because 6 px looks further from the border than 3 px does."""
    config = _config(BorderStage(margin_px=4))
    near = _scene(1920, 1080, x0=3, size=30)
    assert not _run(near, _calib(0.7), config).candidates
    assert not _run(_upscale(near), _calib(0.35), config,
                    frame_scale=2.0).candidates, \
        "4K: the margin stopped meaning µm"

    # ... and the same at the other end: 3 px from the edge in 1080p terms
    # is rejected at BOTH resolutions, which is the property that matters.
    verdict_1080 = bool(_run(near, _calib(0.7), config).candidates)
    verdict_4k = bool(_run(_upscale(near), _calib(0.35), config,
                           frame_scale=2.0).candidates)
    assert verdict_1080 == verdict_4k

    # well inside, both keep it
    inside = _scene(1920, 1080, x0=40, size=30)
    assert _run(inside, _calib(0.7), config).candidates
    assert _run(_upscale(inside), _calib(0.35), config,
                frame_scale=2.0).candidates


def test_the_same_flake_measures_the_same_area_at_4k():
    """µm² comes from the calibration the frame is handed — the per-frame
    calibration the workspace supplies. The 4K frame is the same scene with
    twice the pixels, so its µm/px is half."""
    small = _scene(1920, 1080, x0=100, size=40)
    large = _upscale(small)
    config = _config()
    area_1080 = _run(small, _calib(0.7), config).candidates[0].area_um2
    area_4k = _run(large, _calib(0.35), config,
                   frame_scale=2.0).candidates[0].area_um2
    # 40 x 40 px at 0.7 µm/px (cv2.contourArea counts pixel centres, so the
    # measured square is one pixel narrower than the drawn one)
    assert area_1080 == pytest.approx(40 * 40 * 0.7 * 0.7, rel=0.05)
    # the two agree to within the contour convention (one pixel of
    # half-width is a bigger fraction of a 40 px square than of an 80 px
    # one), which is 2.5 % here and not a resolution effect
    assert area_4k == pytest.approx(area_1080, rel=0.04)


def test_the_merge_gap_is_a_distance_not_a_pixel_count():
    """Two fragments one gap apart merge at either resolution."""
    config = _config(MergeStage(gap_px=20))

    def scene(width, height, part, gap_px, size):
        img = np.zeros((height, width, 3), np.uint8)
        img[size:2 * size, part:part + size] = _FLAKE_RGB
        img[size:2 * size,
            part + size + gap_px:part + 2 * size + gap_px] = _FLAKE_RGB
        return img

    small = scene(1920, 1080, 100, 20, 30)      # 20 reference px apart
    assert len(_run(small, _calib(0.7), config).candidates) == 1
    assert len(_run(_upscale(small), _calib(0.35), config,
                    frame_scale=2.0).candidates) == 1


def test_the_sharpness_gate_does_not_move_with_the_resolution():
    """The score is per-pixel gradient, so the same scene scores half as
    much at 4K — the threshold has to move with it or a 4K scan would
    reject flakes the 1080p scan accepts."""
    small = _scene(1920, 1080, x0=200, size=40)
    large = _upscale(small)
    config = _config(SharpnessStage(min_edge_strength=4.0))
    kept_1080 = _run(small, _calib(0.7), config).candidates
    kept_4k = _run(large, _calib(0.35), config, frame_scale=2.0).candidates
    assert kept_1080 and kept_4k
    # the score is a gradient magnitude, so it is NOT the same number —
    # what has to match is the verdict
    assert kept_1080[0].score == pytest.approx(kept_4k[0].score, rel=0.25)


# ----------------------------------------------------------------------
# who supplies the ratio
# ----------------------------------------------------------------------


class _WorkspaceStub:
    """Just the two attributes ``frame_scale_for`` reads (the real method —
    this is the production rule, not a copy of it)."""

    def __init__(self, reference_width):
        self._reference_width = reference_width


def _scale_for(reference_width, width):
    from talos.ui.workspaces.sample_finding import SampleFindingWorkspace

    frame = np.zeros((width * 9 // 16, width, 3), np.uint8)
    return SampleFindingWorkspace.frame_scale_for(
        _WorkspaceStub(reference_width), frame)


def test_the_scale_is_the_ratio_between_the_tile_and_the_tuned_frame():
    assert _scale_for(1920, 1920) == pytest.approx(1.0)   # the same frame
    assert _scale_for(1920, 3840) == pytest.approx(2.0)   # 4K tile
    assert _scale_for(3840, 1920) == pytest.approx(0.5)   # coarser tile
    assert _scale_for(0, 3840) == pytest.approx(1.0)      # nothing latched


def test_the_reference_is_the_frame_before_a_scan_switches_the_stream():
    """During a 4K run the live stream is 4K too, so the reference cannot
    be read from whatever is streaming at that moment — it is latched from
    before the run, which is what the operator was looking at."""
    from talos.ui.workspaces.sample_finding import SampleFindingWorkspace

    class _State:
        mode = "MANUAL"

    stub = _WorkspaceStub(0)
    stub._state = _State()
    frame_1080 = np.zeros((1080, 1920, 3), np.uint8)
    SampleFindingWorkspace.on_frame(stub, frame_1080)
    assert stub._reference_width == 1920

    _State.mode = "SCAN"                      # a run is walking the axes
    SampleFindingWorkspace.on_frame(stub, np.zeros((2160, 3840, 3), np.uint8))
    assert stub._reference_width == 1920, "the reference moved with the run"
    assert _scale_for(stub._reference_width, 3840) == pytest.approx(2.0)
