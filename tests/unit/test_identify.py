"""The identification pipeline: sources, gates, and what they must not do.

The scenes are synthetic so the ground truth is exact: a known blob of a
known colour at a known place, and a substrate that is none of those.
"""

import cv2
import numpy as np
import pytest

from talos.cv.identify import (FAIL_COLOUR, PASS_COLOUR, BorderStage, ColourStage,
                               IdentifyConfig, IdentifyPipeline,
                               MergeStage, MorphologyStage, SharpnessStage,
                               SizeStage, colour_mask, hex_to_hsv,
                               render_overlay, sample_hex, valid_hex)
from talos.models import ObjectiveCalibration, StagePosition

CALIB = ObjectiveCalibration(objective_id=0, um_per_px_x=0.2, um_per_px_y=0.2)
#: purple-grey substrate — deliberately far from every target used here
SUBSTRATE = (105, 100, 120)


def rgb_of_hsv(h: int, s: int = 200, v: int = 210) -> tuple[int, int, int]:
    """An exact RGB colour for an OpenCV hue, so the band edges are known."""
    pixel = np.uint8([[[h, s, v]]])
    return tuple(int(c) for c in cv2.cvtColor(pixel, cv2.COLOR_HSV2RGB)[0, 0])


def hex_of(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def scene(spots, shape=(240, 320), background=SUBSTRATE, seed=0):
    """A frame with rectangular spots of colour: (x, y, w, h, rgb)."""
    rng = np.random.default_rng(seed)
    img = np.full((shape[0], shape[1], 3), background, np.float32)
    img += rng.normal(0, 2, img.shape)
    for x, y, w, h, colour in spots:
        img[y:y + h, x:x + w] = colour
    return np.clip(img, 0, 255).astype(np.uint8)


def config(**overrides) -> IdentifyConfig:
    """Colour-match chain with selected stages replaced."""
    stages = IdentifyConfig().stages
    for name, stage in overrides.items():
        for i, existing in enumerate(stages):
            if existing.NAME == name:
                stages[i] = stage
    return IdentifyConfig(stages=stages)


MAGENTA = rgb_of_hsv(150)


# --- the colour source -----------------------------------------------------

def test_colour_match_finds_the_planted_blob():
    img = scene([(100, 80, 60, 40, MAGENTA)])
    result = IdentifyPipeline().run(
        img, CALIB, config=config(colour=ColourStage(hex_color=hex_of(MAGENTA),
                                                     tolerance=20.0)))
    assert len(result.candidates) == 1
    cand = result.candidates[0]
    assert cand.x_px == pytest.approx(130, abs=3)
    assert cand.y_px == pytest.approx(100, abs=3)
    # 60 × 40 px at 0.2 µm/px = 12 × 8 µm — the contour sits inside by a pixel
    assert cand.area_um2 == pytest.approx(96.0, rel=0.25)


def test_tolerance_decides_what_is_a_match():
    near = rgb_of_hsv(150)
    far = rgb_of_hsv(120)
    img = scene([(60, 60, 50, 40, near), (200, 60, 50, 40, far)])
    tight = IdentifyPipeline().run(
        img, CALIB, config=config(colour=ColourStage(hex_color=hex_of(near),
                                                     tolerance=8.0)))
    wide = IdentifyPipeline().run(
        img, CALIB, config=config(colour=ColourStage(hex_color=hex_of(near),
                                                     tolerance=60.0)))
    assert len(tight.candidates) == 1
    assert len(wide.candidates) >= 2


def test_hue_wraparound_matches_red_on_both_sides_of_the_seam():
    """A red target's tolerance must not be one-sided.

    Hue 0 and hue 179 are neighbours on the wheel; a band computed without
    wraparound silently drops one of them (the reference implementation
    clipped at 0 and missed everything above 179 - tolerance).
    """
    target = rgb_of_hsv(2)
    below = rgb_of_hsv(175)          # 173 away one way, 7 the other
    above = rgb_of_hsv(10)
    img = scene([(60, 60, 50, 40, below), (200, 60, 50, 40, above)])
    result = IdentifyPipeline().run(
        img, CALIB, config=config(
            colour=ColourStage(hex_color=hex_of(target), tolerance=15.0)))
    assert len(result.candidates) == 2

    mask = colour_mask(img, hex_of(target), tolerance=15.0)
    assert mask[80, 80] == 255 and mask[80, 220] == 255


def test_min_saturation_keeps_grey_out_of_a_hue_band():
    """A pixel with no saturation has no meaningful hue: a wide band around
    a red target would otherwise match every grey in the frame."""
    img = scene([(60, 60, 50, 40, (200, 200, 200))])   # grey, not red
    target = hex_of(rgb_of_hsv(0, s=120, v=200))
    loose = IdentifyPipeline().run(
        img, CALIB, config=config(colour=ColourStage(
            hex_color=target, tolerance=60.0, min_saturation=0.0)))
    strict = IdentifyPipeline().run(
        img, CALIB, config=config(colour=ColourStage(
            hex_color=target, tolerance=60.0, min_saturation=60.0)))
    assert len(loose.candidates) == 1
    assert strict.candidates == []


# --- the source is the colour, and only the colour -------------------------

def test_colour_is_the_only_source():
    """Two sources used to OR into one mask, and the second was contrast.
    It answered "what is here at all" rather than "which of these are the
    same material" — on a frame holding one thick flake and one monolayer
    the histogram follows the thick one, and the monolayer falls out of
    the threshold. The flattening it relied on is a pre-processing stage
    now instead (cv/preprocess.py)."""
    kinds = {stage.NAME: stage.KIND for stage in IdentifyConfig().stages}
    assert [name for name, kind in kinds.items() if kind == "source"] == \
        ["colour"]


def test_a_retired_stage_in_a_stored_config_is_dropped_quietly():
    """A settings file written before the chain narrowed must still load —
    and must come back as the current chain, not as a chain plus a ghost."""
    restored = IdentifyConfig.from_dict({"stages": [
        {"name": "contrast", "enabled": True, "blur_sigma": 30.0},
        {"name": "annotation", "enabled": True, "sat_min": 99.0},
        {"name": "colour", "hex_color": "#0a141e"},
    ]})
    assert [s.NAME for s in restored.stages] == [
        s.NAME for s in IdentifyConfig().stages]
    assert restored.stage("colour").hex_color == "#0a141e"


# --- the gates -------------------------------------------------------------

def test_size_gate_uses_physical_units():
    big = (60, 60, 80, 60, MAGENTA)
    small = (220, 60, 12, 12, MAGENTA)      # ~5.7 µm² — below the floor
    img = scene([big, small])
    base = IdentifyPipeline().run(
        img, CALIB, config=config(colour=ColourStage(hex_color=hex_of(MAGENTA),
                                                     tolerance=20.0)))
    with_size = IdentifyPipeline().run(
        img, CALIB, config=config(colour=ColourStage(hex_color=hex_of(MAGENTA),
                                                     tolerance=20.0),
                                  size=SizeStage(min_area_um2=30.0)))
    without = IdentifyPipeline().run(
        img, CALIB, config=config(colour=ColourStage(hex_color=hex_of(MAGENTA),
                                                     tolerance=20.0),
                                  size=SizeStage(enabled=False)))
    assert len(base.candidates) == 1          # the default floor is 30 µm²
    assert len(with_size.candidates) == 1
    assert len(without.candidates) == 2


def test_border_gate_rejects_a_blob_on_the_frame_edge():
    img = scene([(0, 60, 40, 40, MAGENTA)])   # flush with the left edge
    stage = ColourStage(hex_color=hex_of(MAGENTA), tolerance=20.0)
    loose = IdentifyPipeline().run(
        img, CALIB, config=config(colour=stage,
                                  border=BorderStage(enabled=False)))
    strict = IdentifyPipeline().run(
        img, CALIB, config=config(colour=stage, border=BorderStage(margin_px=4)))
    assert len(loose.candidates) == 1
    assert strict.candidates == []


def test_sharpness_gate_separates_an_edge_from_a_smudge():
    """A soft, defocused blob is not a crystal. The gate scores the mean
    gradient on the candidate's own boundary.

    The cutoff is measured here rather than hardcoded: what matters is that
    a sharp boundary outscores a defocused one and that a threshold between
    the two separates them.
    """
    sharp = scene([(60, 60, 60, 40, MAGENTA)])
    smudge = cv2.GaussianBlur(sharp, (0, 0), 5.0)
    stage = ColourStage(hex_color=hex_of(MAGENTA), tolerance=20.0)
    ungated = config(colour=stage,
                     sharpness=SharpnessStage(min_edge_strength=0.0))
    sharp_score = IdentifyPipeline().run(
        sharp, CALIB, config=ungated).candidates[0].score
    smudge_score = IdentifyPipeline().run(
        smudge, CALIB, config=ungated).candidates[0].score
    assert sharp_score > smudge_score * 1.5

    cutoff = (sharp_score + smudge_score) / 2.0
    gated = config(colour=stage,
                   sharpness=SharpnessStage(min_edge_strength=cutoff))
    assert len(IdentifyPipeline().run(sharp, CALIB, config=gated).candidates) == 1
    assert IdentifyPipeline().run(smudge, CALIB, config=gated).candidates == []


def test_a_red_region_is_a_sample_not_a_scale_bar():
    """A saturated red bar used to be rejected here as the app's own
    annotation. The burn-in only ever touches snapshot copies inside the
    camera backend — the live stream, the frame slot and every scan tile
    are raw — so there is nothing on this path to reject, and an operator
    looking for a genuinely red sample gets it."""
    img = scene([(200, 190, 90, 20, (255, 0, 0))])
    stage = ColourStage(hex_color="#ff0000", tolerance=20.0)
    result = IdentifyPipeline().run(img, CALIB, config=config(colour=stage))
    assert len(result.candidates) == 1


def test_merge_joins_two_fragments_of_one_sample():
    """One sample often fragments into adjacent blobs at the segmentation
    threshold; they are one object, not two."""
    img = scene([(60, 60, 40, 40, MAGENTA), (110, 60, 40, 40, MAGENTA)])
    stage = ColourStage(hex_color=hex_of(MAGENTA), tolerance=20.0)
    merged = IdentifyPipeline().run(img, CALIB, config=config(colour=stage))
    split = IdentifyPipeline().run(
        img, CALIB, config=config(colour=stage,
                                  merge=MergeStage(enabled=False)))
    assert len(split.candidates) == 2
    assert len(merged.candidates) == 1
    assert merged.candidates[0].area_um2 > split.candidates[0].area_um2


def test_no_source_enabled_finds_nothing():
    img = scene([(60, 60, 60, 40, MAGENTA)])
    stages = IdentifyConfig().stages
    for stage in stages:
        if stage.KIND == "source":
            stage.enabled = False
    result = IdentifyPipeline().run(img, CALIB,
                                    config=IdentifyConfig(stages=stages))
    assert result.candidates == []
    assert not result.mask.any()


# --- the chain's readout ---------------------------------------------------

def test_counts_report_what_each_stage_let_through():
    img = scene([(60, 60, 60, 40, MAGENTA)])
    result = IdentifyPipeline().run(
        img, CALIB, config=config(colour=ColourStage(hex_color=hex_of(MAGENTA),
                                                     tolerance=20.0)))
    labels = [label for label, _in, _out in result.counts]
    assert labels[0] == "Colour match"
    assert "Size" in labels and "Merge fragments" in labels
    # the summary reads like the chain
    assert result.summary.startswith("Colour match ")
    by_label = {label: (i, o) for label, i, o in result.counts}
    assert by_label["Size"][0] >= by_label["Size"][1]


# --- resolution independence ----------------------------------------------

def test_results_are_the_same_at_preview_scale():
    """The live preview runs on a downscaled frame. A parameter tuned there
    must mean the same thing on a full-resolution tile — and a candidate
    must come back in the caller's pixels, not the preview's."""
    img = scene([(100, 80, 60, 40, MAGENTA)], shape=(480, 640))
    stage = ColourStage(hex_color=hex_of(MAGENTA), tolerance=20.0)
    cfg = config(colour=stage)
    full = IdentifyPipeline().run(img, CALIB, config=cfg, scale=1.0)
    half = IdentifyPipeline().run(img, CALIB, config=cfg, scale=0.5)

    assert len(full.candidates) == len(half.candidates) == 1
    assert half.candidates[0].x_px == pytest.approx(
        full.candidates[0].x_px, abs=3.0)
    assert half.candidates[0].area_um2 == pytest.approx(
        full.candidates[0].area_um2, rel=0.2)
    assert half.scale == 0.5


def test_x_um_y_um_come_from_the_stage_position():
    """The table's X/Y columns used to print 0.0 for every row: the
    detector never filled them."""
    img = scene([(100, 80, 60, 40, MAGENTA)], shape=(480, 640))
    stage_pos = StagePosition(x_um=1000.0, y_um=2000.0)
    result = IdentifyPipeline().run(
        img, CALIB, config=config(colour=ColourStage(hex_color=hex_of(MAGENTA),
                                                     tolerance=20.0)),
        stage_pos=stage_pos)
    cand = result.candidates[0]
    # Image centre (320, 240) is the stage position. Right of centre is
    # +X, but the mounting runs Y the other way (bench-measured, see
    # cv/orientation.py), so a feature ABOVE centre is at a LARGER stage Y.
    assert cand.x_um == pytest.approx(1000.0 + (130 - 320) * 0.2, abs=1.0)
    assert cand.y_um == pytest.approx(2000.0 + (240 - 100) * 0.2, abs=1.0)


def test_the_camera_flip_mirrors_the_stage_mapping():
    """With the flip on, the frame is rotated 180° about its centre, so a
    feature RIGHT of the centre is at a SMALLER stage X. Ignoring the flip
    (which this mapping did) sent "go to sample" to the mirrored position.
    """
    img = scene([(100, 80, 60, 40, MAGENTA)], shape=(480, 640))
    stage_pos = StagePosition(x_um=1000.0, y_um=2000.0)
    cfg = config(colour=ColourStage(hex_color=hex_of(MAGENTA), tolerance=20.0))
    straight = IdentifyPipeline().run(img, CALIB, config=cfg,
                                      stage_pos=stage_pos, flip=False)
    flipped = IdentifyPipeline().run(img, CALIB, config=cfg,
                                     stage_pos=stage_pos, flip=True)
    a, b = straight.candidates[0], flipped.candidates[0]
    assert b.x_um == pytest.approx(2 * 1000.0 - a.x_um, abs=0.5)
    assert b.y_um == pytest.approx(2 * 2000.0 - a.y_um, abs=0.5)
    # the pixel position is the same; only the stage coordinate mirrors
    assert (b.x_px, b.y_px) == pytest.approx((a.x_px, a.y_px))


# --- config round-trip -----------------------------------------------------

def test_config_round_trips_through_a_settings_dict():
    original = IdentifyConfig()
    original.stage("colour").hex_color = "#123456"
    original.stage("size").min_area_um2 = 12.5
    original.stage("sharpness").enabled = False
    restored = IdentifyConfig.from_dict(original.to_dict())
    assert restored.stage("colour").hex_color == "#123456"
    assert restored.stage("size").min_area_um2 == 12.5
    assert restored.stage("sharpness").enabled is False
    assert [s.NAME for s in restored.stages] == [s.NAME
                                                 for s in original.stages]


def test_config_survives_a_damaged_settings_file():
    """A hand-edited file must not be able to stop the pipeline running."""
    restored = IdentifyConfig.from_dict({"stages": [
        {"name": "colour", "hex_color": "not a colour", "tolerance": 9999},
        {"name": "gone-in-v2", "enabled": True},
        {"name": "size", "min_area_um2": "big"},
        "junk",
    ]})
    assert [s.NAME for s in restored.stages] == [
        s.NAME for s in IdentifyConfig().stages]
    assert restored.stage("colour").hex_color == "#c8a2c8"   # the fallback
    assert restored.stage("size").min_area_um2 == 30.0


def test_valid_hex_normalises_and_falls_back():
    assert valid_hex("#AABBCC") == "#aabbcc"
    assert valid_hex("abc") == "#aabbcc"
    assert valid_hex("nope") == "#c8a2c8"
    assert valid_hex(None) == "#c8a2c8"


def test_hex_to_hsv_agrees_with_the_frame_conversion():
    assert hex_to_hsv("#ff0000")[0] == 0
    assert hex_to_hsv("#00ff00")[0] == 60
    assert hex_to_hsv("#0000ff")[0] == 120


# --- the dropper -----------------------------------------------------------

def test_sample_hex_averages_a_circular_patch():
    img = np.zeros((40, 40, 3), np.uint8)
    img[18:23, 18:23] = (200, 100, 50)
    assert sample_hex(img, 20, 20, radius=2) == "#c86432"


def test_sample_hex_ignores_the_diagonal_corners():
    """A circle, not a square. A square's four corners are the pixels most
    likely to belong to whatever is diagonally adjacent — on a flake edge
    that is the substrate, and the pick lands between the two materials
    instead of on one of them."""
    img = np.zeros((40, 40, 3), np.uint8)
    img[18:23, 18:23] = (100, 100, 100)
    for corner in ((18, 18), (18, 22), (22, 18), (22, 22)):
        img[corner] = (255, 0, 0)                  # only the corners differ
    # a square average would be (21*100 + 4*255) / 25 = 125, not 100
    assert sample_hex(img, 20, 20, radius=2) == "#646464"


def test_sample_hex_refuses_a_point_off_the_frame():
    img = np.zeros((10, 10, 3), np.uint8)
    assert sample_hex(img, 5, 40) is None
    assert sample_hex(img, -1, 5) is None


# --- the processed view ----------------------------------------------------

def test_render_overlay_darkens_everything_but_the_match():
    """The processed view: the frame is darkened EXCEPT where the sources
    matched, so the sample keeps its own pixels inside the match."""
    img = scene([(100, 80, 60, 40, MAGENTA)])
    result = IdentifyPipeline().run(
        img, CALIB, config=config(colour=ColourStage(hex_color=hex_of(MAGENTA),
                                                     tolerance=20.0)))
    out = render_overlay(img, result, darken=0.75)
    assert out.shape == img.shape and out.dtype == np.uint8
    # outside the match: darkened
    assert int(out[5, 5, 0]) < int(img[5, 5, 0])
    assert int(out[5, 5, 0]) == pytest.approx(int(img[5, 5, 0]) * 0.25, abs=3)
    # inside it: the frame's own pixels, so the material is still visible
    assert np.array_equal(out[100, 130], img[100, 130])


def test_render_overlay_outlines_pass_and_fail_differently():
    """Rectangles are gone on purpose: the blob's own outline carries the
    verdict, bright for what survived the chain and dim for what a gate
    threw away."""
    big = (60, 60, 90, 60, MAGENTA)
    small = (240, 60, 14, 14, MAGENTA)      # below the size floor
    img = scene([big, small])
    cfg = config(colour=ColourStage(hex_color=hex_of(MAGENTA), tolerance=20.0))
    result = IdentifyPipeline().run(img, CALIB, config=cfg)
    assert len(result.candidates) == 1
    assert len(result.regions) == 2
    assert sum(1 for r in result.regions if r.passed) == 1

    out = render_overlay(img, result)
    # the surviving blob's boundary is the bright colour, the rejected
    # blob's is the dim one — both ON the darkened background
    passed = [r for r in result.regions if r.passed][0]
    failed = [r for r in result.regions if not r.passed][0]
    px, py = passed.contour[0][0]
    fx, fy = failed.contour[0][0]
    assert tuple(int(v) for v in out[py, px]) == PASS_COLOUR
    assert tuple(int(v) for v in out[fy, fx]) == FAIL_COLOUR


def test_regions_survive_a_preview_downscale():
    """The contours are drawn over the CALLER's frame, so they take the
    same trip back from the preview resolution as the candidates."""
    img = scene([(100, 80, 60, 40, MAGENTA)], shape=(480, 640))
    result = IdentifyPipeline().run(
        img, CALIB, config=config(colour=ColourStage(hex_color=hex_of(MAGENTA),
                                                     tolerance=20.0)),
        scale=0.5)
    assert len(result.regions) == 1
    xs = result.regions[0].contour[:, 0, 0]
    cand = result.candidates[0]
    bx, by, bw, bh = cand.bbox
    # the contour straddles the candidate's box, in the CALLER's pixels
    # (at 0.5 scale it would sit at half these coordinates)
    assert xs.min() == pytest.approx(bx, abs=3)
    assert xs.max() == pytest.approx(bx + bw, abs=3)
    out = render_overlay(img, result)
    assert out.shape == img.shape
