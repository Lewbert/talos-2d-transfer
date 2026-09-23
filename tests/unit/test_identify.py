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
from talos.models import (FlakeCandidate, ObjectiveCalibration,
                          StagePosition)

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


def _ctx_for(img):
    """A minimal _Ctx, so a stage can be asked for its own mask."""
    from talos.cv.identify import _Ctx

    return _Ctx(img, (0.2, 0.2), 1.0)


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
    a red target would otherwise match every grey in the frame.

    Both stages carry ``spread`` as well as ``tolerance`` now, and they have
    to: the grey sits 120 units below the target's saturation, so a shade
    window that only went ±64 would reject it for the wrong reason and this
    test would pass without exercising ``min_saturation`` at all.
    """
    img = scene([(60, 60, 50, 40, (200, 200, 200))])   # grey, not red
    target = hex_of(rgb_of_hsv(0, s=120, v=200))
    loose = IdentifyPipeline().run(
        img, CALIB, config=config(colour=ColourStage(
            hex_color=target, tolerance=60.0, spread=60.0,
            min_saturation=0.0)))
    strict = IdentifyPipeline().run(
        img, CALIB, config=config(colour=ColourStage(
            hex_color=target, tolerance=60.0, spread=60.0,
            min_saturation=60.0)))
    assert len(loose.candidates) == 1
    assert strict.candidates == []


def test_the_shade_spread_is_what_separates_two_layers_of_one_material():
    """The bench report: a dark-blue monolayer target that also accepted the
    lighter layer.

    Dark blue is H113 S201 V132 and the light layer is H106 S78 V245 — 7° of
    hue apart, ~120 units of saturation and value. The hue band cannot
    separate them at any sane tolerance; the shade window is the control
    that can, which is why it has its own number.
    """
    dark = (28, 52, 132)          # the monolayer
    light = (170, 205, 245)       # the layer above it
    img = scene([(60, 60, 50, 40, dark), (200, 60, 50, 40, light)])
    target = hex_of(dark)

    wide = IdentifyPipeline().run(
        img, CALIB, config=config(colour=ColourStage(
            hex_color=target, tolerance=25.0, spread=60.0,
            min_saturation=0.0)))
    narrow = IdentifyPipeline().run(
        img, CALIB, config=config(colour=ColourStage(
            hex_color=target, tolerance=25.0, spread=20.0,
            min_saturation=0.0)))
    # ...and the shipped defaults already separate them, which is why the
    # bench sighting was either a wider shade window or a pick that was not
    # the monolayer at all (see the dropper's patch warning).
    default = IdentifyPipeline().run(
        img, CALIB, config=config(colour=ColourStage(hex_color=target)))

    assert len(wide.candidates) == 2, "the reported symptom, as it behaves"
    assert len(narrow.candidates) == 1, "the dark layer only"
    assert len(default.candidates) == 1
    assert narrow.candidates[0].x_px == pytest.approx(85, abs=6)


def test_the_shade_window_follows_the_spread_not_the_hue_tolerance():
    """Turning up the hue tolerance must not open the shade window: that is
    the whole reason the two numbers exist."""
    dark = (28, 52, 132)
    light = (170, 205, 245)
    img = scene([(60, 60, 50, 40, dark), (200, 60, 50, 40, light)])
    target = hex_of(dark)
    for hue_tolerance in (5.0, 25.0, 60.0, 100.0):
        result = IdentifyPipeline().run(
            img, CALIB, config=config(colour=ColourStage(
                hex_color=target, tolerance=hue_tolerance, spread=20.0,
                min_saturation=0.0)))
        assert len(result.candidates) == 1, f"tolerance {hue_tolerance}"


def test_the_picked_colour_is_always_inside_its_own_mask():
    """A mask that excludes the colour just clicked reads as a broken
    detector — and it is silent. A floor above the target's own saturation
    or value used to do exactly that."""
    for rgb in [(28, 52, 132), (200, 205, 210), (30, 120, 200), (250, 250, 250),
                (60, 60, 60), (150, 155, 160)]:
        img = np.full((8, 8, 3), rgb, np.uint8)
        for spread in (0.0, 15.0, 50.0):
            for min_sat in (0.0, 40.0, 120.0, 200.0):
                for min_val in (0.0, 60.0, 150.0, 240.0):
                    mask = colour_mask(img, hex_of(rgb), 25.0, spread,
                                       min_sat, min_val)
                    assert mask.all(), (
                        f"the pick {rgb} is outside its own mask "
                        f"(spread {spread}, min_sat {min_sat}, "
                        f"min_val {min_val})")


def test_a_stored_colour_stage_keeps_its_exact_window():
    """The split must not move a colour someone tuned on the bench: a file
    written before ``spread`` existed carried both meanings in ``tolerance``,
    and it is folded in unchanged (same expression, same rounding)."""
    stored = {"stages": [{"name": "colour", "enabled": True,
                          "hex_color": "#1c3484", "tolerance": 50.0,
                          "min_saturation": 40.0, "min_value": 0.0}]}
    stage = IdentifyConfig.from_dict(stored).stage("colour")
    assert stage.spread == 50.0
    img = np.full((4, 4, 3), (28, 52, 132), np.uint8)
    assert np.array_equal(colour_mask(img, "#1c3484", 50.0, 50.0, 40.0, 0.0),
                          stage.source_mask(_ctx_for(img)))
    # an explicit spread in the file wins over the fold
    stored["stages"][0]["spread"] = 12.0
    assert IdentifyConfig.from_dict(stored).stage("colour").spread == 12.0


def test_a_colour_stage_round_trips_with_its_spread():
    stage = ColourStage(hex_color="#123456", tolerance=30.0, spread=12.0)
    back = IdentifyConfig.from_dict(IdentifyConfig(stages=[stage]).to_dict())
    again = back.stage("colour")
    assert again.tolerance == 30.0 and again.spread == 12.0


# --- what the dropper sampled, and whether it trusted itself ---------------

def test_the_dropper_reports_a_patch_that_straddles_an_edge():
    """The "sometimes" in the bench report: the dropper averages a 9-px
    disc, so a click near a flake's edge returns a colour NEITHER the flake
    nor the substrate has — and that colour then becomes the mask's target
    and the curve's centre."""
    from talos.cv.identify import sample_hex_stats

    img = np.zeros((40, 60, 3), np.uint8)
    img[:, :30] = (28, 52, 132)          # the dark layer
    img[:, 30:] = (170, 205, 245)        # the light one
    well_inside = sample_hex_stats(img, 10, 20)
    on_the_edge = sample_hex_stats(img, 30, 20)

    assert well_inside[1] < 5.0, "a clean patch of one colour reads clean"
    assert well_inside[0] == "#1c3484"
    assert on_the_edge[1] > 60.0, "a patch spanning both reads as a mixture"
    assert on_the_edge[0] not in ("#1c3484", "#aacdf5")


def test_sample_hex_and_its_stats_agree():
    """``sample_hex`` is the one the mask's target comes from; the stats
    must describe the very same patch."""
    from talos.cv.identify import sample_hex_stats

    img = scene([(60, 60, 50, 40, (28, 52, 132))])
    stats = sample_hex_stats(img, 85, 80)
    assert sample_hex(img, 85, 80) == stats[0]


def test_the_stats_survive_a_point_off_the_frame():
    from talos.cv.identify import sample_hex_stats

    assert sample_hex_stats(np.zeros((4, 4, 3), np.uint8), 10, 10) is None
    assert sample_hex_stats(np.zeros((4, 4, 3), np.uint8), -1, 0) is None
    assert sample_hex(np.zeros((4, 4, 3), np.uint8), 10, 10) is None


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


def test_a_merged_sample_cannot_slip_past_the_size_cap():
    """The gates run before the merge (the canonical order), and merging
    SUMS areas: two fragments that each pass can produce one candidate over
    the operator's cap — with the readout still saying "Size 2 → Merge 1".

    The two fragments here are 40 x 40 px at 0.2 µm/px = 64 µm² each; merged
    they are 128 µm², so a cap of 100 is the case: the fragments fit, the
    merged sample does not.
    """
    img = scene([(60, 60, 40, 40, MAGENTA), (110, 60, 40, 40, MAGENTA)])
    stage = ColourStage(hex_color=hex_of(MAGENTA), tolerance=20.0)
    capped = SizeStage(min_area_um2=1.0, max_area_um2=100.0)

    # the control: without a merge the two fragments each pass the cap
    split = IdentifyPipeline().run(
        img, CALIB, config=config(colour=stage, size=capped,
                                  merge=MergeStage(enabled=False)))
    assert len(split.candidates) == 2

    # merged, the single candidate is over the cap and must not be reported
    merged = IdentifyPipeline().run(img, CALIB,
                                    config=config(colour=stage, size=capped))
    assert merged.candidates == [], \
        "a 128 µm² sample reported under a 100 µm² cap"
    # and the readout still says the merge happened (the counts are per stage)
    assert any(label == "Merge fragments" and out == 1
               for label, _in, out in merged.counts)


def test_a_merged_sample_sits_between_its_fragments_not_beside_them():
    """The merged position is the fragments' area-weighted centroid, not the
    union box's centre: two fragments on a diagonal have a box centre that
    lands between them, on bare substrate — and that value is what the
    table, the map marker and 'go to sample' all point at."""
    from talos.cv.flakes import merge_fragments

    a = FlakeCandidate(x_px=80.0, y_px=80.0, area_px2=1600.0,
                       bbox=(60, 60, 40, 40))
    b = FlakeCandidate(x_px=125.0, y_px=125.0, area_px2=1600.0,
                       bbox=(105, 105, 40, 40))
    merged = merge_fragments([a, b], gap_px=25.0, um2_per_px2=0.04)

    assert len(merged) == 1
    cand = merged[0]
    assert cand.area_px2 == pytest.approx(3200.0)
    # the centroid of two equal fragments is the midpoint of the two CENTRES
    assert cand.x_px == pytest.approx(102.5)
    assert cand.y_px == pytest.approx(102.5)
    # the union box runs 60..145 in both axes, so its centre is (102.5,
    # 102.5) too — the diagonal is the case where they differ, and equal
    # areas are the case where the centroid is the midpoint
    assert cand.bbox == (60, 60, 85, 85)


def test_a_lighter_fragment_does_not_drag_the_merged_position():
    """The centre is weighted by area, so a big fragment outweighs a small
    one — which is the difference between "where the sample is" and "where
    its bounding box happens to be"."""
    from talos.cv.flakes import merge_fragments

    big = FlakeCandidate(x_px=100.0, y_px=100.0, area_px2=4000.0,
                         bbox=(80, 80, 40, 40))
    small = FlakeCandidate(x_px=140.0, y_px=100.0, area_px2=400.0,
                           bbox=(130, 90, 20, 20))
    cand = merge_fragments([big, small], gap_px=25.0, um2_per_px2=0.04)[0]
    assert cand.x_px == pytest.approx((100.0 * 4000 + 140.0 * 400) / 4400.0)
    assert cand.x_px == pytest.approx(103.6, abs=0.1)


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


def test_an_uncalibrated_objective_says_so_once(caplog):
    """The pipeline has to pick a number when there is no calibration, and
    picks 1.0 µm/px — so a "30 µm²" size floor becomes a 30-pixel one. Every
    other consumer of the calibration treats a missing value as "no
    calibration" (the scale bar does not draw); this one cannot, so it is
    logged rather than silently reported as µm²."""
    from talos.cv import identify as ident

    ident._warned_uncalibrated = False          # fresh for the test
    uncalibrated = ObjectiveCalibration(objective_id=0, um_per_px_x=None,
                                        um_per_px_y=None)
    with caplog.at_level("WARNING"):
        result = IdentifyPipeline().run(scene([(60, 60, 40, 40, MAGENTA)]),
                                        uncalibrated,
                                        config=config(colour=ColourStage(
                                            hex_color=hex_of(MAGENTA),
                                            tolerance=20.0)))
    assert result.candidates, "the pipeline still runs"
    assert any("no µm/px calibration" in record.message
               for record in caplog.records)
    caplog.clear()
    ident._warned_uncalibrated = True
    with caplog.at_level("WARNING"):
        IdentifyPipeline().run(scene([(60, 60, 40, 40, MAGENTA)]),
                               uncalibrated,
                               config=config(colour=ColourStage(
                                   hex_color=hex_of(MAGENTA), tolerance=20.0)))
    assert not caplog.records, "it must not repeat itself every frame"


# --- the three matching methods -------------------------------------------

TARGET = "#1c3484"                       # the bench's dark blue


def _probe(**hsv_delta) -> np.ndarray:
    """One pixel at a given HSV offset from TARGET, as real 8-bit RGB.

    Built through cv2 in both directions so the pixel the mask sees really
    has (about) that hue/saturation — an RGB triple typed by hand lands ±1
    unit away after the round trip, which is enough to cross a boundary and
    make a test that claims to be about geometry fail for arithmetic.
    """
    import cv2 as _cv2

    h0, s0, v0 = hex_to_hsv(TARGET)
    want = (max(0, min(179, h0 + hsv_delta.get("dh", 0))),
            max(0, min(255, s0 + hsv_delta.get("ds", 0))),
            max(0, min(255, v0 + hsv_delta.get("dv", 0))))
    return _cv2.cvtColor(np.uint8([[list(want)]]),
                         _cv2.COLOR_HSV2RGB).astype(np.uint8)


def _masked(pixel, method, tolerance=25.0, spread=25.0) -> bool:
    return bool(colour_mask(pixel, TARGET, tolerance, spread, 0.0, 0.0,
                            method=method)[0, 0])


def test_the_window_is_a_box_and_the_hsv_ball_rounds_its_corners():
    """The one thing that separates the first two methods.

    A pixel at the far edge of hue AND of shade at once passes the box
    (each axis is inside its own half-width) and fails the ball (it is
    further from the centre than the radius) — and that diagonal is exactly
    what an illumination gradient looks like.
    """
    corner = _probe(dh=16, ds=-45)
    assert _masked(corner, "window")
    assert not _masked(corner, "hsv_distance")


def test_the_hsv_ball_is_inside_the_window_when_the_spread_matches():
    """The containment the docs claim: at ``spread == tolerance`` the ball
    is inscribed in the box — nothing the ball accepts is outside it, and
    the box still holds the corners the ball cuts. (With a *narrowed* spread
    that stops being true, which is the next test.)"""
    rng = np.random.default_rng(11)
    px = rng.integers(0, 256, (20000, 1, 3)).astype(np.uint8)
    for tol in (15.0, 25.0, 60.0):
        window = colour_mask(px, TARGET, tol, tol, 0.0, 0.0, method="window")
        ball = colour_mask(px, TARGET, tol, tol, 0.0, 0.0,
                           method="hsv_distance")
        assert (ball > 0).sum() > 0, tol
        assert ((ball > 0) & (window == 0)).sum() == 0, tol
        assert ((window > 0) & (ball == 0)).sum() > 0, tol


def test_the_two_distance_methods_ignore_the_spread():
    """Spread belongs to the window. A distance method that quietly used it
    as well would make the same number mean two things."""
    pixel = _probe(ds=-70, dv=40)
    for method in ("hsv_distance", "rgb_distance"):
        assert (_masked(pixel, method, spread=0.0)
                == _masked(pixel, method, spread=60.0))


def test_the_hsv_ball_uses_the_tolerance_for_shade_not_the_spread():
    """The context switch worth knowing about: narrowing the spread to
    separate two layers does NOT narrow a distance method — its shade slack
    is 2.55·tolerance, so the layers come back. The panel warns on the
    switch; this pins the arithmetic behind the warning."""
    far_in_shade = _probe(ds=-100)          # beyond the window at spread 20
    assert not _masked(far_in_shade, "window", tolerance=40.0, spread=20.0)
    assert _masked(far_in_shade, "hsv_distance", tolerance=40.0, spread=20.0)


def test_every_method_contains_the_colour_that_was_picked():
    """The invariant the floors exist to keep, in all three geometries: a
    mask the pick is outside of reads as a broken detector and is silent."""
    for method in ("window", "hsv_distance", "rgb_distance"):
        for colour in ("#1c3484", "#aacdf5", "#d0d0d0", "#3c3c3c"):
            pixel = np.array([[[int(colour[1:3], 16), int(colour[3:5], 16),
                                int(colour[5:7], 16)]]], np.uint8)
            assert _masked(pixel, method) or colour != TARGET, colour
            # ...including with the floors raised above the pick itself
            assert bool(colour_mask(pixel, colour, 25.0, 25.0, 200.0, 150.0,
                                    method=method)[0, 0]), (colour, method)


def test_a_red_target_matches_across_the_seam_in_every_method():
    """Hue is a circle: 179 and 2 are three apart. The window splits its
    band in two; a distance needs the wrapped difference itself, and uint8
    subtraction would wrap silently without the int32 cast."""
    red = "#ff0000"
    near = np.uint8([[[255, 30, 30]]])
    for method in ("window", "hsv_distance", "rgb_distance"):
        h0, s0, v0 = hex_to_hsv(red)
        assert h0 < 5, "the red target's hue moved — check the fixture"
        assert bool(colour_mask(near, red, 25.0, 25.0, 0.0, 0.0,
                                method=method)[0, 0]), method


def test_every_method_returns_the_dtype_the_pipeline_needs():
    """The mask is ORed into a uint8 accumulator and handed to
    cv2.connectedComponents: a bool or int64 mask from one method would fail
    far away from its cause."""
    img = np.zeros((4, 4, 3), np.uint8)
    for method in ("window", "hsv_distance", "rgb_distance"):
        mask = colour_mask(img, TARGET, 25.0, 25.0, 0.0, 0.0, method=method)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)) <= {0, 255}


def test_the_distance_methods_are_not_the_window_under_another_name():
    """A cross-check with real pixels. (A stub that returned the window for
    every method would pass every equality test in this file and fail this
    one.)

    Both balls give up pixels the box keeps — its corners. Only the RGB ball
    also takes pixels the box refuses: its axes are R, G and B, so a
    low-saturation pixel with a wildly different hue is simply *close* to
    another low-saturation pixel, which is precisely why this method exists.
    """
    rng = np.random.default_rng(3)
    px = rng.integers(0, 256, (20000, 1, 3)).astype(np.uint8)
    window = colour_mask(px, TARGET, 25.0, 25.0, 0.0, 0.0, method="window")
    for method in ("hsv_distance", "rgb_distance"):
        other = colour_mask(px, TARGET, 25.0, 25.0, 0.0, 0.0, method=method)
        assert (other > 0).sum() > 0, method
        assert ((window > 0) & (other == 0)).sum() > 0, method
    rgb = colour_mask(px, TARGET, 25.0, 25.0, 0.0, 0.0,
                      method="rgb_distance")
    assert ((rgb > 0) & (window == 0)).sum() > 0


def test_a_colour_stage_round_trips_its_method():
    stage = ColourStage(hex_color="#123456", tolerance=30.0, spread=12.0,
                        method="rgb_distance")
    back = IdentifyConfig.from_dict(IdentifyConfig(stages=[stage]).to_dict())
    assert back.stage("colour").method == "rgb_distance"


def test_a_stored_file_without_a_method_masks_exactly_as_it_did():
    """Every settings file written before this field existed: the window,
    byte for byte."""
    stored = {"stages": [{"name": "colour", "enabled": True,
                          "hex_color": TARGET, "tolerance": 40.0,
                          "spread": 20.0, "min_saturation": 40.0,
                          "min_value": 0.0}]}
    stage = IdentifyConfig.from_dict(stored).stage("colour")
    assert stage.method == "window"
    img = np.full((4, 4, 3), (28, 52, 132), np.uint8)
    assert np.array_equal(colour_mask(img, TARGET, 40.0, 20.0, 40.0, 0.0),
                          stage.source_mask(_ctx_for(img)))


def test_an_unknown_stored_method_falls_back_to_the_window(caplog):
    """A hand-edited (or future) value must not reach the pipeline as a
    string nothing matches — every method check is an equality, so it would
    fall through to the window by accident rather than by decision."""
    stored = {"stages": [{"name": "colour", "hex_color": TARGET,
                          "method": "perceptual"}]}
    with caplog.at_level("INFO"):
        stage = IdentifyConfig.from_dict(stored).stage("colour")
    assert stage.method == "window"
    assert any("unknown colour method" in record.message
               for record in caplog.records)
