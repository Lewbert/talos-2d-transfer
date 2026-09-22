"""Pre-processing: the LUT chain and the local-contrast curve.

The curve has three properties that are not negotiable, and each has a
test named after the failure it prevents:

- **Monotone** — a channel that folds back turns a gradient into a false
  edge, which is structure the sample does not have.
- **Pinned at 0, at the picked colour and at 255** — the picked colour is
  also the colour mask's target, so a curve that moved it would leave the
  mask looking for a colour the frame no longer contains.
- **Gain means gain** — the slope at the pick is the number on the
  slider, including near the ends of the range where the boost has to be
  capped.
"""

from __future__ import annotations

import numpy as np
import pytest

import cv2

from talos.cv.preprocess import (
    DN_MAX,
    MAX_FLAT_RUN,
    Denoise,
    LocalContrast,
    PreprocessConfig,
    apply,
    build_lut,
    channel_curve,
    curve_values,
    cv_lut,
    denoise,
    effective_gain,
    effective_width,
)

CENTRES = (1.0, 8.0, 32.0, 80.0, 128.0, 200.0, 250.0, 254.0)
GAINS = (1.5, 3.0, 8.0)
WIDTHS = (2.0, 16.0, 32.0, 120.0)


def _slope_at(centre: float, gain: float, width: float, h: float = 0.25) -> float:
    """The one-sided slopes at the centre, measured, not assumed."""
    return float(curve_values(centre, gain, width,
                              np.array([centre - h, centre + h]))[1]
                 - curve_values(centre, gain, width,
                                np.array([centre - h, centre + h]))[0]) / (2 * h)


# ----------------------------------------------------------------------
# The curve
# ----------------------------------------------------------------------

@pytest.mark.parametrize("centre", CENTRES)
@pytest.mark.parametrize("gain", GAINS)
@pytest.mark.parametrize("width", WIDTHS)
def test_the_curve_never_folds_back(centre, gain, width):
    v = np.linspace(0.0, DN_MAX, 4096)
    assert np.all(np.diff(curve_values(centre, gain, width, v)) >= -1e-9)


@pytest.mark.parametrize("centre", CENTRES)
@pytest.mark.parametrize("gain", GAINS)
@pytest.mark.parametrize("width", WIDTHS)
def test_the_pick_and_both_ends_are_pinned(centre, gain, width):
    """The three fixed points, to the last decimal the maths can hold.

    The pick is the one that matters: the same hex feeds the colour mask,
    so a curve that shifted it would make the mask miss the very thing
    the operator pointed at.
    """
    values = curve_values(centre, gain, width,
                          np.array([0.0, centre, DN_MAX]))
    assert values[0] == pytest.approx(0.0, abs=1e-6)
    assert values[1] == pytest.approx(centre, abs=1e-6)
    assert values[2] == pytest.approx(DN_MAX, abs=1e-6)


@pytest.mark.parametrize("gain", (1.5, 2.0, 3.0, 5.0))
def test_the_slope_at_the_pick_is_the_gain_on_the_slider(gain):
    """Measured on a mid-tone pick, where both sides agree exactly."""
    assert _slope_at(128.0, gain, 32.0) == pytest.approx(gain, rel=0.02)


@pytest.mark.parametrize("centre", CENTRES)
@pytest.mark.parametrize("gain", GAINS)
@pytest.mark.parametrize("width", WIDTHS)
def test_the_reported_gain_is_the_delivered_gain(centre, gain, width):
    """`effective_gain` is what the panel shows, so it has to be the
    truth — including where the boost is capped by monotonicity."""
    assert _slope_at(centre, gain, width) == pytest.approx(
        effective_gain(centre, gain, width), rel=0.05, abs=0.05)


def test_a_mid_tone_pick_gets_the_full_gain():
    assert effective_gain(128.0, 3.0, 32.0) == pytest.approx(3.0, rel=0.01)


def _worst_flat_run(lut) -> int:
    levels = lut.astype(int)
    runs, run = [], 0
    for step in np.diff(levels):
        run = run + 1 if step == 0 else 0
        runs.append(run)
    return max(runs) if runs else 0


@pytest.mark.parametrize("centre", CENTRES)
@pytest.mark.parametrize("gain", GAINS)
@pytest.mark.parametrize("width", WIDTHS)
def test_no_curve_has_a_dead_shelf(centre, gain, width):
    """The budget is fixed: pinning 0, the pick and 255 means the total
    slope is 255 whatever the parameters. Boost a wide band hard and the
    charge comes back as a shelf — measured at gain 8 over ±32 DN it was
    a hundred input levels sharing one output, twenty DN from the pick.
    The band gives way instead, so no shelf can form where the sample is.
    """
    lut = channel_curve(centre, gain, width)
    assert _worst_flat_run(lut) <= MAX_FLAT_RUN


@pytest.mark.parametrize("centre", CENTRES)
@pytest.mark.parametrize("gain", GAINS)
def test_the_band_narrows_rather_than_the_gain_being_eaten(centre, gain):
    """Width is my invention; the gain is what was asked for. So a
    request that cannot be granted keeps the gain and reports a smaller
    band."""
    used = effective_width(centre, gain, 120.0)
    assert used <= 120.0
    if used > 1.0:
        assert effective_gain(centre, gain, 120.0) == pytest.approx(
            effective_gain(centre, gain, used), rel=1e-9)


def test_a_reasonable_request_is_granted_untouched():
    assert effective_width(128.0, 3.0, 32.0) == pytest.approx(32.0)
    assert effective_width(128.0, 3.0, 16.0) == pytest.approx(16.0)


def test_the_pick_is_the_steepest_point():
    v = np.linspace(0.0, DN_MAX, 4096)
    values = curve_values(32.0, 4.0, 16.0, v)
    slope = np.diff(values) / np.diff(v)
    assert abs(v[np.argmax(slope)] - 32.0) <= 1.0


def test_gain_one_is_the_identity():
    v = np.linspace(0.0, DN_MAX, 256)
    assert np.array_equal(curve_values(128.0, 1.0, 32.0, v), v)
    assert np.array_equal(channel_curve(128.0, 1.0, 32.0), np.arange(256, dtype=np.uint8))


def test_the_uint8_lut_is_the_float_curve_rounded():
    lut = channel_curve(128.0, 3.0, 32.0)
    expected = np.round(curve_values(128.0, 3.0, 32.0, np.arange(256.0)))
    assert lut.dtype == np.uint8
    assert np.all(np.abs(lut.astype(int) - expected.astype(int)) <= 1)


# ----------------------------------------------------------------------
# What the curve is FOR
# ----------------------------------------------------------------------

def test_the_curve_pulls_two_nearby_colours_apart():
    """The whole point: a layer-to-layer difference of a few DN around the
    substrate becomes a difference the eye separates — while the picked
    colour itself does not move.
    """
    substrate, monolayer, bilayer = 150, 156, 162
    curve = curve_values(float(substrate), 3.0, 32.0,
                         np.array([substrate, monolayer, bilayer],
                                  dtype=float))
    before = bilayer - monolayer
    after = curve[2] - curve[1]
    assert after > before * 2
    assert curve[0] == pytest.approx(substrate, abs=1e-6)


def test_the_curve_does_nothing_far_from_the_pick():
    """Well outside the band the curve only compresses — it must not
    invent separation between two colours that had none near the pick."""
    far = np.array([10.0, 11.0])
    out = curve_values(200.0, 3.0, 16.0, far)
    assert 0.0 < out[1] - out[0] <= 1.0 + 1e-6


# ----------------------------------------------------------------------
# The point chain
# ----------------------------------------------------------------------

def test_the_point_chain_is_the_identity_by_default():
    cfg = PreprocessConfig(enabled=True)
    assert cfg.points_identity() is True
    assert np.array_equal(build_lut(cfg), np.tile(np.arange(256), (3, 1)))


def test_the_lut_is_per_channel_and_rgb_ordered():
    cfg = PreprocessConfig(enabled=True)
    cfg.local = LocalContrast(enabled=True, gain=4.0, width=4.0)
    # centre the curve on blue only: red and green stay the identity
    lut = build_lut(cfg, centre_rgb=(0, 0, 200))
    assert lut.shape == (3, 256)
    assert np.array_equal(lut[0], np.arange(256))
    assert np.array_equal(lut[1], np.arange(256))
    assert not np.array_equal(lut[2], np.arange(256))


def test_the_curve_is_applied_after_the_spatial_stage():
    """The pick is a value the operator read off the PRE-PROCESSED frame,
    so the curve has to be applied to the output of the spatial stage —
    not fed the raw frame and told about it. Denoise is that stage now
    (the tone operations were removed in 2026-09-23), and the property is
    the same one: what the LUT sees is what the operator saw."""
    cfg = PreprocessConfig(enabled=True)
    cfg.local = LocalContrast(enabled=True, gain=3.0, width=8.0)
    cfg.denoise = Denoise(enabled=True, diameter=5, sigma_color=40.0,
                          sigma_space=5.0)
    img = _frame(3)
    expected = cv2.LUT(denoise(img, 5, 40.0, 5.0),
                       cv_lut(build_lut(cfg, (128, 128, 128))))
    assert np.array_equal(apply(img, cfg, (128, 128, 128)), expected)


# ----------------------------------------------------------------------
# apply()
# ----------------------------------------------------------------------

def _frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)


def test_off_means_the_very_same_array():
    """Not a copy that happens to match — the same object. That is the
    'pre-processed layer is the original when pre-processing is off'
    contract, in its strongest form."""
    img = _frame()
    assert apply(img, PreprocessConfig()) is img
    assert apply(img, PreprocessConfig(enabled=True)) is img
    assert apply(img, None) is img


def test_a_gain_with_no_colour_picked_is_still_identity():
    """The panel can enable the curve before anything is picked."""
    img = _frame()
    cfg = PreprocessConfig(enabled=True)
    cfg.local = LocalContrast(enabled=True, gain=4.0)
    assert apply(img, cfg) is img
    assert apply(img, cfg, centre_rgb=(128, 128, 128)) is not img


def test_apply_keeps_the_shape_and_dtype():
    img = _frame()
    cfg = PreprocessConfig(enabled=True)
    cfg.local = LocalContrast(enabled=True, gain=3.0, width=16.0)
    cfg.denoise = Denoise(enabled=True, diameter=3)
    out = apply(img, cfg, centre_rgb=(128, 128, 128))
    assert out.shape == img.shape and out.dtype == np.uint8


def test_apply_maps_every_channel_by_its_own_curve():
    """THE test that was missing: the table is built channel-major and
    OpenCV wants it interleaved, and reshaping between the two without the
    transpose permutes it. The image still comes out plausible, so only
    checking actual pixel values catches it — the shape checks did not.
    """
    img = np.array([[[10, 20, 30], [250, 200, 150]]], np.uint8)
    # per-channel, with a curve on one channel only: the picked pixel must
    # survive, and the other must move
    cfg = PreprocessConfig(enabled=True)
    cfg.local = LocalContrast(enabled=True, gain=4.0, width=8.0)
    lut = build_lut(cfg, centre_rgb=(10, 20, 30))
    assert lut[0][10] == 10 and lut[1][20] == 20 and lut[2][30] == 30
    out = apply(img, cfg, centre_rgb=(10, 20, 30))
    assert np.array_equal(out[0, 0], img[0, 0])
    assert not np.array_equal(out[0, 1], img[0, 1])


def test_apply_never_mutates_the_input():
    img = _frame(3)
    before = img.copy()
    cfg = PreprocessConfig(enabled=True)
    cfg.local = LocalContrast(enabled=True, gain=3.0, width=8.0)
    apply(img, cfg, centre_rgb=(128, 128, 128))
    assert np.array_equal(img, before)


# ----------------------------------------------------------------------
# The spatial stage
# ----------------------------------------------------------------------

def test_denoise_smooths_noise_but_keeps_an_edge():
    rng = np.random.default_rng(11)
    flat = np.full((40, 40, 3), 120, dtype=np.uint8)
    noisy = np.clip(flat.astype(int) + rng.integers(-25, 26, flat.shape),
                    0, 255).astype(np.uint8)
    out = denoise(noisy, diameter=5, sigma_color=40, sigma_space=4)
    assert out[16, 16, 0] != 0
    assert out.astype(float).std() < noisy.astype(float).std()

    edge = np.zeros((40, 40, 3), dtype=np.uint8)
    edge[:, 20:] = 200
    kept = denoise(edge, diameter=5, sigma_color=30, sigma_space=3)
    # the step survives: the two plateaus stay far apart
    assert kept[20, 5, 0] < 40 and kept[20, 35, 0] > 160


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

def test_round_trip_through_dict():
    cfg = PreprocessConfig(enabled=True)
    cfg.local = LocalContrast(enabled=True, gain=4.0, width=12.0)
    cfg.denoise = Denoise(enabled=True, diameter=9, sigma_color=50.0,
                          sigma_space=6.0)
    again = PreprocessConfig.from_dict(cfg.to_dict())
    assert again == cfg
    assert again.local == cfg.local
    assert again.denoise == cfg.denoise


def test_from_dict_is_forgiving():
    """A hand-edited settings file must not be able to break the frame
    path, exactly like IdentifyConfig.from_dict."""
    assert PreprocessConfig.from_dict(None) == PreprocessConfig()
    assert PreprocessConfig.from_dict("nonsense") == PreprocessConfig()
    assert PreprocessConfig.from_dict({"local": {"gain": 1e9}}).local.gain == 8.0
    assert PreprocessConfig.from_dict({"denoise": {"diameter": 3.7}}).denoise.diameter == 4
    assert PreprocessConfig.from_dict({"enabled": True}).enabled is True


def test_from_dict_ignores_booleans_where_numbers_are_expected():
    assert PreprocessConfig.from_dict({"local": {"gain": True}}).local.gain \
        == LocalContrast().gain


def test_a_removed_stage_in_a_stored_file_is_ignored():
    """The tone operations and the shade correction were dropped in
    2026-09-23. A settings file that still carries them (or the sections
    they lived in) must load as if they were never there — the migration
    drops the keys, and even if one survived it means nothing here."""
    cfg = PreprocessConfig.from_dict({
        "enabled": True, "exposure": 2.0, "brightness": 20.0,
        "contrast": 0.5, "gamma": 2.0,
        "shade": {"enabled": True, "sigma": 30.0},
        "local": {"enabled": True, "gain": 3.0},
    })
    assert cfg.enabled is True
    assert cfg.local.enabled is True and cfg.local.gain == 3.0
    assert not hasattr(cfg, "exposure") and not hasattr(cfg, "shade")


def test_a_missing_section_keeps_the_defaults():
    cfg = PreprocessConfig.from_dict({"enabled": True, "local": {"enabled": True}})
    assert cfg.local.gain == LocalContrast().gain
    assert cfg.local.width == LocalContrast().width
