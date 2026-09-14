"""Focus-metric and synthetic-stack tests."""

import numpy as np
import pytest

from talos.cv.focus_metric import (
    abs_diff,
    bin2,
    brenner,
    brenner_k,
    default_roi,
    laplacian_variance,
    sharpness_profile,
    tenengrad,
)
from tests.testing.sim_images import defocus_blur, focus_stack, synthetic_flake_image


@pytest.fixture(scope="module")
def sharp():
    return synthetic_flake_image(seed=7, noise=0.0)


def test_all_metrics_decrease_with_defocus(sharp):
    sigmas = (0.0, 1.5, 3.0, 6.0)
    for metric in (laplacian_variance, tenengrad, brenner):
        scores = [metric(defocus_blur(sharp, s)) for s in sigmas]
        assert scores[0] > scores[-1]
        assert scores[0] == max(scores), metric.__name__


def test_metric_is_strictly_peak_shaped(sharp):
    """Monotonic decrease as |z - focus| grows. The 1% tolerance absorbs
    Laplacian-variance wiggle at the heavy-blur noise floor (a nearly
    flat image — the metric is noise-dominated there, per docs/PLAN.md)."""
    sigmas = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
    scores = [laplacian_variance(defocus_blur(sharp, s)) for s in sigmas]
    for a, b in zip(scores, scores[1:]):
        assert a >= b * 0.99


def test_roi_limits_computation():
    frame = synthetic_flake_image(shape=(240, 320))
    roi = default_roi(frame.shape)
    assert roi == (80, 60, 160, 120)
    full = laplacian_variance(frame)
    cropped = laplacian_variance(frame, roi)
    assert cropped >= 0 and full >= 0


def test_focus_stack_peak_at_focus_position(sharp):
    positions = np.arange(-400, 401, 100)
    frames = focus_stack(positions, focus_pos=0,
                         img_gen=lambda: sharp, k_per_step=0.02, base_sigma=0.4)
    scores = sharpness_profile(positions, frames)
    assert int(scores.argmax()) == 4  # position 0 is index 4
    assert scores[4] > scores[0] and scores[4] > scores[-1]


# ---------------------------------------------------------------------------
# Low-frequency coarse metrics (adaptive strategy stage 1)
# ---------------------------------------------------------------------------

def test_low_freq_metrics_peak_shaped_under_defocus(sharp):
    sigmas = (0.0, 2.0, 4.0, 8.0, 16.0)
    for metric in (brenner_k, abs_diff):
        scores = [metric(defocus_blur(sharp, s), k=8) for s in sigmas]
        assert scores[0] > scores[-1], metric.__name__
        assert scores[0] == max(scores), metric.__name__


def test_large_kernel_is_broad_at_heavy_blur(sharp):
    """The blur-tolerance property: at heavy defocus the k=8 kernel keeps
    far more signal than the k=2 variant — its curve falls slower."""
    sigmas = (0.0, 8.0, 16.0)
    wide = [brenner_k(defocus_blur(sharp, s), k=8) for s in sigmas]
    narrow = [brenner(defocus_blur(sharp, s)) for s in sigmas]
    # normalized retention from sharp to the heaviest blur
    wide_retention = wide[-1] / wide[0]
    narrow_retention = narrow[-1] / narrow[0]
    assert wide_retention > narrow_retention


def test_bin2_halves_and_preserves_peak(sharp):
    binned = bin2(sharp)
    assert binned.shape == (sharp.shape[0] // 2, sharp.shape[1] // 2)
    assert binned.ndim == 2                     # gray output
    # peak position on a focus stack is unchanged by binning
    positions = np.arange(-200, 201, 100)
    frames = focus_stack(positions, focus_pos=0,
                         img_gen=lambda: sharp, k_per_step=0.02, base_sigma=0.4)
    native = [brenner_k(f) for f in frames]
    binned_scores = [brenner_k(bin2(f)) for f in frames]
    assert int(np.argmax(native)) == int(np.argmax(binned_scores))


def test_bin2_with_roi():
    frame = synthetic_flake_image(shape=(240, 320))
    roi = default_roi(frame.shape)              # center half
    binned = bin2(frame, roi)
    assert binned.shape == (60, 80)             # (120//2, 160//2)


def test_short_roi_scores_zero_not_nan():
    """A ROI shorter than the kernel left the row-difference array empty:
    mean() of an empty array is NaN (plus a numpy warning) and nothing
    downstream rejects NaN — pick_peak's `hi - lo <= 1e-12` test is False
    for NaN, so the curve failed later with a misleading message."""
    frame = synthetic_flake_image(shape=(240, 320))
    for fn, k in ((brenner_k, 8), (abs_diff, 8), (brenner, 2)):
        score = fn(frame, (0, 0, 40, k))        # height == k
        assert score == 0.0
        assert not np.isnan(score)
        assert not np.isnan(fn(frame, (0, 0, 40, k - 1)))
