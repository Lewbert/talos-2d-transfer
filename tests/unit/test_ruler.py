"""Calibrated tick ruler: spacing ladder, tick offsets and labels."""

import pytest

from talos.cv.ruler import (MIN_MINOR_PX, RulerSpec, minors_per_major,
                            ruler_spec, tick_label, tick_offsets)


def test_minor_divisions_are_round_numbers():
    assert minors_per_major(1.0) == 5      # 0.2 each
    assert minors_per_major(2.0) == 4      # 0.5 each
    assert minors_per_major(5.0) == 5      # 1 each
    assert minors_per_major(0.5) == 5
    assert minors_per_major(200.0) == 4
    assert minors_per_major(0.0) == 1      # degenerate


def test_ruler_spec_uses_the_scale_bar_ladder():
    """Same 1/2/5 ladder as the scale bar, so the two can never disagree
    about what a "round" length is."""
    spec = ruler_spec(1.0, (1080, 1920))     # 1 µm/px, 1920 px wide
    assert spec is not None
    assert spec.major_um in (100.0, 200.0)   # ≤ 16 % of 1920 px = 307 µm
    assert spec.label in ("100 µm", "200 µm")
    assert spec.minor_um == pytest.approx(spec.major_um / 5
                                          if spec.major_um in (100.0, 0.5)
                                          else spec.major_um / 4)


def test_ruler_spec_scales_with_the_objective():
    coarse = ruler_spec(2.0, (1080, 1920))
    fine = ruler_spec(0.1, (1080, 1920))
    assert coarse.major_um > fine.major_um
    # both stay inside the density window
    for spec, um_per_px in ((coarse, 2.0), (fine, 0.1)):
        fraction = spec.major_um / um_per_px / 1920.0
        assert 0.06 <= fraction <= 0.16


def test_ruler_spec_degenerate_inputs():
    assert ruler_spec(None, (1080, 1920)) is None
    assert ruler_spec(0.0, (1080, 1920)) is None
    assert ruler_spec(1.0, ()) is None
    assert ruler_spec(1.0, (0, 0)) is None


def test_tick_offsets_are_centred_on_the_frame():
    """0 is the optical axis (where the crosshair and the stage are) and
    the offsets are symmetric, so the ruler never shifts with resizes."""
    spec = RulerSpec(major_um=100.0, minor_um=20.0, minors_per_major=5,
                     label="100 µm")
    majors, minors = tick_offsets(spec, um_per_px=1.0, frame_px=1000)
    assert majors[0] == -500.0 and majors[-1] == 500.0
    assert 0.0 in majors
    assert all(abs(m) <= 500.0 for m in majors)
    assert 0.0 not in minors                     # majors cover the centre
    assert all(abs(m) <= 500.0 for m in minors)
    # minors that coincide with a major are omitted (every 5th here)
    assert all(abs(m % 100.0) > 1e-9 for m in minors)
    assert len(majors) == 11          # -500 … 500 step 100
    assert len(minors) == 40          # 51 candidates − 11 majors


def test_tick_offsets_drop_minors_that_are_too_dense():
    spec = RulerSpec(major_um=1.0, minor_um=0.2, minors_per_major=5,
                     label="1 µm")
    majors, minors = tick_offsets(spec, um_per_px=1.0, frame_px=4)
    assert minors == []          # 0.2 px apart — a solid line, not a ruler
    assert majors


def test_tick_offsets_degenerate_inputs():
    spec = RulerSpec(major_um=100.0, minor_um=20.0, minors_per_major=5,
                     label="100 µm")
    assert tick_offsets(spec, 0.0, 1000) == ([], [])
    assert tick_offsets(spec, 1.0, 0) == ([], [])


def test_tick_labels_are_compact_and_signed():
    assert tick_label(0.0) == "0"
    assert tick_label(-200.000000001) == "-200"
    assert tick_label(2.5) == "2.5"
    assert tick_label(0.0000001) == "0"


def test_the_minor_step_is_never_below_the_paint_floor():
    from talos.cv.ruler import MIN_MINOR_PX as floor_px
    assert floor_px >= 1.0
    spec = ruler_spec(0.05, (1080, 3840))
    majors, minors = tick_offsets(spec, 0.05, 3840)
    if minors:
        assert (minors[1] - minors[0]) >= floor_px - 1e-9
