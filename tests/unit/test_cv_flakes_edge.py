"""Flake-detector and wafer-edge tests on synthetic scenes."""

import cv2
import numpy as np
import pytest

from talos.cv.edge import detect_wafer_edge
from talos.cv.flakes import ClassicFlakeDetector, FlakeConfig
from talos.models import ObjectiveCalibration

CALIB = ObjectiveCalibration(objective_id=0, um_per_px_x=0.2, um_per_px_y=0.2)


def planted_flake_scene(shape=(720, 960), flake_rects=None, seed=0):
    """Purple-gray substrate with bright sharp flakes at given (x, y, w, h)."""
    rng = np.random.default_rng(seed)
    h, w = shape
    img = np.full((h, w, 3), (105, 100, 120), dtype=np.float32)
    img += rng.normal(0, 3, (h, w, 1))
    for x, y, fw, fh in flake_rects or []:
        img[y:y + fh, x:x + fw] = (170, 140, 90)   # gold-ish flake
        # sharp darker nucleus
        ny, nx = y + fh // 4, x + fw // 4
        img[ny:ny + fh // 2, nx:nx + fw // 2] = (85, 60, 45)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_flake_recall_on_planted_truth():
    rects = [(150, 120, 120, 100), (500, 300, 80, 70), (300, 550, 90, 60),
             (700, 100, 60, 50), (80, 500, 50, 40)]
    img = planted_flake_scene(flake_rects=rects, seed=1)
    flakes = ClassicFlakeDetector().find(img, CALIB, FlakeConfig())
    # Every planted flake is >= 40x50 px = 80 µm² >= min 30 µm².
    assert len(flakes) >= 4  # >=95% of 5: allow one miss at the very worst
    for x, y, fw, fh in rects:
        assert any(abs(f.x_px - (x + fw / 2)) < fw and abs(f.y_px - (y + fh / 2)) < fh
                   for f in flakes)


def test_flake_detector_ignores_clean_substrate():
    img = planted_flake_scene(flake_rects=[], seed=2)
    flakes = ClassicFlakeDetector().find(img, CALIB, FlakeConfig())
    assert flakes == []


def test_flake_size_filter_in_um():
    rects = [(100, 100, 20, 20),      # 4 µm² — below min 30
             (400, 300, 100, 100)]    # 400 µm² — kept
    img = planted_flake_scene(flake_rects=rects, seed=3)
    flakes = ClassicFlakeDetector().find(img, CALIB, FlakeConfig(min_area_um2=30.0))
    assert len(flakes) == 1
    assert flakes[0].area_um2 == pytest.approx(400.0, rel=0.3)


def test_flake_color_gate_rejects_yellow_green_when_enabled():
    rng = np.random.default_rng(4)
    h, w = 480, 640
    img = np.full((h, w, 3), (105, 100, 120), dtype=np.uint8)
    img = img.astype(np.float32) + rng.normal(0, 2, (h, w, 1))
    # One normal flake (gold-ish) and one strongly yellow-green region.
    img[100:200, 150:250] = (170, 140, 90)
    img[250:330, 150:250] = (60, 200, 130)  # green-yellow (hue ~40)
    img = np.clip(img, 0, 255).astype(np.uint8)
    cfg = FlakeConfig(color_gate=True, min_area_um2=5.0)
    flakes = ClassicFlakeDetector().find(img, CALIB, cfg)
    xs = [f.x_px for f in flakes]
    assert any(100 < x < 300 for x in xs)   # the normal flake kept
    # The green-yellow region's box center (~200, 290) must be absent.
    assert not any(270 < f.y_px < 310 and 100 < f.x_px < 300 for f in flakes)


# --- Wafer edge ------------------------------------------------------------

def make_wafer_scene(wafer: tuple | None = (200, 150, 500, 380)):
    """Orange rough copper stage; purple rectangle wafer (or no wafer)."""
    rng = np.random.default_rng(0)
    h, w = 720, 960
    img = np.full((h, w, 3), (60, 110, 160), dtype=np.float32)  # orange (BGR)
    img += rng.normal(0, 8, (h, w, 3))  # rough surface
    if wafer is not None:
        x, y, ww, hh = wafer
        img[y:y + hh, x:x + ww] = (160, 70, 110)  # purple (BGR)
        img[y:y + hh, x:x + ww] += rng.normal(0, 3, (hh, ww, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def test_edge_rectangle_fit_on_wafer():
    img = make_wafer_scene()
    result = detect_wafer_edge(img)
    assert result.method == "rectangle"
    cx, cy = result.center_px
    assert cx == pytest.approx(450, abs=30)
    assert cy == pytest.approx(340, abs=30)
    # minAreaRect may report the rotated rect with either side first.
    assert sorted(result.size_px) == pytest.approx(sorted((500.0, 380.0)), rel=0.1)


def test_edge_none_when_no_wafer():
    result = detect_wafer_edge(make_wafer_scene(wafer=None))
    assert result.method == "none"


def test_edge_none_when_wafer_fills_view():
    # Wafer fills the frame → no edge to find (no false rectangle).
    rng = np.random.default_rng(1)
    img = np.full((720, 960, 3), (160, 70, 110), dtype=np.float32)
    img += rng.normal(0, 3, img.shape)
    result = detect_wafer_edge(np.clip(img, 0, 255).astype(np.uint8))
    assert result.method == "none"
