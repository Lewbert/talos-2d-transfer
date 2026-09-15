"""Camera orientation flip (devices.camera.flip, default ON).

The bench optics present the specimen rotated 180°, so every backend
rotates the decoded frame before anything else sees it. These tests pin
the two things that are easy to get wrong: the flip must be applied by
EVERY backend (a forgotten one leaves the live view corrected but the
autofocus/detection path not), and it must happen BEFORE the scale-bar
burn (otherwise the bar and its label are mirrored into the saved file).
"""

import cv2
import numpy as np
import pytest

from talos.hal.camera_props import map_property
from talos.hal.devices.camera import _BACKENDS
from talos.hal.sim.sim_camera import SimCamera

BACKENDS = ("smartcam", "harvesters", "mmcore", "mcam", "directshow",
            "manual", "sim")


def _scene(w=160, h=120) -> np.ndarray:
    """Asymmetric frame: a bright patch TOP-LEFT, everything else dark."""
    frame = np.zeros((h, w, 3), np.uint8)
    frame[0:20, 0:30] = 255
    return frame


class _StubCamera(SimCamera):
    """SimCamera with a deterministic scene: a bright patch in the TOP-LEFT
    eighth of the frame, everything else black (no noise, no RNG)."""

    def _render_scene(self) -> np.ndarray:  # noqa: D102
        h, w = self.height, self.width
        frame = np.zeros((h, w, 3), np.uint8)
        frame[0:max(1, h // 8), 0:max(1, w // 8)] = 255
        return frame


def _patch_corner(frame: np.ndarray, corner: str) -> int:
    """Max value in the corner patch (1/8 × 1/8 of the frame)."""
    h, w = frame.shape[:2]
    ph, pw = max(1, h // 8), max(1, w // 8)
    return int({
        "tl": frame[:ph, :pw],
        "tr": frame[:ph, -pw:],
        "bl": frame[-ph:, :pw],
        "br": frame[-ph:, -pw:],
    }[corner].max())


def test_default_is_enabled_and_rotates_180():
    cam = _StubCamera()
    assert cam.flip_enabled is True
    out = cam.apply_flip(_scene())
    assert out is not None
    # the bright patch moved from the top-left to the bottom-right corner
    assert out[0:20, 0:30].max() == 0
    assert out[-20:, -30:].max() == 255


def test_flip_off_passes_the_same_object_through():
    cam = _StubCamera({"flip": False})
    frame = _scene()
    assert cam.apply_flip(frame) is frame  # no copy, no transform
    assert cam.apply_flip(None) is None


def test_flip_on_returns_a_fresh_c_contiguous_array():
    """fetch() promises a fresh contiguous array — the flip must not
    hand back a view whose buffer the camera may reuse."""
    cam = _StubCamera()
    frame = _scene()
    out = cam.apply_flip(frame)
    assert out is not frame
    assert out.flags["C_CONTIGUOUS"] and out.base is None


def test_every_registered_backend_applies_the_flip():
    """A backend that ships without APPLIES_FLIP shows the correct live
    image while feeding autofocus/detection the unflipped one."""
    for name, cls in _BACKENDS.items():
        assert getattr(cls, "APPLIES_FLIP", False) is True, \
            f"camera backend {name!r} does not apply the orientation flip"


def test_map_property_accepts_flip_for_every_backend():
    for backend in BACKENDS:
        assert map_property(backend, "flip", True) == ("flip", True)


def test_set_property_flip_is_software_only():
    cam = _StubCamera()
    props_before = cam.get_properties()
    cam.set_property("flip", False)
    assert cam.flip_enabled is False
    assert cam.get_properties()["flip"] is False
    # the hardware property table is untouched (no such parameter exists)
    assert {k: v for k, v in cam.get_properties().items() if k != "flip"} \
        == {k: v for k, v in props_before.items() if k != "flip"}


def test_sim_fetch_frame_is_rotated():
    cam = _StubCamera({"width": 320, "height": 240})
    frame = cam.fetch()
    assert frame is not None
    assert frame.shape == (240, 320, 3)
    assert _patch_corner(frame, "br") == 255   # patch rotated to bottom-right
    assert _patch_corner(frame, "tl") == 0
    assert _patch_corner(frame, "tr") == 0


def test_snapshot_burns_the_bar_AFTER_the_flip(tmp_path):
    """The saved image is flipped first, then burned: a flip applied after
    the burn would mirror the bar and its µm label into the top-left of
    every saved snapshot (and the live view would look right)."""
    cam = _StubCamera({"width": 640, "height": 480, "flip": True})
    path = cam.snapshot(tmp_path / "shot.png", burn={"um_per_px": 1.0})
    img = cv2.imread(str(path))
    assert img is not None and img.shape[:2] == (480, 640)
    # bar present in the bottom-right corner…
    assert img[-60:, -200:].max() > 180
    # …and the TOP-LEFT still holds only the (rotated-away) black corner:
    # a post-burn flip would have mirrored the bar and its label there.
    assert img[:60, :200].max() < 60


def test_snapshot_without_flip_still_burns_bottom_right(tmp_path):
    cam = _StubCamera({"width": 640, "height": 480, "flip": False})
    path = cam.snapshot(tmp_path / "shot.png", burn={"um_per_px": 1.0})
    img = cv2.imread(str(path))
    # the scene patch stays top-left, the bar stays bottom-right
    assert _patch_corner(img, "tl") == 255
    assert img[-60:, -200:].max() > 180
    assert img[:60, -200:].max() < 60  # no bar in the top-right either


@pytest.mark.parametrize("flip", [True, False])
def test_flip_is_per_frame_not_sticky(flip, tmp_path):
    cam = _StubCamera({"width": 320, "height": 240, "flip": flip})
    first = cam.fetch()
    cam.set_property("flip", not flip)
    second = cam.fetch()
    assert np.array_equal(second, first[::-1, ::-1])
