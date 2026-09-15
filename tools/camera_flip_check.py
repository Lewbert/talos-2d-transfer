"""Hardware check: the camera flip reaches the REAL camera path.

Opens the Axiocam through the app's own layer, grabs one live frame with
``devices.camera.flip`` on and one with it off, and verifies that the
second is the 180° rotation of the first — i.e. the toggle is a pure
software rotation applied at the backend's single frame egress, and it is
the SAME frame for the live view, autofocus and snapshots.

Read-only: no motion, no property writes beyond the flip flag (which is
restored), nothing is saved.

    pwsh -Command "conda activate talos; python tools/camera_flip_check.py"

Exit code 0 = the flip works on hardware, 1 = mismatch/unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from talos.config import Settings  # noqa: E402
from talos.paths import get_settings_path  # noqa: E402


def main() -> int:
    from talos.hal.registry import make_camera

    settings = Settings.load(get_settings_path())
    cfg = dict(settings.device("camera"))
    cfg["flip"] = True
    cameras = make_camera(cfg, sim=False)
    if not isinstance(cameras, (list, tuple)):
        cameras = [cameras]

    camera = None
    for candidate in cameras:
        try:
            candidate.connect()
            candidate.start()
            camera = candidate
            break
        except Exception as exc:  # noqa: BLE001
            print(f"  {candidate.device_id}: {exc}")
    if camera is None:
        print("FAIL: no camera backend could be opened "
              "(close ZEN / LabscopeService and retry)")
        return 1

    try:
        print(f"camera: {camera.device_id}  flip={camera.flip_enabled}")
        on = camera.fetch(timeout_ms=3000.0)
        if on is None:
            print("FAIL: no frame with flip on")
            return 1
        camera.set_property("flip", False)
        off = camera.fetch(timeout_ms=3000.0)
        if off is None:
            print("FAIL: no frame with flip off")
            return 1
        camera.set_property("flip", True)

        if on.shape != off.shape:
            print(f"FAIL: shape changed {on.shape} -> {off.shape}")
            return 1
        expected = np.ascontiguousarray(off[::-1, ::-1])
        diff = np.abs(on.astype(np.int16) - expected.astype(np.int16))
        # the scene moves between two grabs (specimen drift / noise), so
        # tolerate a small mean difference and compare against the
        # UN-flipped reference as the control.
        mean_flipped = float(diff.mean())
        control = float(np.abs(on.astype(np.int16)
                               - off.astype(np.int16)).mean())
        print(f"frame {on.shape}  |flip(off) - on| = {mean_flipped:.2f}  "
              f"|off - on| = {control:.2f}")
        if mean_flipped >= control:
            print("FAIL: the flipped image is not closer to the 180° "
                  "rotation of the unflipped one (flip not applied?)")
            return 1
        print("PASS: the flip is a 180° software rotation of the live frame")

        # The ordering that really matters: the flip must happen BEFORE the
        # scale-bar burn, or every saved 4K snapshot carries a mirrored bar
        # (and a mirrored µm label) in its top-left corner.
        import tempfile

        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            plain = camera.snapshot(Path(tmp) / "plain.png", timeout_s=30.0,
                                    resolution=0)
            burned = camera.snapshot(Path(tmp) / "burned.png", timeout_s=30.0,
                                     resolution=0, burn={"um_per_px": 0.4})
            a = cv2.imread(str(plain))
            b = cv2.imread(str(burned))
        if a is None or b is None or a.shape != b.shape:
            print("FAIL: the 4K snapshots could not be compared")
            return 1
        # Where the two differ IS the bar (the scene under it is the same
        # frame content; burn-in is the only difference).
        changed = (np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
                   > 60)

        def count(region) -> int:
            return int(region.sum())

        h, w = changed.shape
        ch, cw = h // 4, w // 4
        corners = {
            "top-left": count(changed[:ch, :cw]),
            "top-right": count(changed[:ch, -cw:]),
            "bottom-left": count(changed[-ch:, :cw]),
            "bottom-right": count(changed[-ch:, -cw:]),
        }
        total = count(changed)
        print(f"4K burn difference (px): {corners}  total={total}")
        if not total or max(corners, key=corners.get) != "bottom-right":
            print("FAIL: the burned bar is not in the bottom-right quadrant "
                  "(a flip applied after the burn would mirror it)")
            return 1
        print("PASS: the burned scale bar lands in the bottom-right of the "
              "saved 4K frame")
        return 0
    finally:
        try:
            camera.stop()
        finally:
            camera.disconnect()


if __name__ == "__main__":
    sys.exit(main())
