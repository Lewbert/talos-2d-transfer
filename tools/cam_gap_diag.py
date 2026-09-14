"""Camera delivery diagnostic: stream for N minutes at a given
exposure/gain/white-balance and report the frame-gap stalls the
SmartCam backend logs (the marginal USB3 extension cable's signature —
irregular 300-580 ms bursts every ~20-40 s). Used to verify a cable/
hub change. Usage:
    python tools/cam_gap_diag.py 30000 10 Off 3
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from talos.config import Settings  # noqa: E402
from talos.cv.frame_slot import LatestFrameSlot  # noqa: E402
from talos.hal.base import DeviceError  # noqa: E402
from talos.hal.devices.camera import camera_chain  # noqa: E402
from tools.af_bench import CameraPump, _safe, tick  # noqa: E402

settings = Settings.load()
exposure = float(sys.argv[1]) if len(sys.argv) > 1 else 30000.0
gain = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
wb = sys.argv[3] if len(sys.argv) > 3 else ""
minutes = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0

camera = None
for candidate in camera_chain(settings.device("camera")):
    try:
        candidate.connect()
        candidate.start()
        camera = candidate
        break
    except DeviceError as exc:
        print(f"backend {candidate.__class__.__name__} failed: {exc}")
if camera is None:
    print("FATAL: no camera")
    sys.exit(1)
camera.set_property("resolution", 1)
camera.set_property("exposure_us", exposure)
camera.set_property("gain", gain)
if wb:
    camera.set_property("white_balance", wb)
tick(0.5)
slot = LatestFrameSlot()
pump = CameraPump(camera, slot)
pump.start()
t0 = time.monotonic()
while time.monotonic() - t0 < minutes * 60.0:
    tick(5.0)
    _safe(f"  +{time.monotonic() - t0:5.0f}s  "
          f"{pump.rate_since(time.monotonic() - 5.0)} frames/5s")
camera.stop()
camera.disconnect()
print("done — count the backend's 'frame gap' warnings above")
