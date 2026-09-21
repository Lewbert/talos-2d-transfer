"""Grid-scan drill through InstrumentManager — the workspace's real path.

The scan used to build its OWN zolix driver. A second handle on the same
COM port cannot open on Windows, so the grid scan only ever ran against
the sim; and a private driver was invisible to STOP ALL. This drill runs
real scans through ManagerStageAdapter (the manager's own proxy) and then
aborts one mid-flight to prove the cooperative abort reaches the moving
stage.

Motion: every scan starts at the CURRENT stage position (never the
origin) with a small grid. Keep the objective clear of the sample.

Usage:
    pwsh -Command "conda activate talos; python tools/scan_manager_bench.py --yes"
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import QApplication  # noqa: E402

from talos.config import Settings  # noqa: E402
from talos.cv.scan import GridScanner, scan_speed_config  # noqa: E402
from talos.hal.proxies.stage_adapter import ManagerStageAdapter  # noqa: E402
from talos.instruments import InstrumentManager  # noqa: E402
from talos.logging_setup import setup_logging  # noqa: E402
from talos.models import ScanParams  # noqa: E402
from talos.paths import get_scan_dir  # noqa: E402


def _pump(app, seconds: float) -> None:
    """Deliver queued signals while a worker thread runs the scan."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)


def _wait_for_zolix(app, manager, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if manager.last_position.get("zolix") is not None:
            return True
        time.sleep(0.05)
    return False


def _run_scan(app, manager, settings, *, width, height, abort_after=None):
    pos = manager.last_position["zolix"]
    speed = float(settings.section("scan").get("speed_pps", 500) or 500)
    cfg = scan_speed_config(settings.device("zolix"), speed)
    stop = threading.Event()
    adapter = ManagerStageAdapter(manager, cfg, abort_check=stop.is_set)
    scanner = GridScanner(adapter, camera=None)
    # telemetry positions are plain dicts (the proxy publishes asdict)
    params = ScanParams(x0_um=float(pos.get("x_um", 0.0)),
                        y0_um=float(pos.get("y_um", 0.0)),
                        width_um=width, height_um=height,
                        overlap=0.1, serpentine=True,
                        settle_ms=int(settings.section("scan")
                                      .get("settle_ms", 100) or 0))
    out_dir = get_scan_dir() / time.strftime("bench_%Y%m%d_%H%M%S")
    box: dict = {}
    t0 = time.monotonic()

    def worker():
        try:
            box["result"] = scanner.run(params, out_dir,
                                        meta={"fov_um": (200.0, 150.0),
                                              "objective_id": 0})
        except Exception as exc:  # noqa: BLE001
            box["error"] = repr(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    if abort_after is not None:
        _pump(app, abort_after)
        print(f"  aborting after {abort_after:.1f} s ...")
        stop.set()
        scanner.request_abort()
        abort_t = time.monotonic()
        while thread.is_alive() and time.monotonic() - abort_t < 10.0:
            app.processEvents()
            time.sleep(0.01)
        print(f"  abort->return: {time.monotonic() - abort_t:.2f} s "
              f"({'unwound' if not thread.is_alive() else 'STILL RUNNING'})")
    else:
        while thread.is_alive():
            app.processEvents()
            time.sleep(0.01)
    adapter.close()
    return box.get("result"), box.get("error"), time.monotonic() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=float, default=300.0, help="grid width um")
    parser.add_argument("--height", type=float, default=150.0, help="grid height um")
    parser.add_argument("--abort-width", type=float, default=4000.0)
    parser.add_argument("--abort-height", type=float, default=4000.0)
    parser.add_argument("--yes", action="store_true", help="confirm the motion")
    args = parser.parse_args()

    setup_logging(verbose=False)
    if not args.yes:
        print("motion drill: re-run with --yes (small grid from the CURRENT position)")
        return 2

    app = QApplication.instance() or QApplication([])
    settings = Settings.load()
    manager = InstrumentManager(settings)
    manager.connect_all()
    if not _wait_for_zolix(app, manager):
        print("zolix never reported a position — is the controller connected?")
        manager.shutdown()
        return 1
    pos = manager.last_position["zolix"]
    print(f"stage start: x={pos.get('x_um', 0.0):.1f} um, "
          f"y={pos.get('y_um', 0.0):.1f} um")

    exit_code = 0
    try:
        print(f"scan 1: {args.width:.0f}x{args.height:.0f} um from here")
        result, error, elapsed = _run_scan(app, manager, settings,
                                           width=args.width, height=args.height)
        if error:
            print("  FAILED:", error)
            exit_code = 1
        else:
            print(f"  {len(result.frames)} waypoints in {elapsed:.1f} s -> "
                  f"{result.manifest_path}")
            print(f"  message: {result.message}")
            # The headline number of the stop-phase work: what each tile
            # costs once the stage has arrived.
            if result.timing.summary:
                print(f"  timing: {result.timing.summary}")
            if len(result.frames) < 4:
                print("  FAILED: expected at least 4 waypoints")
                exit_code = 1

        print("scan 2 (abort drill): large grid, aborted mid-flight")
        result2, error2, _ = _run_scan(app, manager, settings,
                                       width=args.abort_width,
                                       height=args.abort_height,
                                       abort_after=2.0)
        if error2:
            print("  FAILED:", error2)
            exit_code = 1
        else:
            print(f"  frames: {len(result2.frames)}, aborted={result2.aborted}, "
                  f"message={result2.message!r}")
            if not result2.aborted:
                print("  FAILED: the abort did not mark the result")
                exit_code = 1
    finally:
        manager.stop_all()
        _pump(app, 1.0)
        manager.shutdown()
    print("drill done:", "OK" if exit_code == 0 else "FAILED")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
