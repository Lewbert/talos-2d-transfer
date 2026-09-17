# TALOS — Development

Tests, bench tools, packaging, and how to add an instrument.

## Setting up

```powershell
conda env create -f environment.yml
conda activate talos
python -m talos --sim          # the whole app, simulated devices, no hardware
```

`--sim` is not a mock: it swaps every driver for a simulated twin that speaks the same protocol
state machine (busy rejections, limit flags, emergency-stop bits, timing), so the UI, the job model,
the abort paths and the autofocus loops all run for real. Most development can happen here.

All Python in this project is run through PowerShell with the conda environment activated:

```powershell
pwsh -Command "conda activate talos; python -m pytest"
```

## Tests

```powershell
python -m pytest                  # everything — the real-time simulations dominate
python -m pytest -m "not slow"    # the fast subset (order of a minute)
python -m pytest tests/unit/test_af_math.py -q
```

Three kinds of test, deliberately separated:

| Suite | What it covers | Needs hardware |
|---|---|---|
| `tests/unit/` | Drivers against scripted fake serial ports (CRC faults, torn lines, timeouts, late replies), protocol codecs, the settings merge and migrations, the calibration store, CV maths, input resolution, UI wiring | No |
| `tests/integration/` | The manager, the proxies and their life-cycle: submit/stop ordering, reconnect, retirement, the STOP ALL budget | No |
| `tests/sim/` | Closed-loop **simulations**, marked `slow`: the autofocus strategies and the grid scan drive simulated focus curves and stage models in real time and must land on the true focus position. For anything geometric, `SimCamera(wafer=True)` images a synthetic wafer that the stage carries (`hal/sim/bench.py`) — the scene moves with the stage, so a stitched result can be checked instead of merely looking plausible | No |

The suite needs no instruments — that is a design rule, not an accident. Hardware-in-the-loop work
lives in `tools/` instead.

One test, `tests/sim/test_autofocus_adaptive_closed_loop.py::test_adaptive_refine_repeaks_without_coarse`,
is timing-sensitive and can fail under heavy machine load; it passes on a quiet machine and when run
alone.

## Bench tools

Standalone scripts, each with its own `--help`. Most refuse to move anything without an explicit
confirmation flag. **Read-only unless stated otherwise.**

| Tool | Purpose | Hardware |
|---|---|---|
| `smoke_test.py [--motion]` | Ordered, cautious device checklist for a new or reconfigured bench. Read-only by default | Yes |
| `dev_console.py` | Read-only device console: connect, identify, read status. No motion | Yes |
| `focus_firmware_probe.py` | Raw-serial verification of the focus firmware's parser (junk, torn lines, floods) — the post-flash check | Yes |
| `sigmakoki_latency.py --yes` | Continuous-jog ack/latency drill for the transfer stage; prints the raw first reply line | Yes |
| `scan_manager_bench.py --yes` | The grid scan driven through the real manager, plus a mid-flight abort drill | Yes |
| `hardware_scan.py` | Standalone serpentine scan (builds its own driver and camera — the app-accurate route is `scan_manager_bench.py`, or the Scan window itself). Takes `--settle-ms` and `--backlash-um` | Yes |
| `autofocus_hardware.py` | Autofocus and backlash calibration from the command line — the tuning companion to the AF panel | Yes |
| `af_bench3.py` | The current adaptive-v3 autofocus bench, including the σ measurement and the v2/v3 comparison | Yes |
| `af_abort_bench.py` | Input-abort and stay-on-failure drill through the full stack (manager + service + proxies) | Yes |
| `af_refocus.py` | Repeat autofocus until it succeeds, for recovering a multi-peak field | Yes |
| `af_bench.py` | Adaptive-v1 bench. Also a small library (`CameraPump`, timing helpers) imported by several other tools | Yes |
| `af_bench2.py` | Adaptive-v2 bench, kept for reproducing the v2-era measurements | Yes |
| `camera_benchmark.py` | Connects every candidate camera backend and measures resolution, fps, exposure, WB and colour into a decision matrix — the tool to run for a *new* camera | Yes |
| `camera_flip_check.py` | Proves the image flip reaches the camera path (on/off frames differ by 180°) | Yes |
| `ui_hw_bench.py` | 4K/1080p snapshots, scale-bar burn, auto-gain convergence, autofocus regression | Yes |
| `cam_gap_diag.py` | Frame-delivery stall diagnosis at a given exposure/gain/WB over N minutes — used to catch a marginal USB3 cable | Yes |
| `frame_stall_probe.py` | Correlates frame stalls with stage motion (the EMI question, answered) | Yes |
| `smartcam_probe.py` | Dumps the camera's full parameter table plus its function and event lists | Yes |
| `smartcam_live.py` | The SmartCamApi verification ladder (baseline, exposure, colour, sweeps, WB, full resolution) | Yes |
| `smartcam_decode_study.py` | Offline decode study: raw buffers against a ZEN snapshot as ground truth | No |
| `validate_cv.py` | Flake and edge CV on recorded images; writes annotated PNGs and a report | No |
| `ui_shot.py` | Screenshot rig for UI work and visual review — grabs each workspace and window at a chosen size (run it against a real display; offscreen renders placeholder glyphs) | No (sim) |
| `ui_scale_check.py` | End-to-end scale-bar pixel check across the three render paths | No (sim) |
| `make_qss_assets.py` | Regenerates the bundled stylesheet glyph PNGs into `resources/qss/` | No |
| `migrate_settings.py` | Imports the precursor projects' settings into `%APPDATA%\TALOS\settings.json` (idempotent, `--dry-run`) | No |

## Adding an instrument

1. **Pick the interface.** `Camera`, `FocusStage`, `XYRStage`, `XYZStage`, `TemperatureController` or
   plain `AbstractDevice`, all in [talos/hal/base.py](../talos/hal/base.py). If your device is a new
   *kind* of thing, add an ABC there (and a `Null*` implementation, as `Nosepiece` does) rather than
   special-casing it in the UI.
2. **Write the driver** in `talos/hal/devices/<name>.py`. Rules: blocking calls with a timeout on
   every transaction, no Qt imports, `stop()` callable at any time and never raising, and never
   trusting a write without reading back the thing that matters.
3. **Write its simulator** in `talos/hal/sim/` and export it from `hal/sim/__init__.py`. Model the
   behaviour that bites on real hardware — busy rejections, limit flags, latency — not just the happy
   path; that is what makes `--sim` worth running.
4. **Register it** in [talos/hal/registry.py](../talos/hal/registry.py): the real class in `_REAL`,
   the simulator in `_SIM`, and the key in `DEVICE_KEYS` — plus `MOTION_KEYS` if it moves, so STOP
   ALL and the shutdown ladder cover it.
5. **Add settings defaults** under `devices.<key>` in `resources/defaults/default_settings.json`.
   That file is the schema; the user's file only stores differences and is migrated on load.
6. **Optional — live reconnect:** add the keys whose change should rebuild the driver to
   `_CONNECTION_KEYS` in [talos/instruments.py](../talos/instruments.py).
7. **Optional — a Preferences page:** one `pages.append(...)`, implementing `_apply()`.
8. **Test it:** unit tests against `tests/testing/fake_serial.py` (or a fake camera), and a
   closed-loop sim test if it moves. Then run `tools/smoke_test.py` and the relevant bench tool
   against the real instrument before trusting it in a job.

If it speaks Modbus RTU or ASCII lines, the transport is already written in
[talos/protocols/](../talos/protocols/). The job model, abort paths, safety latches and the whole UI
are hardware-independent and do not change.

## Packaging

```powershell
pwsh -File packaging/build.ps1     # -> dist/TALOS/talos.exe (one-dir, portable)
```

[packaging/talos.spec](../packaging/talos.spec) is a one-dir PyInstaller build: it bundles
`resources/defaults`, `resources/qss` and `resources/icons`, and excludes the unwired camera
backends plus Qt WebEngine/QML/Charts. Freeze early if you add a dependency that imports its
extensions lazily — the two bugs the frozen build surfaced (gamepad initialisation and proxy
shutdown) were invisible from source.

Drop an icon at `resources/icons/talos.ico` and it is embedded automatically; see
[resources/icons/README.md](../resources/icons/README.md).

## Documentation rule

Every document under `docs/` should be reachable from the README, and every document should link
back to what it depends on. A doc nobody links to is a doc nobody reads.
