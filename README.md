# TALOS — Transfer and Alignment Laboratory Operating System

Integrated hardware control, computer vision and automation for 2D-material
transfer and stacking on a Zeiss Axiolab microscope. One PySide6 desktop
application drives four serial/Modbus instruments and the camera, runs the
autofocus and grid-scan workflows, and records the data they produce.

Built for a real bench, so the hard-won details — firmware quirks, timing
limits, failure modes — are documented in the code, in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and in the hardware notes under
[docs/](docs/).

## Safety first

This software commands physical motion. Read this before running it against
hardware.

- **`Esc` is STOP ALL** — focus → transfer stage → XYR, verified within a
  1.5 s budget. It works from the main window and from the Stage Control / AF
  dialogs, and it *latches*: motion commands stay suppressed until every input
  source (keys, on-screen holds, gamepad) is at rest. On the gamepad,
  **LB+RB does the same** (hold both bumpers for 0.2 s); its Start button does
  NOT stop everything — it toggles the enable gate of the stage Back selected
  (which halts that stage and drops its commands).
- **The focus stage has NO limit sensor.** A physical restrain protects the
  objective; the firmware's soft limits are the software bound — and the
  firmware ships with those limits **disabled** (`SLIM=0`), which the app now
  reports loudly at autofocus start and compensates for by clamping the search
  window itself. The residual risk is lost steps, not a crash.
- **Autofocus is bounded**: window clamped inside the soft limits, readback
  verification after every move, and an abort that is polled *while the axis
  travels* (a manual jog or STOP ALL stops the stage immediately).
- **Manual motion is gated by mode**: during a grid scan the jog inputs are
  refused, and during autofocus any input aborts the run and takes over.
- **Zolix motion verifies its registers**: every fixed-length move reads the
  distance register back before issuing the motion opcode (a CRC-glitched frame
  must never become a wild move — this follows a real hardware incident).
  Absolute moves are composed from validated relative moves because the
  controller's absolute opcode silently no-ops.
- **The Zolix controller re-zeros on power-cycle** — saved coordinates are
  session-relative; re-establish references after a controller reset.
- **The camera needs exclusive USB access**: close ZEN and LabscopeService
  before starting TALOS (and vice versa).

Run `python -m talos --sim` for a fully simulated bench — it exercises the same
code paths with simulated devices and no hardware.

## Hardware

| Instrument | Interface | Driver module |
|---|---|---|
| Zeiss Axiocam 208 color camera | SmartCamApi over libusb0 (ZEN/Labscope driver) | `talos/hal/devices/camera/smartcam_backend.py` |
| Zolix ZC300 XYR sample stage | Modbus RTU, 115200 8N1 | `talos/hal/devices/zolix.py` |
| DIY SigmaKoki XYZ transfer stage | ASCII line protocol over Arduino | `talos/hal/devices/sigmakoki.py` |
| DIY focus stage (Arduino + CRD5103PB) | ASCII line protocol, **no limit sensor** | `talos/hal/devices/focus.py` |
| Yudian AI-828 temperature controller | Modbus RTU | `talos/hal/devices/yudian.py` |

Ports, baud rates, speeds and limits live in
`resources/defaults/default_settings.json` and are editable in Preferences
(the user's copy is `%APPDATA%\TALOS\settings.json`).

Camera notes: the 208 works over SmartCamApi at ~15–20 fps in 1080p with
exposure/gain/white-balance control, plus 4K live and 4K snapshots. There is
**no hardware fixed-white-balance gain pair** (that parameter is read-only), so
"fixed WB" means AWB off plus a color temperature. See
[docs/CAMERA_208.md](docs/CAMERA_208.md) and [docs/SMARTCAM_API.md](docs/SMARTCAM_API.md).

## Install and run

```powershell
conda env create -f environment.yml
conda activate talos

python -m talos          # real hardware
python -m talos --sim    # simulated devices (safe, no hardware)
```

The first real launch applies the saved defaults to the camera and connects
every enabled device; each device that fails to open is reported in the log and
the rest keep working.

## Workspaces

- **Navigation & Control** — live view, quick actions (snapshot, AF-S, origins),
  and foldable right-panel groups for Capture / Camera / Autofocus / Temperature.
  Manual jogging happens in the **Stage Control** dialbox (Windows menu) or with
  the keyboard/gamepad. The Autofocus group also holds the **AF measurement
  region** (whole frame, or a ROI you drag or type in — drawn on the live view
  as a dashed "AF ROI" box).
- **Sample Finding** — flake detection on the current frame, bounded autofocus,
  and the grid scan (serpentine waypoints + manifest + frames on disk). A
  **scan-path panel** at the top of the viewport previews the serpentine grid
  from the current field values and highlights the row/column the run is on.

Menus: **File**, **Edit → Preferences** (`Ctrl+,`), **Display** (overlays, scale
bar), **Windows** (AF Detail, Stage Control, Log), **Help**.

Display toggles the live-view overlays: scale bar (with an optional burn-in for
snapshots), AF status pill, **crosshairs** (solid, inverse-video — the line
inverts whatever is under it, so it stays visible on any image) with an
optional **crosshair ticks** child that turns them into a calibrated measuring
reticle, **tick ruler** (calibrated major/minor ticks on all four frame edges,
labelled in µm from the frame centre — the centre is left unlabelled, the
crosshair marks it) and the scan path.

Preferences holds General, Objectives & Calibration, AutoFocus, Input & Gamepad
and the Hardware device pages (Camera, Focus, Zolix XYR, SigmaKoki XYZ,
Temperature), grouped by kind (Connection / Manual controls / Axis direction /
Scale …). Ports are picked from the ports actually present (a configured port
that is not detected is kept), and every
page scrolls. Workspace-dependent camera settings (exposure, gain, WB,
auto-gain) live in the right panels, not in Preferences — they differ per
workspace by design.

Two Preferences points worth knowing:

- **Camera flip** (Hardware → Camera → Image orientation, default ON) rotates
  every frame 180° so the optically inverted image reads in real-world
  orientation. It applies to the live view, autofocus, flake detection *and*
  saved snapshots, and is deliberately independent of the stage axis inversion
  (flipping the camera never inverts a stage, and never changes the scan
  direction). Changing it mid-session mirrors the AF region and clears the
  detected-flake table, both of which are tied to the old orientation.
- **Connection settings reconnect on Apply**: changing a port, slave address
  or timeout rebuilds that device's driver immediately (the LED shows
  CONNECTING while it swaps). It is refused — with a log line — while a scan or
  an autofocus run owns the axes; re-apply after the job.

## Input

`Esc` = **STOP ALL** (the only global stop). Gamepad:

| Control | Action |
|---|---|
| left stick | X/Y jog on the transfer (SigmaKoki) stage, analog |
| right stick | XYR stage, 8-direction |
| D-pad | X/Y of the *selected* stage — short press = one step, hold = continuous |
| **Back** | cycles which stage the D-pad drives (transfer ⟷ XYR); the status bar shows the current choice |
| **Start** | toggles the enable gate of the D-pad-selected stage (as in the reference project). Disabling stops that stage immediately and drops its commands; the strip's Enable checkbox follows |
| triggers (L/R) | focus, analog speed |
| **LT + RT** (both) | **autofocus once** |
| **LB + RB** (both) | **STOP ALL** — same as Esc, including the latch |
| A / B / X / Y | temperature presets 1–4 |

The two gestures are edge-triggered after a **0.2 s hold**, so a bump past a
bumper cannot stop a running job by accident. While a gesture is held it owns
its inputs: LT+RT stops any focus jog and emits none (otherwise the gesture's
own trigger imbalance would abort the autofocus it just started), and LB+RB
stops a stick-driven jog. A single bumper keeps its normal meaning (fast
modifier), as does a single trigger (focus jog).

Keyboard: arrows = X/Y, R/F = Z, `+`/`-` = focus (hold for continuous, tap for a
step), Shift = fast. On-screen hold buttons behave the same way and release on
pointer-leave, window hide, or Esc.

### Inverting the controls

Every axis can be inverted for *manual* motion, and each stage can swap its two
axes (X↔Y), in **Preferences → Hardware → the device → Axis direction**; the
focus jog direction and the gamepad's per-stick axes live on their own pages.
A `flip X↔Y` swap means the Stage Control panel's X+ button drives the physical
Y axis — the hint on the page says so.

These affect manual motion only (keyboard, gamepad, on-screen buttons and holds,
the dialbox). Position readback, autofocus, the flake "go to" move and the grid
scan are computed motions and are never inverted — that would silently corrupt
stored coordinates. The camera flip is separate again: it rotates the image and
never touches an axis.

## Calibration

The canonical calibration is **µm per 4K-sensor pixel**. The live 1080p stream
covers 2× that per pixel, and the 4K snapshot path uses it directly; both the
scale bar and the burned-in snapshot bar are derived from one shared layout
module. Objective µm/px values are editable per objective in
**Preferences → Objectives & Calibration** and are stored in
`%APPDATA%\TALOS\calibration.db`.

A measurement wizard exists (`talos/cv/calibration.py`) but is not needed on a
bench whose values are already known: the table is the source of truth, and the
app materializes it once at first launch.

## Tests

```powershell
python -m pytest                  # everything — the real-time simulations dominate
python -m pytest -m "not slow"    # the fast subset (order of a minute)
```

About 750 tests: unit tests for the drivers/CV/settings/UI wiring, integration
tests for the job/stop/reconnect machinery, and closed-loop **simulations**
(marked `slow`) that run the autofocus strategies and the grid scan against
simulated focus curves and check that they land on the true focus position.
Hardware-in-the-loop checks live in `tools/` rather than in the suite, so the
suite needs no instruments.

## Tools

Hardware benches and probes (each is standalone; most refuse to move anything
without an explicit confirmation flag):

| Tool | Purpose |
|---|---|
| `tools/smoke_test.py [--motion]` | ordered, cautious device checklist |
| `tools/dev_console.py` | read-only device console (no motion) |
| `tools/sigmakoki_latency.py --yes` | continuous-jog ack/latency drill |
| `tools/scan_manager_bench.py --yes` | grid scan through the manager + abort drill |
| `tools/ui_hw_bench.py` | camera snapshot/auto-gain/AF regression drills |
| `tools/focus_firmware_probe.py`, `tools/cam_gap_diag.py`, `tools/frame_stall_probe.py` | serial and frame-delivery forensics |
| `tools/af_bench*.py`, `tools/af_abort_bench.py`, `tools/autofocus_hardware.py` | autofocus benches (one per algorithm generation) |
| `tools/smartcam_*.py`, `tools/camera_benchmark.py` | camera backend/decode studies |
| `tools/ui_shot.py`, `tools/ui_scale_check.py` | UI screenshot rig + scale-bar pixel check |
| `tools/camera_flip_check.py` | live view vs a saved snapshot, to confirm the flip orientation by eye |
| `tools/validate_cv.py` | flake/edge CV on recorded images |
| `tools/hardware_scan.py`, `tools/af_refocus.py` | standalone scan / refocus helpers |
| `tools/make_qss_assets.py`, `tools/migrate_settings.py` | asset generation, first-run settings import |

## Packaging

```powershell
pwsh -File packaging/build.ps1     # -> dist/TALOS/talos.exe (one-dir, portable)
```

The frozen build excludes the unwired camera backends. Drop an icon at
`resources/icons/talos.ico` and it is embedded automatically (see
`resources/icons/README.md`).

## Project layout

```
talos/
  app.py, bootstrap.py        application shell, startup wiring
  config.py, paths.py         settings (JSON, layered over defaults), locations
  instruments.py              InstrumentManager: single writer for every device
  models.py                   shared dataclasses (positions, statuses, jobs)
  calibration_store.py        SQLite objective/calibration table
  hal/                        drivers, proxies (one worker thread per device), sim devices
  cv/                         autofocus strategies, flake detection, grid scan, metrics
  input/                      gamepad + keyboard/UI resolver (all motion gating)
  ui/                         windows, dialogs, widgets, workspaces, theme
tests/                        unit / sim / integration suites
tools/                        hardware benches and probes (not part of the app)
arduino_firmware/             focus-controller firmware sources
docs/                         architecture, hardware notes, development journal
```

## Known limitations / deferred

- The grid scan captures frames only if a camera is wired to it; today's
  workspace leaves the camera detached, and missing frames are now counted and
  reported rather than faked (the manifest keeps an empty frame column).
- Flake identification (as opposed to detection) is not implemented yet.
- GenTL / Micro-Manager / MCam / DirectShow camera backends stay in the tree as
  stored knowledge but are unwired: they need a USB3-Vision driver this bench
  does not have, or are far too slow (see
  `talos/hal/devices/camera/__init__.py`).
- The temperature controller is monitor-only when its serial link is flaky;
  setpoint writes are gated by the configured safety range.
- The live view receives frames as queued signal payloads rather than pulling
  from the shared frame slot; the shared slot is already used by autofocus.

## License

TALOS is free software under the GNU General Public License, version 3 (only) — see [LICENSE](LICENSE); for commercial licensing, contact the author.
