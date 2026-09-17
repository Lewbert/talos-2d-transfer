# TALOS — Design

How TALOS is put together, what the bench taught, and where the build diverged from the plan.

This document is the *why*. The *how it works today* — layers, threading, the proxy life-cycle, the
job model, the current known weaknesses — is [ARCHITECTURE.md](ARCHITECTURE.md). The autofocus
algorithm has its own document, [AUTOFOCUS.md](AUTOFOCUS.md).

## Provenance

TALOS was written for one microscope bench — an Olympus BXFM frame carrying a Zeiss camera, with
Arduino-built motion hardware alongside the commercial stages, i.e. a hybrid and heavily DIY setup —
by the person who uses it, starting from a written build plan and then a session-by-session
development record. This document distils that plan and that record, updated to match the current
code: the hardware research, the decisions that survived, the milestones and what each one actually
taught, and the places where the finished application disagrees with the plan it started from. Where
the plan and the code disagree, the code wins — and the interesting disagreements are listed under
[Divergences](#divergences-from-the-original-plan).

The session record itself is not published: it is a working log containing local paths, bench
chatter and superseded states. What survived it is here.

## Scope

TALOS is a single desktop application that owns an entire 2D-material transfer bench:

- the **camera**, and everything derived from its frames (live view, snapshots, autofocus metrics,
  flake detection, calibrated overlays);
- the **XYR sample stage**, the **XYZ transfer stage** and the **focus axis** — all commanded motion;
- the **hot stage**, whose setpoint is safety-clamped;
- the **automation** built on top: bounded autofocus, grid scan, flake finding, calibration.

Two things it deliberately is *not*: it does not replace the microscope's own software for what that
software is good at (ZEN remains the place to configure the camera deeply — TALOS needs the camera's
USB interface exclusively while it runs), and it does not try to be general-purpose. The generality
lives in the HAL, not in the UI.

## The bench

These are the instruments of the bench the project is currently developed against — what the author
works with and has implemented. Nothing in the application architecture depends on them: the HAL is
the extension point, and the movers here are a mix of commercial controllers and DIY Arduino builds.
See [Adapting TALOS to your hardware](../README.md#adapting-talos-to-your-hardware) in the README.

### Zolix ZC300 XYR sample stage — Modbus RTU

115200 8N1, slave 1. Function codes 0x03/0x04 for reads, 0x06 for single-register writes, and
**0x10 only for the opcode block** (three registers for a move, two for a stop, one for a save;
anything else is answered with exception 0x03).

| Item | Where |
|---|---|
| Position (x, y, r) | 30016–30021, IEEE-754 float32 big-endian pairs |
| Status | 30012–30015 (bit 9 = emergency stop) |
| Opcode | 30050 — 0x0064 absolute, 0x0065 fixed-length, 0x0066 continuous, 0x0067 decel stop, 0x0068 immediate stop, 0x0069 home, 0x006D save parameters |
| Axis / direction | 30051 (0x31 X, 0x32 Y, 0x33 R) / 30052 (0x50 +, 0x4E −) |

Scale: **0.625 µm/pulse** in X and Y, **0.00125 deg/pulse** in R.

Two behaviours shape the driver. First, **the absolute-move opcode silently does nothing** on the
real controller — accepted, acknowledged, no motion — so every absolute move is composed from
validated fixed-length moves, each of which reads its distance register back before the motion
opcode is issued. That readback exists because of a real incident: a CRC-glitched frame once turned
a short move into a wild one. Second, a decelerating stop is verified and escalates to an immediate
stop if the axis is still moving.

### DIY SigmaKoki XYZ transfer stage — ASCII over Arduino

115200 8N1, Arduino + Autonics MD5-HD14 drivers. `MV:<axis>:<dir>:<level>` for continuous motion,
`STEP:<axis>:<dir>:<n>` which replies `OK:STEP:X:<actual>`, `SPD`, `HOME`, `STOP:<axis>`/`STOP:ALL`,
`LIMITS?`, `STATUS?`, and asynchronous `ERR:*` / `EV:LIM:*` events. Scale **0.5 µm/step** in X/Y and
**0.25 µm/step** in Z, with per-axis limit switches.

### DIY focus axis — ASCII over Arduino, **no limit sensor**

115200 8N1, Arduino + CRD5103PB driving the fine-focus knob: 0.36°/step, 200 µm per revolution →
**0.2 µm/step**, open loop, no encoder. Commands: `STATUS?` (position, mode, speed, limits, soft
limits), `MOVE:<rel>`, `GOTO:<abs>` (trapezoidal), `SPD:<signed>` (0 = ramp stop), `STOP`, `ZERO`,
`MVSPD`, `SLIM:0|1`, `SLIM:SET:<lo>:<hi>`, `AWOFF`, `CUTB`, `CFG:MAX/ACC/TMO`. Speeds are clamped to
10–5000 steps/s with 20 000 steps/s² acceleration, and the firmware stops the axis after 5 s of
serial inactivity — treated by TALOS as a safety feature, not a fault.

The missing limit sensor is the single biggest safety consideration in the project, and it is why
soft limits, the search-window clamp and the abort-while-moving path exist. The full protocol,
including the boot banner, error codes and EEPROM persistence, is
[hardware/focus/protocol.md](hardware/focus/protocol.md); the firmware is
[arduino_firmware/focus_controller/](../arduino_firmware/focus_controller/).

### Yudian AI-828 temperature controller — Modbus RTU

9600 8N1, slave 1, function codes 0x03/0x06. Setpoint write at register 40001; process value at 75,
setpoint readback at 76, output percentage at 77, decimal point at 13. Setpoint writes are clamped
to a configured safety range before they are sent, and the value is read back afterwards.

### Zeiss Axiocam 208 color — SmartCamApi over libusb0

The camera is bound to the vendor's `SmartCamApi` library, which is why ZEN and LabscopeService must
be closed while TALOS runs: the interface is exclusive. Practical numbers and how they were
established are in [CAMERA_208.md](CAMERA_208.md) and [SMARTCAM_API.md](SMARTCAM_API.md); the short
version is ~15–19 fps in 1080p with exposure/gain/white-balance control, 4K live and 4K snapshots,
and one significant quirk — **every parameter query stalls frame delivery for ~330 ms**, which is
why properties are cached rather than polled.

### The nosepiece

This microscope can neither sense nor rotate its nosepiece, so the active objective is set by hand
in the UI and the HAL ships a `NullNosepiece`. The slot exists so that a future motorised nosepiece
is one adapter and one registry entry, not a change to the calibration model.

## Protocols, not device code, are the shared layer

Two wire protocols carry four of the five instruments: Modbus RTU
([talos/protocols/modbus_rtu.py](../talos/protocols/modbus_rtu.py)) and a line-based ASCII protocol
([talos/protocols/ascii_line.py](../talos/protocols/ascii_line.py)). Both are pure Python with no
Qt, which is what makes the drivers unit-testable against a scripted fake serial port
(`tests/testing/fake_serial.py`) — including the fault cases that matter: CRC corruption, torn
lines, timeouts, and a device that answers late.

Drivers are blocking and synchronous, with a timeout on every transaction, and never import Qt. The
threading lives one layer up, in the proxies. That split is the reason a new instrument can be
developed and tested without touching the application at all.

## Decisions that still hold

| Topic | Decision | Why |
|---|---|---|
| UI framework | PySide6 | LGPL, one dependency for widgets + signals, dark stylesheet |
| Camera | HAL with pluggable backends; hardware decides | The plan expected GenTL; the bench said SmartCamApi (see divergences) |
| Gamepad | In v1, and the recommended input | Best ergonomics for bench work — the whole manual workflow without reaching for anything, and gloved hands do not fight a pointer. Keyboard is an adequate second; the mouse-driven Stage Control window is a deliberate fallback (it also serves as a touch panel), never the primary route |
| Input path | One resolver, every source | Keyboard, gamepad and on-screen buttons all route through the same gate, so the mode locks and the STOP ALL latch cannot be bypassed by using a different input |
| Settings | One JSON file, deep-merged over bundled defaults | The defaults file *is* the schema; the user file only records differences |
| Calibration | SQLite, per objective, measured beats imported | Values measured in TALOS take precedence over values imported from Labscope |
| Packaging | PyInstaller one-dir | Portable, no installer, no self-zipping |
| Threading | GUI thread + one worker thread per device | A blocked serial port can never freeze the UI |

## What the build taught

Each milestone was also an experiment on the hardware. The lessons, in the order they arrived:

| Milestone | What it taught |
|---|---|
| Manual control | Both DIY stages needed speed *levels* rather than a raw rate; the transfer stage's axes are direction-inverted relative to the wiring, which is why inversion is a per-axis setting and not a sign convention in the code |
| Autofocus (first pass) | The Laplacian-variance metric the plan specified is noise-dominated on this camera's grayscale stream; **Tenengrad** (fine) and **Brenner-k** (coarse) replaced it, and the metric is now a setting |
| Camera | GenTL, Micro-Manager and MCam all enumerate zero devices on this machine — the 208 is libusb0-bound and only the vendor API works. The unwired backends stay in the tree as stored knowledge |
| Grid scan | The absolute-move opcode is a silent no-op (see above), and a CRC-glitched frame produced a wild move. Both are now enforced in the driver: composed relative moves, and distance readback before every motion |
| Edge + flake finding | The wafer-vs-stage colour contrast is strong enough for a colour-threshold mask plus a rectangle fit — circle fitting was the wrong model for a hand-cut wafer |
| Grid scan, second pass | The scan could never capture a frame because it held a camera the camera worker owned. Reading the shared frame slot removed the camera from the scan entirely — and the same pass found that STOP ALL stopped the motion without stopping the RUN, because the abort flag was only ever set by the Abort button |
| Sample identification | A colour threshold is easy; making it behave is not. Hue has to wrap across the 0/179 seam, saturation needs its own floor or a red band matches every grey, and a sharpness gate tuned on colour-matched contours silently rejects every contrast-matched one |
| Calibration | The Labscope objective table imports cleanly but its units need resolving at import time; TALOS stores its own provenance for every calibration entry |
| Gamepad | Python's `inputs` package hangs on this machine; the gamepad is read as raw XInput through `ctypes` instead, which also removed a dependency |
| Packaging | The frozen build hung on two bugs (gamepad initialisation and proxy shutdown) that never appear in a source run — worth freezing early, not at the end |

## Divergences from the original plan

The plan was written before any hardware was touched. These are the places where reality won:

- **A third *Calibration* workspace was planned.** There is none: the objective table became
  Preferences → Objectives & Calibration, and the measurement wizards that would have filled the
  workspace were never needed on a bench whose values were already known.
- **The camera backend changed** from GenICam GenTL to SmartCamApi after GenTL enumerated nothing.
- **The focus metric changed** from Laplacian variance to Tenengrad/Brenner, and from a fixed
  constant to a setting.
- **The autofocus strategy registry was deleted.** The application runs adaptive v3 and only v3;
  the classic and v1/v2 controllers remain in the tree as stored knowledge, exercised by the
  closed-loop simulation suites.
- **Absolute Zolix moves became composed relative moves**, with readback verification.
- **Gamepad input moved off the `inputs` package** to raw XInput.
- **The settings schema is at version 7**, not the planned 2 — each bump is a migration in
  `talos/config.py`, which is also where removed keys are recorded.
- **The px→µm mapping stayed orthotropic.** The plan had it upgrade to a measured 2×2 jacobian once
  the calibration wizard existed. The bench settled it the other way: the X↔Y anisotropy is
  negligible and the stage-to-image rotation is extremely small, so the off-diagonal terms would buy
  a correction below the noise of everything downstream at the cost of a calibration step and a
  matrix inversion. The wizard's phase-correlation code stays in the tree as stored knowledge.
- **Flake identification arrived as a filter chain, not a classifier.** The plan left it open; what
  shipped is a stack of stages the operator switches on and off (colour match, contrast, size,
  sharpness, and so on) whose parameters are judged by eye against a processed live view. It finds
  what the operator points at and does not rank material — a deliberate limit, and the one most
  likely to be revisited.
- **The scan captures through the frame slot, not a camera handle.** The first implementation took a
  camera object, which cannot work in an application where the camera belongs to its own worker
  thread; the mailbox the autofocus controller already used turned out to be the right seam, and the
  scan now has no camera dependency at all.

## Risks, and which ones materialised

The plan's risk register, with the outcome attached:

| Risk | Outcome |
|---|---|
| Focus crash into the objective (no limit sensor) | Mitigated and still the top safety concern: bounded window, speed clamp, abort polled while moving. A physical restrain protects the objective; the residual risk is lost steps |
| Camera exclusive access | Real, and handled by the documented rule (close ZEN/LabscopeService) rather than by a workaround |
| Modbus wrong-register writes | Real — it produced the wild-move incident. Now a driver invariant: verified writes, no silent retries on motion |
| COM re-enumeration | Materialised; ports are discovered from what is present and a configured-but-absent port is preserved rather than dropped |
| Zolix unit mismatch | Did not materialise; the scale factors were confirmed against the controller |
| PyInstaller / GenTL failures | Half and half: GenTL was abandoned, and the frozen build did surface two real bugs |
| Autofocus converging to the wrong layer | Real, and the reason the metric, the ROI and the multi-peak handling all exist |
| Shutdown deadlock or orphaned motion | Materialised in the frozen build; the shutdown ladder and the proxy retirement path are the answer |

## Reading order

New to the code? [ARCHITECTURE.md](ARCHITECTURE.md) first, then this document, then
[AUTOFOCUS.md](AUTOFOCUS.md) if you intend to touch anything that moves the focus axis. If you are
adding an instrument, the checklist is in the README and the long form — with the bench tools for
validating a driver against real hardware — is in [DEVELOPMENT.md](DEVELOPMENT.md).
