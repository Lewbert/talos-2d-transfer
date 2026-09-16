# Zeiss Axiocam 208 color — TALOS camera documentation

Applies to the **Axiocam 208 / 208 color** family: every fact below was
established against a 208 color on a real bench, and the API details are
family-specific rather than universal.

Status: **fully operational** — 1080p color live ~15–19 fps (ZEN parity 20.2),
4K live mode, full-res-ish snapshots, exposure/gain/WB all settable and
hardware-verified.

Companion documents: [SMARTCAM_API.md](SMARTCAM_API.md) for the DLL
interoperability notes, [DESIGN.md](DESIGN.md) for where this camera sits in the
bench, and [ARCHITECTURE.md](ARCHITECTURE.md) for the acquisition loop.

## How it works

- The 208 is **not** a USB3-Vision camera. It binds via Zeiss's own
  `axiocam_208.inf` to **libusb0.sys**; ZEN and Labscope both talk to it
  through **SmartCamApi.dll** over that binding. GenTL (`harvesters`),
  pymmcore-plus, and MCam (axcam64.dll — that SDK is the Axiocam 503/506
  family) all enumerate **0 cameras** for the 208. SmartCamApi is the only
  working path — hence it is first in the backend chain.
- The full parameter API was extracted from ZEN's own managed wrapper
  (`Zeiss.Micro.Cameras.SmartCam.dll`, decompiled for interoperability —
  see [SMARTCAM_API.md](SMARTCAM_API.md)). Key facts:
  - `SetParameterValue`/`GetParameterValue` take value **pointers**.
  - Parameter IDs = ZEN's `ParameterKey` enum ordinals
    (`talos/hal/devices/camera/smartcam_params.py`).
  - Live = `StartContinuousAcquisition` + `ImageAcquired` event callback;
    snap = `StartSequenceAcquisition(1)` + poll `GetSequenceImage` + abort.
  - The transfer's "YUV420" is semi-planar with **V-first** chroma pairs;
    ZEN's software white-balance LUT (5500K preset) is applied on decode.
- ZEN defaults are applied at connect (`apply_defaults`): exposure 20 ms,
  gain 4, hardware AWB on. All are settable at runtime; the camera keeps
  values in memory across sessions but re-initializes some at open.

## Verified capabilities (hardware ladder, 2026-09-06)

| Capability | Verified result |
|---|---|
| Live color | purple wafer / orange copper correct (checked against a ZEN reference frame) |
| 1080p fps | ~15–19 fps mean (ZEN: 20.2) |
| 4K live (Resolution=0) | works, decoded through the same pipeline |
| Exposure | 0.061–1000 ms, monotonic luma sweep, fps tracks 1/exposure |
| Gain | 1–22x, monotonic sweep |
| WB | AWB off/auto/once + color temperature 1500–10000K (R/B direction verified) |
| Snapshot | sequence-acquisition still works |
| Parameter dump | 52 params enumerated — reproduce with `tools/smartcam_probe.py` |

## Facts learned after the first ladder (2026-09-14, bench)

- **The acquisition buffer is fixed-size.** `ApiCam_GetAcquisitionBufferSize`
  returns the camera's MAX transfer size (16.6 MB = 4K × 2 bytes) and it is the
  SAME at every resolution; every `ImageAcquired` event carries that size as
  `Param2`. The live frame occupies the buffer's PREFIX and is sliced with the
  current geometry — so 1080p and 4K both decode correctly from the same
  buffer (verified: mean abs diff 2.19/255 between the two after downscaling,
  colour statistics identical). Do not try to match the buffer size against
  `w*h*3/2`; a buffer SMALLER than the payload is the only real error case.
- **Every parameter query stalls frame delivery** (~330 ms, measured) — never
  query in a per-frame loop; `get_properties` caches its readback.
- **AWB does not latch.** The firmware runs its own balance pass during
  open/init (the first ~3–4 s), and setting the 3-state `AutoWhiteBalance`
  parameter to `Once` reads back as `0` rather than staying armed. TALOS
  therefore stores WB as On/Off only and exposes "Balance once" as a button.
- **There is no fixed-WB gain pair**: `WhiteBalance` (param 52) is read-only;
  fixed WB = AWB off + `ColorTemperature` (param 29).
- **The DLL can wedge on an immediate reopen** after a close (observed >4 min).
  `smartcam_backend` enforces a process-global settle delay after every close,
  including the failed-connect cleanup path.

## Usage rules

- **Exclusive access**: ZEN and Labscope must be CLOSED while TALOS uses
  the camera (and vice versa). TALOS closes the camera cleanly on exit.
- The camera resets some parameters (e.g. Resolution) at open — the
  backend re-applies configured defaults on every connect.
- Backend chain (auto): **SmartCamApi only**. The other backends (harvesters /
  GenTL, pymmcore-plus, MCam, DirectShow) are still in the tree as stored
  knowledge and can be selected by name in dev tooling, but they are unwired
  from the application: on this bench they either find no device (the 208 needs
  SmartCamApi over libusb0) or deliver ~2.5 fps / monochrome, which is unusable
  for this workflow. See `talos/hal/devices/camera/__init__.py`.

## Tools

- `tools/smartcam_probe.py` — live parameter dump (names/types/ranges/enums).
- `tools/smartcam_live.py --step baseline|exposure|color|exposure_sweep|gain_sweep|wb|fullres`
  — the verification ladder.
- `tools/smartcam_decode_study.py` — offline raw-buffer decode gallery,
  scored against a ZEN reference capture of the same field.
- `tools/camera_benchmark.py --backend smartcam` — fps/capability report.

## Known limitations / open items

- Full-sensor raw (the odd 5472×3600 buffer from acquisition index 2, seen
  during the sprint) is **not reachable** in the current firmware state —
  index 2 now returns a 1080p frame. 4K (3840×2160) live is the real
  maximum and works. Side note, not blocking.
- Color rendering matches ZEN within ~7–9/10 (scored against ZEN reference
  frames); the residual is brightness/contrast from bench illumination —
  tune exposure/gain in the UI.
- MJPEG transfer (TransferFormat=1) is supported by the camera but not yet
  wired in the decode path (would need cv2.imdecode); YUV (NV12-style) is
  the default and works.
