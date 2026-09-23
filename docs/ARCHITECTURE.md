# TALOS architecture

Written for someone about to change this code. The README covers what the app
does; this covers how it is put together, which rules are load-bearing, and
which failures the existing design already paid for. The *why* behind it — the
bench, the hardware protocols, and where the build diverged from its plan — is
[DESIGN.md](DESIGN.md); the tooling and test story is
[DEVELOPMENT.md](DEVELOPMENT.md).

## Layers

```
talos/ui/            PySide6 windows, dialogs, widgets, workspaces
talos/input/         gamepad/keyboard/on-screen -> command tuples -> gates
talos/instruments.py InstrumentManager: the ONLY component that submits jobs
talos/hal/proxies/   one worker QThread per device (serial I/O lives here)
talos/hal/devices/   drivers: zolix, sigmakoki, focus, yudian, camera backends
talos/hal/sim/       drop-in simulated devices (same interfaces)
talos/cv/            autofocus strategies, flake detection, scan, metrics,
                     pre-processing
talos/config.py      settings: defaults + user file + migrations
talos/calibration_store.py  SQLite objective/calibration table
```

Dependencies point downwards only: UI → input → manager → proxies → drivers.
`cv/` is called from the UI and from the focus worker's autofocus job.

## Threading model

| Thread | Owns | Notes |
|---|---|---|
| GUI thread | widgets, `AppState`, `InstrumentManager`, the 60 Hz input tick, the gamepad poll, `AutoGainController`, `AutofocusService` orchestration | never does blocking I/O |
| one QThread per device (`DeviceProxy`) | the driver instance and its serial port | created, connected and used exclusively inside that thread |
| one QThread (`CameraProxy`) | the camera backend, frame fetch loop | publishes frames and properties |
| focus worker (same thread as the focus proxy) | also runs the autofocus job inline | see below |
| scan `_Worker` QThread | a `GridScanner` run | submitted jobs, waits for completions |
| scan writer thread (`FrameWriter`) | `frames/`, `manifest.csv` | one queue, bounded: the encode, the thumbnail and the row leave the scan thread the moment a tile is captured |
| detection `_DetectWorker` QThread | the pre-processing chain and the identification pipeline, plus the scan's exports | one queue: live preview frames drop, scan tiles never do. Owned by the Sample Finding tab |

**Rules that keep this safe**

1. **Single writer.** Only `InstrumentManager.submit`/`submit_camera` enqueue
   work. Nothing else touches a driver object (the one historical exception —
   a scan building its own Zolix driver — could not even open the port, because
   the manager already owned it).
2. **Queued invocations, never direct calls across threads.** `DeviceProxy`
   moves work with `QMetaObject.invokeMethod(..., QueuedConnection)`; a bare
   `QTimer.singleShot(0, ...)` would run in the *calling* thread (this bug
   produced a multi-GB frame backlog and a frozen GUI).
3. **Bound slots, not bare lambdas, for cross-thread signals.** A lambda with
   no receiver executes in the *emitting* thread.
4. **The device queue is touched from two threads** (`enqueue`/`enqueue_stop`
   from the GUI, `_drain` from the worker), so every critical section of it is
   guarded by `_queue_lock`. The lock is never held across a driver call: a
   blocking serial transaction must not stall the GUI's enqueue path.
5. **A blocking autofocus job owns the focus worker** for its duration, so
   anything queued behind it (a jog, a stop) waits. Abort therefore does not go
   through the queue: `FocusProxy.request_abort()` sets a flag that the
   controller polls, and `move_to_verified` polls it *while the axis travels*.

## Device reconnect (proxy life-cycle)

A driver is constructed **inside its worker thread** from a factory that closes
over the settings dict and runs exactly once, so an edited port never
reaches a live driver. `InstrumentManager.reconnect(key)` is the supported way
to apply one:

1. record what the live driver was built with (`_proxy_cfg`, a *copy*),
2. stop the old worker through its own event loop — queued `enqueue_stop` →
   `cancel_pending` → `request_shutdown` — and join it,
3. build a fresh proxy (new factory → new driver → new `connect()`), re-wire
   the manager's signals, re-broadcast the enable gate, start it.

Rules that make this safe:

- **Retire, don't delete.** The replaced proxy is kept in `_retired` with its
  signals intact: completions already queued to the GUI thread (a
  `sig_af_done`, a `sig_command_done`) must still be delivered — destroying the
  sender drops them and an autofocus run would leave the service busy forever.
- **Identity, not key.** `_sender_key()` returns `""` for a sender that is not
  the live proxy for its key, so a retired proxy cannot speak for its
  replacement (same `device_key`, different device).
- **Queued jobs FAIL.** `cancel_pending()` emits `sig_command_failed` for every
  queued job; `request_shutdown` clears the queue silently, which would leave
  `StageAdapter._call` (and the UI's busy gates) blocked for their full
  timeout. Failure rather than "done" is deliberate: a done-with-None result is
  read as a *value* by the adapter.
- **Refused while a job owns the worker** (`FocusProxy.busy`) and deferred by
  the UI while a scan/autofocus owns the axes.
- Test note: a manager must be torn down deterministically (`del` + `gc.collect()`)
  — letting Python's collector free a proxy/thread graph inside a later nested
  event loop crashes the interpreter.

## Input mapping (manual motion)

`InputSystem._dispatch` is the single choke point every manual source goes
through (keyboard, gamepad sticks/D-pad/triggers, on-screen holds and clicks,
the dialbox). `talos/input/axis_map.py` maps `(axis, direction)` there:
flip X↔Y first, then invert the (possibly swapped) axis — the reference
project's order. Stops carry direction 0, so they follow the axis flip and are
never negated.

Deliberately NOT mapped: position readback, autofocus, the objective focus
offsets, the flake "go to" move and the grid scan. These are computed motions;
inverting them would silently corrupt stored coordinates and scan geometry. The
camera flip is likewise independent — it rotates the image, never an axis.

## Live-view overlays: two layers

- **Baked into the frame pixmap** (`LiveViewWidget._compose`): the
  inverse-video crosshair and the calibrated tick ruler, painted with
  `CompositionMode_Difference` against white (= |dst − 255| = invert). This is
  the only layer where a difference blend is correct: the overlay surface below
  is a translucent child repainted with every frame, so a difference pen there
  would compound against its own previous output. Baking also clips the lines
  to the frame (no drawing across the letterbox bars) and puts them on the
  frame's true centre. `_frame_pixmap` stays CLEAN; `_compose` copies it.
- **On the overlay surface** (a mouse-transparent child stacked above the frame
  label — Qt paints children after the parent): the AF ROI outline + tag, the
  drag rubber band, the scale bar and the AF status pill. The three-view
  switch is a sibling of the live view rather than part of this stack, and it
  is repositioned over the view's top edge on every resize. None of these is
  ever in the pixels the pipeline sees. Their geometry is mapped through
  `fit_transform`/`roi_for_resolution` so they stay frame-anchored at any
  window size.

The ruler's spacing comes from the same 1/2/5 ladder as the scale bar
(`talos/cv/ruler.py`), and its offsets are measured from the FRAME CENTRE, so 0
sits on the optical axis and the labelled µm span does not change with resizes.

## Image orientation (camera flip)

`devices.camera.flip` (default ON) rotates every decoded frame 180° so the
optically inverted bench image reads in real-world orientation. It is applied
in the `Camera` ABC's `apply_flip`, called at each backend's SINGLE frame
egress — `_decode` for smartcam/mcam, `_render` for the sim, `fetch` for the
rest — which is what puts it *before* the scale-bar burn: a flip applied later
would mirror the bar and its label into the corner of every saved snapshot
while the live view looked correct. Each backend declares `APPLIES_FLIP` and a
registry test walks them, because a backend that forgets leaves the live view
corrected and the autofocus/detection path not.

Cost: `cv2.flip` ≈ 3.6 ms at 1080p (~8 % of a 50 ms frame), ≈ 15 ms once per 4K
snapshot. Toggling it mid-session mirrors `autofocus.default_roi_norm` and
clears the flake table — both are frame-space state that a 180° rotation
invalidates.

## The job model

`InstrumentManager.submit(device, method, *args, priority=0)` wraps the call in
a `Job` and hands it to the device's proxy:

- **Priority queue**: `PRIORITY_STOP` (1) beats `PRIORITY_NORMAL` (0), so a
  release-stop always runs first.
- **Coalescing**: for methods listed in `_COALESCE_KEYS`, only the newest
  queued value survives (per axis where axes matter). A held jog re-emits at
  input rate; without this the queue acts on stale positions. Superseded jobs
  complete with `None` so job accounting stays consistent.
- **Stop purge**: `enqueue_stop()` drops queued *continuous* motion commands
  (a stop followed by a stale `set_speed`/`move_continuous` would restart the
  axis with nothing held — hardware-verified). Discrete commands (single steps,
  go-to) survive.
- **Completions** are signals: `sig_command_done(job_id, result)`,
  `sig_command_failed(job_id, type, message)`. The UI gates busy states on
  them; `GridScanner`'s adapter waits on them.

Telemetry is polled by a `QTimer` inside each worker (default 100 ms), skipped
while commands are pending (interleaving status reads with motion commands
garbles the firmware's replies), and backed off on repeated failures. A driver
that exposes `get_telemetry()` is polled once for status+position instead of
twice (sigmakoki's `STATUS?` carries both).

## Safety interlocks

- **Mode gate** (`AppState.mode`): `MANUAL` allows everything; `SCAN` refuses
  manual starts (releases always pass); `AUTOFOCUS` passes input and lets it
  abort the run. Every motion source goes through `InputSystem._dispatch`,
  including on-screen clicks and the ZERO/home button — those used to bypass it.
- **ESC latch**: after STOP ALL, motion stays suppressed until keys, on-screen
  holds and the gamepad are all at rest. On-screen holds are also dropped at
  that moment (a surviving claim would re-command its axis once the latch
  cleared) and expired by a 60 s dead-man timeout.
- **Hold release paths**: pointer-leave, window hide, Esc, and the timeout.
  The widget's `released` signal used to be the only one, so a hidden window
  left the axis jogging.
- **Soft limits**: the focus firmware gates every limit check on a SLIM flag
  that ships OFF, and reading the bounds is not evidence of enforcement.
  Autofocus always clamps its window in software and warns when the firmware
  flag is off.
- **Stop path**: `STOP ALL` walks stop jobs through the motion devices' queues
  (focus first) with a 1.5 s budget, and reports when acks are missing.
- **Session ownership**: one owner at a time, and the gate is
  `AppState.mode`. A scan takes it before the first move (and before the
  camera switch that precedes it) and holds it for the whole run; manual
  motion, autofocus, snapshot and the device enable gates are refused while
  it does — see "What a run owns" in `docs/SCAN.md`. Autofocus takes it for
  its own run and clears it only if the mode is still AUTOFOCUS (a scan that
  started meanwhile keeps it).

## Autofocus

`AutofocusService` (UI-facing) builds a config from settings, submits a
`focus.autofocus` job to the focus worker, and reports progress/results on
signals. Inside the worker, `FocusProxy` constructs the controller and runs it.

**Measurement region.** Which part of the frame gets scored is ONE persisted
preference (`autofocus.default_roi_norm`, null = the whole frame), surfaced by
`ui/af_region.AfRegionController`. Three consumers share it: the right-panel AF
settings, the AF detail window (both embed the same `AfSettingsWidget`, and a
change in one refreshes the other) and the live-view overlay, which draws the
region it is about to measure. `AutofocusService._default_roi()` reads the same
key, so no caller passes a region — the panel used to pass its own copy, which
is how it could show "Full frame" while the run measured a crop.

The current strategy is **AF-S v3** (`cv/af_v3.py` + `cv/af_adaptive.py`
helpers): a 3-point probe classifies near/far, a continuous measure-while-moving
pass with a derivative guard ladder finds the peak, then a fine hill pass and an
overshoot-and-return landing take up the backlash. Older generations
(`cv/af_c.py`, `cv/af_adaptive.py` V1/V2, and the classic flow in
`cv/autofocus.py`) are kept as stored knowledge and are still exercised by the
closed-loop sim suites. Design history and the hardware measurements behind the
constants: [AUTOFOCUS.md](AUTOFOCUS.md) and [DESIGN.md](DESIGN.md).

## Settings

`resources/defaults/default_settings.json` is the schema; the user file is
`%APPDATA%\TALOS\settings.json` (override with the `TALOS_APPDATA` env var).
`Settings.load()` deep-merges the user file over the defaults, then runs
`_normalize()`, which performs versioned migrations (backfills, renames,
dead-key removal). A migration failure no longer aborts startup: the loaded
values are kept and the error is logged.

`Settings.save()` is atomic (temp file + replace) and single-threaded.

**UI contract**: `PreferencesDialog` calls `_apply()` on every page and
*silently skips* pages that do not implement it — every page added to the tree
must implement `_apply()` or its edits are discarded. Pages that edit
workspace-dependent camera settings do not belong here (right panels own those).
After Apply, `sig_applied` tells consumers that cache settings-derived state
(the calibration context, the hardware strip) to refresh.

## Calibration

The canonical unit is **µm per 4K-sensor pixel** (`cv/calibration.py:
SENSOR_WIDTH_PX = 3840`). A 1080p frame pixel covers 2× that; the scale bar,
the snapshot burn and the flake→stage mapping all convert from the same value.
`CalibrationContext` caches per nosepiece position and refreshes on objective
change, on Apply, and (defensively) if the database is unreadable — a corrupt
`calibration.db` degrades to a pixel-pitch estimate instead of blocking startup.

## Camera specifics

- **Profiles** (`ui/camera_profiles.py`): exposure/gain/WB/temperature are per
  workspace. A profile is applied on workspace switch and on reconnect (the
  reconnect path resets the diff baseline first — otherwise the diff is empty
  and the backend's connect-time defaults win).
- **WB semantics**: AWB is 3-state on the hardware, but a *stored* "Once" would
  re-run a balance pass on every profile apply, so TALOS stores On/Off only and
  exposes "Balance once" as a button. There is no hardware fixed-gain WB pair;
  fixed WB = AWB off + color temperature.
- **Frame path**: the DLL hands out a fixed-size acquisition buffer (the
  camera's max transfer size, 16.6 MB at every resolution); the live frame sits
  in its prefix and the decoder slices it with the current geometry.
- **Reopen guard**: the DLL can wedge on an immediate reopen after a close
  (>4 min, hardware-seen), so a process-global timestamp enforces a settle
  delay — stamped both on disconnect and on every failed-connect cleanup.
- **Parameter queries stall frame delivery** (~330 ms each, measured). Never
  query in a per-frame loop; `get_properties` caches its readback.
- **Orientation**: `flip` is a SOFTWARE property — it must never reach
  `ApiCam_SetParameterValue` (each backend's `set_property` routes it through
  `try_set_flip` first) and it is deliberately not a camera-profile key, since
  it is not per-workspace.

## The grid scan and the identification chain

The long form of both — the geometry and its coverage proof, the path orders,
the chain's stages and the tuning ladder — is in [SCAN.md](SCAN.md) and
[IDENTIFICATION.md](IDENTIFICATION.md). This section is the architecture: what
crosses which thread, and the rules that keep the two honest.

Two pieces of automation share one rule and one shape.

**Capture never touches the camera.** The scan runs on its own thread, and the
SmartCamApi binding is exclusive and not thread-safe, so a scan that called
`fetch()` would be reaching into a backend the camera worker owns. Instead it
reads the shared **frame slot** (`cv/frame_slot.py`) — the same mailbox the
autofocus controller polls — and its `LatestFrameSource`
(`cv/frame_source.py`) refuses a frame whose capture timestamp predates the
settle window. That gate is the point: the manifest records where the stage IS
(a readback) next to every frame, so a frame taken before the move would
mislabel the image. A capture that fails returns `None` and the waypoint is
counted **missing**; nothing is ever filed under a position it did not come
from. The CLI benches, which own their camera, use `CameraFrameSource` and the
same gate.

**Identification is a chain of stages** (`cv/identify.py`), not a mode: a
colour-match source stage produces a mask, one stage cleans it, then gates —
size in µm², frame edge, boundary sharpness — decide what survives, and a
merge stage joins fragments. Each stage carries its own parameters, switches
off independently, and reports how many candidates it let through. The engine
is order-respecting and kind-dispatched (`source` / `mask` / `gate` /
`merge`), so a new stage is a dataclass and a `RANGES` entry — no UI code and
no pipeline change.

**In front of the chain is pre-processing** (`cv/preprocess.py`): shade
correction, an edge-preserving denoise, the tone operations and the
local-contrast curve, in that order, off until switched on. It is a pure
transform — it never draws and is never written to disk, so the raw capture is
still what a snapshot, a scan tile and the mosaic contain. The point of it is
that the operator tunes filters against the same pixels the mask segments: the
dropper samples the pre-processed layer, the pipeline runs on it, and the curve
is pinned at the picked colour so that switching it on cannot make the sample
the operator pointed at disappear from the mask.

**The pipeline is fed from the frame slot and from nothing that draws.** The
live stream, the frame slot and every scan tile are raw; the only overlay burn
in the application is the opt-in scale bar on a snapshot copy, applied inside
the camera backend. That is worth stating as a rule rather than as a fact,
because the rule is what a future overlay would have to be checked against —
and it is why there is no longer a stage whose job was to reject the
application's own annotation.

**It always runs on ONE frame.** A scan's tiles are each identified on their
own, at full resolution, and the mosaic is never an input: merging tens or
hundreds of tiles is too expensive in compute and memory for what it would buy,
and per-frame results are what "go to sample" needs. That is also why the
mosaic tolerates a small gap or an imperfect seam — nothing measures from it —
and why the scan's px→µm mapping is orthotropic (see `cv/orientation.py` and
the jacobian note in `flake_to_stage`).

Three rules the implementation exists to keep: **hue wraps** across the 0/179
seam (a red target's tolerance is not one-sided), **the preview is honest**
(the live view processes a downscaled copy, but results come back in the
caller's pixels and µm — pixel-unit gates are scaled with it), and **the
pipeline never sees a half-edited config** (the panel rebuilds the whole config
per job; the worker never reads a widget).

**Stage↔image orientation lives in one place** (`cv/orientation.py`), because
two independent things decide it and everything else must agree with both.
The **mounting** is which way the optics put the specimen on the sensor —
bench-measured as `(x, −y)`: stage +X moves a feature right in the frame and
stage +Y moves it up. The **camera flip** is a 180° rotation of every
delivered frame, so it negates both axes. Together they give the per-axis
signs every consumer uses: the px→µm mapping (`flake_to_stage(..., flip=…)`,
behind the identification pipeline and "go to sample"), the mosaic layout and
the scan map.

The layout matters as much as the offsets: a frame taken at `p` shows the
sample point `s` at image offset `a·(s − p)`, so a mosaic must place that
tile at a position scaled by the SAME `a` — otherwise every feature lands once
per tile and the mosaic looks doubled. Placing tiles with the wrong sign and
rotating their content to compensate is what shipped first, and it produced
exactly that doubling plus a map that read upside down against the live view.

**The map is drawn in the sample's frame** (`ui/widgets/scan_map.py`): tiles at
their readback positions (mirrored with the flip, as above), the camera
footprint walking across them, the route, and a marker per find. Tiles are
drawn AS CAPTURED — they come from the same frame slot the live view does, so
the camera flip is already in the pixels.

**The live view has three modes, and the last two are the worker's output.**
*Original* is the stream. *Pre-processed* is the filter chain's result, which
is also the layer the dropper samples. *Samples* is a verdict rather than a
decoration: every region the source found is drawn with its own OUTLINE —
bright for what survived the chain, dim for what a gate threw away — over a
frame darkened everywhere it did not match. Deliberately not rectangles: boxes
round shapeless blobs overlap and read as one object, and they hide the shape
the operator is judging. The darkening keeps the sample's own pixels visible
inside the match, so it reads as "this part of the wafer", not as a mask
poster.

**Nothing here blocks.** Capture runs on the scan thread; pre-processing and
identification run on ONE detection thread fed by a queue (tiles queue — a
tile not examined is a sample not found; live preview frames drop instead — a
preview lagging the stream is worse than one that skips). That is what makes an
expensive filter acceptable: a denoise costing hundreds of milliseconds delays
the next preview and nothing else. Detection outlives the capture by design, so
the exports wait for the queue to drain. The Sample Finding tab is the only
place the stream is shown, and the detection worker builds both of its
processed views from the same array it fed the pipeline.

**The previews stand down when they would be wrong or in the way.** While an
axis the camera can see is moving (XYR, focus) the two processed views fall
back to the raw stream — a processed frame of a moving stage is a picture of
where the stage *was* — and for the whole of a scan the live feed is suspended
so the worker's time goes to the tiles. Neither touches the dropper (which
keeps reading the pre-processed layer) or the capture path.

## Testing model

- `tests/unit/` — drivers against scripted fake serial ports, CV maths, settings
  migrations, UI construction and wiring.
- `tests/integration/` — the manager's job/stop/shutdown machinery with real
  threads.
- `tests/sim/` — closed-loop simulations marked `slow`: the autofocus
  strategies and the grid scan run against simulated focus curves and stages,
  asserting where they land (within one fine step of the truth) and how they
  behave on abort, limits and flat scenes. The scan suite feeds the simulated
  camera into a frame slot exactly as the camera worker does, so the capture
  path is exercised rather than stubbed.
- `tools/` — hardware-in-the-loop benches. Not part of the suite (they need
  instruments), but they are the evidence behind most of the constants.

`python -m pytest -m "not slow"` is the fast loop; the full suite takes minutes
because the simulations run in real time.

## Known weaknesses (worth fixing deliberately)

- The live view receives whole frames as queued signal payloads; pulling from
  the existing `LatestFrameSlot` would bound the backlog during GUI stalls.
- The autofocus job blocks the focus worker's queue, so a jog submitted during
  a run only executes after it returns (the abort path is unaffected).
