# TALOS architecture

Written for someone about to change this code. The README covers what the app
does; this covers how it is put together, which rules are load-bearing, and
which failures the existing design already paid for.

## Layers

```
talos/ui/            PySide6 windows, dialogs, widgets, workspaces
talos/input/         gamepad/keyboard/on-screen -> command tuples -> gates
talos/instruments.py InstrumentManager: the ONLY component that submits jobs
talos/hal/proxies/   one worker QThread per device (serial I/O lives here)
talos/hal/devices/   drivers: zolix, sigmakoki, focus, yudian, camera backends
talos/hal/sim/       drop-in simulated devices (same interfaces)
talos/cv/            autofocus strategies, flake detection, scan, metrics
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
   from the GUI, `_drain` from the worker). This is currently unsynchronised in
   the original design and is a known weakness — see "Known weaknesses".
5. **A blocking autofocus job owns the focus worker** for its duration, so
   anything queued behind it (a jog, a stop) waits. Abort therefore does not go
   through the queue: `FocusProxy.request_abort()` sets a flag that the
   controller polls, and `move_to_verified` polls it *while the axis travels*.

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
- **Session ownership**: scan and autofocus cannot both own the axes
  (`_set_job`); only the owner clears its mode.

## Autofocus

`AutofocusService` (UI-facing) builds a config from settings, submits a
`focus.autofocus` job to the focus worker, and reports progress/results on
signals. Inside the worker, `FocusProxy` constructs the controller and runs it.

The current strategy is **AF-S v3** (`cv/af_v3.py` + `cv/af_adaptive.py`
helpers): a 3-point probe classifies near/far, a continuous measure-while-moving
pass with a derivative guard ladder finds the peak, then a fine hill pass and an
overshoot-and-return landing take up the backlash. Older generations
(`cv/af_c.py`, `cv/af_adaptive.py` V1/V2, and the classic flow in
`cv/autofocus.py`) are kept as stored knowledge and are still exercised by the
closed-loop sim suites. Design history and the hardware measurements behind the
constants: [AUTOFOCUS.md](AUTOFOCUS.md) and [PLAN.md](PLAN.md).

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

## Testing model

- `tests/unit/` — drivers against scripted fake serial ports, CV maths, settings
  migrations, UI construction and wiring.
- `tests/integration/` — the manager's job/stop/shutdown machinery with real
  threads.
- `tests/sim/` — closed-loop simulations marked `slow`: the autofocus
  strategies and the grid scan run against simulated focus curves and stages,
  asserting where they land (within one fine step of the truth) and how they
  behave on abort, limits and flat scenes.
- `tools/` — hardware-in-the-loop benches. Not part of the suite (they need
  instruments), but they are the evidence behind most of the constants.

`python -m pytest -m "not slow"` is the fast loop; the full suite takes minutes
because the simulations run in real time.

## Known weaknesses (worth fixing deliberately)

- `DeviceProxy._queue` is mutated from the GUI and worker threads without a
  lock. The window is small but real: a command appended between the worker's
  sorted-rebuild and its reassignment is lost (its job never completes, so a
  busy gate can stick).
- The live view receives whole frames as queued signal payloads; pulling from
  the existing `LatestFrameSlot` would bound the backlog during GUI stalls.
- The autofocus job blocks the focus worker's queue, so a jog submitted during
  a run only executes after it returns (the abort path is unaffected).
