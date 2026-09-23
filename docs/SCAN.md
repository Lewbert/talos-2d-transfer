# The grid scan — covering an area without touching the focus

Companion documents: [IDENTIFICATION.md](IDENTIFICATION.md) for what happens to
the frames afterwards, [ARCHITECTURE.md](ARCHITECTURE.md) for the job model and
the threading, [DESIGN.md](DESIGN.md) for the bench and the design decisions.
The panel that drives it is the right-hand column of the **Sample Finding**
tab (`ui/widgets/scan_panel.py`); the map is `ui/widgets/scan_map.py`, and
double-clicking it opens the same widget in its own window
(`ui/widgets/map_window.py`).

## What a scan is, and what it deliberately is not

A scan walks a rectangle of the sample in a fixed pattern, stops at each tile
centre, captures a frame, and records where that frame was taken. That is all.

It does **not** autofocus per tile. Scanning happens at 5–20×, where the depth
of field is tens of µm and the stage's own repeatability is ~1.6 µm; the focus
that was set before the run is still good at the far corner of a few mm of
wafer, and stopping to hunt for focus at every tile would multiply the run time
for a correction nothing needs. Focus is the operator's business before the
scan, not the scan's business during it.

It does not rotate, and it does not change the objective. The scan drives the
XYR stage and nothing else: the focus and transfer axes are left exactly as the
operator had them, and the objective selector is disabled while a run owns the
axes (a manually-changed nosepiece mid-run would silently invalidate the field
of view the run is tiling at, and the objective id its manifest records).

## Geometry: the field of view comes from the calibration

The tile pitch is `FOV × (1 − overlap)` per axis, and **the FOV is derived, not
typed in**: the active objective's calibration (the value the operator
configured in Preferences → Objectives & Calibration) is expressed in µm per
4K-sensor pixel, so

    FOV_x = um_per_px_x × 3840        FOV_y = um_per_px_y × 2160

is resolution-independent — the same field of view whether the frame in hand is
a 1080p live frame or a 4K snapshot. On this bench the 5× is a measured
0.3557 µm/px, giving 1366 × 768 µm.

There is no manual field-of-view field, deliberately: two sources for one number
is a way for them to disagree, and the calibration is the one the rest of the
application measures with (µm² on every candidate, the scale bar, *go to
sample*). So the window **shows** the number and where it came from — `measured
in TALOS`, `imported from Labscope` — and if it can only estimate (no
calibration row for that objective), it says so and points at Preferences →
Objectives & Calibration, where correcting it fixes everything else too. The
field of view and the planned grid follow the objective selector live, so
changing objectives updates the tile count rather than leaving a stale plan.

The operator jogs to a feature they can see and presses *Scan from here*, so
the start is wherever the stage is at that moment — and `origin` says what
that position **means**. `plan_geometry()` is the one function that turns the
area, the origin and the pitch into a grid, and both the run and every
preview of it call it: a preview that disagrees with the plan is worse than
none, and the corner modes make that easy to get wrong by hand (the tile
count depends on the field of view, not just on the area).

| Origin | The start position is | Tiles per axis | Coverage |
|---|---|---|---|
| **Centre** *(default)* | the middle of the first tile | `ceil((area − FOV/2) / pitch) + 1` | half a frame behind the start, the far edge covered |
| **Corner** | a corner of the area | `ceil((area − FOV) / pitch) + 1`, then the step is re-divided to `(area − FOV)/(n−1)` | lands on the far edge **exactly**, and it is the fewest tiles that can |
| **Corner (pitch)** | a corner of the area | `ceil((area − FOV/2) / pitch) + 1` | the last tile's *centre* on the far edge; the union overhangs by up to one step |

*Centre* is what the first version did, and it stays the default because
"scan around what I am looking at" is a real thing to want. *Corner* is for
covering a region: it spends no frame on the space behind the start point
and lands on the far edge to the micron. *Corner (pitch)* is for when the
area typed is the path the stage walks rather than the region the frames
cover — the requested overlap is respected exactly, and the last tile
overhangs.

An area smaller than one frame gets one tile, centred on it.

**The coverage argument was wrong once, and the fix is worth recording.**
The first implementation used `ceil(area / pitch)` tiles from a centre
origin and justified it with `(n−1)·pitch + FOV ≥ area` — true, and
irrelevant: it measures the covered *span*, and half of that span lies
behind the origin. On the bench 5×, a 2000 µm area was covered only to
1912 µm, and the 88 µm strip nobody imaged was invisible until the test
that checks the union against the rectangle was written
(`test_scan_plan.py::test_every_origin_mode_covers_the_area_it_was_given`).
Coverage is now a property of the plan, asserted for every mode, both
directions, and a range of areas and overlaps.

## Path orders

Every order visits the same set of tile centres; only the sequence differs.
`plan_cells()` produces them and `plan_path()` turns cells into stage µm, so the
preview, the map and the run are all the same arithmetic.

| Path | Order | Notes |
|---|---|---|
| **Serpentine** | rows end-to-end, alternating direction | The default: each row continues where the last ended, so the inter-tile moves are the shortest |
| **One-way** | every row from the same side, a return between rows | The same cells as the serpentine with the direction rule off |
| **Spiral** | ring by ring, outermost first | Same tile centres, inward-working order |
| **Hilbert** | the curve, clipped to the rectangle | Complete, but loses its locality on very asymmetric areas |

"Serpentine" and "Serpentine, one-way" were never two paths — they are one
order with a direction rule — so the panel asks **one** question with four
answers, not two questions. The CV layer spells One-way as
`path="serpentine"` plus `serpentine=false`, which is exactly what the old
Order row wrote: a stored configuration means the same thing after the
merge, and `one_way` exists as a path *name* so the settings file and the
map legend have something to say. Spiral and Hilbert ship as experimental:
complete by test, no bench history, and the serpentine's locality is what
keeps the inter-tile moves short.

`start_axis` chooses whether rows advance first along X or along Y, and
`x_dir`/`y_dir` mirror the whole rectangle from the start point. Both are
segmented buttons on the panel rather than drop-downs: a scan setting is
changed while looking down the eyepieces, and a popup is a look away from
the sample.

**Every order is proven complete by test, not by inspection**: for a matrix of
rectangle shapes — including the degenerate 1 × N and N × 1, where a spiral or a
clipped curve loses a cell if the implementation is careless — the tests assert
the visited cells are a *permutation* of the grid. That is the property that
matters: a missing corner in a scan is invisible until someone looks for a flake
that was never imaged.

## Motion, settling and backlash

Per waypoint, in order:

1. **Move** to the tile centre, composed of the driver's validated fixed-length
   moves through `ManagerStageAdapter` — the manager's own Zolix worker, never a
   second serial handle, so STOP ALL and the status polling still cover a run.
   The speed is `scan.speed_pps`, and **only** that: one number, set in
   Preferences → Scan, used for the run and for *go to sample* alike. It is
   not the manual jog speed and it is not scaled by the objective's
   `stage_speed_multiplier` (that one is a jog preference and still applies
   there, through `ActionResolver`). The reason a single speed is enough:
   the scan drives the controller in fixed-steps mode, so the controller
   generates its own acceleration and deceleration ramp. There is no
   stability case left for a slow/fast pair to answer.
2. **`wait_idle`** — until the *controller* reports every axis stopped, from
   its own motion bits (registers 30012-30014), read with a short `get_status`
   job every 50 ms. One poll after the ramp finishes ends it: ~0.08 s against
   the **0.6 s** the previous telemetry rule cost every waypoint.
3. **Readback** the position from the controller. The manifest records *where
   the stage was*, never where it was told to go.
4. **Settle** (`settle_ms`, default 100) — mechanical quiet after the stop.
   It begins when the motion actually ended, which is what step 2 made true:
   the old rule put ~0.6 s of waiting in between, which is why the default
   halved with the fix.
5. **Capture** (below), then hand the frame to the writer thread and go
   straight on to the next move.

**The stop phase is what the operator feels**, and these five steps are where
it went. Per tile at 1080p, shipped defaults, before and after:

| | before | after |
|---|---|---|
| motion-end detection | ≥0.60 s | ~0.08 s |
| settle | 0.20 s | 0.10 s |
| capture gate | 0.06–0.12 s | 0.06–0.12 s |
| readback | 0.02–0.05 s | 0.02–0.05 s |
| PNG + thumbnail + manifest | 0.04–0.12 s **on the scan thread** | 0 (writer thread) |
| **stopped total** | **~1.1 s** | **~0.25 s** |

Travel is untouched and remains the operator's knob: the bench's
0.625 µm/pulse means `speed_pps` 500 is only ~312 µm/s, so on a large area
the *move* is the term that is left. `ScanResult.timing` measures the three
phases separately (to command / travelling / stopped) and the panel logs
their per-tile averages at the end of a run — read that before changing any
of this, and note that the bench checklist verifies the numbers rather than
re-deriving them.

**Why the wait is polled from the caller, not run inside the driver.** The
driver has a `wait_idle` of its own and it would be marginally faster. It is
not used because it would block the device worker for the whole move, which
holds a queued STOP ALL behind it: Esc would look like it did nothing until
the stage finished travelling. Short status jobs leave the worker free
*between* reads, so the priority stop still lands promptly.

Three details of that loop that are the whole point:

- A result that is **not** a `StageStatus` counts as a FAILED poll, never as
  "stopped". Reading an answer that cannot answer the question as a yes is
  the bug the old telemetry key was written about (`stage_adapter.py`).
- Three consecutive failures raise: a blip is retried by the driver's own
  read retries, a link that stays dead must fail the waypoint rather than spin
  to the timeout. The WAIT is not retried at the scan level either — a stage
  that reports "moving" for 120 s is jammed or travelling, and the retry in
  (2) is about the move, not about the wait.
- An abort returns **quietly**, so the scan reports "aborted" — a different
  thing from "stopped early", and it keeps the abort out of the status line
  as an error.

**Backlash take-up** is optional and off by default (`backlash_um = 0`). When
enabled it makes every move *finish from the same side*: an axis travelling the
wrong way first backs off past the target and comes in. Correcting only the
reversals — the obvious implementation — is wrong for a serpentine, whose rows
alternate: it would leave odd rows a backlash away from even ones, which is the
error the feature exists to remove. The cost is one extra short move on the
waypoints where the approach is already wrong, and the tests prove the invariant
over a whole run (`test_scan_backlash.py`), not just for a single move. Each
take-up move now **waits for its own motion to end** before the target move is
issued: the driver refuses to command a move onto a moving axis (by design —
that is what stops a queued move from being executed twice), so the two-move
take-up could not have worked on hardware. Its earlier test passed only
because the stub's `wait_idle` was a no-op.

## Capture: from the frame slot, never from the camera

The scan holds **no camera object**. It reads the shared `LatestFrameSlot` —
the same mailbox the autofocus controller polls, written by the camera worker on
every fetched frame — through `LatestFrameSource` (`cv/frame_source.py`). The
reason is architectural, not stylistic: the SmartCamApi binding is exclusive and
not thread-safe, and the camera belongs to its own worker. A scan that called
`fetch()` would be reaching into a backend another thread owns, which is exactly
why the first implementation never captured a single frame.

The source **refuses a frame whose capture timestamp predates the end of the
settle window**. That gate is what makes the manifest trustworthy: every row
pairs an image with the position the stage held when the image was *taken*. A
stream that has stalled cannot satisfy the gate, so the waypoint is counted
**missing** — an empty frame column, and the count reported — rather than filing
the previous frame under a new position. The CLI benches, which own their
camera outright, use `CameraFrameSource` and the same gate.

### Capture resolution, and the switch around a run

Tiles are whatever the frame slot is publishing, and Preferences → Scan says
**which sensor mode that is**: `scan.resolution` (0 = 4K, 1 = 1080p — the
same encoding the Capture group's snapshot resolution uses), 1080p by
default, i.e. the live resolution and therefore no switch at all. When it
differs, the panel changes the camera's mode **once** before the first move
and puts it back when the run ends (finish, abort or fault — one path), and
the run does not start until the switch's job completes: capturing the first
tiles in the old mode would file 1080p frames as 4K tiles.

Two consequences worth knowing:

- **The live stream runs at the scan's resolution for the duration.** 4K is
  a deliberate quality choice, not the fast path: the frames are four times
  the pixels for the encode, the identification and the mosaic. Nothing
  about *where* the tiles are changes — the plan's pitch comes from the
  objective calibration, which is resolution-independent.
- **A refused switch costs nothing.** A backend with no live mode (the
  snapshot-only ones, or a camera that errors) fails the job, the panel logs
  it, and the run captures at the live resolution — which is *correct*
  because the identification measures each frame with its own pixels (see
  `IDENTIFICATION.md`). A camera that refuses to switch *back* leaves the
  live view at the scan's mode for that session; the next connect
  re-applies the configured live resolution.

### The frames are written by their own thread

`cv/frame_writer.py` owns `frames/` and `manifest.csv` behind a bounded
queue. The scan emits its signals as it always did (the frame and its
thumbnail, on the scan thread — one emitter, and no ordering question
against `sig_done`), then hands over `(index, readback position, frame)` and
moves on; the writer encodes the PNG, appends the row and flushes it. Three
properties the callers rely on:

- **Order and completeness.** One thread, one FIFO, and every waypoint gets
  exactly one row, including the ones with no frame and the ones with no
  position. A row names a file only once that file exists.
- **The join is a contract.** `scan_output` re-reads the manifest from disk
  and `meta.json` counts the frames, so `run()` joins the writer before
  either — and the return-to-start move happens first, so the drain overlaps
  the travel.
- **Bounded memory.** Tiles waiting for the detector are capped
  (`MAX_PENDING_TILES`): tiles are never dropped, so an uncapped queue is how
  a 4K scan runs out of memory (25 MB a frame). Past the cap the *stage*
  waits for the detector instead — slower, and bounded. A safety valve keeps
  a wedged detector from parking the run mid-plan.

A writer failure is a fault like any other: the run stops with
`stopped_early` and the reason. `ScanResult.failed` exists because a failure
on the *last* waypoint would otherwise leave `visited == planned`, which
reads as a completed scan over a truncated dataset.

## What a scan writes

`~/Pictures/TALOS/scans/scan_<timestamp>/` — beside the snapshots, where
the Capture group's folder is, rather than in a second tree under
`Documents`. The folder is chosen on the panel, with the same field +
browse + open row the Capture group uses: one editor for one path.

| File | Contents |
|---|---|
| `frames/frame_NNNNN.png` | One per captured waypoint, at the configured capture resolution |
| `manifest.csv` | One row per waypoint: frame name, **readback** x/y/r, unix time, objective id |
| `meta.json` | The request (area, overlap, path, settle, backlash…), the FOV used, the frame shape, the counts, the phase timings |
| `mosaic.png` | Optional: the tiles assembled at their readback positions |
| `mosaic_annotated.png` | Optional: the same image with every found sample ringed and numbered |
| `candidates.csv` | Optional: every detection with its stage coordinates and the tile it came from |

The manifest is the record of truth: a waypoint visited but not captured keeps
its row with an empty frame column, so the geometry of a run is always
recoverable even when the images are not.

## The mosaic, and which way is up

The mosaic and the map are drawn **in the sample's frame, as the frames show
it** — which is the whole story of `cv/orientation.py`, and it has two parts:

- The **camera flip** is a 180° rotation applied to every delivered frame. A
  frame taken with the stage at `p` therefore shows the sample point `s` at
  image offset `a·(s − p)`, so a mosaic must scale its *layout* by the same `a`,
  not merely its offsets. Placing tiles unmirrored and rotating their content
  instead — the first implementation — puts every feature on the mosaic once per
  tile, which is what a glitched mosaic looks like.
- The **mounting** decides what `a` is. Bench-measured 2026-09-17, flip in its
  default state: stage +X moves a feature right and stage +Y moves it *up*. Both
  the mosaic layout and the px→µm mapping follow it, so "go to sample" and the
  map can never disagree about which way an axis runs.

Tiles are drawn as captured, with no content rotation: they come from the same
frames the live view shows, so the flip is already in the pixels.

The **highlighted box** is the field of view, and it means exactly one of
three things, decided in one place (`ScanMapWidget._highlight`):

| State | What the box is |
|---|---|
| nothing scanned yet | **nothing**. The plan and its start dot, and no box |
| a run is capturing | the tile being captured — the newest frame in the mosaic |
| between runs, tiles on the map | the live stage position: where you are looking, relative to what was scanned |

A legend line under the box names which of the two it is; a colour would
have to be remembered. *Fit* frames the box too, so a stage parked off the
scanned area can be brought back into view.

The middle row is why the box is fed from the **tile signal** and not from
telemetry during a run: the position a caller can read mid-run is a
commanded value that lags the capture by a move, and the scan's own jobs
starve the poll anyway.

The first version had no rule at all — it drew a box from whatever telemetry
last said, whenever a scan was not running — and it was wrong three ways.
The two states that matter are the ones an empty box would lie about: with
nothing scanned, a box says "you are here" about a map that has no
"here" yet. And the box is set only from a position that is actually known:
`set_footprint(None)` means "no position" and draws nothing. That
distinction is not pedantic — the original passed the whole device payload
to `StagePosition.from_telemetry`, which reads a *flat* dict, so the
position came back as `(0, 0, 0)`, and stage (0, 0) is a real place: the box
sat at the origin on every idle frame, unrelated to the plan. A missing
position and a position at the origin must not look the same.

## Accuracy, and why the mosaic is not a measurement

**Nothing measures from the mosaic.** Identification runs on each frame
individually at full resolution (`IDENTIFICATION.md`), and the mosaic exists so
an operator can see where a run went. That sets a low accuracy bar: a small gap
or an imperfect seam is acceptable, so the builder stays simple — placement by
readback position, overlaps averaged (a stamp would leave a seam), and no
registration, no seam blending and no rotation correction. The px→µm mapping is
orthotropic for the same reason; the 2×2 jacobian is kept in reserve and would
come back into play only if a merged image ever had to be metrologically useful.

**`mosaic_annotated.png` is that same image with the samples on it** — one ring
per candidate, drawn as the circle of equal area, with the number the sample
list gives it. It exists because the alternative, a sheet of per-tile
thumbnails with boxes drawn on them, answered "was there a sample in this
tile?" and never "where on the wafer is sample 7?" — it had no positions on it
at all. The rings come from the mosaic's own geometry (`mosaic_geometry`, the
same call `build_mosaic` uses) and the candidates' stage µm, so a ring sits on
the pixels of the sample it names rather than near them.

## What the map keeps, and for how long

A finished run's **mosaic and sample markers stay on the map** until the next
run starts or the operator clears the results. After a scan the map is what a
sample is read off — its position, its neighbours, which way the wafer ran —
and losing that the moment the stage stopped made the map useless for exactly
the job it is best at.

Two mechanisms are behind that, and the second one was a bug:

- `ScanMapWidget.set_plan` no longer drops the tiles. The tiles are the record
  of the run that happened; the plan is a preview of one that might, so
  editing the area after a scan does not (and should not) erase what was
  captured.
- The plan's **origin is latched to the run** while its results are on the
  map. The plan is normally anchored to the stage — "what would a scan from
  here cover?" — and a run ends with the stage back at the start or at the
  last tile, so re-anchoring it changed the plan's identity and took the
  mosaic with it. A single pulse (0.625 µm) of readback noise in where the
  stage came back to was enough. The latch is released by clearing the
  results, and re-taken by the next run.

## Aborting, and STOP ALL

Abort is cooperative. The flag is checked at the top of the waypoint loop and
inside the stage adapter's blocking call, so a run stops between moves and
unwinds normally — `QThread.terminate()` is banned here for a reason that is
recorded in the code: it once killed the scan thread mid-serial-write, leaving
the port open and the controller possibly mid-move.

**STOP ALL aborts the run, not just the motion in flight.** Esc and LB+RB route
through the input layer to `manager.stop_all()`, whose completion signal
(`sig_stop_all_done`) reaches the scan's abort path. Without that wiring the
stage stopped but the run continued at the next waypoint, which made Esc look
like it had done nothing. The reverse case is handled too: the scan's own start
sequence submits a *Zolix-only* stop rather than `manager.stop_all()`, so it
cannot abort itself through that same signal.

**The abort does not stop the stage, and that is the point.** Three things
arrive at the same abort: the panel's Abort button (which does stop
everything), Esc / LB+RB through the input layer (already stopped by the time
the signal lands), and the map window's Esc. The abort path itself only sets
the flag, asks the **adapter** for one priority stop of the axis the run owns,
and drops the input holds.

The first version called `manager.stop_all()` from all three. For the
completion-driven ones that is a cycle: `stop_all` completes by emitting
`sig_stop_all_done`, whose handler called the abort, which stopped everything
again — which emitted the signal again. On the bench it read as a scan that
"aborted halfway and then looped forever", with a trio of
`job #N: zolix.stop()` / `STOP ALL requested` / `All stages stopped` repeating
several times a second for as long as the run took to unwind, each round
putting another stop job on a serial link that is this bench's weak point. **A
completion handler must not re-issue the action that produces it.**

Two properties the unwinding has, both of which were missing:

- **An abort is not waited out.** `ManagerStageAdapter._call` watches the abort
  flag in the same loop that watches its deadline, so an abort arriving while a
  move is in flight ends the run in a poll or two, not at the job's timeout
  (120 s for a move) with the stage already stopped and the operator watching.
- **An abort always reaches the scanner**, including one that lands while the
  scan worker is still starting up (the adapter's abort check is the same
  Event, so the run was already refusing to move — but it used to *end* as
  "stopped early", i.e. the operator's own abort reported as a fault).

Two known gaps here, both older than the retry and neither fixed by it:

- **A queued move survives a stop.** `device_proxy._MOTION` does not list
  `move_abs_um`, so `enqueue_stop` purges *continuous* motion and not a queued
  absolute move — a move submitted just before the stop still executes. It is
  harmless for the dataset (the run ends as an abort, no tile is filed) but it
  means "the stage stops instantly" is not literally true. Fixing it needs
  `_purge_continuous`'s "completed with None" signal to be told apart from a
  landed move, which is why it is a known gap rather than a patch.
- **The abort flag is checked between waypoints and between attempts**, so it
  cannot interrupt a job already on the wire; the adapter's own check covers
  the wait for that job.

## Three ways a run can end, and why they must not be confused

A scan ends because the operator aborted it, because it finished, or because
something **failed** part-way. The third one used to be reported as the second,
which is the worst possible confusion: on 2026-09-18 a single truncated Modbus
reply — two bytes and then silence, from a perfectly healthy controller, during
one `get_position()` readback — stopped a run at tile 3 of 9 and the panel said
**"Scan done"**. A third of a dataset looked complete, and the reason was
carried in a field nothing displayed.

Four fixes, at four layers:

1. **A garbled frame is retried — three times, at the read level.** A
   *truncated* reply never reaches the length its function code promises
   (visible in `_transact`), while a *CRC mismatch* arrives at the right
   length with corrupted bytes (visible only once the parser validates it).
   Both are retried in `_read_registers`, the layer that knows the request and
   the answer together. Note the asymmetry: a **silent** line is not retried
   there — `_transact` raises `DeviceTimeoutError` before the retry loop is
   entered, so a line that says nothing at all fails on the first attempt.
   That is the fault the approach retry (below) exists for. Writes do **not**
   retry: a truncated write response means we cannot know whether the write
   landed, and re-sending a motion command on a guess is how the controller
   gets the same move twice. A device's *answer* is not retried either —
   `LimitHitError`, `DeviceBusyError` and `EStopError` are the controller
   replying, and asking again does not change a limit switch.
2. **The approach is retried, when the fault is one a re-issue can clear.**
   The scan re-issues the take-up and the target move (up to
   `_MAX_APPROACH_ATTEMPTS = 3`) on a `DeviceTimeoutError`, a
   `DeviceBusyError` or a `ProtocolError` — the link and the timing — because
   an ABSOLUTE move recomposed from a fresh readback is safe to repeat. It
   never retries a limit switch, an e-stop, a dead port or a refused command,
   and it does not retry a stage that will not settle: that one would be
   re-issued *onto a moving axis*, which the driver composes as
   `target + (target − where it was)` — the manifest stays honest (it records
   the readback) but the tile grid gains a gap and a doubled tile and nothing
   on screen says so. Between attempts the scan waits for the previous motion
   to end (`_RETRY_SETTLE_S`) and **abandons the retry if it cannot**.
   `ScanTiming.retries` counts them, `meta.json` records the count, and the
   summary line gains "N retried approach(es)" so a run that fought the link
   is not read as a slow one. Nothing else — the readback, the capture, the
   tile bookkeeping — is retried: those have their own tolerance (one lost
   tile) and re-entering them would spend the settle window twice.
3. **A malformed frame leaves the driver as a `ProtocolError`, and a failed
   job arrives at the scan as the class the driver raised.**
   `ManagerStageAdapter` rebuilds the exception from the `(job_id, name,
   message)` triple the proxy reports, because that is the only thing the
   retry decision in (2) can be made on. A name it does not know — a driver
   bug, an AttributeError — falls back to `DeviceError`, which is never
   retried.
4. **The result knows whether it finished.** `ScanResult` carries `planned` and
   `visited`; `aborted` means the operator asked for it, `stopped_early` means
   something else did, and the panel reports each in its own words, in its own
   colour, with the reason. The export summary *appends* its file list to that
   verdict rather than replacing it — it was the message actually on screen.

A run that finishes while capturing nothing is called out too ("is the camera
streaming?"): the geometry is recorded and the scan did complete, but "done"
alone would send the operator looking for images that were never taken.

**A waypoint that survives the retries costs a tile, not the run.** If the
position readback fails after all three attempts, there is no position to file
a frame under — so the manifest row is written with an **empty position and an
empty frame**, the waypoint is counted in `missing`, and the scan walks on.
That is the same honesty as the missing-frame case: the manifest may not
invent a position, but it may say it does not have one, and the rest of the
dataset is still worth having. Three losses *in a row* is not a blip — it is a
dead link, and grinding through the remaining tiles at two seconds each would
produce nothing — so the run stops there and says so.

While a scan runs, `AppState.mode` is `SCAN` — the single gate every manual
input source funnels through — so jogging is refused, with the mode badge in
the status bar saying why. STOP ALL always works, on the panel and in the
map's own window alike: Esc does what it does everywhere else, so there is
no window in this application where the panic key has been quietly
repurposed.

## What the operator changes, and what they set once

The panel carries the settings that change *between* runs: the area, the
origin, the directions, the path (one selector, four walks), the start axis,
whether to come back, and the output folder. Overlap, settle time, scan
speed, backlash, the capture resolution and which extra files a run writes
are in **Preferences → Scan** — set once and then not thought about again.

Which key belongs to which is pinned by a test
(`test_preferences.py::test_the_scan_page_owns_what_the_panel_does_not`):
the two sets must be disjoint and together cover the section. A key in
neither is a value nothing can change; a key in both is somewhere for the
two editors to disagree. Writing that test caught `scan.dir` on both sides.

## Bench status

- **Direction: verified** (2026-09-17) — the mounting above is the measurement.
- **Pitch: implied** by that run's mosaic being continuous; a mis-scaled FOV
  would have shown up as visible gaps or double-coverage rather than a
  continuous image.
- **Scale: verified separately** against a calibration glass slide.
- **The origin modes have no bench history yet.** They are pure geometry with
  the coverage proof asserted in `test_scan_plan.py`, and the properties that
  matter at the bench — the tile count and the union landing on the far edge —
  are visible on the map before the run starts.
- **The stop-phase rewrite has no bench history yet** (2026-09-22). The
  motion bits it now relies on are the controller's own and already drive the
  hardware strip, and the sim test pins the wall-clock cost; what is *not*
  measured is a real run's per-tile stop time and whether 100 ms of settle is
  enough on this frame. `tools/scan_manager_bench.py` prints the phase
  timings for exactly that, and the checklist item is to run it twice — once
  at the shipped settings, once with `settle_ms` raised — and compare.
- **A 4K scan has never run on hardware.** It is supported (the switch, the
  per-frame calibration, the resolution-invariant gates are all tested), but
  nothing has captured a 4K tile on this bench.
- **Still owed**: Esc mid-scan end-to-end, a deliberately stalled camera (the
  missing-frame path), and a first identification run on a real wafer. The full
  checklist is in `docs/journal/BENCH_TODO.md`.
