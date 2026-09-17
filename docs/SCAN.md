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
| **Serpentine** | rows end-to-end, alternating direction | The default. *Order: One-way* runs every row the same way |
| **Spiral** | ring by ring, outermost first | Same tile centres, inward-working order |
| **Hilbert** | the curve, clipped to the rectangle | Complete, but loses its locality on very asymmetric areas |

"Serpentine" and "Serpentine, one-way" were never two paths — they are one
order with a direction rule — so the panel asks the two questions
separately: **Path** (Serpentine / Spiral / Hilbert) and **Order**
(Serpentine / One-way). Spiral and Hilbert ship as experimental: complete by
test, no bench history, and the serpentine's locality is what keeps the
inter-tile moves short.

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
2. **`wait_idle`** — position stable across three consecutive telemetry samples
   at 0.2 s, i.e. at least 0.6 s after the motion stops. This is the single
   largest *fixed* cost per tile: with the settle and the capture it comes to
   about 0.9 s before any motion is counted, which is what the progress readout's
   measured ETA is built from.
3. **Readback** the position from the controller. The manifest records *where
   the stage was*, never where it was told to go.
4. **Settle** (`settle_ms`, default 200) — mechanical quiet after the stop.
5. **Capture** (below).

**Backlash take-up** is optional and off by default (`backlash_um = 0`). When
enabled it makes every move *finish from the same side*: an axis travelling the
wrong way first backs off past the target and comes in. Correcting only the
reversals — the obvious implementation — is wrong for a serpentine, whose rows
alternate: it would leave odd rows a backlash away from even ones, which is the
error the feature exists to remove. The cost is one extra short move on the
waypoints where the approach is already wrong, and the tests prove the invariant
over a whole run (`test_scan_backlash.py`), not just for a single move.

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

## What a scan writes

`~/Pictures/TALOS/scans/scan_<timestamp>/` — beside the snapshots, where
the Capture group's folder is, rather than in a second tree under
`Documents`. The folder is chosen on the panel, with the same field +
browse + open row the Capture group uses: one editor for one path.

| File | Contents |
|---|---|
| `frames/frame_NNNNN.png` | One per captured waypoint, 1080p |
| `manifest.csv` | One row per waypoint: frame name, **readback** x/y/r, unix time, objective id |
| `meta.json` | The request (area, overlap, path, settle, backlash…), the FOV used, the frame shape, the counts |
| `mosaic.png` | Optional: the tiles assembled at their readback positions |
| `overview.png` | Optional: a thumbnail per tile with the detections drawn on it |
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

The **footprint** — the outlined box with a faint fill — is the field of
view at the current stage position, and it is set only from a position that
is actually known: `set_footprint(None)` means "no telemetry yet" and draws
nothing. That distinction is not pedantic. The first version passed the
whole device payload to `StagePosition.from_telemetry`, which reads a *flat*
dict, so the position came back as `(0, 0, 0)` — and stage (0, 0) is a real
place, so the box sat at origin on every idle frame, unrelated to the plan.
A missing position and a position at the origin must not look the same.

## Accuracy, and why the mosaic is not a measurement

**Nothing measures from the mosaic.** Identification runs on each frame
individually at full resolution (`IDENTIFICATION.md`), and the mosaic exists so
an operator can see where a run went. That sets a low accuracy bar: a small gap
or an imperfect seam is acceptable, so the builder stays simple — placement by
readback position, overlaps averaged (a stamp would leave a seam), and no
registration, no seam blending and no rotation correction. The px→µm mapping is
orthotropic for the same reason; the 2×2 jacobian is kept in reserve and would
come back into play only if a merged image ever had to be metrologically useful.

## Aborting, and STOP ALL

Abort is cooperative. The flag is checked at the top of the waypoint loop and
inside the stage adapter's blocking call, so a run stops between moves and
unwinds normally — `QThread.terminate()` is banned here for a reason that is
recorded in the code: it once killed the scan thread mid-serial-write, leaving
the port open and the controller possibly mid-move.

**STOP ALL aborts the run, not just the motion in flight.** Esc and LB+RB route
through the input layer to `manager.stop_all()`, whose completion signal
(`sig_stop_all_done`) reaches the same abort path as the Abort button. Without
that wiring the stage stopped but the run continued at the next waypoint, which
made Esc look like it had done nothing. The reverse case is handled too: the
scan's own start sequence submits a *Zolix-only* stop rather than
`manager.stop_all()`, so it cannot abort itself through that same signal.

While a scan runs, `AppState.mode` is `SCAN` — the single gate every manual
input source funnels through — so jogging is refused, with the mode badge in
the status bar saying why. STOP ALL always works, on the panel and in the
map's own window alike: Esc does what it does everywhere else, so there is
no window in this application where the panic key has been quietly
repurposed.

## What the operator changes, and what they set once

The panel carries the settings that change *between* runs: the area, the
origin, the directions, the path order, the start axis, whether to come
back, and the output folder. Overlap, settle time, scan speed, backlash and
which extra files a run writes are in **Preferences → Scan** — set once and
then not thought about again.

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
- **Still owed**: Esc mid-scan end-to-end, a deliberately stalled camera (the
  missing-frame path), and a first identification run on a real wafer. The full
  checklist is in `docs/journal/BENCH_TODO.md`.
