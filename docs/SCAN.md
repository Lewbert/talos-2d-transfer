# The grid scan — covering an area without touching the focus

Companion documents: [IDENTIFICATION.md](IDENTIFICATION.md) for what happens to
the frames afterwards, [ARCHITECTURE.md](ARCHITECTURE.md) for the job model and
the threading, [DESIGN.md](DESIGN.md) for the bench and the design decisions.
The console that drives it is `Windows ▸ Scan` (`ui/widgets/scan_window.py`);
the map is `ui/widgets/scan_map.py`.

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
0.3557 µm/px, giving 1366 × 768 µm. The window prints the source beside the
number (`measured in TALOS` / `imported from Labscope` / `estimated from the
sensor pitch`) so an estimate is never mistaken for a measurement, and the
manual FOV fields are a fallback for an uncalibrated bench, not the normal path.

The grid is `ceil(area / pitch)` tiles per axis, anchored at the **first
waypoint** — the operator jogs to a feature they can see and presses *Scan from
here*, so the origin is wherever the stage is at that moment. Coverage is
guaranteed rather than exact: `(n−1)·pitch + FOV ≥ area` holds for any
`n = ceil(area/pitch)` because `FOV > pitch`, so the last column or row simply
overhangs the requested rectangle. The overhang is what makes the run robust to
a slightly optimistic area; the alternative (stretching the pitch to land
exactly on the far corner) would silently lose coverage whenever the calibration
is a little off.

## Path orders

Every order visits the same set of tile centres; only the sequence differs.
`plan_cells()` produces them and `plan_path()` turns cells into stage µm, so the
preview, the map and the run are all the same arithmetic.

| Path | Order | Notes |
|---|---|---|
| **Serpentine** | rows end-to-end, alternating direction | The default. `serpentine off` runs every row the same way |
| **Spiral** | ring by ring, outermost first | Same tile centres, inward-working order |
| **Hilbert** | the curve, clipped to the rectangle | Complete, but loses its locality on very asymmetric areas |

`start_axis` chooses whether rows advance first along X or along Y, and
`x_dir`/`y_dir` mirror the whole rectangle from the start point.

**Every order is proven complete by test, not by inspection**: for a matrix of
rectangle shapes — including the degenerate 1 × N and N × 1, where a spiral or a
clipped curve loses a cell if the implementation is careless — the tests assert
the visited cells are a *permutation* of the grid. That is the property that
matters: a missing corner in a scan is invisible until someone looks for a flake
that was never imaged.

Spiral and Hilbert are labelled experimental in the UI: they are complete by
test, but they have no bench history and the serpentine's locality (each tile
adjacent to the last) is what keeps the inter-tile moves short.

## Motion, settling and backlash

Per waypoint, in order:

1. **Move** to the tile centre, composed of the driver's validated fixed-length
   moves through `ManagerStageAdapter` — the manager's own Zolix worker, never a
   second serial handle, so STOP ALL and the status polling still cover a run.
   Speed is the configured pps scaled by the active objective's
   `stage_speed_multiplier` (clamped to [0.05, 1.0], floor 10 pps).
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

`~/Documents/TALOS_scans/scan_<timestamp>/`:

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
input source funnels through — so jogging is refused, with the mode badge in the
status bar saying why. STOP ALL always works.

## Bench status

- **Direction: verified** (2026-09-17) — the mounting above is the measurement.
- **Pitch: implied** by that run's mosaic being continuous; a mis-scaled FOV
  would have shown up as visible gaps or double-coverage rather than a
  continuous image.
- **Scale: verified separately** against a calibration glass slide.
- **Still owed**: Esc mid-scan end-to-end, a deliberately stalled camera (the
  missing-frame path), and a first identification run on a real wafer. The full
  checklist is in handoff #32 of the development journal.
