# Sample identification — a filter chain, judged by eye

Companion documents: [SCAN.md](SCAN.md) for how the frames are captured,
[ARCHITECTURE.md](ARCHITECTURE.md) for the threading and the job model,
[DESIGN.md](DESIGN.md) for why the bench is built the way it is. The code is
`cv/identify.py` (the chain) with shared helpers in `cv/flakes.py`; the panel
and the processed view are in `ui/widgets/scan_window.py` and
`ui/widgets/live_view.py`.

## What it is for, and what it refuses to do

The operator points at a colour — a dropper click on the live view, or a typed
hex — and the chain finds regions of the frame that match it, filtering out the
things that are not samples: speckle, the frame edge, diffuse smudges, the
application's own red scale bar. It answers *"where are the flakes that look like
this?"*, which is the question a transfer workflow actually starts from.

It does **not** judge thickness, rank materials, or decide whether a flake is
worth transferring. Colour is not thickness — interference colours overlap
across layer counts under varying illumination — and a number that pretends
otherwise would be worse than no number. The result is a list of places to look,
with a marker on the map and a *go to* that centres one under the crosshair.

It also always runs on **one frame**. A scan's tiles are identified
individually, at full resolution; the mosaic is never an input. Merging tens or
hundreds of tiles is too expensive in compute and memory for what it would buy,
and the per-frame result is what *go to* needs. A consequence worth stating:
the mosaic may have a small gap or an imperfect seam without affecting anything
that matters (see [SCAN.md](SCAN.md#accuracy-and-why-the-mosaic-is-not-a-measurement)).

## The chain

Eight stages, in a fixed order. Each has `enabled` plus its own parameters,
reports how many candidates it let through, and is described to the UI by the
stage itself (`LABEL`, `RANGES`), so a new stage needs no UI code.

| # | Stage | Kind | Default | What it does |
|---|---|---|---|---|
| 1 | **Colour match** | source | **on** | Pixels within a tolerance of the picked colour |
| 2 | **Contrast** | source | off | Illumination-flattened contrast, Otsu-thresholded |
| 3 | **Clean up** | mask | **on** | Open then close: removes speckle, joins broken edges |
| 4 | **Size** | gate | **on** | Area in µm², min and max |
| 5 | **Frame edge** | gate | **on** | Rejects blobs touching the field of view's edge |
| 6 | **Sharpness** | gate | **on** | Mean boundary gradient: crystals are sharp, smudges are diffuse |
| 7 | **Scale bar** | gate | **on** | Rejects the app's own saturated red annotation |
| 8 | **Merge fragments** | merge | **on** | Joins boxes within a gap — one sample often segments as several blobs |

The two sources are OR'd when both are enabled, so "colour **or** contrast" is a
supported combination rather than a mode to choose between. With no source
enabled the chain finds nothing, by construction.

Each stage reports `(in, out)`, and the panel renders the chain as the readout
it is: `Colour match 812 → Size 12 → Sharpness 2`. That string is the fastest
way to answer "why is my flake missing" — the stage whose count collapses is the
one to change.

## Three rules the implementation exists to keep

**Hue wraps.** The colour band is split across the 0/179 seam, so a red target
(hue ≈ 0 or ≈ 179) gets the same tolerance as any other hue. The obvious
implementation — a `low ≤ h ≤ high` box — silently leaves red one-sided, which
is the bug the reference project shipped with and this one has a test against.

**Saturation has its own floor.** An unsaturated pixel's hue is meaningless, so
a wide band around a red target would otherwise match every grey in the frame.
`min_saturation` (default 40) is separate from the tolerance slider for exactly
that reason; a grey or white target needs it lowered deliberately, and the UI
gives it a row of its own.

**The preview is honest.** The live view processes a *downscaled* copy of the
frame for speed (50 % by default), but every result comes back in the caller's
pixels and µm, and the pixel-unit parameters — the edge margin, the merge gap,
the sharpness floor — are scaled with it. A knob turned on the preview means the
same thing on a full-resolution tile, which is asserted by test: the same blob
must yield the same µm at 1.0 and 0.5 scale.

The pipeline also never sees a half-edited configuration: the panel rebuilds the
whole config per job and hands it over, so the worker never reads a widget.

## Configuration

The chain serialises to `identify.stages` in the settings file — the bundled
defaults *are* the schema — and `IdentifyConfig.from_dict` is deliberately
forgiving: an unknown stage is dropped, a missing one returns at its default, a
parameter with the wrong type falls back to its default, and a malformed hex
falls back to a colour that works. A hand-edited settings file must not be able
to stop the pipeline from running.

## The processed view

**Processed** in the live view's two-button switch shows what the chain makes of
the frame: everything the sources did *not* match is darkened, and every region
they did match is outlined by its verdict — **bright** for what survived the
whole chain, **dim** for what a gate rejected. The operator sees not only what
was found but what was thrown away and where, which is what makes tuning a
parameter feel like tuning a filter chain rather than guessing.

Deliberately not rectangles: a box round a shapeless blob says little, boxes
from neighbouring regions overlap and read as one object, and a box hides the
shape the operator is judging. The darkening keeps the sample's own pixels
visible inside the match, so the view reads as "this part of the wafer" rather
than as a mask poster.

The **dropper always samples the original frame**, never what is on screen: in
processed mode the display is darkened and outlined, so picking from it would
return a colour the sample does not have.

The processed view is a **display choice**: the camera stream, the autofocus and
a running scan never wait on it. The identification runs on its own thread, fed
by the frames that were delivered anyway.

## Performance and scheduling

- **Live preview**: one job in flight, newest wins — a frame that arrives while
  the previous one is still being processed is *dropped*, because a preview that
  lags the stream is worse than one that skips frames. It runs only while the
  console is on screen or the tab is showing Processed.
- **Scan tiles**: queued unconditionally. A tile that is not examined is a
  sample that was not found, so nothing is dropped; the queue drains while the
  scan runs and after it finishes, and the exports wait for it (detection
  deliberately outlives the capture).
- A failure inside the pipeline is logged and the next job runs: it can never
  stop a scan.

## Tuning on the bench

Start with colour alone, then add the chain one stage at a time and watch the
counts — each stage's `(in, out)` says whether it is doing anything useful.

1. **Tolerance** until the flake family is caught without the substrate.
2. **Min saturation** if the substrate (or the illumination gradient) comes in
   with it — this is the parameter that does the most work on a real wafer.
3. **Clean up**: raise the kernel if a flake fragments into speckle; it also
   merges broken edges, so watch that it does not eat small samples.
4. **Size**: the µm² floor is the honest filter, and it is in physical units, so
   it means the same thing at every objective.
5. **Sharpness**: raise it to reject defocused blobs. Its default (4.0) is
   deliberately low — a colour-matched contour traces the colour boundary and
   scores in the tens, while a contrast-matched one wanders through the noise
   around the object and scores in single digits, so a threshold tuned on one
   source will silently reject everything from the other.
6. **Merge**: raise the gap when one flake arrives as several boxes.

The unit tests build synthetic frames with known truth (a blob of a known colour
at a known place, and a substrate that is none of those), which is the fastest
place to check a change of behaviour; `tools/validate_cv.py` renders annotated
PNGs for real frames.

## Bench status

The thresholds have only ever met synthetic blobs. The chain is exercised
end-to-end in simulation — a wafer that moves with the stage, per-tile
identification, the exports — but the first real-wafer tuning session is still
owed, and it is the next item after the scan's own bench campaign
(handoff #32 in the development journal).
