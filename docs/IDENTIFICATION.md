# Sample identification — a filter chain, judged by eye

Companion documents: [SCAN.md](SCAN.md) for how the frames are captured,
[ARCHITECTURE.md](ARCHITECTURE.md) for the threading and the job model,
[DESIGN.md](DESIGN.md) for why the bench is built the way it is. The code is
`cv/identify.py` (the chain) with shared helpers in `cv/flakes.py`; the
pre-processing chain that runs in front of it is `cv/preprocess.py`, and the
panels and the three live-view modes are in `ui/widgets/identify_panel.py` and
`ui/widgets/live_view.py`.

## What it is for, and what it refuses to do

The operator points at a colour — a dropper click on the live view, or a typed
hex — and the chain finds regions of the frame that match it, filtering out the
things that are not samples: speckle, the frame edge, diffuse smudges. It
answers *"where are the flakes that look like this?"*, which is the question a
transfer workflow actually starts from.

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

**The frames it runs on are prepared, never composited.** The input is the raw
camera frame with the operator's pre-processing chain applied — no scale bar,
no crosshair, no annotation of any kind. There used to be a stage here whose
job was to reject the application's own red scale bar; tracing the burn-in
showed it only ever touched snapshot copies inside the camera backend, so
there was nothing on this path to reject. The rule that stage stood for is
kept in its place, and it is now structural: the pipeline is fed from the
frame slot and from nothing that draws.

## The chain

Six stages, in a fixed order. Each has `enabled` plus its own parameters,
reports how many candidates it let through, and is described to the UI by the
stage itself (`LABEL`, `RANGES`, `CHOICES`), so a new stage — or a new *named
choice* on an existing one — needs no UI code.

| # | Stage | Kind | Default | What it does |
|---|---|---|---|---|
| 1 | **Colour match** | source | **on** | Pixels within a tolerance of the picked colour — one of three methods, below |
| 2 | **Clean up** | mask | **on** | Open then close: removes speckle, joins broken edges |
| 3 | **Size** | gate | **on** | Area in µm², min and max |
| 4 | **Frame edge** | gate | **on** | Rejects blobs touching the field of view's edge |
| 5 | **Sharpness** | gate | **on** | Mean boundary gradient: crystals are sharp, smudges are diffuse |
| 6 | **Merge fragments** | merge | **on** | Joins boxes within a gap — one sample often segments as several blobs |

With the source disabled the chain finds nothing, by construction.

Each stage reports `(in, out)`, and the panel renders the chain as the readout
it is: `Colour match 812 → Size 12 → Sharpness 2`. That string is the fastest
way to answer "why is my flake missing" — the stage whose count collapses is the
one to change.

### Two stages that used to be here, and why they are not

**Contrast** was a second source: Otsu on an illumination-flattened frame. It
answers *what is here at all* — dust, residue, the wafer edge — which is not
the question a transfer workflow asks. On the thin samples this is built for
it is worse than unhelpful: the threshold follows the bulk of the histogram
rather than the object of interest, so one thick flake in the same frame can
push a monolayer under it, and a monolayer's deviation from the substrate can
sit at the flattening's own residual, where Otsu is segmenting texture rather
than material. Its useful half — the flattening — is now a **pre-processing**
stage, where it prepares the frame instead of segmenting it.

**Scale bar** rejected saturated red blobs as the app's own annotation. The
burn-in only ever touches snapshot copies inside the camera backend, so the
pipeline never saw one; the stage was defending against something that could
not arrive, which is a defence that quietly stops being checked. The
prevention is structural now (see above). Both names are recorded in
`_RETIRED_STAGES` and a stored config that mentions them loads without them.

## Three ways to match, and which to use

The colour match is one of three geometries, chosen in the colour card and
stored with the colour. `Tolerance` keeps the **same per-axis meaning** in all
three (hue ×0.9, saturation and value ×2.55) so switching method does not
re-scale the number — only the shape of what it accepts:

| Method | Accepts | Shape |
|---|---|---|
| **Window** (default) | `dH ≤ 0.9·T` and `dS ≤ 2.55·Spread` and `dV ≤ 2.55·Spread` | a **box** in HSV |
| **HSV dist.** | `√((dH/0.9)² + (dS/2.55)² + (dV/2.55)²) ≤ T` | a **ball** of the same weighting |
| **RGB dist.** | `√(ΔR² + ΔG² + ΔB²) ≤ 2.55·T` | a ball in RGB |

`dH` is circular in every method (H is 0–179: 179 and 2 are three degrees
apart), and `min_saturation`/`min_value` apply in all three as a *pre-filter* on
the pixel, clamped to the pick so the picked colour is always inside its own
mask.

**The shapes are not interchangeable, and that is the point of having them.**
With `Spread == Tolerance` the HSV ball is inscribed in the box: it gives up the
box's corners — a pixel that is at the far edge of hue *and* of shade at once —
and that diagonal is exactly what an illumination gradient looks like, which is
why the ball is the one to try when the substrate keeps creeping in. RGB has no
hue/brightness decomposition at all: it is the sharpest tool for "this shade on
this wafer under this lamp", the first to fail when the exposure or the colour
temperature moves, and — because a low-saturation pixel has a meaningless hue in
HSV but is simply *close* to another low-saturation pixel in RGB — the one to
reach for on a **grey or near-white** background.

**Two switches that change more than the shape**, both of which the panel says
out loud when they happen:

- **Spread is superseded by a distance method.** The window's shade slack is
  `spread × 2.55`; a ball's is `tolerance × 2.55`. So switching to a distance
  method with a deliberately narrowed spread *re-opens* the shade window — for
  the two-layer case the split exists for, the lighter layer comes back. The
  `Spread` row is disabled with a tooltip saying so, and the card's own line
  warns on the switch.
- **The RGB ball is not the box.** It is stricter on a brightness change (three
  channels at once) and looser across hue at low saturation. The same number is
  the same *slack per axis*, not the same region.

## Three rules the implementation exists to keep

**Hue wraps.** The colour band is split across the 0/179 seam, so a red target
(hue ≈ 0 or ≈ 179) gets the same tolerance as any other hue. The obvious
implementation — a `low ≤ h ≤ high` box — silently leaves red one-sided, which
is the bug the reference project shipped with and this one has a test against.
The distance methods need the *wrapped difference* rather than a split band, and
it is computed in int32 because uint8 subtraction wraps silently — turning every
red target inside out.

**Hue and shade are two numbers, and they must be.** The match is an HSV box:
`Hue tolerance` sets the hue half-width (× 0.9), `Shade spread` sets the
saturation **and** value half-widths (× 2.55). They were one number until
2026-09-23, and on a blue-on-blue sample that was a trap worth documenting:

| | H | S | V |
|---|---|---|---|
| monolayer, `#1c3484` | 113 | 201 | 132 |
| the layer above it, `#aacdf5` | 106 | 78 | 245 |

Seven degrees of hue apart, and ~120 units of saturation and value. **The hue
band is irrelevant to that pair** — the shade window is the only thing that
separates them, so widening the single old slider to tolerate illumination also
let the other layer in (from ≈ 49 on). Two sliders mean hue can be opened
without opening the shade, which is the whole point of having them.

**Two floors, and the pick cannot be outside its own window.** An unsaturated
pixel's hue is meaningless, so a wide band around a red target would otherwise
match every grey in the frame: that is `min_saturation` (default 40), with
`min_value` (default 0) as the brightness equivalent. Both are *clamped to the
picked colour*, because a mask the pick is outside of reads as a broken
detector and is silent — which is exactly what a raised `min_saturation` used
to do to the very flake it was raised from. The consequence is deliberate and
is reported in the colour card's own line: for a pick paler than the floor, the
floor no longer applies to that pick.

**The preview is honest, and so is the resolution.** A pixel-unit parameter —
the edge margin, the merge gap, the sharpness floor — means pixels of the frame
the operator was looking at when they set it, and both things that can change
that frame are compensated:

- the **preview downscale** (50 % by default): the pipeline runs on a smaller
  copy, and every result comes back in the caller's pixels and µm;
- the **frame's own resolution**: a scan can capture at 1080p or at 4K
  (Preferences → Scan), and the same field of view sampled with twice the
  pixels per axis is twice the pixels for a margin and *half* the per-pixel
  gradient for an edge. The caller that knows both frames says so
  (`IdentifyPipeline.run(frame_scale=...)`), and `_Ctx.ref_px`/`ref_grad`
  scale the lengths and the gradient floors accordingly.

Both are asserted by test: the same blob yields the same µm at 1.0 and 0.5
scale, and the same scene at 1080p and 4K gets the same verdict from every
gate. Deliberately *not* scaled: `_MIN_AREA_PX2` and the morphology kernel.
Sensor speckle is a per-pixel phenomenon rather than a physical size, and the
physical size gate is `SizeStage`, which is µm² and resolution-independent
already.

`frame_scale` is **never inferred from the frame width**. A 640×480 camera and
a 4K one are not the same field of view, and guessing would move the gates of
every setup that is not the one on this bench. It defaults to 1.0 — which is
every 1080p path, i.e. the arithmetic every earlier version had.

**The measurement is per frame too.** `SampleFindingWorkspace.calibration_for(frame)`
scales the objective calibration (µm per 4K-sensor pixel) by *that frame's*
width, for the live preview and for each tile alike. Using the live frame's
scale for a tile captured at another resolution gets every µm² wrong by the
square of the ratio and every go-to-sample offset wrong in proportion — a
4× error in the size gate that nothing in a single-resolution build can show
you.

The pipeline also never sees a half-edited configuration: the panel rebuilds the
whole config per job and hands it over, so the worker never reads a widget.

## Pre-processing — the layer in front of the chain

Everything here runs on the detection worker, once per job, before the chain
sees the frame. Two stages, in a fixed order that the panel states:
**denoise → curve**. It is not arbitrary: the local-contrast curve multiplies
small differences around the picked colour, *including* the differences that
sensor noise put there, so the noise is removed first.

| Stage | What it does | Default |
|---|---|---|
| **Denoise** | Edge-preserving (bilateral) smoothing, for the noise the curve would otherwise amplify. It must not soften flake edges — they are what is being identified. | off |
| **Local contrast** | The curve below. | off |

**Pre-processing is a pure transform, and it is never written to disk.**
Snapshots, scan tiles and the mosaic are all raw captures; the chain exists so
that the operator can see — and the identification can segment — the same
prepared frame, and for nothing else.

### The three stages that were removed, and why

Recorded because "why is there no brightness control?" is a question somebody
will ask with a screenshot in hand (removed 2026-09-23).

- **Tone operations** — exposure, brightness, contrast, gamma. The camera's own
  exposure, gain and white balance already set the frame, and a second set of
  the same controls behind the operator's back made every bench session a
  question about *which layer* was being tuned: the live view is the
  pre-processed one, so a brightness change looked exactly like a change to the
  camera. With the identification chain also having lost its Contrast stage,
  they were the last thing standing between the sample and the operator.
- **Shade correction** (illumination flattening). It was the useful half of the
  old Contrast stage, and on this bench the lighting is even enough that it had
  nothing to correct. **A vignette correction is the version of it worth
  having** if a wider field or a different lamp ever makes the corners matter —
  that is a different filter (a calibrated per-pixel gain, not a blurred-copy
  division) and it is deliberately not this one.
- The old contrast stage's *other* half — the one that decided what is a sample
  — is still gone from the chain as it has been since #34. The mask's source is
  the picked colour.

### The local-contrast curve

The idea is a *matched gain*: steepen the tone curve at the colour the operator
picked, and flatten it everywhere else. The substrate plateau, which carries no
layer information, is compressed into a narrow output band; the few levels that
separate one layer from the next get the display's range instead. In 8-bit
terms a monolayer and a bilayer might be five or ten levels apart — above the
sensor's noise but below what the eye separates at a glance — so amplifying
them recovers information that is genuinely there.

Two properties make it usable rather than merely plausible, and both are
asserted in `tests/unit/test_preprocess.py`:

**The curve is pinned at 0, at the picked colour, and at 255.** The pick is
the one that matters, and not for aesthetics: the *same hex* is the colour
mask's target. A curve that moved the picked value would leave the mask
searching for a colour the frame no longer contains — you would point at a
flake, switch the filter on, and watch it disappear. That is why the curve is
applied **last**, after the spatial stage: its fixed point lands on the value
the dropper actually read, which is a value in output space.

**The gain means the gain.** "×3" is the slope at the picked colour, solved for
rather than assumed — the naive construction delivers about 2.9 and calls it 4,
because pinning the endpoints has to take something back. Where a request
cannot be honoured the *band* gives way rather than the gain: pinning three
points fixes the total slope at 255 whatever the parameters, so a wide band
boosted hard has to pay for itself with a shelf somewhere, and at ×8 over
±32 DN that shelf was a hundred input levels collapsing onto one output,
twenty DN from the colour under examination. The band is narrowed until no more
than a few levels share an output, and the panel reports what it used
(`band ±18 DN (asked ±32)`). Only when even the narrowest band cannot carry
the gain does the delivered gain come out lower, and it says so.

**What it does not fix.** It is a point operation, so it cannot tell a layer
difference from a *lighting* difference — there is no illumination flattening
behind it any more (see above), so a field with a real gradient in it is a
field this chain cannot help with. It also cannot separate two layers whose
colour difference sits under the sensor's noise: it multiplies what is there,
and if what is there is noise, the noise is what gets multiplied — which is
what the denoise is for, and why the denoise runs first.

## Configuration

The chain serialises to `identify.stages` in the settings file — the bundled
defaults *are* the schema — and `IdentifyConfig.from_dict` is deliberately
forgiving: an unknown stage is dropped, a missing one returns at its default, a
parameter with the wrong type falls back to its default, a malformed hex falls
back to a colour that works, and an unknown `method` falls back to `window`
(the behaviour every file written before the field existed had) with a log line.
A hand-edited settings file must not be able to stop the pipeline from running.

## The three views

The live view's floating switch offers **Original**, **Pre-processed** and
**Samples**.

*Original* is the camera's own frame. *Pre-processed* is what the chain above
makes of it — the layer the dropper samples. *Samples* is the identification
result drawn over that same layer: everything the source did *not* match is
darkened, and every region it did match is outlined by its verdict — **bright**
for what survived the whole chain, **dim** for what a gate rejected. The
operator sees not only what was found but what was thrown away and where, which
is what makes tuning a parameter feel like tuning a filter chain rather than
guessing.

Deliberately not rectangles: a box round a shapeless blob says little, boxes
from neighbouring regions overlap and read as one object, and a box hides the
shape the operator is judging. The darkening keeps the sample's own pixels
visible inside the match, so the view reads as "this part of the wafer" rather
than as a mask poster.

**The dropper samples the layer that is on screen.** Normally that is the
pre-processed one: it IS the array the identification ran on, so a colour
picked off the screen is a colour the mask will look for, by construction
rather than by coincidence. It is never the *composited* display — in Samples
mode the screen is darkened and outlined, and sampling that would return a
colour the sample does not have. And while the processed layers are **held
back** (a moving stage, a running scan) it is the raw frame instead, because
the held layer is a picture of where the stage *was* while the click is mapped
with the frame actually on screen. A pick while an axis is moving is refused
outright, with the reason, for the same reason.

**...and the pointer is the patch.** While the dropper is armed the mouse
cursor *is* a circle of the region a click will average, with a cross at the
sampling pixel — the platform pointer is hidden, and comes back off the image
or when the pick lands. The size is the colour card's **Patch** row (a radius
in frame pixels; the row shows the diameter, since that is what the disc
spans). It exists because at fit-to-window scale the default 9-px patch is
about two screen pixels: the operator could not see what they were averaging,
which is the same problem the warning below reports after the fact.

**...and it judges the patch it averaged, out loud.** The sample is a
circular mean a few pixels across, not a pixel, so a click near a flake's edge
returns the mean of
the flake and its substrate — a colour *neither* material has, which then
becomes the mask's target and the curve's centre. It is the "sometimes it finds
the other layer" case, and it is now visible rather than silent: the colour
card's line reports the patch's own spread, in warning colour above 12 DN,
with the fix ("click further inside"). Two nearby consequences of the same
click are reported there too: a patch spanning an edge, and a pick paler than
the Min saturation/Min brightness floors (see the clamp above).

### The two processed views stand down while a stage moves, or a scan runs

They cost a pre-process and a full identification per job, and while the
picture underneath them is moving they are a picture of where the stage *WAS*.
So the display falls back to the raw stream — the mode bar keeps its selection
and the view returns to it — whenever an axis the camera can see is moving
(XYR: it blurs the image; focus: it changes what is in the image) and for the
whole of a scan, which additionally **suspends the live feed** so the worker's
time goes to the tiles.

A caption over the image says why the frame on screen is not the layer the
button names; a mode bar that is silently wrong is worse than a pause. A
*tile's* result is deliberately not affected: it no longer writes the display
buffers at all — a tile is a different part of the sample, and letting it
become the layer on screen (or the layer the dropper reads) is how the view
jumps to a region nobody is looking at.

### ...and the chain only runs for the view that is showing

The same costs are what made the feed worth accounting for, so the live feed
computes exactly what the visible view needs and nothing else:

| On screen | What the worker does per tick |
|---|---|
| **Original** | nothing — the raw frame is already there |
| **Pre-processed** | the pre-processing chain only (no mask, no overlay) |
| **Samples** | the chain, the identification and the overlay |
| the Navigation workspace | nothing — that tab has no chain to feed |

Two consequences, both visible rather than silent. The live stage counts and
the live sample list come from the identification, so they are **cleared** when
the pipeline stops rather than left on screen describing an older frame — the
identify group's readout is for the Samples view. And the scan's tiles are
unaffected in every case: a tile that is not identified is a sample that was
not found, so those are always run in full.

The transfer (XYZ) axes are excluded on purpose: they never appear in the
image, and their firmware has no busy flag, so a motion inferred from a
position delta would pause the previews on a noisy sample for no reason.

## Performance and scheduling

- **Live preview**: one job in flight, newest wins — a frame that arrives while
  the previous one is still being processed is *dropped*, because a preview that
  lags the stream is worse than one that skips frames.
- **Scan tiles**: queued unconditionally. A tile that is not examined is a
  sample that was not found, so nothing is dropped; the queue drains while the
  scan runs and after it finishes, and the exports wait for it (detection
  deliberately outlives the capture).
- A failure inside the pipeline is logged and the next job runs: it can never
  stop a scan.

**Everything here runs on one worker thread, and never on the GUI thread.**
That is what makes an expensive pre-processing stage acceptable, and it is why
the two processed views are slight previews rather than the live stream: they
cost the worker's cadence, and the camera's capture sequence, the frame slot,
the autofocus and a running scan never wait on them. A preview that lags is
recoverable; a capture that waits on a bilateral filter is not.

## Tuning on the bench

Look at **Pre-processed** while setting the filters up, then switch to
**Samples** to see what the chain made of it. Start with colour alone, then add
the chain one stage at a time and watch the counts — each stage's `(in, out)`
says whether it is doing anything useful.

1. **Pre-processing**, if the sample needs it. Pick the colour first (the
   dropper on the Pre-processed view), then try the local-contrast curve: the
   gain is the slope at the picked colour, the band is how far either side it
   stays steep. A wide band with a high gain will be narrowed automatically;
   the readout under the controls says what it actually used. A vignette
   correction is the intended future of the removed shade stage (see "The
   three stages that were removed").
2. **Match by**: leave it on **Window** to start with — it is the only method
   that separates two *thicknesses* of one material (see "Three ways to
   match"). Reach for **HSV dist.** when the substrate keeps creeping in along
   a lighting diagonal, and for **RGB dist.** on a grey or near-white
   background, or when the shade is the whole question and the lamp is stable.
   Remember the switch re-opens the shade window if you had narrowed the
   spread — the card says so.
3. **Hue tolerance** until the flake family is caught without the substrate.
4. **Shade spread** (Window only) when two *thicknesses* of the same material
   arrive together — the monolayer and the bilayer are close in hue and far
   apart in brightness, so this is the slider that separates them. Narrow it
   until the layer you do not want drops out, and check the colour card's line
   first: if the pick itself was a mixture, fix the pick before touching this.
5. **Min saturation** if the substrate (or the illumination gradient) comes in
   with it — this is the parameter that does the most work on a real wafer.
   It is a *floor*: it rejects anything less saturated than the value, which is
   the cheap way to cut a substrate paler than the flake, and it applies under
   every method. **Min brightness** is the same guard on the dark side, and it
   is off (0) as shipped: lower it never, raise it for a flake that is the
   bright thing on a dark field, or to kill the dark rim at the field's edge.
6. **Clean up**: raise the kernel if a flake fragments into speckle; it also
   merges broken edges, so watch that it does not eat small samples.
7. **Size**: the µm² floor is the honest filter, and it is in physical units, so
   it means the same thing at every objective. The µm² floor is the one filter
   here whose threshold is a fact about the sample rather than about the image.
8. **Sharpness**: raise it to reject defocused blobs. Its default (4.0) is
   deliberately low, because a colour-matched contour traces the colour
   boundary and scores in the tens — a threshold picked without measuring would
   reject most real samples.
9. **Merge**: raise the gap when one flake arrives as several boxes.

The unit tests build synthetic frames with known truth (a blob of a known colour
at a known place, and a substrate that is none of those), which is the fastest
place to check a change of behaviour; `tools/validate_cv.py` renders annotated
PNGs for real frames.

## Bench status

The thresholds have only ever met synthetic blobs, and so has the curve: its
properties are proven on synthetic level ramps, not on a real monolayer/bilayer
pair. The first real-wafer tuning session is still owed — it is the next
item after the scan's own bench campaign, and the checklist is
`docs/journal/BENCH_TODO.md`.
