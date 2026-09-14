# Autofocus — Agile + Accurate on an Open-Loop Focus Axis

## Physics of the axis

- Arduino + CRD5103PB stepper, **open-loop** (no encoder). The firmware's
  position counter counts COMMANDED steps; the load (what the camera
  sees) lags the counter by the mechanism backlash B in the current
  direction after each reversal: `load = counter − dir×B`.
- 0.36°/step 5-phase stepper directly driving the Olympus BXFM fine-focus
  knob, 200 µm/rot → **0.2 µm/step** (`devices.focus.um_per_step`).
- Positions reset on reconnect/ZERO — everything is session-relative.
  This is WHY the autofocus range limit is **relative to the arm
  position** (commanded travel this run), not an absolute bookkeeping
  coordinate: an absolute limit is only as trustworthy as the accumulated
  lost-step drift; a relative window is exactly what the axis physically
  executes. Firmware soft limits (SLIM) remain as a backstop.

## Per-objective optics (user-editable, global)

The objectives table lives in `settings → objectives` (toolbar
**Objectives…**, or **Edit objectives…** in the Focus window) and is
GLOBAL: name, optics, the speed multipliers, and the manual px→µm
calibration. The selected objective (top-bar combo) persists across
restarts. Saving the dialog syncs the registry one-way into the
CalibrationStore (the manual `px_um` lands as a `settings_manual`
calibration entry, so Sample Finding and the Calibration workspace see
it through the measured > Labscope composition).

DOF ≈ λn/NA² + n·e/(M·NA) (λ = 0.55 µm, e = 2.4 µm pixel). Seeds from the
Olympus LMPlanFL series; **tune each row on hardware** (see the tuning
ladder below):

| Objective | NA | DOF | window | coarse | fine | AF mult | man mult | stage mult | eff. coarse |
|---|---|---|---|---|---|---|---|---|---|
| 5× | 0.15 | ~28 µm | 1000 µm | 5 µm | 1.0 µm | 1.0 | 1.0 | 1.0 | 100 µm/s |
| 10× | 0.30 | ~7 µm | 60 µm | 2 µm | 0.4 µm | 0.25 | 0.25 | 0.5 | 25 µm/s |
| 20× | 0.46 | ~2.9 µm | 30 µm | 1 µm | 0.2 µm | 0.1063 | 0.1063 | 0.25 | 10.6 µm/s |
| 50× | 0.55 | ~2 µm | 20 µm | 0.4 µm | 0.2 µm | 0.0744 | 0.0744 | 0.1 | 7.4 µm/s |
| 100× | 0.85 | ~0.8 µm | 12 µm | 0.2 µm | 0.2 µm | 0.0311 | 0.0311 | 0.05 | 3.1 µm/s |

- **AF mult** (`af_speed_multiplier`): scales the AUTOFOCUS scan speeds —
  global `coarse_speed_base_um_s` (100 µm/s — aggressive; it exceeds the
  sampling-density bound at 5× and the warning says so — the operator's
  choice) × the multiplier (default ∝ 1/NA² — the DOF scales as λ/NA², so
  the tolerable blur budget and with it the usable scan speed scale the
  same way).
- **Man mult** (`focus_manual_multiplier`): scales the MANUAL focus jog
  speeds (keyboard +/−, gamepad triggers) — default the same NA ratio.
- **Stage mult** (`stage_speed_multiplier`): scales the manual XYR/XYZ
  high/low jog speeds (keyboard, gamepad, on-screen buttons) — default
  `5/mag` so the apparent on-screen motion speed stays roughly constant
  as the field of view shrinks. Single-step distances are NOT scaled.
- **px→µm** (`px_um`): the manual pixel calibration (0 = unset); see the
  store sync above.

Rules enforced by `build_config` (warnings only — the operator tunes):
- `fine_step ≤ DOF/4`, `coarse_step ≤ DOF`;
- coarse motion blur `v × exposure ≤ DOF/4` (the low-frequency coarse
  metric tolerates far more blur than the fine metric);
- **sampling density** `v ≤ coarse_step × camera_fps` — the coarse peak
  localization needs ≥1 sample per coarse step; above the bound the
  curve gets too sparse to localize the peak (hardware-verified at 5×:
  150 µm/s failed outright, ~75-100 µm/s is the practical ceiling at
  ~15-21 fps).

## How a run works (AF-S — the classic flow, kept as shared machinery)

The app runs the **adaptive v3** pipeline (below); this 6-phase flow is
the classic controller's — stored knowledge, still reachable from the
sim suites and the bench tools. Its preflight, peak-pick, landing and
safety invariants are shared by every controller.

1. **Preflight**: window = arm position ± span/2, clamped inside SLIM
   (10×fine-step margin). Empty window → fail without motion.
   **Caveat (2026-09-14)**: SLIM ships DISABLED on this firmware, so the
   firmware is not enforcing those bounds — the clamp above is a software
   bound, and it is now applied whether or not the firmware flag is on.
   Autofocus reports the real flag state at run start and warns when it is
   off. Do not treat "the bounds read back" as evidence that a sweep
   cannot leave them.
2. **Coarse — one continuous pass** across the window (nearest edge
   first), scoring every fresh camera frame *while moving*: the firmware
   reports live POS during motion, and each frame's position is
   interpolated from the (t, POS) history at its CAPTURE timestamp
   (frames carry backend capture timestamps through a shared
   LatestFrameSlot). One monotonic direction keeps backlash constant
   inside the curve. Frames older than `freshness_ms` are skipped
   (camera stalls drop points, never fake valleys).
3. **Peak pick**: multi-peak selection prefers the candidate NEAREST the
   arm position (wafer surface vs the mount below — the global max is
   NOT followed); edge peaks → "widen the window"; flat curves fail
   without a blind move; a plateau top is centroid-averaged.
4. **Fine — slow step-and-shoot** around the fitted peak, every point
   scored with a frame captured AFTER the move ended (freshness gate).
   If the final peak sits at the fine window's edge, the window is
   EXTENDED once and re-swept (sparse-coarse localization cannot kill
   the run).
5. **Landing — overshoot-and-return** from the SAME direction the fine
   sweep measured: the fitted counter position is only valid on that
   side of the backlash. Slowest reliable speed (50 steps/s).
6. **Verify**: ±2-step counter readback.

~15–20 s per run at the seed values. Agile levers (hardware-tuned):
shorter `af_exposure_us` (allows faster sweeps within the blur budget),
smaller windows once your defocus habit is known.

## Adaptive strategy (v3 — the only strategy the app runs)

The app runs **one** autofocus strategy: the derivative-hybrid
**Adaptive v3** (hardcoded in the service/proxy — the strategy selector
and the strategy registry are gone). The other algorithms stay in the
codebase as **stored knowledge**, reachable only from their sim suites
and the bench tools: Adaptive v2 (probe-driven, frozen as the rollout
backup), Adaptive v1 (the hardware-verified 2026-09-07 baseline), the
Classic 6-phase flow above, and the AF-C state machine
(`talos/cv/af_c.py` + its unit tests). They share the preflight,
peak-pick, landing and safety invariants.

### Adaptive v3 — derivative-hybrid (the default)

The sharpness-vs-defocus curve is near-Gaussian (S = A·exp(−d²/2σ²),
σ ≈ DOF/3): its second derivative is strongly negative near the peak,
zero at ≈ 1.1σ, positive beyond. v3 exploits it WITHOUT replacing the v2
machinery — each derivative signal ADDS a hybrid branch, and every one
that misfires degrades to the v2 behavior (a config of zeros makes v3
run exactly the v2 pipeline; the sim suite pins that). The numbers
below are audit-verified on the Gaussian model and hardware-tuned via
the bench's Step-0 σ measurement.

**Probe classification** (pure math in `af_math.probe_classify`, motion
unchanged): the 3-point curvature C = (S₋δ − 2S_c + S₊δ)/(δ²·S_c) —
≈ −0.64/σ² at the peak, zero at d* = arcosh(e^(δ²/2σ²))·σ²/δ (≈
1.1–1.2σ for δ/σ in 1.1–1.6), positive beyond; linear backgrounds
cancel exactly in C. The verdict:
- **near (curvature+max)**: C < −`probe_curv_in` AND the center is the
  three-point max — the **valley gate** (audit #8/#9: without it the
  between-planes valley reads "near" and the mount wins). This branch
  catches peaks where the v2 weaker-side test fails (small δ/σ).
- **near (plateau)**: the v2 weaker-side test (C ≈ 0 on flat tops).
- **near (cluster)**: the center-max gate fails (a neighbor sub-peak
  within δ beats the center) BUT all three points are HIGH and
  genuinely spread — S_c ≥ 0.5× the best side, the worst side ≥ 0.4×
  S_c, the best ≥ 1.2× the worst, AND the best clears
  `probe_cluster_min_score` ABSOLUTELY (0 = the branch disabled; a
  per-field knob the user tunes to their peak's magnitude — ratio
  gates cannot tell a flat far tail from a plateau shoulder,
  sim-found). Stage 2 enters directly, anchored at the BEST probe
  point (the `_probe_anchor` hook). Hardware-found: the multi-peak
  shoulder misread as blind.
- **flank**: C > +`probe_curv_out` — past the inflection; the direction
  call is the low-freq slope's job either way.
- below the RELATIVE floor (`probe_score_floor_ratio` × the preflight
  baseline) or no signal → the v2 low-freq slope / blind.
The "curvature-wins-where-weaker-side-fails" disagreement branch is
deliberately DROPPED (audit #9: it loses on the mount-within-δ geometry
— that case falls through to the v2 directed coarse + nearest-center).
**User-verified on the multi-peak field**: the 3-point direction call
is unreliable (conflicting slopes → blind; a wrong away-side call was
observed) — the probe keeps the NEAR-PEAK role; the stage-1 direction
comes from the pass itself (below).

**Stage-1 stop ladder** (guard first, then): (1) the **curvature stop**
— a sliding 5-sample PASS-ONLY window (the stationary probe seeds are
5×-score outliers with huge x-leverage — they seed classification only)
fit with a centered parabola; it fires when the normalized metric
2a·h²/S_max < −`coarse_curv_stop` AND the sweep-aware detrended slope
gate b_eff × sweep_direction > 0 holds (the b-kernel is (−2,−1,0,1,2)/10h;
the background slope from the pass's own first samples is subtracted —
audit L3) AND two consecutive windows were negative (the debounce —
adjacent 5-sample curvature estimates are exactly uncorrelated, so the
persistence genuinely squares the false rate); halt = ramp stop,
NORMAL stop. (2) the v2 post-peak early-stop (backstop), (3) the v2
walk-away, (4) the direction guard. The curvature state lives inside
each cont_pass invocation (auto-reset on the guard reversal; the stop
also runs on the reversed pass and the blind sweep).
On the curvature-stop reason the peak analysis **BYPasses** pick_peak
and the edge check entirely (the curve ends climbing — a picked peak
would be flagged at_edge and aborted — audit #1): the quality gate
runs on the pass-only series (settle frames AND probe prefix excluded),
and the anchor is the window's **trusted vertex** when the metric is
below the `coarse_curv_vertex` floor (the curvature AT 0.7σ — the
vertex estimate is only valid within ≤ 0.7σ, and beyond ~0.9σ the
window parabola opens upward), else the stop position. `_coarse_anchor`
uses the bypass anchor directly — no parabolic re-fit over the stage-1
curve (the settle outlier + climbing end would drag it off — audit
#12/#13).

**Cascade direction guard**: the v2 15%-drop test and sharp-rise gate
stay EXACTLY as v2 (the fallback); an ADDITIONAL trigger fits the
pass-only low-freq series vs POSITION and reverses when the slope's
2σ interval lies entirely below zero: b̂ + z·σₙ/√Sxx < 0 with σₙ the
POOLED STATIONARY noise from the probe's three points (a 3-parameter
fit's residual SE has N−3 = 2 DOF — z=2 on it is a 9.2% one-sided test,
so the residual SE is sanity-only — audit A10), CAPPED at 5% of the
mean low-freq score (on structured fields the probe's points straddle
sub-peaks and the detrended residual reads 10-50× the honest noise —
the trigger would be permanently disabled), and the sharp-rise gate
unmet (audit #7). One reversal, same as v2. Every guarded pass logs a
one-shot diagnostic: "direction guard: low-freq fall N%, sharp rise
gate MET/unmet — no reversal" (why it did not fire). Hardware-proven:
the guard evaluates (28% fall seen) but the rise gate blocks it on the
multi-peak field — the away-side sub-peaks climb even the wrong way,
and the gate NEVER reverses a climb (conservative by design; the
guard is a time-saver for single-peak fields, not a correctness
dependency).

**The blind fallback (stage 1 with no probe direction)**: starts AT
THE ARM sweeping the DEFAULT direction (`coarse_direction`: 0 = +1
away from the sample — the safe default; ±1 forces it) with the early
direction check armed — the drop test + the 2σ trigger reverse the
pass toward the peak side from its OWN early samples (the user's
conclusion: the 3-point probe is not trustworthy for the direction on
structured fields; the pass decides). Strictly better than the v2
nearest-edge walk (frozen in V2): the peak region sits near the arm,
so the pass reaches structure in ~100s of steps instead of ~2500, and
the walk-away/early-stop/curvature rungs stop at the first structure
crossed either way. Sim-pinned in both geometries (ahead: no reversal,
the − side untouched; behind: the 2σ reversal recovers the peak).

**Per-objective thresholds** (audit L1): `coarse_curv_stop`,
`coarse_curv_vertex`, `probe_curv_in/out` are computed in
`build_config` from DOF (σ ≈ DOF/3) and the coarse sampling spacing
(h = speed/fps): the stop ladder reproduces the audit's per-mag values
(5× ≈ 0.25, 10× ≈ 0.10, 20× ≈ 0.06), the vertex floor is the closed-form
SG curvature at 0.7σ, the probe thresholds 0.3/σ² and 0.1/σ².
Per-objective-row and per-config overrides win (0 = auto). The bench's
Step-0 measured σ goes into `af_cfg["sigma_steps"]` — it REPLACES the
DOF model (hardware: 107-114 steps at 5× vs the DOF/3 estimate of 47 —
the model underestimates by 2.3×). A **vertex-in-window gate** rejects
the metric's far-tail oscillations (their parabola vertex lies many
spacings outside the window — hardware-verified at 483 steps = 14h;
a true ≤0.7σ approach puts it ~0.25σ ahead).

**The near-entry stage-2 window** (`near_window_steps`, 0 = the default
fw): the near verdict tolerates the arm up to ~±0.7σ off the true
peak — the default fw is sized for the COARSE localization error
(max(coarse_step, 3×fine_step, 3×coarse sampling spacing)), so a hill
fit 74 steps from the anchor was distrusted (|fit − anchor| > fw//2)
and the re-measure around the anchor failed on the flat shoulder
(hardware-found: "hitting window edge when already very near-focus").
On a near/cluster verdict the stage-2 window becomes max(fw,
near_window_steps), scaling the distrust tolerance with it.
build_config computes max(fw, 1.5σ) from the DOF model or the measured
`sigma_steps`; overrides win. The user's 5× settings carry 170.

**Hardware tuning state (2026-09-08)**: the Step-0 gate measured
σ = 107.8-113.6 steps with R² = 0.995. With the σ-corrected thresholds
the curvature stop fires for real — but the rollout gate said STAND
DOWN: the field is MULTI-PEAK (a ~190-step-wide bump of comparable
sub-peaks — the 20-step σ sweep had smoothed them over) — the stop
fires on a sub-peak's flank, the vertex lands BETWEEN peaks, and the
inherited stage-2 fit-distrust/end-max chain fails on the flat zones.
`sigma_steps` stays UNSET in the shipped settings → the stop stays
dead at 5× (the DOF/3 threshold is unreachable) and v3 runs the v2
pipeline + the probe classification. The delivery gaps (300-580 ms
every ~20-40 s on the marginal USB3 extension cable) were FIXED by the
user's wiring separation — `tools/cam_gap_diag.py` verifies (1
connect-time gap per 3 min after the fix). Current hardware numbers:
the cluster entry fires in the wild (5.84 s direct stage 2),
repeatability 8 steps = 1.6 µm; the deep-defocus family (±200/±400/
+600) remains fail-safe. **The tuning resumes: set sigma_steps ≈ 110
and re-run tools/af_bench3 on a single-plane field, one edit at a
time, vision-verified.**

**Failure semantics (v3, the user's policy)**: failures and stops end
at the CURRENT position — no arm restore on ANY failure path. The base
controller's `_restore_on_fail` policy attribute is set False by v3
(the stored-knowledge controllers keep the arm restore); aborts never
move the axis either way. Failed results carry the note "— stopped at
current position".

### Adaptive v2 — probe-driven dispatcher (frozen rollout backup)

Every run starts with a **probe**, then enters the stage the probe
dictates — the mode no longer branches inside the controller:

0. **Probe** — three points at [center−δ, center, center+δ] (δ =
   `probe_step_steps` or 3×coarse_step, TRAP moves at the staging speed),
   measured CENTER-FIRST with a return to center (the monotonic ordering
   broke the near-focus logic when armed at the peak — user-reverted).
   Each point is scored with BOTH metrics: the
   SHARP metric decides **near-focus** (the center is a clear local
   maximum ≥ `probe_peak_ratio` above the weaker side → skip straight to
   stage 2, anchor = center); the LOW-FREQUENCY metric (Brenner k=8 on a
   2×-binned crop — its broad response gives a usable slope far from
   focus where the sharp metric is flat) estimates the **hill
   direction**. No information → the blind v1 sweep below.
1. **Coarse** — with a direction, the CONT pass starts AT the arm
   position and sweeps toward the rising side, stopping at the FIRST
   peak (`coarse_early_stop_samples`). A **direction guard** watches the
   first few samples: the low-freq series falling while the sharp series
   never rose (the peak-seen gate not met) means the probe pointed the
   wrong way → the sweep REVERSES once, from the current position. A
   **start-on-peak walk-away** stop covers the corner where the pass
   starts ON the peak (the peak-seen gate can never fire there). Blind
   fallback (no probe information): the v1 nearest-edge full-window
   sweep, unchanged. The probe's sharp samples seed the stage-1 curve
   (position-sorted before analysis) so a peak at a probe point is never
   an "edge peak".
2. **Hill climb** — v1's CONT pass across the fine window with the
   position-based velocity profile. **Boundary-hit salvage**: a pass cut
   short by a limit/budget reason (never an abort) keeps its curve when
   it contains a COMPLETE peak — samples on both sides of the picked
   peak + enough points to fit. The hit-window fail only fires when the
   pass genuinely ended while still climbing.
3. **Lock-on** — v1's direct fit / stationary re-measure, plus the
   complete-peak gate on the edge check: a peak near a window edge with
   samples on both sides is accepted instead of failing. The stationary
   end-max discipline (never chase a climbing curve's end) is untouched.

A stage-2 failure (phase "fine", not aborted) retries the stage LOCALLY
up to `stage2_retries` times around the CURRENT position (stage 2 is
already near focus by definition): never a return to the arm position,
never a stage-1 fallback. After the retries the run stops AT the
current position — the failure result carries `restore_on_fail=False`,
so the axis stays where the last attempt stopped (the user's policy:
an AF that is near focus must not run away or undo itself).

The Focus window plots both metric series: sharp (blue) and low-freq
(amber, per-series normalization — the magnitudes differ by orders of
magnitude).

### Adaptive v1 (frozen baseline)

1. **Guided coarse sweep** — one CONT-mode pass across the window at the
   aggressive per-objective speed, scored with Tenengrad (the
   low-frequency metric's broad plateau cannot localize the peak on a
   DOF-wide scene; hardware-verified), nearest-edge-first. Early-stop
   after passing the peak (peak-seen gate: the running max must rise
   `early_stop_rise` above the curve start).
2. **Velocity-proportional hill climb** — a CONT pass across the fine
   window (max(coarse_step, 3×fine_step, 3× the coarse sampling
   spacing)) scored with Tenengrad at full resolution. The speed schedule
   is POSITION-based: v_cap at the pass edges, `hill_v_min` (blur ≤
   fine_step/3) within 2×fine_step of the anchor, linear ramp between —
   deterministic where the score-based schedule collapsed on DOF
   plateaus (hardware-verified). Early-stop 2 frames after the peak.
3. **Parabolic lock-on** — fit the hill curve near the peak when ≥4
   samples sit within 85% of it; otherwise a 5-point stationary
   step-and-shoot re-measure (classic machinery, staged past the first
   point so every scored point shares one backlash state) with a
   classic-style one-shot window extension. A curve still climbing at
   its edge FAILS (never chases the stronger plane below — the mount
   must not win).

### Motion architecture (shared)

Both passes run in firmware **CONT mode** (signed `SPD` = direction;
live `STATUS?` polls build the (t, POS) history). Stops are **ramp
stops** (`SPD:0` decelerates at ACC) — the axis is open-loop, so an
instant STOP at speed loses steps silently; the predictive edge stop
begins the ramp with the decel distance + poll slip + safety in hand,
allowing an 8×fine-step overshoot into the SLIM margin (a legitimate
peak can sit inside the stop zone). `set_speed` updates are throttled to
10 Hz (SPD floods wedge the firmware RX). Verified firmware behaviors:
mid-CONT SPD changes speed live; SPD:0 ramps to IDLE with no event;
MOVE/GOTO during CONT → ERR:BUSY (passes are staged with TRAP moves).

**Camera timestamps are content-corrected**: the SmartCam event stamp is
the DELIVERY time — the backend subtracts exposure/2 + a transfer
allowance (`smartcam.capture_latency_ms`). Without the correction the
interpolated peak shifts by v×latency (~25 steps at 500 sps) and
freshness gates accept frames whose content still shows the previous
position. **Hardware tools must pump the camera** (greedy fetch into a
LatestFrameSlot, like the app's camera worker): the on-demand fetch
pattern starves the SmartCam delivery to ~3 fps and empties the
interpolation history.

## AF args (scalability)

- **`bounds`** — an absolute-stage search-window override on
  `run()`/`AutofocusRequest`/`AutofocusService.start_af_s`: replaces
  the symmetric center ± span/2 window (still clamped inside the soft
  limits) — asymmetric, scenario-specific windows. None = the symmetric
  config window.
- **`coarse_direction`** — the DEFAULT coarse sweep direction when the
  probe has no direction info: 0 = auto (+1 away from the sample, the
  safe default); ±1 forces the blind pass's sweep (+1 away/up, −1
  toward/down). Scenario overrides; the guard's early direction check
  still reverses on a strong fall.
- **The speed reference**: every objective's effective speed = the
  global `coarse_speed_base_um_s` × its `af_speed_multiplier`. The
  auto multiplier = (na_min/na)² where na_min = the LOWEST-POWER
  objective's NA from the table (passed by the callers) — that
  objective gets exactly the base speed (the table's max); explicit
  per-row multipliers win (user-modifiable). The multiplier flows into
  the stage-2 fine window through the coarse sampling spacing
  (speed/fps); the TOTAL search span stays the per-objective user
  window — the args own the bounds. Unit tests pin the whole model
  (the speed table, the window scaling, the span independence).

## Modes

- **AF-S (focus once)**: the v3 adaptive flow; lands and stops. The
  app's only autofocus mode — AF-C was unwired from the app; its state
  machine (`talos/cv/af_c.py`), the controller-level AF_REFINE mode and
  their tests stay as stored knowledge.
- **Measure area**: full frame, or a user-drawn ROI (rubber band on the
  live view, letterbox-correct mapping, resolution-safe normalization).
  ROI selects WHICH plane the metric sees when the field contains
  features at several depths.
- **Focus window**: the autofocus panel lives in its own window, toggled
  from the toolbar (**Focus**). Esc/close hides it (the global STOP ALL
  shortcut stays on the main window). It plots the sharp curve (blue)
  and the low-freq series (amber), each with its own score
  normalization.

## Range limit (the wafer-on-copper scenario)

The window is symmetric around the position where AF was armed. Focus on
the wafer, arm AF, keep the window smaller than the wafer thickness →
the mount is unreachable by construction. Extra guards: nearest-center
peak selection (never the global max) and `fail_on_edge_peak` (the truth
is outside the window → fail, don't guess).

## Backlash auto-calibration

"Calibrate backlash" (panel) or `--calibrate-backlash` (CLI): on a
feature-rich field, two monotonic sweeps (up then down) with a
forced-state staging dance (drop past the start, return — the return
always reverses, taking the dead zone up deterministically). The up-sweep
peaks at truth+B in counter coordinates, the down-sweep at truth−B;
parabolic fits of both peaks give B = offset/2. Result is stored into
the active objective row (`backlash_um` + `backlash_measured_at`).
Expect ~0.2–2 µm; repeatability ≤ 0.4 µm on a clean field.

## Manual jog gating (input aborts the run)

While a job runs (mode AUTOFOCUS), user input **aborts** it and the
input then proceeds — input wins. The service's `sig_job_submitted`
hook is the single abort point: stage moves (jogs, go-to, home/zero),
manual focus adjust, and snapshots abort; camera exposure/gain/WB
changes and the yudian setpoint do NOT. The InputSystem gate passes
AUTOFOCUS-mode inputs through instead of dropping them (SCAN mode still
drops); inputs arriving during the 350 ms arm window cancel the pending
arm instead of starting (an abort landing between the arm submit and
the controller's construction latches in the proxy and is polled by
the controller's `_check`). Esc / STOP ALL stays live (STOP ALL aborts a
running job through the FocusProxy and cancels a pending arm through
the service). Enqueueing a release-stop purges queued CONTINUOUS motion
commands (a quick tap during an AF run would otherwise re-start the
axis after its release — hardware-verified and fixed). The autofocus
NEVER alters camera exposure/gain (user rule).

## Tuning ladder (hardware, user present)

1. Backlash calibration (3×, feature-rich field) → store.
2. AF-S from ±5×DOF defocus, 5 repetitions — must land within ±1 fine
   step of the best manual focus (compare Tenengrad of both).
3. Tune per objective: window (smallest that never misses your defocus
   habit), coarse speed (fastest without visible streaking), fine step
   (≤ DOF/4, ≥ 0.2 µm), backlash row.
4. ROI AF on a flake; two-height field → nearest-arm plane wins; tiny
   window + far-away plane → refusal without motion.

## CLI

```
python tools/autofocus_hardware.py                          # step-based defaults
python tools/autofocus_hardware.py --objective 2            # µm table row
python tools/autofocus_hardware.py --objective 0 --defocus 200 --mode AF_REFINE
python tools/autofocus_hardware.py --calibrate-backlash --store --objective 0

# v2 hardware bench (vision pre-flight + probe/near-focus/salvage
# protocol; snapshots + curve CSVs into docs/bench/)
python tools/af_bench2.py --preflight-only                  # capture + exit
python tools/af_bench2.py                                   # full protocol
python tools/af_bench2.py --runs 3 --skip-salvage --skip-preflight
```
Curve CSV → `%APPDATA%\TALOS\autofocus_curve.csv` (position,score).
The bench's camera health gate rejects noisy EMI windows (periodic
SmartCam delivery gaps) — retry in a quiet window.

## Safety invariants (enforced by code + tests)

- No motion on empty window / flat curve / edge peak (fail, never guess).
- Abort checks every ~50–100 ms inside the coarse scan; STOP ALL aborts.
- Since 2026-09-14 the abort is also polled **while an axis move is in
  flight** (`move_to_verified` → `_wait_idle_checked`), not only between
  moves: a jog or STOP ALL that landed during a move used to wait out
  `wait_idle`'s 60 s budget with the axis still travelling. Bench drill:
  abort → stage stopped and the run unwound in 0.20 s.
- Every move readback-verified (±2 steps); EV:TMO re-issued once.
- Landing at the slowest reliable speed; approach direction matches the
  measurement direction.
- Camera exposure/gain never altered on any path; the mode lock is
  released on every completion path (success, abort, failure, arm
  cancel).
- User motion/snapshot input aborts a running job (the service hook);
  failures and stops end at the current position (v3 — no arm restore).
