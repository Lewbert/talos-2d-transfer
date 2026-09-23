"""Sample identification as a chain of stages.

The operator builds a filter chain rather than picking a mode: a SOURCE
stage turns the frame into a mask (the colour match), one stage cleans the
mask up, then gates decide which blobs survive (size, frame edge,
sharpness) and a final stage merges fragments of the same object. Every
stage can be switched off, and each reports how many candidates it let
through — so the panel reads like the chain it is: ``colour 812 → size 12 →
sharpness 2``.

Colour is now the only source, deliberately. There used to be a contrast
source (Otsu on an illumination-flattened frame) and it answered a
different question — *what is here at all* — which is not the question a
transfer workflow asks. On the thin samples this is built for, the
threshold follows the bulk of the histogram rather than the object of
interest, and a monolayer's deviation from the substrate can sit at the
flattening's own residual. Its useful half, the flattening, is a
pre-processing stage now (``cv/preprocess.py``), where it prepares the
frame instead of segmenting it.

**The frames this runs on are pre-processed, never composited.** The
input is the raw camera frame with the operator's pre-processing chain
applied and nothing else — no scale bar, no crosshair, no annotation of
any kind. There used to be a stage here whose job was to reject the app's
own red scale bar from the results; it was removed when the burn-in was
traced and found to touch only snapshot copies inside the camera backend.
The rule it stood for is worth keeping in its place: the pipeline is fed
from the frame slot and from nothing that draws.

Design rules worth keeping:

- **Resolution independence.** The live preview runs on a downscaled copy
  of the frame for speed, but every geometric result — centroids, boxes,
  areas, thresholds in pixels — is expressed in FULL-frame units, so a
  parameter tuned on the preview means the same thing on a 4K snapshot.
  ``run(..., scale=s)`` does the conversion in one place.
- **Hue wraparound.** A red target's tolerance must not be one-sided; the
  hue band is split across the 0/179 seam (see ``colour_mask``).
- **No thickness, no ranking by material.** This identifies what the
  operator pointed at. It does not decide whether a flake is worth
  transferring — that is what the map markers and the table are for.

Qt-free and camera-free: usable from a worker thread, a CLI tool or a test.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

import cv2
import numpy as np

from talos.cv.flakes import flake_to_stage, merge_fragments
from talos.models import FlakeCandidate, ObjectiveCalibration, StagePosition

logger = logging.getLogger(__name__)

#: A connected region smaller than this many pixels is noise, at any scale.
#: Deliberately NOT scaled by the frame's resolution (unlike the lengths
#: below): sensor speckle is a per-pixel phenomenon, not a physical one, and
#: this is a noise floor rather than a size gate — the µm² gate is
#: ``SizeStage``, which is physical and resolution-independent already.
_MIN_AREA_PX2 = 4.0

_STAGE_TYPES: dict[str, type] = {}


def _register(cls):
    _STAGE_TYPES[cls.NAME] = cls
    return cls


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


#: The scalar types a stage parameter can have, for coercing a hand-edited
#: settings file. ``from __future__ import annotations`` makes the dataclass
#: field types strings, so the map is by name.
_TYPES = {"float": float, "int": int, "bool": _as_bool, "str": str}


def valid_hex(text: Any, fallback: str = "#c8a2c8") -> str:
    """Normalise a hex colour, falling back rather than raising — a settings
    file must never be able to stop the pipeline from running."""
    raw = str(text or "").strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6:
        return fallback
    try:
        int(raw, 16)
    except ValueError:
        return fallback
    return "#" + raw.lower()


def hex_to_rgb(text: str) -> tuple[int, int, int]:
    raw = valid_hex(text).lstrip("#")
    return (int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))


def sample_hex(img: np.ndarray, x_px: int, y_px: int,
               radius: int = 4) -> str | None:
    """The colour under a click, averaged over a small CIRCULAR patch.

    A circle rather than a square: a square's corners bias the average
    toward whatever is diagonally adjacent, which on a flake edge is the
    substrate. Returns ``#rrggbb``, or None when the point is off-frame.

    The frame handed in must be the one the operator is judging colour
    in — the pre-processed frame, which IS the raw frame when
    pre-processing is switched off. Sampling a composited display instead
    would pick up the darkening and the outlines rather than the sample.
    """
    stats = sample_hex_stats(img, x_px, y_px, radius)
    return stats[0] if stats is not None else None


def sample_hex_stats(img: np.ndarray, x_px: int, y_px: int,
                     radius: int = 4) -> tuple[str, float] | None:
    """``(hex, spread)`` for the same patch :func:`sample_hex` averages.

    The spread is the largest per-channel standard deviation in the patch —
    how much the 9-px disc disagrees with itself. On a clean flake it is a
    few DN of sensor noise; straddle a flake's edge and it is tens, because
    half the disc is substrate. That is the case worth telling the operator
    about: the mean of two materials is a colour NEITHER of them has, and it
    then becomes the mask's target and the curve's centre.
    """
    height, width = img.shape[:2]
    if not (0 <= x_px < width and 0 <= y_px < height):
        return None
    x0, x1 = max(0, x_px - radius), min(width, x_px + radius + 1)
    y0, y1 = max(0, y_px - radius), min(height, y_px + radius + 1)
    patch = img[y0:y1, x0:x1].astype(np.float32)
    if patch.size == 0:
        return None
    yy, xx = np.mgrid[y0 - y_px:y1 - y_px, x0 - x_px:x1 - x_px]
    mask = (xx * xx + yy * yy) <= radius * radius
    if not mask.any():
        mask = np.ones(patch.shape[:2], bool)
    inside = patch[mask]
    if inside.shape[0] == 0:
        return None
    mean = inside.mean(axis=0)
    spread = float(np.max(inside.std(axis=0))) if inside.shape[0] > 1 else 0.0
    hex_color = "#{:02x}{:02x}{:02x}".format(
        *(int(round(v)) for v in mean))
    return hex_color, spread


def hex_to_hsv(text: str) -> tuple[int, int, int]:
    """The colour in OpenCV's HSV units (H 0-179, S/V 0-255) — the same
    units the frame is converted into, so the band is exact."""
    r, g, b = hex_to_rgb(text)
    pixel = np.array([[[r, g, b]]], np.uint8)
    h, s, v = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0, 0]
    return int(h), int(s), int(v)


def colour_mask(img: np.ndarray, hex_color: str, tolerance: float,
                spread: float = 25.0, min_saturation: float = 40.0,
                min_value: float = 0.0) -> np.ndarray:
    """Binary mask of the pixels within ``tolerance`` of the target colour.

    Two numbers, because they answer two different questions and the bench
    proved they must be separable:

    - ``tolerance`` is how far the HUE may drift — "the same material under
      this illumination". Its half-width is ``tolerance × 0.9`` (so 100 =
      ±90 = half the wheel), and the band WRAPS across the 0/179 seam: a red
      target (hue ≈ 0 or 179) gets exactly the same tolerance as any other.
      The reference implementation clipped instead, which silently left red
      one-sided.
    - ``spread`` is how far the SHADE may drift — "the same thickness of
      it": the saturation and value half-widths, ``spread × 2.55`` each.

    They were one number until 2026-09-23, and for a blue-on-blue sample
    that was a trap: dark blue ``#1c3484`` is H113 S201 V132 and the lighter
    layer ``#aacdf5`` is H106 S78 V245 — 7° of hue apart and about 120 units
    of S and V. The hue band is irrelevant to that pair; what separates them
    is the shade window, so widening the tolerance to tolerate illumination
    also let the other layer in (it does so from ``tolerance ≈ 49``). With
    the split, hue can be opened without opening the shade.

    The window can never exclude the colour that was PICKED: the floors are
    clamped to the target's own S and V. A mask the pick is outside of reads
    as a broken detector and is silent — which is what a raised
    ``min_saturation`` used to do to the very flake it was raised from. The
    ceilings cannot do it (``sat + ds ≥ sat`` for any ``ds ≥ 0``), so only
    the floors need this, and the consequence is deliberate: for a pale pick
    ``min_saturation`` no longer applies to the pick itself.
    """
    hue, sat, val = hex_to_hsv(hex_color)
    tol = max(0.0, min(100.0, float(tolerance)))
    sp = max(0.0, min(100.0, float(spread)))
    dh = int(round(tol * 0.9))
    ds = dv = int(round(sp * 2.55))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    if dh >= 90:
        hue_ok = np.ones(img.shape[:2], bool)     # the whole wheel
    else:
        lo, hi = hue - dh, hue + dh
        if lo < 0:
            hue_ok = (h >= 180 + lo) | (h <= hi)
        elif hi > 179:
            hue_ok = (h <= hi - 180) | (h >= lo)
        else:
            hue_ok = (h >= lo) & (h <= hi)

    sat_lo = max(min(float(min_saturation), float(sat)), sat - ds)
    sat_hi = min(255.0, sat + ds)
    val_lo = max(min(float(min_value), float(val)), val - dv)
    val_hi = min(255.0, val + dv)
    keep = (hue_ok & (s >= sat_lo) & (s <= sat_hi)
            & (v >= val_lo) & (v <= val_hi))
    return (keep.astype(np.uint8) * 255)


# ----------------------------------------------------------------------
# The stages
# ----------------------------------------------------------------------

@dataclass
class Stage:
    """Base: a name, a UI label, and an on/off switch.

    ``KIND`` decides where the pipeline calls it:
    ``source`` → mask, ``mask`` → mask in/mask out, ``gate`` → candidate
    in/out, ``merge`` → candidates in/out.
    """

    enabled: bool = True

    NAME: ClassVar[str] = ""
    LABEL: ClassVar[str] = ""
    KIND: ClassVar[str] = "gate"
    #: (min, max, step) per parameter — the UI reads these to build its
    #: spin boxes, so a stage describes its own editor.
    RANGES: ClassVar[dict] = {}

    def params(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)
                if f.name != "enabled"}

    def to_dict(self) -> dict:
        return {"name": self.NAME, "enabled": bool(self.enabled), **self.params()}


@_register
@dataclass
class ColourStage(Stage):
    """Source: pixels within a tolerance of the picked colour."""

    hex_color: str = "#c8a2c8"
    #: Hue half-width (× 0.9) — "the same material under this illumination".
    tolerance: float = 25.0
    #: Saturation AND value half-widths (× 2.55) — "the same shade of it",
    #: which is where layer thickness shows up. See :func:`colour_mask` for
    #: why this is not the tolerance.
    spread: float = 25.0
    min_saturation: float = 40.0
    min_value: float = 0.0

    NAME = "colour"
    RANGES = {"tolerance": (0.0, 100.0, 5.0), "spread": (0.0, 100.0, 5.0),
              "min_saturation": (0, 255, 5), "min_value": (0, 255, 5)}
    LABEL = "Colour match"
    KIND = "source"

    def source_mask(self, ctx: "_Ctx") -> np.ndarray:
        return colour_mask(ctx.img, self.hex_color, self.tolerance,
                           self.spread, self.min_saturation, self.min_value)


@_register
@dataclass
class MorphologyStage(Stage):
    """Mask: open then close — removes speckle, joins broken edges."""

    kernel: int = 3

    NAME = "morphology"
    RANGES = {"kernel": (1, 15, 2)}
    LABEL = "Clean up"
    KIND = "mask"

    def apply_mask(self, mask: np.ndarray, ctx: "_Ctx") -> np.ndarray:
        size = max(1, int(self.kernel))
        if size <= 1 or not mask.any():
            return mask
        kernel = np.ones((size, size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


@_register
@dataclass
class SizeStage(Stage):
    """Gate: physical area in µm² — the units the operator thinks in."""

    min_area_um2: float = 30.0
    max_area_um2: float = 100000.0

    NAME = "size"
    RANGES = {"min_area_um2": (0.0, 100000.0, 10.0),
              "max_area_um2": (0.0, 10000000.0, 100.0)}
    LABEL = "Size"
    KIND = "gate"

    def keep(self, cand: FlakeCandidate, ctx: "_Ctx") -> bool:
        return self.min_area_um2 <= cand.area_um2 <= self.max_area_um2


@_register
@dataclass
class BorderStage(Stage):
    """Gate: reject blobs touching the field of view's edge — a sliver of a
    neighbour tile is not a sample."""

    margin_px: int = 4

    NAME = "border"
    RANGES = {"margin_px": (0, 100, 1)}
    LABEL = "Frame edge"
    KIND = "gate"

    def keep(self, cand: FlakeCandidate, ctx: "_Ctx") -> bool:
        margin = max(1, int(round(ctx.ref_px(self.margin_px))))
        x, y, w, h = cand.bbox
        return not (x <= margin or y <= margin
                    or x + w >= ctx.width - margin
                    or y + h >= ctx.height - margin)


@_register
@dataclass
class SharpnessStage(Stage):
    """Gate: mean boundary gradient. Crystals have sharp edges; tape
    residue, dust and defocused blobs are diffuse. The threshold is
    normalised by the preview scale AND by the frame's own resolution (a
    downscaled or coarser frame has smaller gradients), so it means the
    same thing in the preview, in a 1080p tile and in a 4K one.

    Raise it to reject diffuse blobs; the default is deliberately low,
    because a colour-matched contour traces the colour boundary and
    scores in the tens while a diffuse blob scores in single digits, so a
    threshold picked without measuring would reject most real samples.
    """

    min_edge_strength: float = 4.0

    NAME = "sharpness"
    RANGES = {"min_edge_strength": (0.0, 200.0, 1.0)}
    LABEL = "Sharpness"
    KIND = "gate"

    def keep(self, cand: FlakeCandidate, ctx: "_Ctx") -> bool:
        strength = ctx.edge_strength(cand)
        cand.score = strength          # the table's ranking number
        return strength >= ctx.ref_grad(self.min_edge_strength)


@_register
@dataclass
class MergeStage(Stage):
    """Merge: one sample often fragments into adjacent blobs."""

    gap_px: float = 25.0

    NAME = "merge"
    RANGES = {"gap_px": (0.0, 500.0, 5.0)}
    LABEL = "Merge fragments"
    KIND = "merge"

    def merge(self, candidates: list[FlakeCandidate],
              ctx: "_Ctx") -> list[FlakeCandidate]:
        return merge_fragments(candidates, ctx.ref_px(self.gap_px),
                               ctx.um2_per_px2)


#: The canonical order — how a config is written, read and shown.
CANONICAL_STAGES = (ColourStage, MorphologyStage, SizeStage, BorderStage,
                    SharpnessStage, MergeStage)


_warned_uncalibrated = False


def _warn_uncalibrated() -> None:
    """Say ONCE that sizes are pixels wearing a µm label.

    Every other consumer of the calibration treats a missing value as "no
    calibration": the scale bar simply does not draw, the field of view falls
    back to the pixel pitch. The pipeline has to pick a number, and picks
    1.0 µm/px — so ``area_um2 == area_px2`` and the 30 µm² size floor becomes
    a 30-PIXEL floor (four times too permissive at 0.5 µm/px), while the
    table reports the result as µm². Saying so once is the difference
    between a wrong number and a known one.
    """
    global _warned_uncalibrated
    if _warned_uncalibrated:
        return
    _warned_uncalibrated = True
    logger.warning(
        "identification: this objective has no µm/px calibration — areas and "
        "sizes are measured in PIXELS and labelled µm. Calibrate the "
        "objective (Preferences → Objectives) before trusting a size.")

#: Stage names that used to ship and no longer do. ``from_dict`` already
#: drops an unknown name, so these need no migration — they are listed so
#: the drop is a decision with a record rather than a silent absence:
#:
#: - ``contrast`` — Otsu on illumination-flattened contrast. It answers
#:   "what is here at all", never "which of these are the same material",
#:   and on the thin samples this is for, the threshold follows the bulk
#:   of the histogram rather than the object of interest. The useful half
#:   (the flattening) is now a pre-processing stage, where it prepares
#:   the frame instead of segmenting it.
#: - ``annotation`` — rejected saturated red blobs as the app's scale bar.
#:   The bar is drawn only into snapshot copies inside the camera backend,
#:   never into the live stream, the frame slot or a scan tile, so there
#:   was never anything for it to catch on this path.
_RETIRED_STAGES = {"contrast", "annotation"}


@dataclass
class IdentifyConfig:
    """The whole chain. Serialises to the settings file verbatim."""

    stages: list = field(default_factory=lambda: [
        cls() for cls in CANONICAL_STAGES
    ])

    def stage(self, name: str) -> Stage | None:
        return next((s for s in self.stages if s.NAME == name), None)

    @property
    def active(self) -> list:
        return [s for s in self.stages if s.enabled]

    def to_dict(self) -> dict:
        return {"stages": [s.to_dict() for s in self.stages]}

    @classmethod
    def from_dict(cls, data: Any) -> "IdentifyConfig":
        """Rebuild from a settings dict, keeping the canonical stage set and
        order: an unknown stage is dropped, a missing one comes back at its
        default, and a stage's own new parameters keep their defaults."""
        stored = {}
        for entry in (data or {}).get("stages") or []:
            if not isinstance(entry, dict) or "name" not in entry:
                continue
            name = str(entry["name"])
            stage_cls = _STAGE_TYPES.get(name)
            if stage_cls is None:
                if name not in _RETIRED_STAGES:
                    logger.info("identification: dropping unknown stage %r",
                                name)
                continue
            declared = {f.name: f.type for f in fields(stage_cls)}
            kwargs = {}
            for key, value in entry.items():
                if key not in declared:
                    continue
                coerce = _TYPES.get(str(declared[key]))
                try:
                    kwargs[key] = coerce(value) if coerce else value
                except (TypeError, ValueError):
                    pass        # a bad value falls back to the default
            if stage_cls is ColourStage:
                kwargs["hex_color"] = valid_hex(
                    kwargs.get("hex_color", ColourStage.hex_color))
                if "spread" not in kwargs:
                    # A file written before the shade window had a number of
                    # its own carried both meanings in ``tolerance`` (hue
                    # × 0.9 AND saturation/value × 2.55). Folding it into
                    # ``spread`` reproduces the stored mask EXACTLY — the
                    # same expression, the same rounding — because a colour
                    # someone tuned on the bench must not move because the
                    # controls were split. Logged, because it means the
                    # screen will show a second number that was never set.
                    kwargs["spread"] = kwargs.get("tolerance",
                                                  ColourStage.spread)
                    logger.info(
                        "identification: colour tolerance %r split into hue "
                        "tolerance and shade spread %r to keep the stored "
                        "mask identical", kwargs.get("tolerance"),
                        kwargs["spread"])
            try:
                stored[stage_cls.NAME] = stage_cls(**kwargs)
            except (TypeError, ValueError):
                stored[stage_cls.NAME] = stage_cls()
        return cls(stages=[stored.get(c.NAME, c()) for c in CANONICAL_STAGES])


@dataclass
class Region:
    """One blob the sources found, and whether it survived the chain.

    The processed view draws these: the operator needs to see not only what
    passed but what DIDN'T, because "why is this flake missing" is answered
    by the dim outline sitting on it. Rectangles are not drawn — a box round
    a shapeless blob says less than the blob's own outline.
    """

    contour: np.ndarray
    passed: bool = True


@dataclass
class IdentifyResult:
    candidates: list[FlakeCandidate] = field(default_factory=list)
    mask: np.ndarray | None = None            # full-frame size, uint8
    #: Every source region with its verdict (see Region).
    regions: list = field(default_factory=list)
    #: (label, in, out) per enabled stage, in pipeline order.
    counts: list = field(default_factory=list)
    scale: float = 1.0

    @property
    def summary(self) -> str:
        """``colour 812 → size 12 → sharpness 2`` — the chain's readout."""
        return " → ".join(f"{label} {out}" for label, _in, out in self.counts)


class _Ctx:
    """Everything a stage needs, computed at most once per run."""

    def __init__(self, img: np.ndarray, um_per_px: tuple[float, float],
                 scale: float, frame_scale: float = 1.0):
        self.img = img                       # the (possibly scaled) frame
        self.height, self.width = img.shape[:2]
        self.scale = float(scale) or 1.0
        # µm per pixel OF THIS FRAME (a downscaled frame covers more µm/px)
        self.um_per_px = (um_per_px[0], um_per_px[1])
        self.um2_per_px2 = um_per_px[0] * um_per_px[1]
        #: How finely this frame samples compared with the frame the
        #: operator tunes on (see :meth:`ref_px`). 1.0 unless a caller says
        #: otherwise: the live preview IS the frame the parameters were
        #: judged on, and only another resolution of the same field of view
        #: needs the ratio.
        self.frame_scale = float(frame_scale) or 1.0
        self._grad = None
        self._contours: dict[int, Any] = {}
        self._regions: list = []          # (candidate id, contour)
        self._failed: set[int] = set()

    @property
    def grad(self) -> np.ndarray:
        if self._grad is None:
            gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            self._grad = np.sqrt(
                cv2.Sobel(gray, cv2.CV_64F, 1, 0) ** 2
                + cv2.Sobel(gray, cv2.CV_64F, 0, 1) ** 2)
        return self._grad

    def attach_contour(self, cand: FlakeCandidate, contour) -> None:
        """Keyed by id AND holding the candidate: a dropped candidate is
        only referenced from here, so without the reference its id could be
        recycled by the next object allocated — and a later lookup would
        hand back somebody else's contour."""
        self._contours[id(cand)] = (cand, contour)

    def contour_for(self, cand: FlakeCandidate):
        entry = self._contours.get(id(cand))
        return entry[1] if entry is not None else None

    def attach_region(self, cand: FlakeCandidate, contour) -> None:
        """Record the blob and the candidate that stands for it, so a gate
        that drops the candidate can mark the blob as failed — the
        processed view draws both verdicts."""
        self._regions.append((id(cand), contour))

    def mark_failed(self, cand: FlakeCandidate) -> None:
        self._failed.add(id(cand))

    # --- pixel-unit parameters, in the frame actually in hand ------------

    def ref_px(self, value: float) -> float:
        """A LENGTH the operator set, in work-frame pixels.

        The parameter is expressed in pixels of the frame they judged it on
        (the live preview), which is not necessarily the frame in hand: a
        scan can capture at another resolution of the same field of view
        (Preferences → Scan). Scaling by ``frame_scale`` keeps the number
        meaning the same DISTANCE on the sample — 4 px of a 1080p frame and
        8 px of the 4K one are both the same µm. With ``frame_scale`` at
        its default this is exactly ``value * scale``, which is what the
        code did before there was a second resolution at all.
        """
        return float(value) * self.scale * self.frame_scale

    def ref_grad(self, value: float) -> float:
        """A per-pixel GRADIENT threshold, in work-frame-pixel units.

        Gradient magnitude is per pixel, so this one scales INVERSELY to the
        sampling: the same physical edge spread over twice the pixels has
        half the per-pixel gradient. ``ref_px``'s inverse, for the same
        reason.
        """
        return float(value) * self.scale / self.frame_scale

    def regions(self) -> list:
        return [Region(contour=contour, passed=key not in self._failed)
                for key, contour in self._regions]

    def edge_strength(self, cand: FlakeCandidate) -> float:
        """Mean gradient magnitude on the candidate's own boundary.

        Drawn on a bbox-sized canvas rather than the full frame: zeroing a
        1080p buffer per candidate cost 2 MB of allocation each."""
        contour = self.contour_for(cand)
        if contour is None:
            return 0.0
        x, y, w, h = cand.bbox
        pad = 2
        rx, ry = max(0, x - pad), max(0, y - pad)
        rw = min(self.width - rx, w + 2 * pad)
        rh = min(self.height - ry, h + 2 * pad)
        if rw <= 0 or rh <= 0:
            return 0.0
        ring = np.zeros((rh, rw), np.uint8)
        cv2.drawContours(ring, [contour - (rx, ry)], -1, 255, 2)
        values = self.grad[ry:ry + rh, rx:rx + rw][ring > 0]
        return float(values.mean()) if values.size else 0.0


def _component_count(mask: np.ndarray) -> int:
    count, _labels = cv2.connectedComponents(mask, connectivity=8)
    return max(0, count - 1)           # label 0 is the background


def _candidates_from_mask(mask: np.ndarray, ctx: _Ctx) -> list[FlakeCandidate]:
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
    out: list[FlakeCandidate] = []
    for contour in contours:
        area_px2 = float(cv2.contourArea(contour))
        if area_px2 < _MIN_AREA_PX2:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cand = FlakeCandidate(
            x_px=moments["m10"] / moments["m00"],
            y_px=moments["m01"] / moments["m00"],
            area_px2=area_px2,
            area_um2=area_px2 * ctx.um2_per_px2,   # physical: scale-free
            bbox=tuple(int(v) for v in cv2.boundingRect(contour)))
        ctx.attach_contour(cand, contour)
        ctx.attach_region(cand, contour)
        out.append(cand)
    return out


class IdentifyPipeline:
    """Runs a :class:`IdentifyConfig` over a frame."""

    def __init__(self, config: IdentifyConfig | None = None):
        self.config = config or IdentifyConfig()

    def run(self, img: np.ndarray,
            calib: ObjectiveCalibration | None = None,
            config: IdentifyConfig | None = None,
            stage_pos: StagePosition | None = None,
            scale: float = 1.0,
            flip: bool = False,
            frame_scale: float = 1.0) -> IdentifyResult:
        """Identify samples in ``img`` (RGB uint8).

        ``scale`` < 1 processes a downscaled copy (the live preview) while
        every result comes back in FULL-frame pixels and µm. ``stage_pos``
        (the position the frame was taken at) additionally fills each
        candidate's ``x_um``/``y_um`` through the px→stage mapping, which
        needs ``flip`` (the camera flip — it decides which way the image
        axes point relative to the stage).

        ``frame_scale`` says how finely THIS frame samples the field of
        view compared with the frame the pixel-unit parameters were tuned
        on — supplied by a caller that captures at more than one resolution
        (the scan), and 1.0 for everything else. See ``_Ctx.ref_px``.
        """
        config = config or self.config
        scale = float(scale)
        if not 0.0 < scale <= 1.0:
            scale = 1.0

        work = img
        if scale < 1.0:
            work = cv2.resize(img, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)

        um_x = float(getattr(calib, "um_per_px_x", None) or 1.0) / scale
        um_y = float(getattr(calib, "um_per_px_y", None) or 1.0) / scale
        if calib is not None and not (getattr(calib, "um_per_px_x", None)
                                      and getattr(calib, "um_per_px_y", None)):
            _warn_uncalibrated()
        ctx = _Ctx(work, (um_x, um_y), scale, frame_scale)

        counts: list = []
        mask = np.zeros(work.shape[:2], np.uint8)
        candidates: list[FlakeCandidate] | None = None
        for stage in config.stages:
            if not stage.enabled:
                continue
            if stage.KIND == "source":
                produced = stage.source_mask(ctx)
                if produced is None:
                    continue
                n = _component_count(produced)
                counts.append((stage.LABEL, n, n))
                mask = cv2.bitwise_or(mask, produced)
            elif stage.KIND == "mask":
                before = _component_count(mask)
                mask = stage.apply_mask(mask, ctx)
                counts.append((stage.LABEL, before, _component_count(mask)))
            else:
                if candidates is None:
                    candidates = _candidates_from_mask(mask, ctx)
                before = len(candidates)
                if stage.KIND == "merge":
                    candidates = stage.merge(candidates, ctx)
                else:
                    kept = [c for c in candidates if stage.keep(c, ctx)]
                    for dropped in candidates:
                        if dropped not in kept:
                            ctx.mark_failed(dropped)
                    candidates = kept
                counts.append((stage.LABEL, before, len(candidates)))

        if candidates is None:              # no source enabled at all
            candidates = []

        # A merge runs AFTER the gates (the canonical order), and merging
        # SUMS areas: two fragments that each pass the size floor and cap can
        # produce one candidate OVER the cap, silently — the readout still
        # says "Size 2 → Merge 1". So the area gate is re-applied to the
        # merged set, and only that one:
        #
        # - the sharpness gate cannot be re-run, because the merge consumed
        #   the contour it measures (a merged candidate has none), and its
        #   merged score is the best of its fragments by construction;
        # - the frame-edge gate cannot be violated by a union: boxes that each
        #   satisfy `margin < x` and `x + w < width - margin` satisfy the
        #   union's bounds too.
        if any(s.KIND == "merge" for s in config.stages if s.enabled):
            for stage in config.stages:
                if not stage.enabled or stage.NAME != "size":
                    continue
                kept = [c for c in candidates if stage.keep(c, ctx)]
                if len(kept) != len(candidates):
                    candidates = kept

        regions = ctx.regions()
        _to_full_frame(candidates, 1.0 / scale)
        _regions_to_full_frame(regions, 1.0 / scale)
        if mask is not None and mask.shape[:2] != img.shape[:2]:
            # The mask is handed to whoever draws it, and they draw over
            # the CALLER's frame — a mask left at preview resolution is not
            # just mis-sized, it silently disables whatever tests it
            # (the darkening did exactly that for every preview-scaled run).
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        if stage_pos is not None and calib is not None:
            for cand in candidates:
                cand.x_um, cand.y_um = flake_to_stage(
                    cand.x_px, cand.y_px, img.shape, calib, stage_pos, flip)
        candidates.sort(key=lambda c: c.area_um2, reverse=True)

        return IdentifyResult(candidates=candidates, mask=mask,
                              regions=regions, counts=counts, scale=scale)


def _regions_to_full_frame(regions: list, inv: float) -> None:
    """The region contours are in the processed frame too (they are drawn
    over the caller's frame), so they take the same trip back."""
    if inv == 1.0:
        return
    for region in regions:
        region.contour = (region.contour * inv).astype(np.int32)


def _to_full_frame(candidates: list[FlakeCandidate], inv: float) -> None:
    """Undo the preview downscale: the caller's frame is the full one, so
    every pixel quantity must be expressed in ITS pixels. ``area_um2`` is
    already physical and is left alone."""
    if inv == 1.0:
        return
    for cand in candidates:
        cand.x_px *= inv
        cand.y_px *= inv
        cand.area_px2 *= inv * inv
        x, y, w, h = cand.bbox
        cand.bbox = (int(round(x * inv)), int(round(y * inv)),
                     int(round(w * inv)), int(round(h * inv)))


def identify(img: np.ndarray, calib: ObjectiveCalibration | None = None,
             config: IdentifyConfig | None = None,
             stage_pos: StagePosition | None = None,
             scale: float = 1.0, flip: bool = False) -> IdentifyResult:
    """Convenience wrapper (one-shot pipeline)."""
    return IdentifyPipeline(config).run(img, calib, config=config,
                                        stage_pos=stage_pos, scale=scale,
                                        flip=flip)


#: The processed view's two verdicts. The bright outline is what survived
#: the chain; the dim one is what a gate threw away. Both are drawn on the
#: DARKENED part of the frame, so "dimmer" is still clearly visible.
PASS_COLOUR = (0, 255, 160)
FAIL_COLOUR = (0, 130, 95)


def render_overlay(img: np.ndarray, result: IdentifyResult,
                   darken: float = 0.75,
                   pass_colour: tuple = PASS_COLOUR,
                   fail_colour: tuple = FAIL_COLOUR) -> np.ndarray:
    """The processed view: everything except the matched regions darkened,
    every region outlined by its verdict.

    Deliberately NOT rectangles. A box round a shapeless blob says little
    and, worse, boxes from neighbouring regions overlap and read as one
    object; the blob's own outline is the shape the operator is judging,
    and its brightness carries the verdict. The darkened background is what
    makes that readable at a glance — and it keeps the sample's own pixels
    visible inside the match, so the operator still sees the material, not
    a mask poster.
    """
    out = img.copy()
    mask = result.mask
    if mask is not None and mask.shape[:2] == out.shape[:2] and mask.any():
        # One multiply over the whole frame rather than a fancy-index copy,
        # a float upcast and a clip of the OUTSIDE pixels only: at 4K that
        # path allocated a few hundred megabytes per frame (three full-size
        # temporaries), on the thread that also runs the scan's tiles.
        factor = np.where(mask[:, :, None] == 0, 1.0 - float(darken), 1.0)
        out = (out.astype(np.float32) * factor).astype(np.uint8)
    for region in result.regions:
        contour = region.contour
        if contour is None or not len(contour):
            continue
        colour = pass_colour if region.passed else fail_colour
        cv2.drawContours(out, [contour], -1, colour, 2)
    return out


__all__ = ["CANONICAL_STAGES", "ColourStage", "IdentifyConfig",
           "Region",
           "IdentifyPipeline", "IdentifyResult", "MorphologyStage",
           "BorderStage", "MergeStage", "SharpnessStage",
           "SizeStage", "Stage", "colour_mask", "hex_to_hsv", "hex_to_rgb",
           "identify", "render_overlay", "sample_hex", "valid_hex"]
