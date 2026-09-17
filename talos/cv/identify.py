"""Sample identification as a chain of stages.

The operator builds a filter chain rather than picking a mode: two SOURCE
stages turn the frame into a mask (colour match, contrast), one cleans the
mask up, then gates decide which blobs survive (size, border, sharpness,
annotation) and a final stage merges fragments of the same object. Every
stage can be switched off, and each reports how many candidates it let
through — so the panel reads like the chain it is: ``colour 812 → size 12 →
sharpness 2``.

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

from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

import cv2
import numpy as np

from talos.cv.flakes import (flake_to_stage, flatten_contrast,
                             is_red_annotation, merge_fragments)
from talos.models import FlakeCandidate, ObjectiveCalibration, StagePosition

#: A connected region smaller than this many pixels is noise, at any scale.
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


def hex_to_hsv(text: str) -> tuple[int, int, int]:
    """The colour in OpenCV's HSV units (H 0-179, S/V 0-255) — the same
    units the frame is converted into, so the band is exact."""
    r, g, b = hex_to_rgb(text)
    pixel = np.array([[[r, g, b]]], np.uint8)
    h, s, v = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0, 0]
    return int(h), int(s), int(v)


def colour_mask(img: np.ndarray, hex_color: str, tolerance: float,
                min_saturation: float = 40.0, min_value: float = 0.0
                ) -> np.ndarray:
    """Binary mask of the pixels within ``tolerance`` of the target colour.

    The hue half-width is ``tolerance × 0.9`` (so 100 = ±90 = half the hue
    wheel) and the saturation/value half-widths are ``tolerance × 2.55``.
    The hue band WRAPS across the 0/179 seam: a red target (hue ≈ 0 or
    179) gets exactly the same tolerance as any other — the reference
    implementation clipped instead, which silently left red one-sided.
    """
    hue, sat, val = hex_to_hsv(hex_color)
    tol = max(0.0, min(100.0, float(tolerance)))
    dh = int(round(tol * 0.9))
    ds = int(round(tol * 2.55))
    dv = ds
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

    sat_lo = max(float(min_saturation), sat - ds)
    sat_hi = min(255.0, sat + ds)
    val_lo = max(float(min_value), val - dv)
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
    tolerance: float = 25.0
    min_saturation: float = 40.0
    min_value: float = 0.0

    NAME = "colour"
    RANGES = {"tolerance": (0.0, 100.0, 5.0), "min_saturation": (0, 255, 5),
              "min_value": (0, 255, 5)}
    LABEL = "Colour match"
    KIND = "source"

    def source_mask(self, ctx: "_Ctx") -> np.ndarray:
        return colour_mask(ctx.img, self.hex_color, self.tolerance,
                           self.min_saturation, self.min_value)


@_register
@dataclass
class ContrastStage(Stage):
    """Source: illumination-flattened contrast, Otsu-thresholded."""

    blur_sigma: float = 15.0

    NAME = "contrast"
    RANGES = {"blur_sigma": (1.0, 60.0, 1.0)}
    LABEL = "Contrast"
    KIND = "source"

    def source_mask(self, ctx: "_Ctx") -> np.ndarray:
        return flatten_contrast(ctx.img, self.blur_sigma)[1]


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
        margin = max(1, int(round(self.margin_px * ctx.scale)))
        x, y, w, h = cand.bbox
        return not (x <= margin or y <= margin
                    or x + w >= ctx.width - margin
                    or y + h >= ctx.height - margin)


@_register
@dataclass
class SharpnessStage(Stage):
    """Gate: mean boundary gradient. Crystals have sharp edges; tape
    residue, dust and defocused blobs are diffuse. The threshold is
    normalised by the preview scale (a downscaled frame has smaller
    gradients), so it means the same thing in the preview and in a
    full-resolution tile.

    The default is deliberately LOW. A colour-matched contour traces the
    colour boundary exactly (scores in the tens), but a contrast-matched
    one wanders through the noise around the object, so its mean boundary
    gradient is only a few counts — the first version of this gate, tuned
    on colour matches, silently rejected every contrast-source candidate.
    Raise it to reject diffuse blobs.
    """

    min_edge_strength: float = 4.0

    NAME = "sharpness"
    RANGES = {"min_edge_strength": (0.0, 200.0, 1.0)}
    LABEL = "Sharpness"
    KIND = "gate"

    def keep(self, cand: FlakeCandidate, ctx: "_Ctx") -> bool:
        strength = ctx.edge_strength(cand)
        cand.score = strength          # the table's ranking number
        return strength >= self.min_edge_strength * ctx.scale


@_register
@dataclass
class AnnotationStage(Stage):
    """Gate: saturated pure-red overlays are the scale bar, not a sample."""

    sat_min: float = 120.0

    NAME = "annotation"
    RANGES = {"sat_min": (0, 255, 5)}
    LABEL = "Scale bar"
    KIND = "gate"

    def keep(self, cand: FlakeCandidate, ctx: "_Ctx") -> bool:
        contour = ctx.contour_for(cand)
        if contour is None:
            return True
        return not is_red_annotation(ctx.hsv, contour, self.sat_min)


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
        return merge_fragments(candidates, self.gap_px * ctx.scale,
                               ctx.um2_per_px2)


#: The canonical order — how a config is written, read and shown.
CANONICAL_STAGES = (ColourStage, ContrastStage, MorphologyStage, SizeStage,
                    BorderStage, SharpnessStage, AnnotationStage, MergeStage)


@dataclass
class IdentifyConfig:
    """The whole chain. Serialises to the settings file verbatim."""

    stages: list = field(default_factory=lambda: [
        # contrast is a second opinion, off until the operator wants it:
        # the simple path is "point at the colour".
        cls(enabled=not (cls is ContrastStage)) for cls in CANONICAL_STAGES
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
            stage_cls = _STAGE_TYPES.get(str(entry["name"]))
            if stage_cls is None:
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
            try:
                stored[stage_cls.NAME] = stage_cls(**kwargs)
            except (TypeError, ValueError):
                stored[stage_cls.NAME] = stage_cls()
        return cls(stages=[stored.get(c.NAME, c()) for c in CANONICAL_STAGES])


@dataclass
class IdentifyResult:
    candidates: list[FlakeCandidate] = field(default_factory=list)
    mask: np.ndarray | None = None            # full-frame size, uint8
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
                 scale: float):
        self.img = img                       # the (possibly scaled) frame
        self.height, self.width = img.shape[:2]
        self.scale = scale
        # µm per pixel OF THIS FRAME (a downscaled frame covers more µm/px)
        self.um_per_px = (um_per_px[0], um_per_px[1])
        self.um2_per_px2 = um_per_px[0] * um_per_px[1]
        self._hsv = None
        self._grad = None
        self._contours: dict[int, Any] = {}

    @property
    def hsv(self) -> np.ndarray:
        if self._hsv is None:
            self._hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        return self._hsv

    @property
    def grad(self) -> np.ndarray:
        if self._grad is None:
            gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            self._grad = np.sqrt(
                cv2.Sobel(gray, cv2.CV_64F, 1, 0) ** 2
                + cv2.Sobel(gray, cv2.CV_64F, 0, 1) ** 2)
        return self._grad

    def attach_contour(self, cand: FlakeCandidate, contour) -> None:
        self._contours[id(cand)] = contour

    def contour_for(self, cand: FlakeCandidate):
        return self._contours.get(id(cand))

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
            scale: float = 1.0) -> IdentifyResult:
        """Identify samples in ``img`` (RGB uint8).

        ``scale`` < 1 processes a downscaled copy (the live preview) while
        every result comes back in FULL-frame pixels and µm. ``stage_pos``
        (the position the frame was taken at) additionally fills each
        candidate's ``x_um``/``y_um`` through the px→stage mapping.
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
        ctx = _Ctx(work, (um_x, um_y), scale)

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
                    candidates = [c for c in candidates if stage.keep(c, ctx)]
                counts.append((stage.LABEL, before, len(candidates)))

        if candidates is None:              # no source enabled at all
            candidates = []

        _to_full_frame(candidates, 1.0 / scale)
        if stage_pos is not None and calib is not None:
            for cand in candidates:
                cand.x_um, cand.y_um = flake_to_stage(
                    cand.x_px, cand.y_px, img.shape, calib, stage_pos)
        candidates.sort(key=lambda c: c.area_um2, reverse=True)

        return IdentifyResult(candidates=candidates, mask=mask,
                              counts=counts, scale=scale)


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
             scale: float = 1.0) -> IdentifyResult:
    """Convenience wrapper (one-shot pipeline)."""
    return IdentifyPipeline(config).run(img, calib, config=config,
                                        stage_pos=stage_pos, scale=scale)


def render_overlay(img: np.ndarray, result: IdentifyResult,
                   alpha: float = 0.45,
                   colour: tuple[int, int, int] = (0, 200, 255)) -> np.ndarray:
    """The processed view: matched regions tinted, SURVIVORS outlined.

    The tint is what the sources matched; the boxes are what made it
    through every gate. Watching one change as the other is edited is the
    whole point of the processed view.
    """
    out = img.copy()
    mask = result.mask
    if mask is not None and mask.shape[:2] == out.shape[:2] and mask.any():
        selected = mask > 0
        tint = np.array(colour, np.float32)
        blended = (out[selected].astype(np.float32) * (1.0 - alpha)
                   + tint * alpha)
        out[selected] = np.clip(blended, 0, 255).astype(np.uint8)
    for cand in result.candidates:
        x, y, w, h = cand.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, 1)
    return out


__all__ = ["CANONICAL_STAGES", "ColourStage", "ContrastStage", "IdentifyConfig",
           "IdentifyPipeline", "IdentifyResult", "MorphologyStage",
           "AnnotationStage", "BorderStage", "MergeStage", "SharpnessStage",
           "SizeStage", "Stage", "colour_mask", "hex_to_hsv", "hex_to_rgb",
           "identify", "render_overlay", "valid_hex"]
