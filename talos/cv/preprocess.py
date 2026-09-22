"""Pre-processing: the layer between the camera and everything that looks
at the frame.

Two consumers, one implementation. The identification chain and the
processed views both work on the *pre-processed* frame, so what the
operator tunes against is exactly what the pipeline segments — a colour
picked off the screen is a colour the mask will look for, by construction
rather than by coincidence.

**Pre-processing is a pure transform.** It never draws into the frame it
is given, and it is never written to disk: snapshots, scan tiles and the
mosaic are all the raw capture. That is the whole overlay rule — a scale
bar, a timestamp or a crosshair must not be able to enter the pipeline,
so nothing that could draw is allowed in here.

**Two stages, and both of them earn their place.** ``denoise -> local
contrast``, in that order, for the reason the curve exists at all: it
multiplies small differences around the picked colour, and it would
multiply the sensor noise just as eagerly. Smoothing first is what makes
the curve multiply signal.

What is NOT here is as deliberate. Tone operations (exposure, brightness,
contrast, gamma) were removed in 2026-09-23: the camera's own exposure,
gain and white balance already set the frame, and a second set of them
behind the operator's back made every bench session a question about
which layer was being tuned. Illumination flattening went the same way —
this bench's lighting is even enough that the correction had nothing to
correct — with a note that vignette correction is the version of it worth
having if the field ever needs one.

The curve is a 256-entry lookup table per channel and one :func:`cv2.LUT`
call, which is why it is cheap enough to run on every previewed frame.
Denoise is not, which is why :func:`apply` runs on the detection worker
and never on the GUI thread (see ``ui/detect_engine.py``): a preview that
lags is recoverable, a camera capture sequence that waits on a bilateral
filter is not.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: The value range the point curves are defined on, and the LUT length.
#: The curves are specified in 8-bit DN regardless of the frame's own
#: depth, because the operator picks colours in DN and types slider
#: numbers in DN.
DN_MAX = 255
LUT_SIZE = 256

#: Bounds for every tunable, shared by ``from_dict`` and the panel so a
#: hand-edited settings file and a slider cannot disagree about the range.
RANGES = {
    "denoise_diameter": (1.0, 25.0),
    "denoise_sigma_color": (0.0, 150.0),
    "denoise_sigma_space": (1.0, 25.0),
    "gain": (1.0, 8.0),
    "width": (2.0, 120.0),
}


#: The longest run of input levels that may be collapsed onto a single
#: output level. The curve trades range outside the band for range
#: inside it, so some collapse is the point — but a long shelf destroys
#: every difference inside it, and this is the number that keeps a shelf
#: from forming where the sample is (see ``_fit_width``).
MAX_FLAT_RUN = 4


# ----------------------------------------------------------------------
# The local-contrast curve
# ----------------------------------------------------------------------

def _hann_cdf(u: np.ndarray) -> np.ndarray:
    """∫ of the raised-cosine kernel, from −∞ to ``u``.

    Zero at or below −1, one at or above +1, and smooth in between, so a
    band that does not reach an end of the range contributes nothing at
    that end.
    """
    inside = np.clip(np.asarray(u, dtype=np.float64), -1.0, 1.0)
    return 0.5 * (inside + np.sin(np.pi * inside) / np.pi) + 0.5


def _band_width(centre: float, width: float) -> float:
    """The usable half-width: never wider than the room the band has.

    A band that runs off the end of the range is not a wider band, it is
    a shorter one that has been clipped — and clipping it is what makes
    the boost lopsided. Keeping it inside the range is also what keeps
    the curve monotone (see :func:`_amplitude`).
    """
    return max(1.0, min(float(width), 0.9 * centre, 0.9 * (DN_MAX - centre)))


def _reference_nodes(centre: float, width: float) -> list[tuple[float, float]]:
    """The corners the boosted band has to lift away from.

    The curve is built as ``f(v) = v + amp·B(v)`` where ``B`` is the
    accumulating boost minus the straight lines that join **(0, 0)**,
    **(centre, centre)** and **(255, 255)**. Subtracting those lines is
    what makes all three *pinned*: the operator's colour renders exactly
    where they picked it, which is not a nicety — the same hex drives the
    colour mask, so a curve that moved the picked value would leave the
    mask searching for a colour the frame no longer contains.
    """
    p0 = float(_hann_cdf(np.array(-centre / width)))
    p2 = float(_hann_cdf(np.array((DN_MAX - centre) / width)))
    nodes = [(0.0, p0)]
    if 0.0 < centre < DN_MAX:
        nodes.append((centre, 0.5))
    nodes.append((DN_MAX, p2))
    return nodes


def _slopes_at(nodes, centre: float) -> list[float]:
    """The reference's one-sided slopes at the centre pin."""
    out: list[float] = []
    for (x0, y0), (x1, y1) in zip(nodes, nodes[1:]):
        slope = (y1 - y0) / (x1 - x0)
        if x0 < centre <= x1 or x0 <= centre < x1:
            out.append(slope)
    return out


def _amplitude(centre: float, gain: float, width: float,
               slopes: list[float]) -> float:
    """The boost amplitude that delivers ``gain`` at the picked colour.

    The slope at the centre is ``1 + amp·(1/width − reference_slope)``,
    once per side of the pin. Solving for the amplitude so the *average*
    of those comes out at ``gain`` is what makes the number on the slider
    the number in the image: for a colour away from the ends of the range
    both sides land on it exactly, and near an end the two sides bracket
    it instead of one of them silently eating the boost.

    A lopsided band can be boosted only so hard before the curve stops
    rising somewhere else, so the amplitude is capped by what keeps the
    slope non-negative — the delivered gain is reported by
    :func:`effective_gain` rather than assumed.
    """
    terms = [1.0 / width - slope for slope in slopes]
    total = sum(terms)
    if total <= 1e-12:
        return 0.0
    amp = (gain - 1.0) * len(terms) / total
    worst = _worst_boost_slope(centre, width, slopes)
    if worst < -1e-12:
        amp = min(amp, -1.0 / worst)
    return max(0.0, amp)


def _worst_boost_slope(centre: float, width: float,
                       slopes: list[float]) -> float:
    """``min B'`` over the range — how far the boost pulls the curve down.

    ``B`` rises inside the band and falls outside it, and the falling part
    is what could push the total slope below zero. Measured on a grid
    rather than bounded analytically: the grid is 1024 points on a curve
    that is resolved to 256, so it cannot miss the minimum that matters.
    """
    grid = np.linspace(0.0, DN_MAX, 1024)
    kernel = _hann_kernel((grid - centre) / width) / width
    left = slopes[0] if slopes else 0.0
    right = slopes[-1] if slopes else 0.0
    ref_slope = np.where(grid < centre, left, right)
    return float(np.min(kernel - ref_slope))


def _hann_kernel(u: np.ndarray) -> np.ndarray:
    inside = np.clip(np.asarray(u, dtype=np.float64), -1.0, 1.0)
    return np.where(np.abs(u) < 1.0, 0.5 * (1.0 + np.cos(np.pi * inside)),
                    0.0)


def _raw_curve(centre: float, gain: float, width: float,
               v: np.ndarray) -> np.ndarray:
    """The curve at a given width — no width fitting, no recursion."""
    v = np.asarray(v, dtype=np.float64)
    width = _band_width(centre, width)
    nodes = _reference_nodes(centre, width)
    slopes = _slopes_at(nodes, centre)
    amp = _amplitude(centre, gain, width, slopes)
    if amp <= 1e-12:
        return v
    boost = _hann_cdf((v - centre) / width) - np.interp(
        v, [x for x, _ in nodes], [y for _, y in nodes])
    return np.clip(v + amp * boost, 0.0, DN_MAX)


def _segment_lengths(curve: np.ndarray) -> list[int]:
    """How many input levels each output level swallows, worst first."""
    levels = np.round(curve).astype(np.int64)
    runs: list[int] = []
    start = 0
    for index in range(1, len(levels) + 1):
        if index == len(levels) or levels[index] != levels[start]:
            runs.append(index - start)
            start = index
    return sorted(runs, reverse=True)


@lru_cache(maxsize=512)
def _fit_width(centre: float, gain: float, width: float) -> float:
    """The widest band that does not need a dead shelf to pay for itself.

    The endpoints and the pick are all pinned, so the total slope over
    the range is fixed at 255 no matter what the parameters are. A wide
    band boosted hard spends that budget inside the band and has to
    charge it back elsewhere — at gain 8 over ±32 DN the compensation is
    a shelf where *a hundred* input levels share one output level, and it
    can sit twenty DN from the colour being examined.

    Rather than model that, it is measured: build the curve, count the
    longest run of levels that collapse together, and narrow the band
    until the answer is a handful. Width is the parameter that gives way
    — the gain is what the operator asked for in the first place, and a
    narrower band is also the conservative direction.
    """
    limit = float(RANGES["width"][0])
    top = _band_width(centre, width)
    if _segment_lengths(_raw_curve(centre, gain, top,
                                   np.linspace(0.0, DN_MAX, LUT_SIZE)))[0] \
            <= MAX_FLAT_RUN:
        return top
    low = limit if limit < top else 1.0
    for _ in range(24):
        mid = 0.5 * (low + top)
        worst = _segment_lengths(_raw_curve(
            centre, gain, mid, np.linspace(0.0, DN_MAX, LUT_SIZE)))[0]
        if worst <= MAX_FLAT_RUN:
            low = mid
        else:
            top = mid
    return low


def curve_values(centre: float, gain: float, width: float,
                 v: np.ndarray) -> np.ndarray:
    """The local-contrast curve, in float DN, evaluated at ``v``.

    Steepest at the picked colour, pinned at 0, at the pick and at 255,
    and never folding back on itself. Monotonicity is the property that
    matters most and is worth stating twice: a channel that folds back
    turns a smooth gradient into a false edge, which is structure the
    sample does not have, and nothing about looking at the image would
    tell the operator it was not real.

    The band may be narrower than requested — see :func:`_fit_width` and
    :func:`effective_width`.
    """
    gain = max(1.0, float(gain))
    centre = float(np.clip(centre, 0.0, DN_MAX))
    if gain <= 1.0 + 1e-9 or not 0.0 < centre < DN_MAX:
        return np.asarray(v, dtype=np.float64)
    return _raw_curve(centre, gain, _fit_width(centre, gain, width), v)


def effective_width(centre: float, gain: float, width: float) -> float:
    """The band the curve actually uses, after :func:`_fit_width`."""
    gain = max(1.0, float(gain))
    centre = float(np.clip(centre, 0.0, DN_MAX))
    if gain <= 1.0 + 1e-9 or not 0.0 < centre < DN_MAX:
        return _band_width(centre, width)
    return _fit_width(centre, gain, width)


def effective_gain(centre: float, gain: float, width: float) -> float:
    """The slope the curve actually delivers at the pick.

    Equal to ``gain`` except when even the narrowest band cannot carry it
    — a colour within a few DN of black or white, where the boost has no
    room. The panel shows this number rather than the slider's, so a
    curve that could not deliver is visible rather than discovered later
    from an image that would not line up.
    """
    gain = max(1.0, float(gain))
    centre = float(np.clip(centre, 0.0, DN_MAX))
    if gain <= 1.0 + 1e-9 or not 0.0 < centre < DN_MAX:
        return 1.0
    width = _fit_width(centre, gain, width)
    slopes = _slopes_at(_reference_nodes(centre, width), centre)
    if not slopes:
        return 1.0
    amp = _amplitude(centre, gain, width, slopes)
    terms = [1.0 / width - slope for slope in slopes]
    return 1.0 + amp * sum(terms) / len(terms)


def channel_curve(centre: float, gain: float, width: float,
                  n: int = LUT_SIZE) -> np.ndarray:
    """One channel's curve as an ``n``-entry uint8 lookup table."""
    v = np.linspace(0.0, DN_MAX, n)
    out = np.clip(np.round(curve_values(centre, gain, width, v)), 0, DN_MAX)
    # Rounding a monotone function cannot make it decrease, so this is
    # belt-and-braces — but it makes the guarantee unconditional rather
    # than an argument the next reader has to re-make.
    return np.maximum.accumulate(out).astype(np.uint8)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

@dataclass
class LocalContrast:
    """The matched gain: steep at the picked colour, flat everywhere else.

    ``gain`` is the slope at the picked colour (1.0 = off), ``width`` how
    many DN either side of it stay boosted. The substrate plateau — which
    carries no layer information — is compressed into a narrow output
    band, and the display's range is spent on the few DN that separate
    one layer from the next.
    """

    enabled: bool = False
    gain: float = 3.0
    width: float = 32.0

    #: field → key in :data:`RANGES`. The keys are flat and prefixed so
    #: there is exactly one bounds table to keep honest, while the
    #: serialised form stays the plain field names ``asdict`` produces.
    RANGE_KEYS = {"gain": "gain", "width": "width"}


@dataclass
class Denoise:
    """Edge-preserving smoothing, for the noise the curve would amplify.

    Bilateral rather than Gaussian or median: the flake *edges* are the
    thing being identified, and a smoother that softens them trades one
    problem for a worse one. This is the only stage here with a real
    cost — tens of milliseconds at 1080p, hundreds at 4K — which is
    affordable only because it runs on the detection worker.
    """

    enabled: bool = False
    diameter: int = 7
    sigma_color: float = 40.0
    sigma_space: float = 5.0

    RANGE_KEYS = {"diameter": "denoise_diameter",
                  "sigma_color": "denoise_sigma_color",
                  "sigma_space": "denoise_sigma_space"}


@dataclass
class PreprocessConfig:
    """The whole chain, including the master switch.

    ``enabled`` False means the output *is* the input — the same array,
    not a copy that happens to match. That is the "pre-processed layer is
    the original image when pre-processing is off" contract, and it is
    cheap to keep and easy to test.
    """

    enabled: bool = False
    local: LocalContrast = field(default_factory=LocalContrast)
    denoise: Denoise = field(default_factory=Denoise)

    # -- serialisation (the bundled defaults ARE the schema) ------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data) -> "PreprocessConfig":
        """Forgiving by design, like ``IdentifyConfig.from_dict``: an
        unreadable value falls back to its default rather than stopping
        the pipeline. A hand-edited settings file must not be able to
        break the live view.
        """
        cfg = cls()
        if not isinstance(data, dict):
            return cfg
        cfg.enabled = _as_bool(data.get("enabled"), cfg.enabled)
        cfg.local = _as_section(LocalContrast, data.get("local"))
        cfg.denoise = _as_section(Denoise, data.get("denoise"))
        return cfg

    # ------------------------------------------------------------------

    def local_active(self, centre_rgb=None) -> bool:
        """Whether the local-contrast curve will do anything.

        The centre is part of the question: a gain above 1 with no colour
        picked yet is still the identity, which is what lets the panel
        enable the curve before the operator has chosen anything.
        """
        return (self.local.enabled
                and self.local.gain > 1.0 + 1e-9
                and centre_rgb is not None)

    def points_identity(self, centre_rgb=None) -> bool:
        """True when the point chain changes nothing."""
        return not self.local_active(centre_rgb)

    def is_identity(self, centre_rgb=None) -> bool:
        """True when :func:`apply` would return the frame unchanged."""
        return (not self.enabled
                or (self.points_identity(centre_rgb)
                    and not self.denoise.enabled))


def _as_bool(value, default: bool) -> bool:
    return bool(value) if isinstance(value, bool) else default


def _as_float(value, default: float, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    lo, hi = RANGES[key]
    return float(min(max(float(value), lo), hi))


def _as_int(value, default: int, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    lo, hi = RANGES[key]
    return int(round(min(max(float(value), lo), hi)))


def _as_section(cls, data):
    """Build a sub-config from one section of the settings dict."""
    out = cls()
    if not isinstance(data, dict):
        return out
    out.enabled = _as_bool(data.get("enabled"), out.enabled)
    for name, current in vars(out).items():
        if name == "enabled":
            continue
        limit = getattr(cls, "RANGE_KEYS", {}).get(name, name)
        if isinstance(current, int) and not isinstance(current, bool):
            setattr(out, name, _as_int(data.get(name), current, limit))
        elif isinstance(current, float):
            setattr(out, name, _as_float(data.get(name), current, limit))
    return out


# ----------------------------------------------------------------------
# The point operations, as one LUT per channel
# ----------------------------------------------------------------------

def build_lut(cfg: PreprocessConfig,
              centre_rgb: tuple[int, int, int] | None = None) -> np.ndarray:
    """The point chain as a (3, 256) uint8 table, RGB order.

    Channel-major — row 0 is red's 256 outputs — because that is the shape
    a reader can check by indexing. :func:`cv_lut` converts it to the one
    OpenCV wants; the two are NOT the same memory layout, and reshaping
    between them without the transpose silently applies red's curve to
    green.

    The curve is the only point stage left, so this is the identity ramp
    when it is off (or when no colour has been picked yet — a gain with no
    centre is still the identity, which is what lets the panel enable the
    curve before the operator has chosen anything).
    """
    gain = float(cfg.local.gain)
    if not cfg.local.enabled or gain <= 1.0 + 1e-9 or centre_rgb is None:
        ramp = np.arange(LUT_SIZE, dtype=np.uint8)
        return np.stack([ramp, ramp, ramp])
    return np.stack([
        channel_curve(float(centre_rgb[channel]), gain, cfg.local.width)
        for channel in range(3)
    ])


# ----------------------------------------------------------------------
# The spatial stages
# ----------------------------------------------------------------------

def cv_lut(lut: np.ndarray) -> np.ndarray:
    """A channel-major (3, 256) table as the (1, 256, 3) OpenCV wants.

    The transpose is the whole point. ``(3, 256).reshape(1, 256, 3)``
    reinterprets the same bytes in a different order rather than moving
    them, so the table arrives permuted and every channel is filtered by
    another channel's curve — a bug that leaves the image looking like a
    plausible, wrongly-tinted photograph.
    """
    return np.ascontiguousarray(lut.T).reshape(1, LUT_SIZE, 3)


def denoise(img: np.ndarray, diameter: int = 7, sigma_color: float = 40.0,
            sigma_space: float = 5.0) -> np.ndarray:
    """Edge-preserving smoothing (bilateral)."""
    d = int(min(max(int(diameter), 1), int(RANGES["denoise_diameter"][1])))
    return cv2.bilateralFilter(img, d, float(sigma_color),
                               float(sigma_space))


# ----------------------------------------------------------------------

def apply(img: np.ndarray, cfg: PreprocessConfig | None,
          centre_rgb: tuple[int, int, int] | None = None) -> np.ndarray:
    """Run the chain. Returns ``img`` itself when nothing is enabled.

    Returning the input for the identity case is deliberate: it makes
    "pre-processing off means the pre-processed layer IS the original"
    true in the strongest sense, and it costs nothing.
    """
    if img is None or cfg is None or cfg.is_identity(centre_rgb):
        return img
    out = img
    if cfg.denoise.enabled:
        out = denoise(out, cfg.denoise.diameter, cfg.denoise.sigma_color,
                      cfg.denoise.sigma_space)
    if not cfg.points_identity(centre_rgb):
        out = cv2.LUT(out, cv_lut(build_lut(cfg, centre_rgb)))
    return out


__all__ = [
    "LUT_SIZE",
    "MAX_FLAT_RUN",
    "RANGES",
    "Denoise",
    "LocalContrast",
    "PreprocessConfig",
    "apply",
    "build_lut",
    "channel_curve",
    "curve_values",
    "cv_lut",
    "denoise",
    "effective_gain",
    "effective_width",
]
