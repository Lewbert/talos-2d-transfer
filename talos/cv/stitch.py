"""Assemble what a scan captured: the mosaic, and the rings on it.

Both are OVERVIEWS, not the data. The frames in ``frames/`` are the data, and
they are the only thing anything measures from: **identification runs per
single frame, never on a merged image** — merging tens or hundreds of tiles
is too expensive in compute and memory for what it would buy, and the
per-frame result is what "go to sample" needs anyway. A mosaic here exists
so an operator can see at a glance where a scan went and what it covered.

That sets the accuracy bar, and it is a low one: **a small gap or a slightly
imperfect seam is acceptable** because nothing downstream consumes the
stitched pixels. So the builder deliberately stays simple — placement by the
manifest readback positions, overlaps AVERAGED (a stamp would leave a seam,
a seam is what the overlap exists to avoid) — with no registration, no seam
blending and no rotation correction. If a future upgrade ever needs the
mosaic to be measurement-grade, that is when the jacobian work in
cv/calibration.py comes back into play.

Pure numpy/cv2, no Qt, no camera — unit-testable and usable from a CLI.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from talos.cv.orientation import mosaic_offset

#: Longest edge of the stitched mosaic. The tiles live on disk at full
#: resolution; this is a summary, and 2048 px keeps it a few MB.
DEFAULT_MAX_PX = 2048


def mosaic_geometry(tiles, fov_um: tuple[float, float],
                    max_px: int = DEFAULT_MAX_PX,
                    flip: bool = False) -> tuple[float, float, float]:
    """``(px_per_um, x0_um, y0_um, shrink)`` — where a sample point lands.

    The single source of the mosaic's layout: ``build_mosaic`` places tiles
    with it, and a caller that wants to draw on top of a mosaic (or check
    where a feature ended up) uses it instead of re-deriving the arithmetic.

    Tiles are laid out in the coordinates of the SAMPLE as the frames show
    it, so with the camera flip on the layout is mirrored (see
    cv/orientation.py for why that is what makes the seams line up).
    """
    items = [(float(x), float(y), np.asarray(img))
             for x, y, img in tiles if img is not None and img.size]
    items = [(mosaic_offset(x, y, flip)[0], mosaic_offset(x, y, flip)[1], img)
             for x, y, img in items]
    fov_x, fov_y = float(fov_um[0]), float(fov_um[1])
    tile_w = min(img.shape[1] for _x, _y, img in items)
    tile_h = min(img.shape[0] for _x, _y, img in items)
    px_per_um = min(tile_w / fov_x, tile_h / fov_y)
    xs = [x for x, _y, _img in items]
    ys = [y for _x, y, _img in items]
    x0, y0 = min(xs) - fov_x / 2.0, min(ys) - fov_y / 2.0
    span_x = max(max(xs) + fov_x / 2.0 - x0, 1e-6)
    span_y = max(max(ys) + fov_y / 2.0 - y0, 1e-6)
    out_w = max(1, int(round(span_x * px_per_um)))
    out_h = max(1, int(round(span_y * px_per_um)))
    shrink = min(1.0, float(max_px) / float(max(out_w, out_h)))
    return px_per_um, x0, y0, shrink


def build_mosaic(tiles, fov_um: tuple[float, float],
                 max_px: int = DEFAULT_MAX_PX,
                 flip: bool = False) -> np.ndarray | None:
    """Stitch ``tiles`` into one image, placed by their readback positions.

    ``tiles``: ``[(x_um, y_um, rgb_uint8)]`` where (x_um, y_um) is the stage
    position the frame was taken at — i.e. the CENTRE of that tile in the
    sample frame. ``fov_um`` is the field of view those frames cover.
    ``flip`` is the camera flip, which mirrors the layout (cv/orientation.py).

    Overlapping pixels are averaged rather than overwritten, so a tile that
    is slightly offset (or a few µm out because of backlash) blends instead
    of showing a seam. Returns None when there is nothing to draw.
    """
    items = [(float(x), float(y), np.asarray(img))
             for x, y, img in tiles if img is not None and img.size]
    if not items:
        return None
    fov_x, fov_y = float(fov_um[0]), float(fov_um[1])
    if fov_x <= 0 or fov_y <= 0:
        return None

    tile_w = min(img.shape[1] for _x, _y, img in items)
    tile_h = min(img.shape[0] for _x, _y, img in items)
    px_per_um, x0, y0, shrink = mosaic_geometry(items, fov_um, max_px, flip)
    items = [(mosaic_offset(x, y, flip)[0], mosaic_offset(x, y, flip)[1], img)
             for x, y, img in items]
    out_w = max(1, int(round((max(x for x, _y, _i in items) + fov_x / 2.0 - x0)
                             * px_per_um * shrink)))
    out_h = max(1, int(round((max(y for _x, y, _i in items) + fov_y / 2.0 - y0)
                             * px_per_um * shrink)))

    # Accumulate as integers and divide at the end: with a 1-9 tile overlap
    # the sum cannot overflow uint16, and it is a third of float32's memory.
    total = np.zeros((out_h, out_w, 3), np.uint16)
    counts = np.zeros((out_h, out_w, 1), np.uint16)
    for x_um, y_um, img in items:
        tile = img
        if img.shape[0] != tile_h or img.shape[1] != tile_w:
            tile = cv2.resize(img, (tile_w, tile_h),
                              interpolation=cv2.INTER_AREA)
        # the tile covers fov_x µm, so at the mosaic's scale it occupies
        # fov_x × px_per_um pixels — NOT its own pixel count
        tw = max(1, int(round(fov_x * px_per_um * shrink)))
        th = max(1, int(round(fov_y * px_per_um * shrink)))
        if (tw, th) != (tile_w, tile_h):
            tile = cv2.resize(tile, (tw, th), interpolation=cv2.INTER_AREA)
        left = int(round((x_um - fov_x / 2.0 - x0) * px_per_um * shrink))
        top = int(round((y_um - fov_y / 2.0 - y0) * px_per_um * shrink))
        # clip to the canvas: a tile half outside the planned area
        src_x0, src_y0 = max(0, -left), max(0, -top)
        dst_x0, dst_y0 = max(0, left), max(0, top)
        w = min(tw - src_x0, out_w - dst_x0)
        h = min(th - src_y0, out_h - dst_y0)
        if w <= 0 or h <= 0:
            continue
        patch = tile[src_y0:src_y0 + h, src_x0:src_x0 + w].astype(np.uint16)
        total[dst_y0:dst_y0 + h, dst_x0:dst_x0 + w] += patch
        counts[dst_y0:dst_y0 + h, dst_x0:dst_x0 + w] += 1

    covered = counts[:, :, 0] > 0
    out = np.zeros((out_h, out_w, 3), np.uint8)
    if covered.any():
        divisor = np.maximum(counts, 1)
        out[covered] = (total[covered] // divisor[covered]).astype(np.uint8)
    return out


def draw_sample_rings(mosaic: np.ndarray, samples, tiles,
                      fov_um: tuple[float, float], *,
                      max_px: int = DEFAULT_MAX_PX, flip: bool = False,
                      colour: tuple[int, int, int] = (0, 255, 90),
                      label: bool = True) -> np.ndarray:
    """Ring every found sample on the mosaic — in place, returns it.

    Takes the SAME tiles, field of view and flip the mosaic was built from,
    and derives their geometry rather than accepting one: a geometry from a
    different call, or the flip left out, would put every ring somewhere
    plausible and wrong. (The layout is the mosaic's own — the rule the map
    follows too.)

    A sample is ringed by the circle of equal AREA (``area_um2`` is on every
    candidate already and is measurement-based, unlike a bbox corner), and
    each ring carries its number from the sample list — the point of the
    image is to go back and look at a specific one. Drawn dark-then-bright
    so a ring reads on a dark field and on a bright one.
    """
    if mosaic is None or mosaic.size == 0 or not samples:
        return mosaic
    px_per_um, x0, y0, shrink = mosaic_geometry(tiles, fov_um, max_px, flip)
    scale = float(px_per_um) * float(shrink)
    if scale <= 0:
        return mosaic
    height, width = mosaic.shape[:2]
    font_scale = max(0.45, min(1.0, height / 900.0))
    for index, cand in enumerate(samples):
        x_um, y_um = mosaic_offset(cand.x_um, cand.y_um, flip)
        cx = int(round((x_um - float(x0)) * scale))
        cy = int(round((y_um - float(y0)) * scale))
        radius_um = math.sqrt(max(float(cand.area_um2), 1.0) / math.pi)
        radius = max(4, int(round(radius_um * scale)))
        if not (-radius <= cx <= width + radius
                and -radius <= cy <= height + radius):
            continue                      # a sample off the mosaic's canvas
        cv2.circle(mosaic, (cx, cy), radius, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.circle(mosaic, (cx, cy), radius, colour, 2, cv2.LINE_AA)
        if not label:
            continue
        text = str(index + 1)
        at = (cx + radius + 3, cy - radius - 3)
        cv2.putText(mosaic, text, at, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(mosaic, text, at, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    colour, 2, cv2.LINE_AA)
    return mosaic


__all__ = ["DEFAULT_MAX_PX", "build_mosaic", "draw_sample_rings",
           "mosaic_geometry"]
