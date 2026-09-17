"""Assemble what a scan captured: a mosaic, and a tile overview.

Both are OVERVIEWS, not the data — the frames in ``frames/`` are the data,
and they are the only thing a measurement should ever be taken from. A
mosaic is what makes a scan legible at a glance: the whole visited area in
one image, at whatever resolution fits.

Pure numpy/cv2, no Qt, no camera — unit-testable and usable from a CLI.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Longest edge of the stitched mosaic. The tiles live on disk at full
#: resolution; this is a summary, and 2048 px keeps it a few MB.
DEFAULT_MAX_PX = 2048

#: Thumbnail width in the tile overview.
DEFAULT_THUMB_W = 240


def build_mosaic(tiles, fov_um: tuple[float, float],
                 max_px: int = DEFAULT_MAX_PX) -> np.ndarray | None:
    """Stitch ``tiles`` into one image, placed by their readback positions.

    ``tiles``: ``[(x_um, y_um, rgb_uint8)]`` where (x_um, y_um) is the stage
    position the frame was taken at — i.e. the CENTRE of that tile in the
    sample frame. ``fov_um`` is the field of view those frames cover.

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
    px_per_um = min(tile_w / fov_x, tile_h / fov_y)

    xs = [x for x, _y, _img in items]
    ys = [y for _x, y, _img in items]
    x0, x1 = min(xs) - fov_x / 2.0, max(xs) + fov_x / 2.0
    y0, y1 = min(ys) - fov_y / 2.0, max(ys) + fov_y / 2.0
    span_x, span_y = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)

    out_w = max(1, int(round(span_x * px_per_um)))
    out_h = max(1, int(round(span_y * px_per_um)))
    shrink = min(1.0, float(max_px) / float(max(out_w, out_h)))
    if shrink < 1.0:
        out_w = max(1, int(round(out_w * shrink)))
        out_h = max(1, int(round(out_h * shrink)))

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


def build_overview(entries, cols: int | None = None,
                   thumb_w: int = DEFAULT_THUMB_W,
                   colour: tuple[int, int, int] = (0, 200, 255)
                   ) -> np.ndarray | None:
    """A thumbnail per captured tile, in scan order, with boxes drawn.

    ``entries``: ``[(rgb_uint8, [FlakeCandidate, ...]), ...]`` in the order
    the scan visited them. Easier to read than a mosaic when the area is
    large and the samples are small — and it shows WHICH tile a detection
    came from.
    """
    usable = [(img, cands) for img, cands in entries
              if img is not None and getattr(img, "size", 0)]
    if not usable:
        return None
    cols = max(1, int(cols or round(len(usable) ** 0.5)))
    scale = thumb_w / float(max(img.shape[1] for img, _ in usable))
    thumb_h = max(1, int(round(max(img.shape[0] for img, _ in usable) * scale)))
    rows = (len(usable) + cols - 1) // cols
    pad = 4
    sheet = np.full((rows * (thumb_h + pad) + pad, cols * (thumb_w + pad) + pad, 3),
                    18, np.uint8)
    for index, (img, cands) in enumerate(usable):
        thumb = cv2.resize(img, (thumb_w, thumb_h),
                           interpolation=cv2.INTER_AREA)
        for cand in cands:
            x, y, w, h = cand.bbox
            cv2.rectangle(thumb,
                          (int(x * scale), int(y * scale)),
                          (int((x + w) * scale), int((y + h) * scale)),
                          colour, 1)
        row, col = divmod(index, cols)
        top = pad + row * (thumb_h + pad)
        left = pad + col * (thumb_w + pad)
        sheet[top:top + thumb_h, left:left + thumb_w] = thumb
    return sheet


__all__ = ["DEFAULT_MAX_PX", "DEFAULT_THUMB_W", "build_mosaic",
           "build_overview"]
