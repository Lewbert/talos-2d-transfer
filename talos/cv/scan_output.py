"""What a scan leaves behind besides the frames.

``frames/`` and ``manifest.csv`` are written by :class:`talos.cv.scan.GridScanner`
as the run goes. This module writes the three optional summaries afterwards
— the mosaic, the tile overview and the sample list — because detection
deliberately outlives the capture: identification must never slow the
stage down, so "the scan finished" and "the results are in" are different
moments and these wait for the second one.

Pure: no Qt, no camera, no widgets. The caller copies the results and the
export flags out of its controls before handing them over, which is both
simpler to test and safe to run on a thread.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2

from talos.cv.stitch import build_mosaic, build_overview


def candidate_rows(candidates) -> list[list[str]]:
    """Table/CSV rows for candidates — one formatter, so what is exported
    is what was shown."""
    return [[str(index + 1), f"{c.x_um:.1f}", f"{c.y_um:.1f}",
             f"{c.area_um2:.1f}", f"{c.score:.1f}"]
            for index, c in enumerate(candidates)]


def frames_from_manifest(manifest_path: Path, out_dir: Path
                         ) -> list[tuple[int, Path]]:
    """(waypoint index, frame path) for every waypoint that captured one.

    The MANIFEST is the alignment, not the order frames arrived in: a
    waypoint that captured nothing keeps its row with an empty frame cell,
    so zipping the frame list against the tile list would shift every
    frame after a miss onto the wrong position.
    """
    frames: list[tuple[int, Path]] = []
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            name = (row.get("frame") or "").strip()
            if name:
                frames.append((index, out_dir / "frames" / name))
    return frames


def write_outputs(out_dir: Path, manifest_path: Path,
                  tiles: dict, hits: dict, fov_um, exports: dict,
                  flip: bool = False) -> dict:
    """Write the requested summaries and report what was written.

    ``tiles`` maps waypoint index → the readback position it was taken at,
    ``hits`` maps waypoint index → the candidates found in that frame.
    """
    out_dir = Path(out_dir)
    written: list[str] = []
    frames = frames_from_manifest(Path(manifest_path), out_dir)

    if exports.get("candidates"):
        path = out_dir / "candidates.csv"
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["tile", "x_um", "y_um", "area_um2",
                             "edge", "tile_x_um", "tile_y_um"])
            for index in sorted(hits):
                x_um, y_um = tiles.get(index, (0.0, 0.0))
                for cand in hits[index]:
                    writer.writerow([index, f"{cand.x_um:.3f}",
                                     f"{cand.y_um:.3f}",
                                     f"{cand.area_um2:.2f}",
                                     f"{cand.score:.1f}",
                                     f"{x_um:.3f}", f"{y_um:.3f}"])
        written.append(path.name)

    # Read each frame at most once for the two image summaries.
    images = {}
    if exports.get("mosaic") or exports.get("overview"):
        for index, path in frames:
            image = cv2.imread(str(path))
            if image is not None:
                images[index] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if exports.get("mosaic"):
        mosaic_tiles = [(tiles.get(index, (0.0, 0.0))[0],
                         tiles.get(index, (0.0, 0.0))[1], image)
                        for index, image in images.items()]
        mosaic = build_mosaic(mosaic_tiles, fov_um, flip=flip)
        if mosaic is not None:
            path = out_dir / "mosaic.png"
            cv2.imwrite(str(path), cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))
            written.append(f"{path.name} ({mosaic.shape[1]}×"
                           f"{mosaic.shape[0]})")

    if exports.get("overview"):
        entries = [(image, hits.get(index, []))
                   for index, image in images.items()]
        sheet = build_overview(entries)
        if sheet is not None:
            path = out_dir / "overview.png"
            cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
            written.append(path.name)

    return {"dir": out_dir, "written": written,
            "samples": sum(len(items) for items in hits.values())}


__all__ = ["candidate_rows", "frames_from_manifest", "write_outputs"]
