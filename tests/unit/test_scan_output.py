"""What a scan leaves behind: the summaries, and the alignment rule.

The manifest is the record of truth — a waypoint that captured nothing
keeps its row with an empty frame cell — so anything that pairs a frame
with a position has to read it rather than count arrivals. Getting that
wrong shifts every frame after a miss onto the wrong place, and the
result still *looks* like a scan.
"""

from __future__ import annotations

import csv

import cv2
import numpy as np
import pytest

from talos.cv.scan_output import (candidate_rows, frames_from_manifest,
                                  write_outputs)
from talos.models import FlakeCandidate


def _candidate(x_um: float, y_um: float, area: float = 50.0,
               score: float = 12.0) -> FlakeCandidate:
    return FlakeCandidate(x_px=1.0, y_px=2.0, area_px2=area, area_um2=area,
                          x_um=x_um, y_um=y_um, score=score)


def _scan(tmp_path, frames=("frame_00000.png", "frame_00001.png",
                            "", "frame_00003.png")):
    """A tiny scan folder: a manifest with a MISS in it, and the PNGs."""
    out = tmp_path / "scan_x"
    (out / "frames").mkdir(parents=True)
    for row, name in enumerate(frames):
        if not name:
            continue
        image = np.full((60, 80, 3), 40 + row * 20, np.uint8)
        cv2.imwrite(str(out / "frames" / name), image)
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "x_um", "y_um", "r_deg", "t_unix",
                         "objective_id", "focus_pos"])
        for row, name in enumerate(frames):
            writer.writerow([name, row * 100.0, 0.0, 0.0, 0.0, 0, 0])
    return out, out / "manifest.csv"


def test_the_manifest_decides_which_frame_is_which(tmp_path):
    """A miss keeps its row with an empty frame cell, so the indices of
    the frames that DID arrive are not 0,1,2,3."""
    _out, manifest = _scan(tmp_path)
    frames = frames_from_manifest(manifest, manifest.parent)
    assert [index for index, _ in frames] == [0, 1, 3]
    assert [path.name for _, path in frames] == [
        "frame_00000.png", "frame_00001.png", "frame_00003.png"]


def test_candidate_rows_are_the_table_and_the_csv(tmp_path):
    rows = candidate_rows([_candidate(1.234, 5.678, area=9.87, score=3.0)])
    assert rows == [["1", "1.2", "5.7", "9.9", "3.0"]]


def test_nothing_is_written_unless_it_was_asked_for(tmp_path):
    out, manifest = _scan(tmp_path)
    summary = write_outputs(out, manifest, tiles={0: (0.0, 0.0)},
                            hits={}, fov_um=(100.0, 100.0), exports={})
    assert summary["written"] == []
    assert not (out / "mosaic.png").exists()
    assert not (out / "candidates.csv").exists()


def test_the_three_summaries_are_written_where_the_scan_is(tmp_path):
    out, manifest = _scan(tmp_path)
    tiles = {0: (0.0, 0.0), 1: (100.0, 0.0), 3: (0.0, 100.0)}
    hits = {0: [_candidate(0.0, 0.0)], 3: [_candidate(20.0, 90.0)]}
    summary = write_outputs(out, manifest, tiles=tiles, hits=hits,
                            fov_um=(100.0, 100.0),
                            exports={"mosaic": True, "candidates": True,
                                     "annotated": True})
    assert set(summary["written"]) != set()
    for name in ("mosaic.png", "candidates.csv", "mosaic_annotated.png"):
        assert (out / name).exists(), name
    assert summary["samples"] == 2


def test_the_annotated_mosaic_is_the_mosaic_with_the_samples_on_it(tmp_path):
    """The same pixels, plus a ring per sample — that is what makes it
    usable for "where is sample 7 on the wafer", which the tile overview
    could not answer (it had no positions on it at all)."""
    out, manifest = _scan(tmp_path)
    tiles = {0: (0.0, 0.0), 1: (100.0, 0.0), 3: (0.0, 100.0)}
    hits = {0: [_candidate(0.0, 0.0, area=200.0)],
            3: [_candidate(20.0, 90.0, area=200.0)]}
    write_outputs(out, manifest, tiles=tiles, hits=hits,
                  fov_um=(100.0, 100.0),
                  exports={"mosaic": True, "annotated": True})
    plain = cv2.imread(str(out / "mosaic.png"))
    marked = cv2.imread(str(out / "mosaic_annotated.png"))
    assert plain.shape == marked.shape
    assert not np.array_equal(plain, marked)
    # the rings are drawn in the ring colour, and they are on the mosaic
    # (BGR here: the colour is BGR-swapped on the way to disk)
    ring = np.array([90, 255, 0], np.uint8)
    assert (np.abs(marked.astype(int) - ring).sum(axis=2) < 60).any()


def test_an_annotated_mosaic_with_no_samples_is_still_the_mosaic(tmp_path):
    out, manifest = _scan(tmp_path)
    tiles = {0: (0.0, 0.0), 1: (100.0, 0.0)}
    write_outputs(out, manifest, tiles=tiles, hits={},
                  fov_um=(100.0, 100.0), exports={"annotated": True})
    assert (out / "mosaic_annotated.png").exists()
    assert not (out / "mosaic.png").exists(), "only what was asked for"


def test_candidates_csv_carries_the_tile_the_sample_came_from(tmp_path):
    out, manifest = _scan(tmp_path)
    tiles = {0: (1000.0, 2000.0), 1: (1100.0, 2000.0), 3: (0.0, 100.0)}
    hits = {1: [_candidate(1010.5, 2002.25, area=42.0, score=7.5)]}
    write_outputs(out, manifest, tiles=tiles, hits=hits,
                  fov_um=(100.0, 100.0), exports={"candidates": True})
    with open(out / "candidates.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["tile"] == "1"
    assert rows[0]["x_um"] == "1010.500"
    assert rows[0]["area_um2"] == "42.00"
    # the tile's own readback position, so a reader can re-derive the
    # sample's position from the frame it was found in
    assert rows[0]["tile_x_um"] == "1100.000"
    assert rows[0]["tile_y_um"] == "2000.000"


def test_a_scan_with_no_frames_still_reports_cleanly(tmp_path):
    out, manifest = _scan(tmp_path, frames=("", ""))
    summary = write_outputs(out, manifest, tiles={}, hits={},
                            fov_um=(100.0, 100.0),
                            exports={"mosaic": True, "annotated": True})
    assert summary["written"] == []
    assert summary["samples"] == 0


def test_the_mosaic_lands_at_the_tile_positions(tmp_path):
    """Not just "a file was written": the picture has to be one image,
    which is what the readback placement is for."""
    out, manifest = _scan(tmp_path)
    tiles = {0: (0.0, 0.0), 1: (60.0, 0.0), 3: (30.0, 60.0)}
    write_outputs(out, manifest, tiles=tiles, hits={},
                  fov_um=(40.0, 40.0), exports={"mosaic": True})
    mosaic = cv2.imread(str(out / "mosaic.png"))
    assert mosaic is not None
    assert mosaic.shape[0] > 60 and mosaic.shape[1] > 60


@pytest.mark.parametrize("flip", (False, True))
def test_the_mosaic_follows_the_camera_flip(tmp_path, flip):
    out, manifest = _scan(tmp_path)
    tiles = {0: (0.0, 0.0), 1: (60.0, 0.0)}
    write_outputs(out, manifest, tiles=tiles, hits={},
                  fov_um=(40.0, 40.0), exports={"mosaic": True}, flip=flip)
    assert (out / "mosaic.png").exists()
