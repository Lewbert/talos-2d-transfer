"""The scan's frame writer: the manifest is 1:1 with the plan, the frames
are on disk before the manifest says they are, and a slow disk costs speed
rather than memory."""

import csv
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from talos.cv.frame_writer import MANIFEST_HEADER, FrameWriter


def _frame(w=16, h=8):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _writer(tmp_path, **kwargs):
    kwargs.setdefault("max_queue", 8)
    return FrameWriter(tmp_path / "scan", {"objective_id": "5x",
                                           "focus_pos": 1234}, **kwargs)


def test_rows_follow_submission_order_and_frames_land_on_disk(tmp_path):
    writer = _writer(tmp_path)
    writer.start()
    for index in (0, 1, 2):
        writer.submit_frame(index, (float(index), 2.0, 0.0), _frame(),
                            t_unix=100.0 + index)
    assert writer.close() is None

    rows = _rows(writer.manifest_path)
    assert [row["frame"] for row in rows] == [f"frame_{i:05d}.png"
                                              for i in (0, 1, 2)]
    assert [p.name for p in writer.frames] == [row["frame"] for row in rows]
    for path in writer.frames:
        assert path.exists() and path.parent.name == "frames"
    assert rows[1]["x_um"] == "1.000" and rows[1]["objective_id"] == "5x"
    assert rows[1]["focus_pos"] == "1234"


def test_a_waypoint_with_nothing_to_record_still_gets_its_row(tmp_path):
    """The manifest records the geometry of the run even when the images
    (or the positions) did not arrive — the row is the only place the
    visited waypoint exists."""
    writer = _writer(tmp_path)
    writer.start()
    writer.submit_frame(0, (0.0, 0.0, 0.0), _frame(), 1.0)
    writer.submit_missing(1, None, 2.0)               # no readback at all
    writer.submit_missing(2, (3.0, 4.0, 0.0), 3.0)    # readback, no frame
    writer.submit_frame(3, (5.0, 6.0, 0.0), _frame(), 4.0)
    assert writer.close() is None

    rows = _rows(writer.manifest_path)
    assert len(rows) == 4
    assert rows[1]["frame"] == "" and rows[1]["x_um"] == ""
    assert rows[2]["frame"] == "" and rows[2]["x_um"] == "3.000"
    assert rows[3]["frame"] == "frame_00003.png"
    assert len(writer.frames) == 2


def test_a_failed_encode_never_names_a_file_that_is_not_there(tmp_path,
                                                             monkeypatch):
    writer = _writer(tmp_path)
    writer.start()
    monkeypatch.setattr("talos.cv.frame_writer.cv2.imwrite",
                        lambda *a, **k: False)
    writer.submit_frame(0, (0.0, 0.0, 0.0), _frame(), 1.0)
    error = writer.close()

    assert error is not None and "could not write" in error
    rows = _rows(writer.manifest_path)
    assert rows[0]["frame"] == ""        # the row is kept, the name is not
    assert writer.frames == []
    assert not (writer.frames_dir / "frame_00000.png").exists()


def test_the_first_error_is_the_one_reported(tmp_path, monkeypatch):
    writer = _writer(tmp_path)
    writer.start()
    monkeypatch.setattr("talos.cv.frame_writer.cv2.imwrite",
                        lambda *a, **k: False)
    writer.submit_frame(0, (0.0, 0.0, 0.0), _frame(), 1.0)
    writer.submit_frame(1, (1.0, 1.0, 0.0), _frame(), 2.0)
    assert "could not write" in writer.close()


def test_the_writer_emits_nothing_of_its_own(tmp_path):
    """The scan's signals belong to the SCAN thread: one emitter, and a
    synchronous caller (the sim tests, the CLI benches) sees its own emits
    delivered instead of queued to an event loop that is not running."""
    writer = _writer(tmp_path)
    assert not hasattr(writer, "sig_frame")
    assert not hasattr(writer, "sig_tile")


def test_a_full_queue_blocks_the_caller_rather_than_growing(tmp_path):
    """Backpressure, not memory: the third submit cannot return while the
    writer is wedged and the queue (size 1) is full."""
    writer = _writer(tmp_path, max_queue=1)
    gate = threading.Event()
    real_write = FrameWriter._write

    def held(self, item):
        gate.wait(5.0)
        real_write(self, item)

    writer._write = held.__get__(writer, FrameWriter)
    writer.start()
    writer.submit_frame(0, (0.0, 0.0, 0.0), _frame(), 1.0)   # writer takes it
    time.sleep(0.05)                                        # it is inside held()
    writer.submit_frame(1, (1.0, 1.0, 0.0), _frame(), 2.0)   # fills the queue

    finished = threading.Event()

    def third():
        writer.submit_frame(2, (2.0, 2.0, 0.0), _frame(), 3.0)
        finished.set()

    submitter = threading.Thread(target=third)
    submitter.start()
    submitter.join(0.2)
    assert not finished.is_set(), "the writer let the queue grow"

    gate.set()
    submitter.join(5.0)
    assert finished.is_set()
    assert writer.close() is None
    assert len(_rows(writer.manifest_path)) == 3


def test_close_waits_for_everything_queued(tmp_path):
    """scan_output re-reads the manifest from disk the moment the scan
    returns, so close() must mean 'the dataset is complete'."""
    writer = _writer(tmp_path)
    writer.start()
    for index in range(20):
        writer.submit_frame(index, (float(index), 0.0, 0.0), _frame(), 1.0)
    assert writer.close() is None
    rows = _rows(writer.manifest_path)
    assert len(rows) == 20
    assert len(list(writer.frames_dir.glob("*.png"))) == 20


def test_the_manifest_header_is_the_documented_one(tmp_path):
    writer = _writer(tmp_path)
    writer.start()
    writer.close()
    with open(writer.manifest_path, newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == MANIFEST_HEADER
