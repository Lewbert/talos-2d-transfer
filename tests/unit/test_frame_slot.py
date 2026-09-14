"""LatestFrameSlot: newest-wins handoff, gated reads, threaded smoke."""

import threading
import time

import numpy as np

from talos.cv.frame_slot import LatestFrameSlot


def _frame(n: int) -> np.ndarray:
    return np.full((4, 4, 3), n, dtype=np.uint8)


def test_empty_slot_reads_none():
    slot = LatestFrameSlot()
    assert slot.read() is None
    assert slot.read_since() is None
    assert slot.read_since(min_seq=0) is None


def test_write_read_roundtrip_by_reference():
    slot = LatestFrameSlot()
    frame = _frame(7)
    slot.write(frame, t_capture=100.0, seq=1)
    out, meta = slot.read()
    assert out is frame  # handed over by reference, no copy
    assert meta.t_capture == 100.0
    assert meta.seq == 1
    assert meta.shape == (4, 4, 3)


def test_newest_wins_overwrites():
    slot = LatestFrameSlot()
    slot.write(_frame(1), 10.0, 1)
    slot.write(_frame(2), 11.0, 2)
    _, meta = slot.read()
    assert meta.seq == 2
    assert meta.t_capture == 11.0


def test_read_since_seq_gate():
    slot = LatestFrameSlot()
    slot.write(_frame(1), 10.0, 1)
    assert slot.read_since(min_seq=1) is None    # not NEWER than seq 1
    assert slot.read_since(min_seq=0) is not None
    slot.write(_frame(2), 11.0, 2)
    _, meta = slot.read_since(min_seq=1)
    assert meta.seq == 2


def test_read_since_time_gate():
    slot = LatestFrameSlot()
    slot.write(_frame(1), 10.0, 1)
    assert slot.read_since(min_t=11.0) is None
    _, meta = slot.read_since(min_t=9.0)
    assert meta.seq == 1


def test_read_since_both_gates_must_pass():
    slot = LatestFrameSlot()
    slot.write(_frame(1), 10.0, 1)
    slot.write(_frame(2), 12.0, 2)
    _, meta = slot.read_since(min_t=11.5, min_seq=1)
    assert meta.seq == 2
    assert slot.read_since(min_t=12.5, min_seq=1) is None  # time gate fails
    assert slot.read_since(min_t=11.5, min_seq=2) is None  # seq gate fails


def test_threaded_producer_consumer_smoke():
    slot = LatestFrameSlot()
    stop = threading.Event()
    seen: list[int] = []

    def producer() -> None:
        n = 0
        while not stop.is_set():
            slot.write(_frame(n % 256), time.monotonic(), n)
            n += 1
            time.sleep(0.001)

    def consumer() -> None:
        last_seq = -1
        while not stop.is_set():
            item = slot.read_since(min_seq=last_seq)
            if item is not None:
                _, meta = item
                seen.append(meta.seq)
                last_seq = meta.seq
            time.sleep(0.0005)

    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=consumer)
    t1.start()
    t2.start()
    time.sleep(0.2)
    stop.set()
    t1.join()
    t2.join()
    assert seen
    assert seen == sorted(set(seen))  # strictly increasing, no duplicates
