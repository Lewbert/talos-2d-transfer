"""ASCII line protocol tests."""

import pytest

from talos.protocols.ascii_line import LineIO, parse_kv
from tests.testing.fake_serial import LineFakeSerial


def test_parse_kv():
    out = parse_kv("POS:-589,MODE:IDLE,V:0,SPD:0,LIM:0,SLIM:0")
    assert out == {"POS": "-589", "MODE": "IDLE", "V": "0", "SPD": "0",
                   "LIM": "0", "SLIM": "0"}


def test_parse_kv_ignores_items_without_colon():
    assert parse_kv("a:1,bad,c:2") == {"a": "1", "c": "2"}


def test_read_reply_collects_events():
    fake = LineFakeSerial({})
    fake._pending = b"EV:LIM:+:123\nS:POS:5,MODE:IDLE,V:0,SPD:0,LIM:0,SLIM:0\n"
    io = LineIO(fake, timeout_s=0.2)
    reply = io.read_reply("S:")
    assert reply.line.startswith("S:POS:5")
    assert reply.events == ["EV:LIM:+:123"]


def test_read_reply_err_is_terminal():
    fake = LineFakeSerial({})
    fake._pending = b"ERR:BUSY\n"
    io = LineIO(fake, timeout_s=0.2)
    reply = io.read_reply("S:")
    assert reply.line == "ERR:BUSY"


def test_read_reply_timeout():
    fake = LineFakeSerial({})
    io = LineIO(fake, timeout_s=0.05)
    with pytest.raises(TimeoutError):
        io.read_reply("S:")


def test_read_reply_multi_response_list():
    fake = LineFakeSerial({"STATUS?": [b"S:POS:1,MODE:IDLE,V:0,SPD:0,LIM:0,SLIM:0\n",
                                       b"S:POS:2,MODE:TRAP,V:0,SPD:0,LIM:0,SLIM:0\n"]})
    io = LineIO(fake, timeout_s=0.2)
    io.send("STATUS?")
    first = io.read_reply("S:")
    io.send("STATUS?")
    second = io.read_reply("S:")
    assert parse_kv(first.line[2:])["POS"] == "1"
    assert parse_kv(second.line[2:])["POS"] == "2"


def test_drain_events():
    fake = LineFakeSerial({})
    fake._pending = b"EV:LIM:X+\nEV:LIM:X+\n"
    io = LineIO(fake, timeout_s=0.2)
    assert io.drain_events() == ["EV:LIM:X+", "EV:LIM:X+"]


def test_send_never_discards_pending_input():
    """Regression: send() used to reset_input_buffer() before every write,
    purging a late reply MID-LINE and permanently desynchronizing the
    reply pairing (hardware-verified: the focus firmware garble-spammed
    ERR:UNKNOWN on every subsequent command until a DTR reset)."""

    class NoDiscardFake:
        def __init__(self):
            self.pending = b"EV:OLD:1\nPART"  # stale line + a partial tail
            self.writes: list[bytes] = []
            self.timeout = 0.1

        @property
        def in_waiting(self):
            return len(self.pending)

        def write(self, data):
            self.writes.append(data)

        def read(self, n):
            chunk, self.pending = self.pending[:n], self.pending[n:]
            return chunk

        def reset_input_buffer(self):
            raise AssertionError("send() must never discard pending input")

    port = NoDiscardFake()
    io = LineIO(port, timeout_s=0.2)
    io.send("STATUS?")
    assert port.writes == [b"STATUS?\n"]
    # the stale complete line survives and stays line-aligned
    assert io.read_line() == "EV:OLD:1"
    # the partial tail stays buffered for the next read (no desync)
    assert io._buf == b"PART"
