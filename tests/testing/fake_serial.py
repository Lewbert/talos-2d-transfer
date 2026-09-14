"""Minimal pyserial-compatible fakes for driver unit tests."""

from __future__ import annotations

from typing import Any


class FakeSerial:
    """Byte-level fake: scripted request-bytes -> response-bytes."""

    def __init__(self, script: dict[bytes, bytes] | None = None):
        self._script = dict(script or {})
        self._pending = b""
        self.writes: list[bytes] = []
        self.timeout = 0.1
        self.is_open = True
        self.port = "FAKE"

    def write(self, data: bytes) -> None:
        self.writes.append(bytes(data))
        resp = self._script.get(bytes(data), b"")
        if isinstance(resp, list):
            resp = resp.pop(0) if resp else b""
        self._pending = resp

    def reset_input_buffer(self) -> None:
        self._pending = b""

    def reset_output_buffer(self) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.is_open = False

    @property
    def in_waiting(self) -> int:
        return len(self._pending)

    def read(self, n: int = 1) -> bytes:
        if not self._pending:
            return b""
        chunk, self._pending = self._pending[:n], self._pending[n:]
        return chunk


class LineFakeSerial(FakeSerial):
    """Line-level fake: script maps ASCII command strings to reply bytes
    (or to lists of reply bytes consumed one per call)."""

    def __init__(self, script: dict[str, Any] | None = None):
        self._line_script = dict(script or {})
        super().__init__(script={})

    def write(self, data: bytes) -> None:
        line = data.decode("ascii").strip()
        self.writes.append(line)
        resp = self._line_script.get(line, b"")
        if isinstance(resp, list):
            resp = resp.pop(0) if resp else b""
        self._pending = resp
