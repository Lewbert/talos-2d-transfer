"""Line-framed serial protocol helpers (SigmaKoki XYZ stage, focus stage).

Protocol: ASCII lines terminated by '\\n' (possibly '\\r\\n'). Replies are
single lines (``OK:STEP:X:...``, ``S:POS:...``) while unsolicited lines
(``EV:...``, ``BOOT``, ``ERR:...``) may arrive at any time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class LineReply:
    """A matching reply line plus any unsolicited lines seen while waiting."""

    line: str
    events: list[str]


class LineIO:
    """Line-oriented reader/writer over a pyserial port."""

    def __init__(self, port, timeout_s: float = 0.3):
        self.port = port
        self.timeout_s = timeout_s
        self._buf = b""

    def send(self, line: str) -> None:
        # NEVER discard pending input here: a late reply still in the RX
        # buffer would be purged MID-LINE, permanently desynchronizing the
        # reply pairing (hardware-verified: the firmware garble-spams
        # ERR:UNKNOWN on every subsequent command until a DTR reset).
        # Stale lines are handled by read_reply, which simply skips
        # non-matching lines — no purge needed.
        self.port.write((line.rstrip() + "\n").encode("ascii"))

    def read_line(self) -> str | None:
        """Read one line (may block up to timeout_s)."""
        deadline = time.monotonic() + self.timeout_s
        while True:
            if b"\n" in self._buf:
                raw, self._buf = self._buf.split(b"\n", 1)
                return raw.decode("ascii", errors="replace").strip()
            if time.monotonic() >= deadline:
                return None
            n = self.port.in_waiting or 1
            chunk = self.port.read(n)
            if chunk:
                self._buf += chunk
            else:
                time.sleep(0.005)

    def read_reply(self, expected_prefix: str,
                   error_prefixes: tuple[str, ...] = ("ERR:",)) -> LineReply:
        """Read lines until one starts with ``expected_prefix`` (or an
        error prefix, which is returned as the reply for the caller to
        dispatch). Raises TimeoutError if nothing matches within timeout_s.
        """
        prefixes = (expected_prefix,) + tuple(error_prefixes)
        events: list[str] = []
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            line = self.read_line()
            if line is None:
                break
            if line.startswith(prefixes):
                return LineReply(line=line, events=events)
            if line.startswith(("EV:", "BOOT")):
                events.append(line)
        raise TimeoutError(
            f"No '{expected_prefix}' reply within {self.timeout_s}s; events={events}"
        )

    def drain_events(self) -> list[str]:
        """Non-blocking read of any pending unsolicited lines."""
        events: list[str] = []
        saved_timeout = self.port.timeout
        self.port.timeout = 0.0
        try:
            chunk = self.port.read(self.port.in_waiting or 1024)
            if chunk:
                self._buf += chunk
        finally:
            self.port.timeout = saved_timeout
        while b"\n" in self._buf:
            raw, self._buf = self._buf.split(b"\n", 1)
            line = raw.decode("ascii", errors="replace").strip()
            if line and line.startswith(("EV:", "BOOT", "ERR:")):
                events.append(line)
        return events


def parse_kv(payload: str, sep: str = ",") -> dict[str, str]:
    """Parse 'K1:v1,K2:v2,...' into a dict (values kept as strings)."""
    out: dict[str, str] = {}
    for item in payload.split(sep):
        if ":" in item:
            k, v = item.split(":", 1)
            out[k.strip()] = v.strip()
    return out
