"""Minimal Modbus RTU codec (own implementation — no pymodbus dependency).

Rules baked in (researched for Zolix ZC300 / Yudian AI-828):
- Frame: addr + fn + payload + CRC16 (low byte first).
- Registers are 1-based in documentation; the address on the wire is reg - 1.
- 32-bit floats span 2 registers, IEEE 754 big-endian.
- Exception responses: fn | 0x80 followed by an exception code.
"""

from __future__ import annotations

import struct

# Function codes
READ_HOLDING = 0x03
READ_INPUT = 0x04
WRITE_SINGLE = 0x06
WRITE_MULTIPLE = 0x10

_EXC_CODES = {
    0x01: "illegal function",
    0x02: "illegal data address",
    0x03: "illegal data value",
    0x04: "slave device failure",
    0x06: "slave device busy",
    0x07: "negative acknowledge / limit",
    0x08: "estop",
    0x09: "not enabled",
    0x0A: "gateway path unavailable",
}


class ModbusException(Exception):
    """Device returned a Modbus exception response."""

    def __init__(self, code: int, fn: int):
        self.code = code
        self.fn = fn
        super().__init__(
            f"Modbus exception 0x{code:02X} on fn 0x{fn:02X}: {_EXC_CODES.get(code, 'unknown')}"
        )


class FrameError(Exception):
    """CRC mismatch or malformed frame."""


def crc16(data: bytes) -> int:
    """Modbus CRC-16 (polynomial 0xA001), as an int."""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


def frame(slave: int, fn: int, payload: bytes) -> bytes:
    """Assemble a full RTU frame with CRC."""
    body = bytes([slave, fn]) + payload
    return body + crc16(body).to_bytes(2, "little")


def build_read(slave: int, fn: int, reg: int, count: int) -> bytes:
    """Build a read-holding/read-input frame (reg is 1-based)."""
    return frame(slave, fn, struct.pack(">HH", reg - 1, count))


def build_write_single(slave: int, reg: int, value: int) -> bytes:
    """Build a write-single frame (reg is 1-based)."""
    return frame(slave, WRITE_SINGLE, struct.pack(">HH", reg - 1, value))


def build_write_multiple(slave: int, reg: int, values: list[int]) -> bytes:
    """Build a write-multiple (fn 0x10) frame (reg is 1-based)."""
    payload = struct.pack(">HHB", reg - 1, len(values), len(values) * 2)
    payload += b"".join(struct.pack(">H", v) for v in values)
    return frame(slave, WRITE_MULTIPLE, payload)


def parse_frame(data: bytes, slave: int, fn: int) -> bytes:
    """Validate a response frame; returns the payload (excluding CRC).

    Raises ModbusException on exception responses, FrameError otherwise.
    """
    if len(data) < 4:
        raise FrameError(f"Frame too short: {data!r}")
    if data[-2:] != crc16(data[:-2]).to_bytes(2, "little"):
        raise FrameError("CRC mismatch")
    if data[0] != slave:
        raise FrameError(f"Slave address mismatch: {data[0]} != {slave}")
    if data[1] == fn | 0x80:
        raise ModbusException(code=data[2], fn=fn)
    if data[1] != fn:
        raise FrameError(f"Unexpected function code 0x{data[1]:02X}")
    return data[2:-2]


def parse_read_response(data: bytes, slave: int, fn: int) -> list[int]:
    """Validate a read response and return the 16-bit register values."""
    payload = parse_frame(data, slave, fn)
    if not payload or payload[0] != len(payload) - 1:
        raise FrameError("Bad read response byte count")
    return list(struct.unpack(f">{len(payload[1:]) // 2}H", payload[1:]))


def parse_write_response(data: bytes, slave: int, fn: int) -> None:
    """Validate a write acknowledgement (echo of reg/value)."""
    parse_frame(data, slave, fn)


def pack_f32(value: float) -> tuple[int, int]:
    """IEEE 754 big-endian float → (hi_reg, lo_reg)."""
    hi, lo = struct.unpack(">HH", struct.pack(">f", value))
    return hi, lo


def unpack_f32(hi: int, lo: int) -> float:
    """(hi_reg, lo_reg) → IEEE 754 big-endian float."""
    return struct.unpack(">f", struct.pack(">HH", hi, lo))[0]


def to_signed16(value: int) -> int:
    """Interpret an unsigned 16-bit register as signed two's complement."""
    return value - 0x10000 if value & 0x8000 else value
