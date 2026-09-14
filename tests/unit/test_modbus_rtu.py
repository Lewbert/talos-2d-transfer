"""Codec tests: CRC vectors, frame round-trips, exception parsing, floats."""

import struct

import pytest

from talos.protocols.modbus_rtu import (
    FrameError,
    ModbusException,
    READ_HOLDING,
    READ_INPUT,
    WRITE_MULTIPLE,
    WRITE_SINGLE,
    build_read,
    build_write_multiple,
    build_write_single,
    crc16,
    frame,
    pack_f32,
    parse_frame,
    parse_read_response,
    parse_write_response,
    to_signed16,
    unpack_f32,
)


def test_crc16_spec_vector():
    # Classic Modbus spec example: 01 03 00 00 00 0A -> CRC C5 CD (low byte first).
    data = bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x0A])
    assert crc16(data) == 0xCDC5
    assert frame(0x01, 0x03, b"\x00\x00\x00\x0A") == data + b"\xC5\xCD"


def test_build_read_uses_zero_based_wire_address():
    # Register 1 (doc number 40001 in holding area) -> wire address 0.
    req = build_read(1, READ_HOLDING, 1, 1)
    assert req[2:4] == b"\x00\x00"
    # Register 30050 (input area) -> wire address 30049 = 0x7561.
    req = build_read(1, READ_INPUT, 30050, 1)
    assert req[2:4] == b"\x75\x61"


def test_read_response_roundtrip():
    resp = frame(1, READ_INPUT, b"\x06\x12\x34\x56\x78\x9A\xBC")
    values = parse_read_response(resp, 1, READ_INPUT)
    assert values == [0x1234, 0x5678, 0x9ABC]


def test_write_single_and_multiple_roundtrip():
    req = build_write_single(2, 40001, 250)
    assert req[0] == 2 and req[1] == WRITE_SINGLE
    parse_write_response(frame(2, WRITE_SINGLE, req[2:-2]), 2, WRITE_SINGLE)  # echo

    req = build_write_multiple(1, 30050, [0x0064, 0x31, 0x50])
    assert req[1] == WRITE_MULTIPLE
    parse_write_response(frame(1, WRITE_MULTIPLE, req[2:-2]), 1, WRITE_MULTIPLE)


def test_exception_response():
    resp = frame(1, 0x03 | 0x80, b"\x06")
    with pytest.raises(ModbusException) as exc_info:
        parse_read_response(resp, 1, READ_HOLDING)
    assert exc_info.value.code == 0x06
    assert "busy" in str(exc_info.value)


def test_crc_mismatch_raises():
    resp = frame(1, READ_INPUT, b"\x02\x00\x01")
    resp = resp[:-2] + b"\x00\x00"  # corrupt CRC
    with pytest.raises(FrameError, match="CRC"):
        parse_read_response(resp, 1, READ_INPUT)


def test_slave_mismatch_raises():
    resp = frame(1, READ_INPUT, b"\x02\x00\x01")
    with pytest.raises(FrameError, match="Slave"):
        parse_read_response(resp, 2, READ_INPUT)


@pytest.mark.parametrize("value", [0.625, 0.00125, -1.5e-3, 10_000_000.0, 1422.767216977564])
def test_f32_roundtrip(value):
    hi, lo = pack_f32(value)
    assert unpack_f32(hi, lo) == pytest.approx(value, rel=1e-6)


def test_f32_big_endian_encoding():
    hi, lo = pack_f32(1.0)
    assert hi == 0x3F80 and lo == 0x0000  # IEEE 754 BE 1.0


def test_signed16():
    assert to_signed16(0x0000) == 0
    assert to_signed16(0x00FA) == 250
    assert to_signed16(0xFF38) == -200
    assert to_signed16(0x7FFF) == 32767
    assert to_signed16(0x8000) == -32768


def test_write_multiple_packs_registers():
    req = build_write_multiple(1, 30050, [0x0064, 0x31, 0x50])
    # addr hi/lo (30049), count 3, byte count 6, data
    assert struct.unpack(">HHB", req[2:7]) == (30049, 3, 6)
    assert req[7:] == struct.pack(">3H", 0x0064, 0x31, 0x50) + crc16(req[:-2]).to_bytes(2, "little")
