"""YudianTempController tests against a scripted fake serial."""

import struct

import pytest

from talos.hal.base import CommandRejectedError
from talos.hal.devices.yudian import REG_DPT, REG_PV, REG_SV_READ, REG_SV_WRITE, YudianTempController
from talos.protocols.modbus_rtu import (
    READ_HOLDING,
    build_read,
    build_write_single,
    crc16,
)
from tests.testing.fake_serial import FakeSerial


def read_resp(slave: int, fn: int, values: list[int]) -> bytes:
    payload = bytes([len(values) * 2]) + b"".join(struct.pack(">H", v) for v in values)
    body = bytes([slave, fn]) + payload
    return body + crc16(body).to_bytes(2, "little")


def write_ack(slave: int, fn: int, reg: int) -> bytes:
    body = bytes([slave, fn, (reg - 1) >> 8, (reg - 1) & 0xFF, 0x00, 0x01])
    return body + crc16(body).to_bytes(2, "little")


def make_connected(extra_script=None) -> tuple[YudianTempController, FakeSerial]:
    script = {
        build_read(1, READ_HOLDING, REG_DPT, 1): read_resp(1, READ_HOLDING, [1]),
        build_read(1, READ_HOLDING, REG_PV, 1): read_resp(1, READ_HOLDING, [250]),  # 25.0 C
    }
    script.update(extra_script or {})
    fake = FakeSerial(script)
    driver = YudianTempController({"port": "COM5"})
    driver._ser = fake
    driver._connected = True
    return driver, fake


def test_connect_reads_dpt_and_pv():
    script = {
        build_read(1, READ_HOLDING, REG_DPT, 1): read_resp(1, READ_HOLDING, [1]),
        build_read(1, READ_HOLDING, REG_PV, 1): read_resp(1, READ_HOLDING, [250]),
    }
    fake = FakeSerial(script)
    # Factory injection: the driver must use the fake, never a real port.
    driver = YudianTempController({"port": "COM5", "serial_factory": lambda **kw: fake})
    driver.connect()
    assert driver.is_connected
    assert driver._ser is fake
    assert driver._dpt == 1


def test_read_pv_scaling():
    driver, _ = make_connected(extra_script={
        build_read(1, READ_HOLDING, REG_PV, 1): read_resp(1, READ_HOLDING, [250]),
    })
    assert driver.read_pv() == pytest.approx(25.0)


def test_read_pv_negative():
    driver, _ = make_connected(extra_script={
        build_read(1, READ_HOLDING, REG_PV, 1): read_resp(1, READ_HOLDING, [0xFF38]),
    })
    assert driver.read_pv() == pytest.approx(-20.0)


def test_set_sv_clamps_to_safety_range():
    driver, fake = make_connected()
    with pytest.raises(CommandRejectedError):
        driver.set_sv(500.0)
    with pytest.raises(CommandRejectedError):
        driver.set_sv(-150.0)
    assert fake.writes == []


def test_set_sv_writes_and_verifies_readback():
    driver, fake = make_connected(extra_script={
        build_write_single(1, REG_SV_WRITE, 1800): write_ack(1, 0x06, REG_SV_WRITE),
        build_read(1, READ_HOLDING, REG_SV_READ, 1): read_resp(1, READ_HOLDING, [1800]),
    })
    driver.set_sv(180.0)
    assert build_write_single(1, REG_SV_WRITE, 1800) in fake.writes


def test_set_sv_readback_mismatch_raises():
    driver, _ = make_connected(extra_script={
        build_write_single(1, REG_SV_WRITE, 1800): write_ack(1, 0x06, REG_SV_WRITE),
        build_read(1, READ_HOLDING, REG_SV_READ, 1): read_resp(1, READ_HOLDING, [999]),
    })
    with pytest.raises(CommandRejectedError, match="mismatch"):
        driver.set_sv(180.0)


def test_read_output_percent_low_byte():
    driver, _ = make_connected(extra_script={
        build_read(1, READ_HOLDING, 77, 1): read_resp(1, READ_HOLDING, [0x1234]),
    })
    assert driver.read_output_percent() == pytest.approx(0x34)
