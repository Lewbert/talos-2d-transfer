"""ZolixXYRStage tests against a scripted fake serial (Modbus frames)."""

import struct

import pytest

from talos.hal.base import (CommandRejectedError, DeviceBusyError,
                        DeviceError, EStopError, ProtocolError)
from talos.hal.devices.zolix import (
    AXIS_SEL,
    DIR_NEG,
    DIR_POS,
    AXIS_ALL,
    OP_FIXED_LENGTH,
    OP_IMMEDIATE_STOP,
    REG_ACC_X,
    REG_DIST_X,
    REG_DEVICE_MODEL,
    REG_ENABLE_X,
    REG_MOTION_STATE,
    REG_OPCODE,
    REG_POS_X,
    REG_SPEED_CONST_X,
    REG_UNIT_X,
    ZolixXYRStage,
)
from talos.protocols.modbus_rtu import (
    READ_HOLDING,
    READ_INPUT,
    build_read,
    build_write_multiple,
    build_write_single,
    crc16,
    pack_f32,
    unpack_f32,
)
from tests.testing.fake_serial import FakeSerial


def read_resp(slave: int, fn: int, values: list[int]) -> bytes:
    payload = bytes([len(values) * 2]) + b"".join(struct.pack(">H", v) for v in values)
    body = bytes([slave, fn]) + payload
    return body + crc16(body).to_bytes(2, "little")


def write_ack(slave: int, fn: int, reg: int, count: int) -> bytes:
    body = bytes([slave, fn, (reg - 1) >> 8, (reg - 1) & 0xFF, count >> 8, count & 0xFF])
    return body + crc16(body).to_bytes(2, "little")


def exc_resp(slave: int, fn: int, code: int) -> bytes:
    body = bytes([slave, fn | 0x80, code])
    return body + crc16(body).to_bytes(2, "little")


def make_connected(extra_script=None) -> tuple[ZolixXYRStage, FakeSerial]:
    script = {
        # Status block (10 regs, 30012-30021) — used by get_status/get_position.
        build_read(1, READ_INPUT, REG_MOTION_STATE, 10): read_resp(1, READ_INPUT, [0] * 10),
        build_read(1, READ_INPUT, REG_POS_X, 6): read_resp(1, READ_INPUT, [0] * 6),
        build_read(1, READ_INPUT, REG_DEVICE_MODEL, 7): read_resp(
            1, READ_INPUT, [0x5A43, 0x3330, 0x3020, 0x2020, 0, 0, 0]),
        build_read(1, READ_INPUT, REG_UNIT_X, 3): read_resp(1, READ_INPUT, [0, 0, 0]),
        build_write_single(1, REG_ENABLE_X, 1): write_ack(1, 0x06, REG_ENABLE_X, 1),
        build_write_single(1, REG_ENABLE_X + 1, 1): write_ack(1, 0x06, REG_ENABLE_X + 1, 1),
        build_write_single(1, REG_ENABLE_X + 2, 1): write_ack(1, 0x06, REG_ENABLE_X + 2, 1),
    }
    for axis in range(3):
        reg = REG_ACC_X + 2 * axis
        hi, lo = pack_f32(10_000_000.0)
        script[build_write_multiple(1, reg, [hi, lo])] = write_ack(1, 0x10, reg, 2)
    # Motion-config readback (holding reads of accel registers).
    for axis in range(3):
        reg = REG_ACC_X + 2 * axis
        hi, lo = pack_f32(10_000_000.0)
        script[build_read(1, READ_HOLDING, reg, 2)] = read_resp(1, READ_HOLDING, [hi, lo])
    script.update(extra_script or {})
    fake = FakeSerial(script)
    driver = ZolixXYRStage({"port": "COM3", "timeout_s": 0.05})
    driver._ser = fake
    driver._connected = True
    return driver, fake


def test_connect_flow():
    fake = FakeSerial({})
    script = {
        build_read(1, READ_INPUT, REG_MOTION_STATE, 3): read_resp(1, READ_INPUT, [0, 0, 0]),
        build_read(1, READ_INPUT, REG_DEVICE_MODEL, 7): read_resp(
            1, READ_INPUT, [0x5A43, 0x3330, 0x3020, 0x2020, 0, 0, 0]),
        build_read(1, READ_INPUT, REG_UNIT_X, 3): read_resp(1, READ_INPUT, [0, 0, 0]),
    }
    for axis in range(3):
        reg = REG_ENABLE_X + axis
        script[build_write_single(1, reg, 1)] = write_ack(1, 0x06, reg, 1)
    for axis in range(3):
        reg = REG_ACC_X + 2 * axis
        hi, lo = pack_f32(10_000_000.0)
        script[build_write_multiple(1, reg, [hi, lo])] = write_ack(1, 0x10, reg, 2)
        script[build_read(1, READ_HOLDING, reg, 2)] = read_resp(1, READ_HOLDING, [hi, lo])
    fake._script = script
    # Factory injection: the driver must use the fake, never a real port.
    driver = ZolixXYRStage({"port": "COM3", "timeout_s": 0.05,
                            "serial_factory": lambda **kw: fake})
    driver.connect()
    assert driver.is_connected
    assert driver._ser is fake


def test_get_status_estop_and_motion():
    values = [0, 1, 0, 0x0200]  # Y moving, estop bit 9
    values += [*pack_f32(100.0), *pack_f32(-50.0), *pack_f32(90.0)]
    driver, _ = make_connected(extra_script={
        build_read(1, READ_INPUT, REG_MOTION_STATE, 10): read_resp(1, READ_INPUT, values),
    })
    status = driver.get_status()
    assert status.y_moving and not status.x_moving
    assert status.estop
    assert driver.check_estop()


def test_get_position_converts_to_um():
    values = [*pack_f32(100.0), *pack_f32(-50.0), *pack_f32(90.0)]
    driver, _ = make_connected(extra_script={
        build_read(1, READ_INPUT, REG_POS_X, 6): read_resp(1, READ_INPUT, values),
    })
    pos = driver.get_position()
    assert pos.x_pulses == 100 and pos.x_um == pytest.approx(62.5)
    assert pos.y_um == pytest.approx(-31.25)
    assert pos.r_deg == pytest.approx(90 * 0.00125)


def test_move_rel_um_uses_fixed_length_opcode():
    driver, fake = make_connected()
    # Speed write for X (500 pps), distance write (1000 pulses = 625 um),
    # distance READBACK (verify), then the opcode.
    hi_s, lo_s = pack_f32(500.0)
    hi_d, lo_d = pack_f32(1000.0)
    fake._script[build_write_multiple(1, REG_SPEED_CONST_X, [hi_s, lo_s])] = (
        write_ack(1, 0x10, REG_SPEED_CONST_X, 2))
    fake._script[build_write_multiple(1, REG_DIST_X, [hi_d, lo_d])] = (
        write_ack(1, 0x10, REG_DIST_X, 2))
    fake._script[build_read(1, READ_HOLDING, REG_DIST_X, 2)] = (
        read_resp(1, READ_HOLDING, [hi_d, lo_d]))
    fake._script[build_write_multiple(1, REG_OPCODE, [OP_FIXED_LENGTH, AXIS_SEL["x"], DIR_POS])] = (
        write_ack(1, 0x10, REG_OPCODE, 3))
    driver.move_rel_um(625.0, 0.0)
    opcode_frame = build_write_multiple(1, REG_OPCODE, [OP_FIXED_LENGTH, AXIS_SEL["x"], DIR_POS])
    assert opcode_frame in fake.writes


def test_move_rel_um_blocks_on_distance_readback_mismatch():
    driver, fake = make_connected()
    hi_s, lo_s = pack_f32(500.0)
    hi_d, lo_d = pack_f32(1000.0)
    fake._script[build_write_multiple(1, REG_SPEED_CONST_X, [hi_s, lo_s])] = (
        write_ack(1, 0x10, REG_SPEED_CONST_X, 2))
    fake._script[build_write_multiple(1, REG_DIST_X, [hi_d, lo_d])] = (
        write_ack(1, 0x10, REG_DIST_X, 2))
    # Corrupted distance lands in the register (e.g. glitched write).
    bad_hi, bad_lo = pack_f32(999999.0)
    fake._script[build_read(1, READ_HOLDING, REG_DIST_X, 2)] = (
        read_resp(1, READ_HOLDING, [bad_hi, bad_lo]))
    with pytest.raises(ProtocolError, match="readback mismatch"):
        driver.move_rel_um(625.0, 0.0)
    # The motion opcode must NEVER have been issued.
    opcode_frame = build_write_multiple(1, REG_OPCODE, [OP_FIXED_LENGTH, AXIS_SEL["x"], DIR_POS])
    assert opcode_frame not in fake.writes


def test_move_abs_um_composes_fixed_length_moves():
    """Hardware finding: opcode 0x0064 no-ops on the real controller, so
    move_abs_um must compose validated 0x0065 fixed-length moves from the
    controller's own position readback."""
    driver, fake = make_connected(extra_script={
        build_read(1, READ_INPUT, REG_POS_X, 6): read_resp(
            1, READ_INPUT, [*pack_f32(100.0), *pack_f32(-50.0), *pack_f32(90.0)]),
    })
    hi_s, lo_s = pack_f32(500.0)
    fake._script[build_write_multiple(1, REG_SPEED_CONST_X, [hi_s, lo_s])] = (
        write_ack(1, 0x10, REG_SPEED_CONST_X, 2))
    fake._script[build_write_multiple(1, REG_SPEED_CONST_X + 2, [hi_s, lo_s])] = (
        write_ack(1, 0x10, REG_SPEED_CONST_X + 2, 2))
    hi_d100, lo_d100 = pack_f32(100.0)
    hi_d80, lo_d80 = pack_f32(80.0)  # -50 µm = -80 pulses at 0.625 µm/pulse
    fake._script[build_write_multiple(1, REG_DIST_X, [hi_d100, lo_d100])] = (
        write_ack(1, 0x10, REG_DIST_X, 2))
    fake._script[build_write_multiple(1, REG_DIST_X + 2, [hi_d80, lo_d80])] = (
        write_ack(1, 0x10, REG_DIST_X + 2, 2))
    fake._script[build_read(1, READ_HOLDING, REG_DIST_X, 2)] = (
        read_resp(1, READ_HOLDING, [hi_d100, lo_d100]))
    fake._script[build_read(1, READ_HOLDING, REG_DIST_X + 2, 2)] = (
        read_resp(1, READ_HOLDING, [hi_d80, lo_d80]))
    fake._script[build_write_multiple(1, REG_OPCODE, [OP_FIXED_LENGTH, AXIS_SEL["x"], DIR_POS])] = (
        write_ack(1, 0x10, REG_OPCODE, 3))
    fake._script[build_write_multiple(1, REG_OPCODE, [OP_FIXED_LENGTH, AXIS_SEL["y"], DIR_NEG])] = (
        write_ack(1, 0x10, REG_OPCODE, 3))
    # Target 125.0 µm X / -81.25 µm Y: deltas are +100 pulses X, -80 pulses Y.
    driver.move_abs_um(125.0, -81.25)
    assert build_write_multiple(1, REG_OPCODE, [OP_FIXED_LENGTH, AXIS_SEL["x"], DIR_POS]) in fake.writes
    assert build_write_multiple(1, REG_OPCODE, [OP_FIXED_LENGTH, AXIS_SEL["y"], DIR_NEG]) in fake.writes
    # Never the unvalidated absolute opcode.
    assert not any(b[2:4] == b"\x00d" and b[4] == 0x10 and OP_FIXED_LENGTH not in b
                   for b in fake.writes)


def test_move_while_moving_raises_busy():
    driver, _ = make_connected(extra_script={
        build_read(1, READ_INPUT, REG_MOTION_STATE, 10): read_resp(
            1, READ_INPUT, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    })
    with pytest.raises(DeviceBusyError):
        driver.move_rel_um(10.0, 0.0)


def test_move_with_estop_raises():
    driver, _ = make_connected(extra_script={
        build_read(1, READ_INPUT, REG_MOTION_STATE, 10): read_resp(
            1, READ_INPUT, [0, 0, 0, 0x0200, 0, 0, 0, 0, 0, 0]),
    })
    with pytest.raises(EStopError):
        driver.move_rel_um(10.0, 0.0)


def test_write_exception_translates_to_command_rejected():
    driver, fake = make_connected()
    fake._script[build_write_multiple(1, REG_SPEED_CONST_X, list(pack_f32(500.0)))] = (
        exc_resp(1, 0x10, 0x03))
    with pytest.raises(CommandRejectedError):
        driver.move_rel_um(625.0, 0.0)


def test_wait_idle_polls_until_stopped():
    driver, _ = make_connected(extra_script={
        build_read(1, READ_INPUT, REG_MOTION_STATE, 10): [
            read_resp(1, READ_INPUT, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            read_resp(1, READ_INPUT, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ],
    })
    driver.wait_idle(timeout_s=2.0, poll_s=0.01)


def test_stop_never_raises_disconnected():
    driver = ZolixXYRStage({"port": "COM3"})
    driver._connected = False
    driver.stop()  # must not raise


def test_stop_returns_as_soon_as_the_axes_halt():
    """The stop path slept a fixed 0.3 s + 0.2 s waiting for the axes to
    halt — on EVERY release-stop and inside the 1.5 s STOP ALL budget.
    It now returns as soon as the controller reports no axis moving."""
    import time

    driver, _ = make_connected(extra_script={
        build_write_multiple(1, REG_OPCODE, [OP_IMMEDIATE_STOP, AXIS_ALL]):
            write_ack(1, 0x10, REG_OPCODE, 2),
        build_read(1, READ_INPUT, REG_MOTION_STATE, 10):
            read_resp(1, READ_INPUT, [0] * 10),   # already stopped
    })
    started = time.monotonic()
    driver.stop()
    assert time.monotonic() - started < 0.2, "stop must not sleep the full settle"


def test_stop_escalates_when_axes_keep_moving():
    driver, _ = make_connected(extra_script={
        build_write_multiple(1, REG_OPCODE, [OP_IMMEDIATE_STOP, AXIS_ALL]):
            write_ack(1, 0x10, REG_OPCODE, 2),
        # the status reads keep reporting X moving, then it halts
        build_read(1, READ_INPUT, REG_MOTION_STATE, 10): [
            read_resp(1, READ_INPUT, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            read_resp(1, READ_INPUT, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            read_resp(1, READ_INPUT, [0] * 10),
        ],
    })
    driver.stop()  # must not raise, must not hang


# --- transient link faults --------------------------------------------------

def test_a_truncated_read_reply_is_retried():
    """Bench, 2026-09-18: a `get_position()` came back as two bytes and
    then silence — the slave address and the function code, nothing else —
    from a controller that was healthy. That ONE bad exchange failed the
    read, the adapter wrapped it as a device error, and the scan stopped
    halfway through.

    A read has no side effects, so the honest response to a fragment is to
    ask again.
    """
    values = [*pack_f32(100.0), *pack_f32(-50.0), *pack_f32(90.0)]
    request = build_read(1, READ_INPUT, REG_POS_X, 6)
    driver, fake = make_connected(extra_script={
        # first exchange: header only; second: the real reply
        request: [b"\x01\x04", read_resp(1, READ_INPUT, values)],
    })
    pos = driver.get_position()
    assert pos.x_pulses == 100
    assert fake.writes.count(request) == 2, "the read must be re-sent once"


def test_a_reply_that_never_completes_still_fails_the_read():
    """Retrying is not the same as inventing an answer: a link that keeps
    truncating must fail, so the caller knows the position is unknown."""
    request = build_read(1, READ_INPUT, REG_POS_X, 6)
    driver, fake = make_connected(extra_script={request: b"\x01\x04"})
    with pytest.raises(DeviceError):
        driver.get_position()
    assert fake.writes.count(request) == 2      # tried, then gave up


def test_a_truncated_WRITE_reply_is_not_retried():
    """The other half of the rule, and the more important one: a truncated
    write response means we do not know whether the write landed, so the
    driver must NOT send it again (a motion command issued twice is a
    second move). The fragment goes to the parser and the command fails."""
    driver, fake = make_connected()
    req = fake.writes
    with pytest.raises(DeviceError):
        # enable X: scripted to answer with a fragment
        fake._script[build_write_single(1, REG_ENABLE_X, 1)] = b"\x01\x06"
        driver._write_single(REG_ENABLE_X, 1)
    assert fake.writes.count(build_write_single(1, REG_ENABLE_X, 1)) == 1
    del req
