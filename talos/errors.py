"""Map device exceptions to concise user-facing messages."""

from __future__ import annotations

from talos.hal.base import (
    CommandRejectedError,
    DeviceBusyError,
    DeviceConnectionError,
    DeviceError,
    DeviceTimeoutError,
    EStopError,
    LimitHitError,
    NotConnectedError,
    ProtocolError,
)


def user_message(exc: Exception) -> str:
    """Return a short, user-facing description of a device error."""
    device = getattr(exc, "device_id", "") or ""
    where = f" ({device})" if device else ""
    if isinstance(exc, LimitHitError):
        return f"Limit reached — motion blocked{where}."
    if isinstance(exc, EStopError):
        return f"Emergency stop active{where}."
    if isinstance(exc, DeviceBusyError):
        return f"Device busy — try again after the current motion{where}."
    if isinstance(exc, CommandRejectedError):
        return f"Command rejected by device{where}: {exc}"
    if isinstance(exc, DeviceConnectionError):
        return f"Cannot connect{where}: {exc}"
    if isinstance(exc, DeviceTimeoutError):
        return f"No response from device{where}: {exc}"
    if isinstance(exc, ProtocolError):
        return f"Protocol error{where}: {exc}"
    if isinstance(exc, NotConnectedError):
        return f"Device not connected{where}."
    if isinstance(exc, DeviceError):
        return f"Device error{where}: {exc}"
    return str(exc)
