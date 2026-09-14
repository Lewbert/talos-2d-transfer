"""Shared helpers for simulated devices."""

from __future__ import annotations

import time


def now() -> float:
    return time.monotonic()


def move_duration(steps: float, speed: float) -> float:
    """Simulated travel time in seconds (never zero, never huge)."""
    if speed <= 0:
        return 0.0
    return max(0.001, min(30.0, abs(steps) / speed))
