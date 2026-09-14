"""Software auto-gain: adjusts the camera gain toward a target mean luma
at a fixed exposure. (Auto-EXPOSURE changes the framerate — auto-GAIN
does not, which is why the UI defaults to this.)

The pure math lives in ``mean_luma``/``next_gain``; the QObject runs a
GUI-thread timer and submits gain writes through the manager. The loop
suspends whenever the app mode leaves MANUAL (autofocus/scan own the
camera) and during snapshots (a 4K capture must not fight the loop).
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

logger = logging.getLogger(__name__)

GAIN_MIN = 1.0
GAIN_MAX = 22.0


def mean_luma(frame: np.ndarray) -> float:
    """Mean luma of an RGB uint8 frame (strided sample — a 1080p frame
    costs ~2 ms, not 6)."""
    if frame is None or frame.size == 0:
        return 0.0
    sample = frame[::8, ::8]
    return float(sample.mean())


def next_gain(luma: float, target: float, current: float, *,
              min_gain: float = GAIN_MIN, max_gain: float = GAIN_MAX,
              deadband: float = 10.0, max_step: float = 2.0) -> float | None:
    """The next gain for the observed luma, or None when no change is
    needed. Half-geometric step toward the ideal ratio (stable, bounded),
    clamped, rounded to 0.1."""
    if luma <= 0 or target <= 0:
        return None
    if abs(luma - target) <= deadband:
        return None
    ratio = target / luma
    step = current * (math.sqrt(ratio) - 1.0)
    step = max(-max_step, min(step, max_step))
    candidate = max(min_gain, min(current + step, max_gain))
    candidate = round(candidate, 1)
    return candidate if abs(candidate - current) >= 0.1 else None


def should_adjust(last_adjust_t: float | None, now: float, settle_s: float,
                  samples: list[float], stability: float) -> bool:
    """Standby gating: adjust only when the settle interval since the
    last adjustment has elapsed AND the sample buffer is full and stable
    (max − min ≤ stability). The first adjustment (last_adjust_t=None)
    also requires the full buffer — no blind first step on noise."""
    if len(samples) < 3:
        return False
    if max(samples) - min(samples) > stability:
        return False  # the scene is still changing — wait it out
    if last_adjust_t is not None and now - last_adjust_t < settle_s:
        return False
    return True


class AutoGainController(QObject):
    """Ticks at ``auto_gain_interval_ms``; enabled from the camera
    profile (Navigation on, Sample Finding off)."""

    sig_gain_changed = Signal(float)

    def __init__(self, manager, settings, state, parent=None):
        super().__init__(parent)
        self._manager = manager
        self._settings = settings
        self._state = state
        cfg = settings.device("camera")
        self._target = float(cfg.get("auto_gain_target", 120.0))
        self._enabled = bool(cfg.get("auto_gain", True))
        self._capture_busy = False
        self._last_frame: np.ndarray | None = None
        # Standby pacing: the gain WRITE stalls the camera's frame
        # delivery (~330 ms per parameter query — hardware-measured), so
        # adjustments are gated to at most one per settle_s, on a stable
        # luma sample set. The loop measures every tick; it only rarely
        # writes.
        self._settle_s = float(cfg.get("auto_gain_settle_s", 1.0))
        self._stability = float(cfg.get("auto_gain_stability", 8.0))
        self._samples: list[float] = []
        self._last_adjust_t: float | None = None
        # The gain the controller last commanded/knows. manager.camera_props
        # is a connect-time snapshot (get_properties runs once) — stale
        # after live edits, so the loop tracks its own value.
        self._applied: float | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(int(cfg.get("auto_gain_interval_ms", 500)))
        self._timer.timeout.connect(self._tick)

    def set_enabled(self, on: bool) -> None:
        self._enabled = bool(on)
        if on:
            self._applied = None  # re-learn from the props/settings
            self._samples = []
            self._last_adjust_t = None  # re-gate the first adjustment

    def set_target(self, luma: float) -> None:
        self._target = float(luma)

    def note_manual_gain(self, value: float) -> None:
        """The user (or a profile apply) set the gain externally —
        restart the settle so the loop never fights the edit."""
        self._applied = float(value)
        self._samples = []
        self._last_adjust_t = time.monotonic()

    def on_frame(self, frame) -> None:
        self._last_frame = frame

    def once(self) -> bool:
        """One-shot adjustment (the "Gain once" button): measure the
        current frame and make ONE gain step toward the target, ignoring
        the settle/standby gating — an explicit user action. The
        auto-gain setting itself is untouched (a transient action, like
        the WB Once button). Returns True when a gain was written."""
        if self._capture_busy or self._last_frame is None:
            return False
        next_value = next_gain(mean_luma(self._last_frame), self._target,
                               self._current_gain())
        self._last_adjust_t = time.monotonic()
        if next_value is None:
            return False
        self._manager.submit_camera("set_property", "gain", next_value)
        self._applied = next_value
        self.sig_gain_changed.emit(next_value)
        return True

    def notify_capture_busy(self, busy: bool) -> None:
        self._capture_busy = bool(busy)

    def _current_gain(self) -> float:
        if self._applied is not None:
            return self._applied
        props = self._manager.camera_props or {}
        return float(props.get("gain", self._settings.device("camera")
                               .get("gain", 4.0)))

    def _tick(self) -> None:
        if not self._enabled or self._capture_busy or self._last_frame is None:
            return
        mode = self._state.mode if self._state is not None else "MANUAL"
        if mode != "MANUAL":
            # autofocus/scan own the camera — and the pre-suspension
            # luma samples go stale while it runs: drop them so the
            # first post-AF adjustment re-gates on fresh frames.
            self._samples = []
            return
        now = time.monotonic()
        luma = mean_luma(self._last_frame)
        self._samples.append(luma)
        if len(self._samples) > 3:
            self._samples.pop(0)
        if not should_adjust(self._last_adjust_t, now, self._settle_s,
                             self._samples, self._stability):
            return
        next_value = next_gain(sum(self._samples) / len(self._samples),
                               self._target, self._current_gain())
        if next_value is None:
            return
        self._manager.submit_camera("set_property", "gain", next_value)
        self._applied = next_value
        self._last_adjust_t = now
        self.sig_gain_changed.emit(next_value)

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
