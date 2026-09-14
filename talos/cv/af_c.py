"""AF-C (keep-focused) state machine — pure, no Qt, fully unit-testable.

HOLD → REPEAK → HOLD on slow degradation; PAUSED on user/XY interference;
STOPPED on scene change or a failed re-peak.

The user-requested scene-change guard: during HOLD there is NO motion
(XY motion and manual focus input pause AF-C), so a violent single-sample
score collapse is not drift — it means the scene itself changed (lamp
off, nosepiece rotated, sample removed). That EXITS AF-C instead of
hunting a peak that no longer exists. Slow drift (thermal/settling)
degrades gradually and triggers a small re-peak instead.

The service drives it: monitor ticks call sample(score); job results
call job_done(ok); interference calls activity(source) with a settle
timer afterwards calling idle_after_pause().
"""

from __future__ import annotations

HOLD = "HOLD"
REPEAK = "REPEAK"
PAUSED = "PAUSED"
STOPPED = "STOPPED"


class AfCStateMachine:
    def __init__(self, hysteresis: float = 0.25, degrade_samples: int = 3,
                 scene_change_ratio: float = 0.95,
                 pause_on_xy_motion: bool = True,
                 pause_on_manual_focus: bool = True,
                 resume_refocus: bool = True):
        self._hysteresis = hysteresis
        self._degrade_samples = degrade_samples
        self._scene_change_ratio = scene_change_ratio
        self._pause_on_xy_motion = pause_on_xy_motion
        self._pause_on_manual_focus = pause_on_manual_focus
        self._resume_refocus = resume_refocus
        self._state = HOLD
        self._baseline: float | None = None
        self._degrade_streak = 0

    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def has_baseline(self) -> bool:
        return self._baseline is not None

    @property
    def baseline(self) -> float | None:
        return self._baseline

    def set_baseline(self, score: float) -> None:
        """(Re-)baseline after a successful focus job."""
        self._baseline = score
        self._degrade_streak = 0

    # ------------------------------------------------------------------

    def sample(self, score: float) -> str | None:
        """One monitor tick. Returns an action: "refine",
        "stop:scene_changed", or None."""
        if self._state in (STOPPED, PAUSED, REPEAK):
            return None
        if self._baseline is None:
            return None
        # Single-sample collapse = the scene changed (lamp off, nosepiece
        # rotated, sample removed) — nothing moves during HOLD, so no
        # mechanical cause can drop the score this far this fast.
        if score < self._baseline * (1.0 - self._scene_change_ratio):
            self._state = STOPPED
            return "stop:scene_changed"
        if score < self._baseline * (1.0 - self._hysteresis):
            self._degrade_streak += 1
            if self._degrade_streak >= self._degrade_samples:
                self._state = REPEAK
                return "refine"
            return None
        self._degrade_streak = 0
        return None

    def activity(self, source: str) -> str | None:
        """User/XY interference while holding. Returns "pause" or None."""
        if self._state != HOLD:
            return None
        if source == "xy" and self._pause_on_xy_motion:
            self._state = PAUSED
            return "pause"
        if source == "manual" and self._pause_on_manual_focus:
            self._state = PAUSED
            return "pause"
        return None

    def idle_after_pause(self) -> str | None:
        """Activity sources have been idle for the settle window. Returns
        "resume_refocus" (re-peak because the sample may have moved), or
        None when just resuming the hold."""
        if self._state != PAUSED:
            return None
        if self._resume_refocus:
            self._state = REPEAK
            return "resume_refocus"
        self._state = HOLD
        return None

    def job_done(self, success: bool) -> str | None:
        """An AF job finished (REPEAK). Returns "stop:refine_failed" or
        None (the caller re-baselines on success)."""
        if self._state == REPEAK:
            if success:
                self._state = HOLD
                self._degrade_streak = 0
                return None
            self._state = STOPPED
            return "stop:refine_failed"
        return None

    def stop(self) -> None:
        self._state = STOPPED
