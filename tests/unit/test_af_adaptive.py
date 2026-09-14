"""cont_pass unit tests: a scripted CONT-mode fake focus + a scripted
frame reader + a fake clock (monkeypatched time) so every loop of the
pass engine is deterministic — interpolation, speed-schedule throttling,
early-stop ramp-halts, predictive edge stop, events, aborts, and the
belt-and-braces stop path."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from talos.cv.af_adaptive import cont_pass
from talos.hal.base import DeviceError, DeviceTimeoutError


class Clock:
    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += s


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr("time.monotonic", c.monotonic)
    monkeypatch.setattr("time.sleep", c.sleep)
    return c


class FakeContFocus:
    """CONT-mode model: pos advances by v×poll_s per status poll;
    set_speed(0) idles instantly (like SimFocusStage); scripted
    EV events; scriptable poll failures and a wait_idle failure."""

    def __init__(self, clock, poll_s=0.02):
        self.clock = clock
        self.poll_s = poll_s
        self.pos = 0
        self.v = 0
        self.mode = "IDLE"
        self.speeds: list[int] = []        # set_speed values
        self.speed_times: list[float] = []
        self.events: list[str] = []
        self.stopped = 0
        self.move_calls: list[tuple] = []
        self.status_calls = 0
        self.fail_status = 0               # fail the first N status polls
        self.fail_wait_idle = False
        self.base_pos = 0                  # position folded at the last speed change
        self.v_start_t = None

    def move_abs(self, position, speed=None):
        self.move_calls.append((int(position), speed))
        self.base_pos = int(position)
        self.mode = "IDLE"
        self.v = 0
        self.v_start_t = None

    def set_speed(self, v):
        # Continuous-position model: pos(t) = base + v×(t − v_start_t),
        # folded per segment. Per-poll int() truncation of a small float
        # difference (1000.02 − 1000.0 = 0.019999… → int = 1!) would
        # silently halve the speed — the fake must integrate exactly.
        if self.mode == "CONT" and self.v and self.v_start_t is not None:
            self.base_pos += int(round(
                self.v * (self.clock.t - self.v_start_t)))
        self.speeds.append(int(v))
        self.speed_times.append(self.clock.t)
        self.v = int(v)
        self.v_start_t = self.clock.t
        self.mode = "IDLE" if v == 0 else "CONT"

    def wait_idle(self, timeout_s=1.5, poll_s=0.02):
        if self.fail_wait_idle:
            raise DeviceTimeoutError("fake wait_idle timeout")
        if self.v and self.v_start_t is not None:
            self.base_pos += int(round(
                self.v * (self.clock.t - self.v_start_t)))
        self.v = 0
        self.mode = "IDLE"

    def get_status(self):
        self.status_calls += 1
        if self.fail_status:
            self.fail_status -= 1
            raise DeviceError("fake poll noise")
        pos = self.base_pos
        if self.mode == "CONT" and self.v and self.v_start_t is not None:
            pos += int(round(self.v * (self.clock.t - self.v_start_t)))
        return SimpleNamespace(pos=pos, mode=self.mode,
                               is_idle=(self.mode == "IDLE"), v=abs(self.v),
                               spd=self.v)

    def drain_events(self):
        events, self.events = self.events, []
        return events

    def stop(self):
        self.stopped += 1
        self.v = 0
        self.mode = "IDLE"


class ScriptedFrames:
    def __init__(self, items):
        self._items = list(items)
        self._i = 0

    def read_since(self, min_t=None, min_seq=-1):
        if self._i >= len(self._items):
            return None
        item = self._items[self._i]
        self._i += 1
        return item


def make_frames(clock, scores, poll_s=0.02):
    """Frame i is delivered at loop iteration i, captured halfway between
    the (i-1)-th and i-th status polls — inside the interpolation
    history."""
    t0 = clock.t
    items = []
    for i, score in enumerate(scores):
        t_cap = t0 + (i - 0.5) * poll_s
        meta = SimpleNamespace(t_capture=t_cap, seq=i)
        items.append((np.zeros((8, 8), np.uint8), meta))
    return ScriptedFrames(items)


def run_pass(clock, focus, frames, scores_fn, span=200, v0=100,
             speed_schedule=None, check=None, early_stop_ratio=0.0,
             early_stop_samples=2, guard=None):
    curve, reason, end_pos = cont_pass(
        focus, frames, start=0, direction=1, span=span, v0=v0,
        speed_schedule=speed_schedule,
        check=check or (lambda: None),
        on_score=lambda pos, frame: scores_fn(),
        on_log=None, poll_s=0.02, freshness_ms=400.0, interp_max_gap_ms=300.0,
        early_stop_ratio=early_stop_ratio, early_stop_samples=early_stop_samples,
        stop_accel=20000.0, stop_latency_s=0.05, stop_safety_steps=10,
        guard=guard)
    return curve, reason, end_pos


# ---------------------------------------------------------------------------

def test_cont_pass_scores_at_interpolated_positions(clock):
    focus = FakeContFocus(clock)
    frames = make_frames(clock, scores=[1, 2, 3, 4])
    it = iter([10.0, 20.0, 30.0, 40.0])
    curve, reason, end_pos = run_pass(clock, focus, frames, lambda: next(it))
    assert reason is None
    # frame i interpolates between history[i-1] and history[i]: pos = 2i-1
    assert len(curve) == 3  # frame 0 predates the history — skipped
    assert curve[0][0] == pytest.approx(1.0)
    assert curve[1][0] == pytest.approx(3.0)
    assert curve[2][0] == pytest.approx(5.0)
    assert focus.speeds[0] == 100           # v0 issued, no schedule changes


def test_cont_pass_early_stop_ramp_halts(clock):
    focus = FakeContFocus(clock)
    # rise to 100 then fall below 40 (0.4 × max) twice → early stop
    scores = [10, 15, 30, 60, 90, 100, 80, 55, 42, 35, 38, 40]
    frames = make_frames(clock, scores=scores)
    it = iter(scores)
    curve, reason, end_pos = run_pass(clock, focus, frames, lambda: next(it),
                                      early_stop_ratio=0.6,
                                      early_stop_samples=2)
    assert reason is None
    assert focus.speeds == [100, 0]         # v0 then the ramp stop
    assert focus.stopped == 0               # STOP is NOT the primary stop
    assert focus.mode == "IDLE"
    # the peak is inside the scored curve; the pass stopped just past it
    assert max(s for _, s in curve) == 100
    assert end_pos < 200


def test_cont_pass_speed_schedule_throttled_to_10hz(clock):
    focus = FakeContFocus(clock)
    scores = [50] * 15
    frames = make_frames(clock, scores=scores)
    calls = []

    def schedule(pos, frame, score, running_max):
        calls.append(score)
        return 100 - 10 * len(calls)

    it = iter(scores)
    run_pass(clock, focus, frames, lambda: next(it), speed_schedule=schedule)
    # updates issue only when the value changes AND ≥ 0.1 s since the
    # last: with 15 frames at 20 ms the schedule fires at frames 1, 6, 11
    assert focus.speeds == [100, 90, 80, 70, 0]
    # exclude the initial v0 and the ramp-stop 0 — schedule updates only
    update_times = focus.speed_times[1:-1]
    gaps = [b - a for a, b in zip(update_times, update_times[1:])]
    assert gaps and all(g >= 0.1 - 1e-9 for g in gaps)


def test_cont_pass_noise_dip_during_climb_does_not_stop(clock):
    """The peak-seen gate: a noise dip below the early-stop threshold
    BEFORE any real rise above the curve start must not stop the pass
    (user-observed: the climb stopped too early on noisy fields). The
    pass runs to the predictive edge stop instead."""
    focus = FakeContFocus(clock)
    scores = [50, 52, 48, 55, 51, 20, 21, 53, 50, 47, 54, 50, 52, 49, 51]
    frames = make_frames(clock, scores=scores)
    it = iter(scores)
    curve, reason, end_pos = run_pass(clock, focus, frames, lambda: next(it),
                                      span=200, early_stop_ratio=0.6)
    assert reason is None
    # the dip (20/21 below 0.4×55) did NOT stop the pass: every scripted
    # frame was scored and the pass ran on to the far edge stop
    # (without the gate it would have stopped at the 7th frame)
    assert len(curve) == 14
    assert end_pos > 150
    assert focus.speeds == [100, 0]


def test_cont_pass_predictive_edge_stop(clock):
    focus = FakeContFocus(clock)
    frames = make_frames(clock, scores=[10] * 40)
    it = iter([10] * 40)
    curve, reason, end_pos = run_pass(clock, focus, frames, lambda: next(it),
                                      span=30, early_stop_ratio=0.0)
    assert reason is None
    # stop_at = 100²/40000 + 100×0.07 + 10 ≈ 17.25 — the halt fired
    # before the 30-step edge
    assert end_pos < 30
    assert focus.speeds == [100, 0]
    assert focus.stopped == 0


def test_cont_pass_ev_tmo_and_ev_lim_hard_stop(clock):
    for event in ("EV:TMO:120", "EV:LIM:+"):
        focus = FakeContFocus(clock)
        focus.events.append(event)
        frames = make_frames(clock, scores=[10] * 6)
        it = iter([10] * 6)
        _curve, reason, _end = run_pass(clock, focus, frames,
                                        lambda: next(it))
        assert reason is not None and event.split(":")[0] in reason
        assert focus.stopped == 1           # events hard-stop


def test_cont_pass_abort_halts_and_returns_reason(clock):
    focus = FakeContFocus(clock)
    frames = make_frames(clock, scores=[10] * 20)
    it = iter([10] * 20)
    state = {"aborted": False}

    def check():
        if focus.status_calls >= 5:
            state["aborted"] = True
        return "aborted by user" if state["aborted"] else None

    _curve, reason, _end = run_pass(clock, focus, frames, lambda: next(it),
                                    check=check)
    assert reason == "aborted by user"
    assert focus.speeds[-1] == 0            # ramp halt
    assert focus.mode == "IDLE"


def test_cont_pass_transient_poll_errors_tolerated(clock):
    focus = FakeContFocus(clock)
    focus.fail_status = 2                   # first 2 polls fail
    frames = make_frames(clock, scores=[10] * 10)
    it = iter([10] * 10)
    curve, reason, _end = run_pass(clock, focus, frames, lambda: next(it))
    assert reason is None
    assert len(curve) >= 6                  # scoring resumed after the noise


def test_ramp_halt_belt_and_braces_stop_on_timeout(clock):
    focus = FakeContFocus(clock)
    focus.fail_wait_idle = True
    frames = make_frames(clock, scores=[10] * 8)
    it = iter([10] * 8)
    curve, reason, end_pos = run_pass(clock, focus, frames, lambda: next(it),
                                      early_stop_ratio=0.6)
    assert reason is None
    assert focus.stopped == 1               # wait_idle failed → hard stop
    assert focus.mode == "IDLE"


def test_cont_pass_guard_halts_and_preserves_curve(clock):
    """The guard hook halts the pass exactly like a check() reason: ramp
    stop (not STOP), stage at rest, curve preserved for the caller."""
    focus = FakeContFocus(clock)
    scores = [10] * 20
    frames = make_frames(clock, scores=scores)
    it = iter(scores)
    curve, reason, _end = run_pass(
        clock, focus, frames, lambda: next(it),
        guard=lambda c: "direction guard: peak on the other side"
        if len(c) >= 4 else None)
    assert reason == "direction guard: peak on the other side"
    assert len(curve) == 4                  # curve handed back, not discarded
    assert focus.speeds == [100, 0]         # ramp stop only
    assert focus.stopped == 0
    assert focus.mode == "IDLE"


def test_cont_pass_guard_none_runs_on(clock):
    """guard=None (and a None-returning guard) change nothing — the pass
    runs to the predictive edge stop."""
    focus = FakeContFocus(clock)
    frames = make_frames(clock, scores=[10] * 40)
    it = iter([10] * 40)
    curve, reason, end_pos = run_pass(
        clock, focus, frames, lambda: next(it), span=30, guard=lambda c: None)
    assert reason is None
    assert end_pos < 30                     # edge stop as usual
    assert focus.speeds == [100, 0]
