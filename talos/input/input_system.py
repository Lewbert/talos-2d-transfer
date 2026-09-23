"""Input system: 60 Hz driver running the (reference-ported) ActionResolver
and dispatching its command tuples to the InstrumentManager.

Command tuple: (stage_id, axis, mode, direction, speed, source).
Dispatch table mirrors the reference instruments.execute() semantics.
Keyboard keys are tracked in Qt (MainWindow feeds press/release events);
Esc = STOP ALL + latch.
"""

from __future__ import annotations

import logging
import time

from PySide6.QtCore import QObject, QTimer, Signal

from talos.hal.base import Axis, AxisMask, Direction
from talos.hal.devices.sigmakoki import hz_to_level
from talos.input.action_resolver import ActionResolver
from talos.input.axis_map import IDENTITY, axis_maps
from talos.input.gamepad import GamepadController, GamepadState

logger = logging.getLogger(__name__)

POLL_HZ = 60
RELEASE_CHURN_S = 0.05  # release+press gap below this = auto-repeat churn
# Gamepad combo gestures: hold BOTH controls this long before the action
# fires (a brush past a bumper must not stop a running job), and both
# inputs must pass this level for a trigger pair to count as a gesture.
COMBO_HOLD_S = 0.20
#: DEFAULT for the ``input.gamepad.trigger_threshold`` setting (Preferences →
#: Input & Gamepad). The level a trigger pair must reach to count as a
#: gesture is user-editable because worn triggers rest well above zero.
COMBO_TRIGGER_LEVEL = 0.5
#: A fired gesture keeps owning its inputs until they fall below this —
#: releasing LT/RT a few ms apart must not leave one of them jogging (see
#: InputSystem._check_combos). Deliberately NOT the press threshold: the
#: release is about "the trigger is physically back", which does not move
#: when the operator tunes the press level.
COMBO_RELEASE_LEVEL = 0.05
# Dead-man switch for on-screen holds: a hold whose release signal is lost
# must not jog the axis forever. Longer than any plausible manual hold.
MAX_UI_HOLD_S = 60.0


class InputSystem(QObject):
    sig_log = Signal(str)
    sig_dpad_stage = Signal(str)
    sig_af_requested = Signal()   # gamepad LT+RT: run autofocus once

    def __init__(self, manager, settings, state=None, parent: QObject | None = None):
        super().__init__(parent)
        self._manager = manager
        self._settings = settings
        # Optional AppState: while SCAN owns the axes manual jog is
        # BLOCKED; during AUTOFOCUS the input passes and ABORTS the run
        # (the service's sig_job_submitted hook). Esc / STOP ALL bypass
        # the dispatch (they do not go through _dispatch).
        self._state = state
        self._key_state: dict[str, float] = {}
        self._prev_key_state: dict[str, float] = {}
        self._released_times: dict[str, float] = {}
        self._esc_latch = False
        self._last_focus_speed: int | None = None
        self._last_focus_submit_t = 0.0
        self._last_continuous: dict[str, tuple] = {}
        self._ui_state: dict[str, tuple] = {}
        self._last_ui_step: dict[str, float] = {}
        # Combo-gesture state: when each gesture started, and which ones
        # have already fired for the current press.
        self._combo_since: dict[str, float] = {}
        self._combo_fired: set[str] = set()
        # Gestures that FIRED and whose inputs are not released yet — they
        # keep owning those inputs (see _check_combos).
        self._combo_latch: set[str] = set()
        self._resolver = ActionResolver(settings, state=state)
        # Manual-control axis mapping (invert / flip X↔Y), refreshed by
        # reload_settings() — see talos.input.axis_map for the scope.
        self._axis_maps = axis_maps(settings)
        self._trigger_level = COMBO_TRIGGER_LEVEL
        self._load_settings()
        self.gamepad = GamepadController(settings)
        self.gamepad.sig_state.connect(self._on_state)
        # Seed a disconnected state so the tick loop never depends on a
        # gamepad being present: sig_state is only emitted while one is
        # connected, so without this seed keyboard/UI input was dead until
        # the first gamepad poll (forever, with no gamepad plugged in).
        self._gamepad_state = GamepadState()
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(1000 // self._loop_rate_hz())
        self._tick_timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------

    def start(self) -> None:
        available = self.gamepad.start()
        self.sig_log.emit("gamepad connected" if available
                          else "no gamepad — keyboard/mouse only")
        self._tick_timer.start()

    def stop(self) -> None:
        """Teardown: the 60 Hz tick and the XInput poll both keep running
        otherwise (they only go inert because the event loop has returned,
        which is not true on the ragged-exit path). Called from app.shutdown
        BEFORE the manager's workers are torn down — the tick dispatches into
        them."""
        self._tick_timer.stop()
        self.gamepad.stop()
        self.cancel_all_holds("shutdown")

    def _loop_rate_hz(self) -> int:
        """The settings' input loop rate, clamped to a sane band (the key
        used to be ignored — the loop always ran at the POLL_HZ constant)."""
        try:
            rate = int(self._settings.section("input").get(
                "loop_rate_hz", POLL_HZ) or POLL_HZ)
        except (TypeError, ValueError):
            rate = POLL_HZ
        return max(10, min(rate, 240))

    def reload_settings(self) -> None:
        """Pick up edited manual-control settings without a restart: the
        axis mapping (invert / flip X↔Y), the jog speeds, the focus
        trigger curve, the gamepad curve, the combo trigger threshold and
        the loop rate — all of them were construction-cached before, so a
        Preferences Apply needed an app restart to take effect."""
        self._axis_maps = axis_maps(self._settings)
        self._load_settings()
        self._resolver.reload_settings()
        self.gamepad.reload_settings()
        self._tick_timer.setInterval(1000 // self._loop_rate_hz())

    def _load_settings(self) -> None:
        """The combo gesture's trigger level — the one input setting the
        dispatcher itself reads (the rest live in the resolver and the
        gamepad controller). ``input.gamepad.trigger_threshold`` shipped as a
        Preferences knob that nothing read: the gestures used the module
        constant, so the field was inert."""
        cfg = self._settings.section("input").get("gamepad", {})
        try:
            level = float(cfg.get("trigger_threshold", COMBO_TRIGGER_LEVEL))
        except (TypeError, ValueError):
            level = COMBO_TRIGGER_LEVEL
        # A resting trigger sits slightly above zero; a level at or below
        # that would read as "held" from the moment the pad is polled.
        self._trigger_level = min(max(level, 0.05), 1.0)

    # --- keyboard feeding (from MainWindow) ------------------------------

    def key_down(self, keysym: str) -> None:
        now = time.monotonic()
        if keysym in self._key_state:
            return
        # Auto-repeat churn (some IMEs deliver repeats as release+press
        # pairs) must not refresh the press timestamp: the release would
        # then look like a short press and fire a phantom single step
        # after a long jog (user-found on hardware: zolix always added a
        # step after continuous movement stopped). A re-press within
        # RELEASE_CHURN_S of the release keeps the ORIGINAL press time.
        prev_t = self._released_times.pop(keysym, 0.0)
        self._key_state[keysym] = (prev_t if prev_t and now - prev_t < RELEASE_CHURN_S
                                   else now)

    def key_up(self, keysym: str) -> None:
        t = self._key_state.pop(keysym, None)
        if t is not None:
            self._released_times[keysym] = t

    @property
    def esc_latched(self) -> bool:
        """Is the Esc latch up? Motion is suppressed until every source is
        released — and any command that does NOT come through ``_dispatch``
        (the stage panel's ZERO/home) has to ask this itself."""
        return self._esc_latch

    def on_escape(self) -> None:
        self._esc_latch = True
        # Drop every on-screen hold too: a claim that survives ESC would
        # re-command its axis as soon as the latch clears (the latch only
        # suppresses NEW commands, and the resolver re-emits a held claim
        # every tick).
        self.cancel_all_holds()
        self._manager.stop_all()

    def cancel_all_holds(self, reason: str = "stop requested") -> None:
        """Fail-safe: drop every on-screen hold.

        The resolver emits a ``continuous_stop`` for a claim that has
        disappeared, so clearing here stops the axis on the next tick.
        Needed because a hold's only release path is the widget's
        ``released`` signal — a lost one (window hidden mid-hold, focus
        stolen, widget destroyed) left the axis jogging with no button
        pressed, and a later speed change re-commanded it.
        """
        if not self._ui_state:
            return
        held = ", ".join(sorted(self._ui_state))
        self._ui_state.clear()
        logger.info("UI holds cancelled (%s): %s", reason, held)
        self.sig_log.emit(f"on-screen holds released ({reason})")

    # --- on-screen button feeding (reference UI-button branch) ------------

    def ui_hold(self, claim_key: str, direction: int,
                fast: bool = False) -> None:
        """Register an on-screen hold (claim_key = 'stage:axis'). The
        optional fast flag accelerates focus holds (the dialbox's
        double-arrow buttons); existing 2-arg callers stay slow."""
        self._ui_state[claim_key] = (time.monotonic(), direction, bool(fast))

    def ui_release(self, claim_key: str) -> None:
        self._ui_state.pop(claim_key, None)

    def ui_click(self, stage_id: str, axis: str, direction: int) -> None:
        """Short click on a UI button → single step (0.2 s cooldown).

        Routed through ``_dispatch`` like every other motion source: the
        click used to call ``_do_single_step`` directly, bypassing both
        the mode gate and the ESC latch — a jog button could step the
        stage in the middle of a grid scan (which the workspace docstring
        promises is refused)."""
        now = time.monotonic()
        key = f"{stage_id}:{axis}"
        if now - self._last_ui_step.get(key, 0) < 0.2:
            return
        # Only a command that actually goes out starts the cooldown: a
        # click refused by the mode gate must not swallow the next one.
        if self._dispatch((stage_id, axis, "single_step", direction, 0.0, "ui_click")):
            self._last_ui_step[key] = now

    # ------------------------------------------------------------------

    def _on_state(self, state) -> None:
        # Reference mapping: Back cycles the D-pad stage, Start toggles the
        # enable gate of the stage Back selected. Disabling stops that stage
        # and drops its commands (manager.set_enabled), so Start still halts
        # the axis you are driving — but the GLOBAL emergency stop is Esc.
        if state.edges.get("back"):
            new_stage = self._resolver.toggle_dpad_stage()
            self.sig_dpad_stage.emit(new_stage)
            self.sig_log.emit(
                f"gamepad D-pad stage: {new_stage} "
                f"({'enabled' if self._manager.is_enabled(new_stage) else 'DISABLED'})")
        if state.edges.get("start"):
            stage_id = self._resolver.dpad_stage
            mode = getattr(self._state, "mode", "MANUAL")
            if mode != "MANUAL" and self._manager.is_enabled(stage_id):
                # Disabling a stage mid-scan drops every later command
                # (manager.submit refuses), which the scan reports as a dead
                # device and stops the run on. Enabling one back is harmless.
                # The emergency stop is LB+RB, not this.
                self.sig_log.emit(
                    f"gamepad Start: refused — {mode} owns the axes")
            else:
                enabled = not self._manager.is_enabled(stage_id)
                self._manager.set_enabled(stage_id, enabled)
                self.sig_log.emit(
                    f"gamepad Start: {stage_id} "
                    f"{'enabled' if enabled else 'DISABLED — commands dropped'}")
        self._check_combos(state)
        self._gamepad_state = state

    # --- combo gestures (both triggers = AF, both bumpers = stop) ---------

    def _check_combos(self, state) -> None:
        """Two-finger gestures, held briefly so a brush cannot fire them.

        LT+RT: autofocus once (the triggers' focus jog is suppressed — see
        ActionResolver.resolve). LB+RB: STOP ALL, with the Esc latch, so
        nothing restarts until everything is released.

        A gesture OWNS its inputs until they are actually released, not
        just while both are held: releasing LT and RT a few milliseconds
        apart leaves the residual trigger commanding a focus jog, and any
        focus command ABORTS the autofocus the gesture just asked for —
        usually inside its 350 ms arm window, so the run never starts and
        nothing appears on screen at all (hardware symptom).
        """
        now = time.monotonic()
        lt = float(state.left_trigger)
        rt = float(state.right_trigger)
        lb = bool(state.button_left_shoulder)
        rb = bool(state.button_right_shoulder)
        both_triggers = lt >= self._trigger_level and rt >= self._trigger_level
        both_bumpers = lb and rb
        self._arm_combo("af", both_triggers,
                        lt < COMBO_RELEASE_LEVEL and rt < COMBO_RELEASE_LEVEL,
                        now, self._fire_af_combo)
        self._arm_combo("stop", both_bumpers, not lb and not rb, now,
                        self._fire_stop_combo)

    def _arm_combo(self, name: str, active: bool, released: bool, now: float,
                   fire) -> None:
        if released:
            self._combo_since.pop(name, None)
            self._combo_fired.discard(name)
            self._combo_latch.discard(name)
            return
        if not active or name in self._combo_fired:
            return                      # one action per press
        since = self._combo_since.setdefault(name, now)
        if now - since >= COMBO_HOLD_S:
            self._combo_fired.add(name)
            self._combo_latch.add(name)
            fire()

    def _fire_af_combo(self) -> None:
        mode = getattr(self._state, "mode", "MANUAL") if self._state else "MANUAL"
        if mode != "MANUAL":
            self.sig_log.emit(
                f"gamepad LT+RT: {mode.lower()} owns the axes — "
                "autofocus not started")
            return
        self.sig_log.emit("gamepad LT+RT: autofocus once")
        self.sig_af_requested.emit()

    def _fire_stop_combo(self) -> None:
        self.sig_log.emit("gamepad LB+RB: STOP ALL")
        self.on_escape()

    def _tick(self) -> None:
        gamepad = self._gamepad_state
        self._expire_stale_holds()
        if self._esc_latch:
            # Latch: suppress motion commands until EVERY source is at
            # rest (reference ESC-latch semantics). The predicate used to
            # read `(not keys and not gamepad.connected) or released` —
            # `and` binds tighter than `or`, so with a connected gamepad
            # and centred sticks the latch cleared on the very next tick
            # regardless of held keys or on-screen holds, and a still-held
            # source could then restart motion right after STOP ALL.
            if (not self._key_state and not self._ui_state
                    and self._sources_released(gamepad)):
                self._esc_latch = False
            self._prev_key_state = dict(self._key_state)
            return
        # The combo gestures own their inputs while held AND until they are
        # released (see _check_combos): no focus jog under LT+RT, no
        # stick/D-pad jog under LB+RB.
        both_triggers = (gamepad.left_trigger >= self._trigger_level
                         and gamepad.right_trigger >= self._trigger_level)
        both_bumpers = bool(gamepad.button_left_shoulder
                            and gamepad.button_right_shoulder)
        try:
            self._resolver.resolve(
                self._key_state, self._prev_key_state, gamepad,
                on_command=self._dispatch, ui_state=self._ui_state,
                suppress_focus=both_triggers or "af" in self._combo_latch,
                suppress_jog=both_bumpers or "stop" in self._combo_latch)
        except Exception as exc:  # noqa: BLE001 - input must never crash the GUI
            # _dispatch is guarded but the resolver itself was not: it is
            # called from a QTimer slot with a public ui_state API (a bogus
            # "stage:axis" claim raises inside it) and an unhandled exception
            # here kills the tick that also feeds _prev_key_state.
            logger.warning("Input resolve failed: %s", exc)
        self._prev_key_state = dict(self._key_state)

    def _expire_stale_holds(self) -> None:
        """Drop on-screen holds that outlived MAX_UI_HOLD_S (see the
        constant) — the fail-safe behind ``cancel_all_holds``."""
        if not self._ui_state:
            return
        now = time.monotonic()
        for key, entry in list(self._ui_state.items()):
            if now - float(entry[0]) > MAX_UI_HOLD_S:
                del self._ui_state[key]
                logger.warning("UI hold %s expired after %.0f s — stopping",
                               key, MAX_UI_HOLD_S)
                self.sig_log.emit(f"hold on {key} expired — stopping")

    def _sources_released(self, gamepad) -> bool:
        """True when all motion sources are at rest (latch clears)."""
        sticks_centered = (abs(gamepad.left_x) < 0.10 and abs(gamepad.left_y) < 0.10
                           and abs(gamepad.right_x) < 0.50
                           and abs(gamepad.right_y) < 0.50)
        triggers_released = (abs(gamepad.left_trigger) < COMBO_RELEASE_LEVEL
                             and abs(gamepad.right_trigger) < COMBO_RELEASE_LEVEL)
        buttons_released = not any((gamepad.button_a, gamepad.button_b,
                                    gamepad.button_x, gamepad.button_y,
                                    gamepad.dpad_up, gamepad.dpad_down,
                                    gamepad.dpad_left, gamepad.dpad_right))
        return sticks_centered and triggers_released and buttons_released

    # ------------------------------------------------------------------

    def _dispatch(self, command: tuple) -> bool:
        """Send one resolved command. Returns False when a gate refused it."""
        stage_id, axis, mode, direction, speed, source = command
        # Manual-control axis mapping (invert / flip X↔Y) — applied here,
        # at the one point every manual source goes through, so keyboard,
        # gamepad and on-screen buttons can never disagree. Stops carry
        # direction 0, so they follow the axis flip but are never negated.
        axis, direction = self._axis_maps.get(stage_id, IDENTITY).apply(
            axis, direction)
        if self._esc_latch and mode != "continuous_stop":
            # The latch suppresses motion until every source is released.
            # _tick returns early while latched, so this only matters for
            # sources that bypass the tick (a UI single-step click).
            return False
        if self._state is not None and self._state.mode != "MANUAL" \
                and mode != "continuous_stop":
            if self._state.mode != "AUTOFOCUS":
                # Scan owns the axes — manual motion is gated (unchanged).
                return False
            # AUTOFOCUS: user input ABORTS the run (the autofocus
            # service's sig_job_submitted hook fires on the resulting
            # submit) and the action proceeds — input wins; the queued
            # focus motion executes after the AF job returns. RELEASES
            # always pass below: a jog started before (or during) an AF
            # run must be stoppable (hardware-verified: the gated release
            # left the zolix unstoppable during autofocus).
        try:
            if mode == "continuous_stop":
                self._do_continuous_stop(stage_id, axis)
            elif mode == "continuous_start":
                self._do_continuous_start(stage_id, axis, direction, speed)
            elif mode == "single_step":
                self._do_single_step(stage_id, axis, direction)
            else:
                logger.warning("Unknown input mode: %s", mode)
                return False
        except Exception as exc:  # noqa: BLE001 - input must never crash the GUI
            logger.warning("Input dispatch failed %s: %s", command, exc)
            return False
        return True

    def _do_continuous_start(self, stage_id, axis, direction, speed) -> None:
        if stage_id == "focus":
            signed = int(speed) if direction > 0 else -int(speed)
            # Dedupe: the reference re-emits every tick and relies on
            # driver-side rate-limiting — TALOS rate-limits at dispatch.
            if self._last_focus_speed == signed:
                return
            # Throttle to 10 Hz: a trigger curve sweep floods set_speed
            # commands and wedges the firmware's RX (hardware-verified).
            # The release stop is a separate path and is never throttled.
            now = time.monotonic()
            # The first command after a release is EXEMPT: re-pressing a
            # dialbox hold used to be swallowed by the throttle and felt
            # up to 100 ms late (a sweep keeps _last_focus_speed set, so
            # it stays throttled).
            if (self._last_focus_speed is not None
                    and now - self._last_focus_submit_t < 0.1):
                return
            self._last_focus_speed = signed
            self._last_focus_submit_t = now
            self._manager.submit("focus", "set_speed", signed)
            return
        key = f"{stage_id}:{axis}"
        if stage_id == "sigmakoki":
            level = hz_to_level(speed)
            signature = (direction, level)
            target = ("sigmakoki", "move", _axis_enum(axis),
                      Direction.POSITIVE if direction > 0 else Direction.NEGATIVE,
                      level)
        elif stage_id == "zolix":
            from talos.hal.devices.zolix import DIR_NEG, DIR_POS

            pps = int(speed)
            signature = (direction, pps)
            target = ("zolix", "move_continuous", axis,
                      DIR_POS if direction > 0 else DIR_NEG, pps)
        else:
            return
        # Dedupe on the QUANTISED command, not the raw analog value: the
        # stick emits a fresh float speed every 16 ms tick, so comparing
        # floats re-sent the same firmware speed level dozens of times a
        # second and the queue acted on commands the operator had left.
        if self._last_continuous.get(key) == signature:
            return
        self._last_continuous[key] = signature
        self._manager.submit(*target)

    def _do_continuous_stop(self, stage_id, axis) -> None:
        if stage_id == "focus":
            if self._last_focus_speed is None:
                return
            self._last_focus_speed = None
            # The driver's stop() is belt-and-braces fire-and-forget
            # (STOP + SPD:0, no ACK round-trip): releasing the trigger must
            # never depend on a wedged firmware answering. The proxy purge
            # in enqueue_stop drops stale speed commands so nothing
            # re-starts the axis afterwards.
            self._manager.submit("focus", "stop", priority=1)
            return
        key = f"{stage_id}:{axis}"
        if self._last_continuous.pop(key, None) is None:
            return  # nothing was running — skip the redundant stop
        if stage_id == "sigmakoki":
            self._manager.submit("sigmakoki", "stop", AxisMask[_axis_enum(axis).name],
                                 priority=1)
        elif stage_id == "zolix":
            self._manager.submit("zolix", "stop_axis", axis, priority=1)

    def _do_single_step(self, stage_id, axis, direction) -> None:
        if stage_id == "focus":
            return  # focus has no single-step (continuous-immediate)
        if stage_id == "sigmakoki":
            cfg = self._settings.device("sigmakoki")
            steps = int(cfg.get("single_step_z" if axis == "z" else "single_step", 3))
            self._manager.submit("sigmakoki", "step", _axis_enum(axis),
                                 Direction.POSITIVE if direction > 0 else Direction.NEGATIVE,
                                 steps)
        elif stage_id == "zolix":
            cfg = self._settings.device("zolix")
            if axis == "r":
                step = int(cfg.get("single_step_r", 80))
                self._manager.submit("zolix", "move_rel_um", 0.0, 0.0,
                                     direction * step * float(cfg.get("um_per_pulse_r", 0.00125)))
            else:
                step = int(cfg.get("single_step", 5))
                um = step * float(cfg.get("um_per_pulse_xy", 0.625))
                if axis == "x":
                    self._manager.submit("zolix", "move_rel_um",
                                         direction * um, 0.0)
                else:
                    self._manager.submit("zolix", "move_rel_um",
                                         0.0, direction * um)


def _axis_enum(axis: str) -> Axis:
    return {"x": Axis.X, "y": Axis.Y, "z": Axis.Z}[axis]
