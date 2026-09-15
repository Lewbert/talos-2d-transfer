"""Faithful port of the reference transfer-stage-control ActionResolver
(input_system/action_resolver.py) adapted to TALOS's command layer.

Behavioral contract (identical to the reference):
- Short press (< 300 ms) → single step (0.2 s cooldown per stage:axis);
  long press → continuous; release after continuous → stop.
- Focus: continuous IMMEDIATELY on press (short taps nudge);
  slow = max(fw floor, min_speed, max/4), fast (Shift/LB/RB) = max.
- Keyboard priority over gamepad per axis; ESC latch until re-center.
- Gamepad: left stick → SigmaKoki XY (analog per-axis, 10% deadzone),
  right stick → Zolix XY (8-direction, 50% deadzone, 2-tick hysteresis),
  X/Y → Zolix R, A/B → SigmaKoki Z (short/long press),
  D-pad → selected stage XY (Back toggles stage), triggers → focus.
- Speeds from settings; LB/RB = fast for their stage.

Instead of the reference's StageCommand objects this emits plain tuples:
(stage_id, axis, mode, direction, speed, source) into ``on_command``.
"""

from __future__ import annotations

import logging
import math
import time

logger = logging.getLogger(__name__)

LONG_PRESS_S = 0.300
SINGLE_STEP_COOLDOWN_S = 0.2
FOCUS_FIRMWARE_MIN_SPS = 10

# 8-direction → (axis, direction) for the Zolix right stick.
_DIR_MAP = {
    0: (("x", 1),),           1: (("x", 1), ("y", 1)),
    2: (("y", 1),),           3: (("x", -1), ("y", 1)),
    4: (("x", -1),),          5: (("x", -1), ("y", -1)),
    6: (("y", -1),),          7: (("x", 1), ("y", -1)),
}


def focus_trigger_to_speed(lt: float, rt: float, *,
                           min_speed: float, max_speed: float,
                           gamma: float = 2.2, deadzone: float = 0.05,
                           invert: bool = False) -> int:
    """Signed focus speed from LT/RT (net = RT − LT)."""
    net = rt - lt
    if invert:
        net = -net
    mag = abs(net)
    if mag <= deadzone:
        return 0
    x = (mag - deadzone) / (1.0 - deadzone)
    lo = max(FOCUS_FIRMWARE_MIN_SPS, float(min_speed))
    hi = max(lo, float(max_speed))
    speed = lo + (hi - lo) * (x ** float(gamma))
    speed = min(speed, hi)
    return int(round(speed)) if net > 0 else -int(round(speed))


class ActionResolver:
    """Stateful resolver: one resolve() call per input tick (60 Hz)."""

    def __init__(self, settings, state=None):
        self._settings = settings
        self._state = state  # AppState (the selected objective; optional)
        # Set by resolve() from the gamepad combo gestures (see resolve).
        self._suppress_focus = False
        self._suppress_jog = False
        self._load_settings()
        self._dpad_stage = "sigmakoki"

        # Continuous-motion tracking: key → source token.
        self._continuous_keys: dict[str, str] = {}
        self._continuous_speed: dict[str, float] = {}
        self._continuous_stick: dict[str, bool] = {}
        self._focus_active = False
        self._focus_paused = False
        self._stick_pause: set[str] = set()
        self._last_stick_dir: dict[str, int] = {}
        self._last_stick_fast: dict[str, bool] = {}
        self._stick_dir_counter: dict[str, int] = {}
        self._press_times: dict[str, float] = {}
        self._last_single_step_time: dict[str, float] = {}

    def _load_settings(self) -> None:
        """(Re)read every settings-derived value. Called from __init__ and
        from ``reload_settings`` — the jog speeds and the focus trigger
        curve used to be cached for the process lifetime, so a Preferences
        Apply only took effect after a restart."""
        settings = self._settings
        z = settings.device("zolix")
        sk = settings.device("sigmakoki")
        fc = settings.device("focus")
        self._long_press_s = float(settings.section("input").get(
            "long_press_threshold_ms", 300)) / 1000.0
        self._slow_speed = {"sigmakoki": int(sk.get("slow_speed_hz", 100)),
                            "zolix": int(z.get("slow_speed_pps", 500))}
        self._fast_speed = {"sigmakoki": int(sk.get("fast_speed_hz", 1000)),
                            "zolix": int(z.get("fast_speed_pps", 2000))}
        self._slow_z = {"sigmakoki": int(sk.get("slow_speed_z", 99))}
        self._fast_z = {"sigmakoki": int(sk.get("fast_speed_z", 500))}
        self._slow_r = {"zolix": int(z.get("slow_speed_r", 1000))}
        self._fast_r = {"zolix": int(z.get("fast_speed_r", 10000))}
        self._focus_min = int(fc.get("min_speed", 50))
        self._focus_max = int(fc.get("max_speed", 2000))
        self._focus_gamma = float(fc.get("gamma", 2.2))
        self._focus_deadzone = float(fc.get("deadzone", 0.05))

    def reload_settings(self) -> None:
        """Apply edited speeds / focus curve to the NEXT resolve tick."""
        self._load_settings()

    @property
    def dpad_stage(self) -> str:
        return self._dpad_stage

    def toggle_dpad_stage(self) -> str:
        self._dpad_stage = "zolix" if self._dpad_stage == "sigmakoki" else "sigmakoki"
        return self._dpad_stage

    # ------------------------------------------------------------------

    def resolve(self, key_state: dict[str, float], prev_key_state: dict[str, float],
                gamepad, now: float | None = None, on_command=None,
                ui_state: dict[str, tuple] | None = None,
                suppress_focus: bool = False,
                suppress_jog: bool = False) -> list:
        """Resolve one input tick into command tuples.

        ``key_state``: keysym → press timestamp for keys CURRENTLY down.
        ``prev_key_state``: the previous tick's map (for short-press
        release detection — identical to the reference design).
        ``gamepad``: GamepadState-like object with analog fields.
        ``ui_state``: "stage:axis" → (press_time, direction) for on-screen
        hold buttons (reference UI-button branch).
        ``on_command``: callable(command_tuple) — defaults to collecting.

        ``suppress_focus`` / ``suppress_jog``: the gamepad COMBO gestures
        (LT+RT autofocus, LB+RB stop) own those inputs while they are held —
        starts are not emitted, and the normal release path stops whatever
        was already moving (an AF run must not be aborted by the residual
        trigger imbalance of the very gesture that started it).
        """
        self._suppress_focus = bool(suppress_focus)
        self._suppress_jog = bool(suppress_jog)
        now = now if now is not None else time.monotonic()
        commands: list = []
        emit = on_command or commands.append
        ui_state = ui_state or {}

        kb_claimed: set[str] = set()
        ui_claimed: set[str] = set()

        # ---- Keyboard ---------------------------------------------------
        shift = self._shift_held(key_state)
        for keysym, press_time in key_state.items():
            if press_time <= 0:
                continue
            action = self._key_action(keysym)
            if action is None:
                continue
            stage_id, axis, direction = action
            claim = f"{stage_id}:{axis}"
            if stage_id == "focus":
                speed = self._focus_key_speed(fast=shift)
                # The KEY's direction matters here (unlike the gamepad
                # triggers, where direction comes from the SIGNED speed):
                # +/− map to ±1 and the key speed is always positive —
                # deriving direction from the speed sign made both keys
                # jog the same way.
                emit((stage_id, axis, "continuous_start",
                      direction if speed >= 0 else -direction,
                      float(abs(speed)), "keyboard"))
                if claim not in self._continuous_keys:
                    self._continuous_keys[claim] = keysym
                    self._continuous_speed[claim] = speed
                kb_claimed.add(claim)
                continue
            duration = now - press_time
            if duration >= self._long_press_s:
                speed = self._get_speed(stage_id, axis, fast=shift)
                if claim not in self._continuous_keys:
                    emit((stage_id, axis, "continuous_start", direction,
                          speed, "keyboard"))
                    self._continuous_keys[claim] = keysym
                    self._continuous_speed[claim] = speed
                elif speed != self._continuous_speed.get(claim, 0):
                    emit((stage_id, axis, "continuous_start", direction,
                          speed, "keyboard"))
                    self._continuous_speed[claim] = speed
                kb_claimed.add(claim)

        # ---- On-screen UI buttons (long-press holds) ---------------------
        for claim_key, entry in ui_state.items():
            press_time, direction = entry[:2]
            # 3-tuple entries carry an explicit fast flag (the stage
            # dialbox's double-arrow buttons); 2-tuples (legacy callers)
            # fall back to the keyboard Shift state.
            fast = bool(entry[2]) if len(entry) > 2 else False
            if press_time <= 0 or direction == 0:
                continue
            stage_id, axis = claim_key.split(":", 1)
            if claim_key in kb_claimed:
                continue  # keyboard has priority over UI
            ui_key = f"ui:{claim_key}"
            speed = self._get_speed(stage_id, axis, fast=(fast or shift))
            emit((stage_id, axis, "continuous_start", direction, speed, "ui_button"))
            self._continuous_keys[ui_key] = str(direction)
            self._continuous_speed[ui_key] = speed
            ui_claimed.add(claim_key)

        claimed = kb_claimed | ui_claimed

        # ---- Gamepad -----------------------------------------------------
        if gamepad.connected:
            self._handle_sticks(gamepad, claimed, emit)
            self._handle_dpad(gamepad, now, claimed, emit)
            self._handle_face_buttons(gamepad, now, claimed, emit)
        self._handle_triggers(gamepad, claimed, emit)
        self._handle_stops(key_state, gamepad, now, emit, claimed=claimed,
                           ui_state=ui_state)

        # ---- Short-press releases → single steps --------------------------
        for keysym, prev_time in list(prev_key_state.items()):
            if prev_time <= 0 or key_state.get(keysym, 0) > 0:
                continue
            action = self._key_action(keysym)
            if action is None:
                continue
            stage_id, axis, direction = action
            if stage_id == "focus":
                continue  # no single-step for focus
            claim = f"{stage_id}:{axis}"
            if now - prev_time < self._long_press_s:
                if self._axis_claimed(stage_id, axis):
                    continue
                if now - self._last_single_step_time.get(claim, 0) <= SINGLE_STEP_COOLDOWN_S:
                    continue
                self._last_single_step_time[claim] = now
                emit((stage_id, axis, "single_step", direction,
                      self._get_speed(stage_id, axis, fast=False), "keyboard"))

        return commands

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    _KEY_MAP = {
        # Zolix X/Y (WASD), R (Q/E)
        "w": ("zolix", "y", 1), "s": ("zolix", "y", -1),
        "a": ("zolix", "x", -1), "d": ("zolix", "x", 1),
        "q": ("zolix", "r", -1), "e": ("zolix", "r", 1),
        # SigmaKoki X/Y (arrows), Z (R/F)
        "Up": ("sigmakoki", "y", 1), "Down": ("sigmakoki", "y", -1),
        "Left": ("sigmakoki", "x", -1), "Right": ("sigmakoki", "x", 1),
        "r": ("sigmakoki", "z", 1), "f": ("sigmakoki", "z", -1),
        # Focus (+/−)
        "equal": ("focus", "z", 1), "plus": ("focus", "z", 1),
        "minus": ("focus", "z", -1), "underscore": ("focus", "z", -1),
    }

    def _key_action(self, keysym: str):
        return self._KEY_MAP.get(keysym)

    def _shift_held(self, key_state: dict[str, float]) -> bool:
        return key_state.get("Shift_L", 0) > 0 or key_state.get("Shift_R", 0) > 0

    def _objective_multiplier(self, kind: str) -> float:
        """Per-objective manual speed scale: "stage" → the XYR/XYZ jog
        multiplier, "focus" → the manual focus multiplier. 0/missing →
        1.0 (tests without objectives rows stay unscaled)."""
        rows = self._settings.get("objectives") or []
        idx = getattr(self._state, "objective", 0) \
            if self._state is not None else 0
        row = rows[min(int(idx), len(rows) - 1)] if rows else {}
        key = ("stage_speed_multiplier" if kind == "stage"
               else "focus_manual_multiplier")
        m = float(row.get(key, 0.0) or 0.0)
        return m if m > 0 else 1.0

    def _focus_key_speed(self, fast: bool) -> float:
        mult = self._objective_multiplier("focus")
        if fast:
            return float(self._focus_max) * mult
        return float(max(FOCUS_FIRMWARE_MIN_SPS, self._focus_min,
                         self._focus_max // 4)) * mult

    def _get_speed(self, stage_id: str, axis: str, fast: bool) -> float:
        if stage_id == "focus":
            # UI focus holds (StageControlWindow) share the keyboard
            # focus speed: min/max × objective focus multiplier (the
            # stage tables below use the wrong multiplier for focus).
            return self._focus_key_speed(fast=fast)
        mult = self._objective_multiplier("stage")
        if axis == "r":
            table = self._fast_r if fast else self._slow_r
            return float(table.get(stage_id, 500)) * mult
        if axis == "z":
            table = self._fast_z if fast else self._slow_z
            return float(table.get(stage_id, 200)) * mult
        return float(self._fast_speed[stage_id] if fast else self._slow_speed[stage_id]) \
            * mult

    def _axis_claimed(self, stage_id: str, axis: str) -> bool:
        for key in self._continuous_keys:
            if key == f"{stage_id}:{axis}" or key.endswith(f":{stage_id}:{axis}"):
                return True
        return f"{stage_id}:{axis}" in self._continuous_stick

    # ------------------------------------------------------------------
    # Gamepad sticks / dpad / buttons / triggers / stops (reference port)
    # ------------------------------------------------------------------

    def _handle_sticks(self, gamepad, claimed, emit) -> None:
        if self._suppress_jog:
            # The LB+RB gesture owns the sticks: stop what a stick is
            # driving and emit no new jogs (the stick branch's own stop
            # lives in the deflection check below, which this return would
            # skip — releasing the claims here is what actually halts the
            # axis). Holding both bumpers is an unambiguous stop gesture,
            # not a fast modifier.
            for ck in list(self._continuous_stick):
                stage_id, axis = ck.split(":", 1)
                emit((stage_id, axis, "continuous_stop", 0, 0.0,
                      "gamepad_stick"))
                del self._continuous_stick[ck]
            return
        # Left stick → SigmaKoki (analog per-axis, 10% deadzone).
        fast = gamepad.button_left_shoulder
        mult = self._objective_multiplier("stage")
        max_spd = (self._fast_speed["sigmakoki"] if fast
                   else self._slow_speed["sigmakoki"]) * mult
        self._stick_analog(gamepad.left_x, "sigmakoki", "x", max_spd, claimed, emit)
        self._stick_analog(gamepad.left_y, "sigmakoki", "y", max_spd, claimed, emit)
        # Right stick → Zolix (8-direction, 50% deadzone, hysteresis).
        fast = gamepad.button_right_shoulder
        self._stick_8dir(gamepad.right_x, gamepad.right_y, "zolix", "right",
                         fast, claimed, emit)

    def _stick_analog(self, value, stage_id, axis, max_speed, claimed, emit) -> None:
        ck = f"{stage_id}:{axis}"
        if ck in claimed:
            return
        if abs(value) < 0.10:
            if ck in self._continuous_stick:
                emit((stage_id, axis, "continuous_stop", 0, 0.0, "gamepad_stick"))
                del self._continuous_stick[ck]
            return
        direction = 1 if value > 0 else -1
        speed = abs(value) * max_speed
        emit((stage_id, axis, "continuous_start", direction, speed, "gamepad_stick"))
        self._continuous_stick[ck] = True

    def _stick_8dir(self, x, y, stage_id, stick_id, fast, claimed, emit) -> None:
        magnitude = math.sqrt(x * x + y * y)
        speed = (self._fast_speed[stage_id] if fast
                 else self._slow_speed[stage_id]) \
            * self._objective_multiplier("stage")
        if magnitude < 0.50:
            self._stick_pause.discard(stick_id)
            prev = self._last_stick_dir.pop(stick_id, -1)
            self._stick_dir_counter.pop(stick_id, None)
            if prev >= 0:
                for ax, _dr in _DIR_MAP.get(prev, ()):
                    ck = f"{stage_id}:{ax}"
                    emit((stage_id, ax, "continuous_stop", 0, 0.0, "gamepad_stick"))
                    self._continuous_stick.pop(ck, None)
                self._last_stick_fast.pop(stick_id, None)
            return
        if stick_id in self._stick_pause:
            return
        ax, ay = abs(x), abs(y)
        if ax > 2.0 * ay:
            direction = 0 if x > 0 else 4
        elif ay > 2.0 * ax:
            direction = 2 if y > 0 else 6
        elif x > 0 and y > 0:
            direction = 1
        elif x < 0 and y > 0:
            direction = 3
        elif x < 0 and y < 0:
            direction = 5
        else:
            direction = 7
        prev_dir = self._last_stick_dir.get(stick_id, -1)
        prev_fast = self._last_stick_fast.get(stick_id, False)
        if prev_dir == direction and prev_fast == fast:
            for ax, dr in _DIR_MAP[direction]:
                ck = f"{stage_id}:{ax}"
                if ck in claimed:
                    continue
                emit((stage_id, ax, "continuous_start", dr, speed, "gamepad_stick"))
                self._continuous_stick[ck] = True
            return
        cnt = self._stick_dir_counter.get(stick_id, 0) + 1
        self._stick_dir_counter[stick_id] = cnt
        threshold = 1 if prev_dir < 0 else 2
        if cnt < threshold:
            return
        self._stick_dir_counter[stick_id] = 0
        if prev_dir >= 0:
            for ax, _dr in _DIR_MAP.get(prev_dir, ()):
                ck = f"{stage_id}:{ax}"
                emit((stage_id, ax, "continuous_stop", 0, 0.0, "gamepad_stick"))
                self._continuous_stick.pop(ck, None)
        self._last_stick_dir[stick_id] = direction
        self._last_stick_fast[stick_id] = fast
        for ax, dr in _DIR_MAP[direction]:
            ck = f"{stage_id}:{ax}"
            if ck in claimed:
                continue
            emit((stage_id, ax, "continuous_start", dr, speed, "gamepad_stick"))
            self._continuous_stick[ck] = True

    def _handle_dpad(self, gamepad, now, claimed, emit) -> None:
        if self._suppress_jog:
            return   # the LB+RB stop gesture owns the pad too
        stage_id = self._dpad_stage
        fast = (gamepad.button_left_shoulder if stage_id == "sigmakoki"
                else gamepad.button_right_shoulder)
        speed = (self._fast_speed[stage_id] if fast
                 else self._slow_speed[stage_id]) \
            * self._objective_multiplier("stage")
        for pressed, axis, direction in (
            (gamepad.dpad_up, "y", 1), (gamepad.dpad_down, "y", -1),
            (gamepad.dpad_left, "x", -1), (gamepad.dpad_right, "x", 1),
        ):
            claim = f"{stage_id}:{axis}"
            press_key = f"dpad:{claim}:{direction}"
            if claim in claimed:
                self._press_times.pop(press_key, None)
                continue
            if pressed:
                if press_key not in self._press_times:
                    self._press_times[press_key] = now
                duration = now - self._press_times[press_key]
                if duration >= self._long_press_s:
                    key = f"dpad:{claim}"
                    if key not in self._continuous_keys:
                        emit((stage_id, axis, "continuous_start", direction,
                              speed, "gamepad_dpad"))
                        self._continuous_keys[key] = str(direction)
                        self._continuous_speed[key] = speed
                    elif speed != self._continuous_speed.get(key, 0):
                        emit((stage_id, axis, "continuous_start", direction,
                              speed, "gamepad_dpad"))
                        self._continuous_speed[key] = speed

    def _handle_face_buttons(self, gamepad, now, claimed, emit) -> None:
        # X/Y → Zolix R (RB fast), A/B → SigmaKoki Z (LB fast).
        mult = self._objective_multiplier("stage")
        fast = gamepad.button_right_shoulder
        speed = (self._fast_r["zolix"] if fast else self._slow_r["zolix"]) * mult
        for pressed, direction, name in ((gamepad.button_x, -1, "X"),
                                         (gamepad.button_y, 1, "Y")):
            self._handle_button(pressed, "zolix", "r", direction, name,
                                now, claimed, emit, speed)
        fast = gamepad.button_left_shoulder
        speed = (self._fast_z["sigmakoki"] if fast
                 else self._slow_z["sigmakoki"]) * mult
        for pressed, direction, name in ((gamepad.button_a, 1, "A"),
                                         (gamepad.button_b, -1, "B")):
            self._handle_button(pressed, "sigmakoki", "z", direction, name,
                                now, claimed, emit, speed)

    def _handle_button(self, pressed, stage_id, axis, direction, name,
                       now, claimed, emit, speed) -> None:
        press_key = f"btn:{stage_id}:{axis}:{direction}"
        claim_key = f"btn:{stage_id}:{axis}"
        if f"{stage_id}:{axis}" in claimed:
            self._press_times.pop(press_key, None)
            return
        if pressed:
            if press_key not in self._press_times:
                self._press_times[press_key] = now
            duration = now - self._press_times[press_key]
            if duration >= self._long_press_s:
                if claim_key not in self._continuous_keys:
                    emit((stage_id, axis, "continuous_start", direction,
                          speed, "gamepad_button"))
                    self._continuous_keys[claim_key] = str(direction)
                    self._continuous_speed[claim_key] = speed
                elif speed != self._continuous_speed.get(claim_key, 0):
                    emit((stage_id, axis, "continuous_start", direction,
                          speed, "gamepad_button"))
                    self._continuous_speed[claim_key] = speed

    def _handle_triggers(self, gamepad, claimed, emit) -> None:
        if self._suppress_focus:
            # The LT+RT gesture owns the triggers: stop whatever the
            # triggers were driving and emit no starts. Both halves matter —
            # a jog still running when the AF starts would abort the run it
            # just asked for (any focus command aborts a running AF), and
            # the gesture's own residual imbalance (LT 0.8 / RT 1.0 = a
            # small net jog) must not become one either.
            if self._focus_active:
                emit(("focus", "z", "continuous_stop", 0, 0.0, "gamepad_trigger"))
                self._focus_active = False
            return
        mult = self._objective_multiplier("focus")
        focus_min = self._focus_min * mult
        focus_max = self._focus_max * mult
        # NOTE: no direction inversion here. `devices.focus.invert` is
        # applied once, in InputSystem._dispatch, together with the stage
        # axis maps — inverting in both places cancelled out for the
        # gamepad only, so the triggers jogged the opposite way from the
        # keyboard and the dialbox buttons.
        if self._focus_paused:
            if focus_trigger_to_speed(
                    gamepad.left_trigger, gamepad.right_trigger,
                    min_speed=focus_min, max_speed=focus_max,
                    gamma=self._focus_gamma,
                    deadzone=self._focus_deadzone) == 0:
                self._focus_paused = False
            return
        if "focus:z" in claimed:
            return
        speed = focus_trigger_to_speed(
            gamepad.left_trigger, gamepad.right_trigger,
            min_speed=focus_min, max_speed=focus_max,
            gamma=self._focus_gamma, deadzone=self._focus_deadzone)
        if speed == 0:
            if self._focus_active:
                emit(("focus", "z", "continuous_stop", 0, 0.0, "gamepad_trigger"))
                self._focus_active = False
            return
        emit(("focus", "z", "continuous_start", 1 if speed > 0 else -1,
              float(abs(speed)), "gamepad_trigger"))
        self._focus_active = True

    def _handle_stops(self, key_state, gamepad, now, emit, claimed,
                      ui_state: dict | None = None) -> None:
        """Reference-ported stops: emit continuous_stop for axes that are no
        longer commanded, and single_step for gamepad dpad/buttons released
        before the long-press threshold (0.2 s per-axis cooldown)."""
        ui_state = ui_state or {}

        # ---- Keyboard stops ----
        for claim_key, keysym in list(self._continuous_keys.items()):
            if claim_key.startswith(("dpad:", "btn:", "ui:")):
                continue
            if key_state.get(keysym, 0) > 0:
                continue
            stage_id, axis = claim_key.split(":", 1)
            if ui_state.get(claim_key, (0.0, 0))[0] > 0:
                del self._continuous_keys[claim_key]
                self._continuous_speed.pop(claim_key, None)
                continue  # UI button still held — its branch re-emits
            emit((stage_id, axis, "continuous_stop", 0, 0.0, "keyboard"))
            del self._continuous_keys[claim_key]
            self._continuous_speed.pop(claim_key, None)

        # ---- UI-button stops ----
        for claim_key in list(self._continuous_keys):
            if not claim_key.startswith("ui:"):
                continue
            parts = claim_key.split(":")
            stage_id, axis = parts[1], parts[2]
            plain = f"{stage_id}:{axis}"
            if ui_state.get(plain, (0.0, 0))[0] > 0:
                continue
            if plain not in claimed:
                emit((stage_id, axis, "continuous_stop", 0, 0.0, "ui_button"))
            del self._continuous_keys[claim_key]
            self._continuous_speed.pop(claim_key, None)

        # ---- D-pad stops (single step on short press) ----
        for press_key, press_time in list(self._press_times.items()):
            if not press_key.startswith("dpad:"):
                continue
            parts = press_key.split(":")
            stage_id, axis, dir_str = parts[1], parts[2], parts[3]
            direction = int(dir_str)
            claim_key = f"dpad:{stage_id}:{axis}"
            claimed_axis = f"{stage_id}:{axis}" in claimed
            if self._is_dpad_direction_pressed(gamepad, axis, direction):
                continue
            held = now - press_time
            if held < self._long_press_s:
                if (not claimed_axis
                        and now - self._last_single_step_time.get(
                            f"{stage_id}:{axis}", 0) > SINGLE_STEP_COOLDOWN_S):
                    self._last_single_step_time[f"{stage_id}:{axis}"] = now
                    emit((stage_id, axis, "single_step", direction,
                          self._get_speed(stage_id, axis, fast=False),
                          "gamepad_dpad"))
            elif claim_key in self._continuous_keys:
                if not claimed_axis:
                    emit((stage_id, axis, "continuous_stop", 0, 0.0, "gamepad_dpad"))
                del self._continuous_keys[claim_key]
                self._continuous_speed.pop(claim_key, None)
            del self._press_times[press_key]

        # ---- Face-button stops (single step on short press) ----
        for press_key, press_time in list(self._press_times.items()):
            if not press_key.startswith("btn:"):
                continue
            parts = press_key.split(":")
            stage_id, axis, dir_str = parts[1], parts[2], parts[3]
            direction = int(dir_str)
            claim_key = f"btn:{stage_id}:{axis}"
            claimed_axis = f"{stage_id}:{axis}" in claimed
            if self._is_face_button_pressed(gamepad, stage_id, axis, direction):
                continue
            held = now - press_time
            if held < self._long_press_s:
                if (not claimed_axis
                        and now - self._last_single_step_time.get(
                            f"{stage_id}:{axis}", 0) > SINGLE_STEP_COOLDOWN_S):
                    self._last_single_step_time[f"{stage_id}:{axis}"] = now
                    emit((stage_id, axis, "single_step", direction,
                          self._get_speed(stage_id, axis, fast=False),
                          "gamepad_button"))
            elif claim_key in self._continuous_keys:
                if not claimed_axis:
                    emit((stage_id, axis, "continuous_stop", 0, 0.0, "gamepad_button"))
                del self._continuous_keys[claim_key]
                self._continuous_speed.pop(claim_key, None)
            del self._press_times[press_key]

    # Helpers ---------------------------------------------------------

    def _is_dpad_direction_pressed(self, gamepad, axis, direction) -> bool:
        if axis == "y":
            return gamepad.dpad_up if direction > 0 else gamepad.dpad_down
        return gamepad.dpad_right if direction > 0 else gamepad.dpad_left

    @staticmethod
    def _is_face_button_pressed(gamepad, stage_id, axis, direction) -> bool:
        if stage_id == "zolix" and axis == "r":
            return gamepad.button_y if direction > 0 else gamepad.button_x
        if stage_id == "sigmakoki" and axis == "z":
            return gamepad.button_a if direction > 0 else gamepad.button_b
        return False
