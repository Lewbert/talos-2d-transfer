"""Bottom hardware strip: compact XYR/XYZ/Focus/Temp status (the
simplified transfer-stage-control bottom bar). Status + enable toggles
only — all control buttons live in the StageControlWindow.

The parsers use the CORRECT telemetry keys (the old stage panels read
stale ones): zolix position is ``x_pulses/.../x_um/...`` and limits are
``limit_x_pos/...`` (``any_moving`` is a property and never serialized —
moving is derived from the per-axis flags); sigmakoki position is
``dict[Axis, int]`` and its status is a ``dict[str, str]``.

LAYOUT RULE (the user's): a panel is a row of components separated by ONE
fixed gap (``FIELD_GAP``) and holds EXACTLY ONE elastic component that
absorbs the panel's leftover width — the position readout for the stages,
the trigger bar for FOCUS, and the three equal value fields for TEMP. The
fixed components therefore keep their size at every window width and no
flexible gap can open a hole; the four panels are the same width (equal
stretch + equal minimum in ``HardwareStrip``).
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QStyle,
    QStyleOption,
    QVBoxLayout,
    QWidget,
)

from talos.hal.base import Axis
from talos.hal.devices.sigmakoki import SPEED_LEVEL_TO_HZ
from talos.ui.theme import DANGER, LED_OFF, TEXT_DIM
from talos.ui.widgets.trigger_bar import TriggerBarWidget

#: All four sections are the SAME width (the user's requirement): equal
#: stretch, equal explicit minimum. The explicit minimum is what lets them
#: compress at all — a section's own content minimum is larger.
_SECTION_STRETCH = 1

#: The one gap between neighbouring components. Every panel uses exactly
#: this, so the rhythm is identical across the four of them.
FIELD_GAP = 8

#: The stage readout's minimum, and the smaller one used in compact mode:
#: at the 1024 px window floor a panel cannot give the readout its full
#: width, and a lower minimum keeps the BOX inside the panel (the text
#: truncates in the middle) instead of overflowing the frame's border.
READOUT_MIN = 180
COMPACT_READOUT_MIN = 92

#: The words the stage/focus state indicator can show. ONE slot width fits
#: them all, measured in the BOLD font the lit state uses, so a word change
#: never shifts the components beside it.
STATE_WORDS = ("IDLE", "MOVE", "CONT", "TRAP", "BLOCKED")


def state_slot_width(font: QFont) -> int:
    """Fixed width of an ``IDLE``/``MOVE``-style state word."""
    bold = QFont(font)
    bold.setBold(True)
    metrics = QFontMetrics(bold)
    return max(metrics.horizontalAdvance(word) for word in STATE_WORDS) + 6


def centred_readout(text: str) -> str:
    """Re-pad a space-padded readout symmetrically.

    The numeric fields are right-aligned inside a fixed width, which is
    what stops the text jittering as digits come and go — but it also puts
    all of the slack on the LEFT. A centred label aligns the string, spaces
    included, so the numbers used to land ~6 px right of the middle of the
    elastic box (measured). Redistributing the same padding keeps the
    constant width AND centres the ink.
    """
    return text.strip().center(len(text))


# --- pure parsers (unit-tested) ----------------------------------------


def parse_zolix(payload: dict) -> dict:
    pos = payload.get("position") or {}
    status = payload.get("status") or {}
    return {
        "x_um": float(pos.get("x_um", 0.0)),
        "y_um": float(pos.get("y_um", 0.0)),
        "r_deg": float(pos.get("r_deg", 0.0)),
        "moving": bool(status.get("x_moving") or status.get("y_moving")
                       or status.get("r_moving")),
        "limits": {
            "x+": bool(status.get("limit_x_pos")),
            "x-": bool(status.get("limit_x_neg")),
            "y+": bool(status.get("limit_y_pos")),
            "y-": bool(status.get("limit_y_neg")),
        },
        "estop": bool(status.get("estop")),
    }


def _limit_flags(payload: dict) -> dict[str, bool] | None:
    """The four X/Y limit switches, or None when the payload has none."""
    limits = payload.get("limits")
    if not isinstance(limits, dict) or not limits:
        return None
    return {key: bool(limits.get(key)) for key in ("x+", "x-", "y+", "y-")}


def parse_sigmakoki(payload: dict) -> dict:
    """STATUS? carries POSITIONS as x/y/z and SPEED LEVELS as xspd/yspd/
    zspd.

    The levels are a SETTING, not a motion flag: the firmware's default is
    level 2 and ``stopAxis()`` clears ``moving`` but NOT ``speed_level``
    (verified in transfer_stage_controller.ino), so a level > 0 would light
    a "moving" lamp forever. Motion is inferred from the position changing
    between polls — the same settle detection the driver's ``wait_idle``
    uses (the firmware has no per-axis busy flag).
    """
    status = payload.get("status") or {}
    pos = payload.get("position") or {}

    def steps(key: str) -> int:
        value = pos.get(getattr(Axis, key, None))
        if value is not None:
            return int(value)
        try:
            # the status dict uses lowercase keys ("x"/"y"/"z")
            return int(status.get(key.lower(), 0))
        except (TypeError, ValueError):
            return 0

    def level(key: str) -> int:
        try:
            return int(status.get(key, 0))
        except (TypeError, ValueError):
            return 0

    levels = {axis: level(f"{axis}spd") for axis in ("x", "y", "z")}
    return {"x": steps("X"), "y": steps("Y"), "z": steps("Z"),
            "levels": levels,
            "speed_hz": float(SPEED_LEVEL_TO_HZ.get(levels["z"], 0.0)),
            "limits": _limit_flags(payload)}


def parse_focus(payload: dict) -> dict:
    status = payload.get("status") or {}
    return {
        "pos": int(status.get("pos", 0)),
        "mode": str(status.get("mode", "—")),
        "slim_bounds": payload.get("slim_bounds"),
        "slim_on": bool(status.get("slim_on")),
        "blocked": str(status.get("blocked_dir", "0")),
    }


def parse_yudian(payload: dict) -> dict:
    return {
        "pv": payload.get("pv"),
        "sv": payload.get("sv"),
        "out": payload.get("output_percent"),
    }


# --- widgets ------------------------------------------------------------


class _ValueLabel(QLabel):
    """A ``#readout`` box whose text carries a DATA colour (a heat ramp).

    The box, font and padding come from the QSS rule — the same rule every
    other value field in the strip uses — so there is one source of truth
    for the look. Only the TEXT colour is data-driven, which no stylesheet
    can express: a per-widget palette does not survive the app stylesheet
    (Qt's stylesheet style republishes the widget palette during polish and
    paints the text from its own rule — measured, handoff #28b), and a
    plain QWidget subclass never gets the QSS background at all. So the box
    is drawn explicitly through ``PE_Widget`` and the text is painted here.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setObjectName("readout")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._colour: str | None = None

    def set_colour(self, colour: str | None) -> None:
        if colour != self._colour:
            self._colour = colour
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        # NOT chaining to QLabel.paintEvent: that would draw the text a
        # second time in the rule's accent colour, over the data colour.
        option = QStyleOption()
        option.initFrom(self)
        painter = QPainter(self)
        self.style().drawPrimitive(QStyle.PrimitiveElement.PE_Widget,
                                   option, painter, self)
        painter.setPen(QColor(self._colour or TEXT_DIM))
        # contentsRect(), not rect(): honour the rule's 2px/6px padding.
        painter.drawText(self.contentsRect(), Qt.AlignmentFlag.AlignCenter,
                         self.text())


def _set_state(widget: QWidget, object_name: str) -> None:
    """Swap a state colour by objectName. Changing objectName does NOT
    re-evaluate the stylesheet — Qt only repolishes when asked."""
    if widget.objectName() == object_name:
        return
    widget.setObjectName(object_name)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)


class _StateWord(QLabel):
    """The ``IDLE``/``MOVE`` state word shared by the stage and focus
    panels — one widget, so the two are the same treatment by construction:
    dim at rest, green while the axis moves, warn when a direction is
    blocked, centred in a slot sized for the longest word so the components
    beside it never shift.

    The slot is measured from the widget's FONT, which the app stylesheet
    only installs at polish time — hence the recompute on FontChange.
    """

    def __init__(self, text: str = "IDLE", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setObjectName("strip_mov_idle")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._apply_metrics()

    def _apply_metrics(self) -> None:
        self.setFixedWidth(state_slot_width(self.font()))

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self._apply_metrics()

    def set_state(self, text: str, object_name: str) -> None:
        if self.text() != text:
            self.setText(text)
        _set_state(self, object_name)


def temp_power_color(pv: float | None, sv: float | None,
                     out: float | None) -> str | None:
    """Heat-status colour for the PWR readout.

    The ramp is the reference project's temperature panel verbatim (its PV
    colour ladder): full power red → orange → amber, then — once the heater
    is idling — green when settled on the setpoint, light green when close,
    blue otherwise. None = no colour (uncalibrated/no reading).
    """
    if pv is None or sv is None or out is None:
        return None
    delta = float(pv) - float(sv)
    if out > 80:
        return "#e53935"        # red — full power
    if out > 40:
        return "#fb8c00"        # orange — moderate heating
    if out > 10:
        return "#fdd835"        # amber — gentle heating
    if abs(delta) < 0.5:
        return "#43a047"        # green — stable at the setpoint
    if abs(delta) < 2.0:
        return "#66bb6a"        # light green — approaching it
    return "#1e88e5"            # blue — heating up / cooling down


class _MiniDot(QWidget):
    """8px status dot (lit = triggered)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._on = False
        self.setFixedSize(8, 8)

    def set_on(self, on: bool) -> None:
        if on != self._on:
            self._on = on
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(DANGER if self._on else LED_OFF))
        painter.drawEllipse(0, 0, 8, 8)


class _Section(QFrame):
    """A strip section: title row + body. Styled by objectName (theme QSS)
    — inline stylesheets with hardcoded colours are the house anti-pattern."""

    def __init__(self, title: str, tooltip: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("strip_section")
        self._title = QLabel(title)
        self._title.setObjectName("dim")
        if tooltip:
            self.setToolTip(tooltip)
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 3, 6, 3)
        root.setSpacing(2)
        root.addWidget(self._title)
        self._body = QHBoxLayout()
        self._body.setSpacing(FIELD_GAP)   # the panel's fixed component gap
        root.addLayout(self._body)

    def add(self, widget: QWidget, stretch: int = 0) -> None:
        """Add a component. Exactly ONE component per panel is added with
        ``stretch=1`` — that is the panel's elastic field."""
        self._body.addWidget(widget, stretch)


class _StageSection(_Section):
    """XYR/XYZ compact status — the SAME layout and behaviour on both
    stages: enable toggle, axis readout, four limit dots, E-STOP (XYR
    only: the XYZ firmware has no such bit) and MOV.

    Both stages run ONE code path here; only the readout text differs
    (XYR reports degrees on R, XYZ a third linear axis). The optional
    indicators keep their slots reserved so lighting up never re-flows the
    bar, and MOV means the same thing on both — the axis is turning.
    """

    #: Both firmwares report the four X/Y limit switches (zolix from its
    #: status registers, sigmakoki from LIMITS?); the dots are identical.
    LIMIT_KEYS = ("x+", "x-", "y+", "y-")

    def __init__(self, title: str, device_key: str, tooltip: str,
                 manager, settings, *, estop: bool, parent=None):
        super().__init__(title, tooltip, parent)
        self._enable = QCheckBox("Enable")
        self._enable.setChecked(True)  # matches the manager's default gate
        self._enable.setToolTip(f"Enable commands for {tooltip or title}")
        self._enable.toggled.connect(
            lambda on: manager.set_enabled(device_key, on))
        self.add(self._enable)

        # Status word right after the toggle — the same widget the focus
        # panel uses, so both read identically (dim grey at rest, green
        # while the axis turns).
        self._moving = _StateWord("IDLE")
        self._moving.setToolTip("Green while the axis is turning")
        self.add(self._moving)

        # THE elastic field: the other components keep their size and this
        # one takes whatever width the panel has left.
        self._pos = QLabel("—")
        self._pos.setObjectName("readout")
        self._pos.setMinimumWidth(READOUT_MIN)
        self._pos.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.add(self._pos, stretch=1)

        # The four limit dots are ONE component, so they sit in their own
        # tightly-spaced row rather than being spread by the panel's gap.
        self._dots_box = QWidget(self)
        dots_row = QHBoxLayout(self._dots_box)
        dots_row.setContentsMargins(0, 0, 0, 0)
        dots_row.setSpacing(4)
        self._dots: dict[str, _MiniDot] = {}
        for key in self.LIMIT_KEYS:
            dot = _MiniDot(self._dots_box)
            dot.setToolTip(f"{device_key} limit {key}")
            self._dots[key] = dot
            dots_row.addWidget(dot)
        self.add(self._dots_box)

        self._has_estop = bool(estop)
        self._estop = QLabel("")
        self._estop.setObjectName("strip_estop")
        self._estop.setToolTip("Emergency-stop bit is set on the controller")
        if self._has_estop:
            self.add(self._estop)
            self._estop.hide()   # NO reserved slot: an E-STOP appearing must
            # shove the neighbours — that is what makes it noticed.
        self._last_pos: tuple | None = None

    def set_compact(self, compact: bool) -> None:
        """Drop the optional indicators when the window is too narrow for
        all four sections (1024 px minimum): the enable toggle, the state
        word, the readout and the E-STOP survive; the limit dots go."""
        if compact == getattr(self, "_compact", None):
            return
        self._compact = compact
        # The whole cluster, not each dot: a visible-but-empty container
        # would still cost the panel a gap.
        self._dots_box.setVisible(not compact)
        self._pos.setMinimumWidth(COMPACT_READOUT_MIN if compact else READOUT_MIN)
        if self._has_estop:
            # never un-hide an E-STOP that is not actually active
            self._estop.setVisible(not compact and bool(self._estop.text()))

    def _is_moving(self, parsed: dict) -> bool:
        """zolix reports real per-axis moving bits; sigmakoki has none, so
        its motion is the position changing between polls (the driver's own
        settle detection — the speed level persists after a stop)."""
        if "moving" in parsed:
            return bool(parsed["moving"])
        pos = (parsed["x"], parsed["y"], parsed["z"])
        moved = self._last_pos is not None and pos != self._last_pos
        self._last_pos = pos
        return moved

    def set_telem(self, parsed: dict) -> None:
        # The values are padded to a FIXED field width: the readout keeps
        # the compact "·" style while the text stops jittering horizontally
        # as digits come and go (a monospace font alone does not fix that).
        if "r_deg" in parsed:  # zolix (its r_deg is the discriminator)
            self._pos.setText(centred_readout(
                f"{parsed['x_um']:6.1f} · {parsed['y_um']:6.1f} µm · "
                f"{parsed['r_deg']:5.2f}°"))
        else:  # sigmakoki (steps → µm done by the strip's caller)
            self._pos.setText(centred_readout(
                f"{parsed['x_um']:6.1f} · {parsed['y_um']:6.1f} · "
                f"{parsed['z_um']:6.1f} µm"))
        # ONE code path for both stages from here down.
        limits = parsed.get("limits") or {}
        for key, dot in self._dots.items():
            dot.set_on(bool(limits.get(key)))
        if self._has_estop:
            estop = bool(parsed.get("estop"))
            self._estop.setText("E-STOP" if estop else "")
            self._estop.setVisible(estop)
        moving = self._is_moving(parsed)
        self._moving.set_state("MOVE" if moving else "IDLE",
                               "strip_mov" if moving else "strip_mov_idle")


def format_focus_pos(pos_steps: int, um_per_step: float) -> str:
    """The strip's FOCUS readout: µm + steps."""
    return f"{pos_steps * um_per_step:.1f} µm · {pos_steps} st"


class HardwareStrip(QWidget):
    """Shared bottom strip — one instance, owned by MainWindow."""

    def __init__(self, manager, settings, input_system=None, state=None, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._manager = manager
        self.reload_settings()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(6)

        self._xyr = _StageSection("XYR STAGE", "zolix",
                                  "Zolix XYR sample stage",
                                  manager, settings, estop=True)
        self._xyr.setMinimumWidth(240)
        layout.addWidget(self._xyr, stretch=_SECTION_STRETCH)
        self._xyz = _StageSection("XYZ STAGE", "sigmakoki",
                                  "SigmaKoki XYZ transfer stage",
                                  manager, settings, estop=False)
        self._xyz.setMinimumWidth(240)
        layout.addWidget(self._xyz, stretch=_SECTION_STRETCH)

        focus = _Section("FOCUS", "Focus stage (no limit sensor)")
        focus.setMinimumWidth(240)
        self._focus_pos = QLabel("—")
        self._focus_pos.setObjectName("readout")
        self._focus_pos.setMinimumWidth(160)  # fits "123.4 µm · 123456 st"
        self._focus_pos.setAlignment(Qt.AlignmentFlag.AlignCenter)
        focus.add(self._focus_pos)
        # The same widget as the stages' state word — same slot, same
        # colours, same wording.
        self._focus_state = _StateWord("—")
        focus.add(self._focus_state)
        # THE elastic field of this panel. It gets the AppState because its
        # jog-speed readout must show the speed the dispatcher commands (the
        # objective's focus multiplier scales that curve — see the widget).
        self._triggers = TriggerBarWidget(settings, state=state)
        focus.add(self._triggers, stretch=1)
        layout.addWidget(focus, stretch=_SECTION_STRETCH)

        # TEMP carries only three short values, in a panel as wide as the
        # stages': all three are elastic with the SAME stretch and the same
        # minimum, so they always come out the same width, and each is a
        # "NAME: VALUE" field in the same box style as every other readout.
        temp = _Section("TEMP", "Yudian AI-828 temperature controller")
        temp.setMinimumWidth(240)
        self._temp_pv = QLabel("PV: —")
        self._temp_pv.setObjectName("readout")
        self._temp_sv = QLabel("SV: —")
        self._temp_sv.setObjectName("readout")
        self._temp_out = _ValueLabel("PWR: —")
        for field in (self._temp_pv, self._temp_sv, self._temp_out):
            field.setMinimumWidth(90)
            field.setAlignment(Qt.AlignmentFlag.AlignCenter)
            temp.add(field, stretch=1)
        layout.addWidget(temp, stretch=_SECTION_STRETCH)

        layout.addStretch(0)
        self.setFixedHeight(56)
        self.setMinimumWidth(0)   # the sections compress; nothing clips

    #: Below this width the four sections cannot show every indicator —
    #: see _StageSection.set_compact. The number is the widest a
    #: NON-compact stage panel needs: 12 margins + 57 enable + 8 + 59 state
    #: word + 8 + 180 readout + 8 + 44 dots = 376, so 4×376 + 3×6 + 12 =
    #: 1534 for the whole row. (It used to be 1150, which let the panels
    #: overflow their own frames between 1150 and ~1500.)
    COMPACT_WIDTH = 1540

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        compact = self.width() < self.COMPACT_WIDTH
        for section in (self._xyr, self._xyz):
            section.set_compact(compact)
        # Compact minimums are chosen so each row still ADDS UP inside its
        # panel at the 1024 px floor with room to spare: focus
        # 12+100+8+59+8+50, TEMP 12+3×72+2×8, stage 12+57+8+59+8+92 (the
        # state word's 59 is fixed and cannot give).
        self._triggers.setMinimumWidth(50 if compact else 130)
        self._focus_pos.setMinimumWidth(100 if compact else 140)
        for field in (self._temp_pv, self._temp_sv, self._temp_out):
            field.setMinimumWidth(72 if compact else 90)

    def reload_settings(self) -> None:
        """Re-read the scale factors the readouts convert with.

        They are cached at construction, so a Preferences edit to
        µm-per-pulse would otherwise keep converting with the old value
        until the next restart."""
        cfg_z = self._settings.device("zolix")
        cfg_s = self._settings.device("sigmakoki")
        self._um_xy_zolix = float(cfg_z.get("um_per_pulse_xy", 0.625))
        self._um_xy_sig = float(cfg_s.get("um_per_step_xy", 0.5))
        self._um_z_sig = float(cfg_s.get("um_per_step_z", 0.25))
        self._um_focus = float(self._settings.device("focus").get("um_per_step", 0.2))
        # The trigger readout caches the focus curve + jog inversion too.
        # getattr: reload_settings() also runs from __init__, BEFORE the
        # trigger bar exists.
        triggers = getattr(self, "_triggers", None)
        if triggers is not None:
            triggers.reload_settings()

    def trigger_bar(self) -> TriggerBarWidget:
        return self._triggers

    def update_telem(self, device_key: str, payload: dict) -> None:
        # The enable gate can change from outside this checkbox (the
        # gamepad's Start button toggles the D-pad-selected stage).
        if "enabled" in payload:
            section = {"zolix": self._xyr, "sigmakoki": self._xyz}.get(device_key)
            if section is not None:
                box = section._enable
                box.blockSignals(True)
                box.setChecked(bool(payload["enabled"]))
                box.blockSignals(False)
        if device_key == "zolix":
            self._xyr.set_telem(parse_zolix(payload))
        elif device_key == "sigmakoki":
            parsed = parse_sigmakoki(payload)
            parsed["x_um"] = parsed["x"] * self._um_xy_sig
            parsed["y_um"] = parsed["y"] * self._um_xy_sig
            parsed["z_um"] = parsed["z"] * self._um_z_sig
            self._xyz.set_telem(parsed)
        elif device_key == "focus":
            parsed = parse_focus(payload)
            self._focus_pos.setText(
                format_focus_pos(parsed["pos"], self._um_focus))
            blocked = parsed["blocked"] not in ("0", "")
            # The whole word is one slot wide, so the blocked state reads
            # BLOCKED rather than "⚠ BLOCKED": the warn colour and weight
            # carry the alarm, and the trigger bar never has to give up
            # 40 px to a glyph.
            self._focus_state.set_state(
                "BLOCKED" if blocked else parsed["mode"],
                # Same state styling as the stages' word: dim at rest, green
                # while the axis moves (CONT = continuous jog, TRAP = a
                # positioned move), warn when a direction is blocked.
                "strip_warn" if blocked
                else ("strip_mov" if parsed["mode"] in ("CONT", "TRAP")
                      else "strip_mov_idle"))
        elif device_key == "yudian":
            parsed = parse_yudian(payload)
            if parsed["pv"] is not None:
                self._temp_pv.setText(f"PV: {parsed['pv']:.1f} °C")
            if parsed["sv"] is not None:
                self._temp_sv.setText(f"SV: {parsed['sv']:.1f} °C")
            if parsed["out"] is not None:
                self._temp_out.setText(f"PWR: {parsed['out']:.0f}%")
            # Heat-status colour (the reference project's ladder).
            self._temp_out.set_colour(
                temp_power_color(parsed["pv"], parsed["sv"], parsed["out"]))
