"""The camera-and-CV half of the Sample Finding tab.

Three groups and one plot, all of them about the same question: what is in
this frame, and how do I make it easier to see?

- **Sample colour** (the quick-access group, pinned above the scroll) is the
  colour stage's editor, promoted out of the chain because it is the one
  control an operator touches constantly. It is also the curve's centre —
  one colour, two uses, so what you pick is what you amplify.
- **Pre-processing** is the chain that runs before anything looks at the
  frame: shade correction, denoise, the tone operations, and the
  local-contrast curve. That curve is a *matched gain* — steep at the
  picked colour and pinned everywhere it matters — and the plot above the
  controls draws it, because a tone curve that cannot be seen is a
  parameter you can only guess at.
- **Identification** is the rest of the chain: the gates that decide which
  blobs survive, with the per-stage counts that make tuning it feel like
  tuning a filter rather than guessing.

The stage editors are data-driven from the stages themselves (``LABEL``,
``RANGES``, the dataclass fields), so adding a stage needs no UI code.
"""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from talos.cv.identify import (METHOD_WINDOW, IdentifyConfig, Stage,
                               valid_hex)
from talos.cv.preprocess import (DN_MAX, LocalContrast, PreprocessConfig,
                                 channel_curve, effective_gain,
                                 effective_width)
from talos.ui import theme
from talos.ui.widgets.segmented import SegmentedToggle

#: Parameter names as the operator reads them. Anything not listed falls
#: back to the field name with underscores turned into spaces.
PARAM_LABELS = {
    "min_area_um2": "Min area (µm²)",
    "max_area_um2": "Max area (µm²)",
    "margin_px": "Margin (px)",
    "gap_px": "Join within (px)",
    "min_saturation": "Min saturation",
    "min_value": "Min brightness",
    "min_edge_strength": "Min edge",
    # The two halves of what used to be one "Tolerance": the axis each one
    # moves is in the name, because that is the whole reason they are two.
    "tolerance": "Hue tolerance",
    "spread": "Shade spread",
    "kernel": "Kernel",
}

#: The dropper's patch: a radius in FRAME pixels. 4 is the 9-px disc the
#: sampler has always used; 16 is a 33-px patch, past which "which flake did
#: I click" stops being a question the operator can answer on screen.
_PATCH_DEFAULT_PX = 4
_PATCH_MIN_PX = 1
_PATCH_MAX_PX = 16
_PATCH_READOUT_W = 72


def _patch_diameter_bounds() -> tuple[int, int]:
    """The row works in diameters (2r + 1), the setting in radii."""
    return (2 * _PATCH_MIN_PX + 1, 2 * _PATCH_MAX_PX + 1)

#: What a parameter does, for the row's tooltip. Only where the name cannot
#: carry it — the colour's shade window is the one an operator is most
#: likely to set by trial and error.
PARAM_TIPS = {
    "tolerance": "How far the hue may drift and still count as the same "
                 "material — illumination, not thickness.",
    "spread": "How much lighter or darker a pixel may be and still count. "
              "Layer thickness shows up here, not in the hue: narrow it to "
              "keep a thin flake and reject the thicker one next to it.",
    "min_saturation": "Reject anything less saturated than this — a floor "
                      "for grey and white backgrounds.",
    "min_value": "Reject anything darker than this.",
}


def param_label(name: str) -> str:
    return PARAM_LABELS.get(name, name.replace("_", " ").capitalize())


#: What each matching method means, under its own rows — the two lines an
#: operator needs to read the parameters they are looking at.
METHOD_NOTES = {
    "window": "A box: hue ±0.9 × Tolerance, saturation and shade ±2.55 × "
              "Spread. Only method that separates two thicknesses.",
    "hsv_distance": "One radius: hue counts as 0.9 × Tolerance, saturation "
                    "and shade as 2.55 ×. Spread is not used.",
    "rgb_distance": "One radius in RGB (2.55 × Tolerance DN). No hue axis: "
                    "sharpest for one shade under one lamp, first to fail "
                    "when the illumination moves. Spread is not used.",
}


def _set_quietly(box, value) -> None:
    """Move a spin box without telling anyone (see ``_number_and_slider``)."""
    box.blockSignals(True)
    box.setValue(value if isinstance(box, QSpinBox) else float(value))
    box.blockSignals(False)


def _spin(value, limits, suffix: str = "") -> QWidget:
    """A spin box matching the parameter's own type."""
    lo, hi, step = limits
    if isinstance(value, float):
        box = QDoubleSpinBox()
        box.setDecimals(2)
    else:
        box = QSpinBox()
    box.setRange(lo, hi)
    box.setSingleStep(step)
    box.setValue(value)
    if suffix:
        box.setSuffix(suffix)
    return box


# ----------------------------------------------------------------------
# The curve plot
# ----------------------------------------------------------------------

class CurvePlot(QWidget):
    """The local-contrast curve, drawn: three channels and the identity.

    Worth the pixels because the curve is invisible otherwise. Every
    parameter here is a number that describes a shape — where the steep
    part is, how steep, how wide — and the shape is what the operator is
    actually choosing. It also shows what the fitting did: when a request
    is impossible the band narrows, and that is visible here as a steeper,
    narrower S rather than as a parameter that quietly stopped working.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("curve_plot")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumHeight(96)
        self._centre: tuple[int, int, int] | None = None
        self._gain = 1.0
        self._width = 32.0
        self.setToolTip("The three channel curves. The diagonal is the "
                        "identity; where the colour curves rise above it, "
                        "differences are being amplified.")

    def set_curves(self, centre_rgb, gain: float, width: float) -> None:
        self._centre = tuple(centre_rgb) if centre_rgb else None
        self._gain = float(gain)
        self._width = float(width)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(2.0, 2.0, max(1.0, self.width() - 4.0),
                      max(1.0, self.height() - 4.0))
        painter.fillRect(self.rect(), QColor(theme.BG))

        def at(value: float, axis: str) -> float:
            fraction = max(0.0, min(1.0, value / DN_MAX))
            if axis == "x":
                return rect.left() + fraction * rect.width()
            return rect.bottom() - fraction * rect.height()

        painter.setPen(QPen(QColor(theme.BORDER), 1))
        painter.drawRect(rect)
        # Quarters, so a curve can be read rather than only recognised:
        # without them "it rises about there" is the most the eye can say.
        painter.setPen(QPen(QColor(theme.BORDER), 1))
        for fraction in (0.25, 0.5, 0.75):
            x = rect.left() + fraction * rect.width()
            y = rect.top() + fraction * rect.height()
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        # the identity, for reference — what "no change" looks like
        painter.setPen(QPen(QColor(theme.TEXT_DIM), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(at(0, "x"), at(0, "y")),
                         QPointF(at(DN_MAX, "x"), at(DN_MAX, "y")))

        if self._centre is None or self._gain <= 1.0 + 1e-9:
            painter.setPen(QColor(theme.TEXT_DIM))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             "Pick a colour to centre the curve")
            painter.end()
            return

        samples = np.linspace(0.0, DN_MAX, 128)
        for channel, colour in enumerate((theme.DANGER, "#3fb950", "#58a6ff")):
            curve = channel_curve(float(self._centre[channel]), self._gain,
                                  self._width, n=128)
            painter.setPen(QPen(QColor(colour), 1))
            previous = QPointF(at(samples[0], "x"), at(float(curve[0]), "y"))
            for index in range(1, len(samples)):
                point = QPointF(at(samples[index], "x"),
                                at(float(curve[index]), "y"))
                painter.drawLine(previous, point)
                previous = point

        # the fixed point: the picked colour renders as itself
        centre = float(sum(self._centre)) / 3.0
        painter.setPen(QPen(QColor(theme.ACCENT), 1))
        painter.setBrush(QColor(theme.ACCENT))
        painter.drawEllipse(QPointF(at(centre, "x"), at(centre, "y")), 3.0, 3.0)
        painter.end()


# ----------------------------------------------------------------------
# Stage editing (data-driven)
# ----------------------------------------------------------------------

class StageEditor(QFrame):
    """One stage: an enable box and a row per parameter.

    Built from the stage's own dataclass fields and ``RANGES``, so a new
    stage appears here without this file changing. ``read()`` rebuilds the
    stage with ``dataclasses.replace`` — the worker is handed a fresh
    object every job and never sees a half-edited one.
    """

    sig_changed = Signal()
    sig_dropper = Signal()

    def __init__(self, stage: Stage, parent: QWidget | None = None,
                 show_enable: bool = True, framed: bool = True,
                 sliders: bool = False,
                 only: tuple[str, ...] | None = None):
        super().__init__(parent)
        self._stage = stage
        #: Build only these fields. The colour stage is edited in two places
        #: — the pinned picker (its colour, nothing else) and the pipeline
        #: card (everything) — and the second editor must be a PROJECTION of
        #: the same stage rather than a second copy of it.
        self._only = tuple(only) if only else None
        #: Give every numeric row a slider under its number box, the way the
        #: camera's controls are laid out. Off by default: the values in a
        #: gate are typed once and left, while the colour's three are
        #: hunted for by eye while watching the mask change.
        self._sliders = bool(sliders)
        #: True while a programmatic load is writing the widgets: those
        #: writes must not look like edits (a workspace switch would persist
        #: the settings file and re-run the mask for a change nobody made).
        self._loading = False
        if framed:
            self.setObjectName("card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 6)
        outer.setSpacing(2)

        self.enable = QCheckBox(stage.LABEL)
        self.enable.setChecked(bool(stage.enabled))
        font = self.enable.font()
        font.setBold(True)          # the stage name outranks its parameters
        self.enable.setFont(font)
        self.enable.setVisible(show_enable)
        if show_enable:
            outer.addWidget(self.enable)
        form = QFormLayout()
        form.setContentsMargins(2, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setVerticalSpacing(4)
        outer.addLayout(form)

        self._editors: dict[str, tuple] = {}
        #: The row's label and its container, beside ``_editors`` rather than
        #: inside it: two tests pin the shape of those tuples (a 2- or
        #: 3-tuple whose [1] is the value widget), and a label is not a value.
        self._rows: dict[str, tuple[QLabel, QWidget]] = {}
        #: The build order, so a row taken out for one method can be put back
        #: in its place when another method needs it.
        self._row_order: list[str] = []
        self._form = form
        declared = [f.name for f in fields(stage) if f.name != "enabled"]
        if self._only is not None:
            declared = [n for n in declared if n in self._only]
        choices = getattr(stage, "CHOICES", None) or {}
        # A named choice LEADS the form: it decides what the numbers under it
        # mean. (Built in that order rather than moved afterwards — Qt's
        # QFormLayout.removeRow deletes the widgets it removes.)
        ordered = ([n for n in declared if n in choices]
                   + [n for n in declared if n not in choices])
        self._row_order = list(ordered)
        for name in ordered:
            self._add_field(form, stage, name, getattr(stage, name))

        #: What the current method means, under its rows. Only a stage with a
        #: named choice gets one (see _sync_dependent_rows).
        self._method_note = QLabel("")
        self._method_note.setObjectName("dim")
        self._method_note.setWordWrap(True)
        self._method_note.setVisible(False)
        outer.addWidget(self._method_note)
        # Slack goes to the bottom of the card, not into whichever label or
        # row can grow: a stretched card otherwise draws the surplus as a
        # blank band between two parameters, which reads as a missing row.
        outer.addStretch(1)

        self.enable.toggled.connect(self._on_changed)
        self._sync_dependent_rows()

    # ------------------------------------------------------------------

    def _sync_dependent_rows(self) -> None:
        """Which rows the current method HAS, and what they mean.

        The colour match's parameters are not the same set for every method:
        a distance method has no separate shade window, so ``spread`` is not
        greyed out but *gone*, and the tolerance is a radius rather than a
        hue half-width. A row that is merely disabled still reads as "part of
        this method, temporarily unavailable" — and a label saying "Hue
        tolerance" over an RGB radius is the same class of lie as a button
        that refuses a click.
        """
        entry = self._editors.get("method")
        if entry is None:
            return
        method = entry[1].value()
        window = method == METHOD_WINDOW

        def show(name: str, visible: bool) -> None:
            """A row the method does not have is TAKEN OUT of the form.

            Hiding its widgets leaves the row's geometry slot behind, and a
            card that is stretched taller than its content then draws the
            freed space as a blank band exactly the height of the parameter
            that is gone — which reads as a rendering fault rather than as
            "this method has no shade window". ``takeRow`` keeps the widgets
            (they hold the operator's value) and re-inserts them in field
            order when the method comes back.
            """
            if name not in self._rows:
                return
            label, container = self._rows[name]
            row, _role = self._form.getWidgetPosition(container)
            if not visible:
                if row >= 0:
                    self._form.takeRow(row)
                label.hide()          # out of the layout, still a child
                container.hide()
                return
            if row < 0:
                self._form.insertRow(self._visible_index(name), label,
                                     container)
            label.show()
            container.show()

        if "tolerance" in self._rows:
            label, container = self._rows["tolerance"]
            label.setText("Hue tolerance" if window else "Tolerance")
            tip = (PARAM_TIPS.get("tolerance", "") if window else
                   "How far a pixel may be from the picked colour, as one "
                   "radius: hue counts as 0.9× and saturation/brightness as "
                   "2.55×, but a pixel must be close on the whole rather "
                   "than close on each axis.")
            for widget in (container, *self._editors["tolerance"][1:]):
                widget.setToolTip(tip)

        show("spread", window)
        note = METHOD_NOTES.get(method, "")
        if self._method_note is not None:
            self._method_note.setText(note)
            self._method_note.setVisible(bool(note))

    def _visible_index(self, name: str) -> int:
        """Where ``name`` goes among the rows currently IN the form."""
        index = 0
        for other in self._row_order:
            if other == name:
                break
            if other not in self._rows:
                continue
            _label, widget = self._rows[other]
            if self._form.getWidgetPosition(widget)[0] >= 0:
                index += 1
        return index

    # ------------------------------------------------------------------

    def _add_field(self, form: QFormLayout, stage: Stage, name: str,
                   value) -> None:
        if name == "hex_color":
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            edit = QLineEdit(str(value))
            edit.setMaxLength(7)
            swatch = QLabel()
            swatch.setFixedSize(18, 18)
            swatch.setToolTip("The sampled colour")
            dropper = QPushButton("Drop")
            dropper.setObjectName("compact")
            dropper.setToolTip("Click the live view to sample a colour")
            dialog = QPushButton("…")
            dialog.setObjectName("compact")
            dialog.setFixedWidth(24)
            dialog.setToolTip("Choose the colour from a dialog")
            layout.addWidget(edit, 1)
            layout.addWidget(swatch)
            layout.addWidget(dropper)
            layout.addWidget(dialog)
            self._editors[name] = ("hex", edit, swatch, dropper, dialog)
            label = QLabel("Colour")
            form.addRow(label, row)
            self._rows[name] = (label, row)
            edit.editingFinished.connect(self._normalise_hex)
            dropper.clicked.connect(self.sig_dropper)
            dialog.clicked.connect(self._pick_from_dialog)
            self._refresh_swatch()
        elif isinstance(value, str):
            # A named choice, declared by the stage itself (CHOICES, the
            # sibling of RANGES) — a plain string with no declaration is a
            # programming error, and guessing would give it a spin box.
            options = getattr(stage, "CHOICES", {}).get(name)
            if not options:
                raise ValueError(
                    f"{type(stage).__name__}.{name} is a string field with no "
                    f"CHOICES entry — the editor cannot build a row for it")
            toggle = SegmentedToggle(list(options), value)
            toggle.sig_changed.connect(self._on_changed)
            self._editors[name] = ("choice", toggle)
            label = QLabel(param_label(name))
            form.addRow(label, toggle)
            self._rows[name] = (label, toggle)
        elif isinstance(value, bool):
            check = QCheckBox()
            check.setChecked(bool(value))
            self._editors[name] = ("bool", check)
            label = QLabel(param_label(name))
            form.addRow(label, check)
            self._rows[name] = (label, check)
            check.toggled.connect(self._on_changed)
        else:
            limits = getattr(stage, "RANGES", {}).get(name, (0, 255, 1))
            box = _spin(value, limits)
            tip = PARAM_TIPS.get(name)
            if tip:
                box.setToolTip(tip)
            if self._sliders and limits[1] > limits[0]:
                container, slider = self._number_and_slider(box, limits)
                # Registered like every other row: ``stage()`` reads its
                # parameters back out of ``_editors``, so a row that skips
                # it is a parameter the operator sets and the pipeline
                # never sees.
                self._editors[name] = ("num", box, slider)
                label = QLabel(param_label(name))
                form.addRow(label, container)
                self._rows[name] = (label, container)
                box.valueChanged.connect(self._on_changed)
                return
            self._editors[name] = ("num", box)
            label = QLabel(param_label(name))
            form.addRow(label, box)
            self._rows[name] = (label, box)
            box.valueChanged.connect(self._on_changed)

    def _number_and_slider(self, box, limits) -> tuple[QWidget, QSlider]:
        """The camera's row: the number, and a slider under it.

        The slider moves the NUMBER and nothing else; the change is applied
        once, on release. Dragging must not re-save the settings file and
        re-run the mask on every pixel — that is why the camera's exposure
        row is built the same way — while the number box still applies as
        it is typed, which is how the value is set precisely.
        """
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(int(limits[0]), int(limits[1]))
        step = max(1, int(limits[2]))
        slider.setSingleStep(step)
        slider.setPageStep(step * 4)
        slider.setValue(int(round(box.value())))
        slider.valueChanged.connect(
            lambda value, b=box: _set_quietly(b, value))
        slider.sliderReleased.connect(self._on_changed)
        box.valueChanged.connect(
            lambda value, s=slider: s.setValue(int(round(value))))

        container = QWidget()
        # Fixed height: a container that can grow swallows whatever slack the
        # card has, and the space appears as a gap between this row's number
        # and the row below it — which reads as a missing parameter.
        container.setSizePolicy(QSizePolicy.Policy.Preferred,
                                QSizePolicy.Policy.Fixed)
        row = QVBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(box)
        row.addWidget(slider)
        if box.toolTip():
            container.setToolTip(box.toolTip())
            slider.setToolTip(box.toolTip())
        return container, slider

    def add_row(self, label: str, widget: QWidget) -> QLabel:
        """One more row in the form, for a control the stage does not own.

        It goes through the same QFormLayout as the parameters, so it lines
        up with them: a control beside the stage's own values that starts at
        a different x reads as belonging to something else.
        """
        text = QLabel(label)
        self._form.addRow(text, widget)
        return text

    def load(self, stage: Stage) -> None:
        """Point the editor at ``stage`` and show it — the one path every
        reload takes. SILENT: the widgets are written with signals off, so a
        reload cannot look like an edit.

        The kinds are dispatched here ONCE. A widget kind added to
        ``_add_field`` and missed here would work until the first workspace
        switch and then raise — or, worse, show a stale value next to a
        pipeline that is using the new one.
        """
        self._stage = stage
        self._loading = True
        try:
            self.enable.setChecked(bool(stage.enabled))
            for name, spec in self._editors.items():
                value = getattr(stage, name, None)
                if spec[0] == "hex":
                    if value is not None:
                        spec[1].setText(valid_hex(str(value)))
                elif spec[0] == "bool":
                    spec[1].setChecked(bool(value))
                elif spec[0] == "choice":
                    spec[1].set_value(value)
                elif value is not None:
                    spec[1].setValue(value)
        finally:
            self._loading = False
        self._refresh_swatch()
        self._sync_dependent_rows()

    def _normalise_hex(self) -> None:
        """Put the CLAMPED colour back in the field, then apply.

        The field is free text and the swatch shows whatever ``valid_hex``
        made of it, so a typo left the two disagreeing — the colour the mask
        would search was the swatch's while the box showed what was typed.
        """
        entry = self._editors.get("hex_color")
        if entry is not None:
            text = entry[1].text()
            clean = valid_hex(text, self.hex_color())
            if clean != text:
                entry[1].setText(clean)
        self._on_changed()

    def _refresh_swatch(self) -> None:
        entry = self._editors.get("hex_color")
        if entry is None:
            return
        colour = QColor(valid_hex(entry[1].text(),
                                  getattr(self._stage, "hex_color",
                                          "#c8a2c8")))
        entry[2].setStyleSheet(
            f"background: {colour.name()}; border: 1px solid #3f3f46;")

    def show_hex(self, text: str) -> None:
        """Set the colour field WITHOUT notifying — a mirror being refreshed.

        The two editors that show the colour (the pinned picker and the
        pipeline card) have to agree, and whichever was edited must be able
        to tell the other without the other reporting it as a new edit.
        """
        entry = self._editors.get("hex_color")
        if entry is None or not text:
            return
        self._loading = True
        try:
            entry[1].setText(valid_hex(text))
        finally:
            self._loading = False
        self._refresh_swatch()

    def _on_changed(self, *_args) -> None:
        if self._loading:
            return              # a reload, not an edit (see load/show_hex)
        self._refresh_swatch()
        self._sync_dependent_rows()
        self.sig_changed.emit()

    def _pick_from_dialog(self) -> None:
        self.set_hex(QColorDialog.getColor(
            QColor(self.hex_color()), self, "Sample colour").name())

    # ------------------------------------------------------------------

    def hex_color(self) -> str:
        entry = self._editors.get("hex_color")
        if entry is None:
            return getattr(self._stage, "hex_color", "#c8a2c8")
        return valid_hex(entry[1].text(), getattr(self._stage, "hex_color",
                                                  "#c8a2c8"))

    def set_hex(self, text: str) -> None:
        entry = self._editors.get("hex_color")
        if entry is None or not text:
            return
        entry[1].setText(valid_hex(text))
        self._on_changed()

    def stage(self) -> Stage:
        """The stage as the widgets currently describe it."""
        kwargs = {"enabled": self.enable.isChecked()}
        for name, spec in self._editors.items():
            if spec[0] == "hex":
                kwargs[name] = valid_hex(spec[1].text(),
                                         getattr(self._stage, name))
            elif spec[0] == "bool":
                kwargs[name] = spec[1].isChecked()
            else:
                kwargs[name] = spec[1].value()
        try:
            return replace(self._stage, **kwargs)
        except Exception:  # noqa: BLE001 - a bad edit must not stop a run
            return self._stage

    def set_enabled_controls(self, enabled: bool) -> None:
        for spec in self._editors.values():
            for widget in spec[1:]:
                widget.setEnabled(enabled)


# ----------------------------------------------------------------------
# The three groups
# ----------------------------------------------------------------------

class ColourGroup(QGroupBox):
    """The PICKER: the colour that is being looked for, and how it is taken.

    Pinned at the top because reaching for a colour is the one thing done
    constantly at the microscope — and because it is what the curve is
    centred on, so the swatch is also a readout of the fixed point.

    It is a PROJECTION of the colour stage, not a second copy of it: only the
    colour is here (plus the patch size, which is the dropper's own setting),
    and the parameters that shape the match live in the Identification card
    where the pipeline is. Editing the colour in either place updates the
    other; the Identification card's editor is the one the config comes from.
    """

    sig_changed = Signal()
    sig_dropper = Signal()
    #: The dropper's patch size changed (radius, in frame pixels). Its own
    #: signal, not ``sig_changed``: the mask does not read it, so re-running
    #: the pipeline for it would be work for nobody.
    sig_patch_changed = Signal(int)

    def __init__(self, settings, parent: QWidget | None = None):
        super().__init__("Sample colour", parent)
        self._settings = settings
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(4)

        # Only the colour: the method and its parameters belong with the
        # pipeline they shape (see the class docstring).
        self.editor = StageEditor(self._colour_stage(), self,
                                  show_enable=False, framed=False,
                                  only=("hex_color",))
        self.editor.sig_changed.connect(self._on_changed)
        self.editor.sig_dropper.connect(self.sig_dropper)
        layout.addWidget(self.editor)
        # After the editor exists: the row is added to ITS form, so it lines
        # up with the stage's own rows.
        self._build_patch_row()

        self._hint = QLabel(self.DEFAULT_HINT)
        self._hint.setObjectName("dim")
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)
        layout.addStretch(1)      # see IdentifyGroup: slack goes to the bottom

    def _build_patch_row(self) -> None:
        """How big a disc the dropper averages.

        A sampling aid, NOT a ``ColourStage`` field: the mask never reads it,
        and a parameter in the stage is a parameter the pipeline is expected
        to use. It lives in ``ui`` beside the view mode, and the circle the
        cursor draws is this number — which is the point of having it, since
        at fit-to-window scale a 9-px patch is about two screen pixels.
        """
        # The row is in DIAMETERS, because that is what the disc spans and
        # what the operator judges on screen; the stored value is the radius
        # the sampler takes. Odd numbers only (2r + 1), like the camera's
        # exposure row: number and slider, the slider applying on release.
        self._patch_slider = QSlider(Qt.Orientation.Horizontal)
        self._patch_slider.setRange(*_patch_diameter_bounds())
        self._patch_slider.setSingleStep(2)
        self._patch_slider.setPageStep(4)
        self._patch_slider.setValue(2 * self._stored_patch_radius() + 1)
        self._patch_box = QSpinBox()
        self._patch_box.setRange(*_patch_diameter_bounds())
        self._patch_box.setSingleStep(2)
        self._patch_box.setSuffix(" px")
        self._patch_box.setMaximumWidth(_PATCH_READOUT_W)
        self._patch_box.setAlignment(Qt.AlignmentFlag.AlignRight)
        _set_quietly(self._patch_box, self._patch_slider.value())
        self._patch_slider.valueChanged.connect(
            lambda value: _set_quietly(self._patch_box, value))
        self._patch_slider.sliderReleased.connect(self._on_patch_released)
        self._patch_box.valueChanged.connect(self._on_patch_moved)
        self._patch_box.editingFinished.connect(self._on_patch_released)

        tip = ("How large a disc the dropper averages, in FRAME pixels — the "
               "circle follows the cursor so you can see it. Bigger averages "
               "away sensor noise and makes it easier to stay inside one "
               "material; smaller is more precise on a small flake.")
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self._patch_slider, 1)
        row.addWidget(self._patch_box)
        for widget in (self._patch_slider, self._patch_box, holder):
            widget.setToolTip(tip)
        self.editor.add_row("Patch", holder)
        self._sync_patch_pair()

    def _stored_patch_radius(self) -> int:
        try:
            value = int(self._settings.section("ui").get("pick_radius_px",
                                                         _PATCH_DEFAULT_PX))
        except (TypeError, ValueError):
            return _PATCH_DEFAULT_PX
        return max(_PATCH_MIN_PX, min(_PATCH_MAX_PX, value))

    def _sync_patch_pair(self) -> None:
        """The slider and the number box show the same diameter, and a box
        typed out of step snaps to an odd one (the disc spans 2r + 1)."""
        value = self._patch_slider.value()
        if value % 2 == 0:
            value += 1
            self._patch_slider.setValue(value)
        _set_quietly(self._patch_box, value)

    def _on_patch_moved(self, _value: int) -> None:
        self._sync_patch_pair()
        self.sig_patch_changed.emit(self.pick_radius())

    def _on_patch_released(self) -> None:
        self._settings.section("ui")["pick_radius_px"] = self.pick_radius()
        self._settings.save()

    def pick_radius(self) -> int:
        return max(_PATCH_MIN_PX, (int(self._patch_slider.value()) - 1) // 2)

    def set_pick_radius(self, radius: int) -> None:
        """Set the patch size (a stored value being applied)."""
        radius = max(_PATCH_MIN_PX, min(_PATCH_MAX_PX, int(radius)))
        self._patch_slider.setValue(2 * radius + 1)
        self._sync_patch_pair()

    #: What the card's own line says when it has nothing else to report.
    DEFAULT_HINT = "Also centres the local-contrast curve."

    def note(self, text: str, tone: str | None = None) -> None:
        """The card's one line, where the operator is already looking.

        Used by the dropper to say what it sampled and whether the patch it
        averaged was uniform: a 9-px disc that straddles a flake's edge
        returns a colour neither the flake nor the substrate has, and the
        click that produced it looked perfectly reasonable on screen.
        """
        self._hint.setText(text or self.DEFAULT_HINT)
        wanted = tone or "dim"
        if self._hint.objectName() != wanted:
            self._hint.setObjectName(wanted)
            # Qt only re-evaluates the stylesheet when told the widget
            # changed; without this the colour never moves.
            style = self._hint.style()
            style.unpolish(self._hint)
            style.polish(self._hint)

    def _colour_stage(self):
        stages = IdentifyConfig.from_dict(
            self._settings.section("identify")).stages
        return next((s for s in stages if s.NAME == "colour"),
                    IdentifyConfig().stage("colour"))

    def stage(self):
        return self.editor.stage()

    def hex_color(self) -> str:
        return self.editor.hex_color()

    def rgb(self) -> tuple[int, int, int]:
        raw = self.hex_color().lstrip("#")
        return (int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))

    def set_hex(self, text: str) -> None:
        self.editor.set_hex(text)

    def show_hex(self, text: str) -> None:
        """Mirror a colour edited in the chain card, without reporting it
        back — the two views have to agree, and neither may treat the other's
        refresh as a new edit."""
        self.editor.show_hex(text)

    def reload(self) -> None:
        """Re-read the stored colour (a settings change made elsewhere)."""
        self.editor.load(self._colour_stage())

    def _on_changed(self) -> None:
        self.sig_changed.emit()


class PreprocessGroup(QGroupBox):
    """The pre-processing chain: what happens to the frame before anyone
    looks at it. Off entirely until it is switched on."""

    sig_changed = Signal()

    def __init__(self, settings, parent: QWidget | None = None):
        super().__init__("Pre-processing", parent)
        self._settings = settings
        self.setCheckable(False)
        cfg = PreprocessConfig.from_dict(settings.section("preprocess"))
        self._building = True
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(4)

        self.enable = QCheckBox("Enable pre-processing")
        self.enable.setChecked(cfg.enabled)
        font = self.enable.font()
        font.setBold(True)
        self.enable.setFont(font)
        self.enable.setToolTip(
            "The chain runs on the detection worker, never on the GUI\n"
            "thread — a slow filter delays the preview and nothing else.")
        layout.addWidget(self.enable)

        self.plot = CurvePlot(self)
        layout.addWidget(self.plot)

        self.local = _LocalContrastRows(cfg.local, self)
        self.local.sig_changed.connect(self._changed)
        layout.addWidget(self.local)

        self.spatial = _SpatialRows(cfg, self)
        self.spatial.sig_changed.connect(self._changed)
        layout.addWidget(self.spatial)

        order = QLabel("Order: denoise → curve. The curve is last so the "
                       "pick stays where it was picked, and the denoise "
                       "runs first because the curve amplifies noise as "
                       "eagerly as it amplifies a layer.")
        order.setObjectName("hint")
        order.setWordWrap(True)
        layout.addWidget(order)

        reset = QPushButton("Reset pre-processing")
        reset.setObjectName("compact")
        reset.clicked.connect(self._reset)
        layout.addWidget(reset)

        self.enable.toggled.connect(self._changed)
        self._building = False
        self._refresh_curve()

    # ------------------------------------------------------------------

    def config(self) -> PreprocessConfig:
        cfg = PreprocessConfig(enabled=self.enable.isChecked(),
                               **self.spatial.values())
        cfg.local = self.local.value()
        return cfg

    #: The picked colour, handed in by the workspace — the curve's centre
    #: and the mask's target are the same colour on purpose.
    _centre: tuple[int, int, int] | None = None

    def set_centre(self, centre_rgb) -> None:
        self._centre = tuple(centre_rgb) if centre_rgb else None
        self._refresh_curve()

    def reload(self) -> None:
        """Re-read the stored chain (a settings change made elsewhere)."""
        cfg = PreprocessConfig.from_dict(self._settings.section("preprocess"))
        self._building = True
        try:
            self.enable.setChecked(cfg.enabled)
            self.spatial.load(cfg)
            self.local.load(cfg.local)
        finally:
            self._building = False
        self._refresh_curve()

    def _refresh_curve(self) -> None:
        local = self.local.value()
        self.plot.set_curves(self._centre, local.gain, local.width)
        self.local.set_readout(self._centre, local.gain, local.width)

    def _changed(self, *_args) -> None:
        if self._building:
            return
        self._refresh_curve()
        self.sig_changed.emit()

    def _reset(self) -> None:
        fresh = PreprocessConfig()
        self._building = True
        try:
            self.enable.setChecked(fresh.enabled)
            self.spatial.load(fresh)
            self.local.load(fresh.local)
        finally:
            self._building = False
        self._refresh_curve()
        self.sig_changed.emit()


class _LocalContrastRows(QFrame):
    """The curve's own controls, and what it actually delivered."""

    sig_changed = Signal()

    def __init__(self, local: LocalContrast, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.enable = QCheckBox("Local contrast (matched gain)")
        self.enable.setToolTip(
            "Steepens the tone curve at the picked colour and flattens it\n"
            "elsewhere, so the few levels that separate one layer from the\n"
            "next get the display's range instead of the substrate.")
        layout.addWidget(self.enable)
        form = QFormLayout()
        form.setContentsMargins(14, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setVerticalSpacing(4)
        self.gain = _spin(local.gain, (1.0, 8.0, 0.5))
        self.gain.setToolTip("How much steeper, at the picked colour")
        self.width = _spin(local.width, (2.0, 120.0, 2.0))
        self.width.setToolTip("How many levels either side stay boosted")
        form.addRow("Gain", self.gain)
        form.addRow("Band ±", self.width)
        layout.addLayout(form)
        self.readout = QLabel("")
        self.readout.setObjectName("dim")
        self.readout.setWordWrap(True)
        layout.addWidget(self.readout)

        self.enable.setChecked(local.enabled)
        self.enable.toggled.connect(self.sig_changed)
        self.gain.valueChanged.connect(self.sig_changed)
        self.width.valueChanged.connect(self.sig_changed)

    def value(self) -> LocalContrast:
        return LocalContrast(enabled=self.enable.isChecked(),
                             gain=float(self.gain.value()),
                             width=float(self.width.value()))

    def load(self, local: LocalContrast) -> None:
        self.enable.setChecked(local.enabled)
        self.gain.setValue(local.gain)
        self.width.setValue(local.width)

    def set_readout(self, centre, gain: float, width: float) -> None:
        """What the curve will really do.

        The band narrows when the request cannot be honoured (a wide band
        boosted hard has to pay for itself with a shelf somewhere), and the
        gain is capped only when even the narrowest band cannot carry it.
        Saying so here is the difference between a slider that lies and one
        that reports.
        """
        if centre is None:
            self.readout.setText("Pick a colour to centre the curve.")
            return
        mean = float(sum(centre)) / 3.0
        used = effective_width(mean, gain, width)
        delivered = effective_gain(mean, gain, width)
        parts = [f"Steepest at #{centre[0]:02x}{centre[1]:02x}{centre[2]:02x}"]
        parts.append(f"· band ±{used:.0f} DN")
        if used < width - 0.5:
            parts.append(f"(asked ±{width:.0f})")
        parts.append(f"· gain ×{delivered:.2f}")
        if delivered < gain - 0.05:
            parts.append(f"(asked ×{gain:.2f})")
        self.readout.setText(" ".join(parts))


class _SpatialRows(QFrame):
    """Denoise — the one stage here with a real cost, and the one that has
    to run before the curve."""

    sig_changed = Signal()

    def __init__(self, cfg: PreprocessConfig, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.denoise_enable = QCheckBox("Denoise")
        self.denoise_enable.setToolTip(
            "Edge-preserving (bilateral): the flake's edges are what is\n"
            "being identified, so a smoother that softens them trades one\n"
            "problem for a worse one.")
        self.denoise_d = _spin(cfg.denoise.diameter, (1, 25, 2), " px")
        self.denoise_sc = _spin(cfg.denoise.sigma_color, (0.0, 150.0, 5.0))
        self.denoise_ss = _spin(cfg.denoise.sigma_space, (1.0, 25.0, 1.0))
        denoise_form = QFormLayout()
        denoise_form.setContentsMargins(14, 0, 0, 0)
        denoise_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        denoise_form.setVerticalSpacing(4)
        denoise_form.addRow("Diameter", self.denoise_d)
        denoise_form.addRow("σ colour", self.denoise_sc)
        denoise_form.addRow("σ space", self.denoise_ss)
        layout.addWidget(self.denoise_enable)
        layout.addLayout(denoise_form)

        self.denoise_enable.setChecked(cfg.denoise.enabled)
        self.denoise_enable.toggled.connect(self.sig_changed)
        for widget in (self.denoise_d, self.denoise_sc, self.denoise_ss):
            widget.valueChanged.connect(self.sig_changed)

    def values(self) -> dict:
        return {"denoise": self._denoise()}

    def _denoise(self):
        from talos.cv.preprocess import Denoise
        return Denoise(enabled=self.denoise_enable.isChecked(),
                       diameter=int(self.denoise_d.value()),
                       sigma_color=float(self.denoise_sc.value()),
                       sigma_space=float(self.denoise_ss.value()))

    def load(self, cfg: PreprocessConfig) -> None:
        self.denoise_enable.setChecked(cfg.denoise.enabled)
        self.denoise_d.setValue(cfg.denoise.diameter)
        self.denoise_sc.setValue(cfg.denoise.sigma_color)
        self.denoise_ss.setValue(cfg.denoise.sigma_space)


class IdentifyGroup(QGroupBox):
    """The whole chain, in canonical order — colour first, then the gates —
    with the counts that make it readable: ``Colour match 812 → Size 12 →
    Sharpness 2``.

    The colour editor LEADS it because that is the order the pipeline runs
    in: the source that produces the mask, then what cleans it, then the
    gates. The pinned picker above the scroll edits the same stage's colour;
    this one is what the config is built from.
    """

    sig_changed = Signal()
    sig_dropper = Signal()

    def __init__(self, settings, parent: QWidget | None = None):
        super().__init__("Identification", parent)
        self._settings = settings
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(4)
        self._config = IdentifyConfig.from_dict(settings.section("identify"))
        self._editors: list[StageEditor] = []
        for stage in self._config.stages:
            editor = StageEditor(stage, self, sliders=stage.NAME == "colour")
            editor.sig_changed.connect(self.sig_changed)
            editor.sig_dropper.connect(self.sig_dropper)
            self._editors.append(editor)
            layout.addWidget(editor)
        self.counts = QLabel("")
        self.counts.setObjectName("dim")
        self.counts.setWordWrap(True)
        layout.addWidget(self.counts)
        reset = QPushButton("Reset the chain")
        reset.setObjectName("compact")
        reset.clicked.connect(self.reset)
        layout.addWidget(reset)
        # Slack belongs at the bottom of the card. Without this the group
        # stretches the editors inside it, and a form that is given more
        # height than it needs spends it between two rows — a blank band that
        # reads as a parameter that went missing.
        layout.addStretch(1)

    # ------------------------------------------------------------------

    def colour_editor(self) -> StageEditor | None:
        """The colour stage's own editor — first in this card, because the
        source that produces the mask is first in the pipeline."""
        for editor in self._editors:
            if editor._stage.NAME == "colour":
                return editor
        return None

    def colour_stage(self):
        """The colour stage as the pipeline card describes it — THE source of
        truth for the match (the pinned picker mirrors its colour)."""
        editor = self.colour_editor()
        return editor.stage() if editor is not None else IdentifyConfig() \
            .stage("colour")

    def show_colour(self, hex_color: str) -> None:
        """Mirror a colour picked elsewhere, without reporting it back."""
        editor = self.colour_editor()
        if editor is not None:
            editor.show_hex(hex_color)

    def stages(self) -> list:
        """Every stage as the widgets describe them, in canonical order —
        colour first, which is also the order the pipeline runs them."""
        return [editor.stage() for editor in self._editors]

    def set_counts(self, summary: str) -> None:
        self.counts.setText(summary)

    def reload(self) -> None:
        """Re-read the stored chain (a settings change made elsewhere)."""
        self._config = IdentifyConfig.from_dict(
            self._settings.section("identify"))
        for editor, stage in zip(self._editors, self._config.stages):
            editor.load(stage)

    def reset(self) -> None:
        fresh = IdentifyConfig()
        for editor, stage in zip(self._editors, fresh.stages):
            # load() applies the values AND the rows that depend on a named
            # choice, so a reset cannot leave a label describing the old one.
            editor.load(stage)
        self.sig_changed.emit()


__all__ = ["ColourGroup", "CurvePlot", "IdentifyGroup", "PARAM_LABELS",
           "PreprocessGroup", "StageEditor", "param_label"]
