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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from talos.cv.identify import IdentifyConfig, Stage, valid_hex
from talos.cv.preprocess import (DN_MAX, LocalContrast, PreprocessConfig,
                                 channel_curve, effective_gain,
                                 effective_width)
from talos.ui import theme

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
    "tolerance": "Tolerance",
    "kernel": "Kernel",
}


def param_label(name: str) -> str:
    return PARAM_LABELS.get(name, name.replace("_", " ").capitalize())


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
                 show_enable: bool = True, framed: bool = True):
        super().__init__(parent)
        self._stage = stage
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
        for field_info in fields(stage):
            name = field_info.name
            if name == "enabled":
                continue
            self._add_field(form, stage, name, getattr(stage, name))

        self.enable.toggled.connect(self._on_changed)

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
            form.addRow("Colour", row)
            edit.editingFinished.connect(self._on_changed)
            dropper.clicked.connect(self.sig_dropper)
            dialog.clicked.connect(self._pick_from_dialog)
            self._refresh_swatch()
        elif isinstance(value, bool):
            check = QCheckBox()
            check.setChecked(bool(value))
            self._editors[name] = ("bool", check)
            form.addRow(param_label(name), check)
            check.toggled.connect(self._on_changed)
        else:
            limits = getattr(stage, "RANGES", {}).get(name, (0, 255, 1))
            box = _spin(value, limits)
            self._editors[name] = ("num", box)
            form.addRow(param_label(name), box)
            box.valueChanged.connect(self._on_changed)

    def _refresh_swatch(self) -> None:
        entry = self._editors.get("hex_color")
        if entry is None:
            return
        colour = QColor(valid_hex(entry[1].text(),
                                  getattr(self._stage, "hex_color",
                                          "#c8a2c8")))
        entry[2].setStyleSheet(
            f"background: {colour.name()}; border: 1px solid #3f3f46;")

    def _on_changed(self, *_args) -> None:
        self._refresh_swatch()
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
    """The sample colour: the colour stage, promoted to quick access.

    One colour drives two things — the mask's target and the curve's
    centre — so it is edited once, here, and both read it.
    """

    sig_changed = Signal()
    sig_dropper = Signal()

    def __init__(self, settings, parent: QWidget | None = None):
        super().__init__("Sample colour", parent)
        self._settings = settings
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(4)

        self.editor = StageEditor(self._colour_stage(), self,
                                  show_enable=True, framed=False)
        self.editor.sig_changed.connect(self._on_changed)
        self.editor.sig_dropper.connect(self.sig_dropper)
        layout.addWidget(self.editor)

        hint = QLabel("Also centres the local-contrast curve.")
        hint.setObjectName("dim")
        hint.setWordWrap(True)
        layout.addWidget(hint)

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

    def reload(self) -> None:
        """Re-read the stored colour (a settings change made elsewhere)."""
        self.editor._stage = self._colour_stage()
        self.editor.enable.setChecked(bool(self.editor._stage.enabled))
        for name, spec in self.editor._editors.items():
            value = getattr(self.editor._stage, name, None)
            if spec[0] == "hex":
                spec[1].setText(str(value))
            elif spec[0] == "bool":
                spec[1].setChecked(bool(value))
            elif value is not None:
                spec[1].setValue(value)

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

        self.tone = _ToneRows(cfg, self)
        self.tone.sig_changed.connect(self._changed)
        layout.addWidget(self.tone)

        order = QLabel("Order: shade → denoise → tone → curve. The curve is "
                       "last so the pick stays where it was picked.")
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
        cfg = PreprocessConfig(
            enabled=self.enable.isChecked(),
            **self.tone.values(), **self.spatial.values())
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
            self.tone.load(cfg)
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
            self.tone.load(fresh)
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
    """Shade correction and denoise — the stages that run before the tone
    operations, and the expensive ones."""

    sig_changed = Signal()

    def __init__(self, cfg: PreprocessConfig, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.shade_enable = QCheckBox("Shade correction")
        self.shade_enable.setToolTip(
            "Divides out a blurred copy of the frame. The curve amplifies\n"
            "illumination unevenness exactly as eagerly as a flake, so this\n"
            "removes it first.")
        self.shade_sigma = _spin(cfg.shade.sigma, (4.0, 200.0, 4.0), " px")
        self.shade_strength = _spin(cfg.shade.strength, (0.0, 1.0, 0.05))
        shade_form = QFormLayout()
        shade_form.setContentsMargins(14, 0, 0, 0)
        shade_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        shade_form.setVerticalSpacing(4)
        shade_form.addRow("σ", self.shade_sigma)
        shade_form.addRow("Strength", self.shade_strength)
        layout.addWidget(self.shade_enable)
        layout.addLayout(shade_form)

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

        self.shade_enable.setChecked(cfg.shade.enabled)
        self.denoise_enable.setChecked(cfg.denoise.enabled)
        for widget in (self.shade_enable, self.denoise_enable):
            widget.toggled.connect(self.sig_changed)
        for widget in (self.shade_sigma, self.shade_strength,
                       self.denoise_d, self.denoise_sc, self.denoise_ss):
            widget.valueChanged.connect(self.sig_changed)

    def values(self) -> dict:
        return {"shade": self._shade(), "denoise": self._denoise()}

    def _shade(self):
        from talos.cv.preprocess import Shade
        return Shade(enabled=self.shade_enable.isChecked(),
                     sigma=float(self.shade_sigma.value()),
                     strength=float(self.shade_strength.value()))

    def _denoise(self):
        from talos.cv.preprocess import Denoise
        return Denoise(enabled=self.denoise_enable.isChecked(),
                       diameter=int(self.denoise_d.value()),
                       sigma_color=float(self.denoise_sc.value()),
                       sigma_space=float(self.denoise_ss.value()))

    def load(self, cfg: PreprocessConfig) -> None:
        self.shade_enable.setChecked(cfg.shade.enabled)
        self.shade_sigma.setValue(cfg.shade.sigma)
        self.shade_strength.setValue(cfg.shade.strength)
        self.denoise_enable.setChecked(cfg.denoise.enabled)
        self.denoise_d.setValue(cfg.denoise.diameter)
        self.denoise_sc.setValue(cfg.denoise.sigma_color)
        self.denoise_ss.setValue(cfg.denoise.sigma_space)


class _ToneRows(QFrame):
    """The plain point operations. They are all one LUT with the curve, so
    they cost nothing to leave switched on but the identity."""

    sig_changed = Signal()

    def __init__(self, cfg: PreprocessConfig, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        layout.setVerticalSpacing(4)
        self.exposure = _spin(cfg.exposure, (0.25, 4.0, 0.05), " ×")
        self.brightness = _spin(cfg.brightness, (-64.0, 64.0, 1.0))
        self.contrast = _spin(cfg.contrast, (0.25, 3.0, 0.05))
        self.gamma = _spin(cfg.gamma, (0.2, 3.0, 0.05))
        for label, widget in (("Exposure", self.exposure),
                              ("Brightness", self.brightness),
                              ("Contrast", self.contrast),
                              ("Gamma", self.gamma)):
            layout.addRow(label, widget)
            widget.valueChanged.connect(self.sig_changed)

    def values(self) -> dict:
        return {"exposure": float(self.exposure.value()),
                "brightness": float(self.brightness.value()),
                "contrast": float(self.contrast.value()),
                "gamma": float(self.gamma.value())}

    def load(self, cfg: PreprocessConfig) -> None:
        self.exposure.setValue(cfg.exposure)
        self.brightness.setValue(cfg.brightness)
        self.contrast.setValue(cfg.contrast)
        self.gamma.setValue(cfg.gamma)


class IdentifyGroup(QGroupBox):
    """The gates, in canonical order, with the counts that make the chain
    readable — ``Colour match 812 → Size 12 → Sharpness 2``."""

    sig_changed = Signal()

    def __init__(self, settings, parent: QWidget | None = None):
        super().__init__("Identification", parent)
        self._settings = settings
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(4)
        self._config = IdentifyConfig.from_dict(settings.section("identify"))
        self._editors: list[StageEditor] = []
        for stage in self._config.stages:
            if stage.NAME == "colour":
                continue            # promoted to the quick-access group
            editor = StageEditor(stage, self)
            editor.sig_changed.connect(self.sig_changed)
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

    def stages(self) -> list:
        """The gates as the widgets describe them, in canonical order."""
        return [editor.stage() for editor in self._editors]

    def set_counts(self, summary: str) -> None:
        self.counts.setText(summary)

    def reset(self) -> None:
        fresh = IdentifyConfig()
        for editor, stage in zip(self._editors, fresh.stages[1:]):
            editor._stage = stage
            editor.enable.setChecked(stage.enabled)
            for name, spec in editor._editors.items():
                value = getattr(stage, name, None)
                if spec[0] == "hex":
                    continue
                if spec[0] == "bool":
                    spec[1].setChecked(bool(value))
                elif value is not None:
                    spec[1].setValue(value)
        self.sig_changed.emit()


__all__ = ["ColourGroup", "CurvePlot", "IdentifyGroup", "PARAM_LABELS",
           "PreprocessGroup", "StageEditor", "param_label"]
