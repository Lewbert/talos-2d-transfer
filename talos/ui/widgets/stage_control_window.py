"""StageControlWindow: the dialbox hosting ALL manual stage control
buttons, laid out like the precursor transfer-stage-control: three
panels side by side — Zolix XYR | SigmaKoki XYZ | Focus. The focus
panel is four stacked hold buttons with arrow count = speed:
▲▲ fast up, ▲ slow up, ▼ slow down, ▼▼ fast down.
Toggleable from the Windows menu; Esc/close HIDE it (FocusWindow
pattern). The main interface keeps status/enable only (HardwareStrip).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QSizePolicy,
    QVBoxLayout,
)

from talos.ui.widgets.stage_panel import ReferenceStagePanel, _HoldButton


class FocusHoldPanel(QGroupBox):
    """Manual focus hold buttons: hold = jog at the resolver's focus
    speed (min/max × objective multiplier — same path as the keyboard
    keys), release = stop. Arrow count = speed. Claim key ``focus:z``
    is the resolver's focus axis."""

    def __init__(self, input_system, settings, parent=None):
        super().__init__("FOCUS", parent)
        self._input = input_system
        self._settings = settings
        self._buttons: list[_HoldButton] = []
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        # (text, direction, fast) — listed from up to down, arrow count
        # indicates the speed
        for text, direction, fast in (("▲▲", 1, True), ("▲", 1, False),
                                      ("▼", -1, False), ("▼▼", -1, True)):
            btn = _HoldButton(text, threshold_ms=self._hold_threshold_ms)
            btn.setToolTip("Hold to jog focus"
                           + (" (fast)" if fast else " (slow)"))
            btn.sig_press.connect(lambda d=direction, f=fast:
                                  self._hold(d, f))
            btn.sig_release.connect(self._release)
            self._buttons.append(btn)
            # stretch + Expanding: QPushButton's vertical policy is
            # Fixed, which blocks stretch growth — the four buttons
            # share the column height (big hold targets, no dead space)
            btn.setSizePolicy(QSizePolicy.Policy.Preferred,
                              QSizePolicy.Policy.Expanding)
            layout.addWidget(btn, stretch=1)

    def _hold_threshold_ms(self) -> int:
        """The resolver's tap-vs-hold threshold — same number the keys and
        the D-pad use (see _HoldButton)."""
        return int(self._settings.section("input").get(
            "long_press_threshold_ms", 300) or 300)

    def _hold(self, direction: int, fast: bool = False) -> None:
        if self._input is not None:
            self._input.ui_hold("focus:z", direction, fast)

    def _release(self) -> None:
        if self._input is not None:
            self._input.ui_release("focus:z")


class StageControlWindow(QDialog):
    def __init__(self, manager, settings, input_system=None, state=None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stage Control")
        self._toggle_action = None
        self._input = input_system
        self._panels: dict[str, ReferenceStagePanel] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Fixed panel widths + no stretch factors: the window hugs its
        # content (sizeHint), and the top-aligned panels never balloon
        # vertically into dead whitespace. 320 px gives the jog grid
        # ~145 px columns → 85 px buttons ≈ the 2.5:1 reference ratio.
        for key, title, axes in (("zolix", "ZOLIX XYR STAGE", ["x", "y", "r"]),
                                 ("sigmakoki", "SIGMAKOKI XYZ STAGE", ["x", "y", "z"])):
            panel = ReferenceStagePanel(key, title, axes, input_system,
                                        manager, settings, state=state,
                                        parent=self)
            panel.setFixedWidth(320)
            self._panels[key] = panel
            layout.addWidget(panel)
            layout.setAlignment(panel, Qt.AlignmentFlag.AlignTop)
        if input_system is not None:
            focus = FocusHoldPanel(input_system, settings, self)
            focus.setFixedWidth(120)  # the narrow third column
            # expand vertically: the hold buttons share the extra height
            # (stretch=1 each) so the column matches its stage siblings
            # and no dead space opens below the panel
            focus.setSizePolicy(QSizePolicy.Policy.Preferred,
                                QSizePolicy.Policy.Expanding)
            layout.addWidget(focus)
        self.resize(self.layout().sizeHint())

    def set_toggle_action(self, action) -> None:
        self._toggle_action = action

    def update_telem(self, device_key: str, payload: dict) -> None:
        panel = self._panels.get(device_key)
        if panel is not None:
            panel.update_telem(payload)

    def _hide_and_uncheck(self) -> None:
        # Hiding a window that holds a jog button must release the claim:
        # the release signal never arrives for a hidden widget, and the
        # axis would keep jogging with no button under the pointer.
        if self._input is not None:
            self._input.cancel_all_holds("stage control hidden")
        self.hide()
        if self._toggle_action is not None:
            self._toggle_action.setChecked(False)

    def reject(self) -> None:  # Esc
        # Esc is the global STOP ALL. This dialog hosts the jog buttons —
        # swallowing Esc into a mere hide left the operator with no
        # emergency stop while their hand was on this window.
        if self._input is not None:
            self._input.on_escape()
        self._hide_and_uncheck()

    def closeEvent(self, event) -> None:  # noqa: N802
        event.ignore()  # never destroyed — toggled back via the menu
        self._hide_and_uncheck()
