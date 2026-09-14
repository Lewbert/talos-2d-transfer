"""StageControlWindow: construction smoke (the hold buttons route
through the InputSystem's ui_hold/ui_release)."""

import pytest
from PySide6.QtWidgets import QApplication

from talos.ui.widgets.stage_control_window import (
    FocusHoldPanel,
    StageControlWindow,
)


class StubInput:
    def __init__(self):
        self.holds: list = []
        self.releases: list = []
        self.cancels: list = []
        self.escapes: int = 0

    def ui_hold(self, claim_key, direction, fast=False):
        self.holds.append((claim_key, direction, fast))

    def ui_release(self, claim_key):
        self.releases.append(claim_key)

    def ui_click(self, stage_id, axis, direction):
        pass

    def cancel_all_holds(self, reason="stop requested"):
        self.cancels.append(reason)

    def on_escape(self):
        self.escapes += 1


class StubManager:
    def set_enabled(self, key, on):
        pass

    def stop_all(self):
        pass

    def submit(self, *args):
        return 0


class FakeSettings:
    def device(self, key):
        return {"zolix": {"um_per_pulse_xy": 0.625, "um_per_pulse_r": 0.00125},
                "sigmakoki": {"um_per_step_xy": 0.5, "um_per_step_z": 0.25},
                "focus": {}}[key]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def window(qapp):
    return StageControlWindow(StubManager(), FakeSettings(), StubInput())


def test_window_has_both_stage_panels_and_focus_holds(window):
    assert set(window._panels) == {"zolix", "sigmakoki"}
    focus = window.findChild(type(window._panels["zolix"]))
    assert focus is not None


def test_focus_hold_routes_through_input_system(window):
    focus_panel = window.findChild(FocusHoldPanel)
    assert focus_panel is not None
    input_system = window._input
    # Simulate the hold buttons (sig_press / sig_release are on the
    # _HoldButton; drive the panel's handlers directly).
    focus_panel._hold(1)
    assert ("focus:z", 1, False) in input_system.holds
    focus_panel._hold(1, fast=True)
    assert ("focus:z", 1, True) in input_system.holds
    focus_panel._release()
    assert "focus:z" in input_system.releases


def test_focus_buttons_ordered_upfast_downsfast(window):
    """▲▲ / ▲ / ▼ / ▼▼ — arrow count = speed, listed up to down."""
    focus_panel = window.findChild(FocusHoldPanel)
    texts = [b.text() for b in focus_panel._buttons]
    assert texts == ["▲▲", "▲", "▼", "▼▼"]
    # the horizontal three-panel layout: zolix | sigmakoki | focus
    layout = window.layout()
    assert layout.count() == 3


def test_close_hides_and_never_destroys(window):
    window.show()
    window.close()
    assert not window.isVisible()
    window.show()  # still alive — the hide-not-destroy contract
    assert window.isVisible()


def test_hiding_releases_held_jogs(window):
    """A hidden window never delivers a release signal: without the
    cancel, the input system keeps the claim and the axis jogs on with no
    button under the pointer."""
    window.show()
    window._input.ui_hold("zolix:x", 1)
    window.close()
    assert window._input.cancels == ["stage control hidden"]


def test_escape_stops_all_before_hiding(window):
    """Esc must stay the global STOP ALL: this dialog hosts the jog
    buttons, so swallowing Esc into a mere hide left the operator without
    an emergency stop while their hand was on this window."""
    window.show()
    window.reject()
    assert window._input.escapes == 1
    assert not window.isVisible()


def test_zero_is_refused_while_a_job_owns_the_axes(qapp, monkeypatch):
    """ZERO submits a home opcode straight to the manager — it used to be
    reachable mid-scan and mid-autofocus."""
    from PySide6.QtWidgets import QMessageBox

    from talos.ui.widgets.stage_panel import ReferenceStagePanel

    class _State:
        mode = "SCAN"

    panel = ReferenceStagePanel("zolix", "ZOLIX", ["x", "y", "r"], StubInput(),
                                StubManager(), FakeSettings(), state=_State())
    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: asked.append(a) or QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    panel._on_zero()
    assert asked == [], "the confirmation must never be reached"


def test_zero_is_allowed_in_manual_mode(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from talos.ui.widgets.stage_panel import ReferenceStagePanel

    class _State:
        mode = "MANUAL"

    class _Manager(StubManager):
        def __init__(self):
            self.submitted: list = []

        def submit(self, *args, **kwargs):
            self.submitted.append(args)

    manager = _Manager()
    panel = ReferenceStagePanel("zolix", "ZOLIX", ["x", "y", "r"], StubInput(),
                                manager, FakeSettings(), state=_State())
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Ok)
    panel._on_zero()
    assert manager.submitted == [("zolix", "home")]
