"""InstrumentManager integration: proxy wiring, config isolation, shutdown."""

import pytest
from PySide6.QtWidgets import QApplication

from talos.hal.registry import DEVICE_KEYS
from talos.instruments import InstrumentManager

_EXPECTED_PORTS = {"zolix": "COM3", "sigmakoki": "COM6",
                   "focus": "COM10", "yudian": "COM5"}


class StubSettings:
    def device(self, key):
        if key == "camera":
            return {"backend": "manual", "manual_folder": ".", "fps": 1.0}
        return {"port": _EXPECTED_PORTS[key], "enabled": True}

    def section(self, key):
        return {}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_each_proxy_gets_its_own_config(qapp):
    """Regression: a closure bug made every device use the LAST config
    section (all-on-COM5 on real hardware)."""
    settings = StubSettings()
    manager = InstrumentManager(settings, sim=False)
    for key in DEVICE_KEYS:
        proxy = manager._proxies[key]
        driver = proxy._factory()  # builds the (unconnected) driver
        assert driver.port_name == _EXPECTED_PORTS[key], key
        assert driver.device_id == f"{key}@{_EXPECTED_PORTS[key]}"


def test_focus_gets_the_focus_proxy(qapp):
    from talos.hal.proxies import FocusProxy

    settings = StubSettings()
    manager = InstrumentManager(settings, sim=False)
    assert isinstance(manager._proxies["focus"], FocusProxy)
    assert "autofocus" in manager._proxies["focus"]._special_methods


def test_manager_stop_all_budget_completes(qapp):
    """stop_all without proxies running must still emit within budget."""
    settings = StubSettings()
    manager = InstrumentManager(settings, sim=True)
    manager.stop_all()
    # In sim mode the proxies are not started; the budget timer fires.
    from PySide6.QtCore import QEventLoop, QTimer

    done = []
    manager.sig_stop_all_done.connect(lambda: done.append(True))
    loop = QEventLoop()
    QTimer.singleShot(2500, loop.quit)
    loop.exec()
    assert done


def test_stop_acks_report_only_a_real_stop_all(qapp):
    """Regression (2026-09-16): every driver-level "stop" acks here —
    including each jog release (sigmakoki's stop, the focus trigger release) —
    so the counter reached its threshold without any STOP ALL and then
    re-fired "All stages stopped" on every later ack."""
    from talos.hal.registry import MOTION_KEYS

    settings = StubSettings()
    manager = InstrumentManager(settings, sim=True)
    reports = []
    manager.sig_stop_all_done.connect(lambda: reports.append(True))

    # three unrelated jog releases (one per motion device) — no round open
    for key in MOTION_KEYS:
        manager._proxies[key].sig_all_stopped.emit()
    assert reports == [], "no STOP ALL was requested"

    # a real STOP ALL reports ONCE, when every ack is in
    manager.stop_all()
    for key in MOTION_KEYS:
        manager._proxies[key].sig_all_stopped.emit()
    assert len(reports) == 1

    # ...and a later release-stop must not report again
    manager._proxies["focus"].sig_all_stopped.emit()
    assert len(reports) == 1
    manager._stop_budget_timer.stop()
