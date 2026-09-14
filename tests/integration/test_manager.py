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

    def sim(self):  # attribute, not method — handled below
        return False


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_each_proxy_gets_its_own_config(qapp):
    """Regression: a closure bug made every device use the LAST config
    section (all-on-COM5 on real hardware)."""
    settings = StubSettings()
    settings.sim = False  # manager reads settings.sim? — reads its own flag
    manager = InstrumentManager(settings, sim=False)
    for key in DEVICE_KEYS:
        proxy = manager._proxies[key]
        driver = proxy._factory()  # builds the (unconnected) driver
        assert driver.port_name == _EXPECTED_PORTS[key], key
        assert driver.device_id == f"{key}@{_EXPECTED_PORTS[key]}"


def test_focus_gets_the_focus_proxy(qapp):
    from talos.hal.proxies import FocusProxy

    settings = StubSettings()
    settings.sim = False
    manager = InstrumentManager(settings, sim=False)
    assert isinstance(manager._proxies["focus"], FocusProxy)
    assert "autofocus" in manager._proxies["focus"]._special_methods


def test_manager_stop_all_budget_completes(qapp):
    """stop_all without proxies running must still emit within budget."""
    settings = StubSettings()
    settings.sim = False
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
