"""Application assembly: AppState, InstrumentManager, MainWindow wiring."""

from __future__ import annotations

import logging
import os

from PySide6.QtCore import QObject, Signal

from talos.config import Settings
from talos.cv.autofocus_service import AutofocusService
from talos.cv.frame_slot import LatestFrameSlot
from talos.hal.proxies.focus_proxy import FocusProxy
from talos.instruments import InstrumentManager
from talos.models import StagePosition
from talos.ui.main_window import MainWindow

logger = logging.getLogger(__name__)


class AppState(QObject):
    """Single source of truth for UI state (positions/temps live in manager
    telemetry; this holds session-level selections).

    Rule: only InstrumentManager-driven signals and explicit user actions
    (objective selector) write AppState; widgets only read it.
    """

    sig_objective_changed = Signal(int)   # nosepiece position 0..4
    sig_mode_changed = Signal(str)
    sig_stage_origin_changed = Signal(object)   # StagePosition
    sig_focus_origin_changed = Signal(int)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._objective = 0
        self._mode = "MANUAL"
        self._stage_origin: StagePosition | None = None
        self._focus_origin: int | None = None

    @property
    def objective(self) -> int:
        return self._objective

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def stage_origin(self) -> StagePosition | None:
        """The software XYR origin (set from the quick actions; consumed
        by the future scanning function)."""
        return self._stage_origin

    @property
    def focus_origin(self) -> int | None:
        return self._focus_origin

    def set_stage_origin(self, position: StagePosition) -> None:
        self._stage_origin = position
        self.sig_stage_origin_changed.emit(position)

    def set_focus_origin(self, steps: int) -> None:
        self._focus_origin = int(steps)
        self.sig_focus_origin_changed.emit(int(steps))

    def set_objective(self, objective: int) -> None:
        if objective != self._objective:
            self._objective = objective
            self.sig_objective_changed.emit(objective)
            logger.info("Objective (manual): nosepiece position %d", objective)

    def set_mode(self, mode: str) -> None:
        if mode != self._mode:
            self._mode = mode
            self.sig_mode_changed.emit(mode)
            logger.info("Mode: %s", mode)


class TALOSApplication:
    """Top-level object graph: Settings + AppState + InstrumentManager + UI."""

    def __init__(self, qapp, sim: bool = False):
        self.qapp = qapp
        self.sim = sim
        # The app icon (windows + taskbar): picked up automatically once
        # an icon file lands in resources/icons/ (talos.ico / talos.png).
        from PySide6.QtGui import QIcon

        from talos.paths import app_icon_path
        icon_path = app_icon_path()
        if icon_path is not None:
            qapp.setWindowIcon(QIcon(str(icon_path)))
        self.settings = Settings.load()
        # One-shot Labscope unwire: materialize the Zeiss table values as
        # proper calibration entries and backfill the settings px_um
        # columns (idempotent; never raises) — BEFORE any DB read below.
        from talos.migration import materialize_labscope_calibration
        materialize_labscope_calibration(self.settings)
        if sim:
            self.settings.update("device_mode", "sim")
        from talos import debug_console
        from talos.ui import theme

        # The Blender-style debug console (separate system window) and
        # verbose logging default ON while core features are in-dev.
        debug_console.setup_from_settings(self.settings)
        try:
            debug_cfg = self.settings.section("debug")
        except Exception:  # noqa: BLE001
            debug_cfg = {}
        verbose = bool(debug_cfg.get("verbose_logging", True))
        level = logging.DEBUG if verbose else logging.INFO
        logging.getLogger().setLevel(level)
        for handler in logging.getLogger().handlers:
            handler.setLevel(level)
        try:
            ui_cfg = self.settings.section("ui")
            font_size = int(ui_cfg.get("font_size", 12))
            accent = str(ui_cfg.get("accent", "#00BCBC"))
        except Exception:  # noqa: BLE001
            font_size, accent = 12, "#00BCBC"
        theme.set_accent(qapp, accent)
        theme.set_font_size(qapp, font_size)
        self.state = AppState()
        self.manager = InstrumentManager(self.settings, sim=sim)
        # Shared frame mailbox: the camera worker publishes every fetched
        # frame (with capture timestamps); the focus worker's autofocus
        # jobs and the AF-C monitor read it.
        self.frame_slot = LatestFrameSlot()
        self.autofocus = AutofocusService(self.manager, self.settings,
                                          self.state, self.frame_slot)
        self.input = None
        try:
            from talos.input import InputSystem

            self.input = InputSystem(self.manager, self.settings,
                                     state=self.state)
        except Exception as exc:  # noqa: BLE001 - input layer is optional
            logging.getLogger(__name__).warning("Input system unavailable: %s", exc)
        self.window = MainWindow(self.manager, self.settings, self.state,
                                 input_system=self.input,
                                 autofocus_service=self.autofocus)

    def run(self) -> int:
        from PySide6.QtCore import QTimer

        self.window.show()
        self.manager.connect_all()
        # Wire the shared frame slot into the proxies AFTER connect_all so
        # both workers exist; the camera worker publishes from its first
        # streamed frame on.
        self.manager.camera.set_frame_slot(self.frame_slot)
        focus = self.manager.device("focus")
        if isinstance(focus, FocusProxy):
            focus.set_frame_slot(self.frame_slot)
        if self.input is not None:
            self.input.start()
        autoquit_ms = os.environ.get("TALOS_AUTOQUIT_MS")
        if autoquit_ms:
            # Headless smoke-test hook: quit after N ms.
            QTimer.singleShot(int(autoquit_ms), self.qapp.quit)
        return self.qapp.exec()

    def shutdown(self) -> None:
        self.autofocus.shutdown()
        self.manager.shutdown()
        logger.info("TALOS exited cleanly")
        # Close the debug-console tailer on BOTH exit paths — otherwise
        # its console window lingers after the app quits.
        from talos import debug_console
        debug_console.disable()
        if getattr(self.manager, "shutdown_ragged", False):
            # A device worker is still inside a blocking call; walking
            # through normal interpreter teardown would destroy a live
            # QThread, which aborts Qt (CRITICAL + exit code 1). Exit
            # at the OS level instead — the file log is already flushed.
            logger.warning("Ragged shutdown: exiting past Qt teardown "
                           "(a device thread is still inside a call)")
            os._exit(0)
