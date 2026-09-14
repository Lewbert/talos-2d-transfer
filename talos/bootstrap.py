"""Frozen-safe bootstrap: logging, QApplication, theme, Qt message bridge.

GenTL producer discovery for the camera lands here in M4 (before
``import harvesters``).
"""

from __future__ import annotations

import logging
import sys

import talos
from talos.logging_setup import (
    install_qt_bridge,
    install_stderr_tee,
    setup_logging,
)

logger = logging.getLogger(__name__)


def _install_excepthook() -> None:
    """Uncaught exceptions must reach the FILE log — after a console
    detach (or with the debug console disabled) stderr may be devnull
    or a freed handle, and PySide6 slot crashes die without a trace."""

    def hook(exc_type, exc, tb) -> None:
        logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = hook


def bootstrap(argv: list[str] | None = None) -> "object":
    from PySide6.QtWidgets import QApplication

    log_path = setup_logging()
    _install_excepthook()
    # mirror raw stderr (slot/worker-thread tracebacks) into stderr.log —
    # the debug console window swallows them otherwise
    install_stderr_tee(log_path.parent)
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("TALOS")
    app.setApplicationVersion(talos.__version__)

    from talos.ui.theme import apply_theme

    apply_theme(app)
    install_qt_bridge()
    return app
