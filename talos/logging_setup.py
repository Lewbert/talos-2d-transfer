"""Rotating file logging + optional Qt message bridge + stderr tee."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from talos.paths import get_log_dir

_tee_file = None
_log_path: Path | None = None


def current_log_path() -> Path:
    """The active talos.log path (for the debug-console log tailer)."""
    return _log_path


def setup_logging(log_dir: Path | None = None, verbose: bool = False) -> Path:
    global _log_path
    log_dir = log_dir or get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "talos.log"
    _log_path = log_path

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=5,
                                       encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.addHandler(console)
    return log_path


class _TeeStderr:
    """Stderr writes are mirrored to a file — PySide6 prints slot
    exceptions and threading prints worker-thread crashes directly to
    stderr, which lives on the debug console window and vanishes with
    the process. The tee leaves those traces on disk."""

    def __init__(self, stream, file):
        self._stream = stream
        self._file = file

    def write(self, s) -> None:
        for target in (self._stream, self._file):
            try:
                target.write(s)
            except Exception:  # noqa: BLE001 - a dead console fd must not kill
                pass
        try:
            self._file.flush()
        except Exception:  # noqa: BLE001
            pass

    def flush(self) -> None:
        try:
            self._stream.flush()
        except Exception:  # noqa: BLE001
            pass

    def isatty(self) -> bool:
        return False


def install_stderr_tee(log_dir: Path) -> None:
    """Mirror all future stderr writes into <log_dir>/stderr.log."""
    import sys

    global _tee_file
    _tee_file = open(log_dir / "stderr.log", "a", encoding="utf-8",  # noqa: SIM115
                     errors="replace")
    sys.stderr = _TeeStderr(sys.stderr, _tee_file)


def retee_stderr(stream) -> None:
    """Re-wrap a REPLACED sys.stderr (the debug console swaps streams)
    so the mirror keeps working."""
    import sys

    if _tee_file is not None:
        sys.stderr = _TeeStderr(stream, _tee_file)


def install_qt_bridge():
    """Route Qt messages into the Python logging system.

    Must be called after QApplication exists. Guarded import so headless
    tools can use setup_logging without PySide6.
    """
    from PySide6.QtCore import QMessageLogContext, QtMsgType, qInstallMessageHandler

    type_map = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }

    def handler(msg_type: QtMsgType, context: QMessageLogContext, message: str) -> None:
        logging.getLogger("qt").log(type_map.get(msg_type, logging.DEBUG), "%s", message)

    qInstallMessageHandler(handler)
