"""Blender-style separate debug console (Windows): a CHILD PROCESS tails
the log file in its own console window (CREATE_NEW_CONSOLE).

The app itself never owns that console — closing the window's X only
kills the tailer, so no CTRL_CLOSE_EVENT can ever terminate TALOS (the
previous in-process AllocConsole design had exactly that failure mode:
an X-click silently killed the app with exit 1).

``disable()`` kills the tailer (its console closes with it); ``enable()``
spawns a fresh one. Toggle from Preferences → General; honored at
startup via settings ``debug.console_enabled`` (default true while core
features are still in development — flip to false at release).
"""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)

_tailer: subprocess.Popen | None = None


def enable() -> bool:
    """Spawn the log-tailer console window. Never raises; True when a
    tailer is running afterwards."""
    global _tailer
    if _tailer is not None and _tailer.poll() is None:
        return True
    try:
        from talos.logging_setup import current_log_path

        # -Wait tails like `tail -f`; -Tail seeds the window with the
        # recent history. The window title is set for recognizability.
        cmd = [
            "powershell", "-NoExit", "-Command",
            "$Host.UI.RawUI.WindowTitle='TALOS Debug Console'; "
            f"Get-Content '{current_log_path()}' -Wait -Tail 100",
        ]
        _tailer = subprocess.Popen(
            cmd,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        logger.info("Debug console attached (log tailer)")
        return True
    except Exception as exc:  # noqa: BLE001 - startup must never crash on this
        logger.warning("Debug console enable failed: %s", exc)
        return False


def disable() -> None:
    """Kill the tailer — its console window closes with it."""
    global _tailer
    if _tailer is None:
        return
    try:
        _tailer.terminate()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Debug console disable failed: %s", exc)
    finally:
        _tailer = None


def is_enabled() -> bool:
    """True while the tailer is alive."""
    return _tailer is not None and _tailer.poll() is None


def setup_from_settings(settings) -> None:
    """Honor settings.debug.console_enabled at startup."""
    try:
        enabled = bool(settings.section("debug").get("console_enabled", True))
    except Exception:  # noqa: BLE001 - section may not exist yet
        enabled = True
    if enabled:
        enable()
