"""Logging configuration for the application."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ..data import paths

_CONFIGURED = False
_FORMAT = "%(asctime)s %(levelname)-7s %(name)-24s %(message)s"
_DATE_FORMAT = "%H:%M:%S"

#: Separate file for monitoring lines: only MONITOR records, key=value, no prose.
_MONITOR_FORMAT = "%(asctime)s %(message)s"
_MONITOR_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Keep the monitoring log bounded - it is appended to on every run.
_MONITOR_MAX_BYTES = 2 * 1024 * 1024
_MONITOR_BACKUPS = 3


def setup_logging(verbose: bool = False, log_file: Path | None = None) -> logging.Logger:
    """Configure root logging once and return the package logger.

    ``verbose`` switches the console handler to ``DEBUG``; otherwise only
    ``INFO`` and above reach the terminal.  Warnings and errors are also
    mirrored to a rotating log file inside the data directory.

    Monitoring records additionally go to their own ``monitoring.log`` so that
    stage timings stay machine-readable and separate from the prose log.
    """
    global _CONFIGURED

    root = logging.getLogger("bookrag")
    if _CONFIGURED:
        root.setLevel(logging.DEBUG if verbose else logging.INFO)
        return root

    level = logging.DEBUG if verbose else logging.INFO
    root.setLevel(logging.DEBUG)
    root.propagate = False

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console)

    try:
        paths.logs_dir().mkdir(parents=True, exist_ok=True)
        target = log_file or (paths.logs_dir() / "bookrag.log")
        file_handler = logging.FileHandler(target, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(file_handler)
    except OSError:
        # A read-only data directory must never stop the app from starting.
        pass

    _configure_monitor(log_file)
    _CONFIGURED = True
    return root


def _configure_monitor(log_file: Path | None) -> None:
    """Give the monitor its own rotating file, isolated from the prose log."""
    monitor = logging.getLogger("bookrag.monitor")
    monitor.setLevel(logging.DEBUG)
    # Stop monitor records reaching the console or bookrag.log - the point of
    # this handler is one parseable stream on its own.
    monitor.propagate = False

    try:
        base = log_file.parent if log_file else paths.logs_dir()
        base.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = RotatingFileHandler(
            base / "monitoring.log",
            maxBytes=_MONITOR_MAX_BYTES,
            backupCount=_MONITOR_BACKUPS,
            encoding="utf-8",
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(_MONITOR_FORMAT, datefmt=_MONITOR_DATE_FORMAT))
        monitor.addHandler(handler)
    except OSError:
        pass


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``bookrag`` namespace."""
    if name.startswith("bookrag"):
        return logging.getLogger(name)
    return logging.getLogger(f"bookrag.{name}")
