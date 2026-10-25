"""Filesystem locations used by BookRAG.

Everything the app writes lives under a single platform-appropriate data
directory so a project can be backed up or moved as one unit.  The location
can be overridden with the ``BOOKRAG_HOME`` environment variable, which is
what the test-suite uses to stay hermetic.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "BookRAG"

#: Root of the installed package (``src/bookrag``).
#: paths.py now lives in bookrag/data/, so .parent.parent reaches bookrag/.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

#: Repository root when running from a source checkout.
PROJECT_ROOT = PACKAGE_ROOT.parent.parent


def _data_root() -> Path:
    override = os.environ.get("BOOKRAG_HOME")
    if override:
        return Path(override).expanduser().resolve()

    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
    return base / APP_NAME


def data_dir() -> Path:
    """Root directory for application data (database, images, exports)."""
    return _data_root()


def settings_file() -> Path:
    """Location of the persisted settings file."""
    return data_dir() / "settings.json"


def db_dir() -> Path:
    """Directory holding the persistent ChromaDB store."""
    return data_dir() / "db"


def images_dir() -> Path:
    """Directory where images extracted from source documents are cached."""
    return data_dir() / "images"


def exports_dir() -> Path:
    """Directory for generated reports and exports."""
    return data_dir() / "exports"


def logs_dir() -> Path:
    """Directory for application logs.

    Precedence, highest first:

    1. ``BOOKRAG_LOG_DIR`` - an explicit override, for when logs belong
       somewhere the data directory cannot follow them to.
    2. ``BOOKRAG_HOME`` - hermetic mode.  The test-suite sets this, and it
       must not leave log files behind in a developer's checkout.
    3. a source checkout - the project's own ``Log`` folder, so that running
       from the repo keeps every artefact together in one place.
    4. an installed package - the platform data directory.
    """
    override = os.environ.get("BOOKRAG_LOG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if os.environ.get("BOOKRAG_HOME"):
        return data_dir() / "logs"
    checkout = _source_checkout_root()
    if checkout is not None:
        return checkout / "Log"
    return data_dir() / "logs"


def _source_checkout_root() -> Path | None:
    """The repository root when running from a checkout, else ``None``.

    A wheel install has neither ``pyproject.toml`` nor ``src/bookrag`` at that
    depth, which is exactly what distinguishes it from a source tree.
    """
    if (PROJECT_ROOT / "pyproject.toml").is_file() and (PROJECT_ROOT / "src" / "bookrag").is_dir():
        return PROJECT_ROOT
    return None


def model_cache_dir() -> Path:
    """Directory used to cache the downloaded HuggingFace model weights."""
    return data_dir() / "models"


def ensure_data_dirs() -> Path:
    """Create the data directory tree and return its root."""
    root = data_dir()
    for path in (root, db_dir(), images_dir(), exports_dir(), logs_dir()):
        path.mkdir(parents=True, exist_ok=True)
    return root
