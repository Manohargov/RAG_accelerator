"""Exception hierarchy shared across the application."""

from __future__ import annotations


class BookRagError(Exception):
    """Base class for every error raised by BookRAG."""


class ConfigError(BookRagError):
    """Raised when settings are missing or contradictory."""


class DependencyError(BookRagError):
    """A required third-party package is not installed.

    Carries an ``install_hint`` so the CLI can tell the user exactly what to
    run instead of dumping a raw ``ImportError`` traceback.
    """

    def __init__(self, message: str, install_hint: str = "") -> None:
        super().__init__(message)
        self.install_hint = install_hint


class ModelUnavailableError(DependencyError):
    """The embedding model could not be loaded (missing weights, no network)."""


class StoreError(BookRagError):
    """The vector store rejected an operation.

    Carries an ``install_hint`` in the same shape as :class:`DependencyError`,
    because the failures that reach this level - a collection built by a
    different checkpoint, an index that needs rebuilding before it can be
    read - each have one specific command that resolves them.  A message that
    only describes the wall leaves the user stuck at it.
    """

    def __init__(self, message: str, install_hint: str = "") -> None:
        super().__init__(message)
        self.install_hint = install_hint


class IngestError(BookRagError):
    """A source document could not be parsed or ingested."""


class QueryError(BookRagError):
    """A query could not be executed."""
