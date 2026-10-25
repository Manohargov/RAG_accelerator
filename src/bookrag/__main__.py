"""Command line entry point: ``python -m bookrag``."""

from __future__ import annotations

import sys


def main() -> int:
    from .Rag_app.cli import main as run_cli

    return run_cli()


if __name__ == "__main__":
    sys.exit(main())
