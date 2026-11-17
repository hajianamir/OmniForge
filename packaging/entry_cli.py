"""PyInstaller entry point for the lightweight OmniForge CLI executable (OmniForge-CLI.exe)."""

from __future__ import annotations

import multiprocessing


def _bootstrap() -> None:
    """Initialize frozen-safe multiprocessing and invoke the Typer CLI."""
    multiprocessing.freeze_support()
    from omniforge.cli.main import main

    main()


if __name__ == "__main__":
    _bootstrap()
