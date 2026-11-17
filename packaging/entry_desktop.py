"""PyInstaller entry point for the full OmniForge desktop executable (OmniForge.exe)."""

from __future__ import annotations

import multiprocessing
import sys


def _bootstrap() -> int:
    """Initialize frozen-safe multiprocessing and launch the unified OmniForge desktop runtime."""
    multiprocessing.freeze_support()
    from omniforge.launcher import main

    return main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(_bootstrap())
