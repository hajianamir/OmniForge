"""OmniForge Enterprise Workbench UI (FastAPI + Pre-Compiled React/Vite TypeScript SPA)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omniforge.ui.app import STUDIOS, create_workbench_app, get_static_app_dir, run_workbench_server

__all__ = [
    "STUDIOS",
    "create_workbench_app",
    "get_static_app_dir",
    "run_workbench_server",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from omniforge.ui import app as _ui_app

        return getattr(_ui_app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
