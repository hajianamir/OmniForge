"""OmniForge Enterprise Workbench Server (FastAPI + Compiled React/Vite TypeScript SPA)."""

from __future__ import annotations

import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

from omniforge.core.logging import get_logger

logger = get_logger(__name__)

STUDIOS: list[tuple[str, str, str]] = [
    ("01", "Data Ingestion, Quality & Cleaning", "database"),
    ("02", "Clustering & Anomaly Screening", "scatter"),
    ("03", "Universal Diagnostics, Baselines & Risk", "activity"),
    ("04", "Automated EDA & Hypothesis Lab", "bar-chart"),
    ("05", "Feature Engineering & Selection", "sliders"),
    ("06", "AutoML, Deep Learning & Predictions", "cpu"),
    ("07", "Responsible AI", "shield"),
    ("08", "Forecasting, Causal, Opt & Simulation", "trending-up"),
    ("09", "Specialized AI", "sparkles"),
    ("10", "Visual DAG Pipeline Builder", "git-branch"),
    ("11", "MLOps", "server"),
    ("12", "Executive Reporting & Export Center", "file-text"),
    ("13", "AI Analytics Copilot", "message-square"),
    ("14", "Live Prediction & Supervisory Control-Room Monitor", "gauge"),
]


def create_workbench_app() -> Any:
    """Create and return the unified FastAPI + React SPA Workbench ASGI application."""
    from omniforge.api.main import create_app

    return create_app()


def get_static_app_dir() -> Path:
    """Return the path to the pre-compiled React/Vite SPA static bundle."""
    return Path(__file__).resolve().parent / "static_app"


def run_workbench_server(
    host: str = "127.0.0.1",
    port: int = 8501,
    debug: bool = False,
    open_browser: bool = False,
) -> None:
    """Launch the OmniForge FastAPI + React TypeScript Enterprise Workbench via Uvicorn."""
    import uvicorn

    from omniforge.api.auth import is_loopback_host, set_loopback_workbench_bootstrap

    prev_bootstrap = set_loopback_workbench_bootstrap(is_loopback_host(host))
    try:
        app = create_workbench_app()
        display_host = "127.0.0.1" if host == "0.0.0.0" else host
        url = f"http://{display_host}:{port}"

        if open_browser:

            def _opener() -> None:
                time.sleep(1.0)
                try:
                    if not webbrowser.open(url):
                        logger.warning(
                            "workbench_browser_open_failed",
                            reason="browser_open_returned_false",
                            fallback="server_only",
                        )
                except Exception as exc:
                    # Exception messages can contain URLs or local paths; avoid logging them.
                    logger.warning(
                        "workbench_browser_open_failed",
                        reason="browser_open_exception",
                        error_type=type(exc).__name__,
                        fallback="server_only",
                    )

            threading.Thread(target=_opener, daemon=True, name="omniforge-browser").start()

        uvicorn.run(
            app,
            host=host,
            port=int(port),
            log_level="debug" if debug else "info",
        )
    finally:
        set_loopback_workbench_bootstrap(prev_bootstrap)


# Backwards-compatible aliases preserved for external scripts and desktop launchers
create_dash_app = create_workbench_app
run_dash_server = run_workbench_server


if __name__ == "__main__":
    run_workbench_server(host="127.0.0.1", port=8501, debug=False, open_browser=False)
