"""Unified Desktop, Server, UI, CLI, and Frozen-Executable Launcher for OmniForge.

Designed to work seamlessly both from a standard Python installation and from a
PyInstaller-frozen Windows standalone executable (``OmniForge.exe`` / ``OmniForge-CLI.exe``).
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing
import os
import platform
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

from omniforge.core.logging import get_logger
from omniforge.version import get_version_info

logger = get_logger(__name__)


def is_frozen() -> bool:
    """Return True when running inside a PyInstaller / frozen executable bundle."""
    return bool(getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"))


def get_bundle_root() -> Path:
    """Return the primary root directory for bundled or source resources."""
    if is_frozen():
        return Path(str(getattr(sys, "_MEIPASS", ""))).resolve()
    return Path(__file__).resolve().parents[2]


def get_executable_dir() -> Path:
    """Return the directory containing the executable or project root."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def resolve_resource_path(relative_path: str | Path) -> Path:
    """Resolve a resource path across frozen PyInstaller bundles and source layouts.

    Args:
        relative_path: Path relative to package root, repository root, or bundle root.

    Returns:
        First existing resolved path, or the most likely candidate path if none exists.
    """
    rel = Path(relative_path)
    if rel.is_absolute() and rel.exists():
        return rel

    candidates: list[Path] = []
    if is_frozen():
        meipass = Path(str(getattr(sys, "_MEIPASS", ""))).resolve()
        exe_dir = Path(sys.executable).resolve().parent
        candidates.extend(
            [
                exe_dir / rel,
                meipass / rel,
                meipass / "omniforge" / rel,
            ]
        )

    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent.parent
    candidates.extend(
        [
            Path.cwd() / rel,
            pkg_dir / rel,
            repo_root / rel,
            repo_root / "src" / rel,
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_dotenv_file(env_path: Path | None = None) -> dict[str, str]:
    """Load key=value pairs from a ``.env`` file into ``os.environ`` without overwriting existing values.

    Args:
        env_path: Optional explicit path to a ``.env`` file. Defaults to searching
            executable directory and current working directory.

    Returns:
        Dictionary of variables loaded from the file.
    """
    loaded: dict[str, str] = {}
    search_paths = (
        [env_path]
        if env_path is not None
        else [Path.cwd() / ".env", get_executable_dir() / ".env"]
    )
    target: Path | None = next((p for p in search_paths if p and p.is_file()), None)
    if target is None:
        return loaded

    for raw_line in target.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def is_port_available(host: str, port: int) -> bool:
    """Return True if a TCP port is available to bind on the given host."""
    bind_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, port))
            return True
        except OSError:
            return False


def find_available_port(host: str, preferred_port: int, max_attempts: int = 25) -> int:
    """Return ``preferred_port`` if available, otherwise the next free TCP port."""
    for offset in range(max_attempts):
        candidate = preferred_port + offset
        if is_port_available(host, candidate):
            return candidate
    return preferred_port


def wait_for_port(host: str, port: int, timeout_seconds: float = 15.0) -> bool:
    """Block until a TCP port starts accepting connections or timeout elapses."""
    check_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((check_host, port)) == 0:
                return True
        time.sleep(0.25)
    return False


def run_environment_check() -> dict[str, Any]:
    """Verify Python runtime, dependencies, 18 domain plugins, resources, and permissions.

    Returns:
        Structured diagnostic report dictionary.
    """
    info = get_version_info()
    required_modules = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "pydantic",
        "pydantic_settings",
        "yaml",
        "plotly",
        "networkx",
        "typer",
        "rich",
        "fastapi",
        "uvicorn",
        "pyarrow",
        "duckdb",
        "sqlalchemy",
        "statsmodels",
        "openpyxl",
        "docx",
        "pptx",
        "reportlab",
        "nbformat",
        "defusedxml",
    ]
    optional_modules = [
        "torch",
        "xgboost",
        "lightgbm",
        "catboost",
        "optuna",
        "shap",
        "PyInstaller",
        "PIL",
    ]

    module_status: dict[str, bool] = {}
    for mod in required_modules:
        try:
            importlib.import_module(mod)
            module_status[mod] = True
        except Exception:
            module_status[mod] = False

    optional_status: dict[str, bool] = {}
    for mod in optional_modules:
        try:
            importlib.import_module(mod)
            optional_status[mod] = True
        except Exception:
            optional_status[mod] = False

    from omniforge.domains import default_domain_registry

    domains = [p.metadata.name for p in default_domain_registry.list_plugins()]
    ui_app_path = resolve_resource_path("omniforge/ui/app.py")
    if not ui_app_path.exists():
        ui_app_path = resolve_resource_path("ui/app.py")
    config_path = resolve_resource_path("configs/default.yaml")
    static_index = Path(__file__).resolve().parent / "ui" / "static_app" / "index.html"

    missing_required = [m for m, ok in module_status.items() if not ok]
    python_ok = sys.version_info >= (3, 12)
    domains_ok = len(domains) >= 18
    ui_ok = ui_app_path.exists() and static_index.is_file()
    overall_ok = python_ok and domains_ok and ui_ok and len(missing_required) == 0

    return {
        "status": "ok" if overall_ok else "degraded",
        "platform": info,
        "runtime": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "frozen": is_frozen(),
            "bundle_root": str(get_bundle_root()),
            "cwd": str(Path.cwd()),
        },
        "resources": {
            "ui_app_path": str(ui_app_path),
            "ui_app_exists": ui_ok,
            "static_app_exists": static_index.is_file(),
            "default_config_path": str(config_path),
            "default_config_exists": config_path.exists(),
        },
        "domains": {
            "count": len(domains),
            "registered": sorted(domains),
        },
        "required_modules": module_status,
        "missing_required_modules": missing_required,
        "optional_modules": optional_status,
    }


def _print_check_report(report: dict[str, Any], as_json: bool = False) -> None:
    """Render environment diagnostic report to stdout."""
    if as_json:
        print(json.dumps(report, indent=2))
        return

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    info = report["platform"]
    rt = report["runtime"]
    status_tag = "[bold green]READY[/bold green]" if report["status"] == "ok" else "[bold yellow]DEGRADED[/bold yellow]"
    console.print(
        Panel.fit(
            f"[bold cyan]{info['name']}[/bold cyan] v{info['version']} ({info['edition']})\n"
            f"Diagnostic Status: {status_tag}\n"
            f"Python: {rt['python_version']} | OS: {rt['os']} | Frozen EXE: {rt['frozen']}\n"
            f"Industry Domain Plugins Loaded: {report['domains']['count']}/18",
            title="OmniForge Environment Self-Check",
        )
    )

    table = Table(title="Dependency & Resource Verification")
    table.add_column("Component", style="cyan")
    table.add_column("Category")
    table.add_column("Status")
    table.add_row(
        "FastAPI + React Workbench Entry",
        "Resource",
        "[green]OK[/green]" if report["resources"]["ui_app_exists"] else "[red]MISSING[/red]",
    )
    table.add_row(
        "Default YAML Config",
        "Resource",
        "[green]OK[/green]" if report["resources"]["default_config_exists"] else "[yellow]FALLBACK[/yellow]",
    )
    for mod, ok in report["required_modules"].items():
        table.add_row(mod, "Required", "[green]OK[/green]" if ok else "[red]MISSING[/red]")
    for mod, ok in report["optional_modules"].items():
        table.add_row(mod, "Optional", "[green]INSTALLED[/green]" if ok else "[dim]not installed[/dim]")
    console.print(table)


def launch_api(host: str = "127.0.0.1", port: int = 8000, log_level: str = "info") -> None:
    """Run the FastAPI REST API server directly (safe inside PyInstaller bundles)."""
    import uvicorn

    from omniforge.api.main import app as fastapi_app

    chosen_port = find_available_port(host, port)
    if chosen_port != port:
        logger.warning("Requested API port busy; switched to available port", requested=port, chosen=chosen_port)

    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    print(f"[OmniForge API] Starting Enterprise FastAPI Backend at http://{display_host}:{chosen_port}")
    print(f"[OmniForge API] Interactive Swagger Docs: http://{display_host}:{chosen_port}/docs")
    uvicorn.run(fastapi_app, host=host, port=chosen_port, log_level=log_level)


def launch_ui(
    host: str = "127.0.0.1",
    port: int = 8501,
    open_browser: bool = True,
) -> None:
    """Launch the OmniForge FastAPI + React TypeScript Enterprise Workbench in-process (PyInstaller-safe)."""
    from omniforge.ui.app import run_workbench_server

    chosen_port = find_available_port(host, port)
    if chosen_port != port:
        logger.warning("Requested UI port busy; switched to available port", requested=port, chosen=chosen_port)
        print(
            f"[WARN] Port {port} is ALREADY IN USE by another process (possibly an older OmniForge instance)!\n"
            f"       Launching this updated OmniForge UI on port {chosen_port} instead.\n"
            f"       Open http://{('127.0.0.1' if host == '0.0.0.0' else host)}:{chosen_port} (NOT port {port})!"
        )

    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    url = f"http://{display_host}:{chosen_port}"
    static_dir = Path(__file__).resolve().parent / "ui" / "static_app"
    js_files = sorted(p.name for p in (static_dir / "assets").glob("index-*.js")) if (static_dir / "assets").is_dir() else []
    print(f"[OmniForge UI] Package Path  : {Path(__file__).resolve().parent}")
    print(f"[OmniForge UI] SPA Bundle    : {js_files[-1] if js_files else 'MISSING'}")
    print(f"[OmniForge UI] Starting FastAPI + React Enterprise Workbench at {url}")

    if open_browser:
        def _open_when_ready() -> None:
            if wait_for_port(display_host, chosen_port, timeout_seconds=25.0):
                try:
                    webbrowser.open(url)
                except Exception as exc:
                    logger.debug("browser_auto_open_skipped", url=url, error=str(exc))

        threading.Thread(target=_open_when_ready, daemon=True, name="omniforge-browser-opener").start()

    run_workbench_server(host=host, port=chosen_port, debug=False, open_browser=False)


def launch_full(
    host: str = "127.0.0.1",
    api_port: int = 8000,
    ui_port: int = 8501,
    open_browser: bool = True,
) -> None:
    """Launch both FastAPI backend (background thread) and OmniForge React SPA Workbench (foreground)."""
    import uvicorn

    from omniforge.api.main import app as fastapi_app

    chosen_api_port = find_available_port(host, api_port)
    chosen_ui_port = find_available_port(host, ui_port if ui_port != chosen_api_port else ui_port + 1)
    display_host = "127.0.0.1" if host == "0.0.0.0" else host

    config = uvicorn.Config(
        fastapi_app,
        host=host,
        port=chosen_api_port,
        log_level=os.environ.get("OMNIFORGE_LOG_LEVEL", "warning").lower(),
    )
    api_server = uvicorn.Server(config=config)
    api_thread = threading.Thread(
        target=api_server.run,
        daemon=True,
        name="omniforge-fastapi-server",
    )
    api_thread.start()

    ready = wait_for_port(display_host, chosen_api_port, timeout_seconds=15.0)
    v_info = get_version_info()
    print("=" * 72)
    print(f"  {v_info['name']} — {v_info['edition']} v{v_info['version']}")
    print("=" * 72)
    print(f"  REST API Server : http://{display_host}:{chosen_api_port} ({'READY' if ready else 'STARTING'})")
    print(f"  Swagger Docs    : http://{display_host}:{chosen_api_port}/docs")
    print(f"  Enterprise UI   : http://{display_host}:{chosen_ui_port}")
    print("  Press Ctrl+C in this window to stop all OmniForge services cleanly.")
    print("=" * 72)

    os.environ["OMNIFORGE_API_URL"] = f"http://{display_host}:{chosen_api_port}"

    try:
        launch_ui(host=host, port=chosen_ui_port, open_browser=open_browser)
    except (KeyboardInterrupt, SystemExit):
        api_server.should_exit = True
        api_thread.join(timeout=5.0)


def launch_cli(cli_args: list[str]) -> None:
    """Delegate execution to the Typer CLI."""
    from omniforge.cli.main import cli

    sys.argv = ["omniforge", *cli_args]
    cli()


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for the unified OmniForge launcher."""
    parser = argparse.ArgumentParser(
        prog="omniforge-desktop",
        description="OmniForge Unified Desktop, Server, UI & CLI Launcher.",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "ui", "api", "cli", "check"],
        default=os.environ.get("OMNIFORGE_LAUNCH_MODE", "full"),
        help="Launch mode: full (API+UI), ui (FastAPI + React Workbench), api (FastAPI), cli (Typer CLI), check (diagnostics).",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("OMNIFORGE_HOST", "127.0.0.1"),
        help="Bind host address (default: 127.0.0.1; use 0.0.0.0 for LAN access).",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=int(os.environ.get("OMNIFORGE_API_PORT", "8000")),
        help="FastAPI REST API port (default: 8000).",
    )
    parser.add_argument(
        "--ui-port",
        type=int,
        default=int(os.environ.get("OMNIFORGE_UI_PORT", "8501")),
        help="OmniForge FastAPI + React Workbench UI port (default: 8501).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the web browser on launch.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output diagnostic check report as JSON (only applies to --mode check).",
    )
    parser.add_argument(
        "cli_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to the Typer CLI when --mode cli is selected.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Primary entry point for ``omniforge-desktop`` and standalone ``OmniForge.exe``."""
    multiprocessing.freeze_support()
    load_dotenv_file()

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "check":
        report = run_environment_check()
        _print_check_report(report, as_json=args.json)
        return 0 if report["status"] == "ok" else 1

    if args.mode == "cli":
        forwarded = list(args.cli_args)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        launch_cli(forwarded)
        return 0

    if args.mode == "api":
        launch_api(host=args.host, port=args.api_port)
        return 0

    if args.mode == "ui":
        launch_ui(host=args.host, port=args.ui_port, open_browser=not args.no_browser)
        return 0

    launch_full(
        host=args.host,
        api_port=args.api_port,
        ui_port=args.ui_port,
        open_browser=not args.no_browser,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
