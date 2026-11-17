# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for OmniForge (OmniForge.exe).

Supports both directory mode (fast startup, recommended for enterprise scientific stacks)
and single-file mode via environment variable:
    set OMNIFORGE_BUILD_MODE=onedir   (default)
    set OMNIFORGE_BUILD_MODE=onefile  (single standalone OmniForge.exe)
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

PROJECT_ROOT = Path(SPECPATH).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

ENTRY_SCRIPT = PROJECT_ROOT / "packaging" / "entry_desktop.py"
ICON_PATH = PROJECT_ROOT / "packaging" / "assets" / "omniforge.ico"
VERSION_FILE = PROJECT_ROOT / "packaging" / "omniforge_version_info.txt"

BUILD_MODE = os.environ.get("OMNIFORGE_BUILD_MODE", "onedir").strip().lower()
ONEFILE = BUILD_MODE == "onefile"
USE_UPX = os.environ.get("OMNIFORGE_USE_UPX", "0").strip() == "1"

block_cipher = None

datas: list[tuple[str, str]] = [
    (str(PROJECT_ROOT / "configs"), "configs"),
    (str(PROJECT_ROOT / ".env.example"), "."),
    (str(PROJECT_ROOT / "LICENSE"), "."),
]
# Bundle pre-compiled React/Vite TypeScript SPA static bundle in _MEIPASS (zero Node.js at runtime)
if (SRC_DIR / "omniforge" / "ui" / "static_app").is_dir():
    datas.append((str(SRC_DIR / "omniforge" / "ui" / "static_app"), "omniforge/ui/static_app"))
binaries: list[tuple[str, str]] = []
hiddenimports: list[str] = []

# Skip optional/test/example subpackages whose imports depend on uninstalled extras
# (e.g. ipywidgets, torch, cupy, pytest) so PyInstaller collects zero spurious submodule errors.
_SKIP_SUBMODULE_MARKERS: tuple[str, ...] = (
    ".tests",
    ".test_",
    ".testing",
    ".examples",
    ".benchmarks",
    ".conftest",
    "sklearn.externals.array_api_compat",
    "scipy._lib.array_api_compat",
    "statsmodels.sandbox",
)


def _keep_submodule(name: str) -> bool:
    if name.endswith(".__main__"):
        return False
    return not any(marker in name for marker in _SKIP_SUBMODULE_MARKERS)


def _is_installed(package_name: str) -> bool:
    try:
        return importlib.util.find_spec(package_name) is not None
    except Exception:
        return False


def _safe_collect_package(package_name: str, *, include_submodules: bool = True) -> None:
    """Collect data files, dynamic libs, and filtered submodules only for installed packages."""
    if not _is_installed(package_name):
        return
    try:
        datas.extend(collect_data_files(package_name, include_py_files=False))
    except Exception:
        pass
    try:
        binaries.extend(collect_dynamic_libs(package_name))
    except Exception:
        pass
    if include_submodules:
        try:
            hiddenimports.extend(collect_submodules(package_name, filter=_keep_submodule))
        except Exception:
            pass


def _safe_copy_metadata(dist_name: str) -> None:
    try:
        datas.extend(copy_metadata(dist_name))
    except Exception:
        pass


# 1. Collect ALL OmniForge submodules explicitly (all 13 UI studios, clustering, models, 18 domains, reporting, API, CLI)
hiddenimports.extend(collect_submodules("omniforge", filter=lambda n: not n.endswith(".__main__")))

# 2. Packages lacking official PyInstaller hooks — collect data + filtered submodules + native libs
for pkg in (
    "typer",
    "fastapi",
    "starlette",
    "pydantic_settings",
    "reportlab",
    "statsmodels",
    "seaborn",
    "networkx",
):
    _safe_collect_package(pkg, include_submodules=True)

# 3. Packages WITH official PyInstaller hooks or pure runtime data assets
#    (grammar files, JSON schemas, timezone data, TLS certs) — collect data/libs cleanly
#    without brute-forcing thousands of test/optional submodules (sklearn.tests, torch compat, etc.)
for hooked_pkg in (
    "plotly",
    "rich",
    "uvicorn",
    "pydantic",
    "docx",
    "pptx",
    "openpyxl",
    "nbformat",
    "duckdb",
    "pyarrow",
    "sklearn",
    "scipy",
    "rfc3987_syntax",
    "lark",
    "jsonschema",
    "jsonschema_specifications",
    "tzdata",
    "certifi",
):
    _safe_collect_package(hooked_pkg, include_submodules=False)

# 4. Optional ML/DL/Boosting libraries — only collected when installed (Full profile)
for optional_pkg in ("xgboost", "lightgbm", "catboost", "optuna", "shap", "torch"):
    _safe_collect_package(optional_pkg, include_submodules=True)

# 5. Distribution metadata required by importlib.metadata at runtime
for dist_name in (
    "omniforge",
    "plotly",
    "pydantic",
    "pydantic-settings",
    "fastapi",
    "uvicorn",
    "typer",
    "rich",
    "scikit-learn",
    "numpy",
    "pandas",
    "scipy",
    "statsmodels",
    "pyarrow",
    "duckdb",
    "nbformat",
    "jsonschema",
    "jsonschema-specifications",
    "rfc3987-syntax",
    "lark",
):
    _safe_copy_metadata(dist_name)

# 6. Verified runtime hidden imports (only added when resolvable in active Python environment)
for mod_name in (
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "multipart",
    "python_multipart",
    "jwt",
    "cryptography",
    "cryptography.fernet",
    "sqlalchemy.sql.default_comparator",
    "sklearn.utils._typedefs",
    "sklearn.utils._heap",
    "sklearn.utils._sorting",
    "sklearn.utils._vector_sentinel",
    "sklearn.neighbors._partition_nodes",
    "scipy._lib.messagestream",
    "scipy.special._ufuncs_cxx",
):
    if _is_installed(mod_name):
        hiddenimports.append(mod_name)

# Deduplicate hidden imports while preserving order
hiddenimports = list(dict.fromkeys(hiddenimports))

a = Analysis(
    [str(ENTRY_SCRIPT)],
    pathex=[str(PROJECT_ROOT), str(SRC_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter.test",
        "pytest",
        "IPython",
        "jupyterlab",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

icon_arg = str(ICON_PATH) if ICON_PATH.is_file() else None
version_arg = str(VERSION_FILE) if (sys.platform == "win32" and VERSION_FILE.is_file()) else None

if ONEFILE:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name="OmniForge",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=USE_UPX,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=icon_arg,
        version=version_arg,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="OmniForge",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=USE_UPX,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=icon_arg,
        version=version_arg,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=USE_UPX,
        upx_exclude=[],
        name="OmniForge",
    )
