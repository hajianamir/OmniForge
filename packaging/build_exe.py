"""Enterprise Executable Build Orchestrator for OmniForge.

Builds standalone Windows executables (``OmniForge.exe`` and optionally ``OmniForge-CLI.exe``)
using PyInstaller, generates multi-resolution application icons, bundles configuration
templates, writes SHA-256 checksums, and optionally compiles an Inno Setup installer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGING_DIR = PROJECT_ROOT / "packaging"
PYINSTALLER_DIR = PACKAGING_DIR / "pyinstaller"
ASSETS_DIR = PACKAGING_DIR / "assets"
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_icon_assets() -> tuple[Path, Path]:
    """Ensure ``omniforge.ico`` and ``omniforge_logo.png`` exist, generating them if absent."""
    ico_path = ASSETS_DIR / "omniforge.ico"
    png_path = ASSETS_DIR / "omniforge_logo.png"
    if ico_path.is_file() and png_path.is_file():
        return ico_path, png_path

    if importlib.util.find_spec("PIL") is None:
        raise RuntimeError(
            "Pillow is required to generate omniforge.ico. Install build requirements via: "
            "pip install -r requirements-build.txt"
        )

    sys.path.insert(0, str(PACKAGING_DIR))
    from generate_icon import generate_omniforge_icon

    return generate_omniforge_icon(ASSETS_DIR)


def validate_build_environment(require_pyinstaller: bool = True) -> dict[str, Any]:
    """Validate specs, entry scripts, OmniForge package, icon, and PyInstaller availability."""
    checks: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "project_root": str(PROJECT_ROOT),
        "pyinstaller_installed": importlib.util.find_spec("PyInstaller") is not None,
        "pillow_installed": importlib.util.find_spec("PIL") is not None,
        "omniforge_spec_exists": (PYINSTALLER_DIR / "omniforge.spec").is_file(),
        "omniforge_cli_spec_exists": (PYINSTALLER_DIR / "omniforge_cli.spec").is_file(),
        "desktop_entry_exists": (PACKAGING_DIR / "entry_desktop.py").is_file(),
        "cli_entry_exists": (PACKAGING_DIR / "entry_cli.py").is_file(),
        "version_info_exists": (PACKAGING_DIR / "omniforge_version_info.txt").is_file(),
    }

    missing = [k for k, v in checks.items() if k.endswith("_exists") and not v]
    if missing:
        raise FileNotFoundError(f"Missing required build assets: {', '.join(missing)}")

    # Verify spec syntax using Python's compiler (catches syntax regressions without invoking PyInstaller)
    for spec_name in ("omniforge.spec", "omniforge_cli.spec"):
        spec_path = PYINSTALLER_DIR / spec_name
        compile(spec_path.read_text(encoding="utf-8"), str(spec_path), "exec")

    if require_pyinstaller and not checks["pyinstaller_installed"]:
        raise RuntimeError(
            "PyInstaller is not installed in the active Python environment. "
            "Run: pip install -r requirements-build.txt"
        )

    return checks


def run_pyinstaller(spec_path: Path, build_mode: str, use_upx: bool = False, dry_run: bool = False) -> list[str]:
    """Invoke PyInstaller for a given ``.spec`` file."""
    env = os.environ.copy()
    env["OMNIFORGE_BUILD_MODE"] = build_mode
    env["OMNIFORGE_USE_UPX"] = "1" if use_upx else "0"
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(BUILD_DIR / "pyinstaller"),
        str(spec_path),
    ]
    print(f"[BUILD] Mode={build_mode} | Spec={spec_path.name}")
    print(f"[BUILD] Command: {' '.join(cmd)}")

    if dry_run:
        return cmd

    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=True)
    return cmd


def write_checksums(dist_root: Path) -> Path:
    """Write ``SHA256SUMS.txt`` for built executables in ``dist/``."""
    checksum_lines: list[str] = []
    candidates = [
        dist_root / "OmniForge.exe",
        dist_root / "OmniForge" / "OmniForge.exe",
        dist_root / "OmniForge-CLI.exe",
        dist_root / "OmniForge-CLI" / "OmniForge-CLI.exe",
        dist_root / "OmniForge",
        dist_root / "OmniForge" / "OmniForge",
        dist_root / "OmniForge-CLI",
        dist_root / "OmniForge-CLI" / "OmniForge-CLI",
    ]
    seen: set[Path] = set()
    for path in candidates:
        if path.is_file() and path.resolve() not in seen:
            seen.add(path.resolve())
            rel = path.relative_to(dist_root).as_posix()
            checksum_lines.append(f"{sha256_file(path)}  {rel}")

    manifest_path = dist_root / "SHA256SUMS.txt"
    manifest_path.write_text("\n".join(checksum_lines) + ("\n" if checksum_lines else ""), encoding="utf-8")
    return manifest_path


def run_smoke_test(build_mode: str) -> None:
    """Execute ``OmniForge.exe --mode check`` to validate the built executable."""
    exe_name = "OmniForge.exe" if platform.system() == "Windows" else "OmniForge"
    exe_path = (DIST_DIR / exe_name) if build_mode == "onefile" else (DIST_DIR / "OmniForge" / exe_name)
    if not exe_path.is_file():
        print(f"[WARN] Executable not found for smoke test: {exe_path}")
        return
    print(f"[VERIFY] Running smoke test: {exe_path} --mode check")
    subprocess.run([str(exe_path), "--mode", "check"], check=True)
    print("[VERIFY] Executable smoke test PASSED.")


def compile_inno_setup(dry_run: bool = False) -> None:
    """Compile Inno Setup installer if ``iscc`` is available on Windows."""
    iss_path = PACKAGING_DIR / "inno_setup" / "omniforge_installer.iss"
    iscc = shutil.which("iscc") or shutil.which("ISCC.exe")
    if not iscc:
        default_iscc = Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe")
        if default_iscc.is_file():
            iscc = str(default_iscc)

    if not iscc:
        print("[INFO] Inno Setup 6 (ISCC.exe) not found; skipping Setup Wizard installer generation.")
        return

    cmd = [iscc, str(iss_path)]
    print(f"[INNO] Compiling Windows Setup Wizard: {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the executable build orchestrator."""
    parser = argparse.ArgumentParser(
        description="Build standalone executables for OmniForge."
    )
    parser.add_argument(
        "--mode",
        choices=["onedir", "onefile"],
        default="onedir",
        help="PyInstaller bundle mode: onedir (fast startup, default) or onefile (single portable .exe).",
    )
    parser.add_argument(
        "--target",
        choices=["desktop", "cli", "all"],
        default="desktop",
        help="Target executable(s) to build: desktop (OmniForge.exe), cli (OmniForge-CLI.exe), or all.",
    )
    parser.add_argument(
        "--upx",
        action="store_true",
        help="Enable UPX binary compression if UPX is installed on PATH.",
    )
    parser.add_argument(
        "--with-inno",
        action="store_true",
        help="Also compile the Inno Setup 6 Windows Setup Wizard (requires onedir build + Inno Setup 6).",
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Skip post-build executable verification.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate specs, assets, and print PyInstaller commands without invoking PyInstaller.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute the full OmniForge executable build pipeline."""
    args = parse_args(argv)

    print("=" * 72)
    print("  OmniForge v2.0.0 — Executable Builder")
    print("=" * 72)

    ico_path, png_path = ensure_icon_assets()
    print(f"[ASSET] Icon ready: {ico_path} | Logo PNG: {png_path}")

    checks = validate_build_environment(require_pyinstaller=not args.dry_run)
    print(f"[CHECK] Python {checks['python_version']} on {checks['platform']} | Specs validated OK")

    if args.target in ("desktop", "all"):
        run_pyinstaller(PYINSTALLER_DIR / "omniforge.spec", build_mode=args.mode, use_upx=args.upx, dry_run=args.dry_run)

    if args.target in ("cli", "all"):
        cli_mode = "onefile" if args.mode == "onefile" else "onedir"
        run_pyinstaller(PYINSTALLER_DIR / "omniforge_cli.spec", build_mode=cli_mode, use_upx=args.upx, dry_run=args.dry_run)

    if args.dry_run:
        print("[DRY-RUN] Validation complete. No binaries were compiled.")
        return 0

    manifest = write_checksums(DIST_DIR)
    print(f"[BUILD] SHA-256 manifest written to: {manifest}")

    if not args.skip_smoke_test and args.target in ("desktop", "all"):
        run_smoke_test(args.mode)

    if args.with_inno and args.mode == "onedir":
        compile_inno_setup(dry_run=args.dry_run)

    print("=" * 72)
    print(f"  Build finished successfully! Output directory: {DIST_DIR}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
