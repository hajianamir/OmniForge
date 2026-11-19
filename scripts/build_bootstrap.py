"""Build the small OmniForge bootstrap payload (``dist/omniforge-bootstrap``).

This produces everything the Inno Setup script copies — and nothing else:

    dist/omniforge-bootstrap/
    ├── app/                     OmniForge itself (src, configs, docs, SPA bundle, launcher)
    │   ├── src/omniforge/...
    │   ├── omniforge.py         thin entry point: `OmniForge.exe` -> `python omniforge.py`
    │   ├── configs/  docs/  examples/  notebooks/
    │   ├── pyproject.toml  requirements.txt  LICENSE  README.md  .env.example
    ├── runtime/
    │   ├── python/              CPython embeddable (pinned 3.13.x) + `_omegaf_ardenv.pth`
    │   └── uv.exe               self-contained resolver (extracted from the uv wheel)
    ├── locks/                   hash-pinned lock per tier (simple/advanced/everything)
    └── manifest.json            sizes + sha256 of every staged file (audit trail)

Usage (Windows, from the repo root):

    python scripts/build_bootstrap.py                 # full payload
    python scripts/build_bootstrap.py --skip-python   # reuse an already downloaded embeddable zip
    python scripts/build_bootstrap.py --dry-run       # print the plan, touch nothing

Offline/CI: ``--python-zip <file>`` and ``--uv-whl <file>`` avoid any network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "dist" / "omniforge-bootstrap"

#: Pinned embeddable runtime — must match the platform tags in ``locks/*.lock`` (cp313).
DEFAULT_PY_VERSION = "3.13.9"

#: Pinned uv release (the resolver shipped inside the installer).
DEFAULT_UV_VERSION = "0.13.9"

#: Files/directories copied into ``app/`` (payload of the installer).
APP_INCLUDE: tuple[str, ...] = (
    "src",
    "configs",
    "docs",
    "examples",
    "notebooks",
    "packaging",
    "pyproject.toml",
    "requirements.txt",
    "requirements-build.txt",
    "README.md",
    "LICENSE",
    ".env.example",
    "installer/provision.py",
)

#: Never staged (build outputs, caches, VCS, tests, frontend sources).
EXCLUDE_DIRS: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        ".venv",
        "dist",
        "build",
        ".omniforge_workspace",
        "tests",
    }
)
EXCLUDE_SUFFIXES: tuple[str, ...] = (".pyc", ".pyo", ".map")

ENTRY_TEMPLATE = '''"""OmniForge desktop entry point (invoked by OmniForge.exe in the installed bundle)."""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE_ROOT / "src"))


def main() -> int:
    """Freeze-safe launch of the unified desktop runtime."""
    multiprocessing.freeze_support()
    from omniforge.launcher import main as launcher_main

    return launcher_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
'''


class StagingError(RuntimeError):
    """Raised when the payload cannot be assembled."""


def log(msg: str) -> None:
    """Print a progress line."""
    print(f"[stage] {msg}", flush=True)


def uv_wheel_url(version: str) -> str:
    """Resolve the exact PyPI download URL for the win_amd64 uv wheel.

    ``files.pythonhosted.org`` paths carry a content hash, so they must be looked up through
    the JSON API instead of being constructed by hand.
    """
    api = f"https://pypi.org/pypi/uv/{version}/json"
    with urllib.request.urlopen(api, timeout=60) as response:
        payload = json.load(response)
    for item in payload.get("urls", []):
        if item["filename"] == f"uv-{version}-py3-none-win_amd64.whl":
            return str(item["url"])
    raise StagingError(f"uv {version} has no py3-none-win_amd64 wheel on PyPI")


def download(url: str, target: Path) -> Path:
    """Download ``url`` to ``target`` unless it already exists."""
    if target.is_file() and target.stat().st_size > 0:
        log(f"reusing {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading {url}")
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(target)
    log(f"saved {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def _is_excluded(path: Path) -> bool:
    """Return True when a path must not be staged."""
    if any(part in EXCLUDE_DIRS for part in path.parts):
        return True
    return path.suffix in EXCLUDE_SUFFIXES


def app_payload_complete(app_dir: Path) -> bool:
    """Is an existing ``app/`` tree usable, or does it need a (slow) full re-copy?

    A re-copy of the application tree is what invalidated a half-built installer on Windows:
    replacing ``app/`` while trusting a stale ``LICENSE.txt`` produced
    ``Line 73: Could not read ...app\\LICENSE.txt``. Prefer a full copy unless the tree is
    provably complete and carries every derived file we require.
    """
    required = (
        app_dir / "omniforge.py",
        app_dir / "LICENSE",
        app_dir / "LICENSE.txt",
        app_dir / "installer" / "provision.py",
        app_dir / "omniforge" / "__init__.py",
    )
    return all(path.exists() for path in required)


def refresh_derived_app_files(app_dir: Path) -> list[str]:
    """Re-create the files that are generated *from* the payload (never hand-maintained)."""
    derived: list[str] = []
    license_src = app_dir / "LICENSE"
    if license_src.is_file():
        shutil.copy2(license_src, app_dir / "LICENSE.txt")
        derived.append("LICENSE.txt")
    entry = app_dir / "omniforge.py"
    if not entry.is_file():
        entry.write_text(ENTRY_TEMPLATE, encoding="utf-8")
        derived.append("omniforge.py")
    return derived


def copy_app(app_dir: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Copy the application payload into ``app/``."""
    copied: list[str] = []
    for rel in APP_INCLUDE:
        src = PROJECT_ROOT / rel
        if not src.exists():
            log(f"WARN missing optional payload entry: {rel}")
            continue
        dest = app_dir / rel
        if dry_run:
            copied.append(rel)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(
                src,
                dest,
                dirs_exist_ok=True,
                ignore=lambda _d, names: [n for n in names if _is_excluded(Path(n))],
            )
            # ``copytree``'s ignore only filters the names it is handed, so nested caches can
            # slip through when a parent directory was copied wholesale — prune defensively.
            for cache_dir in list(dest.rglob("__pycache__")) + list(dest.rglob(".pytest_cache")):
                shutil.rmtree(cache_dir, ignore_errors=True)
            for compiled in dest.rglob("*.pyc"):
                compiled.unlink(missing_ok=True)
        else:
            shutil.copy2(src, dest)
        copied.append(rel)

    entry = app_dir / "omniforge.py"
    if not dry_run:
        entry.write_text(ENTRY_TEMPLATE, encoding="utf-8")

    # Inno Setup's [Setup] LicenseFile accepts ONLY .txt or .rtf (jrsoftware.org/ishelp:
    # "license agreement file, in .txt or .rtf format"), while the payload keeps the
    # extension-less LICENSE. Stage a byte-identical twin so ISCC, whose message is
    # otherwise just "Line N: ... LICENSE", finds a file it will actually accept.
    license_src = app_dir / "LICENSE"
    license_txt = app_dir / "LICENSE.txt"
    if dry_run:
        copied.append("LICENSE.txt (derived)")
    elif license_src.is_file():
        # Refresh unconditionally: a stale or missing twin must never survive into a build.
        shutil.copy2(license_src, license_txt)
        copied.append("LICENSE.txt (derived)")
    else:
        log("ERROR no LICENSE in the payload - the installer cannot reference it")
    log(f"app payload: {len(copied)} entries -> {app_dir}")
    return {"entries": copied, "entry_point": str(entry)}


def extract_python(embed_zip: Path, python_dir: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Unpack the embeddable CPython and make it venv-capable (``.pth`` import hook)."""
    if dry_run:
        log(f"[dry-run] would extract {embed_zip.name} -> {python_dir}")
        return {"zip": embed_zip.name, "entries": 0}

    if python_dir.exists():
        shutil.rmtree(python_dir)
    python_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(embed_zip) as archive:
        names = archive.namelist()
        archive.extractall(python_dir)

    # The embeddable distribution ships ``python313._pth`` which disables site handling.
    # Adding ``import site`` keeps the runtime isolated (no %APPDATA% user site) while still
    # letting ``venv`` / ``Site`` initialise inside the private environment we create.
    pth_files = sorted(python_dir.glob("python*._pth"))
    if not pth_files:
        raise StagingError("embeddable distribution has no python*._pth file")
    pth = pth_files[0]
    content = pth.read_text(encoding="utf-8", errors="ignore")
    if "#import site" in content:
        content = content.replace("#import site", "import site")
    elif "import site" not in content:
        content += "\nimport site\n"
    pth.write_text(content, encoding="utf-8")

    log(f"python runtime: {len(names)} files -> {python_dir}")
    return {"zip": embed_zip.name, "entries": len(names), "pth": pth.name}


def extract_uv(uv_wheel: Path, runtime_dir: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Pull ``uv.exe`` out of the published wheel (no installer, no PATH changes)."""
    if dry_run:
        log(f"[dry-run] would extract uv.exe from {uv_wheel.name}")
        return {"wheel": uv_wheel.name}
    target = runtime_dir / "uv.exe"
    with zipfile.ZipFile(uv_wheel) as archive:
        candidates = [n for n in archive.namelist() if n.endswith(("scripts/uv.exe", "scripts/uv"))]
        if not candidates:
            raise StagingError(f"no uv binary found inside {uv_wheel.name}")
        with archive.open(candidates[0]) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    log(f"uv resolver: {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
    return {"binary": str(target), "size_mb": round(target.stat().st_size / 1e6, 1)}


def stage_locks(locks_dir: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Verify the hash-pinned locks exist (they are generated by ``scripts/build_locks.py``)."""
    found = sorted(p.name for p in (PROJECT_ROOT / "locks").glob("*.lock"))
    if not found:
        raise StagingError("no lock files found — run `python scripts/build_locks.py` first")
    if not dry_run:
        locks_dir.mkdir(parents=True, exist_ok=True)
        for name in found:
            shutil.copy2(PROJECT_ROOT / "locks" / name, locks_dir / name)
    log(f"locks: {', '.join(found)}")
    return {"locks": found}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(out_dir: Path, parts: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    """Write ``manifest.json`` (sizes + sha256 of every staged file) — the installer's audit trail."""
    files: list[dict[str, Any]] = []
    total = 0
    if not dry_run:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                size = path.stat().st_size
                total += size
                files.append(
                    {"path": str(path.relative_to(out_dir)), "size": size, "sha256": _sha256(path)}
                )
    manifest: dict[str, Any] = {
        "product": "OmniForge",
        "payload_format": 1,
        "python": parts.get("python"),
        "uv": parts.get("uv"),
        "locks": parts.get("locks", {}).get("locks", []),
        "app_entries": parts.get("app", {}).get("entries", []),
        "total_files": len(files),
        "total_bytes": total,
        "total_mb": round(total / 1e6, 1),
        "files": files,
    }
    if not dry_run:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        log(f"manifest: {len(files)} files, {manifest['total_mb']} MB total")
    return manifest


def build(args: argparse.Namespace) -> dict[str, Any]:
    """Assemble the bootstrap payload."""
    out_dir = Path(args.out).resolve()
    dry_run = bool(args.dry_run)
    parts: dict[str, Any] = {}

    # Make sibling build helpers importable when run as a script (python scripts/build_bootstrap.py)
    if str(PROJECT_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

    if dry_run:
        log(f"[dry-run] target: {out_dir}")
        parts["app"] = copy_app(out_dir / "app", dry_run=True)
        parts["locks"] = stage_locks(out_dir / "locks", dry_run=True)
        parts["python"] = {"zip": f"python-{args.python_version}-embed-amd64.zip", "entries": 0}
        parts["uv"] = {"wheel": f"uv-{args.uv_version}-py3-none-win_amd64.whl"}
        return write_manifest(out_dir, parts, dry_run=True)

    # Incremental mode: never wipe, and only copy app/ when the existing tree is incomplete.
    incremental = bool(getattr(args, "skip_staged", False)) and not args.keep
    reuse_app = incremental and app_payload_complete(out_dir / "app")
    if reuse_app:
        log(f"reusing staged app payload: {out_dir / 'app'} (missing nothing)")
    elif out_dir.exists() and not args.keep:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = Path(args.cache).resolve() if args.cache else PROJECT_ROOT / ".cache" / "bootstrap"
    cache.mkdir(parents=True, exist_ok=True)

    # 1. application payload
    if reuse_app:
        refreshed = refresh_derived_app_files(out_dir / "app")
        parts["app"] = {"entries": ["(reused)"], "derived": refreshed}
    else:
        parts["app"] = copy_app(out_dir / "app")

    # 2. embeddable CPython
    if args.skip_python and (out_dir / "runtime" / "python").is_dir():
        log("python runtime already staged; skipping")
    else:
        embed_zip = (
            Path(args.python_zip).resolve()
            if args.python_zip
            else download(
                f"https://www.python.org/ftp/python/{args.python_version}/"
                f"python-{args.python_version}-embed-amd64.zip",
                cache / f"python-{args.python_version}-embed-amd64.zip",
            )
        )
        parts["python"] = extract_python(embed_zip, out_dir / "runtime" / "python")

    # 3. uv resolver
    uv_wheel = (
        Path(args.uv_whl).resolve()
        if args.uv_whl
        else download(uv_wheel_url(args.uv_version), cache / f"uv-{args.uv_version}-py3-none-win_amd64.whl")
    )
    parts["uv"] = extract_uv(uv_wheel, out_dir / "runtime")

    # 4. lock files
    parts["locks"] = stage_locks(out_dir / "locks")

    # 5. native launcher (OmniForge.exe) — venvlauncher.exe retargeted, no PyInstaller needed.
    #    Output goes INTO runtime\python: the Windows loader resolves python3XX.dll and the
    #    vcruntime DLLs from the executable's own directory, so co-locating is mandatory.
    from make_launcher_shim import build as build_launcher  # noqa: PLC0415 - sibling build script
    from make_launcher_shim import find_launcher_sources  # noqa: PLC0415

    launcher_zip = Path(args.launcher_zip).resolve() if args.launcher_zip else None
    console_launcher, gui_launcher = find_launcher_sources(None, None, launcher_zip)
    parts["launcher"] = build_launcher(
        out_dir / "runtime" / "python",
        console_launcher=console_launcher,
        gui_launcher=gui_launcher,
    )

    # 6. provenance
    return write_manifest(out_dir, parts)


def build_parser() -> argparse.ArgumentParser:
    """CLI for the staging builder."""
    parser = argparse.ArgumentParser(prog="build_bootstrap", description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory")
    parser.add_argument("--cache", default=None, help="Downloads cache (default .cache/bootstrap)")
    parser.add_argument("--python-version", default=DEFAULT_PY_VERSION, help="Embeddable CPython version")
    parser.add_argument("--uv-version", default=DEFAULT_UV_VERSION, help="uv release to bundle")
    parser.add_argument("--python-zip", default=None, help="Use a local embeddable zip (offline)")
    parser.add_argument("--uv-whl", default=None, help="Use a local uv wheel (offline)")
    parser.add_argument("--skip-python", action="store_true", help="Keep an already staged runtime")
    parser.add_argument(
        "--launcher-zip",
        default=None,
        help="CPython win-amd64 zip to extract venvlauncher.exe from (for OmniForge.exe)",
    )
    parser.add_argument(
        "--skip-staged",
        action="store_true",
        help="Reuse an existing, complete app/ tree; only runtime, locks and launchers are rebuilt",
    )
    parser.add_argument("--keep", action="store_true", help="Do not wipe the output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without writing")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    try:
        manifest = build(args)
    except StagingError as exc:
        log(f"ERROR {exc}")
        return 1
    if args.dry_run:
        log("dry-run complete (nothing written)")
    else:
        log(f"payload ready: {manifest.get('total_mb')} MB, {manifest.get('total_files')} files")
        log("next: compile installer\\omniforge_installer_tiers.iss with Inno Setup 6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
