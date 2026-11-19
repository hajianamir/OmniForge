"""OmniForge Windows installer bootstrapper — small installer, on-demand provisioning.

Runs from the Inno Setup ``[Run]`` section, or manually for repair:

    provision.py --tier advanced --app-dir "C:\\Program Files\\OmniForge" --check-only
    provision.py --tier advanced --packs boosting,responsible --wheels \\\\srv\\wheels
    provision.py --repair
    provision.py --list-packs --json

Design rules (see docs/INSTALLER_PROVISIONING_PROPOSAL.md):
  * Never touch the user's global Python / conda / site-packages.
  * Install strictly from a hash-pinned lock with ``uv sync --frozen --require-hashes``.
  * Reuse artifacts that already exist locally (wheelhouse / uv cache / mirror) before downloading.
  * Idempotent: safe to re-run, and used by the uninstaller's "Repair".
  * Keeps the previous environment so a failed upgrade can roll back.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

# The pack catalogue lives in the application package so there is exactly one source of truth.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from omniforge.capabilities import (  # noqa: E402
    ALL_PACKS,
    OPTIONAL_PACKS,
    TIERS,
    Tier,
    profile_summary,
    requirements_for,
    resolve_selection,
)

PROVISION_STATE = "provision.json"
VENV_DIR = "venv"
VENV_BACKUP = "venv.prev"
LOCK_DIR = "locks"
MIN_PYTHON = (3, 12)
EXIT_OK, EXIT_USAGE, EXIT_NETWORK, EXIT_VERIFY, EXIT_INTERRUPTED = 0, 2, 3, 4, 5


# ---------------------------------------------------------------------------
# Console helpers (the installer may hide stdout; every write is guarded)
# ---------------------------------------------------------------------------


def log(msg: str, *, level: str = "INFO") -> None:
    """Emit a line the Inno Setup progress page can forward to its log view."""
    try:
        print(f"[{level}] {msg}", flush=True)
    except Exception:  # pragma: no cover - detached console
        pass


def _write_status(status_path: Path | None, **payload: Any) -> None:
    """Atomically publish progress so the installer window can render a progress bar."""
    if status_path is None:
        return
    payload = {"ts": time.time(), **payload}
    tmp = status_path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, status_path)
    except OSError:
        # Progress reporting must never fail the install.
        pass


# ---------------------------------------------------------------------------
# Environment discovery
# ---------------------------------------------------------------------------


@dataclass
class ProvisionPlan:
    """Everything resolved before a single byte is downloaded."""

    app_dir: Path
    data_dir: Path
    tier: str
    packs: list[str]
    requirements: list[str]
    python_exe: Path
    uv_exe: Path
    lock_file: Path | None
    wheelhouse: Path | None = None
    index_url: str | None = None
    offline: bool = False
    status_path: Path | None = None
    summary: dict[str, Any] = field(default_factory=dict)


def _candidate_pythons(app_dir: Path) -> list[Path]:
    """Bundled embeddable runtime first, then a discovered system interpreter."""
    out = [
        app_dir / "runtime" / "python" / "python.exe",
        app_dir / "runtime" / "python.exe",
        app_dir / "python" / "python.exe",
    ]
    for name in ("python3.13", "python3.12", "python3", "python"):
        found = shutil.which(name)
        if found:
            out.append(Path(found))
    return out


def find_python(app_dir: Path) -> Path:
    """Return the bundled interpreter, else the newest compatible system Python."""
    for cand in _candidate_pythons(app_dir):
        if not cand.is_file():
            continue
        try:
            proc = subprocess.run(
                [str(cand), "-c", "import sys;print(sys.version_info[0], sys.version_info[1])"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        parts = (proc.stdout or "").split()
        if len(parts) == 2 and tuple(int(p) for p in parts) >= MIN_PYTHON:
            return cand
    raise FileNotFoundError(
        "No compatible Python interpreter found. The installer must ship the embeddable runtime "
        "in <app>\\runtime\\python\\ or the machine needs Python 3.12+."
    )


def find_uv(app_dir: Path) -> Path | None:
    """Return the bundled ``uv`` binary, else one on PATH, else ``None`` (fall back to pip)."""
    for cand in (app_dir / "runtime" / "uv.exe", app_dir / "runtime" / "uv"):
        if cand.is_file():
            return cand
    found = shutil.which("uv") or shutil.which("uv.exe")
    return Path(found) if found else None


def _lock_for(tier: str, app_dir: Path, requested: list[str]) -> Path | None:
    """Pick the lock file that matches the selection.

    Resolution order:
      1. ``pack-<id>.lock`` when exactly ONE optional pack was requested on top of the tier
         (this is the "install this capability from inside the app" path);
      2. ``<tier>.lock`` — the tier preset (which already includes everything in that tier);
      3. ``everything.lock`` — last-resort superset, still hash-pinned.
    """
    base = app_dir / LOCK_DIR
    optional = [p for p in requested if p in OPTIONAL_PACKS]
    candidates: list[str] = []
    if len(optional) == 1:
        candidates.append(f"pack-{optional[0]}.lock")
    candidates.extend([f"{tier}.lock", "everything.lock"])
    for name in candidates:
        cand = base / name
        if cand.is_file():
            return cand
    return None


def _tier(value: str) -> Tier:
    """Validate/narrow the CLI tier flag (argparse already restricts the choices)."""
    if value not in TIERS:
        raise KeyError(f"Unknown tier: {value!r}")
    return cast(Tier, value)


def build_plan(args: argparse.Namespace) -> ProvisionPlan:
    """Resolve interpreter, uv, packs and lock file without touching the network."""
    app_dir = Path(args.app_dir).resolve()
    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
    elif os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        data_dir = Path(os.environ["LOCALAPPDATA"]) / "OmniForge"
    else:
        data_dir = Path.home() / ".omniforge" / "desktop"
    requested = [p.strip() for p in (args.packs or "").split(",") if p.strip()]
    tier = _tier(args.tier)
    packs = resolve_selection(tier, requested)
    python_exe = find_python(app_dir)
    uv_exe = find_uv(app_dir)
    lock = _lock_for(args.tier, app_dir, requested)
    wheelhouse = Path(args.wheels).resolve() if args.wheels else None
    return ProvisionPlan(
        app_dir=app_dir,
        data_dir=data_dir,
        tier=tier,
        packs=[p.id for p in packs],
        requirements=requirements_for(packs),
        python_exe=python_exe,
        uv_exe=uv_exe or Path("uv"),
        lock_file=lock,
        wheelhouse=wheelhouse,
        index_url=args.index_url,
        offline=bool(args.offline),
        status_path=Path(args.status) if args.status else None,
        summary=profile_summary(tier, requested),
    )


# ---------------------------------------------------------------------------
# Artifact reuse ("check what is already available, download only the rest")
# ---------------------------------------------------------------------------


def local_artifact_report(plan: ProvisionPlan) -> dict[str, int]:
    """Count how many wheels we already have locally before downloading anything."""
    report = {"wheelhouse_wheels": 0, "cache_wheels": 0, "needs_download_estimate": 0}
    if plan.wheelhouse and plan.wheelhouse.is_dir():
        report["wheelhouse_wheels"] = sum(1 for _ in plan.wheelhouse.glob("*.whl"))
    cache_root = Path(os.environ.get("UV_CACHE_DIR", Path.home() / ".cache" / "uv"))
    if cache_root.is_dir():
        report["cache_wheels"] = sum(1 for _ in cache_root.rglob("*.whl"))
    report["needs_download_estimate"] = max(
        0, len(plan.requirements) - report["wheelhouse_wheels"]
    )
    return report


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _run(cmd: list[str], *, env: dict[str, str], timeout: int | None = None) -> int:
    """Run a child process, streaming output; return the exit code."""
    log(" ".join(str(c) for c in cmd), level="CMD")
    try:
        proc = subprocess.run(cmd, env=env, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        log(f"Executable not found: {exc}", level="ERROR")
        return 127
    except subprocess.SubprocessError as exc:
        log(f"Subprocess failure: {exc}", level="ERROR")
        return 1
    return proc.returncode


def create_venv(plan: ProvisionPlan) -> Path:
    """Create the private virtual environment (idempotent)."""
    venv_dir = plan.data_dir / VENV_DIR
    python_in_venv = venv_dir / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    if python_in_venv.is_file():
        log(f"Reusing existing environment: {venv_dir}")
        return venv_dir
    if venv_dir.exists():
        shutil.rmtree(venv_dir, ignore_errors=True)
    log(f"Creating private environment at {venv_dir}")
    code = _run([str(plan.python_exe), "-m", "venv", "--without-pip", str(venv_dir)], env=os.environ.copy(), timeout=300)
    if code != 0 or not python_in_venv.is_file():
        raise RuntimeError("Failed to create the virtual environment from the bundled runtime.")
    return venv_dir


def install_packs(plan: ProvisionPlan, venv_dir: Path) -> None:
    """Install the selected packs, reusing local artifacts before downloading."""
    _write_status(plan.status_path, phase="resolving", detail="Reading local artifact cache")
    report = local_artifact_report(plan)
    log(f"Local artifacts: {report}")

    env = os.environ.copy()
    env["UV_LINK_MODE"] = "copy"          # wheels land inside the app's own venv
    env["UV_PYTHON_DOWNLOADS"] = "never"  # never fetch a second interpreter

    if plan.offline:
        env["UV_OFFLINE"] = "1"
    if plan.index_url:
        env["UV_INDEX_URL"] = plan.index_url

    common: list[str] = []
    if plan.wheelhouse:
        common += ["--find-links", str(plan.wheelhouse)]
    if plan.lock_file is not None:
        # Reproducible, hash-verified install — the only path we ship to customers.
        # ``uv pip sync`` installs the lock exactly (removing anything not in it) and
        # ``--require-hashes`` makes every wheel hash-checked against the lock.
        cmd = [str(plan.uv_exe), "pip", "sync", "--require-hashes",
               "--python", str(venv_dir), str(plan.lock_file), *common]
        if plan.offline:
            cmd.append("--offline")
    else:
        # No lock present (developer machine): plain resolution, still isolated.
        log("No lock file packaged; falling back to direct resolution (not for release builds).", level="WARN")
        req_file = plan.data_dir / "requirements.selection.txt"
        req_file.write_text("\n".join(plan.requirements) + "\n", encoding="utf-8")
        cmd = [str(plan.uv_exe), "pip", "install", "--python", str(venv_dir), "-r", str(req_file), *common]

    _write_status(plan.status_path, phase="downloading", detail=f"{len(plan.packs)} pack(s)")
    code = _run(cmd, env=env, timeout=None)
    if code == 127:
        raise RuntimeError(
            "The bundled 'uv' resolver is missing. Reinstall OmniForge with the bundled runtime, "
            "or provide --wheels/--index-url for an offline install."
        )
    if code != 0:
        raise RuntimeError(
            "Dependency resolution failed. Common causes: no internet access, a proxy blocking PyPI, "
            "or a lock file built for a different Python version. Re-run with --repair, or point "
            "--wheels at a local wheelhouse for an offline install."
        )


def write_receipt(plan: ProvisionPlan, venv_dir: Path) -> None:
    """Persist what was installed so the app can verify and offer targeted repairs."""
    receipt = {
        "omniforge": _app_version(plan),
        "tier": plan.tier,
        "packs": plan.packs,
        "requirements": plan.requirements,
        "python": str(plan.python_exe),
        "venv": str(venv_dir),
        "lock": str(plan.lock_file) if plan.lock_file else None,
        "lock_sha256": _sha256(plan.lock_file) if plan.lock_file else None,
        "uv": str(plan.uv_exe),
        "platform": platform.platform(),
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": "offline" if plan.offline else (plan.index_url or "pypi"),
        "wheelhouse": str(plan.wheelhouse) if plan.wheelhouse else None,
        "local_artifacts": local_artifact_report(plan),
    }
    (plan.data_dir / PROVISION_STATE).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    log(f"Provisioning receipt written to {plan.data_dir / PROVISION_STATE}")


def _app_version(plan: ProvisionPlan) -> str:
    try:
        sys.path.insert(0, str(plan.app_dir / "src"))
        from omniforge.version import __version__ as version

        return str(version)
    except Exception:
        return "unknown"


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(plan: ProvisionPlan, venv_dir: Path) -> dict[str, Any]:
    """Import-probe every pack in the fresh environment (the honest post-install check)."""
    probe = (
        "import importlib, json, sys\n"
        "missing = []\n"
        "for name in json.loads(sys.argv[1]):\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception as exc:\n"
        "        missing.append([name, type(exc).__name__])\n"
        "print(json.dumps({'missing': missing}))\n"
    )
    python_in_venv = venv_dir / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    imports = plan.summary.get("imports", [])
    proc = subprocess.run(
        [str(python_in_venv), "-c", probe, json.dumps(imports)],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    try:
        result: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        result = {"missing": [["<probe>", "probe_failed"]], "stderr": proc.stderr[-500:]}
    result["ok"] = not result.get("missing")
    return result


def rollback(plan: ProvisionPlan) -> None:
    """Restore the previous environment when an upgrade breaks imports."""
    venv_dir = plan.data_dir / VENV_DIR
    backup = plan.data_dir / VENV_BACKUP
    if not backup.is_dir():
        log("No previous environment to roll back to.", level="WARN")
        return
    if venv_dir.is_dir():
        shutil.rmtree(venv_dir, ignore_errors=True)
    backup.rename(venv_dir)
    log("Rolled back to the previous environment.", level="WARN")


def snapshot_previous(plan: ProvisionPlan) -> None:
    """Keep the working environment so ``--rollback`` can restore it."""
    venv_dir = plan.data_dir / VENV_DIR
    backup = plan.data_dir / VENV_BACKUP
    if venv_dir.is_dir():
        shutil.rmtree(backup, ignore_errors=True)
        try:
            shutil.copytree(venv_dir, backup, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
        except OSError as exc:  # pragma: no cover - disk full / locks
            log(f"Could not snapshot the previous environment: {exc}", level="WARN")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the installer-facing CLI."""
    parser = argparse.ArgumentParser(prog="provision", description="OmniForge dependency provisioner")
    parser.add_argument("--tier", choices=sorted(TIERS), default="advanced", help="Installation profile")
    parser.add_argument("--packs", default="", help="Comma-separated extra optional pack ids")
    parser.add_argument("--app-dir", default=str(Path(__file__).resolve().parent.parent), help="Directory containing runtime/ and locks/")
    parser.add_argument("--data-dir", default=None, help="Where the private venv lives (defaults to %%LOCALAPPDATA%%\\OmniForge)")
    parser.add_argument("--wheels", default=None, help="Local directory or UNC share with pre-downloaded wheels")
    parser.add_argument("--index-url", default=None, help="Corporate PyPI mirror (Artifactory/Nexus/devpi)")
    parser.add_argument("--offline", action="store_true", help="Never touch the network; require --wheels")
    parser.add_argument("--status", default=None, help="Path for the JSON progress file used by the installer UI")
    parser.add_argument("--check-only", action="store_true", help="Resolve the plan and exit without installing")
    parser.add_argument("--repair", action="store_true", help="Re-run the install over an existing environment")
    parser.add_argument("--rollback", action="store_true", help="Restore the previous environment")
    parser.add_argument("--list-packs", action="store_true", help="Print the pack catalogue")
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point used by Inno Setup's ``[Run]`` section."""
    args = build_parser().parse_args(argv)

    if args.list_packs:
        catalogue = [
            {
                "id": p.id,
                "title": p.title,
                "description": p.description,
                "kind": p.kind,
                "approx_mb": p.approx_mb,
                "dists": list(p.dists),
                "unlocks": p.unlocks,
            }
            for p in ALL_PACKS
        ]
        payload = {"tiers": {k: list(v) for k, v in TIERS.items()}, "packs": catalogue}
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for tier, ids in TIERS.items():
                print(f"{tier}: {', '.join(ids)}")
            for p in ALL_PACKS:
                print(f"  [{p.kind:8s}] {p.id:14s} ~{p.approx_mb:5d} MB  {p.title}")
        return EXIT_OK

    try:
        plan = build_plan(args)
    except (FileNotFoundError, KeyError) as exc:
        log(str(exc), level="ERROR")
        return EXIT_USAGE

    log(f"Profile   : {plan.tier}")
    log(f"Packs     : {', '.join(plan.packs)}")
    log(f"Python    : {plan.python_exe}")
    log(f"Resolver  : {plan.uv_exe}")
    log(f"Lock file : {plan.lock_file or 'MISSING (release builds must ship one)'}")
    log(f"Data dir  : {plan.data_dir}")

    if args.check_only:
        resolver = str(plan.uv_exe)
        if not (resolver and (shutil.which(resolver) or Path(resolver).exists())):
            log("uv resolver not found (bundle it at <app>\\runtime\\uv.exe)", level="ERROR")
            return EXIT_USAGE
        if plan.lock_file is None:
            log("No lock file for this selection", level="ERROR")
            return EXIT_VERIFY
        if args.json:
            print(json.dumps({"plan": plan.summary, "python": str(plan.python_exe), "lock": str(plan.lock_file)}, indent=2))
        return EXIT_OK

    if args.rollback:
        rollback(plan)
        return EXIT_OK

    plan.data_dir.mkdir(parents=True, exist_ok=True)
    try:
        if not args.repair:
            snapshot_previous(plan)
        _write_status(plan.status_path, phase="venv", detail="Creating private environment")
        venv_dir = create_venv(plan)
        install_packs(plan, venv_dir)
        _write_status(plan.status_path, phase="verify", detail="Import-probing installed packs")
        result = verify(plan, venv_dir)
        if not result["ok"]:
            log(f"Post-install verification failed: {result['missing']}", level="ERROR")
            _write_status(plan.status_path, phase="failed", detail=str(result["missing"]))
            rollback(plan)
            return EXIT_VERIFY
        write_receipt(plan, venv_dir)
        _write_status(plan.status_path, phase="done", detail="Installation complete")
        log("Installation complete.")
        return EXIT_OK
    except RuntimeError as exc:
        log(str(exc), level="ERROR")
        _write_status(plan.status_path, phase="failed", detail=str(exc))
        return EXIT_NETWORK
    except KeyboardInterrupt:  # pragma: no cover - user cancelled
        log("Cancelled by user.", level="WARN")
        _write_status(plan.status_path, phase="cancelled")
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    raise SystemExit(main())
