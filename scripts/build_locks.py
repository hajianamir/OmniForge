"""Regenerate the hash-pinned lock files that the installer ships.

Run this in CI (or locally) whenever a dependency pin changes:

    uv pip install uv                      # only needed on the build machine
    python scripts/build_locks.py          # all tiers + the optional pack locks
    python scripts/build_locks.py --check  # verify the committed locks are up to date

The generated files are the *only* things the customer installer installs. They are produced
for ``--python-platform windows --python-version 3.13`` so that every wheel is a prebuilt
binary — the user's PC never needs a C compiler.

Layout:
    locks/simple.lock                    tier preset
    locks/advanced.lock
    locks/everything.lock
    locks/pack-<id>.lock                 one optional pack, installable from inside the app
    locks/dev.lock                       pytest/mypy/ruff (CI + developer machines only)
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from omniforge.capabilities import (  # noqa: E402
    ALL_PACKS,
    OPTIONAL_PACKS,
    TIERS,
    requirements_for,
    resolve_selection,
)

LOCKS_DIR = PROJECT_ROOT / "locks"
PYTHON_PLATFORM = "windows"
PYTHON_VERSION = "3.13"

#: Developer/CI-only packages (never installed on a user's machine).
DEV_REQUIREMENTS: tuple[str, ...] = (
    "pytest>=9.2.0",
    "pytest-cov>=7.1.0",
    "mypy>=2.4.1",
    "ruff>=0.17.5",
)


def _uv() -> str:
    """Locate the uv binary used for compiling locks."""
    exe = shutil.which("uv") or shutil.which("uv.exe")
    if not exe:
        raise SystemExit(
            "uv is not installed on this build machine.\n"
            "  pip install uv      (or download a release from https://github.com/astral-sh/uv)"
        )
    return exe


def compile_lock(requirements: list[str], target: Path, *, check: bool = False) -> dict[str, object]:
    """Compile one requirement set into a hash-pinned lock file."""
    if not requirements:
        return {"target": target.name, "skipped": True}
    prior_digest = (
        hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None
    )
    # A *stable* input filename keeps the generated locks byte-identical between runs
    # (uv records the input path in provenance comments, so a random temp name would make
    # every build look like a diff). It also makes `--check` meaningful in CI.
    req_path = LOCKS_DIR / f".requirements.{target.stem}.txt"
    out_path = LOCKS_DIR / f".check.{target.name}" if check else target
    if check and target.is_file():
        shutil.copyfile(target, out_path)
    req_path.write_text("\n".join(requirements) + "\n", encoding="utf-8")

    cmd = [
        _uv(), "pip", "compile", str(req_path),
        "--generate-hashes",
        "--python-platform", PYTHON_PLATFORM,
        "--python-version", PYTHON_VERSION,
        "--no-header", "--quiet",
        "-o", str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    req_path.unlink(missing_ok=True)
    if result.returncode != 0:
        out_path.unlink(missing_ok=True)
        raise SystemExit(f"lock generation failed for {target.name}:\n{result.stderr[-2000:]}")

    text = out_path.read_text(encoding="utf-8")
    # Replace the build-machine input path with a stable marker (keeps diffs reviewable).
    text = text.replace(str(req_path), "(requirements)")
    text = text.replace(f"-r locks/.requirements.{target.stem}.txt", "-r requirements.txt")
    if check:
        out_path.unlink(missing_ok=True)
    else:
        target.write_text(text, encoding="utf-8")
    pins = sum(1 for line in text.splitlines() if "==" in line and not line.startswith("    "))
    hashes = text.count("--hash=sha256:")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    is_stale = check and (prior_digest != digest or pins == 0)
    print(f"  {target.name:26s} {pins:4d} pins  {hashes:5d} hashes  sha256:{digest[:16]}>")
    return {"target": target.name, "pins": pins, "hashes": hashes, "sha256": digest, "stale": is_stale}


def main(argv: list[str] | None = None) -> int:
    """Generate (or verify) every lock the installer needs."""
    parser = argparse.ArgumentParser(prog="build_locks", description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Verify committed locks match the pins")
    parser.add_argument("--only", default=None, help="Comma-separated subset: simple,advanced,everything,dev,pack-<id>")
    args = parser.parse_args(argv)

    LOCKS_DIR.mkdir(exist_ok=True)
    targets: dict[str, list[str]] = {}

    for tier in TIERS:
        targets[f"{tier}.lock"] = requirements_for(resolve_selection(tier))  # type: ignore[arg-type]

    for pack_id in OPTIONAL_PACKS:
        # Every optional pack is compiled on top of "advanced": inside the app a user installs
        # one pack into an environment that already satisfies the advanced tier.
        targets[f"pack-{pack_id}.lock"] = requirements_for(resolve_selection("advanced", [pack_id]))

    targets["dev.lock"] = list(DEV_REQUIREMENTS)

    if args.only:
        wanted = {f"{name.strip()}.lock" for name in args.only.split(",")}
        targets = {k: v for k, v in targets.items() if k in wanted}
        if not targets:
            raise SystemExit(f"nothing matched --only {args.only!r}")

    print(f"compiling {len(targets)} lock file(s) for {PYTHON_PLATFORM}/cp{PYTHON_VERSION.replace('.', '')}")
    summary: list[dict[str, object]] = []
    for name, reqs in targets.items():
        summary.append(compile_lock(reqs, LOCKS_DIR / name, check=args.check))

    if args.check:
        # A lock is stale when recompiling produces a different digest.
        stale = [s["target"] for s in summary if s.get("stale")]
        if stale:
            raise SystemExit(f"stale locks: {stale}")
    total = len(ALL_PACKS)
    print(f"done — {len(summary)} lock(s), catalogue has {total} packs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
