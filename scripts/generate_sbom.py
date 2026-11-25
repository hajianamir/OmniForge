"""Generate or verify CycloneDX 1.5 JSON SBOMs & SHA-256 lockfile provenance (RR-022).

Usage:
    python scripts/generate_sbom.py --out sbom.cdx.json
    python scripts/generate_sbom.py --image omniforge:local --out docker-sbom.cdx.json
    python scripts/generate_sbom.py --verify
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from omniforge.core.sbom import generate_cyclonedx_sbom, verify_lockfiles_provenance, write_cyclonedx_sbom  # noqa: E402


def write_verified_image_sbom(image: str, output: Path) -> None:
    """Check the image's installed Python distributions against its architecture lock."""
    identity = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Os}} {{.Architecture}} {{.Id}}", image], text=True
    ).split()
    os_name, architecture, image_id = identity
    if os_name != "linux" or architecture not in {"amd64", "arm64"}:
        raise ValueError(f"No runtime lock for {os_name}/{architecture}.")
    lock = PROJECT_ROOT / "docker" / f"requirements-{architecture}.lock"
    sbom = generate_cyclonedx_sbom(lock)
    distributions = json.loads(subprocess.check_output(
        [
            "docker", "run", "--rm", "--network", "none", "--entrypoint", "/opt/venv/bin/python", image_id,
            "-c", 'import importlib.metadata as m, json; print(json.dumps([(d.metadata["Name"], d.version) for d in m.distributions()]))',
        ],
        text=True,
    ))
    installed = {re.sub(r"[-_.]+", "-", name.lower()): version for name, version in distributions}
    if len(installed) != len(distributions):
        raise ValueError("The image has duplicate installed distribution names.")
    # pip ships in the base image and is bootstrapped into the runtime virtualenv.
    pip_version = installed.pop("pip", "")
    components = [*sbom["components"], sbom["metadata"]["component"]]
    expected = {component["name"]: component["version"] for component in components}
    if not all(c["hashes"] for c in sbom["components"]):
        raise ValueError("The runtime lock contains a dependency without wheel hashes.")
    if installed != expected or not pip_version:
        missing = sorted(expected.keys() - installed.keys())
        unexpected = sorted(installed.keys() - expected.keys())
        mismatched = sorted(name for name in expected.keys() & installed.keys() if expected[name] != installed[name])
        raise ValueError(
            f"Image inventory differs from {lock.name}. Missing={missing}, unexpected={unexpected}, "
            f"version_mismatch={mismatched}, bootstrap_pip={pip_version or 'missing'}."
        )
    sbom["metadata"]["properties"].extend([
        {"name": "omniforge:verified_container_image_id", "value": image_id},
        {"name": "omniforge:verified_platform", "value": f"{os_name}/{architecture}"},
        {"name": "omniforge:base_image_pip_version", "value": pip_version},
    ])
    output.write_text(json.dumps(sbom, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CycloneDX 1.5 SBOM & verify lockfile SHA-256 provenance.")
    parser.add_argument("--tier", default="everything", help="Lockfile tier name in locks/ (default: everything).")
    parser.add_argument("--out", type=Path, default=None, help="Destination CycloneDX JSON path.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="Verify 100% SHA-256 hash-pinning across all locks/*.lock.")
    mode.add_argument("--image", help="Verify the installed inventory of a local Docker image against its runtime lock.")
    args = parser.parse_args()

    locks_dir = PROJECT_ROOT / "locks"
    if args.verify:
        rep = verify_lockfiles_provenance(locks_dir)
        print(json.dumps(rep, indent=2))
        return 0 if rep["verified"] else 1

    out_path = args.out or (PROJECT_ROOT / "sbom.cdx.json")
    if args.image:
        write_verified_image_sbom(args.image, out_path)
    else:
        write_cyclonedx_sbom(out_path, locks_dir=locks_dir, tier=args.tier)
    print(f"CycloneDX 1.5 SBOM written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
