"""Generate or verify CycloneDX 1.5 JSON SBOMs & SHA-256 lockfile provenance (RR-022).

Usage:
    python scripts/generate_sbom.py --out sbom.cdx.json
    python scripts/generate_sbom.py --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from omniforge.core.sbom import verify_lockfiles_provenance, write_cyclonedx_sbom  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CycloneDX 1.5 SBOM & verify lockfile SHA-256 provenance.")
    parser.add_argument("--tier", default="everything", help="Lockfile tier name in locks/ (default: everything).")
    parser.add_argument("--out", type=Path, default=None, help="Destination CycloneDX JSON path.")
    parser.add_argument("--verify", action="store_true", help="Verify 100% SHA-256 hash-pinning across all locks/*.lock.")
    args = parser.parse_args()

    locks_dir = PROJECT_ROOT / "locks"
    if args.verify:
        rep = verify_lockfiles_provenance(locks_dir)
        print(json.dumps(rep, indent=2))
        return 0 if rep["verified"] else 1

    out_path = args.out or (PROJECT_ROOT / "sbom.cdx.json")
    written = write_cyclonedx_sbom(out_path, locks_dir=locks_dir, tier=args.tier)
    print(f"CycloneDX 1.5 SBOM written to {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
