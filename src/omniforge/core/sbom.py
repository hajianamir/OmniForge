"""CycloneDX 1.5 Software Bill of Materials (SBOM) & Cryptographic Lockfile Provenance (RR-022)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from omniforge.core.exceptions import OmniForgeError
from omniforge.version import __version__, get_version_info

_PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]+\])?==([^\s\\]+)")
_HASH_RE = re.compile(r"--hash=sha256:([0-9a-fA-F]{64})")
_NORM_PKG_RE = re.compile(r"[-_.]+")


def _normalize_pep503(name: str) -> str:
    return _NORM_PKG_RE.sub("-", name.strip()).lower()


@dataclass(frozen=True)
class LockedPackage:
    """Single hash-pinned dependency parsed from a compiled uv/pip-tools lockfile."""

    name: str
    version: str
    sha256_hashes: tuple[str, ...] = field(default_factory=tuple)
    via: tuple[str, ...] = field(default_factory=tuple)

    @property
    def purl(self) -> str:
        """Return canonical Package URL (purl) per https://github.com/package-url/purl-spec."""
        return f"pkg:pypi/{_normalize_pep503(self.name)}@{self.version}"


def _default_locks_dir() -> Path:
    repo_candidate = Path(__file__).resolve().parents[3] / "locks"
    if repo_candidate.is_dir():
        return repo_candidate
    return Path.cwd() / "locks"


def parse_uv_lockfile(lock_path: str | Path) -> list[LockedPackage]:
    """Parse a hash-pinned ``locks/*.lock`` file into structured ``LockedPackage`` records."""
    path = Path(lock_path)
    if not path.is_file():
        raise OmniForgeError(f"Lockfile not found: '{path}'.")

    packages: list[LockedPackage] = []
    cur_name: str | None = None
    cur_ver: str | None = None
    cur_hashes: list[str] = []
    cur_via: list[str] = []
    in_via_block = False

    def _flush() -> None:
        nonlocal cur_name, cur_ver, cur_hashes, cur_via, in_via_block
        if cur_name is not None and cur_ver is not None:
            packages.append(
                LockedPackage(
                    name=_normalize_pep503(cur_name),
                    version=cur_ver,
                    sha256_hashes=tuple(dict.fromkeys(h.lower() for h in cur_hashes)),
                    via=tuple(dict.fromkeys(cur_via)),
                )
            )
        cur_name = None
        cur_ver = None
        cur_hashes = []
        cur_via = []
        in_via_block = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            in_via_block = False
            continue
        if stripped.startswith("#"):
            comment = stripped.lstrip("#").strip()
            if comment == "via":
                in_via_block = True
            elif comment.startswith("via "):
                in_via_block = False
                parent = comment[4:].strip().split()[0]
                if not parent.startswith("-r"):
                    cur_via.append(_normalize_pep503(parent))
            elif in_via_block and comment and not comment.startswith("-r"):
                parent = comment.split()[0]
                cur_via.append(_normalize_pep503(parent))
            continue

        in_via_block = False
        pin_match = _PIN_RE.match(stripped)
        if pin_match:
            _flush()
            cur_name = pin_match.group(1)
            cur_ver = pin_match.group(2)
        for h_match in _HASH_RE.finditer(stripped):
            cur_hashes.append(h_match.group(1))

    _flush()
    packages.sort(key=lambda p: (p.name, p.version))
    return packages


def generate_cyclonedx_sbom(
    lock_path: str | Path | None = None,
    *,
    locks_dir: str | Path | None = None,
    tier: str = "everything",
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Generate a CycloneDX 1.5 JSON SBOM with SHA-256 artifact hashes and dependency graph."""
    if lock_path is not None:
        target_lock = Path(lock_path)
    else:
        base_dir = Path(locks_dir) if locks_dir is not None else _default_locks_dir()
        safe_tier = re.sub(r"[^A-Za-z0-9_-]", "", tier) or "everything"
        target_lock = base_dir / f"{safe_tier}.lock"

    lock_bytes = target_lock.read_bytes() if target_lock.is_file() else None
    if lock_bytes is None:
        raise OmniForgeError(f"Cannot generate SBOM; lockfile does not exist: '{target_lock}'.")
    lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
    locked_pkgs = parse_uv_lockfile(target_lock)

    # Deterministic build timestamp: honor SOURCE_DATE_EPOCH (reproducible builds);
    # otherwise derive a stable timestamp from the lockfile content hash so that
    # repeated generations of the same lockfile produce byte-identical SBOMs.
    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_date_epoch:
        resolved_ts = datetime.fromtimestamp(int(source_date_epoch), tz=UTC)
    else:
        resolved_ts = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(
            seconds=int(lock_sha256[:12], 16) % (365 * 10 * 24 * 3600)
        )
    resolved_timestamp = resolved_ts.isoformat()

    ver_info = get_version_info()
    root_ref = f"pkg:pypi/omniforge@{__version__}"
    # Deterministic UUIDv5 derived from root package + lockfile SHA-256 ensures reproducible builds
    det_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"https://omniforge.ai/sbom/{__version__}/{target_lock.name}/{lock_sha256}")

    purl_by_name = {pkg.name: pkg.purl for pkg in locked_pkgs}
    children_by_parent: dict[str, list[str]] = {root_ref: []}
    for pkg in locked_pkgs:
        children_by_parent.setdefault(pkg.purl, [])

    components: list[dict[str, Any]] = []
    for pkg in locked_pkgs:
        comp: dict[str, Any] = {
            "type": "library",
            "bom-ref": pkg.purl,
            "name": pkg.name,
            "version": pkg.version,
            "purl": pkg.purl,
            "hashes": [{"alg": "SHA-256", "content": h} for h in pkg.sha256_hashes],
        }
        components.append(comp)
        linked_to_parent = False
        for parent_name in pkg.via:
            parent_purl = purl_by_name.get(parent_name)
            if parent_purl and parent_purl != pkg.purl:
                children_by_parent.setdefault(parent_purl, []).append(pkg.purl)
                linked_to_parent = True
        if not linked_to_parent:
            children_by_parent[root_ref].append(pkg.purl)

    dependencies = [
        {"ref": ref, "dependsOn": sorted(set(deps))}
        for ref, deps in sorted(children_by_parent.items(), key=lambda kv: kv[0])
    ]

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{det_uuid}",
        "version": 1,
        "metadata": {
            "timestamp": timestamp or resolved_timestamp,
            "tools": [
                {
                    "vendor": "OmniForge",
                    "name": "omniforge-sbom-generator",
                    "version": __version__,
                }
            ],
            "component": {
                "type": "application",
                "bom-ref": root_ref,
                "name": "omniforge",
                "version": __version__,
                "purl": root_ref,
                "description": f"{ver_info['name']} ({ver_info['edition']})",
            },
            "properties": [
                {"name": "omniforge:lockfile:name", "value": target_lock.name},
                {"name": "omniforge:lockfile:sha256", "value": lock_sha256},
                {"name": "omniforge:component_count", "value": str(len(components))},
            ],
        },
        "components": components,
        "dependencies": dependencies,
    }


def verify_lockfiles_provenance(locks_dir: str | Path | None = None) -> dict[str, Any]:
    """Verify cryptographic SHA-256 hash-pinning completeness across all committed lockfiles."""
    base_dir = Path(locks_dir) if locks_dir is not None else _default_locks_dir()
    if not base_dir.is_dir():
        raise OmniForgeError(f"Locks directory not found: '{base_dir}'.")

    lock_files = sorted(p for p in base_dir.glob("*.lock") if not p.name.startswith("."))
    if not lock_files:
        raise OmniForgeError(f"No .lock files found in '{base_dir}'.")

    manifests: list[dict[str, Any]] = []
    unpinned_packages: list[str] = []
    for lf in lock_files:
        digest = hashlib.sha256(lf.read_bytes()).hexdigest()
        pkgs = parse_uv_lockfile(lf)
        missing = [f"{lf.name}:{p.name}=={p.version}" for p in pkgs if len(p.sha256_hashes) == 0]
        unpinned_packages.extend(missing)
        manifests.append(
            {
                "lockfile": lf.name,
                "sha256": digest,
                "package_count": len(pkgs),
                "total_wheel_hashes": sum(len(p.sha256_hashes) for p in pkgs),
                "all_packages_hash_pinned": len(missing) == 0,
            }
        )

    return {
        "verified": len(unpinned_packages) == 0 and len(manifests) > 0,
        "lockfile_count": len(manifests),
        "unpinned_packages": unpinned_packages,
        "lockfiles": manifests,
    }


def write_cyclonedx_sbom(
    output_path: str | Path,
    *,
    lock_path: str | Path | None = None,
    locks_dir: str | Path | None = None,
    tier: str = "everything",
    timestamp: str | None = None,
) -> Path:
    """Write a formatted CycloneDX 1.5 JSON SBOM to ``output_path``."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = generate_cyclonedx_sbom(lock_path, locks_dir=locks_dir, tier=tier, timestamp=timestamp)
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return out
