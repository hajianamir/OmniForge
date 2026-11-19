"""Guards for the Inno Setup installer script.

These tests exist because ISCC only reports one error at a time and only on Windows:
a 30-second edit/compile loop turned into three separate aborted builds while authoring the
tiered installer. `scripts/check_iss.py` encodes Inno's rules so CI (and developers on any OS)
catch the whole class before invoking ISCC.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ISS_PATH = PROJECT_ROOT / "installer" / "omniforge_installer_tiers.iss"


def _load_checker():
    """Import scripts/check_iss.py without requiring it to be a package."""
    spec = importlib.util.spec_from_file_location("check_iss", PROJECT_ROOT / "scripts" / "check_iss.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    return _load_checker()


def test_identifier_rules_match_inno(checker):
    """Inno allows only alphanumerics, '_', '/' and '\\'; hyphens are rejected."""
    assert checker.check_identifier("core_runtime", "component") is None
    assert checker.check_identifier("data_io", "component") is None
    assert "may not start with a digit" in checker.check_identifier("1core", "component")
    assert "reserved" in checker.check_identifier("and", "component")
    for bad in ("core-runtime", "data-io", "api-server"):
        error = checker.check_identifier(bad, "component")
        assert error is not None and "'-'" in error, bad


def test_real_installer_has_no_identifier_errors(checker):
    """The shipped script itself must be free of identifier errors (regression guard)."""
    issues = checker.validate(ISS_PATH)
    errors = [i for i in issues if i.level == "error" and "name" in i.message]
    assert errors == [], "\n".join(str(i) for i in errors)


def test_installer_has_no_setup_directive_typos(checker):
    issues = [i for i in checker.validate(ISS_PATH) if "unknown [Setup] directive" in i.message]
    assert issues == []


def test_every_code_reference_exists(checker):
    issues = [i for i in checker.validate(ISS_PATH) if "{code:" in i.message]
    assert issues == [], "\n".join(str(i) for i in issues)


def test_optional_packs_are_all_wired_into_pascal(checker):
    """Every optional component must be readable by GetPackList, else ticking it is a no-op."""
    issues = [i for i in checker.validate(ISS_PATH) if "never referenced in [Code]" in i.message]
    assert issues == [], "\n".join(str(i) for i in issues)


def test_missing_payload_is_reported_not_crashed(checker, tmp_path):
    """A fresh clone has no dist/ payload: validation must flag it, not explode."""
    issues = checker.validate(ISS_PATH, stage=tmp_path)
    assert any("payload" in i.message for i in issues)


# ---------------------------------------------------------------------------
# Payload/installer consistency. Every assertion below corresponds to a real
# aborted ISCC run, not a hypothetical.
# ---------------------------------------------------------------------------


def _staged_payload() -> Path | None:
    """Return the staged payload if one exists (CI builds it; a fresh clone has none)."""
    staged = PROJECT_ROOT / "dist" / "omniforge-bootstrap"
    return staged if (staged / "app" / "LICENSE").exists() else None


def test_license_file_points_at_a_txt_twin(checker):
    """Regression: `Line 73: Could not read ...app\\LICENSE.txt`.

    Inno's [Setup] LicenseFile accepts only .txt/.rtf, so the payload derives LICENSE.txt
    from LICENSE. The .iss and the payload must agree on that name.
    """
    text = ISS_PATH.read_text(encoding="utf-8")
    match = __import__("re").search(r"^LicenseFile=(.+)$", text, __import__("re").MULTILINE)
    assert match, "LicenseFile directive missing"
    value = match.group(1).strip()
    assert value.lower().endswith((".txt", ".rtf")), value
    assert "LICENSE.txt" in value, value


def test_staged_payload_satisfies_every_iss_reference(checker):
    """Every Source:/LicenseFile/SetupIconFile the installer reads must exist."""
    staged = _staged_payload()
    if staged is None:
        pytest.skip("no staged payload (run scripts/build_bootstrap.py first)")
    issues = [i for i in checker.validate(ISS_PATH, stage=staged) if i.level == "error"]
    assert issues == [], "\n".join(str(i) for i in issues)
    required = [
        "app/LICENSE",
        "app/LICENSE.txt",
        "app/omniforge.py",
        "app/installer/provision.py",
        "app/packaging/assets/omniforge.ico",
        "runtime/python/python.exe",
        "runtime/python/OmniForge.exe",
        "locks/advanced.lock",
    ]
    missing = [rel for rel in required if not (staged / rel).exists()]
    assert missing == [], f"payload is missing {missing}"


def test_payload_completeness_helper_matches_reality(tmp_path):
    """``app_payload_complete`` decides whether a re-copy is needed; it must be honest."""
    module = _load_checker.__globals__  # noqa: F841  (keeps ruff quiet about unused import)
    spec = importlib.util.spec_from_file_location(
        "build_bootstrap", PROJECT_ROOT / "scripts" / "build_bootstrap.py"
    )
    assert spec and spec.loader
    bb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bb)

    assert bb.app_payload_complete(tmp_path) is False
    (tmp_path / "omniforge.py").write_text("", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("", encoding="utf-8")
    assert bb.app_payload_complete(tmp_path) is False, "LICENSE.txt twin is required"
    (tmp_path / "LICENSE.txt").write_text("", encoding="utf-8")
    (tmp_path / "installer").mkdir()
    (tmp_path / "installer" / "provision.py").write_text("", encoding="utf-8")
    (tmp_path / "omniforge").mkdir()
    (tmp_path / "omniforge" / "__init__.py").write_text("", encoding="utf-8")
    assert bb.app_payload_complete(tmp_path) is True


def test_refresh_derived_files_recreates_the_licence_twin(tmp_path):
    """`--skip-staged` must repair a stale app/ tree instead of trusting it."""
    spec = importlib.util.spec_from_file_location(
        "build_bootstrap2", PROJECT_ROOT / "scripts" / "build_bootstrap.py"
    )
    assert spec and spec.loader
    bb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bb)

    (tmp_path / "LICENSE").write_text("Apache License 2.0\n", encoding="utf-8")
    derived = bb.refresh_derived_app_files(tmp_path)
    assert "LICENSE.txt" in derived
    assert (tmp_path / "LICENSE.txt").read_text(encoding="utf-8") == (
        "Apache License 2.0\n"
    ), "the twin must be a byte copy of LICENSE"
