"""P1-13 regression tests: per-tenant pseudonymization salt from env/KMS."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import pandas as pd
import pytest

import omniforge.core.security as sec
from omniforge.core.exceptions import SecurityError
from omniforge.core.security import PIIRedactor, resolve_pseudonym_salt

SALT_A = "salt-tenant-a-" + "x" * 40
SALT_B = "salt-tenant-b-" + "y" * 40
SALT_GLOBAL = "global-salt-" + "z" * 40


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    for var in ("OMNIFORGE_PSEUDONYM_SALTS", "OMNIFORGE_PSEUDONYM_SALT", "OMNIFORGE_PSEUDONYM_SALT__TENANT_A"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("OMNIFORGE_ENVIRONMENT", raising=False)
    # Stable, isolated dev key store for tests that exercise the dev fallback.
    monkeypatch.setattr(sec, "_default_key_store_dir", lambda: tmp_path / "keys")
    return monkeypatch


def _pii_df() -> pd.DataFrame:
    return pd.DataFrame({"email": ["a@example.com", "b@example.com", "a@example.com"], "n": [1, 2, 3]})


def test_env_global_salt_deterministic_and_tenant_bound(clean_env) -> None:
    clean_env.setenv("OMNIFORGE_PSEUDONYM_SALT", SALT_GLOBAL)
    assert resolve_pseudonym_salt("tenant-a") == SALT_GLOBAL.encode()
    df1 = PIIRedactor().anonymize_dataframe(_pii_df(), columns=["email"], tenant_id="tenant-a")
    df2 = PIIRedactor().anonymize_dataframe(_pii_df(), columns=["email"], tenant_id="tenant-b")
    # Same value, different tenants -> different pseudonyms (tenant binding).
    assert list(df1["email"]) != list(df2["email"])
    # Deterministic within tenant.
    df3 = PIIRedactor().anonymize_dataframe(_pii_df(), columns=["email"], tenant_id="tenant-a")
    assert list(df1["email"]) == list(df3["email"])
    # Stable pairing: identical raw values share a pseudonym.
    assert df1["email"].iloc[0] == df1["email"].iloc[2]


def test_per_tenant_env_var_takes_precedence(clean_env) -> None:
    clean_env.setenv("OMNIFORGE_PSEUDONYM_SALT", SALT_GLOBAL)
    clean_env.setenv("OMNIFORGE_PSEUDONYM_SALT__TENANT_A", SALT_A)
    assert resolve_pseudonym_salt("tenant-a") == SALT_A.encode()
    assert resolve_pseudonym_salt("tenant-b") == SALT_GLOBAL.encode()


def test_json_map_salt_resolution(clean_env) -> None:
    clean_env.setenv("OMNIFORGE_PSEUDONYM_SALTS", json.dumps({"tenant-a": SALT_A, "tenant-b": SALT_B}))
    assert resolve_pseudonym_salt("tenant-a") == SALT_A.encode()
    assert resolve_pseudonym_salt("tenant-b") == SALT_B.encode()


def test_per_tenant_env_var_beats_json_map(clean_env) -> None:
    clean_env.setenv("OMNIFORGE_PSEUDONYM_SALTS", json.dumps({"tenant-a": SALT_B}))
    clean_env.setenv("OMNIFORGE_PSEUDONYM_SALT__TENANT_A", SALT_A)
    assert resolve_pseudonym_salt("tenant-a") == SALT_A.encode()


def test_invalid_json_map_fails_closed(clean_env) -> None:
    clean_env.setenv("OMNIFORGE_PSEUDONYM_SALTS", "{not-json")
    with pytest.raises(SecurityError):
        resolve_pseudonym_salt("tenant-a")


def test_json_map_wrong_type_fails_closed(clean_env) -> None:
    clean_env.setenv("OMNIFORGE_PSEUDONYM_SALTS", json.dumps(["not", "a", "map"]))
    with pytest.raises(SecurityError):
        resolve_pseudonym_salt("tenant-a")


def test_production_requires_salt(clean_env) -> None:
    clean_env.setenv("OMNIFORGE_ENVIRONMENT", "production")
    with pytest.raises(SecurityError, match="No pseudonymization salt"):
        resolve_pseudonym_salt("tenant-a")


def test_production_rejects_short_and_placeholder_salts(clean_env) -> None:
    clean_env.setenv("OMNIFORGE_ENVIRONMENT", "production")
    clean_env.setenv("OMNIFORGE_PSEUDONYM_SALT", "short")
    with pytest.raises(SecurityError, match="too weak"):
        resolve_pseudonym_salt("tenant-a")
    clean_env.setenv("OMNIFORGE_PSEUDONYM_SALT", "x" * 40 + "change-me")
    with pytest.raises(SecurityError, match="too weak"):
        resolve_pseudonym_salt("tenant-a")


def test_production_accepts_strong_global_salt(clean_env) -> None:
    clean_env.setenv("OMNIFORGE_ENVIRONMENT", "production")
    clean_env.setenv("OMNIFORGE_PSEUDONYM_SALT", SALT_GLOBAL)
    assert resolve_pseudonym_salt("tenant-a") == SALT_GLOBAL.encode()


def test_dev_fallback_persists_per_tenant_keyfiles(clean_env, tmp_path: Path) -> None:
    salt1 = resolve_pseudonym_salt("tenant-a")
    salt2 = resolve_pseudonym_salt("tenant-a")
    assert salt1 == salt2  # persisted, stable across calls
    other = resolve_pseudonym_salt("tenant-b")
    assert other != salt1  # per-tenant distinct
    assert (tmp_path / "keys" / "pseudonym-salt-TENANT_A.key").is_file()


def test_explicit_salt_argument_bypasses_env(clean_env) -> None:
    clean_env.setenv("OMNIFORGE_ENVIRONMENT", "production")  # would otherwise raise
    df = PIIRedactor().anonymize_dataframe(_pii_df(), columns=["email"], salt="kms-fetched-explicit", tenant_id="tenant-x")
    assert len(df) == 3


def test_pseudonym_format_and_algorithm_pinning(clean_env) -> None:
    clean_env.setenv("OMNIFORGE_PSEUDONYM_SALT", SALT_GLOBAL)
    df = PIIRedactor().anonymize_dataframe(_pii_df(), columns=["email"], tenant_id="tenant-a")
    expected = hmac.new(
        SALT_GLOBAL.encode(),
        b"omniforge:pseudonym:v1\x00tenant-a\x00a@example.com",
        hashlib.sha256,
    ).hexdigest()[:16]
    assert df["email"].iloc[0] == expected
    assert all(len(v) == 16 for v in df["email"])


def test_mask_and_drop_strategies_still_work(clean_env) -> None:
    df = _pii_df()
    masked = PIIRedactor().anonymize_dataframe(df, columns=["email"], strategy="mask", salt="s" * 20)
    assert list(masked["email"]) == ["***MASKED***"] * 3
    dropped = PIIRedactor().anonymize_dataframe(df, columns=["email"], strategy="drop", salt="s" * 20)
    assert "email" not in dropped.columns


def test_unknown_strategy_rejected(clean_env) -> None:
    with pytest.raises(ValueError, match="Unknown anonymization strategy"):
        PIIRedactor().anonymize_dataframe(_pii_df(), columns=["email"], strategy="encrypt", salt="s" * 20)


def test_mask_and_drop_work_in_production_without_salt(clean_env) -> None:
    """Salt resolution is lazy — mask/drop must not require KMS configuration."""
    clean_env.setenv("OMNIFORGE_ENVIRONMENT", "production")
    masked = PIIRedactor().anonymize_dataframe(_pii_df(), columns=["email"], strategy="mask", tenant_id="tenant-nosalt")
    assert list(masked["email"]) == ["***MASKED***"] * 3
    dropped = PIIRedactor().anonymize_dataframe(_pii_df(), columns=["email"], strategy="drop", tenant_id="tenant-nosalt")
    assert "email" not in dropped.columns


def test_hash_in_production_without_salt_fails_closed(clean_env) -> None:
    clean_env.setenv("OMNIFORGE_ENVIRONMENT", "production")
    with pytest.raises(SecurityError, match="No pseudonymization salt"):
        PIIRedactor().anonymize_dataframe(_pii_df(), columns=["email"], tenant_id="tenant-nosalt")
