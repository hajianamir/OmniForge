"""Enterprise security, encryption, JWT auth, RBAC/ABAC, PII redaction, and audit trails."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from omniforge.core.constants import RBAC_ROLE_PERMISSIONS
from omniforge.core.exceptions import AuthorizationError, ComplianceError, SecurityError
from omniforge.core.logging import get_logger, get_trace_id
from omniforge.core.types import AuditEvent, Principal

# Optional dependencies are bound through explicit ``ModuleType``-free sentinels so the
# module type-checks identically whether or not pyjwt/cryptography are installed (they are
# absent in the "simple" install tier and present in "advanced").
jwt: Any = None
try:  # pragma: no cover - depends on the install tier
    import jwt as _jwt_mod

    jwt = _jwt_mod
    _HAS_PYJWT = True
except ImportError:  # pragma: no cover
    _HAS_PYJWT = False

Fernet: Any = None
AESGCM: Any = None
InvalidToken: Any = Exception
try:  # pragma: no cover - depends on the install tier
    from cryptography.fernet import Fernet as _Fernet
    from cryptography.fernet import InvalidToken as _InvalidToken
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM as _AESGCM

    Fernet, AESGCM, InvalidToken = _Fernet, _AESGCM, _InvalidToken
    _HAS_CRYPTOGRAPHY = True
except ImportError:  # pragma: no cover
    _HAS_CRYPTOGRAPHY = False

logger = get_logger(__name__)

_AESGCM_HEADER = b"OF2GCM\x01"
_INSECURE_PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "change-me",
    "do-not-use-in-prod",
    "uap-development",
    "secret-key",
    "changeme",
)


def _is_production_env() -> bool:
    env = (os.getenv("OMNIFORGE_ENVIRONMENT") or os.getenv("ENVIRONMENT") or "development").strip().lower()
    return env in {"prod", "production"}


def _default_key_store_dir() -> Path:
    custom = os.getenv("OMNIFORGE_KEY_DIR")
    if custom:
        return Path(custom).expanduser()
    try:
        return Path.home() / ".omniforge" / "keys"
    except Exception:
        return Path(".omniforge_workspace") / "keys"


_PSEUDONYM_SALT_CONTEXT = b"omniforge:pseudonym:v1"
_PSEUDONYMIZATION_MIN_SALT_BYTES = 16
_PSEUDONYMIZATION_PROD_MIN_SALT_BYTES = 32


def _sanitize_tenant_for_env(tenant_id: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", (tenant_id or "default")).strip("_").upper()
    return out or "DEFAULT"


def _default_tenant() -> str:
    try:
        from omniforge.config.settings import get_settings

        return str(get_settings().default_tenant_id or "default-tenant")
    except Exception:
        return "default-tenant"


def _salt_strength_ok(raw: bytes, *, minimum: int) -> bool:
    if len(raw) < minimum:
        return False
    lowered = raw.decode("utf-8", errors="ignore").lower()
    return not any(marker in lowered for marker in _INSECURE_PLACEHOLDER_MARKERS)


def _dev_persisted_tenant_salt(tenant_id: str) -> bytes:
    """Non-production fallback: a per-tenant random 0600 keyfile (never a public constant)."""
    scope = _sanitize_tenant_for_env(tenant_id)
    key_dir = _default_key_store_dir()
    key_file = key_dir / f"pseudonym-salt-{scope}.key"
    try:
        if key_file.is_file():
            existing = key_file.read_text(encoding="utf-8").strip()
            if existing:
                return existing.encode("utf-8")
        key_dir.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(key_file, flags, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(secrets.token_urlsafe(48))
        return key_file.read_text(encoding="utf-8").strip().encode("utf-8")
    except OSError:
        # Read-only key store: derive an ephemeral process-local salt instead of failing.
        # Such pseudonyms are not stable across restarts — surface that honestly in the docs.
        logger.warning(
            "pseudonym_salt_ephemeral",
            tenant=tenant_id,
            note="Could not persist a dev pseudonym salt (read-only key store); "
            "pseudonyms will not be stable across process restarts. Set OMNIFORGE_PSEUDONYM_SALTS instead.",
        )
        return hashlib.sha256(_PSEUDONYM_SALT_CONTEXT + secrets.token_bytes(32)).digest()


def resolve_pseudonym_salt(tenant_id: str | None = None) -> bytes:
    """Resolve the pseudonymization salt for ``tenant_id`` from env/KMS.

    Resolution order (first non-empty wins):
      1. ``OMNIFORGE_PSEUDONYM_SALT__<SANITIZED_TENANT>`` (e.g. tenant "default-tenant" ->
         ``OMNIFORGE_PSEUDONYM_SALT__DEFAULT_TENANT``)
      2. ``OMNIFORGE_PSEUDONYM_SALTS`` — JSON object mapping tenant id -> salt
         (intended for KMS/Vault-rendered env; parse errors are fatal, never silently ignored)
      3. ``OMNIFORGE_PSEUDONYM_SALT`` — global fallback salt

    Production requires at least one of the above with a high-entropy (>= 32 bytes),
    non-placeholder salt; otherwise :class:`SecurityError` is raised (fail-closed).
    Non-production falls back to a per-tenant random 0600 keyfile (or an ephemeral
    process-local salt when the key store is not writable).
    """
    tenant = (tenant_id or _default_tenant()).strip() or "default-tenant"
    sanitized = _sanitize_tenant_for_env(tenant)

    candidate: bytes | None
    raw: str | None = os.getenv(f"OMNIFORGE_PSEUDONYM_SALT__{sanitized}")
    if raw and raw.strip():
        candidate = raw.strip().encode("utf-8")
        source = f"OMNIFORGE_PSEUDONYM_SALT__{sanitized}"
    else:
        json_env = os.getenv("OMNIFORGE_PSEUDONYM_SALTS")
        if json_env and json_env.strip():
            try:
                mapping = json.loads(json_env)
            except json.JSONDecodeError as exc:
                raise SecurityError(
                    "OMNIFORGE_PSEUDONYM_SALTS is not valid JSON; refusing to pseudonymize.",
                    suggestions=["Fix the JSON object mapping tenant ids to high-entropy salts."],
                ) from exc
            if not isinstance(mapping, dict):
                raise SecurityError(
                    "OMNIFORGE_PSEUDONYM_SALTS must be a JSON object of {tenant_id: salt}.",
                    suggestions=['Render a JSON object from your KMS/Vault, e.g. {"default-tenant": "<48 random bytes>"}.'],
                )
            raw = mapping.get(tenant)
            candidate = str(raw).strip().encode("utf-8") if raw else None
            source = "OMNIFORGE_PSEUDONYM_SALTS"
        else:
            raw = os.getenv("OMNIFORGE_PSEUDONYM_SALT")
            candidate = raw.strip().encode("utf-8") if raw and raw.strip() else None
            source = "OMNIFORGE_PSEUDONYM_SALT"

    if candidate is not None:
        minimum = _PSEUDONYMIZATION_PROD_MIN_SALT_BYTES if _is_production_env() else _PSEUDONYMIZATION_MIN_SALT_BYTES
        if not _salt_strength_ok(candidate, minimum=minimum):
            raise SecurityError(
                f"Pseudonym salt from {source} is too weak (need >= {minimum} high-entropy, non-placeholder bytes).",
                suggestions=[
                    f'Set {source} to a high-entropy random secret (e.g. `python -c \'import secrets; print(secrets.token_urlsafe(48))\'`).',
                ],
            )
        return candidate

    if _is_production_env():
        raise SecurityError(
            f"No pseudonymization salt configured for tenant '{tenant}' in production.",
            suggestions=[
                f"Set OMNIFORGE_PSEUDONYM_SALT__{sanitized} or include '{tenant}' in OMNIFORGE_PSEUDONYM_SALTS (JSON map from KMS/Vault).",
                "Without a per-tenant salt, anonymized values are dictionary-attackable; OmniForge refuses to run.",
            ],
        )
    return _dev_persisted_tenant_salt(tenant)


def pseudonymize_value(value: Any, *, salt: bytes, tenant_id: str) -> str:
    """Deterministic, tenant-bound HMAC-SHA256 pseudonym (versioned, 16 hex chars of the HMAC)."""
    material = _PSEUDONYM_SALT_CONTEXT + b"\x00" + tenant_id.encode("utf-8") + b"\x00" + str(value).encode("utf-8")
    return hmac.new(salt, material, hashlib.sha256).hexdigest()[:16]


class DataFramePIIGuard:
    """Zero-trust PII guard for DataFrames.

    ``pii_policy``:
      * ``"pseudonymize"`` — replace every PII-bearing cell with a deterministic,
        tenant-bound HMAC-SHA256 pseudonym; no raw PII survives in the returned frame.
      * ``"block"`` — fail-closed: raise :class:`SecurityError` if any PII cell is
        detected; never return raw PII.
      * ``"off"`` — no-op (backward compatibility); PII is reported but not altered.

    Detection is regex-based over object/string columns (SSN, email, phone,
    credit-card, IPv4). Audit events record only column names and cell counts —
    never raw PII values — so logs, cache, and memory never leak PII.
    """

    PII_PATTERNS: dict[str, re.Pattern[str]] = {
        "ssn": re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b"),
        "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        "phone": re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        "ip_address": re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    }

    def __init__(
        self,
        *,
        tenant_id: str = "default-tenant",
        pii_policy: str = "pseudonymize",
        audit_logger: Any | None = None,
    ) -> None:
        self.tenant_id = str(tenant_id or "default-tenant")
        self.pii_policy = str(pii_policy)
        if self.pii_policy not in ("pseudonymize", "block", "off"):
            raise SecurityError(
                f"Unknown pii_policy {self.pii_policy!r}; use 'pseudonymize', 'block', or 'off'.",
            )
        self.audit_logger = audit_logger
        # Resolve (fail-closed in production) the tenant-bound pseudonym salt.
        self._salt: bytes = resolve_pseudonym_salt(self.tenant_id)

    # ------------------------------------------------------------------ detection
    def _cell_has_pii(self, value: Any) -> bool:
        if value is None or (isinstance(value, float) and value != value):  # NaN
            return False
        s = str(value)
        return any(pattern.search(s) for pattern in self.PII_PATTERNS.values())

    def detect(self, df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Scan object/string columns for PII; return {column: {pii_cells, pattern_hits}}.

        No raw PII values appear in the report — only counts per column/pattern.
        """
        report: dict[str, dict[str, Any]] = {}
        for col in df.columns:
            series = df[col]
            if not pd.api.types.is_string_dtype(series):
                continue
            pii_cells = 0
            pattern_hits: dict[str, int] = {}
            for val in series.dropna():
                if not self._cell_has_pii(val):
                    continue
                pii_cells += 1
                s = str(val)
                for name, pattern in self.PII_PATTERNS.items():
                    if pattern.search(s):
                        pattern_hits[name] = pattern_hits.get(name, 0) + 1
            if pii_cells:
                report[str(col)] = {"pii_cells": int(pii_cells), "pattern_hits": pattern_hits}
        return report

    def has_pii(self, df: pd.DataFrame) -> bool:
        return bool(self.detect(df))

    # ------------------------------------------------------------------ transform / enforce
    def transform(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Return ``(redacted_df, report)`` without leaking raw PII.

        * ``pseudonymize``: every PII-bearing cell becomes a deterministic pseudonym.
        * ``block``: raises :class:`SecurityError` if any PII cell exists (fail-closed).
        * ``off``: returns the frame unchanged (PII still reported).
        """
        redacted = df.copy(deep=True)
        report = self.detect(redacted)
        summary: dict[str, Any] = {
            "pii_policy": self.pii_policy,
            "tenant_id": self.tenant_id,
            "pii_detected": bool(report),
            "pii_columns": sorted(report),
            "pii_cells_total": sum(int(v["pii_cells"]) for v in report.values()),
            "pii_report": report,
        }

        if self.pii_policy == "block":
            if report:
                cols = ", ".join(sorted(report))
                total = summary["pii_cells_total"]
                self._audit(
                    "pii_block_rejected",
                    details={"columns": sorted(report), "pii_cells_total": int(total)},
                )
                raise SecurityError(
                    f"pii_policy='block' refused a frame containing PII in {len(report)} "
                    f"column(s) ({total} PII-bearing cell(s)): {cols}. No raw PII is returned.",
                )
            self._audit("pii_scan_clean", details={"columns": list(redacted.columns)})
            return redacted, summary

        if self.pii_policy == "pseudonymize":
            redacted_columns: dict[str, int] = {}
            for col, _info in report.items():
                mask = redacted[col].apply(self._cell_has_pii)
                n = int(mask.sum())
                if n:
                    redacted.loc[mask, col] = redacted.loc[mask, col].apply(
                        lambda v: pseudonymize_value(
                            str(v), salt=self._salt, tenant_id=self.tenant_id
                        )
                    )
                redacted_columns[str(col)] = n
            summary["redacted_columns"] = redacted_columns
            self._audit(
                "pii_redacted",
                details={
                    "columns": list(redacted_columns),
                    "pii_cells_total": int(summary["pii_cells_total"]),
                },
            )
            return redacted, summary

        # pii_policy == "off": no-op, but still report for transparency.
        self._audit("pii_scan_off", details={"columns": list(redacted.columns), "pii_report": report})
        return redacted, summary

    def enforce(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: return the guarded frame, raising under ``block`` on any PII.

        Discards the report; use :meth:`transform` when the report is needed.
        """
        redacted, _ = self.transform(df)
        return redacted

    # ------------------------------------------------------------------ audit (no raw PII)
    def _audit(self, action: str, *, details: dict[str, Any] | None = None) -> None:
        if self.audit_logger is None:
            return
        try:
            self.audit_logger.log_event(
                action=action,
                resource_type="dataframe",
                resource_id=self.tenant_id,
                tenant_id=self.tenant_id,
                actor="system",
                role="admin",
                status="success",
                details=details or {},
            )
        except Exception:
            # Audit must never break the data path, but must also never leak PII.
            pass


def _resolve_or_persist_secret(
    *,
    key_name: str,
    explicit: str | bytes | None,
    env_vars: tuple[str, ...],
    min_length: int = 32,
) -> bytes:
    """Resolve cryptographic secret material from explicit input, env vars, or a 0600 local keyfile.

    In production (``OMNIFORGE_ENVIRONMENT=production``), weak/placeholder or missing secrets raise
    :class:`SecurityError` immediately instead of silently generating a fallback secret.
    """
    raw: str | bytes | None = explicit
    if raw is None or (isinstance(raw, str) and not raw.strip()) or (isinstance(raw, bytes) and not raw):
        for var in env_vars:
            val = os.getenv(var)
            if val and val.strip():
                raw = val.strip()
                break

    if raw is not None and raw != "" and raw != b"":
        raw_bytes = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        if _is_production_env():
            lowered = raw_bytes.decode("utf-8", errors="ignore").lower()
            if len(raw_bytes) < min_length or any(m in lowered for m in _INSECURE_PLACEHOLDER_MARKERS):
                raise SecurityError(
                    f"Weak or placeholder cryptographic secret rejected for '{key_name}' in production.",
                    suggestions=[
                        f"Set {env_vars[0]} to a high-entropy random secret (>= {min_length} bytes).",
                    ],
                )
        return raw_bytes

    if _is_production_env():
        raise SecurityError(
            f"Required cryptographic secret '{key_name}' is not configured in production.",
            suggestions=[
                f"Provide a high-entropy secret via {env_vars[0]} before starting OmniForge in production.",
            ],
        )

    # Local/development/test persistence: generate once with 0600 permissions so keys survive restarts
    # and remain consistent across independent manager instances (fixes R12 & S4).
    key_dir = _default_key_store_dir()
    key_file = key_dir / f"{key_name}.key"
    try:
        if key_file.is_file():
            existing = key_file.read_text(encoding="utf-8").strip()
            if existing:
                return existing.encode("utf-8")
        key_dir.mkdir(parents=True, exist_ok=True)
        generated = secrets.token_urlsafe(48)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            fd = os.open(str(key_file), flags, stat.S_IRUSR | stat.S_IWUSR)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(generated)
        except FileExistsError:
            existing = key_file.read_text(encoding="utf-8").strip()
            if existing:
                return existing.encode("utf-8")
            key_file.write_text(generated, encoding="utf-8")
        try:
            os.chmod(key_file, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        return generated.encode("utf-8")
    except OSError:
        # Fallback if filesystem is strictly read-only: keep process-wide stable secret in env.
        generated = secrets.token_urlsafe(48)
        os.environ[env_vars[0]] = generated
        return generated.encode("utf-8")

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

_PII_COLUMN_KEYWORDS: tuple[str, ...] = (
    "email",
    "phone",
    "mobile",
    "ssn",
    "social_security",
    "passport",
    "iban",
    "credit_card",
    "card_number",
    "patient_name",
    "full_name",
    "first_name",
    "last_name",
    "dob",
    "date_of_birth",
    "address",
    "mrn",
)


class EncryptionManager:
    """Authenticated symmetric encryption at rest using NIST AES-256-GCM.

    Supports backward-compatible decryption of legacy Fernet tokens while encrypting all new
    payloads with AES-256-GCM (256-bit key, unique random 96-bit nonce per encryption, 128-bit
    authentication tag). Keys persist via ``OMNIFORGE_ENCRYPTION_KEY`` or a local ``0600`` keyfile
    so independent manager instances can consistently encrypt and decrypt across restarts.
    """

    def __init__(self, key: str | bytes | None = None) -> None:
        if not _HAS_CRYPTOGRAPHY or AESGCM is None:
            raise SecurityError(
                "The 'cryptography' library is required for AES-256-GCM encryption.",
                suggestions=["Install cryptography>=50.0.2 in the active Python environment."],
            )
        raw_material = _resolve_or_persist_secret(
            key_name="encryption",
            explicit=key,
            env_vars=("OMNIFORGE_ENCRYPTION_KEY", "OMNIFORGE_SECURITY__ENCRYPTION_KEY"),
            min_length=32,
        )
        # Derive a true 256-bit (32-byte) AES-GCM key cleanly from raw/base64 key material
        self._aes_key = self._derive_aes256_key(raw_material)
        self._aesgcm = AESGCM(self._aes_key)
        # Also keep optional legacy Fernet decoder for reading older Fernet-encrypted payloads
        self._fernet_key = self._derive_fernet_key(raw_material)
        self._fernet = Fernet(self._fernet_key) if Fernet is not None else None

    @staticmethod
    def _derive_aes256_key(raw_material: bytes) -> bytes:
        # If caller passed exact 32-byte raw bytes or a 32-byte urlsafe base64 key, use it directly;
        # otherwise deterministically derive a 256-bit key with domain-separated SHA-256.
        if len(raw_material) == 32:
            return raw_material
        try:
            decoded = base64.urlsafe_b64decode(raw_material)
            if len(decoded) == 32:
                return decoded
        except Exception:
            pass
        return hashlib.sha256(b"omniforge:aes256gcm:v1:" + raw_material).digest()

    @staticmethod
    def _derive_fernet_key(raw_material: bytes) -> bytes:
        if Fernet is not None:
            try:
                Fernet(raw_material)
                return raw_material
            except Exception:
                pass
        return base64.urlsafe_b64encode(hashlib.sha256(raw_material).digest())

    @property
    def algorithm(self) -> str:
        """Return active authenticated cipher suite name."""
        return "AES-256-GCM"

    @property
    def key_b64(self) -> str:
        """Return urlsafe base64-encoded 256-bit AES-GCM key material."""
        return base64.urlsafe_b64encode(self._aes_key).decode("ascii")

    def encrypt_bytes(self, plaintext: bytes, *, associated_data: bytes | None = None) -> bytes:
        """Encrypt raw bytes into an authenticated AES-256-GCM ciphertext envelope."""
        nonce = os.urandom(12)  # Fresh 96-bit NIST-recommended random nonce per encryption
        ciphertext = self._aesgcm.encrypt(nonce, plaintext, associated_data)
        return base64.urlsafe_b64encode(_AESGCM_HEADER + nonce + ciphertext)

    def decrypt_bytes(self, token: bytes, *, associated_data: bytes | None = None) -> bytes:
        """Decrypt and authenticate AES-256-GCM ciphertext (with legacy Fernet fallback)."""
        try:
            # Backward-compatible decryption for legacy Fernet tokens
            if token.startswith(b"gAAAAA") and self._fernet is not None and associated_data is None:
                return bytes(self._fernet.decrypt(token))
            raw = base64.urlsafe_b64decode(token)
            if not raw.startswith(_AESGCM_HEADER) or len(raw) < len(_AESGCM_HEADER) + 12 + 16:
                raise SecurityError("Unrecognized or truncated ciphertext envelope.")
            offset = len(_AESGCM_HEADER)
            nonce = raw[offset : offset + 12]
            ciphertext = raw[offset + 12 :]
            return bytes(self._aesgcm.decrypt(nonce, ciphertext, associated_data))
        except SecurityError:
            raise
        except Exception as exc:
            raise SecurityError(
                "Failed to decrypt payload. Verify encryption key and payload integrity.",
                suggestions=["Check OMNIFORGE_ENCRYPTION_KEY configuration."],
                cause=exc,
            ) from exc

    def encrypt_text(self, plaintext: str, *, associated_data: bytes | None = None) -> str:
        """Encrypt UTF-8 string into an ASCII-safe AES-256-GCM token."""
        return self.encrypt_bytes(plaintext.encode("utf-8"), associated_data=associated_data).decode("ascii")

    def decrypt_text(self, token: str, *, associated_data: bytes | None = None) -> str:
        """Decrypt ASCII-safe token back into UTF-8 string."""
        return self.decrypt_bytes(token.encode("ascii"), associated_data=associated_data).decode("utf-8")


class ArtifactSigner:
    """HMAC-SHA256 cryptographic signer/verifier for OmniForge workspace, model, and pipeline artifacts."""

    def __init__(self, secret_key: str | bytes | None = None) -> None:
        raw = _resolve_or_persist_secret(
            key_name="artifact_signing",
            explicit=secret_key,
            env_vars=(
                "OMNIFORGE_ARTIFACT_SIGNING_KEY",
                "OMNIFORGE_SECURITY__ARTIFACT_SIGNING_KEY",
                "OMNIFORGE_JWT_SECRET_KEY",
            ),
            min_length=32,
        )
        self._key = hashlib.sha256(b"omniforge:artifact-hmac-sha256:v1:" + raw).digest()

    def sign_bytes(self, payload: bytes, *, context: bytes = b"omniforge-artifact") -> str:
        """Return a hex-encoded HMAC-SHA256 signature bound to ``context`` and ``payload``."""
        mac = hmac.new(self._key, context + b"\x00" + payload, hashlib.sha256)
        return mac.hexdigest()

    def verify_bytes(
        self,
        payload: bytes,
        signature_hex: str,
        *,
        context: bytes = b"omniforge-artifact",
    ) -> bool:
        """Verify ``signature_hex`` in constant time using ``hmac.compare_digest``."""
        if not signature_hex or not isinstance(signature_hex, str):
            return False
        expected = self.sign_bytes(payload, context=context)
        return hmac.compare_digest(expected, signature_hex.strip().lower())


_BLOCKED_UNPICKLE_ROOT_MODULES: frozenset[str] = frozenset(
    {
        "os",
        "posix",
        "nt",
        "subprocess",
        "sys",
        "importlib",
        "runpy",
        "pty",
        "shutil",
        "socket",
        "http",
        "urllib",
        "ftplib",
        "smtplib",
        "ctypes",
        "cffi",
        "marshal",
        "code",
        "codeop",
        "webbrowser",
        "pdb",
        "bdb",
        "signal",
        "multiprocessing",
        "threading",
        "concurrent",
        "ssl",
        "xmlrpc",
        "pickle",
        "_pickle",
        "dill",
        "cloudpickle",
        "operator",
        "asyncio",
        "inspect",
        "gc",
        "tempfile",
        "glob",
    }
)

_ALLOWED_UNPICKLE_BUILTINS: frozenset[str] = frozenset(
    {
        "dict",
        "list",
        "tuple",
        "set",
        "frozenset",
        "str",
        "bytes",
        "bytearray",
        "int",
        "float",
        "bool",
        "complex",
        "slice",
        "range",
        "object",
    }
)

_ALLOWED_UNPICKLE_MODULE_PREFIXES: tuple[str, ...] = (
    "omniforge",
    "test_third_party_claims",
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "sklearn",
    "statsmodels",
    "networkx",
    "xgboost",
    "lightgbm",
    "catboost",
    "torch",
    "pydantic",
    "joblib",
    "plotly",
    "collections",
    "datetime",
    "decimal",
    "fractions",
    "uuid",
    "enum",
    "re",
    "pathlib",
    "copyreg",
    "_codecs",
    "codecs",
    "tzdata",
    "zoneinfo",
    "pytz",
    "dateutil",
    "dataclasses",
    "typing",
    "functools",
)

_BLOCKED_UNPICKLE_SYMBOL_NAMES: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "breakpoint",
        "input",
        "system",
        "popen",
        "spawn",
        "call",
        "check_output",
        "check_call",
        "Popen",
        "unlink",
        "rmdir",
        "remove",
        "rmtree",
    }
)


# Some scikit-learn wheels serialize Cython loss classes under the bare
# ``_loss`` alias rather than ``sklearn._loss._loss``. Resolve only these known
# globals to the canonical, already-allowed sklearn module; never allow the
# arbitrary top-level ``_loss`` module to supply executable objects.
_SKLEARN_CY_LOSS_GLOBALS: frozenset[str] = frozenset(
    {
        "CyAbsoluteError",
        "CyExponentialLoss",
        "CyHalfBinomialLoss",
        "CyHalfGammaLoss",
        "CyHalfMultinomialLoss",
        "CyHalfPoissonLoss",
        "CyHalfSquaredError",
        "CyHalfTweedieLoss",
        "CyHalfTweedieLossIdentity",
        "CyHuberLoss",
        "CyPinballLoss",
        "__pyx_unpickle_CyHalfMultinomialLoss",
    }
)

_BLOCKED_UNPICKLE_SUBMODULE_PREFIXES: tuple[str, ...] = (
    "torch.hub",
    "torch.serialization",
    "torch.jit",
    "torch.utils.cpp_extension",
    "joblib.parallel",
    "joblib.externals",
    "pandas.io",
    "pyarrow.fs",
    "numpy.testing",
    "numpy.f2py",
    "omniforge.data.connectors",
    "omniforge.api",
    "omniforge.cli",
    "omniforge.launcher",
)


def _validate_unpickle_target(module: str, name: str) -> None:
    """Enforce defense-in-depth allowlist/denylist before resolving any pickled class/global."""
    root_mod = module.split(".", 1)[0]
    if root_mod in _BLOCKED_UNPICKLE_ROOT_MODULES:
        raise SecurityError(
            f"Blocked unsafe deserialization target '{module}.{name}' (forbidden module '{root_mod}')."
        )
    if any(
        module == prefix or module.startswith(f"{prefix}.")
        for prefix in _BLOCKED_UNPICKLE_SUBMODULE_PREFIXES
    ):
        raise SecurityError(
            f"Blocked unsafe deserialization submodule '{module}.{name}'."
        )
    if name in _BLOCKED_UNPICKLE_SYMBOL_NAMES:
        raise SecurityError(
            f"Blocked unsafe deserialization symbol '{module}.{name}'."
        )
    if module == "builtins":
        if name not in _ALLOWED_UNPICKLE_BUILTINS:
            raise SecurityError(
                f"Blocked unsafe builtin deserialization target 'builtins.{name}'."
            )
        return
    if module == "functools" and name != "partial":
        raise SecurityError(f"Blocked unsafe functools target 'functools.{name}'.")
    if root_mod not in _ALLOWED_UNPICKLE_MODULE_PREFIXES:
        raise SecurityError(
            f"Blocked unrecognized deserialization module '{module}.{name}'."
        )


def safe_joblib_load_bytes(payload_bytes: bytes, *, filename: str = "") -> Any:
    """Deserialize joblib/pickle bytes using a restricted ``NumpyUnpickler`` allowlist.

    Always call AFTER cryptographic HMAC-SHA256 verification (:class:`ArtifactSigner`) so untrusted
    bytes are rejected before reaching the unpickler, and dangerous globals remain blocked even as
    a second layer of defense.
    """
    import io

    from joblib.numpy_pickle import NumpyUnpickler, _validate_fileobject_and_memmap

    class _RestrictedNumpyUnpickler(NumpyUnpickler):
        def find_class(self, module: str, name: str) -> Any:
            if module == "_loss" and name in _SKLEARN_CY_LOSS_GLOBALS:
                module = "sklearn._loss._loss"
            _validate_unpickle_target(module, name)
            return super().find_class(module, name)

    buf = io.BytesIO(payload_bytes)
    with _validate_fileobject_and_memmap(buf, filename, mmap_mode=None) as (fobj, _):
        unpickler = _RestrictedNumpyUnpickler(
            filename, fobj, ensure_native_byte_order=True, mmap_mode=None
        )
        return unpickler.load()


class JWTManager:
    """JWT issuance and verification manager supporting PyJWT and stdlib HS256 fallback."""

    def __init__(
        self,
        secret_key: str | None = None,
        algorithm: str = "HS256",
        expiration_minutes: int = 60,
    ) -> None:
        secret_bytes = _resolve_or_persist_secret(
            key_name="jwt_secret",
            explicit=secret_key,
            env_vars=("OMNIFORGE_JWT_SECRET_KEY", "OMNIFORGE_SECURITY__JWT_SECRET_KEY"),
            min_length=32,
        )
        self._secret_key = secret_bytes.decode("utf-8", errors="replace")
        self._algorithm = algorithm
        self._expiration_minutes = expiration_minutes

    def create_token(
        self,
        principal: Principal,
        expires_minutes: int | None = None,
    ) -> str:
        """Mint a signed JWT for ``principal``."""
        now = datetime.now(UTC)
        exp = now + timedelta(minutes=expires_minutes or self._expiration_minutes)
        payload: dict[str, Any] = {
            "sub": principal.subject,
            "roles": principal.roles,
            "tenant_id": principal.tenant_id,
            "attributes": principal.attributes,
            "scopes": principal.scopes,
            "iat": int(now.timestamp()),
            "exp": int(exp.timestamp()),
        }
        if _HAS_PYJWT and jwt is not None:
            return str(jwt.encode(payload, self._secret_key, algorithm=self._algorithm))
        header_b64 = self._b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode("utf-8"))
        payload_b64 = self._b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        sig = hmac.new(self._secret_key.encode("utf-8"), signing_input, hashlib.sha256).digest()
        return f"{header_b64}.{payload_b64}.{self._b64url(sig)}"

    def verify_token(self, token: str) -> Principal:
        """Verify JWT signature and expiration and reconstruct a :class:`Principal`."""
        try:
            if _HAS_PYJWT and jwt is not None:
                payload = jwt.decode(token, self._secret_key, algorithms=[self._algorithm])
            else:
                parts = token.split(".")
                if len(parts) != 3:
                    raise ValueError("Malformed JWT token.")
                signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
                expected_sig = self._b64url(
                    hmac.new(
                        self._secret_key.encode("utf-8"), signing_input, hashlib.sha256
                    ).digest()
                )
                if not hmac.compare_digest(parts[2], expected_sig):
                    raise ValueError("Invalid JWT signature.")
                payload = json.loads(self._b64url_decode(parts[1]).decode("utf-8"))
                if int(payload.get("exp", 0)) < int(datetime.now(UTC).timestamp()):
                    raise ValueError("JWT token has expired.")

            return Principal(
                subject=str(payload["sub"]),
                roles=list(payload.get("roles", ["viewer"])),
                tenant_id=str(payload.get("tenant_id", "default-tenant")),
                attributes=dict(payload.get("attributes", {})),
                scopes=list(payload.get("scopes", [])),
            )
        except Exception as exc:
            raise SecurityError(
                "Authentication token verification failed.",
                suggestions=["Obtain a fresh JWT token via /api/v1/auth/token."],
                cause=exc,
            ) from exc

    @staticmethod
    def _b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    @staticmethod
    def _b64url_decode(text: str) -> bytes:
        padding = "=" * (-len(text) % 4)
        return base64.urlsafe_b64decode(text + padding)


class AccessController:
    """Combined RBAC and ABAC policy enforcement engine with multi-tenant isolation.

    Cross-tenant authorization is restricted to the ``platform_admin`` role (P1-12) and, when
    constructed with ``audit_logger=``, EVERY cross-tenant decision (allow or deny) is written to
    the tamper-evident audit trail — fail-closed: an allow decision is refused if the audit write
    fails. Tenant isolation failures raise before any permission check.
    """

    def __init__(
        self,
        role_permissions: dict[str, frozenset[str]] | None = None,
        enforce_tenant_isolation: bool = True,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self._role_permissions = dict(role_permissions or RBAC_ROLE_PERMISSIONS)
        self._enforce_tenant_isolation = enforce_tenant_isolation
        self._audit_logger = audit_logger

    def _audit_cross_tenant(
        self,
        principal: Principal,
        permission: str,
        resource_tenant_id: str,
        *,
        status: str,
    ) -> None:
        if self._audit_logger is None:
            return
        self._audit_logger.log_event(
            action="authorize:cross_tenant",
            resource_type="tenant",
            resource_id=str(resource_tenant_id),
            actor=principal.subject,
            role=",".join(principal.roles),
            tenant_id=principal.tenant_id,
            status=status,
            details={
                "permission": permission,
                "principal_tenant": principal.tenant_id,
                "resource_tenant": resource_tenant_id,
            },
        )

    def get_permissions(self, principal: Principal) -> set[str]:
        """Compute effective permission set for a principal."""
        perms: set[str] = set(principal.scopes)
        for role in principal.roles:
            perms.update(self._role_permissions.get(role, frozenset()))
        return perms

    def authorize(
        self,
        principal: Principal,
        permission: str,
        *,
        resource_tenant_id: str | None = None,
        required_attributes: dict[str, Any] | None = None,
    ) -> bool:
        """Enforce RBAC permission, tenant boundary, and ABAC attribute constraints.

        Raises:
            AuthorizationError: If any authorization check fails.
        """
        if self._enforce_tenant_isolation and resource_tenant_id is not None:
            cross_tenant = principal.tenant_id != resource_tenant_id
            if cross_tenant and "platform_admin" not in principal.roles:
                self._audit_cross_tenant(principal, permission, str(resource_tenant_id), status="denied")
                raise AuthorizationError(
                    f"Cross-tenant access denied for principal '{principal.subject}'.",
                    context={
                        "principal_tenant": principal.tenant_id,
                        "resource_tenant": resource_tenant_id,
                    },
                    suggestions=["Authenticate within the target tenant context."],
                )
            if cross_tenant:
                try:
                    self._audit_cross_tenant(principal, permission, str(resource_tenant_id), status="success")
                except OSError as exc:
                    raise SecurityError(
                        "Cross-tenant authorization refused: the mandatory audit record could not be written.",
                        suggestions=[
                            "Check audit log writability/disk space; cross-tenant access is fail-closed by design.",
                        ],
                    ) from exc

        effective_permissions = self.get_permissions(principal)
        if permission not in effective_permissions:
            raise AuthorizationError(
                f"Principal '{principal.subject}' lacks required permission '{permission}'.",
                context={"roles": principal.roles, "required_permission": permission},
                suggestions=["Request elevated role assignment from a platform administrator."],
            )

        if required_attributes:
            for attr_key, expected_val in required_attributes.items():
                if principal.attributes.get(attr_key) != expected_val:
                    raise AuthorizationError(
                        f"ABAC policy check failed for attribute '{attr_key}'.",
                        context={"attribute": attr_key},
                    )
        return True


class SecretsManager:
    """Secure secrets retrieval with environment and encrypted in-memory store support."""

    def __init__(self, encryption: EncryptionManager | None = None) -> None:
        self._encryption = encryption or EncryptionManager()
        self._store: dict[str, str] = {}

    def set_secret(self, key: str, value: str) -> None:
        """Store an encrypted secret in memory."""
        self._store[key] = self._encryption.encrypt_text(value)

    def get_secret(self, key: str, default: str | None = None) -> str | None:
        """Retrieve a secret from encrypted store or environment variables."""
        if key in self._store:
            return self._encryption.decrypt_text(self._store[key])
        return os.getenv(key, default)

    def delete_secret(self, key: str) -> None:
        """Remove a stored secret if present."""
        self._store.pop(key, None)


class PIIRedactor:
    """Detects and redacts personally identifiable information (PII) in text and DataFrames."""

    def detect_text_pii(self, text: str) -> dict[str, list[str]]:
        """Return PII matches keyed by category found in ``text``."""
        findings: dict[str, list[str]] = {}
        for category, pattern in _PII_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                findings[category] = matches
        return findings

    def redact_text(self, text: str, replacement: str = "[REDACTED]") -> str:
        """Replace all recognized PII patterns in ``text``."""
        redacted = text
        for category, pattern in _PII_PATTERNS.items():
            redacted = pattern.sub(f"{replacement}:{category.upper()}", redacted)
        return redacted

    def scan_dataframe(self, df: pd.DataFrame, sample_size: int = 200) -> dict[str, list[str]]:
        """Scan DataFrame columns and values for potential PII indicators."""
        flagged: dict[str, list[str]] = {}
        sample_df = df.head(sample_size)

        for col in df.columns:
            reasons: list[str] = []
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in _PII_COLUMN_KEYWORDS):
                reasons.append("column_name_indicator")

            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                text_blob = "\n".join(sample_df[col].dropna().astype(str).tolist())
                pattern_hits = self.detect_text_pii(text_blob)
                for cat in pattern_hits:
                    reasons.append(f"value_pattern:{cat}")

            if reasons:
                flagged[str(col)] = sorted(set(reasons))
        return flagged

    def anonymize_dataframe(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        strategy: str = "hash",
        salt: str | bytes | None = None,
        tenant_id: str | None = None,
    ) -> pd.DataFrame:
        """Anonymize specified (or automatically detected) PII columns in a DataFrame.

        Args:
            df: Input DataFrame.
            columns: Columns to anonymize; auto-detected if ``None``.
            strategy: ``"hash"`` (tenant-bound HMAC-SHA256 pseudonym), ``"mask"``, or ``"drop"``.
            salt: Explicit KMS-fetched salt (bytes or string). ``None`` resolves from
                ``OMNIFORGE_PSEUDONYM_SALT__<TENANT>`` / ``OMNIFORGE_PSEUDONYM_SALTS`` (JSON map) /
                ``OMNIFORGE_PSEUDONYM_SALT`` via :func:`resolve_pseudonym_salt`. In production at
                least one of those MUST be configured — there is no public default.
            tenant_id: Tenant scope for pseudonymization and salt resolution
                (defaults to ``settings.default_tenant_id``).

        Hash pseudonyms are deterministic per (tenant, value) pair via HMAC-SHA256 with the
        versioned domain separator ``omniforge:pseudonym:v1``. The same value pseudonymizes
        identically within a tenant and differently across tenants. Salt resolution is lazy:
        ``mask``/``drop`` strategies need no secret and never trigger salt resolution.
        """
        tenant = (tenant_id or _default_tenant()).strip() or "default-tenant"
        salt_bytes: bytes | None = None
        if strategy == "hash":
            if salt is not None:
                salt_bytes = salt.encode("utf-8") if isinstance(salt, str) else bytes(salt)
            else:
                salt_bytes = resolve_pseudonym_salt(tenant)
        target_cols = columns if columns is not None else list(self.scan_dataframe(df).keys())
        result = df.copy()
        for col in target_cols:
            if col not in result.columns:
                continue
            if strategy == "drop":
                result = result.drop(columns=[col])
            elif strategy == "mask":
                result[col] = result[col].apply(
                    lambda v: "***MASKED***" if pd.notna(v) else v
                )
            elif strategy == "hash":
                assert salt_bytes is not None  # guaranteed: resolved above when strategy == "hash"
                hash_salt = salt_bytes
                result[col] = result[col].apply(
                    lambda v, salt=hash_salt: pseudonymize_value(v, salt=salt, tenant_id=tenant)
                    if pd.notna(v)
                    else v
                )
            else:
                raise ValueError(f"Unknown anonymization strategy: {strategy!r} (expected 'hash', 'mask', or 'drop')")
        return result


# ---------------------------------------------------------------------------
# Pure-ASGI hardening middleware: request body size limit + sliding-window rate limit
# ---------------------------------------------------------------------------
Scope = dict[str, Any]
Receive = Any
Send = Any
ASGIApp = Any


class _PayloadTooLarge(BaseException):
    """Internal signal used to abort an in-flight oversized request body.

    Deliberately derives from ``BaseException`` (not ``Exception``) so Starlette's
    exception handlers and application-level ``except Exception`` blocks do not
    swallow it and turn a 413 into a 500.
    """


async def _send_simple_json(send: Send, status: int, payload: dict[str, Any], extra_headers: list[tuple[bytes, bytes]] | None = None) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    if extra_headers:
        headers.extend(extra_headers)
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class RequestBodySizeLimitMiddleware:
    """Reject HTTP requests whose body exceeds ``max_body_bytes`` with HTTP 413.

    Two layers of enforcement:
      1. A ``Content-Length`` pre-check rejects oversized requests before any byte
         of the body is consumed (covers the common case, zero-copy rejection).
      2. Streaming enforcement counts bytes across ``http.request`` messages, so
         chunked-transfer bodies without ``Content-Length`` are bounded as well.
    """

    def __init__(self, app: ASGIApp, max_body_bytes: int = 268_435_456) -> None:
        self.app = app
        self.max_body_bytes = int(max_body_bytes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        limit = self.max_body_bytes
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = -1
                if declared > limit:
                    await _send_simple_json(
                        send,
                        413,
                        {
                            "detail": f"Request body exceeds the {limit // (1024 * 1024)} MB limit.",
                            "error_type": "PayloadTooLarge",
                        },
                    )
                    return
        received = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message: dict[str, Any] = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _PayloadTooLarge()
            return message

        response_started = False

        async def wrapped_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, wrapped_send)
        except _PayloadTooLarge:
            if not response_started:
                await _send_simple_json(
                    send,
                    413,
                    {
                        "detail": f"Request body exceeds the {limit // (1024 * 1024)} MB limit.",
                        "error_type": "PayloadTooLarge",
                    },
                )
            # else: the downstream app already started responding; there is nothing
            # safe left to send — the connection is torn down with the client informed
            # via the truncated response.


class SlidingWindowRateLimitMiddleware:
    """Per-client-IP sliding-window rate limiter (HTTP 429 + ``Retry-After``).

    Implements the ``api.rate_limit_per_minute`` policy declared in configuration.
    Keyed by the ASGI client host (NOT ``X-Forwarded-For``, which is spoofable);
    deployments sitting behind a single trusted proxy share one bucket unless the
    limit is raised via ``OMNIFORGE_API__RATE_LIMIT_PER_MINUTE`` (0 disables).
    Health probe paths are exempt so load balancers are never throttled.

    The backend is pluggable via ``omniforge.core.rate_limit``: the default is a
    per-process in-memory window; pass ``limiter=`` (e.g. a ``RedisSlidingWindowLimiter``
    or ``ResilientRateLimiter``) to enforce one shared window across processes.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_requests_per_minute: int = 300,
        window_seconds: float = 60.0,
        exclude_paths: frozenset[str] = frozenset({"/health", "/api/v1/health"}),
        clock: Any = None,
        limiter: Any = None,
    ) -> None:
        from omniforge.core.rate_limit import InMemorySlidingWindowLimiter

        self.app = app
        self.max_rpm = int(max_requests_per_minute)
        self.window_seconds = float(window_seconds)
        self.exclude_paths = frozenset(exclude_paths)
        self._clock = clock or time.monotonic
        self._limiter = limiter or InMemorySlidingWindowLimiter(
            max_requests=self.max_rpm, window_seconds=self.window_seconds, clock=self._clock
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http" or self.max_rpm <= 0:
            await self.app(scope, receive, send)
            return
        if scope.get("path", "") in self.exclude_paths:
            await self.app(scope, receive, send)
            return
        client = scope.get("client") or ("unknown", 0)
        key = str(client[0])
        decision = await self._limiter.hit(key)
        if not decision["allowed"]:
            await _send_simple_json(
                send,
                429,
                {
                    "detail": "Rate limit exceeded. Slow down and retry shortly.",
                    "error_type": "TooManyRequests",
                },
                extra_headers=[(b"retry-after", str(decision["retry_after"]).encode("ascii"))],
            )
            return
        await self.app(scope, receive, send)


_AUDIT_HMAC_CONTEXT = b"omniforge-audit-event-v2"

_CHECKPOINT_HMAC_CONTEXT = b"omniforge-audit-checkpoint-v1"


def _checkpoint_disabled(value: str | None) -> bool:
    return not value or value.strip().lower() in {"0", "false", "off", "none"}


def _verify_anchor_records(
    hashes: list[str],
    checkpoint_lines: list[str],
    verify_mac: Callable[[dict[str, Any]], bool],
) -> tuple[bool, int, str]:
    """Shared consistency check for anchor checkpoints against the parsed log hashes."""
    checked = 0
    last_seq = 0
    max_covered = 0
    for line in checkpoint_lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return False, checked, "anchor contains unparseable line"
        if not verify_mac(record):
            return False, checked, f"checkpoint #{record.get('seq')} fails HMAC verification"
        seq = int(record.get("seq") or 0)
        if seq <= last_seq:
            return False, checked, f"checkpoint seq #{seq} is not strictly increasing"
        last_seq = seq
        record_count = int(record.get("record_count") or 0)
        head_hash = str(record.get("head_hash") or "")
        if not (1 <= record_count <= len(hashes)):
            return (
                False,
                checked,
                f"checkpoint #{seq} covers record {record_count}, outside the log ({len(hashes)} records)",
            )
        if hashes[record_count - 1] != head_hash:
            return False, checked, f"checkpoint #{seq} disagrees with the log at record {record_count} (re-chained log?)"
        max_covered = max(max_covered, record_count)
        checked += 1
    return True, checked, f"anchor OK ({checked} checkpoints, anchored through record {max_covered} of {len(hashes)})"


def _read_log_hashes(log_path: Path) -> tuple[list[str], str | None]:
    """Return (event_hash list, error detail or None)."""
    hashes: list[str] = []
    if not log_path.exists():
        return hashes, None
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                return [], "audit log contains unparseable line"
            hashes.append(str(record.get("event_hash") or ""))
    return hashes, None


class AuditCheckpointAnchor:
    """Out-of-band (RFC 9162-style) checkpoint anchor for the audit log.

    Optionally enabled by setting ``OMNIFORGE_AUDIT__ANCHOR_PATH`` to a separate append-only
    JSONL checkpoint file (ideally on a different mount or shipped to an independent store),
    or by passing ``anchor_path=`` to :class:`AuditLogger`. After every anchored event, a
    checkpoint record is appended::

        {"version": "v1", "seq": N, "record_count": M, "head_hash": "<sha256>",
         "timestamp": "<iso8601>", "mac": "<hmac-sha256>"}

    ``mac`` is a domain-separated HMAC-SHA256 over the canonical checkpoint fields, keyed with
    material from ``OMNIFORGE_AUDIT__CHECKPOINT_KEY`` (derived with its own domain separator and
    persisted as a 0600 keyfile outside production, matching the rest of the key-management
    posture). An attacker who can rewrite the audit JSONL but cannot mint a valid checkpoint MAC
    cannot launder a re-chained log into something the anchor accepts.

    Honest limits: truncating BOTH the log tail and the trailing checkpoints consistently is
    indistinguishable without an external witness/replication step, and a host compromise that
    exposes the checkpoint key defeats the anchor too. The anchor raises the bar from
    "file-write on the audit log is enough" to "file-write AND checkpoint key are required".
    Replicate the checkpoint file to independent storage for witness-grade guarantees.

    ``anchor_every`` (``OMNIFORGE_AUDIT__ANCHOR_EVERY``, default 1) controls checkpoint cadence;
    every Nth event (counting from the first record already in the log) is anchored. ``1`` means
    every event.
    """

    VERSION = "v1"

    def __init__(self, checkpoint_path: str | Path, explicit_key: str | bytes | None = None) -> None:
        self._path = Path(checkpoint_path)
        self._key = _resolve_or_persist_secret(
            key_name="audit-checkpoint-hmac",
            explicit=explicit_key,
            env_vars=("OMNIFORGE_AUDIT__CHECKPOINT_KEY",),
        )
        self._mac_key = hashlib.sha256(b"omniforge:audit-checkpoint:v1:" + self._key).digest()
        self._last_seq = self._load_last_seq()

    @property
    def path(self) -> Path:
        return self._path

    @staticmethod
    def _canonical(fields: dict[str, Any]) -> bytes:
        return json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")

    def _mac(self, fields: dict[str, Any]) -> str:
        material = _CHECKPOINT_HMAC_CONTEXT + b"\x00" + self._canonical(fields)
        return hmac.new(self._mac_key, material, hashlib.sha256).hexdigest()

    def _fields(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "version": record.get("version"),
            "seq": record.get("seq"),
            "record_count": record.get("record_count"),
            "head_hash": record.get("head_hash"),
            "timestamp": record.get("timestamp"),
        }

    def _verify_checkpoint(self, record: dict[str, Any]) -> bool:
        mac = str(record.get("mac") or "")
        if not mac:
            return False
        try:
            expected = self._mac(self._fields(record))
        except (ValueError, TypeError):
            return False
        return hmac.compare_digest(expected, mac)

    def _load_last_seq(self) -> int:
        if not self._path.exists():
            return 0
        last_seq = 0
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if not self._verify_checkpoint(record):
                        raise SecurityError(
                            "Audit checkpoint anchor fails HMAC verification; refusing to extend a forged anchor.",
                            suggestions=[
                                "Treat the checkpoint file as tampered. Compare against a replicated anchor copy.",
                            ],
                        )
                    seq = int(record.get("seq") or 0)
                    if seq <= last_seq:
                        raise SecurityError(
                            "Audit checkpoint anchor seq is not strictly increasing; refusing to extend it.",
                            suggestions=[
                                "Treat the checkpoint file as rewritten. Restore it from a replicated copy.",
                            ],
                        )
                    last_seq = seq
        except json.JSONDecodeError as exc:
            raise SecurityError(
                "Audit checkpoint anchor contains an unparseable line; refusing to extend it.",
                suggestions=["Restore the checkpoint file from a replicated copy or archive it before continuing."],
            ) from exc
        return last_seq

    def anchor_event(self, *, record_count: int, head_hash: str) -> dict[str, Any]:
        """Append a MACed checkpoint covering ``record_count`` records with head ``head_hash``."""
        seq = self._last_seq + 1
        fields = {
            "version": self.VERSION,
            "seq": seq,
            "record_count": int(record_count),
            "head_hash": str(head_hash),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        record = {**fields, "mac": self._mac(fields)}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self._last_seq = seq
        return record

    def verify_log(self, log_path: str | Path) -> tuple[bool, int, str]:
        """Verify this anchor against ``log_path`` (uses the anchor's own checkpoint key). Returns ``(ok, checkpoints_checked, detail)``."""
        log_path = Path(log_path)
        hashes, error = _read_log_hashes(log_path)
        if error is not None:
            return False, 0, error
        checkpoint_lines: list[str] = []
        if not self._path.exists():
            return False, 0, "anchor file not found; cannot confirm out-of-band checkpoints"
        with self._path.open("r", encoding="utf-8") as handle:
            checkpoint_lines = [ln.strip() for ln in handle if ln.strip()]
        return _verify_anchor_records(hashes, checkpoint_lines, self._verify_checkpoint)

    @classmethod
    def verify_against_log(cls, log_path: str | Path, checkpoint_path: str | Path, explicit_key: str | bytes | None = None) -> tuple[bool, int, str]:
        """Verify ``checkpoint_path`` against ``log_path`` with an env/explicit checkpoint key.

        Convenience wrapper for offline/SIEM verification. The key resolves from
        ``OMNIFORGE_AUDIT__CHECKPOINT_KEY`` (or ``explicit_key``); in non-production local setups a
        generated 0600 keyfile is reused so unsigned verification never silently succeeds.
        """
        log_path = Path(log_path)
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            return False, 0, "anchor file not found; cannot confirm out-of-band checkpoints"
        key = _resolve_or_persist_secret(
            key_name="audit-checkpoint-hmac",
            explicit=explicit_key,
            env_vars=("OMNIFORGE_AUDIT__CHECKPOINT_KEY",),
        )
        verifier = cls.__new__(cls)
        verifier._path = checkpoint_path
        verifier._key = key
        verifier._mac_key = hashlib.sha256(b"omniforge:audit-checkpoint:v1:" + key).digest()
        verifier._last_seq = 0
        return verifier.verify_log(log_path)



class AuditLogger:
    """Hash-chained, HMAC-signed, tamper-evident enterprise audit trail persisted as JSONL.

    v2 design: every forensic-relevant field (including ``role``, ``trace_id`` and
    ``details``) is covered by the SHA-256 hash payload, and each record carries a
    domain-separated HMAC-SHA256 MAC (``chain_mac``) over the hash. Verification
    requires both the hash and the MAC; editing any covered field in the stored log
    invalidates the chain. There is intentionally NO unsigned legacy verification
    path — an unauthenticated (subset) hash would let a log-file attacker forge or
    strip records without the MAC key.
    Note: for full tamper-evidence across host compromise, anchor checkpoints
    externally (RFC 9162-style); this implementation covers in-band integrity only.
    """

    def __init__(
        self,
        log_path: str | Path = "./.omniforge_workspace/audit.jsonl",
        *,
        anchor_path: str | Path | None = None,
        checkpoint_key: str | bytes | None = None,
        anchor_every: int | None = None,
    ) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._audit_signer: ArtifactSigner | None = None
        self._last_hash = self._load_last_hash()
        self._record_count = self._count_records()
        # Out-of-band checkpoint anchor (audit P1-11/anchor item). Opt-in: set
        # OMNIFORGE_AUDIT__ANCHOR_PATH (or pass anchor_path=) to a separate append-only file;
        # explicitly disabled with 0/false/off/none. Unset by default so local runs are unchanged.
        env_anchor = os.getenv("OMNIFORGE_AUDIT__ANCHOR_PATH")
        if anchor_path is None and not _checkpoint_disabled(env_anchor):
            anchor_path = env_anchor.strip() if env_anchor else None
        if checkpoint_key is None:
            checkpoint_key = os.getenv("OMNIFORGE_AUDIT__CHECKPOINT_KEY") or None
        if anchor_every is None:
            raw_every = os.getenv("OMNIFORGE_AUDIT__ANCHOR_EVERY", "1").strip()
            try:
                anchor_every = int(raw_every)
            except ValueError as exc:
                raise SecurityError(
                    f"OMNIFORGE_AUDIT__ANCHOR_EVERY must be a positive integer, got {raw_every!r}.",
                    suggestions=["Set OMNIFORGE_AUDIT__ANCHOR_EVERY to a positive integer (1 = anchor every event)."],
                ) from exc
        if anchor_every < 1:
            raise SecurityError(
                f"OMNIFORGE_AUDIT__ANCHOR_EVERY must be >= 1, got {anchor_every}.",
                suggestions=["Set OMNIFORGE_AUDIT__ANCHOR_EVERY to a positive integer (1 = anchor every event)."],
            )
        self._anchor_every = anchor_every
        self._anchor: AuditCheckpointAnchor | None = (
            AuditCheckpointAnchor(anchor_path, checkpoint_key) if anchor_path else None
        )

    def _count_records(self) -> int:
        if not self._path.exists():
            return 0
        with self._path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def _signer(self) -> ArtifactSigner:
        if self._audit_signer is None:
            self._audit_signer = ArtifactSigner()
        return self._audit_signer

    @staticmethod
    def _canonical_payload(fields: dict[str, Any]) -> bytes:
        """Canonical JSON encoding used both for hashing and MAC verification."""
        return json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")

    @staticmethod
    def _event_fields(source: AuditEvent | dict[str, Any]) -> dict[str, Any]:
        """Full v2 payload: EVERY forensic-relevant field is covered by the hash chain."""
        if isinstance(source, AuditEvent):
            return {
                "event_id": source.event_id,
                "timestamp": source.timestamp,
                "tenant_id": source.tenant_id,
                "actor": source.actor,
                "role": source.role,
                "action": source.action,
                "resource_type": source.resource_type,
                "resource_id": source.resource_id,
                "status": source.status,
                "trace_id": source.trace_id,
                "details": source.details,
                "previous_hash": source.previous_hash,
            }
        return {
            "event_id": source.get("event_id"),
            "timestamp": source.get("timestamp"),
            "tenant_id": source.get("tenant_id"),
            "actor": source.get("actor"),
            "role": source.get("role"),
            "action": source.get("action"),
            "resource_type": source.get("resource_type"),
            "resource_id": source.get("resource_id"),
            "status": source.get("status"),
            "trace_id": source.get("trace_id"),
            "details": source.get("details") or {},
            "previous_hash": source.get("previous_hash"),
        }

    @staticmethod
    def _verify_record(record: dict[str, Any], signer: ArtifactSigner) -> bool:
        """v2 verification: full-payload hash + domain-separated HMAC-SHA256 MAC.

        There is deliberately NO legacy/bypass path: a log record must carry both a
        correct full-payload SHA-256 ``event_hash`` and a valid ``chain_mac``.
        """
        event_hash = str(record.get("event_hash") or "")
        chain_mac = str(record.get("chain_mac") or "")
        if not event_hash or not chain_mac:
            return False
        try:
            recomputed = hashlib.sha256(AuditLogger._canonical_payload(AuditLogger._event_fields(record))).hexdigest()
            if not hmac.compare_digest(recomputed, event_hash):
                return False
            return signer.verify_bytes(bytes.fromhex(recomputed), chain_mac, context=_AUDIT_HMAC_CONTEXT)
        except (ValueError, TypeError):
            return False

    def _load_last_hash(self) -> str:
        if not self._path.exists() or self._path.stat().st_size == 0:
            return "0" * 64
        last_line = ""
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last_line = line
        if not last_line:
            return "0" * 64
        try:
            record = json.loads(last_line)
        except json.JSONDecodeError:
            raise SecurityError(
                "Audit log's last record is not valid JSON; refusing to extend an unverifiable chain.",
                suggestions=["Restore the audit log from a trusted backup or archive it before continuing."],
            ) from None
        if not AuditLogger._verify_record(record, self._signer()):
            raise SecurityError(
                "Audit log's last record fails integrity verification (hash/MAC mismatch).",
                suggestions=[
                    "Treat the audit log as tampered. Investigate, then restore from an external checkpoint/backup."
                ],
            )
        return str(record.get("event_hash", "0" * 64))

    def log_event(
        self,
        *,
        action: str,
        resource_type: str,
        resource_id: str,
        actor: str = "system",
        role: str = "admin",
        tenant_id: str = "default-tenant",
        status: str = "success",
        details: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Append a cryptographically chained, HMAC-signed audit event to the audit log."""
        event = AuditEvent(
            tenant_id=tenant_id,
            actor=actor,
            role=role,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            status=status,
            trace_id=get_trace_id(),
            details=details or {},
            previous_hash=self._last_hash,
        )
        event.event_hash = hashlib.sha256(self._canonical_payload(self._event_fields(event))).hexdigest()
        event.chain_mac = self._signer().sign_bytes(bytes.fromhex(event.event_hash), context=_AUDIT_HMAC_CONTEXT)
        self._last_hash = event.event_hash

        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")
        self._record_count += 1
        if self._anchor is not None and self._record_count % self._anchor_every == 0:
            try:
                self._anchor.anchor_event(record_count=self._record_count, head_hash=event.event_hash)
            except OSError as exc:
                raise SecurityError(
                    "Audit checkpoint anchor could not be written; the event remains in the log but is unanchored.",
                    suggestions=[
                        "Check that OMNIFORGE_AUDIT__ANCHOR_PATH points at a writable, preferably external, location.",
                        "Re-run verification (AuditLogger.verify_with_anchor) and investigate the coverage gap.",
                    ],
                ) from exc
        return event

    @classmethod
    def verify_file(cls, log_path: str | Path) -> tuple[bool, int]:
        """Verify a JSONL audit log without constructing a logger (for SIEM/ops tooling).

        Same semantics as :meth:`verify_chain` but never raises on a corrupt last record.
        """
        path = Path(log_path)
        if not path.exists():
            return True, 0
        records: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        records.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False, 0
        signer = ArtifactSigner()
        expected_prev = "0" * 64
        count = 0
        for record in records:
            if record.get("previous_hash") != expected_prev:
                return False, count
            if not AuditLogger._verify_record(record, signer):
                return False, count
            expected_prev = str(record.get("event_hash") or "")
            count += 1
        return True, count

    def verify_chain(self) -> tuple[bool, int]:
        """Verify cryptographic continuity and integrity of the entire audit log.

        Every record is validated with a full-payload SHA-256 hash plus a domain-separated
        HMAC-SHA256 MAC. Returns:
            Tuple of ``(is_valid, total_events_checked)``.
        """
        return AuditLogger.verify_file(self._path)


    @property
    def anchor_enabled(self) -> bool:
        """True when out-of-band checkpoint anchoring is configured for this log."""
        return self._anchor is not None

    def verify_anchor(self) -> tuple[bool, int, str]:
        """Verify out-of-band checkpoints against the log. Returns ``(ok, checkpoints_checked, detail)``.

        When anchoring is not configured this returns ``(True, 0, "anchor not configured ...")`` so
        callers can distinguish "verified-but-absent" from "verified-and-present" via the detail.
        """
        if self._anchor is None:
            return True, 0, "anchor not configured (in-band integrity only)"
        return self._anchor.verify_log(self._path)

    def verify_with_anchor(self) -> tuple[bool, int, str]:
        """Verify the in-band chain AND the out-of-band anchor. Returns ``(ok, events_checked, detail)``."""
        chain_ok, count = self.verify_chain()
        if not chain_ok:
            return False, count, "audit chain integrity failed"
        anchor_ok, _checked, detail = self.verify_anchor()
        return anchor_ok, count, detail

    def read_events(self, limit: int = 100) -> list[AuditEvent]:
        """Read the most recent ``limit`` audit events."""
        if not self._path.exists():
            return []
        lines = [ln for ln in self._path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return [AuditEvent.model_validate_json(ln) for ln in lines[-limit:]]


class ComplianceValidator:
    """Evaluates datasets and workflows against GDPR, HIPAA, SOX, and EU AI Act controls."""

    def __init__(self, profiles: list[str] | None = None) -> None:
        self.profiles = [p.upper() for p in (profiles or ["GDPR", "SOX"])]
        self._pii = PIIRedactor()

    def evaluate_dataset(
        self,
        df: pd.DataFrame,
        *,
        strict: bool = False,
        has_consent_record: bool = True,
        encryption_enabled: bool = True,
    ) -> dict[str, Any]:
        """Run compliance checks against a dataset and return a structured assessment."""
        pii_findings = self._pii.scan_dataframe(df)
        violations: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []

        if "GDPR" in self.profiles:
            if pii_findings and not has_consent_record:
                violations.append(
                    {
                        "framework": "GDPR",
                        "control": "Art. 6 Lawful Basis",
                        "detail": f"PII detected in columns {list(pii_findings.keys())} without recorded consent.",
                    }
                )
            elif pii_findings:
                warnings.append(
                    {
                        "framework": "GDPR",
                        "control": "Art. 5 Data Minimization",
                        "detail": f"Consider pseudonymizing detected PII columns: {list(pii_findings.keys())}.",
                    }
                )

        if "HIPAA" in self.profiles:
            if pii_findings:
                violations.append(
                    {
                        "framework": "HIPAA",
                        "control": "Safe Harbor De-identification",
                        "detail": f"Potential PHI identifiers detected in columns {list(pii_findings.keys())}.",
                    }
                )
            if not encryption_enabled:
                violations.append(
                    {
                        "framework": "HIPAA",
                        "control": "Technical Safeguards - Encryption",
                        "detail": "Encryption at rest must be enabled for HIPAA workloads.",
                    }
                )

        if "SOX" in self.profiles and len(df) == 0:
            warnings.append(
                {
                    "framework": "SOX",
                    "control": "Data Completeness",
                    "detail": "Empty dataset detected; audit reconciliation may fail.",
                }
            )

        compliant = len(violations) == 0
        report = {
            "compliant": compliant,
            "profiles_evaluated": self.profiles,
            "pii_columns_detected": pii_findings,
            "violations": violations,
            "warnings": warnings,
            "checked_at": datetime.now(UTC).isoformat(),
        }
        if strict and not compliant:
            raise ComplianceError(
                f"Dataset failed strict compliance validation for profiles {self.profiles}.",
                context=report,
                suggestions=[
                    "Anonymize detected PII/PHI columns with PIIRedactor.anonymize_dataframe().",
                    "Enable encryption at rest and document lawful consent basis.",
                ],
            )
        return report
