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
    """Combined RBAC and ABAC policy enforcement engine with multi-tenant isolation."""

    def __init__(
        self,
        role_permissions: dict[str, frozenset[str]] | None = None,
        enforce_tenant_isolation: bool = True,
    ) -> None:
        self._role_permissions = dict(role_permissions or RBAC_ROLE_PERMISSIONS)
        self._enforce_tenant_isolation = enforce_tenant_isolation

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
            if "admin" not in principal.roles and principal.tenant_id != resource_tenant_id:
                raise AuthorizationError(
                    f"Cross-tenant access denied for principal '{principal.subject}'.",
                    context={
                        "principal_tenant": principal.tenant_id,
                        "resource_tenant": resource_tenant_id,
                    },
                    suggestions=["Authenticate within the target tenant context."],
                )

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
        salt: str = "uap-pii-salt",
    ) -> pd.DataFrame:
        """Anonymize specified (or automatically detected) PII columns in a DataFrame.

        Args:
            df: Input DataFrame.
            columns: Columns to anonymize; auto-detected if ``None``.
            strategy: ``"hash"`` (salted SHA-256 prefix), ``"mask"``, or ``"drop"``.
            salt: Cryptographic salt for deterministic pseudonymization.
        """
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
            else:
                result[col] = result[col].apply(
                    lambda v: hashlib.sha256(f"{salt}:{v}".encode()).hexdigest()[:16]
                    if pd.notna(v)
                    else v
                )
        return result


class AuditLogger:
    """Hash-chained, tamper-evident enterprise audit trail persisted as JSONL."""

    def __init__(self, log_path: str | Path = "./.omniforge_workspace/audit.jsonl") -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = self._load_last_hash()

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
            return str(json.loads(last_line).get("event_hash", "0" * 64))
        except json.JSONDecodeError:
            return "0" * 64

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
        """Append a cryptographically chained audit event to the audit log."""
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
        hash_payload = json.dumps(
            {
                "event_id": event.event_id,
                "timestamp": event.timestamp,
                "tenant_id": event.tenant_id,
                "actor": event.actor,
                "action": event.action,
                "resource_type": event.resource_type,
                "resource_id": event.resource_id,
                "status": event.status,
                "previous_hash": event.previous_hash,
            },
            sort_keys=True,
        ).encode("utf-8")
        event.event_hash = hashlib.sha256(hash_payload).hexdigest()
        self._last_hash = event.event_hash

        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")
        return event

    def verify_chain(self) -> tuple[bool, int]:
        """Verify cryptographic continuity of the entire audit log.

        Returns:
            Tuple of ``(is_valid, total_events_checked)``.
        """
        if not self._path.exists():
            return True, 0
        expected_prev = "0" * 64
        count = 0
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("previous_hash") != expected_prev:
                    return False, count
                recomputed = hashlib.sha256(
                    json.dumps(
                        {
                            "event_id": record["event_id"],
                            "timestamp": record["timestamp"],
                            "tenant_id": record["tenant_id"],
                            "actor": record["actor"],
                            "action": record["action"],
                            "resource_type": record["resource_type"],
                            "resource_id": record["resource_id"],
                            "status": record["status"],
                            "previous_hash": record["previous_hash"],
                        },
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()
                if record.get("event_hash") != recomputed:
                    return False, count
                expected_prev = recomputed
                count += 1
        return True, count

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
