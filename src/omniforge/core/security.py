"""Enterprise security, encryption, JWT auth, RBAC/ABAC, PII redaction, and audit trails."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from omniforge.core.constants import RBAC_ROLE_PERMISSIONS
from omniforge.core.exceptions import AuthorizationError, ComplianceError, SecurityError
from omniforge.core.logging import get_logger, get_trace_id
from omniforge.core.types import AuditEvent, Principal

try:
    import jwt

    _HAS_PYJWT = True
except ImportError:
    jwt = None  # type: ignore[assignment]
    _HAS_PYJWT = False

try:
    from cryptography.fernet import Fernet, InvalidToken

    _HAS_CRYPTOGRAPHY = True
except ImportError:
    Fernet = None  # type: ignore[assignment,misc]
    InvalidToken = Exception  # type: ignore[assignment,misc]
    _HAS_CRYPTOGRAPHY = False

logger = get_logger(__name__)

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
    """Symmetric encryption at rest using Fernet (AES-128-CBC + HMAC-SHA256) with XOR-HMAC fallback."""

    def __init__(self, key: str | bytes | None = None) -> None:
        raw_key = key or os.getenv("OMNIFORGE_ENCRYPTION_KEY")
        if _HAS_CRYPTOGRAPHY and Fernet is not None:
            if raw_key is None or raw_key == "":
                self._fernet_key = Fernet.generate_key()
            else:
                key_bytes = raw_key.encode("utf-8") if isinstance(raw_key, str) else raw_key
                try:
                    Fernet(key_bytes)
                    self._fernet_key = key_bytes
                except Exception:
                    digest = hashlib.sha256(key_bytes).digest()
                    self._fernet_key = base64.urlsafe_b64encode(digest)
            self._fernet = Fernet(self._fernet_key)
            self._raw_secret = self._fernet_key
        else:
            seed = (
                raw_key.encode("utf-8")
                if isinstance(raw_key, str) and raw_key
                else (raw_key if isinstance(raw_key, bytes) and raw_key else os.urandom(32))
            )
            self._raw_secret = hashlib.sha256(seed).digest()
            self._fernet_key = base64.urlsafe_b64encode(self._raw_secret)
            self._fernet = None

    @property
    def key_b64(self) -> str:
        """Return base64-encoded key material."""
        return self._fernet_key.decode("utf-8")

    def encrypt_bytes(self, plaintext: bytes) -> bytes:
        """Encrypt raw bytes into an authenticated ciphertext envelope."""
        if self._fernet is not None:
            return bytes(self._fernet.encrypt(plaintext))
        nonce = os.urandom(16)
        keystream = self._derive_stream(nonce, len(plaintext))
        cipher = bytes(a ^ b for a, b in zip(plaintext, keystream, strict=False))
        mac = hmac.new(self._raw_secret, nonce + cipher, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(nonce + mac + cipher)

    def decrypt_bytes(self, token: bytes) -> bytes:
        """Decrypt and verify authenticated ciphertext bytes."""
        try:
            if self._fernet is not None:
                return bytes(self._fernet.decrypt(token))
            raw = base64.urlsafe_b64decode(token)
            nonce, mac, cipher = raw[:16], raw[16:48], raw[48:]
            expected = hmac.new(self._raw_secret, nonce + cipher, hashlib.sha256).digest()
            if not hmac.compare_digest(mac, expected):
                raise SecurityError("Ciphertext integrity verification failed.")
            keystream = self._derive_stream(nonce, len(cipher))
            return bytes(a ^ b for a, b in zip(cipher, keystream, strict=False))
        except SecurityError:
            raise
        except Exception as exc:
            raise SecurityError(
                "Failed to decrypt payload. Verify encryption key and payload integrity.",
                suggestions=["Check OMNIFORGE_ENCRYPTION_KEY configuration."],
                cause=exc,
            ) from exc

    def encrypt_text(self, plaintext: str) -> str:
        """Encrypt UTF-8 string into an ASCII-safe token."""
        return self.encrypt_bytes(plaintext.encode("utf-8")).decode("ascii")

    def decrypt_text(self, token: str) -> str:
        """Decrypt ASCII-safe token back into UTF-8 string."""
        return self.decrypt_bytes(token.encode("ascii")).decode("utf-8")

    def _derive_stream(self, nonce: bytes, length: int) -> bytes:
        out = bytearray()
        counter = 0
        while len(out) < length:
            block = hmac.new(
                self._raw_secret, nonce + counter.to_bytes(4, "big"), hashlib.sha256
            ).digest()
            out.extend(block)
            counter += 1
        return bytes(out[:length])


class JWTManager:
    """JWT issuance and verification manager supporting PyJWT and stdlib HS256 fallback."""

    def __init__(
        self,
        secret_key: str | None = None,
        algorithm: str = "HS256",
        expiration_minutes: int = 60,
    ) -> None:
        self._secret_key = (
            secret_key
            or os.getenv("OMNIFORGE_JWT_SECRET_KEY")
            or "uap-development-jwt-secret-key-do-not-use-in-prod-0123456789"
        )
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
