# OmniForge Security Reference

> Verified against commit `b7dc7ae` (2026-11-24); see `../SKILL.md` for provenance and authority policy.

> Read this before touching auth, serialization, SQL connectors, audit, or
> tenancy code. Each control below states the **threat it mitigates** — the
> review checklist in `SKILL.md` assumes this framing.

## Threat model (one line per control)

| Control (`core/security.py` unless noted) | Threat mitigated |
|---|---|
| `safe_joblib_load_bytes` + `_validate_unpickle_target` (restricted unpickler allowlist) | **RCE via crafted model artifacts** — `joblib.load` on untrusted bytes executes arbitrary code; the allowlist refuses non-OmniForge globals. |
| `ArtifactSigner` (HMAC-SHA256, domain-separated `context`) | **Artifact tampering / substitution** — a modified `.joblib` fails signature verification on `BaseOmniForgeModel.load`. |
| `EncryptionManager` (AES-256-GCM, Fernet fallback) | **Confidentiality at rest** — stolen artifact files are ciphertext without the key. |
| `JWTManager` + `api/auth.py` (`get_current_principal`, `require_permission`) | **Impersonation / privilege escalation** — every state-changing route requires an authenticated `Principal` with the least-privilege permission. |
| `AccessController.authorize` (RBAC/ABAC, mandatory audit, fail-closed) | **Cross-tenant data access (BOLA/IDOR)** — tenant `admin` cannot reach another tenant's sessions/jobs/registry; `platform_admin` bypass is audited. |
| Workbench cookie HMAC (`{nonce}.{ts}.{mac}`, 8h, `HttpOnly; SameSite=Strict`) + CSRF double-submit | **Session hijack + CSRF** — cookies are unforgeable, expiring, and state-changing requests need the `X-CSRF-Token` header. |
| `RequestBodySizeLimitMiddleware` (413) + `read_upload_bounded` | **Memory-exhaustion DoS** — huge uploads are rejected before allocation. |
| `SlidingWindowRateLimitMiddleware` + `core/rate_limit.py` (429 + `Retry-After`) | **Brute-force / abuse** — per-client sliding window (in-memory or Redis). |
| `validate_readonly_sql` + SQLite authorizer + identifier quoting (`data/connectors.py`) | **SQL injection / destructive queries** — only single read-only statements; identifiers allowlisted per dialect. |
| `_resolve_allowed_local_db_path` (confined roots) | **Path traversal / local-file disclosure** — SQLite/Access files must live under allowed directories. |
| `_validate_remote_db_host` + `_assert_ip_not_ssrf` + `validate_outbound_http_url` | **SSRF / DNS-rebinding** — private/internal hosts blocked unless explicitly allowed. |
| `DataFramePIIGuard` + `PIIRedactor` + `resolve_pseudonym_salt` | **PII leakage into logs/models** — salted, tenant-scoped pseudonymization; salt strength enforced (dev keyfile vs prod env). |
| `AuditLogger` (hash-chained JSONL, `event_hash` + `chain_mac`) | **Audit-log tampering** — rewriting history breaks the chain; `verify_file`/`verify_chain` detect it. |
| `AuditCheckpointAnchor` (out-of-band HMAC checkpoints) | **Log laundering** — an attacker who rewrites the JSONL but lacks the checkpoint key cannot mint valid checkpoints. "RFC 9162-style" means the same pattern as Certificate Transparency: a separate append-only checkpoint file carrying `(seq, record_count, head_hash, MAC)` so truncation/re-chaining is detectable. Honest limit (documented in the class docstring): truncating *both* log and checkpoints consistently is indistinguishable without external replication — replicate the checkpoint file for witness-grade guarantees. Off by default; key required in prod when enabled. |
| `ComplianceValidator` | **Regulatory violation** — GDPR/SOX-style dataset checks before processing. |
| Log redaction (`SENSITIVE_KEY_PATTERNS`, `RedactingJsonFormatter`) | **Secret leakage via logs** — passwords/tokens/keys never reach log output. |
| CORS/CSP/HSTS (`APISettings` + `main.py` middleware) | **Clickjacking / XSS / downgrade** — `nosniff`, `SAMEORIGIN`, report-only CSP, HSTS; `*` never combined with credentials (CWE-942). |
| Spreadsheet sanitization (`sanitize_spreadsheet_value`) | **CSV/Excel formula injection** — `=cmd` cells neutralized on export. |
| `.uap` archive validation (`_read_zip_member_bounded`, manifest SHA-256) | **Zip-slip / decompression bomb** — bounded reads + structure validation on project load. |

## RBAC

Permissions (`core/constants.py::RBAC_ROLE_PERMISSIONS`): `data:read/write/
delete`, `model:read/train/deploy/promote/delete`, `pipeline:read/execute/
write`, `mlops:read/write`, `copilot:use`, `report:generate`, `audit:read`,
`security:manage`. Roles: `admin` (full, tenant-scoped), `platform_admin`
(same set **plus** cross-tenant bypass — grant sparingly via IdP group),
`data_scientist`, `mlops_engineer`, `analyst`, `viewer`. Enforcement:
`api/auth.py::require_permission("model:deploy")` as FastAPI `Depends`;
`access_controller` is built eagerly with a mandatory `AuditLogger` and
**refuses to boot unaudited**. Tests: `test_bola_session_isolation.py`,
`test_platform_admin_role.py`, `test_p0_security_and_runtime_hardening.py`.

## Secrets & keys

Resolution order (`_resolve_or_persist_secret`): explicit setting → env var
→ 0600 keyfile under `.omniforge_workspace/.secrets/` (dev only). Production
(`OMNIFORGE_ENVIRONMENT=production`) requires explicit ≥32-byte values for
`OMNIFORGE_SECURITY__JWT_SECRET_KEY`, `__ENCRYPTION_KEY`,
`__ARTIFACT_SIGNING_KEY`, `__API_KEY` — generate with
`python -c "import secrets; print(secrets.token_urlsafe(48))"`. Never commit
`.env` or `.omniforge_workspace/`. Rotate all four on upgrade; old artifacts
signed with a previous key will fail verification by design.

## Fallbacks vs. silent failure (reconciliation)

The codebase deliberately degrades when optional packs are absent
(NumPy instead of torch, HistGB instead of XGBoost, permutation importance
instead of SHAP). This is **not** the `except: pass` anti-pattern when three
conditions hold — verify all three in review:

1. **Observable:** the active backend is recorded (`model.framework` /
   `metadata_.framework`, e.g. `omniforge_numpy_autograd`) and reported by
   `probe_*_backends` (`probe_foundation_backends`,
   `probe_automl_backends`, `probe_nlp_backends`, `probe_vision_backends`).
2. **Documented:** the pack's `unlocks` field in `capabilities.py` states
   what the fallback loses.
3. **Tested:** `test_third_party_claims.py` asserts fallback behavior.

What remains forbidden: bare `except: pass` that swallows errors without
logging, metadata, or a test. If you add a fallback, add the probe entry and
the `unlocks` text with it.
