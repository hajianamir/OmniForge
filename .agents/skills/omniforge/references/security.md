# OmniForge security reference

> Reverified at the code revision in [SKILL.md](../SKILL.md). These are
> scoped controls and known limitations, not a security certification. Review
> the implementation and deployment boundary before relying on any control.

## Authentication, permissions and ownership

`src/omniforge/api/auth.py::get_current_principal` accepts Bearer JWT,
`X-API-Key`/`X-Workbench-Token`, or the local workbench cookie, in that order.
The API key resolves to an admin principal for the default tenant and is also
authorized to mint tokens: treat it as a privileged secret. `/api/v1/auth/token`
requires a valid API key or an existing admin token; tenant admins cannot mint
platform-admin or other-tenant tokens. The workbench router globally requires
a principal (`src/omniforge/api/workbench.py`), but **many workbench handlers have no
additional per-action permission dependency**. Studio 11 promotion uses
`require_permission("model:promote")`; registration now requires `mlops:write`
and additionally `model:promote` for Production, and rollback requires
`model:promote`. Audit authorization route by route, including
`/api/v1` handlers (`src/omniforge/api/main.py`).

`_sid` namespaces a supplied session ID by tenant and subject. Job status is
also owner-scoped for ordinary principals, but `platform_admin` job queries and
cancellation intentionally use an unscoped owner key via a separate path; do
not claim that path is automatically covered by `AccessController` auditing.
Unit tests include `tests/unit/test_bola_session_isolation.py`,
`tests/unit/test_platform_admin_role.py`, and
`tests/unit/test_persistent_workbench_jobs.py`.

`src/omniforge/core/security.py::AccessController` evaluates roles/scopes and
optional resource tenant/attributes. With tenant isolation enabled, an audit
logger **and** a different `resource_tenant_id`, it records cross-tenant
boundary checks; an audit-write failure prevents an allow in the handled error
path. The `success` audit event is written before the later permission/attribute
checks, so it does not prove overall authorization succeeded. A controller
without a logger provides no such audit. `src/omniforge/api/auth.py` builds its
shared controller with an `AuditLogger`, but `require_permission` calls
`authorize` without a resource-tenant argument. Verify the actual call path
before claiming a decision is audited or fail-closed.

Cookie auth uses an eight-hour HMAC-authenticated `HttpOnly`, `SameSite=Strict`
cookie and a separate CSRF double-submit token on mutations. Auto-bootstrap is
disabled by the settings default and in production; if explicitly enabled in
non-production, SPA responses issue admin cookies. The bootstrap check is not
a network perimeter: do not enable it on a server exposed to untrusted users.
Note that `.env.example` enables it, despite the safe settings default.

## Secrets, encryption and artifacts

`src/omniforge/core/security.py::_resolve_or_persist_secret` resolves explicit material,
then applicable environment variables, then (outside production) a local
keyfile. The normal development key directory is `~/.omniforge/keys`, or
`OMNIFORGE_KEY_DIR` if set; the fallback when a home directory cannot be
resolved is `.omniforge_workspace/keys`. `.env.example`'s
`.omniforge_workspace/.secrets/` comment is not the implementation. Keyfiles
are created with 0600 permissions where supported. If the store is unwritable,
the implementation can generate process-local material, so persistence across
restarts is not guaranteed.

In production, provision distinct high-entropy secrets through a secret
manager before using the relevant component. JWT, encryption and artifact
signing resolvers enforce at least **32 bytes** and reject placeholder markers;
`src/omniforge/api/auth.py::get_api_key` calls the resolver with a **24-byte** minimum.
Using at least 32 random bytes for every key is a sensible deployment policy.
The corresponding nested settings names are
`OMNIFORGE_SECURITY__JWT_SECRET_KEY`, `OMNIFORGE_SECURITY__ENCRYPTION_KEY`,
`OMNIFORGE_SECURITY__ARTIFACT_SIGNING_KEY` and `OMNIFORGE_SECURITY__API_KEY`.
The implementation also accepts legacy un-nested names, and the signer has a
JWT-key fallback. Inspect `src/omniforge/core/security.py` and
`src/omniforge/api/auth.py` for exact precedence; avoid relying on fallback key
reuse. Never put real secrets into the repository or logs.

`ArtifactSigner` authenticates payloads with context-separated HMAC-SHA256;
`src/omniforge/models/base.py::save/load` signs persisted model bytes, verifies by default,
and then uses `safe_joblib_load_bytes` with allow/deny restrictions. The
unpickler admits selected non-OmniForge libraries as well as OmniForge types;
it is defense in depth, not permission to load arbitrary untrusted pickle.
`EncryptionManager` writes AES-256-GCM ciphertext and supports legacy Fernet
decryption; model encryption is optional, not the default.

`src/omniforge/core/workspace_session.py` writes a ZIP with a JSON manifest and a joblib
state payload. The **payload** is HMAC-signed and the default loader checks the
signature before restricted deserialization, with archive size/structure
limits. The manifest metadata itself is not covered by that payload HMAC;
`inspect_bytes()` validates/reads the manifest without verifying the payload.
Workspace exports are **not encrypted** and may contain datasets, models,
lineage and user inputs. Protect them as sensitive data. `.uap` is a recognized
legacy extension, not a promise that unsigned/old/incompatible files will
load. Do not disable signature checks or broaden the unpickler to make an
untrusted artifact work; recover a trusted backup and its original keys.

Before any upgrade or rotation, inventory and securely back up the needed
workspace, model metadata, and key material; plan/test a trusted migration and
retain old keys as needed to verify/decrypt existing artifacts and audit logs.
Blind rotation or deleting development keys can make data inaccessible.

## Connectors, uploads and privacy

- `RequestBodySizeLimitMiddleware` checks declared and streamed request sizes;
  `read_upload_bounded` additionally bounds upload reads. The default cap is
  268,435,456 bytes; bounded uploads can still consume substantial memory.
- `src/omniforge/data/connectors.py` validates single read-only `SELECT`/`WITH` queries,
  quotes inspected identifiers, confines local DB files to allowed roots and
  uses a SQLite read-only authorizer. Remote DB and HTTP validation checks
  resolved addresses for metadata/link-local and, unless configured, private
  hosts. `src/omniforge/data/ingestion.py::_fetch_http_safe` revalidates redirects, strips
  sensitive headers on cross-origin redirects and bounds responses. These are
  application-level guards, not a universal SQL or SSRF proof: validation and
  connection can resolve DNS separately, and other connectors may have
  different paths. Use network egress controls and read-only DB credentials.
  `OMNIFORGE_ALLOWED_DB_DIRS`, `OMNIFORGE_ALLOWED_DB_HOSTS`,
  `OMNIFORGE_ALLOW_PRIVATE_DB_HOSTS`, `OMNIFORGE_ALLOWED_HTTP_HOSTS` and
  `OMNIFORGE_ALLOW_PRIVATE_HTTP_HOSTS` expand trust and should be narrowly
  configured.
- `src/omniforge/copilot/tools.py::_validate_copilot_ingest_source` confines local paths to
  `OMNIFORGE_COPILOT_INGEST_ROOT` (default current directory), rejects
  traversal/dotfiles, and allows HTTP(S) through the ingestion path. Test
  symlinks, redirects and tenant boundaries when modifying this code.
- `DataFramePIIGuard` and `PIIRedactor` use patterns/heuristics, not exhaustive
  PII detection. The guard's `block` and `pseudonymize` policies apply to
  recognized cells when invoked; `off` leaves data unchanged, but the guard
  still resolves a salt on construction. Pseudonyms include the tenant ID;
  `resolve_pseudonym_salt` can use per-tenant salts or a global fallback. Its
  production resolver requires at least 32 non-placeholder bytes; dev may use
  a persisted or ephemeral salt. `PIIRedactor.anonymize_dataframe` also accepts
  an explicit salt without that resolver's strength check. Supply strong salts
  and review the actual call path and env precedence before migration.
- `src/omniforge/reporting/exporters.py::sanitize_spreadsheet_value` is used in report
  CSV/XLSX paths and preserves signed numeric literals. It is not automatically
  applied to every export: for example, `src/omniforge/api/workbench.py::s01_export_clean_dataset`
  calls pandas CSV/Excel writers directly. Audit each export for formula
  injection and test headers and cell values.

## Audit, headers and operational fallbacks

`AuditLogger` writes hash-chained, HMAC-authenticated JSONL events. Optional
`AuditCheckpointAnchor` uses a separately keyed checkpoint file; without an
independent replicated witness, coordinated truncation of both log and
checkpoint tail cannot be detected. `ModelRegistry` blocks Production for a
failed fairness result unless an override reason is supplied; a missing result
blocks only when the fairness gate is required. Registry gate/override audit
writes in `_audit_registry_event` are **best effort**, not fail-closed.

Rate limiting keys the ASGI client host, not an independently parsed
`X-Forwarded-For` header. The in-memory limiter is per process; Redis can be
configured, but `ResilientRateLimiter` can fall back to memory. Account for
trusted proxy settings, shared IPs and degraded Redis. Default CSP is
**report-only**, not XSS enforcement. The app also emits frame, content-type,
referrer and HSTS headers; HSTS only has browser effect over HTTPS. Wildcard
CORS disables credential support (`src/omniforge/config/settings.py`, `src/omniforge/api/main.py`).

`src/omniforge/core/logging.py` provides secret-pattern redaction, but it is not exhaustive:
other paths log `str(exc)` or include exception causes and user-derived data.
Review what each call emits rather than assuming all logs/errors are safe.
The reviewed fallbacks in `src/omniforge/ui/app.py`, `src/omniforge/models/mining/miner.py`,
`src/omniforge/models/nlp/advanced_nlp.py` and `src/omniforge/models/computer_vision/foundation.py` emit
structured warnings for attempted failures without exception messages or
dataset headers; mining records a zero-based column index. Optional backend
absence/intentional disabling does not warn. This is a scoped improvement, not
a claim about every fallback in the repo. Regression tests are in
`tests/unit/test_fallback_observability.py`.
