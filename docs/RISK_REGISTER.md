# OmniForge Risk Register — NIST AI RMF (AI 100-1) Mapping

**Scope:** OmniForge v3.2.0 platform (API, workbench SPA, 18 industry domain plugins, AutoML/MLOps,
responsible-AI modules, packaging/installer). **Framework:** NIST AI Risk Management Framework 1.0
(GOVERN / MAP / MEASURE / MANAGE) and its trustworthy-AI characteristics. **Source of findings:**
internal audit ([docs/AUDIT_REPORT.md](AUDIT_REPORT.md)) plus standing operational risks.

**Method:** each risk names the component, the NIST function(s) affected, likelihood × impact (H/M/L),
existing controls (with file/env references), residual rating, owner, and status. Ratings are
pre-treatment → residual. Owners are *roles*, not individuals. Review cadence: every release, plus
after any P0/P1 incident, plus quarterly tabletop for MANAGE-category rows.

**Status legend:** `Mitigated` (controls verified by tests/CI) · `Accepted` (documented, deliberate)
· `Open` (tracked, not yet fully mitigated) · `Watch` (monitor, no action unless triggered).

---

## GOVERN — culture, accountability, policies

| ID | Trustworthy AI char. | Risk | Component | L | I | Rating | Existing controls | Residual | Owner | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| RR-001 | Valid & Reliable / Accountable | No single accountable role for AI-system residual risk | Org process | M | H | High | This register + mandatory release review; `platform_admin` role restricted to a handful of IdP-group-mapped operators (`core/constants.py`) | Low | Eng leadership | Mitigated |
| RR-002 | Explainable | Executives treat AutoML leaderboard as ground truth; challenged by "lift over strongest baseline" gaps | AutoML | M | M | Medium | Mandatory strong baselines; lift reporting in every AutoML result (`automl/`); `MultiBackendAutoMLResult` carries `lift_over_best_baseline` | Low | ML governance | Mitigated |
| RR-003 | Privacy | PII leakage from ingestion into training artifacts, logs, or reports | Data + MLOps | M | H | High | `PIIRedactor` scan at profiling; HMAC pseudonymization with per-tenant KMS salt (`OMNIFORGE_PSEUDONYM_SALT__<TENANT>`); log redaction of secrets; artifact signing (HMAC) on save; model cards | Medium | Security eng | Mitigated |
| RR-004 | Privacy | Audit trail unreadable/forged after host compromise — accountability collapses | Audit | L | H | Medium | v2 hash chain + HMAC; optional RFC 9162-style anchor; `docs/AUDIT_ANCHOR_REPLICATION.md` replication/witness patterns; fail-closed anchor writes | Medium (truncation with consistent log+anchor deletion needs external witness — accepted with Option A/B replication) | Security eng | Watch |
| RR-005 | Safe / Accountable | Cross-tenant data exposure via over-privileged `admin` (historical single-role design) | AuthZ | L | H | Medium | `platform_admin` (only bypass) vs tenant-scoped `admin`; mandatory `authorize:cross_tenant` audit on every allow/deny; fail-closed allow on audit-write failure; API refuses boot on audit sink failure | Low | Security eng | Mitigated |

## MAP — context, categorization, impact assessment

| ID | Trustworthy AI char. | Risk | Component | L | I | Rating | Existing controls | Residual | Owner | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| RR-010 | Fair | No systematic fairness measurement in the AutoML/Registry promotion path for regulated domains (finance/HR) | Responsible AI + Registry | M | H | High | `ModelRegistry._enforce_fairness_gate` (`mlops/registry.py`) blocks `ModelStage.PRODUCTION` when `fairness_passed is False` (or when `OMNIFORGE_REQUIRE_FAIRNESS_FOR_PRODUCTION=true` and `fairness_passed is None`) unless an auditable `override_reason` is logged via `AuditLogger`; Studio 11 (`s11_mlops.py`) auto-binds `uap_fairness_result` | Low | Domain leads | Mitigated |
| RR-011 | Explainable | SHAP/LIME explanations computed on training distribution; production skew makes them misleading | Responsible AI | M | M | Medium | Drift detection (PSI/KS/JS/Wasserstein) in `mlops/drift.py`; promotion gate on unseen holdout; cards document training data | Medium | ML eng | Watch |
| RR-012 | Valid & Reliable | Synthetic (Gaussian-copula) data exported as "real" without marking | Synthetic | L | M | Medium | Synthetic engine flags `is_synthetic`; model cards include data lineage section | Low | Data eng | Mitigated |
| RR-013 | Safe | Forecasts presented without uncertainty; consumers ignore point estimates | Forecasting | M | M | Medium | Prediction intervals/conformal variants in forecasting stack; model cards carry intended-use + limits | Medium | ML eng | Watch |
| RR-014 | Privacy | Embeddings/vector stores retain PII beyond retention policy | Feature store / vector | M | M | Medium | `LocalFeatureStore.get_online_features` enforces `spec.ttl_seconds` (Feast-compatible freshness vs. `event_timestamp`/`updated_at`); `purge_expired_features()` removes expired offline Parquet + online SQLite rows; `delete_entity_data()` implements GDPR Art. 17 right-to-erasure across all views (`data/feature_store.py`) | Low | Platform eng | Mitigated |

## MEASURE — metrics, verification, monitoring

| ID | Trustworthy AI char. | Risk | Component | L | I | Rating | Existing controls | Residual | Owner | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| RR-020 | Valid & Reliable | Behavioral regressions ship silently (256 tests, no perf gate historically) | CI | L | M | Medium | 320+ test suite; ruff+mypy gates; `latency-budget` job (p95 budgets); `lock-verify` job (`--require-hashes`); `e2e` job (fresh SPA build + drift gate) | Low | Eng | Mitigated |
| RR-021 | Valid & Reliable | Drift detector false-negatives → stale champion serves silently degrading predictions | MLOps | M | H | High | Scale-equivariant ADWIN + PSI/KS/JS/Wasserstein (`mlops/drift.py`); async paired-feedback `ModelMonitor.evaluate_health(auto_rollback=True)` + `ContinuousTrainingOrchestrator.monitor_and_rollback` automatically demote failing champions | Low | MLOps | Mitigated |
| RR-022 | Secure & Resilient | Supply-chain compromise (dep confusion / tampered wheel) | Packaging | L | H | Medium | `requirements*.lock` hash-pinned; `lock-verify` CI job; deterministic CycloneDX 1.5 SBOM generator/verifier with SHA-256 lockfile provenance (`core/sbom.py`, `scripts/generate_sbom.py`, `omniforge sbom` CLI); `build_locks.py --check` enforces 100% hash-pinning | Low | Platform eng | Mitigated |
| RR-023 | Privacy | Benchmark/CI fixtures contain real-looking PII; committed sample data leaks | Repo | L | L | Low | Synthetic fixtures only; `.gitignore` blocks `*.jsonl` audit files; secret redaction in logs | Low | Repo hygiene | Mitigated |
| RR-024 | Secure & Resilient | Rate-limit naive per-process behind multi-replica deployment → effective cap ×N | API | M | M | Medium | `rate_limit.py` Redis backend (atomic ZSET+Lua) with in-memory degradation; `OMNIFORGE_API__RATE_LIMIT_BACKEND=redis`; X-Forwarded-For not trusted | Low (single-process) / Low (Redis) | Platform eng | Mitigated |

## MANAGE — prioritize, respond, recover

| ID | Trustworthy AI char. | Risk | Component | L | I | Rating | Existing controls | Residual | Owner | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| RR-030 | Safe | Closed-loop CT auto-promotes a degraded challenger to PRODUCTION unnoticed | Continuous training | L | H | Medium | Kill-switch `OMNIFORGE_CT_DISABLED=true` halts ALL CT activity; `OMNIFORGE_CT_AUTO_PROMOTE=false` caps at STAGING; audit events `ct:kill_switch_blocked`, `ct:auto_promotion_withheld`; promotion requires ≥ champion + min_improvement on unseen holdout; register stage STAGING by default when withheld | Low | MLOps | Mitigated |
| RR-031 | Safe | Incident response has no automated/operator path for "model behaving badly in prod" | Ops | M | H | High | Atomic `ModelRegistry.rollback_production` (`mlops/registry.py`), `POST /api/workbench/s11/rollback`, and live SLA breach auto-rollback in `ModelMonitor.evaluate_health(..., auto_rollback=True)` (`OMNIFORGE_AUTO_ROLLBACK_DISABLED` override) with `registry:production_rollback` tamper-evident audit trail | Low | MLOps + SRE | Mitigated |
| RR-032 | Accountable | No kill-switch for autonomous copilot actions in a loop | Copilot | L | M | Low | Copilot tools are request-scoped, authenticated, RBAC-checked; every tool call audited; offline planner cannot shell out | Low | Product + Security | Mitigated |
| RR-033 | Secure & Resilient | Host compromise → attacker rewrites log+anchor consistently; org cannot distinguish | Audit | L | H | Medium | Anchor + replication guide (Option A/B/C); checkpoint-key in KMS; verification runbook (`verify_against_log`); hourly verification recommended | Medium | Security eng | Accepted (documented limit) |
| RR-034 | Valid & Reliable | Dependence on optional heavy backends (AutoGluon/H2O/TPOT) unavailability breaks AutoML SLO silently | AutoML | M | L | Low | `AutoMLBackendHub` probes backends and falls back to native engines; fallback reported in result (`used_fallback=True`) | Low | ML eng | Mitigated |
| RR-035 | Privacy | `allow_unsafe_pickle=True` path re-introduces arbitrary code execution if a caller opts in | Ingestion | L | H | High | Gate defaults False with explicit message; `safe_joblib_load_bytes` HMAC-verifies artifacts first; no in-repo callers enable it | Medium | Security eng | Accepted (explicit opt-in, in audit §4.6) |
| RR-036 | Secure & Resilient | Untrusted XML via `ET.fromstring` fallback (defusedxml-first, but fallback exists) | Ingestion | L | M | Low | DOCTYPE/ENTITY pre-rejected; defusedxml attempted first; fallback documented in audit §4.6 | Low | Data eng | Watch |
| RR-037 | Secure & Resilient | Binding on `0.0.0.0` in container/dev defaults widens exposure | Deployment | M | L | Low | Non-root uid 1000 in image; healthcheck; compose secrets fail-fast in prod; docs recommend 127.0.0.1 dev binding | Low | Platform eng | Watch |
| RR-038 | Accountable | Model cards describe intent; no enforced registry accountability metadata or promotion gate on model change | Governance | M | M | Medium | `RegisteredModelVersion` persists NIST AI RMF fields (`owner`, `intended_use`, `approved_domains`, `risk_tier`, `fairness_passed`) with SQLite schema migration and auditable `override_reason` enforcement on `PRODUCTION` transitions (`mlops/registry.py`) | Low | Eng leadership | Mitigated |
| RR-039 | Secure & Resilient | DuckDB SQL connector / tabular ingestion could invoke external file-scan functions (`read_csv_auto`, `read_parquet`) | Data + Security | L | H | High | `_resolve_duckdb_url_path` workspace confinement, `duckdb.connect(..., read_only=True, config={"enable_external_access": False})`, `_FORBIDDEN_SQL_RE` scan-function denylist, and SQL-free `duckdb.read_parquet`/`duckdb.read_csv` (`data/connectors.py`, `data/ingestion.py`) | Low | Security eng | Mitigated |

---

## CT kill-switch quick reference (RR-030)

```bash
# Halt ALL continuous-training activity immediately (drift check, training, registration, promotion):
export OMNIFORGE_CT_DISABLED=true

# Or cap the loop at STAGING — challengers train but never auto-promote:
export OMNIFORGE_CT_AUTO_PROMOTE=false
```

Behavioral contract (enforced in `omniforge/mlops/continuous_training.py`):

- `OMNIFORGE_CT_DISABLED=true` → `evaluate_and_retrain()` returns immediately with
  `{"retrained": False, "kill_switch": True}`; no drift detection, training, or registry writes occur.
  Audited as `ct:kill_switch_blocked` (best-effort, never raises).
- `OMNIFORGE_CT_AUTO_PROMOTE=false` → challenger is registered at `STAGING` only when it beats the
  champion; audited as `ct:auto_promotion_withheld`. Promotion to `PRODUCTION` is a human decision
  (registry stage flip).
- Both flags are read at orchestrator construction; restart the worker/API to change them.
  Explicit constructor args (`kill_switch=`, `auto_promote=`) override the environment (useful in tests).

## Review & maintenance

- **Trigger-based:** any P0/P1 security/audit finding adds or updates a row here before release.
- **Calendar-based:** quarterly review of all `Open`/`Watch` rows; every release re-validates
  `Mitigated` rows with a sample test.
- **Metrics:** % rows Mitigated (target ≥80%), % `Open` with owner + date, audit coverage of
  safety decisions (CT kill-switch, cross-tenant allow/deny, anchor verification).

*Companion docs: [docs/AUDIT_REPORT.md](AUDIT_REPORT.md) · [docs/AUDIT_ANCHOR_REPLICATION.md](AUDIT_ANCHOR_REPLICATION.md) ·
[docs/architecture.md](architecture.md) · [docs/P2_REMEDIATION_REPORT.md](P2_REMEDIATION_REPORT.md) (historical).*
