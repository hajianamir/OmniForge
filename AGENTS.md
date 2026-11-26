---
document: AGENTS.md
title: OmniForge - Agent Operating Instructions
project: OmniForge v4.0.0 (package: omniforge, import root: src/omniforge/)
audience: Codex and other AI coding agents
status: Active
owner: OmniForge maintainers
last_reviewed: 2026-11-24
next_review: 2027-02-24
companion: docs/AGENT_REFERENCE.md (full manual; read only the sections your task needs)
---

# OmniForge - Agent Operating Instructions

> **Scope:** `github.com/hajianamir/OmniForge` (package `omniforge`, import root `src/omniforge/`).
> **How to use this file:** This file is auto-loaded. It states the rules and where to look. The full manual is `docs/AGENT_REFERENCE.md`. Do not read the whole manual by default - use the task-routing table in section 4.
> **Working rule:** Read the complete module you are about to change before you change it. The test suite (`tests/`) is the executable specification.

## 0. Before you start (60 seconds)

Record the baseline so pre-existing failures are not attributed to you:

```bash
pip install -e ".[dev,api,ui,reporting,data,forecasting]"
ruff check src tests
mypy src/omniforge
pytest -q tests
```

Note the results. If gates are already red, do not fix unrelated failures - only ensure your change adds no new findings.

## 1. Project identity

- **Name:** OmniForge - "Enterprise Analytics & AI Workbench". `src/omniforge/version.py`: `__version__ = "4.0.0"`, `__api_version__ = "v2"`, `__schema_version__ = "2.0"`.
- **Version alignment:** Keep `src/omniforge/version.py`, `pyproject.toml`, and `frontend/package.json` in step (all `4.0.0`).
- **Runtime:** Python `>= 3.12`. CI runs 3.12 and 3.13. Licence: Apache-2.0.
- **Legacy name "UAP" is load-bearing.** Session keys are `uap_*`; `BaseOmniForgeModel.framework = "uap"`; Kafka group `uap-consumer`; `.uap` is read-only legacy. Do not rename casually. Full detail: reference section 1.

## 2. Non-negotiable rules

1. **Python 3.12+.** New modules start with `from __future__ import annotations`.
2. **No casual hard dependencies.** Heavy/optional libraries (torch, xgboost, lightgbm, catboost, shap, optuna, statsmodels, pyarrow, duckdb, sqlalchemy, polars, pymongo, kafka, fastavro, codecarbon, transformers, cv2, spacy) are imported lazily inside functions and guarded. Raise `DependencyMissingError` only when the user explicitly requested that backend. Prefer `try/except ImportError` around the real import over `is_package_available()` (which only proves installation).
3. **Keep `import omniforge` cheap.** `__init__.py` uses PEP 562 lazy `__getattr__`. `version.py` and `capabilities.py` stay stdlib-only. `omniforge.core` is NOT lightweight (`core/security.py` imports pandas at top).
4. **Library errors are `OmniForgeError` subclasses** with `suggestions=[...]`, optional `context={...}` and `cause=exc`. Never raise bare `ValueError`/`RuntimeError` from library layers. Never put secrets, PII, or filesystem internals in `context`/`cause` (`to_dict()` exposes them to clients).
5. **API payloads must be JSON-safe.** Wrap in `core.utils.to_json_object` / `to_json_safe` (NaN/Inf/NaT -> `None`, numpy -> Python, pydantic -> `model_dump`, sets -> sorted lists, `Path` -> `str`).
6. **Security is fail-closed.** Never weaken SSRF validation, read-only SQL, pickle bans, path confinement, CSRF, tenant/session namespacing, audit-chain verification, or production secret requirements. Add no "temporary" bypasses. Full checklist: section 6 below.
7. **Never log or audit raw PII or secrets.** Use `get_logger(__name__)` (it redacts); record column names and counts only.
8. **Reproducibility.** Pass `random_seed` (default `DEFAULT_RANDOM_SEED = 42`). `BaseOmniForgeModel.fit` calls `set_global_seed` (note: it writes `PYTHONHASHSEED` at run time, which only affects child processes).
9. **Baseline-first modeling is a product invariant.** Any model-comparison or AutoML path must compare against baselines (`BaselineBenchmarkSuite`) and report lift over the best baseline. `ModelingSettings.always_compare_baselines` is declared but never read - enforcement lives in call sites.
10. **Leakage safety is a product invariant.** Fit transformers on train only, respect the Split Lock, and use `LeakageSafeSplitter` / `LeakageGuard`.
11. **Do not hand-edit generated artifacts:** `requirements.lock`, `requirements-build.lock`, `locks/*.lock`, and the compiled SPA in `src/omniforge/ui/static_app/`. Regenerate them (reference sections 3.1, 15.4).
12. **Keep the gates green for the code you touch.** Add tests under `tests/unit`, `tests/integration`, or `tests/e2e` with registered markers only (`unit, integration, e2e, slow`).
13. **Do not trust a settings field because it exists.** Most of `config/settings.py` is declared but unused. Grep for a consumer before relying on a setting. Consumed settings are listed in reference section 7.2.

## 3. Commands and quality gates

| Task | Command |
|---|---|
| Install (matches CI) | `pip install -e ".[dev,api,ui,reporting,data,forecasting]"` |
| Install everything except `experimental` | `pip install -e ".[all]"` (what `make install-dev` runs) |
| Lint | `ruff check src tests` (line length 120, target `py312`, rules `E, F, I, W, B, UP`, ignore `E501`/`B008`) |
| Typecheck | `mypy src/omniforge` (not `strict`, but `disallow_untyped_defs`, `warn_unused_ignores`, etc.; stale `# type: ignore` is an error) |
| Test | `pytest -q tests` (`-ra --strict-markers`; unregistered marker fails the run) |
| Single test | `pytest -q tests/unit/<file>.py::test_name` |
| Run API | `uvicorn omniforge.api.main:app --host 0.0.0.0 --port 8000 --reload` (`make run-api`) |
| Run UI | `python -m omniforge.launcher --mode ui --host 0.0.0.0 --ui-port 8501 --no-browser` (`make run-ui`) |
| Frontend build | `cd frontend && npm ci && npm run build` (writes to `src/omniforge/ui/static_app`) |
| CLI | `omniforge = "omniforge.cli:app"`; `python -m omniforge` runs `cli.main.main`; commands: `version, domains, quality, clean, eda, automl, drift, copilot, check, desktop, monitor, sbom, serve, ui` |

Notes:

- CLI `serve`/`ui` default to `--host 127.0.0.1`. Only `make run-api`/`run-ui` and `.env.example` bind `0.0.0.0`. Do not conflate them.
- `launcher.main()` is the **only** place `.env` is loaded. `uvicorn omniforge.api.main:app` and the `omniforge` CLI do **not** read `.env`.
- Lock staleness is **not** enforced by CI (`lock-verify` proves installability, not freshness). Regenerate locks with `uv` after any dependency change; never hand-edit them.
- CI `e2e` checks `git diff --quiet -- src/omniforge/ui/static_app` - it catches modified/deleted tracked files, **not** newly added untracked ones. Always `git add` new `static_app` files.

## 4. Task routing - read only what you need

| If you are... | Read first | Key files |
|---|---|---|
| Adding a studio endpoint | Reference 14.5 (handler pattern + route map) | `api/studios/schemas.py`, `api/studios/sNN_*.py`, `api/workbench.py` (bottom router block) |
| Changing workbench state / dataset mutations | Reference 14.4 | `ui/state.py` (`update_active_dataset`, `get_analysis_dataframe`, `_DOWNSTREAM_KEYS`) |
| Adding a model | Reference 9.1-9.3, 19 | `models/base.py`, `models/registry.py` (`ModelCatalog`) |
| Adding a Copilot tool | Reference 12 | `copilot/tools.py` (`CopilotToolRegistry`) |
| Adding a data source | Reference 8.1, 19 | `data/ingestion.py`, `data/connectors.py` |
| Adding a setting / env var | Reference 7.5, 17.1 | `config/settings.py`, `.env.example` |
| Adding a dependency | Reference 3.1, 19 | `requirements.in`, `pyproject.toml` extra, `capabilities.py` pack |
| Adding a built-in domain | Reference 13 | `domains/<slug>.py`, `domains/__init__.py`, `configs/domains/<slug>.yaml` |
| Touching auth / security | Reference 6.6, 14.2, 17 | `core/security.py`, `api/auth.py`, `api/main.py` |
| Touching frontend | Reference 15.3 | `frontend/src/` + `types.ts` / `studioTypes.ts` (payload contracts) |
| Touching drift / MLOps | Reference 10 | `mlops/drift.py`, `mlops/registry.py`, `mlops/root_cause.py` |
| Touching MPC / simulation | Reference 11 (read the gradient warning first) | `simulation/closed_loop.py` |

Areas not covered in depth (read the module and its tests in full before editing): `eda/*`, `features/*`, `forecasting/*`, `causal/*`, `optimization/*`, `evaluation/*`, `automl/*`, `reporting/*`, `pipelines/{dag, executor}`, `models/*/` subpackages, `mlops/{tracking, serving, monitoring, continuous_training, job_queue, future_scoring, supervisory_monitor}`, `simulation/engines.py`, `data/{cleaning, quality, validation, versioning, feature_store, synthetic, multimodal}`, `copilot/{agent, memory, prompts}`, `frontend/src/**` internals, `installer/`, `packaging/`, `docker/`, `configs/**`. Full list: reference section 21.

## 5. Architecture traps

- **Layering guardrails.** `api/workbench.py` imports `omniforge.ui.state` at module level, while `ui/__init__.py` uses lazy PEP 562 `__getattr__` and `ui/app.py` defers `omniforge.api.main` imports inside `create_workbench_app()` so importing `omniforge.ui.state` or `omniforge.ui` does not construct the FastAPI app at import time. Do not add top-level imports of `omniforge.api` or `omniforge.ui` into `core` or `data`. Keep the studio `_wb()` lazy-import pattern. Full detail: reference section 5.
- **REST and Workbench share the tenant model registry.** Obtain the registry via `get_tenant_registry()` and the Workbench tracker via `get_tenant_tracker()`. The shared registry imports legacy REST catalog entries while preserving existing version IDs and signed artifact paths.
- **State is per-process and lost on restart.** `ServerSessionRegistry` (in-process, `RLock`-guarded) and `BackgroundJobManager` do not share across workers.
- **Canonical workspace extension is `.omniforge`.** `.uap` is read-only legacy. Signing keys are host-local - workspaces do not verify across hosts unless keys are shared.
- **Remediated defects & remaining architectural boundaries:** D1 (multi-writer audit locking & tail sync), D2 (exact MPC gradient, initial-step dynamics & soft state slack), D3 (`.env.example` `AUTO_BOOTSTRAP_LOCAL_UI=false` & loopback `Makefile` defaults), D4 (`model:promote` enforced on `/api/v1/registry/.../stage` Production transitions and request-scoped Copilot tools), D5 (`list_studios` OpenAPI-verified routes), D6 (per-hop SSRF DNS-rebinding IP pinning + TLS SNI), D7 (class/instance domain entry-point discovery), D8 (atomic in-memory encrypted `save()`), D12, D14 (`/api/system/capabilities`), D15 (JWT `exp`/`sub` parity & sanitized `OmniForgeError` HTTP responses), D16, and D17 (CI untracked `static_app` check) are remediated and covered by `tests/unit/test_remediation_audit_loop.py`. Full register: reference section 20.

## 6. Security invariants (verify on every change)

1. Every `/api/workbench` route keeps `require_workbench_auth`, and Production model promotions enforce `model:promote` across `/s11/promote`, `/api/v1/registry/.../stage`, and request-scoped Copilot tools.
2. Workbench state is reached only through `_sid()`. Cookie and API-key callers share one namespace.
3. State-changing, cookie-authenticated requests need a matching `X-CSRF-Token`.
4. Uploads use `read_upload_bounded`. Pickle-like extensions (`.pkl .pickle .joblib .dill`) are rejected.
5. Outbound HTTP goes only through `resolve_and_validate_outbound_http_url` / `validate_outbound_http_url` and `_fetch_http_safe`, with no `follow_redirects=True` and per-hop IP pinning + TLS SNI verification.
6. SQL goes only through `validate_readonly_sql`, with allow-listed and quoted identifiers. SQLite is read-only with an authorizer. DuckDB is read-only without external access.
7. Model and workspace artifacts are HMAC-signed, saved atomically via sibling temp files + `os.replace` (plaintext never touches disk when `encrypt=True`), and loaded through the restricted unpickler. Enums that can be pickled use `SerializableStrEnum`. See reference 6.6.
8. Secrets resolve through `_resolve_or_persist_secret` (race-free across threads/processes). Production requires explicit, strong, non-placeholder values.
9. The audit log is append-only, hash-chained and MACed, synchronized across threads and processes via `_locked_file_path` + prefix digest verification. Never add an unsigned verification path.
10. Copilot file ingestion stays confined to `OMNIFORGE_COPILOT_INGEST_ROOT`, with no hidden files and no system directories.
11. Rate limiting is keyed by the ASGI client host. Only `/health` and `/api/v1/health` are exempt.
12. CORS never combines a wildcard with credentials.
13. Registry promotions respect the fairness gate. Overrides are tagged and audited.
14. Tenant isolation: per-tenant registries and trackers; cross-tenant access only for `platform_admin`, and always audited.
15. Logs, audits and `to_dict()` outputs contain no raw PII or secrets.
16. Anything that accepts a path or URL from a client must add its own confinement. `DataIngestionEngine.ingest` does not confine local paths.

Environment variables: reference section 17.1 (direct `os.getenv` reads) plus settings-driven vars in `.env.example`.

## 7. Recipes (condensed - full steps in reference section 19)

- **Studio endpoint:** request model in `api/studios/schemas.py`, handler in the right `sNN_*.py` (reference 14.5 pattern), cache under `sNN_<name>`, dataset mutations via `update_active_dataset(..., preserve_stage=...)`, new derived keys in `_DOWNSTREAM_KEYS`, update frontend types, rebuild and commit `static_app`, add a test.
- **Model:** subclass `BaseOmniForgeModel`, register in `ModelCatalog`, confirm save/load round-trip under the restricted unpickler (no lambdas, no unsafe globals), include in baseline comparisons.
- **Drift metric:** extend `DriftDetector.detect` and `FeatureDriftResult`; thresholds are constructor arguments, not settings.
- **Data source:** extend `DataIngestionEngine` routing and `SUPPORTED_*` constants; reuse `_fetch_http_safe` and read-only SQL helpers; add `DependencyMissingError` with install hint; sanitize URIs in lineage.
- **Setting/env var:** add field with safe default, mirror in `.env.example`, **wire a consumer and add a test** (otherwise it is another dead setting).
- **Dependency:** edit `requirements.in`, the `pyproject.toml` extra, and the `capabilities.py` pack (when user-facing); regenerate **all** affected locks with `uv`, including `locks/` via `scripts/build_locks.py`; confirm the `lock-verify` smoke import.

## 8. Pre-commit checklist

- [ ] I read every file I modified in full, and I re-ran the baseline gates first (section 0).
- [ ] `ruff check src tests` and `mypy src/omniforge` show **no new findings**, with full annotations and no stale `# type: ignore`.
- [ ] `pytest -q` shows no new failures, and new behaviour has tests with registered markers.
- [ ] No new hard or heavy dependency. Optional imports are lazy with fallbacks. `import omniforge` stays fast, and `version.py` and `capabilities.py` stay stdlib-only.
- [ ] Errors are `OmniForgeError` subclasses with suggestions. Payloads are JSON-safe. Nothing logs PII or secrets. `context` and `cause` hold nothing sensitive.
- [ ] Section 6 still holds. No weakening of auth, CSRF, SSRF, SQL, pickle, path confinement or audit verification. Any new client-supplied path or URL has its own confinement.
- [ ] State changes updated `_DOWNSTREAM_KEYS` and the preserve groups. Dataset mutations go through `update_active_dataset`, and analysis uses `get_analysis_dataframe`.
- [ ] Any new setting has a real consumer, a test and an entry in `.env.example`.
- [ ] Baseline-first and leakage-safe behaviour is preserved (Split Lock respected).
- [ ] If `frontend/` changed: I rebuilt it and committed `src/omniforge/ui/static_app` (including new files).
- [ ] If dependencies changed: I regenerated all locks (including `locks/`) with `uv`, not by hand.
- [ ] Version strings in `version.py`, `pyproject.toml` and `frontend/package.json` are aligned.

## 9. Maintaining this file

- Update `last_reviewed` / `next_review` and the companion reference in the same change.
- Verify claims by reading source (e.g. `grep -rn "always_compare_baselines" src/`), not by trusting this file. Confidence tags (`[V]`, `[X]`, `[C]`, `[P]`, `[!]`) are defined in reference section 0.1.
- Keep this file short. Move detail to `docs/AGENT_REFERENCE.md`; link to it instead of duplicating it.

*End of instructions. Full manual: `docs/AGENT_REFERENCE.md`.*
