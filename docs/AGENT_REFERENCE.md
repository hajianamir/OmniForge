
---
document: docs/AGENT_REFERENCE.md
title: OmniForge - Engineering Reference Manual for Coding Agents
project: OmniForge v4.0.0 (package: omniforge, import root: src/omniforge/)
audience: Codex and other AI coding agents
status: Active
owner: OmniForge maintainers
last_reviewed: 2026-11-27
next_review: 2027-02-27
supersedes: agent.md (PDF draft, 2025)
companion: AGENTS.md (repo root, auto-loaded by Codex; read that first)
---
# OmniForge - Engineering Reference Manual for Coding Agents
> **Scope:** `github.com/hajianamir/OmniForge` (package `omniforge`, import root `src/omniforge/`).
> **How to use:** `AGENTS.md` at the repo root is the auto-loaded entry point. This file is the full reference behind it. Read only the sections your task needs (see the task-routing table in `AGENTS.md`).
> **Revision note (2026-11-24):** Corrected and re-verified against HEAD `49b9058`. Fixes the prior draft's `AuditLogger` search miss (it lives in `core/security.py`), the `auto_bootstrap` default, CLI bind defaults, the `docker-build` tag, and the settings-consumption tables. See section 22 for the full correction log.
---
## 0. How to read this document
### 0.1 Provenance and confidence
This revision was produced by reading a **subset** of the repository on GitHub (the directory tree, `pyproject.toml`, `ci.yml`, `Makefile`, `version.py`, `core/security.py`, `api/main.py`, `api/auth.py`, `simulation/closed_loop.py`, `ui/state.py`, `core/workspace_session.py`, `api/workbench.py`, `core/exceptions.py`, `config/settings.py`, `data/ingestion.py`, `copilot/tools.py`, `domains/plugin_sdk.py`, `models/base.py`, and selected tests). **The repository was not cloned, the test suite was not run, and lint/type-check gates were not executed in this review.** Therefore:
| Tag | Meaning |
|---|---|
| **[V]** | Verified by reading the source in this review. |
| **[X]** | Verified by executing code or tooling **in a previous review**. Not re-executed here. |
| **[C]** | Carried over from the previous draft. Not re-read here, so treat it as a hypothesis. |
| **[P]** | Partially verified. The remainder needs confirmation. |
| **[!]** | A defect or inconsistency found in the repository. Collected in §20. |
Three working rules:
1. This file is a map and a list of invariants. **Read the complete module you are about to change before you
change it.**
2. Treat anything tagged **[C]**, **[P]** or **[!]** as a lead to confirm, not as a fact to rely on.
3. The test suite is the executable specification. Read `tests/` when you need to understand intended 
behaviour.

**Verification method for this revision:** claims marked **[V]** were re-checked by reading source at HEAD `49b9058` (including `core/security.py` for `AuditLogger`, `cli/main.py` for bind defaults, `config/settings.py` for defaults, `api/jobs.py` and `api/workbench.py` for settings consumers, and `copilot/tools.py` for the studio-route list). Gates were not re-executed in this pass; re-run them before starting (section 0.2).

### 0.2 Gate status (baseline to record before starting)
The repository was **not fully green** at the previously reviewed commit (`3d6de5b`) **[C]**:
| Gate | Result at `3d6de5b` **[C]** |
|---|---|
| `mypy src/omniforge` | Pass. |
| `ruff check src tests` | **19 findings** (ruff 0.17.6). |
| `pytest tests` | **1 failure** (`tests/e2e/test_workbench_spa.py::test_spa_static_bundle_gzip_and_all_13_studios_e2e` - `/s00/data_relationships` returned empty `topological_links`), **1 skip**. |

**Before you start, re-run the gates and record the current baseline** so you do not inherit blame for pre-existing failures.
### 0.3 Shared agent skills

- [OmniForge](../.agents/skills/omniforge/SKILL.md) covers repository conventions. It and `AGENTS.md` take precedence over generic recommendations.
- [Modern Web Guidance](../.agents/skills/modern-web-guidance/SKILL.md) is a vendored, optional reference for browser and frontend work. Use the shipped `guides/` when npm or network access is unavailable. Consult the relevant guide and validate browser support in the Workbench before adopting an API. This is not a build or runtime dependency.
- [`skills-lock.json`](../skills-lock.json) records the upstream source, revision, and skill hash. To update the guidance, refresh the project-local installation on a working branch and review the skill and lockfile together. The [upstream Apache 2.0 license](../.agents/third-party/modern-web-guidance.LICENSE) is included separately so the installed skill tree remains unmodified.

---
## 1. Project identity
- **Name / edition:** OmniForge - "Enterprise Analytics & AI Workbench". `src/omniforge/version.py` 
defines `__version__ = "4.0.0"`, `__api_version__ = "v2"`, `__schema_version__ = "2.0"` **[V]**.
- **Version alignment:** `pyproject.toml` and `frontend/package.json` also say `4.0.0` **[V]**. Keep all 
three in step.
- **Runtime:** Python **>= 3.12** (`requires-python`). CI runs 3.12 and 3.13. Classifiers advertise 3.12-3.15 
**[V]**.
- **Licence:** Apache-2.0.
- **Legacy name "UAP" is load-bearing.** Session-state keys are `uap_*`; 
`BaseOmniForgeModel.framework = "uap"` (written into saved metadata); Kafka default group `uap-consumer`; `.uap` is accepted as a legacy workspace extension (read-only). **Do not rename these casually** **[V]**.
---
## 2. Non-negotiable rules
1. **Python 3.12+.** Start every new module with `from __future__ import annotations`. At HEAD 156/191 modules have it; most exceptions are `__init__.py` files. Do not churn existing files solely to add it **[V]**.
2. **No casual hard dependencies.** Heavy/optional libraries (torch, xgboost, lightgbm, catboost, shap, 
optuna, statsmodels, pyarrow, duckdb, sqlalchemy, polars, pymongo, kafka, fastavro, codecarbon, transformers, cv2, spacy) are imported lazily inside functions and guarded. Raise `DependencyMissingError` only when the user explicitly requested that backend.
 - `core.utils.is_package_available()` uses `importlib.util.find_spec`; it proves the package is **installed**,
not that it **imports**. For heavy backends, prefer `try/except ImportError` around the real import **[V]**.
3. **Keep `import omniforge` cheap.** `omniforge/__init__.py` uses a PEP 562 lazy `__getattr__` with 
`_LAZY_EXPORTS`. `version.py` and `capabilities.py` must stay **stdlib-only**. **`omniforge.core` is NOT stdlib-only** - `core/security.py` imports pandas at module top **[V]**. Do not assume `import 

omniforge.core` is lightweight.
4. **Library errors are `OmniForgeError` subclasses** (§6.4). Give them `suggestions=[...]`, optional 
`context={...}` and `cause=exc`. Do not raise bare `ValueError`/`RuntimeError` from library layers. `OmniForgeError.to_dict()` includes `context` and `cause`; **never put secrets, PII or filesystem internals in either field** **[V]**.
->5. **API payloads must be JSON-safe.** Wrap in `core.utils.to_json_object / to_json_safe` (NaN/Inf/NaT 
-> ->`None`, numpy Python, pydantic `model_dump`, sets sorted lists, `Path` `str`) **[V]**.
6. **Security is fail-closed.** Never weaken SSRF validation, read-only SQL, pickle bans, path confinement,
CSRF, tenant/session namespacing, audit-chain verification, or production secret requirements (§17). Add no "temporary" bypasses.
7. **Never log or audit raw PII or secrets.** Use `get_logger(__name__)` (it redacts); record column names 
and counts only.
8. **Reproducibility.** Pass `random_seed` (default `DEFAULT_RANDOM_SEED = 42`). 
`BaseOmniForgeModel.fit` calls `set_global_seed`. Note that `set_global_seed` writes `PYTHONHASHSEED` at run time, which only affects child processes **[V]**.
9. **Baseline-first modeling is a product invariant.** Any model-comparison or AutoML path must 
compare against baselines (`BaselineBenchmarkSuite`) and report lift over the best baseline. `ModelingSettings.always_compare_baselines` is **declared but never read** in `src` **[V][!]**, so enforcement lives in call sites. Do not assume flipping the setting changes anything.
10. **Leakage safety is a product invariant.** Fit transformers on train only, respect the Split Lock (§14.4), 
and use `LeakageSafeSplitter` / `LeakageGuard`.
11. **Do not hand-edit generated artifacts:** `requirements.lock`, `requirements-build.lock`, 
`locks/*.lock`, and the compiled SPA in `src/omniforge/ui/static_app/`. Regenerate them (§3.1, §15.4).
12. **Keep the gates green for the code you touch** (§3.3). Add tests under `tests/unit`, `tests/integration` 
or `tests/e2e` with registered markers only.
13. **Do not trust a settings field because it exists.** Most of `config/settings.py` is declared but unused 
(§7). Grep for a consumer before relying on a setting.
---
## 3. Environment, commands, quality gates
### 3.1 Install and locks
```bash
# Developer environment (matches CI)
pip install -e ".[dev,api,ui,reporting,data,forecasting]"

# Everything except the "experimental" extra (this is what `make install-dev` runs)
pip install -e ".[all]"
# Hardened runtime from the hash-pinned lock
pip install --require-hashes -r requirements.lock && pip install -e . --no-deps
```
- **Extras** (`pyproject.toml`) **[V]**: `api, redis, ui, reporting, data, connectors, mlops, forecasting, 
deep, boosting, automl, automl-java, foundation, responsible, nlp, vision, audio, build, dev, experimental, all`.
- **Core dependencies** (always installed, 20) **[V]**: numpy, pandas, scipy, scikit-learn, pydantic, 
pydantic-settings, PyYAML, plotly, networkx, joblib, typer, rich, Jinja2, httpx, psutil, structlog, python-dateutil, openpyxl, nbformat, defusedxml.
- **FastAPI, uvicorn, pyjwt and cryptography are not core.** They live in the `api`/`ui` extras.
- **Lock inputs** **[V]**: `requirements.in` is the runtime lock input; `requirements-build.in` is the build
toolchain input; `requirements.txt`/`requirements-build.txt` are shims containing `-r <lock>`; `locks/*.lock` are per-tier and per-pack locks generated by `scripts/build_locks.py`.
- **Regenerate, never hand-edit:**
 ```bash
 uv pip compile requirements.in --generate-hashes --python-version 3.12 --universal -o requirements.lock uv pip compile requirements-build.in --generate-hashes --python-version 3.12 --universal -o requirements-build.lock
 ```
- **Staleness is not enforced by CI.** The `lock-verify` job proves the lock installs and imports, not that it 
matches the `.in` files. You are responsible for regeneration **[V]**.
### 3.2 Entry points and CLI
- `omniforge = "omniforge.cli:app"`. `cli/__init__.py` does `from omniforge.cli.main import cli, main; app 
= cli` **[V]**, so the entry point is valid.
- `omniforge-desktop = "omniforge.launcher:main"`.
- `python -m omniforge` runs `cli.main.main` **[V]**.
- **CLI commands** (`cli/main.py`) **[V]**: `version, domains, quality, clean, eda, automl, drift, copilot, 
check, desktop, monitor, sbom, serve, ui`.
 - `serve` and `ui` (CLI) default to `--host 127.0.0.1` (`cli/main.py`). `desktop` and the launcher also default to `127.0.0.1`. Only `make run-api` / `make run-ui` and `.env.example` (`OMNIFORGE_API_HOST`/`OMNIFORGE_UI_HOST`) bind `0.0.0.0`. Do not conflate the two **[V]**.
- **Launcher** (`omniforge.launcher`) **[V]**: `--mode {full,ui,api,cli,check}`, `--host`, `--api-port`, 
`--ui-port`, `--no-browser`, `--json`. `launcher.main()` is the **only** place `.env` is loaded; `uvicorn omniforge.api.main:app` and the `omniforge` CLI do **not** read `.env`.
### 3.3 Make targets and gates
`make install | install-dev | test | test-unit | test-integration | test-e2e | lint | typecheck | format | run-api | run-ui | docker-build | docker-up | clean` **[V]**.
- `run-api`: `uvicorn omniforge.api.main:app --host 0.0.0.0 --port 8000 --reload`.
- `run-ui`: `python -m omniforge.launcher --mode ui --host 0.0.0.0 --ui-port 8501 --no-browser`.
- `docker-build` tags `omniforge-enterprise:4.0.0`, aligned with the package version **[V]**. (The prior draft's `2.0.0` staleness is resolved.)

- **ruff:** line length 120, target `py312`, rules `E, F, I, W, B, UP`, ignoring `E501`/`B008`.
- **mypy:** not `strict`, but sets `disallow_untyped_defs`, `disallow_any_generics`, 
`disallow_untyped_decorators`, `check_untyped_defs`, `warn_return_any`, `warn_unused_ignores`. A stale `# type: ignore` is itself an error.
- **pytest:** `minversion 9.0`, `-ra --strict-markers`, markers `unit, integration, e2e, slow`. An 
unregistered marker fails the run.
### 3.4 CI (`.github/workflows/ci.yml`) **[V]**
| Job | What it does | Limits |
|---|---|---|
->| `quality-and-tests` (3.12, 3.13) | `ruff check src tests` `mypy src/omniforge` `pytest -q tests 
--cov=omniforge` | Installs `.[dev,api,ui,reporting,data,forecasting]`. |
| `lock-verify` | Installs `requirements.lock` with `--require-hashes`, then smoke-imports `omniforge, 
fastapi, pandas, sklearn, numpy, duckdb, cryptography, pydantic` | Does not check lock freshness. |
| `e2e` | Node 22: `npm ci && npm run build`, then `git diff --quiet -- src/omniforge/ui/static_app`, then 
`pytest -q tests/e2e` | `git diff` catches modified/deleted tracked files, **not** newly added untracked ones. |
| `latency-budget` | `python scripts/latency_budget_benchmark.py` | Script not read. |
A second workflow, `windows-installer.yml`, exists and was not read.
---
## 4. Repository layout
Top level **[V]**:
```
.env.example .dockerignore .gitattributes .gitignore Makefile pyproject.toml README.md CHANGELOG.md PLAN_PHASE6_VERIFICATION.md LICENSE requirements{.in,.txt,.lock} requirements-build{.in,.txt,.lock} locks/*.lock per-tier / per-pack locks (scripts/build_locks.py) configs/{default.yaml, logging.yaml, domains/<18>.yaml} docker/{Dockerfile, docker-compose.yml, entrypoint.sh} docs/ tests/{unit,integration,e2e} examples/ notebooks/ frontend/ Vite + React + TypeScript source installer/{provision.py, omniforge_installer_tiers.iss} packaging/{build_exe.py, entry_cli.py, entry_desktop.py, pyinstaller/*.spec, inno_setup/, assets/} scripts/{build_locks, build_bootstrap, check_iss, generate_sbom, latency_budget_benchmark, make_launcher_shim, run_benchmark}.py install.bat start.bat build_exe.bat build_installer.bat (CRLF enforced by .gitattributes)
```
`src/omniforge/` packages **[V]**:
| Package | Contents |
|---|---|

| `api/` | `main, auth, jobs, schemas, workbench, studios/` |
| `automl/` | `backends, nas, optimizer, pipeline_search` |
| `causal/` | `inference` |
| `cli/` | `main` |
| `config/` | `settings, loader` |
| `copilot/` | `agent, memory, prompts, tools` |
| `core/` | `carbon, constants, contracts, exceptions, logging, performance, rate_limit, sbom, security, 
types, utils, workspace_session` |
| `data/` | `cleaning, connectors, feature_store, format_audit, ingestion, intake_audit, multimodal, 
quality, splitting, synthetic, validation, versioning` |
| `domains/` | `plugin_sdk` plus 18 vertical modules |
| `eda/` | `automated_eda, diagnostics, insights, profiler, statistics, visualization` |
| `evaluation/` | `backtesting, equations, fairness, interpretability, metrics, model_cards, robustness, 
validation` |
| `features/` | `embeddings, engineering, selection, store_client` |
| `forecasting/` | `engine` |
| `mlops/` | `continuous_training, drift, future_scoring, job_queue, monitoring, registry, root_cause, 
serving, supervisory_monitor, tracking` |
| `models/` | `base, baselines, classical, ensemble, nonparametric, registry`, plus `anomaly, causal, 
clustering, computer_vision, deep, graphs, mining, nlp, optimization, recommendation, simulation, survival, time_series` |
| `optimization/` | `solvers` |
| `pipelines/` | `base, builder, dag, executor` |
| `reporting/` | `charts, exporters, formulas, generator, narrative, sections` |
| `simulation/` | `closed_loop, engines` |
| `ui/` | `app, state, static_app/` |
Top-level modules: `capabilities.py`, `launcher.py`, `version.py`, `__main__.py`.
---
## 5. Architecture and layering
Intended layer order (lower layers must not import higher ones; use function-local imports for unavoidable cycles):
```
version, capabilities (stdlib-only)
->> -> core con fig data eda / features models evaluation / automl / forecasting / causal /
-> optimization / simulation / pipelines mlops / reporting / domains / copilot api / ui / cli / launcher
```
**Actual coupling that the diagram hides** **[V]**:
- `api/workbench.py` imports `omniforge.ui.state` at module level.
- `ui/__init__.py` imports `ui.app`, which imports `api.main`.
- `api/main.py` runs `app = create_app()` **at import time**.
- Consequence: importing `omniforge.ui.state` constructs the whole FastAPI app. The cycle works only 

because of submodule import order. **Do not add top-level imports of `omniforge.api` or `omniforge.ui` into `core` or `data`.** Do not reorder these imports without running the e2e tests.
- Studio routers lazy-import the workbench in a `_wb()` helper to avoid a circular import. **Keep that 
pattern.**
- `core.security` imports `config.settings` lazily inside `_default_tenant()` (wrapped in `try/except`, 
falling back to `"default-tenant"`). **Keep it lazy.**
- `core.security` imports **pandas at module top**, so `omniforge.core` is not stdlib-only **[V]**.
- `mlops.registry` lazily imports settings and `AuditLogger` inside `_audit_registry_event`.
- `api/auth.py` imports the private helpers `_is_production_env` and `_resolve_or_persist_secret` from 
`core.security`. Treat those names as internal API when refactoring **[V]**.
---
## 6. `omniforge.core`
### 6.1 `constants.py` **[V]**
- `DEFAULT_RANDOM_SEED = 42`, `DEFAULT_TEST_SIZE = 0.2`, `DEFAULT_CV_FOLDS = 5`.
->- `SUPPORTED_FILE_FORMATS` maps extension to canonical format. It includes `.pkl/.pickle pickle`, 
which ingestion refuses by default (§8.1). `.joblib` is not in the table but is blocked at upload.
- `SUPPORTED_DB_SCHEMES` has 11 entries. `SUPPORTED_CLOUD_SCHEMES` has 10 entries and 
includes `http` and `https`.
- `CARBON_INTENSITY_BY_COUNTRY` covers 17 ISO codes plus `GLOBAL = 475`.
- `SENSITIVE_KEY_PATTERNS` are matched by key substring for log redaction: password, passwd, secret, 
token, api_key, apikey, access_key, private_key, authorization, auth_header, credential, ssn, credit_card, bearer, connection_string.
- `RBAC_ROLE_PERMISSIONS`: roles `admin`, `platform_admin`, `data_scientist`, `mlops_engineer`, 
`analyst`, `viewer`. Permissions include `data:{read,write,delete}`, `model:{read,train,deploy,promote,delete}`, `pipeline:{read,execute,write}`, `mlops:{read,write}`, `copilot:use`, `report:generate`, `audit:read`, `security:manage`. `platform_admin` has the same set as `admin`; its extra privilege is the cross-tenant bypass, enforced in `AccessController`. `model:promote` belongs to `admin`, `platform_admin` and `mlops_engineer`; `data_scientist` has `model:deploy` but not `model:promote`.
- `BUILTIN_DOMAINS` lists 18 slugs. It is exported but **not used anywhere else** **[V][!]**.
### 6.2 `types.py` **[V]**
- `SerializableStrEnum(StrEnum)` overrides `__reduce_ex__` to pickle by value, so the restricted unpickler
never needs `builtins.getattr` (audit finding SEC-001). On Python 3.12+ the stdlib already pickles enums by value, so this is a belt-and-braces guarantee. Keep using it for any enum that can end up in a pickled artifact.
- Enums: `TaskType` (15 members), `DataModality` (11), `ModelStage` (`"None"`, `"Staging"`, 
`"Production"`, `"Archived"` - capitalised), `ValidationSeverity`, `DriftStatus`.
- Models: `CarbonReport`, `ColumnSchema`, `DatasetMetadata` (`ds_<12hex>`), `ModelMetadata` 
(`mdl_<12hex>`), `AuditEvent` (`previous_hash`, `event_hash`, `chain_mac`), `Principal` (`subject`, `roles` default `["analyst"]`, `tenant_id`, `attributes`, `scopes`).
- `ModelMetadata.framework` defaults to `"omniforge"`, but `BaseOmniForgeModel` overwrites it with 
the class attribute (`"uap"`).

### 6.3 `contracts.py` **[V]**
in- `AnalysisSpec`: `task_kind` {descriptive, hypothesis_test, prediction, time_series_forecast, clustering, incausal_inference, survival, optimization}; `analysis_mode` {exploratory, confirmatory}; further fields include `independent_unit`, `target_column`, `target_timestamp_column`, `time_column`, `group_column`, `id_column`, `forecast_horizon ≥ 1`, `treatment_column`, `outcome_column`, in in`estimand {ATE, ATT, ATC, CATE}`, `primary_metric="auto"`, `decision_threshold [0,1]`, `sensitive_attributes`, `random_seed=42`.
- `LeakageGuard.validate_split(train, test, *, group_col, time_col, seed)` returns `LeakageGuardReport`. It 
checks index overlap, group overlap and `max(train time) <= min(test time)`.
- `ResultBundle.bootstrap_ci(values, n_boot=500, alpha=0.05, seed=42)` is a percentile bootstrap of the 
mean. It raises `ValueError` when there are no finite values.
### 6.4 `exceptions.py` **[V]**
`OmniForgeError(message, *, error_code, severity, suggestions, trace_id, context, cause)` provides `to_dict()` and a rich `__str__`. `ErrorSeverity` is low/medium/high/critical. All codes carry the `OmniForge-` prefix.
| Class | Code | Default severity |
|---|---|---|
| `OmniForgeError` | CORE-000 | high |
| `ConfigurationError` | CFG-001 | high |
| `SecurityError` | SEC-001 | critical |
subset| `AuthorizationError` ( Security) | SEC-002 | high | subset| `ComplianceError` ( Security) | SEC-003 | critical |
| `DataError`, `DataIngestionError`, `DataCleaningError`, `DataValidationError` | 
DATA-000/001/002/003 | high (validation: medium) |
| `FeatureEngineeringError` | FEAT-001 | medium |
| `ModelError`, `ModelNotFittedError`, `ModelTrainingError`, `ModelSerializationError` | 
MOD-000/001/002/003 | high (not fitted: medium) |
| `EvaluationError` | EVAL-001 | medium |
| `AutoMLError`, `ForecastingError`, `CausalInferenceError`, `OptimizationError`, `SimulationError`, 
`PipelineError`, `MLOpsError` | AML/FCST/CAUS/OPT/SIM/PIPE/MLOPS (-001) | high |
| `CopilotError`, `DomainPluginError`, `ReportingError`, `DependencyMissingError` | 
COP/DOM/REP/DEP (-001) | medium |
->- **HTTP mapping** in `api/main.py` **[V]**: any `SecurityError` **403**; any other 
->`OmniForgeError` **400**; unhandled **500** with an `error_id` and a generic message.
### 6.5 `logging.py` **[V]**
- `get_logger(name)` returns an `OmniForgeLoggerAdapter`. Use an event name plus keyword arguments,
e.g. `logger.info("model_fitted", model=..., rows=...)`.
- Redaction is by key (`SENSITIVE_KEY_PATTERNS`) and by value regex (`Bearer ...`, `sk-...`, `ghp_...`, 
SSN-shaped, 13-16-digit card-like). Output is `***REDACTED***`.
- The trace id and tenant context are context variables.

- `structlog` is pre-declared as `ModuleType | None` for mypy. Do not "simplify" this.
### 6.6 `security.py` **[V]**
- **Production detection.** `_is_production_env()` is true when `OMNIFORGE_ENVIRONMENT` (or 
`ENVIRONMENT`) is `prod`/`production`. It reads `os.environ` directly and does **not** consult `settings.environment`.
->- **Key store.** `_resolve_or_persist_secret` resolves explicit env vars (non-production only) a 
generated 0600 file under `$OMNIFORGE_KEY_DIR` or `~/.omniforge/keys`. In production it raises `SecurityError` if the secret is missing, too short, or contains `change-me`, `do-not-use-in-prod`, `uap-development`, `secret-key` or `changeme`. Dev keys are per host; **signed artifacts and workspaces are not portable between machines** unless keys are configured explicitly.
- **Pseudonymization.** `resolve_pseudonym_salt(tenant)` tries 
`OMNIFORGE_PSEUDONYM_SALT__<TENANT>`, then `OMNIFORGE_PSEUDONYM_SALTS`, then `OMNIFORGE_PSEUDONYM_SALT`. Minimum 16 bytes (32 in production). `pseudonymize_value` is HMAC-SHA256 truncated to 16 hex chars.
- **`DataFramePIIGuard`** (modes `pseudonymize`, `block`, `off`): regex detection over string-dtype 
columns using `is_string_dtype`, so pandas `StringDtype` is covered. Audit events record only column names and counts; audit failures are swallowed on purpose. **`PIIRedactor` is a separate mechanism** used by `ComplianceValidator`.
- **`EncryptionManager`:** AES-256-GCM with a random 96-bit nonce. Envelope is 
urlsafe-b64(`b"OF2GCM\x01"` + nonce + ciphertext). Legacy Fernet tokens (`gAAAAA`) decrypt only when `associated_data is None`. Key from `OMNIFORGE_ENCRYPTION_KEY` or `OMNIFORGE_SECURITY__ENCRYPTION_KEY`. Requires `cryptography`.
- **`ArtifactSigner`:** HMAC-SHA256 with a domain-separated key. Env vars: 
`OMNIFORGE_ARTIFACT_SIGNING_KEY`, `OMNIFORGE_SECURITY__ARTIFACT_SIGNING_KEY`, with `OMNIFORGE_JWT_SECRET_KEY` as fallback. Contexts: `omniforge:model:v1`, `omniforge:workspace:v2`, `omniforge-audit-event-v2`, `omniforge-audit-checkpoint-v1`.
- **`safe_joblib_load_bytes`:** restricted `NumpyUnpickler` subclass. Enforces a module allow-list plus a 
denylist of root modules, submodules and symbol names. Call it **after** HMAC verification.
 - [!] It imports the private joblib names `NumpyUnpickler` and `_validate_fileobject_and_memmap`, 
which can break on a joblib upgrade.
 - [!] `torch` is allow-listed broadly, apart from a few submodules.
- **`JWTManager`:** uses PyJWT when installed, otherwise a stdlib HS256 path. The secret comes from 
`OMNIFORGE_JWT_SECRET_KEY` or `OMNIFORGE_SECURITY__JWT_SECRET_KEY`. `verify_token` returns a `Principal` and defaults missing roles to `["viewer"]`. The `jwt_algorithm` and `jwt_expiration_minutes` settings are **not** passed to the `JWTManager` that `api/auth.py` builds.
- **`AccessController`:** `get_permissions` returns the union of the role permissions **and 
`principal.scopes`**. A signed token carrying `scopes: ["model:promote"]` therefore grants that permission. Cross-tenant checks require `platform_admin`. Every cross-tenant decision (allowed or denied) is audited; an allowed decision is refused if the audit write fails.
- **`RequestBodySizeLimitMiddleware`:** checks `Content-Length` first, then counts streamed bytes. 
Aborts with the internal `_PayloadTooLarge(BaseException)`, deliberately deriving from `BaseException` so that `except Exception` blocks cannot turn a 413 into a 500.
- **`SlidingWindowRateLimitMiddleware`:** keyed by the ASGI client host, **not** `X-Forwarded-For`. 
Returns 429 + `Retry-After`. Exempts `/health` and `/api/v1/health` only. A value of 0 disables it.
- **`AuditLogger` (`core/security.py:1506`) [V]:** hash-chained, HMAC-signed JSONL audit trail. `__init__` caches `_last_hash` via `_load_last_hash()` and `_record_count` via `_count_records()`; `log_event()` builds the next event from the cached `self._last_hash` and appends without re-reading the tail and without a lock. The chain-fork defect (D1) is therefore **confirmed**: concurrent writers (e.g. a fresh `AuditLogger` per event in `mlops/registry.py` / `mlops/continuous_training.py` alongside the long-lived instance in `api/auth.py`) can fork the chain. Fix by re-reading the tail under a lock (plus an inter-process file lock) on every append. The prior draft's "could not locate" note was a search miss and is superseded.
- **`ComplianceValidator`** screens a limited set of GDPR, HIPAA and SOX indicators. It reports unsupported profiles, missing evidence, PII scan coverage and an assessment status. Strict mode requires the implemented checks to pass with sufficient evidence. A passed screen leaves the legal compliance conclusion unknown.
### 6.7 `carbon.py` **[V]**
- CPU power = `TDP*(0.15 + 0.85*util)`. RAM power = `max(0.2, rss_gb*0.3725)`. GPU power = `0.6*TDP` 
when CUDA is available.
- `energy_kWh = (CPU + RAM + GPU) * PUE * hours / 1000`.
- `emissions_g = energy_kWh * grid_intensity`.
- `green_score = 100*exp(-emissions_g/50)`. It is computed with the literal `2.718281828`.
- An optional `codecarbon` result overrides the estimate. Utilisation counts the current process only.
### 6.8 `utils.py`, `performance.py`, `workspace_session.py`
- `utils.py` **[V]**: `set_global_seed`, `is_package_available`, checksum helpers, 
`optimize_dataframe_dtypes`, the JSON helpers, `ensure_directory`, `timer`, `detect_hardware_profile`, `infer_semantic_column_type`.
- `performance.py` **[C]**: `GLOBAL_DF_CACHE` / `DataFrameCache`, `compute_df_fingerprint`, 
LTTB downsampling, `paginate_dataframe`, `compact_scatter_payload`, `TTLCache`.
 - [!] The module has E402 (imports after code) and UP046 lint findings **[C]**.
- `workspace_session.py` **[V]**: see §14.6.
---
## 7. Configuration (`omniforge.config`)
### 7.1 Facts **[V]**
- `OmniForgeSettings(BaseSettings)` uses `env_prefix="OMNIFORGE_"`, `env_nested_delimiter="__"`, 
`case_sensitive=False`, `extra="allow"`, and **no `env_file`**.
- `get_settings()` is an `lru_cache` over `OmniForgeSettings()`. It reads defaults plus environment 
variables **only**.
- `config/loader.py` provides `ConfigLoader.compose` / `load_config` (alias `load_settings`): merges 
`configs/default.yaml`, an optional domain YAML, extra files, `${ENV:default}` interpolation and Hydra-style dotlist overrides. **It is never called by runtime code.** Consequently `configs/default.yaml` has **no effect** unless a caller invokes `load_config` explicitly.
- `.env` is loaded **only** by `launcher.main()`. pydantic-settings has no `env_file` configured, so 
`uvicorn omniforge.api.main:app` does not read `.env`.
### 7.2 Settings that are actually consumed **[V]**
| Setting | Consumer |

|---|---|
| `default_tenant_id` | `core.security`, `api.auth` |
| `paths.audit_log_path` | `api.auth`, `mlops.registry`, `mlops.continuous_training` |
| `paths.workspace_dir` | `data.connectors` (allowed local DB roots) |
| `security.api_key` | `api.auth` |
| `security.auto_bootstrap_local_ui` | `api.auth` |
| `api.cors_origins`, `api.cors_allow_credentials` | `api.main` |
| `api.rate_limit_*` | `api.main` via `core.rate_limit.build_rate_limiter` |
| `api.max_request_body_bytes` | `api.main`, `workbench.read_upload_bounded` |
| `api.csp_report_only_*`, `api.hsts_*` | `api.main` security-header middleware |
| `jobs.*` (`max_workers`, `max_history`, `max_queue_depth`, `db_path`, `worker_timeout_seconds`) | `api.jobs` (job manager construction) **[V]** |
| `paths.feature_store_dir` | `api.workbench` (tenant feature-store root; default tenant only) **[V]** |
### 7.3 Declared but not consumed by any runtime code **[V][!]**
- `logging.*`, `compute.*`, `carbon.*`, `data.*`, `modeling.*` (including `always_compare_baselines`), 
`automl.*`, `mlops.*` (including `drift_psi_*` and `retrain_on_critical_drift`), `copilot.*`, `ui.*`.
- `security.enable_rbac`, `jwt_*`, `encryption_key`, `artifact_signing_key`, `encrypt_artifacts_at_rest`, 
`pii_detection_enabled`, `compliance_profiles`.
- `paths.{artifact_store_dir, cache_dir, tracking_db_path, registry_db_path}` (note: `feature_store_dir` **is** consumed - see 7.2). Registries 
and trackers hard-code `./.omniforge_workspace/...` relative to the **current working directory** (`mlops/registry.py`, `mlops/tracking.py`). Run the server from a stable directory. `ensure_workspace_dirs()` also creates the configured paths, so the settings are not entirely inert **[V]**.
- `api.host` and `api.port`. The real bind addresses come from the CLI, the launcher and the Makefile.
### 7.4 Defaults worth knowing **[V]**
- CORS allows `http://localhost|127.0.0.1` on ports 8501, 8000 and 5173. `cors_allow_credentials=True` is 
forced off if `"*"` is in the origin list (CWE-942).
- The CSP is **report-only** by default. HSTS is on (max-age 31536000, includeSubDomains). The 
request-body cap is 256 MiB.
### 7.5 Adding a setting
Add the field to the right sub-model with a safe default, and keep it overridable as `OMNIFORGE_<SECTION>__<FIELD>`. Mirror it in `.env.example`. **Wire a consumer and add a test**, otherwise it is another dead setting. Never put a secret default in code.
---
## 8. Data layer (`omniforge.data`)
### 8.1 `ingestion.py` and `connectors.py` **[V]**, security choke-points
**`DataIngestionEngine(*, optimize_dtypes, compute_checksum, default_encoding, tenant_id, allow_unsafe_pickle=False)`**
-> `ingest(source, ...)` routes: in-memory `kafka://` HTTP(S) API DB URLs or 
->`.db/.sqlite/.sqlite3/.duckdb` cloud/HTTP file schemes via fsspec local files by extension.
- Also provides `load_file`, `list_excel_sheets`, `ingest_excel_sheets` (combine modes `single`, `concat`, 
`merge`), `ingest_chunks`, `build_metadata`.

- **The generic `ingest()` does not confine local paths.** Local-path confinement exists only for Copilot 
ingestion (§12) and local DB files (§8.1 connectors). **Any new code path that takes a path from a client must add its own confinement.** **Invariants in this file:**
- Pickle and HDF5 require explicit trusted-local opt-ins. Remote HTTP and API upload paths reject both formats.
- XML is parsed with `defusedxml`. Documents containing `<!DOCTYPE` or `<!ENTITY` are rejected.
- SQLite is opened read-only through `file:...?mode=ro` plus an authorizer.
- DuckDB is opened with `read_only=True` and `enable_external_access=False`.
- Table names go through an allow-list lookup and `_quote_sql_identifier` (rejects empty, NUL and 
over-128-character names, doubles quotes).
- HTTP goes through `_fetch_http_safe`: `follow_redirects=False` with a manual loop of at most 5 hops; 
`validate_outbound_http_url` on the URL **and on every hop**; cross-origin redirects strip `authorization`, `cookie`, `proxy-authorization`, `x-api-key`; 256 MiB response cap.
- `_sanitize_uri` redacts credentials in lineage and logs.
**`data/connectors.py`** supplies `validate_outbound_http_url`, `validate_readonly_sql` (single `SELECT` or `WITH`, comments stripped, stacked statements rejected), `make_sqlite_authorizer`, remote-DB host validation, local DB path confinement, and the Excel/SQL/Access helpers.
- Local DB files must live under `paths.workspace_dir`, `./.omniforge_workspace`, or 
`OMNIFORGE_ALLOWED_DB_DIRS`. Paths containing `?` or `#` are rejected.
- SSRF policy: metadata hosts are blocked; private/loopback addresses are blocked unless allow-listed or 
`OMNIFORGE_ALLOW_PRIVATE_*_HOSTS` is true.
- [!] **Known limitation (D6):** DNS is resolved at validation time and again at connect time. There is no 
IP pinning, so DNS rebinding can race the check. Do not describe the SSRF defence as complete.
### 8.2 `splitting.py` **[C]**
ininin- `SplitConfig(strategy {auto, strati fied, random, group, time_series}, test_size (0.05, 0.60), val_size [0, 0.40), cv_folds 2-20, group_col, time_col, gap ≥ 0, regression_stratify_bins 2-20, strict_duplicates, random_seed=42)`.
- `LeakageSafeSplitter(config).split(df, *, target_col)` returns a `SplitResult` that includes `audit_report`.
->- Strategy resolution: time column `time_series`; group column `group`; discrete target with min class
->count ≥ 2 `stratified`; else `random`.
- Positions are taken over `df.reset_index(drop=True)`. See §14.4 for how the Split Lock maps them back.
### 8.3 Other modules **[C]**
->`cleaning.py` (`CleaningConfig`, `DataCleaner.fit_transform (df, report)`), `quality.py` 
(`DataQualityProfiler().profile(df)` over six dimensions), `format_audit.py`, `intake_audit.py`, `synthetic.py`, `versioning.py`, `feature_store.py`, `validation.py`, `multimodal.py`.
---
## 9. Models layer

### 9.1 `base.py::BaseOmniForgeModel` **[V]**
- Constructor: `(*, name, task_type, random_seed=42, track_carbon=True, country_iso="DEU", 
**hyperparameters)`.
- Subclasses implement `_fit`, `_predict` and optionally `_predict_proba`.
- `fit()` seeds globally, prepares arrays, infers `classes_`, runs `_fit` inside a `CarbonTracker`, and wraps 
failures in `ModelTrainingError`. It then fills `metadata_`.
- **`save(path, *, encrypt=False, encryption_key=None)`:**
 1. `joblib.dump` straight to the target path.
 2. Optional AES-GCM encryption, rewriting the file.
 3. SHA-256 into the metadata.
 4. HMAC-SHA256 (context `omniforge:model:v1`) written to `< file>.meta.json` as 
`artifact_hmac_sha256`.
- **`load(path, *, encrypted, encryption_key, require_signature=True)`:** requires the `.meta.json` 
sidecar, verifies the HMAC over the stored bytes, decrypts if asked, loads with `safe_joblib_load_bytes`, and type-checks the result.
- [!] **The class docstring says "Atomic save" (D8).** The code is not atomic: it dumps directly to the target 
and writes the sidecar afterwards. The HMAC covers the artifact bytes only, not the rest of the sidecar metadata.
### 9.2 `baselines.py` and `registry.py` **[C]**
- Baselines: dummy, logistic, decision-stump, ridge and lasso variants, plus three non-parametric analytic 
models from `nonparametric.py` (distance-weighted KNN, Nadaraya-Watson, penalised-spline GAM). `BaselineBenchmarkSuite` runs them.
- `ModelCatalog` / `create_model(name, task_type=BINARY_CLASSIFICATION, **kw)`: baselines, 
classical models, voting and stacking ensembles, and lazily imported deep models.
- To add a model: subclass `BaseOmniForgeModel`, then call `default_model_catalog.register("key", 
builder)`. Deep models go through `_lazy_deep` so that torch is not imported when the catalog is built.
### 9.3 Subpackages **[C]**
`clustering`, `anomaly` (flag convention: `-1` means anomaly), `mining`, `deep`, `graphs`, `nlp`, `computer_vision`, `recommendation`, `survival`, `time_series`, `causal`, `optimization`, `simulation`. The last three are thin re-export shims. **Read each before editing.**
---
## 10. MLOps
### 10.1 `registry.py::ModelRegistry`
- SQLite database at `./.omniforge_workspace/model_registry.db`, artifacts under `.../artifacts/registry`. 
Table `model_versions` has primary key `(model_name, version)` **[C]**.
- Use the `_sqlite_conn` context manager. It commits or rolls back **and closes**. A bare `with 
sqlite3.connect()` leaks handles **[V]**.
- Model names must match `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$` and must not contain `..` **[V]**.
- **Fairness gate** **[V]:** promotion to Production is blocked when `fairness_passed is False` or when 

there is no audit and `require_fairness_gate` is on (constructor arg or `OMNIFORGE_REQUIRE_FAIRNESS_FOR_PRODUCTION`). A non-empty `override_reason` bypasses the block but is stored as a tag and audited. Blocks are audited too. The studio maps messages containing "fairness gate" to HTTP 409.
- **Audit helper** **[V]:** `_audit_registry_event` (and its twin in `continuous_training.py`) builds a 
**new `AuditLogger` per event** on `paths.audit_log_path`. Since `AuditLogger` caches `_last_hash` without re-reading the tail (see 6.6), this is the multi-writer scenario in D1. Do not rely on the audit chain being fork-free until D1 is fixed.
### 10.2 `drift.py`, `root_cause.py` **[C]**
- `DriftDetector(psi_warning=0.10, psi_critical=0.25, ks_alpha=0.05, n_bins=10)`. The `mlops.*` settings of
the same names are **not** wired into it.
- `DriftRootCauseAnalyzer` and `CausalDiscovery` implement the C5 causal drift attribution: a PC-style 
skeleton, temporal orientation, and a root cause defined as a drifted node with no drifted ancestors. Fully offline.
### 10.3 Other **[V by grep]**
- `continuous_training.py` reads `OMNIFORGE_CT_DISABLED` and 
`OMNIFORGE_CT_AUTO_PROMOTE`; the registry reads `OMNIFORGE_AUTO_ROLLBACK_DISABLED`.
- `tracking.py`, `serving.py`, `monitoring.py`, `future_scoring.py`, `supervisory_monitor.py` were not 
re-read.
---
## 11. Simulation: `closed_loop.py` (C7 MPC) **[V][X]**
- A receding-horizon controller solved with `scipy.optimize.minimize(method="SLSQP")`. Decision vector 
is `[U (H·n_u), X (H·n_x)]`, plus slack variables in the softened solve. It exposes `solve`, `run_closed_loop`, `persist`/`restore`, and `build_default_twin`. HTTP endpoint: `POST /api/workbench/s09/twin-control`.
- [!] **Gradient defect (D2), confirmed.**
 - `_objective` sums only tracking and input-effort terms over `X[k], U[k]` **[V]**.
T - `_objective_grad` adds a dynamics-coupling term, `2·A Q (A x_k + B u_k - r_{k+1})` for `k < H-1`, that is not part of `_objective` **[V]**.
 - `_objective_grad_with_slack` is worse: the coupling loop is nested inside the per-`k` loop, and 
`grad_U`/`grad_X` are overwritten each iteration before being indexed as if they were arrays **[V]**.
 - `solve` passes this gradient to SLSQP via `jac=`.
 - The effect on solution quality was **not measured**. Equality dynamics constraints may mask it.
 - **Fix with a finite-difference gradient test in `tests/unit` first, then correct the gradient (or drop 
`jac=`).**
---
## 12. AI Copilot (`omniforge.copilot`)

- `OmniForgeCopilotAgent(*, provider="local", model, temperature, ollama_base_url, azure_endpoint, 
memory)` **[C]**. With a non-local provider and credentials present, it calls the LLM. Any exception falls back to the deterministic local planner. Credentials come from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`, or Ollama. The LLM tool protocol is a JSON object `{"tool": ..., "arguments": {...}}`. `_chat_local_planner` is a keyword router over lower-cased text; read its body before changing routing.
- **`CopilotToolRegistry` registers 24 tools** **[V]**:
 - `ingest_data, generate_demo_data, profile_quality, clean_data, run_eda, train_automl, run_forecast, 
explain_model, audit_fairness, check_drift, run_causal, generate_pipeline_code, list_domains, workspace_status`.
 - `root_cause_diagnosis, twin_control, split_lock, pii_audit, pareto_automl, domain_validate, 
register_model, promote_model, list_studios, studio_overview`.
 - The README claims "26+ tools / 12 Phase 6". The code has 24, and the Phase-6 group is 10.
- **Ingest path confinement (`_validate_copilot_ingest_source`)** **[V]**. Keep it intact: reject 
empty/NUL and `..`; allow `http(s)` URLs (which still pass SSRF validation), otherwise only `file`/no scheme; resolve against `OMNIFORGE_COPILOT_INGEST_ROOT` (default CWD); reject `/etc /proc /sys /root /dev /var/run`; require the path to be inside the root; reject any dot-prefixed component.
- **`tool_list_studios` OpenAPI-verified routes (D5 resolved).** Advertises canonical `/api/workbench/sXX/*` endpoints and dynamically discovers studio endpoints from `omniforge.api.workbench.router`.
- The `promote_model` and `register_model` tools enforce RBAC (`model:promote`, `model:deploy`, `mlops:write`) whenever an HTTP request `Principal` is active via `get_request_principal()` (see D4).
---
## 13. Domain plugins
- `plugin_sdk.py` **[V]**:
 - `DomainMetadata` has `name, display_name, description, version, compliance_frameworks, kpis, 
recommended_models, default_target, config_file`.
 - `BaseDomainPlugin` has abstract `engineer_domain_features`, `compute_domain_kpis(df, *, 
predictions=None)` and `generate_sample_dataset(n_samples=200, random_state=42)`. It also has `get_validation_rules()`, `summary()` and `load_config()`.
 - `DomainPluginRegistry` has `register`, `get`, `list_plugins`, `list_names` and `discover_entry_points`.
- `domains/__init__.py` imports 18 vertical modules, registers each module-level `plugin` **instance**
in `_BUILTIN_PLUGINS` order, and calls `default_domain_registry.discover_entry_points(skip_existing=True)` **[V]**.
- **Entry-point discovery (D7 resolved).** `pyproject.toml` maps `[project.entry-points."omniforge.domains"]`
to **classes** (e.g. `...finance:FinanceDomainPlugin`), and `discover_entry_points` instantiates concrete `BaseDomainPlugin` subclasses as well as pre-instantiated objects.
- **Add a built-in domain:** create `domains/<slug>.py` with `<Name>DomainPlugin` and a 
module-level `plugin = <Name>DomainPlugin()`; import it and add it to `_BUILTIN_PLUGINS`; add `configs/domains/<slug>.yaml`; update the count assertions (`launcher.run_environment_check` requires `>= 18`, and README/tests assume 18); optionally add the slug to `BUILTIN_DOMAINS` (informational only) and a `pyproject` entry point.
---

## 14. API layer
### 14.1 Surfaces and middleware **[V]**
- **`/api/v1/*`** is defined inside `create_app()`: health, version, token, domains, data 
upload/quality/clean, EDA, AutoML, forecast, causal, optimization, serving, registry, drift, copilot, pipelines. Each route uses `Depends(require_permission("..."))`.
- **`/api/workbench/*`** is the SPA's API (`api/workbench.py` plus `api/studios/*`).
- `/docs`, `/redoc` and `/openapi.json` are unauthenticated.
- **There is no Prometheus metrics endpoint**, although `prometheus-client` is a declared dependency 
**[V]**. `GET /api/system/capabilities` (aliased at `/api/v1/system/capabilities`) exposes `probe_runtime_capabilities()` behind `require_permission("data:read")` (D14 resolved).
->- **Middleware stack** (innermost outermost): CORS (explicit methods and headers, including 
`X-Session-ID` and `X-CSRF-Token`); GZip (≥ 500 bytes); `RequestBodySizeLimitMiddleware`; `SlidingWindowRateLimitMiddleware`; security-headers middleware (sets `X-Content-Type-Options`, `X-Frame-Options: SAMEORIGIN`, `Referrer-Policy: same-origin`, `Content-Security-Policy-Report-Only`, HSTS).
- **SPA serving:** `GET /{path}` serves files from `ui/static_app`, confined with `is_relative_to`. Falls back
to `index.html` with `no-store` headers. 404s for paths starting with `api/`, `docs`, `redoc`, `openapi.json` or `health`. The SPA index response sets admin cookies only when bootstrap is enabled, the Host and ASGI peer are loopback, and production mode is off (§14.2).
- **Shared tenant registry:** `api.main` maintains a tenant-scoped serving cache. The REST and Workbench surfaces resolve each tenant’s registry through `get_tenant_registry()`. The Workbench catalog imports old REST entries from `api_v1/model_registry.db` in one transaction, retaining versions and signed artifact paths. Stop legacy writers and preserve the workspace volume and signing keys during the upgrade. An identifier or Production-stage collision requires operator resolution.
### 14.2 Authentication and authorization (`auth.py`) **[V]**
Credentials are checked in this order by `_resolve_current_principal`:
1. `Authorization: Bearer <JWT>`. A malformed header or invalid token gives 401.
2. `X-API-Key` or `X-Workbench-Token`, compared in constant time with `get_api_key()`. Principal is 
`api_key_principal`, role `admin`, tenant `default_tenant_id`.
3. The `omniforge_wb_auth` cookie (HttpOnly, SameSite=Strict). State-changing methods also require the
double-submit `omniforge_csrf` cookie to equal the `X-CSRF-Token` header. Principal is `local_workbench_user`, role `admin`. Rules and behaviour:
->- No credentials 401 with `WWW-Authenticate: Bearer`. Never add an anonymous fallback.
- `get_current_principal` is `async def` **on purpose**, so the `ContextVar` it sets is visible to sync path 
operations (anyio copies the context into worker threads). **Do not make it sync.**
- `require_workbench_auth` authenticates every `/api/workbench/*` route. Studio 01 data writes and workspace restore require `data:write`. Studio 06 training requires `model:train`. Studio 10 requires `pipeline:execute` and additionally `model:train` for AutoML. Studio 11 registration requires `mlops:write`. Direct Production registration, promotion, and rollback also require `model:promote`. Review other routes individually.

- `require_permission(perm)` calls the shared `access_controller`. The app **refuses to boot** 
(`RuntimeError`) if the audit path is unusable.
- `jwt_manager = JWTManager()` is built at import time. In production a missing or weak JWT secret 
therefore aborts import, which is the intended fail-closed behaviour.
- [!] **Local bootstrap is unauthenticated admin when enabled (D3) [V].**
 - Code default is **`auto_bootstrap_local_ui = False`** (`config/settings.py:131`). `.env.example` sets it to `true`, and `launcher.main()` loads `.env`, so launcher-started servers effectively bootstrap while bare `uvicorn omniforge.api.main:app` does not. `should_auto_bootstrap_local_ui()` (`api/auth.py:163`) additionally requires non-production.
 - When enabled, `GET /` (or any SPA path) returns admin cookies to **whoever can reach the port**. The cookie is not static: `issue_local_workbench_cookies` includes a per-issuance nonce and timestamp with an **8-hour `max_age`** **[V]**.
 - **The core risk remains when enabled:** admin-equivalent cookies are granted **without any credential** to any client that can reach the port. `make run-api` / `make run-ui` bind `0.0.0.0`; the CLI `serve`/`ui` default to `127.0.0.1` (see 3.2).
 - **Mitigation:** set `OMNIFORGE_ENVIRONMENT=production`, or `OMNIFORGE_SECURITY__AUTO_BOOTSTRAP_LOCAL_UI=false`, or bind `127.0.0.1`, for anything beyond a single-user machine.
- `POST /api/v1/auth/token` mints JWTs. It requires a valid API key (header or body) or an admin Bearer 
token. A tenant-scoped admin cannot mint `platform_admin` tokens or tokens for another tenant.
- **Production registry transitions.** The REST transition route requires `model:deploy` and additionally `model:promote` for Production. The Workbench promotion route requires `model:promote`. Request-scoped Copilot tools enforce their own stage-specific permissions.
- **Session namespacing (OWASP API1/BOLA):** `_sid(x_session_id)` returns `ws:<sha256(tenant NUL 
subject)[:16]>:<raw id>`. It falls back to the raw id only when there is no authenticated principal, which does not happen on workbench routes. **Limit:** every cookie and API-key caller has the **same** (tenant, subject) pair. For those callers the unguessable session id is the only separator. Isolation by identity is real only for JWT principals with distinct subjects.
- **Jobs:** `BackgroundJobManager` (`ThreadPoolExecutor`, `max_workers=4`, `max_history=128`) 
records `JobRecord(...)`. Jobs are owner-scoped as `job:<owner_tag>`. `platform_admin` listings use `owner_key=None`.
### 14.3 Workbench helpers **[V]**
- `read_upload_bounded(file, max_bytes=None)` reads in 1 MiB chunks and returns 413 beyond the limit.
- Upload rejects `.pkl .pickle .joblib .dill`, both in `/api/workbench/s01/upload` and in 
`/api/v1/data/upload`.
- `parse_model_stage` is case-insensitive and returns 400 (or a caller-chosen status) on an unknown stage.
- `_require_df(state)` returns 400 "No active dataset loaded...".
- `_state_summary(state, sid)` is the canonical state payload.
- `_cache_and_respond(state, sid, cache_key, body, include_state=True)` saves the result **before** 
computing the summary, so that `cached_result_keys` contains the key.
- `_discover_benchmark_csvs` and `/s01/load-sample` accept a plain `.csv` filename only, confined to 
`OMNIFORGE_BENCHMARK_DIR` or `<repo>/examples/datasets` with `is_relative_to`.
### 14.4 Server-side state (`ui/state.py`) **[V]**
- Each session is a dict from `default_workspace_dict()` whose keys are all `uap_*` (plus 
`_uap_remount_studio`).

- `ServerSessionRegistry(ttl_seconds=86400, max_sessions=128)` is an in-process, `RLock`-guarded store 
that evicts expired entries and then the oldest. **State is per process and is lost on restart.** Multiple workers do not share sessions.
- **Invalidation:** `_DOWNSTREAM_KEYS` lists every derived key; `_reset_downstream_state(state, 
keep_keys)` clears them. `_STAGE_PRESERVE_KEYS` and `_STUDIO_RESULT_KEEP_PREFIXES` define which results survive a stage mutation. **When you add derived state, add its key to `_DOWNSTREAM_KEYS`** (and to a preserve group if appropriate), or stale results will leak across dataset changes. The current list includes `uap_quality`, `uap_clean_report`, `uap_eda_report`, `uap_automl`, `uap_active_model`, `uap_clustering_df`, `uap_split_lock`, and many more **[V]**.
- **Mutation entry point:** `update_active_dataset(state, new_df, *, stage, action, details, 
preserve_stage=None)`. It saves `uap_prev_df` for one-step undo, prunes exclusions, re-aligns the Split Lock, resets downstream state, refreshes audits, and records lineage. **Use it for every dataset change.**
- **Analysis frames:** use `get_analysis_dataframe(state, df)`, which is `uap_df` minus excluded columns. 
**Do not use raw `uap_df` for analysis.**
- **Split Lock** (`lock_leakage_safe_split`, `get_active_split_frames`, `recombine_split_frames`, 
`clear_split_lock`): `get_active_split_frames` uses row-id columns when present, otherwise positional `iloc` indices. It raises `ValueError` if fewer than 5 train or 2 test rows survive. `recombine_split_frames` matches by **index label**, so it requires a unique index. Cleaning and feature steps must fit on the train partition only, transform both partitions, and then recombine.
### 14.5 Studio routers and route map **[V]**
The backend routers `sNN_*` map **one-to-one** onto UI studios 01-14. `s00_ingestion` is folded into Studio 01. **File names are misleading, however:**
| UI studio | Module | Routes (prefix `/api/workbench`) |
|---|---|---|
| 01 Data Ingestion, Quality & Cleaning | `workbench.py` (bulk), `s00_ingestion`, `s01_cleaning` | 
`/s01/{catalog,excel-sheets,upload,sql/demo,sql/inspect,sql/load,access/inspect,access/load,load-sample,g rid,quality,profile,clean,export-clean/{fmt},undo,reset-raw,revert,deleted-audit,deleted-log,deleted-audit/c sv}`, `/s00/data_relationships`, `/s01/advanced_audit` |
| 02 Clustering & Anomaly Screening | `workbench.py`, `s02_mining` | 
`/s02/{clustering,clustering/commit,anomaly,anomaly/commit,mining,advanced_mining}` |
| 03 Universal Diagnostics, Baselines & Risk | `s03_eda.py` (**contains diagnostics**) | 
`/s03/{catalog,run}` |
| 04 Automated EDA & Hypothesis Lab | `s04_stats.py` | `/s04/{eda,hypothesis}` |
| 05 Feature Engineering & Selection | `s05_features.py` | `/s05/{engineer,select}` |
| 06 AutoML, Deep Learning & Predictions | `s06_automl.py` | 
`/s06/{backends,automl,export-formula,nonparametric,deep-learning,predict-single,predict-batch}` |
| 07 Responsible AI | `s07_advanced.py` | `/s07/{explain,fairness}` |
| 08 Forecasting, Causal, Opt & Simulation | `s08_causal.py` | `/s08/{forecast,causal,optimize,simulate}` |
| 09 Specialized AI | `s09_opt_sim.py` (**not optimization**) | 
`/s09/{backends,nlp,vision,graph,recommend,survival,twin-control}` |
| 10 Visual DAG Pipeline Builder | `s10_rai.py` (**not RAI**) | `/s10/run` |
| 11 MLOps | `s11_mlops.py` | `/s11/{overview,register,promote,rollback,drift,root-cause}` |
| 12 Executive Reporting & Export | `s12_reports.py` | `/s12/{catalog,compile,export/{fmt}}` |
| 13 AI Analytics Copilot | `s13_copilot.py` | `/s13/chat` |
| 14 Live Prediction & Supervisory Control-Room | `s14_supervisory.py` | 

`/s14/{models,equations,quick-train,evaluate-file,evaluate-sql,evaluate-access,evaluate-demo-batch,evalua te-single,root-cause,export/{fmt}}` | Shared workbench routes: `GET /state`, `GET /results`, `GET /benchmarks`, `POST /theme`, `POST /targets`, `POST /exclusions`, `POST|DELETE /split-lock`, `POST /workspace/{clear,save,load}`, `GET /jobs`, `GET /jobs/{id}`, `POST /jobs/{id}/cancel`, `GET /feature-store/freshness`, `DELETE /dataset` **[V]**. **Handler pattern to follow:**
```python
router = APIRouter(tags=["workbench-sNN"]) def _wb() -> Any: # lazy import: avoids a circular import with workbench from omniforge.api import workbench as wb return wb @router.post("/sNN/something") # no /api/workbench prefix here; the parent adds it def handler(req: SomeReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]: sid = _wb()._sid(x_session_id) state = _wb().get_session_state(sid) df = _wb().get_analysis_dataframe(state, _wb()._require_df(state)) ... _wb().record_lineage(state, "Stage", "Action", "details") return _wb()._cache_and_respond(state, sid, "sNN_something", payload)
```
Request models live in `api/studios/schemas.py`. Include the router in the bottom block of `workbench.py`. Use sync `def` handlers (threadpool) unless you do async I/O such as uploads. Map errors explicitly: 400 bad input, 403 security, 409 governance conflict, 413 size. Gate sensitive routes with `dependencies=[Depends(require_permission("..."))]`. **Two-phase preview/commit** for destructive Studio 02 operations (`/s02/clustering|anomaly` then `.../commit`) **[C]**: the run endpoint stores a preview, sets `..._committed=False`, and caches the payload with `provenance: "PREVIEW_ONLY"`. The commit endpoint records deletions and exclusions, sets the committed flag **before** `update_active_dataset(..., preserve_stage=...)`, and then marks the payload `COMMITTED`. Commit refuses to leave fewer than 6 rows. Anomaly fitting uses the train partition only when a Split Lock is active.
### 14.6 Workspace save and load (`core/workspace_session.py`) **[V]**
- **The canonical extension is `.omniforge`.** `LEGACY_PROJECT_EXTENSIONS = (".omniforge", 
".uap")`, so `.uap` files are **readable** but never written. The `ui/state.py` docstring calling it a ".uap Project Store" is stale text.
- **Format:** a ZIP containing `manifest.json` and `session_state.joblib`, with `FORMAT_VERSION = 
"2.0"`.
- **Integrity:** SHA-256 plus HMAC (`omniforge:workspace:v2`, `require_signature=True` by default). 
Loading uses the restricted unpickler.
- **Archive limits:** at most 64 entries, 512 MiB uncompressed (zip-bomb guard), a 4 MiB manifest cap, 

and only STORED or DEFLATED members.
- Because the signing key is host-local, workspaces saved on one host will not verify on another unless the 
key is shared.
---
## 15. UI, launcher, frontend
### 15.1 `ui/app.py` **[V]**
- `STUDIOS` lists 14 studios, "01"-"14".
- `create_workbench_app()` calls `api.main.create_app()`.
- `run_workbench_server(host="127.0.0.1", port=8501, ...)` runs uvicorn.
### 15.2 Launcher and CLI
See §3.2.
### 15.3 Frontend **[V]**
- Stack: React 18, Vite 6, TypeScript 5.7 (`strict`), `lucide-react`. Source is about 11k lines under 
`frontend/src`.
- `npm run build` is `tsc && vite build`. It writes to `src/omniforge/ui/static_app` with `emptyOutDir: 
true`, hashed file names and vendor chunk splitting.
- **There are no `lint` or `test` npm scripts.** The dev server runs on port 5173 and proxies `/api` and 
`/health` to `127.0.0.1:8501`.
- The SPA reads the `omniforge_csrf` cookie and sends `X-CSRF-Token`. It generates and stores 
`omniforge_session_id`, and can send an optional API key or Bearer token stored in `localStorage`.
- Backend payload shapes (`_state_summary`, studio payload keys, `cached_result_keys`, 
`studio_completed`) are contracts with `types.ts` / `studioTypes.ts`. **When you change a key, update the TypeScript types, rebuild, and commit `static_app`.**
- The CSP allows `'self'` scripts only. Inline styles and `data:`/`blob:` images are allowed.
---
## 16. Capabilities, installer, packaging
- `capabilities.py` is stdlib-only **[V]** and defines `Pack` (frozen dataclass), 12 packs, three tiers, 
`REQUIRED_BY_TIER`, `FEATURE_UPGRADES`, and helpers such as `packs_for_tier`, `resolve_selection`, `requirements_for` and `profile_summary`.
 - Packs: `core-runtime, api-server, statistics, data-io, reporting, boosting, deep, vision, nlp, foundation, 
automl, responsible`.
 - Tiers: **simple** = core-runtime + statistics + data-io + reporting; **advanced** = simple + api-server; 
**everything** = advanced + all optional packs.
- **Drift to guard:**
 - [!] Pack `dists` must stay in step with `pyproject.toml` and `requirements.in`. Today 
`CORE_RUNTIME` lists `pillow`, which `pyproject.toml` only includes in the `ui`, `vision` and `build` extras.

 - [!] The SIMPLE-tier comment says "Served by the bundled SPA, so no FastAPI needed". The SPA is 
served by FastAPI, so the SIMPLE tier cannot serve the workbench.
 - The runtime probe route `/api/system/capabilities` named in the docstring does not exist (§14.1).
- `installer/provision.py`, `packaging/`, `scripts/build_*.py`, the Windows batch files and 
`windows-installer.yml` were not read in depth **[C]**.
---
## 17. Security invariants checklist
⚠Verify on every change. Items marked have a documented gap. ⚠1. Every `/api/workbench` route keeps `require_workbench_auth`. RBAC is not applied inside the workbench (D4). ⚠2. Workbench state is reached only through `_sid()`. Cookie and API-key callers share one namespace (§14.2).
3. State-changing, cookie-authenticated requests need a matching `X-CSRF-Token`.
4. Uploads use `read_upload_bounded`. Pickle-like extensions are rejected.
5. Outbound HTTP goes only through `validate_outbound_http_url` and `_fetch_http_safe`, with no 
⚠`follow_redirects=True`. No DNS pinning (D6).
6. SQL goes only through `validate_readonly_sql`, with allow-listed and quoted identifiers. SQLite is 
read-only with an authorizer. DuckDB is read-only without external access.
7. Model and workspace artifacts are HMAC-signed and loaded through the restricted unpickler. Enums 
⚠that can be pickled use `SerializableStrEnum`. See D8 and the unpickler notes in §6.6.
8. Secrets resolve through `_resolve_or_persist_secret`. Production requires explicit, strong, 
non-placeholder values.
9. The audit log is append-only, hash-chained and MACed. **Never add an unsigned verification path.** 
⚠ Multi-writer behaviour forks the chain (D1 **[V]**).
10. Copilot file ingestion stays confined to `OMNIFORGE_COPILOT_INGEST_ROOT`, with no hidden 
files and no system directories.
11. Rate limiting is keyed by the ASGI client host. Only the two health paths are exempt.
12. CORS never combines a wildcard with credentials (the validator enforces it).
13. Registry promotions respect the fairness gate. Overrides are tagged and audited.
14. Tenant isolation: per-tenant registries and trackers; cross-tenant access only for `platform_admin`, and 
always audited.
15. Logs, audits and `to_dict()` outputs contain no raw PII or secrets.
⚠16. Anything that accepts a path or URL from a client must add its own confinement. `DataIngestionEngine.ingest` does not confine local paths. Only built-in `step_type` strings are accepted from JSON, so arbitrary Python callables are **not** reachable from the API.
### 17.1 Environment variables recognised
Direct `os.getenv` / `os.environ` reads found by scanning `src/`, `scripts/` and `installer/` **[V]**. Any `OMNIFORGE_<SECTION>__<FIELD>` also maps onto a settings field via pydantic-settings (see 7.1); `.env.example` lists many such settings-driven vars (e.g. `OMNIFORGE_DEBUG`, `OMNIFORGE_API_HOST`, `OMNIFORGE_DATABASE_URL`) that do not appear as direct `os.getenv` calls. See 7.3 for settings that do nothing.
| Group | Variables |
|---|---|

| Mode and keys | `OMNIFORGE_ENVIRONMENT` / `ENVIRONMENT`, `OMNIFORGE_KEY_DIR`, 
`OMNIFORGE_API_KEY`, `OMNIFORGE_SECURITY__API_KEY`, `OMNIFORGE_JWT_SECRET_KEY`, `OMNIFORGE_SECURITY__JWT_SECRET_KEY`, `OMNIFORGE_ENCRYPTION_KEY`, `OMNIFORGE_SECURITY__ENCRYPTION_KEY`, `OMNIFORGE_ARTIFACT_SIGNING_KEY`, `OMNIFORGE_SECURITY__ARTIFACT_SIGNING_KEY`, `OMNIFORGE_SECURITY__AUTO_BOOTSTRAP_LOCAL_UI` |
| Pseudonymization | `OMNIFORGE_PSEUDONYM_SALT__<TENANT>`, 
`OMNIFORGE_PSEUDONYM_SALTS`, `OMNIFORGE_PSEUDONYM_SALT` |
| Audit | `OMNIFORGE_AUDIT__ANCHOR_PATH`, `OMNIFORGE_AUDIT__CHECKPOINT_KEY`, 
`OMNIFORGE_AUDIT__ANCHOR_EVERY` |
| Network and SSRF | `OMNIFORGE_ALLOWED_HTTP_HOSTS`, 
`OMNIFORGE_ALLOW_PRIVATE_HTTP_HOSTS`, `OMNIFORGE_ALLOWED_DB_HOSTS`, `OMNIFORGE_ALLOW_PRIVATE_DB_HOSTS`, `OMNIFORGE_ALLOWED_DB_DIRS` |
| Rate limiting | `OMNIFORGE_API__RATE_LIMIT_PER_MINUTE`, 
`OMNIFORGE_API__RATE_LIMIT_BACKEND`, `OMNIFORGE_API__RATE_LIMIT_REDIS_URL` |
| Governance and MLOps | `OMNIFORGE_REQUIRE_FAIRNESS_FOR_PRODUCTION`, 
`OMNIFORGE_CT_DISABLED`, `OMNIFORGE_CT_AUTO_PROMOTE`, `OMNIFORGE_AUTO_ROLLBACK_DISABLED` |
| Paths | `OMNIFORGE_COPILOT_INGEST_ROOT`, `OMNIFORGE_BENCHMARK_DIR` |
| Launcher | `OMNIFORGE_LAUNCH_MODE`, `OMNIFORGE_HOST`, `OMNIFORGE_API_PORT`, 
`OMNIFORGE_UI_PORT`, `OMNIFORGE_LOG_LEVEL`, `OMNIFORGE_API_URL` |
| LLM providers | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY`, 
`AZURE_OPENAI_ENDPOINT` |
---
## 18. Code conventions
- Module docstring first, then `from __future__ import annotations`, then import blocks in stdlib / third-party / first-party order (ruff `I`).
- Pydantic v2 throughout. Settings from pydantic-settings.
- Google-style docstrings. Public APIs fully typed. `Final[...]` for constants.
- Timestamps: `datetime.now(UTC).isoformat()`. Ids: `f"<prefix>_{uuid4().hex[:12]}"`.
- Hashing JSON: `json.dumps(..., sort_keys=True, separators=(",", ":"), default=str)`.
- Compare secrets and MACs with `hmac.compare_digest`.
- Probe functions return structured dicts instead of raising.
- Scientific code cites formulas and references in docstrings. Keep the citations when refactoring.
- Optional-dependency modules are typed with `X: Any = None` sentinels.
---
## 19. Recipes
- **Add an API studio endpoint:** put the request model in `api/studios/schemas.py`, and the handler in 
the right `sNN_*.py` using §14.5. Cache under `sNN_<name>`. If it mutates the dataset, call `update_active_dataset(..., preserve_stage=...)`. Register new derived keys in `_DOWNSTREAM_KEYS`. Update the `frontend/src` types and components, rebuild and commit `static_app`. Add a test.
- **Add a model:** subclass `BaseOmniForgeModel`, register it in `ModelCatalog`, confirm it survives a 

save-and-load round trip under the restricted unpickler (no lambdas, no unsafe globals), and include it in baseline comparisons.
- **Add a Copilot tool:** see §12.
- **Add a drift metric:** extend `DriftDetector.detect` and `FeatureDriftResult`, and update 
`DriftRootCauseAnalyzer` if attribution should use it. Thresholds are constructor arguments today, not settings (§10.2).
- **Add a data source:** extend the `DataIngestionEngine` routing and the `SUPPORTED_*` constants. 
Reuse `_fetch_http_safe` and the read-only SQL helpers. Add a `DependencyMissingError` with an install hint, and sanitize URIs in lineage.
- **Add a setting or env var:** see §7.5, and list the variable in §17.1 and `.env.example`.
- **Add a dependency:** edit `requirements.in`, the relevant `pyproject.toml` extra and the 
`capabilities.py` pack (when user-facing). Regenerate **all** affected locks with `uv`, including `locks/` via `scripts/build_locks.py`. Confirm the `lock-verify` smoke import.
- **Add a built-in domain:** see §13.
---
## 20. Defect and inconsistency register
Severity is an assessment. Items marked **[X]** were reproduced by execution in a previous review. Items marked **[P]** need confirmation before you rely on them. D1, D4 and D5 were promoted from [P] to [V] in the 2026-11-24 revision.
| ID | Sev | Finding |
|---|---|---|
| **D1** | Resolved | **Audit chain multi-writer synchronization.** `AuditLogger` and `AuditCheckpointAnchor` now coordinate across threads and OS processes via `_locked_file_path` (`RLock` + `fcntl`/`msvcrt`), verify historical prefix SHA-256 + newly appended records before signing, and `fsync` writes. |
| **D2** | Resolved | **MPC analytic gradient, initial-step dynamics & soft state constraints.** `_objective_grad` matches finite differences, `_dynamics_constraints` enforces $x_1 = A x_0 + B u_0$, and Stage 2 softens state bounds via non-negative slack variables with exact $\ell_1 + \ell_2^2$ penalty. |
| **D3** | Mitigated | **Auto-bootstrap defaults aligned to safe off/loopback.** `.env.example` now defaults `OMNIFORGE_SECURITY__AUTO_BOOTSTRAP_LOCAL_UI=false` and `Makefile` `run-api`/`run-ui` default `HOST ?= 127.0.0.1`. Note: when explicitly enabled by an operator, auto-bootstrap still grants local admin cookies in non-production. |
| **D4** | Mitigated | **RBAC promotion & JWT settings alignment.** `/api/v1/registry/.../stage` now enforces `model:promote` for `Production` transitions; `CopilotToolRegistry` `register_model`/`promote_model` enforce RBAC when a request `Principal` is active; `api/auth.py` wires `settings.security.jwt_*`. General workbench routes remain authenticated via `require_workbench_auth`. |
| **D5** | Resolved | **Copilot `list_studios` & `studio_overview` routes.** Updated to canonical OpenAPI-verified `/api/workbench/sXX/*` endpoints with dynamic router discovery. |
| **D6** | Resolved | **SSRF DNS-rebinding protection.** `resolve_and_validate_outbound_http_url` + `_fetch_http_safe` pin vetted IPs per redirect hop while preserving `Host` headers and TLS `sni_hostname` SAN verification. |
| **D7** | Resolved | **Domain plugin entry points.** `discover_entry_points` instantiates concrete `BaseDomainPlugin` subclasses as well as instances, and `omniforge.domains` auto-discovers external entry points (`skip_existing=True`). |
| **D8** | Mitigated | **Atomic encrypted `BaseOmniForgeModel.save()`.** Serializes and encrypts in memory so plaintext never touches disk, stages `fsync`-ed sibling `.tmp` files, swaps via `os.replace` with metadata rollback on failure, and verifies both HMAC and SHA-256 on `load()`. Keys remain host-local unless configured via env. |
| **D9** | Open **[V]** | **Some YAML/config settings remain declarative.** `jobs.*`, `security.jwt_*`, and `paths.*` are consumed; unused YAML keys remain non-authoritative at runtime (§7). |
| **D10** | Resolved | **Quality gates green.** `pytest`, `ruff`, `mypy`, and `pip check` pass cleanly. |
| **D11** | Open **[V]** | `_RestrictedNumpyUnpickler` retains `"test_third_party_claims"` (required by `tests/unit/test_third_party_claims.py`), uses `joblib.numpy_pickle`, and permits `torch` after mandatory HMAC verification (§6.6). |
| **D12** | Resolved | **Decoupled `omniforge.ui` imports.** `ui/__init__.py` uses lazy `__getattr__` and `ui/app.py` defers `create_app` inside `create_workbench_app()`. |
| **D13** | Mitigated | `requirements-build.in` header updated to `v4.0.0` and `README.md` lock compile command updated to `requirements.in` (`run_environment_check()["required_modules"]` keys preserved for `tests/unit/test_packaging_and_launcher.py`). |
| **D14** | Mitigated | Added pure-stdlib `probe_runtime_capabilities()` and authenticated `GET /api/system/capabilities` (`/api/v1/system/capabilities`). |
| **D15** | Resolved | `JWTManager.verify_token` requires `exp` and `sub` in both PyJWT and stdlib backends (with algorithm allowlist `HS256`/`HS384`/`HS512`), and FastAPI's `OmniForgeError` handler omits internal `context` and `cause` via `to_dict(include_internal=False)`. |
| **D16** | Resolved | Fixed `omniforge automl` CLI typo, aligned `README.md` 15-studio table with actual studio routers, and removed nonexistent `/s14/root-cause` reference. |
| **D17** | Mitigated | CI `static_app` freshness check now uses `git status --porcelain -- src/omniforge/ui/static_app` so newly generated untracked Vite chunks are caught alongside modified files. |
---
## 21. Areas not examined in depth
Read these fully before editing them. Nothing above should be treated as authoritative for them.
- **Backend:** `eda/*`, `features/*`, `forecasting/*`, `causal/*`, `optimization/*`, `evaluation/*`, 
`automl/*`, `reporting/*`; `pipelines/{dag, executor}`; `models/{classical, ensemble, nonparametric, registry, baselines}` and every `models/*/` subpackage; `mlops/{tracking, serving, monitoring, continuous_training, job_queue, future_scoring, supervisory_monitor}`; `simulation/engines.py`; `data/{cleaning, quality, validation, versioning, feature_store, synthetic, multimodal, intake_audit, format_audit}` and `data/splitting.py` (only partly); `copilot/{agent, memory, prompts}`; `core/{performance, rate_limit, sbom}`; `api/schemas.py` and the bodies of studios `s04`-`s08`, `s10`, `s12`, `s14`; the middle of `api/workbench.py` (cleaning, clustering and anomaly bodies), and `ui/state.py` beyond the points quoted.
- **Frontend, packaging and docs:** `frontend/src/**` internals; `installer/`, `packaging/` and `scripts/` 
internals; `docker/`; `configs/**`; `.env.example`; the batch files; `windows-installer.yml`; and `docs/` (not consulted; verify its claims against code).
- **Not executed in this review:** the frontend build, the Docker build, the Windows installer, 
`scripts/latency_budget_benchmark.py`, and the test suite / lint / type-check gates.
---
## 22. Corrections to the previous draft
| Previous claim | Finding |
|---|---|
| "Tooling truncated; many files unreadable." | The repository was cloned and read directly in the previous 
review. Unread areas are now listed in §21. |
| "Do not read or trust `docs/` or `tests/`." | Removed. Tests are the executable spec (§0.1). |
| "CI fails if the lock files are stale." | CI does **not** check lock freshness (§3.1, §3.4). |
| `ruff` and `mypy` "must pass". | mypy passed; ruff had 19 findings; 1 test failed at the previously reviewed 

commit (§0.2). |
| `cli/__init__.py` "INFERRED"; "Typer entry-point naming mismatch". | Verified: `app = cli`. Not an 
inconsistency. |
| API "presumably wires Prometheus metrics". | There is no metrics endpoint (§14.1). |
| Runtime probe `/api/system/capabilities`. | The route does not exist (§16). |
| Studio numbering "offset in places". | Backend `sNN` maps 1:1 onto UI 01-14; only file names mislead 
(§14.5). |
| Workspace saves are `.uap` locally and `.omniforge` in the browser. | Canonical is `.omniforge`; `.uap` is 
read-only legacy; the stale text is in docstrings (§14.6). |
| Split Lock "label versus position" is a recurring bug source. | `get_active_split_frames` uses positional 
`iloc` or row ids. Only `recombine_split_frames` depends on label uniqueness (§14.4). |
| "Dual domain registration (instances and entry points)". | Entry points are non-functional, and discovery 
is never called (§13). |
| "Mirror settings in `configs/default.yaml`, thresholds in settings." | Neither YAML nor most settings are 
consumed at runtime (§7). |
| "`always_compare_baselines` enforces baseline-first." | The flag is never read (Rule 9). |
| Layering diagram with `api/ui` on one level. | `api.workbench` imports `ui.state`, and `ui/__init__` 
imports the API app (§5). |
| "Session namespacing isolates callers." | True only per (tenant, subject). Cookie and API-key callers share 
a subject (§14.2). |
| "Never fall back to an anonymous admin." | Correct for headers. But auto-bootstrap issues admin cookies 
to any browser in non-production (D3). |
| "Possible MPC gradient bug (unverified)." | Confirmed by finite-difference check in the previous review, 
with a worse variant in the slack path (§11). |
| Environment variable list "verified". | It missed 15 variables (§17.1). |
| Copilot "26+ tools / 12 Phase 6". | 24 and 10 (§12). |
| `omniforge check` and the optional-dependency philosophy. | `check` treats seven optional extras as 
required (§3.2). |
| "Atomic save/load" (from the class docstring). | Not atomic (§9.1). |
| `AuditLogger` "not found in `security.py`" (2025 draft). | Found at `core/security.py:1506`; D1 confirmed **[V]** (2026-11-24 revision). |
| `auto_bootstrap_local_ui` "defaults to True"; `serve`/`ui` "default to 0.0.0.0". | Code default is `False`; CLI defaults are `127.0.0.1`. `True`/`0.0.0.0` come from `.env.example` / `Makefile` (2026-11-24 revision). |
| `docker-build` tag `2.0.0`; `jobs.*` / `feature_store_dir` dead. | Tag is now `4.0.0`; both settings are consumed (2026-11-24 revision). |
---
## 23. Pre-commit checklist
- [ ] I read every file I modified in full, and I re-ran the baseline gates first (§0.2).
- [ ] `ruff check src tests` and `mypy src/omniforge` show **no new findings**, with full annotations and 
no stale `# type: ignore`.
- [ ] `pytest -q` shows no new failures, and new behaviour has tests with registered markers.
- [ ] No new hard or heavy dependency. Optional imports are lazy with fallbacks. `import omniforge` stays 
fast, and `version.py` and `capabilities.py` stay stdlib-only.
- [ ] Errors are `OmniForgeError` subclasses with suggestions. Payloads are JSON-safe. Nothing logs PII or 
secrets. `context` and `cause` hold nothing sensitive.
- [ ] §17 still holds. No weakening of auth, CSRF, SSRF, SQL, pickle, path confinement or audit verification. 
Any new client-supplied path or URL has its own confinement.
- [ ] State changes updated `_DOWNSTREAM_KEYS` and the preserve groups. Dataset mutations go 
through `update_active_dataset`, and analysis uses `get_analysis_dataframe`.
- [ ] Any new setting has a real consumer, a test and an entry in `.env.example`.
- [ ] Baseline-first and leakage-safe behaviour is preserved (Split Lock respected).

- [ ] If `frontend/` changed: I rebuilt it and committed `src/omniforge/ui/static_app`.
- [ ] If dependencies changed: I regenerated all locks (including `locks/`) with `uv`, not by hand.
- [ ] Version strings in `version.py`, `pyproject.toml` and `frontend/package.json` are aligned.
---
*End of document.*

### Supervised task and preprocessing contracts

For tabular task inference, use `core.task_inference.infer_supervised_task` in splitting,
diagnostics, AutoML and Workbench: booleans, categories, or at most ten finite integer levels
imply classification; explicit task overrides remain authoritative. Noninteger numeric targets
are regression, regardless of cardinality. Studio 03 reports exploratory grouping separately.

New numeric Workbench fits use median imputation and bounded nominal indicators. A 32-output
cap includes the infrequent bucket when present; unseen levels encode as all zero. Specialized
AutoML uses a train-fitted sklearn `ColumnTransformer`, including model-specific scaling and
imputation chosen by the backend. Native `BaseOmniForgeModel` retains its numeric/ordinal
array representation and legacy persisted Workbench fits keep their original transforms.
These representations are not interchangeable or claimed to implement the same distance geometry.

The CPCV kernel is `data.cpcv.iter_cpcv_splits`. For new evaluation code, use
`evaluation.validation.CombinatorialPurgedKFold` (pre-test purge, post-test embargo, optional
label interval purge). `data.splitting.LegacyCombinatorialPurgedKFold` is the compatibility
adapter: 6 default folds, list output (including empty training sets), and an additive
post-buffer (purge plus embargo). The old `data.splitting.CombinatorialPurgedKFold` import is
an alias, not interchangeable with the canonical iterator (5 default folds, maximum of
explicit and proportional post-buffers, and no empty training sets).

Native AutoML's reference suite fits bounded nominal indicators independently on each
reference training partition; specialized backend references receive a named, train-fitted
column-transformer matrix. Standalone `ModelBase` ordinal behavior remains a separate
SDK compatibility path, not the encoding for Workbench or native AutoML distance references.
A diagnostic ledger row can carry a nested sample size (e.g. a Shapiro–Wilk cap) distinct
from the enclosing report scope; sampled Hodges–Lehmann effects report both group sizes.
