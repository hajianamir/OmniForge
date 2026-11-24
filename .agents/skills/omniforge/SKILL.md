---
name: omniforge
description: >-
  Review, test, upgrade, develop, and improve OmniForge v4 — an enterprise
  analytics platform with a Python SDK, FastAPI backend, React SPA, and 18
  domain plugins. Use when working in this repository. Do not use for generic
  Python/pandas questions unrelated to OmniForge, or for other projects that
  share the OmniForge name.
version: "4.0.0"
repository: "github.com/hajianamir/OmniForge"
commit: "b7dc7ae4416c6db9b38048c8827dfbd27fb229a1"
last_verified: "2026-11-24"
derived_by: "Arena.ai coding agent; tool-assisted source inspection via shell/grep/Python AST (src/omniforge, frontend/src, tests, configs, packaging, installer, docker, scripts)"
---

# OmniForge — Codex Agent Skill

Repository-local skill: `.agents/skills/omniforge/`. Reference paths are
relative to this skill directory; source, command, and workflow paths are
relative to the repository root. Run commands from the repository root
unless a command explicitly changes directories.

**Authority policy (single rule):** this skill is a **starting index, not an
exhaustive spec**. Before inventing or changing any API, route, env var,
class, or behavior, **read the cited file** (the named function/class, not
just its name — grep finds it, reading it proves it). If the skill and the
code disagree, **code wins** and the skill must be updated (see §7).

**Provenance:** derived by reading code only (no README/docs prose) against
commit `b7dc7ae` on 2026-11-24. Line counts and test counts in references
were verified with `wc -l` / `grep -c` at that commit; re-verify before
quoting them. If `git rev-parse HEAD` differs, treat this skill as
potentially stale and spot-check the sections you rely on.

## Contents

1. [Identity](#1-identity) · 2. [Invariants](#2-invariants) ·
3. [Mental model](#3-mental-model) · 4. [Known limitations](#4-known-limitations) ·
5. [Review checklist](#5-review-checklist) · 6. [Quick reference](#6-quick-reference) ·
7. [How to update this skill](#7-how-to-update-this-skill) ·
8. [References](#8-references-load-on-demand)

## 1. Identity

| Fact | Value | Source |
|---|---|---|
| Package / version | `omniforge` **4.0.0**, *Enterprise Analytics & AI Workbench* | `pyproject.toml`, `src/omniforge/version.py` (`__api_version__="v2"`, `__schema_version__="2.0"`) |
| Python | `>=3.12`; `pyproject.toml` declares 3.12–3.15 classifiers and `requirements.in` (header dated Nov 2026) claims 3.15.2 stable — treat 3.12/3.13 as the CI-proven range | `pyproject.toml`, `requirements.in`, `.github/workflows/ci.yml` |
| Layout | `src/` layout, setuptools; Apache-2.0 | `pyproject.toml` |
| Entry points | `omniforge` → `omniforge.cli:app`; `omniforge-desktop` → `omniforge.launcher:main`; 18 `omniforge.domains` entry points | `pyproject.toml` |
| Backend | FastAPI + Uvicorn: REST `/api/v1/*`, workbench `/api/workbench/*`, compiled SPA | `src/omniforge/api/main.py::create_app` |
| Frontend | React 18 + Vite 6 + TypeScript 5, built into `src/omniforge/ui/static_app/` | `frontend/package.json`, `frontend/vite.config.ts` |
| Workspace | `./.omniforge_workspace/` (gitignored: SQLite DBs, artifacts, audit log, secrets) | `src/omniforge/config/settings.py` |
| Ports | API `8000`, workbench/UI `8501`, Vite dev `5173` — one app serves REST+SPA; `serve` binds :8000, `ui` binds :8501 + browser, `desktop` runs both (see `references/api-and-state.md`) | `configs/default.yaml`, `launcher.py` |

## 2. Invariants

1. **Lazy SDK facade.** `src/omniforge/__init__.py` uses PEP 562 `__getattr__`
   + `_LAZY_EXPORTS` so `import omniforge` works with zero third-party deps.
   New public names go in `_LAZY_EXPORTS` + `__all__` + `TYPE_CHECKING` together.
2. **Stdlib-only `capabilities.py`.** The installer imports it before wheels
   exist. Keep it dependency-free.
3. **Every model subclasses `BaseOmniForgeModel`** (`models/base.py`) and
   implements `_fit/_predict/_predict_proba`. The base owns seeding, carbon,
   metadata, and signed serialization — never reimplement `fit/save/load`.
4. **Anti-leakage is architectural.** Transformers are `fit`/`transform`
   pairs (`DataCleaner`, `FeatureEngineeringPipeline`, `LeakageSafeSplitter`,
   `LeakageGuard`, `AnalysisSpec`). Fit on train only; cover with a leakage test.
5. **Enums pickle by value.** All enums subclass `SerializableStrEnum`
   (`core/types.py`); never plain `StrEnum` (restricted unpickler, SEC-001).
6. **Structured errors only.** Everything raises `OmniForgeError` subclasses
   with `error_code/severity/suggestions/trace_id/context`
   (`core/exceptions.py`). No bare `ValueError`/`RuntimeError` across public
   boundaries.
7. **Tenancy is fail-closed.** Sessions, jobs, registry, and copilot agents
   are keyed by tenant+subject (`api/workbench.py::_sid`, `api/jobs.py`
   `owner_key`). Never key state by raw `X-Session-ID` alone.
8. **SPA freshness is CI-enforced.** `ui/static_app/` is a committed build
   artifact (required so the `simple` tier ships without Node). After any
   `frontend/` change: `cd frontend && npm ci && npm run build`, commit the
   assets, or the `e2e` job fails.
9. **Locks are generated.** `requirements.lock` (universal, 3.12) and
   `locks/*.lock` (Windows, 3.13) come from `uv pip compile` /
   `scripts/build_locks.py`. Change intent (`requirements.in`,
   `pyproject.toml`, `capabilities.py`), then regenerate.
10. **Secrets never log or ship.** Logging redacts `SENSITIVE_KEY_PATTERNS`;
    production requires explicit ≥32-byte secrets. Never commit `.env` or
    `.omniforge_workspace/`.

## 3. Mental model

```
React SPA (15 studios 00–14, keep-alive tabs)
  │ fetch + X-Session-ID + CSRF / API-Key / Bearer
FastAPI create_app() — CORS · GZip · body-cap · rate-limit · security headers
  │ /api/v1/*  /api/workbench/*  SPA fallback
ui/state.py (uap_* keys) · data/* · models/* · automl/* · mlops/*
  │ core/* (types, contracts, security, logging, carbon) · config/* · capabilities.py
```

**Lifecycle:** ingest → format/intake audit → clean (fit/transform) →
quality/validate → split-lock (`LeakageSafeSplitter` + `LeakageGuard`) →
EDA/diagnostics → features → AutoML/model (baselines always compared) →
RAI (fairness/SHAP/conformal/model card) → forecast/causal/optimize/simulate
→ DAG pipeline → registry/serving/drift → report → copilot throughout →
supervisory live scoring (Studio 14) → `.uap` project save. Detail:
`references/architecture.md`, `references/api-and-state.md`.

## 4. Known limitations

- **Process-local execution and workspaces.** Workbench sessions, job
  callables, and copilot agents remain in memory. Workbench job status and
  results are persisted in SQLite with bounded queue and history, but lost
  workers' callables are not resumed. Docker defaults to **1 worker**;
  multi-worker workspace operations need sticky sessions or shared workspace
  state. Shared job status requires a filesystem with reliable SQLite locking.
  Restart loses unsaved sessions (job details rechecked against `49b9058`).
- **Tier-gated features.** `simple` has no FastAPI server; `foundation`
  models need the `deep` pack + API keys/weights; SHAP/codecarbon, native
  boosting, OpenCV, spaCy, Optuna/FLAML are all optional-pack upgrades with
  documented fallbacks (see `references/security.md`).
- **Offline copilot is heuristic.** The `local` provider is a deterministic
  planner, not an LLM — expect rule-based routing, not open-ended reasoning.
- **Audit anchor is not witness-grade by default.** Off unless
  `OMNIFORGE_AUDIT__ANCHOR_PATH` is set; truncating log + checkpoints
  together is undetectable without replicating checkpoints externally.
- **CSP is report-only.** `Content-Security-Policy-Report-Only` surfaces
  violations without blocking; enforcement is a future step.
- **SQLite-backed MLOps.** Tracking/registry/feature-store default to local
  SQLite/Parquet — suitable for single-host use, not multi-writer production.
- **Python version skew.** CI proves 3.12/3.13; 3.14/3.15 classifiers exist
  but several optional deps are version-gated (`python_version < '3.14'/'3.15'`)
  and may be unavailable there.

### What this skill does not know

This skill is an index, not a census. It does **not** enumerate every
`uap_*` key (~90 — read `default_workspace_dict()` in `ui/state.py`), every
CLI flag (read `cli/main.py`), every studio request field (read
`api/studios/schemas.py`), or every workflow beyond the CI jobs named in
`references/testing.md` (read `.github/workflows/`). If you need a complete
list of anything, grep the owning file — do not assume absence from this
skill means absence from the code.

## 5. Review checklist

**Security & tenancy** — no raw `joblib/pickle.load` (use
`safe_joblib_load_bytes`); SQL via identifier quoting + `validate_readonly_sql`
+ confined paths + SSRF checks; state keyed by tenant+subject; cross-tenant
via `AccessController` (audited, fail-closed); no secret in logs/errors;
CORS/CSP/HSTS unchanged or consciously updated; uploads/ZIP/exports bounded
and sanitized. Threat per control: `references/security.md`.

**Methodology** — new transformer is `fit`/`transform` split, no test/future
leakage, split-lock fail-closed; canonical metrics (`MetricsEvaluator`), CIs
via `bootstrap_ci`/conformal; baselines compared, AutoML lift reported;
fairness gate still blocks promotion.

**API & state** — routes use `require_permission`/`require_workbench_auth`
with least privilege; schemas in `schemas.py`; errors are `OmniForgeError`
subclasses; new `uap_*` keys follow the 6-step lifecycle (declare → write →
invalidate → persist → surface → frontend) in `references/api-and-state.md`;
`studioTypes.ts` updated for any response-key change.

**Frontend & packaging** — SPA rebuilt + `static_app` committed; no
`localhost` in browser code; `capabilities.py` stdlib-only; new optional dep
added as a `Pack` + `pyproject.toml` extra + regenerated locks; Docker
1-worker rule preserved.

**Tests & quality** — `ruff check`, `mypy`, full `pytest` green; new behavior
covered (positive + negative); latency benchmark within budget; no `print`
in library code; no bare `except: pass` without logging/metadata/test
(fallbacks must satisfy the 3 conditions in `references/security.md`).

## 6. Quick reference

- **Taxonomies.** Studios (15, `00`–`14`) and domains (18) are enumerated in
  `references/architecture.md` (Taxonomies) — read there for the full lists;
  they are the fastest to go stale, so they live in one place.
  Packs (12): `core-runtime, api-server, statistics, data-io, reporting`
  (required) + `boosting, deep, vision, nlp, foundation, automl, responsible`
  (optional). Tiers are dependency selections: `simple`
  (core+statistics+data-io+reporting), `advanced` (+api-server),
  `everything` (+all optionals). See `references/packaging.md` for
  installation details and the studio-coverage caveat.
- **Commands.** `pip install -e ".[dev,api,ui,reporting,data,forecasting]"` ·
  `pytest -q tests --cov=omniforge` · `ruff check src tests` ·
  `mypy src/omniforge` · `omniforge check|ui|serve|desktop` ·
  `cd frontend && npm ci && npm run build` ·
  `python scripts/build_locks.py` · `python scripts/latency_budget_benchmark.py`.
  Full map: `references/testing.md`, `references/recipes.md`.
- **Env (essentials).** `OMNIFORGE_ENVIRONMENT=production` +
  `OMNIFORGE_SECURITY__JWT_SECRET_KEY/ENCRYPTION_KEY/ARTIFACT_SIGNING_KEY/API_KEY`
  (≥32 bytes); `OMNIFORGE_API__RATE_LIMIT_PER_MINUTE` (300; `0` disables);
  `OMNIFORGE_API__CORS_ORIGINS`; `OMNIFORGE_PATHS__WORKSPACE_DIR`. Full
  template: `.env.example`.
- **Paths.** `.omniforge_workspace/{mlops_tracking.db, model_registry.db,
  audit.jsonl, artifacts/, cache/, feature_store/, tenants/, .secrets/}`.
  Health: `GET /health`, `GET /api/v1/health`, `GET /api/v1/version`,
  `omniforge check`. Docs: `/docs`, `/redoc`, `/openapi.json`.

## 7. How to update this skill

1. Note the new commit: `git rev-parse HEAD` + date. Update frontmatter
   `commit` / `last_verified` / `version` in the same PR as any code change
   that invalidates a claim.
2. Edit the **smallest** file that owns the claim: invariants/checklist here;
   subsystem detail in the owning `references/*.md`. Never re-expand
   `SKILL.md` with per-file detail — add it to references and link it.
3. Re-verify quantitative claims (`wc -l`, `grep -c "def test_"`,
   `grep -n "Programming Language" pyproject.toml`) and update or delete
   stale numbers. If a claim cannot be verified, delete it.
4. Keep `SKILL.md` ≤ ~2,500 words. If it grows past that, move content to
   references. Keep frontmatter `description` ≤ ~80 words with a negative
   trigger.
5. Run `ruff`/`mypy`/`pytest` if you changed code alongside the skill; the
   skill itself needs no build step.
6. **Re-derive from scratch** (no generator script exists): read the code in
   this order — `pyproject.toml` + `version.py`, `core/`, `config/`,
   `data/`, `eda/features/evaluation`, `models/`, decision engines,
   `pipelines/mlops/copilot/reporting`, `domains/`, `api/` + `ui/state.py`,
   `frontend/src/`, `tests/`, `configs/`, `packaging/installer/docker/scripts/`
   — then rewrite the owning section. The staleness workflow
   (`.github/workflows/skill-staleness.yml`) warns when code has moved past
   the pinned commit.

## 8. References (load on demand)

| File | Load when you need… |
|---|---|
| `references/architecture.md` | subsystem detail: core, config, data, EDA/features/evaluation, models, AutoML/forecasting/causal/optimization/simulation, pipelines, MLOps, copilot, reporting, domains |
| `references/security.md` | threat per control, RBAC, secrets/keys, audit anchor, fallback-vs-silent-failure rule |
| `references/api-and-state.md` | FastAPI routes, auth, jobs, workbench + studio router table, `uap_*` key lifecycle, frontend file map, port semantics |
| `references/testing.md` | test commands, category → file map, CI jobs, how to add tests |
| `references/recipes.md` | step-by-step: add model/domain/endpoint/tool/pack/report-section/config; run locally; CLI/launcher reference |
| `references/packaging.md` | packs/tiers/locks (both lock systems), why `static_app` is committed, PyInstaller/installer/Docker/scripts, upgrade playbook, troubleshooting, end-to-end flow |

*End of skill. When in doubt, read the cited file — this skill describes the
code as it exists at the pinned commit, not as any external documentation claims.*
