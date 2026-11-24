---
name: omniforge
description: >-
  Review, test, develop, package, or improve the OmniForge analytics workbench
  in this repository: its Python SDK, FastAPI API, React SPA, and domain plugins.
  Do not use for unrelated Python questions or other projects named OmniForge.
version: "4.0.0"
repository: "github.com/hajianamir/OmniForge"
commit: "f63d3f76b5aefbd8e902fdab419061dd91b6afa5"
last_verified: "2026-11-21"
derived_by: "Arena.ai coding agent; reverified all seven skill files against source, tests, configuration, frontend, packaging, and workflows, with targeted runtime probes"
---

# OmniForge repository skill

Use this as a **source index and review guide**, not a substitute for reading
implementation. Paths to code, commands, and workflows are relative to the
repository root; links to references are relative to this file. Code and tests
win over this skill, comments, and other documentation. Inspect the actual
function, its callers, and tests before changing or relying on behavior.

The pin above identifies the **code revision inspected**, not the later
documentation/maintenance-only commit. All seven files were re-derived for
this revision. The coverage is the claims in this skill, not an assertion
that every behavior in this repository was exhaustively audited. If the
working tree or a newer commit changes a cited area, re-check it. See
[maintenance](#maintaining-this-skill).

## Identity and entry points

- `pyproject.toml` and `src/omniforge/version.py` identify package
  `omniforge` version **4.0.0**, Python **>=3.12**, setuptools `src/` layout,
  and Apache-2.0 licensing. CI exercises Python 3.12 and 3.13; classifiers for
  later versions are not proof that all optional packages work there.
- The version metadata says `api_version="v2"`, but actual REST routes are
  `/api/v1/*` and `/api/workbench/*` (`src/omniforge/api/main.py`); do not
  infer route prefixes from the metadata.
- `omniforge` runs the Typer CLI; `omniforge-desktop` runs the launcher.
  `create_app()` serves both REST and the compiled React/Vite SPA. Defaults
  are port 8000 for `serve`, 8501 for `ui`, and 5173 for the Vite dev server.
  `desktop` starts API and UI listeners. See [API and state](references/api-and-state.md).
- React defines 15 studios (`00`–`14`) in `frontend/src/types.ts`; the older
  Python `src/omniforge/ui/app.py::STUDIOS` metadata begins at `01`. Eighteen
  built-in domain plugin instances are registered in
  `src/omniforge/domains/__init__.py`.
  The [architecture reference](references/architecture.md) maps subsystems.

## Guardrails for changes

1. **Imports and models.** `src/omniforge/__init__.py` is a lazy SDK facade;
   keep `src/omniforge/capabilities.py` stdlib-only for the installer. Model
   catalog entries normally use `BaseOmniForgeModel`
   (`src/omniforge/models/base.py`), whose abstract hooks are `_fit` and
   `_predict`; `_predict_proba` has a default. Other engines are not
   necessarily model subclasses. See [recipes](references/recipes.md).
2. **Evaluation.** Fit learned transformations on training data and validate
   held-out data independently. Split Lock is honored by some paths, not a
   blanket guarantee across all studios. Studio 05 can report a split lock
   even when fitting the full dataset without one; Studio 06 nonparametric and
   Studio 14 quick-train fit preprocessing before their evaluation splits.
   Do not present these paths as leakage-free without fixes and tests. See
   [API and state](references/api-and-state.md).
3. **State and ownership.** Workbench session IDs are scoped to authenticated
   tenant and subject; job ownership uses a separate owner key. Preserve those
   boundaries. Dataset/exclusion changes invalidate selected downstream state,
   but target and split-lock changes currently do not reliably invalidate
   models/results. Inspect and test each mutation rather than assuming a
   universal invalidation rule.
4. **Security.** Keep authentication, route-specific permissions, tenant
   isolation, bounded uploads, connector guards, signatures and restricted
   deserialization intact. Do not assume every authenticated workbench route
   has an action-specific RBAC check or that PII/log redaction is exhaustive.
   Follow [security](references/security.md) for exact scope and key handling.
5. **Persistence.** Workspace state payloads are HMAC-signed, not encrypted;
   manifest metadata is not authenticated and exports may contain data and
   fitted models. Keep signing/encryption keys available
   during trusted migration. Never bypass verification or blindly rotate or
   delete keys to make an old artifact load. See [packaging](references/packaging.md).
6. **Frontend and dependencies.** Rebuild and commit
   `src/omniforge/ui/static_app/` after frontend changes. Browser requests
   should use relative API URLs. Installer tiers and pip extras differ;
   current per-pack lock selection is not a reliable arbitrary union. Inspect
   the selected lock and test real provisioning before promising combinations.
7. **Observable degradation.** Preserve intentional fallbacks, but report
   failed attempted backends without exposing sensitive exception messages or
   dataset content; test both failure and expected absence. Do not claim all
   existing fallback paths already meet this standard.

## Operational limits worth checking first

- Workbench sessions and job callables are process-local. Job status/results
  and cancellation are SQLite-backed; expired workers fail unfinished jobs
  rather than resume them. Multiple workers require shared, reliably locking
  SQLite for job status **and** sticky routing or shared workspace state for
  workspace operations. Docker defaults to one API worker.
- `simple` tier omits the `api-server` pack even though the React workbench
  needs FastAPI/Uvicorn. Tier/pack catalogue text is not proof of a working
  installation or of frontend install buttons.
- Local Copilot is a deterministic intent planner, not an LLM. Optional
  backends may be absent, disabled, fail, or need separately available weights;
  inspect the active backend and its own fallback behavior.
- Workspace archives and the default SQLite/Parquet stores hold user data;
  they are not disposable unless the owner has explicitly accepted data loss.
  Docker's workspace volume does not automatically capture every hard-coded
  or independently configured store path; verify mounts and backups. PII
  detection is heuristic, and CSP is report-only by default.

## Quick validation

On a supported Python version, from the repository root:

```bash
python -m pip install -e ".[dev,api,ui,reporting,data,forecasting]"
ruff check src tests
mypy src/omniforge
pytest -q tests --cov=omniforge --cov-report=term-missing --tb=short
python scripts/latency_budget_benchmark.py
```

For frontend changes: `cd frontend && npm ci && npm run build`, then review
and commit the generated static assets. [Testing](references/testing.md)
describes focused suites and the actual CI triggers; [recipes](references/recipes.md)
has local launch and extension examples.

## Maintaining this skill

1. Check `git status --short` and `git rev-parse HEAD`; identify the exact
   committed code revision and any uncommitted changes being inspected.
2. For a **partial correction**, edit the affected guidance but retain the
   existing full-verification pin and say what was rechecked. Do not advance
   the global pin after merely changing a few claims.
3. For a **full refresh**, inspect every retained claim in this file and all
   six references against current source, callers, tests, frontend, config,
   packaging, and workflows. Replace or remove unsupported claims; include
   limitations and source pointers. Review security and compatibility advice
   particularly carefully.
4. Only after that review, set exactly one frontmatter line of the form
   `commit: "<40 lowercase hex digits>"` to the inspected **committed code**
   revision and update `last_verified` and provenance. If code and skill
   change together, first establish a committed code revision that matches
   what was inspected; do not pin the subsequent documentation-only commit.
5. Check reference links, source paths, example commands, `git diff --check`,
   and relevant tests. The warn-only `.github/workflows/skill-staleness.yml`
   checks commits since the pin touching `src/omniforge`, `frontend`, `tests`,
   or `configs` on PRs to `main` and pushes to `main`. It does **not** check
   uncommitted changes, pin ancestry, or changes limited to installer,
   packaging, Docker, scripts, `pyproject.toml`, or workflows. Inspect those
   separately; a green check is not proof of a complete review.

## References (load as needed)

| Reference | Use for |
|---|---|
| [Architecture](references/architecture.md) | Studios, domains, config, subsystem and model map |
| [API and state](references/api-and-state.md) | Routes, ownership, jobs, scoring, state lifecycle, frontend |
| [Security](references/security.md) | Auth, secrets, serialization, connectors, privacy, fallbacks |
| [Packaging](references/packaging.md) | Packs, lock selection, installers, Docker, safe migration |
| [Testing](references/testing.md) | Commands, regression suites, CI and staleness semantics |
| [Recipes](references/recipes.md) | Local launch and adding models, domains, routes, tools, packs and reports |
