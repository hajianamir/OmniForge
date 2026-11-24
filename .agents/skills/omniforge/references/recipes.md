# OmniForge development recipes

> Reverified at the code revision in [SKILL.md](../SKILL.md). These are
> change checklists; read the named implementations and tests first.

## Run locally

On Python 3.12+ in an isolated environment:

```bash
python -m pip install -e ".[dev,api,ui,reporting,data,forecasting]"
omniforge check --json
omniforge ui --host 127.0.0.1 --port 8501 --no-browser
```

The check command can exit nonzero with a degraded report when dependencies or
resources are missing. The UI command serves both API and SPA; authentication
is required for the workbench. For a **trusted loopback-only development
session**, one option is
to set `OMNIFORGE_SECURITY__AUTO_BOOTSTRAP_LOCAL_UI=true` before starting the
server and visit `http://127.0.0.1:8501/` to receive local cookies. This
bootstrap grants an admin session, is disabled in production and must not be
used on an untrusted/exposed server. Otherwise configure an appropriate API
key or Bearer token; see [security](security.md).

For frontend development, keep the backend on 8501 and in another terminal run:

```bash
cd frontend
npm ci
npm run dev
```

Vite serves port 5173 and proxies `/api` and `/health` to 8501. With local
cookie bootstrap, first visit the backend SPA on the same hostname, then open
`http://127.0.0.1:5173/`; cookies are host-scoped. Use relative URLs in browser
code. For an externally proxied preview, configure host/origin acceptance and
use an appropriate authentication boundary; do not assume browser localhost
means the sandbox.

Other entry points: `omniforge serve --port 8000` serves the same API+SPA app;
`omniforge desktop` starts both listeners; `python -m omniforge --help` lists
CLI commands; `omniforge-desktop --help` describes launcher modes.
`examples/quickstart_end_to_end.py` writes synthetic demonstration outputs under
`.omniforge_workspace/quickstart_outputs`; it is not a substitute for an
independent holdout evaluation.

## Add a model

1. For a `ModelCatalog` model, subclass `src/omniforge/models/base.py::BaseOmniForgeModel`,
   implement `_fit` and `_predict`, set suitable family/framework metadata and
   honor task semantics. Override `_predict_proba` only when appropriate: the
   base already has a default projection. Not every engine needs to subclass it.
2. Register a constructor in `src/omniforge/models/registry.py`; keep optional heavy imports
   lazy where practical. Add backend-specific equation extraction or API/UI
   exposure only if needed; catalog registration alone does not expose it in
   every studio.
3. Test fit/predict and task guards, model parameters, training-only
   transformations, held-out scoring and signed save/load. Test optional
   backend availability, failure and fallback separately; record the effective
   backend without logging input data or sensitive exception messages.

## Add a domain

1. Implement `BaseDomainPlugin`'s `engineer_domain_features`,
   `compute_domain_kpis` and `generate_sample_dataset`; provide a module-level
   `plugin` instance and, where useful, validation rules and a domain YAML.
2. For a new built-in, update `src/omniforge/domains/__init__.py`,
   `src/omniforge/core/constants.py::BUILTIN_DOMAINS`, corresponding configuration, and any
   distribution metadata. Test registration, generated data and KPIs.
3. For external discovery, `DomainPluginRegistry.discover_entry_points()`
   accepts only a loaded **instance**. Export the instance, explicitly call or
   wire discovery, and test it. Existing `pyproject.toml` built-in entries
   export classes and are skipped by this method; built-ins work through
   manual registration. Do not assume discovery is invoked automatically.

## Add a studio endpoint or state key

1. Add/extend the request model in `src/omniforge/api/studios/schemas.py`; put the handler
   in the appropriate router and ensure the router is included from
   `src/omniforge/api/workbench.py`. Use `_sid` and session helpers for tenant/subject
   ownership. Decide and enforce the required action permission explicitly;
   router-level auth alone is insufficient for granular RBAC.
2. If caching is appropriate, follow `wb_cache_and_respond` and keep the
   response contract synchronized with `frontend/src/studioTypes.ts` and the
   calling component. Record lineage/exclusions where meaningful.
3. For new state, review defaults, dataset/target/split invalidation,
   stage-preservation, serialization filters, save/load, completion markers
   and frontend hydration. Do not rely on target/split setters to clear stale
   models today. Test unauthorized and cross-owner requests, mutation and
   restore, plus the supported non-happy paths.
4. If the endpoint trains a model, fit learned preprocessing on each training
   partition/fold and reuse that fitted pipeline for inference. Studio 06
   nonparametric and Studio 14 quick-train currently need particular care;
   follow [API and state](api-and-state.md), not their current leakage pattern.

## Add a Copilot tool

Implement/register a `CopilotTool` in `src/omniforge/copilot/tools.py`, following the
existing name, description, parameter metadata and handler conventions.
`list_tools()` returns this metadata for the provider prompt; it is not, by
itself, proof of provider-native JSON Schema validation. Registration alone
does not change the deterministic local planner: add intent routing in
`src/omniforge/copilot/agent.py` if local chat should select it. Review authentication,
allowed data sources, tool arguments, output/log privacy and provider
fallbacks. Test direct invocation and offline planning, plus any provider path
being changed.

## Add a pack or change dependencies

Update the catalogue and any intended tier in `src/omniforge/capabilities.py`,
and separately update the applicable `pyproject.toml` extras and lock inputs.
Regenerate and review the locks. Implement/test optional backend behavior and
capability probes where relevant. **First fix and test provisioner lock selection** if
multiple packs must install together; existing tier/pack locks and `uv pip
sync` do not guarantee the requested union. Test clean install, upgrade,
repair and actual offline execution, not just `profile_summary()`. See
[packaging](packaging.md).

## Add a report section or exporter

Add a `ReportSectionSpec` to `src/omniforge/reporting/sections.py`; update
`ReportGenerator` and only the chart, narrative, formula or exporter code the
feature actually needs. Test selection filtering and each affected output
format. Use `sanitize_spreadsheet_value` for untrusted spreadsheet cells and
headers, and audit other exports separately; not all dataset exports use the
reporting helper.

## Change configuration

Add a typed field to the relevant settings class where appropriate, update
`configs/default.yaml` and `.env.example` when applicable, and document the
actual environment name and precedence. `get_settings()` is cached, so tests
that change environment variables must manage/reset the cache. Some components
intentionally read env vars directly; trace the real consumer instead of
assuming every setting is uniformly applied. Test defaults, overrides,
production secret validation and deployment paths relevant to the change.
