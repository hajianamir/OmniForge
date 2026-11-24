# OmniForge Development Recipes

> Verified against commit `b7dc7ae` (2026-11-24); see `../SKILL.md` for provenance and authority policy.

## Add a model

1. Create `src/omniforge/models/<family>/<name>.py` with
   `class XxxModel(BaseOmniForgeModel)` implementing `_fit/_predict/
   _predict_proba`; set `model_family`; accept `**hyperparameters`.
2. Register in `models/registry.py` (`_register_defaults` eager, or lazy
   `_lazy_deep` pattern for heavy deps).
3. Add equation extraction in `evaluation/equations.py` if Studio 14 should
   show it; add metrics in `evaluation/metrics.py` if new.
4. Tests: fit/predict smoke, `get_params/set_params` round-trip,
   `save`/`load` with signature verification, leakage check, golden metric.
5. Optional: expose via Studio 06 (`s06_automl.py`) or Studio 09.
6. If the model needs an optional dependency, follow the fallback pattern in
   `security.md` (probe + `framework` metadata + pack `unlocks` + test).

## Add a domain

Create `domains/<slug>.py` (plugin class + module-level `plugin` instance
implementing `engineer_domain_features`, `compute_domain_kpis`,
`generate_sample_dataset`, optional `get_validation_rules`),
`configs/domains/<slug>.yaml`, register in `domains/__init__.py`
(`_BUILTIN_PLUGINS`), add entry point in `pyproject.toml`, add slug to
`BUILTIN_DOMAINS` in `core/constants.py`, extend domain coverage tests
(`test_p3_all_domain_plugins_load_and_generate_samples` pattern in
`test_audit_regressions.py`).

## Add a studio endpoint

Add the Pydantic request class to `api/studios/schemas.py`, implement the
handler in the matching `sNN_*.py` using `_wb()` + `wb_cache_and_respond`,
add the frontend call in `frontend/src/api.ts` + payload type in
`frontend/src/studioTypes.ts`, follow the `uap_*` key lifecycle in
`api-and-state.md` (declare → write → invalidate → persist → surface →
frontend), and cover it with tenancy + unauthenticated tests (see
`testing.md`).

## Add a Copilot tool

Implement `tool_*` in `copilot/tools.py`, register in
`_register_platform_tools` with name/description/JSON schema, ensure the
local planner can select it (deterministic routing), add a prompt hint in
`copilot/prompts.py`, test offline (`provider="local"`) with no API keys.

## Add a capability pack

Add `Pack(...)` to `capabilities.py` (`ALL_PACKS`, `TIERS` if preselected,
`FEATURE_UPGRADES` if in-app installable), mirror the extra in
`pyproject.toml`, regenerate locks (`scripts/build_locks.py`), add the
fallback path in code (`is_package_available` probe + NumPy/sklearn
fallback + `probe_*` entry + `unlocks` text), extend
`test_capability_packs.py`.

## Add a report section / exporter

Add `ReportSectionSpec` in `reporting/sections.py`, builder in
`reporting/charts.py`, exporter method in `reporting/exporters.py`
(sanitize spreadsheet values), narrative in `reporting/narrative.py`,
formula in `reporting/formulas.py`; verify section filtering + all formats
in `test_comprehensive_reporting.py` style.

## Change config

Add the field to the matching `*Settings` class in `config/settings.py`,
mirror in `configs/default.yaml` + `.env.example`, document the env var
(`OMNIFORGE_<SECTION>__<FIELD>`), test composition in
`test_core_and_config.py` style. Never read `os.environ` directly in library
code — go through `get_settings()`.

## Run locally

```bash
pip install -e ".[dev,api,ui,reporting,data,forecasting]"
omniforge check                       # environment diagnostics
omniforge ui --port 8501 --no-browser # workbench (API+SPA on :8501)
omniforge serve --port 8000           # REST API only (same app, :8000)
omniforge desktop                     # API (:8000) + UI (:8501) together
cd frontend && npm ci && npm run dev  # SPA dev on :5173 (proxied to :8501)
python examples/quickstart_end_to_end.py
```

CLI (`cli/main.py`): `version`, `domains [--sample --out --rows]`,
`quality`, `clean`, `eda`, `automl`, `drift`, `copilot`, `check [--json]`,
`desktop`, `monitor`, `sbom [--tier --verify]`, `serve`, `ui`. Heavy imports
are function-local (fast `--help`, `simple`-tier usable). Launcher
(`launcher.py`): `--mode {full,ui,api,cli,check}`, `--host`, `--api-port`,
`--ui-port`, `--no-browser`, `--json`, `--cli-args`; `python -m omniforge`
→ CLI; `omniforge-desktop` → `launcher:main`.
