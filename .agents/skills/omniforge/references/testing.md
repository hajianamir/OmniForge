# OmniForge testing reference

> Reverified at the code revision in [SKILL.md](../SKILL.md). A test proves
> the exercised path and fixtures, not every route or deployment configuration.

## Local commands

Use a supported Python (package metadata requires **3.12+**), with the
repository root as the current directory:

```bash
python -m pip install -e ".[dev,api,ui,reporting,data,forecasting]"
ruff check src tests
mypy src/omniforge
pytest -q tests --cov=omniforge --cov-report=term-missing --tb=short
python scripts/latency_budget_benchmark.py
```

Focused examples:

```bash
pytest -q tests/unit/test_workbench_inference_preprocessing.py
pytest -q tests/unit/test_persistent_workbench_jobs.py tests/unit/test_persistent_workbench_processes.py
pytest -q tests/unit/test_feature_freshness_monitoring.py tests/unit/test_registry_stage_and_fairness.py
pytest -q tests/unit/test_fallback_observability.py
pytest -q tests/e2e --tb=short
```

For frontend changes: `cd frontend && npm ci && npm run build`, then inspect
and commit `src/omniforge/ui/static_app/`. `Makefile` also supplies `make test`,
`make test-unit`, `make test-integration`, `make test-e2e`, `make lint`, and
`make typecheck`; its pytest targets use different flags and do not replicate
the coverage invocation. `pyproject.toml` configures strict markers, Ruff and
mypy. `tests/conftest.py` disables the API rate limit for the suite and seeds
fixtures; direct limiter tests remain relevant. A local run with an unsupported
Python/dependency combination is supplementary, not evidence that the
supported CI matrix passes.

## Representative coverage and limits

All filenames below are in `tests/unit/` unless otherwise indicated.

| Area | Files | Scope to inspect |
|---|---|---|
| Split/evaluation | `test_anti_leakage_and_pipeline.py`, `test_phase2_statistical_and_pipeline_hardening.py` | Exercises specific splitter, cleaner, anomaly and modeling cases. It does **not** establish leakage-free behavior for every studio; see [API and state](api-and-state.md). |
| Scoring | `test_workbench_inference_preprocessing.py` | Studio 06 fitted preprocessing, mixed data scoring and cancellation before publication. Add per-path holdout/fold leakage tests when changing training. |
| Jobs | `test_persistent_workbench_jobs.py`, `test_persistent_workbench_processes.py` | Owner scoping, capacity, cooperative cancellation, cross-process status and lost-worker failure. |
| Freshness | `test_feature_freshness_monitoring.py` | TTL boundaries, unknown timestamps, purges and tenant-scoped endpoint. |
| Governance | `test_registry_stage_and_fairness.py`, `test_p1_mathematical_and_statistical_integrity.py` | Stage validation, fairness result serialization and selected statistical regressions. |
| Auth/tenancy | `test_bola_session_isolation.py`, `test_platform_admin_role.py`, `test_p0_security_and_runtime_hardening.py` | Representative unauthenticated, role and isolation cases; add a negative and positive test for each new route. |
| Audit and privacy | `test_audit_anchor_and_browser_headers.py`, `test_audit_regressions.py`, `test_p1_13_pseudonym_salt.py` | Anchors/headers, audit behavior and salt handling; verify deployment controls separately. |
| Serialization | `test_model_serialization_and_unpickler.py`, `test_workspace_session_and_themes.py` | Signed model/workspace behavior, unpickler restrictions and theme/session cases. |
| Data and connectors | `test_data_layer.py`, `test_format_audit_and_exclusions.py`, `test_p1_14_sql_identifier_allowlist.py` | Selected ingestion, cleaning, exclusions and SQL identifier cases. |
| Fallbacks | `test_fallback_observability.py`, `test_third_party_claims.py` | Specific fallback behaviors; the former guards four reviewed files, not every exception handler in the repository. |
| Packs and installer | `test_capability_packs.py`, `test_installer_iss.py`, `test_packaging_and_launcher.py` | Catalogue and installer/launcher checks, not proof that arbitrary tier + pack unions or offline upgrades work. |
| Reports | `test_comprehensive_reporting.py`, `test_model_card_pdf.py` | Selected report sections and export behavior, not all independent data exports. |
| Integration/E2E | `tests/integration/test_full_platform.py`, `tests/e2e/test_api_and_cli.py`, `tests/e2e/test_workbench_spa.py` | Broad lifecycle and API/CLI/SPA smoke tests. |

When adding a feature, add the right failure cases as well as success cases:

- **Data/modeling:** distinguish raw input schema, training partition,
  evaluation partition and each CV fold. Use holdout-only categories and
  extreme holdout values to detect leaked encoders/medians. Test invalid
  split locks and changes to target/dataset separately.
- **Endpoint/state:** verify missing credentials, least-privilege decisions,
  same session ID across different tenants/subjects, cache invalidation and
  save/restore. Do not assume router-level authentication implies RBAC.
- **Job:** test owner mismatch, saturation, queued/running cancellation,
  commit races, worker expiry and cross-process polling with a shared DB.
- **Fallback:** simulate backend initialization and inference failure; assert
  the result, structured diagnostic, absence of sensitive input and exception
  text, and no spurious warning for expected absence/disabled behavior.
- **Installer:** inspect the selected lock and compare actual installed
  distributions/imports for Simple, Advanced, Everything, single/multiple
  packs, upgrade/repair and network-isolated wheelhouse provisioning.
- **Security/export:** test tampered and unsigned artifacts, key continuity,
  archive limits, input confinement, redirects and spreadsheet formula strings
  at the actual export endpoint.

## CI and skill freshness

`.github/workflows/ci.yml` runs on PRs to `main` and pushes to `main`:

1. `quality-and-tests`: Python 3.12 and 3.13 matrix; editable install with the
   extras above, Ruff, mypy, and the full pytest command with coverage.
2. `lock-verify`: Python 3.12 installs `requirements.lock` with hashes and
   smoke-imports selected packages. It is not a recompile of all installer
   locks.
3. `e2e`: Node 22 runs `npm ci` and `npm run build`, checks tracked changes
   under `static_app` with `git diff --quiet`, then runs the E2E tests on
   Python 3.12. That tracked-diff command alone does not detect untracked files.
4. `latency-budget`: runs `scripts/latency_budget_benchmark.py`.

`.github/workflows/windows-installer.yml` is separate, configured for manual
runs and version-tag pushes; it regenerates Windows locks, stages the
bootstrap, tests the catalogue/ISS, provisions Simple and Advanced and builds
an Inno installer. It is not a PR matrix of all selections.

`.github/workflows/skill-staleness.yml` is **warn-only** on PRs/pushes to
`main`. It extracts one quoted 40-hex pin and counts commits in
`<pin>..HEAD` touching only `src/omniforge`, `frontend`, `tests` and `configs`.
It does not check uncommitted files or validate pin ancestry; packaging,
installer, scripts, Docker, `pyproject.toml` and workflows are outside that
path filter. Invalid/missing pins also warn rather than fail. A green check
therefore cannot prove the skill is current; follow the full-review procedure
in [SKILL.md](../SKILL.md#maintaining-this-skill).
