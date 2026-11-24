# OmniForge Testing Reference

> Verified against commit `b7dc7ae` (2026-11-24); see `../SKILL.md` for provenance and authority policy.

## Commands

```bash
pip install -e ".[dev,api,ui,reporting,data,forecasting]"   # CI parity
pytest -q tests --cov=omniforge --cov-report=term-missing --tb=short  # CI command
pytest -v tests/unit            # fast unit only
pytest -v tests/integration     # cross-module
pytest -v tests/e2e             # API + CLI + SPA
ruff check src tests            # line-length 120, py312, E/F/I/W/B/UP (E501/B008 off)
mypy src/omniforge              # disallow_untyped_defs, warn_return_any, … (api/cli/ui decorator override)
python scripts/latency_budget_benchmark.py   # p95 budget gate
cd frontend && npm ci && npm run build       # SPA rebuild (then commit static_app)
make test / test-unit / test-integration / test-e2e / lint / typecheck
```

**Make vs CI:** `Makefile` targets wrap the same tools with local flags
(`pytest -v`, no coverage). CI (`.github/workflows/ci.yml`) runs
`pytest -q tests --cov=omniforge --cov-report=term-missing` plus ruff/mypy
with failure-summary annotations. If `make test` passes but CI fails, check
coverage/lint/typecheck first — CI is stricter, not different.

Pytest config (`pyproject.toml`): `testpaths=["tests"]`,
`addopts="-ra --strict-markers"`, markers `unit|integration|e2e|slow`.
`tests/conftest.py`: sets `OMNIFORGE_API__RATE_LIMIT_PER_MINUTE=0` (suite
issues >300 req/min; limiter is unit-tested directly — never disable it
elsewhere), autouse seed fixture (`set_global_seed(42)`), fixtures
`classification_df` (220×6), `regression_df` (200×6), `timeseries_df`
(110 periods), `causal_df` (280 rows, ATE 3.2) via `SyntheticDataGenerator`,
`workspace_tmp`.

## What proves what (categories → representative files)

| Category | Representative test file(s) | Proves |
|---|---|---|
| Leakage-safety | `test_anti_leakage_and_pipeline.py`, `test_phase2_statistical_and_pipeline_hardening.py` | splitter/cleaner/anomaly/clustering/AutoML never learn from test/future rows |
| Security & tenancy | `test_bola_session_isolation.py`, `test_platform_admin_role.py`, `test_p0_security_and_runtime_hardening.py`, `test_p1_13_pseudonym_salt.py`, `test_p1_14_sql_identifier_allowlist.py` | tenant/subject isolation, unauthenticated rejection, salt strength, SQL allowlist |
| Audit & headers | `test_audit_anchor_and_browser_headers.py`, `test_audit_regressions.py` (large) | checkpoint anchor, CSP/HSTS, hash-chain verification, tamper detection |
| Statistical correctness | `test_p1_mathematical_and_statistical_integrity.py`, `test_pass2_ml_and_statistics.py`, `test_phase3_econometrics_conformal_and_governance.py` | golden values (AIPW/PSM SE, association-rule metrics), conformal, econometrics |
| Data layer | `test_data_layer.py`, `test_format_audit_and_exclusions.py`, `test_p1_10_dataframe_fingerprint.py` | ingestion/cleaning/quality/format audit, fingerprint stability |
| EDA & features | `test_eda_and_features.py`, `test_phase5_green_ai_explainability_pii.py` | profiling, selection, SHAP/PII/carbon |
| Modeling & decision | `test_modeling_and_decision_engines.py`, `test_phase2_advanced_engines.py`, `test_phase4_pillars.py`, `test_phase6.py` | baselines, AutoML, forecasting, causal, optimization, simulation |
| API & studios | `test_pass3_api_tenancy_and_studios.py`, `test_pass4_reporting_cli_docker_spa.py` | tenancy, studio endpoints, CLI/Docker/SPA |
| Reporting | `test_comprehensive_reporting.py`, `test_model_card_pdf.py` | DOCX/PDF/XLSX/CSV exports, section filtering, model-card PDF |
| Serialization | `test_model_serialization_and_unpickler.py` | HMAC verify + restricted unpickler refusal |
| Packs & installer | `test_capability_packs.py`, `test_installer_iss.py`, `test_packaging_and_launcher.py` | tier cumulativeness, dedup pins, ISS validity, launcher |
| Rate limits | `test_request_limits.py`, `test_rate_limit_shared.py` | sliding window, shared (Redis) behavior |
| Third-party claims | `test_third_party_claims.py` | capability claims match installed backends/fallbacks |
| Workspace & themes | `test_workspace_session_and_themes.py` | `.uap` save/load, session isolation, themes |
| Regressions | `test_regression_items_10_23.py`, `test_phase1_critical_modifications.py`, `test_ct_kill_switch.py` | historical bug fixes stay fixed; CT kill-switches |
| Full platform | `tests/integration/test_full_platform.py` | end-to-end lifecycle |
| E2E | `tests/e2e/test_api_and_cli.py`, `test_omniforge_task33/34/35/37.py`, `test_workbench_spa.py` | live API/CLI/SPA serving |

No per-file test counts are given here — they drift as tests are added.
Use `grep -c "def test_" <file>` at the pinned commit if you need one.

## CI (`.github/workflows/ci.yml`, 4 jobs)

`quality-and-tests` (py3.12/3.13: install `[dev,api,ui,reporting,data,
forecasting]` → ruff → mypy → pytest+cov), `lock-verify`
(`pip install --require-hashes -r requirements.lock` + smoke import),
`e2e` (Node 22 `npm ci` → `npm run build` → **fail if `static_app` diff**
→ pytest `tests/e2e`), `latency-budget`
(`latency_budget_benchmark.py`). `windows-installer.yml` covers the installer.

## Adding tests

- **New engine/model:** unit test with `SyntheticDataGenerator` fixtures + a
  leakage assertion (fit on train, transform test; assert no test-row
  influence) + a golden-value test where math is hand-computable (follow
  `test_association_rules_exact_hand_calculated_metrics`,
  `test_causal_aipw_and_psm_golden_se`).
- **New endpoint:** tenancy test (same header, two principals → isolated
  state) + unauthenticated-rejection test + `wb_cache_and_respond` envelope
  check.
- **New security control:** negative test first (attack must fail), then
  positive.
- **New fallback:** assert the fallback path explicitly (framework metadata +
  probe output), per `security.md`.
