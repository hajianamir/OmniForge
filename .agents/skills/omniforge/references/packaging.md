# OmniForge Packaging, Upgrade & Troubleshooting Reference

> Verified against commit `b7dc7ae` (2026-11-24); see `../SKILL.md` for provenance and authority policy.

## Capability packs, tiers & locks

`src/omniforge/capabilities.py` is the single source of truth for the
Windows installer (stdlib-only — see SKILL.md invariant 2). 12 packs:

| Pack id | Kind | Unlocks |
|---|---|---|
| `core-runtime` | required | numpy/pandas/sklearn/scipy/pydantic/structlog/typer/rich/plotly/networkx… |
| `api-server` | required (Advanced+) | FastAPI/Uvicorn/JWT/crypto/prometheus |
| `statistics` | required | statsmodels — forecasting, causal DML/IV, hypothesis tests |
| `data-io` | required | pyarrow/duckdb/SQLAlchemy/fsspec/lxml |
| `reporting` | required | docx/pptx/reportlab/nbformat/openpyxl/matplotlib |
| `boosting` | optional | XGBoost/LightGBM/CatBoost (fallback: HistGB) |
| `deep` | optional | torch/lightning (fallback: NumPy) |
| `vision` | optional | OpenCV/scikit-image (fallback: Pillow/NumPy) |
| `nlp` | optional | NLTK/spaCy/gensim (fallback: TF-IDF) |
| `foundation` | optional | transformers/timm/CLIP/ultralytics (needs `deep`) |
| `automl` | optional | Optuna/FLAML (GP-UCB + random/grid included) |
| `responsible` | optional | SHAP + codecarbon (fallback: permutation + estimated carbon) |

Tiers: `simple` = core+statistics+data-io+reporting; `advanced` = simple +
api-server; `everything` = advanced + all optionals.

**Coverage caveat:** `capabilities.py`'s “Studio 01–05 + Universal
Diagnostics” comment uses older numbering where 01 combined ingestion and
cleaning; React now separates 00 and 01, while legacy `/s01/*` routes remain.
Tiers select dependencies, not studio access: `simple` omits FastAPI, so
verify a clean install before promising React workflows without `api-server`.

**Two lock systems (do not confuse them):**
- `requirements.lock` — hash-pinned runtime lock for pip
  (`pip install --require-hashes`), compiled **universal / Python 3.12**
  (`requirements.txt` header). `requirements.in` is the floating intent.
- `locks/*.lock` — installer locks compiled by `scripts/build_locks.py`
  for **Windows / Python 3.13** (`PYTHON_PLATFORM`/`PYTHON_VERSION` in that
  script) so every wheel is a prebuilt binary. `locks/{simple,advanced,
  everything,dev}.lock` + `pack-*.lock`.

Regenerate: `python scripts/build_locks.py` (needs `uv`); verify with
`--check`. `pyproject.toml` extras mirror the packs (`api, redis, ui,
reporting, data, connectors, mlops, forecasting, deep, boosting, automl,
automl-java, foundation, responsible, nlp, vision, audio, build, dev,
experimental, all`).

## Why `static_app/` is committed

`src/omniforge/ui/static_app/` is a **committed build artifact** (invariant
8). This is deliberate, not an oversight: the `simple` tier and the desktop
installer ship **without Node.js**, so the prebuilt SPA must be in the repo
for `create_app()` to serve it. CI (`e2e` job) fails if the committed bundle
is stale vs `frontend/` source. After any frontend change:
`cd frontend && npm ci && npm run build` and commit the rebuilt assets.

## PyInstaller / installer / Docker / scripts

| Area | Files | Notes |
|---|---|---|
| PyInstaller | `packaging/build_exe.py`, `packaging/pyinstaller/*.spec`, `packaging/entry_{desktop,cli}.py`, `packaging/generate_icon.py`, `packaging/assets/` | Validates env, ensures icons (Pillow), builds onedir, writes SHA-256 checksums, optional Inno compile. Entries call `multiprocessing.freeze_support()`. |
| Inno Setup | `packaging/inno_setup/omniforge_installer.iss` | Full-exe variant. |
| Tiered installer | `installer/omniforge_installer_tiers.iss`, `installer/provision.py` | Small payload (app + embeddable CPython + `uv.exe` + `locks/*.lock`) + wizard (Simple/Advanced/Everything + per-pack checkboxes). `provision.py` bootstraps a private venv, installs only selected packs (local wheels first), atomic `provision.json`, exit codes 0/2/3/4/5, offline `/WHEELHOUSE`. `LaunchMode=py` (default shim) vs `=exe` (PyInstaller). |
| Bootstrap | `scripts/build_bootstrap.py` | Stages `dist/omniforge-bootstrap/`; ISS guards fail fast on stale payload. |
| Launcher shim | `scripts/make_launcher_shim.py` | Retargets `venvlauncher.exe` (0 MB, offline) + `.cmd`/`.lnk` fallback. Refuses the fake “copy pythonw.exe” trick. |
| SBOM | `scripts/generate_sbom.py` | CycloneDX via `core/sbom.py`. |
| Benchmarks | `scripts/run_benchmark.py`, `scripts/latency_budget_benchmark.py` | Domain/baseline/AutoML/carbon smoke; p95 budget gate in CI. |
| ISS check | `scripts/check_iss.py` | Validates Inno scripts. |
| Docker | `docker/Dockerfile`, `docker/docker-compose.yml`, `docker/entrypoint.sh` | Multi-stage `python:3.12-slim`, non-root `omniforge:1000`, healthchecks. Compose: `omniforge-api` (:8000) + `omniforge-ui` (:8501, waits for healthy API), shared volume, secrets via env. Entrypoint modes `api` (uvicorn defaults to **1 worker** — sessions and job execution are process-local), `ui`, `worker`, `cli`. Job status is SQLite-backed; see `api-and-state.md`. |
| Windows helpers | `install.bat`, `start.bat`, `build_exe.bat`, `build_installer.bat` | Thin wrappers. |
| Build reqs | `requirements-build.in/.txt/.lock` | PyInstaller + Pillow + setuptools/wheel/build. |

## Upgrade / migration playbook

1. **Version bump:** `src/omniforge/version.py`, `pyproject.toml`,
   `frontend/package.json`, Docker image tag, installer `MyAppVersion`,
   `CHANGELOG.md`. Bump `__api_version__`/`__schema_version__` only on
   breaking change; migrate `.uap` loaders and registry DBs accordingly.
2. **Dependencies:** edit `requirements.in` / `pyproject.toml` /
   `capabilities.py` → regenerate `requirements.lock` (universal, 3.12) and
   `locks/*.lock` (Windows, 3.13) → verify (`--check`, `omniforge sbom
   --verify`) → CI `lock-verify` must pass.
3. **SPA:** rebuild + commit `static_app` (above).
4. **Installer:** `python scripts/build_bootstrap.py` → ISCC compile →
   `scripts/check_iss.py` → `windows-installer.yml`. Test all tiers +
   per-pack checkboxes + offline wheelhouse.
5. **Data/DB:** `model_registry.db` / `mlops_tracking.db` are SQLite;
   `.omniforge_workspace/` is disposable — never migrate secrets keyfiles;
   rotate `OMNIFORGE_SECURITY__*` on upgrade.
6. **Breaking API:** update `api/schemas.py` + `studios/schemas.py`,
   `frontend/src/studioTypes.ts`, handbook, `docs/api_reference.md`; keep
   old routes as deprecated shims for one minor version where feasible.
7. **Deprecation:** `DeprecationWarning`, keep `uap_*` aliases, remove only
   after tests + SPA migrated. Never silently rename a `uap_*` key or a
   `ModelCatalog` id — both are persisted.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `static_app stale` CI failure | `frontend/` changed without rebuild | rebuild + commit `static_app` |
| 429s in manual testing | limiter at 300/min | raise `OMNIFORGE_API__RATE_LIMIT_PER_MINUTE` or `0` locally; never in prod |
| 413 on upload | body cap 256 MB | raise `OMNIFORGE_API__MAX_REQUEST_BODY_BYTES` |
| `OmniForgeError` on model load | HMAC mismatch / unpickler refusal | re-save with current key; add class to allowlist (`core/security.py`) |
| Studio results vanish after target change | downstream invalidation (by design) | re-run dependent studios; check `_reset_downstream_state` |
| `torch` import error in simple tier | heavy dep leaked to top level | move behind `is_package_available` probe + NumPy fallback |
| Copilot returns local-planner text | no provider keys | set `OPENAI_API_KEY` etc. or keep `provider="local"` (offline by design) |
| Registry promotion blocked | fairness gate | run `FairnessAuditor`, fix disparate impact, re-register |
| Docker sessions lost on restart | in-process state + >1 worker | keep `OMNIFORGE_API_WORKERS=1` or add sticky sessions/external state |
| `omniforge check` fails | missing pack for tier | install the tier's packs (`capabilities.profile_summary`) or pick a lower tier |

## End-to-end flow (one pass)

Ingest (`DataIngestionEngine` → format/intake audit → `load_new_dataset`)
→ plan & split-lock (`AnalysisSpec` + `LeakageSafeSplitter` +
`LeakageGuard`) → clean (`DataCleaner.fit_transform_split`, deleted-row
provenance) → diagnose/EDA → features (fit on train only) → model
(`AutoMLPipelineSearch`, baselines + lift, equations) → RAI (SHAP/fairness/
conformal/model card) → decide (forecast/causal/optimize/simulate/specialized)
→ pipeline (DAG executor, cached/carbon-tracked) → ship (registry fairness
gate → serving; drift) → report (section-filtered exports) → copilot
throughout → supervise (Studio 14) → persist (`.uap` ZIP + SHA-256).
