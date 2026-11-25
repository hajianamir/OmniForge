# OmniForge — Enterprise Analytics & AI Workbench v4.0.0

**OmniForge** is a production-grade, enterprise-scale, modular Python 3.12+ platform that unifies the complete data-to-decision lifecycle — from multi-source ingestion and 6-dimensional data quality profiling to automated EDA, leakage-safe feature engineering, baseline-first modeling, deep learning, time-series forecasting, causal inference, mathematical optimization, simulation, responsible AI, AutoML, DAG orchestration, MLOps, an air-gapped AI Copilot, 18 industry domain plugins, a FastAPI backend, a **FastAPI + React TypeScript Enterprise Workbench** (pre-compiled, code-split SPA — featuring 60fps Canvas plots, server-side paginated data grid, interactive SVG DAG/network graphs; Data Ingestion/Cleaning → Clustering & Anomaly Screening → Universal Diagnostics/Baselines/Risk → EDA → Feature Engineering → AutoML & Deep Learning vs. Baselines → Responsible AI → Decision Science → Specialized AI → DAG Pipeline → MLOps → Executive Reporting → AI Copilot → Live Supervisory Control Room), a Typer CLI, and one-click executive reporting.

OmniForge runs **fully offline and air-gapped** with zero third-party LLM dependency for its deterministic Copilot planner, digital twin MPC controller, and causal root-cause engine — making it suitable for regulated, secure, and disconnected environments.

---

## What's New in v4.0.0 (Phase 6 — Future Developments II)

v4.0.0 is a major SemVer bump from 3.3.0. It delivers three contracted capability clusters plus version alignment across every artifact:

| Capability Cluster | Code | What It Delivers | Where It Lives |
|---|------|------------------|----------------|
| **C5 — Causal Drift Root-Cause Attribution** | `DriftRootCauseAnalyzer` + `CausalDiscovery` | Connects `DriftDetector` PSI/Wasserstein shifts → Delta-attribution per drifted feature → structural DAG upstream-ancestor traversal to isolate root-cause upstream sensor/source shifts from downstream correlated symptoms; emits remediation prescriptions | `src/omniforge/mlops/root_cause.py`; Studio 11 (`/s11/root-cause`) |
| **C7 — Closed-Loop Digital Twin MPC** | `ClosedLoopMPCController` | Constrained receding-horizon quadratic control `argmin ||Hx̂−r||_Q² + ||u||_R²` subject to input/state safety bounds via SLSQP; constraint softening with slack variables, solve-time tracking, warm-start alignment, R-matrix dimension fix, bounded relaxation fallback; exposed via `POST /s09/twin-control` with interactive trajectory charts | `src/omniforge/simulation/closed_loop.py`; Studio 09 (`/s09/twin-control`) |
| **C4 — Air-Gapped Deterministic Copilot** | `CopilotToolRegistry` + `_chat_local_planner` | 10 new deterministic offline tools spanning all 15 Studios: intake/PII auditing (`pii_audit`), Split Lock enforcement (`split_lock`), honest data cleaning, 18-domain validation (`domain_validate`), Green-AI Pareto AutoML (`pareto_automl`), causal root-cause diagnosis (`root_cause_diagnosis`), closed-loop twin control (`twin_control`), model register/promote (`register_model`/`promote_model`), studio listing/overview (`list_studios`/`studio_overview`) | `src/omniforge/copilot/tools.py`; Studio 13 (`/s13/chat`) |
| **Version Alignment** | `pyproject.toml`, `src/omniforge/version.py` | Canonical SemVer 3.3.0 → 4.0.0 across SDK, pyproject.toml, native hubs, exporters, SPA manifests, and tests | All packages |

**Also included (bug-fix and hardening commits merged into v4.0.0):**
- PII `detect()` now uses `pd.api.types.is_string_dtype()` instead of `dtype != object`, correctly catching pandas `StringDtype` columns (pandas 3.x default)
- PII `_audit()` now passes `tenant_id=self.tenant_id` to `log_event()` (was passing `resource_id`, causing a silently swallowed `TypeError`)
- `structlog` pre-declared as `ModuleType | None` with alias import, resolving mypy type errors
- Ruff import sort fixed in Studio 14 supervisory router
- Removed unused `best_latency`/`best_carbon` and renamed ambiguous `l` → `latency` in AutoML pipeline search
- `interpretability.explain_diff` returns all features (removed `[:10]` cap); `conformal_explain` `half_width` computed from already-rounded interval bounds

---

## Core Architecture & Modules (`src/omniforge/`)

| Subsystem | Package | Key Capabilities |
|---|---|---|
| **Core & Security** | `omniforge.core`, `omniforge.config` | Pydantic v2 Settings + YAML composition, structured logging with secret redaction, rich custom exception hierarchy (`error_code`, `severity`, `suggestions`, `trace_id`), AES/Fernet encryption, JWT + RBAC/ABAC, hash-chained audit logs, GDPR/HIPAA/SOX compliance checks, Green AI Carbon Tracker |
| **Data Layer** | `omniforge.data` | 30+ data sources (CSV/TSV/Excel/Parquet/Feather/JSON/JSONL/ORC/Avro/HDF5/Pickle/XML/YAML/TOML/SAS/SPSS/Stata/SQL/DuckDB/MongoDB/REST/S3/GCS/Azure/In-memory), fit/transform `DataCleaner`, `DataValidator`, 6D `DataQualityProfiler`, content-addressed `DatasetVersionManager`, Feast-compatible `LocalFeatureStore`, Gaussian Copula `SyntheticDataGenerator`, Multimodal extractors |
| **EDA & Insights** | `omniforge.eda` | Statistical profiling, parametric/non-parametric hypothesis tests, interactive Plotly dark/light visualizations, rule-based executive narratives, standalone HTML EDA reports |
| **Feature Engineering** | `omniforge.features` | Leakage-safe imputation/encoding/scaling, interaction/polynomial/datetime/time-series lags & rolling windows, 8 feature selectors (Variance, Correlation, Mutual Info, Chi2, ANOVA, Lasso, Tree, RFE), dense SVD/TF-IDF embeddings, feature store client |
| **Modeling Ecosystem** | `omniforge.models` | Unified `BaseOmniForgeModel` (scikit-learn compatible + atomic save/load + metadata + carbon), mandatory `BaselineBenchmarkSuite`, Unsupervised Clustering Engine (`KMeans`, `MiniBatchKMeans`, `Agglomerative`, `GMM`, `DBSCAN`, `OPTICS`, `BIRCH` + Auto-K + PCA + ANOVA), Classical ML (sklearn/XGBoost/LightGBM/CatBoost), Voting & Stacking Ensembles, Deep Learning (MLP, CNN1D, RNN/LSTM/GRU, Tabular Transformer, GNN, Autoencoders, Multimodal) with loss curves & direct Baseline comparisons, Recommendation, Survival (KM/CoxPH), Anomaly Detection, NLP, Computer Vision, Graph Analytics |
| **Forecasting, Causal, OR & Simulation** | `omniforge.forecasting`, `omniforge.causal`, `omniforge.optimization`, `omniforge.simulation` | Statistical/ML/Deep/Hierarchical/Foundation probabilistic forecasting; IPW/AIPW/PSM/DML/IV-2SLS/DiD/CATE causal inference + DoWhy-style refutations; LP/MILP/Knapsack/Hungarian/Markowitz/Transportation/CVRP/SLSQP solvers; Monte Carlo, Discrete-Event, Agent-Based & Digital Twin simulation |
| **Evaluation & Responsible AI** | `omniforge.evaluation` | Comprehensive metrics, stratified/group/purged/combinatorial-purged CV, walk-forward backtesting, Permutation/SHAP/LIME/PDP/Counterfactual explainability, Fairness auditing & Kamiran-Calders mitigation, Conformal Prediction & Adversarial Robustness, automated Model Cards & Data Cards |
| **AutoML & Pipelines** | `omniforge.automl`, `omniforge.pipelines` | GP-UCB Bayesian/Optuna/Random/Grid HPO, baseline-first `AutoMLPipelineSearch`, evolutionary NAS; Fluent & YAML `PipelineBuilder`, cycle-checked `PipelineDAG` (Mermaid & Python export), `PipelineExecutor` with caching & retries |
| **MLOps** | `omniforge.mlops` | SQLite MLflow-compatible `ExperimentTracker`, semantic-versioned `ModelRegistry` (None/Staging/Production/Archived), `ModelServingEngine` (canary A/B routing + latency telemetry), `DriftDetector` (PSI/KS/Wasserstein/JS/Chi2/Page-Hinkley), `ModelMonitor`, `ContinuousTrainingOrchestrator`, **C5 `DriftRootCauseAnalyzer`** |
| **AI Copilot & Domains** | `omniforge.copilot`, `omniforge.domains` | Multi-turn `OmniForgeCopilotAgent` (OpenAI, Anthropic, Azure, Ollama + offline deterministic planner) wired to 24 platform tools including 10 Phase 6 C4 tools; Plugin SDK + 18 ready industry plugins |
| **API, UI, CLI & Reporting** | `omniforge.api`, `omniforge.ui`, `omniforge.cli`, `omniforge.reporting` | FastAPI REST backend with JWT/RBAC, **15-studio CRISP-DM FastAPI + React TypeScript Workbench** (s00–s14), Rich Typer CLI, one-click export to HTML, PDF, DOCX, PPTX, XLSX, CSV, Markdown, JSON, Jupyter Notebook (.ipynb), and Docker Microservice Bundle |

---

## 15-Studio CRISP-DM Workbench Map

| Studio | ID | Name | Primary Purpose | Phase 6 Enhancement |
|---|---|---|---|---|
| Studio 00 | `s00` | Data Relationships | Multi-source data loading, schema relationships & initial intake audit | — |
| Studio 01 | `s01` | Cleaning | Leakage-safe cleaning, imputation, outlier handling | — |
| Studio 02 | `s02` | Mining | Clustering, anomaly screening, data mining | — |
| Studio 03 | `s03` | Industry Domains | 18 vertical industry domain solutions, KPIs & diagnostics | — |
| Studio 04 | `s04` | EDA & Hypothesis | Automated exploratory data analysis & statistical hypothesis lab | — |
| Studio 05 | `s05` | Feature Engineering | Feature creation & multi-method consensus selection | — |
| Studio 06 | `s06` | AutoML | Multi-backend AutoML, non-parametric analytical, deep learning & Green-AI Pareto | — |
| Studio 07 | `s07` | Responsible AI | Explainability (SHAP, PDP), conformal intervals, fairness & model cards | — |
| Studio 08 | `s08` | Forecasting & Causal | Forecasting, causal ATE, optimization & simulation | — |
| Studio 09 | `s09` | Specialized AI | NLP, Vision, Graph, RecSys, Survival + **C7 Twin MPC** | ✅ C7 `/s09/twin-control` |
| Studio 10 | `s10` | Pipeline DAG | Visual DAG pipeline orchestration & code export | — |
| Studio 11 | `s11` | MLOps | Model registry, experiments, drift & **C5 root-cause** | ✅ C5 `/s11/root-cause` |
| Studio 12 | `s12` | Reports | Executive reporting & multi-format export | — |
| Studio 13 | `s13` | Copilot | AI analytics copilot + **C4 deterministic tools** | ✅ C4 `/s13/chat` |
| Studio 14 | `s14` | Supervisory | Algorithm catalog, quick-train, live prediction & supervisory control | — |

---

## Quick Installation

### Windows One-Click Installation, Launcher & Standalone `.exe` Build

OmniForge ships with complete Windows automation scripts and PyInstaller/Inno Setup packaging:

- **`install.bat`** — Detects Python 3.12–3.15 (latest stable: Python 3.15.2 / 3.14.10; offers automatic `winget` installation if absent), checks Visual C++ runtime prerequisites, creates an isolated `.venv`, installs the latest 2026 runtime & build dependencies, initializes `.env` and `packaging/assets/omniforge.ico`, verifies all 18 industry domain plugins, optionally builds standalone `OmniForge.exe` / `OmniForge-CLI.exe`, and creates a Windows Desktop shortcut.
- **`start.bat`** — Enterprise Launch Center with interactive menu and direct CLI flags (`--full`, `--ui`, `--api`, `--exe`, `--cli`, `--check`, `--test`). Launches both the FastAPI REST API (`:8000/docs`) and FastAPI + React Enterprise Workbench (`:8501`) and automatically opens your browser.
- **`build_exe.bat`** — Builds or rebuilds standalone Windows executables (`dist\\OmniForge\\OmniForge.exe` in fast `--onedir` mode or `dist\\OmniForge.exe` in single-file `--onefile` mode, plus optional `OmniForge-CLI.exe` and Inno Setup 6 Windows installer `OmniForge-4.0.0-Enterprise-Setup.exe`) with embedded Windows PE version metadata, multi-size icon, and `dist\\SHA256SUMS.txt`.

```bat
REM 1. Interactive full installation + optional .exe build
install.bat

REM 2. Unattended enterprise install + build standalone OmniForge.exe + Desktop shortcut
install.bat --profile standard --build-exe --onedir --shortcut

REM 3. Launch Enterprise Platform (interactive menu or direct mode)
start.bat
start.bat --full
start.bat --exe

REM 4. Rebuild standalone .exe anytime (onedir or onefile)
build_exe.bat --onedir --all
build_exe.bat --onefile
```

### Linux / macOS / Python Developer Installation

```bash
cd OmniForge
pip install -r requirements.txt
pip install -e .
```

Optional extras (advanced install tier):
```bash
pip install -e ".[api,ui,reporting,data,forecasting,deep,boosting,automl,responsible,build,dev]"
```

### Reproducible & Hardened Installs

For production/deployments, install the hash-pinned runtime lock instead of floating ranges (target: Python 3.12, generated with `uv pip compile requirements.in --generate-hashes --python-version 3.12 --universal -o requirements.lock`):

```bash
pip install --require-hashes -r requirements.lock
pip install -e . --no-deps
```

`requirements.in` (floating minimums) is the source of truth for intent; `requirements.lock` (90 pinned distributions, sha256 hashes) and `requirements-build.lock` (build toolchain) are the generated, committed deployment artifacts — `requirements.txt` / `requirements-build.txt` are thin compatibility shims that include the locks, so `pip install -r requirements.txt` is now hash-checked too. CI verifies the lock on every PR (`lock-verify` job). Regenerate after touching the `.in` files — do not edit locks by hand:

```bash
uv pip compile requirements.in --generate-hashes --python-version 3.12 --universal -o requirements.lock
uv pip compile requirements-build.in --generate-hashes --python-version 3.12 --universal -o requirements-build.lock
```

---

## Quickstart (Python SDK)

```python
import omniforge
print(omniforge.__version__)  # '4.0.0'

# 1. Load an industry domain plugin (18 available)
finance = omniforge.default_domain_registry.get("finance")
df = finance.engineer_domain_features(finance.generate_sample_dataset(n_samples=300))

# 2. Profile 6-dimensional data quality & clean
from omniforge.data import DataQualityProfiler, DataCleaner, CleaningConfig
quality = DataQualityProfiler().profile(df)
df_clean, clean_report = DataCleaner(CleaningConfig(numeric_imputation="median")).fit_transform(df)

# 3. Automated EDA
from omniforge.eda import AutomatedEDA
eda_report = AutomatedEDA().run(df_clean, target_column="default_flag")

# 4. Baseline-first AutoML with Green AI carbon tracking
from omniforge.automl import AutoMLPipelineSearch
X = df_clean.drop(columns=["default_flag"])
y = df_clean["default_flag"]
automl = AutoMLPipelineSearch(time_budget_seconds=30).fit(X, y)
print(f"Champion: {automl.best_model_name} | Lift over best baseline: {automl.lift_over_best_baseline:+.4f}")

# 5. C5 Causal Drift Root-Cause (Phase 6)
from omniforge.mlops import DriftRootCauseAnalyzer
ref = df_clean.sample(n=200, random_state=1)
cur = df_clean.sample(n=200, random_state=2)
cur["income"] = cur["income"] + 1.5e5  # inject upstream shift
rc_report = DriftRootCauseAnalyzer().analyze(ref, cur, feature_cols=["income", "debt", "credit_score"])
print(f"Root causes: {rc_report.root_causes}")
print(f"Symptoms: {rc_report.symptoms}")
for rx in rc_report.remediation_prescriptions:
    print(rx)

# 6. C7 Closed-Loop Digital Twin MPC (Phase 6)
from omniforge.simulation import ClosedLoopMPCController, build_default_twin
import numpy as np
twin = build_default_twin(n_x=2, n_u=1, dt=1.0)
ctrl = ClosedLoopMPCController(twin, horizon=10,
    u_min=np.array([-1.0]), u_max=np.array([1.0]),
    x_min=np.array([-10.0, -10.0]), x_max=np.array([10.0, 10.0]))
measurements = [np.zeros(2) for _ in range(8)]
report = ctrl.run_closed_loop(measurements, setpoint=np.array([1.0, 0.0]))
print(f"Steps applied: {report.n_steps_applied}, constraints satisfied: {report.all_constraints_satisfied}")
print(f"Solve time: {report.stage_costs} | Total cost: {report.total_cost}")

# 7. Export executive reports (HTML, PDF, PPTX, DOCX, XLSX, IPYNB)
from omniforge.reporting import ReportGenerator
reporter = ReportGenerator()
payload = reporter.build_report(
    title="Credit Risk Analytics Executive Briefing",
    domain="finance",
    quality_report=quality.model_dump(),
    model_summary={
        "best_model_name": automl.best_model_name,
        "primary_metric": automl.primary_metric,
        "best_score": automl.best_score,
        "lift_over_best_baseline": automl.lift_over_best_baseline,
    },
    carbon_report=automl.carbon_report.model_dump(),
)
reporter.export(payload, "credit_risk_report.html")
reporter.export(payload, "credit_risk_deck.pptx")
```

---

## CLI, REST API & Enterprise Web UI

```bash
# List all 18 industry domain plugins
omniforge domains

# Run AutoML from the terminal
omniforge automl finance.csv --target default_flag --budget 20

# Ask the AI Copilot in plain English (air-gapped deterministic planner)
omniforge copilot "profile data quality" --data finance.csv

# Launch the FastAPI REST Backend (Swagger UI at /docs)
omniforge serve --host 0.0.0.0 --port 8000

# Launch the FastAPI + React Enterprise Workbench
omniforge ui --port 8501

# Generate SBOM (CycloneDX 1.5)
omniforge sbom

# Run system health check
omniforge check
```

- **Interactive Enterprise Workbench**: `http://localhost:8501`
- **FastAPI OpenAPI / Swagger Docs**: `http://localhost:8501/docs` (Healthcheck: `/api/v1/health`)
- **REST API base**: `http://localhost:8000`

---

## Reproducible container deployment

The Docker image pins its Python base by digest and installs hash-verified
binary dependencies for Linux amd64 and arm64. The local project wheel builds
offline. Regenerate the runtime and build locks from `pyproject.toml` with
`bash scripts/build_docker_locks.sh`.

```bash
docker build -f docker/Dockerfile -t omniforge:local .
python3 scripts/smoke_docker_registry.py omniforge:local
```

The smoke test registers tenant-scoped models through the REST and Workbench
APIs in one container, removes it, and verifies registered versions, signed
artifacts, and saved fairness evidence from a replacement container on the same
volume. Production containers must reuse their signing and encryption keys
with the volume. When moving an existing registry to a new workspace directory,
back up its database and signed artifacts and preserve the absolute paths stored
in the registry, or migrate them together before replacing the old container.

When `OMNIFORGE_REQUIRE_FAIRNESS_FOR_PRODUCTION=true`, run the Studio 07
fairness audit on the active model's holdout before registering it in Studio 11.
The registry stores the holdout fingerprint and assessment outcome alongside
the model version and signed-artifact checksum. Older unbound fairness records
require a fresh evaluation and registration to satisfy the strict gate.

## Running Tests

```bash
pytest -v
```

Run only Phase 6 acceptance tests:
```bash
pytest tests/unit/test_phase6.py -v
```

---

## Documentation

- [`README.md`](README.md) — This file: project overview, installation, quickstart, studio map, CLI.
- [`docs/user_guide.md`](docs/user_guide.md) — End-to-end user and operator guide: complete step-by-step workflow for all 15 studios, algorithm reference, threshold compendium, mathematical formulas, Python SDK patterns.
- [`docs/architecture.md`](docs/architecture.md) — Layered architecture, directory map, architectural guarantees, migration notes.
- [`docs/AUDIT_ANCHOR_REPLICATION.md`](docs/AUDIT_ANCHOR_REPLICATION.md) — Audit anchor semantics, off-host replication, witness patterns, runbook + rotation.
- [`docs/RISK_REGISTER.md`](docs/RISK_REGISTER.md) — NIST AI RMF (AI 100-1) risk register: 18 risks across GOVERN/MAP/MEASURE/MANAGE with controls, residuals and owners.
- [`docs/P2_REMEDIATION_REPORT.md`](docs/P2_REMEDIATION_REPORT.md) — P2 hardening report: every gate, item, and measured before/after number (ruff, mypy, pytest, `tsc`, bundle split, correlation checks).
- [`docs/MIGRATION_V3.md`](docs/MIGRATION_V3.md) — Migration guide from v3.x to v4.0.0.

---

## License

Licensed under the Apache License 2.0. See [`LICENSE`](LICENSE) for details.