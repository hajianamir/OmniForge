# OmniForge — Enterprise Analytics & AI Workbench v2.0.0

**OmniForge** is a production-grade, enterprise-scale, modular Python 3.12+ (3.12–3.15) platform that unifies the complete data-to-decision lifecycle — from multi-source ingestion and 6-dimensional data quality profiling to automated EDA, leakage-safe feature engineering, baseline-first modeling, deep learning, time-series forecasting, causal inference, mathematical optimization, simulation, responsible AI, AutoML, DAG orchestration, MLOps, an AI Copilot, 18 industry domain plugins, a FastAPI backend, a **FastAPI + React/Vite TypeScript Enterprise Workbench** (pre-compiled, code-split SPA — ~63 KB gzipped for the initial entry graph, with vendor libraries and each studio group lazy-loaded on demand — featuring 60fps Canvas plots, a server-side paginated grid, GZip compression & zero Node.js at runtime), a Typer CLI, and one-click executive reporting.

---

## Core Architecture & Modules (`src/omniforge/`)

| Subsystem | Package | Key Capabilities |
| :--- | :--- | :--- |
| **Core & Security** | `omniforge.core`, `omniforge.config` | Pydantic v2 Settings + YAML composition, structured logging with secret redaction, rich custom exception hierarchy (`error_code`, `severity`, `suggestions`, `trace_id`), AES/Fernet encryption, JWT + RBAC/ABAC, hash-chained audit logs, GDPR/HIPAA/SOX compliance checks, Green AI Carbon Tracker |
| **Data Layer** | `omniforge.data` | 30+ data sources (CSV/TSV/Excel/Parquet/Feather/JSON/JSONL/ORC/Avro/HDF5/Pickle/XML/YAML/TOML/SAS/SPSS/Stata/SQL/DuckDB/MongoDB/REST/S3/GCS/Azure/In-memory), fit/transform `DataCleaner`, `DataValidator`, 6D `DataQualityProfiler`, content-addressed `DatasetVersionManager`, Feast-compatible `LocalFeatureStore`, Gaussian Copula `SyntheticDataGenerator`, Multimodal extractors |
| **EDA & Insights** | `omniforge.eda` | Statistical profiling, parametric/non-parametric hypothesis tests, interactive Plotly dark/light visualizations, rule-based + LLM executive narratives, standalone HTML EDA reports |
| **Feature Engineering** | `omniforge.features` | Leakage-safe imputation/encoding/scaling, interaction/polynomial/datetime/time-series lags & rolling windows, 8 feature selectors (Variance, Correlation, Mutual Info, Chi2, ANOVA, Lasso, Tree, RFE), dense SVD/TF-IDF embeddings, feature store client |
| **Modeling Ecosystem** | `omniforge.models` | Unified `BaseOmniForgeModel` (scikit-learn compatible + atomic save/load + metadata + carbon), mandatory `BaselineBenchmarkSuite`, **Unsupervised Clustering Engine** (`KMeans`, `MiniBatch KMeans`, `Agglomerative`, `GMM`, `DBSCAN`, `OPTICS`, `BIRCH` + Auto-K + PCA + ANOVA), Classical ML (sklearn/XGBoost/LightGBM/CatBoost), Voting & Stacking Ensembles, Deep Learning (MLP, CNN1D, RNN/LSTM/GRU, Tabular Transformer, GNN, Autoencoders, Multimodal) with loss curves & direct Baseline comparisons, Recommendation, Survival (KM/CoxPH), Anomaly Detection, NLP, Computer Vision, Graph Analytics |
| **Forecasting, Causal, OR & Simulation** | `omniforge.forecasting`, `omniforge.causal`, `omniforge.optimization`, `omniforge.simulation` | Statistical/ML/Deep/Hierarchical/Foundation probabilistic forecasting; IPW/AIPW/PSM/DML/IV-2SLS/DiD/CATE causal inference + DoWhy-style refutations; LP/MILP/Knapsack/Hungarian/Markowitz/Transportation/CVRP/SLSQP solvers; Monte Carlo, Discrete-Event, Agent-Based & Digital Twin simulation |
| **Evaluation & Responsible AI** | `omniforge.evaluation` | Comprehensive metrics, stratified/group/purged/combinatorial-purged CV, walk-forward backtesting, Permutation/SHAP/LIME/PDP/Counterfactual explainability, Fairness auditing & Kamiran-Calders mitigation, Conformal Prediction & Adversarial Robustness, automated Model Cards & Data Cards |
| **AutoML & Pipelines** | `omniforge.automl`, `omniforge.pipelines` | GP-UCB Bayesian/Optuna/Random/Grid HPO, baseline-first `AutoMLPipelineSearch`, evolutionary NAS; Fluent & YAML `PipelineBuilder`, cycle-checked `PipelineDAG` (Mermaid & Python export), `PipelineExecutor` with caching & retries |
| **MLOps** | `omniforge.mlops` | SQLite MLflow-compatible `ExperimentTracker`, semantic-versioned `ModelRegistry` (None/Staging/Production/Archived), `ModelServingEngine` (canary A/B routing + latency telemetry), `DriftDetector` (PSI/KS/Wasserstein/JS/Chi2/Page-Hinkley), `ModelMonitor`, `ContinuousTrainingOrchestrator` |
| **AI Copilot & Domains** | `omniforge.copilot`, `omniforge.domains` | Multi-turn `OmniForgeCopilotAgent` (OpenAI, Anthropic, Azure, Ollama + offline deterministic planner) wired to 14 platform tools; Plugin SDK + **18 ready industry plugins** (`finance`, `energy`, `healthcare`, `manufacturing`, `retail`, `transportation`, `agriculture`, `climate`, `telecom`, `hr`, `education`, `real_estate`, `sports`, `cybersecurity`, `government`, `bioinformatics`, `materials`, `media`) |
| **API, UI, CLI & Reporting** | `omniforge.api`, `omniforge.ui`, `omniforge.cli`, `omniforge.reporting` | FastAPI REST backend with JWT/RBAC, **14-studio CRISP-DM FastAPI + React TypeScript Workbench** (native Dark/Light UI + server-side paginated Data Grid + 60fps Canvas scatter + interactive SVG DAG/network graphs; Data Ingestion/Cleaning → Clustering & Anomaly Screening → Universal Diagnostics/Baselines/Risk → EDA → Feature Engineering → AutoML & Deep Learning vs. Baselines → Responsible AI → Decision Science → Specialized AI → DAG Pipeline → MLOps → Executive Reporting → AI Copilot, with live `RAW/PROCESSED` status tracking, `.uap` project save/reload & Undo-safe `Apply` controls), Rich Typer CLI, one-click export to **HTML, PDF, DOCX, PPTX, XLSX, CSV, Markdown, JSON, Jupyter Notebook (.ipynb), and Docker Microservice Bundle** |

---

## Quick Installation

### Windows One-Click Installation, Launcher & Standalone `.exe` Build

OmniForge ships with complete Windows automation scripts and PyInstaller/Inno Setup packaging:

- **`install.bat`** — Detects Python 3.12–3.15 (latest stable: Python 3.15.2 / 3.14.10; offers automatic `winget` installation if absent), checks Visual C++ runtime prerequisites, creates an isolated `.venv`, installs the latest 2026 runtime & build dependencies (`requirements.txt`, `requirements-build.txt`), initializes `.env` and `packaging/assets/omniforge.ico`, verifies all 18 industry domain plugins, optionally builds standalone **`OmniForge.exe`** / **`OmniForge-CLI.exe`**, and creates a Windows Desktop shortcut.
- **`start.bat`** — Enterprise Launch Center with interactive menu and direct CLI flags (`--full`, `--ui`, `--api`, `--exe`, `--cli`, `--build-exe`, `--check`, `--test`). Launches both the FastAPI REST API (`:8000/docs`) and FastAPI + React Enterprise Workbench (`:8501`) and automatically opens your browser.
- **`build_exe.bat`** — Builds or rebuilds standalone Windows executables (`dist\OmniForge\OmniForge.exe` in fast `--onedir` mode or `dist\OmniForge.exe` in single-file `--onefile` mode, plus optional `OmniForge-CLI.exe` and Inno Setup 6 Windows installer `OmniForge-1.0.0-Enterprise-Setup.exe`) with embedded Windows PE version metadata, multi-size icon, and `dist\SHA256SUMS.txt`.

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

Optional extras:
```bash
pip install -e ".[api,ui,reporting,data,forecasting,deep,boosting,automl,responsible,build,dev]"
```

---

## Quickstart (Python SDK)

```python
import omniforge

# 1. Load an industry domain plugin (18 available)
finance = omniforge.default_domain_registry.get("finance")
df = finance.engineer_domain_features(finance.generate_sample_dataset(n_samples=300))

# 2. Profile 6-dimensional data quality & clean
quality = omniforge.DataQualityProfiler().profile(df)
df_clean, clean_report = omniforge.DataCleaner().fit_transform(df)

# 3. Automated EDA
eda_report = omniforge.AutomatedEDA().run(df_clean, target_column="default_flag")

# 4. Baseline-first AutoML with Green AI carbon tracking
X = df_clean.drop(columns=["default_flag"])
y = df_clean["default_flag"]
automl = omniforge.AutoMLPipelineSearch(time_budget_seconds=30).fit(X, y)
print(f"Champion: {automl.best_model_name} | Lift over best baseline: {automl.lift_over_best_baseline:+.4f}")

# 5. Export executive reports (HTML, PDF, PPTX, DOCX, XLSX, IPYNB)
reporter = omniforge.ReportGenerator()
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
omniforge domains --sample finance --out finance.csv
omniforge automl finance.csv --target default_flag --budget 20

# Ask the AI Copilot in plain English
omniforge copilot "profile data quality" --data finance.csv

# Launch the FastAPI REST Backend (Swagger UI at /docs)
omniforge serve --host 0.0.0.0 --port 8000

# Launch the FastAPI + React Enterprise Workbench
omniforge ui --port 8501
```

---

## Running Tests

```bash
pytest -v
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — layered architecture, directory map, guarantees.
- [`docs/user_guide.md`](docs/user_guide.md) — end-to-end user and operator guide.
- [`docs/P2_REMEDIATION_REPORT.md`](docs/P2_REMEDIATION_REPORT.md) — P2 hardening report: every gate, item
  and measured before/after number (ruff, mypy, pytest, `tsc`, bundle split, correlation checks).

---

## License

Licensed under the Apache License 2.0. See [`LICENSE`](LICENSE) for details.
