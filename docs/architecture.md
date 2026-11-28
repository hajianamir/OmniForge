# OmniForge — Enterprise Architecture (v4.0.0)

**Canonical Version:** `4.0.0` | **API Version:** `v2` | **Schema Version:** `2.0`  
**Python:** `3.12` – `3.15` | **Edition:** Enterprise Analytics & AI Workbench

---

## 1. Layered Monorepo Architecture

```text
+-----------------------------------------------------------------------+
|  PRESENTATION & INTERFACES                                            |
|                                                                       |
|   FastAPI + React TypeScript Enterprise Workbench (15 CRISP-DM        |
|   Studios s00–s14) | FastAPI REST API (OpenAPI/Swagger at /docs)     |
|   Typer + Rich CLI (`omniforge`) | Desktop Launcher (`OmniForge.exe`)|
|   One-click executive reporting (HTML/PDF/PPTX/DOCX/XLSX/IPYNB)      |
+-----------------------------------------------------------------------+
|  INTELLIGENCE & DOMAIN LAYER                                          |
|                                                                       |
|   AI Copilot: Multi-LLM (OpenAI/Anthropic/Azure/Ollama) +             |
|   Offline deterministic local planner (C4, Phase 6)                   |
|   24 deterministic Copilot tools (14 base + 10 Phase 6 C4)           |
|   Automated Narrative Reporting                                       |
|   18 Enterprise Industry Domain Plugins + Extensible Plugin SDK       |
+-----------------------------------------------------------------------+
|  ORCHESTRATION & MLOPS LAYER                                          |
|                                                                       |
|   Fluent & YAML DAG Pipelines | Content-Addressed Artifact Caching    |
|   Experiment Tracking | Staged Model Registry (None/Staging/Prod/Arch)|
|   Real-Time / Batch / ONNX Model Serving                              |
|   Drift Detection (PSI/KS/JS/Wasserstein) | SLA Monitor | Auto-CT     |
|   C5 Causal Drift Root-Cause Analyzer (Phase 6)                       |
+-----------------------------------------------------------------------+
|  MODELING, DECISION INTELLIGENCE, AUTOML & RESPONSIBLE AI             |
|                                                                       |
|   Mandatory Strong Baselines → Classical ML → Ensembles               |
|   Deep Learning (MLP, 1D-CNN, RNN/LSTM/GRU, Transformer, GNN, AE)     |
|   Forecasting (Statistical, ML, Deep, Hierarchical, Foundation)       |
|   Causal Inference (IPW/AIPW/PSM/DML/IV/DiD/CATE + Refutations)       |
|   Optimization (LP/MILP/Knapsack/Hungarian/Markowitz/CVRP/NLP/SLSQP) |
|   Simulation (Monte Carlo, Discrete-Event, Agent-Based, Digital Twin) |
|   C7 Closed-Loop Digital Twin MPC (Phase 6)                           |
|   Specialized (Anomaly, Survival KM/CoxPH, RecSys, NLP, Vision, Graph)|
|   Responsible AI (SHAP/LIME/PDP/CF, Fairness, Conformal, Model Cards) |
+-----------------------------------------------------------------------+
|  DATA & FEATURE FOUNDATION                                            |
|                                                                       |
|   30+ Source Ingestion | Leakage-Safe Cleaning | Declarative Rules   |
|   6-Dimension Quality Profiler | Content-Addressed Versioning        |
|   Feast-Compatible Feature Store | Gaussian Copula Synthetic Engine  |
|   Multimodal Extractors | Automated EDA & Statistical Hypothesis Suite|
+-----------------------------------------------------------------------+
|  CORE INFRASTRUCTURE & GOVERNANCE (Python 3.12–3.15)                  |
|                                                                       |
|   Pydantic v2 Settings + YAML/Env/Dotlist Composition                 |
|   Structured Logging + Secret Redaction | OmniForgeError Hierarchy   |
|   AES-256-GCM / Fernet Encryption | JWT + RBAC/ABAC                   |
|   Hash-Chained Audit Trail | Optional Checkpoint Anchor (RFC 9162)   |
|   GDPR/HIPAA/SOX Compliance Validator | Green AI Carbon Tracker       |
|   C8 PII Guard with StringDtype support (Phase 6 fix)                 |
+-----------------------------------------------------------------------+
```

---

## 2. Canonical Directory Structure

```
src/omniforge/
├── __init__.py              # Package entry point, version re-export
├── __main__.py              # `python -m omniforge` entry point
├── capabilities.py          # Platform capability discovery
├── launcher.py              # Desktop application launcher
├── version.py               # __version__ = "4.0.0", get_version_info()
│
├── api/                     # FastAPI REST backend & Workbench routers
│   ├── __init__.py
│   ├── auth.py              # JWT authentication
│   ├── jobs.py              # Background job management
│   ├── main.py              # FastAPI app factory + middleware
│   ├── schemas.py           # Pydantic request/response schemas
│   │   # Phase 6 schemas: TwinControlReq, RootCauseReq,
│   │   # CopilotToolExecReq, NonParametricRunReq
│   ├── workbench.py         # Workbench session state, caching, lineage
│   └── studios/             # 15 CRISP-DM Studio routers (s00–s14)
│       ├── __init__.py
│       ├── s00_ingestion.py         # Data ingestion
│       ├── s01_cleaning.py          # Leakage-safe cleaning
│       ├── s02_mining.py            # Clustering & anomaly mining
│       ├── s03_eda.py               # Automated EDA
│       ├── s04_stats.py             # Diagnostics & baselines
│       ├── s05_features.py          # Feature engineering
│       ├── s06_automl.py            # AutoML training
│       ├── s07_advanced.py          # Responsible AI
│       ├── s08_causal.py            # Causal inference & forecasting
│       ├── s09_opt_sim.py           # Optimization, simulation, C7 Twin MPC
│       ├── s10_rai.py               # DAG pipeline builder
│       ├── s11_mlops.py             # MLOps, drift, C5 root-cause
│       ├── s12_reports.py           # Executive reporting
│       ├── s13_copilot.py           # AI Copilot chat
│       ├── s14_supervisory.py       # Live supervisory control, C5 root-cause
│       └── schemas.py               # Shared Pydantic schemas
│
├── automl/                  # AutoML pipeline search
│   ├── __init__.py
│   ├── backends.py          # 4 native SOTA engines + 5 external adapters
│   ├── nas.py               # Evolutionary neural architecture search
│   └── pipeline_search.py   # AutoMLPipelineSearch (baseline-first, HPO)
│
├── causal/                  # Causal inference
│   ├── __init__.py
│   └── inference.py         # IPW/AIPW/PSM/DML/IV/DiD/CATE + refutations
│
├── cli/                     # Typer CLI
│   ├── __init__.py
│   └── main.py              # `omniforge` command groups
│
├── config/                  # Pydantic v2 settings
│   ├── __init__.py
│   ├── loader.py            # Multi-layer YAML/env/dotlist ConfigLoader
│   └── settings.py          # OmniForgeSettings
│
├── copilot/                 # AI Copilot (C4, Phase 6)
│   ├── __init__.py
│   ├── agent.py             # OmniForgeCopilotAgent (multi-provider + local planner)
│   ├── memory.py            # CopilotMemory (datasets, models, working state)
│   ├── prompts.py           # SYSTEM_PROMPT, TOOL_CALLING_INSTRUCTION
│   └── tools.py             # CopilotToolRegistry (24 tools, 10 Phase 6 C4 tools)
│
├── core/                    # Core infrastructure & governance
│   ├── __init__.py
│   ├── carbon.py            # Green AI Carbon Tracker
│   ├── constants.py         # SENSITIVE_KEY_PATTERNS, RBAC_ROLE_PERMISSIONS
│   ├── contracts.py         # LeakageGuard, Split Lock contracts
│   ├── exceptions.py        # OmniForgeError hierarchy (error_code, severity, suggestions)
│   ├── logging.py           # Structured logging with secret redaction
│   │   # Phase 6 fix: structlog pre-declared as ModuleType | None
│   ├── performance.py       # TTLCache, performance utilities
│   ├── rate_limit.py        # Sliding-window rate limiting
│   ├── sbom.py              # CycloneDX 1.5 SBOM generator/verifier
│   ├── security.py          # PII guard, encryption, JWT, RBAC/ABAC, audit
│   │   # Phase 6 fix: detect() uses is_string_dtype(); _audit() passes tenant_id
│   ├── types.py             # Pydantic types, DriftStatus, ModelStage, AuditEvent, Principal
│   ├── utils.py             # to_json_object, set_global_seed, etc.
│   └── workspace_session.py # Workbench session management
│
├── data/                    # Data layer
│   ├── __init__.py
│   ├── cleaning.py          # DataCleaner (11-stage symmetric pipeline)
│   ├── connectors.py        # Multi-source connectors (DuckDB, SQL, MongoDB, etc.)
│   ├── feature_store.py     # Feast-compatible LocalFeatureStore (TTL, purge)
│   ├── format_audit.py      # Format auditing utilities
│   ├── ingestion.py         # DataIngestionEngine (30+ sources)
│   ├── intake_audit.py      # InitialIntakeAuditor (health grade A–F)
│   ├── multimodal.py        # Multimodal extractors
│   ├── quality.py           # DataQualityProfiler (6 dimensions), DataValidator
│   ├── splitting.py         # Leakage-safe splitting, LeakageGuard
│   ├── synthetic.py         # SyntheticDataGenerator (Gaussian Copula, DP)
│   └── validation.py        # Validation rules, expectations
│
├── domains/                 # Industry domain plugins (18)
│   ├── __init__.py
│   ├── plugin_sdk.py        # Plugin SDK + default_domain_registry
│   ├── finance.py
│   ├── energy.py
│   ├── healthcare.py
│   ├── manufacturing.py
│   ├── retail.py
│   ├── transportation.py
│   ├── agriculture.py
│   ├── climate.py
│   ├── telecom.py
│   ├── hr.py
│   ├── education.py
│   ├── real_estate.py
│   ├── sports.py
│   ├── cybersecurity.py
│   ├── government.py
│   ├── bioinformatics.py
│   ├── materials.py
│   └── media.py
│
├── eda/                     # Automated EDA & statistics
│   ├── __init__.py
│   ├── automated_eda.py     # AutomatedEDA
│   ├── diagnostics.py       # UniversalDiagnosticsEngine
│   ├── insights.py          # Rule-based executive narratives
│   ├── profiler.py          # Statistical profiling
│   ├── statistics.py        # 17 hypothesis tests, corrections
│   └── visualization.py     # Plotly visualizations
│
├── evaluation/              # Evaluation & Responsible AI
│   ├── __init__.py
│   ├── backtesting.py       # Walk-forward backtesting
│   ├── equations.py         # Model equation extraction
│   ├── fairness.py          # FairnessAuditor, Kamiran-Calders mitigation
│   ├── interpretability.py  # InterpretabilityEngine (SHAP, LIME, PDP, CF, conformal)
│   │   # Phase 6 fix: explain_diff returns all features; conformal_explain half_width
│   │   # from rounded bounds
│   └── robustness.py        # ConformalPredictor, split_conformal_quantile
│
├── features/                # Feature engineering
│   ├── __init__.py
│   ├── engineering.py       # FeatureEngineeringPipeline
│   └── selection.py         # 8 feature selectors + 4-method consensus
│
├── forecasting/             # Time-series forecasting
│   ├── __init__.py
│   └── engine.py            # ForecastingEngine (statistical, ML, deep, hierarchical)
│
├── mlops/                   # MLOps lifecycle
│   ├── __init__.py
│   ├── continuous_training.py   # ContinuousTrainingOrchestrator
│   ├── drift.py               # DriftDetector, DriftReport, FeatureDriftResult
│   ├── future_scoring.py        # FutureTargetPredictor
│   ├── monitoring.py          # ModelMonitor, AlertEvent, SLA
│   ├── registry.py            # ModelRegistry, RegisteredModelVersion
│   ├── root_cause.py          # C5 DriftRootCauseAnalyzer, CausalDiscovery (Phase 6)
│   ├── serving.py             # ModelServingEngine (canary A/B)
│   ├── supervisory_monitor.py # SupervisoryPredictionMonitor
│   └── tracking.py            # ExperimentTracker, RunRecord
│
├── models/                  # Modeling ecosystem
│   ├── __init__.py
│   ├── base.py              # BaseOmniForgeModel
│   ├── baselines.py         # BaselineBenchmarkSuite, RidgeLinearBaseline, LassoLinearBaseline
│   ├── classical.py         # ClassicalMLModel
│   ├── ensemble.py          # VotingEnsembleModel, StackingEnsembleModel
│   ├── clustering/          # 7 unsupervised algorithms + Auto-K/PCA/ANOVA
│   ├── deep/                # 7 neural architectures (MLP, CNN1D, RNN, Transformer, GNN, AE)
│   ├── nonparametric/       # 5 True Non-Parametric Analytical Models
│   ├── registry.py          # Model registry
│   ├── recommendation.py    # Recommender systems
│   ├── survival/            # Kaplan-Meier, Cox PH
│   ├── anomaly/             # Anomaly detection
│   ├── nlp/                 # NLP models
│   ├── computer_vision/     # CV models
│   └── graphs/              # Graph analytics
│
├── optimization/            # Mathematical optimization
│   ├── __init__.py
│   └── solvers.py           # LP/MILP/Knapsack/Hungarian/Markowitz/CVRP/SLSQP
│
├── pipelines/               # DAG pipelines
│   ├── __init__.py
│   ├── builder.py           # PipelineBuilder, PipelineDAG
│   └── executor.py          # PipelineExecutor (caching, retries)
│
├── reporting/               # Executive reporting
│   ├── __init__.py
│   └── ...                  # ReportGenerator, multi-format export
│
├── simulation/              # Simulation engines
│   ├── __init__.py          # Exports MPC classes + DigitalTwinEngine
│   ├── closed_loop.py       # C7 ClosedLoopMPCController (Phase 6)
│   │   # Phase 6: constraint softening, solve_time_ms, warm-start fix,
│   │   # R matrix fix (n_u), bounded relaxation (1.2x), setpoint padding fix
│   └── engines.py           # MonteCarlo, DiscreteEvent, AgentBased, DigitalTwinEngine
│
└── ui/                      # Pre-compiled React Workbench SPA
    └── static_app/          # ~63 KB gzipped entry, code-split studio chunks
```

**External:**
- `packaging/` — PyInstaller & Inno Setup packaging specs
- `docker/` — Multi-stage Python 3.15 Docker containers
- `tests/` — Unit tests (338+ tests), including `tests/unit/test_phase6.py` (21 Phase 6 tests)
- `examples/datasets/` — Sample datasets
- `notebooks/` — Jupyter notebooks
- `scripts/` — Build and utility scripts

---

## 3. Architectural Guarantees

### 3.1 Mandatory Baseline Benchmarking
Every AutoML run and forecasting tournament evaluates strong baselines first and reports explicit lift over the strongest baseline (`lift_over_best_baseline`).

### 3.2 Leakage-Safe Fit/Transform Semantics
`DataCleaner` and `FeatureEngineeringPipeline` strictly learn statistics on training splits and transform unseen validation/production batches deterministically. `Split Lock` enforces this at the workbench level.

### 3.3 Modern Python 3.12–3.15 & 2026 Ecosystem Alignment
- Native `StrEnum`, `datetime.UTC`, PEP 604/585 typing
- Pydantic v2, Pandas 3 Copy-on-Write / `StringDtype` / `datetime64[us]` compatibility
- NumPy 2.5+, SciPy 1.18+, scikit-learn 1.9+, FastAPI 0.163+
- Pre-compiled React 18 / Vite TypeScript SPA that code-splits vendor libraries and all 15 studios

### 3.4 Defense-in-Depth Governance
- Automatic secret redaction in logs (bearer tokens, API keys, SSN, credit cards)
- NIST AES-256-GCM authenticated encryption with Fernet backward compatibility
- HMAC-SHA256 artifact signing
- Immutable hash-chained audit logs with optional out-of-band checkpoint anchor
- RBAC enforcement with tenant-scoped roles and `platform_admin` for cross-tenant
- Automated Model Cards & Data Cards

### 3.5 RBAC Role Migration (admin → platform_admin, P1-12)
Previously, any holder of tenant role `admin` could bypass tenant isolation everywhere. After this change:
- Tenant `admin` is strictly tenant-scoped (administrative within own tenant only)
- Cross-tenant authorization is restricted to `platform_admin` role
- Every cross-tenant allow/deny decision is written to the tamper-evident audit trail (fail-closed)
- The API refuses to boot if the audit sink cannot initialize
- Deployments that relied on cross-tenant admin access must create/rename platform operators to `platform_admin`

### 3.6 Air-Gapped Operation (Phase 6)
All Phase 6 components run offline with zero third-party LLM dependency:
- C5 Causal Drift Root-Cause: pure NumPy/SciPy/pandas/scikit-learn
- C7 Digital Twin MPC: pure SciPy SLSQP + NumPy
- C4 Copilot deterministic planner: pure Python string matching + tool invocation

---

## 4. Phase 6 Component Architecture

### 4.1 C5 — Causal Drift Root-Cause Attribution

```
DriftDetector (PSI/Wasserstein/JS)
        │
        ▼
InterpretabilityEngine (Delta-SHAP attribution)
        │
        ▼
CausalDiscovery (PC skeleton + orientation + cycle breaking)
        │
        ▼
upstream_ancestors() / downstream_descendants() traversal
        │
        ▼
Root-cause vs symptom classification + confidence scoring
        │
        ▼
Remediation prescriptions → Studios 11 & 14
```

**Key Files:**
- `src/omniforge/mlops/root_cause.py` — `DriftRootCauseAnalyzer`, `CausalDiscovery`, `DriftedFeature`, `RootCauseReport`
- `src/omniforge/api/studios/s11_mlops.py` — `POST /s11/root-cause`
- `src/omniforge/api/studios/s14_supervisory.py` — `POST /s14/root-cause`, `GET /s14/equations`
- `src/omniforge/copilot/tools.py` — `tool_root_cause_diagnosis`

### 4.2 C7 — Closed-Loop Digital Twin MPC

```
DigitalTwinEngine (A, B, H, Q, R, Kalman filter)
        │
        ▼
ClosedLoopMPCController (SLSQP, constraint softening, warm-start)
        │
        ▼
MPCSolveResult (solve_time_ms, slack_used, constraint_violations)
        │
        ▼
TwinControlReport (trajectory charts for Studio 09)
```

**Key Files:**
- `src/omniforge/simulation/closed_loop.py` — `ClosedLoopMPCController`, `MPCSolveResult`, `TwinControlReport`, `build_default_twin`
- `src/omniforge/simulation/engines.py` — `DigitalTwinEngine` (Kalman filter)
- `src/omniforge/simulation/__init__.py` — Exports
- `src/omniforge/api/studios/s09_opt_sim.py` — `POST /s09/twin-control` (`TwinControlReq`)
- `src/omniforge/api/studios/schemas.py` — `TwinControlReq`, `RootCauseReq`, `CopilotToolExecReq`
- `src/omniforge/copilot/tools.py` — `tool_twin_control` (with `session_key` persistence)

### 4.3 C4 — Air-Gapped Deterministic Copilot

```
OmniForgeCopilotAgent (provider="local" by default)
        │
        ▼
_chat_local_planner() — English keyword → tool routing
        │
        ▼
CopilotToolRegistry.invoke(tool_name, arguments)
        │
        ▼
24 deterministic tools (14 base + 10 Phase 6 C4)
```

**Key Files:**
- `src/omniforge/copilot/agent.py` — `OmniForgeCopilotAgent`, `_chat_local_planner`
- `src/omniforge/copilot/tools.py` — `CopilotToolRegistry`, all 24 tools
- `src/omniforge/copilot/memory.py` — `CopilotMemory`
- `src/omniforge/copilot/prompts.py` — `SYSTEM_PROMPT`, `TOOL_CALLING_INSTRUCTION`
- `src/omniforge/api/studios/s13_copilot.py` — `POST /s13/chat`

---

## 5. API Surface Summary

### 5.1 Python Public API

| Module | Key Classes |
|---|---|
| `omniforge` | `__version__`, `get_version_info()` |
| `omniforge.mlops` | `DriftDetector`, `DriftRootCauseAnalyzer`, `ModelRegistry`, `ModelMonitor`, `ExperimentTracker` |
| `omniforge.simulation` | `ClosedLoopMPCController`, `DigitalTwinEngine`, `MonteCarloSimulator`, `build_default_twin` |
| `omniforge.copilot` | `OmniForgeCopilotAgent`, `CopilotToolRegistry`, `CopilotMemory` |
| `omniforge.evaluation` | `InterpretabilityEngine`, `FairnessAuditor`, `ConformalPredictor` |
| `omniforge.automl` | `AutoMLPipelineSearch`, `HyperparameterOptimizer` |
| `omniforge.data` | `DataIngestionEngine`, `DataCleaner`, `DataQualityProfiler` |
| `omniforge.core.security` | `DataFramePIIGuard`, `EncryptionManager`, `JWTManager`, `AccessController`, `AuditLogger` |
| `omniforge.core.logging` | `get_logger`, `configure_logging`, `set_trace_id`, `set_tenant_context` |
| `omniforge.domains` | `default_domain_registry` (18 plugins) |

### 5.2 REST API Endpoints (15 Studios)

| Method | Path | Studio | Description |
|---|---|---|---|
| POST | `/s00/ingest` | S00 | Data ingestion |
| POST | `/s01/clean` | S01 | Data cleaning |
| POST | `/s02/mine` | S02 | Clustering & mining |
| GET | `/s03/catalog` | S03 | List executable diagnostics modules |
| POST | `/s03/run` | S03 | Run Universal Diagnostics |
| POST | `/s04/diagnostics` | S04 | Diagnostics |
| POST | `/s05/features` | S05 | Feature engineering |
| POST | `/s06/run` | S06 | AutoML |
| POST | `/s07/advanced` | S07 | Responsible AI |
| POST | `/s08/causal` | S08 | Causal inference |
| POST | `/s09/twin-control` | S09 | **C7 Twin MPC** |
| POST | `/s10/dag` | S10 | DAG pipeline |
| POST | `/s11/root-cause` | S11 | **C5 Root-cause** |
| POST | `/s11/drift` | S11 | Drift detection |
| POST | `/s12/report` | S12 | Executive reporting |
| POST | `/s13/chat` | S13 | **C4 Copilot** |
| GET | `/s14/models` | S14 | Supervisory models |
| POST | `/s14/root-cause` | S14 | **C5 Root-cause** |

---

## 6. Version History

| Version | Date | Key Changes |
|---|---|---|
| **4.0.0** | 2026-11-22 | Phase 6: C5 root-cause, C7 Twin MPC, C4 Copilot tools; version bump 3.3.0→4.0.0; PII StringDtype fix; audit tenant_id fix; structlog mypy fix; explain_diff cap removal; conformal half_width fix |
| 3.3.0 | 2026-11-21 | Phase 5: C3 Green-AI Pareto AutoML, C10 Explainability Diff + Conformal, C8 Zero-Trust PII |
| 3.2.0 | 2026-11-21 | Phase 4: Statistical mining FDR, feature-store TTL/erasure, CycloneDX SBOM, Studio 10 DAG |
| 3.1.0 | 2026-11-21 | Exact hypothesis testing, non-central t power, conformal APS/Mondrian/ACI, omitted-variable sensitivity, HLN Diebold-Mariano, copula mapping, counterfactuals, MLOps governance |
| 3.0.1 | 2026-11-21 | Feature selector suite, EM MLE for MCAR test, AutoML HPO fixes, pipeline caching, ADWIN scaling, ONNX export |
| 3.0.0 | 2026-11-21 | Platform major release: Ridge/Lasso baselines, DataCleaner stage symmetry, DP synthetic data, validation fixes, PROV-DM provenance, CWE-22 path confinement |

---

*End of Architecture Reference. For the complete user guide, see [`docs/user_guide.md`](docs/user_guide.md).*