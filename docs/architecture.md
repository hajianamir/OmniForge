# OmniForge — Enterprise Architecture

## 1. Layered Monorepo Architecture

```text
+-----------------------------------------------------------------------+
| Presentation & Interfaces:                                            |
|   FastAPI + React Enterprise Workbench (14 CRISP-DM Studios) | FastAPI API    |
|   Typer + Rich CLI (`omniforge`) | Desktop Launcher (`OmniForge.exe` / `start.bat`)|
+-----------------------------------------------------------------------+
| Intelligence & Domain Layer:                                          |
|   AI Copilot (Multi-LLM + Offline Planner + 14 Tools)                 |
|   Automated Narrative Reporting (HTML/PDF/PPTX/DOCX/XLSX/IPYNB/Docker)|
|   18 Enterprise Industry Domain Plugins + Extensible Plugin SDK       |
+-----------------------------------------------------------------------+
| Orchestration & MLOps Layer:                                          |
|   Fluent & YAML DAG Pipelines | Content-Addressed Artifact Caching    |
|   Experiment Tracking | Staged Model Registry | Real-Time/Batch/ONNX  |
|   Drift Detection (PSI/KS/JS/Wasserstein) | SLA Monitor | Auto-CT     |
+-----------------------------------------------------------------------+
| Modeling, Decision Intelligence, AutoML & Responsible AI:             |
|   Mandatory Strong Baselines -> Classical ML -> Ensembles             |
|   Deep Learning (MLP, 1D-CNN, RNN/LSTM/GRU, Transformer, GNN, AE, MM) |
|   Forecasting (Statistical, ML, Deep, Hierarchical, Foundation)       |
|   Causal Inference (IPW/AIPW/PSM/DML/IV/DiD/CATE + Refutations)       |
|   Optimization (LP/MILP/Knapsack/Hungarian/Markowitz/CVRP/NLP)        |
|   Simulation (Monte Carlo, Discrete-Event, Agent-Based, Digital Twin) |
|   Specialized (Anomaly, Survival KM/CoxPH, RecSys, NLP, Vision, Graph)|
|   Responsible AI (SHAP/LIME/PDP/CF, Fairness, Conformal, Model Cards) |
+-----------------------------------------------------------------------+
| Data & Feature Foundation:                                            |
|   30+ Source Ingestion | Leakage-Safe Cleaning | Declarative Rules    |
|   6-Dimension Quality Profiler | Content-Addressed Versioning         |
|   Feast-Compatible Feature Store | Gaussian Copula Synthetic Engine   |
|   Multimodal Extractors | Automated EDA & Statistical Hypothesis Suite|
+-----------------------------------------------------------------------+
| Core Infrastructure & Governance (Python 3.12–3.15):                  |
|   Pydantic v2 Settings + YAML/Env/Dotlist Composition                 |
|   Structured Logging + Secret Redaction | Custom OmniForgeError Hierarchy   |
|   AES/Fernet Encryption | JWT + RBAC/ABAC | Hash-Chained Audit Trail  |
|   GDPR/HIPAA/SOX Compliance Validator | Green AI Carbon Tracker       |
+-----------------------------------------------------------------------+
```

## 2. Canonical Directory Structure

- `src/omniforge/core/`: Constants, custom `OmniForgeError` hierarchy, StrEnum/Pydantic types, structured logging with secret redaction, security/RBAC/encryption/compliance, Green AI carbon tracker, utilities.
- `src/omniforge/config/`: Pydantic v2 `OmniForgeSettings` and multi-layer YAML/env/dotlist `ConfigLoader`.
- `src/omniforge/data/`: Multi-source `DataIngestionEngine`, `DataCleaner`, `DataValidator`, 6D `DataQualityProfiler`, `DatasetVersionManager`, `LocalFeatureStore`, `SyntheticDataGenerator`, multimodal processors.
- `src/omniforge/eda/` & `src/omniforge/features/`: Automated EDA, statistics, Plotly visualizations, narrative insights; leakage-safe feature engineering, 8 feature selectors, dense embeddings, feature store client.
- `src/omniforge/models/`: `BaseOmniForgeModel`, mandatory `BaselineBenchmarkSuite`, `clustering/` (7 unsupervised algorithms + Auto-K/PCA/ANOVA), classical ML, ensembles, `deep/` (7 neural architectures with `loss_history_`), `time_series/` (5 paradigms), `anomaly/`, `survival/`, `recommendation/`, `nlp/`, `computer_vision/`, `graphs/`, and internal registry.
- `src/omniforge/{forecasting,causal,optimization,simulation}/`: Decision-intelligence engines.
- `src/omniforge/{evaluation,automl,pipelines,mlops,copilot,reporting}/`: Full evaluation/Responsible AI, HPO/AutoML/NAS, DAG pipelines, MLOps lifecycle (including batch & ONNX/edge serving), AI Copilot, and multi-format reporting.
- `src/omniforge/domains/` & `configs/domains/`: Plugin SDK + 18 ready vertical plugins and YAML profiles.
- `src/omniforge/{api,cli,ui,launcher.py}`: FastAPI backend, Typer CLI, 14-studio CRISP-DM FastAPI + React TypeScript workbench (`src/omniforge/api/workbench.py` + pre-compiled `src/omniforge/ui/static_app/`), and frozen-safe desktop launcher.
- `packaging/`, `install.bat`, `start.bat`, `build_exe.bat`, `docker/`: Windows `.exe` PyInstaller & Inno Setup packaging plus multi-stage Python 3.15 Docker containers.

## 3. Architectural Guarantees

1. **Mandatory Baseline Benchmarking**: Every AutoML run and forecasting tournament evaluates strong baselines first and reports explicit lift over the strongest baseline.
2. **Leakage-Safe Fit/Transform Semantics**: `DataCleaner` and `FeatureEngineeringPipeline` strictly learn statistics on training splits and transform unseen validation/production batches deterministically.
3. **Modern Python 3.12–3.15 & 2026 Ecosystem Alignment**: Native `StrEnum`, `datetime.UTC`, PEP 604/585 typing, Pydantic v2, Pandas 3 Copy-on-Write/StringDtype/datetime64[us] compatibility, NumPy 2.5+, SciPy 1.18+, scikit-learn 1.9+, FastAPI 0.163+, and a pre-compiled React 18/Vite TypeScript SPA that code-splits vendor libraries and all 14 studios into lazily-loaded chunks.
4. **Defense-in-Depth Governance**: Automatic secret redaction in logs, NIST AES-256-GCM authenticated encryption, HMAC-SHA256 artifact signing, immutable hash-chained audit logs, RBAC enforcement, and automated Model & Data Cards.
