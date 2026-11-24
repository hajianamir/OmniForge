# OmniForge architecture reference

> Source map reverified at the code revision in [SKILL.md](../SKILL.md).
> Code paths are relative to the repository root; file links are relative to
> this reference. A bare filename following a full source path refers to that
> same directory. Names locate implementations, not guarantees that all
> optional backends are installed or all workflow paths share their behavior.

## Interfaces and navigation

| Surface | Source | Purpose |
|---|---|---|
| SDK | `src/omniforge/__init__.py` | Lazy top-level exports; subsystem code lives under `src/omniforge/`. |
| CLI and desktop | `src/omniforge/cli/main.py`, `src/omniforge/launcher.py` | Typer commands and frozen/source launcher; see [recipes](recipes.md). |
| HTTP and SPA | `src/omniforge/api/main.py`, `src/omniforge/api/workbench.py`, `frontend/` | One FastAPI app serves REST and the compiled React workbench; see [API and state](api-and-state.md). |
| Installer | `src/omniforge/capabilities.py`, `installer/provision.py` | stdlib-only pack catalogue and private-environment provisioning; see [packaging](packaging.md) for lock-selection limitations. |

React navigation (`frontend/src/types.ts::STUDIOS`) has 15 studios:

| ID | Workbench area | Main backend location |
|---|---|---|
| `00` | Data ingestion and Split Lock | `src/omniforge/api/studios/s00_ingestion.py`; shared `/s01/*` ingestion and `/split-lock` handlers in `src/omniforge/api/workbench.py` |
| `01` | Data preparation and cleaning | `src/omniforge/api/workbench.py`, `src/omniforge/api/studios/s01_cleaning.py` |
| `02` | Clustering, anomaly and mining | `src/omniforge/api/workbench.py`, `src/omniforge/api/studios/s02_mining.py` |
| `03` | Diagnostics, baselines and risk | `src/omniforge/api/studios/s03_eda.py`, `src/omniforge/eda/diagnostics.py` |
| `04` | EDA and hypothesis tests | `src/omniforge/api/studios/s04_stats.py`, `src/omniforge/eda/` |
| `05` | Feature engineering and selection | `src/omniforge/api/studios/s05_features.py`, `src/omniforge/features/` |
| `06` | AutoML, deep learning and scoring | `src/omniforge/api/studios/s06_automl.py`, `src/omniforge/automl/` |
| `07` | Responsible AI | `src/omniforge/api/studios/s07_advanced.py`, `src/omniforge/evaluation/` |
| `08` | Forecasting, causal, optimization and simulation | `src/omniforge/api/studios/s08_causal.py` and decision engines |
| `09` | Specialized AI | `src/omniforge/api/studios/s09_opt_sim.py`, `src/omniforge/models/` |
| `10` | Visual DAG pipelines | `src/omniforge/api/studios/s10_rai.py`, `src/omniforge/pipelines/` |
| `11` | MLOps | `src/omniforge/api/studios/s11_mlops.py`, `src/omniforge/mlops/` |
| `12` | Reports | `src/omniforge/api/studios/s12_reports.py`, `src/omniforge/reporting/` |
| `13` | Copilot | `src/omniforge/api/studios/s13_copilot.py`, `src/omniforge/copilot/` |
| `14` | Supervisory monitoring | `src/omniforge/api/studios/s14_supervisory.py`, `src/omniforge/mlops/supervisory_monitor.py` |

`src/omniforge/ui/app.py::STUDIOS` is older exported metadata for `01`–`14`,
not the React navigation source. Route numbers, legacy labels, and current UI
numbers are not always identical; verify the actual caller before renaming.

## Core and configuration

- `src/omniforge/core/types.py` defines shared enums and Pydantic models including
  `TaskType`, `ModelStage`, and `Principal`. `src/omniforge/core/exceptions.py` defines
  `OmniForgeError` and domain-specific subclasses, but ordinary Python and
  FastAPI exceptions are also raised throughout the repository.
- `src/omniforge/core/contracts.py` contains `AnalysisSpec` and `LeakageGuard`;
  `src/omniforge/data/splitting.py` provides `LeakageSafeSplitter` and pipeline helpers.
  A split utility or a UI lock does not itself prove every consumer uses it.
- `src/omniforge/core/logging.py` provides `get_logger` and redacting formatters;
  `src/omniforge/core/security.py` contains authentication, signing, encryption,
  restricted deserialization, audit and privacy helpers. Logging/redaction is
  not a universal confidentiality guarantee. See [security](security.md).
- `src/omniforge/core/workspace_session.py::WorkspaceSessionManager` filters state, creates
  signed project archives, and validates/restores them. `src/omniforge/core/performance.py`
  holds caches and dataframe/plot helpers; `src/omniforge/core/sbom.py` provides SBOM and
  lock-hash checks; `src/omniforge/core/rate_limit.py` provides memory and Redis limiters.
- `src/omniforge/config/settings.py::OmniForgeSettings` uses `OMNIFORGE_` env prefix and
  `__` nested delimiter; `get_settings()` caches an instance. The settings
  sections include paths, logging, compute, carbon, data, modeling, AutoML,
  MLOps, jobs, security, copilot, API and UI. Defaults include development
  environment, `debug=False`, local hosts, 300 requests/minute, and a
  256 MiB request-body limit. Check `settings.py` and `configs/default.yaml`
  rather than inferring enforcement from a setting's name.
- `src/omniforge/config/loader.py::ConfigLoader.compose` merges base YAML, extra files and
  dotlist overrides, and loads optional domain YAML into `domain_overrides`;
  `load_settings` is an alias of `load_config`. The launcher `main()` reads
  a `.env` file without overwriting existing environment variables; do not
  assume every entry point reads it the same way. Some security/connectors
  settings are resolved directly from the environment. See `.env.example`
  as a template, but trust implementation for key paths and validation.

## Data, features and evaluation

| Subsystem | Principal files and objects | Review notes |
|---|---|---|
| Ingestion | `src/omniforge/data/ingestion.py::DataIngestionEngine`, `src/omniforge/data/connectors.py` | Local files, HTTP and database paths have different protections and optional dependencies. |
| Quality and cleaning | `src/omniforge/data/quality.py::DataQualityProfiler`, `src/omniforge/data/cleaning.py::DataCleaner`, `src/omniforge/data/validation.py::DataValidator` | Cleaner has fit/transform and split-aware APIs; check the actual route and selected mode. |
| Audits and splits | `src/omniforge/data/format_audit.py`, `src/omniforge/data/intake_audit.py`, `src/omniforge/data/splitting.py` | Audits and Split Lock report useful evidence; they do not establish that downstream evaluation is unbiased. |
| Features | `src/omniforge/features/engineering.py::FeatureEngineeringPipeline`, `src/omniforge/features/selection.py::FeatureSelector`, `src/omniforge/features/inference.py::WorkbenchInputPreprocessor` | The fitted scoring schema is distinct from feature engineering; see [API and state](api-and-state.md). |
| Feature store | `src/omniforge/data/feature_store.py::LocalFeatureStore`, `src/omniforge/features/store_client.py` | Local Parquet offline and SQLite online data; freshness is a point-in-time report on currently materialized vectors. |
| Other data | `src/omniforge/data/versioning.py`, `src/omniforge/data/synthetic.py`, `src/omniforge/data/multimodal.py` | Dataset versions, sample generators and multimodal helpers. |
| EDA | `src/omniforge/eda/profiler.py`, `src/omniforge/eda/statistics.py`, `src/omniforge/eda/diagnostics.py`, `src/omniforge/eda/automated_eda.py`, `src/omniforge/eda/visualization.py` | Profiling, tests, diagnostics and visualizations. |
| Evaluation | `src/omniforge/evaluation/metrics.py`, `validation.py`, `fairness.py`, `interpretability.py`, `robustness.py`, `backtesting.py`, `model_cards.py`, `equations.py` | Metrics, validation, explanations, governance and equation extraction. Inspect task-specific limits. |

## Modeling and decision engines

`src/omniforge/models/base.py::BaseOmniForgeModel` wraps fitted-model operations and signed
serialization. Implement `_fit` and `_predict`; `_predict_proba` has a
non-abstract default. `src/omniforge/models/registry.py::ModelCatalog` registers baselines,
classical algorithms, ensembles, and lazy deep constructors. Other engines,
including specialized analytics, need not subclass the base class. A
`NotImplementedError` can be an intentional task guard, for example in
regression-only models; review the call site instead of treating every
occurrence as unfinished code.

Useful entry points:

- `src/omniforge/models/baselines.py`, `classical.py`, `ensemble.py` and `src/omniforge/models/deep/`
  contain model families; `src/omniforge/models/nonparametric.py` contains the nonparametric
  benchmark suite. Some optional backends use local fallbacks, with behavior
  specific to each implementation.
- `src/omniforge/models/time_series/`, `src/omniforge/models/anomaly/`, `src/omniforge/models/clustering/`,
  `src/omniforge/models/nlp/`, `src/omniforge/models/computer_vision/`, `src/omniforge/models/graphs/`,
  `src/omniforge/models/mining/`, `src/omniforge/models/survival/`, and
  `src/omniforge/models/recommendation/` contain specialized engines.
- `src/omniforge/automl/pipeline_search.py::AutoMLPipelineSearch` and
  `src/omniforge/automl/backends.py::AutoMLBackendHub` orchestrate searches/backends;
  `src/omniforge/forecasting/engine.py`, `src/omniforge/causal/inference.py`,
  `src/omniforge/optimization/solvers.py`, and `src/omniforge/simulation/`
  implement decision workflows. Verify fit and evaluation
  boundaries for each route; the existence of CV does not rule out preprocessing
  leakage before the folds.

## Pipelines, MLOps, Copilot and reporting

- `src/omniforge/pipelines/base.py`, `dag.py`, `builder.py`, `executor.py` define steps,
  DAG validation, building and execution.
- `src/omniforge/mlops/registry.py::ModelRegistry` stores versions in SQLite and applies a
  Production fairness gate; `tracking.py`, `serving.py`, `drift.py`,
  `monitoring.py`, `continuous_training.py`, and `supervisory_monitor.py`
  cover tracking, serving and monitoring. The separate in-memory
  `src/omniforge/mlops/job_queue.py` is not the workbench's SQLite-backed job manager.
- `src/omniforge/copilot/agent.py::OmniForgeCopilotAgent` supports configured external
  providers and a deterministic local planner. `src/omniforge/copilot/tools.py` registers
  callable tools; `memory.py` owns bounded conversation and working state.
  An advertised tool description is not necessarily an implemented route.
- `src/omniforge/reporting/sections.py`, `generator.py`, `charts.py`, `exporters.py`,
  `narrative.py`, and `formulas.py` define selectable sections and report
  exports. Spreadsheet sanitization in `src/omniforge/reporting/exporters.py` does not
  automatically cover unrelated dataset export endpoints.

## Domain plugins

`src/omniforge/domains/__init__.py` registers module-level `plugin` **instances** for 18
built-ins: `finance`, `energy`, `healthcare`, `manufacturing`, `retail`,
`transportation`, `agriculture`, `climate`, `telecom`, `hr`, `education`,
`real_estate`, `sports`, `cybersecurity`, `government`, `bioinformatics`,
`materials`, and `media`. `src/omniforge/core/constants.py::BUILTIN_DOMAINS` and
`configs/domains/` provide corresponding catalogue/configuration data.

`src/omniforge/domains/plugin_sdk.py::BaseDomainPlugin` requires feature engineering,
KPI calculation and sample generation; `DomainPluginRegistry` registers
instances. Its `discover_entry_points()` registers only objects that are
already `BaseDomainPlugin` instances. The built-in entries in `pyproject.toml`
export **classes**, so that discovery method does not instantiate them; built-ins
work because they are explicitly registered in `src/omniforge/domains/__init__.py`. No
in-repository caller of `discover_entry_points()` was found at this revision.
For an external plugin, export an instance and explicitly invoke or wire
and test discovery; do not rely on existing class entries as proof of it.
