# OmniForge Architecture Reference

> Load this file when you need subsystem detail. The slim `SKILL.md` is the
> entry point; this file is the index into the code. All paths are relative to
> the repo root. Line counts below were verified against commit `b7dc7ae`
> (2026-11-24); see `../SKILL.md` for provenance — re-verify with `wc -l` before quoting them.

## Taxonomies (canonical lists — update here, link from `SKILL.md` §6)

**Studios (15, frontend `00`–`14`; backend `s00`–`s14`):** `00` Ingestion &
Split Lock, `01` Preparation & Cleaning, `02` Cluster·Mining, `03`
Diagnostics & Risk, `04` EDA & Hypothesis, `05` Feature Studio, `06` AutoML
& Deep AI, `07` Responsible AI, `08` Decision & Forecast, `09`
NLP·Vision·Graph, `10` DAG Pipelines, `11` MLOps Registry, `12` Executive
Reports, `13` AI Copilot, `14` Supervisory Monitor. Sources:
`frontend/src/types.ts::STUDIOS`, `api/workbench.py` + `api/studios/`,
`ui/app.py::STUDIOS` (lists `01`–`14`; legacy `s01` combines ingestion and cleaning,
see `packaging.md` coverage caveat).
`ui/app.py::STUDIOS` is retained, publicly re-exported metadata with no
in-repo UI consumer; React’s `frontend/src/types.ts::STUDIOS` drives the
current navigation. When adding a studio, update React and audit the Python
export for compatibility rather than assuming the two lists are synchronized.

**Domains (18):** `finance, energy, healthcare, manufacturing, retail,
transportation, agriculture, climate, telecom, hr, education, real_estate,
sports, cybersecurity, government, bioinformatics, materials, media`.
Sources: `core/constants.py::BUILTIN_DOMAINS`, `domains/__init__.py`,
`pyproject.toml` entry points, `configs/domains/*.yaml`.

**Packs (12) & tiers:** see `packaging.md` (single source:
`capabilities.py`). Required: `core-runtime, api-server, statistics,
data-io, reporting`. Optional: `boosting, deep, vision, nlp, foundation,
automl, responsible`. Tiers: `simple` → `advanced` → `everything`
(cumulative; `TIERS` in `capabilities.py`).

## Core — `src/omniforge/core/`

- **`types.py`** — `SerializableStrEnum` (pickle-by-value base; see SKILL.md
  invariant 5), `TaskType` (15 values), `DataModality` (11), `ModelStage`
  (`None/Staging/Production/Archived`), `ValidationSeverity`, `DriftStatus`;
  Pydantic models `CarbonReport`, `ColumnSchema`, `DatasetMetadata`,
  `ModelMetadata`, `AuditEvent` (hash-chained + `chain_mac` HMAC v2),
  `Principal`.
- **`constants.py`** — `DEFAULT_RANDOM_SEED=42`, `DEFAULT_TEST_SIZE=0.2`,
  `DEFAULT_CV_FOLDS=5`; `SUPPORTED_FILE_FORMATS` (30+ extensions),
  `SUPPORTED_DB_SCHEMES`, `SUPPORTED_CLOUD_SCHEMES`,
  `CARBON_INTENSITY_BY_COUNTRY` (18 regions), `SENSITIVE_KEY_PATTERNS` (14),
  `RBAC_ROLE_PERMISSIONS` (6 roles), `BUILTIN_DOMAINS` (18 slugs).
- **`exceptions.py`** — `OmniForgeError` base (`error_code`, `severity`,
  `suggestions`, `trace_id`, `context`, `to_dict()`) → `ConfigurationError`,
  `SecurityError` → `AuthorizationError`/`ComplianceError`, `DataError`
  family, `ModelError` family, plus evaluation/AutoML/forecasting/causal/
  optimization/simulation/pipeline/MLOps/copilot/domain/reporting and
  `DependencyMissingError`. Each subclass sets `default_error_code`
  (`OmniForge-XXX-NNN`) and `default_severity`.
- **`contracts.py`** — `AnalysisSpec` (shared `s00_plan`: `task_kind` 8
  literals, `analysis_mode`, `independent_unit`, target/time/group/id
  columns, `forecast_horizon`, treatment/outcome/estimand
  `ATE/ATT/ATC/CATE`, `primary_metric`, `decision_threshold`,
  `sensitive_attributes`, `random_seed`); `LeakageGuard.validate_split`
  (row/group/temporal overlap + target-leakage exact-copy and
  near-perfect-predictor checks) → `LeakageGuardReport`; `ResultBundle`,
  `bootstrap_ci`.
- **`logging.py`** — `configure_logging()`, `get_logger(name)` →
  `OmniForgeLoggerAdapter`, `RedactingJsonFormatter`,
  `redact_sensitive_value`, trace-id and tenant context vars. Library code
  logs via `get_logger(__name__)`; never `print`.
- **`carbon.py`** — `CarbonTracker` context manager (hardware-estimated
  kWh/CO₂ from power, PUE, grid intensity) + `track_carbon` decorator.
  Measured mode via `codecarbon` when the `responsible` pack is installed.
- **`rate_limit.py`** — `RateLimiter` protocol; `InMemorySlidingWindowLimiter`,
  `RedisSlidingWindowLimiter`, `ResilientRateLimiter`;
  `build_rate_limiter(settings, …)` (`auto`/`memory`/`redis`). `0` disables
  (test suite sets it in `tests/conftest.py`).
- **`sbom.py`** — `parse_uv_lockfile`, `generate_cyclonedx_sbom`,
  `write_cyclonedx_sbom`, `verify_lockfiles_provenance`. Backs
  `omniforge sbom` and `scripts/generate_sbom.py`.
- **`performance.py`** — `parallel_n_jobs`, `compute_df_fingerprint`,
  `DataFrameCache`, `TTLCache`, `smart_sample_df`, `lttb_downsample`,
  `optimize_plotly_dict`, `paginate_dataframe`,
  `downsample_table_for_plot`, `compact_scatter_payload`.
- **`utils.py`** — `set_global_seed`, `is_package_available`,
  `compute_bytes/file/dataframe_checksum` (SHA-256),
  `optimize_dataframe_dtypes`, `safe_json_default`, `to_json_object/list/safe`,
  `ensure_directory`, `timer`, `detect_hardware_profile`,
  `infer_semantic_column_type`.
- **`workspace_session.py`** — `WorkspaceSessionManager`
  (`export_to_bytes`, `save_to_file`, `save_local_project`,
  `load_from_bytes/file`, `inspect_bytes`, `list/delete_local_projects`);
  `WorkspaceManifest` (SHA-256 integrity), `WorkspaceSnapshotSummary`.
  ZIP-based with bounded member reads and archive validation. See
  `api-and-state.md` for the `uap_*` key lifecycle.

Security (`security.py`, 1852 lines) is documented separately in
`security.md` — read it before touching auth, serialization, or audit code.

## Config — `src/omniforge/config/` + `configs/`

`settings.py` defines `OmniForgeSettings` (`BaseSettings`,
`env_prefix="OMNIFORGE_"`, `env_nested_delimiter="__"`, `extra="allow"`,
`get_settings()` cached). Sections and defaults:

| Section | Key defaults |
|---|---|
| top-level | `environment=development`, `debug=False`, `random_seed=42`, `default_tenant_id=default-tenant` |
| `PathsSettings` | workspace/artifacts/cache/feature_store dirs, `mlops_tracking.db`, `model_registry.db`, `audit.jsonl` |
| `LoggingSettings` | `level=INFO`, `format=json`, `redact_secrets=True`, `include_trace_id=True` |
| `ComputeSettings` | `n_jobs=-1`, `prefer_gpu=True`, `backend=local`, `chunk_size=100000`, `memory_limit_mb=8192` |
| `CarbonSettings` | `enabled=True`, `country_iso=DEU`, `intensity=380.0`, `pue=1.25`, TDPs |
| `DataSettings` | `optimize_dtypes=True`, `compute_checksums=True`, `missing_threshold_drop_col=0.95`, outlier `iqr/1.5`, 6 quality thresholds |
| `ModelingSettings` | `always_compare_baselines=True`, `cv_folds=5`, `test_size=0.2`, conformal/fairness/interpretability on, metrics `f1_weighted/rmse/smape` |
| `AutoMLSettings` | `time_budget=120s`, `max_trials=30`, `bayesian`, ensembling on, `early_stopping=10` |
| `MLOpsSettings` | `auto_track=True`, PSI warn `0.1` / critical `0.25`, KS α `0.05`, `retrain_on_critical_drift=True` |
| `SecuritySettings` | `enable_rbac=True`, JWT HS256/60min, `auto_bootstrap_local_ui=False`, `encrypt_artifacts=False`, PII on, profiles `[GDPR, SOX]` |
| `CopilotSettings` | `provider=local`, `model=gpt-4o-mini`, `temperature=0.1`, `max_tokens=2048`, `max_tool_steps=8`, `memory_window=20`, Ollama/Azure fields |
| `APISettings` | host `127.0.0.1:8000`, CORS localhost 8501/8000/5173, `rate_limit_per_minute=300`, `rate_limit_backend=auto`, `max_request_body_bytes=256MB`, CSP report-only on, HSTS on |
| `UISettings` | host `127.0.0.1:8501`, `default_theme=dark`, `locale=en` |

Note: `debug=False` even in `development` is the shipped default
(`settings.py:219`, `configs/default.yaml:7`) — verbose logging must be
enabled explicitly. `APISettings._prevent_wildcard_credentials` forces
credentials off when `*` is in CORS origins (CWE-942).

`loader.py` — `ConfigLoader` (`load_yaml`, `compose`: base → domain →
overlay → dotlist overrides), `deep_merge`, `parse_dotlist_overrides`
(`section.key=value`), `_resolve_env_interpolation` (`${ENV:default}`).
Public entry points `load_config` / `load_settings` (re-exported from
`omniforge.config` and the top-level SDK).

YAML: `configs/default.yaml` (mirrors the table), `configs/logging.yaml`
(stdlib `dictConfig`), `configs/domains/*.yaml` (18 files: `name`,
`compliance_frameworks`, `primary_tasks`, `key_metrics`,
`protected_attributes`, `default_cv_strategy`, `drift_sensitivity`,
`fairness_thresholds` — e.g. finance: SOX/Basel-III/SR-11-7/GDPR/MiFID-II/
Fair-Lending, `purged_kfold`, disparate-impact ≥ 0.80).

Env vars: prefix `OMNIFORGE_`, nesting via `__`
(`OMNIFORGE_API__RATE_LIMIT_PER_MINUTE`,
`OMNIFORGE_SECURITY__JWT_SECRET_KEY`, `OMNIFORGE_PATHS__WORKSPACE_DIR`).
Provider keys (no prefix): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`AZURE_OPENAI_API_KEY/ENDPOINT`, `OLLAMA_BASE_URL`. Connector controls:
`OMNIFORGE_ALLOWED_DB_DIRS/HOSTS`, `OMNIFORGE_ALLOW_PRIVATE_DB_HOSTS`,
`OMNIFORGE_DATABASE_URL`, `OMNIFORGE_MLFLOW_TRACKING_URI`,
`OMNIFORGE_FEAST_REPO_PATH`, `OMNIFORGE_KAFKA_BOOTSTRAP_SERVERS`. Audit
anchor: `OMNIFORGE_AUDIT__ANCHOR_PATH/CHECKPOINT_KEY/ANCHOR_EVERY`.
Production requires `OMNIFORGE_ENVIRONMENT=production` + all four
`OMNIFORGE_SECURITY__*` secrets (≥32 bytes); dev auto-generates 0600
keyfiles under `.omniforge_workspace/.secrets/`. Full template:
`.env.example` (55 keys).

## Data — `src/omniforge/data/`

| File | Key class(es) | Contract |
|---|---|---|
| `ingestion.py` | `DataIngestionEngine`, `IngestionResult` | `ingest(source)` / `load_file` / `ingest_chunks` / `list_excel_sheets` / `ingest_excel_sheets`; 30+ formats, 11 DB schemes, S3/GS/Azure/HTTP, REST APIs, Kafka. `fast_read_tabular` (Polars/DuckDB fast path, pandas fallback). Identifiers via `_quote_sql_identifier`; canonical table names for SQLite/DuckDB. |
| `connectors.py` | module functions | Read-only SQL: `validate_readonly_sql` (strip comments, reject DDL/DML/multi-statement), `make_sqlite_authorizer`, `_resolve_allowed_local_db_path` (confined roots), `_validate_remote_db_host` + `_assert_ip_not_ssrf`, `validate_outbound_http_url`, `sql_inspect_database`, `sql_load_dataframe`, `access_inspect/load` (Access), `ensure_demo_sqlite_database`, Excel sheet readers. |
| `cleaning.py` | `DataCleaner`, `CleaningConfig`, `CleaningReport` | `fit_transform` / `fit_transform_split` / `fit`+`transform`; missing (incl. MICE/Rubin, MissForest), strings, type inference, fuzzy dedup (LSH), outliers (IQR/z/Mahalanobis), `apply_domain_bounds`, `benford_digit_test`, `cleaning_distortion_audit`. Fit on train only. |
| `quality.py` | `DataQualityProfiler`, `DataQualityReport` | `profile(df)` → 6 dimensions, score + grade, recommendations; `little_mcar_test`. |
| `validation.py` | `DataValidator`, `ValidationRule`, `DataContract` | GE/Pandera-style declarative rules; `auto_generate_suite`, `register_custom_validator`, `validate`. Domain plugins supply rules. |
| `splitting.py` | `LeakageSafeSplitter`, `SplitConfig`, `LeakageSafePipelineBuilder`, `CombinatorialPurgedKFold` | `split` (stratified/group/time_series/purged), `verify_no_leakage`, `build_cv_splitter`, `build_preprocessor`, `build_ml_pipeline`, `cross_validate_leakage_free`. |
| `format_audit.py` | `DataFormatAuditor`, `DatasetFormatAuditReport` | Post-import defect audit (numeric-as-string, date-family, encoding); severity-ranked. Auto-run on ingest. |
| `intake_audit.py` | `InitialIntakeAuditor` | Post-ingestion diagnostics + `auto_clean_preset`. Surfaced in Studio 01. |
| `feature_store.py` | `LocalFeatureStore`, `FeatureEntity`, `FeatureViewSpec` | Feast-compatible offline (Parquet) + online (SQLite); confined paths. |
| `versioning.py` | `DatasetVersionManager`, `DatasetVersion` | Content-addressed versions, lineage, `commit/checkout/list/diff`, DVC export. |
| `synthetic.py` | `SyntheticDataGenerator` | classification/regression/time_series/survival/causal generators, Gaussian-copula synthesis, DP histograms. Backs test fixtures and demos. |
| `multimodal.py` | `MultimodalDataset`, `MultimodalProcessor` | `extract_image/audio/video/geospatial/sensor_iot/genomic_features`. |

## EDA / Features / Evaluation

**`eda/`** — `profiler.py` (`DataProfiler` → `DatasetProfile`);
`statistics.py` (`StatisticalAnalyzer`: parametric/non-parametric,
`distance_correlation`, VIF, stationarity; `friedman_nemenyi_test`);
`diagnostics.py` (`UniversalDiagnosticsSuite` → hypothesis ledger, baseline
benchmarking, risk engine; Studio 03); `automated_eda.py`
(`AutomatedEDA.run(df, target_column, generate_figures)` → `EDAReport`,
`export_html`); `visualization.py` (`VisualizationEngine`, Plotly
dark/light); `insights.py` (`InsightGenerator` → `AnalyticalInsight`).

**`features/`** — `engineering.py` (`FeatureEngineeringPipeline`:
`fit`/`transform`/`fit_transform_split`, leakage-safe tabular + time-series);
`selection.py` (`FeatureSelector`: variance/correlation/MI/F-test/L1/tree/
RFE/Boruta/mRMR); `embeddings.py` (`EmbeddingGenerator` + semantic search);
`store_client.py` (`FeatureStoreClient` unifying local + Feast).

**`evaluation/`** — `metrics.py` (`MetricsEvaluator`:
classification/regression/forecasting/clustering/financial,
`calibration_metrics`, `deflated_sharpe_ratio`; `ProbabilityCalibrator`;
`evaluate_anomaly_adbench`); `validation.py` (`CrossValidator`,
`PurgedTimeSeriesSplit`, `CombinatorialPurgedKFold` with purge+embargo);
`interpretability.py` (`permutation_importance`, `explain_shap_local`,
`explain_lime_local`, `partial_dependence`, `generate_counterfactual`,
`explain_diff`, `conformal_explain`); `fairness.py` (`FairnessAuditor.audit`
→ demographic parity/equalized odds/disparate impact;
`compute_reweighing_weights`, `optimize_group_thresholds`);
`robustness.py` (`RobustnessTester`, `ConformalPredictor`:
`split_conformal_quantile`, `adaptive_conformal_intervals`,
`venn_abers_calibrate`, `knn_shapley_valuation`, `nested_cv_evaluate`);
`backtesting.py` (`BacktestingEngine`: walk-forward + embargo + causal
imputation); `model_cards.py` (`ModelCardGenerator` → `ModelCard`/`DataCard`,
EU AI Act / NIST AI RMF); `equations.py` (`ModelEquationExtractor.extract`
→ `ModelEquationReport`: linear/tree/deep/forecasting/baseline/nonparametric
extraction, LaTeX, Excel + external-deployment formulas; Studio 14).

## Models — `src/omniforge/models/`

**`base.py`** — `BaseOmniForgeModel(ABC)`: `__init__(*, name, task_type,
random_seed, track_carbon, country_iso, **hyperparameters)`; `fit` (seeds,
carbon-tracks, records `ModelMetadata`, wraps errors in
`ModelTrainingError`), `predict`, `predict_proba`, `score`,
`get_feature_importance`, `get_params`/`set_params`,
`save(path, *, encrypt, encryption_key)` (joblib + SHA-256 + HMAC +
optional AES-GCM + `.meta.json` sidecar), `load(path, *, encrypted,
encryption_key, require_signature=True)` (HMAC verify +
`safe_joblib_load_bytes`). Subclasses implement `_fit/_predict/
_predict_proba`. `SklearnEstimatorAdapter` wraps foreign estimators.

**`registry.py`** — `ModelCatalog` (`register`, `list_models`, `create`) +
`default_model_catalog` + `create_model(name, task_type, **kwargs)`. Eager:
9 baselines, 13 classical algorithms, 2 ensembles. Lazy (import-on-create):
`mlp`, `cnn`, `rnn`, `lstm`, `gru`, `transformer`, `gnn`, `autoencoder`,
`multimodal_net`. Unknown name → `ModelError` with suggestions.

| Module | Key class(es) | Notes |
|---|---|---|
| `baselines.py` | `DummyClassifierBaseline`, `LogisticBaseline`, `DecisionStump*`, `DummyRegressorBaseline`, `Ridge/LassoLinearBaseline`, `ForecastingBaseline`, `BaselineBenchmarkSuite` | Always compared; AutoML reports `lift_over_best_baseline`. |
| `classical.py` | `ClassicalMLModel(algorithm=…)` | sklearn + XGBoost/LightGBM/CatBoost dispatch; HistGB fallback. |
| `ensemble.py` | `VotingEnsembleModel`, `StackingEnsembleModel` | voting/stacking/blending/weighted averaging. |
| `deep/mlp.py` | `MLPModel` | torch or NumPy Adam backprop; `framework` recorded (`pytorch` vs `omniforge_numpy_autograd`). |
| `deep/cnn.py` | `CNN1DModel` | 1D/2D conv. |
| `deep/rnn_lstm.py` | `RecurrentSequenceModel` | RNN/LSTM/GRU, NumPy fallback. |
| `deep/transformers.py` | `TabularTransformerModel` | FT-Transformer-style `_PreNormGEGLUBlock` (Gorishniy et al., NeurIPS 2021); NumPy forward/backward with cache. |
| `deep/tabnet.py` | `TabNetModel` | Arik & Pfister (AAAI 2021); `sparsemax_numpy`, prior-scale relaxation. |
| `deep/resnet_tabular.py` | `ResNetTabularModel` | Gorishniy et al. pre-activation residual blocks. |
| `deep/gnn.py` | `GraphNeuralNetworkModel`, `GraphAutoencoderLinkModel` | GCN/GraphSAGE/GAT + GAE/VGAE. |
| `deep/autoencoders.py` | `AutoencoderModel` | reconstruction-error anomaly scoring. |
| `deep/multimodal.py` | `MultimodalFusionModel` | gated cross-modal fusion. |
| `deep/__init__.py` | `probe_foundation_backends` | reports `native` vs `transparent_fallback` per backend. |
| `time_series/statistical.py` | `StatisticalForecaster` | ETS, ARIMA/SARIMAX, Theta, Croston, Fourier-Prophet. |
| `time_series/ml_forecast.py` | `MLForecaster` | lag features + conformal intervals. |
| `time_series/deep_forecast.py` | `DeepForecaster` | N-BEATS, PatchTST, TFT. |
| `time_series/hierarchical.py` | `HierarchicalReconciler` | bottom-up/top-down/OLS/WLS/MinT-shrink; `verify_coherence`. |
| `time_series/foundation.py` | `FoundationForecaster` | Chronos-style zero-shot. |
| `anomaly/detector.py` | `AnomalyDetector` | IF/LOF/OCSVM/PCA/MAD/z/EWMA/ensemble; zero-batch-leakage. |
| `clustering/clusterer.py` | `ClusteringEngine` | KMeans/Hierarchical/GMM/DBSCAN/OPTICS/BIRCH; `auto_select_k`, gap, `bootstrap_stability_ari`, `hopkins_statistic`. |
| `nlp/nlp_engine.py` | `NLPEngine` | sentiment/NER/keywords/TextRank/LDA/classifier. |
| `nlp/advanced_nlp.py` | `AdvancedNLPSuite` | zero-shot/QA/semantic search/emotion/language/toxicity; `probe_nlp_backends`. |
| `computer_vision/vision.py` | `ComputerVisionEngine` | HOG/color/texture, Otsu detection. |
| `computer_vision/foundation.py` | `VisionFoundationHub` | CLIP/DINOv2/YOLO; `probe_vision_backends`. |
| `graphs/analytics.py` | `GraphAnalyticsEngine` | centralities/communities/embeddings. |
| `graphs/link_prediction.py` | `GraphLinkPredictionSuite` | 8 topological scorers + RRF, supervised edge operators, GAE. |
| `mining/miner.py` | `AssociationRuleMiner`, `SubgroupDiscoveryMiner`, `ManifoldProjector`, `SequentialPatternMiner` | Apriori/FP-Growth, PrefixSpan, beam search. |
| `nonparametric.py` | `PenalizedSplineGAMModel`, `NadarayaWatsonKernelModel`, `RBFKernelRidgeModel`, `DistanceWeightedKNNModel`, `IsotonicMonotonicModel`, `NonParametricAnalyticalSuite` | closed-form models; leak-free val split. |
| `survival/survival.py` | `KaplanMeierEstimator`, `CoxPHModel` | KM/log-rank/Nelson-Aalen/Cox PH/PH test/RMST. |
| `recommendation/recommender.py` | `HybridRecommender` | SVD/Item-KNN/content/hybrid. |
| `causal/`, `optimization/`, `simulation/` (under `models/`) | re-exports | shims re-exporting the top-level packages. |

**Torch-optional pattern:** deep models check
`is_package_available("torch")` and fall back to NumPy with identical API;
the active backend is recorded in `model.framework` / `metadata_.framework`
and surfaced by `probe_*_backends`. Never import `torch` at module top level
in a way that breaks the `simple` tier.

## Decision engines

**`automl/`** — `pipeline_search.py` (`AutoMLPipelineSearch.fit(X, y)` →
`AutoMLResult`: `best_model`, `leaderboard`, `lift_over_best_baseline`;
Stratified/Group/TimeSeries CV, ColumnTransformer, baselines, HPO,
ensembling, Pareto frontier, latency measurement); `backends.py`
(`AutoMLBackendHub`, `probe_automl_backends`: multi-layer stacking with
Caruana weights, genetic pipeline search, meta warm-start, external
AutoGluon/TPOT/H2O/FLAML adapters); `optimizer.py`
(`HyperparameterOptimizer`: GP-EI Bayesian, random, grid, Optuna hook);
`nas.py` (`NeuralArchitectureSearcher`: evolutionary MLP mutation).

**`forecasting/engine.py`** — `ForecastingEngine.forecast(df, target_col,
time_col, horizon)` → `ForecastingReport`; causal imputation
(`_causal_impute_split`), TimeSeriesSplit rolling CV, `diebold_mariano_test`.

**`causal/inference.py`** — `CausalInferenceEngine.estimate_ate(...)` →
`CausalEffectResult`; IPW/AIPW/PSM/DML/IV-2SLS/DiD/CATE; refutations, SMD
balance, sensitivity bounds, DoWhy/GML export. IV-2SLS **does** report
first-stage partial F with Stock–Yogo (2005) critical value 16.38
(`inference.py:419-424`) — weak-instrument diagnostics are implemented.

**`optimization/solvers.py`** — `OptimizationEngine`: `solve_lp/milp/
knapsack/assignment/transportation/vrp_savings/nonlinear`,
`optimize_portfolio` (Markowitz).

**`simulation/`** — `engines.py` (`MonteCarloSimulator`,
`DiscreteEventSimulator`, `AgentBasedSimulator`, `DigitalTwinEngine`);
`closed_loop.py` (`ClosedLoopMPCController`, `build_default_twin`;
Studio 09 twin-control).

## Pipelines / MLOps / Copilot / Reporting / Domains

**`pipelines/`** — `base.py` (`PipelineStep`, `FunctionStep`,
`PipelineContext`, `StepExecutionRecord`); `dag.py` (`PipelineDAG`:
`validate`, `topological_order`, `execution_levels`, `to_mermaid`);
`builder.py` (`PipelineBuilder`: fluent steps, `from_dict/from_yaml`,
`export_python_code`); `executor.py` (`PipelineExecutor`: fingerprint
caching, retries, carbon, lineage).

**`mlops/`** — `tracking.py` (`ExperimentTracker`, SQLite);
`registry.py` (`ModelRegistry` with fairness gate `_enforce_fairness_gate`);
`serving.py` (`ModelServingEngine`: deploy/predict/ONNX/canary);
`drift.py` (`DriftDetector`: PSI/KS/Wasserstein/JS/chi², Page-Hinkley,
ADWIN); `monitoring.py` (`ModelMonitor`, Prometheus);
`continuous_training.py` (`ContinuousTrainingOrchestrator`, env kill-switches);
`job_queue.py` (`BackgroundJobQueue`, MLflow remote config);
`future_scoring.py` (`FutureTargetPredictor`, file/SQL scoring);
`root_cause.py` (`DriftRootCauseAnalyzer`, causal discovery);
`supervisory_monitor.py` (`SupervisoryPredictionMonitor`, Studio 14).

**`copilot/`** — `agent.py` (`OmniForgeCopilotAgent`: OpenAI/Anthropic/
Azure/Ollama + offline deterministic local planner; `load_dataframe`,
`chat` → `CopilotResponse`); `tools.py` (`CopilotToolRegistry`: every major
subsystem as a callable tool); `memory.py` (`CopilotMemory`, windowed);
`prompts.py` (system prompts + orchestration templates).

**`reporting/`** — `sections.py` (`ReportSectionSpec`,
`AnalysisAuditRecord`, `ExclusionRecord`, `FormulaRuleRecord`,
`ReportChartArtifact`, `ReportSelectionConfig`; user selection strictly
filters output); `generator.py` (`ReportGenerator.build_report/export`);
`charts.py` (`ReportChartBuilder`, lazy pyplot); `exporters.py`
(`ReportExporters`: markdown/html/json/csv/csv_bundle/excel/docx/pdf/pptx/
notebook/microservice bundle; spreadsheet sanitization);
`narrative.py` (`NarrativeGenerator`); `formulas.py`
(`FormulaRulesCatalog`).

**`domains/`** — SDK in `plugin_sdk.py`: `DomainMetadata`,
`BaseDomainPlugin` (abstract `engineer_domain_features`,
`compute_domain_kpis`, `generate_sample_dataset`; concrete `load_config`,
`get_validation_rules`, `summary`), `DomainPluginRegistry`
(`register/get/list_plugins/list_names/discover_entry_points`) +
`default_domain_registry`. `domains/__init__.py` registers 18 module-level
`plugin` instances. Each plugin has a matching `configs/domains/<slug>.yaml`
and a `pyproject.toml` entry point under `omniforge.domains`. Example
pattern (`finance.py`): engineers `debt_to_income_ratio`,
`loan_to_value_ratio`, `credit_utilization`; KPIs `sharpe_ratio`,
`value_at_risk_95`, `expected_shortfall_95`, `default_rate`,
`expected_loss_total`.
