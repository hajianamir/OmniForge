# OmniForge v4.0.0 — Complete Enterprise User Guide & Analytical Reference

**OmniForge v4.0.0** — Enterprise Analytics & AI Workbench  
Canonical SemVer: `4.0.0` | API Version: `v2` | Schema Version: `2.0`  
Python: `3.12` – `3.15` | License: Apache-2.0

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Installation & Launch](#2-installation--launch)
3. [Version & Compatibility](#3-version--compatibility)
4. [Exact Step-by-Step End-to-End Workflow](#4-exact-step-by-step-end-to-end-workflow)
5. [Complete Guide to All 15 Studios](#5-complete-guide-to-all-15-studios)
6. [Phase 6 — C5: Causal Drift Root-Cause Attribution Engine](#6-phase-6-c5-causal-drift-root-cause-attribution-engine)
7. [Phase 6 — C7: Closed-Loop Digital Twin MPC Controller](#7-phase-6-c7-closed-loop-digital-twin-mpc-controller)
8. [Phase 6 — C4: Air-Gapped Deterministic Copilot Tools](#8-phase-6-c4-air-gapped-deterministic-copilot-tools)
9. [Complete Algorithm, Statistical Test & Rule Selection Reference](#9-complete-algorithm-statistical-test--rule-selection-reference)
10. [Core Mathematical Formulas & Universal Threshold Compendium](#10-core-mathematical-formulas--universal-threshold-compendium)
11. [Python SDK Reference — Core Patterns](#11-python-sdk-reference--core-patterns)
12. [REST API Reference](#12-rest-api-reference)
13. [Copilot Agent & Tool Reference](#13-copilot-agent--tool-reference)
14. [Security, PII, Audit & Encryption Guide](#14-security-pii-audit--encryption-guide)
15. [Logging & Observability](#15-logging--observability)
16. [Troubleshooting & FAQ](#16-troubleshooting--faq)

---

## 1. System Requirements

| Component | Requirement | Notes |
|---|---|---|
| **Python** | 3.12, 3.13, 3.14, or 3.15 (64-bit) | Recommended: Python 3.15.2 or 3.14.10 |
| **OS** | Windows 10/11 (x64), Linux (x86_64/ARM64), macOS (Apple Silicon/Intel) | Fully cross-platform |
| **NumPy** | >= 2.5.5 | NumPy 2.x API |
| **Pandas** | >= 3.1.1 | Pandas 3.x with `StringDtype` default |
| **SciPy** | >= 1.18.1 | For statistics, optimization, signal processing |
| **scikit-learn** | >= 1.9.1 | For classical ML baselines |
| **Pydantic** | >= 2.14.3 | Pydantic v2 settings & data models |
| **FastAPI** | >= 0.163.1 | REST API backend (optional extra) |
| **Plotly** | >= 7.1.0 | Interactive visualizations |
| **DuckDB** | >= 2.0.0 | Embedded analytical SQL |
| **PyArrow** | >= 26.0.1 | Columnar memory format |
| **structlog** | >= 26.1.0 | Structured logging (optional) |
| **Zero Runtime Node.js** | Not required | React 18 / TypeScript Workbench pre-compiled into `src/omniforge/ui/static_app/` |

---

## 2. Installation & Launch

### 2.1 Windows One-Click Installation

OmniForge ships complete Windows automation:

| Script | Purpose |
|---|---|
| `install.bat` | Detects Python 3.12–3.15, creates `.venv`, installs dependencies, initializes `.env`, verifies 18 domain plugins, optionally builds `OmniForge.exe`, creates desktop shortcut |
| `start.bat` | Enterprise Launch Center with interactive menu and CLI flags (`--full`, `--ui`, `--api`, `--exe`, `--cli`, `--check`, `--test`). Launches FastAPI REST API (`:8000/docs`) and React Workbench (`:8501`) |
| `build_exe.bat` | Builds standalone `OmniForge.exe` (`--onedir` or `--onefile`), optional `OmniForge-CLI.exe`, and Inno Setup installer with SHA256SUMS |

```bat
REM Interactive full installation
install.bat

REM Unattended enterprise install + standalone .exe + shortcut
install.bat --profile standard --build-exe --onedir --shortcut

REM Launch platform
start.bat
start.bat --full
start.bat --ui
start.bat --api

REM Rebuild standalone .exe
build_exe.bat --onedir --all
build_exe.bat --onefile
```

### 2.2 Linux / macOS / Python Developer

```bash
cd OmniForge
pip install --upgrade -r requirements.txt
pip install -e .

# Optional advanced extras
pip install -e ".[api,ui,reporting,data,forecasting,deep,boosting,automl,responsible,build,dev]"

# Verify installation
omniforge check

# Launch REST API
omniforge serve --host 0.0.0.0 --port 8000

# Launch Enterprise Workbench
omniforge ui --port 8501
```

### 2.3 Docker

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

| Service | URL |
|---|---|
| Enterprise Workbench | `http://localhost:8501` |
| FastAPI Swagger Docs | `http://localhost:8501/docs` |
| Healthcheck | `/api/v1/health` |

The API and Workbench services share the named workspace volume at `/var/lib/omniforge`. Keep that volume and the existing signing and encryption keys when rebuilding the containers. For SDK or single-process deployments with explicitly configured registry, artifact, or tracking paths, carry those settings forward to keep using the same data. The registry records artifact locations as absolute paths. Preserve those mount points when moving an existing deployment, and verify that a saved model loads before retiring the old storage.

To verify the installed Python packages in a locally built image against its platform lock and export a CycloneDX inventory, run `python scripts/generate_sbom.py --image omniforge-enterprise:4.0.0 --out docker-sbom.cdx.json`.

### Repeated-unit fairness and forecast interval calibration

In Studio 07, choose a **Cluster ID** column when multiple records belong to the same independent unit, such as a customer, device, or site. The automatic option uses the active Split Lock group. The Python equivalent is `FairnessAuditor.audit(..., cluster_ids=...)`. Clustered intervals allow dependence within each unit and assume independence between units. For evaluation on unseen units, use a group-disjoint split.

In Studio 08, **Interval calibration origins** enables rolling calibration for the selected champion. Arrange the source rows in chronological order. Use **Evaluation windows** to choose how many later non-overlapping windows to assess. The report shows coverage and counts by forecast lead. Native and calibrated comparisons use the same observed outcomes with valid bounds for both.

At the default 95% level, calibration requires at least 19 usable error observations per lead. Each evaluation window uses another full forecast horizon, in addition to the history reserved for selection and calibration. Observed coverage describes those dates and can change under drift. The SDK exposes both controls as `ForecastingEngine.forecast(..., interval_calibration_origins=20, interval_evaluation_origins=3)`.

For retrospective drift tests, add `compare_rolling_intervals=True` to the Python call. The calibration report includes a `rolling_comparison` with paired coverage, mean width, and interval score for fixed and rolling widths. Rolling widths use only completed windows, with a trailing window the same length as the original calibration. The returned forecast continues to use the fixed calibration.

In Studio 01, upload a **Trusted reference CSV**, select the record key shared by both datasets, and click **Compare with Trusted Reference**. Each dataset needs one observed, unique value per key. The audit displays the number of matched rows and comparable non-key values.

In the Studio 08 causal tab, select **Clustered difference-in-differences** for a two-period study with a fixed treatment group within each independent cluster. Supply the cluster ID and binary post-period column. The result displays when clustered uncertainty cannot be estimated.

### 2.4 Reproducible / Hash-Pinned Installs (Production)

```bash
pip install --require-hashes -r requirements.lock
pip install -e . --no-deps

# Regenerate locks after touching .in files (never edit locks by hand)
uv pip compile requirements.in --generate-hashes --python-version 3.12 --universal -o requirements.lock
uv pip compile requirements-build.in --generate-hashes --python-version 3.12 --universal -o requirements-build.lock
```

---

## 3. Version & Compatibility

v4.0.0 bumps the canonical SemVer from **3.3.0 → 4.0.0**. The version is defined in three places and kept in sync:

| File | Symbol | Value |
|---|---|---|
| `pyproject.toml` | `version` | `"4.0.0"` |
| `src/omniforge/version.py` | `__version__` | `"4.0.0"` |
| `src/omniforge/version.py` | `get_version_info()` | `{"name": "OmniForge", "short_name": "OmniForge", "version": "4.0.0", ...}` |

Check version at runtime:
```python
import omniforge
print(omniforge.__version__)                    # '4.0.0'
print(omniforge.get_version_info())             # full metadata dict
```

**Backward compatibility:** All Phase 1–5 APIs are preserved. v4.0.0 is additive — new classes, methods, tools, and endpoints are introduced without removing or changing existing public signatures. The `Split Lock` role migration (`admin` → `platform_admin` for cross-tenant access) is the only behavioral change; tenant-scoped `admin` remains fully functional within its own tenant.

---

## 4. Exact Step-by-Step End-to-End Workflow

Use this iterative CRISP-DM workflow to track lineage, compare baselines, and reserve data for final evaluation. Revisit preparation and modeling when the evidence calls for it. Choose a split that matches deployment and verify feature availability at prediction time.

| Step | Studio / Control | Requirement | Exact Action | Why & What Happens |
|---:|---|---|---|---|
| **1** | **Studio 00 · Ingestion** | Mandatory | Load CSV, TSV, multi-sheet Excel (`.xlsx`), Parquet, JSON, SQL DB, MS Access (`.accdb`/`.mdb`), or a built-in benchmark domain dataset. | `InitialIntakeAuditor` executes immediately (health grade A–F, missingness, duplicates, outliers, constant/mixed-type columns, skewness, leakage warnings). |
| **2** | **Topbar · Targets & Exclusions** | Mandatory | Confirm Target Column(s) in **Targets** and mask raw IDs / post-outcome leakage fields in **Exclusions**. | Reversibly scopes all 15 studios without modifying Raw Data. |
| **3** | **Topbar · Split Lock** | **Recommended BEFORE Cleaning** | Click **Split Lock**, select Strategy (`Auto`, `Stratified`, `Random`, or `Chronological Time-Series`), Test Ratio (e.g., 20%), CV Folds, then click **Lock Train/Test Split**. | **Anti-Leakage Gold Standard**: Forces Studio 01 Cleaning and Studio 02 Clustering to fit statistics (medians, means, outlier fences, scalers, cluster centroids) strictly on Train and transform Test without leakage. |
| **4** | **Studio 01 · Quality & Cleaning** | Mandatory | Run **Quality & Format Audit**, inspect **Dry-Run Preview**, pick `Auto` / `Complete` / `Custom` mode, click **Apply & Commit Cleaning**. | Updates Active Data (`RAW` → `CLEANED`), preserves Raw Data separately (with Undo & Revert to Raw), logs every deleted row/column to the Forensic Deleted-Records Audit Log. |
| **5** | **Studio 02 · Clustering, Anomaly & Mining** | Recommended | Run Clustering, Anomaly Screening, and/or Data Mining in `PREVIEW` mode. Optionally **Remove Selected Corrupt Clusters / Anomalies** or **Attach `cluster_id` / `is_anomaly`**. | Isolates faulty operating regimes/outliers before modeling. Lights up Studio 02 green status LED (`CLEANED + CLUSTER-SCREENED`). |
| **6** | **Studio 03 · Diagnostics, Baselines & Risk** | Recommended | Run diagnostics and compare the inclusive baseline using the active Split Lock. Use the holdout ratio to form a diagnostic holdout when no lock is active. | Establishes a reference from the eligible study predictors before optional Studio 05 selection. |
| **7** | **Studio 04 · Automated EDA & Hypothesis Lab** | Recommended | Run Automated EDA or the hypothesis battery. For a custom test, select a method matching the sampling design. | Reports evidence and skipped tests. Battery decisions use Benjamini–Hochberg adjusted p-values. A fitted-null test run on a family selected from the same sample is exploratory. |
| **8** | **Studio 05 · Feature Engineering & Selection** | Conditional | Optionally generate skew-corrected, interaction, polynomial, or datetime cyclical features; run 4-Method Consensus Selection. | Adds engineered features (`ENGINEERED` provenance) and reversibly excludes low-signal noise columns. |
| **9** | **Studio 06 · AutoML, Non-Parametric & Deep Learning** | Mandatory | Train Multi-Backend AutoML, 5 True Non-Parametric Analytical Models (closed-form formulas & GAM decompositions), and/or Paper-Grade Deep Learning architectures. | Benchmarks every model strictly against strong baselines, promotes Champion to Active Model, unlocks Studios 07, 11, 12, 14. |
| **10** | **Studio 07 · Responsible AI** | Recommended | Run Global Permutation Importance, Local SHAP, Partial Dependence Plots (PDP), Fairness Audit, Model Card generation. | Verifies explainability, domain plausibility, and fairness compliance (Disparate Impact ≥ 0.80). |
| **11** | **Studios 08–11 · Decision AI, Specialized AI, DAG & MLOps** | Conditional | Run Forecasting / Causal ATE / Markowitz / Monte Carlo (S08), NLP / Vision / Graph / RecSys / Survival + **C7 Twin MPC** (S09), DAG Pipeline (S10), or **Model Registry & PSI/KS Drift Monitor** + **C5 Root-Cause** (S11). | Extends analysis to forecasting, causal policy impact, unstructured modalities, and production governance. |
| **12** | **Studio 12 · Executive Reporting & Export** | Mandatory | Select desired sections from 14 available chapters, preview narratives/tables/charts, export **PDF, XLSX, DOCX, HTML, Markdown, CSV, JSON, LaTeX**. | Compiles full end-to-end audit evidence into publication-ready deliverables. |
| **13** | **Studio 13 · AI Copilot** | Operational | Ask the air-gapped deterministic Copilot in plain English: "profile data quality", "train automl", "root cause diagnosis", "twin control", "pii audit", "pareto automl", "register model", etc. | Executes real OmniForge analytics end-to-end offline via 24 deterministic tools. |
| **14** | **Studio 14 · Live Supervisory Control Room** | Operational | Choose Champion + Baseline models, set Tolerance Rules (`% Error`, `Abs Error`, `σ-Z`, `Baseline Ratio`, `Batch Breach %`), connect live CSV/XLSX/SQL/MS Access or Probe input, evaluate. | Monitors real-time operational accuracy, plots Actual vs Predicted vs Baseline SPC limits, triggers `HEALTHY` / `WARNING` / `CRITICAL_ALARM` LEDs, exports scored audit logs. **C5 Root-Cause** available at `/s14/root-cause`. |

### 4.1 Split Lock — Mandatory Before Cleaning for Predictive Modeling

**YES** for predictive modeling (AutoML, Non-Parametric Analytical Models, Deep Learning, Diagnostics Baselines, Supervisory Deployment). If numeric imputation (mean/median/KNN/MICE), outlier fences (IQR/Z-score/MAD), high-missing column thresholds, or scaling/clustering are computed on the entire dataset before splitting, test-set statistics leak into training and inflate validation accuracy.

Row-independent formatting fixes can run before the split. Group repeated records by entity before splitting when independence between entities matters. Split Lock is optional for exploratory cleanup and spreadsheet exports that will not feed predictive evaluation.

---

## 5. Complete Guide to All 15 Studios

### Studio 00 — Ingestion

**Purpose:** Ingest multi-source datasets, automatically grade data quality at intake, control Train/Test Split Lock, and apply auditable cleaning rules.

**Supported Sources (30+):** CSV, TSV, Excel (`.xlsx`, multi-sheet), Parquet, Feather, JSON, JSONL, ORC, Avro, HDF5, Pickle, XML, YAML, TOML, SAS, SPSS, Stata, SQL (any SQLAlchemy dialect), DuckDB, MongoDB, REST APIs, S3, GCS, Azure Blob, In-memory DataFrames.

**Key Controls:**
- **Initial Intake Auditor:** Health grade A–F, missingness, duplicates, outliers, constant/mixed-type columns, skewness, leakage warnings
- **Targets & Exclusions:** Scope target columns; mask leakage fields (reversible, no raw data modification)
- **Split Lock:** Anti-leakage train/test split before any statistics are fit
- **Undo / Revert to Raw:** Full reversibility of cleaning operations
- **Forensic Deleted-Records Audit Log:** Every deleted row/column logged with stage and reason

**Applications:** Spreadsheet QA, industrial SCADA/telemetry preprocessing, SQL/Access ETL validation, forensic record exclusion auditing.

---

### Studio 01 — Cleaning

**Purpose:** Leakage-safe data cleaning with 11-stage symmetric pipeline.

**11-Stage Order (identical across `fit()` and `transform()`):**

1. `clean_strings` — HTML/whitespace stripping, missing-token standardization
2. `standardize_missing_tokens` — `"N/A"`, `"NA"`, `"null"`, `""` → `NaN`
3. `infer_and_cast_types` — Column type inference and casting
4. `replace_nonfinite_with_nan` — `inf`, `-inf` → `NaN`
5. `apply_domain_bounds` — Mark values outside explicit domain limits as missing
6. `drop_high_missing_columns` — Drop columns above the configured missingness threshold
7. `drop_exact_duplicates` — Remove fully duplicate rows
8. `fuzzy_deduplicate` — Near-duplicate detection and removal
9. `add_missing_indicators` — Add `_is_missing` flag columns
10. `outliers` — Outlier detection (IQR, MAD, Z-score, Isolation Forest, LOF)
11. `impute_missing` — Missing value imputation (median, mean, KNN, MICE)

**Modes.** Studio 01 `Auto` and `Complete` flag outliers for review. `Custom` exposes treatment rules. The SDK defaults to IQR detection and clipping, with a configurable action. The fitted cleaner screens predictors for outliers before applying imputation. An active temporal split supplies the time column and enables causal forward filling for time-based numeric imputation. Cleaning preserves selected outcomes, split key names, and rows reserved for the temporal gap. Reserved rows stay outside the training and holdout transforms.

**Workflow.** Profile and validate the incoming data, use the health assessment to prioritize fixes, preview cleaning rules, and then re-profile and validate the result before committing. Use the deletion log, versioning, and exports to record the changes. Keep multicollinearity, skewness, and class imbalance in the statistical review. Compare reference and current cohorts for drift and track operational degradation in Studios 11 and 14. Studio 03 is diagnostic. Apply any independently justified domain features before Studio 05, using predictors available at scoring time.

The dataset quality profile reports completeness, uniqueness, validity, consistency, reference accuracy, and timeliness. Reference accuracy requires observed matches against trusted values. Timeliness requires an evaluable timestamp. The score renormalizes the weights over evaluable dimensions and returns them as `effective_weights`. Rule conformance and statistical plausibility are reported separately where evaluated.

Missingness indicators and outlier flags generated by Workbench cleaning are audit-only. They remain visible in Active Data and clean exports. The Workbench excludes them from study inputs, model feature engineering and training. The exclusion follows the recorded generated column names, including after renaming or workspace restore. Uploaded columns with similar names remain eligible. The SDK returns its configured indicator columns to callers, who control which columns to pass to models.

**Provenance:** Updates Active Data while keeping Raw Data intact. Supports Undo, Revert to Raw, clean CSV/XLSX export.

---

### Studio 02 — Clustering, Anomaly Screening & Data Mining Lab

**Purpose:** Discover subpopulations, flag/remove corrupt regimes or multivariate anomalies, mine co-occurrence rules, discover high-signal subgroups, visualize high-dimensional manifolds.

**Clustering Algorithms:**
| Algorithm | Paradigm | Best For |
|---|---|---|
| K-Means | Centroid | Fast, roughly spherical numeric clusters |
| MiniBatch KMeans | Centroid | Large-scale streaming clusters |
| Agglomerative | Hierarchical | Nested hierarchies (n < 5,000) |
| GMM | Probabilistic elliptical | Overlapping/elongated clusters |
| DBSCAN | Density-based + noise | Arbitrary non-convex shapes, noise isolation |
| OPTICS | Density-based | Varying density clusters |
| BIRCH | Hierarchical tree | Large n, incremental clustering |

**Anomaly Detection:** Isolation Forest, Local Outlier Factor (LOF), One-Class SVM.

**Data Mining:** Apriori Association Rules (Lift > 1.2), Subgroup Discovery (WRAcc), PCA/SVD/t-SNE/Isomap/MDS dimensionality reduction.

When the study has multiple selected targets, choose one outcome for subgroup and advanced mining. Rerun for another outcome. Selected outcomes are excluded from predictor and projection inputs. A selected outcome can color a projection.

**Safety Model:** Always previews first (`PREVIEW_ONLY`). Nothing modifies Active Data until you explicitly click **Remove Selected Clusters**, **Remove Anomalies**, or **Attach Cluster/Anomaly Features**.

---

### Studio 03 — Universal Diagnostics, Ranked Baseline Equations & Risk

**Purpose:** Test parametric assumptions, measure tail risk, rank mathematical baseline models against naive floors.

**Diagnostics:** Normality (Shapiro-Wilk, D'Agostino K², Anderson-Darling, KS), correlation (Pearson, Spearman, Kendall), multicollinearity (VIF), financial/operational tail risk (VaR/CVaR).

**Baseline Models (ranked with Lift over Naive Floor):**
- Ridge linear regression and logistic regression
- All results include LaTeX and Python code for reproducibility

**Thresholds:** `VIF > 10` indicates severe collinearity; `CVaR` measures expected tail loss beyond `VaR`.

---

### Studio 04 — Automated EDA & Hypothesis Lab

**Purpose:** Visual distribution, correlation, and missingness profiling with parametric and nonparametric hypothesis tests.

Each studio's live-chart picker lists its assigned charts. Studio 12 provides the cross-studio gallery for reporting and export.

**Test families:**
| Test | Data Type | Non-Parametric Alternative |
|---|---|---|
| Shapiro-Wilk | 1 Numeric | D'Agostino K² |
| D'Agostino K² | 1 Numeric | Anderson-Darling |
| Anderson-Darling | 1 Numeric | KS Normality |
| Lilliefors normality | 1 Numeric | — |
| Welch's t-Test | Numeric × 2 Groups | Mann-Whitney U |
| Paired t-Test | 2 Paired Numeric | Wilcoxon Signed-Rank |
| 2-Sample KS Test | Two independent numeric samples | — |
| One-Way ANOVA | Numeric × 3+ Groups | Kruskal-Wallis H |
| Levene / Fligner / Bartlett | Numeric × 2+ Groups | Fligner-Killeen |
| Pearson r | 2 Numeric | Spearman ρ / Kendall τ |
| Chi-Square χ² | 2 Categorical | Fisher's Exact Test |

The Studio 04 battery shares observations across tests. Interpret its Benjamini–Hochberg adjusted results as exploratory when the dependence assumptions have not been established. Use a prespecified test family and check the sampling assumptions for confirmatory work. Studio 03 separately offers Bonferroni correction.

---

### Studio 05 — Feature Engineering & 4-Method Consensus Selection

**Purpose:** Create Yeo-Johnson skew corrections, pairwise interactions, polynomial expansions, datetime cyclical features; rank predictors via Mutual Information, F-Test, Tree Importance, L1 Lasso, Variance Threshold, or 4-Method Consensus.

**Feature Types:**
- Skew corrections (Yeo-Johnson)
- Pairwise interactions
- Polynomial expansions
- Datetime cyclical features (sin/cos encoding)
- Time-series lags & rolling windows

**Selectors (8):** Variance, Correlation, Mutual Info, Chi2, ANOVA, Lasso, Tree, RFE + 4-Method Consensus.

Studio 05 ranks features on the training partition when Split Lock is active. Preview shows encoded features and their source columns. Commit retains those source predictors in Active Data for Studio 06. Native `AutoMLPipelineSearch(select_k_features=...)` affects fold-local CV diagnostics for supported candidates. Use committed or explicitly supplied predictor columns to control production inputs.

---

### Studio 06 — Multi-Backend AutoML, Non-Parametric Analytical & Deep Learning

**Purpose:** Unified predictive modeling studio combining four layers:

**1. Multi-Backend AutoML Hub:** 4 native SOTA engines (`native_stacking`, `native_multilayer_stacker`, `native_evolutionary`, `native_fast_gbdt`) + 5 optional external adapters (FLAML, AutoGluon, H2O, TPOT, Auto-Sklearn). Mandatory strong baselines first, explicit lift over strongest baseline.

**2. 5 True Non-Parametric Analytical Models:**
| Model | Functional Form | Best For |
|---|---|---|
| Penalized B-Spline GAM | ŷ = β₀ + Σⱼ Σₘ wⱼₘ Bⱼₘ(xⱼ) + λ\|w\|₂² | Curved/non-Gaussian relationships with transparent per-feature curves |
| Nadaraya-Watson Kernel Smoother | ŷ(x) = Σ Kₕ(x-xᵢ)yᵢ / Σ Kₕ(x-xᵢ) | Low-dimensional (p < 15) distribution-free smooth local expectations |
| RBF Kernel Ridge (RKHS) | ŷ(x) = Σ αᵢ exp(-γ\|x-xᵢ\|²) | Smooth nonlinear response surfaces with multi-way interactions (n ≤ 5,000) |
| Distance-Weighted k-NN | ŷ(x) = Σ wᵢyᵢ / Σ wᵢ | Multimodal local manifolds with zero global shape assumptions |
| Isotonic Monotonic (PAVA) | Non-decreasing/non-increasing step | Guaranteed monotonic physics (dose-response, sensor calibration) |

**3. Paper-Grade Deep Learning:** FT-Transformer, TabNet, ResNet-Tabular, MLP, 1D-CNN, GCN, GAT, GraphSAGE with loss curves & feature importances.

The deep benchmark uses the active Split Lock or creates a separate holdout before fitting its input schema. Its row cap samples training and holdout separately. The response records the partition source, available and evaluated row counts, and the fitted predictor schema. Deep candidates and baselines are scored on the same holdout rows.

**4. Single-Row & Batch CSV Scoring.** Models trained by Studio 06 retain a fitted input schema with the workspace. Both scoring endpoints accept the original columns, including categorical values and datetime-typed features, and apply the same input preparation and model-fitted encoding used in training. Blank or omitted inputs follow the model's missing-value handling; unknown categories follow the trained encoder's policy. Numeric-only workbench models retain fitted medians and category encodings. Invalid numeric/datetime values return a 422 error rather than silently becoming zero. Older saved workspaces without a fitted schema retain their legacy numeric scoring behavior until retrained.

**Background AutoML jobs:** Set `async_job: true` on `POST /api/workbench/s06/automl`, then poll `GET /api/workbench/jobs/{job_id}` or list `GET /api/workbench/jobs`. Cancel via `POST /api/workbench/jobs/{job_id}/cancel`. Queued jobs cancel immediately; running AutoML jobs return 202 (`cancelling`) and stop at the next checkpoint before the model is published. A training step already in progress cannot be forcibly interrupted. Once publication begins, or for running jobs that do not opt into cooperative cancellation, the endpoint returns 409. A full queue returns 429 with `Retry-After`.

Job status and JSON results are stored in SQLite (default `paths.workspace_dir/workbench_jobs.db`). Configure `OMNIFORGE_JOBS__MAX_WORKERS` (per process), `OMNIFORGE_JOBS__MAX_QUEUE_DEPTH` (shared waiting-job limit), `OMNIFORGE_JOBS__MAX_HISTORY` (terminal records), `OMNIFORGE_JOBS__WORKER_TIMEOUT_SECONDS`, and optionally `OMNIFORGE_JOBS__DB_PATH`. API workers must use the same settings and a shared filesystem with reliable SQLite locking. A vanished worker's queued/running jobs are marked failed after the lease expires; Python callables are **not** persisted or automatically resumed. Workspace datasets and active models remain process-local, so multi-worker workspace operations still require sticky routing or a separate shared workspace implementation. The separate `omniforge.mlops.job_queue.BackgroundJobQueue` helper remains in-memory.

On graceful API shutdown the worker drains accepted work and then releases its SQLite lease. Set the deployment stop timeout to accommodate the remaining jobs. SDK queue owners can call `shutdown(wait=True, cancel_pending=True)` to cancel waiting tasks and join running threads. The SDK queue defaults to 128 waiting tasks and 128 terminal records. The in-memory API rate limiter also bounds tracked client keys at 10,000 per process. When all slots hold active windows, it returns HTTP 429 to new clients until a slot expires.

**External Deployment Formula Kit:** After AutoML completes, OmniForge extracts governing mathematical relationships and exposes:
- Standalone Python (`.py`) — zero-dependency scoring function
- Excel Formula (`=...`) — spreadsheet formula mapped to columns
- SQL `SELECT` Expression — PostgreSQL, SQL Server, DuckDB, Snowflake, SQLite
- MATLAB / C / PLC / SCADA — embedded controllers, Simulink, engineering scripts
- LaTeX — publication-ready notation + Surrogate Fidelity R² verification

---

### Studio 07 — Responsible AI (Explainability, Fairness & Governance)

**Purpose:** Global Permutation Importance, Local SHAP explanations, Partial Dependence Plots (PDP), Algorithmic Fairness audits, ISO/NIST-style Model Cards.

**Explainability:** Permutation Importance, Monte Carlo SHAP, LIME local surrogate, PDP/ICE profiles, Counterfactuals (Wachter/DiCE inverse-MAD L₁).

**Fairness:** Disparate Impact, Equalized Odds, Demographic Parity. **EEOC 4/5ths rule:** DI in [0.80, 1.25] is compliant; < 0.80 is adverse impact violation.

**Mitigation:** Kamiran-Calders reweighing. Compliance Model Cards generated automatically.

---

### Studio 08 — Forecasting, Causal Inference, Optimization & Simulation

**Forecasting:** Statistical, ML, Deep (TFT, N-BEATS, PatchTST, LSTM), Hierarchical, Foundation probabilistic forecasting.

**Causal Inference:** IPW, AIPW (Doubly Robust), PSM, DML, IV-2SLS, DiD, CATE + DoWhy-style refutations. Sensitivity bounds (Cinelli-Hazlett OVB, Robustness Values).

**Optimization:** LP, MILP, Knapsack, Hungarian, Markowitz Mean-Variance, Transportation, CVRP, SLSQP. Ledoit-Wolf covariance shrinkage + PSD projection for portfolios.

**Simulation:** Monte Carlo (with Gaussian copula for correlated inputs), Discrete-Event, Agent-Based, Digital Twin.

---

### Studio 09 — Specialized AI (NLP, Vision, Graphs, RecSys & Survival) + C7 Twin MPC

**Purpose:** Multi-modal analytics covering NLP, Computer Vision, Graph Analytics, Hybrid Recommender Systems, Survival/Reliability Analysis.

**NLP:** Zero-Shot, QA, PII Redaction, Topic Modeling, NER.

**Computer Vision:** CLIP/DINOv2/YOLO/Segmentation (optional deep backends; native fallbacks when offline).

**Graph Analytics:** Link Prediction (RRF/VGAE), Graph Neural Networks (GCN, GAT, GraphSAGE).

**RecSys:** Hybrid Recommender Systems.

**Survival:** Kaplan-Meier Estimator, Cox Proportional Hazards (C-Index ≥ 0.75 strong discrimination).

**✅ C7 — Closed-Loop Digital Twin MPC (Phase 6):**

Endpoint: `POST /s09/twin-control`

Solves the constrained receding-horizon quadratic program:
```
u*_{t:t+H} = argmin  Σ ||H x̂_{t+k|t} - r_{t+k}||_Q² + ||u_{t+k}||_R²
subject to u_min ≤ u ≤ u_max
           x_min ≤ x̂ ≤ x_max        (state safety bounds)
           x̂_{t+k+1|t} = A x̂_{t+k|t} + B u_{t+k}
```

**Request Schema (`TwinControlReq`):**
| Field | Type | Default | Description |
|---|---|---|---|
| `n_x` | int | 2 | State dimension (1–16) |
| `n_u` | int | 1 | Control dimension (1–8) |
| `horizon` | int | 10 | MPC receding horizon H (1–100) |
| `n_steps` | int | 8 | Closed-loop steps to apply (1–500) |
| `dt` | float | 1.0 | Sampling period (0 < dt ≤ 3600) |
| `setpoint` | list[float] \| null | null | Flat setpoint path; tiled across horizon if short |
| `u_min` / `u_max` | list[float] \| null | null | Control input bounds |
| `x_min` / `x_max` | list[float] \| null | null | State safety bounds |
| `Q_diag` / `R_diag` | list[float] \| null | null | State/control cost diagonals |
| `measurements` | list[list[float]] \| null | null | Per-step measurements (n_steps × n_x) |

**Response:** `twin_control` report with `state_trajectory`, `control_history`, `setpoint`, `stage_costs`, `total_cost`, `all_constraints_satisfied`, `constraint_violations`, `solve_time_ms`, `slack_used`.

**C7 Enhancements (v4.0.0):**
- **Constraint softening (slack variables):** When hard bounds make the QP infeasible, slack variables (bounded ≥ 0, heavily penalized at 1e⁴ weight) let the solver find a near-feasible solution instead of failing completely
- **Solve time tracking (`solve_time_ms`):** Every solve reports execution time for observability
- **Slack usage reporting (`slack_used`):** Reports max slack value so operators know when bounds were too tight
- **Bounded relaxation fallback:** Max 1.2× bound relaxation (was 1.5×), preventing dangerous safety bound violations
- **Fixed R matrix dimension:** `R` now correctly matches control dimension (`n_u × n_u`)
- **Fixed warm-start alignment:** Slice-and-concatenate instead of `np.roll`, preventing control/state misalignment
- **Fixed setpoint padding:** Clean tiling logic instead of confusing slice-then-pad

---

### Studio 10 — Visual DAG Pipeline Builder

**Purpose:** Composable visual Directed Acyclic Graph orchestrator executing `quality → clean → eda → features → automl` in one auditable sequence.

**Features:** Fluent & YAML `PipelineBuilder`, cycle-checked `PipelineDAG` (Mermaid & Python export), `PipelineExecutor` with caching & retries, tenant-scoped cache keys, CWE-94-safe code export.

---

### Studio 11 — MLOps (Model Registry, Experiments & Drift Monitor) + C5 Root-Cause

**Purpose:** Register versioned models across lifecycle stages and audit covariate shift.

**Model Registry Stages:** None → Staging → Production → Archived. Semantic versioning. SQLite MLflow-compatible.

**Drift Detection:** PSI, KS, Wasserstein, JS, Chi², Page-Hinkley. PSI thresholds: < 0.10 stable, 0.10–0.25 moderate, ≥ 0.25 major drift.

**Feature-store freshness:** Studios 11 and 14 show a tenant-scoped snapshot from `GET /api/workbench/feature-store/freshness`, refreshed every minute. It reports online vector counts, currently expired vectors and per-view stale alerts. Timestamp parsing falls back from event time to materialization time, matching online TTL retrieval; unknown timestamps are counted separately and nonpositive TTL disables expiry. Counts are snapshots of records still in the online store, not cumulative expiration events, and exclude purged records. The default tenant uses `paths.feature_store_dir`; other tenants use their workspace-isolated feature-store directory.

**✅ C5 — Causal Drift Root-Cause Attribution (Phase 6):**

Endpoint: `POST /s11/root-cause`

Connects `DriftDetector` feature shifts → Delta-attribution per drifted feature → structural DAG upstream-ancestor traversal to isolate root-cause upstream sensor/source shifts from downstream correlated symptoms.

**Request Schema (`RootCauseReq`):**
| Field | Type | Default | Description |
|---|---|---|---|
| `reference_dataset` | string | "reference_dataset" | Copilot/workbench dataset name for reference |
| `current_dataset` | string | "current_dataset" | Copilot/workbench dataset name for current |
| `target_col` | string \| null | null | Target column to exclude from features |
| `time_col` | string \| null | null | Time column for temporal DAG orientation |
| `psi_warning` | float | 0.10 | PSI warning threshold (0–1) |
| `psi_critical` | float | 0.25 | PSI critical threshold (0–1) |
| `max_cond_set_size` | int | 2 | Max conditioning set size for PC skeleton (0–4) |

**Response:** `root_cause_report` with `overall_status`, `root_causes`, `symptoms`, `drifted_features` (each with `column`, `status`, `psi`, `delta_prediction`, `is_root_cause`, `is_symptom`, `root_cause_confidence`, `upstream_ancestors`, `downstream_descendants`), `dag_edges`, `prediction_shift`, `total_attribution`, `remediation_prescriptions`.

---

### Studio 12 — Executive Reporting & Export Center

**Purpose:** Compile up to 14 customizable report sections with narrative insights, embedded tables, and charts.

**Export Formats:** PDF, XLSX, DOCX, PPTX, HTML, Markdown, CSV, JSON, LaTeX, Jupyter Notebook (.ipynb), Docker Microservice Bundle.

**Sections:** 14 available chapters covering the full CRISP-DM workflow from ingestion to supervisory deployment.

---

### Studio 13 — Context-Aware AI Analytics Copilot + C4 Deterministic Tools

**Purpose:** Workspace-grounded AI assistant that inspects live dataset statistics, intake defects, diagnostics, active model performance, and supervisory alerts.

**Providers:** OpenAI, Anthropic, Azure, Ollama + **offline deterministic local planner** (zero third-party LLM dependency).

**Endpoint:** `POST /s13/chat` with `CopilotReq{message: string}`.

**✅ C4 — Air-Gapped Deterministic Copilot Tools (Phase 6):**

10 new deterministic offline tools:

| Tool | Purpose | Key Parameters |
|---|---|---|
| `root_cause_diagnosis` | C5 causal drift root-cause: drift → Delta-attribution → DAG ancestors → remediation | `reference_dataset`, `current_dataset`, `target_col?`, `time_col?` |
| `twin_control` | C7 closed-loop Digital Twin MPC control (constrained receding horizon) | `n_x`, `n_u`, `horizon`, `n_steps`, `dt`, `setpoint?`, `measurements?`, `session_key?` |
| `split_lock` | Enforce leakage-safe Split Lock train/test integrity (LeakageGuard) | `strategy?`, `test_size?`, `cv_folds?`, `time_col?`, `group_col?`, `target_col?`, `gap?`, `dataset_name?` |
| `pii_audit` | C8 zero-trust PII audit: detect + pseudonymize or block (fail-closed) | `dataset_name?`, `pii_policy?` |
| `pareto_automl` | C3 Green-AI Pareto AutoML with carbon / inference-latency budgets | `target_column`, `time_budget_seconds?`, `max_carbon_grams_co2e?`, `max_inference_latency_ms?` |
| `domain_validate` | Validate active dataset against an industry domain plugin (18 domains) | `domain`, `dataset_name?` |
| `register_model` | Register active model to the MLOps Model Registry (Studio 11) | `model_name?`, `stage?` |
| `promote_model` | Promote a registered model to a new stage (Studio 11) | `model_name`, `version`, `stage?` |
| `list_studios` | List all 15 OmniForge Workbench studios and routes | — |
| `studio_overview` | Get overview payload for a specific studio (s11, s13, ...) | `studio_id` |

**C4 Enhancements (v4.0.0):**
- **Twin control session persistence (`session_key`):** Twin state + controller warm-start checkpointed into workbench session state; restorable across calls via `ClosedLoopMPCController.restore()`
- **Correct model lookup (`register_model`):** Looks up by `model_name` parameter first, falls back to `pareto_champion` then `automl_champion`
- **Model existence validation (`promote_model`):** Checks registry before `transition_stage`; returns structured error if not found
- **Dead code removed:** Duplicate if/else branches eliminated; variable shadowing fixed
- **Structured error handling:** Invalid stage returns `{"error": ..., "available_stages": [...]}` instead of raising

**Local Planner Keyword Routing:** The deterministic planner maps English keywords to tools: "root cause" → `root_cause_diagnosis`, "twin control"/"mpc" → `twin_control`, "split lock" → `split_lock`, "pii audit" → `pii_audit`, "pareto"/"green-ai" → `pareto_automl`, "domain validate" → `domain_validate`, "register model"/"promote model" → `register_model`/`promote_model`, "list studios" → `list_studios`, and many more.

---

### Studio 14 — Live Prediction & Engineering Supervisory Control Room + C5 Root-Cause

**Purpose:** Production monitoring & engineering digital-twin dashboard connecting Excel/CSV, SQL Databases, MS Access, Simulated Fault Streams, or Single Sensor Probes to score Champion vs. Baseline predictions against live actuals.

**Tolerance Rules (Studio 14):**
| Rule | Formula | Use Case |
|---|---|---|
| Max % Error | \|y − ŷ\| / max(\|y\|, ε) × 100% | Scale-invariant engineering KPIs |
| Max Absolute Error | \|y − ŷ\| | Temperature, pressure, voltage, currency caps |
| Max σ Z-Score | zᵢ = (yᵢ − ŷᵢ) / σ_train | SPC (\|z\| ≥ 3 signals rare abnormal excursion) |
| Baseline Ratio Envelope | Rᵢ = yᵢ / ŷ_baseline,ᵢ within [min, max] | Regime drift detection |
| Warning Fraction | Triggers WARNING at 0.75 × critical limit | Early operator intervention |

**Status LEDs:** `HEALTHY` (‖Z‖ < 2.0), `WARNING` (2.0 ≤ ‖Z‖ < 3.0), `CRITICAL_ALARM` (‖Z‖ ≥ 3.0).

**✅ C5 — Signed Root-Cause Baseline Deviation Attribution (Phase 6):**

Endpoint: `GET /s14/equations` and `POST /s14/root-cause`

Displays BOTH the Active Champion Governing Equation and Reference Baseline Equation live. On every batch evaluation and per-alarm drilldown, computes Antithetic Interventional Root-Cause Attribution:

```
φ_j = ½ [ (f(x_ref[j ← x_live,j]) − f(x_ref)) + (f(x_live) − f(x_live[j ← x_ref,j])) ]
```

Explicitly separates **Factors Driving INCREASE (+Δ Above Baseline)** from **Factors Driving DECREASE (−Δ Below Baseline)** on a signed **Diverging Tornado Chart**, quantifies **Unexplained Residual** (y_actual − ŷ_champion) to distinguish measured input shifts from unmeasured external faults or sensor drift, and generates a prioritized **P1..P6 Corrective Engineering Remediation Playbook** plus an interactive **Single-Alarm Record Inspector**.

---

## 6. Phase 6 — C5: Causal Drift Root-Cause Attribution Engine

### 6.1 Overview

The C5 engine (`src/omniforge/mlops/root_cause.py`) implements a complete **drift → attribution → root-cause** pipeline:

```
Step 1: DriftDetector (PSI / Wasserstein / JS) on reference vs current
   → per-feature drift status (HEALTHY / WARNING / CRITICAL)
Step 2: Delta-SHAP attribution (InterpretabilityEngine.explain_shap_local)
   → prediction-shift per drifted feature
Step 3: CausalDiscovery structural DAG (PC skeleton + orientation)
   → parent map: {child: set of parents}
Step 4: Root-cause / symptom classification via upstream-ancestor traversal
   → root cause = drifted node with no drifted ancestors
   → symptom = drifted node with drifted ancestors
Step 5: Remediation prescriptions
   → actionable guidance for operators
```

### 6.2 Core Classes

#### `DriftedFeature` (Pydantic model)

| Field | Type | Description |
|---|---|---|
| `column` | str | Feature column name |
| `status` | DriftStatus | HEALTHY / WARNING / CRITICAL |
| `psi` | float \| null | Population Stability Index |
| `wasserstein_distance` | float \| null | Wasserstein distance |
| `js_divergence` | float \| null | Jensen-Shannon divergence |
| `delta_prediction` | float | Delta-SHAP attribution (prediction shift) |
| `abs_delta_prediction` | float | Absolute attribution magnitude |
| `is_root_cause` | bool | True if classified as root cause |
| `is_symptom` | bool | True if classified as symptom |
| `root_cause_confidence` | float | 0.0–1.0 confidence score |
| `upstream_ancestors` | list[str] | All upstream DAG ancestors |
| `downstream_descendants` | list[str] | All downstream drifted descendants |
| `remediation` | str | Remediation guidance |

#### `RootCauseReport` (Pydantic model)

| Field | Type | Description |
|---|---|---|
| `overall_status` | DriftStatus | Overall drift status |
| `n_reference` | int | Reference dataset row count |
| `n_current` | int | Current dataset row count |
| `drifted_features` | list[DriftedFeature] | Per-feature details |
| `root_causes` | list[str] | Root cause column names |
| `symptoms` | list[str] | Symptom column names |
| `dag_edges` | list[list[str]] | [parent, child] edge pairs |
| `prediction_shift` | float | Net signed prediction shift |
| `total_attribution` | float | Total absolute attribution |
| `remediation_prescriptions` | list[str] | Actionable remediation guidance |
| `evaluated_at` | str | ISO timestamp |

#### `CausalDiscovery`

PC-style constraint-based causal discovery:

1. **PC Skeleton:** Start with complete graph, drop edges where X ⟂ Y \| S (partial correlation CI test, α = 0.05, max conditioning set size = 2)
2. **Orientation:** Temporal drift onset → drift magnitude → naming heuristics → column order (deterministic tiebreak)
3. **Cycle Detection & Removal:** DFS-based cycle detection; breaks cycles by removing weakest-orientation edge (smallest \|drift_a − drift_b\|)

**Orientation Priority Chain:**
1. **Temporal causality:** Earlier drift onset = upstream (strongest signal, when `time_col` provided)
2. **Drift magnitude:** Higher PSI shift = more likely upstream source
3. **Naming heuristics:** Sensor/process keywords (temp, pressure, flow, voltage, etc.) score upstream; derived/KPI keywords (ratio, diff, kpi, efficiency, etc.) score downstream
4. **Column order:** Lower index = upstream (deterministic tiebreak)

#### `DriftRootCauseAnalyzer`

```python
analyzer = DriftRootCauseAnalyzer(
    random_seed=42,
    psi_warning=0.10,
    psi_critical=0.25,
    max_cond_set_size=2,
)

report = analyzer.analyze(
    reference_df, current_df,
    model=model,                    # optional, for Delta-SHAP
    feature_cols=["source", "derived", "noise"],
    target_col="target",            # optional
    time_col="timestamp",           # optional, enables temporal orientation
)
```

**Confidence Scoring (0.0–1.0):**

| Classification | Confidence Formula |
|---|---|
| Root cause with descendants | min(0.5 + 0.3 × drift_strength + 0.2 × attribution_strength, 1.0) |
| Isolated root cause (no descendants) | min(0.4 + 0.3 × drift_strength + 0.2 × attribution_strength, 1.0) |
| Symptom | min(0.5 + 0.2 × min(n_ancestors, 3) / 3.0 + 0.2 × drift_strength, 1.0) |
| Not classified | 0.0 |

**Remediation Prescriptions:**
- Root causes: `"ROOT CAUSE 'source' (PSI=2.234, confidence=0.85): investigate upstream sensor/source feed for 'source' (attributed prediction shift +0.4521); check calibration, sensor drift, or upstream ETL change."`
- Symptoms: `"SYMPTOM 'derived' (confidence=0.70): downstream effect of drifted upstream cause(s) [source]; do not remediate 'derived' directly — fix the upstream root cause(s)."`
- No drift: `"No material drift detected; no remediation required."`
- No root cause isolated: `"Drift detected but no clear upstream root cause isolated; inspect all drifted features for shared upstream source changes."`

### 6.3 Studio 11 Endpoint (`POST /s11/root-cause`)

Accepts `RootCauseReq`. Splits the active dataframe into reference (first 50%) and current (last 50%), runs the analyzer, returns `root_cause_report`.

**Requires:** At least 20 numeric rows.

---

## 7. Phase 6 — C7: Closed-Loop Digital Twin MPC Controller

### 7.1 Overview

The C7 controller (`src/omniforge/simulation/closed_loop.py`) implements a constrained receding-horizon Model Predictive Controller for a linear state-space Digital Twin:

```
u*_{t:t+H} = argmin  Σ ||H x̂_{t+k|t} - r_{t+k}||_Q² + ||u_{t+k}||_R²
subject to u_min ≤ u ≤ u_max
           x_min ≤ x̂ ≤ x_max
           x̂_{t+k+1|t} = A x̂_{t+k|t} + B u_{t+k}
```

Solved with `scipy.optimize.minimize` (SLSQP). Deterministic, offline, air-gapped safe.

### 7.2 Core Classes

#### `ClosedLoopMPCController`

**Constructor:**
```python
ctrl = ClosedLoopMPCController(
    twin: DigitalTwinEngine,
    horizon=10,
    Q=None,           # defaults to eye(n_x)
    R=None,           # defaults to eye(n_u) [FIXED: was n_x]
    u_min=None,       # defaults to -inf
    u_max=None,       # defaults to +inf
    x_min=None,       # defaults to -inf
    x_max=None,       # defaults to +inf
    random_seed=42,
)
```

**`solve(x0, r_path=None, *, warm_start=None, slack_penalty=1e4) → MPCSolveResult`:**

Three-stage solving strategy:
1. **Stage 1 — Hard constraints:** SLSQP with original bounds, no slack
2. **Stage 2 — Constraint softening:** If Stage 1 fails, retry with slack variables (bounded ≥ 0, penalized at 1e⁴ weight in objective)
3. **Stage 3 — Bound relaxation:** If Stage 2 fails, progressively relax bounds up to 1.2× (was 1.5×)

**`MPCSolveResult` fields:**
| Field | Type | Description |
|---|---|---|
| `optimal_control` | ndarray | First control action u_t (receding-horizon apply) |
| `full_control_sequence` | ndarray | All H control steps u_{t:t+H} |
| `state_trajectory` | ndarray | x̂_{t:t+H+1} under optimal control |
| `cost` | float | Objective cost |
| `success` | bool | Solver convergence |
| `solver_message` | str | SLSQP message |
| `constraint_violations` | list[str] | Any bound violations vs original bounds |
| `solve_time_ms` | float | Execution time in milliseconds |
| `slack_used` | float | Max slack value (0 if no softening needed) |

**`run_closed_loop(measurements, setpoint=None, *, state=None, state_key=None) → TwinControlReport`:**

Closed-loop simulation: at each step, solve MPC from current twin state, apply first control, assimilate measurement via Kalman filter, record trajectory. When `state` and `state_key` are supplied, checkpoints twin state + warm-start after every step.

**`persist(*, warm_start=None) → dict`:** Checkpoint twin state (x, P), horizon, bounds, warm-start into serializable dict.

**`restore(twin, payload) → (controller, warm_start)`:** Restore controller from persist payload, re-bound to a (possibly new) twin.

#### `build_default_twin(n_x=2, n_u=1, dt=1.0, random_seed=42)`

Constructs a default stable second-order linear Digital Twin (discretized double integrator):
```
x_{k+1} = [1 dt; 0 1] x_k + [0.5 dt²; dt] u_k + w
y_k     = [1 0] x_k + v
```

### 7.3 C7 Enhancements (v4.0.0)

| Enhancement | Before | After |
|---|---|---|
| Constraint softening | Hard bounds only; solver fails on infeasible QP | Slack variables allow near-feasible solutions |
| Solve time tracking | Not reported | `solve_time_ms` on every result |
| Slack usage reporting | Not reported | `slack_used` reports max slack value |
| R matrix dimension | `np.eye(n_x)` — wrong for control inputs | `np.eye(n_u)` — correctly dimensioned |
| Warm-start alignment | `np.roll` — caused misalignment | Slice-and-concatenate — correct alignment |
| Setpoint padding | Confusing slice-then-pad | Clean `np.tile` logic |
| Bound relaxation | Up to 1.5× | Limited to 1.2× (safer) |
| Dead parameter | `controls_override` accepted but unused | Removed |
| Dead code | Unused RNG in root_cause | Removed |

### 7.4 Studio 09 Endpoint (`POST /s09/twin-control`)

Accepts `TwinControlReq`. Builds or reuses a cached Digital Twin, runs closed-loop MPC, returns trajectory chart data suitable for the interactive Studio 09 panel.

---

## 8. Phase 6 — C4: Air-Gapped Deterministic Copilot Tools

### 8.1 Overview

The C4 Copilot (`src/omniforge/copilot/tools.py`) extends `CopilotToolRegistry` with 10 new deterministic offline tools. All tools execute without network access or third-party LLM dependency. The `OmniForgeCopilotAgent._chat_local_planner()` maps English natural-language keywords to these tools.

### 8.2 All Copilot Tools (24 total)

**Pre-Phase 6 Tools (14):**
`ingest_data`, `generate_demo_data`, `profile_quality`, `clean_data`, `run_eda`, `train_automl`, `run_forecast`, `explain_model`, `audit_fairness`, `check_drift`, `run_causal`, `generate_pipeline_code`, `list_domains`, `workspace_status`

**Phase 6 C4 Tools (10):**

| Tool | Category | Description |
|---|---|---|
| `root_cause_diagnosis` | C5 | Causal drift root-cause: drift → Delta-attribution → DAG ancestors → remediation |
| `twin_control` | C7 | Closed-loop Digital Twin MPC control with optional session persistence |
| `split_lock` | Data Integrity | Leakage-safe Split Lock train/test integrity (LeakageGuard) |
| `pii_audit` | C8 Security | Zero-trust PII audit: detect + pseudonymize or block (fail-closed) |
| `pareto_automl` | C3 Green AI | Green-AI Pareto AutoML with carbon / inference-latency budgets |
| `domain_validate` | Domains | Validate active dataset against an industry domain plugin (18 domains) |
| `register_model` | MLOps | Register active model to the MLOps Model Registry (Studio 11) |
| `promote_model` | MLOps | Promote a registered model to a new stage (Studio 11) |
| `list_studios` | Workbench | List all 15 OmniForge Workbench studios and routes |
| `studio_overview` | Workbench | Get overview payload for a specific studio |

### 8.3 C4 Tool Details

#### `twin_control(session_key=...)`

```python
result = registry.invoke("twin_control", {
    "n_x": 2, "n_u": 1, "horizon": 10, "n_steps": 8, "dt": 1.0,
    "setpoint": [1.0, 0.0],
    "session_key": "my_session_1"
})
```

**Session Persistence:** When `session_key` is supplied, twin state + controller warm-start are checkpointed into `memory.working_state["twin_sessions"][session_key]`. Restorable via `ClosedLoopMPCController.restore()`.

#### `register_model(model_name=..., stage=...)`

```python
result = registry.invoke("register_model", {
    "model_name": "my_champion",
    "stage": "Staging"
})
```

**Lookup Chain:** `state["models"][model_name]` → `state["models"]["pareto_champion"]` → `state["models"]["automl_champion"]`

**Invalid Stage:** Returns `{"error": "Invalid model stage 'X'.", "available_stages": ["none", "staging", "production", "archived"]}` instead of raising.

#### `promote_model(model_name, version, stage=...)`

```python
result = registry.invoke("promote_model", {
    "model_name": "my_champion",
    "version": 1,
    "stage": "Production"
})
```

**Existence Check:** Validates model exists in registry via `reg.list_versions()` before `transition_stage`. Returns structured error if not found.

#### `pii_audit(dataset_name=..., pii_policy=...)`

```python
result = registry.invoke("pii_audit", {
    "dataset_name": "my_dataset",
    "pii_policy": "pseudonymize"
})
```

**Policies:** `pseudonymize` (deterministic HMAC-SHA256 pseudonym), `block` (fail-closed SecurityError if PII detected), `off` (no-op, still reports).

---

## 9. Complete Algorithm, Statistical Test & Rule Selection Reference

### 9.1 Data Cleaning, Imputation & Outlier Rules

| Method / Rule | What It Does | When to Use | Key Caution |
|---|---|---|---|
| Median Imputation | Fills missing with column median | Best general default for skewed numeric | Ignores cross-column correlations; fit on Train only |
| Mean Imputation | Fills missing with arithmetic mean | Roughly symmetric Gaussian columns | Distorted by extreme tails; shrinks variance at >15% missing |
| KNN / MICE | Estimates missing from nearest rows / chained regressions | Moderate missingness (5–30%) with correlated columns | Slower on large wide datasets; requires Split Lock first |
| Forward-Fill / Interpolate | Carries last value forward / linear interpolation | Chronological time-series / IoT streams | Never use `bfill` for causal forecasting (look-ahead leakage) |
| IQR Fence (1.5× IQR) | Flags outside [Q₁ − 1.5·IQR, Q₃ + 1.5·IQR] | Skewed or non-Gaussian univariate | Prefer Clip/Winsorize over Remove unless confirmed corrupt |
| Modified Z-Score (MAD > 3.5) | Robust median/MAD deviation | Heavy-tailed data | Safe default when outliers occur in clusters |
| Z-Score (‖z‖ > 3) | Standard deviations from mean | Approximately Gaussian only | Unreliable on skewed distributions |
| Isolation Forest | Random partition depth | Fast global tabular anomaly detection | May miss tiny local density pockets |
| LOF | Local reachability density | Anomalies in varying-density clusters | Sensitive to neighborhood size k |
| One-Class SVM | Smooth kernel envelope | Tight boundary around clean baseline | Slower on large n; sensitive to scaling |

### 9.2 Clustering Algorithms

| Algorithm | Paradigm | When to Use | Key Caution |
|---|---|---|---|
| K-Means / BIRCH | Centroid / Hierarchical tree | Fast, roughly spherical numeric clusters | Requires selecting k (use Silhouette sweep) |
| GMM | Probabilistic elliptical | Overlapping or elongated clusters | Sensitive when features ≫ samples |
| DBSCAN | Density-based + noise | Arbitrary non-convex shapes, noise isolation | Sensitive to sharp density variations |
| Agglomerative / Spectral | Hierarchical / Graph Laplacian | Nested hierarchies or non-convex rings (n < 5,000) | Quadratic scaling on large tables |
| Isolation Forest | Random partition | Fast general-purpose anomaly detector | May miss tiny local density pockets |
| LOF | Local reachability | Anomalies hiding in varying-density clusters | Sensitive to neighborhood size k |
| One-Class SVM | Kernel envelope | Tight boundary around clean baseline | Slower on large n |

### 9.3 All 17 Statistical & Hypothesis Tests

(See Section 5 of the original guide for the full table — reproduced in Studio 04 section above.)

**Multiple Testing Corrections:** Benjamini-Hochberg (FDR), Bonferroni, Holm. Always pair with effect size.

### 9.4 Predictive Models Reference

(See Studio 06 section above for the complete model family table.)

---

## 10. Core Mathematical Formulas & Universal Threshold Compendium

### 10.1 Core Analytical Formulas

| Formula | Studio | Equation |
|---|---|---|
| Modified Z-Score (MAD) | S01 | Mᵢ = 0.6745 · (xᵢ − median(x)) / MAD(x) |
| Silhouette Coefficient | S02 | s(i) = (b(i) − a(i)) / max(a(i), b(i)) ∈ [−1, 1] |
| Calinski-Harabasz | S02 | CH = tr(Bₖ)/(k−1) / tr(Wₖ)/(n−k) |
| Davies-Bouldin | S02 | DB = (1/k) Σᵢ maxⱼ≠ᵢ (σᵢ + σⱼ) / d(cᵢ, cⱼ) |
| Isolation Forest Score | S02 | s(x, n) = 2^(−E[h(x)] / c(n)) |
| Apriori Lift | S02 | Lift(A ⇒ B) = P(A ∩ B) / (P(A)P(B)) |
| Subgroup WRAcc | S02 | WRAcc(S) = \|S\|/N · (p_S − p₀) |
| Variance Inflation Factor | S03 | VIFⱼ = 1 / (1 − Rⱼ²) |
| Historical VaR / CVaR | S03 | VaR_α = −Q_α(r); CVaR_α = −E[r \| r ≤ −VaR_α] |
| Benjamini-Hochberg FDR | S03/04 | Reject H₍₁₎…H₍ₖ*₎ for largest k with p₍ₖ₎ ≤ (k/m)α |
| Welch's t-test | S04 | t = (x̄₁ − x̄₂) / √(s₁²/n₁ + s₂²/n₂) |
| Cohen's d | S04 | d = (x̄₁ − x̄₂) / s_pooled |
| Yeo-Johnson & Mutual Information | S05 | I(X;Y) = ∬ p(x,y) log(p(x,y)/(p(x)p(y))) dx dy |
| Penalized B-Spline GAM | S06 | ŷ(x) = β₀ + Σⱼ Σₘ wⱼₘ Bⱼₘ(xⱼ) + λ\|w\|₂² |
| Nadaraya-Watson Kernel | S06 | ŷ(x) = Σ exp(−\|x−xᵢ\|²/(2h²))·yᵢ / Σ exp(−\|x−xᵢ\|²/(2h²)) |
| RBF Kernel Ridge (RKHS) | S06 | ŷ(x) = Σ αᵢ exp(−γ\|x−xᵢ\|²), α = (K + λI)⁻¹y |
| SHAP | S07 | φⱼ = Σ_{S⊆F\{j\}} |S|!(|F|−|S|−1)! / |F|! · [f(S∪{j}) − f(S)] |
| Disparate Impact | S07 | DI = P(Ŷ=1 \| A=unprivileged) / P(Ŷ=1 \| A=privileged) |
| AIPW Causal ATE | S08 | τ̂_AIPW = (1/n) Σ [μ̂₁(Xᵢ) − μ̂₀(Xᵢ) + Tᵢ(Yᵢ−μ̂₁(Xᵢ))/ê(Xᵢ) − (1−Tᵢ)(Yᵢ−μ̂₀(Xᵢ))/(1−ê(Xᵢ))] |
| Kaplan-Meier | S09 | Ŝ(t) = ∏_{tᵢ≤t} (1 − dᵢ/nᵢ) |
| Cox PH | S09 | h(t\|x) = h₀(t)·exp(βᵀx) |
| Population Stability Index | S11 | PSI = Σ (P_b − Q_b)·ln(P_b / Q_b) |
| Antithetic Root-Cause Attribution | S14 | φ_j = ½ [(f(x_ref[j←x_live,j]) − f(x_ref)) + (f(x_live) − f(x_live[j←x_ref,j]))] |

### 10.2 Universal Recommended Numeric Threshold Compendium

| Metric | Green / Healthy | Caution / Review | Critical / Action |
|---|---|---|---|
| Missing Rate per Column | < 5% | 5%–30% (KNN/MICE + indicator) | > 40–50% (drop unless domain-mandatory) |
| Skewness \|γ₁\| | < 0.5 (symmetric) | 0.5–1.0 (moderate) | > 1.0 (Yeo-Johnson / non-parametric) |
| Outlier Fences (IQR / MAD / Z) | Within bounds | 1.5×IQR / MAD > 3.5 / \|Z\| > 3.0 | 3.0×IQR / \|Z\| > 4.0 (extreme) |
| Silhouette Score | ≥ 0.50 (strong) | 0.25–0.50 (moderate) | < 0.25 (weak) |
| Multicollinearity (VIF) | < 5 | 5–10 (regularize) | > 10 (severe — prune/PCA) |
| Significance (α) | 0.05 (standard) | 0.01 (confirmatory) | 0.001 (safety-critical) |
| Effect Size (Cohen's d / \|r\|) | d ≥ 0.8 / \|r\| ≥ 0.5 | d = 0.5 / \|r\| = 0.3 | d ≤ 0.2 / \|r\| ≤ 0.1 |
| Apriori Lift | ≥ 1.5 (strong) | 1.1–1.5 (modest) | ≤ 1.0 (independent) |
| Disparate Impact | 0.80–1.25 (EEOC compliant) | 0.70–0.80 (borderline) | < 0.80 (adverse impact) |
| Covariate Drift (PSI) | < 0.10 (stable) | 0.10–0.25 (investigate) | ≥ 0.25 (major — retrain) |
| SPC Residual Z-Score (S14) | \|Z\| < 2.0 (HEALTHY) | 2.0 ≤ \|Z\| < 3.0 (WARNING) | \|Z\| ≥ 3.0 (CRITICAL_ALARM) |
| Survival C-Index (S09) | ≥ 0.75 (strong) | 0.65–0.75 (useful) | ≤ 0.55 (near random) |

---

## 11. Python SDK Reference — Core Patterns

Built-in domain YAML and the SDK typing marker are included in the wheel and source distribution. `BaseDomainPlugin.load_config()` uses `configs/domains/<slug>.yaml` in the working directory or the supplied override directory first. `ConfigLoader` looks in the `domains` subdirectory of its configuration directory. Both use the packaged default when there is no local override.

### 11.1 Version Check
```python
import omniforge
omniforge.__version__                    # "4.0.0"
omniforge.get_version_info()             # full metadata
```

### 11.2 Domain Plugins (18 available)
```python
from omniforge.domains import default_domain_registry
finance = default_domain_registry.get("finance")
df = finance.engineer_domain_features(finance.generate_sample_dataset(n_samples=300))
# Available: finance, energy, healthcare, manufacturing, retail, transportation,
# agriculture, climate, telecom, hr, education, real_estate, sports,
# cybersecurity, government, bioinformatics, materials, media
```

### 11.3 Data Quality & Cleaning
```python
from omniforge.data import DataQualityProfiler, DataCleaner, CleaningConfig

quality = DataQualityProfiler().profile(df)  # 6-dimensional score
cleaned, report = DataCleaner(CleaningConfig(
    numeric_imputation="median",
    outlier_method="iqr",
    outlier_action="clip"
)).fit_transform(df)
```

### 11.4 AutoML
```python
from omniforge.automl import AutoMLPipelineSearch
X = df.drop(columns=["target"])
y = df["target"]
automl = AutoMLPipelineSearch(
    time_budget_seconds=30,
    tune_trials=4,
    include_deep=True
).fit(X, y)
print(automl.best_model_name, automl.best_score, automl.lift_over_best_baseline)
```

### 11.5 C5 Root-Cause Analysis
```python
from omniforge.mlops import DriftRootCauseAnalyzer
report = DriftRootCauseAnalyzer().analyze(
    ref_df, cur_df,
    model=model,
    feature_cols=["source", "derived", "noise"],
    time_col="timestamp"
)
for rc in report.root_causes:
    print(f"ROOT CAUSE: {rc}")
for sym in report.symptoms:
    print(f"SYMPTOM: {sym}")
for rx in report.remediation_prescriptions:
    print(rx)
```

### 11.6 C7 Twin MPC
```python
from omniforge.simulation import ClosedLoopMPCController, build_default_twin
import numpy as np

twin = build_default_twin(n_x=2, n_u=1, dt=1.0)
ctrl = ClosedLoopMPCController(
    twin, horizon=10,
    u_min=np.array([-1.0]), u_max=np.array([1.0]),
    x_min=np.array([-10.0, -10.0]), x_max=np.array([10.0, 10.0])
)
measurements = [np.zeros(2) for _ in range(8)]
report = ctrl.run_closed_loop(measurements, setpoint=np.array([1.0, 0.0]))

# Persistence across sessions
snapshot = ctrl.persist(warm_start=None)
twin2 = build_default_twin(n_x=2, n_u=1, dt=1.0)
ctrl2, warm_start = ClosedLoopMPCController.restore(twin2, snapshot)
```

### 11.7 C4 Copilot Agent
```python
from omniforge.copilot import OmniForgeCopilotAgent
copilot = OmniForgeCopilotAgent(provider="local")
copilot.load_dataframe(df, name="active_dataset")
response = copilot.chat("profile data quality")
print(response.reply)
print(response.tools_executed)
```

### 11.8 Reporting
```python
from omniforge.reporting import ReportGenerator
reporter = ReportGenerator()
payload = reporter.build_report(
    title="Executive Report",
    domain="finance",
    quality_report=quality,
    model_summary=summary,
)
reporter.export(payload, "report.html")   # or .pdf, .pptx, .docx, .xlsx, .ipynb
```

---

## 12. REST API Reference

### 12.1 Base URLs
| Service | Port | Base |
|---|---|---|
| FastAPI REST API | 8000 | `http://localhost:8000` |
| Enterprise Workbench (API + SPA) | 8501 | `http://localhost:8501` |
| Swagger Docs | 8501 | `http://localhost:8501/docs` |

### 12.2 Studio Endpoints (Workbench)

| Endpoint | Method | Studio | Description |
|---|---|---|---|
| `/s00/ingest` | POST | S00 | Ingest dataset |
| `/s01/clean` | POST | S01 | Clean data |
| `/s02/mine` | POST | S02 | Clustering & mining |
| `/s03/eda` | POST | S03 | Automated EDA |
| `/s04/diagnostics` | POST | S04 | Diagnostics & baselines |
| `/s05/features` | POST | S05 | Feature engineering |
| `/s06/run` | POST | S06 | AutoML |
| `/s07/advanced` | POST | S07 | Responsible AI |
| `/s08/causal` | POST | S08 | Causal inference |
| `/s09/twin-control` | POST | S09 | **C7 Twin MPC** |
| `/s10/dag` | POST | S10 | DAG pipeline |
| `/s11/root-cause` | POST | S11 | **C5 Root-cause** |
| `/s11/drift` | POST | S11 | Drift detection |
| `/s12/report` | POST | S12 | Executive reporting |
| `/s13/chat` | POST | S13 | **C4 Copilot chat** |
| `/s14/models` | GET | S14 | Supervisory models |
| `/s14/root-cause` | POST | S14 | **C5 Root-cause (supervisory)** |
| `/s14/equations` | GET | S14 | Champion vs baseline equations |

### 12.3 Copilot Tool Execution

Endpoint: `POST /s13/chat` with `CopilotReq{message: string}` — routes English keywords to deterministic tools via the local planner.

Direct tool execution is also available through the `CopilotToolRegistry.invoke(tool_name, arguments)` Python API.

---

## 13. Copilot Agent & Tool Reference

### 13.1 `OmniForgeCopilotAgent`

```python
agent = OmniForgeCopilotAgent(
    provider="local",           # "local", "openai", "anthropic", "azure", "ollama"
    model="gpt-4o-mini",
    temperature=0.1,
    memory=CopilotMemory(),
)
response = agent.chat("train automl on my dataset with target=churn")
# response.reply: str
# response.provider_used: str
# response.tools_executed: list[dict]
# response.suggested_next_steps: list[str]
```

### 13.2 `CopilotToolRegistry`

```python
from omniforge.copilot import CopilotToolRegistry, CopilotMemory
registry = CopilotToolRegistry(CopilotMemory())

# List all tools
tools = registry.list_tools()  # OpenAI/Anthropic-compatible specs

# Invoke any tool
result = registry.invoke("root_cause_diagnosis", {
    "reference_dataset": "ref",
    "current_dataset": "cur"
})

# Tool names
registry.invoke("ingest_data", {"source": "data.csv", "dataset_name": "ds"})
registry.invoke("profile_quality", {"dataset_name": "ds"})
registry.invoke("clean_data", {"dataset_name": "ds"})
registry.invoke("run_eda", {"dataset_name": "ds", "target_column": "target"})
registry.invoke("train_automl", {"target_column": "target", "time_budget_seconds": 30})
registry.invoke("run_forecast", {"target_column": "y", "horizon": 14})
registry.invoke("explain_model", {"model_name": None, "target_column": "target"})
registry.invoke("audit_fairness", {"sensitive_column": "group", "target_column": "target"})
registry.invoke("check_drift", {"reference_dataset": "ref", "current_dataset": "cur"})
registry.invoke("run_causal", {"treatment_col": "treatment", "outcome_col": "outcome", "method": "aipw"})
registry.invoke("generate_pipeline_code", {"target_column": "target"})
registry.invoke("list_domains")
registry.invoke("workspace_status")
registry.invoke("split_lock", {"dataset_name": "ds", "test_size": 0.2})
registry.invoke("pii_audit", {"dataset_name": "ds", "pii_policy": "pseudonymize"})
registry.invoke("pareto_automl", {"target_column": "target", "time_budget_seconds": 30})
registry.invoke("domain_validate", {"domain": "finance", "dataset_name": "ds"})
registry.invoke("register_model", {"model_name": "champion", "stage": "Staging"})
registry.invoke("promote_model", {"model_name": "champion", "version": 1, "stage": "Production"})
registry.invoke("list_studios")
registry.invoke("studio_overview", {"studio_id": "s11"})
registry.invoke("root_cause_diagnosis", {"reference_dataset": "ref", "current_dataset": "cur"})
registry.invoke("twin_control", {"n_x": 2, "n_u": 1, "horizon": 10, "n_steps": 8, "dt": 1.0})
```

### 13.3 Local Planner Keyword Routing

| Keywords in Message | Tool Invoked |
|---|---|
| "domain", "industry", "plugin" | `list_domains` |
| "pipeline", "code", "export dag" | `generate_pipeline_code` |
| "forecast", "time series" | `run_forecast` |
| "causal", "treatment", "ate", "uplift" | `run_causal` |
| "clean", "impute", "outlier", "dedup" | `clean_data` |
| "quality", "scorecard" | `profile_quality` |
| "eda", "explor", "profile", "insight" | `run_eda` |
| "fairness", "bias", "disparate" | `audit_fairness` |
| "explain", "shap", "interpret", "why" | `explain_model` |
| "train", "automl", "model", "predict" | `train_automl` |
| "root cause", "root-cause", "drift attribution", "causal drift" | `root_cause_diagnosis` |
| "twin control", "mpc", "closed-loop", "digital twin control" | `twin_control` |
| "split lock", "split_lock", "leakage guard" | `split_lock` |
| "pii audit", "pii_audit", "pseudonymize", "pii guard" | `pii_audit` |
| "pareto", "green-ai", "carbon budget", "latency budget" | `pareto_automl` |
| "domain validate", "industry domain", "compliance" | `domain_validate` |
| "register model", "model registry", "promote model" | `register_model` / `promote_model` |
| "list studios", "studios", "studio overview" | `list_studios` / `studio_overview` |

---

## 14. Security, PII, Audit & Encryption Guide

### 14.1 DataFramePIIGuard (`src/omniforge/core/security.py`)

**Policies:**
| Policy | Behavior |
|---|---|
| `pseudonymize` | Every PII-bearing cell → deterministic HMAC-SHA256 pseudonym (tenant-bound) |
| `block` | Fail-closed: raises `SecurityError` if any PII cell exists |
| `off` | No-op; PII still reported but not altered |

**Detection Patterns:** SSN, email, phone, credit card, IPv4 (regex-based over string columns).

**v4.0.0 Fix:** `detect()` now uses `pd.api.types.is_string_dtype(series)` instead of `dtype != object`, correctly catching pandas `StringDtype` columns (pandas 3.x default).

**Audit:** `_audit()` now correctly passes `tenant_id=self.tenant_id` to `log_event()` (was passing `resource_id`, causing a silently swallowed `TypeError`).

```python
from omniforge.core.security import DataFramePIIGuard

guard = DataFramePIIGuard(tenant_id="my-tenant", pii_policy="pseudonymize")
redacted, report = guard.transform(df)
# report: {"pii_detected": bool, "pii_columns": [...], "pii_cells_total": int, "redacted_columns": {...}}
```

### 14.2 Encryption

- **AES-256-GCM** authenticated encryption for all new payloads
- **Fernet** backward-compatible decryption for legacy tokens
- Key resolution: `OMNIFORGE_ENCRYPTION_KEY` env → `OMNIFORGE_SECURITY__ENCRYPTION_KEY` → local 0600 keyfile
- Production requires high-entropy (≥ 32 bytes) non-placeholder secret

### 14.3 JWT Authentication

- HS256 JWT with PyJWT or stdlib HMAC-SHA256 fallback
- Secret resolution: `OMNIFORGE_JWT_SECRET_KEY` → `OMNIFORGE_SECURITY__JWT_SECRET_KEY`
- Claims: sub, roles, tenant_id, attributes, scopes, iat, exp

### 14.4 RBAC/ABAC

**Roles:** `admin` (tenant-scoped), `data_scientist`, `mlops_engineer`, `analyst`, `viewer`, `platform_admin` (cross-tenant, P1-12).

**Migration:** Tenant `admin` is now strictly tenant-scoped. Cross-tenant access requires `platform_admin`. Every cross-tenant allow/deny is audited (fail-closed).

### 14.5 Audit Trail

- Hash-chained, HMAC-signed JSONL audit log
- Every record: event_hash + chain_mac (both required for verification)
- Optional out-of-band checkpoint anchor (RFC 9162-style) via `OMNIFORGE_AUDIT__ANCHOR_PATH`
- `AuditLogger.verify_chain()` / `verify_anchor()` / `verify_with_anchor()`

---

## 15. Logging & Observability

### 15.1 Structured Logging (`src/omniforge/core/logging.py`)

**v4.0.0 Fix:** `structlog` pre-declared as `ModuleType | None` with alias import, resolving mypy type errors.

**Features:**
- Automatic secret redaction (bearer tokens, sk-*, ghp_*, SSN, credit cards)
- Trace ID and tenant ID context propagation
- JSON or console output
- structlog integration when available, stdlib fallback

```python
from omniforge.core.logging import get_logger, set_trace_id, set_tenant_context
logger = get_logger(__name__)
set_trace_id("trace-123")
set_tenant_context("tenant-456")
logger.info("processing_started", rows=1000, columns=25)
```

### 15.2 Carbon Tracking

Green AI Carbon Tracker reports `emissions_g_co2eq` for every AutoML and forecasting run.

---

## 16. Troubleshooting & FAQ

### Q: Split Lock — must it be enabled before cleaning?
**A:** YES for predictive modeling. If imputation, outlier fences, or scaling are computed on the full dataset before splitting, test statistics leak into training. For pure BI cleaning (formatting only), it's optional.

### Q: PII detect() misses StringDtype columns?
**A:** Fixed in v4.0.0 — now uses `pd.api.types.is_string_dtype()` which catches pandas `StringDtype` (the pandas 3.x default).

### Q: MPC solver fails on tight bounds?
**A:** C7 constraint softening with slack variables provides near-feasible solutions. If still infeasible, bounded relaxation up to 1.2× is attempted. Check `slack_used` and `constraint_violations` in the result.

### Q: Twin state lost between Copilot calls?
**A:** Use `session_key` with `twin_control`. State is checkpointed via `ClosedLoopMPCController.persist()` and restorable via `restore()`.

### Q: Invalid model stage in register_model/promote_model?
**A:** Returns structured error `{"error": ..., "available_stages": [...]}` instead of raising. `promote_model` also validates model existence in the registry first.

### Q: Version mismatch after upgrade?
**A:** v4.0.0 is defined in `pyproject.toml` and `src/omniforge/version.py`. Run `omniforge check` to verify. Import `omniforge` and check `__version__` at runtime.

### Q: Can OmniForge run fully offline?
**A:** Yes. The deterministic Copilot planner, C5 root-cause engine, C7 MPC controller, and all core analytics run with zero third-party LLM or network dependency. Optional LLM providers (OpenAI, Anthropic, Azure, Ollama) require network and API keys.

---

*End of User Guide. For architecture details, see [`docs/architecture.md`](docs/architecture.md). For audit anchor replication, see [`docs/AUDIT_ANCHOR_REPLICATION.md`](docs/AUDIT_ANCHOR_REPLICATION.md). For risk register, see [`docs/RISK_REGISTER.md`](docs/RISK_REGISTER.md).*
