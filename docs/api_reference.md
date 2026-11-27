# OmniForge v4.0.0 — Complete REST API Reference

**Base URL:** `http://localhost:8501` (Enterprise Workbench, includes FastAPI)  
**Swagger UI:** `http://localhost:8501/docs`  
**Healthcheck:** `GET /api/v1/health`  
**API Version:** `v2` | **Schema Version:** `2.0`

---

## Table of Contents

1. [Authentication & Middleware](#1-authentication--middleware)
2. [Studio Endpoints s00–s14](#2-studio-endpoints-s00-s14)
3. [Phase 6 Endpoints (C4, C5, C7)](#3-phase-6-endpoints-c4-c5-c7)
4. [Pydantic Request/Response Schemas](#4-pydantic-request-response-schemas)
5. [Error Handling](#5-error-handling)
6. [Workbench Session State](#6-workbench-session-state)

---

## 1. Authentication & Middleware

### 1.1 JWT Authentication

The API supports JWT-based authentication via the `Authorization: Bearer <token>` header. Tokens are issued by `JWTManager` and contain claims: `sub`, `roles`, `tenant_id`, `attributes`, `scopes`, `iat`, `exp`.

```bash
curl -X POST http://localhost:8501/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "operator", "password": "secret"}'
```

### 1.2 RBAC / Tenant Isolation

- Roles: `admin` (tenant-scoped), `data_scientist`, `mlops_engineer`, `analyst`, `viewer`, `platform_admin` (cross-tenant)
- Tenant ID passed via `X-Tenant-Id` header or derived from JWT
- Cross-tenant access requires `platform_admin` role and is audited (fail-closed)

### 1.3 Rate Limiting

Sliding-window rate limiter (default 300 requests/minute per client IP). Health probe paths (`/health`, `/api/v1/health`) are exempt.

---

## 2. Studio Endpoints s00–s14

Each studio follows the pattern: `POST /api/workbench/sXX/<action>` with an `X-Session-Id` header for session scoping.

### 2.1 Studio 00 — Ingestion (`s00`)

| Endpoint | Method | Description |
|---|---|---|
| `/s00/ingest` | POST | Load dataset from CSV, Excel, Parquet, JSON, SQL, etc. |

**Request:** Multipart form or JSON with `source` (file path / URL / connection string), `dataset_name`, `source_type`.

**Response:** `{dataset_name, n_rows, n_columns, column_names, checksum_sha256, intake_grade}`

---

### 2.2 Studio 01 — Cleaning (`s01`)

| Endpoint | Method | Description |
|---|---|---|
| `/s01/clean` | POST | Leakage-safe cleaning with 11-stage symmetric pipeline |

**Request Body (`CleanReq`):**
```json
{
  "mode": "custom",
  "impute_numeric": "median",
  "impute_categorical": "most_frequent",
  "drop_duplicates": true,
  "handle_outliers": "clip",
  "outlier_method": "iqr",
  "outlier_threshold": 1.5,
  "drop_high_missing_cols": true,
  "missing_col_threshold": 0.90,
  "preview_only": false
}
```

**Response:** Cleaned data summary, deleted records audit, quality report.

---

### 2.3 Studio 02 — Mining (`s02`)

| Endpoint | Method | Description |
|---|---|---|
| `/s02/mine` | POST | Clustering, anomaly screening, association rules |

**Request Body (`ClusterReq`):** `algorithm`, `n_clusters`, `n_steps`, `preview_only`, etc.

**Response:** Cluster assignments, anomaly flags, association rules, silhouette scores.

---

### 2.4 Studio 03 — Universal Diagnostics (`s03`)

| Endpoint | Method | Description |
|---|---|---|
| `/s03/catalog` | GET | Module descriptions, including `executable_modules` for Studio 03 (not downstream suggestions) |
| `/s03/run` | POST | Run diagnostics with `mode: all` (or `complete`), `auto`, or `custom` |

Custom mode requires a nonempty `selected_modules` array of unique executable IDs.
Other modes must omit the list or send an empty list. Invalid selections or modes return 422.
The response includes `report.executed_modules`, `report.module_status`, the hypothesis ledger,
and any available baseline or risk evidence. Modules without evaluable results are marked `unavailable`.
Risk uses `structural_risk_index`, `risk_band`, and `tail_risk_table`.

---

### 2.5 Studio 04 — Diagnostics (`s04`)

| Endpoint | Method | Description |
|---|---|---|
| `/s04/diagnostics` | POST | Universal diagnostics, baseline equations, risk metrics |

**Request:** `alpha` (significance level), `holdout_ratio`, `mode` (complete/auto/custom).

**Response:** Normality tests, correlation matrix, VIF, VaR/CVaR, ranked baseline equations with LaTeX/Python.

---

### 2.6 Studio 05 — Feature Engineering (`s05`)

| Endpoint | Method | Description |
|---|---|---|
| `/s05/features` | POST | Feature creation & selection |

**Request:** `engineering_steps` (skew correction, interactions, polynomials, datetime), `selector_method` (variance, mutual_info, f_test, lasso, tree, rfe, consensus).

**Response:** Engineered feature names, selection scores, transformed data preview.

---

### 2.7 Studio 06 — AutoML (`s06`)

| Endpoint | Method | Description |
|---|---|---|
| `/s06/run` | POST | Multi-backend AutoML training |

**Request:** `target_column`, `time_budget_seconds`, `tune_trials`, `include_deep`.

**Response:** `best_model_name`, `best_score`, `primary_metric`, `lift_over_best_baseline`, `carbon_g_co2eq`, `leaderboard_top5`.

**Formula Export:** `GET /s06/export-formula` — Standalone Python, Excel formula, SQL expression, MATLAB/C/PLC, LaTeX.

---

### 2.8 Studio 07 — Responsible AI (`s07`)

| Endpoint | Method | Description |
|---|---|---|
| `/s07/explain` | POST | SHAP, LIME, permutation importance |
| `/s07/pdp` | POST | Partial dependence plots |
| `/s07/fairness` | POST | Disparate impact, equalized odds |
| `/s07/counterfactual` | POST | Wachter/DiCE counterfactuals |
| `/s07/model-card` | POST | ISO/NIST Model Card generation |

**Fairness Threshold:** Disparate Impact in [0.80, 1.25] is EEOC 4/5ths compliant.

---

### 2.9 Studio 08 — Causal & Forecasting (`s08`)

| Endpoint | Method | Description |
|---|---|---|
| `/s08/forecast` | POST | Time-series forecasting tournament |
| `/s08/causal` | POST | Causal ATE (IPW/AIPW/PSM/DML/IV/DiD) |
| `/s08/optimize` | POST | LP/MILP/Markowitz/CVRP optimization |

**Causal Methods:** `ipw`, `psm`, `aipw`, `dml`, `iv_2sls`, `did`, `cate`.

**Response:** ATE estimate, confidence interval, p-value, refutation checks, sensitivity bounds.

---

### 2.10 Studio 09 — Specialized AI + Twin MPC (`s09`)

| Endpoint | Method | Description |
|---|---|---|
| `/s09/nlp` | POST | NLP: zero-shot, QA, topics, NER |
| `/s09/vision` | POST | Computer vision (optional deep backends) |
| `/s09/graph` | POST | Graph analytics & link prediction |
| `/s09/recsys` | POST | Hybrid recommender systems |
| `/s09/survival` | POST | Kaplan-Meier, Cox PH |
| `/s09/recommend` | POST | Algorithm recommendations |
| `/s09/twin-control` | POST | **C7 Closed-Loop Digital Twin MPC** |

#### `POST /s09/twin-control` (C7, Phase 6)

**Request Schema (`TwinControlReq`):**

| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `n_x` | int | 2 | 1–16 | State dimension |
| `n_u` | int | 1 | 1–8 | Control dimension |
| `horizon` | int | 10 | 1–100 | MPC receding horizon H |
| `n_steps` | int | 8 | 1–500 | Closed-loop steps |
| `dt` | float | 1.0 | 0 < dt ≤ 3600 | Sampling period |
| `setpoint` | list[float] \| null | null | — | Flat setpoint path |
| `u_min` / `u_max` | list[float] \| null | null | — | Control input bounds |
| `x_min` / `x_max` | list[float] \| null | null | — | State safety bounds |
| `Q_diag` / `R_diag` | list[float] \| null | null | — | Cost diagonals |
| `measurements` | list[list[float]] \| null | null | — | Per-step measurements |

**Response:**
```json
{
  "twin_control": {
    "horizon": 10,
    "n_steps_applied": 8,
    "setpoint": [[1.0, 0.0], ...],
    "state_trajectory": [[0.0, 0.0], ...],
    "control_history": [[0.5], ...],
    "stage_costs": [0.25, ...],
    "total_cost": 2.5,
    "all_constraints_satisfied": true,
    "constraint_violations": [],
    "solved_at": "2026-11-22T..."
  },
  "twin_dimensions": {"n_x": 2, "n_u": 1},
  "horizon": 10,
  "n_steps_applied": 8,
  "all_constraints_satisfied": true,
  "constraint_violations": [],
  "total_cost": 2.5,
  "trajectory_chart": {
    "state_trajectory": [...],
    "control_history": [...],
    "setpoint": [...],
    "stage_costs": [...]
  }
}
```

---

### 2.11 Studio 10 — DAG Pipeline (`s10`)

| Endpoint | Method | Description |
|---|---|---|
| `/s10/dag` | POST | Visual DAG pipeline builder & executor |

**Response:** Mermaid DAG, Python code, execution results, caching info.

---

### 2.12 Studio 11 — MLOps (`s11`)

| Endpoint | Method | Description |
|---|---|---|
| `/s11/overview` | GET | Registered models list |
| `/s11/register` | POST | Register model version |
| `/s11/promote` | POST | Promote model stage |
| `/s11/rollback` | POST | Rollback production model |
| `/s11/drift` | POST | Drift detection (PSI/KS/Wasserstein) |
| `/s11/root-cause` | POST | **C5 Causal Drift Root-Cause** |

#### `POST /s11/root-cause` (C5, Phase 6)

**Request Schema (`RootCauseReq`):**

| Field | Type | Default | Description |
|---|---|---|---|
| `reference_dataset` | string | "reference_dataset" | Reference dataset name |
| `current_dataset` | string | "current_dataset" | Current dataset name |
| `target_col` | string \| null | null | Target column to exclude |
| `time_col` | string \| null | null | Time column for temporal orientation |
| `psi_warning` | float | 0.10 | PSI warning threshold (0–1) |
| `psi_critical` | float | 0.25 | PSI critical threshold (0–1) |
| `max_cond_set_size` | int | 2 | PC skeleton conditioning set size (0–4) |

**Response:** `{root_cause_report: {...}}` with overall_status, root_causes, symptoms, drifted_features, dag_edges, prediction_shift, total_attribution, remediation_prescriptions.

**Requires:** At least 20 numeric rows in the active dataframe.

---

### 2.13 Studio 12 — Reporting (`s12`)

| Endpoint | Method | Description |
|---|---|---|
| `/s12/report` | POST | Executive report generation & export |

**Export Formats:** PDF, XLSX, DOCX, PPTX, HTML, Markdown, CSV, JSON, LaTeX, IPYNB.

---

### 2.14 Studio 13 — AI Copilot (`s13`)

| Endpoint | Method | Description |
|---|---|---|
| `/s13/chat` | POST | **C4 Air-Gapped Deterministic Copilot** |

#### `POST /s13/chat` (C4, Phase 6)

**Request Schema (`CopilotReq`):**
```json
{
  "message": "profile data quality on my dataset"
}
```

**Response:**
```json
{
  "reply": "Six-dimensional Data Quality Score: 87/100 (Grade B)...",
  "provider_used": "local_deterministic_planner",
  "suggested_actions": ["Clean detected data issues", "Run AutoML"],
  "tool_calls": [{"tool": "profile_quality", "result": {...}}],
  "history": [...]
}
```

**Local Planner Keyword Routing:** See [`docs/user_guide.md`](docs/user_guide.md) Section 13.3 for the complete keyword→tool mapping.

---

### 2.15 Studio 14 — Supervisory Control (`s14`)

| Endpoint | Method | Description |
|---|---|---|
| `/s14/models` | GET | Available Champion/Baseline models |
| `/s14/evaluate` | POST | Live batch evaluation |
| `/s14/root-cause` | POST | **C5 Root-Cause (supervisory context)** |
| `/s14/equations` | GET | Champion vs Baseline governing equations |

#### `GET /s14/equations` (C5, Phase 6)

Returns the Active Champion Governing Equation and Reference Baseline Equation for signed root-cause deviation attribution.

**Response:**
```json
{
  "champion_equation": "ŷ = β₀ + Σ βⱼxⱼ (Ridge, alpha=1.0)",
  "baseline_equation": "ŷ = median(y_train)",
  "surrogate_fidelity_r2": 0.94,
  "attribution_formula": "φ_j = ½[(f(x_ref[j←x_live,j]) − f(x_ref)) + (f(x_live) − f(x_live[j←x_ref,j]))]"
}
```

---

## 3. Phase 6 Endpoints Summary (C4, C5, C7)

| Capability | Endpoint | Method | Studio | Schema |
|---|---|---|---|---|
| **C7 Twin MPC** | `/s09/twin-control` | POST | S09 | `TwinControlReq` |
| **C5 Root-Cause (MLOps)** | `/s11/root-cause` | POST | S11 | `RootCauseReq` |
| **C4 Copilot Chat** | `/s13/chat` | POST | S13 | `CopilotReq` |
| **C5 Root-Cause (Supervisory)** | `/s14/root-cause` | POST | S14 | `RootCauseReq` |
| **C5 Equations** | `/s14/equations` | GET | S14 | — |

---

## 4. Pydantic Request/Response Schemas

All schemas are defined in `src/omniforge/api/studios/schemas.py`.

### Key Schemas

```python
class TwinControlReq(BaseModel):
    n_x: int = Field(default=2, ge=1, le=16)
    n_u: int = Field(default=1, ge=1, le=8)
    horizon: int = Field(default=10, ge=1, le=100)
    n_steps: int = Field(default=8, ge=1, le=500)
    dt: float = Field(default=1.0, gt=0.0, le=3600.0)
    setpoint: list[float] | None = None
    u_min: list[float] | None = None
    u_max: list[float] | None = None
    x_min: list[float] | None = None
    x_max: list[float] | None = None
    Q_diag: list[float] | None = None
    R_diag: list[float] | None = None
    measurements: list[list[float]] | None = None

class RootCauseReq(BaseModel):
    reference_dataset: str = "reference_dataset"
    current_dataset: str = "current_dataset"
    target_col: str | None = None
    time_col: str | None = None
    psi_warning: float = Field(default=0.10, ge=0.0, le=1.0)
    psi_critical: float = Field(default=0.25, ge=0.0, le=1.0)
    max_cond_set_size: int = Field(default=2, ge=0, le=4)

class CopilotReq(BaseModel):
    message: str

class CopilotToolExecReq(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)

class CleanReq(BaseModel):
    mode: str = "custom"
    impute_numeric: str = "median"
    impute_categorical: str = "most_frequent"
    drop_duplicates: bool = True
    handle_outliers: str = "clip"
    outlier_method: str = "iqr"
    outlier_threshold: float = 1.5
    drop_high_missing_cols: bool = True
    missing_col_threshold: float = 0.90
    preview_only: bool = False
    # ... (11-stage cleaning parameters)

class SplitLockReq(BaseModel):
    strategy: str = "auto"
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    cv_folds: int = Field(default=5, ge=2, le=10)
    time_col: str | None = None
    group_col: str | None = None
    target_col: str | None = None
    gap: int = 0

class TargetsReq(BaseModel):
    targets: list[str]

class ExclusionsReq(BaseModel):
    excluded_columns: list[str]
    reason: str = "User excluded column from analysis"

class LoadSampleReq(BaseModel):
    source_type: str = "domain"
    name: str = "finance"
    n_samples: int = Field(default=600, ge=30, le=50000)

class SQLLoadReq(BaseModel):
    connection_url: str
    table_name: str | None = None
    query: str | None = None
    limit: int = Field(default=50000, ge=10, le=250000)

class AccessLoadReq(BaseModel):
    file_path: str
    table_name: str
    limit: int = Field(default=50000, ge=10, le=250000)
```

---

## 5. Error Handling

OmniForge uses a rich custom exception hierarchy (`src/omniforge/core/exceptions.py`):

| Exception | HTTP Status | Meaning |
|---|---|---|
| `OmniForgeError` | 500 | Base platform error |
| `ValidationError` | 400 | Invalid request parameters |
| `SecurityError` | 403 | Security policy violation (PII block, weak salt) |
| `AuthorizationError` | 403 | RBAC/ABAC authorization failure |
| `SimulationError` | 400 | MPC/simulation failure |
| `CopilotError` | 400 | Copilot tool execution failure |
| `ModelError` | 400 | Model training/inference failure |
| `ComplianceError` | 403 | GDPR/HIPAA/SOX compliance violation |

**Error Response Format:**
```json
{
  "error": "Human-readable error message",
  "error_type": "ExceptionClassName",
  "error_code": "specific_error_code",
  "suggestions": ["Actionable remediation steps"],
  "context": {"key": "value"},
  "trace_id": "uuid-trace-id"
}
```

**C4 Structured Errors (Phase 6):**
- `register_model` with invalid stage: `{"error": "Invalid model stage 'X'.", "available_stages": ["none", "staging", "production", "archived"]}`
- `promote_model` for non-existent model: `{"error": "Model 'X' not found in registry.", "available_models": [...]}`
- Copilot tool not found: `CopilotError` with `suggestions=["Available tools: [...]]"`

---

## 6. Workbench Session State

Each studio request includes an `X-Session-Id` header (default: `"default"`). Session state is managed by `omniforge.api.workbench` and persists:

- Active dataset (`uap_df`)
- Active model (`uap_active_model`)
- Active feature columns (`uap_active_feature_cols`)
- Selected targets (`uap_selected_targets`)
- Excluded columns
- Split lock configuration
- Studio results cache
- Chat history (Studio 13)
- Digital twin (`uap_digital_twin`, Studio 09)
- MPC controller (`uap_mpc_controller`, Studio 09)
- Twin control report (`uap_twin_control_report`, Studio 09)
- Copilot agent (`uap_copilot`, Studio 13)

**Utility:** `wb_cache_and_respond(state, sid, cache_key, body)` — typed wrapper for caching studio responses with lineage tracking.

---

## 7. Python SDK API Reference (Key Classes)

See [`docs/user_guide.md`](docs/user_guide.md) Section 11 for complete Python SDK patterns including:

- `omniforge.mlops.DriftRootCauseAnalyzer` (C5)
- `omniforge.simulation.ClosedLoopMPCController` (C7)
- `omniforge.copilot.OmniForgeCopilotAgent` (C4)
- `omniforge.copilot.CopilotToolRegistry` (C4)
- `omniforge.core.security.DataFramePIIGuard` (C8)
- `omniforge.evaluation.InterpretabilityEngine`
- `omniforge.automl.AutoMLPipelineSearch`
- `omniforge.reporting.ReportGenerator`

---

*End of API Reference. For installation and quickstart, see [`README.md`](README.md). For the complete user guide, see [`docs/user_guide.md`](docs/user_guide.md).*
### Run-level model inputs

`DiagnosticsReq`, `AutoMLRunReq`, `NonParametricRunReq`, and `DeepBenchReq` accept `ignored_columns`.
This filters predictor inputs, not the session dataset. Studio 03 still scans the entire analysis
scope, subject to its diagnostic caps. The list must not name an outcome. Call `/s05/select` with
`apply_to_run: true` on a locked training cohort to obtain a `run_scope` object to pass into one
modelling request. This cannot be combined with a session-wide `commit: true`.
