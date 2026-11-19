"""Enterprise FastAPI Workbench Router Powering All 13 OmniForge CRISP-DM Studios & Phase 2 AI Engines."""

from __future__ import annotations

import dataclasses
import io
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

import networkx as nx
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from omniforge.api.auth import require_workbench_auth
from omniforge.api.jobs import job_manager
from omniforge.automl.backends import AutoMLBackendHub, probe_automl_backends
from omniforge.causal.inference import CausalInferenceEngine
from omniforge.copilot.agent import OmniForgeCopilotAgent
from omniforge.core.logging import get_logger
from omniforge.core.performance import (
    compact_scatter_payload,
    downsample_table_for_plot,
    smart_sample_df,
)
from omniforge.core.types import ModelMetadata, ModelStage, TaskType
from omniforge.core.utils import to_json_object, to_json_safe
from omniforge.data.cleaning import CleaningConfig, DataCleaner
from omniforge.data.connectors import (
    access_inspect_database,
    access_load_table,
    ensure_demo_sqlite_database,
    list_excel_sheets,
    read_excel_sheet,
    sql_inspect_database,
    sql_load_dataframe,
)
from omniforge.data.ingestion import DataIngestionEngine
from omniforge.data.quality import DataQualityProfiler
from omniforge.domains import default_domain_registry
from omniforge.eda.automated_eda import AutomatedEDA
from omniforge.eda.diagnostics import (
    DIAGNOSTIC_TEST_CATALOG,
    DOWNSTREAM_MODULE_CATALOG,
    StatisticalCriteriaConfig,
    UniversalDiagnosticsSuite,
)
from omniforge.eda.statistics import StatisticalAnalyzer
from omniforge.evaluation.equations import ModelEquationExtractor
from omniforge.evaluation.fairness import FairnessAuditor
from omniforge.evaluation.interpretability import InterpretabilityEngine
from omniforge.evaluation.model_cards import ModelCardGenerator
from omniforge.features.engineering import FeatureConfig, FeatureEngineeringPipeline
from omniforge.features.selection import FeatureSelector
from omniforge.forecasting.engine import ForecastingEngine
from omniforge.mlops.drift import DriftDetector
from omniforge.mlops.registry import ModelRegistry
from omniforge.mlops.supervisory_monitor import SupervisoryPredictionMonitor
from omniforge.mlops.tracking import ExperimentTracker
from omniforge.models.anomaly.detector import AnomalyDetector
from omniforge.models.baselines import BaselineBenchmarkSuite
from omniforge.models.clustering.clusterer import ClusteringEngine
from omniforge.models.computer_vision.foundation import VisionFoundationHub, probe_vision_backends
from omniforge.models.deep import (
    CNN1DModel,
    FTTransformerModel,
    GraphNeuralNetworkModel,
    MLPModel,
    ResNetTabularModel,
    TabNetModel,
)
from omniforge.models.graphs.link_prediction import GraphLinkPredictionSuite
from omniforge.models.mining import AssociationRuleMiner, ManifoldProjector, SubgroupDiscoveryMiner
from omniforge.models.nlp.advanced_nlp import AdvancedNLPSuite, probe_nlp_backends
from omniforge.models.nonparametric import (
    NadarayaWatsonKernelModel,
    NonParametricAnalyticalSuite,
    PenalizedSplineGAMModel,
)
from omniforge.models.recommendation.recommender import HybridRecommender
from omniforge.models.survival.survival import CoxPHModel, KaplanMeierEstimator
from omniforge.models.time_series.deep_forecast import DeepForecaster
from omniforge.optimization.solvers import OptimizationEngine
from omniforge.pipelines.builder import PipelineBuilder
from omniforge.pipelines.executor import PipelineExecutor
from omniforge.reporting.generator import ReportGenerator
from omniforge.reporting.sections import REPORT_PRESETS, REPORT_SECTIONS_CATALOG
from omniforge.simulation.engines import MonteCarloSimulator
from omniforge.ui.state import (
    clear_split_lock,
    clear_workspace_dataset,
    export_workspace_project_bytes,
    get_active_split_frames,
    get_all_deleted_records_df,
    get_analysis_dataframe,
    get_dataset_processing_status,
    get_excluded_columns,
    get_selected_targets,
    get_session_state,
    get_studio_results,
    load_new_dataset,
    lock_leakage_safe_split,
    recombine_split_frames,
    record_deleted_data,
    record_exclusion,
    record_lineage,
    restore_workspace_project_bytes,
    revert_to_raw_dataset,
    run_format_audit,
    save_studio_result,
    set_excluded_columns,
    set_selected_targets,
    studio_completion_map,
    undo_last_dataset_change,
    update_active_dataset,
    workflow_completion_status,
)
from omniforge.version import get_version_info

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/workbench",
    tags=["OmniForge Workbench"],
    dependencies=[Depends(require_workbench_auth)],
)

_SHARED_REGISTRY = ModelRegistry()
_SHARED_TRACKER = ExperimentTracker()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sid(x_session_id: str | None) -> str:
    return (x_session_id or "default").strip() or "default"


def _require_df(state: dict[str, Any]) -> pd.DataFrame:
    df = state.get("uap_df")
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise HTTPException(
            status_code=400,
            detail="No active dataset loaded. Upload a dataset or load a sample in Studio 01 first.",
        )
    return df


def _benchmark_datasets_dir() -> Path:
    """Resolve the canonical directory for local benchmark CSV datasets."""
    env_dir = os.getenv("OMNIFORGE_BENCHMARK_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / "examples" / "datasets").resolve()


def _discover_benchmark_csvs() -> list[dict[str, Any]]:
    """Locate benchmark CSV datasets under the canonical benchmark datasets directory."""
    ds_dir = _benchmark_datasets_dir()
    if not ds_dir.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for p in sorted(ds_dir.glob("*.csv")):
        if not p.is_file():
            continue
        try:
            resolved = p.resolve()
            if not resolved.is_relative_to(ds_dir):
                continue
        except OSError:
            continue
        size_mb = round(resolved.stat().st_size / (1024 * 1024), 2)
        items.append(
            {
                "filename": p.name,
                "size_mb": size_mb,
                "label": f"{p.stem.replace('_', ' ').title()} ({size_mb} MB)",
            }
        )
    return items


def _state_summary(state: dict[str, Any], session_id: str) -> dict[str, Any]:
    df = state.get("uap_df")
    raw_df = state.get("uap_raw_df")
    has_df = isinstance(df, pd.DataFrame) and not df.empty
    df_ok: pd.DataFrame | None = df if isinstance(df, pd.DataFrame) and not df.empty else None
    cols = list(df_ok.columns) if df_ok is not None else []
    num_cols = list(df_ok.select_dtypes(include=[np.number]).columns) if df_ok is not None else []
    cat_cols = [c for c in cols if c not in num_cols]
    excluded = get_excluded_columns(state, df) if has_df else []
    targets = get_selected_targets(state, df) if has_df else []
    status = get_dataset_processing_status(state) if has_df else {}
    lock = state.get("uap_split_lock")
    split_summary = None
    if isinstance(lock, dict) and lock.get("active"):
        split_summary = {
            "active": True,
            "strategy": lock.get("strategy"),
            "n_train": lock.get("n_train"),
            "n_test": lock.get("n_test"),
            "cv_folds": lock.get("cv_folds"),
            "test_size": lock.get("test_size"),
            "passed": bool((lock.get("audit_report") or {}).get("passed", True)),
        }
    deleted_log = state.get("uap_deleted_records_log") or []
    total_deleted = int(sum(int(entry.get("rows_removed", entry.get("rows_deleted", 0))) for entry in deleted_log))
    fmt_audit = state.get("uap_format_audit") or {}

    return to_json_object(
        {
            "session_id": session_id,
            "platform": get_version_info(),
            "has_dataset": has_df,
            "dataset_name": state.get("uap_dataset_name"),
            "dataset_source": state.get("uap_dataset_source"),
            "n_rows": int(len(df_ok)) if df_ok is not None else 0,
            "n_cols": int(df_ok.shape[1]) if df_ok is not None else 0,
            "raw_rows": int(len(raw_df)) if isinstance(raw_df, pd.DataFrame) else 0,
            "raw_cols": int(raw_df.shape[1]) if isinstance(raw_df, pd.DataFrame) else 0,
            "columns": cols,
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols,
            "targets": targets,
            "primary_target": targets[0] if targets else None,
            "excluded_columns": excluded,
            "analysis_cols_count": max(0, len(cols) - len(excluded)),
            "can_undo": state.get("uap_prev_df") is not None,
            "processing_status": status,
            "split_lock": split_summary,
            "total_deleted_rows": total_deleted,
            "format_audit_issues": int(fmt_audit.get("total_issues", 0)) if isinstance(fmt_audit, dict) else 0,
            "intake_audit": state.get("uap_intake_audit"),
            "s02_cluster_committed": bool(state.get("uap_s02_cluster_committed")),
            "s02_cluster_summary": state.get("uap_s02_cluster_summary"),
            "s02_anomaly_committed": bool(state.get("uap_s02_anomaly_committed")),
            "s02_anomaly_summary": state.get("uap_s02_anomaly_summary"),
            "active_model_name": state.get("uap_active_model_name"),
            "active_model_metrics": state.get("uap_active_model_metrics"),
            "active_feature_cols": state.get("uap_active_feature_cols") or [],
            "available_models_count": len(SupervisoryPredictionMonitor.catalog_available_models(state)),
            "supervisory_status": (
                state["uap_supervisory_result"].get("system_status", "IDLE")
                if isinstance(state.get("uap_supervisory_result"), dict)
                else "IDLE"
            ),
            "studio_completed": studio_completion_map(state),
            "cached_result_keys": sorted(get_studio_results(state).keys()),
            "workflow_checklist": [{"label": lbl, "done": ok} for lbl, ok in workflow_completion_status(state)],
            "lineage": (state.get("uap_lineage") or [])[-30:],
            "theme": state.get("uap_theme", "dark"),
            "project_name": state.get("uap_project_name") or state.get("uap_dataset_name") or "omniforge_project",
        }
    )


def _cache_and_respond(
    state: dict[str, Any],
    sid: str,
    cache_key: str,
    body: dict[str, Any],
    *,
    include_state: bool = True,
) -> dict[str, Any]:
    """Save the JSON-safe result payload BEFORE computing ``_state_summary`` so ``cached_result_keys`` always includes ``cache_key``."""
    safe_body = to_json_object(body)
    save_studio_result(state, cache_key, safe_body)
    if include_state:
        safe_body["state"] = to_json_safe(_state_summary(state, sid))
    return safe_body


def _infer_task(y: pd.Series) -> TaskType:
    s = y.dropna()
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return TaskType.BINARY_CLASSIFICATION if s.nunique() <= 2 else TaskType.MULTICLASS_CLASSIFICATION
    if s.nunique() <= 10 and np.all(np.isclose(s.to_numpy(dtype=float), np.round(s.to_numpy(dtype=float)))):
        return TaskType.BINARY_CLASSIFICATION if s.nunique() <= 2 else TaskType.MULTICLASS_CLASSIFICATION
    return TaskType.REGRESSION


def _prepare_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, TaskType]:
    work = df.dropna(subset=[target]).copy()
    if len(work) < 10:
        raise HTTPException(status_code=400, detail=f"Target '{target}' has fewer than 10 non-null observations.")
    y = work[target]
    X_raw = work.drop(columns=[target])
    num_df = X_raw.select_dtypes(include=[np.number]).copy()
    cat_cols = [c for c in X_raw.columns if c not in num_df.columns]
    for c in cat_cols[:12]:
        if X_raw[c].nunique(dropna=True) <= 20:
            dummies = pd.get_dummies(X_raw[c].astype(str), prefix=c, drop_first=False, dtype=float)
            num_df = pd.concat([num_df, dummies], axis=1)
        else:
            codes, _ = pd.factorize(X_raw[c].astype(str))
            num_df[f"{c}_code"] = codes.astype(float)
    num_df = num_df.replace([np.inf, -np.inf], np.nan)
    medians = num_df.median(numeric_only=True).fillna(0.0)
    num_df = num_df.fillna(medians)
    if num_df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="No usable predictor columns remain after excluding target.")
    return num_df, y, _infer_task(y)


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------
class LoadSampleReq(BaseModel):
    source_type: str = Field(default="domain", description="'benchmark' or 'domain'")
    name: str = Field(default="finance")
    n_samples: int = Field(default=600, ge=30, le=50000)


class TargetsReq(BaseModel):
    targets: list[str]


class ExclusionsReq(BaseModel):
    excluded_columns: list[str]
    reason: str = "User excluded column from analysis"


class SplitLockReq(BaseModel):
    strategy: str = "auto"
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    cv_folds: int = Field(default=5, ge=2, le=10)
    time_col: str | None = None
    group_col: str | None = None
    target_col: str | None = None
    gap: int = 0


class CleanReq(BaseModel):
    mode: str = Field(default="custom", description="'auto', 'complete', or 'custom'")
    impute_numeric: str = "median"
    impute_categorical: str = "most_frequent"
    constant_fill_numeric: float = 0.0
    constant_fill_categorical: str = "MISSING"
    drop_duplicates: bool = True
    handle_outliers: str = "clip"
    outlier_method: str = "iqr"
    outlier_threshold: float = 1.5
    contamination: float = Field(default=0.05, gt=0.0, lt=0.5)
    standardize_missing_tokens: bool = True
    clean_strings: bool = True
    lowercase_strings: bool = False
    strip_html: bool = True
    infer_types: bool = True
    drop_high_missing_cols: bool = True
    missing_col_threshold: float = Field(default=0.90, ge=0.2, le=1.0)
    add_missing_indicators: bool = False
    drop_constant_columns: bool = False
    standardize_column_names: bool = False
    preview_only: bool = False


class SQLInspectReq(BaseModel):
    connection_url: str


class SQLLoadReq(BaseModel):
    connection_url: str
    table_name: str | None = None
    query: str | None = None
    limit: int = Field(default=50000, ge=10, le=250000)


class AccessInspectReq(BaseModel):
    file_path: str


class AccessLoadReq(BaseModel):
    file_path: str
    table_name: str
    limit: int = Field(default=50000, ge=10, le=250000)


class ClusterReq(BaseModel):
    mode: str = Field(default="custom", description="'auto', 'complete', or 'custom'")
    algorithm: str = "kmeans"
    algorithms: list[str] = Field(default_factory=list)
    n_clusters: int = Field(default=4, ge=2, le=20)
    run_auto_k: bool = True
    k_min: int = 2
    k_max: int = 8
    scaler: str = "standard"
    linkage: str = "ward"

    features: list[str] | None = None


# ClusteringEngine requires Literal-typed hyper-parameters; the HTTP layer accepts
# free-form strings, so normalise (and defensively fall back) before construction.
_CLUSTER_SCALERS: dict[str, Literal["standard", "robust", "minmax", "none"]] = {
    "standard": "standard",
    "robust": "robust",
    "minmax": "minmax",
    "none": "none",
}
_CLUSTER_LINKAGES: dict[str, Literal["ward", "complete", "average", "single"]] = {
    "ward": "ward",
    "complete": "complete",
    "average": "average",
    "single": "single",
}


class ClusterCommitReq(BaseModel):
    mode: str = Field(default="attach", description="'attach' or 'remove_clusters'")
    remove_cluster_ids: list[int] = Field(default_factory=list)
    attach_cluster_id: bool = False


class AnomalyReq(BaseModel):
    mode: str = Field(default="custom", description="'auto', 'complete', or 'custom'")
    method: str = "isolation_forest"
    methods: list[str] = Field(default_factory=list)
    contamination: float = Field(default=0.05, ge=0.005, le=0.45)
    features: list[str] | None = None


class AnomalyCommitReq(BaseModel):
    mode: str = Field(default="attach", description="'attach' or 'remove_anomalies'")


class MiningReq(BaseModel):
    mode: str = Field(default="complete", description="'auto', 'complete', or 'custom'")
    tasks: list[str] = Field(default_factory=lambda: ["association_rules", "subgroups", "manifold"])
    manifold_method: str = "pca"
    min_support: float = Field(default=0.08, ge=0.01, le=0.9)
    min_confidence: float = Field(default=0.55, ge=0.1, le=1.0)
    min_lift: float = Field(default=1.1, ge=0.5, le=20.0)
    color_col: str | None = None


class DiagnosticsReq(BaseModel):
    mode: str = "all"  # all | auto | custom
    alpha: float = 0.05
    confidence_level: float = 0.95
    fdr_method: str = "fdr_bh"
    parametric_mode: str = "auto"
    normality_method: str = "auto"
    correlation_method: str = "auto"
    outlier_method: str = "iqr"
    vif_threshold: float = 5.0
    risk_tail_alpha: float = 0.05
    holdout_ratio: float = 0.20
    selected_modules: list[str] = Field(default_factory=list)


class HypothesisReq(BaseModel):
    mode: str = "custom"  # custom | auto | complete
    test_type: str = "welch_t"
    col_a: str | None = None
    col_b: str | None = None
    alpha: float = 0.05


class FeatureEngReq(BaseModel):
    mode: str = "custom"  # custom | auto | complete
    polynomial_degree: int = 2
    add_interactions: bool = True
    add_polynomials: bool = False
    correct_skew: bool = True
    add_datetime_features: bool = True
    commit: bool = True


class FeatureSelReq(BaseModel):
    mode: str = "custom"  # custom | auto | complete
    method: str = "mutual_info"
    top_k: int = 10
    target: str | None = None
    commit: bool = False


class AutoMLRunReq(BaseModel):
    target: str | None = None
    backend: str = "omniforge_native"
    time_budget_seconds: float = Field(default=30.0, ge=5.0, le=300.0)
    cv_folds: int = Field(default=3, ge=2, le=10)
    async_job: bool = False


class DeepBenchReq(BaseModel):
    target: str | None = None
    architectures: list[str] = Field(
        default_factory=lambda: ["ft_transformer", "tabnet", "resnet_tabular", "mlp", "cnn1d", "gnn_gcn"]
    )
    epochs: int = Field(default=25, ge=5, le=120)
    max_rows: int = Field(default=2500, ge=100, le=15000)


class SinglePredictReq(BaseModel):
    record: dict[str, Any]


class ExplainReq(BaseModel):
    row_index: int = 0
    feature_for_pdp: str | None = None


class FairnessReq(BaseModel):
    sensitive_col: str


class ForecastReq(BaseModel):
    target_col: str
    time_col: str | None = None
    horizon: int = Field(default=14, ge=3, le=120)
    deep_arch: str = "nbeats"  # nbeats | patchtst | tft | lstm
    epochs: int = Field(default=30, ge=5, le=120)


class CausalReq(BaseModel):
    treatment_col: str
    outcome_col: str
    confounders: list[str] = Field(default_factory=list)
    method: str = "aipw"


class OptimizeReq(BaseModel):
    asset_cols: list[str] = Field(default_factory=list)
    risk_aversion: float = 2.0


class SimulateReq(BaseModel):
    target_col: str
    n_simulations: int = Field(default=2000, ge=200, le=25000)
    horizon_steps: int = Field(default=12, ge=1, le=120)


class NLPReq(BaseModel):
    task: str = "overview"
    text_col: str | None = None
    custom_texts: list[str] = Field(default_factory=list)
    candidate_labels: list[str] = Field(default_factory=lambda: ["Finance", "Operations", "Risk", "Customer Support"])
    question: str = ""
    context: str = ""
    query: str = ""
    aspects: list[str] = Field(default_factory=list)
    n_topics: int = 4
    n_clusters: int = 3
    use_pretrained: bool = False


class GraphReq(BaseModel):
    source_col: str | None = None
    target_col: str | None = None
    link_method: str = "ensemble_rrf"
    run_supervised: bool = True
    run_neural_gae: bool = True
    edge_operator: str = "hadamard"
    top_k: int = 12


class RecommendReq(BaseModel):
    user_col: str | None = None
    item_col: str | None = None
    rating_col: str | None = None
    top_k: int = 8


class SurvivalReq(BaseModel):
    duration_col: str
    event_col: str


class DAGRunReq(BaseModel):
    stages: list[str] = Field(default_factory=lambda: ["quality", "clean", "eda", "features", "automl"])
    target: str | None = None


class DriftReq(BaseModel):
    split_ratio: float = Field(default=0.5, ge=0.2, le=0.8)


class RegisterModelReq(BaseModel):
    model_name: str = "omniforge_champion"
    stage: str = "Staging"


class PromoteModelReq(BaseModel):
    model_name: str
    version: int
    stage: str


class ReportCompileReq(BaseModel):
    title: str = "OmniForge Executive Analytics Report"
    author: str = "OmniForge Enterprise"
    selected_sections: list[str] | None = None
    include_charts: bool = True
    include_tables: bool = True
    include_dataset_appendix: bool = True
    max_table_rows: int = Field(default=100, ge=10, le=500)


class SupervisoryQuickTrainReq(BaseModel):
    method: str = "ridge"  # ridge | logistic | decision_stump | random_forest | gradient_boosting | tabnet | resnet_tabular | ft_transformer | mlp
    target: str | None = None
    epochs: int = Field(default=15, ge=5, le=100)


class SupervisoryTolBase(BaseModel):
    model_id: str = "active_champion"
    baseline_model_id: str | None = None
    actual_col: str | None = None
    id_col: str | None = None
    max_pct_error: float = Field(default=10.0, ge=0.1, le=500.0)
    max_abs_error: float | None = None
    max_sigma_z: float = Field(default=3.0, ge=0.5, le=10.0)
    baseline_ratio_min: float = Field(default=0.85, ge=0.01, le=2.0)
    baseline_ratio_max: float = Field(default=1.15, ge=0.1, le=10.0)
    warning_fraction: float = Field(default=0.75, ge=0.2, le=0.99)
    max_batch_breach_rate_pct: float = Field(default=5.0, ge=0.0, le=100.0)
    min_classification_accuracy_pct: float = Field(default=85.0, ge=10.0, le=100.0)


class SupervisorySQLReq(SupervisoryTolBase):
    connection_url: str
    table_name: str | None = None
    query: str | None = None
    limit: int = Field(default=10000, ge=5, le=100000)


class SupervisoryAccessReq(SupervisoryTolBase):
    file_path: str
    table_name: str
    limit: int = Field(default=10000, ge=5, le=100000)


class SupervisoryDemoReq(SupervisoryTolBase):
    n_rows: int = Field(default=80, ge=15, le=500)
    inject_fault_pct: float = Field(default=10.0, ge=0.0, le=50.0)
    noise_scale: float = Field(default=0.04, ge=0.0, le=1.0)


class SupervisorySingleReq(SupervisoryTolBase):
    record: dict[str, Any]
    actual_value: Any | None = None


class CopilotReq(BaseModel):
    message: str


class NonParametricRunReq(BaseModel):
    target: str | None = None
    cv_folds: int = Field(default=5, ge=2, le=10)
    models: list[str] | None = None
    n_knots: int = Field(default=5, ge=3, le=12)
    spline_degree: int = Field(default=3, ge=1, le=3)
    bandwidth: float | None = Field(default=None, gt=0.0)
    n_neighbors: int = Field(default=11, ge=3, le=51)
    rbf_gamma: float | None = Field(default=None, gt=0.0)
    include_parametric_ref: bool = True


# ---------------------------------------------------------------------------
# Workspace & Session Endpoints
# ---------------------------------------------------------------------------
@router.get("/state")
def get_workbench_state(x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    return _state_summary(get_session_state(sid), sid)


@router.get("/results")
def get_cached_studio_results(x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    """Return all cached JSON-serializable studio UI results so navigating back to any stage restores immediately."""
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    cached = get_studio_results(state)
    return to_json_object(
        {
            "results": cached,
            "studio_results": cached,
            "state": _state_summary(state, sid),
        }
    )


@router.get("/benchmarks")
def list_benchmarks() -> dict[str, Any]:
    domains = [p.summary() for p in default_domain_registry.list_plugins()]
    return to_json_object({"benchmarks": _discover_benchmark_csvs(), "domains": domains})


@router.post("/theme")
def set_theme(payload: dict[str, str], x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    t = str(payload.get("theme", "dark")).lower()
    state["uap_theme"] = "light" if t == "light" else "dark"
    return {"theme": state["uap_theme"]}


@router.post("/targets")
def update_targets(req: TargetsReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = _require_df(state)
    valid = set_selected_targets(state, req.targets, df)
    return {"targets": valid, "state": _state_summary(state, sid)}


@router.post("/exclusions")
def update_exclusions(req: ExclusionsReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = _require_df(state)
    updated = set_excluded_columns(state, req.excluded_columns, df=df)
    if req.reason:
        record_lineage(state, "Column Scope", "Exclusion reason noted", req.reason)
    return {"excluded_columns": updated, "state": _state_summary(state, sid)}


@router.post("/split-lock")
def apply_split_lock(req: SplitLockReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = _require_df(state)
    lock = lock_leakage_safe_split(
        state,
        df,
        strategy=req.strategy,
        test_size=req.test_size,
        cv_folds=req.cv_folds,
        time_col=req.time_col,
        group_col=req.group_col,
        target_col=req.target_col,
        gap=req.gap,
    )
    return to_json_object({"split_lock": lock, "state": _state_summary(state, sid)})


@router.delete("/split-lock")
def unlock_split(x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    clear_split_lock(state)
    return {"state": _state_summary(state, sid)}


@router.post("/workspace/clear")
def reset_workspace(x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    clear_workspace_dataset(state)
    return {"state": _state_summary(state, sid)}


@router.post("/workspace/save")
def save_workspace_uap(
    project_name: str = Form(default="omniforge_project"),
    notes: str = Form(default=""),
    x_session_id: str | None = Header(default="default"),
) -> Response:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    raw_bytes, manifest = export_workspace_project_bytes(state, project_name=project_name, notes=notes)
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in manifest.project_name).strip("_") or "omniforge_project"
    return Response(
        content=raw_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.omniforge"'},
    )


@router.post("/workspace/load")
async def load_workspace_uap(
    file: UploadFile = File(...),
    x_session_id: str | None = Header(default="default"),
) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    raw = await file.read()
    try:
        info = restore_workspace_project_bytes(state, raw, source_label=file.filename or "Uploaded .omniforge")
    except Exception as exc:
        from omniforge.core.exceptions import DataValidationError, SecurityError

        if isinstance(exc, SecurityError):
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if isinstance(exc, DataValidationError):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise
    return to_json_object({"restored": info, "state": _state_summary(state, sid)})


@router.get("/jobs")
def list_jobs() -> dict[str, Any]:
    return {"jobs": job_manager.list_recent()}


@router.get("/jobs/{job_id}")
def poll_job(job_id: str) -> dict[str, Any]:
    rec = job_manager.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return rec.to_dict()


# ---------------------------------------------------------------------------
# Studio 01 — Data Ingestion, Quality & Cleaning
# ---------------------------------------------------------------------------
@router.get("/s01/catalog")
def s01_catalog() -> dict[str, Any]:
    """Return available benchmark CSV datasets and instant domain generators."""
    benchmarks = _discover_benchmark_csvs()
    domains = [
        {"name": plugin.metadata.name, "display_name": plugin.metadata.display_name}
        for plugin in sorted(default_domain_registry.list_plugins(), key=lambda p: p.metadata.name)
    ]
    return to_json_object({"benchmarks": benchmarks, "domains": domains})


@router.delete("/dataset")
def delete_active_dataset(x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    """Clear the active dataset and reset session workspace state."""
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    clear_workspace_dataset(state)
    return to_json_object({"cleared": True, "state": _state_summary(state, sid)})


@router.post("/s01/excel-sheets")
async def s01_excel_sheets(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    sheets = list_excel_sheets(raw)
    return to_json_object({"filename": file.filename or "workbook.xlsx", "sheets": sheets})


@router.post("/s01/upload")
async def s01_upload_dataset(
    file: UploadFile = File(...),
    sheet_name: str | None = Form(default=None),
    x_session_id: str | None = Header(default="default"),
) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    fname = file.filename or "uploaded.csv"
    suffix = Path(fname).suffix.lower() or ".csv"
    if suffix in {".pkl", ".pickle", ".joblib", ".dill"}:
        raise HTTPException(
            status_code=400,
            detail="Pickle/joblib uploads are blocked for security (CWE-502). Please upload CSV, Parquet, Excel, or JSON.",
        )
    raw = await file.read()
    t0 = time.perf_counter()
    if suffix in {".xlsx", ".xls", ".xlsm"} and sheet_name:
        df = read_excel_sheet(raw, sheet_name=sheet_name)
        src_label = f"Excel Upload ({fname} · Sheet '{sheet_name}')"
    else:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)
        try:
            df = DataIngestionEngine().ingest(tmp_path).data
        finally:
            tmp_path.unlink(missing_ok=True)
        src_label = f"File Upload ({fname})"
    load_new_dataset(state, df, name=Path(fname).stem, source=src_label)
    return to_json_object(
        {
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
            "format_audit": state.get("uap_format_audit"),
            "intake_audit": state.get("uap_intake_audit"),
            "state": _state_summary(state, sid),
        }
    )


@router.post("/s01/sql/demo")
def s01_sql_demo_db() -> dict[str, Any]:
    """Ensure the built-in SQLite enterprise demo database exists and return its connection info & tables."""
    db_path = ensure_demo_sqlite_database()
    conn_url = f"sqlite:///{db_path.as_posix()}"
    info = sql_inspect_database(conn_url)
    return to_json_object({"connection_url": conn_url, "db_path": str(db_path), **info})


@router.post("/s01/sql/inspect")
def s01_sql_inspect(req: SQLInspectReq) -> dict[str, Any]:
    return to_json_object(sql_inspect_database(req.connection_url))


@router.post("/s01/sql/load")
def s01_sql_load(req: SQLLoadReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    t0 = time.perf_counter()
    df = sql_load_dataframe(req.connection_url, table_name=req.table_name, query=req.query, limit=req.limit)
    label = req.table_name or "sql_query_result"
    load_new_dataset(state, df, name=label, source=f"SQL Database ({label})")
    return to_json_object(
        {
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
            "format_audit": state.get("uap_format_audit"),
            "intake_audit": state.get("uap_intake_audit"),
            "state": _state_summary(state, sid),
        }
    )


@router.post("/s01/access/inspect")
def s01_access_inspect(req: AccessInspectReq) -> dict[str, Any]:
    return to_json_object(access_inspect_database(req.file_path))


@router.post("/s01/access/load")
def s01_access_load(req: AccessLoadReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    t0 = time.perf_counter()
    df = access_load_table(req.file_path, req.table_name, limit=req.limit)
    load_new_dataset(state, df, name=req.table_name, source=f"MS Access ({Path(req.file_path).name} · {req.table_name})")
    return to_json_object(
        {
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
            "format_audit": state.get("uap_format_audit"),
            "intake_audit": state.get("uap_intake_audit"),
            "state": _state_summary(state, sid),
        }
    )


@router.post("/s01/load-sample")
def s01_load_sample(req: LoadSampleReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    t0 = time.perf_counter()
    if req.source_type == "benchmark":
        raw_name = (req.name or "").strip()
        if (
            not raw_name
            or "\x00" in raw_name
            or "/" in raw_name
            or "\\" in raw_name
            or Path(raw_name).name != raw_name
            or not raw_name.lower().endswith(".csv")
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid benchmark dataset name. Provide a plain '.csv' filename without path separators.",
            )
        ds_root = _benchmark_datasets_dir().resolve()
        csv_path = (ds_root / raw_name).resolve()
        if not csv_path.is_relative_to(ds_root):
            raise HTTPException(
                status_code=400,
                detail="Benchmark dataset path escapes the allowed datasets directory.",
            )
        if not csv_path.is_file():
            raise HTTPException(status_code=404, detail=f"Benchmark dataset '{raw_name}' not found.")
        df = DataIngestionEngine().ingest(csv_path).data
        load_new_dataset(state, df, name=csv_path.stem, source=f"Benchmark CSV ({csv_path.name})")
    else:
        plugin = default_domain_registry.get(req.name)
        df = plugin.generate_sample_dataset(n_samples=req.n_samples)
        load_new_dataset(
            state,
            df,
            name=f"{plugin.metadata.name}_sample",
            source=f"Domain Generator ({plugin.metadata.display_name})",
        )
    return to_json_object(
        {
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
            "format_audit": state.get("uap_format_audit"),
            "intake_audit": state.get("uap_intake_audit"),
            "state": _state_summary(state, sid),
        }
    )


@router.get("/s01/grid")
def s01_paginated_grid(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=10, le=250),
    sort_col: str | None = Query(default=None),
    sort_dir: str = Query(default="asc"),
    search: str = Query(default=""),
    view_mode: str = Query(default="active"),
    x_session_id: str | None = Header(default="default"),
) -> dict[str, Any]:
    """Server-side paginated, searchable & sortable DataFrame view (~8 KB even for 1M rows)."""
    state = get_session_state(_sid(x_session_id))
    base_df = state.get("uap_raw_df") if view_mode == "raw" else state.get("uap_df")
    if not isinstance(base_df, pd.DataFrame) or base_df.empty:
        return {"columns": [], "rows": [], "total_rows": 0, "page": 1, "page_size": page_size, "total_pages": 0}
    df = base_df
    if search.strip():
        q = search.strip().lower()
        scan_df = df if len(df) <= 100_000 else df.iloc[:100_000]
        mask = pd.Series(False, index=scan_df.index)
        for col in scan_df.columns[:25]:
            mask = mask | scan_df[col].astype(str).str.lower().str.contains(q, regex=False, na=False)
        df = scan_df.loc[mask]
    if sort_col and sort_col in df.columns:
        df = df.sort_values(by=sort_col, ascending=(sort_dir.lower() != "desc"), kind="mergesort", na_position="last")
    total = int(len(df))
    total_pages = max(1, (total + page_size - 1) // page_size)
    page_clamped = min(page, total_pages)
    start = (page_clamped - 1) * page_size
    page_df = df.iloc[start : start + page_size]
    excluded = set(get_excluded_columns(state, base_df))
    targets = set(get_selected_targets(state, base_df))
    col_meta = [
        {
            "name": str(c),
            "dtype": str(base_df[c].dtype),
            "excluded": str(c) in excluded,
            "is_target": str(c) in targets,
        }
        for c in base_df.columns
    ]
    return to_json_object(
        {
            "columns": col_meta,
            "rows": page_df.to_dict(orient="records"),
            "total_rows": total,
            "page": page_clamped,
            "page_size": page_size,
            "total_pages": total_pages,
        }
    )


@router.post("/s01/quality")
@router.post("/s01/profile")
def s01_run_quality(x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = _require_df(state)
    q_report = DataQualityProfiler().profile(df)
    state["uap_quality"] = q_report
    fmt = state.get("uap_format_audit") or run_format_audit(state, df)
    record_lineage(state, "Quality", "Ran Data Quality & Format Audit", f"Score={q_report.overall_score:.1f}/100")
    return _cache_and_respond(state, sid, "s01_quality", {"quality": q_report.model_dump(), "format_audit": fmt})


@router.post("/s01/clean")
def s01_run_cleaning(req: CleanReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = _require_df(state)
    before_cols = list(df.columns)

    mode = req.mode.lower()
    if mode == "auto":
        preset = (state.get("uap_intake_audit") or {}).get("auto_clean_preset") or {}
        cfg = CleaningConfig(
            standardize_missing_tokens=bool(preset.get("standardize_missing_tokens", True)),
            clean_strings=bool(preset.get("clean_strings", True)),
            lowercase_strings=False,
            strip_html=bool(preset.get("strip_html", True)),
            infer_types=bool(preset.get("infer_types", True)),
            drop_high_missing_cols=bool(preset.get("drop_high_missing_cols", True)),
            missing_col_threshold=float(preset.get("missing_col_threshold", 0.85)),
            numeric_imputation=str(preset.get("numeric_imputation", "median")),
            categorical_imputation=str(preset.get("categorical_imputation", "most_frequent")),
            drop_exact_duplicates=bool(preset.get("drop_exact_duplicates", True)),
            outlier_method=str(preset.get("outlier_method", "iqr")),
            outlier_action=str(preset.get("outlier_action", "clip")),
            outlier_threshold=float(preset.get("outlier_threshold", 1.5)),
            exclude_columns=get_excluded_columns(state, df),
        )
        drop_const = bool(preset.get("drop_constant_columns", True))
        std_col_names = False
    elif mode == "complete":
        cfg = CleaningConfig(
            standardize_missing_tokens=True,
            clean_strings=True,
            lowercase_strings=False,
            strip_html=True,
            infer_types=True,
            drop_high_missing_cols=True,
            missing_col_threshold=0.80,
            numeric_imputation="median",
            categorical_imputation="most_frequent",
            add_missing_indicators=True,
            drop_exact_duplicates=True,
            outlier_method="iqr",
            outlier_action="clip",
            outlier_threshold=1.5,
            exclude_columns=get_excluded_columns(state, df),
        )
        drop_const = True
        std_col_names = False
    else:
        _lock_raw = state.get("uap_split_lock")
        lock_cfg: dict[str, Any] = _lock_raw if isinstance(_lock_raw, dict) else {}
        has_time_lock = bool(lock_cfg.get("active")) and bool(lock_cfg.get("time_col"))
        cfg = CleaningConfig(
            standardize_missing_tokens=req.standardize_missing_tokens,
            clean_strings=req.clean_strings,
            lowercase_strings=req.lowercase_strings,
            strip_html=req.strip_html,
            infer_types=req.infer_types,
            drop_high_missing_cols=req.drop_high_missing_cols,
            missing_col_threshold=req.missing_col_threshold,
            numeric_imputation=req.impute_numeric,
            categorical_imputation=req.impute_categorical,
            constant_fill_numeric=req.constant_fill_numeric,
            constant_fill_categorical=req.constant_fill_categorical,
            add_missing_indicators=req.add_missing_indicators,
            drop_exact_duplicates=req.drop_duplicates,
            outlier_method=req.outlier_method,
            outlier_action=req.handle_outliers,
            outlier_threshold=req.outlier_threshold,
            contamination=req.contamination,
            causal_time_series=has_time_lock,
            time_column=lock_cfg.get("time_col") if has_time_lock else None,
            exclude_columns=get_excluded_columns(state, df),
        )
        drop_const = req.drop_constant_columns
        std_col_names = req.standardize_column_names

    # Replace +/-Inf before cleaning so every imputer works cleanly
    work_in = df.copy()
    for nc in work_in.select_dtypes(include=[np.number]).columns:
        work_in[nc] = work_in[nc].replace([np.inf, -np.inf], np.nan)

    cleaner = DataCleaner(cfg)
    train_df, test_df, lock = get_active_split_frames(state, work_in)
    split_safe_used = False
    if test_df is not None and lock is not None:
        split_safe_used = True
        cleaned_tr, cleaned_te, rep_tr, rep_te = cleaner.fit_transform_split(train_df, test_df)
        cleaned = recombine_split_frames(cleaned_tr, cleaned_te, work_in.index)
        rep_dict = rep_tr.model_dump()
        rep_dict["exact_duplicates_removed"] = int(rep_tr.exact_duplicates_removed + rep_te.exact_duplicates_removed)
        rep_dict["fuzzy_duplicates_removed"] = int(rep_tr.fuzzy_duplicates_removed + rep_te.fuzzy_duplicates_removed)
        rep_dict["removed_rows_count"] = int(rep_tr.removed_rows_count + rep_te.removed_rows_count)
        merged_outliers: dict[str, int] = dict(rep_tr.outliers_detected)
        for k_o, v_o in rep_te.outliers_detected.items():
            merged_outliers[k_o] = int(merged_outliers.get(k_o, 0) + v_o)
        rep_dict["outliers_detected"] = merged_outliers
        rep_dict["split_lock_leakage_safe"] = True
    else:
        cleaned, report = cleaner.fit_transform(work_in)
        rep_dict = report.model_dump() if hasattr(report, "model_dump") else dict(report)
        rep_dict["split_lock_leakage_safe"] = False

    dropped_const_cols: list[str] = []
    if drop_const:
        excl_set = set(get_excluded_columns(state, cleaned)) | set(get_selected_targets(state, cleaned))
        for col in list(cleaned.columns):
            if col not in excl_set and int(cleaned[col].nunique(dropna=True)) <= 1 and len(cleaned) > 1:
                dropped_const_cols.append(str(col))
        if dropped_const_cols and len(dropped_const_cols) < cleaned.shape[1]:
            cleaned = cleaned.drop(columns=dropped_const_cols)

    # Compute genuinely dropped columns BEFORE optional column-name standardization
    dropped_cols = [c for c in before_cols if c not in cleaned.columns]
    rename_map: dict[str, str] = {}
    if std_col_names:
        import re as _re

        seen_names: set[str] = set()
        new_names: list[str] = []
        for i, c in enumerate(cleaned.columns):
            base = _re.sub(r"[^a-zA-Z0-9]+", "_", str(c).strip()).strip("_").lower() or f"col_{i}"
            cand = base
            suffix = 2
            while cand in seen_names:
                cand = f"{base}_{suffix}"
                suffix += 1
            seen_names.add(cand)
            rename_map[str(c)] = cand
            new_names.append(cand)
        cleaned.columns = new_names

    removed_idx = [idx for idx in df.index if idx not in set(cleaned.index)]
    rep_dict["mode_used"] = mode
    rep_dict["dropped_constant_columns"] = dropped_const_cols
    rep_dict["renamed_columns"] = {k: v for k, v in rename_map.items() if k != v}
    rep_dict["rows_before"] = int(len(df))
    rep_dict["rows_after"] = int(len(cleaned))
    rep_dict["cols_before"] = int(df.shape[1])
    rep_dict["cols_after"] = int(cleaned.shape[1])
    rep_dict["missing_before"] = int(df.isna().sum().sum())
    rep_dict["missing_after"] = int(cleaned.isna().sum().sum())
    rep_dict["split_safe_used"] = split_safe_used

    if req.preview_only:
        return _cache_and_respond(
            state,
            sid,
            "s01_clean",
            {
                "preview_only": True,
                "cleaning_report": rep_dict,
                "rows_removed": len(removed_idx),
                "columns_dropped": dropped_cols,
            },
        )

    # Remap targets & excluded columns if column standardization renamed them
    if rename_map:
        state["uap_targets"] = [rename_map.get(c, c) for c in (state.get("uap_targets") or [])]
        if state.get("uap_target"):
            state["uap_target"] = rename_map.get(str(state["uap_target"]), state["uap_target"])
        state["uap_excluded_columns"] = [rename_map.get(c, c) for c in (state.get("uap_excluded_columns") or [])]

    if removed_idx:
        removed_forensic = (
            cleaner.last_removed_rows_.copy()
            if isinstance(cleaner.last_removed_rows_, pd.DataFrame) and not cleaner.last_removed_rows_.empty
            else df.loc[removed_idx].copy()
        )
        record_deleted_data(
            state,
            stage="Studio 01 Cleaning",
            reason=f"Cleaning ({mode}): duplicates / outlier removal",
            removed_df=removed_forensic,
            removed_columns=dropped_cols,
        )
        record_exclusion(
            state,
            stage="Studio 01 Cleaning",
            exclusion_type="Row Removal",
            affected_scope=f"{len(removed_idx):,} rows",
            count=len(removed_idx),
            total_before=len(df),
            criterion_rule=f"mode={mode}, drop_dups={cfg.drop_exact_duplicates}, outlier={cfg.outlier_method}/{cfg.outlier_action}",
            reason="Removed duplicate and/or extreme outlier rows during Studio 01 Data Cleaning.",
            sample_identifiers=removed_idx[:10],
        )
    elif dropped_cols:
        record_deleted_data(
            state,
            stage="Studio 01 Cleaning",
            reason=f"Cleaning ({mode}): high-missing / constant column removal",
            removed_df=None,
            removed_columns=dropped_cols,
        )
    if dropped_cols:
        record_exclusion(
            state,
            stage="Studio 01 Cleaning",
            exclusion_type="Column Removal",
            affected_scope=", ".join(dropped_cols[:10]),
            count=len(dropped_cols),
            total_before=df.shape[1],
            criterion_rule=f"missing>{cfg.missing_col_threshold:.0%} or constant",
            reason="Dropped high-missing or zero-variance columns during Studio 01 Data Cleaning.",
            sample_identifiers=dropped_cols[:10],
        )

    state["uap_s01_cleaned"] = True
    state["uap_clean_report"] = rep_dict
    update_active_dataset(
        state,
        cleaned,
        stage="Cleaning",
        action=f"Applied DataCleaner ({mode})",
        details=f"Rows {len(df):,} -> {len(cleaned):,} | Cols {df.shape[1]} -> {cleaned.shape[1]}",
        preserve_stage="s01",
    )
    return _cache_and_respond(
        state,
        sid,
        "s01_clean",
        {
            "preview_only": False,
            "cleaning_report": rep_dict,
            "rows_removed": len(removed_idx),
            "columns_dropped": dropped_cols,
        },
    )


@router.get("/s01/export-clean/{fmt}")
def s01_export_clean_dataset(fmt: str, x_session_id: str | None = Header(default="default")) -> Response:
    state = get_session_state(_sid(x_session_id))
    df = _require_df(state)
    stem = str(state.get("uap_dataset_name") or "omniforge_cleaned")
    fmt_l = fmt.lower()
    if fmt_l == "xlsx":
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        return Response(
            content=buf.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{stem}_clean.xlsx"'},
        )
    return Response(
        content=df.to_csv(index=False).encode("utf-8"),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{stem}_clean.csv"'},
    )


@router.post("/s01/undo")
def s01_undo(x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    ok = undo_last_dataset_change(state)
    return {"undone": ok, "state": _state_summary(state, sid)}


@router.post("/s01/reset-raw")
@router.post("/s01/revert")
def s01_reset_raw(x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    ok = revert_to_raw_dataset(state)
    return {"reverted": ok, "state": _state_summary(state, sid)}


@router.get("/s01/deleted-audit")
@router.get("/s01/deleted-log")
def s01_deleted_audit(x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = get_session_state(_sid(x_session_id))
    raw_log = state.get("uap_deleted_records_log") or []
    log = [
        {
            "timestamp": entry.get("timestamp", ""),
            "stage": entry.get("stage", ""),
            "reason": entry.get("reason", ""),
            "rows_removed": int(entry.get("rows_removed", entry.get("rows_deleted", 0))),
            "columns_removed": list(entry.get("columns_removed") or []),
        }
        for entry in raw_log
    ]
    del_df = get_all_deleted_records_df(state)
    return to_json_object(
        {
            "log": log,
            "total_deleted": int(len(del_df)),
            "preview_columns": list(del_df.columns) if not del_df.empty else [],
            "preview_rows": del_df.head(100).to_dict(orient="records") if not del_df.empty else [],
        }
    )


@router.get("/s01/deleted-audit/csv")
def s01_deleted_audit_csv(x_session_id: str | None = Header(default="default")) -> Response:
    state = get_session_state(_sid(x_session_id))
    del_df = get_all_deleted_records_df(state)
    csv_bytes = del_df.to_csv(index=False).encode("utf-8")
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="omniforge_deleted_records_audit.csv"'},
    )


# ---------------------------------------------------------------------------
# Studio 02 — Clustering, Anomaly Screening & Data Mining Lab
# ---------------------------------------------------------------------------
@router.post("/s02/clustering")
def s02_run_clustering(req: ClusterReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = get_analysis_dataframe(state, _require_df(state))
    helper_cols = {"cluster_id", "is_anomaly", "anomaly_score"}
    num_cols = req.features or [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in helper_cols
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    if len(num_cols) < 2:
        raise HTTPException(status_code=400, detail="Clustering requires at least 2 numeric columns.")
    train_df, test_df, _ = get_active_split_frames(state, df)

    mode = req.mode.lower()
    if mode == "complete":
        algos_to_run = req.algorithms or ["kmeans", "gmm", "agglomerative", "birch", "dbscan"]
    elif mode == "auto":
        algos_to_run = ["kmeans", "gmm", "dbscan"]
    else:
        algos_to_run = [req.algorithm]

    comparison_rows: list[dict[str, Any]] = []
    best_bundle: tuple[float, Any, pd.DataFrame] | None = None

    for algo_name in algos_to_run:
        try:
            engine = ClusteringEngine(
                algorithm=algo_name,
                n_clusters=req.n_clusters,
                scaler=_CLUSTER_SCALERS.get(req.scaler, "standard"),
                linkage=_CLUSTER_LINKAGES.get(req.linkage, "ward"),
            )
            use_auto_k = req.run_auto_k if len(algos_to_run) == 1 else (algo_name == "kmeans")
            if test_df is not None:
                tr_res, te_labels = engine.fit_predict_split(
                    train_df,
                    test_df,
                    feature_cols=num_cols,
                    run_auto_k=use_auto_k,
                    k_min=req.k_min,
                    k_max=req.k_max,
                )
                cur_res = tr_res
                cur_preview = df.copy()
                cur_preview["cluster_id"] = -1
                cur_preview.loc[train_df.index, "cluster_id"] = cur_res.labels
                cur_preview.loc[test_df.index, "cluster_id"] = te_labels
            else:
                cur_res = engine.fit_predict_with_profile(
                    df,
                    feature_cols=num_cols,
                    run_auto_k=use_auto_k,
                    k_min=req.k_min,
                    k_max=req.k_max,
                )
                cur_preview = df.copy()
                cur_preview["cluster_id"] = cur_res.labels

            sil = float(cur_res.metrics.get("silhouette", -1.0)) if isinstance(cur_res.metrics, dict) else -1.0
            ch = float(cur_res.metrics.get("calinski_harabasz", 0.0)) if isinstance(cur_res.metrics, dict) else 0.0
            db = float(cur_res.metrics.get("davies_bouldin", 99.0)) if isinstance(cur_res.metrics, dict) else 99.0
            comparison_rows.append(
                {
                    "algorithm": algo_name,
                    "clusters_found": int(cur_res.n_clusters_found),
                    "noise_points": int(cur_res.noise_count),
                    "silhouette": round(sil, 4) if math.isfinite(sil) else -1.0,
                    "calinski_harabasz": round(ch, 2) if math.isfinite(ch) else 0.0,
                    "davies_bouldin": round(db, 4) if math.isfinite(db) else 99.0,
                }
            )
            rank_score = sil if math.isfinite(sil) else -2.0
            if best_bundle is None or rank_score > best_bundle[0]:
                best_bundle = (rank_score, cur_res, cur_preview)
        except Exception as exc:
            comparison_rows.append(
                {
                    "algorithm": algo_name,
                    "clusters_found": 0,
                    "noise_points": 0,
                    "silhouette": -1.0,
                    "calinski_harabasz": 0.0,
                    "davies_bouldin": 99.0,
                    "error": str(exc)[:120],
                }
            )

    if best_bundle is None:
        raise HTTPException(status_code=400, detail="Clustering failed on all selected algorithms for this dataset.")

    _, res, preview_df = best_bundle
    comparison_rows.sort(key=lambda r: r.get("silhouette", -1.0), reverse=True)
    for rank_i, row in enumerate(comparison_rows, start=1):
        row["rank"] = rank_i
        row["selected"] = "WINNER" if rank_i == 1 else ""

    state["uap_clustering_result"] = res
    state["uap_clustering_df"] = preview_df
    state["uap_s02_cluster_committed"] = False

    # Build exact clickable cluster choices with row counts & percentages from full preview_df
    vc = preview_df["cluster_id"].value_counts().sort_index()
    total_n = max(len(preview_df), 1)
    cluster_choices = [
        {
            "cluster_id": int(cid),
            "label": "Noise / Outlier Cluster (-1)" if int(cid) == -1 else f"Cluster {int(cid)}",
            "count": int(cnt),
            "pct": round(int(cnt) / total_n * 100.0, 2),
            "is_noise": int(cid) == -1,
        }
        for cid, cnt in vc.items()
    ]

    if isinstance(res.pca_2d, pd.DataFrame) and res.pca_2d.shape[1] >= 2:
        xs = res.pca_2d.iloc[:, 0].to_numpy(dtype=float)
        ys = res.pca_2d.iloc[:, 1].to_numpy(dtype=float)
    else:
        xs = np.zeros(len(res.labels))
        ys = np.zeros(len(res.labels))
    scatter = compact_scatter_payload(xs, ys, groups=res.labels, max_points=5000)

    return _cache_and_respond(
        state,
        sid,
        "s02_clustering",
        {
            "provenance": "PREVIEW_ONLY",
            "mode": mode,
            "algorithm": res.algorithm,
            "n_clusters": res.n_clusters_found,
            "noise_count": res.noise_count,
            "metrics": res.metrics,
            "auto_k_curve": res.k_selection_curve,
            "cluster_profiles": res.cluster_profiles,
            "cluster_choices": cluster_choices,
            "algorithm_benchmark": comparison_rows,
            "scatter": scatter,
            "features_used": num_cols,
        },
    )


@router.post("/s02/clustering/commit")
def s02_commit_clustering(req: ClusterCommitReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = _require_df(state)
    preview_df = state.get("uap_clustering_df")
    if not isinstance(preview_df, pd.DataFrame) or "cluster_id" not in preview_df.columns:
        raise HTTPException(status_code=400, detail="Run clustering preview first before committing.")

    rows_before = int(len(df))
    work = df.copy()
    aligned_labels = preview_df["cluster_id"].reindex(work.index)
    if aligned_labels.isna().all() and len(preview_df) == len(work):
        aligned_labels = pd.Series(preview_df["cluster_id"].to_numpy(), index=work.index)
    cluster_series = aligned_labels.fillna(-999999).astype(int)

    removed = 0
    if req.mode == "remove_clusters":
        if not req.remove_cluster_ids:
            raise HTTPException(
                status_code=400,
                detail="Select at least one cluster ID to remove before clicking 'Remove Selected Clusters'.",
            )
        target_ids = [int(x) for x in req.remove_cluster_ids]
        drop_mask = cluster_series.isin(target_ids)
        removed_df = work.loc[drop_mask].copy()
        removed = int(len(removed_df))
        if removed == 0:
            raise HTTPException(
                status_code=400,
                detail=f"None of the active rows matched cluster IDs {target_ids}. Re-run clustering preview and select valid cluster IDs.",
            )
        if removed >= rows_before - 5:
            raise HTTPException(
                status_code=400,
                detail="Cannot remove nearly all rows in the dataset. Leave at least 6 rows remaining.",
            )
        removed_df["cluster_id"] = cluster_series.loc[drop_mask]
        record_deleted_data(
            state,
            stage="Studio 02 Clustering",
            reason=f"Removed clusters {target_ids}",
            removed_df=removed_df,
        )
        record_exclusion(
            state,
            stage="Studio 02 Clustering",
            exclusion_type="Cluster Row Removal",
            affected_scope=f"Clusters {target_ids} ({removed:,} rows)",
            count=removed,
            total_before=rows_before,
            criterion_rule=f"cluster_id IN {target_ids}",
            reason=f"User removed {removed:,} rows belonging to cluster(s) {target_ids} during Studio 02 screening.",
            sample_identifiers=list(removed_df.index[:10]),
        )
        work = work.loc[~drop_mask].copy()
        if req.attach_cluster_id:
            work["cluster_id"] = cluster_series.loc[~drop_mask]
        action = f"Removed {removed:,} rows in cluster(s) {target_ids} ({rows_before:,} → {len(work):,})"
    else:
        work["cluster_id"] = cluster_series.replace(-999999, -1)
        action = f"Attached cluster_id column to active dataset ({len(work):,} rows)"

    # Persist Studio 02 committed summary BEFORE stage-aware update_active_dataset so green light NEVER turns off!
    prev_res = state.get("uap_clustering_result")
    state["uap_s02_cluster_committed"] = True
    state["uap_s02_last_cluster_deleted"] = removed
    state["uap_s02_cluster_summary"] = {
        "algorithm": getattr(prev_res, "algorithm", "clustering"),
        "n_clusters": int(getattr(prev_res, "n_clusters_found", 0)),
        "mode": req.mode,
        "removed_cluster_ids": [int(x) for x in req.remove_cluster_ids],
        "rows_before": rows_before,
        "rows_after": int(len(work)),
        "rows_removed": removed,
        "action": action,
        "metrics": getattr(prev_res, "metrics", {}),
    }
    update_active_dataset(
        state,
        work,
        stage="Clustering",
        action=action,
        details=f"Rows {rows_before:,} -> {len(work):,}",
        preserve_stage="s02_cluster",
    )
    # Keep preview_df synchronized with remaining rows so repeated operations stay accurate
    synced_preview = preview_df.loc[preview_df.index.intersection(work.index)].copy()
    state["uap_clustering_df"] = synced_preview

    vc_rem = synced_preview["cluster_id"].value_counts().sort_index() if not synced_preview.empty else pd.Series(dtype=int)
    rem_total = max(len(synced_preview), 1)
    remaining_choices = [
        {
            "cluster_id": int(cid),
            "label": "Noise / Outlier Cluster (-1)" if int(cid) == -1 else f"Cluster {int(cid)}",
            "count": int(cnt),
            "pct": round(int(cnt) / rem_total * 100.0, 2),
            "is_noise": int(cid) == -1,
        }
        for cid, cnt in vc_rem.items()
    ]
    cached_c = (state.get("uap_studio_results") or {}).get("s02_clustering")
    if isinstance(cached_c, dict):
        updated_cached = dict(cached_c)
        updated_cached["provenance"] = "COMMITTED"
        updated_cached["committed_action"] = action
        updated_cached["cluster_choices"] = remaining_choices
        updated_cached["n_clusters"] = int(sum(1 for c in remaining_choices if not c["is_noise"]))
        save_studio_result(state, "s02_clustering", updated_cached)

    return to_json_object(
        {
            "committed": True,
            "mode": req.mode,
            "rows_before": rows_before,
            "rows_after": int(len(work)),
            "rows_removed": removed,
            "action": action,
            "remaining_cluster_choices": remaining_choices,
            "state": _state_summary(state, sid),
        }
    )


@router.post("/s02/anomaly")
def s02_run_anomaly(req: AnomalyReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = get_analysis_dataframe(state, _require_df(state))
    helper_cols = {"cluster_id", "is_anomaly", "anomaly_score"}
    num_cols = req.features or [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in helper_cols
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    if not num_cols:
        raise HTTPException(status_code=400, detail="Anomaly detection requires numeric columns.")

    X_num = df[num_cols].replace([np.inf, -np.inf], np.nan)
    X_num = X_num.fillna(X_num.median(numeric_only=True).fillna(0.0))

    mode = req.mode.lower()
    if mode == "complete":
        det_methods = req.methods or ["isolation_forest", "lof", "pca_recon", "mad", "zscore", "ensemble"]
    elif mode == "auto":
        det_methods = ["ensemble"]
    else:
        det_methods = [req.method]

    detector_comparison: list[dict[str, Any]] = []
    selected_flags: np.ndarray | None = None
    selected_scores: np.ndarray | None = None
    chosen_method = det_methods[0]

    for m_name in det_methods:
        try:
            det = AnomalyDetector(method=m_name, contamination=req.contamination)
            det.fit(X_num)
            flags = det.predict(X_num)
            scores = det.score_samples(X_num)
            is_anom = (flags == -1) if np.any(flags == -1) else (flags == 1)
            detector_comparison.append(
                {
                    "detector": m_name,
                    "n_anomalies": int(is_anom.sum()),
                    "anomaly_rate_pct": round(float(is_anom.mean() * 100.0), 3),
                    "mean_score": round(float(np.nanmean(scores)), 4),
                    "max_score": round(float(np.nanmax(scores)), 4),
                }
            )
            if m_name == "ensemble" or selected_flags is None:
                selected_flags = is_anom.astype(int)
                selected_scores = scores
                chosen_method = m_name
        except Exception as exc:
            detector_comparison.append({"detector": m_name, "n_anomalies": 0, "anomaly_rate_pct": 0.0, "error": str(exc)[:100]})

    assert selected_flags is not None and selected_scores is not None
    preview = df.copy()
    preview["is_anomaly"] = selected_flags
    preview["anomaly_score"] = selected_scores
    state["uap_anomaly_df"] = preview
    state["uap_anomaly_meta"] = {
        "method": chosen_method,
        "mode": mode,
        "contamination": req.contamination,
        "n_anomalies": int(selected_flags.sum()),
        "detector_comparison": detector_comparison,
    }
    state["uap_s02_anomaly_committed"] = False

    x_col = num_cols[0]
    y_col = num_cols[1] if len(num_cols) > 1 else num_cols[0]
    scatter = compact_scatter_payload(
        X_num[x_col].to_numpy(),
        X_num[y_col].to_numpy() if y_col != x_col else selected_scores,
        groups=preview["is_anomaly"].to_numpy(),
        max_points=5000,
    )
    top_anoms = preview.loc[preview["is_anomaly"] == 1].sort_values("anomaly_score", ascending=False).head(50)
    return _cache_and_respond(
        state,
        sid,
        "s02_anomaly",
        {
            "provenance": "PREVIEW_ONLY",
            "mode": mode,
            "method": chosen_method,
            "n_anomalies": int(selected_flags.sum()),
            "anomaly_rate_pct": round(float(selected_flags.mean() * 100.0), 3),
            "detector_comparison": detector_comparison,
            "x_axis": x_col,
            "y_axis": y_col if y_col != x_col else "anomaly_score",
            "scatter": scatter,
            "top_anomalies": top_anoms.to_dict(orient="records"),
        },
    )


@router.post("/s02/anomaly/commit")
def s02_commit_anomaly(req: AnomalyCommitReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = _require_df(state)
    anom_df = state.get("uap_anomaly_df")
    if not isinstance(anom_df, pd.DataFrame) or "is_anomaly" not in anom_df.columns:
        raise HTTPException(status_code=400, detail="Run anomaly detection preview first.")
    rows_before = int(len(df))
    work = df.copy()
    aligned_anom = anom_df["is_anomaly"].reindex(work.index)
    aligned_score = anom_df["anomaly_score"].reindex(work.index)
    if aligned_anom.isna().all() and len(anom_df) == len(work):
        aligned_anom = pd.Series(anom_df["is_anomaly"].to_numpy(), index=work.index)
        aligned_score = pd.Series(anom_df["anomaly_score"].to_numpy(), index=work.index)
    is_anom_ser = aligned_anom.fillna(0).astype(int)
    score_ser = aligned_score.fillna(0.0).astype(float)
    removed = 0
    if req.mode == "remove_anomalies":
        drop_mask = is_anom_ser == 1
        removed_df = work.loc[drop_mask].copy()
        removed = int(len(removed_df))
        if removed > 0:
            removed_df["anomaly_score"] = score_ser.loc[drop_mask]
            record_deleted_data(
                state,
                stage="Studio 02 Anomaly Screening",
                reason="Removed detected anomalies",
                removed_df=removed_df,
            )
            record_exclusion(
                state,
                stage="Studio 02 Anomaly Screening",
                exclusion_type="Anomaly Row Removal",
                affected_scope=f"{removed:,} anomalous rows",
                count=removed,
                total_before=rows_before,
                criterion_rule=f"method={(state.get('uap_anomaly_meta') or {}).get('method', 'anomaly')}",
                reason="User removed detected anomalies during Studio 02 Anomaly Screening.",
                sample_identifiers=list(removed_df.index[:10]),
            )
            work = work.loc[~drop_mask].copy()
        action = f"Removed {removed:,} detected anomalies ({rows_before:,} → {len(work):,})"
    else:
        work["is_anomaly"] = is_anom_ser
        work["anomaly_score"] = score_ser
        action = "Attached is_anomaly & anomaly_score to active dataset"

    state["uap_s02_anomaly_committed"] = True
    state["uap_s02_last_anomaly_deleted"] = removed
    state["uap_s02_anomaly_summary"] = {
        "mode": req.mode,
        "rows_before": rows_before,
        "rows_after": int(len(work)),
        "rows_removed": removed,
        "action": action,
        "meta": state.get("uap_anomaly_meta"),
    }
    update_active_dataset(
        state,
        work,
        stage="Anomaly",
        action=action,
        details=f"Rows {rows_before:,} -> {len(work):,}",
        preserve_stage="s02_anomaly",
    )
    state["uap_anomaly_df"] = anom_df.loc[anom_df.index.intersection(work.index)].copy()
    cached_a = (state.get("uap_studio_results") or {}).get("s02_anomaly")
    if isinstance(cached_a, dict):
        updated_a = dict(cached_a)
        updated_a["provenance"] = "COMMITTED"
        updated_a["committed_action"] = action
        save_studio_result(state, "s02_anomaly", updated_a)
    return to_json_object(
        {
            "committed": True,
            "rows_before": rows_before,
            "rows_after": int(len(work)),
            "rows_removed": removed,
            "action": action,
            "state": _state_summary(state, sid),
        }
    )


@router.post("/s02/mining")
def s02_data_mining(req: MiningReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    """Run Association Rule Mining (Apriori), Subgroup Discovery, and Manifold/PCA/t-SNE/Isomap Projection."""
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = get_analysis_dataframe(state, _require_df(state))
    targets = get_selected_targets(state, df)
    target = targets[0] if targets else None
    mode = req.mode.lower()
    tasks = ["association_rules", "subgroups", "manifold"] if mode in ("complete", "auto") else (req.tasks or ["association_rules"])

    out: dict[str, Any] = {"mode": mode, "tasks_executed": tasks}
    if "association_rules" in tasks:
        out["association_rules"] = AssociationRuleMiner().mine(
            df,
            min_support=req.min_support,
            min_confidence=req.min_confidence,
            min_lift=req.min_lift,
        )
    if "subgroups" in tasks and target:
        out["subgroups"] = SubgroupDiscoveryMiner().discover(df, target_col=target)
    if "manifold" in tasks:
        try:
            out["manifold"] = ManifoldProjector().project(
                df,
                method=req.manifold_method,
                color_col=req.color_col or target,
            )
        except Exception as exc:
            out["manifold"] = {"error": str(exc)}
    state["uap_mining_result"] = out
    record_lineage(state, "Data Mining", f"Executed Data Mining ({mode})", f"Tasks={', '.join(tasks)}")
    return _cache_and_respond(state, sid, "s02_mining", dict(out))


# ---------------------------------------------------------------------------
# Studio 03 — Universal Diagnostics, Baselines & Risk
# ---------------------------------------------------------------------------
@router.get("/s03/catalog")
def s03_catalog() -> dict[str, Any]:
    return to_json_object(
        {
            "diagnostic_tests": DIAGNOSTIC_TEST_CATALOG,
            "downstream_modules": DOWNSTREAM_MODULE_CATALOG,
        }
    )


@router.post("/s03/run")
def s03_run_diagnostics(req: DiagnosticsReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = get_analysis_dataframe(state, _require_df(state))
    targets = get_selected_targets(state, df)
    criteria = StatisticalCriteriaConfig(
        alpha=req.alpha,
        confidence_level=req.confidence_level,
        multiple_testing_correction=req.fdr_method,  # type: ignore[arg-type]
        parametric_mode=req.parametric_mode,  # type: ignore[arg-type]
        normality_method=req.normality_method,  # type: ignore[arg-type]
        correlation_method=req.correlation_method,  # type: ignore[arg-type]
        outlier_method=req.outlier_method,  # type: ignore[arg-type]
        vif_threshold=req.vif_threshold,
        risk_tail_alpha=req.risk_tail_alpha,
        holdout_ratio=req.holdout_ratio,
    )
    suite = UniversalDiagnosticsSuite()
    modules: list[str] | None
    if req.mode == "auto":
        modules = list(suite.recommend_modules(df, target_cols=targets).keys())
    elif req.mode == "custom" and req.selected_modules:
        modules = list(req.selected_modules)
    else:
        modules = None  # Complete / Select All
    report = suite.run(df, target_cols=targets, selected_modules=modules, criteria=criteria)
    state["uap_diag_report"] = report
    state["uap_diag_criteria"] = criteria.model_dump() if hasattr(criteria, "model_dump") else dataclasses.asdict(criteria)
    state["uap_baseline_results"] = report.baseline_benchmarks
    state["uap_baseline_equations"] = report.baseline_equations
    state["uap_baseline_models"] = dict(getattr(suite, "last_baseline_models_", {}) or {})
    state["uap_risk_report"] = report.risk_assessment
    record_lineage(state, "Diagnostics", f"Ran Universal Diagnostics ({req.mode})", f"Tests={len(report.hypothesis_ledger)} | Targets={len(targets)}")
    rep_dict = report.model_dump() if hasattr(report, "model_dump") else dataclasses.asdict(report)
    rep_dict["total_tests"] = report.total_tests
    rep_dict["significant_tests"] = report.significant_tests
    return _cache_and_respond(state, sid, "s03_diagnostics", {"report": rep_dict})


# ---------------------------------------------------------------------------
# Studio 04 — Automated EDA & Expanded 17-Test Hypothesis Lab
# ---------------------------------------------------------------------------
@router.post("/s04/eda")
def s04_run_eda(x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = get_analysis_dataframe(state, _require_df(state))
    targets = get_selected_targets(state, df)
    sample_df, was_sampled = smart_sample_df(df, max_rows=25_000, stratify_col=targets[0] if targets else None)
    eda = AutomatedEDA().run(sample_df, target_column=targets[0] if targets else None, generate_figures=False)
    state["uap_eda_report"] = eda

    num_cols = list(sample_df.select_dtypes(include=[np.number]).columns)[:12]
    distributions: list[dict[str, Any]] = []
    for c in num_cols[:6]:
        vals = sample_df[c].dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) >= 5:
            counts, edges = np.histogram(vals, bins=18)
            distributions.append(
                {
                    "column": c,
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "min": round(float(np.min(vals)), 4),
                    "max": round(float(np.max(vals)), 4),
                    "bins": [{"label": f"{edges[i]:.2g}", "count": int(counts[i])} for i in range(len(counts))],
                }
            )
    corr_matrix: dict[str, Any] = {"columns": [], "values": []}
    if len(num_cols) >= 2:
        cm = sample_df[num_cols].corr(numeric_only=True).fillna(0.0)
        corr_matrix = {"columns": list(cm.columns), "values": np.round(cm.to_numpy(dtype=float), 3).tolist()}

    missing_bars = [
        {"column": str(c), "missing_pct": round(float(df[c].isna().mean() * 100.0), 2)}
        for c in df.columns
        if df[c].isna().any()
    ][:20]
    insights = [i.model_dump() if hasattr(i, "model_dump") else i for i in getattr(eda, "insights", [])]
    record_lineage(state, "EDA", "Completed Automated EDA", f"Numeric={len(num_cols)} | Sampled={was_sampled}")
    return _cache_and_respond(
        state,
        sid,
        "s04_eda",
        {
            "sampled": was_sampled,
            "n_rows_analyzed": len(sample_df),
            "insights": insights,
            "distributions": distributions,
            "correlation_matrix": corr_matrix,
            "missing_bars": missing_bars,
        },
    )


def _run_single_stat_test(
    df: pd.DataFrame,
    test_type: str,
    col_a: str,
    col_b: str | None,
    alpha: float,
) -> dict[str, Any]:
    from scipy import stats as sp_stats

    t = test_type.lower()
    if t == "shapiro":
        res = StatisticalAnalyzer.normality_test(df[col_a].dropna().iloc[:5000], alpha=alpha)
        return res.model_dump() if hasattr(res, "model_dump") else dict(res)
    if t == "ks_normality":
        vals = pd.to_numeric(df[col_a], errors="coerce").dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        z = (vals - np.mean(vals)) / max(float(np.std(vals, ddof=1)), 1e-12)
        stat, pval = sp_stats.kstest(z, "norm")
        return {
            "test_name": "Kolmogorov-Smirnov Normality Test",
            "statistic": round(float(stat), 5),
            "p_value": float(pval),
            "alpha": alpha,
            "significant": bool(pval < alpha),
            "interpretation": f"KS D={stat:.4f}, p={pval:.4g} — {'rejects' if pval < alpha else 'consistent with'} normality.",
        }
    if t == "anderson":
        vals = pd.to_numeric(df[col_a], errors="coerce").dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        try:
            ad = sp_stats.anderson(vals[:5000], dist="norm", method="interpolate")
            pval = float(getattr(ad, "pvalue", 0.25))
            sig = bool(pval < alpha)
            interp = f"A²={ad.statistic:.4f}, p={pval:.4g} — {'non-normal' if sig else 'compatible with Gaussian'}."
        except TypeError:
            ad = sp_stats.anderson(vals[:5000], dist="norm")
            crit_5 = float(ad.critical_values[2])
            sig = bool(ad.statistic > crit_5)
            pval = 0.01 if sig else 0.25
            interp = f"A²={ad.statistic:.4f} vs 5% critical={crit_5:.4f} — {'non-normal' if sig else 'compatible with Gaussian'}."
        return {
            "test_name": "Anderson-Darling Normality Test",
            "statistic": round(float(ad.statistic), 5),
            "p_value": round(pval, 6),
            "alpha": alpha,
            "significant": sig,
            "interpretation": interp,
        }
    if t == "dagostino":
        vals = pd.to_numeric(df[col_a], errors="coerce").dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 8:
            raise HTTPException(status_code=400, detail="D'Agostino-Pearson test requires at least 8 numeric observations.")
        stat, pval = sp_stats.normaltest(vals[:5000])
        return {
            "test_name": "D'Agostino-Pearson Omnibus K²",
            "statistic": round(float(stat), 5),
            "p_value": float(pval),
            "alpha": alpha,
            "significant": bool(pval < alpha),
            "interpretation": f"K²={stat:.4f}, p={pval:.4g} — {'significant skew/kurtosis deviation' if pval < alpha else 'passes omnibus normality'}.",
        }
    if t == "distribution_fit":
        vals = pd.to_numeric(df[col_a], errors="coerce").dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)][:4000]
        if len(vals) < 15:
            raise HTTPException(status_code=400, detail="Distribution fitting needs at least 15 values.")
        candidates = [("Normal", sp_stats.norm), ("Student-t", sp_stats.t), ("Laplace", sp_stats.laplace), ("Logistic", sp_stats.logistic)]
        if float(np.min(vals)) > 0:
            candidates.extend([("Log-Normal", sp_stats.lognorm), ("Exponential", sp_stats.expon), ("Gamma", sp_stats.gamma)])
        fits = []
        for dname, dist in candidates:
            try:
                params = dist.fit(vals)
                ks_s, ks_p = sp_stats.kstest(vals, dist.cdf, args=params)
                fits.append({"distribution": dname, "ks_stat": round(float(ks_s), 5), "p_value": round(float(ks_p), 5), "good_fit": bool(ks_p >= alpha)})
            except Exception:
                continue
        fits.sort(key=lambda r: float(str(r["ks_stat"])))
        best: dict[str, Any] = fits[0] if fits else {"distribution": "Normal", "ks_stat": 0.0, "p_value": 1.0}
        return {
            "test_name": f"Distribution Goodness-of-Fit ({col_a})",
            "statistic": best["ks_stat"],
            "p_value": best["p_value"],
            "alpha": alpha,
            "significant": bool(best["p_value"] < alpha),
            "interpretation": f"Best-fitting family is {best['distribution']} (KS={best['ks_stat']:.4f}, p={best['p_value']:.4g}).",
            "extra_table": fits,
        }
    if t in ("pearson", "spearman", "kendall") and col_b and col_b in df.columns:
        res = StatisticalAnalyzer.correlation_test(df[col_a], df[col_b], method=t, alpha=alpha)  # type: ignore[arg-type]
        return res.model_dump() if hasattr(res, "model_dump") else dict(res)
    if t == "chi_square" and col_b and col_b in df.columns:
        res = StatisticalAnalyzer.chi_square_independence(df[col_a], df[col_b], alpha=alpha)
        return res.model_dump() if hasattr(res, "model_dump") else dict(res)
    if t == "fisher_exact" and col_b and col_b in df.columns:
        ct = pd.crosstab(df[col_a].astype(str), df[col_b].astype(str)).iloc[:2, :2]
        if ct.shape != (2, 2):
            raise HTTPException(status_code=400, detail="Fisher Exact test requires two columns with at least 2 categories each.")
        odds, pval = sp_stats.fisher_exact(ct.to_numpy())
        return {
            "test_name": "Fisher's Exact Test (2x2)",
            "statistic": round(float(odds), 5),
            "p_value": float(pval),
            "alpha": alpha,
            "significant": bool(pval < alpha),
            "interpretation": f"Odds Ratio={odds:.4f}, p={pval:.4g} ({'dependent' if pval < alpha else 'independent'}).",
        }
    if t in ("anova", "kruskal") and col_b and col_b in df.columns:
        res = StatisticalAnalyzer.multi_group_test(df, value_col=col_a, group_col=col_b, parametric=(t == "anova"), alpha=alpha)
        return res.model_dump() if hasattr(res, "model_dump") else dict(res)
    if t in ("levene", "bartlett", "fligner") and col_b and col_b in df.columns:
        groups = [pd.to_numeric(g[col_a], errors="coerce").dropna().to_numpy(dtype=float) for _, g in df.groupby(col_b)]
        groups = [g[np.isfinite(g)] for g in groups if len(g) >= 2]
        if len(groups) < 2:
            raise HTTPException(status_code=400, detail=f"Variance test requires >=2 groups in '{col_b}'.")
        fn = {"levene": sp_stats.levene, "bartlett": sp_stats.bartlett, "fligner": sp_stats.fligner}[t]
        stat, pval = fn(*groups[:12])
        return {
            "test_name": f"{t.title()} Homogeneity of Variance Test",
            "statistic": round(float(stat), 5),
            "p_value": float(pval),
            "alpha": alpha,
            "significant": bool(pval < alpha),
            "interpretation": f"Stat={stat:.4f}, p={pval:.4g} — {'heteroscedastic (unequal variances)' if pval < alpha else 'homoscedastic (equal variances)'}.",
        }
    if t in ("paired_t", "wilcoxon", "ks_2samp") and col_b and col_b in df.columns:
        sub = df[[col_a, col_b]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 5:
            raise HTTPException(status_code=400, detail="Need at least 5 paired numeric rows.")
        a_v, b_v = sub[col_a].to_numpy(dtype=float), sub[col_b].to_numpy(dtype=float)
        if t == "paired_t":
            stat, pval = sp_stats.ttest_rel(a_v, b_v)
            label = "Paired Student t-Test"
        elif t == "wilcoxon":
            diff = a_v - b_v
            if np.allclose(diff, 0):
                stat, pval = 0.0, 1.0
            else:
                stat, pval = sp_stats.wilcoxon(diff)
            label = "Wilcoxon Signed-Rank Test"
        else:
            stat, pval = sp_stats.ks_2samp(a_v, b_v)
            label = "Two-Sample Kolmogorov-Smirnov Test"
        return {
            "test_name": label,
            "statistic": round(float(stat), 5),
            "p_value": float(pval),
            "alpha": alpha,
            "significant": bool(pval < alpha),
            "interpretation": f"Stat={stat:.4f}, p={pval:.4g} ({'significant difference' if pval < alpha else 'no significant difference'}).",
        }
    if col_b and col_b in df.columns:
        uniq = df[col_b].dropna().unique()[:2]
        if len(uniq) < 2:
            raise HTTPException(status_code=400, detail=f"Group column '{col_b}' needs at least 2 distinct values.")
        g1 = df.loc[df[col_b] == uniq[0], col_a].dropna()
        g2 = df.loc[df[col_b] == uniq[1], col_a].dropna()
        res = StatisticalAnalyzer.two_sample_test(g1, g2, parametric=(t != "mann_whitney"), alpha=alpha)
        return res.model_dump() if hasattr(res, "model_dump") else dict(res)
    raise HTTPException(status_code=400, detail="Select both Column A and Column B for two-variable tests.")


@router.post("/s04/hypothesis")
def s04_run_hypothesis(req: HypothesisReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = get_analysis_dataframe(state, _require_df(state))
    mode = req.mode.lower()

    if mode in ("auto", "complete"):
        num_cols = list(df.select_dtypes(include=[np.number]).columns)[:6]
        cat_cols = [c for c in df.columns if c not in num_cols and 2 <= df[c].nunique(dropna=True) <= 15][:4]
        battery: list[dict[str, Any]] = []
        tests_norm = ["shapiro", "dagostino", "anderson", "distribution_fit"] if mode == "complete" else ["shapiro", "distribution_fit"]
        for nc in num_cols[:4]:
            for tt in tests_norm:
                try:
                    r = _run_single_stat_test(df, tt, nc, None, req.alpha)
                    battery.append({"scope": nc, "test_type": tt, **{k: v for k, v in r.items() if k != "extra_table"}})
                except Exception:
                    continue
        if len(num_cols) >= 2:
            corr_tests = ["pearson", "spearman", "kendall", "ks_2samp"] if mode == "complete" else ["spearman"]
            for i in range(min(3, len(num_cols))):
                for j in range(i + 1, min(4, len(num_cols))):
                    for tt in corr_tests:
                        try:
                            r = _run_single_stat_test(df, tt, num_cols[i], num_cols[j], req.alpha)
                            battery.append({"scope": f"{num_cols[i]} vs {num_cols[j]}", "test_type": tt, **{k: v for k, v in r.items() if k != "extra_table"}})
                        except Exception:
                            continue
        for nc in num_cols[:3]:
            for cc in cat_cols[:2]:
                grp_tests = ["anova", "kruskal", "levene"] if mode == "complete" else ["kruskal"]
                for tt in grp_tests:
                    try:
                        r = _run_single_stat_test(df, tt, nc, cc, req.alpha)
                        battery.append({"scope": f"{nc} by {cc}", "test_type": tt, **{k: v for k, v in r.items() if k != "extra_table"}})
                    except Exception:
                        continue
        if len(cat_cols) >= 2:
            try:
                r = _run_single_stat_test(df, "chi_square", cat_cols[0], cat_cols[1], req.alpha)
                battery.append({"scope": f"{cat_cols[0]} x {cat_cols[1]}", "test_type": "chi_square", **{k: v for k, v in r.items() if k != "extra_table"}})
            except Exception:
                pass
        state["uap_hypothesis_battery"] = battery
        return _cache_and_respond(
            state,
            sid,
            "s04_hypothesis",
            {
                "mode": mode,
                "total_tests": len(battery),
                "significant_tests": sum(1 for r in battery if r.get("significant")),
                "battery": battery,
                "result": battery[0] if battery else None,
            },
        )

    if not req.col_a or req.col_a not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.col_a}' not found.")
    result = _run_single_stat_test(df, req.test_type, req.col_a, req.col_b, req.alpha)
    return _cache_and_respond(state, sid, "s04_hypothesis", {"mode": "custom", "result": result})


# ---------------------------------------------------------------------------
# Studio 05 — Feature Engineering & Selection
# ---------------------------------------------------------------------------
@router.post("/s05/engineer")
def s05_engineer_features(req: FeatureEngReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = _require_df(state)
    targets = set(get_selected_targets(state, df))
    excluded = set(get_excluded_columns(state, df))
    feat_cols = [c for c in df.columns if c not in targets and c not in excluded]
    mode = req.mode.lower()
    if mode == "complete":
        cfg = FeatureConfig(
            scale_numeric="none",
            correct_skew=True,
            create_interactions=True,
            create_polynomials=True,
            polynomial_degree=max(2, min(req.polynomial_degree, 3)),
            extract_datetime_features=True,
        )
    elif mode == "auto":
        cfg = FeatureConfig(
            scale_numeric="none",
            correct_skew=True,
            create_interactions=True,
            create_polynomials=False,
            polynomial_degree=2,
            extract_datetime_features=True,
        )
    else:
        cfg = FeatureConfig(
            scale_numeric="none",
            correct_skew=req.correct_skew,
            create_interactions=req.add_interactions,
            create_polynomials=req.add_polynomials,
            polynomial_degree=req.polynomial_degree,
            extract_datetime_features=req.add_datetime_features,
        )
    fe = FeatureEngineeringPipeline(cfg)
    train_df = state.get("active_train_df")
    if isinstance(train_df, pd.DataFrame) and not train_df.empty and all(c in train_df.columns for c in feat_cols):
        fe.fit(train_df[feat_cols])
        transformed = fe.transform(df[feat_cols])
    else:
        transformed = fe.fit_transform(df[feat_cols])
    new_cols = [c for c in transformed.columns if c not in df.columns]
    if req.commit and new_cols:
        merged = pd.concat([df, transformed[new_cols]], axis=1)
        state["uap_s05_engineered"] = True
        update_active_dataset(
            state,
            merged,
            stage="Feature Engineering",
            action=f"Engineered {len(new_cols)} new features ({mode})",
            details=", ".join(new_cols[:8]),
            preserve_stage="s05_features",
        )
    return _cache_and_respond(state, sid, "s05_engineer", {"mode": mode, "new_features": new_cols, "n_created": len(new_cols)})


@router.post("/s05/select")
def s05_select_features(req: FeatureSelReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = get_analysis_dataframe(state, _require_df(state))
    targets = get_selected_targets(state, df)
    target = req.target if (req.target and req.target in df.columns) else (targets[0] if targets else None)
    if not target:
        raise HTTPException(status_code=400, detail="Select a target column before running Feature Selection.")
    X, y, task = _prepare_xy(df, target)
    train_df = state.get("active_train_df")
    if isinstance(train_df, pd.DataFrame) and not train_df.empty and target in train_df.columns:
        X_fit, y_fit, _ = _prepare_xy(train_df, target)
    else:
        X_fit, y_fit = X, y
    task_str = "classification" if task in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION) else "regression"
    mode = req.mode.lower()
    methods = ["mutual_info", "f_test", "importance", "l1"] if mode in ("complete", "auto") else [req.method]

    consensus: dict[str, list[float]] = {c: [] for c in X.columns}
    method_breakdown: list[dict[str, Any]] = []
    selected: list[str] = []
    scores: dict[str, float] = {}

    for m_name in methods:
        try:
            selector = FeatureSelector(method=m_name, task=task_str, max_features=min(req.top_k, X_fit.shape[1]))
            selector.fit(X_fit, y_fit)
            s_dict = dict(selector.scores_)
            max_v = max((abs(float(v)) for v in s_dict.values()), default=1.0) or 1.0
            for k, v in s_dict.items():
                if k in consensus:
                    consensus[k].append(abs(float(v)) / max_v)
            method_breakdown.append({"method": m_name, "top_5": list(selector.selected_features_)[:5]})
            if not selected:
                selected = list(selector.selected_features_)
                scores = {k: float(v) for k, v in s_dict.items()}
        except Exception:
            continue

    if len(methods) > 1:
        scores = {k: round(float(np.mean(vs)), 6) if vs else 0.0 for k, vs in consensus.items()}
        selected = [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: min(req.top_k, len(scores))]]

    ranked = sorted(
        [{"feature": k, "score": round(float(v), 6), "selected": k in selected} for k, v in scores.items()],
        key=lambda r: float(str(r["score"])),
        reverse=True,
    )
    state["uap_fs_selected_cols"] = selected
    state["uap_fs_scores"] = scores
    if req.commit:
        full_df = _require_df(state)
        keep_raw = [c for c in full_df.columns if c in selected or c == target]
        dropped = [c for c in full_df.columns if c not in keep_raw]
        if dropped:
            set_excluded_columns(
                state,
                list(set(get_excluded_columns(state, full_df)) | set(dropped)),
                df=full_df,
            )
            record_exclusion(
                state,
                stage="Studio 05 Feature Selection",
                exclusion_type="Feature Selection Exclusion",
                affected_scope=f"{len(dropped)} unselected columns",
                count=len(dropped),
                total_before=full_df.shape[1],
                criterion_rule=f"mode={mode}, method={req.method}, top_k={req.top_k}",
                reason=f"Feature selection kept top {len(selected)} highest-signal predictors.",
                sample_identifiers=dropped[:10],
            )
    return _cache_and_respond(
        state,
        sid,
        "s05_select",
        {
            "mode": mode,
            "method_breakdown": method_breakdown,
            "selected_features": selected,
            "rankings": ranked,
        },
    )


# ---------------------------------------------------------------------------
# Studio 06 — Multi-Backend AutoML, Paper-Grade Deep Learning & Future Predictions
# ---------------------------------------------------------------------------
@router.get("/s06/backends")
def s06_list_backends() -> dict[str, Any]:
    return to_json_object({"automl_backends": probe_automl_backends()})


@router.post("/s06/automl")
def s06_run_automl(req: AutoMLRunReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = get_analysis_dataframe(state, _require_df(state))
    targets = get_selected_targets(state, df)
    target = req.target if (req.target and req.target in df.columns) else (targets[0] if targets else None)
    if not target:
        raise HTTPException(status_code=400, detail="Select a target column first.")

    def _execute() -> dict[str, Any]:
        X, y, task = _prepare_xy(df, target)
        work_df = X.copy()
        work_df[target] = y.to_numpy()
        res = AutoMLBackendHub.run(
            work_df,
            target_col=target,
            backend=req.backend,
            task_type=task,
            time_budget_seconds=req.time_budget_seconds,
            cv_folds=req.cv_folds,
        )
        eq_dict: dict[str, Any] = {}
        try:
            eq_rep = ModelEquationExtractor.extract(
                res.best_model,
                feature_names=list(X.columns),
                target_name=target,
                X_reference=X.iloc[: min(len(X), 500)],
            )
            eq_dict = eq_rep.model_dump() if hasattr(eq_rep, "model_dump") else dict(eq_rep)
        except Exception as exc:
            logger.debug("equation_extract_skipped", error=str(exc))

        baseline_score = round(float(res.best_score - res.lift_over_best_baseline), 4)
        ranked_lb = []
        for idx_lb, row_lb in enumerate(res.leaderboard or [], start=1):
            r_copy = dict(row_lb)
            r_score = float(r_copy.get("cv_score", r_copy.get("score", 0.0)) or 0.0)
            r_lift = r_score - baseline_score
            r_copy["rank"] = idx_lb
            r_copy["selection_criterion"] = f"Max CV {res.primary_metric} + Lift over Baseline ({baseline_score:.4f})"
            r_copy["lift_over_baseline"] = round(r_lift, 4)
            if idx_lb == 1:
                r_copy["acceptance_status"] = "ACCEPTED (CHAMPION)" if r_lift >= 0 else "ACCEPTED (BEST AVAILABLE)"
                r_copy["rationale"] = (
                    f"Rank #1 out of {len(res.leaderboard or [])} candidates with {res.primary_metric}={r_score:.4f} "
                    f"(lift {r_lift:+.4f} vs strong baseline {baseline_score:.4f})."
                )
            else:
                r_copy["acceptance_status"] = "BENCHMARKED (RUNNER-UP)" if idx_lb <= 3 else "REJECTED (SUBOPTIMAL)"
                r_copy["rationale"] = f"Trailed champion by {res.best_score - r_score:.4f} on {res.primary_metric}."
            ranked_lb.append(r_copy)

        state["uap_active_model"] = res.best_model
        state["uap_active_model_name"] = f"{res.best_model_name} ({res.executed_backend})"
        state["uap_active_model_metrics"] = {
            res.primary_metric: res.best_score,
            "baseline_score": baseline_score,
            "lift_over_baseline": res.lift_over_best_baseline,
        }
        state["uap_active_feature_cols"] = list(X.columns)
        state["uap_active_X_test"] = X.iloc[-min(len(X), 400) :]
        state["uap_active_y_test"] = y.iloc[-min(len(y), 400) :]
        state["uap_automl"] = res
        state["uap_active_model_equation"] = eq_dict
        state["uap_model_equations"] = {target: {res.best_model_name: eq_dict}} if eq_dict else {}
        record_lineage(
            state,
            "AutoML",
            f"Trained {res.executed_backend} ({res.best_model_name})",
            f"{res.primary_metric}={res.best_score:.4f} (Lift {res.lift_over_best_baseline:+.4f})",
        )
        return _cache_and_respond(
            state,
            sid,
            "s06_automl",
            {
                "backend_requested": res.requested_backend,
                "backend_used": res.executed_backend,
                "fallback_reason": res.fallback_reason,
                "task_type": res.task_type.value,
                "primary_metric": res.primary_metric,
                "best_model_name": res.best_model_name,
                "best_score": res.best_score,
                "baseline_score": baseline_score,
                "lift_over_baseline": res.lift_over_best_baseline,
                "elapsed_seconds": res.elapsed_seconds,
                "leaderboard": ranked_lb,
                "equation": eq_dict,
                "metadata": res.extra,
            },
        )

    if req.async_job:
        rec = job_manager.submit(f"AutoML ({req.backend})", _execute)
        return {"job": rec.to_dict()}
    return _execute()


@router.get("/s06/export-formula")
def s06_export_external_formula(
    model_id: str | None = None,
    x_session_id: str | None = Header(default="default"),
) -> StreamingResponse:
    """Download a self-contained zero-dependency Python deployment file (.py) containing
    the exact standalone prediction function plus Excel, SQL, C/MATLAB, and LaTeX formulas."""
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    eq: dict[str, Any] | None = None

    if model_id:
        catalog = SupervisoryPredictionMonitor.catalog_available_models(state)
        entry = next((m for m in catalog if m["model_id"] == model_id), None)
        ref_df = state.get("uap_df") if isinstance(state.get("uap_df"), pd.DataFrame) else None
        eq = SupervisoryPredictionMonitor.extract_model_equation(entry, ref_df=ref_df)
    if not eq:
        eq = state.get("uap_active_model_equation")
    if not eq and state.get("uap_active_model") is not None:
        catalog = SupervisoryPredictionMonitor.catalog_available_models(state)
        ref_df = state.get("uap_df") if isinstance(state.get("uap_df"), pd.DataFrame) else None
        eq = SupervisoryPredictionMonitor.extract_model_equation(catalog[0] if catalog else None, ref_df=ref_df)
    if not eq:
        raise HTTPException(status_code=400, detail="No trained model equation available to export yet. Run Studio 03, Studio 06, or Studio 14 first.")

    model_name = str(eq.get("model_name") or "omniforge_model")
    target_name = str(eq.get("target_name") or "target")
    py_code = str(eq.get("formula_python") or "")
    excel_code = str(eq.get("formula_excel") or "")
    sql_code = str(eq.get("formula_sql") or "")
    cm_code = str(eq.get("formula_c_matlab") or "")
    latex_code = str(eq.get("formula_latex") or "")
    plain_eq = str(eq.get("surrogate_plain_equation") or eq.get("formula_plain") or "")
    summary = str(eq.get("external_usage_summary") or "")
    r2_val = eq.get("surrogate_fidelity_r2")
    rules = str(eq.get("decision_rules_text") or "")

    script_parts = [
        'r"""',
        "OmniForge — Standalone External Deployment Equation & Scoring Module",
        f"Model:  {model_name}",
        f"Target: {target_name}",
        f"Type:   {eq.get('equation_type', 'analytical')}",
        f"Summary: {summary}",
        f"Surrogate Fidelity R^2: {r2_val if r2_val is not None else 'Exact (1.0000)'}",
        "",
        "Plain Analytical Relationship:",
        f"  {plain_eq}",
        "",
        "LaTeX Relationship:",
        f"  {latex_code}",
        "",
        "Microsoft Excel Formula:",
        f"  {excel_code}",
        "",
        "ANSI SQL Expression:",
        "\n".join(f"  {ln}" for ln in sql_code.splitlines()),
        "",
        "MATLAB / C / PLC Structured-Text Formula:",
        "\n".join(f"  {ln}" for ln in cm_code.splitlines()),
    ]
    if rules:
        script_parts.extend(["", "Representative Decision Tree Rules:", rules])
    script_parts.extend(['"""', "", py_code, ""])
    payload_bytes = "\n".join(script_parts).encode("utf-8")
    safe_fname = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in f"omniforge_{target_name}_{model_name}")[:60]
    return StreamingResponse(
        io.BytesIO(payload_bytes),
        media_type="text/x-python",
        headers={"Content-Disposition": f'attachment; filename="{safe_fname}_external.py"'},
    )


@router.post("/s06/nonparametric")
def s06_run_nonparametric_analytical(
    req: NonParametricRunReq, x_session_id: str | None = Header(default="default")
) -> dict[str, Any]:
    """Benchmark all Non-Parametric Analytical Models (Spline GAM, Nadaraya-Watson, RBF Kernel Ridge, Weighted k-NN, Isotonic) vs. Parametric Linear."""
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = get_analysis_dataframe(state, _require_df(state))
    targets = get_selected_targets(state, df)
    target = req.target if (req.target and req.target in df.columns) else (targets[0] if targets else None)
    if not target:
        raise HTTPException(status_code=400, detail="Select a target column first.")
    X, y, task = _prepare_xy(df, target)
    if len(X) < 12 or X.shape[1] < 1:
        raise HTTPException(status_code=400, detail="Non-parametric analytical modeling requires at least 12 rows and 1 predictor.")

    res = NonParametricAnalyticalSuite.run(
        X,
        y,
        task_type=task,
        target_name=target,
        cv_folds=req.cv_folds,
        models=req.models,
        n_knots=req.n_knots,
        spline_degree=req.spline_degree,
        bandwidth=req.bandwidth,
        n_neighbors=req.n_neighbors,
        rbf_gamma=req.rbf_gamma,
    )
    fitted_map = res.pop("_fitted_models", {})
    feat_cols = list(X.columns)

    np_store = state.setdefault("uap_nonparametric_models", {})
    tgt_map: dict[str, Any] = {}
    lb_lookup = {r["model_name"]: r for r in (res.get("leaderboard") or [])}
    for m_name, m_obj in fitted_map.items():
        if m_name.startswith("Parametric_"):
            continue
        row_m = lb_lookup.get(m_name, {})
        tgt_map[m_name] = {
            "model": m_obj,
            "feature_cols": feat_cols,
            "metrics": {
                str(row_m.get("primary_metric", "score")): row_m.get("primary_score", 0.0),
                "lift_vs_parametric": row_m.get("lift_vs_parametric", 0.0),
            },
        }
    np_store[target] = tgt_map

    # Promote champion non-parametric model as active if no champion model exists yet
    champ_name = str(res.get("champion_model") or "")
    champ_obj = fitted_map.get(champ_name)
    if champ_obj is not None and state.get("uap_active_model") is None:
        state["uap_active_model"] = champ_obj
        state["uap_active_model_name"] = champ_name
        state["uap_active_model_metrics"] = {
            "score": float(res.get("champion_score", 0.0)),
            "parametric_reference": float(res.get("parametric_reference_score", 0.0)),
        }
        state["uap_active_feature_cols"] = feat_cols
        state["uap_active_X_test"] = X.iloc[-min(len(X), 400) :]
        state["uap_active_y_test"] = y.iloc[-min(len(y), 400) :]

    record_lineage(
        state,
        "Non-Parametric Modeling",
        f"Evaluated {len(tgt_map)} Non-Parametric Analytical Models vs. Parametric Linear",
        f"Champion={champ_name} ({res.get('champion_score', 0.0):.4f})",
    )
    return _cache_and_respond(state, sid, "s06_nonparametric", dict(res))


@router.post("/s06/deep-learning")
def s06_run_deep_benchmark(req: DeepBenchReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    """Benchmark Paper-Grade Deep Learning Architectures (FT-Transformer, TabNet, ResNet-Tabular, MLP, 1D-CNN, GNN) vs. Baselines."""
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = get_analysis_dataframe(state, _require_df(state))
    targets = get_selected_targets(state, df)
    target = req.target if (req.target and req.target in df.columns) else (targets[0] if targets else None)
    if not target:
        raise HTTPException(status_code=400, detail="Select a target column first.")
    X_full, y_full, task = _prepare_xy(df, target)
    if len(X_full) > req.max_rows:
        idx = np.random.default_rng(42).choice(len(X_full), size=req.max_rows, replace=False)
        X_full = X_full.iloc[idx].reset_index(drop=True)
        y_full = y_full.iloc[idx].reset_index(drop=True)
    split = max(10, int(len(X_full) * 0.8))
    X_tr, X_te = X_full.iloc[:split], X_full.iloc[split:]
    y_tr, y_te = y_full.iloc[:split], y_full.iloc[split:]
    is_clf = task in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)

    base_suite = BaselineBenchmarkSuite()
    base_res = base_suite.run_classification(X_tr, y_tr, X_te, y_te) if is_clf else base_suite.run_regression(X_tr, y_tr, X_te, y_te)
    rows: list[dict[str, Any]] = []
    for bname, bm in base_res.items():
        sc = float(bm.get("f1_weighted", bm.get("accuracy", 0.0))) if is_clf else float(bm.get("r2", 0.0))
        rows.append({"model": f"Baseline: {bname}", "family": "Baseline", "framework": "sklearn", "score": round(sc, 5), "elapsed_sec": 0.01, "loss_curve": []})

    arch_map = {
        "ft_transformer": lambda: FTTransformerModel(d_model=24, n_heads=4, n_layers=2, epochs=req.epochs, task_type=task, track_carbon=False),
        "tabnet": lambda: TabNetModel(n_d=16, n_a=16, n_steps=3, epochs=req.epochs, task_type=task, track_carbon=False),
        "resnet_tabular": lambda: ResNetTabularModel(d_hidden=64, n_blocks=3, epochs=req.epochs, task_type=task, track_carbon=False),
        "mlp": lambda: MLPModel(hidden_dims=(64, 32), epochs=req.epochs, task_type=task, track_carbon=False),
        "cnn1d": lambda: CNN1DModel(epochs=req.epochs, task_type=task, track_carbon=False),
        "gnn_gcn": lambda: GraphNeuralNetworkModel(architecture="gcn", hidden_dim=24, epochs=req.epochs, task_type=task, track_carbon=False),
        "gnn_gat": lambda: GraphNeuralNetworkModel(architecture="gat", hidden_dim=24, epochs=req.epochs, task_type=task, track_carbon=False),
        "gnn_graphsage": lambda: GraphNeuralNetworkModel(architecture="graphsage", hidden_dim=24, epochs=req.epochs, task_type=task, track_carbon=False),
    }

    from sklearn.metrics import f1_score, r2_score

    fitted_dl: dict[str, Any] = {}
    feature_importances: dict[str, list[dict[str, Any]]] = {}
    for arch in req.architectures:
        builder = arch_map.get(arch.lower())
        if builder is None:
            continue
        t0 = time.perf_counter()
        try:
            m = builder()
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)
            sc = float(f1_score(y_te, preds, average="weighted", zero_division=0)) if is_clf else float(r2_score(y_te, preds))
            elapsed = round(time.perf_counter() - t0, 3)
            loss_curve = [round(float(v), 5) for v in getattr(m, "loss_history_", [])[-40:]]
            rows.append({"model": m.name, "family": "Deep Learning", "framework": getattr(m, "framework", "pytorch"), "score": round(sc, 5), "elapsed_sec": elapsed, "loss_curve": loss_curve})
            fitted_dl[arch] = {
                "model": m,
                "name": m.name,
                "feature_cols": list(X_tr.columns),
                "metrics": {"f1_weighted" if is_clf else "r2": round(sc, 5), "elapsed_sec": elapsed},
                "target": target,
            }
            _fi_raw = getattr(m, "feature_importances_", None)
            if _fi_raw is not None:
                fi = np.asarray(_fi_raw, dtype=float)
                top_i = np.argsort(fi)[::-1][:10]
                feature_importances[m.name] = [
                    {"feature": str(X_tr.columns[int(i)]), "importance": round(float(fi[int(i)]), 5)}
                    for i in top_i
                    if int(i) < len(X_tr.columns)
                ]
        except Exception as exc:
            rows.append({"model": arch, "family": "Deep Learning", "framework": "error", "score": 0.0, "elapsed_sec": round(time.perf_counter() - t0, 3), "error": str(exc), "loss_curve": []})

    rows.sort(key=lambda r: r["score"], reverse=True)
    state["uap_dl_benchmark_df"] = pd.DataFrame([{k: v for k, v in r.items() if k != "loss_curve"} for r in rows])
    state["uap_dl_models"] = fitted_dl
    record_lineage(state, "Deep Learning", f"Benchmarked {len(fitted_dl)} paper-grade architectures", f"Top={rows[0]['model']} ({rows[0]['score']:.4f})")
    return _cache_and_respond(
        state,
        sid,
        "s06_dl",
        {
            "metric": "f1_weighted" if is_clf else "r2",
            "results": rows,
            "feature_importances": feature_importances,
        },
    )


@router.post("/s06/predict-single")
def s06_predict_single(req: SinglePredictReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    model = state.get("uap_active_model")
    feat_cols = state.get("uap_active_feature_cols") or []
    if model is None or not feat_cols:
        raise HTTPException(status_code=400, detail="Train a model in Studio 06 first before scoring.")
    row: dict[str, Any] = {}
    for c in feat_cols:
        val = req.record.get(c, 0.0)
        if val is None or val == "":
            row[c] = 0.0
        else:
            try:
                fval = float(val)
                row[c] = fval if math.isfinite(fval) else 0.0
            except (TypeError, ValueError):
                row[c] = 0.0
    row_df = pd.DataFrame([row])
    pred = model.predict(row_df)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(row_df)[0].tolist()
        except Exception:
            proba = None
    return _cache_and_respond(
        state,
        sid,
        "s06_predict_single",
        {"prediction": pred, "probabilities": proba, "model": state.get("uap_active_model_name")},
        include_state=False,
    )


@router.post("/s06/predict-batch")
async def s06_predict_batch(
    file: UploadFile = File(...),
    x_session_id: str | None = Header(default="default"),
) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    model = state.get("uap_active_model")
    feat_cols = state.get("uap_active_feature_cols") or []
    if model is None or not feat_cols:
        raise HTTPException(status_code=400, detail="Train a model in Studio 06 before batch scoring.")
    raw = await file.read()
    inc_df = pd.read_csv(io.BytesIO(raw))
    ref_df = state.get("uap_df")
    target = (get_selected_targets(state, ref_df) or ["target"])[0]
    aligned = pd.DataFrame(
        {
            c: pd.to_numeric(inc_df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if c in inc_df.columns
            else 0.0
            for c in feat_cols
        }
    )
    preds = model.predict(aligned)
    scored = inc_df.copy()
    scored[f"predicted_{target}"] = preds
    state["uap_future_prediction_result"] = {"n_rows": len(scored), "preview": scored.head(100).to_dict(orient="records"), "scored_df": scored}
    return _cache_and_respond(
        state,
        sid,
        "s06_predict_batch",
        {"n_rows": len(scored), "columns": list(scored.columns), "preview": scored.head(100).to_dict(orient="records")},
        include_state=False,
    )


# ---------------------------------------------------------------------------
# Studio 07 — Responsible AI (Explainability, Fairness & Governance)
# ---------------------------------------------------------------------------
@router.post("/s07/explain")
def s07_explain(req: ExplainReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = get_session_state(_sid(x_session_id))
    model = state.get("uap_active_model")
    X_te = state.get("uap_active_X_test")
    y_te = state.get("uap_active_y_test")
    if model is None or not isinstance(X_te, pd.DataFrame) or X_te.empty:
        raise HTTPException(status_code=400, detail="Train a model in Studio 06 first to inspect explanations.")
    eng = InterpretabilityEngine()
    perm = eng.permutation_importance(model, X_te, y_te, n_repeats=5)
    local_idx = max(0, min(req.row_index, len(X_te) - 1))
    local_shap = eng.explain_shap_local(model, X_te, X_te.iloc[[local_idx]], n_samples=48)
    pdp_col = req.feature_for_pdp if (req.feature_for_pdp and req.feature_for_pdp in X_te.columns) else str(X_te.columns[0])
    pdp = eng.partial_dependence(model, X_te, feature=pdp_col, grid_resolution=15)
    state["uap_shap_result"] = {"global": perm, "local": local_shap, "pdp": pdp}
    payload = to_json_object({"global_importance": perm, "local_explanation": local_shap, "pdp": pdp, "pdp_feature": pdp_col})
    save_studio_result(state, "s07_explain", payload)
    return payload


@router.post("/s07/fairness")
def s07_fairness(req: FairnessReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = get_session_state(_sid(x_session_id))
    full_df = _require_df(state)
    df = get_analysis_dataframe(state, full_df)
    model = state.get("uap_active_model")
    X_te = state.get("uap_active_X_test")
    y_te = state.get("uap_active_y_test")
    sens_source = df if req.sensitive_col in df.columns else full_df
    if model is None or X_te is None or req.sensitive_col not in sens_source.columns:
        raise HTTPException(status_code=400, detail="Train a model and select a valid sensitive attribute column.")
    sens = sens_source[req.sensitive_col].reindex(X_te.index).fillna("Missing").astype(str).to_numpy()
    preds = np.asarray(model.predict(X_te))
    y_arr = np.asarray(y_te)
    if np.issubdtype(y_arr.dtype, np.number) and len(np.unique(y_arr)) > 2:
        med = float(np.median(y_arr))
        y_bin = (y_arr >= med).astype(int)
        p_bin = (preds.astype(float) >= med).astype(int)
    else:
        pos_val = np.unique(y_arr)[-1]
        y_bin = (y_arr == pos_val).astype(int)
        p_bin = (preds == pos_val).astype(int)
    report = FairnessAuditor().audit(y_bin, p_bin, sens, attribute_name=req.sensitive_col)
    state["uap_fairness_result"] = report.model_dump()
    meta = getattr(model, "metadata_", None)
    if not isinstance(meta, ModelMetadata):
        meta = ModelMetadata(
            name=str(state.get("uap_active_model_name") or "OmniForge Model"),
            model_family="automl",
            framework="omniforge",
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
    card = ModelCardGenerator.build_model_card(
        meta,
        evaluation_metrics=state.get("uap_active_model_metrics") or {},
        fairness_report=report,
    )
    state["uap_model_card"] = card.to_markdown()
    payload = to_json_object({"fairness": report.model_dump(), "model_card_markdown": card.to_markdown()})
    save_studio_result(state, "s07_fairness", payload)
    return payload


# ---------------------------------------------------------------------------
# Studio 08 — Forecasting (N-BEATS / PatchTST / TFT), Causal, Optimization & Simulation
# ---------------------------------------------------------------------------
@router.post("/s08/forecast")
def s08_forecast(req: ForecastReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = get_analysis_dataframe(state, _require_df(state))
    if req.target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.target_col}' not found.")
    work = df.sort_values(req.time_col).reset_index(drop=True) if (req.time_col and req.time_col in df.columns) else df.reset_index(drop=True)
    y_series = pd.to_numeric(work[req.target_col], errors="coerce").dropna().reset_index(drop=True)
    if len(y_series) < 20:
        raise HTTPException(status_code=400, detail="Forecasting requires at least 20 numeric observations.")

    classical_res = ForecastingEngine().forecast(
        work.iloc[: min(len(work), 2000)],
        target_col=req.target_col,
        time_col=req.time_col if (req.time_col and req.time_col in work.columns) else None,
        horizon=req.horizon,
    )
    deep_f = DeepForecaster(architecture=req.deep_arch, epochs=req.epochs)
    deep_f.fit(y_series)
    deep_df = deep_f.predict(req.horizon)

    hist_tail = y_series.tail(min(80, len(y_series))).to_list()
    cl_dump = classical_res.model_dump() if hasattr(classical_res, "model_dump") else classical_res
    state["uap_forecast_report"] = {"classical": cl_dump, "deep_arch": req.deep_arch, "deep_forecast": deep_df.to_dict(orient="records")}
    record_lineage(state, "Forecasting", f"Ran {req.deep_arch.upper()} + Classical Tournament", f"Horizon={req.horizon}")
    return _cache_and_respond(
        state,
        sid,
        "s08_forecast",
        {
            "history": [{"step": i - len(hist_tail) + 1, "actual": round(float(v), 5)} for i, v in enumerate(hist_tail)],
            "deep_arch": req.deep_arch,
            "deep_forecast": deep_df.to_dict(orient="records"),
            "classical_summary": cl_dump,
        },
    )


@router.post("/s08/causal")
def s08_causal(req: CausalReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = get_analysis_dataframe(state, _require_df(state))
    confs = [c for c in req.confounders if c in df.columns and c not in (req.treatment_col, req.outcome_col)]
    if not confs:
        confs = [c for c in df.select_dtypes(include=[np.number]).columns if c not in (req.treatment_col, req.outcome_col)][:5]
    res = CausalInferenceEngine().estimate_ate(
        df,
        treatment_col=req.treatment_col,
        outcome_col=req.outcome_col,
        confounders=confs,
        method=req.method,
    )
    out = res.model_dump() if hasattr(res, "model_dump") else res
    state["uap_causal_result"] = out
    return _cache_and_respond(state, sid, "s08_causal", {"causal": out})


@router.post("/s08/optimize")
def s08_optimize(req: OptimizeReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = get_session_state(_sid(x_session_id))
    df = get_analysis_dataframe(state, _require_df(state))
    cols = [c for c in req.asset_cols if c in df.columns] or list(df.select_dtypes(include=[np.number]).columns)[:5]
    if len(cols) < 2:
        raise HTTPException(status_code=400, detail="Portfolio optimization needs at least 2 numeric columns.")
    sub = df[cols].dropna()
    returns = sub.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 5:
        returns = (sub - sub.mean()) / sub.std().replace(0, 1.0)
    mu = returns.mean().to_numpy(dtype=float)
    cov = returns.cov().to_numpy(dtype=float) + np.eye(len(cols)) * 1e-6
    res = OptimizationEngine().optimize_portfolio(mu, cov, risk_aversion=req.risk_aversion, asset_names=cols)
    out = res.model_dump() if hasattr(res, "model_dump") else res
    state["uap_opt_result"] = out
    payload = to_json_object({"optimization": out})
    save_studio_result(state, "s08_optimize", payload)
    return payload


@router.post("/s08/simulate")
def s08_simulate(req: SimulateReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = get_session_state(_sid(x_session_id))
    df = get_analysis_dataframe(state, _require_df(state))
    if req.target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.target_col}' not found.")
    vals = pd.to_numeric(df[req.target_col], errors="coerce").dropna().to_numpy(dtype=float)
    mu, sigma = float(np.mean(vals)), max(float(np.std(vals)), 1e-6)
    sim_res, _samples = MonteCarloSimulator(n_simulations=req.n_simulations).run(
        input_specs={
            "base": {"dist": "normal", "mean": mu, "std": sigma},
            "drift": {"dist": "normal", "mean": 0.0, "std": sigma * 0.25},
        },
        model_fn=lambda s: s["base"] + np.sqrt(req.horizon_steps) * s["drift"],
    )
    out = sim_res.model_dump()
    state["uap_sim_result"] = out
    payload = to_json_object({"simulation": out})
    save_studio_result(state, "s08_simulate", payload)
    return payload


# ---------------------------------------------------------------------------
# Studio 09 — Specialized AI (Extended NLP, Vision Foundation Hub, Full Graph Link Prediction, RecSys, Survival)
# ---------------------------------------------------------------------------
@router.get("/s09/backends")
def s09_backends() -> dict[str, Any]:
    return to_json_object({"vision": probe_vision_backends(), "nlp": probe_nlp_backends()})


@router.post("/s09/nlp")
def s09_run_nlp(req: NLPReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = get_session_state(_sid(x_session_id))
    suite = AdvancedNLPSuite(use_pretrained=req.use_pretrained)
    texts = [t for t in req.custom_texts if t.strip()]
    if not texts and req.text_col:
        raw_df = state.get("uap_df")
        df = get_analysis_dataframe(state, raw_df) if isinstance(raw_df, pd.DataFrame) else None
        if isinstance(df, pd.DataFrame) and req.text_col in df.columns:
            texts = df[req.text_col].dropna().astype(str).head(250).tolist()
    if not texts:
        texts = [
            "Quarterly revenue grew 18% with strong customer trust and reliable cloud delivery.",
            "Security alert: user john.doe@example.com reported a potential fraud breach from 192.168.1.44.",
            "Forecast anticipates rapid expansion across European markets next quarter with record momentum.",
            "Customer churn increased after terrible service outage and unacceptable billing delays.",
            "Quarterly revenue rose 18% driven by strong customer trust and reliable cloud operations.",
        ]

    t = req.task.lower()
    if t == "zero_shot":
        payload = to_json_object({"task": t, "results": suite.zero_shot_classify(texts[:50], req.candidate_labels)})
    elif t == "qa":
        ctx = req.context.strip() or " ".join(texts[:10])
        q = req.question.strip() or "What grew by 18%?"
        payload = to_json_object({"task": t, "result": suite.answer_question(q, ctx)})
    elif t == "semantic_search":
        q = req.query.strip() or "revenue growth and customer trust"
        payload = to_json_object({"task": t, "results": suite.semantic_search(q, texts[:200], top_k=8)})
    elif t == "emotions":
        payload = to_json_object({"task": t, "results": suite.detect_emotions_and_aspects(texts[:50], aspects=req.aspects or ["revenue", "service", "security"])})
    elif t == "pii_toxicity":
        payload = to_json_object({"task": t, "results": suite.screen_toxicity_and_pii(texts[:50])})
    elif t == "topics":
        payload = to_json_object({"task": t, "lda_topics": suite.extract_topics_lda(texts, n_topics=req.n_topics), "nmf_topics": suite.discover_topics(texts, n_topics=req.n_topics)})
    elif t == "cluster":
        payload = to_json_object({"task": t, "clustering": suite.cluster_documents(texts[:150], n_clusters=req.n_clusters), "near_duplicates": suite.find_near_duplicates(texts[:150])})
    else:
        payload = to_json_object(
            {
                "task": "overview",
                "n_documents": len(texts),
                "emotions": suite.detect_emotions_and_aspects(texts[:20]),
                "languages": suite.detect_language(texts[:20]),
                "pii_toxicity": suite.screen_toxicity_and_pii(texts[:20]),
                "zero_shot": suite.zero_shot_classify(texts[:15], req.candidate_labels),
                "topics": suite.extract_topics_lda(texts, n_topics=min(req.n_topics, max(1, len(texts) // 2))),
                "near_duplicates": suite.find_near_duplicates(texts[:100]),
                "summary": suite.summarize(" ".join(texts[:15]), max_sentences=3),
                "entities": suite.extract_entities(" ".join(texts[:15]))[:25],
            }
        )
    save_studio_result(state, "s09_nlp", payload)
    return payload


@router.post("/s09/vision")
async def s09_run_vision(
    file: UploadFile | None = File(default=None),
    candidate_labels: str = Form(default="sky, forest, building, document, dark, industrial"),
    n_segments: int = Form(default=4),
    backbone: str = Form(default="resnet18"),
    x_session_id: str | None = Header(default="default"),
) -> dict[str, Any]:
    state = get_session_state(_sid(x_session_id))
    hub = VisionFoundationHub()
    if file is not None:
        raw = await file.read()
        from PIL import Image

        img_arr = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    else:
        img_arr = np.zeros((160, 160, 3), dtype=np.uint8)
        img_arr[:70, :, :] = [60, 130, 220]
        img_arr[70:, :, :] = [45, 145, 70]
        img_arr[45:115, 45:115, :] = [235, 95, 65]
    labels = [s.strip() for s in candidate_labels.split(",") if s.strip()]
    zs = hub.zero_shot_classify(img_arr, labels)
    emb = hub.extract_foundation_embedding(img_arr, backbone=backbone, pretrained=False)
    dets = hub.detect_objects(img_arr)
    segs = hub.segment_image(img_arr, n_segments=n_segments)
    payload = to_json_object(
        {
            "image_shape": list(img_arr.shape),
            "zero_shot": zs,
            "embedding": {"backend": emb["backend"], "dim": emb["embedding_dim"], "sample_values": [round(float(v), 4) for v in emb["embedding"][:16]]},
            "detections": dets,
            "segmentation": segs,
            "backends": probe_vision_backends(),
        }
    )
    save_studio_result(state, "s09_vision", payload)
    return payload


@router.post("/s09/graph")
def s09_run_graph(req: GraphReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = get_session_state(_sid(x_session_id))
    raw_df = state.get("uap_df")
    df = get_analysis_dataframe(state, raw_df) if isinstance(raw_df, pd.DataFrame) else None
    suite = GraphLinkPredictionSuite()
    if isinstance(df, pd.DataFrame) and req.source_col and req.target_col and req.source_col in df.columns and req.target_col in df.columns:
        edge_df = df[[req.source_col, req.target_col]].dropna().head(400)
        g = suite.build_graph(edge_df, source_col=req.source_col, target_col=req.target_col)
    else:
        g = nx.karate_club_graph()
        g = nx.relabel_nodes(g, {n: f"Node_{n:02d}" for n in g.nodes()})

    centrality = suite.compute_centralities(g).head(30).to_dict(orient="records")
    comm_info = suite.detect_communities(g)
    node_to_comm = comm_info.get("node_to_community", {}) if isinstance(comm_info, dict) else {}
    structure = suite.analyze_structure_and_influence(g, top_influencers=6)
    topo_links = suite.predict_links_advanced(g, method=req.link_method, top_k=req.top_k)

    supervised_res: dict[str, Any] | None = None
    if req.run_supervised and g.number_of_edges() >= 8:
        try:
            supervised_res = suite.train_supervised_link_predictor(g, edge_operator=req.edge_operator, top_k=req.top_k)
        except Exception as exc:
            supervised_res = {"error": str(exc)}

    neural_res: dict[str, Any] | None = None
    if req.run_neural_gae and 5 <= g.number_of_nodes() <= 500:
        try:
            neural_res = suite.predict_links_neural_gae(g, variational=True, epochs=50, top_k=req.top_k)
        except Exception as exc:
            neural_res = {"error": str(exc)}

    sub_nodes = list(g.nodes())[:120]
    sub_g = g.subgraph(sub_nodes)
    pos = nx.spring_layout(sub_g, seed=42)
    nodes_vis = [
        {"id": str(n), "x": round(float(pos[n][0]), 4), "y": round(float(pos[n][1]), 4), "degree": int(sub_g.degree(n)), "community": int(node_to_comm.get(n, 0))}
        for n in sub_g.nodes()
    ]
    edges_vis = [{"source": str(u), "target": str(v)} for u, v in list(sub_g.edges())[:400]]

    payload = to_json_object(
        {
            "structure": structure,
            "centrality": centrality,
            "topological_links": topo_links,
            "supervised_link_prediction": supervised_res,
            "neural_gae_link_prediction": neural_res,
            "visualization": {"nodes": nodes_vis, "edges": edges_vis},
        }
    )
    save_studio_result(state, "s09_graph", payload)
    return payload


@router.post("/s09/recommend")
def s09_recommend(req: RecommendReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = get_session_state(_sid(x_session_id))
    raw_df = state.get("uap_df")
    df = get_analysis_dataframe(state, raw_df) if isinstance(raw_df, pd.DataFrame) else None
    if isinstance(df, pd.DataFrame) and req.user_col and req.item_col and req.rating_col and all(c in df.columns for c in (req.user_col, req.item_col, req.rating_col)):
        interactions = df[[req.user_col, req.item_col, req.rating_col]].dropna().head(5000).rename(columns={req.user_col: "user_id", req.item_col: "item_id", req.rating_col: "rating"})
    else:
        rng = np.random.default_rng(42)
        interactions = pd.DataFrame(
            {
                "user_id": [f"U{i}" for i in rng.integers(1, 16, size=120)],
                "item_id": [f"Item_{j}" for j in rng.integers(1, 21, size=120)],
                "rating": rng.uniform(2.5, 5.0, size=120).round(2),
            }
        )
    rec = HybridRecommender().fit(interactions)
    sample_user = str(interactions["user_id"].iloc[0])
    recs = rec.recommend(sample_user, top_k=req.top_k)
    payload = to_json_object({"user_id": sample_user, "recommendations": recs})
    save_studio_result(state, "s09_recommend", payload)
    return payload


@router.post("/s09/survival")
def s09_survival(req: SurvivalReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = get_session_state(_sid(x_session_id))
    df = get_analysis_dataframe(state, _require_df(state))
    dur = pd.to_numeric(df[req.duration_col], errors="coerce").fillna(1.0).clip(lower=0.01).to_numpy(dtype=float)
    evt = (pd.to_numeric(df[req.event_col], errors="coerce").fillna(0.0) > 0).astype(int).to_numpy()
    km = KaplanMeierEstimator().fit(dur, evt)
    curve = [
        {
            "time": round(float(r["time"]), 4),
            "survival": round(float(r["survival_prob"]), 5),
            "ci_lower": round(float(r["ci_lower"]), 5),
            "ci_upper": round(float(r["ci_upper"]), 5),
        }
        for _, r in km.survival_table_.iterrows()
    ]
    cox_summary: dict[str, Any] = {}
    num_covs = [c for c in df.select_dtypes(include=[np.number]).columns if c not in (req.duration_col, req.event_col)][:6]
    if num_covs:
        try:
            cox_df = df[[req.duration_col, req.event_col, *num_covs]].copy()
            cox_df[req.duration_col] = dur
            cox_df[req.event_col] = evt
            for c in num_covs:
                cox_df[c] = pd.to_numeric(cox_df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            cox = CoxPHModel().fit(cox_df, duration_col=req.duration_col, event_col=req.event_col, feature_cols=num_covs)
            cox_summary = {
                "hazard_ratios": dict(cox.hazard_ratios_),
                "concordance_index": round(float(cox.concordance_index_), 5),
            }
        except Exception as exc:
            cox_summary = {"error": str(exc)}
    payload = to_json_object({"km_curve": curve[:120], "median_survival": getattr(km, "median_survival_time_", None), "cox": cox_summary})
    save_studio_result(state, "s09_survival", payload)
    return payload


# ---------------------------------------------------------------------------
# Studio 10 — Visual DAG Pipeline Builder
# ---------------------------------------------------------------------------
@router.post("/s10/run")
def s10_run_dag(req: DAGRunReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = get_analysis_dataframe(state, _require_df(state))
    targets = get_selected_targets(state, df)
    target = req.target if (req.target and req.target in df.columns) else (targets[0] if targets else None)

    builder = PipelineBuilder(name="OmniForge_DAG_Workflow")
    prev: str | None = None
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    for idx, stage in enumerate(req.stages):
        node_id = f"step_{idx + 1}_{stage}"
        if stage == "quality":
            builder.add_step(node_id, lambda ctx: {"quality_score": DataQualityProfiler().profile(ctx.get("data")).overall_score}, depends_on=[prev] if prev else [])
        elif stage == "clean":
            builder.add_step(node_id, lambda ctx: {"rows_after": len(DataCleaner().fit_transform(ctx.get("data"))[0])}, depends_on=[prev] if prev else [])
        elif stage == "eda":
            builder.add_step(node_id, lambda ctx: {"n_rows": len(ctx.get("data")), "n_cols": ctx.get("data").shape[1]}, depends_on=[prev] if prev else [])
        elif stage == "features":
            builder.add_step(node_id, lambda ctx: {"numeric_cols": len(ctx.get("data").select_dtypes(include=[np.number]).columns)}, depends_on=[prev] if prev else [])
        elif stage == "automl" and target:
            def _run_fast(ctx: Any, tgt: str = target) -> dict[str, Any]:
                X_d, y_d, t_d = _prepare_xy(ctx.get("data"), tgt)
                w_df = X_d.copy()
                w_df[tgt] = y_d.to_numpy()
                r = AutoMLBackendHub.run(w_df, target_col=tgt, backend="multi_layer_stacker", task_type=t_d, time_budget_seconds=15.0)
                return {"best_model": r.best_model_name, "score": r.best_score, "lift": r.lift_over_best_baseline}

            builder.add_step(node_id, _run_fast, depends_on=[prev] if prev else [])
        else:
            builder.add_step(node_id, lambda ctx, s=stage: {"stage": s, "status": "ok"}, depends_on=[prev] if prev else [])
        nodes.append({"id": node_id, "label": stage.upper(), "order": idx})
        if prev:
            edges.append({"source": prev, "target": node_id})
        prev = node_id

    dag = builder.build()
    exec_res = PipelineExecutor().execute(dag, initial_artifacts={"data": df})
    state["uap_dag_result"] = exec_res
    dump = exec_res.model_dump() if hasattr(exec_res, "model_dump") else (dataclasses.asdict(exec_res) if dataclasses.is_dataclass(exec_res) else {"run_id": getattr(exec_res, "run_id", ""), "step_outputs": getattr(exec_res, "step_outputs", {}), "status": "completed"})
    record_lineage(state, "Pipeline DAG", f"Executed {len(req.stages)}-stage DAG", f"Status={dump.get('status', 'completed')}")
    return _cache_and_respond(state, sid, "s10_dag", {"execution": dump, "nodes": nodes, "edges": edges})


# ---------------------------------------------------------------------------
# Studio 11 — MLOps (Registry, Tracking & Drift Monitoring)
# ---------------------------------------------------------------------------
@router.get("/s11/overview")
def s11_overview() -> dict[str, Any]:
    models = [v.model_dump() for v in _SHARED_REGISTRY.list_versions()]
    runs = _SHARED_TRACKER.list_runs()
    return to_json_object({"registered_models": models, "experiment_runs": runs})


@router.post("/s11/register")
def s11_register(req: RegisterModelReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = get_session_state(_sid(x_session_id))
    model = state.get("uap_active_model")
    if model is None:
        raise HTTPException(status_code=400, detail="No trained model in active workspace. Run Studio 06 first.")
    ver = _SHARED_REGISTRY.register_model(
        model=model,
        model_name=req.model_name,
        metrics={k: float(v) for k, v in (state.get("uap_active_model_metrics") or {}).items() if isinstance(v, (int, float))},
        stage=ModelStage(req.stage) if req.stage in [s.value for s in ModelStage] else ModelStage.STAGING,
    )
    payload = to_json_object({"registered": ver.model_dump(), "registered_models": [v.model_dump() for v in _SHARED_REGISTRY.list_versions()]})
    save_studio_result(state, "s11_registered", payload)
    return payload


@router.post("/s11/promote")
def s11_promote(req: PromoteModelReq) -> dict[str, Any]:
    updated = _SHARED_REGISTRY.transition_stage(req.model_name, req.version, ModelStage(req.stage))
    return to_json_object({"updated": updated.model_dump(), "registered_models": [v.model_dump() for v in _SHARED_REGISTRY.list_versions()]})


@router.post("/s11/drift")
def s11_drift(req: DriftReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = get_session_state(_sid(x_session_id))
    df = get_analysis_dataframe(state, _require_df(state))
    num_df = df.select_dtypes(include=[np.number]).dropna()
    if len(num_df) < 20 or num_df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Drift detection requires at least 20 numeric rows.")
    cut = int(len(num_df) * req.split_ratio)
    report = DriftDetector().detect(num_df.iloc[:cut], num_df.iloc[cut:])
    payload = to_json_object({"drift_report": report.model_dump()})
    save_studio_result(state, "s11_drift", payload)
    return payload


# ---------------------------------------------------------------------------
# Studio 12 — Customizable Executive Reporting & Export Center (ALL 9 Formats!)
# ---------------------------------------------------------------------------
@router.get("/s12/catalog")
def s12_report_catalog(x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = get_session_state(_sid(x_session_id))
    avail_map = {
        "executive_summary": isinstance(state.get("uap_df"), pd.DataFrame),
        "dataset_overview": isinstance(state.get("uap_df"), pd.DataFrame),
        "intake_audit": bool(state.get("uap_intake_audit")),
        "data_quality": isinstance(state.get("uap_df"), pd.DataFrame),
        "cleaning_lineage": bool(state.get("uap_clean_report") or state.get("uap_exclusions") or state.get("uap_deleted_data_log")),
        "universal_diagnostics": bool(state.get("uap_diag_report")),
        "baseline_benchmarks": bool(state.get("uap_baseline_results") or state.get("uap_baseline_equations")),
        "risk_assessment": bool(state.get("uap_risk_report")),
        "clustering_anomalies": bool(state.get("uap_clustering_result") or state.get("uap_anomaly_meta") or state.get("uap_mining_result")),
        "eda_visuals": bool(state.get("uap_eda_report") or isinstance(state.get("uap_df"), pd.DataFrame)),
        "feature_engineering": bool(state.get("uap_fs_scores") or state.get("uap_s05_engineered")),
        "automl_leaderboard": bool(state.get("uap_automl") or state.get("uap_active_model")),
        "deep_learning_bench": bool(state.get("uap_dl_benchmark_df") is not None),
        "model_equations": bool(state.get("uap_baseline_equations") or state.get("uap_model_equations")),
        "explainability": bool(state.get("uap_shap_result")),
        "fairness_ethics": bool(state.get("uap_fairness_result")),
        "forecasting_causal": bool(state.get("uap_causal_result") or state.get("uap_forecast_result")),
        "future_predictions": bool(state.get("uap_supervisory_result") or state.get("uap_future_scores") is not None),
        "recommendations": isinstance(state.get("uap_df"), pd.DataFrame),
        "dataset_appendix": isinstance(state.get("uap_df"), pd.DataFrame),
    }
    catalog = [
        {
            "key": spec.section_id,
            "title": spec.title,
            "category": spec.category,
            "description": spec.description,
            "default_on": getattr(spec, "default_enabled", True),
            "has_data": bool(avail_map.get(spec.section_id, False)),
        }
        for spec in REPORT_SECTIONS_CATALOG
    ]
    return to_json_object(
        {
            "sections": catalog,
            "presets": REPORT_PRESETS,
            "formats": ["pdf", "xlsx", "docx", "csv", "csv_bundle", "pptx", "html", "md", "json"],
        }
    )


def _compile_report_payload(state: dict[str, Any], req: ReportCompileReq) -> dict[str, Any]:
    df = _require_df(state)
    raw_df = state.get("uap_raw_df")
    quality = state.get("uap_quality")
    if quality is None:
        quality = DataQualityProfiler().profile(df)
        state["uap_quality"] = quality
    q_dict: dict[str, Any] = to_json_object(
        quality.model_dump() if hasattr(quality, "model_dump") else quality
    )
    targets = get_selected_targets(state, df)
    gen = ReportGenerator()
    payload = gen.build_report(
        title=req.title,
        author=req.author,
        selected_sections=req.selected_sections,
        include_charts=req.include_charts,
        include_tables=req.include_tables,
        include_dataset_appendix=req.include_dataset_appendix,
        max_table_rows=req.max_table_rows,
        dataset_summary={"name": state.get("uap_dataset_name"), "rows": len(df), "columns": df.shape[1], "targets": targets},
        quality_report=q_dict,
        model_summary={"active_model": state.get("uap_active_model_name"), "metrics": state.get("uap_active_model_metrics")} if state.get("uap_active_model_name") else None,
        fairness_report=state.get("uap_fairness_result"),
        exclusions=state.get("uap_exclusions"),
        df=downsample_table_for_plot(df, max_rows=15_000),
        raw_df=downsample_table_for_plot(raw_df, max_rows=15_000) if isinstance(raw_df, pd.DataFrame) else None,
        target_columns=targets,
        cleaning_report=state.get("uap_clean_report"),
        diagnostics_report=state.get("uap_diag_report"),
        clustering_result=state.get("uap_clustering_result"),
        anomaly_meta=state.get("uap_anomaly_meta"),
        eda_report=state.get("uap_eda_report"),
        fs_scores=state.get("uap_fs_scores"),
        fs_selected=state.get("uap_fs_selected_cols"),
        model_equations=state.get("uap_model_equations"),
        automl_result=state.get("uap_automl"),
        dl_benchmark_df=state.get("uap_dl_benchmark_df"),
        causal_result=state.get("uap_causal_result"),
    )

    # Enrich payload with Task 33 sections/tables when selected (Intake Audit, Data Mining, Hypothesis Battery, Supervisory Alarms)
    selected_set = set(req.selected_sections or [s.section_id for s in REPORT_SECTIONS_CATALOG])
    tables = payload.setdefault("tables", {})
    sections = payload.setdefault("sections", [])

    if "intake_audit" in selected_set and isinstance(state.get("uap_intake_audit"), dict):
        ia = state["uap_intake_audit"]
        defects = ia.get("defects") or []
        pre_tests = ia.get("statistical_pretests") or []
        dc = ia.get("defect_counts") or {}
        sections.insert(
            min(2, len(sections)),
            {
                "section_id": "intake_audit",
                "title": "Automatic Post-Ingestion Defect Audit & Statistical Pre-Screening",
                "category": "Data",
                "narrative": (
                    f"Automatic intake audit detected {dc.get('total', len(defects))} data defects "
                    f"({dc.get('critical', 0)} Critical, {dc.get('high', 0)} High, {dc.get('medium', 0)} Medium) "
                    f"with overall readiness score {ia.get('quality_score', 0):.1f}/100 (Grade {ia.get('quality_grade', 'N/A')} — {ia.get('overall_health', '')})."
                ),
                "bullets": [f"[{str(d.get('severity', '')).upper()}] {d.get('category', '')} ({d.get('column', '')}): {d.get('finding', '')}" for d in defects[:8]],
            },
        )
        if req.include_tables and defects:
            tables["Intake Defect Audit"] = defects[: req.max_table_rows]
        if req.include_tables and pre_tests:
            tables["Intake Statistical Pre-Tests"] = pre_tests[: req.max_table_rows]

    if "clustering_anomalies" in selected_set and isinstance(state.get("uap_mining_result"), dict):
        mr = state["uap_mining_result"]
        ar = (mr.get("association_rules") or {}).get("rules") or []
        sg = (mr.get("subgroups") or {}).get("subgroups") or []
        if req.include_tables and ar:
            tables["Association Rules (Apriori)"] = ar[: req.max_table_rows]
        if req.include_tables and sg:
            tables["Subgroup Discovery"] = sg[: req.max_table_rows]

    if "eda_visuals" in selected_set and isinstance(state.get("uap_hypothesis_battery"), list) and state["uap_hypothesis_battery"]:
        if req.include_tables:
            tables["Hypothesis Test Battery"] = state["uap_hypothesis_battery"][: req.max_table_rows]

    if "future_predictions" in selected_set and isinstance(state.get("uap_supervisory_result"), dict):
        sr = state["uap_supervisory_result"]
        kpis = sr.get("kpis") or {}
        alarms = sr.get("alarms") or []
        sections.append(
            {
                "section_id": "future_predictions",
                "title": "Live Prediction, Reality-vs-Baseline Comparison & Tolerance Alarm Audit",
                "category": "Operations",
                "narrative": (
                    f"Supervisory evaluation of model '{kpis.get('model_label', 'Selected Model')}' vs baseline '{kpis.get('baseline_label', 'Baseline')}' "
                    f"across {kpis.get('n_records', 0)} live observations resulted in status {sr.get('system_status')} "
                    f"with {kpis.get('breach_count', 0)} tolerance breaches ({kpis.get('breach_rate_pct', 0):.2f}%) "
                    f"and {kpis.get('compliance_rate_pct', 100):.2f}% compliance."
                ),
                "bullets": [
                    f"System Status: {sr.get('system_status')}",
                    f"Critical Breaches: {kpis.get('breach_count', 0)} | Warnings: {kpis.get('warning_count', 0)} | Compliance: {kpis.get('compliance_rate_pct', 100):.1f}%",
                    f"Model MAE: {kpis.get('mae', 'N/A')} | MAPE: {kpis.get('mape_pct', 'N/A')}% | Baseline MAE: {kpis.get('baseline_mae', 'N/A')}",
                ],
            }
        )
        if req.include_tables and alarms:
            tables["Supervisory Tolerance Alarms"] = alarms[: req.max_table_rows]
        if req.include_tables and isinstance(state.get("uap_supervisory_scored_df"), pd.DataFrame) and not state["uap_supervisory_scored_df"].empty:
            tables["Supervisory Scored Telemetry"] = state["uap_supervisory_scored_df"].head(req.max_table_rows).to_dict(orient="records")

    state["uap_report_payload"] = payload
    return payload


@router.post("/s12/compile")
def s12_compile_report(req: ReportCompileReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    payload = _compile_report_payload(state, req)
    record_lineage(state, "Reporting", f"Compiled '{req.title}'", f"Sections={len(payload.get('sections', []))}")
    chart_previews = []
    for ch in (payload.get("charts") or [])[:12]:
        if isinstance(ch, dict):
            b64 = ch.get("png_base64")
            if not b64 and ch.get("png_bytes"):
                import base64 as _b64

                b64 = _b64.b64encode(ch["png_bytes"]).decode("ascii")
            if b64:
                chart_previews.append(
                    {
                        "title": ch.get("title", "Chart"),
                        "section": ch.get("section_id") or ch.get("section", "General"),
                        "caption": ch.get("caption", ""),
                        "image_data_url": f"data:image/png;base64,{b64}",
                    }
                )
    table_previews = {}
    for tname, trows in (payload.get("tables") or {}).items():
        if isinstance(trows, list) and trows and isinstance(trows[0], dict):
            table_previews[tname] = {
                "rows": len(trows),
                "columns": list(trows[0].keys()),
                "preview": trows[:15],
            }
        elif isinstance(trows, pd.DataFrame) and not trows.empty:
            table_previews[tname] = {
                "rows": int(len(trows)),
                "columns": [str(c) for c in trows.columns],
                "preview": trows.head(15).to_dict(orient="records"),
            }
    return _cache_and_respond(
        state,
        sid,
        "s12_report",
        {
            "title": payload.get("title"),
            "executive_summary": payload.get("executive_summary"),
            "sections": [
                {
                    "id": s.get("section_id") or s.get("id"),
                    "title": s.get("title"),
                    "category": s.get("category", "Report"),
                    "narrative": s.get("narrative"),
                    "highlights": s.get("bullets") or s.get("highlights") or [],
                }
                for s in payload.get("sections", [])
            ],
            "table_names": list((payload.get("tables") or {}).keys()),
            "table_previews": table_previews,
            "n_charts": len(payload.get("charts") or []),
            "chart_previews": chart_previews,
        },
    )


@router.get("/s12/export/{fmt}")
def s12_export_report(fmt: str, x_session_id: str | None = Header(default="default")) -> StreamingResponse:
    state = get_session_state(_sid(x_session_id))
    payload = state.get("uap_report_payload")
    if payload is None:
        payload = _compile_report_payload(state, ReportCompileReq())
    ext_map = {
        "docx": (".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        "pdf": (".pdf", "application/pdf"),
        "xlsx": (".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        "pptx": (".pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
        "html": (".html", "text/html"),
        "md": (".md", "text/markdown"),
        "json": (".json", "application/json"),
        "csv": (".csv", "text/csv"),
        "csv_bundle": (".zip", "application/zip"),
    }
    key = fmt.lower()
    if key not in ext_map:
        raise HTTPException(status_code=400, detail=f"Unsupported export format '{fmt}'. Supported: {sorted(ext_map.keys())}")
    ext, media = ext_map[key]
    with tempfile.TemporaryDirectory() as tmpd:
        out_path = Path(tmpd) / f"omniforge_report{ext}"
        exported = ReportGenerator().export(payload, out_path, fmt=key)
        data = exported.read_bytes()
    return StreamingResponse(
        io.BytesIO(data),
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="omniforge_executive_report{ext}"'},
    )


# ---------------------------------------------------------------------------
# Studio 13 — Context-Aware AI Analytics Copilot
# ---------------------------------------------------------------------------
@router.post("/s13/chat")
def s13_chat(req: CopilotReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    copilot = state.get("uap_copilot")
    if copilot is None:
        copilot = OmniForgeCopilotAgent(provider="local")
        state["uap_copilot"] = copilot
    raw_df = state.get("uap_df")
    if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
        analysis_df = get_analysis_dataframe(state, raw_df)
        copilot.load_dataframe(analysis_df, name=str(state.get("uap_dataset_name") or "active_dataset"))
    reply = copilot.chat(req.message)
    dump = reply.model_dump() if hasattr(reply, "model_dump") else {"content": str(reply)}
    ans = dump.get("answer") or dump.get("content") or str(reply)
    history = state.setdefault("uap_chat_history", [])
    history.append({"role": "user", "text": req.message})
    history.append({"role": "assistant", "text": ans})
    payload = to_json_object(
        {
            "reply": ans,
            "suggested_actions": dump.get("suggested_actions") or [],
            "tool_calls": dump.get("tool_calls") or [],
            "history": history[-60:],
        }
    )
    save_studio_result(state, "s13_chat", payload)
    return payload


# ---------------------------------------------------------------------------
# Studio 14 — Live Prediction, Reality-vs-Baseline Tolerance & Supervisory Dashboard
# ---------------------------------------------------------------------------
@router.get("/s14/models")
def s14_list_models(x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    raw_models = SupervisoryPredictionMonitor.catalog_available_models(state)
    models = [{k: v for k, v in m.items() if k != "_estimator"} for m in raw_models]
    last_eval = state.get("uap_supervisory_result")
    return to_json_object(
        {
            "models": models,
            "last_evaluation": last_eval,
            "history": state.get("uap_supervisory_history") or [],
            "state": _state_summary(state, sid),
        }
    )


@router.get("/s14/equations")
def s14_get_model_equations(
    model_id: str | None = None,
    baseline_model_id: str | None = None,
    x_session_id: str | None = Header(default="default"),
) -> dict[str, Any]:
    """Return exact governing mathematical equations & external deployment formulas for the selected
    Champion Model and Reference Baseline Model in Studio 14."""
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    catalog = SupervisoryPredictionMonitor.catalog_available_models(state)
    if not catalog:
        return to_json_object({"champion_equation": None, "baseline_equation": None})
    primary_entry = next((m for m in catalog if m["model_id"] == model_id), catalog[0])
    baseline_entry = None
    if baseline_model_id:
        baseline_entry = next((m for m in catalog if m["model_id"] == baseline_model_id), None)
    if baseline_entry is None:
        baseline_entry = next(
            (m for m in catalog if m["family"] == "Baseline" and m["model_id"] != primary_entry["model_id"]),
            next((m for m in catalog if m["family"] == "Baseline"), None),
        )
    ref_df = state.get("uap_df") if isinstance(state.get("uap_df"), pd.DataFrame) else None
    champ_eq = SupervisoryPredictionMonitor.extract_model_equation(primary_entry, ref_df=ref_df)
    base_eq = SupervisoryPredictionMonitor.extract_model_equation(baseline_entry, ref_df=ref_df)
    return to_json_object(
        {
            "champion_model_id": primary_entry["model_id"],
            "baseline_model_id": baseline_entry["model_id"] if baseline_entry else None,
            "champion_equation": champ_eq,
            "baseline_equation": base_eq,
        }
    )


@router.post("/s14/quick-train")
def s14_quick_train(req: SupervisoryQuickTrainReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    """Quickly train a selected baseline, ensemble, or neural network directly from Studio 14 so any model family is immediately available."""
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = get_analysis_dataframe(state, _require_df(state))
    targets = get_selected_targets(state, df)
    target = req.target if (req.target and req.target in df.columns) else (targets[0] if targets else None)
    if not target:
        raise HTTPException(status_code=400, detail="Select a target column first before training a supervisory model.")
    X, y, task = _prepare_xy(df, target)
    is_clf = task in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)
    split = max(10, int(len(X) * 0.8))
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    from sklearn.ensemble import (
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import f1_score, r2_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    m_key = req.method.lower()
    if m_key == "ridge":
        model = LogisticRegression(max_iter=600) if is_clf else Ridge(alpha=1.0)
        label = "Linear Baseline (Logistic)" if is_clf else "Linear Baseline (Ridge)"
        family = "Baseline"
    elif m_key == "decision_stump":
        model = DecisionTreeClassifier(max_depth=4, random_state=42) if is_clf else DecisionTreeRegressor(max_depth=4, random_state=42)
        label = "Decision Tree Baseline"
        family = "Baseline"
    elif m_key == "random_forest":
        model = RandomForestClassifier(n_estimators=80, random_state=42) if is_clf else RandomForestRegressor(n_estimators=80, random_state=42)
        label = "Random Forest"
        family = "AutoML"
    elif m_key == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=42) if is_clf else GradientBoostingRegressor(random_state=42)
        label = "Gradient Boosting"
        family = "AutoML"
    elif m_key == "gam_spline":
        model = PenalizedSplineGAMModel(task_type=task)
        label = "Penalized B-Spline GAM (Non-Parametric)"
        family = "Non-Parametric Analytical"
    elif m_key == "nadaraya_watson":
        model = NadarayaWatsonKernelModel(task_type=task)
        label = "Nadaraya-Watson Kernel Estimator (Non-Parametric)"
        family = "Non-Parametric Analytical"
    elif m_key == "tabnet":
        model = TabNetModel(n_d=16, n_a=16, n_steps=3, epochs=req.epochs, task_type=task, track_carbon=False)
        label = "TabNet Neural Network"
        family = "Deep Learning"
    elif m_key == "resnet_tabular":
        model = ResNetTabularModel(d_hidden=64, n_blocks=3, epochs=req.epochs, task_type=task, track_carbon=False)
        label = "ResNet-Tabular Neural Network"
        family = "Deep Learning"
    elif m_key == "ft_transformer":
        model = FTTransformerModel(d_model=24, n_heads=4, n_layers=2, epochs=req.epochs, task_type=task, track_carbon=False)
        label = "FT-Transformer Neural Network"
        family = "Deep Learning"
    else:
        model = MLPModel(hidden_dims=(64, 32), epochs=req.epochs, task_type=task, track_carbon=False)
        label = "MLP Neural Network"
        family = "Deep Learning"

    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    score = float(f1_score(y_te, preds, average="weighted", zero_division=0)) if is_clf else float(r2_score(y_te, preds))
    metric_name = "f1_weighted" if is_clf else "r2"

    if family == "Deep Learning":
        _dl_raw = state.get("uap_dl_models")
        dl_dict: dict[str, Any] = dict(_dl_raw) if isinstance(_dl_raw, dict) else {}
        dl_dict[m_key] = {
            "model": model,
            "name": label,
            "feature_cols": list(X.columns),
            "metrics": {metric_name: round(score, 4)},
            "target": target,
        }
        state["uap_dl_models"] = dl_dict
    elif family == "Baseline":
        _bl_raw = state.get("uap_baseline_models")
        bl_all: dict[str, Any] = dict(_bl_raw) if isinstance(_bl_raw, dict) else {}
        _t_bl_raw = bl_all.get(target)
        t_bl: dict[str, Any] = _t_bl_raw if isinstance(_t_bl_raw, dict) else {}
        t_bl[label] = {
            "model": model,
            "feature_cols": list(X.columns),
            "metrics": {metric_name: round(score, 4)},
        }
        bl_all[target] = t_bl
        state["uap_baseline_models"] = bl_all
    else:
        state["uap_active_model"] = model
        state["uap_active_model_name"] = label
        state["uap_active_model_metrics"] = {metric_name: round(score, 4)}
        state["uap_active_feature_cols"] = list(X.columns)

    record_lineage(state, "Supervisory Monitor", f"Quick-trained {label}", f"{metric_name}={score:.4f}")
    raw_models = SupervisoryPredictionMonitor.catalog_available_models(state)
    models_clean = [{k: v for k, v in m.items() if k != "_estimator"} for m in raw_models]
    return to_json_object(
        {
            "trained_model": label,
            "family": family,
            "metric": metric_name,
            "score": round(score, 4),
            "models": models_clean,
            "state": _state_summary(state, sid),
        }
    )


def _run_supervisory_eval(
    state: dict[str, Any],
    sid: str,
    incoming_df: pd.DataFrame,
    source_label: str,
    *,
    model_id: str,
    baseline_model_id: str | None,
    actual_col: str | None,
    id_col: str | None,
    max_pct_error: float,
    max_abs_error: float | None,
    max_sigma_z: float,
    baseline_ratio_min: float,
    baseline_ratio_max: float,
    warning_fraction: float,
    max_batch_breach_rate_pct: float,
    min_classification_accuracy_pct: float,
) -> dict[str, Any]:
    ref_df = state.get("uap_df")
    if not isinstance(ref_df, pd.DataFrame) or ref_df.empty:
        raise HTTPException(status_code=400, detail="Load a training/reference dataset in Studio 01 first.")
    targets = get_selected_targets(state, ref_df)
    target_col = targets[0] if targets else None
    actual_use = actual_col or (target_col if (target_col and target_col in incoming_df.columns) else None)
    eval_res = SupervisoryPredictionMonitor().evaluate_incoming(
        state=state,
        incoming_df=incoming_df,
        model_id=model_id,
        baseline_model_id=baseline_model_id,
        actual_col=actual_use,
        id_col=id_col,
        source_label=source_label,
        max_pct_error=max_pct_error,
        max_abs_error=max_abs_error,
        max_sigma_z=max_sigma_z,
        baseline_ratio_min=baseline_ratio_min,
        baseline_ratio_max=baseline_ratio_max,
        warning_fraction=warning_fraction,
        max_batch_breach_rate_pct=max_batch_breach_rate_pct,
        min_classification_accuracy_pct=min_classification_accuracy_pct,
    )
    kpis = eval_res.get("kpis") or {}
    record_lineage(
        state,
        "Supervisory Monitor",
        f"Evaluated {kpis.get('n_records', len(incoming_df))} rows ({source_label})",
        f"Status={eval_res.get('system_status')} | Breaches={kpis.get('breach_count', 0)}",
    )
    return _cache_and_respond(
        state,
        sid,
        "s14_supervisory",
        {
            **eval_res,
            "history": state.get("uap_supervisory_history") or [],
        },
    )


@router.post("/s14/evaluate-file")
async def s14_evaluate_file(
    file: UploadFile = File(...),
    sheet_name: str | None = Form(default=None),
    model_id: str = Form(default="active_champion"),
    baseline_model_id: str | None = Form(default=None),
    actual_col: str | None = Form(default=None),
    id_col: str | None = Form(default=None),
    max_pct_error: float = Form(default=10.0),
    max_abs_error: float | None = Form(default=None),
    max_sigma_z: float = Form(default=3.0),
    baseline_ratio_min: float = Form(default=0.85),
    baseline_ratio_max: float = Form(default=1.15),
    warning_fraction: float = Form(default=0.75),
    max_batch_breach_rate_pct: float = Form(default=5.0),
    min_classification_accuracy_pct: float = Form(default=85.0),
    x_session_id: str | None = Header(default="default"),
) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    fname = file.filename or "live_batch.xlsx"
    suffix = Path(fname).suffix.lower() or ".csv"
    if suffix in {".pkl", ".pickle", ".joblib", ".dill"}:
        raise HTTPException(
            status_code=400,
            detail="Pickle/joblib uploads are blocked for security (CWE-502). Please upload CSV, Parquet, Excel, or JSON.",
        )
    raw = await file.read()
    if suffix in {".xlsx", ".xls", ".xlsm"} and sheet_name:
        incoming = read_excel_sheet(raw, sheet_name=sheet_name)
    else:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)
        try:
            incoming = DataIngestionEngine().ingest(tmp_path).data
        finally:
            tmp_path.unlink(missing_ok=True)
    return _run_supervisory_eval(
        state,
        sid,
        incoming,
        f"File ({fname})",
        model_id=model_id,
        baseline_model_id=baseline_model_id or None,
        actual_col=actual_col or None,
        id_col=id_col or None,
        max_pct_error=max_pct_error,
        max_abs_error=max_abs_error,
        max_sigma_z=max_sigma_z,
        baseline_ratio_min=baseline_ratio_min,
        baseline_ratio_max=baseline_ratio_max,
        warning_fraction=warning_fraction,
        max_batch_breach_rate_pct=max_batch_breach_rate_pct,
        min_classification_accuracy_pct=min_classification_accuracy_pct,
    )


@router.post("/s14/evaluate-sql")
def s14_evaluate_sql(req: SupervisorySQLReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    incoming = sql_load_dataframe(req.connection_url, table_name=req.table_name, query=req.query, limit=req.limit)
    return _run_supervisory_eval(
        state,
        sid,
        incoming,
        f"SQL ({req.table_name or 'custom query'})",
        model_id=req.model_id,
        baseline_model_id=req.baseline_model_id,
        actual_col=req.actual_col,
        id_col=req.id_col,
        max_pct_error=req.max_pct_error,
        max_abs_error=req.max_abs_error,
        max_sigma_z=req.max_sigma_z,
        baseline_ratio_min=req.baseline_ratio_min,
        baseline_ratio_max=req.baseline_ratio_max,
        warning_fraction=req.warning_fraction,
        max_batch_breach_rate_pct=req.max_batch_breach_rate_pct,
        min_classification_accuracy_pct=req.min_classification_accuracy_pct,
    )


@router.post("/s14/evaluate-access")
def s14_evaluate_access(req: SupervisoryAccessReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    incoming = access_load_table(req.file_path, req.table_name, limit=req.limit)
    return _run_supervisory_eval(
        state,
        sid,
        incoming,
        f"MS Access ({Path(req.file_path).name} · {req.table_name})",
        model_id=req.model_id,
        baseline_model_id=req.baseline_model_id,
        actual_col=req.actual_col,
        id_col=req.id_col,
        max_pct_error=req.max_pct_error,
        max_abs_error=req.max_abs_error,
        max_sigma_z=req.max_sigma_z,
        baseline_ratio_min=req.baseline_ratio_min,
        baseline_ratio_max=req.baseline_ratio_max,
        warning_fraction=req.warning_fraction,
        max_batch_breach_rate_pct=req.max_batch_breach_rate_pct,
        min_classification_accuracy_pct=req.min_classification_accuracy_pct,
    )


@router.post("/s14/evaluate-demo-batch")
def s14_evaluate_demo_batch(req: SupervisoryDemoReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    """Generate a realistic live operational batch from the active dataset with optional injected process faults so engineers can immediately test tolerance alarms."""
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = get_analysis_dataframe(state, _require_df(state))
    targets = get_selected_targets(state, df)
    target_col = req.actual_col or (targets[0] if targets else None)

    rng = np.random.default_rng(2026)
    n_take = min(req.n_rows, len(df))
    batch = df.sample(n=n_take, replace=(n_take > len(df)), random_state=2026).reset_index(drop=True).copy()

    # Add light operational measurement noise + explicit process fault excursions on inject_fault_pct% of rows
    num_cols = list(batch.select_dtypes(include=[np.number]).columns)
    for col in num_cols:
        if col == target_col:
            continue
        std = float(batch[col].std())
        if math.isfinite(std) and std > 0 and req.noise_scale > 0:
            batch[col] = batch[col].astype(float) + rng.normal(0.0, std * req.noise_scale, size=len(batch))

    if target_col and target_col in batch.columns and req.inject_fault_pct > 0:
        n_faults = max(1, int(round(len(batch) * req.inject_fault_pct / 100.0)))
        fault_idx = rng.choice(len(batch), size=min(n_faults, len(batch)), replace=False)
        if pd.api.types.is_numeric_dtype(batch[target_col]) and batch[target_col].nunique() > 8:
            t_std = max(float(batch[target_col].std()), abs(float(batch[target_col].mean())) * 0.15, 1.0)
            batch[target_col] = batch[target_col].astype(float)
            signs = rng.choice([-1.0, 1.0], size=len(fault_idx))
            batch.loc[fault_idx, target_col] = batch.loc[fault_idx, target_col] + signs * t_std * 2.8
        else:
            uniq = list(batch[target_col].dropna().unique())
            if len(uniq) >= 2:
                for fi in fault_idx:
                    cur = batch.at[fi, target_col]
                    alt = [u for u in uniq if u != cur][0]
                    batch.at[fi, target_col] = alt

    return _run_supervisory_eval(
        state,
        sid,
        batch,
        f"Live Telemetry Simulation ({len(batch)} rows, {req.inject_fault_pct:.0f}% fault injection)",
        model_id=req.model_id,
        baseline_model_id=req.baseline_model_id,
        actual_col=target_col,
        id_col=req.id_col,
        max_pct_error=req.max_pct_error,
        max_abs_error=req.max_abs_error,
        max_sigma_z=req.max_sigma_z,
        baseline_ratio_min=req.baseline_ratio_min,
        baseline_ratio_max=req.baseline_ratio_max,
        warning_fraction=req.warning_fraction,
        max_batch_breach_rate_pct=req.max_batch_breach_rate_pct,
        min_classification_accuracy_pct=req.min_classification_accuracy_pct,
    )


@router.post("/s14/evaluate-single")
def s14_evaluate_single(req: SupervisorySingleReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    rec = dict(req.record)
    actual_col = req.actual_col
    if req.actual_value is not None:
        df = state.get("uap_df")
        targets = get_selected_targets(state, df) if isinstance(df, pd.DataFrame) else []
        actual_col = actual_col or (targets[0] if targets else "__actual__")
        rec[actual_col] = req.actual_value
    incoming = pd.DataFrame([rec])
    return _run_supervisory_eval(
        state,
        sid,
        incoming,
        "Single Operational Reading",
        model_id=req.model_id,
        baseline_model_id=req.baseline_model_id,
        actual_col=actual_col,
        id_col=req.id_col,
        max_pct_error=req.max_pct_error,
        max_abs_error=req.max_abs_error,
        max_sigma_z=req.max_sigma_z,
        baseline_ratio_min=req.baseline_ratio_min,
        baseline_ratio_max=req.baseline_ratio_max,
        warning_fraction=req.warning_fraction,
        max_batch_breach_rate_pct=req.max_batch_breach_rate_pct,
        min_classification_accuracy_pct=req.min_classification_accuracy_pct,
    )


@router.get("/s14/export/{fmt}")
def s14_export_results(fmt: str, x_session_id: str | None = Header(default="default")) -> Response:
    state = get_session_state(_sid(x_session_id))
    res_df = state.get("uap_supervisory_scored_df")
    if not isinstance(res_df, pd.DataFrame) or res_df.empty:
        raise HTTPException(status_code=400, detail="Run a supervisory evaluation first before exporting.")
    if fmt.lower() == "xlsx":
        buf = io.BytesIO()
        res_df.to_excel(buf, index=False)
        return Response(
            content=buf.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": 'attachment; filename="omniforge_supervisory_evaluation.xlsx"'},
        )
    return Response(
        content=res_df.to_csv(index=False).encode("utf-8"),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="omniforge_supervisory_evaluation.csv"'},
    )
