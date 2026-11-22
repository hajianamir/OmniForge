"""Shared Pydantic request schemas for OmniForge Workbench Studio routers (Studios 00..14)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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
    time_budget_seconds: float = Field(default=30.0, ge=5.0, le=3600.0)
    cv_folds: int = Field(default=3, ge=2, le=10)
    async_job: bool = False
    # C3 Green-AI budgets: None means unconstrained (backward compatible).
    max_carbon_grams_co2e: float | None = Field(default=None, ge=0.0)
    max_inference_latency_ms: float | None = Field(default=None, ge=0.0)


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
    instrument_col: str | None = None


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
    owner: str = ""
    intended_use: str = ""
    approved_domains: list[str] = Field(default_factory=list)
    risk_tier: str = "medium"
    override_reason: str | None = None


class PromoteModelReq(BaseModel):
    model_name: str
    version: int
    stage: str
    override_reason: str | None = None


class RollbackModelReq(BaseModel):
    model_name: str
    reason: str = "Automated or operator-initiated SLA degradation rollback"


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


def wb_cache_and_respond(
    state: dict[str, Any],
    sid: str,
    cache_key: str,
    body: dict[str, Any],
    *,
    include_state: bool = True,
) -> dict[str, Any]:
    """Typed wrapper around ``workbench._cache_and_respond`` for strict mypy studio routers."""
    from omniforge.api import workbench as wb

    out: dict[str, Any] = wb._cache_and_respond(state, sid, cache_key, body, include_state=include_state)
    return out

