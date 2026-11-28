"""Enterprise FastAPI Workbench Router Powering All 15 OmniForge CRISP-DM Studios (Studio 00 Ingestion through Studio 14 Supervisory)."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import re
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import Response

from omniforge.api.auth import require_permission, require_workbench_auth
from omniforge.api.jobs import job_manager
from omniforge.config.settings import get_settings
from omniforge.core.logging import get_logger
from omniforge.core.performance import (
    compact_scatter_payload,
)
from omniforge.core.task_inference import infer_supervised_task
from omniforge.core.types import ModelStage, Principal, TaskType
from omniforge.core.utils import sanitize_spreadsheet_frame, to_json_object, to_json_safe
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
from omniforge.data.feature_store import LocalFeatureStore
from omniforge.data.ingestion import DataIngestionEngine
from omniforge.data.quality import DataQualityProfiler
from omniforge.domains import default_domain_registry
from omniforge.features.inference import WorkbenchInputPreprocessor, nonconstant_predictors
from omniforge.mlops.registry import ModelRegistry
from omniforge.mlops.supervisory_monitor import SupervisoryPredictionMonitor
from omniforge.mlops.tracking import ExperimentTracker
from omniforge.models.anomaly.detector import AnomalyDetector
from omniforge.models.clustering.clusterer import ClusteringEngine
from omniforge.models.mining import AssociationRuleMiner, ManifoldProjector, SubgroupDiscoveryMiner
from omniforge.ui.state import (
    clear_split_lock,
    clear_workspace_dataset,
    export_workspace_project_bytes,
    get_active_split_frames,
    get_all_deleted_records_df,
    get_analysis_dataframe,
    get_audit_anomaly_dataframe,
    get_audit_only_columns,
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
    set_audit_only_columns,
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
_TENANT_STORES_LOCK = threading.RLock()
_TENANT_REGISTRIES: OrderedDict[str, ModelRegistry] = OrderedDict()
_TENANT_TRACKERS: OrderedDict[str, ExperimentTracker] = OrderedDict()
_TENANT_FEATURE_STORES: OrderedDict[str, LocalFeatureStore] = OrderedDict()
_MAX_TENANT_STORES = 128


def _current_principal() -> Principal | None:
    try:
        from omniforge.api.auth import get_request_principal

        return get_request_principal()
    except Exception:
        return None


def _tenant_slug(tenant: str) -> str:
    return hashlib.sha256(tenant.encode("utf-8", errors="replace")).hexdigest()[:16]


def get_tenant_registry(principal: Principal | None = None) -> ModelRegistry:
    """Return a tenant-isolated ModelRegistry (defaulting to _SHARED_REGISTRY for default-tenant)."""
    p = principal if principal is not None else _current_principal()
    tenant = str(getattr(p, "tenant_id", "") or "default-tenant") if p is not None else "default-tenant"
    if tenant == "default-tenant":
        return _SHARED_REGISTRY
    with _TENANT_STORES_LOCK:
        if tenant in _TENANT_REGISTRIES:
            _TENANT_REGISTRIES.move_to_end(tenant)
            return _TENANT_REGISTRIES[tenant]
        slug = _tenant_slug(tenant)
        base = get_settings().paths.workspace_dir / "tenants" / slug
        reg = ModelRegistry(
            db_path=base / "model_registry.db",
            artifact_dir=base / "artifacts" / "registry",
        )
        reg.import_legacy_registry(base / "api_v1" / "model_registry.db")
        _TENANT_REGISTRIES[tenant] = reg
        while len(_TENANT_REGISTRIES) > _MAX_TENANT_STORES:
            _TENANT_REGISTRIES.popitem(last=False)
        return reg


def get_tenant_tracker(principal: Principal | None = None) -> ExperimentTracker:
    """Return a tenant-isolated ExperimentTracker (defaulting to _SHARED_TRACKER for default-tenant)."""
    p = principal if principal is not None else _current_principal()
    tenant = str(getattr(p, "tenant_id", "") or "default-tenant") if p is not None else "default-tenant"
    if tenant == "default-tenant":
        return _SHARED_TRACKER
    with _TENANT_STORES_LOCK:
        if tenant in _TENANT_TRACKERS:
            _TENANT_TRACKERS.move_to_end(tenant)
            return _TENANT_TRACKERS[tenant]
        slug = _tenant_slug(tenant)
        base = get_settings().paths.workspace_dir / "tenants" / slug
        trk = ExperimentTracker(
            db_path=base / "mlops_tracking.db",
            artifact_root=base / "artifacts",
        )
        _TENANT_TRACKERS[tenant] = trk
        while len(_TENANT_TRACKERS) > _MAX_TENANT_STORES:
            _TENANT_TRACKERS.popitem(last=False)
        return trk


def get_tenant_feature_store(principal: Principal | None = None) -> LocalFeatureStore:
    """Return the default or tenant-isolated feature store used by monitoring."""
    from omniforge.config.settings import get_settings

    p = principal if principal is not None else _current_principal()
    tenant = str(getattr(p, "tenant_id", "") or "default-tenant") if p is not None else "default-tenant"
    with _TENANT_STORES_LOCK:
        if tenant in _TENANT_FEATURE_STORES:
            _TENANT_FEATURE_STORES.move_to_end(tenant)
            return _TENANT_FEATURE_STORES[tenant]
        settings = get_settings()
        path = (
            settings.paths.feature_store_dir if tenant == "default-tenant"
            else settings.paths.workspace_dir / "tenants" / _tenant_slug(tenant) / "feature_store"
        )
        store = LocalFeatureStore(repo_path=path)
        _TENANT_FEATURE_STORES[tenant] = store
        while len(_TENANT_FEATURE_STORES) > _MAX_TENANT_STORES:
            _TENANT_FEATURE_STORES.popitem(last=False)
        return store


def parse_model_stage(raw_stage: str, *, error_status: int = 400) -> ModelStage:
    """Case-insensitively parse a ModelStage enum or raise HTTPException."""
    lookup = {s.value.lower(): s for s in ModelStage}
    key = str(raw_stage or "").strip().lower()
    if key not in lookup:
        valid = [s.value for s in ModelStage]
        raise HTTPException(
            status_code=error_status,
            detail=f"Unknown model stage {raw_stage!r}. Valid stages: {valid}",
        )
    return lookup[key]


def current_job_owner_key(principal: Principal | None = None, *, for_query: bool = False) -> str | None:
    """Compute the job ownership scope for the active caller (`None` only for `platform_admin` queries)."""
    p = principal if principal is not None else _current_principal()
    if p is None or not getattr(p, "subject", None):
        return "default"
    if for_query and "platform_admin" in set(getattr(p, "roles", []) or []):
        return None
    tenant = str(getattr(p, "tenant_id", "") or "default-tenant")
    return f"job:{_owner_tag(tenant, str(p.subject))}"


async def read_upload_bounded(file: UploadFile, max_bytes: int | None = None) -> bytes:
    """Read an UploadFile in bounded chunks, raising HTTP 413 if it exceeds max_request_body_bytes."""
    from omniforge.config.settings import get_settings

    limit = int(max_bytes) if max_bytes is not None else int(get_settings().api.max_request_body_bytes)
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if limit > 0 and total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file exceeds maximum allowed size of {limit} bytes.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def parse_tabular_upload(raw: bytes, filename: str, *, sheet_name: str | None = None) -> pd.DataFrame:
    """Read an untrusted tabular upload through the same format policy on all API surfaces."""
    from omniforge.core.constants import SUPPORTED_FILE_FORMATS

    suffix = Path(filename).suffix.lower() or ".csv"
    fmt = SUPPORTED_FILE_FORMATS.get(suffix)
    if fmt not in {
        "csv", "tsv", "excel", "parquet", "feather", "arrow", "json", "jsonl",
        "orc", "avro", "sas", "spss", "stata", "yaml", "toml", "xml",
    }:
        raise HTTPException(status_code=400, detail="Unsupported upload format. Use a supported tabular format such as CSV, Excel, Parquet, or JSON.")
    if fmt == "excel":
        return read_excel_sheet(raw, sheet_name=sheet_name or 0)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"upload{suffix}"
        path.write_bytes(raw)
        return DataIngestionEngine().ingest(path).data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Session-key namespacing (OWASP API1: Broken Object Level Authorization).
# The raw X-Session-ID supplied by a client is never trusted as a global key: it is scoped to
# the authenticated (tenant, subject) pair, so two callers presenting the same header value
# always land in isolated session stores (each gets an empty, private session).
_SESSION_KEY_RE = re.compile(r"^ws:[0-9a-f]{16}:(.*)$")


def _display_session_id(sid: str) -> str:
    """Return the client-facing session id (strip the internal ``ws:<owner_tag>:`` prefix).

    The namespaced key is an internal storage detail; echoing it back would leak the
    ownership-tag scheme and confuse clients that round-trip the value.
    """
    m = _SESSION_KEY_RE.match(sid or "")
    return m.group(1) if m else (sid or "default")


def _owner_tag(tenant: str, subject: str) -> str:
    digest = hashlib.sha256(f"{tenant}\x00{subject}".encode("utf-8", errors="replace")).hexdigest()
    return digest[:16]


def _sid(x_session_id: str | None, principal: Principal | None = None) -> str:
    raw = (x_session_id or "default").strip() or "default"
    if principal is None:
        # Fall back to the request-scoped principal stashed by the auth dependency.
        try:
            from omniforge.api.auth import get_request_principal

            principal = get_request_principal()
        except Exception:
            principal = None
    if principal is None or not getattr(principal, "subject", None):
        # No authenticated context available (unit tests / internal tooling): legacy behavior.
        return raw
    tenant = str(getattr(principal, "tenant_id", "") or "default-tenant")
    return f"ws:{_owner_tag(tenant, str(principal.subject))}:{raw}"


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
    audit_only = get_audit_only_columns(state, df) if has_df else []
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
            "exact_duplicate_leak_count": (lock.get("audit_report") or {}).get("exact_duplicate_leak_count"),
            "passed": bool((lock.get("audit_report") or {}).get("passed", False))
            and not (lock.get("audit_report") or {}).get("exact_duplicate_leak_count"),
        }
    deleted_log = state.get("uap_deleted_records_log") or []
    total_deleted = int(sum(int(entry.get("rows_removed", entry.get("rows_deleted", 0))) for entry in deleted_log))
    fmt_audit = state.get("uap_format_audit") or {}
    input_preprocessor = state.get("uap_active_input_preprocessor")

    return to_json_object(
        {
            "session_id": _display_session_id(session_id),
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
            "audit_only_columns": audit_only,
            "analysis_cols_count": len(cols) - len(set(excluded) | set(audit_only)),
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
            "active_input_feature_cols": (
                input_preprocessor.input_columns if isinstance(input_preprocessor, WorkbenchInputPreprocessor)
                else state.get("uap_active_feature_cols") or []
            ),
            "active_input_feature_types": (
                input_preprocessor.column_types if isinstance(input_preprocessor, WorkbenchInputPreprocessor) else {}
            ),
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
    return infer_supervised_task(y)


def _prepare_xy(df: pd.DataFrame, target: str, *, raw_features: bool = False) -> tuple[pd.DataFrame, pd.Series, TaskType]:
    """Compatibility entry point using the same training preparation as inference."""
    X, y, task, _ = _prepare_xy_fitted(df, target, raw_features=raw_features)
    return X, y, task


def _prepare_xy_fitted(
    df: pd.DataFrame, target: str, *, raw_features: bool = False, prune_constants: bool = True
) -> tuple[pd.DataFrame, pd.Series, TaskType, WorkbenchInputPreprocessor]:
    """Prepare training data and retain the fitted raw-input transformation for scoring."""
    work = df.dropna(subset=[target]).copy()
    if len(work) < 10:
        raise HTTPException(status_code=400, detail=f"Target '{target}' has fewer than 10 non-null observations.")
    raw = work.drop(columns=[target])
    if not raw_features and prune_constants:
        raw = raw.loc[:, nonconstant_predictors(raw)]
    if raw.shape[1] == 0:
        raise HTTPException(status_code=400, detail="No usable predictor columns remain after excluding target.")
    try:
        preprocessor = WorkbenchInputPreprocessor(mode="raw" if raw_features else "numeric").fit(raw)
        prepared = preprocessor.transform(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if prepared.shape[1] == 0:
        raise HTTPException(status_code=400, detail="No usable predictor columns remain after excluding target.")
    y = work[target]
    return prepared, y, _infer_task(y), preprocessor


# ---------------------------------------------------------------------------
# Request Models (shared via omniforge.api.studios.schemas)
# ---------------------------------------------------------------------------
from omniforge.api.studios.schemas import (  # noqa: E402, F401
    AccessInspectReq,
    AccessLoadReq,
    AnomalyCommitReq,
    AnomalyReq,
    AutoMLRunReq,
    CausalReq,
    CleanReq,
    ClusterCommitReq,
    ClusterReq,
    CopilotReq,
    DAGRunReq,
    DeepBenchReq,
    DiagnosticsReq,
    DriftReq,
    ExclusionsReq,
    ExplainReq,
    FairnessReq,
    FeatureEngReq,
    FeatureSelReq,
    ForecastReq,
    GraphReq,
    HypothesisReq,
    LoadSampleReq,
    MiningReq,
    NLPReq,
    NonParametricRunReq,
    OptimizeReq,
    PromoteModelReq,
    RecommendReq,
    RegisterModelReq,
    ReportCompileReq,
    SimulateReq,
    SinglePredictReq,
    SplitLockReq,
    SQLInspectReq,
    SQLLoadReq,
    SupervisoryAccessReq,
    SupervisoryDemoReq,
    SupervisoryQuickTrainReq,
    SupervisorySingleReq,
    SupervisorySQLReq,
    SupervisoryTolBase,
    SurvivalReq,
    TargetsReq,
)

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


@router.post("/targets", dependencies=[Depends(require_permission("data:write"))])
def update_targets(req: TargetsReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = _require_df(state)
    if set(req.targets) & set(get_audit_only_columns(state, df)):
        raise HTTPException(status_code=400, detail="Cleaning audit flags cannot be selected as study targets.")
    valid = set_selected_targets(state, req.targets, df)
    return {"targets": valid, "state": _state_summary(state, sid)}


@router.post("/exclusions", dependencies=[Depends(require_permission("data:write"))])
def update_exclusions(req: ExclusionsReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = _require_df(state)
    updated = set_excluded_columns(state, req.excluded_columns, df=df)
    if req.reason:
        record_lineage(state, "Column Scope", "Exclusion reason noted", req.reason)
    return {"excluded_columns": updated, "state": _state_summary(state, sid)}


@router.post("/split-lock", dependencies=[Depends(require_permission("data:write"))])
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


@router.delete("/split-lock", dependencies=[Depends(require_permission("data:write"))])
def unlock_split(x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    clear_split_lock(state)
    return {"state": _state_summary(state, sid)}


@router.post("/workspace/clear", dependencies=[Depends(require_permission("data:write"))])
def reset_workspace(x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    clear_workspace_dataset(state)
    return {"state": _state_summary(state, sid)}


@router.post("/workspace/save", dependencies=[Depends(require_permission("report:generate"))])
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


@router.post("/workspace/load", dependencies=[Depends(require_permission("data:write"))])
async def load_workspace_uap(
    file: UploadFile = File(...),
    x_session_id: str | None = Header(default="default"),
) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    raw = await read_upload_bounded(file)
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


@router.get("/feature-store/freshness")
def feature_store_freshness() -> dict[str, Any]:
    """Return tenant-scoped, point-in-time TTL and stale-vector monitoring."""
    return to_json_object(get_tenant_feature_store().freshness_report())


@router.get("/jobs")
def list_jobs() -> dict[str, Any]:
    return {"jobs": job_manager.list_recent(owner_key=current_job_owner_key(for_query=True))}


@router.get("/jobs/{job_id}")
def poll_job(job_id: str) -> dict[str, Any]:
    rec = job_manager.get(job_id, owner_key=current_job_owner_key(for_query=True))
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return rec.to_dict()


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, response: Response) -> dict[str, Any]:
    owner = current_job_owner_key(for_query=True)
    status = job_manager.cancel(job_id, owner_key=owner)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if status == "cancelling":
        response.status_code = 202
    elif status != "cancelled":
        raise HTTPException(status_code=409, detail=f"Job is {status}; cancellation is unavailable or already too late.")
    rec = job_manager.get(job_id, owner_key=owner)
    if rec is None:
        # A concurrently completed job may trigger history pruning after cancellation.
        return {"job_id": job_id, "status": status}
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


@router.delete("/dataset", dependencies=[Depends(require_permission("data:write"))])
def delete_active_dataset(x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    """Clear the active dataset and reset session workspace state."""
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    clear_workspace_dataset(state)
    return to_json_object({"cleared": True, "state": _state_summary(state, sid)})


@router.post("/s01/excel-sheets")
async def s01_excel_sheets(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await read_upload_bounded(file)
    sheets = list_excel_sheets(raw)
    return to_json_object({"filename": file.filename or "workbook.xlsx", "sheets": sheets})


@router.post("/s01/upload", dependencies=[Depends(require_permission("data:write"))])
async def s01_upload_dataset(
    file: UploadFile = File(...),
    sheet_name: str | None = Form(default=None),
    x_session_id: str | None = Header(default="default"),
) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    fname = file.filename or "uploaded.csv"
    suffix = Path(fname).suffix.lower() or ".csv"
    raw = await read_upload_bounded(file)
    t0 = time.perf_counter()
    df = parse_tabular_upload(raw, fname, sheet_name=sheet_name)
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        src_label = (
            f"Excel Upload ({fname} · Sheet '{sheet_name}')"
            if sheet_name
            else f"Excel Upload ({fname})"
        )
    else:
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


@router.post("/s01/sql/demo", dependencies=[Depends(require_permission("data:write"))])
def s01_sql_demo_db() -> dict[str, Any]:
    """Ensure the built-in SQLite enterprise demo database exists and return its connection info & tables."""
    db_path = ensure_demo_sqlite_database()
    conn_url = f"sqlite:///{db_path.as_posix()}"
    info = sql_inspect_database(conn_url)
    return to_json_object({"connection_url": conn_url, "db_path": str(db_path), **info})


@router.post("/s01/sql/inspect", dependencies=[Depends(require_permission("data:write"))])
def s01_sql_inspect(req: SQLInspectReq) -> dict[str, Any]:
    return to_json_object(sql_inspect_database(req.connection_url))


@router.post("/s01/sql/load", dependencies=[Depends(require_permission("data:write"))])
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


@router.post("/s01/access/inspect", dependencies=[Depends(require_permission("data:write"))])
def s01_access_inspect(req: AccessInspectReq) -> dict[str, Any]:
    return to_json_object(access_inspect_database(req.file_path))


@router.post("/s01/access/load", dependencies=[Depends(require_permission("data:write"))])
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


@router.post("/s01/load-sample", dependencies=[Depends(require_permission("data:write"))])
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
    return _respond_with_quality(state, sid, df)


@router.post("/s01/profile-reference", dependencies=[Depends(require_permission("data:write"))])
async def s01_profile_reference(
    file: UploadFile = File(...),
    key_column: str = Form(...),
    x_session_id: str | None = Header(default="default"),
) -> dict[str, Any]:
    """Compare a bounded trusted CSV against the active dataset without replacing it."""
    if Path(file.filename or "").suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Upload a trusted reference in CSV format.")
    raw = await read_upload_bounded(file)
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = _require_df(state)
    if key_column not in df or not df.columns.is_unique:
        raise HTTPException(status_code=400, detail="Select a unique key column from the active dataset.")
    try:
        dtype = {key_column: "string"} if pd.api.types.is_string_dtype(df[key_column]) else None
        with io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])
            if not header or len(set(header)) != len(header):
                raise ValueError("Reference CSV must have unique header labels.")
            stream.seek(0)
            reference = pd.read_csv(stream, dtype=dtype, keep_default_na=False, na_values=[""])
    except (csv.Error, UnicodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="The trusted CSV could not be read.") from exc
    return _respond_with_quality(state, sid, df, reference=reference, reference_keys=[key_column])


def _respond_with_quality(
    state: dict[str, Any], sid: str, df: pd.DataFrame, *, reference: pd.DataFrame | None = None,
    reference_keys: list[str] | None = None,
) -> dict[str, Any]:
    q_report = DataQualityProfiler().profile(df, reference_data=reference, reference_key_columns=reference_keys)
    state["uap_quality"] = q_report
    fmt = state.get("uap_format_audit") or run_format_audit(state, df)
    record_lineage(state, "Quality", "Ran Data Quality & Format Audit", f"Score={q_report.overall_score:.1f}/100")
    return _cache_and_respond(state, sid, "s01_quality", {"quality": q_report.model_dump(), "format_audit": fmt})


@router.post("/s01/clean", dependencies=[Depends(require_permission("data:write"))])
def s01_run_cleaning(req: CleanReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    df = _require_df(state)
    before_cols = list(df.columns)
    work_in = df.copy()
    train_df, test_df, lock = get_active_split_frames(state, work_in)
    locked_time = lock.get("time_col") if lock and lock.get("strategy") == "time_series" else None
    lock_columns = {
        lock[key] for key in ("time_col", "group_col", "row_id_col", "target_col") if isinstance(lock.get(key), str)
    } if lock else set()
    protected_columns = list(dict.fromkeys([
        *get_excluded_columns(state, df), *get_selected_targets(state, df), *get_audit_only_columns(state, df),
        *sorted(lock_columns),
    ]))

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
            outlier_action="flag",
            outlier_threshold=float(preset.get("outlier_threshold", 1.5)),
            exclude_columns=protected_columns,
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
            outlier_action="flag",
            outlier_threshold=1.5,
            exclude_columns=protected_columns,
        )
        drop_const = True
        std_col_names = False
    else:
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
            exclude_columns=protected_columns,
        )
        drop_const = req.drop_constant_columns
        std_col_names = req.standardize_column_names

    cfg.causal_time_series = bool(locked_time)
    cfg.time_column = locked_time
    cleaner = DataCleaner(cfg)
    split_safe_used = False
    if test_df is not None and lock is not None:
        split_safe_used = True
        cleaned_tr, cleaned_te, rep_tr, rep_te = cleaner.fit_transform_split(train_df, test_df)
        cleaned = recombine_split_frames(cleaned_tr, cleaned_te, work_in.index)
        reserved = work_in.loc[~work_in.index.isin(train_df.index.union(test_df.index))]
        if not reserved.empty:
            cleaned = recombine_split_frames(cleaned, reserved.reindex(columns=cleaned.columns), work_in.index)
        rep_dict = rep_tr.model_dump()
        rep_dict["columns_added"] = list(dict.fromkeys([*rep_tr.columns_added, *rep_te.columns_added]))
        rep_dict["rows_reserved_by_split"] = len(reserved)
        rep_dict["exact_duplicates_removed"] = int(rep_tr.exact_duplicates_removed + rep_te.exact_duplicates_removed)
        rep_dict["fuzzy_duplicates_removed"] = int(rep_tr.fuzzy_duplicates_removed + rep_te.fuzzy_duplicates_removed)
        rep_dict["removed_rows_count"] = int(rep_tr.removed_rows_count + rep_te.removed_rows_count)
        merged_outliers: dict[str, int] = dict(rep_tr.outliers_detected)
        for k_o, v_o in rep_te.outliers_detected.items():
            merged_outliers[k_o] = int(merged_outliers.get(k_o, 0) + v_o)
        rep_dict["outliers_detected"] = merged_outliers
        merged_imputed: dict[str, int] = dict(rep_tr.imputed_columns)
        for k_i, v_i in rep_te.imputed_columns.items():
            merged_imputed[k_i] = int(merged_imputed.get(k_i, 0) + v_i)
        rep_dict["imputed_columns"] = merged_imputed
        merged_reasons: dict[str, int] = dict(rep_tr.removed_rows_by_reason)
        for k_r, v_r in rep_te.removed_rows_by_reason.items():
            merged_reasons[k_r] = int(merged_reasons.get(k_r, 0) + v_r)
        rep_dict["removed_rows_by_reason"] = merged_reasons
        executed_codes = list(dict.fromkeys([*rep_tr.steps_executed, *rep_te.steps_executed]))
        rep_dict["split_lock_leakage_safe"] = True
    else:
        cleaned, report = cleaner.fit_transform(work_in)
        rep_dict = report.model_dump() if hasattr(report, "model_dump") else dict(report)
        executed_codes = list(report.steps_executed)
        rep_dict["split_lock_leakage_safe"] = False

    dropped_const_cols: list[str] = []
    if drop_const:
        excl_set = set(protected_columns)
        excl_set.update(rep_dict["columns_added"])
        const_source = (
            cleaned_tr
            if (rep_dict.get("split_lock_leakage_safe") and isinstance(cleaned_tr, pd.DataFrame) and not cleaned_tr.empty)
            else cleaned
        )
        for col in list(cleaned.columns):
            if col in const_source.columns and col not in excl_set and int(const_source[col].nunique(dropna=True)) <= 1 and len(const_source) > 1:
                dropped_const_cols.append(str(col))
        if dropped_const_cols and len(dropped_const_cols) < cleaned.shape[1]:
            cleaned = cleaned.drop(columns=dropped_const_cols)
            executed_codes.append("drop_constant_columns")

    # Compute genuinely dropped columns BEFORE optional column-name standardization
    dropped_cols = [c for c in before_cols if c not in cleaned.columns]
    rename_map: dict[str, str] = {}
    if std_col_names:
        import re as _re

        seen_names = set(lock_columns)
        new_names: list[str] = []
        for i, c in enumerate(cleaned.columns):
            if c in lock_columns:
                new_names.append(c)
                continue
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
        if any(k != v for k, v in rename_map.items()):
            executed_codes.append("standardize_column_names")

    removed_idx = [idx for idx in df.index if idx not in set(cleaned.index)]
    renamed_non_trivial = {k: v for k, v in rename_map.items() if k != v}
    rep_dict["columns_dropped"] = list(dict.fromkeys([*(rep_dict.get("columns_dropped") or []), *dropped_cols]))
    rep_dict["mode_used"] = mode
    rep_dict["dropped_constant_columns"] = dropped_const_cols
    rep_dict["renamed_columns"] = renamed_non_trivial
    audit_only = [rename_map.get(c, c) for c in [*get_audit_only_columns(state, df), *rep_dict["columns_added"]]]
    set_audit_only_columns(cleaned, audit_only)
    rep_dict["audit_only_columns"] = get_audit_only_columns(state, cleaned)
    rep_dict["rows_before"] = int(len(df))
    rep_dict["rows_after"] = int(len(cleaned))
    rep_dict["cols_before"] = int(df.shape[1])
    rep_dict["cols_after"] = int(cleaned.shape[1])
    rep_dict["missing_before"] = int(df.isna().sum().sum())
    rep_dict["missing_after"] = int(cleaned.isna().sum().sum())
    rep_dict["split_safe_used"] = split_safe_used
    rep_dict["steps_executed"] = executed_codes

    def _describe_executed_step(idx_num: int, code: str) -> str:
        if code == "clean_strings":
            return f"{idx_num}. Clean & normalize string columns ({len(rep_dict.get('strings_cleaned_columns') or [])} column(s))"
        if code == "standardize_missing_tokens":
            return f"{idx_num}. Standardize missing-value sentinel tokens to NaN"
        if code == "infer_and_cast_types":
            return f"{idx_num}. Infer & coerce numeric/datetime column types ({len(rep_dict.get('type_conversions') or {})} converted)"
        if code == "replace_nonfinite_with_nan":
            return f"{idx_num}. Replace non-finite (+/-Inf) numeric values with NaN"
        if code.startswith("apply_domain_bounds:"):
            detail = code.split(":", 1)[1].replace("_", " ")
            return f"{idx_num}. Coerce out-of-bounds domain values to NaN ({detail})"
        if code == "drop_high_missing_columns":
            return f"{idx_num}. Drop columns exceeding missing-value threshold ({len(cleaner._state.dropped_cols)} dropped)"
        if code == "drop_exact_duplicates":
            return f"{idx_num}. Check & remove exact duplicate rows ({int(rep_dict.get('exact_duplicates_removed', 0))} removed)"
        if code == "fuzzy_deduplicate":
            return f"{idx_num}. Fuzzy-deduplicate text records ({int(rep_dict.get('fuzzy_duplicates_removed', 0))} removed)"
        if code == "add_missing_indicators":
            return f"{idx_num}. Add missingness indicator flag columns"
        if code.startswith("outliers:"):
            parts = code.split(":")
            m_name = parts[1].upper() if len(parts) > 1 else "OUTLIER"
            a_name = parts[2] if len(parts) > 2 else "handled"
            tot_out = sum(int(v) for v in (rep_dict.get("outliers_detected") or {}).values())
            return f"{idx_num}. Detect & handle statistical outliers ({m_name} -> {a_name}; {tot_out} detected)"
        if code == "impute_missing":
            imp_cols = rep_dict.get("imputed_columns") or {}
            tot_imp = sum(int(v) for v in imp_cols.values())
            return f"{idx_num}. Impute remaining missing values ({tot_imp} cell(s) across {len(imp_cols)} column(s))"
        if code == "drop_constant_columns":
            return f"{idx_num}. Drop zero-variance constant columns ({len(dropped_const_cols)} dropped)"
        if code == "standardize_column_names":
            return f"{idx_num}. Standardize column names ({len(renamed_non_trivial)} renamed)"
        return f"{idx_num}. {code}"

    rep_dict["steps_applied"] = [
        _describe_executed_step(idx_num, step_code)
        for idx_num, step_code in enumerate(executed_codes, start=1)
    ]
    deleted_records_list: list[dict[str, Any]] = []
    if isinstance(cleaner.last_removed_rows_, pd.DataFrame) and not cleaner.last_removed_rows_.empty:
        for idx_val, row_val in cleaner.last_removed_rows_.head(200).iterrows():
            deleted_records_list.append(
                {
                    "row_index": str(row_val.get("_original_row_index", row_val.get("_orig_index", idx_val))),
                    "stage": str(row_val.get("_deletion_stage", row_val.get("_removal_stage", "Data Cleaning"))),
                    "reason": str(row_val.get("_deletion_reason", row_val.get("_removal_reason", "Removed during cleaning"))),
                }
            )
    rep_dict["deleted_records"] = deleted_records_list

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
    update_active_dataset(
        state,
        cleaned,
        stage="Cleaning",
        action=f"Applied DataCleaner ({mode})",
        details=f"Rows {len(df):,} -> {len(cleaned):,} | Cols {df.shape[1]} -> {cleaned.shape[1]}",
        preserve_stage="s01",
        row_index_preserved=True,  # DataCleaner and recombine_split_frames retain surviving row labels.
    )
    state["uap_clean_report"] = rep_dict
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
    df = sanitize_spreadsheet_frame(_require_df(state))
    stem = str(state.get("uap_dataset_name") or "omniforge_cleaned")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")[:120] or "omniforge_cleaned"
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


@router.post("/s01/undo", dependencies=[Depends(require_permission("data:write"))])
def s01_undo(x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    ok = undo_last_dataset_change(state)
    return {"undone": ok, "state": _state_summary(state, sid)}


@router.post("/s01/reset-raw", dependencies=[Depends(require_permission("data:write"))])
@router.post("/s01/revert", dependencies=[Depends(require_permission("data:write"))])
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
    csv_bytes = sanitize_spreadsheet_frame(del_df).to_csv(index=False).encode("utf-8")
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="omniforge_deleted_records_audit.csv"'},
    )


# ---------------------------------------------------------------------------
# Studio 02 — Clustering, Anomaly Screening & Data Mining Lab
# ---------------------------------------------------------------------------
def _prepare_numeric_feature_frame(
    df: pd.DataFrame,
    helper_cols: set[str],
    requested_features: list[str] | None = None,
    min_cols: int = 2,
) -> tuple[pd.DataFrame, list[str]]:
    """Return a working DataFrame and numeric feature column list, encoding string/categorical columns if needed."""
    if requested_features:
        cand = [c for c in requested_features if c in df.columns]
    else:
        cand = [c for c in df.select_dtypes(include=[np.number]).columns if c not in helper_cols]
    if len(cand) >= min_cols and all(pd.api.types.is_numeric_dtype(df[c]) for c in cand):
        return df, cand

    work = df.copy()
    encoded_cols: list[str] = list(cand)
    pool = requested_features or [c for c in work.columns if c not in helper_cols]
    for col in pool:
        if col not in work.columns:
            continue
        ser = work[col]
        if pd.api.types.is_numeric_dtype(ser):
            if col not in encoded_cols:
                encoded_cols.append(col)
            continue
        if pd.api.types.is_bool_dtype(ser):
            work[col] = ser.astype(float)
            if col not in encoded_cols:
                encoded_cols.append(col)
            continue
        coerced = pd.to_numeric(
            ser.astype("string").str.strip().str.replace(",", "", regex=False),
            errors="coerce",
        )
        non_na = int(ser.notna().sum())
        if non_na >= 3 and float(coerced.notna().sum()) / max(non_na, 1) >= 0.6:
            work[col] = coerced.astype(float)
            if col not in encoded_cols:
                encoded_cols.append(col)
            continue
        if non_na >= 3 and int(ser.nunique(dropna=True)) >= 2:
            codes, _ = pd.factorize(ser.astype(str), sort=True)
            num_ser = pd.Series(codes, index=ser.index, dtype=float)
            num_ser.loc[ser.isna()] = np.nan
            work[col] = num_ser
            if col not in encoded_cols:
                encoded_cols.append(col)
        if len(encoded_cols) >= max(min_cols, 12) and not requested_features:
            break
    return work, encoded_cols


def _study_predictor_columns(
    df: pd.DataFrame, excluded: set[str], requested: list[str] | None,
) -> list[str]:
    candidates = requested if requested else list(df.columns)
    return [
        col for col in dict.fromkeys(candidates)
        if col in df.columns and col not in excluded and df[col].notna().any()
    ]


@router.post("/s02/clustering")
def s02_run_clustering(req: ClusterReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    raw_analysis_df = get_analysis_dataframe(state, _require_df(state))
    targets = set(get_selected_targets(state, raw_analysis_df))
    helper_cols = {"cluster_id", "is_anomaly", "anomaly_score"} | targets
    df = raw_analysis_df
    train_df, test_df, lock = get_active_split_frames(state, _require_df(state))
    train_df = train_df.loc[:, df.columns]
    if test_df is not None:
        test_df = test_df.loc[:, df.columns]
    if lock:
        helper_cols.update(c for c in (lock.get("row_id_col"), lock.get("group_col"), lock.get("time_col")) if isinstance(c, str))
    num_cols = _study_predictor_columns(train_df, helper_cols, req.features)
    if not num_cols:
        raise HTTPException(status_code=400, detail="Clustering requires an observed predictor in the current study.")

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
                cur_preview = raw_analysis_df.copy()
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
                cur_preview = raw_analysis_df.copy()
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
                    "error": f"Clustering could not evaluate these rows ({type(exc).__name__}).",
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

    scatter = None
    scatter_error = None
    if isinstance(res.pca_2d, pd.DataFrame) and res.pca_2d.shape[1] >= 2:
        xs = res.pca_2d.iloc[:, 0].to_numpy(dtype=float)
        ys = res.pca_2d.iloc[:, 1].to_numpy(dtype=float)
        scatter = compact_scatter_payload(xs, ys, groups=res.labels, max_points=5000)
    else:
        scatter_error = "The clustering result did not include a two-dimensional PCA projection."

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
            "scatter_error": scatter_error,
            "features_used": num_cols,
        },
    )


@router.post("/s02/clustering/commit", dependencies=[Depends(require_permission("data:write"))])
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
        row_index_preserved=True,  # work is filtered from df without resetting its index.
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
        # The scatter, benchmark and metrics describe the original preview.
        # Keep their original counts rather than mixing them with post-commit rows.
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
            "result": get_studio_results(state).get("s02_clustering"),
            "state": _state_summary(state, sid),
        }
    )


@router.post("/s02/anomaly")
def s02_run_anomaly(req: AnomalyReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _sid(x_session_id)
    state = get_session_state(sid)
    raw_audit_df = get_audit_anomaly_dataframe(state, _require_df(state))
    helper_cols = {"cluster_id", "is_anomaly", "anomaly_score"} | set(get_selected_targets(state, raw_audit_df))
    train_df, _test_df, lock = get_active_split_frames(state, _require_df(state))
    if lock:
        helper_cols.update(c for c in (lock.get("row_id_col"), lock.get("group_col"), lock.get("time_col")) if isinstance(c, str))
        row_id = lock.get("row_id_col")
        if isinstance(row_id, str) and row_id in raw_audit_df and raw_audit_df[row_id].is_unique:
            train_df = raw_audit_df.loc[raw_audit_df[row_id].isin(train_df[row_id])]
        elif raw_audit_df.index.is_unique and train_df.index.is_unique and train_df.index.isin(raw_audit_df.index).all():
            train_df = raw_audit_df.loc[train_df.index]
        else:
            raise HTTPException(status_code=400, detail="The raw audit data cannot be aligned with the locked training rows.")
    else:
        train_df = raw_audit_df
    if train_df.empty:
        raise HTTPException(status_code=400, detail="Anomaly detection requires a non-empty training partition.")
    candidate_cols = _study_predictor_columns(train_df, helper_cols, req.features)
    if not candidate_cols:
        raise HTTPException(status_code=400, detail="Anomaly detection requires at least one observed predictor.")
    try:
        preprocessor = WorkbenchInputPreprocessor(mode="numeric").fit(train_df.loc[:, candidate_cols])
        X_fit = preprocessor.transform(train_df.loc[:, candidate_cols])
        X_num = preprocessor.transform(raw_audit_df.loc[:, candidate_cols])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="The anomaly predictor schema could not be fitted.") from exc
    num_cols = list(X_num.columns)

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
            det.fit(X_fit)
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
            logger.warning("anomaly_detector_failed", method=m_name, error_type=type(exc).__name__)
            detector_comparison.append({"detector": m_name, "error": "The detector could not be fitted or the method is unsupported."})

    if selected_flags is None or selected_scores is None:
        raise HTTPException(status_code=400, detail="None of the requested anomaly detectors could evaluate this data.")
    preview = raw_audit_df.copy()
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


@router.post("/s02/anomaly/commit", dependencies=[Depends(require_permission("data:write"))])
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
        if rows_before > 0 and drop_mask.all():
            raise HTTPException(status_code=400, detail="Anomaly removal would leave an empty dataset.")
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
        row_index_preserved=True,  # work is filtered from df without resetting its index.
    )
    state["uap_anomaly_df"] = anom_df.loc[anom_df.index.intersection(work.index)].copy()
    cached_a = (state.get("uap_studio_results") or {}).get("s02_anomaly")
    if isinstance(cached_a, dict):
        updated_a = dict(cached_a)
        updated_a["provenance"] = "COMMITTED"
        updated_a["committed_action"] = action
        updated_a["committed_mode"] = req.mode
        save_studio_result(state, "s02_anomaly", updated_a)
    return to_json_object(
        {
            "committed": True,
            "rows_before": rows_before,
            "rows_after": int(len(work)),
            "rows_removed": removed,
            "action": action,
            "result": get_studio_results(state).get("s02_anomaly"),
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
    mode = req.mode.lower()
    tasks = ["association_rules", "subgroups", "manifold"] if mode in ("complete", "auto") else (req.tasks or ["association_rules"])
    target = resolve_mining_target(targets, req.target) if req.target is not None or "subgroups" in tasks else None
    predictors = df.drop(columns=targets)
    if predictors.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Data mining requires an eligible predictor.")

    if "subgroups" in tasks and target is None:
        raise HTTPException(status_code=400, detail="Select a study target before running subgroup discovery.")
    out: dict[str, Any] = {"mode": mode, "target": target, "tasks_executed": tasks}
    if "association_rules" in tasks:
        out["association_rules"] = AssociationRuleMiner().mine(
            predictors,
            min_support=req.min_support,
            min_confidence=req.min_confidence,
            min_lift=req.min_lift,
        )
    if "subgroups" in tasks and target:
        out["subgroups"] = SubgroupDiscoveryMiner().discover(df.drop(columns=[c for c in targets if c != target]), target_col=target)
    if "manifold" in tasks:
        try:
            color = req.color_col or target or (targets[0] if len(targets) == 1 else None)
            if color is not None and color not in df.columns:
                raise ValueError("Choose a color column inside the study scope.")
            proj_df = predictors.drop(columns=[color], errors="ignore")
            if proj_df.select_dtypes(include=[np.number]).shape[1] < 2:
                proj_df, _ = _prepare_numeric_feature_frame(proj_df, set(), None, min_cols=2)
            if color is not None:
                proj_df = proj_df.assign(**{color: df[color]})
            out["manifold"] = ManifoldProjector().project(
                proj_df,
                method=req.manifold_method,
                color_col=color,
            )
        except Exception as exc:
            logger.warning("manifold_projection_failed", error_type=type(exc).__name__)
            out["manifold"] = {"error": "The manifold method could not evaluate the selected study columns."}
    state["uap_mining_result"] = out
    record_lineage(state, "Data Mining", f"Executed Data Mining ({mode})", f"Tasks={', '.join(tasks)}")
    return _cache_and_respond(state, sid, "s02_mining", dict(out))


# ---------------------------------------------------------------------------
# Studio 03 — Universal Diagnostics, Baselines & Risk
# ---------------------------------------------------------------------------
from omniforge.api.studios.charts_hub import router as charts_hub_router  # noqa: E402
from omniforge.api.studios.s00_ingestion import router as s00_router  # noqa: E402
from omniforge.api.studios.s01_cleaning import router as s01_router  # noqa: E402
from omniforge.api.studios.s02_mining import resolve_mining_target  # noqa: E402
from omniforge.api.studios.s02_mining import router as s02_router  # noqa: E402
from omniforge.api.studios.s03_eda import router as s03_router  # noqa: E402
from omniforge.api.studios.s04_stats import router as s04_router  # noqa: E402
from omniforge.api.studios.s05_features import router as s05_router  # noqa: E402
from omniforge.api.studios.s06_automl import router as s06_router  # noqa: E402
from omniforge.api.studios.s07_advanced import router as s07_router  # noqa: E402
from omniforge.api.studios.s08_causal import router as s08_router  # noqa: E402
from omniforge.api.studios.s09_opt_sim import router as s09_router  # noqa: E402
from omniforge.api.studios.s10_rai import router as s10_router  # noqa: E402
from omniforge.api.studios.s11_mlops import router as s11_router  # noqa: E402
from omniforge.api.studios.s12_reports import router as s12_router  # noqa: E402
from omniforge.api.studios.s13_copilot import router as s13_router  # noqa: E402
from omniforge.api.studios.s14_supervisory import router as s14_router  # noqa: E402

router.include_router(charts_hub_router)
router.include_router(s00_router)
router.include_router(s01_router)
router.include_router(s02_router)
router.include_router(s03_router)
router.include_router(s04_router)
router.include_router(s05_router)
router.include_router(s06_router)
router.include_router(s07_router)
router.include_router(s08_router)
router.include_router(s09_router)
router.include_router(s10_router)
router.include_router(s11_router)
router.include_router(s12_router)
router.include_router(s13_router)
router.include_router(s14_router)


