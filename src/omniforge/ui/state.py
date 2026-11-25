"""Thread-Safe Server-Side Workspace State, Multi-Target, Split Lock, Exclusions, Lineage & .uap Project Store."""

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from omniforge.core.logging import get_logger
from omniforge.core.types import TaskType
from omniforge.data.format_audit import DataFormatAuditor
from omniforge.data.intake_audit import InitialIntakeAuditor
from omniforge.data.splitting import LeakageSafeSplitter, SplitConfig

logger = get_logger(__name__)

_DOWNSTREAM_KEYS: tuple[str, ...] = (
    "uap_quality",
    "uap_clean_report",
    "uap_validation_report",
    "uap_eda_report",
    "uap_feature_report",
    "uap_automl",
    "uap_target_automls",
    "uap_target_models",
    "uap_active_model",
    "uap_active_model_name",
    "uap_active_model_metrics",
    "uap_active_feature_cols",
    "uap_active_input_preprocessor",
    "uap_active_X_test",
    "uap_active_y_test",
    "uap_model_equations",
    "uap_baseline_equations",
    "uap_dl_benchmark_df",
    "uap_dl_models",
    "uap_future_prediction_result",
    "uap_future_source_type",
    "uap_shap_result",
    "uap_fairness_result",
    "uap_model_card",
    "uap_forecast_report",
    "uap_forecast_result",
    "uap_hypothesis_battery",
    "uap_future_scores",
    "uap_causal_result",
    "uap_opt_result",
    "uap_sim_result",
    "uap_dag_result",
    "uap_report_payload",
    "uap_report_config",
    "uap_diag_report",
    "uap_diag_criteria",
    "uap_baseline_results",
    "uap_baseline_models",
    "uap_nonparametric_models",
    "uap_risk_report",
    "uap_clustering_result",
    "uap_clustering_df",
    "uap_anomaly_df",
    "uap_anomaly_meta",
    "uap_mining_result",
    "uap_supervisory_result",
    "uap_supervisory_scored_df",
    "uap_supervisory_models",
    "uap_s02_pending_feedback",
    "uap_fs_selected_cols",
    "uap_fs_scores",
    "uap_fs_dropped",
    "uap_split_lock",
    "uap_split_audit_report",
    "uap_serving_engine",
)

# Stage-aware preservation groups so committing an earlier CRISP-DM stage never wipes completed upstream status
_STAGE_PRESERVE_KEYS: dict[str, tuple[str, ...]] = {
    "s01": ("uap_quality", "uap_clean_report", "uap_s01_cleaned"),
    "s02_cluster": (
        "uap_quality",
        "uap_clean_report",
        "uap_s01_cleaned",
        "uap_s02_cluster_committed",
        "uap_s02_cluster_summary",
        "uap_clustering_result",
        "uap_s02_anomaly_committed",
        "uap_s02_anomaly_summary",
    ),
    "s02_anomaly": (
        "uap_quality",
        "uap_clean_report",
        "uap_s01_cleaned",
        "uap_s02_cluster_committed",
        "uap_s02_cluster_summary",
        "uap_s02_anomaly_committed",
        "uap_s02_anomaly_summary",
        "uap_anomaly_meta",
    ),
    "s05_features": (
        "uap_quality",
        "uap_clean_report",
        "uap_s01_cleaned",
        "uap_s02_cluster_committed",
        "uap_s02_cluster_summary",
        "uap_s02_anomaly_committed",
        "uap_s02_anomaly_summary",
    ),
}


def default_workspace_dict() -> dict[str, Any]:
    """Return a clean initial server-side workspace state dictionary."""
    return {
        "uap_df": None,
        "uap_raw_df": None,
        "uap_prev_df": None,
        "uap_dataset_name": None,
        "uap_dataset_source": None,
        "uap_target": None,
        "uap_targets": [],
        "uap_excluded_columns": [],
        "uap_format_audit": None,
        "uap_intake_audit": None,
        "uap_deleted_records_log": [],
        "uap_s01_cleaned": False,
        "uap_s02_last_cluster_deleted": None,
        "uap_s02_last_anomaly_deleted": None,
        "uap_s02_cluster_committed": False,
        "uap_s02_cluster_summary": None,
        "uap_s02_anomaly_committed": False,
        "uap_s02_anomaly_summary": None,
        "uap_mining_result": None,
        "uap_baseline_models": {},
        "uap_nonparametric_models": {},
        "uap_studio_results": {},
        "uap_supervisory_result": None,
        "uap_supervisory_scored_df": None,
        "uap_supervisory_history": [],
        "uap_supervisory_models": {},
        "uap_s02_pending_feedback": None,
        "uap_target_models": {},
        "uap_target_automls": {},
        "uap_model_equations": {},
        "uap_baseline_equations": {},
        "uap_dl_benchmark_df": None,
        "uap_dl_models": {},
        "uap_future_prediction_result": None,
        "uap_future_source_type": None,
        "uap_lineage": [],
        "uap_exclusions": [],
        "uap_quality": None,
        "uap_clean_report": None,
        "uap_validation_report": None,
        "uap_eda_report": None,
        "uap_feature_report": None,
        "uap_automl": None,
        "uap_active_model": None,
        "uap_active_model_name": None,
        "uap_active_model_metrics": {},
        "uap_active_feature_cols": [],
        "uap_active_input_preprocessor": None,
        "uap_active_X_test": None,
        "uap_active_y_test": None,
        "uap_shap_result": None,
        "uap_fairness_result": None,
        "uap_model_card": None,
        "uap_forecast_report": None,
        "uap_causal_result": None,
        "uap_opt_result": None,
        "uap_sim_result": None,
        "uap_dag_result": None,
        "uap_dag_nodes": [
            {"id": "clean", "label": "Clean Data", "op": "clean"},
            {"id": "features", "label": "Engineer Features", "op": "features"},
            {"id": "train", "label": "Train AutoML", "op": "train"},
        ],
        "uap_dag_edges": [
            {"source": "clean", "target": "features", "label": "cleaned_df"},
            {"source": "features", "target": "train", "label": "feature_matrix"},
        ],
        "uap_report_payload": None,
        "uap_report_config": None,
        "uap_diag_report": None,
        "uap_diag_criteria": None,
        "uap_baseline_results": None,
        "uap_risk_report": None,
        "uap_clustering_result": None,
        "uap_clustering_df": None,
        "uap_anomaly_df": None,
        "uap_anomaly_meta": None,
        "uap_fs_selected_cols": None,
        "uap_fs_scores": None,
        "uap_fs_dropped": None,
        "uap_split_lock": None,
        "uap_split_audit_report": None,
        "uap_pending_excel": None,
        "uap_copilot": None,
        "uap_copilot_history": [],
        "uap_chat_history": [],
        "uap_active_studio": "01",
        "uap_selected_studio": "01",
        "uap_studio_tabs": {},
        "uap_s01_src_mode": "upload",
        "uap_s06_fut_src": "manual",
        "uap_last_clean_diff": None,
        "uap_s01_pending_feedback": None,
        "_uap_remount_studio": False,
        "uap_theme": "dark",
        "uap_project_name": "omniforge_analytics_project",
        "uap_project_notes": "",
        "uap_last_project_notice": None,
        "uap_experiment_runs": [],
        "uap_serving_engine": None,
    }


class ServerSessionRegistry:
    """Thread-safe in-process store mapping browser ``session_id`` values to server-side workspaces."""

    def __init__(self, ttl_seconds: int = 86400, max_sessions: int = 128) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._last_access: dict[str, float] = {}
        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [sid for sid, ts in self._last_access.items() if now - ts > self._ttl_seconds]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._last_access.pop(sid, None)
        while len(self._sessions) > self._max_sessions:
            oldest = min(self._last_access, key=lambda k: self._last_access[k])
            self._sessions.pop(oldest, None)
            self._last_access.pop(oldest, None)

    def get(self, session_id: str | None) -> dict[str, Any]:
        """Return existing workspace dict for ``session_id``, creating one cleanly if absent."""
        sid = session_id or "default"
        with self._lock:
            self._evict_expired()
            if sid not in self._sessions:
                self._sessions[sid] = default_workspace_dict()
            self._last_access[sid] = time.monotonic()
            return self._sessions[sid]

    def reset(self, session_id: str | None) -> dict[str, Any]:
        """Replace the session workspace with a fresh default state."""
        sid = session_id or "default"
        with self._lock:
            self._sessions[sid] = default_workspace_dict()
            self._last_access[sid] = time.monotonic()
            return self._sessions[sid]

    def delete(self, session_id: str) -> None:
        """Remove a session from memory."""
        with self._lock:
            self._sessions.pop(session_id, None)
            self._last_access.pop(session_id, None)

    def new_session_id(self) -> str:
        """Generate a unique browser-tab session identifier."""
        return str(uuid.uuid4())


SESSION_STORE = ServerSessionRegistry()


def get_session_state(session_id: str | None, *, tenant_id: str | None = None) -> dict[str, Any]:
    """Retrieve the active server-side workspace dict scoped by ``tenant_id`` and ``session_id``."""
    raw_sid = (session_id or "default").strip() or "default"
    scoped_sid = f"{tenant_id}::{raw_sid}" if tenant_id else raw_sid
    return SESSION_STORE.get(scoped_sid)


def _reset_downstream_state(state: dict[str, Any], *, keep_keys: set[str] | None = None) -> None:
    """Clear cached downstream reports/models whenever the dataset changes, optionally preserving upstream keys."""
    preserve = keep_keys or set()
    for key in _DOWNSTREAM_KEYS:
        if key in preserve:
            continue
        if key in {
            "uap_target_automls",
            "uap_target_models",
            "uap_model_equations",
            "uap_baseline_equations",
            "uap_baseline_models",
            "uap_nonparametric_models",
            "uap_dl_models",
            "uap_supervisory_models",
        }:
            state[key] = {}
        elif key == "uap_active_model_metrics":
            state[key] = {}
        elif key == "uap_active_feature_cols":
            state[key] = []
        else:
            state[key] = None


_STUDIO_RESULT_KEEP_PREFIXES: dict[str, tuple[str, ...]] = {
    "s01": ("s01_",),
    # The committed operation remains as a labelled historical preview; other
    # Studio 02 analyses no longer describe the changed dataset.
    "s02_cluster": ("s01_", "s02_clustering"),
    "s02_anomaly": ("s01_", "s02_anomaly"),
    # Adding or dropping features changes the study used by earlier analyses.
    "s05_features": ("s01_",),
}

# These operate on explicitly supplied text/images, not the workspace dataset.
_DATASET_INDEPENDENT_RESULTS = frozenset({"s09_nlp", "s09_vision"})


def save_studio_result(state: dict[str, Any], key: str, payload: dict[str, Any]) -> None:
    """Persist the latest serialized UI result payload for a studio card/panel so revisits restore instantly."""
    cache = state.setdefault("uap_studio_results", {})
    if isinstance(cache, dict):
        # Avoid nesting bulky 'state' snapshots inside cached studio results
        cleaned = {k: v for k, v in payload.items() if k != "state"}
        cache[str(key)] = cleaned


def get_studio_results(state: dict[str, Any]) -> dict[str, Any]:
    """Return all cached JSON-serializable studio UI results for the current workspace session."""
    cache = state.get("uap_studio_results")
    return dict(cache) if isinstance(cache, dict) else {}


def _prune_studio_results_for_stage(state: dict[str, Any], stage_key: str | None) -> None:
    """Preserve upstream cached studio payloads while clearing genuinely invalidated downstream results."""
    cache = state.get("uap_studio_results")
    if not isinstance(cache, dict):
        state["uap_studio_results"] = {}
        return
    prefixes = _STUDIO_RESULT_KEEP_PREFIXES.get(stage_key or "", ())
    state["uap_studio_results"] = {
        k: v for k, v in cache.items()
        if (k in _DATASET_INDEPENDENT_RESULTS or any(str(k).startswith(p) for p in prefixes))
        and not (k == "s01_clean" and isinstance(v, dict) and v.get("preview_only"))
    }


def _prune_scope_results(state: dict[str, Any], *, split_only: bool = False) -> None:
    """Keep only results whose inputs are unchanged by a study or split change."""
    cache = get_studio_results(state)
    keep = set(_DATASET_INDEPENDENT_RESULTS) | {"s01_quality"}
    clean = cache.get("s01_clean")
    if isinstance(clean, dict) and not clean.get("preview_only"):
        # A committed cleaning report is historical provenance, not a preview.
        keep.add("s01_clean")
    if split_only:
        # These analyses use the same full study data, not the train/test lock.
        keep.update({
            "s02_mining", "s02_advanced_mining", "s04_eda", "s04_hypothesis",
            "s08_forecast", "s08_causal", "s08_optimize", "s08_simulate",
            "s09_graph", "s09_survival",
        })
    state["uap_studio_results"] = {k: v for k, v in cache.items() if k in keep}


def _invalidate_study_scope(state: dict[str, Any], *, preserve_split: bool) -> None:
    """Invalidate study-dependent previews/models while preserving committed provenance."""
    lock = state.get("uap_split_lock")
    audit = state.get("uap_split_audit_report")
    _reset_downstream_state(state, keep_keys={"uap_quality", "uap_clean_report"})
    _prune_scope_results(state)
    if preserve_split and isinstance(lock, dict) and lock.get("active"):
        state["uap_split_lock"] = lock
        state["uap_split_audit_report"] = audit


def _invalidate_split_results(state: dict[str, Any]) -> None:
    """Invalidate fits based on the old train/test split, retaining full-data analyses."""
    _reset_downstream_state(
        state,
        keep_keys={
            "uap_quality", "uap_clean_report", "uap_mining_result", "uap_eda_report",
            "uap_hypothesis_battery", "uap_forecast_report", "uap_forecast_result",
            "uap_causal_result", "uap_opt_result", "uap_sim_result",
        },
    )
    _prune_scope_results(state, split_only=True)


def record_lineage(state: dict[str, Any], stage: str, action: str, details: str = "") -> None:
    """Append an auditable event to the session data-lineage log."""
    df: pd.DataFrame | None = state.get("uap_df")
    state.setdefault("uap_lineage", []).append(
        {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "stage": stage,
            "action": action,
            "details": details,
            "rows": int(len(df)) if df is not None else 0,
            "columns": int(df.shape[1]) if df is not None else 0,
        }
    )


def record_exclusion(
    state: dict[str, Any],
    *,
    stage: str,
    exclusion_type: str,
    affected_scope: str,
    count: int,
    total_before: int,
    criterion_rule: str,
    reason: str,
    sample_identifiers: list[Any] | None = None,
) -> None:
    """Record a formal auditable data exclusion event for executive & forensic reporting."""
    pct = round((float(count) / max(float(total_before), 1.0)) * 100.0, 3) if total_before > 0 else 0.0
    state.setdefault("uap_exclusions", []).append(
        {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "stage": str(stage),
            "exclusion_type": str(exclusion_type),
            "affected_scope": str(affected_scope),
            "count": int(count),
            "pct_of_dataset": pct,
            "criterion_rule": str(criterion_rule),
            "reason": str(reason),
            "sample_identifiers": [str(x) for x in (sample_identifiers or [])[:12]],
        }
    )


def get_excluded_columns(state: dict[str, Any], df: pd.DataFrame | None = None) -> list[str]:
    """Return user-excluded columns currently present in ``df``."""
    active_df = df if df is not None else state.get("uap_df")
    if active_df is None or active_df.empty:
        return []
    cols = set(active_df.columns)
    raw = state.get("uap_excluded_columns") or []
    valid = [c for c in raw if c in cols]
    # Never allow all columns to be excluded simultaneously
    if len(valid) >= len(active_df.columns):
        valid = valid[: max(0, len(active_df.columns) - 1)]
    state["uap_excluded_columns"] = valid
    return valid


def get_audit_anomaly_dataframe(state: dict[str, Any], df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the unclipped raw audit/anomaly view when available, falling back to the active analysis view."""
    raw = state.get("uap_raw_df")
    if isinstance(raw, pd.DataFrame) and not raw.empty:
        return get_analysis_dataframe(state, raw)
    return get_analysis_dataframe(state, df)


def get_analysis_dataframe(state: dict[str, Any], df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the active DataFrame with user-excluded columns removed for tests & studies.

    Excluded columns remain intact in ``state['uap_df']`` and ``state['uap_raw_df']`` so
    exclusions are completely non-destructive and reversible at any time.
    """
    active_df = df if df is not None else state.get("uap_df")
    if active_df is None:
        return pd.DataFrame()
    excluded = get_excluded_columns(state, active_df)
    if not excluded:
        return active_df
    remaining = [c for c in active_df.columns if c not in set(excluded)]
    if not remaining:
        return active_df
    return active_df[remaining].copy()


def get_study_df(state: dict[str, Any]) -> pd.DataFrame | None:
    """Return the active study DataFrame (with excluded columns omitted), or ``None`` if no dataset is loaded."""
    if state.get("uap_df") is None:
        return None
    return get_analysis_dataframe(state)


def set_excluded_columns(
    state: dict[str, Any],
    excluded_cols: list[str],
    df: pd.DataFrame | None = None,
) -> list[str]:
    """Update the list of columns excluded from all downstream statistical tests & studies."""
    active_df = df if df is not None else state.get("uap_df")
    if active_df is None or active_df.empty:
        state["uap_excluded_columns"] = []
        return []
    all_cols = list(active_df.columns)
    valid = [c for c in dict.fromkeys(excluded_cols) if c in all_cols]
    if len(valid) >= len(all_cols):
        valid = valid[: max(0, len(all_cols) - 1)]

    prev_excl = list(state.get("uap_excluded_columns") or [])
    previous_target = state.get("uap_target")
    previous_lock = state.get("uap_split_lock")
    state["uap_excluded_columns"] = valid

    # Ensure active targets remain inside the included study columns
    incl_cols = [c for c in all_cols if c not in set(valid)]
    cur_targets = [c for c in (state.get("uap_targets") or []) if c in incl_cols]
    if not cur_targets and incl_cols:
        cur_targets = [incl_cols[-1]]
    state["uap_targets"] = cur_targets
    state["uap_target"] = cur_targets[0] if cur_targets else None

    if valid != prev_excl:
        # Cluster/EDA/diagnostic previews also use study scope. Preserve the
        # committed stage flags, but not analyses computed with old columns.
        locked_target = previous_lock.get("target_col") if isinstance(previous_lock, dict) else None
        preserve_split = previous_target == state["uap_target"] and locked_target not in valid
        _invalidate_study_scope(state, preserve_split=preserve_split)
        if valid:
            record_exclusion(
                state,
                stage="Studio 01 · Study Scope",
                exclusion_type="User-Excluded Study Columns",
                affected_scope=f"{len(valid)} column(s): {', '.join(valid[:8])}",
                count=len(valid),
                total_before=len(all_cols),
                criterion_rule="Manual / auto study column exclusion",
                reason="Excluded by user from downstream tests, clustering, EDA, feature engineering & modeling.",
                sample_identifiers=valid,
            )
            record_lineage(
                state,
                "Column Exclusion",
                f"Excluded {len(valid)} column(s) from studies",
                f"Excluded: {', '.join(valid[:8])} | Active study cols: {len(incl_cols)}/{len(all_cols)}",
            )
        else:
            record_lineage(
                state,
                "Column Exclusion",
                "Cleared column exclusions (all columns included)",
                f"Active study cols: {len(all_cols)}/{len(all_cols)}",
            )
        request_studio_remount(state)
    return valid


def run_format_audit(state: dict[str, Any], df: pd.DataFrame | None = None) -> dict[str, Any] | None:
    """Run :class:`DataFormatAuditor` on ``df`` and cache its serialized report in workspace state."""
    target_df = df if df is not None else state.get("uap_df")
    if not isinstance(target_df, pd.DataFrame) or target_df.empty:
        state["uap_format_audit"] = None
        return None
    try:
        report = DataFormatAuditor().audit(target_df)
        dumped = report.model_dump()
        state["uap_format_audit"] = dumped
        return dumped
    except Exception:
        state["uap_format_audit"] = None
        return None


def record_deleted_data(
    state: dict[str, Any],
    *,
    stage: str,
    reason: str,
    removed_df: pd.DataFrame | None = None,
    removed_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Store exact removed rows and/or dropped columns in the forensic workspace deletion log."""
    clean_removed = pd.DataFrame()
    if isinstance(removed_df, pd.DataFrame) and not removed_df.empty:
        clean_removed = removed_df.copy()
        if "_original_row_index" not in clean_removed.columns:
            clean_removed.insert(0, "_original_row_index", list(clean_removed.index))
        if "_deletion_stage" not in clean_removed.columns:
            clean_removed.insert(1, "_deletion_stage", str(stage))
        if "_deletion_reason" not in clean_removed.columns:
            clean_removed.insert(2, "_deletion_reason", str(reason))
        clean_removed = clean_removed.reset_index(drop=True)

    dropped_cols = [str(c) for c in (removed_columns or [])]
    entry: dict[str, Any] = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "stage": str(stage),
        "reason": str(reason),
        "rows_removed": int(len(clean_removed)),
        "columns_removed": dropped_cols,
        "removed_df": clean_removed,
    }
    state.setdefault("uap_deleted_records_log", []).append(entry)
    return entry


def get_all_deleted_records_df(state: dict[str, Any]) -> pd.DataFrame:
    """Return a single unified DataFrame of all rows deleted across Cleaning & Clustering/Anomaly stages."""
    log = state.get("uap_deleted_records_log") or []
    frames = [
        entry["removed_df"]
        for entry in log
        if isinstance(entry, dict) and isinstance(entry.get("removed_df"), pd.DataFrame) and not entry["removed_df"].empty
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def get_selected_targets(state: dict[str, Any], df: pd.DataFrame | None = None) -> list[str]:
    """Return valid selected target column(s) present in ``df`` (respecting study exclusions)."""
    active_df = df if df is not None else state.get("uap_df")
    if active_df is None or active_df.empty:
        return []
    cols = list(active_df.columns)
    excluded_set = set(get_excluded_columns(state, active_df))
    included_cols = [c for c in cols if c not in excluded_set] or cols
    raw = state.get("uap_targets") or []
    valid = [c for c in raw if c in included_cols]
    if not valid:
        single = state.get("uap_target")
        if single in included_cols:
            valid = [single]
        else:
            valid = [included_cols[-1]]
    state["uap_targets"] = valid
    state["uap_target"] = valid[0]
    return valid


def get_studio_tab(state: dict[str, Any], studio_code: str, default: str) -> str:
    """Return the persisted active tab value for ``studio_code`` (or ``default`` if unset)."""
    tabs = state.get("uap_studio_tabs")
    if isinstance(tabs, dict):
        val = tabs.get(studio_code)
        if isinstance(val, str) and val:
            return val
    return default


def set_studio_tab(state: dict[str, Any], studio_code: str, tab_value: str | None) -> None:
    """Persist the user's currently selected tab for ``studio_code`` across re-renders."""
    if not tab_value:
        return
    tabs = state.setdefault("uap_studio_tabs", {})
    if isinstance(tabs, dict):
        tabs[str(studio_code)] = str(tab_value)


def request_studio_remount(state: dict[str, Any]) -> None:
    """Mark the active studio container for a structural refresh on the next state revision.

    Used only when dataset columns, preview rows, split lock, pending Excel sheets,
    DAG nodes, or saved project/version lists structurally change. Non-structural
    analytics callbacks refresh sidebar/banner telemetry without destroying the studio DOM.
    """
    state["_uap_remount_studio"] = True


def set_selected_targets(state: dict[str, Any], targets: list[str], df: pd.DataFrame | None = None) -> list[str]:
    """Update selected target column(s) safely and un-exclude any newly selected target."""
    active_df = df if df is not None else state.get("uap_df")
    if active_df is None or active_df.empty:
        state["uap_targets"] = []
        state["uap_target"] = None
        return []
    cols = list(active_df.columns)
    previous = get_selected_targets(state, active_df)
    cur_excl = list(state.get("uap_excluded_columns") or [])
    previous_lock = state.get("uap_split_lock")
    valid = [c for c in dict.fromkeys(targets) if c in cols]
    if not valid:
        valid = [cols[-1]]
    # Selecting a column as target automatically includes it in study scope.
    remaining_exclusions = [c for c in cur_excl if c not in set(valid)]
    state["uap_excluded_columns"] = remaining_exclusions
    state["uap_targets"] = valid
    state["uap_target"] = valid[0]
    if valid != previous or remaining_exclusions != cur_excl:
        locked_target = previous_lock.get("target_col") if isinstance(previous_lock, dict) else None
        preserve_split = bool(previous and previous[0] == valid[0] and locked_target not in remaining_exclusions)
        _invalidate_study_scope(state, preserve_split=preserve_split)
        request_studio_remount(state)
    return valid


def _ensure_unique_str_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame column labels are non-empty unique strings so JSON/API lookups never mismatch."""
    raw_cols = list(df.columns)
    if (
        all(isinstance(c, str) and bool(c.strip()) for c in raw_cols)
        and len(set(raw_cols)) == len(raw_cols)
    ):
        return df
    out = df.copy()
    seen: dict[str, int] = {}
    clean_names: list[str] = []
    for idx, c in enumerate(raw_cols):
        base = str(c).strip() if c is not None and str(c).strip() else f"column_{idx + 1}"
        cnt = seen.get(base, 0) + 1
        seen[base] = cnt
        clean_names.append(base if cnt == 1 else f"{base}_{cnt}")
    out.columns = pd.Index(clean_names)
    return out


def load_new_dataset(
    state: dict[str, Any],
    df: pd.DataFrame,
    name: str,
    source: str,
    default_target: str | None = None,
    default_targets: list[str] | None = None,
) -> None:
    """Store a newly ingested DataFrame, snapshot raw copy, run format audit, and clear stale caches."""
    clean_df = _ensure_unique_str_columns(df.copy())
    state["uap_df"] = clean_df
    state["uap_raw_df"] = clean_df.copy()
    state["uap_prev_df"] = None
    state["uap_dataset_name"] = name
    state["uap_dataset_source"] = source
    state["uap_excluded_columns"] = []
    state["uap_exclusions"] = []
    state["uap_deleted_records_log"] = []
    state["uap_s01_cleaned"] = False
    state["uap_s02_last_cluster_deleted"] = None
    state["uap_s02_last_anomaly_deleted"] = None
    state["uap_s02_cluster_committed"] = False
    state["uap_s02_cluster_summary"] = None
    state["uap_s02_anomaly_committed"] = False
    state["uap_s02_anomaly_summary"] = None
    state["uap_supervisory_history"] = []
    state["uap_supervisory_models"] = {}
    _prune_studio_results_for_stage(state, None)
    state["uap_s05_engineered"] = False
    state["uap_pending_excel"] = None
    state["uap_last_clean_diff"] = None

    cols = list(clean_df.columns)
    if default_targets:
        valid_t = [c for c in default_targets if c in cols]
    elif default_target and default_target in cols:
        valid_t = [default_target]
    elif cols:
        valid_t = [cols[-1]]
    else:
        valid_t = []

    state["uap_targets"] = valid_t
    state["uap_target"] = valid_t[0] if valid_t else None
    _reset_downstream_state(state)
    audit = run_format_audit(state, clean_df)
    try:
        intake = InitialIntakeAuditor().audit(clean_df, target_col=state["uap_target"])
    except Exception:
        intake = None
    state["uap_intake_audit"] = intake
    issue_note = (
        f" | Intake defects: {intake['defect_counts']['total']} ({intake['overall_health']})"
        if isinstance(intake, dict)
        else (f" | Format issues: {audit['total_issues']}" if isinstance(audit, dict) else "")
    )
    request_studio_remount(state)
    record_lineage(state, "Ingest", f"Loaded '{name}' from {source}", f"Shape: {len(clean_df):,} × {clean_df.shape[1]}{issue_note}")


def clear_workspace_dataset(state: dict[str, Any]) -> None:
    """Clear the active dataset and all downstream analysis artifacts cleanly."""
    prev_name = state.get("uap_dataset_name") or "dataset"
    state["uap_df"] = None
    state["uap_raw_df"] = None
    state["uap_prev_df"] = None
    state["uap_dataset_name"] = None
    state["uap_dataset_source"] = None
    state["uap_target"] = None
    state["uap_targets"] = []
    state["uap_excluded_columns"] = []
    state["uap_format_audit"] = None
    state["uap_intake_audit"] = None
    state["uap_exclusions"] = []
    state["uap_deleted_records_log"] = []
    state["uap_s01_cleaned"] = False
    state["uap_s02_last_cluster_deleted"] = None
    state["uap_s02_last_anomaly_deleted"] = None
    state["uap_s02_cluster_committed"] = False
    state["uap_s02_cluster_summary"] = None
    state["uap_s02_anomaly_committed"] = False
    state["uap_s02_anomaly_summary"] = None
    state["uap_supervisory_history"] = []
    state["uap_supervisory_models"] = {}
    _prune_studio_results_for_stage(state, None)
    state["uap_s05_engineered"] = False
    state["uap_pending_excel"] = None
    state["uap_last_clean_diff"] = None
    _reset_downstream_state(state)
    request_studio_remount(state)
    record_lineage(state, "Workspace", f"Cleared active dataset '{prev_name}'", "Ready for next dataset upload")


def _locked_positions(values: Any, n_rows: int) -> list[int] | None:
    """Validate stored positional indices without silently dropping bad entries."""
    if not isinstance(values, (list, tuple, np.ndarray, pd.Index)):
        return None
    try:
        positions = list(values)
    except TypeError:
        return None
    if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= int(i) < n_rows
           for i in positions):
        return None
    result = [int(i) for i in positions]
    return result if len(set(result)) == len(result) else None


def _locked_row_id_sets(lock: dict[str, Any]) -> tuple[set[Any], set[Any]] | None:
    train = lock.get("train_row_ids")
    test = lock.get("test_row_ids")
    if not isinstance(train, (list, tuple, np.ndarray, pd.Index)) or not isinstance(test, (list, tuple, np.ndarray, pd.Index)):
        return None
    try:
        train_set, test_set = set(train), set(test)
    except TypeError:
        return None
    if len(train_set) != len(train) or len(test_set) != len(test) or train_set & test_set:
        return None
    return train_set, test_set


def _realign_split_lock(
    lock: dict[str, Any], previous: pd.DataFrame, updated: pd.DataFrame, *, row_index_preserved: bool,
) -> dict[str, Any] | None:
    """Map a locked split to surviving rows; fail closed if identity is ambiguous.

    Explicit unique row IDs are authoritative. Otherwise the caller must know
    the transformation preserves index labels: equal labels alone do not prove
    identity, since filtering/reordering followed by reset_index can reuse them.
    Stored train/test indices are *positions*, not index labels.
    """
    for key in ("target_col", "group_col", "time_col"):
        column = lock.get(key)
        if column and column not in updated.columns:
            return None

    row_id_col = lock.get("row_id_col")
    has_row_ids = row_id_col is not None or lock.get("train_row_ids") is not None or lock.get("test_row_ids") is not None
    if has_row_ids:
        if not isinstance(row_id_col, str) or row_id_col not in previous.columns or row_id_col not in updated.columns:
            return None
        old_ids, new_ids = previous[row_id_col], updated[row_id_col]
        row_id_sets = _locked_row_id_sets(lock)
        if row_id_sets is None:
            return None
        try:
            if (not old_ids.is_unique or not new_ids.is_unique or old_ids.isna().any()
                    or new_ids.isna().any() or not new_ids.isin(old_ids).all()):
                return None
            train_ids, test_ids = row_id_sets
            if not (train_ids | test_ids).issubset(set(old_ids)):
                return None
            train_idx = [i for i, row_id in enumerate(new_ids) if row_id in train_ids]
            test_idx = [i for i, row_id in enumerate(new_ids) if row_id in test_ids]
        except TypeError:
            return None
    else:
        if not row_index_preserved or not previous.index.is_unique or not updated.index.is_unique:
            return None
        try:
            old_positions = previous.index.get_indexer(updated.index)
        except (TypeError, ValueError):
            return None
        if (old_positions < 0).any():
            return None
        old_to_new = {int(old): new for new, old in enumerate(old_positions)}
        train_old = _locked_positions(lock.get("train_indices"), len(previous))
        test_old = _locked_positions(lock.get("test_indices"), len(previous))
        if train_old is None or test_old is None or set(train_old) & set(test_old):
            return None
        train_idx = [old_to_new[i] for i in train_old if i in old_to_new]
        test_idx = [old_to_new[i] for i in test_old if i in old_to_new]

    if len(train_idx) < 5 or len(test_idx) < 2:
        return None
    result = dict(lock)
    result.update(train_indices=train_idx, test_indices=test_idx, n_train=len(train_idx), n_test=len(test_idx))
    if has_row_ids:
        result["train_row_ids"] = updated[row_id_col].iloc[train_idx].tolist()
        result["test_row_ids"] = updated[row_id_col].iloc[test_idx].tolist()

    # An audit calculated on the old rows/columns cannot describe the new data.
    try:
        config = SplitConfig(group_col=lock.get("group_col"), time_col=lock.get("time_col"))
        audit = LeakageSafeSplitter(config).verify_no_leakage(
            updated,
            train_idx=np.asarray(train_idx, dtype=int),
            test_idx=np.asarray(test_idx, dtype=int),
            strategy_used=str(lock.get("strategy") or "random"),
        )
        if not audit.passed:
            logger.warning("split_lock_reaudit_not_passed", warnings=audit.warnings)
            return None
        result["audit_report"] = audit.model_dump()
    except Exception as exc:
        logger.warning("split_lock_reaudit_failed", error=str(exc))
        return None
    return result


def update_active_dataset(
    state: dict[str, Any],
    new_df: pd.DataFrame,
    stage: str,
    action: str,
    details: str = "",
    preserve_stage: str | None = None,
    *,
    row_index_preserved: bool = False,
) -> None:
    """Update the dataset with Undo, safe Split Lock realignment and stage-aware invalidation.

    Set ``row_index_preserved`` only for transformations known to keep each
    surviving row's original index label. Otherwise a lock without explicit
    row IDs is invalidated even if the new index happens to match the old one.
    """
    previous = state.get("uap_df")
    if isinstance(previous, pd.DataFrame):
        state["uap_prev_df"] = previous.copy()
    updated = new_df.copy()
    state["uap_df"] = updated

    # Prune excluded columns if any were dropped by cleaning/feature selection
    state["uap_excluded_columns"] = [
        c for c in (state.get("uap_excluded_columns") or []) if c in updated.columns
    ]

    prev_lock = state.get("uap_split_lock")
    had_lock = isinstance(prev_lock, dict) and bool(prev_lock.get("active"))
    preserved_lock = (
        _realign_split_lock(prev_lock, previous, updated, row_index_preserved=row_index_preserved)
        if isinstance(prev_lock, dict) and had_lock and isinstance(previous, pd.DataFrame) else None
    )

    # Auto-infer stage preservation key from stage name if not explicitly provided
    stage_key = preserve_stage
    if stage_key is None:
        s_low = stage.lower()
        if "clean" in s_low:
            stage_key = "s01"
        elif "cluster" in s_low:
            stage_key = "s02_cluster"
        elif "anomal" in s_low:
            stage_key = "s02_anomaly"
        elif "feature" in s_low or "select" in s_low:
            stage_key = "s05_features"

    keep_keys = set(_STAGE_PRESERVE_KEYS.get(stage_key or "", ()))
    saved_vals = {k: state.get(k) for k in keep_keys if k in state}

    get_selected_targets(state, updated)
    _reset_downstream_state(state, keep_keys=keep_keys)
    for k, v in saved_vals.items():
        state[k] = v
    _prune_studio_results_for_stage(state, stage_key)

    if preserved_lock is not None:
        state["uap_split_lock"] = preserved_lock
        state["uap_split_audit_report"] = preserved_lock.get("audit_report")
    elif had_lock:
        record_lineage(state, "Split Lock", "Unlocked split after dataset update", "Row identity, split membership or leakage audit could not be safely preserved.")
    run_format_audit(state, updated)
    try:
        state["uap_intake_audit"] = InitialIntakeAuditor().audit(updated, target_col=state.get("uap_target"))
    except Exception as exc:
        state["uap_intake_audit"] = None
        logger.warning("intake_audit_refresh_failed", stage=stage, error=str(exc))
    request_studio_remount(state)
    record_lineage(state, stage, action, details)


def undo_last_dataset_change(state: dict[str, Any]) -> bool:
    """Revert the active dataset by one step."""
    prev = state.get("uap_prev_df")
    if prev is None:
        return False
    state["uap_df"] = prev.copy()
    state["uap_prev_df"] = None
    state["uap_excluded_columns"] = [
        c for c in (state.get("uap_excluded_columns") or []) if c in state["uap_df"].columns
    ]
    get_selected_targets(state, state["uap_df"])
    _reset_downstream_state(state)
    _prune_studio_results_for_stage(state, None)
    run_format_audit(state, state["uap_df"])
    try:
        state["uap_intake_audit"] = InitialIntakeAuditor().audit(state["uap_df"], target_col=state.get("uap_target"))
    except Exception as exc:
        state["uap_intake_audit"] = None
        logger.warning("intake_audit_refresh_failed", stage="Undo", error=str(exc))
    request_studio_remount(state)
    record_lineage(state, "Undo", "Reverted last dataset transformation")
    return True


def revert_to_raw_dataset(state: dict[str, Any]) -> bool:
    """Restore the untouched raw dataset originally ingested."""
    raw = state.get("uap_raw_df")
    if raw is None:
        return False
    state["uap_prev_df"] = state["uap_df"].copy() if state.get("uap_df") is not None else None
    state["uap_df"] = raw.copy()
    state["uap_excluded_columns"] = []
    state["uap_exclusions"] = []
    state["uap_deleted_records_log"] = []
    state["uap_s01_cleaned"] = False
    state["uap_s02_last_cluster_deleted"] = None
    state["uap_s02_last_anomaly_deleted"] = None
    state["uap_s02_cluster_committed"] = False
    state["uap_s02_cluster_summary"] = None
    state["uap_s02_anomaly_committed"] = False
    state["uap_s02_anomaly_summary"] = None
    state["uap_s05_engineered"] = False
    state["uap_last_clean_diff"] = None
    get_selected_targets(state, state["uap_df"])
    _reset_downstream_state(state)
    _prune_studio_results_for_stage(state, None)
    run_format_audit(state, state["uap_df"])
    try:
        state["uap_intake_audit"] = InitialIntakeAuditor().audit(state["uap_df"], target_col=state.get("uap_target"))
    except Exception as exc:
        state["uap_intake_audit"] = None
        logger.warning("intake_audit_refresh_failed", stage="Reset", error=str(exc))
    request_studio_remount(state)
    record_lineage(state, "Reset", "Restored original raw dataset")
    return True


def get_dataset_processing_status(state: dict[str, Any]) -> dict[str, Any]:
    """Determine truthfully whether the active dataset is RAW or PROCESSED with exact stage provenance.

    Read-only exploration runs (e.g., running Clustering Analysis or Anomaly Detection
    before clicking Remove or Attach) are NEVER reported as mutating the active dataset.
    """
    df: pd.DataFrame | None = state.get("uap_df")
    raw_df: pd.DataFrame | None = state.get("uap_raw_df")
    lineage: list[dict[str, Any]] = state.get("uap_lineage") or []

    # Only stages that genuinely mutated state['uap_df'] or active study columns
    mutation_stages = {
        "Clean",
        "Cleaning",
        "Data Cleaning",
        "Clustering",
        "Studio 02 Clustering",
        "Clustering Cleaning",
        "Cluster Feature Attachment",
        "Anomaly",
        "Studio 02 Anomaly Screening",
        "Anomaly Cleaning",
        "Anomaly Feature Attachment",
        "Features",
        "Feature Engineering",
        "Feature Selection",
        "Column Exclusion",
        "Embeddings",
        "Version Restore",
    }
    applied: list[str] = []
    for entry in lineage:
        stg = str(entry.get("stage", ""))
        if stg in {"Ingest", "Reset"}:
            applied = []
        elif stg == "Undo" and applied:
            applied.pop()
        elif stg in mutation_stages:
            if stg == "Column Exclusion" and not (state.get("uap_excluded_columns") or []):
                applied = [s for s in applied if s != "Column Exclusion"]
            else:
                applied.append(stg)

    if state.get("uap_s01_cleaned") and not any(s in {"Clean", "Cleaning", "Data Cleaning"} for s in applied):
        applied.insert(0, "Cleaning")
    if state.get("uap_s02_cluster_committed") and not any(
        s in {"Clustering", "Studio 02 Clustering", "Clustering Cleaning", "Cluster Feature Attachment"} for s in applied
    ):
        applied.append("Clustering")
    if state.get("uap_s02_anomaly_committed") and not any(
        s in {"Anomaly", "Studio 02 Anomaly Screening", "Anomaly Cleaning", "Anomaly Feature Attachment"} for s in applied
    ):
        applied.append("Anomaly")

    is_processed = bool(applied)
    if not is_processed and df is not None and raw_df is not None and (df.shape != raw_df.shape or list(df.columns) != list(raw_df.columns)):
        is_processed = True
        applied.append("Transformed")

    excluded_cols = get_excluded_columns(state, df) if df is not None else []
    if excluded_cols and "Column Exclusion" not in applied:
        applied.append("Column Exclusion")
        is_processed = True

    deleted_log = state.get("uap_deleted_records_log") or []
    deleted_rows_total = sum(int(e.get("rows_removed", 0)) for e in deleted_log if isinstance(e, dict))
    study_cols_count = (int(df.shape[1]) - len(excluded_cols)) if df is not None else 0

    is_cleaned_flag = bool(state.get("uap_s01_cleaned")) or any(
        s in {"Clean", "Cleaning", "Data Cleaning"} for s in applied
    )
    is_cluster_flag = (
        bool(state.get("uap_s02_cluster_committed"))
        or any(s in {"Clustering", "Studio 02 Clustering", "Clustering Cleaning", "Cluster Feature Attachment"} for s in applied)
        or (df is not None and "cluster_id" in df.columns)
    )
    is_anomaly_flag = (
        bool(state.get("uap_s02_anomaly_committed"))
        or any(s in {"Anomaly", "Studio 02 Anomaly Screening", "Anomaly Cleaning", "Anomaly Feature Attachment"} for s in applied)
        or (df is not None and "is_anomaly" in df.columns)
    )

    label_parts: list[str] = []
    if is_cleaned_flag:
        label_parts.append("CLEANED")
    if is_cluster_flag:
        label_parts.append("CLUSTER-SCREENED")
    if is_anomaly_flag:
        label_parts.append("ANOMALY-FILTERED")
    if any(s in {"Features", "Feature Engineering"} for s in applied) or state.get("uap_s05_engineered"):
        label_parts.append("ENGINEERED")
    if excluded_cols:
        label_parts.append(f"{len(excluded_cols)} EXCLUDED")
    detailed_label = " + ".join(label_parts) if label_parts else ("PROCESSED" if is_processed else "RAW")

    return {
        "is_processed": is_processed,
        "mode_label": "PROCESSED" if is_processed else "RAW",
        "detailed_label": detailed_label,
        "applied_stages": applied,
        "is_cleaned": is_cleaned_flag,
        "is_cluster_applied": is_cluster_flag,
        "is_anomaly_applied": is_anomaly_flag,
        "has_pending_clustering": state.get("uap_clustering_result") is not None and not bool(state.get("uap_s02_cluster_committed")),
        "has_pending_anomaly": state.get("uap_anomaly_df") is not None and not bool(state.get("uap_s02_anomaly_committed")),
        "excluded_columns": excluded_cols,
        "deleted_rows_total": deleted_rows_total,
        "raw_shape": (int(len(raw_df)), int(raw_df.shape[1])) if raw_df is not None else (0, 0),
        "active_shape": (int(len(df)), int(df.shape[1])) if df is not None else (0, 0),
        "study_shape": (int(len(df)), max(study_cols_count, 0)) if df is not None else (0, 0),
    }


def _infer_split_task(series: pd.Series) -> TaskType:
    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 15:
        return TaskType.REGRESSION
    return TaskType.BINARY_CLASSIFICATION if series.nunique(dropna=True) == 2 else TaskType.MULTICLASS_CLASSIFICATION


def lock_leakage_safe_split(
    state: dict[str, Any],
    df: pd.DataFrame,
    *,
    strategy: str = "auto",
    test_size: float = 0.2,
    cv_folds: int = 5,
    time_col: str | None = None,
    group_col: str | None = None,
    target_col: str | None = None,
    gap: int = 0,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Lock a leakage-free Train/Test holdout split BEFORE cleaning, feature engineering, or modeling."""
    targets = get_selected_targets(state, df)
    resolved_target = target_col if (target_col and target_col in df.columns) else (targets[0] if targets else None)
    cfg = SplitConfig(
        strategy=strategy,  # type: ignore[arg-type]
        test_size=float(test_size),
        cv_folds=int(cv_folds),
        time_col=time_col if (time_col and time_col != "(None)") else None,
        group_col=group_col if (group_col and group_col != "(None)") else None,
        gap=int(gap),
        random_seed=int(random_seed),
    )
    res = LeakageSafeSplitter(cfg).split(df, target_col=resolved_target)
    lock_payload: dict[str, Any] = {
        "active": True,
        "strategy": res.strategy_used,
        "configured_strategy": strategy,
        "test_size": float(test_size),
        "cv_folds": int(cv_folds),
        "time_col": cfg.time_col,
        "group_col": cfg.group_col,
        "target_col": resolved_target,
        "gap": int(gap),
        "random_seed": int(random_seed),
        "train_indices": list(res.train_indices),
        "test_indices": list(res.test_indices),
        "n_train": int(len(res.train_indices)),
        "n_test": int(len(res.test_indices)),
        "audit_report": (res.audit_report.model_dump() if res.audit_report is not None else {}),
        "locked_at": datetime.now().strftime("%H:%M:%S"),
    }
    _invalidate_split_results(state)
    state["uap_split_lock"] = lock_payload
    state["uap_split_audit_report"] = lock_payload["audit_report"]
    request_studio_remount(state)
    record_lineage(
        state,
        "Split Lock",
        f"Locked leakage-safe {res.strategy_used.upper()} split",
        f"Train={len(res.train_indices):,} | Test={len(res.test_indices):,} | CV={cv_folds}-fold | "
        f"Audit={'PASS' if (res.audit_report is not None and res.audit_report.passed) else 'WARN'}",
    )
    return lock_payload


def clear_split_lock(state: dict[str, Any]) -> None:
    """Unlock the active Train/Test split and invalidate fits made under it."""
    lock = state.get("uap_split_lock")
    if isinstance(lock, dict) and lock.get("active"):
        _invalidate_split_results(state)
    state["uap_split_lock"] = None
    state["uap_split_audit_report"] = None
    request_studio_remount(state)
    record_lineage(state, "Split Lock", "Unlocked Train/Test split")


def get_active_split_frames(state: dict[str, Any], df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any] | None]:
    """Return ``(train_df, test_df, lock_dict)`` when a valid Split Lock is active, else ``(df, None, None)``."""
    active_df = df if df is not None else state.get("uap_df")
    if active_df is None:
        return pd.DataFrame(), None, None
    lock = state.get("uap_split_lock")
    if not isinstance(lock, dict) or not lock.get("active"):
        return active_df, None, None
    row_id_col = lock.get("row_id_col")
    has_row_ids = row_id_col is not None or lock.get("train_row_ids") is not None or lock.get("test_row_ids") is not None
    if has_row_ids:
        row_id_sets = _locked_row_id_sets(lock)
        if not isinstance(row_id_col, str) or row_id_col not in active_df.columns or row_id_sets is None:
            raise ValueError("Active Split Lock is invalid or corrupted (ambiguous train/test row_ids).")
        try:
            ids = active_df[row_id_col]
            if not ids.is_unique or ids.isna().any():
                raise ValueError("Active Split Lock is invalid or corrupted (ambiguous train/test row_ids).")
            tr_set, te_set = row_id_sets
            tr_mask = ids.isin(tr_set)
            te_mask = ids.isin(te_set)
        except TypeError as exc:
            raise ValueError("Active Split Lock is invalid or corrupted (ambiguous train/test row_ids).") from exc
        if (int(tr_mask.sum()) != len(tr_set) or int(te_mask.sum()) != len(te_set)
                or len(tr_set) < 5 or len(te_set) < 2):
            raise ValueError("Active Split Lock is invalid or corrupted (missing or insufficient train/test row_ids).")
        return active_df.loc[tr_mask].copy(), active_df.loc[te_mask].copy(), lock
    # The splitter stores zero-based positions, regardless of DataFrame labels.
    tr_idx = _locked_positions(lock.get("train_indices"), len(active_df))
    te_idx = _locked_positions(lock.get("test_indices"), len(active_df))
    if tr_idx is None or te_idx is None or len(tr_idx) < 5 or len(te_idx) < 2 or set(tr_idx) & set(te_idx):
        raise ValueError("Active Split Lock is invalid or corrupted (invalid or overlapping train/test indices).")
    return active_df.iloc[tr_idx].copy(), active_df.iloc[te_idx].copy(), lock


def recombine_split_frames(train_df: pd.DataFrame, test_df: pd.DataFrame, original_index: pd.Index) -> pd.DataFrame:
    """Recombine transformed Train & Test partitions back into their canonical index order."""
    combined = pd.concat([train_df, test_df], axis=0)
    ordered_idx = [i for i in original_index if i in combined.index]
    extra_idx = [i for i in combined.index if i not in set(ordered_idx)]
    return combined.loc[ordered_idx + extra_idx]


def workflow_completion_status(state: dict[str, Any]) -> list[tuple[str, bool]]:
    """Return ordered CRISP-DM workflow checklist completion booleans (persistent across downstream steps)."""
    s01_done = bool(
        state.get("uap_s01_cleaned")
        or state.get("uap_quality") is not None
        or state.get("uap_clean_report") is not None
        or state.get("uap_split_lock") is not None
    )
    s02_done = bool(
        state.get("uap_s02_cluster_committed")
        or state.get("uap_s02_anomaly_committed")
        or state.get("uap_s02_cluster_summary") is not None
        or state.get("uap_s02_anomaly_summary") is not None
        or state.get("uap_clustering_result") is not None
        or state.get("uap_anomaly_df") is not None
        or state.get("uap_mining_result") is not None
    )
    s03_s04_done = bool(state.get("uap_diag_report") is not None or state.get("uap_eda_report") is not None)
    s05_done = bool(state.get("uap_feature_report") is not None or state.get("uap_fs_selected_cols") is not None)
    s06_done = bool(
        state.get("uap_active_model") is not None
        or bool(state.get("uap_target_models"))
        or bool(state.get("uap_dl_models"))
    )
    s07_s12_done = bool(
        state.get("uap_model_card") is not None
        or state.get("uap_shap_result") is not None
        or state.get("uap_report_payload") is not None
    )
    s14_done = bool(state.get("uap_supervisory_result") is not None)
    return [
        ("1. Data Ingested & Audited", state.get("uap_df") is not None),
        ("2. Quality / Cleaned / Split", s01_done),
        ("3. Clustered / Anomaly / Mining", s02_done),
        ("4. Diagnostics / Baselines / EDA", s03_s04_done),
        ("5. Features Engineered", s05_done),
        ("6. Model / Neural Net Trained", s06_done),
        ("7. Responsible AI / Reported", s07_s12_done),
        ("8. Live Supervisory Monitored", s14_done),
    ]


def studio_completion_map(state: dict[str, Any]) -> dict[str, bool]:
    """Return persistent per-studio green completion flags for sidebar status dots (Studios 01..14)."""
    res = state.get("uap_studio_results") or {}
    return {
        "01": state.get("uap_df") is not None,
        "02": bool(
            state.get("uap_s02_cluster_committed")
            or state.get("uap_s02_anomaly_committed")
            or state.get("uap_s02_cluster_summary") is not None
            or state.get("uap_s02_anomaly_summary") is not None
            or state.get("uap_clustering_result") is not None
            or state.get("uap_anomaly_df") is not None
            or state.get("uap_mining_result") is not None
        ),
        "03": state.get("uap_diag_report") is not None,
        "04": bool(state.get("uap_eda_report") is not None or "s04_hypothesis" in res),
        "05": bool(
            state.get("uap_feature_report") is not None
            or state.get("uap_fs_selected_cols") is not None
            or state.get("uap_s05_engineered")
            or "s05_engineer" in res
        ),
        "06": bool(
            state.get("uap_active_model") is not None
            or bool(state.get("uap_dl_models"))
            or bool(state.get("uap_nonparametric_models"))
            or "s06_nonparametric" in res
        ),
        "07": bool(state.get("uap_shap_result") is not None or state.get("uap_fairness_result") is not None),
        "08": bool(
            state.get("uap_forecast_report") is not None
            or state.get("uap_causal_result") is not None
            or state.get("uap_opt_result") is not None
            or state.get("uap_sim_result") is not None
        ),
        "09": any(k.startswith("s09_") for k in res),
        "10": state.get("uap_dag_result") is not None,
        "11": "s11_drift" in res or "s11_registered" in res,
        "12": state.get("uap_report_payload") is not None,
        "13": bool(state.get("uap_chat_history") or state.get("uap_copilot_history")),
        "14": state.get("uap_supervisory_result") is not None,
    }


def export_workspace_project_bytes(
    state: dict[str, Any],
    *,
    project_name: str | None = None,
    notes: str | None = None,
) -> tuple[bytes, Any]:
    """Serialize the complete active workspace into portable ``.uap`` bytes + manifest."""
    from omniforge.core.workspace_session import WorkspaceSessionManager

    resolved_name = (
        project_name
        if project_name is not None
        else str(state.get("uap_project_name") or state.get("uap_dataset_name") or "omniforge_project")
    )
    resolved_notes = notes if notes is not None else str(state.get("uap_project_notes") or "")
    return WorkspaceSessionManager.export_to_bytes(state, project_name=resolved_name, notes=resolved_notes)


def save_workspace_project_local(
    state: dict[str, Any],
    *,
    project_name: str,
    notes: str = "",
    projects_dir: str | Path | None = None,
) -> tuple[Path, Any]:
    """Save the complete active workspace snapshot to ``.omniforge_workspace/projects/<name>.uap``."""
    from omniforge.core.workspace_session import WorkspaceSessionManager

    path, manifest = WorkspaceSessionManager.save_local_project(
        state,
        project_name=project_name,
        notes=notes,
        projects_dir=projects_dir,
    )
    state["uap_project_name"] = manifest.project_name
    state["uap_project_notes"] = manifest.notes
    request_studio_remount(state)
    record_lineage(
        state,
        "Project",
        f"Saved project '{manifest.project_name}'",
        f"File: {path.name} | {len(manifest.saved_keys)} artifacts saved | Stages: {len(manifest.completed_stages)}",
    )
    return path, manifest


def restore_workspace_project_bytes(
    state: dict[str, Any],
    raw_bytes: bytes,
    *,
    restore_studio: bool = True,
    source_label: str = "Uploaded .uap File",
) -> dict[str, Any]:
    """Validate and atomically restore a complete ``.uap`` workspace snapshot into ``state``."""
    from omniforge.core.workspace_session import WorkspaceSessionManager

    # Validate the signed archive before touching the existing workspace, then
    # build a fresh replacement. Merging into the old dict leaves stale cached
    # charts behind when a legacy archive lacks uap_studio_results entirely.
    restored_state, manifest = WorkspaceSessionManager.load_from_bytes(raw_bytes)
    target_state = state
    state = default_workspace_dict()
    if not restore_studio:
        for key in ("uap_selected_studio", "uap_active_studio"):
            if key in target_state:
                state[key] = target_state[key]

    for key, val in restored_state.items():
        if not restore_studio and key in {"uap_selected_studio", "uap_active_studio"}:
            continue
        state[key] = val

    # Older archives stored the latest monitor evaluation but had no studio
    # cache. Migrate that real result, never one absent from an explicit cache.
    if "uap_studio_results" not in restored_state and isinstance(state.get("uap_supervisory_result"), dict):
        save_studio_result(
            state,
            "s14_supervisory",
            {**state["uap_supervisory_result"], "history": state.get("uap_supervisory_history") or []},
        )

    # Normalize legacy studio labels (e.g., "06 · AutoML...") to canonical 2-digit studio code "06"
    raw_studio = str(state.get("uap_selected_studio") or state.get("uap_active_studio") or "01").strip()
    studio_code = raw_studio[:2] if len(raw_studio) >= 2 and raw_studio[:2].isdigit() else "01"
    state["uap_selected_studio"] = studio_code
    state["uap_active_studio"] = studio_code

    restored_theme = str(restored_state.get("uap_theme") or manifest.theme or "dark").lower()
    state["uap_theme"] = "light" if restored_theme == "light" else "dark"

    if isinstance(state.get("uap_df"), pd.DataFrame):
        state.setdefault("uap_excluded_columns", [])
        state.setdefault("uap_deleted_records_log", [])
        get_selected_targets(state, state["uap_df"])
        if not state.get("uap_format_audit"):
            run_format_audit(state, state["uap_df"])

    state["uap_project_name"] = manifest.project_name
    state["uap_project_notes"] = manifest.notes
    record_lineage(
        state,
        "Project",
        f"Loaded project '{manifest.project_name}'",
        f"Source: {source_label} | Restored {len(restored_state)} workspace items | Saved at {manifest.saved_at_utc}",
    )
    state["uap_last_project_notice"] = (
        f"Restored full project '{manifest.project_name}' ({len(restored_state)} saved workspace items, "
        f"dataset: {manifest.dataset_name or 'none'}, {len(manifest.completed_stages)} completed stages)."
    )
    request_studio_remount(state)
    target_state.clear()
    target_state.update(state)
    return manifest.model_dump()


def safe_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Safely coerce a DataFrame column to numeric without raising."""
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def has_truthy_trigger(ctx_obj: Any) -> bool:
    """Return True only when at least one triggered Dash Input has a truthy value.

    Prevents phantom callback executions when Dash dynamically re-mounts
    multi-Input buttons or uploads inside a studio/sidebar container, where
    ``prevent_initial_call=True`` still fires once on dynamic mount with all
    ``n_clicks=None`` / ``contents=None``.
    """
    triggered = getattr(ctx_obj, "triggered", None)
    if not triggered:
        return False
    return any(bool(item.get("value")) for item in triggered)

