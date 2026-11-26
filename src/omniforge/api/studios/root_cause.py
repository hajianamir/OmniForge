"""Resolve study-scoped reference and current cohorts for drift diagnostics."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import HTTPException

from omniforge.api.studios.schemas import RootCauseReq
from omniforge.features.inference import WorkbenchInputPreprocessor
from omniforge.mlops.root_cause import DriftRootCauseAnalyzer
from omniforge.mlops.supervisory_monitor import SupervisoryPredictionMonitor
from omniforge.ui.state import get_active_split_frames, get_analysis_dataframe, get_selected_targets


class _FittedScorer:
    """Score raw cohort features using the recorded model representation."""

    def __init__(self, entry: dict[str, Any]) -> None:
        self._entry = entry
        self.task_type = SupervisoryPredictionMonitor._model_task(entry["_estimator"])

    def predict(self, frame: pd.DataFrame) -> Any:
        prepared = SupervisoryPredictionMonitor._model_input(frame, self._entry, None, strict=True)
        return self._entry["_estimator"].predict(prepared)

    def predict_proba(self, frame: pd.DataFrame) -> Any:
        prepared = SupervisoryPredictionMonitor._model_input(frame, self._entry, None, strict=True)
        return self._entry["_estimator"].predict_proba(prepared)


def analyze_study_cohorts(state: dict[str, Any], req: RootCauseReq) -> dict[str, Any]:
    """Compare named local cohorts or the study reference and latest operational input."""
    active = state.get("uap_df")
    if not isinstance(active, pd.DataFrame) or active.empty:
        raise HTTPException(status_code=400, detail="Load a study dataset before comparing cohorts.")
    analysis = get_analysis_dataframe(state, active)
    copilot = state.get("uap_copilot")
    memory = getattr(copilot, "memory", None)
    named_datasets = getattr(memory, "working_state", {}).get("datasets", {})

    if req.reference_dataset in {"reference_dataset", "active_dataset"}:
        try:
            train, _, lock = get_active_split_frames(state, active)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="The active Split Lock cannot be verified.") from exc
        reference = train.loc[:, analysis.columns]
        reference_label = "active_training_partition" if lock else "active_study"
    else:
        reference = named_datasets.get(req.reference_dataset)
        reference_label = req.reference_dataset

    if req.current_dataset in {"current_dataset", "supervisory_telemetry"}:
        current = state.get("uap_supervisory_input_df")
        current_label = "latest_supervisory_input"
    else:
        current = named_datasets.get(req.current_dataset)
        current_label = req.current_dataset

    if not isinstance(reference, pd.DataFrame) or reference.empty:
        raise HTTPException(status_code=400, detail="The selected reference cohort is unavailable.")
    if not isinstance(current, pd.DataFrame) or current.empty:
        raise HTTPException(status_code=400, detail="The selected current cohort is unavailable. Evaluate an operational batch or select a named cohort.")
    if reference.equals(current) or min(len(reference), len(current)) < 5:
        raise HTTPException(status_code=400, detail="Select two distinct cohorts with at least five records each.")
    if req.time_col and (req.time_col not in reference.columns or req.time_col not in current.columns):
        raise HTTPException(status_code=400, detail="The time column must be present in both cohorts.")

    latest = state.get("uap_supervisory_result") or {}
    model_id = latest.get("kpis", {}).get("model_id") if current_label == "latest_supervisory_input" else "active_champion"
    entry = next((item for item in SupervisoryPredictionMonitor.catalog_available_models(state) if item["model_id"] == model_id), None)
    target = req.target_col or (entry["target"] if entry else None)
    excluded = {*get_selected_targets(state, active), target, req.time_col}
    excluded.update(set(active.columns) - set(analysis.columns))
    active_lock = state.get("uap_split_lock") or {}
    excluded.update(active_lock.get(key) for key in ("row_id_col", "time_col", "group_col"))
    columns = [col for col in reference.columns if col in current.columns and col not in excluded]
    if not columns:
        raise HTTPException(status_code=400, detail="The selected cohorts have no shared eligible predictor columns.")

    scorer = None
    if entry and entry["target"] == target:
        preprocessor = entry.get("_input_preprocessor")
        inputs = preprocessor.input_columns if isinstance(preprocessor, WorkbenchInputPreprocessor) else entry["feature_cols"]
        if inputs and set(inputs).issubset(columns):
            scorer = _FittedScorer(entry)

    analyzer = DriftRootCauseAnalyzer(
        psi_warning=req.psi_warning,
        psi_critical=req.psi_critical,
        max_cond_set_size=req.max_cond_set_size,
    )
    report = analyzer.analyze(
        reference, current, model=scorer, feature_cols=columns, target_col=target, time_col=req.time_col,
    )
    return {
        "root_cause_report": report.model_dump(),
        "cohorts": {
            "reference": reference_label,
            "current": current_label,
            "n_reference": len(reference),
            "n_current": len(current),
            "model_id": model_id if scorer else None,
            "method": "observational_drift_diagnostics",
        },
    }
