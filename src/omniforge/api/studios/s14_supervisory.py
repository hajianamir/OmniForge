"""Studio 14 (Supervisory Prediction Monitor) router."""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Response, UploadFile

from omniforge.api.auth import require_permission
from omniforge.api.studios.root_cause import analyze_study_cohorts
from omniforge.api.studios.schemas import (
    RootCauseReq,
    SupervisoryAccessReq,
    SupervisoryDemoReq,
    SupervisoryQuickTrainReq,
    SupervisorySingleReq,
    SupervisorySQLReq,
    wb_cache_and_respond,
)
from omniforge.core.types import TaskType
from omniforge.core.utils import sanitize_spreadsheet_frame, to_json_object
from omniforge.features.inference import WorkbenchInputPreprocessor
from omniforge.mlops.supervisory_monitor import SupervisoryPredictionMonitor
from omniforge.models.deep import FTTransformerModel, MLPModel, ResNetTabularModel, TabNetModel
from omniforge.models.nonparametric import NadarayaWatsonKernelModel, PenalizedSplineGAMModel
from omniforge.ui.state import (
    clear_supervisory_evaluation,
    get_analysis_dataframe,
    get_selected_targets,
    get_session_state,
    record_active_holdout,
    record_lineage,
)

router: APIRouter = APIRouter(tags=["workbench-s14"])


def _wb() -> Any:
    from omniforge.api import workbench as wb
    return wb


@router.get("/s14/models")
def s14_list_models(x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _wb()._sid(x_session_id)
    state = get_session_state(sid)
    raw_models = SupervisoryPredictionMonitor.catalog_available_models(state)
    models = [{k: v for k, v in m.items() if not k.startswith("_")} for m in raw_models]
    last_eval = state.get("uap_supervisory_result")
    return to_json_object(
        {
            "models": models,
            "last_evaluation": last_eval,
            "history": state.get("uap_supervisory_history") or [],
            "state": _wb()._state_summary(state, sid),
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
    sid = _wb()._sid(x_session_id)
    state = get_session_state(sid)
    catalog = SupervisoryPredictionMonitor.catalog_available_models(state)
    if not catalog:
        return to_json_object({"champion_equation": None, "baseline_equation": None})
    primary_entry, baseline_entry = SupervisoryPredictionMonitor.model_pair(catalog, model_id, baseline_model_id)
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


@router.post("/s14/quick-train", dependencies=[Depends(require_permission("model:train"))])
def s14_quick_train(req: SupervisoryQuickTrainReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    """Quickly train a selected baseline, ensemble, or neural network directly from Studio 14 so any model family is immediately available."""
    sid = _wb()._sid(x_session_id)
    state = get_session_state(sid)
    active_df = _wb()._require_df(state)
    df = get_analysis_dataframe(state, active_df)
    targets = get_selected_targets(state, active_df)
    scope_lock = state.get("uap_split_lock")
    scope_targets = tuple(targets)
    scope_exclusions = tuple(_wb().get_excluded_columns(state, active_df))
    target = req.target if req.target is not None else (targets[0] if len(targets) == 1 else None)
    if target is None or target not in df.columns:
        raise HTTPException(status_code=400, detail="Select one target from the active study before training.")
    train_df, test_df, lock = _wb().get_active_split_frames(state, active_df)
    train_df = train_df.loc[:, df.columns].dropna(subset=[target])
    excluded = {*targets, target}
    if lock:
        excluded.update(col for key in ("row_id_col", "group_col", "time_col") if isinstance(col := lock.get(key), str))
    if len(train_df) < 4 or (test_df is None and len(train_df) < 6):
        raise HTTPException(status_code=400, detail="Supervisory quick-train requires four training rows and two test rows.")
    task = _wb()._infer_task(train_df[target])
    is_clf = task in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)
    split_source = "split_lock"
    if test_df is None:
        from sklearn.model_selection import train_test_split

        test_size = max(2, len(train_df) - int(len(train_df) * 0.8))
        if is_clf:
            test_size = max(test_size, int(train_df[target].nunique()))
        try:
            train_df, test_df = train_test_split(
                train_df, test_size=test_size, random_state=42, stratify=train_df[target] if is_clf else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="The observed classes cannot support a stratified holdout.") from exc
        split_source = "stratified_split" if is_clf else "random_split"
    else:
        test_df = test_df.loc[:, df.columns].dropna(subset=[target])
    if len(train_df) < 4 or len(test_df) < 2:
        raise HTTPException(status_code=400, detail="Supervisory quick-train requires four training rows and two test rows.")
    y_tr, y_te = train_df[target], test_df[target]
    if is_clf and (y_tr.nunique() < 2 or not y_te.isin(y_tr.unique()).all()):
        raise HTTPException(status_code=400, detail="The training partition must contain all classes evaluated on the test partition.")
    predictors = _wb()._study_predictor_columns(train_df, excluded, None)
    if not predictors:
        raise HTTPException(status_code=400, detail="No observed predictor columns remain in the active study.")
    try:
        raw_train = train_df.loc[:, predictors]
        input_preprocessor = WorkbenchInputPreprocessor(mode="numeric").fit(raw_train)
        X_tr = input_preprocessor.transform(raw_train)
        X_te = input_preprocessor.transform(test_df.loc[:, predictors])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="The selected predictor columns could not be prepared.") from exc

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

    if (
        state.get("uap_df") is not active_df or state.get("uap_split_lock") is not scope_lock
        or tuple(get_selected_targets(state, active_df)) != scope_targets
        or tuple(_wb().get_excluded_columns(state, active_df)) != scope_exclusions
    ):
        raise HTTPException(status_code=409, detail="The workspace changed during training. Start a new quick-train run.")
    clear_supervisory_evaluation(state)
    if family == "Deep Learning":
        _dl_raw = state.get("uap_dl_models")
        dl_dict: dict[str, Any] = dict(_dl_raw) if isinstance(_dl_raw, dict) else {}
        dl_dict[m_key] = {
            "model": model,
            "name": label,
            "feature_cols": list(X_tr.columns),
            "metrics": {metric_name: round(score, 4)},
            "target": target,
            "input_preprocessor": input_preprocessor,
            "evaluation_holdout": (X_te, y_te),
        }
        state["uap_dl_models"] = dl_dict
    elif family == "Baseline":
        _bl_raw = state.get("uap_baseline_models")
        bl_all: dict[str, Any] = dict(_bl_raw) if isinstance(_bl_raw, dict) else {}
        _t_bl_raw = bl_all.get(target)
        t_bl: dict[str, Any] = _t_bl_raw if isinstance(_t_bl_raw, dict) else {}
        t_bl[label] = {
            "model": model,
            "feature_cols": list(X_tr.columns),
            "metrics": {metric_name: round(score, 4)},
            "input_preprocessor": input_preprocessor,
            "evaluation_holdout": (X_te, y_te),
        }
        bl_all[target] = t_bl
        state["uap_baseline_models"] = bl_all
    else:
        state["uap_active_model"] = model
        state["uap_active_model_target"] = target
        state["uap_active_model_name"] = label
        state["uap_active_model_metrics"] = {metric_name: round(score, 4)}
        state["uap_active_feature_cols"] = list(X_tr.columns)
        state["uap_active_input_preprocessor"] = input_preprocessor
        state["uap_active_X_test"], state["uap_active_y_test"] = X_te, y_te
        record_active_holdout(state)

    record_lineage(state, "Supervisory Monitor", f"Quick-trained {label}", f"{metric_name}={score:.4f}")
    raw_models = SupervisoryPredictionMonitor.catalog_available_models(state)
    models_clean = [{k: v for k, v in m.items() if not k.startswith("_")} for m in raw_models]
    return to_json_object(
        {
            "trained_model": label,
            "family": family,
            "metric": metric_name,
            "score": round(score, 4),
            "evaluation": {"partition": "test", "source": split_source,
                           "n_train": len(X_tr), "n_test": len(X_te)},
            "models": models_clean,
            "state": _wb()._state_summary(state, sid),
        }
    )


def _run_supervisory_eval(
    state: dict[str, Any],
    sid: str,
    incoming_df: pd.DataFrame,
    source_label: str,
    *,
    model_id: str | None,
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
    eval_res = SupervisoryPredictionMonitor().evaluate_incoming(
        state=state,
        incoming_df=incoming_df,
        model_id=model_id,
        baseline_model_id=baseline_model_id,
        actual_col=actual_col,
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
    return wb_cache_and_respond(
        state,
        sid,
        "s14_supervisory",
        {
            **eval_res,
            "history": state.get("uap_supervisory_history") or [],
        },
    )


@router.post("/s14/evaluate-file", dependencies=[Depends(require_permission("data:write"))])
async def s14_evaluate_file(
    file: UploadFile = File(...),
    sheet_name: str | None = Form(default=None),
    model_id: str | None = Form(default=None),
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
    sid = _wb()._sid(x_session_id)
    state = get_session_state(sid)
    fname = file.filename or "live_batch.xlsx"
    raw = await _wb().read_upload_bounded(file)
    incoming = _wb().parse_tabular_upload(raw, fname, sheet_name=sheet_name)
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


@router.post("/s14/evaluate-sql", dependencies=[Depends(require_permission("data:write"))])
def s14_evaluate_sql(req: SupervisorySQLReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _wb()._sid(x_session_id)
    state = get_session_state(sid)
    incoming = _wb().sql_load_dataframe(req.connection_url, table_name=req.table_name, query=req.query, limit=req.limit)
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


@router.post("/s14/evaluate-access", dependencies=[Depends(require_permission("data:write"))])
def s14_evaluate_access(req: SupervisoryAccessReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _wb()._sid(x_session_id)
    state = get_session_state(sid)
    incoming = _wb().access_load_table(req.file_path, req.table_name, limit=req.limit)
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


@router.post("/s14/evaluate-demo-batch", dependencies=[Depends(require_permission("mlops:write"))])
def s14_evaluate_demo_batch(req: SupervisoryDemoReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    """Generate a realistic live operational batch from the active dataset with optional injected process faults so engineers can immediately test tolerance alarms."""
    sid = _wb()._sid(x_session_id)
    state = get_session_state(sid)
    df = get_analysis_dataframe(state, _wb()._require_df(state))
    catalog = SupervisoryPredictionMonitor.catalog_available_models(state)
    model = next((entry for entry in catalog if req.model_id is None or entry["model_id"] == req.model_id), None)
    if model is None:
        raise HTTPException(status_code=400, detail="Select an available model before running a simulation.")
    target_col = req.actual_col or model["target"]
    selected_targets = set(get_selected_targets(state, df))
    if target_col not in df.columns or (target_col != model["target"] and target_col in selected_targets):
        raise HTTPException(status_code=400, detail="The simulation outcome does not match the selected model.")

    rng = np.random.default_rng(2026)
    n_take = min(req.n_rows, len(df))
    batch = df.sample(n=n_take, replace=(n_take > len(df)), random_state=2026).reset_index(drop=True).copy()

    # Add light operational measurement noise + explicit process fault excursions on inject_fault_pct% of rows
    num_cols = list(batch.select_dtypes(include=[np.number]).columns)
    for col in num_cols:
        if col == target_col or col in selected_targets:
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


@router.post("/s14/evaluate-single", dependencies=[Depends(require_permission("mlops:write"))])
def s14_evaluate_single(req: SupervisorySingleReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _wb()._sid(x_session_id)
    state = get_session_state(sid)
    rec = dict(req.record)
    actual_col = req.actual_col
    if req.actual_value is not None:
        actual_col = actual_col or "__actual__"
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


@router.post("/s14/root-cause", dependencies=[Depends(require_permission("mlops:read"))])
def s14_root_cause(
    req: RootCauseReq, x_session_id: str | None = Header(default="default")
) -> dict[str, Any]:
    """C5 causal drift root-cause attribution in the Supervisory studio (live telemetry vs reference)."""
    sid = _wb()._sid(x_session_id)
    state = get_session_state(sid)
    return wb_cache_and_respond(state, sid, "s14_root_cause", analyze_study_cohorts(state, req))


@router.get("/s14/export/{fmt}", dependencies=[Depends(require_permission("report:generate"))])
def s14_export_results(fmt: str, x_session_id: str | None = Header(default="default")) -> Response:
    state = get_session_state(_wb()._sid(x_session_id))
    res_df = state.get("uap_supervisory_scored_df")
    if not isinstance(res_df, pd.DataFrame) or res_df.empty:
        raise HTTPException(status_code=400, detail="Run a supervisory evaluation first before exporting.")
    res_df = sanitize_spreadsheet_frame(res_df)
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


