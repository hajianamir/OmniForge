"""Studio 06 (Multi-Backend AutoML, Non-Parametric Analytical & Deep Learning) router."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from omniforge.api.auth import require_permission
from omniforge.api.jobs import JobCancellationToken, JobQueueFullError
from omniforge.api.studios.schemas import (
    AutoMLRunReq,
    DeepBenchReq,
    NonParametricRunReq,
    SinglePredictReq,
    wb_cache_and_respond,
)
from omniforge.automl.backends import AutoMLBackendHub, probe_automl_backends
from omniforge.core.utils import to_json_object
from omniforge.data.splitting import LeakageSafeSplitter, SplitConfig
from omniforge.evaluation.equations import ModelEquationExtractor
from omniforge.features.inference import WorkbenchInputPreprocessor
from omniforge.models.baselines import BaselineBenchmarkSuite
from omniforge.models.deep import (
    CNN1DModel,
    FTTransformerModel,
    GraphNeuralNetworkModel,
    MLPModel,
    ResNetTabularModel,
    TabNetModel,
)
from omniforge.models.nonparametric import NonParametricAnalyticalSuite

router = APIRouter(tags=["workbench-s06"])


def _wb() -> Any:
    from omniforge.api import workbench as wb
    return wb


def _prepare_holdout(
    frame: pd.DataFrame | None, target: str, preprocessor: WorkbenchInputPreprocessor
) -> tuple[pd.DataFrame, pd.Series] | None:
    if frame is None:
        return None
    observed = frame.dropna(subset=[target])
    if observed.empty:
        raise HTTPException(status_code=400, detail=f"The locked test partition has no observed values for '{target}'.")
    return preprocessor.transform(observed.drop(columns=[target])), observed[target]


@router.get("/s06/backends")
def s06_list_backends() -> dict[str, Any]:
    return to_json_object({"automl_backends": probe_automl_backends()})


@router.post("/s06/automl", dependencies=[Depends(require_permission("model:train"))])
def s06_run_automl(req: AutoMLRunReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    active_df = _wb()._require_df(state)
    df = _wb().get_analysis_dataframe(state, active_df)
    targets = _wb().get_selected_targets(state, active_df)
    active_targets = [req.target] if (req.target and req.target in df.columns) else targets
    if not active_targets:
        raise HTTPException(status_code=400, detail="Select a target column first.")
    target = active_targets[0]

    # Honor Split Lock: train only on train partition when lock is active
    train_df, test_df, lock = _wb().get_active_split_frames(state, active_df)
    automl_source = train_df.loc[:, df.columns]
    holdout_source = test_df.loc[:, df.columns] if test_df is not None else None
    native_backend = req.backend.strip().lower() == "omniforge_native"
    strategy = lock.get("strategy", "auto") if lock else None

    def split_options(target_name: str) -> dict[str, Any]:
        if not lock or not native_backend:
            return {}
        rows = train_df.loc[train_df[target_name].notna()]
        options: dict[str, Any] = {"split_strategy": strategy}
        if strategy == "group":
            options["groups"] = rows[lock["group_col"]].to_numpy()
        elif strategy == "time_series":
            time_col = lock.get("time_col")
            if time_col:
                options["time_values"] = rows[time_col].to_numpy()
            options["gap"] = lock.get("gap", 0)
        return options

    def sources_for(target_name: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        if native_backend or holdout_source is not None:
            return automl_source, holdout_source
        observed = automl_source.dropna(subset=[target_name])
        if len(observed) < 13:
            return observed, None  # Preserve the minimum 10 development rows needed by the adapter.
        reserved = LeakageSafeSplitter(SplitConfig(test_size=0.2)).split(observed, target_col=target_name)
        return observed.iloc[reserved.train_indices], observed.iloc[reserved.test_indices]

    def _execute(*, cancellation_token: JobCancellationToken | None = None) -> dict[str, Any]:
        multi_target_summary: list[dict[str, Any]] = []
        if len(active_targets) > 1:
            for extra_t in active_targets[1:]:
                if cancellation_token is not None:
                    cancellation_token.checkpoint()
                try:
                    extra_source, extra_test = sources_for(extra_t)
                    Xt, yt, tt, extra_preprocessor = _wb()._prepare_xy_fitted(extra_source, extra_t, raw_features=True)
                    extra_holdout = _prepare_holdout(extra_test, extra_t, extra_preprocessor) if native_backend else None
                    wdf_t = Xt.copy()
                    wdf_t[extra_t] = yt.to_numpy()
                    res_t = AutoMLBackendHub.run(
                        wdf_t,
                        target_col=extra_t,
                        backend=req.backend,
                        task_type=tt,
                        time_budget_seconds=max(5, req.time_budget_seconds // len(active_targets)),
                        cv_folds=req.cv_folds,
                        max_carbon_grams_co2e=req.max_carbon_grams_co2e,
                        max_inference_latency_ms=req.max_inference_latency_ms,
                        holdout=extra_holdout,
                        **split_options(extra_t),
                    )
                    multi_target_summary.append(
                        {
                            "target": extra_t,
                            "task_type": tt,
                            "best_model": getattr(res_t, "best_model_name", None),
                            "best_score": getattr(res_t, "best_score", None),
                        }
                    )
                except Exception:
                    multi_target_summary.append({"target": extra_t, "task_type": "error", "best_model": None, "best_score": None})
        if cancellation_token is not None:
            cancellation_token.checkpoint()
        fit_source, test_source = sources_for(target)
        X, y, task, input_preprocessor = _wb()._prepare_xy_fitted(fit_source, target, raw_features=True)
        holdout = _prepare_holdout(test_source, target, input_preprocessor)
        work_df = X.copy()
        work_df[target] = y.to_numpy()
        res = AutoMLBackendHub.run(
            work_df,
            target_col=target,
            backend=req.backend,
            task_type=task,
            time_budget_seconds=req.time_budget_seconds,
            cv_folds=req.cv_folds,
            max_carbon_grams_co2e=req.max_carbon_grams_co2e,
            max_inference_latency_ms=req.max_inference_latency_ms,
            holdout=holdout if native_backend else None,
            **split_options(target),
        )
        if cancellation_token is not None:
            cancellation_token.checkpoint()
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
            _wb().logger.debug("equation_extract_skipped", error=str(exc))

        regression = task == _wb().TaskType.REGRESSION
        baseline_score = round(float(
            res.best_score + res.lift_over_best_baseline if regression
            else res.best_score - res.lift_over_best_baseline
        ), 4)
        evaluation = res.extra.get("evaluation") or {}
        ranked_lb = []
        for idx_lb, row_lb in enumerate(res.leaderboard or [], start=1):
            r_copy = dict(row_lb)
            name = r_copy.get("model") or r_copy.get("model_name")
            raw_score = r_copy.get(res.primary_metric, r_copy.get("cv_score", r_copy.get("score")))
            r_copy["rank"] = idx_lb
            if name == res.best_model_name:
                r_copy["acceptance_status"] = "ACCEPTED (CHAMPION)" if res.lift_over_best_baseline >= 0 else "ACCEPTED (BEST AVAILABLE)"
            else:
                r_copy["acceptance_status"] = "BENCHMARKED"
            if evaluation and raw_score is not None:
                r_score = float(raw_score)
                validation_baseline = evaluation.get("validation_baseline_score")
                r_copy["selection_criterion"] = f"Validation {res.primary_metric} ({'lower' if regression else 'higher'} is better)"
                if validation_baseline is not None:
                    r_lift = validation_baseline - r_score if regression else r_score - validation_baseline
                    r_copy["lift_over_baseline"] = round(r_lift, 4)
                r_copy["rationale"] = f"Validation {res.primary_metric}={r_score:.4f}."
            ranked_lb.append(r_copy)

        # Once model publication begins, cancellation is no longer accepted.
        if cancellation_token is not None:
            cancellation_token.begin_commit()
        state["uap_active_model"] = res.best_model
        state["uap_active_model_name"] = f"{res.best_model_name} ({res.executed_backend})"
        state["uap_active_model_metrics"] = {
            res.primary_metric: res.best_score,
            "baseline_score": baseline_score,
            "lift_over_baseline": res.lift_over_best_baseline,
        }
        state["uap_active_feature_cols"] = list(X.columns)
        state["uap_active_input_preprocessor"] = input_preprocessor
        evaluation_indices = getattr(res, "evaluation_indices", None)
        if holdout is not None:
            state["uap_active_X_test"], state["uap_active_y_test"] = holdout
        elif evaluation_indices is not None and evaluation:
            state["uap_active_X_test"] = X.iloc[evaluation_indices]
            state["uap_active_y_test"] = y.iloc[evaluation_indices]
        else:
            state["uap_active_X_test"] = None
            state["uap_active_y_test"] = None
        state["uap_automl"] = res
        state["uap_active_model_equation"] = eq_dict
        state["uap_model_equations"] = {target: {res.best_model_name: eq_dict}} if eq_dict else {}
        _wb().record_lineage(
            state,
            "AutoML",
            f"Trained {res.executed_backend} ({res.best_model_name})",
            f"{res.primary_metric}={res.best_score:.4f} (Lift {res.lift_over_best_baseline:+.4f})",
        )
        return wb_cache_and_respond(
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
        owner = _wb().current_job_owner_key() or "default"
        try:
            rec = _wb().job_manager.submit(f"AutoML ({req.backend})", _execute, owner_key=owner, cancellable=True)
        except JobQueueFullError as exc:
            raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "5"}) from exc
        return {"job": rec.to_dict()}
    return _execute()


@router.get("/s06/export-formula")
def s06_export_external_formula(
    model_id: str | None = None,
    x_session_id: str | None = Header(default="default"),
) -> StreamingResponse:
    """Download a self-contained zero-dependency Python deployment file (.py) containing
    the exact standalone prediction function plus Excel, SQL, C/MATLAB, and LaTeX formulas."""
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    eq: dict[str, Any] | None = None

    if model_id:
        catalog = _wb().SupervisoryPredictionMonitor.catalog_available_models(state)
        entry = next((m for m in catalog if m["model_id"] == model_id), None)
        ref_df = state.get("uap_df") if isinstance(state.get("uap_df"), pd.DataFrame) else None
        eq = _wb().SupervisoryPredictionMonitor.extract_model_equation(entry, ref_df=ref_df)
    if not eq:
        eq = state.get("uap_active_model_equation")
    if not eq and state.get("uap_active_model") is not None:
        catalog = _wb().SupervisoryPredictionMonitor.catalog_available_models(state)
        ref_df = state.get("uap_df") if isinstance(state.get("uap_df"), pd.DataFrame) else None
        eq = _wb().SupervisoryPredictionMonitor.extract_model_equation(catalog[0] if catalog else None, ref_df=ref_df)
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


@router.post("/s06/nonparametric", dependencies=[Depends(require_permission("model:train"))])
def s06_run_nonparametric_analytical(
    req: NonParametricRunReq, x_session_id: str | None = Header(default="default")
) -> dict[str, Any]:
    """Benchmark all Non-Parametric Analytical Models (Spline GAM, Nadaraya-Watson, RBF Kernel Ridge, Weighted k-NN, Isotonic) vs. Parametric Linear."""
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    active_df = _wb()._require_df(state)
    df = _wb().get_analysis_dataframe(state, active_df)
    targets = _wb().get_selected_targets(state, df)
    target = req.target if (req.target and req.target in df.columns) else (targets[0] if targets else None)
    if not target:
        raise HTTPException(status_code=400, detail="Select a target column first.")
    train_df, test_df, _ = _wb().get_active_split_frames(state, active_df)
    X, y, task, input_preprocessor = _wb()._prepare_xy_fitted(train_df.loc[:, df.columns], target)
    holdout = _prepare_holdout(test_df.loc[:, df.columns] if test_df is not None else None, target, input_preprocessor)
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
        state["uap_active_input_preprocessor"] = input_preprocessor
        state["uap_active_X_test"], state["uap_active_y_test"] = holdout if holdout is not None else (None, None)

    _wb().record_lineage(
        state,
        "Non-Parametric Modeling",
        f"Evaluated {len(tgt_map)} Non-Parametric Analytical Models vs. Parametric Linear",
        f"Champion={champ_name} ({res.get('champion_score', 0.0):.4f})",
    )
    return wb_cache_and_respond(state, sid, "s06_nonparametric", dict(res))


@router.post("/s06/deep-learning", dependencies=[Depends(require_permission("model:train"))])
def s06_run_deep_benchmark(req: DeepBenchReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    """Benchmark Paper-Grade Deep Learning Architectures (FT-Transformer, TabNet, ResNet-Tabular, MLP, 1D-CNN, GNN) vs. Baselines."""
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    df = _wb().get_analysis_dataframe(state, _wb()._require_df(state))
    targets = _wb().get_selected_targets(state, df)
    target = req.target if (req.target and req.target in df.columns) else (targets[0] if targets else None)
    if not target:
        raise HTTPException(status_code=400, detail="Select a target column first.")
    X_full, y_full, task = _wb()._prepare_xy(df, target)
    if len(X_full) < 6:
        raise HTTPException(status_code=400, detail="Deep learning benchmark requires at least 6 rows.")
    if len(X_full) > req.max_rows:
        idx = np.random.default_rng(42).choice(len(X_full), size=req.max_rows, replace=False)
        X_full = X_full.iloc[idx].reset_index(drop=True)
        y_full = y_full.iloc[idx].reset_index(drop=True)
    split = min(len(X_full) - 2, max(2, int(len(X_full) * 0.8)))
    X_tr, X_te = X_full.iloc[:split], X_full.iloc[split:]
    y_tr, y_te = y_full.iloc[:split], y_full.iloc[split:]
    is_clf = task in (_wb().TaskType.BINARY_CLASSIFICATION, _wb().TaskType.MULTICLASS_CLASSIFICATION)

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
        t0 = _wb().time.perf_counter()
        try:
            m = builder()
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)
            sc = float(f1_score(y_te, preds, average="weighted", zero_division=0)) if is_clf else float(r2_score(y_te, preds))
            elapsed = round(_wb().time.perf_counter() - t0, 3)
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
            rows.append({"model": arch, "family": "Deep Learning", "framework": "error", "score": 0.0, "elapsed_sec": round(_wb().time.perf_counter() - t0, 3), "error": str(exc), "loss_curve": []})

    rows.sort(key=lambda r: r["score"], reverse=True)
    state["uap_dl_benchmark_df"] = pd.DataFrame([{k: v for k, v in r.items() if k != "loss_curve"} for r in rows])
    state["uap_dl_models"] = fitted_dl
    _wb().record_lineage(state, "Deep Learning", f"Benchmarked {len(fitted_dl)} paper-grade architectures", f"Top={rows[0]['model']} ({rows[0]['score']:.4f})")
    return wb_cache_and_respond(
        state,
        sid,
        "s06_dl",
        {
            "metric": "f1_weighted" if is_clf else "r2",
            "results": rows,
            "feature_importances": feature_importances,
        },
    )


def _prediction_frame(state: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """Use the exact raw-input preparation retained by the active training run."""
    preprocessor = state.get("uap_active_input_preprocessor")
    if isinstance(preprocessor, WorkbenchInputPreprocessor):
        try:
            return preprocessor.transform(frame)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    # Older saved workspaces may only contain already-encoded numeric feature names.
    # Preserve their historic scoring behavior until they are retrained.
    feat_cols = state.get("uap_active_feature_cols") or []
    aligned = frame.reindex(columns=feat_cols)
    return aligned.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


@router.post("/s06/predict-single")
def s06_predict_single(req: SinglePredictReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    model = state.get("uap_active_model")
    feat_cols = state.get("uap_active_feature_cols") or []
    if model is None or not feat_cols:
        raise HTTPException(status_code=400, detail="Train a model in Studio 06 first before scoring.")
    if any(isinstance(value, (list, dict)) for value in req.record.values()):
        raise HTTPException(status_code=422, detail="Prediction values must be scalar.")
    row_df = _prediction_frame(state, pd.DataFrame([req.record]))
    try:
        pred = model.predict(row_df)[0]
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Unable to score the provided features: {exc}") from exc
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(row_df)[0].tolist()
        except Exception:
            proba = None
    return wb_cache_and_respond(
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
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    model = state.get("uap_active_model")
    feat_cols = state.get("uap_active_feature_cols") or []
    if model is None or not feat_cols:
        raise HTTPException(status_code=400, detail="Train a model in Studio 06 before batch scoring.")
    raw = await _wb().read_upload_bounded(file)
    try:
        # Preserve raw category strings (e.g. "NA" or "001") exactly as single-record
        # JSON does; the fitted schema parses numeric and datetime columns afterward.
        inc_df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}") from exc
    if inc_df.empty:
        raise HTTPException(status_code=400, detail="The CSV must contain at least one record.")
    ref_df = state.get("uap_df")
    target = (_wb().get_selected_targets(state, ref_df) or ["target"])[0]
    aligned = _prediction_frame(state, inc_df)
    try:
        preds = model.predict(aligned)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Unable to score the provided features: {exc}") from exc
    scored = inc_df.copy()
    scored[f"predicted_{target}"] = preds
    state["uap_future_prediction_result"] = {"n_rows": len(scored), "preview": scored.head(100).to_dict(orient="records"), "scored_df": scored}
    return wb_cache_and_respond(
        state,
        sid,
        "s06_predict_batch",
        {"n_rows": len(scored), "columns": list(scored.columns), "preview": scored.head(100).to_dict(orient="records")},
        include_state=False,
    )


# ---------------------------------------------------------------------------
# Studio 07 — Responsible AI (Explainability, Fairness & Governance)
# ---------------------------------------------------------------------------
