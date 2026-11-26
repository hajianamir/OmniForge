"""Studio 10 (Visual DAG Pipeline Builder) router."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Header

from omniforge.api.auth import authorize_principal, require_permission
from omniforge.api.studios.schemas import DAGRunReq, wb_cache_and_respond
from omniforge.automl.backends import AutoMLBackendHub
from omniforge.core.performance import compute_df_fingerprint
from omniforge.core.types import Principal
from omniforge.data.cleaning import CleaningConfig
from omniforge.features.engineering import FeatureConfig, FeatureEngineeringPipeline
from omniforge.pipelines.builder import PipelineBuilder
from omniforge.pipelines.executor import PipelineExecutor

router = APIRouter(tags=["workbench-s10"])


def _wb() -> Any:
    from omniforge.api import workbench as wb

    return wb


def _initial_split_frames(state: dict[str, Any], active_df: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract active Split-Lock train/test frames or deterministic 80/20 holdout frames for leak-free DAG execution."""
    train_df, test_df, _ = _wb().get_active_split_frames(state, active_df)
    if test_df is not None:
        return train_df.loc[:, df.columns], test_df.loc[:, df.columns]
    n = len(df)
    if n < 5:
        return df.copy(), pd.DataFrame(columns=df.columns)
    cut = max(2, int(round(0.8 * n)))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


@router.post("/s10/run")
def s10_run_dag(
    req: DAGRunReq,
    x_session_id: str | None = Header(default="default"),
    principal: Principal = Depends(require_permission("pipeline:execute")),
) -> dict[str, Any]:
    if "automl" in req.stages:
        authorize_principal(principal, "model:train")
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    active_df = _wb()._require_df(state)
    df = _wb().get_analysis_dataframe(state, active_df)
    targets = _wb().get_selected_targets(state, df)
    target = req.target if (req.target and req.target in df.columns) else (targets[0] if targets else None)
    if target is not None and ("features" in req.stages or "automl" in req.stages):
        df = df.drop(columns=[col for col in targets if col != target])

    init_train, init_test = _initial_split_frames(state, active_df, df)

    builder = PipelineBuilder(name="OmniForge_DAG_Workflow")
    prev: str | None = None
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    for idx, stage in enumerate(req.stages):
        node_id = f"step_{idx + 1}_{stage}"
        if stage == "quality":

            def _step_quality(ctx: Any) -> dict[str, Any]:
                cur_df: pd.DataFrame = ctx.get("data")
                prof = _wb().DataQualityProfiler().profile(cur_df)
                return {
                    "quality_score": prof.overall_score,
                    "grade": prof.grade,
                    "input_fingerprint": compute_df_fingerprint(cur_df),
                }

            builder.add_step(node_id, _step_quality, depends_on=[prev] if prev else [])
        elif stage == "clean":

            def _step_clean(ctx: Any) -> dict[str, Any]:
                tr_in: pd.DataFrame = ctx.get("train_df")
                te_in: pd.DataFrame = ctx.get("test_df")
                cleaner = _wb().DataCleaner(CleaningConfig(outlier_action="flag"))
                if len(te_in) > 0:
                    clean_tr, clean_te, _, _ = cleaner.fit_transform_split(tr_in, te_in)
                    combined = pd.concat([clean_tr, clean_te], ignore_index=True)
                    ctx.set("train_df", clean_tr)
                    ctx.set("test_df", clean_te)
                else:
                    combined, _ = cleaner.fit_transform(tr_in)
                    ctx.set("train_df", combined)
                ctx.set("data", combined)
                return {
                    "rows_after": int(len(combined)),
                    "train_rows": int(len(ctx.get("train_df"))),
                    "test_rows": int(len(ctx.get("test_df"))),
                    "leakage_safe_split_fit": True,
                    "output_fingerprint": compute_df_fingerprint(combined),
                }

            builder.add_step(node_id, _step_clean, depends_on=[prev] if prev else [])
        elif stage == "eda":

            def _step_eda(ctx: Any) -> dict[str, Any]:
                cur_df: pd.DataFrame = ctx.get("data")
                num_df = cur_df.select_dtypes(include=[np.number])
                missing_cells = int(cur_df.isna().sum().sum())
                return {
                    "n_rows": int(len(cur_df)),
                    "n_cols": int(cur_df.shape[1]),
                    "numeric_cols": int(num_df.shape[1]),
                    "missing_cells": missing_cells,
                    "data_fingerprint": compute_df_fingerprint(cur_df),
                }

            builder.add_step(node_id, _step_eda, depends_on=[prev] if prev else [])
        elif stage == "features":

            def _step_features(ctx: Any, tgt: str | None = target) -> dict[str, Any]:
                tr_df: pd.DataFrame = ctx.get("train_df")
                te_df: pd.DataFrame = ctx.get("test_df")
                has_tgt = bool(tgt and tgt in tr_df.columns)
                feat_cols = [c for c in tr_df.columns if c != tgt] if has_tgt else list(tr_df.columns)
                if not feat_cols:
                    cur_df = ctx.get("data")
                    return {"numeric_cols": int(len(cur_df.select_dtypes(include=[np.number]).columns))}
                fe = FeatureEngineeringPipeline(FeatureConfig(scale_numeric="standard", create_interactions=len(feat_cols) >= 2))
                y_tr = tr_df[tgt] if (has_tgt and tgt is not None) else None
                X_tr_eng = fe.fit_transform(tr_df[feat_cols], y_tr)
                if has_tgt and tgt is not None:
                    X_tr_eng[tgt] = tr_df[tgt].to_numpy()
                if len(te_df) > 0:
                    X_te_eng = fe.transform(te_df[feat_cols])
                    if has_tgt and tgt is not None and tgt in te_df.columns:
                        X_te_eng[tgt] = te_df[tgt].to_numpy()
                    combined = pd.concat([X_tr_eng, X_te_eng], ignore_index=True)
                    ctx.set("train_df", X_tr_eng)
                    ctx.set("test_df", X_te_eng)
                else:
                    combined = X_tr_eng
                    ctx.set("train_df", X_tr_eng)
                ctx.set("data", combined)
                return {
                    "numeric_cols": int(len(combined.select_dtypes(include=[np.number]).columns)),
                    "engineered_feature_count": int(combined.shape[1] - (1 if has_tgt else 0)),
                    "leakage_safe_fit_on_train": True,
                    "output_fingerprint": compute_df_fingerprint(combined),
                }

            builder.add_step(node_id, _step_features, depends_on=[prev] if prev else [])
        elif stage == "automl" and target:

            def _run_fast(ctx: Any, tgt: str = target) -> dict[str, Any]:
                train_df: pd.DataFrame = ctx.get("train_df")
                if tgt not in train_df.columns:
                    return {"stage": "automl", "status": "skipped_missing_target"}
                X_d, y_d, t_d = _wb()._prepare_xy(train_df, tgt, raw_features=True)
                w_df = X_d.copy()
                w_df[tgt] = y_d.to_numpy()
                # Keep the reserved test rows out of model selection.
                r = AutoMLBackendHub.run(
                    w_df,
                    target_col=tgt,
                    backend="multi_layer_stacker",
                    task_type=t_d,
                    time_budget_seconds=15.0,
                )
                return {
                    "best_model": r.best_model_name,
                    "score": r.best_score,
                    "score_partition": "validation",
                    "lift": r.lift_over_best_baseline,
                    "reserved_test_rows": len(ctx.get("test_df")),
                    "upstream_data_fingerprint": compute_df_fingerprint(w_df),
                }

            builder.add_step(node_id, _run_fast, depends_on=[prev] if prev else [])
        else:
            builder.add_step(node_id, lambda ctx, s=stage: {"stage": s, "status": "ok"}, depends_on=[prev] if prev else [])
        nodes.append({"id": node_id, "label": stage.upper(), "order": idx})
        if prev:
            edges.append({"source": prev, "target": node_id})
        prev = node_id

    dag = builder.build()
    exec_res = PipelineExecutor().execute(
        dag,
        initial_artifacts={"data": df.copy(), "train_df": init_train, "test_df": init_test},
    )
    state["uap_dag_result"] = exec_res
    dump: dict[str, Any] = {
        **exec_res.to_dict(),
        "step_outputs": dict(exec_res.context.step_outputs),
    }
    _wb().record_lineage(
        state, "Pipeline DAG", f"Executed {len(req.stages)}-stage DAG", f"Status={dump.get('status', 'completed')}"
    )
    return wb_cache_and_respond(state, sid, "s10_dag", {"execution": dump, "nodes": nodes, "edges": edges})
