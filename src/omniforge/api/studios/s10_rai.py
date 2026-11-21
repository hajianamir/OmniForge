# ruff: noqa: E402
# mypy: ignore-errors
from __future__ import annotations

import dataclasses

"""s10_rai router."""


from typing import Any

import numpy as np
from fastapi import APIRouter, Header

from omniforge.automl.backends import AutoMLBackendHub
from omniforge.pipelines.builder import PipelineBuilder
from omniforge.pipelines.executor import PipelineExecutor

router = APIRouter(tags=["workbench-s10"])


def _wb() -> Any:
    from omniforge.api import workbench as wb
    return wb


@router.post("/s10/run")
def s10_run_dag(req: _wb().DAGRunReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    df = _wb().get_analysis_dataframe(state, _wb()._require_df(state))
    targets = _wb().get_selected_targets(state, df)
    target = req.target if (req.target and req.target in df.columns) else (targets[0] if targets else None)

    builder = PipelineBuilder(name="OmniForge_DAG_Workflow")
    prev: str | None = None
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    for idx, stage in enumerate(req.stages):
        node_id = f"step_{idx + 1}_{stage}"
        if stage == "quality":
            builder.add_step(node_id, lambda ctx: {"quality_score": _wb().DataQualityProfiler().profile(ctx.get("data")).overall_score}, depends_on=[prev] if prev else [])
        elif stage == "clean":
            builder.add_step(node_id, lambda ctx: {"rows_after": len(_wb().DataCleaner().fit_transform(ctx.get("data"))[0])}, depends_on=[prev] if prev else [])
        elif stage == "eda":
            builder.add_step(node_id, lambda ctx: {"n_rows": len(ctx.get("data")), "n_cols": ctx.get("data").shape[1]}, depends_on=[prev] if prev else [])
        elif stage == "features":
            builder.add_step(node_id, lambda ctx: {"numeric_cols": len(ctx.get("data").select_dtypes(include=[np.number]).columns)}, depends_on=[prev] if prev else [])
        elif stage == "automl" and target:
            def _run_fast(ctx: Any, tgt: str = target) -> dict[str, Any]:
                X_d, y_d, t_d = _wb()._prepare_xy(ctx.get("data"), tgt)
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
    _wb().record_lineage(state, "Pipeline DAG", f"Executed {len(req.stages)}-stage DAG", f"Status={dump.get('status', 'completed')}")
    return _wb()._cache_and_respond(state, sid, "s10_dag", {"execution": dump, "nodes": nodes, "edges": edges})


# ---------------------------------------------------------------------------
# Studio 11 — MLOps (Registry, Tracking & Drift Monitoring)
# ---------------------------------------------------------------------------
