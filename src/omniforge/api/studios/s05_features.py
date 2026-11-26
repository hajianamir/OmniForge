"""Studio 05 (Feature Engineering & Selection) router."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Header, HTTPException

from omniforge.api.auth import authorize_principal, require_permission
from omniforge.api.studios.schemas import FeatureEngReq, FeatureSelReq, wb_cache_and_respond
from omniforge.core.types import Principal
from omniforge.features import FeatureConfig, FeatureEngineeringPipeline, FeatureSelector

router = APIRouter(tags=["workbench-s05"])


def _wb() -> Any:
    from omniforge.api import workbench as wb
    return wb


@router.post("/s05/engineer")
def s05_engineer_features(
    req: FeatureEngReq,
    x_session_id: str | None = Header(default="default"),
    principal: Principal = Depends(require_permission("data:read")),
) -> dict[str, Any]:
    if req.commit:
        authorize_principal(principal, "data:write")
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    df = _wb()._require_df(state)
    targets = set(_wb().get_selected_targets(state, df))
    feat_cols = [c for c in _wb().get_analysis_dataframe(state, df).columns if c not in targets]
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
    primary_target = next((t for t in _wb().get_selected_targets(state, df) if t in df.columns), None)
    train_df, _test_df, lock = _wb().get_active_split_frames(state, df)
    split_locked = (
        isinstance(lock, dict) and bool(lock.get("active"))
        and isinstance(train_df, pd.DataFrame) and not train_df.empty
        and all(c in train_df.columns for c in feat_cols)
    )
    if split_locked:
        y_fit_fe = train_df[primary_target] if (primary_target and primary_target in train_df.columns) else None
        fe.fit(train_df[feat_cols], y_fit_fe)
        transformed = fe.transform(df[feat_cols])
        leakage_warning = None
    else:
        y_full_fe = df[primary_target] if primary_target else None
        transformed = fe.fit_transform(df[feat_cols], y_full_fe)
        leakage_warning = "Split Lock is not active; feature engineering was fitted on full dataset."
    new_cols = [c for c in transformed.columns if c not in df.columns]
    if req.commit and new_cols:
        merged = pd.concat([df, transformed[new_cols]], axis=1)
        state["uap_s05_engineered"] = True
        _wb().update_active_dataset(
            state,
            merged,
            stage="Feature Engineering",
            action=f"Engineered {len(new_cols)} new features ({mode})",
            details=", ".join(new_cols[:8]),
            preserve_stage="s05_features",
            row_index_preserved=True,  # concatenation retains the input row labels.
        )
    return wb_cache_and_respond(
        state,
        sid,
        "s05_engineer",
        {
            "mode": mode,
            "new_features": new_cols,
            "n_created": len(new_cols),
            "split_locked": split_locked,
            "leakage_warning": leakage_warning,
        },
    )


@router.post("/s05/select")
def s05_select_features(
    req: FeatureSelReq,
    x_session_id: str | None = Header(default="default"),
    principal: Principal = Depends(require_permission("data:read")),
) -> dict[str, Any]:
    if req.commit:
        authorize_principal(principal, "data:write")
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    full_df = _wb()._require_df(state)
    df = _wb().get_analysis_dataframe(state, full_df)
    targets = _wb().get_selected_targets(state, df)
    target = req.target if (req.target and req.target in df.columns) else (targets[0] if targets else None)
    if not target:
        raise HTTPException(status_code=400, detail="Select a target column before running Feature Selection.")
    train_df, _test_df, lock = _wb().get_active_split_frames(state, full_df)
    train_df = train_df.loc[:, df.columns]
    protected = set(targets) | {target}
    if isinstance(lock, dict) and lock.get("active"):
        protected.update(lock[key] for key in ("time_col", "group_col", "row_id_col") if isinstance(lock.get(key), str))
    dropped_from_inputs = [c for c in protected - {target} if c in train_df.columns]
    try:
        X_fit, y_fit, task, preprocessor = _wb()._prepare_xy_fitted(train_df.drop(columns=dropped_from_inputs), target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="The feature selection input schema could not be fitted.") from exc
    task_str = "classification" if task in (_wb().TaskType.BINARY_CLASSIFICATION, _wb().TaskType.MULTICLASS_CLASSIFICATION) else "regression"
    mode = req.mode.lower()
    methods = ["mutual_info", "f_test", "importance", "l1"] if mode in ("complete", "auto") else [req.method]

    consensus: dict[str, list[float]] = {c: [] for c in X_fit.columns}
    method_breakdown: list[dict[str, Any]] = []
    selected: list[str] = []
    scores: dict[str, float] = {}

    for m_name in methods:
        try:
            selector = FeatureSelector(
                method=cast(Any, m_name), task=cast(Any, task_str), max_features=min(req.top_k, X_fit.shape[1])
            )
            selector.fit(X_fit, y_fit)
            s_dict = dict(selector.scores_)
            max_v = max((abs(float(v)) for v in s_dict.values()), default=1.0) or 1.0
            for k, v in s_dict.items():
                if k in consensus:
                    consensus[k].append(abs(float(v)) / max_v)
            method_breakdown.append({"method": selector.method, "top_5": list(selector.selected_features_)[:5]})
            if not selected:
                selected = list(selector.selected_features_)
                scores = {k: float(v) for k, v in s_dict.items()}
        except Exception as exc:
            _wb().logger.warning("feature_selection_failed", method=m_name, error_type=type(exc).__name__)
            if len(methods) == 1:
                raise HTTPException(status_code=400, detail="The feature selection method could not evaluate this study.") from exc
            continue

    if not method_breakdown:
        raise HTTPException(
            status_code=400,
            detail="No feature selection method could evaluate this study.",
        )

    if len(methods) > 1:
        scores = {k: round(float(np.mean(vs)), 6) if vs else 0.0 for k, vs in consensus.items()}
        selected = [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: min(req.top_k, len(scores))]]

    ranked = sorted(
        [{"feature": k, "score": round(float(v), 6), "selected": k in selected} for k, v in scores.items()],
        key=lambda r: float(str(r["score"])),
        reverse=True,
    )
    sources = preprocessor.feature_sources()
    selected_raw = list(dict.fromkeys(sources[feature] for feature in selected))
    if req.commit and selected_raw:
        keep_raw = protected | set(selected_raw) | set(_wb().get_audit_only_columns(state, full_df))
        dropped = [c for c in full_df.columns if c not in keep_raw]
        if dropped:
            _wb().set_excluded_columns(
                state,
                list(set(_wb().get_excluded_columns(state, full_df)) | set(dropped)),
                df=full_df,
            )
            _wb().record_exclusion(
                state,
                stage="Studio 05 Feature Selection",
                exclusion_type="Feature Selection Exclusion",
                affected_scope=f"{len(dropped)} unselected columns",
                count=len(dropped),
                total_before=full_df.shape[1],
                criterion_rule=f"mode={mode}, method={req.method}, top_k={req.top_k}",
                reason=f"Feature selection kept {len(selected_raw)} source predictors for {len(selected)} ranked features.",
                sample_identifiers=dropped[:10],
            )
            # Only mutate the dataset when columns actually change; a no-op
            # selection should not erase otherwise valid charts or create Undo.
            remaining_cols = [c for c in full_df.columns if c not in set(dropped)]
            updated_df = full_df[remaining_cols].copy()
            _wb().update_active_dataset(
                state,
                updated_df,
                stage="Feature Selection",
                action=f"Dropped {len(dropped)} unselected columns (commit)",
                details=", ".join(dropped[:10]),
                preserve_stage="s05_features",
                row_index_preserved=True,  # selection changes columns, not row labels.
            )
    # Scope/dataset invalidation during a commit clears old selection state.
    state["uap_fs_selected_cols"] = selected
    state["uap_fs_scores"] = scores
    return wb_cache_and_respond(
        state,
        sid,
        "s05_select",
        {
            "mode": mode,
            "method_breakdown": method_breakdown,
            "selected_features": selected,
            "selected_source_columns": selected_raw,
            "fit_scope": "train_only" if isinstance(lock, dict) and lock.get("active") else "full_frame",
            "committed": req.commit,
            "rankings": ranked,
        },
    )


# ---------------------------------------------------------------------------
# Studio 06 — Multi-Backend AutoML, Paper-Grade Deep Learning & Future Predictions
# ---------------------------------------------------------------------------
