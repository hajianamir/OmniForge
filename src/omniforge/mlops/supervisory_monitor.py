"""Engineering & Supervisory Live Prediction, Actual-vs-Baseline Evaluation & Error-Tolerance Alert Engine."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, root_mean_squared_error

from omniforge.core.exceptions import EvaluationError
from omniforge.core.logging import get_logger
from omniforge.core.types import TaskType
from omniforge.core.utils import to_json_safe
from omniforge.evaluation.equations import ModelEquationExtractor
from omniforge.features.inference import WorkbenchInputPreprocessor

logger = get_logger(__name__)


def _is_classification_target(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return True
    n_uniq = int(series.nunique(dropna=True))
    return n_uniq <= 15 and n_uniq < max(5, int(len(series) * 0.15))


def _finite_round(value: float, digits: int) -> float | None:
    return round(float(value), digits) if math.isfinite(value) else None


class SupervisoryPredictionMonitor:
    """Evaluates live incoming data against ANY workspace model (Baseline, AutoML, or Neural Network),
    compares observed Reality vs. Model Prediction vs. Reference Baseline Prediction, checks user-defined
    permissible error tolerances, triggers Warning/Critical Alarms, and produces SPC engineering charts.
    """

    @staticmethod
    def catalog_available_models(state: dict[str, Any]) -> list[dict[str, Any]]:
        """Enumerate every trained Baseline, AutoML, and Deep Learning model available in the workspace."""
        models: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        def _add(mid: str, label: str, family: str, target: str, feature_cols: list[str], obj: Any, metrics: dict[str, Any] | None = None, preprocessor: WorkbenchInputPreprocessor | None = None, evaluation_holdout: tuple[pd.DataFrame, pd.Series] | None = None) -> None:
            if mid in seen_ids or obj is None or not hasattr(obj, "predict"):
                return
            seen_ids.add(mid)
            models.append(
                {
                    "model_id": mid,
                    "label": label,
                    "family": family,
                    "target": target,
                    "task_type": SupervisoryPredictionMonitor._model_task(obj),
                    "n_features": len(feature_cols),
                    "feature_cols": list(feature_cols),
                    "input_columns": list(preprocessor.input_columns) if isinstance(preprocessor, WorkbenchInputPreprocessor) else list(feature_cols),
                    "input_types": dict(preprocessor.column_types) if isinstance(preprocessor, WorkbenchInputPreprocessor) else {},
                    "metrics": metrics or {},
                    "_estimator": obj,
                    "_input_preprocessor": preprocessor,
                    "_evaluation_holdout": evaluation_holdout,
                }
            )

        primary_target = str(state.get("uap_target") or (state.get("uap_targets") or ["target"])[0] if (state.get("uap_target") or state.get("uap_targets")) else "target")

        # 1. Active Champion Model (Studio 06 AutoML or Deep Learning)
        if state.get("uap_active_model") is not None:
            m_name = str(state.get("uap_active_model_name") or "Active_Champion")
            _add(
                "active_champion",
                f"Champion · {m_name}",
                "AutoML / Champion",
                str(state.get("uap_active_model_target") or primary_target),
                list(state.get("uap_active_feature_cols") or []),
                state.get("uap_active_model"),
                state.get("uap_active_model_metrics") or {},
                state.get("uap_active_input_preprocessor"),
            )

        # 2. All Neural Network / Deep Learning Models trained in Studio 06
        dl_models = state.get("uap_dl_models") or {}
        if isinstance(dl_models, dict):
            dl_feats = list(state.get("uap_active_feature_cols") or [])
            for name, entry in dl_models.items():
                if isinstance(entry, dict) and entry.get("model") is not None:
                    _add(
                        f"dl::{name}",
                        f"Neural Net · {name}",
                        "Neural Network / Deep Learning",
                        str(entry.get("target") or primary_target),
                        list(entry.get("feature_cols") or dl_feats),
                        entry["model"],
                        entry.get("metrics") or {},
                        entry.get("input_preprocessor"),
                        entry.get("evaluation_holdout"),
                    )
                elif hasattr(entry, "predict"):
                    _add(
                        f"dl::{name}",
                        f"Neural Net · {name}",
                        "Neural Network / Deep Learning",
                        primary_target,
                        dl_feats,
                        entry,
                    )

        # 3. All Baseline Models fitted in Studio 03 Diagnostics (or Studio 14 Quick-Fit)
        baseline_models = state.get("uap_baseline_models") or {}
        if isinstance(baseline_models, dict):
            for tgt, tgt_dict in baseline_models.items():
                if not isinstance(tgt_dict, dict):
                    continue
                for b_name, entry in tgt_dict.items():
                    if isinstance(entry, dict) and entry.get("model") is not None:
                        _add(
                            f"baseline::{tgt}::{b_name}",
                            f"Baseline · {b_name} ({tgt})",
                            "Baseline",
                            str(tgt),
                            list(entry.get("feature_cols") or []),
                            entry["model"],
                            entry.get("metrics") or {},
                            entry.get("input_preprocessor"),
                        entry.get("evaluation_holdout"),
                        )
                    elif hasattr(entry, "predict"):
                        _add(
                            f"baseline::{tgt}::{b_name}",
                            f"Baseline · {b_name} ({tgt})",
                            "Baseline",
                            str(tgt),
                            list(state.get("uap_active_feature_cols") or []),
                            entry,
                        )

        # 3b. Non-Parametric Analytical Models (Spline GAM, Nadaraya-Watson, RBF Kernel Ridge, Weighted k-NN, Isotonic)
        np_models = state.get("uap_nonparametric_models") or {}
        if isinstance(np_models, dict):
            for tgt, m_map in np_models.items():
                if not isinstance(m_map, dict):
                    continue
                for m_name, entry in m_map.items():
                    if isinstance(entry, dict) and entry.get("model") is not None:
                        _add(
                            f"nonparametric::{tgt}::{m_name}",
                            f"Non-Parametric · {m_name} ({tgt})",
                            "Non-Parametric Analytical",
                            str(tgt),
                            list(entry.get("feature_cols") or []),
                            entry["model"],
                            entry.get("metrics") or {},
                            entry.get("input_preprocessor"),
                        entry.get("evaluation_holdout"),
                        )

        # 4. Studio 14 Quick-Fit Models
        quick_models = state.get("uap_supervisory_models") or {}
        if isinstance(quick_models, dict):
            for qid, entry in quick_models.items():
                if isinstance(entry, dict) and entry.get("model") is not None:
                    _add(
                        f"quick::{qid}",
                        str(entry.get("label") or qid),
                        str(entry.get("family") or "Supervisory Model"),
                        str(entry.get("target") or primary_target),
                        list(entry.get("feature_cols") or []),
                        entry["model"],
                        entry.get("metrics") or {},
                        entry.get("input_preprocessor"),
                        entry.get("evaluation_holdout"),
                    )

        return models

    @staticmethod
    def _model_task(estimator: Any) -> TaskType | None:
        declared = getattr(estimator, "task_type", None)
        if declared in (TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            return TaskType(declared)
        kind = getattr(estimator, "_estimator_type", None)
        classes = getattr(estimator, "classes_", None)
        if kind == "regressor":
            return TaskType.REGRESSION
        if classes is not None and len(classes) >= 2:
            return TaskType.MULTICLASS_CLASSIFICATION if len(classes) > 2 else TaskType.BINARY_CLASSIFICATION
        if kind == "classifier":
            return TaskType.BINARY_CLASSIFICATION
        return None

    @classmethod
    def model_pair(
        cls, catalog: list[dict[str, Any]], model_id: str | None, baseline_model_id: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        primary = next((m for m in catalog if m["model_id"] == model_id), None) if model_id else next(iter(catalog), None)
        if primary is None:
            raise EvaluationError("The requested model is unavailable in this workspace.")
        task = cls._model_task(primary["_estimator"])
        eligible = [
            m for m in catalog
            if m["target"] == primary["target"] and m["model_id"] != primary["model_id"]
            and (task is None or cls._model_task(m["_estimator"]) == task)
        ]
        if baseline_model_id:
            matched = next((m for m in eligible if m["model_id"] == baseline_model_id), None)
            if matched is None:
                raise EvaluationError("The requested comparison model is unavailable or targets a different outcome or task.")
            return primary, matched
        return primary, next((m for m in eligible if m["family"] == "Baseline"), None)

    @staticmethod
    def _evaluation_holdout(state: dict[str, Any], entry: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series] | None:
        from omniforge.mlops.fairness_evidence import holdout_fingerprint

        if entry["model_id"] == "active_champion":
            if entry["_estimator"] is not state.get("uap_active_holdout_model"):
                return None
            digest = state.get("uap_active_holdout_fingerprint")
            if not digest:
                return None
            X, y = state.get("uap_active_X_test"), state.get("uap_active_y_test")
        else:
            saved = entry.get("_evaluation_holdout")
            if not isinstance(saved, tuple) or len(saved) != 2:
                return None
            X, y = saved
            digest = None
        if (
            not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series) or X.empty
            or not X.index.is_unique or not X.index.equals(y.index) or y.isna().any()
            or y.name != entry["target"] or list(X.columns) != entry["feature_cols"]
            or (digest is not None and holdout_fingerprint(X, y) != digest)
        ):
            return None
        return X, y

    @staticmethod
    def _align_features(
        incoming_df: pd.DataFrame,
        feature_cols: list[str],
        ref_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Align incoming records to model expected columns safely (handling missing cols & +/-Inf)."""
        if not feature_cols:
            num_cols = list(incoming_df.select_dtypes(include=[np.number]).columns)
            feature_cols = num_cols
        out: dict[str, Any] = {}
        for col in feature_cols:
            if col in incoming_df.columns:
                ser = incoming_df[col]
                if pd.api.types.is_numeric_dtype(ser) or (
                    ref_df is not None and col in ref_df.columns and pd.api.types.is_numeric_dtype(ref_df[col])
                ):
                    vals = pd.to_numeric(ser, errors="coerce").replace([np.inf, -np.inf], np.nan)
                    fill_v = 0.0
                    if ref_df is not None and col in ref_df.columns:
                        ref_num = pd.to_numeric(ref_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
                        if ref_num.notna().any():
                            fill_v = float(ref_num.median())
                    elif vals.notna().any():
                        fill_v = float(vals.median())
                    out[col] = vals.fillna(fill_v if math.isfinite(fill_v) else 0.0)
                else:
                    if isinstance(ser.dtype, pd.CategoricalDtype) and "MISSING" not in ser.cat.categories:
                        ser = ser.cat.add_categories(["MISSING"])
                    out[col] = ser.fillna("MISSING").astype(str)
            else:
                fill_v = 0.0
                if ref_df is not None and col in ref_df.columns and pd.api.types.is_numeric_dtype(ref_df[col]):
                    ref_num = pd.to_numeric(ref_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
                    if ref_num.notna().any():
                        fill_v = float(ref_num.median())
                out[col] = np.full(len(incoming_df), fill_v if math.isfinite(fill_v) else 0.0)
        return pd.DataFrame(out, index=incoming_df.index)

    @classmethod
    def _model_input(
        cls, incoming_df: pd.DataFrame, entry: dict[str, Any], ref_df: pd.DataFrame | None,
        fallback_features: list[str] | None = None, *, strict: bool = False,
    ) -> pd.DataFrame:
        preprocessor = entry.get("_input_preprocessor")
        feature_cols = entry["feature_cols"] or fallback_features or []
        if isinstance(preprocessor, WorkbenchInputPreprocessor):
            if strict and not set(preprocessor.input_columns).issubset(incoming_df.columns):
                raise ValueError("One or more fitted model inputs are absent.")
            transformed = preprocessor.transform(incoming_df)
            if strict and feature_cols and list(transformed.columns) != feature_cols:
                raise ValueError("Fitted preprocessing output does not match the model feature schema.")
            return transformed
        if strict:
            if not feature_cols or not set(feature_cols).issubset(incoming_df.columns):
                raise ValueError("The raw model input schema could not be verified.")
            return incoming_df.loc[:, feature_cols]
        return cls._align_features(incoming_df, feature_cols, ref_df=ref_df)

    @staticmethod
    def _probabilities(estimator: Any, X: pd.DataFrame) -> tuple[pd.Index, np.ndarray] | None:
        classes = getattr(estimator, "classes_", None)
        if classes is None or len(classes) < 2 or not hasattr(estimator, "predict_proba"):
            return None
        labels = pd.Index(classes)
        try:
            pmat = np.asarray(estimator.predict_proba(X), dtype=float)
        except Exception:
            return None
        if (
            not labels.is_unique or labels.hasnans or pmat.shape != (len(X), len(labels))
            or not np.isfinite(pmat).all() or (pmat < 0).any() or (pmat > 1).any()
            or not np.allclose(pmat.sum(axis=1), 1, atol=1e-6, rtol=0)
        ):
            return None
        return labels, pmat

    @classmethod
    def _predict_scalar_response(cls, estimator: Any, X: pd.DataFrame, response_class: Any) -> np.ndarray:
        """Return a numeric prediction or the probability of one explicitly identified class."""
        if response_class is None:
            values = np.asarray(estimator.predict(X), dtype=float)
            if values.shape == (len(X),):
                return values
        else:
            probabilities = cls._probabilities(estimator, X)
            if probabilities is not None:
                labels, pmat = probabilities
                if response_class in labels:
                    return pmat[:, labels.get_loc(response_class)]
        raise EvaluationError("The model did not provide a comparable numeric response.")

    @classmethod
    def extract_model_equation(
        cls,
        entry: dict[str, Any] | None,
        ref_df: pd.DataFrame | None,
    ) -> dict[str, Any] | None:
        """Extract the full mathematical relationship & external deployment formulas for any cataloged model."""
        if entry is None or entry.get("_estimator") is None:
            return None
        try:
            fcols = list(entry.get("feature_cols") or [])
            X_ref = None
            if ref_df is not None and not ref_df.empty:
                X_ref = cls._model_input(ref_df.head(400), entry, ref_df)
            rep = ModelEquationExtractor.extract(
                entry["_estimator"],
                X_reference=X_ref,
                feature_names=fcols or (list(X_ref.columns) if X_ref is not None else []),
                target_name=str(entry.get("target") or "target"),
                max_terms=12,
            )
            eq_dict = rep.model_dump() if hasattr(rep, "model_dump") else dict(rep)
            eq_dict["model_name"] = str(entry.get("label") or eq_dict.get("model_name") or "Model")
            eq_dict["model_family"] = str(entry.get("family") or eq_dict.get("model_family") or "Model")
            safe_eq: dict[str, Any] = dict(to_json_safe(eq_dict))
            return safe_eq
        except Exception as exc:
            logger.debug("supervisory_equation_extract_skipped", error_type=type(exc).__name__)
            return None

    @classmethod
    def _analyze_root_causes(
        cls,
        *,
        estimator: Any,
        X_primary: pd.DataFrame,
        ref_df: pd.DataFrame | None,
        preds_scalar: np.ndarray,
        base_scalar: np.ndarray | None,
        actuals_scalar: np.ndarray | None,
        response_class: Any,
        target_name: str,
        breach_indices: list[int],
    ) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
        """Compute exact signed feature-level root-cause attributions explaining what caused
        increases (+Δ) or decreases (-Δ) relative to the Baseline reference operating envelope.

        Uses 2nd-order antithetic interventional perturbation (forward insertion + backward removal)
        around the healthy reference operating point x_ref, normalized to preserve exact additivity.
        """
        n_rows = len(X_primary)
        if n_rows == 0 or X_primary.shape[1] == 0:
            return {
                "overall_direction": "ALIGNED_WITH_BASELINE",
                "summary_narrative": "No numeric features available for root-cause attribution.",
                "increase_drivers": [],
                "decrease_drivers": [],
                "all_factor_attributions": [],
                "remediation_playbook": [],
            }, {}

        # 1. Build Baseline Reference Operating Point (x_ref) and Healthy Operating Bands [P05, P95]
        num_cols = [c for c in X_primary.columns if pd.api.types.is_numeric_dtype(X_primary[c])][:18]
        ref_row: dict[str, Any] = {}
        ref_stats: dict[str, dict[str, float]] = {}

        for c in X_primary.columns:
            if ref_df is not None and c in ref_df.columns:
                if c in num_cols:
                    ser = pd.to_numeric(ref_df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                    if len(ser) >= 3:
                        med = float(ser.median())
                        mu = float(ser.mean())
                        raw_sd = float(ser.std())
                        sd = max(raw_sd, 1e-6) if math.isfinite(raw_sd) else 1e-6
                        q05 = float(ser.quantile(0.05))
                        q95 = float(ser.quantile(0.95))
                    else:
                        live_s = pd.to_numeric(X_primary[c], errors="coerce").fillna(0.0)
                        med, mu = float(live_s.median()), float(live_s.mean())
                        raw_sd = float(live_s.std()) if len(live_s) > 1 else 0.0
                        sd = max(raw_sd, 1e-6) if math.isfinite(raw_sd) else 1e-6
                        q05, q95 = float(live_s.quantile(0.05)), float(live_s.quantile(0.95))
                    ref_row[c] = med
                    ref_stats[c] = {"median": med, "mean": mu, "std": sd, "p05": q05, "p95": q95}
                else:
                    modes = ref_df[c].dropna().astype(str).mode()
                    ref_row[c] = str(modes.iloc[0]) if len(modes) else "MISSING"
            else:
                if c in num_cols:
                    live_s = pd.to_numeric(X_primary[c], errors="coerce").fillna(0.0)
                    med, mu = float(live_s.median()), float(live_s.mean())
                    raw_sd = float(live_s.std()) if len(live_s) > 1 else 0.0
                    sd = max(raw_sd, 1e-6) if math.isfinite(raw_sd) else 1e-6
                    q05, q95 = float(live_s.quantile(0.05)), float(live_s.quantile(0.95))
                    ref_row[c] = med
                    ref_stats[c] = {"median": med, "mean": mu, "std": sd, "p05": q05, "p95": q95}
                else:
                    modes = X_primary[c].dropna().astype(str).mode()
                    ref_row[c] = str(modes.iloc[0]) if len(modes) else "MISSING"

        ref_df_one = pd.DataFrame([ref_row], columns=X_primary.columns)
        try:
            y_ref = float(cls._predict_scalar_response(estimator, ref_df_one, response_class)[0])
        except Exception:
            y_ref = float(np.nanmedian(preds_scalar)) if len(preds_scalar) else 0.0

        # Select rows to attribute: ALL breached rows (up to 60) + evenly spaced sample across batch (total <= 80)
        selected_set = set(breach_indices[:60])
        if n_rows <= 80:
            eval_indices = list(range(n_rows))
        else:
            step = max(1, n_rows // 40)
            for idx in range(0, n_rows, step):
                selected_set.add(idx)
            eval_indices = sorted(selected_set)[:80]

        X_sub = X_primary.iloc[eval_indices].copy().reset_index(drop=True)
        y_sub = np.asarray([float(preds_scalar[i]) for i in eval_indices], dtype=float)
        n_sub = len(eval_indices)

        # 2. Vectorized 2nd-Order Antithetic Interventional Perturbation (Exact Sign & Efficiency)
        # Forward: start from x_ref, insert x_{i,j} -> delta_fwd = f(x_ref with col j = x_sub[j]) - y_ref
        # Backward: start from x_sub, replace col j with x_ref[j] -> delta_bwd = y_sub - f(x_sub with col j = x_ref[j])
        phi_mat = np.zeros((n_sub, len(num_cols)), dtype=float)
        base_ref_block = pd.concat([ref_df_one] * n_sub, ignore_index=True)

        for j, col in enumerate(num_cols):
            try:
                X_fwd = base_ref_block.copy()
                X_fwd[col] = X_sub[col].to_numpy()
                y_fwd = cls._predict_scalar_response(estimator, X_fwd, response_class)

                X_bwd = X_sub.copy()
                X_bwd[col] = ref_row[col]
                y_bwd = cls._predict_scalar_response(estimator, X_bwd, response_class)

                phi_j = 0.5 * ((y_fwd - y_ref) + (y_sub - y_bwd))
                phi_mat[:, j] = np.nan_to_num(phi_j, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                phi_mat[:, j] = 0.0

        # Enforce additive efficiency per row: sum_j phi[i, j] == y_sub[i] - y_ref
        total_delta = y_sub - y_ref
        raw_sum = np.sum(phi_mat, axis=1)
        for r_i in range(n_sub):
            if abs(raw_sum[r_i]) > 1e-9 and abs(total_delta[r_i]) > 1e-9:
                # Only rescale proportionally when raw_sum aligns in sign or avoid blowing up tiny sums
                ratio_scale = total_delta[r_i] / raw_sum[r_i]
                if 0.25 <= ratio_scale <= 4.0:
                    phi_mat[r_i, :] *= ratio_scale

        # 3. Per-Row Root-Cause Diagnosis for Alarmed Records (and Single Probe!)
        per_row_diag: dict[int, dict[str, Any]] = {}
        for pos, orig_idx in enumerate(eval_indices):
            row_effects: list[dict[str, Any]] = []
            for j, col in enumerate(num_cols):
                eff = float(phi_mat[pos, j])
                st = ref_stats[col]
                val = float(X_sub.iloc[pos][col])
                z_sh = (val - st["mean"]) / st["std"]
                row_effects.append(
                    {
                        "feature": col,
                        "live_value": round(val, 4),
                        "normal_ref": round(st["median"], 4),
                        "z_shift": round(z_sh, 2),
                        "signed_impact": round(eff, 4),
                        "direction": "INCREASE (+)" if eff > 1e-6 else ("DECREASE (-)" if eff < -1e-6 else "NEUTRAL"),
                        "healthy_band": f"[{st['p05']:.3g}, {st['p95']:.3g}]",
                    }
                )
            inc_rows = sorted([e for e in row_effects if e["signed_impact"] > 1e-6], key=lambda x: x["signed_impact"], reverse=True)
            dec_rows = sorted([e for e in row_effects if e["signed_impact"] < -1e-6], key=lambda x: x["signed_impact"])
            top_inc_str = ", ".join(
                f"{e['feature']}={e['live_value']} ({e['z_shift']:+.1f}σ -> {e['signed_impact']:+.3f})"
                for e in inc_rows[:3]
            ) or "None significant"
            top_dec_str = ", ".join(
                f"{e['feature']}={e['live_value']} ({e['z_shift']:+.1f}σ -> {e['signed_impact']:+.3f})"
                for e in dec_rows[:3]
            ) or "None significant"

            # Determine record deviation vs baseline
            act_v = float(actuals_scalar[orig_idx]) if (actuals_scalar is not None and math.isfinite(float(actuals_scalar[orig_idx]))) else None
            base_v = float(base_scalar[orig_idx]) if (base_scalar is not None and math.isfinite(float(base_scalar[orig_idx]))) else y_ref
            pred_v = float(preds_scalar[orig_idx]) if math.isfinite(float(preds_scalar[orig_idx])) else y_ref
            delta_vs_base = (act_v - base_v) if act_v is not None else (pred_v - base_v)
            unexplained_res = (act_v - pred_v) if act_v is not None else 0.0

            # Build concrete actionable engineering fix for this specific record
            culprit = (inc_rows[0] if delta_vs_base >= 0 and inc_rows else (dec_rows[0] if delta_vs_base < 0 and dec_rows else (inc_rows[0] if inc_rows else (dec_rows[0] if dec_rows else None))))
            if culprit is not None and abs(culprit["z_shift"]) >= 1.2:
                st_c = ref_stats[culprit["feature"]]
                verb = "Reduce" if culprit["live_value"] > st_c["p95"] else ("Increase" if culprit["live_value"] < st_c["p05"] else "Adjust")
                fix_txt = (
                    f"{verb} '{culprit['feature']}' from {culprit['live_value']} back toward normal baseline "
                    f"{st_c['median']:.3g} (healthy band [{st_c['p05']:.3g}, {st_c['p95']:.3g}]) — "
                    f"currently shifting {target_name} by {culprit['signed_impact']:+.3f}."
                )
            elif act_v is not None and abs(unexplained_res) > max(abs(delta_vs_base) * 0.5, 1e-6):
                fix_txt = (
                    f"Measured actual deviated by {unexplained_res:+.3f} beyond what input sensors predict — "
                    f"inspect sensor calibration, unmeasured mechanical wear, or sudden process regime shift."
                )
            else:
                fix_txt = f"Inspect top shifted predictors ({top_inc_str if delta_vs_base >= 0 else top_dec_str}) against baseline operating SOP."

            per_row_diag[orig_idx] = {
                "delta_vs_baseline": round(delta_vs_base, 4),
                "unexplained_residual": round(unexplained_res, 4),
                "deviation_direction": (
                    "INCREASE (+ above Baseline)" if delta_vs_base > 1e-6 else ("DECREASE (- below Baseline)" if delta_vs_base < -1e-6 else "ALIGNED")
                ),
                "top_increase_drivers": top_inc_str,
                "top_decrease_drivers": top_dec_str,
                "recommended_fix": fix_txt,
                "factor_breakdown": sorted(row_effects, key=lambda x: abs(x["signed_impact"]), reverse=True)[:8],
            }

        # 4. Batch-Level Aggregate Root-Cause Attribution
        # Weight breached rows if breaches exist, otherwise all attributed rows
        focus_pos = [p for p, oi in enumerate(eval_indices) if oi in set(breach_indices)] or list(range(n_sub))
        mean_phi_all = np.mean(phi_mat, axis=0)
        mean_phi_focus = np.mean(phi_mat[focus_pos, :], axis=0)
        total_abs_impact = float(np.sum(np.abs(mean_phi_focus))) or 1e-9

        all_factors: list[dict[str, Any]] = []
        for j, col in enumerate(num_cols):
            st = ref_stats[col]
            live_vals_focus = X_sub.iloc[focus_pos][col].to_numpy(dtype=float)
            live_mean = float(np.nanmean(live_vals_focus))
            z_sh = (live_mean - st["mean"]) / st["std"]
            pct_sh = ((live_mean - st["median"]) / max(abs(st["median"]), 1e-6)) * 100.0
            imp_val = float(mean_phi_focus[j])
            imp_batch = float(mean_phi_all[j])
            share = abs(imp_val) / total_abs_impact * 100.0
            out_of_band = live_mean < st["p05"] or live_mean > st["p95"]

            if live_mean > st["p95"]:
                action = f"Bring '{col}' DOWN from {live_mean:.3g} into healthy band [{st['p05']:.3g}, {st['p95']:.3g}] (normal median {st['median']:.3g})."
                band_status = "ABOVE_NORMAL_BAND"
            elif live_mean < st["p05"]:
                action = f"Bring '{col}' UP from {live_mean:.3g} into healthy band [{st['p05']:.3g}, {st['p95']:.3g}] (normal median {st['median']:.3g})."
                band_status = "BELOW_NORMAL_BAND"
            elif abs(z_sh) >= 1.2:
                action = f"Monitor '{col}' (shifted {z_sh:+.2f}σ from normal {st['median']:.3g}); verify actuator/sensor trim."
                band_status = "DRIFTING"
            else:
                action = f"'{col}' is within normal band [{st['p05']:.3g}, {st['p95']:.3g}] (moderate model sensitivity)."
                band_status = "IN_BAND"

            all_factors.append(
                {
                    "feature": col,
                    "live_mean": round(live_mean, 4),
                    "normal_reference": round(st["median"], 4),
                    "healthy_band": f"[{st['p05']:.3g}, {st['p95']:.3g}]",
                    "z_shift": round(z_sh, 2),
                    "pct_shift_from_normal": round(pct_sh, 1),
                    "signed_impact": round(imp_val, 4),
                    "batch_mean_impact": round(imp_batch, 4),
                    "impact_share_pct": round(share, 1),
                    "direction": "INCREASE (+)" if imp_val > 1e-6 else ("DECREASE (-)" if imp_val < -1e-6 else "NEUTRAL"),
                    "band_status": band_status,
                    "out_of_band": out_of_band,
                    "remediation_action": action,
                }
            )

        all_factors.sort(key=lambda r: abs(r["signed_impact"]), reverse=True)
        increase_drivers = sorted([r for r in all_factors if r["signed_impact"] > 1e-6], key=lambda r: r["signed_impact"], reverse=True)
        decrease_drivers = sorted([r for r in all_factors if r["signed_impact"] < -1e-6], key=lambda r: r["signed_impact"])

        # Overall batch shift metrics vs Baseline
        if actuals_scalar is not None and base_scalar is not None:
            fin_m = np.isfinite(actuals_scalar) & np.isfinite(base_scalar) & np.isfinite(preds_scalar)
            mean_act_base = float(np.mean(actuals_scalar[fin_m] - base_scalar[fin_m])) if fin_m.any() else 0.0
            mean_base_ref = float(np.mean(np.abs(base_scalar[fin_m]))) if fin_m.any() else 1.0
            mean_act_base_pct = (mean_act_base / max(mean_base_ref, 1e-6)) * 100.0
            mean_champ_base = float(np.mean(preds_scalar[fin_m] - base_scalar[fin_m])) if fin_m.any() else 0.0
            mean_unexplained = float(np.mean(actuals_scalar[fin_m] - preds_scalar[fin_m])) if fin_m.any() else 0.0
        else:
            b_arr = base_scalar if base_scalar is not None else np.full(n_rows, y_ref)
            fin_m = np.isfinite(preds_scalar) & np.isfinite(b_arr)
            mean_champ_base = float(np.mean(preds_scalar[fin_m] - b_arr[fin_m])) if fin_m.any() else 0.0
            mean_base_ref = float(np.mean(np.abs(b_arr[fin_m]))) if fin_m.any() else 1.0
            mean_act_base = mean_champ_base
            mean_act_base_pct = (mean_champ_base / max(mean_base_ref, 1e-6)) * 100.0
            mean_unexplained = 0.0

        if mean_act_base_pct > 1.0:
            overall_dir = "INCREASE_ABOVE_BASELINE"
        elif mean_act_base_pct < -1.0:
            overall_dir = "DECREASE_BELOW_BASELINE"
        else:
            overall_dir = "ALIGNED_WITH_BASELINE"

        top_up_txt = ", ".join(
            f"{d['feature']} ({d['signed_impact']:+.3f}, live={d['live_mean']} vs ref={d['normal_reference']}, {d['z_shift']:+.1f}σ)"
            for d in increase_drivers[:3]
        ) or "none"
        top_dn_txt = ", ".join(
            f"{d['feature']} ({d['signed_impact']:+.3f}, live={d['live_mean']} vs ref={d['normal_reference']}, {d['z_shift']:+.1f}σ)"
            for d in decrease_drivers[:3]
        ) or "none"

        scope_label = f"{len(breach_indices)} breached/alarmed records" if breach_indices else f"all {n_rows} evaluated records"
        summary_narrative = (
            f"Relative to Baseline, '{target_name}' shifted by {mean_act_base:+.4f} ({mean_act_base_pct:+.2f}%) "
            f"[Status: {overall_dir.replace('_', ' ')}]. Root-cause decomposition across {scope_label}: "
            f"Primary INCREASE (+) drivers: {top_up_txt}. "
            f"Primary DECREASE (-) drivers: {top_dn_txt}. "
            f"Unexplained residual (Actual − Champion): {mean_unexplained:+.4f}."
        )

        # Build prioritized actionable remediation playbook
        remediation_playbook: list[dict[str, Any]] = []
        prioritized = sorted(
            all_factors,
            key=lambda r: (1 if r["out_of_band"] else 0, abs(r["signed_impact"])),
            reverse=True,
        )
        for rank_i, item in enumerate(prioritized[:6], start=1):
            remediation_playbook.append(
                {
                    "priority": f"P{rank_i}",
                    "variable": item["feature"],
                    "impact_on_target": f"{item['signed_impact']:+.4f} ({item['direction']})",
                    "live_vs_normal": f"{item['live_mean']} vs {item['normal_reference']} ({item['z_shift']:+.1f}σ)",
                    "healthy_operating_band": item["healthy_band"],
                    "corrective_action": item["remediation_action"],
                }
            )
        if actuals_scalar is not None and abs(mean_unexplained) > max(abs(mean_champ_base), 1e-4) * 0.75:
            remediation_playbook.append(
                {
                    "priority": f"P{len(remediation_playbook) + 1}",
                    "variable": "Unexplained Process / Sensor Residual",
                    "impact_on_target": f"{mean_unexplained:+.4f} (Actual − Model)",
                    "live_vs_normal": "Measured output deviates beyond measured input shifts",
                    "healthy_operating_band": "Residual near 0 ± 2σ",
                    "corrective_action": "Audit target sensor calibration, check for unmeasured mechanical degradation/fouling, or retrain model on new regime (see Studio 11 Drift Monitor).",
                }
            )

        root_cause_summary = {
            "overall_direction": overall_dir,
            "attribution_scope": scope_label,
            "attribution_method": "2nd-Order Antithetic Interventional Perturbation (Signed Factor Decomposition vs. Reference Baseline)",
            "reference_baseline_output": round(y_ref, 4),
            "mean_actual_vs_baseline_delta": round(mean_act_base, 4),
            "mean_actual_vs_baseline_pct": round(mean_act_base_pct, 2),
            "mean_champion_vs_baseline_delta": round(mean_champ_base, 4),
            "mean_unexplained_residual": round(mean_unexplained, 4),
            "summary_narrative": summary_narrative,
            "increase_drivers": increase_drivers[:10],
            "decrease_drivers": decrease_drivers[:10],
            "all_factor_attributions": all_factors[:14],
            "remediation_playbook": remediation_playbook,
        }
        return root_cause_summary, per_row_diag

    def evaluate_incoming(
        self,
        state: dict[str, Any],
        incoming_df: pd.DataFrame,
        *,
        model_id: str | None,
        baseline_model_id: str | None = None,
        actual_col: str | None = None,
        id_col: str | None = None,
        source_label: str = "Uploaded Batch",
        max_pct_error: float = 10.0,
        max_abs_error: float | None = None,
        max_sigma_z: float = 3.0,
        baseline_ratio_min: float = 0.85,
        baseline_ratio_max: float = 1.15,
        warning_fraction: float = 0.75,
        max_batch_breach_rate_pct: float = 5.0,
        min_classification_accuracy_pct: float = 85.0,
    ) -> dict[str, Any]:
        """Run full live prediction, actual-vs-baseline comparison, tolerance verification & alarm generation."""
        if incoming_df is None or incoming_df.empty:
            raise EvaluationError("Incoming operational dataset is empty.")

        catalog = self.catalog_available_models(state)
        if not catalog:
            raise EvaluationError(
                "No trained model or baseline is available yet. Run Studio 03 (Baselines), Studio 06 (AutoML / Deep Learning), "
                "or click 'Quick-Train Model' inside Studio 14."
            )

        primary_entry, baseline_entry = self.model_pair(catalog, model_id, baseline_model_id)
        target_name = primary_entry["target"]
        ref_df = state.get("uap_df") if isinstance(state.get("uap_df"), pd.DataFrame) else None
        if actual_col and actual_col not in incoming_df.columns:
            raise EvaluationError("The selected actual column is absent from the incoming data.")
        known_outcomes = {m["target"] for m in catalog} | set(state.get("uap_targets") or ())
        if actual_col != target_name and actual_col in known_outcomes:
            raise EvaluationError("The selected actual column belongs to a different trained outcome.")
        resolved_actual = actual_col or (target_name if target_name in incoming_df.columns else None)

        work_df = incoming_df.copy().reset_index(drop=True)
        try:
            X_primary = self._model_input(work_df, primary_entry, ref_df, strict=primary_entry.get("_input_preprocessor") is None)
            preds = np.asarray(primary_entry["_estimator"].predict(X_primary))
        except Exception as exc:
            raise EvaluationError("The fitted model could not score the incoming feature schema.") from exc
        if preds.shape != (len(work_df),):
            raise EvaluationError("The model returned an invalid prediction shape.")
        model_task = self._model_task(primary_entry["_estimator"])
        if model_task is not None:
            is_clf = model_task != TaskType.REGRESSION
        elif ref_df is not None and target_name in ref_df.columns:
            is_clf = _is_classification_target(ref_df[target_name])
        else:
            is_clf = not pd.api.types.is_numeric_dtype(preds)

        baseline_label = baseline_entry["label"] if baseline_entry is not None else None
        X_base = None
        base_preds = np.full(len(work_df), np.nan)
        if baseline_entry is not None:
            try:
                X_base = self._model_input(
                    work_df, baseline_entry, ref_df, strict=baseline_entry.get("_input_preprocessor") is None,
                )
                base_preds = np.asarray(baseline_entry["_estimator"].predict(X_base))
            except Exception as exc:
                raise EvaluationError("The comparison model could not score the incoming feature schema.") from exc
            if base_preds.shape != (len(work_df),):
                raise EvaluationError("The comparison model returned an invalid prediction shape.")

        # Compute reference feature means/stds for root-cause deviation hints on breach rows
        feat_stats: dict[str, tuple[float, float]] = {}
        if ref_df is not None:
            for c in primary_entry["feature_cols"][:20]:
                if c in ref_df.columns and pd.api.types.is_numeric_dtype(ref_df[c]):
                    r_v = pd.to_numeric(ref_df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                    if len(r_v) >= 5:
                        feat_stats[c] = (float(r_v.mean()), max(float(r_v.std()), 1e-6))

        def _top_deviations(row_idx: int) -> str:
            devs: list[tuple[float, str]] = []
            for c, (mu, sd) in feat_stats.items():
                if c in X_primary.columns:
                    try:
                        val = float(X_primary.iloc[row_idx][c])
                        z = (val - mu) / sd
                        if abs(z) >= 2.0 and math.isfinite(z):
                            devs.append((abs(z), f"{c} ({z:+.1f}σ)"))
                    except Exception:
                        pass
            devs.sort(reverse=True)
            return ", ".join(lbl for _, lbl in devs[:3]) if devs else "Within normal feature envelope"

        row_ids = (
            work_df[id_col].astype(str).tolist()
            if (id_col and id_col in work_df.columns)
            else [f"#{i + 1}" for i in range(len(work_df))]
        )

        rows_out: list[dict[str, Any]] = []
        alarms: list[dict[str, Any]] = []
        breach_indices: list[int] = []
        n_breach = 0
        n_warn = 0
        n_in_spec = 0
        n_unavailable = 0

        kpis: dict[str, Any] = {
            "n_records": int(len(work_df)),
            "has_actuals": resolved_actual is not None,
            "task_mode": "classification" if is_clf else "regression",
            "model_label": primary_entry["label"],
            "model_id": primary_entry["model_id"],
            "target": target_name,
            "baseline_model_id": baseline_entry["model_id"] if baseline_entry is not None else None,
            "model_family": primary_entry["family"],
            "baseline_label": baseline_label,
            "baseline_unavailable_reason": "No eligible comparison model is available." if baseline_entry is None else None,
        }

        trend_series: list[dict[str, Any]] = []
        spc_points: list[dict[str, Any]] = []
        ratio_points: list[dict[str, Any]] = []
        error_hist: list[dict[str, Any]] = []
        spc_limits: dict[str, float] = {}

        if resolved_actual is not None and not is_clf:
            actuals = pd.to_numeric(work_df[resolved_actual], errors="coerce").to_numpy(dtype=float)
            p_num = pd.to_numeric(pd.Series(preds), errors="coerce").to_numpy(dtype=float)
            b_num = pd.to_numeric(pd.Series(base_preds), errors="coerce").to_numpy(dtype=float)

            residuals = actuals - p_num
            abs_errors = np.abs(residuals)
            denom = np.where(np.abs(actuals) > 1e-6, np.abs(actuals), np.maximum(np.abs(p_num), 1.0))
            pct_errors = (abs_errors / denom) * 100.0
            base_abs_errors = np.abs(actuals - b_num)
            base_denom = np.where(np.abs(b_num) > 1e-6, b_num, np.nan)
            actual_base_ratio = actuals / base_denom

            recorded = self._evaluation_holdout(state, primary_entry)
            res_z = np.full(len(work_df), np.nan)
            if recorded is not None:
                X_ref, y_ref = recorded
                raw_ref = np.asarray(primary_entry["_estimator"].predict(X_ref))
                if raw_ref.shape == (len(y_ref),):
                    ref_res = pd.to_numeric(y_ref, errors="coerce").to_numpy(dtype=float) - pd.to_numeric(raw_ref, errors="coerce")
                    ref_res = ref_res[np.isfinite(ref_res)]
                    if len(ref_res) >= 3:
                        center = float(np.mean(ref_res))
                        sigma = float(np.std(ref_res, ddof=1))
                        if math.isfinite(center) and math.isfinite(sigma) and sigma > 0:
                            res_z = (residuals - center) / sigma
                            ucl = max_sigma_z * sigma
                            warn_lim = warning_fraction * ucl
                            spc_limits = {
                                "center": center,
                                "ucl": center + ucl,
                                "lcl": center - ucl,
                                "warn_upper": center + warn_lim,
                                "warn_lower": center - warn_lim,
                                "sigma": sigma,
                            }

            for i in range(len(work_df)):
                act = float(actuals[i]) if math.isfinite(actuals[i]) else None
                prd = float(p_num[i]) if math.isfinite(p_num[i]) else None
                bpr = float(b_num[i]) if math.isfinite(b_num[i]) else None
                ae = float(abs_errors[i]) if math.isfinite(abs_errors[i]) else None
                pe = float(pct_errors[i]) if math.isfinite(pct_errors[i]) else None
                rz = float(res_z[i]) if math.isfinite(res_z[i]) else None
                rat = float(actual_base_ratio[i]) if math.isfinite(actual_base_ratio[i]) else None
                bae = float(base_abs_errors[i]) if math.isfinite(base_abs_errors[i]) else None

                breach_reasons: list[str] = []
                warn_reasons: list[str] = []

                if pe is not None and pe > max_pct_error:
                    breach_reasons.append(f"%Err {pe:.2f}% > {max_pct_error:.1f}%")
                elif pe is not None and pe > max_pct_error * warning_fraction:
                    warn_reasons.append(f"%Err {pe:.2f}% near limit")

                if ae is not None and max_abs_error is not None and max_abs_error > 0:
                    if ae > max_abs_error:
                        breach_reasons.append(f"|Err| {ae:.3f} > {max_abs_error:.3f}")
                    elif ae > max_abs_error * warning_fraction:
                        warn_reasons.append(f"|Err| {ae:.3f} near limit")

                if rz is not None and abs(rz) > max_sigma_z:
                    breach_reasons.append(f"SPC |Z|={abs(rz):.2f}σ > {max_sigma_z:.1f}σ")
                elif rz is not None and abs(rz) > max_sigma_z * warning_fraction:
                    warn_reasons.append(f"SPC |Z|={abs(rz):.2f}σ warning")

                if rat is not None:
                    if rat < baseline_ratio_min or rat > baseline_ratio_max:
                        breach_reasons.append(f"Act/Base ratio {rat:.3f} outside [{baseline_ratio_min:.2f}, {baseline_ratio_max:.2f}]")

                if act is None or prd is None:
                    st = "UNAVAILABLE"
                    n_unavailable += 1
                    reason_str = "Observed outcome or model prediction is missing or invalid."
                elif breach_reasons:
                    st = "BREACH"
                    n_breach += 1
                    breach_indices.append(i)
                    reason_str = " | ".join(breach_reasons)
                    if len(alarms) < 80:
                        alarms.append(
                            {
                                "_row_idx": i,
                                "row_id": row_ids[i],
                                "severity": "CRITICAL",
                                "actual": round(act, 4) if act is not None else None,
                                "predicted": round(prd, 4) if prd is not None else None,
                                "baseline_predicted": round(bpr, 4) if bpr is not None else None,
                                "pct_error": round(pe, 2) if pe is not None else None,
                                "actual_to_baseline_ratio": round(rat, 4) if rat is not None else None,
                                "reason": reason_str,
                                "feature_drivers": _top_deviations(i),
                            }
                        )
                elif warn_reasons:
                    st = "WARNING"
                    n_warn += 1
                    reason_str = " | ".join(warn_reasons)
                else:
                    st = "IN_SPEC"
                    n_in_spec += 1
                    reason_str = "Within permissible tolerance"

                if i < 500:
                    rows_out.append(
                        {
                            "row_id": row_ids[i],
                            "actual": round(act, 4) if act is not None else None,
                            "model_pred": round(prd, 4) if prd is not None else None,
                            "baseline_pred": round(bpr, 4) if bpr is not None else None,
                            "abs_error": round(ae, 4) if ae is not None else None,
                            "pct_error": round(pe, 2) if pe is not None else None,
                            "residual": _finite_round(residuals[i], 4),
                            "z_score": round(rz, 2) if rz is not None else None,
                            "actual_vs_baseline_ratio": round(rat, 4) if rat is not None else None,
                            "baseline_error": round(bae, 4) if bae is not None else None,
                            "status": st,
                            "alert_reason": reason_str,
                        }
                    )

            # Report primary performance on all valid rows and lift on matched rows only.
            valid = np.isfinite(actuals) & np.isfinite(p_num) & np.isfinite(residuals)
            kpis["evaluated_rows"] = int(valid.sum())
            if valid.any():
                kpis.update(
                    {
                        "mae": _finite_round(mean_absolute_error(actuals[valid], p_num[valid]), 4),
                        "rmse": _finite_round(root_mean_squared_error(actuals[valid], p_num[valid]), 4),
                        "mape_pct": _finite_round(np.mean(pct_errors[valid]), 2),
                        "mean_bias": _finite_round(np.mean(p_num[valid] - actuals[valid]), 4),
                        "max_pct_error_observed": _finite_round(np.max(pct_errors[valid]), 2),
                        "max_abs_error_observed": _finite_round(np.max(abs_errors[valid]), 4),
                    }
                )
            if valid.sum() >= 2:
                kpis["r2"] = _finite_round(r2_score(actuals[valid], p_num[valid], force_finite=False), 4)
            matched = valid & np.isfinite(b_num) & np.isfinite(base_abs_errors)
            kpis["comparison_rows"] = int(matched.sum())
            if matched.any():
                b_rmse = float(root_mean_squared_error(actuals[matched], b_num[matched]))
                matched_rmse = float(root_mean_squared_error(actuals[matched], p_num[matched]))
                fin_rat = actual_base_ratio[matched & np.isfinite(actual_base_ratio)]
                kpis.update(
                    {
                        "baseline_mae": _finite_round(mean_absolute_error(actuals[matched], b_num[matched]), 4),
                        "baseline_rmse": _finite_round(b_rmse, 4),
                        "baseline_mape_pct": _finite_round(np.mean((base_abs_errors[matched] / denom[matched]) * 100.0), 2),
                        "model_vs_baseline_lift_pct": _finite_round((b_rmse - matched_rmse) / b_rmse * 100.0, 2) if b_rmse > 0 else None,
                        "mean_actual_to_baseline_ratio": _finite_round(np.mean(fin_rat), 4) if len(fin_rat) else None,
                    }
                )

            # Subsample charts cleanly to <= 180 points for crisp engineering rendering
            step = max(1, math.ceil(len(work_df) / 180))
            for idx in range(0, len(work_df), step):
                st_i = "UNAVAILABLE"
                if math.isfinite(res_z[idx]):
                    st_i = "BREACH" if abs(res_z[idx]) > max_sigma_z else (
                        "WARNING" if abs(res_z[idx]) > max_sigma_z * warning_fraction else "IN_SPEC"
                    )
                trend_series.append(
                    {
                        "label": row_ids[idx],
                        "actual": round(float(actuals[idx]), 4) if math.isfinite(actuals[idx]) else None,
                        "predicted": round(float(p_num[idx]), 4) if math.isfinite(p_num[idx]) else None,
                        "baseline": round(float(b_num[idx]), 4) if math.isfinite(b_num[idx]) else None,
                    }
                )
                spc_points.append(
                    {
                        "label": row_ids[idx],
                        "residual": _finite_round(residuals[idx], 4),
                        "ucl": spc_limits.get("ucl"),
                        "lcl": spc_limits.get("lcl"),
                        "status": st_i,
                    }
                )
                if math.isfinite(actual_base_ratio[idx]):
                    ratio_points.append(
                        {
                            "label": row_ids[idx],
                            "ratio": round(float(actual_base_ratio[idx]), 4),
                            "min_bound": baseline_ratio_min,
                            "max_bound": baseline_ratio_max,
                        }
                    )

            # Error distribution histogram
            fin_pe = pct_errors[np.isfinite(pct_errors)]
            if len(fin_pe) >= 4:
                clipped = np.clip(fin_pe, 0.0, max(max_pct_error * 2.5, float(np.percentile(fin_pe, 98))))
                counts, edges = np.histogram(clipped, bins=16)
                error_hist = [
                    {
                        "label": f"{edges[j]:.1f}%",
                        "count": int(counts[j]),
                        "breach": bool(edges[j] >= max_pct_error),
                    }
                    for j in range(len(counts))
                ]

        elif resolved_actual is not None and is_clf:
            actuals_s = work_df[resolved_actual].reset_index(drop=True)
            preds_s = pd.Series(preds)
            base_s = pd.Series(base_preds)
            classes = getattr(primary_entry["_estimator"], "classes_", None)
            valid_labels = actuals_s.notna() & preds_s.notna()
            if classes is not None:
                valid_labels &= actuals_s.isin(classes) & preds_s.isin(classes)
            matches = actuals_s.eq(preds_s).fillna(False)
            base_matches = actuals_s.eq(base_s).fillna(False)
            matched_labels = valid_labels & base_s.notna()
            probas = None
            probabilities = self._probabilities(primary_entry["_estimator"], X_primary)
            if probabilities is not None:
                labels, pmat = probabilities
                indices = labels.get_indexer(preds)
                probas = np.where(indices >= 0, pmat[np.arange(len(preds)), indices] * 100, np.nan)

            for i in range(len(work_df)):
                actual = actuals_s.iloc[i] if pd.notna(actuals_s.iloc[i]) else None
                prediction = preds_s.iloc[i] if pd.notna(preds_s.iloc[i]) else None
                baseline = base_s.iloc[i] if pd.notna(base_s.iloc[i]) else None
                match_b = bool(base_matches.iloc[i]) if matched_labels.iloc[i] else None
                conf = _finite_round(probas[i], 1) if probas is not None else None
                if not valid_labels.iloc[i]:
                    st = "UNAVAILABLE"
                    n_unavailable += 1
                    reason_str = "Observed outcome or prediction is missing or outside the model's class labels."
                elif not matches.iloc[i]:
                    st = "BREACH"
                    n_breach += 1
                    breach_indices.append(i)
                    reason_str = "Predicted class differs from the observed class."
                    if len(alarms) < 80:
                        alarms.append(
                            {
                                "_row_idx": i,
                                "row_id": row_ids[i],
                                "severity": "CRITICAL",
                                "actual": actual,
                                "predicted": prediction,
                                "baseline_predicted": baseline,
                                "pct_error": 100.0,
                                "reason": reason_str,
                                "feature_drivers": _top_deviations(i),
                            }
                        )
                elif conf is not None and conf < 60.0:
                    st = "WARNING"
                    n_warn += 1
                    reason_str = f"Low prediction confidence ({conf:.1f}%)"
                else:
                    st = "IN_SPEC"
                    n_in_spec += 1
                    reason_str = "Exact match"
                if i < 500:
                    rows_out.append(
                        {
                            "row_id": row_ids[i],
                            "actual": actual,
                            "model_pred": prediction,
                            "baseline_pred": baseline,
                            "confidence_pct": conf,
                            "baseline_match": ("Match" if match_b else "Miss") if match_b is not None else None,
                            "status": st,
                            "alert_reason": reason_str,
                        }
                    )

            kpis["evaluated_rows"] = int(valid_labels.sum())
            kpis["comparison_rows"] = int(matched_labels.sum())
            kpis["min_accuracy_threshold_pct"] = min_classification_accuracy_pct
            if valid_labels.any():
                kpis["accuracy_pct"] = round(float(accuracy_score(actuals_s[valid_labels], preds_s[valid_labels])) * 100, 2)
                kpis["f1_weighted_pct"] = round(float(f1_score(actuals_s[valid_labels], preds_s[valid_labels], average="weighted", zero_division=0)) * 100, 2)
            if matched_labels.any():
                b_acc = float(base_matches[matched_labels].mean()) * 100.0
                acc = float(matches[matched_labels].mean()) * 100.0
                kpis["baseline_accuracy_pct"] = round(b_acc, 2)
                kpis["model_vs_baseline_lift_pct"] = round(acc - b_acc, 2)

            win = max(5, len(work_df) // 25)
            for idx in range(win, len(work_df) + 1, max(1, win // 2)):
                window = slice(idx - win, idx)
                valid_window = valid_labels.iloc[window]
                matched_window = matched_labels.iloc[window]
                trend_series.append(
                    {
                        "label": row_ids[idx - 1],
                        "actual": None,
                        "threshold": min_classification_accuracy_pct,
                        "predicted": _finite_round(matches.iloc[window][valid_window].mean() * 100.0, 2),
                        "baseline": _finite_round(base_matches.iloc[window][matched_window].mean() * 100.0, 2),
                    }
                )
        else:
            # Predictions without observed outcomes cannot establish compliance.
            for i in range(len(work_df)):
                prediction = preds[i] if pd.notna(preds[i]) else None
                baseline = base_preds[i] if pd.notna(base_preds[i]) else None
                if not is_clf:
                    prediction = _finite_round(pd.to_numeric(prediction, errors="coerce"), 4)
                    baseline = _finite_round(pd.to_numeric(baseline, errors="coerce"), 4)
                if not is_clf and i % max(1, math.ceil(len(work_df) / 180)) == 0:
                    trend_series.append({"label": row_ids[i], "actual": None, "predicted": prediction, "baseline": baseline})
                n_unavailable += 1
                if i < 500:
                    rows_out.append(
                        {
                            "row_id": row_ids[i],
                            "actual": None,
                            "model_pred": prediction,
                            "baseline_pred": baseline,
                            "status": "UNAVAILABLE",
                            "alert_reason": "An observed outcome is required to evaluate tolerance compliance.",
                        }
                    )

        evaluated = n_breach + n_warn + n_in_spec
        breach_rate = round(n_breach / evaluated * 100.0, 2) if evaluated else None
        compliance_rate = round((evaluated - n_breach) / evaluated * 100.0, 2) if evaluated else None
        kpis.update(
            {
                "in_spec_count": n_in_spec,
                "warning_count": n_warn,
                "breach_count": n_breach,
                "unavailable_count": n_unavailable,
                "breach_rate_pct": breach_rate,
                "compliance_rate_pct": compliance_rate,
            }
        )

        # System-Level Health & Executive Alarm Banner
        system_alerts: list[str] = []
        if breach_rate is not None and breach_rate > max_batch_breach_rate_pct:
            system_alerts.append(
                f"CRITICAL TOLERANCE BREACH: {n_breach:,} / {evaluated:,} evaluable records ({breach_rate:.2f}%) exceeded permissible error threshold (batch limit: {max_batch_breach_rate_pct:.1f}%)."
            )
        elif n_breach > 0:
            system_alerts.append(
                f"TOLERANCE BREACH DETECTED: {n_breach:,} record(s) ({breach_rate:.2f}%) crossed configured error thresholds."
            )
        if is_clf and resolved_actual and kpis.get("accuracy_pct", 100.0) < min_classification_accuracy_pct:
            system_alerts.append(
                f"ACCURACY ALARM: Observed accuracy ({kpis['accuracy_pct']:.2f}%) fell below minimum permissible floor ({min_classification_accuracy_pct:.1f}%)."
            )
        if not system_alerts and n_warn > 0:
            system_alerts.append(
                f"EARLY WARNING: {n_warn:,} record(s) are approaching error tolerance boundaries (>={int(warning_fraction * 100)}% of limit)."
            )

        if n_unavailable:
            system_alerts.append(f"{n_unavailable:,} record(s) could not be evaluated for tolerance compliance.")
        if (breach_rate is not None and breach_rate > max_batch_breach_rate_pct) or (
            is_clf and resolved_actual and kpis.get("accuracy_pct", 100.0) < min_classification_accuracy_pct
        ):
            system_status = "CRITICAL_ALARM"
        elif not evaluated:
            system_status = "UNAVAILABLE"
        elif n_breach or n_warn or n_unavailable:
            system_status = "WARNING"
        else:
            system_status = "NORMAL"

        champion_equation = self.extract_model_equation(primary_entry, ref_df=ref_df)
        baseline_equation = self.extract_model_equation(baseline_entry, ref_df=ref_df)

        # Explain only directly comparable numeric responses. In classification
        # both models must predict the probability of the same explicit class.
        root_cause_analysis: dict[str, Any] | None = None
        per_row_diag: dict[int, dict[str, Any]] = {}
        if baseline_entry is not None and X_base is not None:
            estimator = primary_entry["_estimator"]
            baseline_estimator = baseline_entry["_estimator"]
            classes = getattr(estimator, "classes_", None)
            response_class = classes[-1] if is_clf and classes is not None and len(classes) >= 2 else None
            if not is_clf or response_class is not None:
                try:
                    preds_scalar = self._predict_scalar_response(estimator, X_primary, response_class)
                    base_scalar = self._predict_scalar_response(baseline_estimator, X_base, response_class)
                except (EvaluationError, TypeError, ValueError):
                    pass
                else:
                    if np.isfinite(preds_scalar).all() and np.isfinite(base_scalar).all():
                        actuals_scalar = (
                            pd.to_numeric(work_df[resolved_actual], errors="coerce").to_numpy(dtype=float)
                            if resolved_actual is not None and not is_clf else None
                        )
                        root_cause_analysis, per_row_diag = self._analyze_root_causes(
                            estimator=estimator,
                            X_primary=X_primary,
                            ref_df=ref_df,
                            preds_scalar=preds_scalar,
                            base_scalar=base_scalar,
                            actuals_scalar=actuals_scalar,
                            response_class=response_class,
                            target_name=target_name,
                            breach_indices=breach_indices,
                        )
                        root_cause_analysis["response_class"] = response_class

        # Enrich every alarm record with its exact signed Increase/Decrease root-cause diagnosis & fix
        enriched_alarms: list[dict[str, Any]] = []
        for al in alarms:
            al_copy = dict(al)
            r_idx = al_copy.pop("_row_idx", None)
            if r_idx is not None and r_idx in per_row_diag:
                diag = per_row_diag[r_idx]
                al_copy["delta_vs_baseline"] = diag["delta_vs_baseline"]
                al_copy["deviation_direction"] = diag["deviation_direction"]
                al_copy["top_increase_drivers"] = diag["top_increase_drivers"]
                al_copy["top_decrease_drivers"] = diag["top_decrease_drivers"]
                al_copy["recommended_fix"] = diag["recommended_fix"]
                al_copy["factor_breakdown"] = diag["factor_breakdown"]
            enriched_alarms.append(al_copy)

        result = to_json_safe(
            {
                "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "source": source_label,
                "system_status": system_status,
                "system_alerts": system_alerts,
                "tolerances": {
                    "max_pct_error": max_pct_error,
                    "max_abs_error": max_abs_error,
                    "max_sigma_z": max_sigma_z,
                    "baseline_ratio_min": baseline_ratio_min,
                    "baseline_ratio_max": baseline_ratio_max,
                    "warning_fraction": warning_fraction,
                    "max_batch_breach_rate_pct": max_batch_breach_rate_pct,
                    "min_classification_accuracy_pct": min_classification_accuracy_pct,
                },
                "kpis": kpis,
                "champion_equation": champion_equation,
                "baseline_equation": baseline_equation,
                "root_cause_analysis": root_cause_analysis,
                "spc_limits": spc_limits or None,
                "spc_unavailable_reason": (
                    "SPC requires a verified held-out residual reference with nonzero variation."
                    if resolved_actual is not None and not is_clf and not spc_limits else None
                ),
                "trend_series": trend_series,
                "spc_points": spc_points,
                "ratio_points": ratio_points,
                "error_histogram": error_hist,
                "alarms": sorted(enriched_alarms, key=lambda a: float(a.get("pct_error") or 0.0), reverse=True),
                "telemetry_rows": rows_out,
            }
        )

        # Persist latest result and longitudinal monitoring history for Reporting & Supervisory Dashboard
        state["uap_supervisory_result"] = result
        state["uap_supervisory_scored_df"] = pd.DataFrame(rows_out)
        # Retain the analysis inputs needed for cohort diagnostics. Unrelated
        # columns from operational uploads stay out of saved workspace results.
        study_columns = ref_df.columns if ref_df is not None else X_primary.columns
        shared_columns = [column for column in work_df.columns if column in study_columns]
        state["uap_supervisory_input_df"] = work_df.loc[:, shared_columns].copy()
        hist = state.setdefault("uap_supervisory_history", [])
        hist.append(
            {
                "timestamp": result["timestamp_utc"],
                "source": source_label,
                "model": primary_entry["label"],
                "target": target_name,
                "baseline": baseline_label,
                "records": len(work_df),
                "compliance_pct": compliance_rate,
                "breaches": n_breach,
                "mape_pct": kpis.get("mape_pct"),
                "accuracy_pct": kpis.get("accuracy_pct"),
                "status": system_status,
            }
        )
        state["uap_supervisory_history"] = hist[-30:]
        logger.info(
            "supervisory_evaluation_completed",
            model=primary_entry["label"],
            records=len(work_df),
            status=system_status,
            breaches=n_breach,
        )
        return dict(result)
