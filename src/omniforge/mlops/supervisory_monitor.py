"""Engineering & Supervisory Live Prediction, Actual-vs-Baseline Evaluation & Error-Tolerance Alert Engine."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, root_mean_squared_error

from omniforge.core.logging import get_logger
from omniforge.core.utils import to_json_safe
from omniforge.evaluation.equations import ModelEquationExtractor
from omniforge.features.inference import WorkbenchInputPreprocessor

logger = get_logger(__name__)


def _is_classification_target(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return True
    n_uniq = int(series.nunique(dropna=True))
    return n_uniq <= 15 and n_uniq < max(5, int(len(series) * 0.15))


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

        def _add(mid: str, label: str, family: str, target: str, feature_cols: list[str], obj: Any, metrics: dict[str, Any] | None = None, preprocessor: WorkbenchInputPreprocessor | None = None) -> None:
            if mid in seen_ids or obj is None or not hasattr(obj, "predict"):
                return
            seen_ids.add(mid)
            models.append(
                {
                    "model_id": mid,
                    "label": label,
                    "family": family,
                    "target": target,
                    "n_features": len(feature_cols),
                    "feature_cols": list(feature_cols),
                    "metrics": metrics or {},
                    "_estimator": obj,
                    "_input_preprocessor": preprocessor,
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
                    )

        return models

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
        fallback_features: list[str] | None = None,
    ) -> pd.DataFrame:
        preprocessor = entry.get("_input_preprocessor")
        if isinstance(preprocessor, WorkbenchInputPreprocessor):
            return preprocessor.transform(incoming_df)
        return cls._align_features(incoming_df, entry["feature_cols"] or fallback_features or [], ref_df=ref_df)

    @staticmethod
    def _predict_safe(estimator: Any, X: pd.DataFrame) -> np.ndarray:
        """Predict cleanly across both raw-DataFrame and numeric-only estimators."""
        try:
            return np.asarray(estimator.predict(X))
        except Exception:
            X_num = pd.DataFrame(
                {c: pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0) for c in X.columns},
                index=X.index,
            )
            return np.asarray(estimator.predict(X_num))

    @staticmethod
    def _predict_scalar_response(estimator: Any, X: pd.DataFrame, is_clf: bool) -> np.ndarray:
        """Return a continuous scalar response (y_hat for regression or P(y=1) for classification)."""
        if is_clf and hasattr(estimator, "predict_proba"):
            try:
                pmat = np.asarray(estimator.predict_proba(X), dtype=float)
                if pmat.ndim == 2 and pmat.shape[1] >= 2:
                    return pmat[:, 1].ravel()
                return pmat.ravel()
            except Exception:
                try:
                    X_num = pd.DataFrame(
                        {c: pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0) for c in X.columns},
                        index=X.index,
                    )
                    pmat = np.asarray(estimator.predict_proba(X_num), dtype=float)
                    return pmat[:, 1].ravel() if (pmat.ndim == 2 and pmat.shape[1] >= 2) else pmat.ravel()
                except Exception:
                    pass
        raw = SupervisoryPredictionMonitor._predict_safe(estimator, X)
        num = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=float)
        if np.isnan(num).all():
            codes, _ = pd.factorize(pd.Series(raw).astype(str))
            return np.asarray(codes, dtype=float)
        return np.asarray(np.nan_to_num(num, nan=0.0), dtype=float)

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
            logger.debug("supervisory_equation_extract_skipped", error=str(exc))
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
        is_clf: bool,
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
            y_ref = float(cls._predict_scalar_response(estimator, ref_df_one, is_clf)[0])
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
                y_fwd = cls._predict_scalar_response(estimator, X_fwd, is_clf)

                X_bwd = X_sub.copy()
                X_bwd[col] = ref_row[col]
                y_bwd = cls._predict_scalar_response(estimator, X_bwd, is_clf)

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
        model_id: str,
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
            raise ValueError("Incoming operational dataset is empty.")

        catalog = self.catalog_available_models(state)
        if not catalog:
            raise ValueError(
                "No trained model or baseline is available yet. Run Studio 03 (Baselines), Studio 06 (AutoML / Deep Learning), "
                "or click 'Quick-Train Model' inside Studio 14."
            )

        primary_entry = next((m for m in catalog if m["model_id"] == model_id), catalog[0])
        baseline_entry = None
        if baseline_model_id:
            baseline_entry = next((m for m in catalog if m["model_id"] == baseline_model_id), None)
        if baseline_entry is None:
            baseline_entry = next(
                (m for m in catalog if m["family"] == "Baseline" and m["model_id"] != primary_entry["model_id"]),
                next((m for m in catalog if m["family"] == "Baseline"), None),
            )

        ref_df = state.get("uap_df") if isinstance(state.get("uap_df"), pd.DataFrame) else None
        target_name = primary_entry["target"]
        resolved_actual = actual_col if (actual_col and actual_col in incoming_df.columns) else (
            target_name if target_name in incoming_df.columns else None
        )

        # Cap row telemetry table at 5,000 rows while computing full-batch KPIs
        work_df = incoming_df.copy().reset_index(drop=True)
        X_primary = self._model_input(work_df, primary_entry, ref_df)
        preds = self._predict_safe(primary_entry["_estimator"], X_primary)

        if baseline_entry is not None:
            X_base = self._model_input(work_df, baseline_entry, ref_df, primary_entry["feature_cols"])
            base_preds = self._predict_safe(baseline_entry["_estimator"], X_base)
            baseline_label = baseline_entry["label"]
        else:
            # Fallback statistical mean/mode baseline computed from reference dataset
            baseline_label = "Reference Mean/Mode Floor"
            b_val: Any
            if ref_df is not None and target_name in ref_df.columns:
                if pd.api.types.is_numeric_dtype(ref_df[target_name]):
                    b_val = float(pd.to_numeric(ref_df[target_name], errors="coerce").median())
                else:
                    b_val = str(ref_df[target_name].mode().iloc[0])
            else:
                b_val = float(np.nanmedian(preds)) if np.issubdtype(preds.dtype, np.number) else preds[0]
            base_preds = np.full(len(work_df), b_val)

        # Determine classification vs regression mode
        is_clf = False
        if resolved_actual is not None:
            is_clf = _is_classification_target(work_df[resolved_actual])
        elif not np.issubdtype(preds.dtype, np.number):
            is_clf = True
        elif ref_df is not None and target_name in ref_df.columns:
            is_clf = _is_classification_target(ref_df[target_name])

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

        kpis: dict[str, Any] = {
            "n_records": int(len(work_df)),
            "has_actuals": resolved_actual is not None,
            "task_mode": "classification" if is_clf else "regression",
            "model_label": primary_entry["label"],
            "model_family": primary_entry["family"],
            "baseline_label": baseline_label,
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

            # Estimate baseline residual sigma (from training holdout or robust MAD of current batch)
            valid_res = residuals[np.isfinite(residuals)]
            res_std = float(np.std(valid_res, ddof=1)) if len(valid_res) > 2 else 1.0
            mad_sigma = float(1.4826 * np.median(np.abs(valid_res - np.median(valid_res)))) if len(valid_res) > 2 else res_std
            sigma_ref = max(mad_sigma if mad_sigma > 1e-6 else res_std, 1e-6)
            res_z = residuals / sigma_ref

            ucl = float(max_sigma_z * sigma_ref)
            warn_lim = float(warning_fraction * ucl)
            spc_limits = {
                "center": 0.0,
                "ucl": round(ucl, 4),
                "lcl": round(-ucl, 4),
                "warn_upper": round(warn_lim, 4),
                "warn_lower": round(-warn_lim, 4),
                "sigma": round(sigma_ref, 4),
            }

            for i in range(len(work_df)):
                act = float(actuals[i]) if math.isfinite(actuals[i]) else None
                prd = float(p_num[i]) if math.isfinite(p_num[i]) else 0.0
                bpr = float(b_num[i]) if math.isfinite(b_num[i]) else 0.0
                ae = float(abs_errors[i]) if math.isfinite(abs_errors[i]) else 0.0
                pe = float(pct_errors[i]) if math.isfinite(pct_errors[i]) else 0.0
                rz = float(res_z[i]) if math.isfinite(res_z[i]) else 0.0
                rat = float(actual_base_ratio[i]) if math.isfinite(actual_base_ratio[i]) else None
                bae = float(base_abs_errors[i]) if math.isfinite(base_abs_errors[i]) else 0.0

                breach_reasons: list[str] = []
                warn_reasons: list[str] = []

                if pe > max_pct_error:
                    breach_reasons.append(f"%Err {pe:.2f}% > {max_pct_error:.1f}%")
                elif pe > max_pct_error * warning_fraction:
                    warn_reasons.append(f"%Err {pe:.2f}% near limit")

                if max_abs_error is not None and max_abs_error > 0:
                    if ae > max_abs_error:
                        breach_reasons.append(f"|Err| {ae:.3f} > {max_abs_error:.3f}")
                    elif ae > max_abs_error * warning_fraction:
                        warn_reasons.append(f"|Err| {ae:.3f} near limit")

                if abs(rz) > max_sigma_z:
                    breach_reasons.append(f"SPC |Z|={abs(rz):.2f}σ > {max_sigma_z:.1f}σ")
                elif abs(rz) > max_sigma_z * warning_fraction:
                    warn_reasons.append(f"SPC |Z|={abs(rz):.2f}σ warning")

                if rat is not None:
                    if rat < baseline_ratio_min or rat > baseline_ratio_max:
                        breach_reasons.append(f"Act/Base ratio {rat:.3f} outside [{baseline_ratio_min:.2f}, {baseline_ratio_max:.2f}]")

                if breach_reasons:
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
                                "predicted": round(prd, 4),
                                "baseline_predicted": round(bpr, 4),
                                "pct_error": round(pe, 2),
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
                            "model_pred": round(prd, 4),
                            "baseline_pred": round(bpr, 4),
                            "abs_error": round(ae, 4),
                            "pct_error": round(pe, 2),
                            "residual": round(float(residuals[i]), 4) if math.isfinite(residuals[i]) else 0.0,
                            "z_score": round(rz, 2),
                            "actual_vs_baseline_ratio": round(rat, 4) if rat is not None else None,
                            "baseline_error": round(bae, 4),
                            "status": st,
                            "alert_reason": reason_str,
                        }
                    )

            # Aggregate Regression KPIs
            fin = np.isfinite(actuals) & np.isfinite(p_num) & np.isfinite(b_num)
            if fin.sum() >= 2:
                mae = float(mean_absolute_error(actuals[fin], p_num[fin]))
                rmse = float(root_mean_squared_error(actuals[fin], p_num[fin]))
                r2 = float(r2_score(actuals[fin], p_num[fin]))
                mape = float(np.mean(pct_errors[fin]))
                bias = float(np.mean(p_num[fin] - actuals[fin]))
                b_mae = float(mean_absolute_error(actuals[fin], b_num[fin]))
                b_rmse = float(root_mean_squared_error(actuals[fin], b_num[fin]))
                b_mape = float(np.mean((base_abs_errors[fin] / denom[fin]) * 100.0))
                err_reduction_pct = ((b_rmse - rmse) / max(b_rmse, 1e-9)) * 100.0
                fin_rat = actual_base_ratio[np.isfinite(actual_base_ratio)]
                mean_ratio = float(np.mean(fin_rat)) if len(fin_rat) else 1.0
                kpis.update(
                    {
                        "mae": round(mae, 4),
                        "rmse": round(rmse, 4),
                        "mape_pct": round(mape, 2),
                        "r2": round(r2, 4),
                        "mean_bias": round(bias, 4),
                        "baseline_mae": round(b_mae, 4),
                        "baseline_rmse": round(b_rmse, 4),
                        "baseline_mape_pct": round(b_mape, 2),
                        "model_vs_baseline_lift_pct": round(err_reduction_pct, 2),
                        "mean_actual_to_baseline_ratio": round(mean_ratio, 4),
                        "max_pct_error_observed": round(float(np.max(pct_errors[fin])), 2),
                        "max_abs_error_observed": round(float(np.max(abs_errors[fin])), 4),
                    }
                )

            # Subsample charts cleanly to <= 180 points for crisp engineering rendering
            step = max(1, len(work_df) // 180)
            for idx in range(0, len(work_df), step):
                st_i = "BREACH" if (pct_errors[idx] > max_pct_error or abs(res_z[idx]) > max_sigma_z or (max_abs_error and abs_errors[idx] > max_abs_error)) else "IN_SPEC"
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
                        "residual": round(float(residuals[idx]), 4) if math.isfinite(residuals[idx]) else 0.0,
                        "ucl": spc_limits["ucl"],
                        "lcl": spc_limits["lcl"],
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
            actuals_s = work_df[resolved_actual].astype(str).to_numpy()
            preds_s = pd.Series(preds).astype(str).to_numpy()
            base_s = pd.Series(base_preds).astype(str).to_numpy()
            probas = None
            if hasattr(primary_entry["_estimator"], "predict_proba"):
                try:
                    pmat = primary_entry["_estimator"].predict_proba(X_primary)
                    probas = np.max(pmat, axis=1)
                except Exception:
                    probas = None

            for i in range(len(work_df)):
                match_m = actuals_s[i] == preds_s[i]
                match_b = actuals_s[i] == base_s[i]
                conf = float(probas[i] * 100.0) if probas is not None else None
                if not match_m:
                    st = "BREACH"
                    n_breach += 1
                    breach_indices.append(i)
                    reason_str = f"Misclassification: Actual={actuals_s[i]} vs Pred={preds_s[i]}"
                    if len(alarms) < 80:
                        alarms.append(
                            {
                                "_row_idx": i,
                                "row_id": row_ids[i],
                                "severity": "CRITICAL",
                                "actual": actuals_s[i],
                                "predicted": preds_s[i],
                                "baseline_predicted": base_s[i],
                                "pct_error": 100.0,
                                "actual_to_baseline_ratio": 1.0 if match_b else 0.0,
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
                            "actual": actuals_s[i],
                            "model_pred": preds_s[i],
                            "baseline_pred": base_s[i],
                            "confidence_pct": round(conf, 1) if conf is not None else None,
                            "baseline_match": "Match" if match_b else "Miss",
                            "status": st,
                            "alert_reason": reason_str,
                        }
                    )

            acc = float(accuracy_score(actuals_s, preds_s) * 100.0)
            f1 = float(f1_score(actuals_s, preds_s, average="weighted", zero_division=0) * 100.0)
            b_acc = float(accuracy_score(actuals_s, base_s) * 100.0)
            kpis.update(
                {
                    "accuracy_pct": round(acc, 2),
                    "f1_weighted_pct": round(f1, 2),
                    "baseline_accuracy_pct": round(b_acc, 2),
                    "model_vs_baseline_lift_pct": round(acc - b_acc, 2),
                    "min_accuracy_threshold_pct": min_classification_accuracy_pct,
                }
            )
            # Rolling accuracy trend
            win = max(5, len(work_df) // 25)
            matches_arr = (actuals_s == preds_s).astype(float) * 100.0
            b_matches_arr = (actuals_s == base_s).astype(float) * 100.0
            for idx in range(win, len(work_df) + 1, max(1, win // 2)):
                trend_series.append(
                    {
                        "label": row_ids[idx - 1],
                        "actual": min_classification_accuracy_pct,
                        "predicted": round(float(np.mean(matches_arr[max(0, idx - win) : idx])), 2),
                        "baseline": round(float(np.mean(b_matches_arr[max(0, idx - win) : idx])), 2),
                    }
                )
        else:
            # Score-only mode (no actual column in incoming data): compare Model vs Baseline deviation!
            p_num = pd.to_numeric(pd.Series(preds), errors="coerce").to_numpy(dtype=float) if not is_clf else None
            b_num = pd.to_numeric(pd.Series(base_preds), errors="coerce").to_numpy(dtype=float) if not is_clf else None
            for i in range(len(work_df)):
                if not is_clf and p_num is not None and b_num is not None and math.isfinite(p_num[i]) and math.isfinite(b_num[i]):
                    ratio = p_num[i] / b_num[i] if abs(b_num[i]) > 1e-6 else 1.0
                    pct_gap = abs(p_num[i] - b_num[i]) / max(abs(b_num[i]), 1.0) * 100.0
                    if ratio < baseline_ratio_min or ratio > baseline_ratio_max:
                        st = "WARNING"
                        n_warn += 1
                        reason_str = f"Model/Baseline divergence ratio={ratio:.3f} (no observed actual yet)"
                    else:
                        st = "IN_SPEC"
                        n_in_spec += 1
                        reason_str = "Aligned with baseline expectation"
                    if i < 500:
                        rows_out.append(
                            {
                                "row_id": row_ids[i],
                                "actual": None,
                                "model_pred": round(float(p_num[i]), 4),
                                "baseline_pred": round(float(b_num[i]), 4),
                                "model_vs_baseline_gap_pct": round(pct_gap, 2),
                                "model_to_baseline_ratio": round(ratio, 4),
                                "status": st,
                                "alert_reason": reason_str,
                            }
                        )
                    if i % max(1, len(work_df) // 150) == 0:
                        trend_series.append(
                            {
                                "label": row_ids[i],
                                "actual": None,
                                "predicted": round(float(p_num[i]), 4),
                                "baseline": round(float(b_num[i]), 4),
                            }
                        )
                else:
                    n_in_spec += 1
                    if i < 500:
                        rows_out.append(
                            {
                                "row_id": row_ids[i],
                                "actual": None,
                                "model_pred": str(preds[i]),
                                "baseline_pred": str(base_preds[i]),
                                "status": "IN_SPEC",
                                "alert_reason": "Scored (awaiting observed actual)",
                            }
                        )

        breach_rate = round(n_breach / max(len(work_df), 1) * 100.0, 2)
        compliance_rate = round((len(work_df) - n_breach) / max(len(work_df), 1) * 100.0, 2)
        kpis.update(
            {
                "in_spec_count": n_in_spec,
                "warning_count": n_warn,
                "breach_count": n_breach,
                "breach_rate_pct": breach_rate,
                "compliance_rate_pct": compliance_rate,
            }
        )

        # System-Level Health & Executive Alarm Banner
        system_alerts: list[str] = []
        if breach_rate > max_batch_breach_rate_pct:
            system_alerts.append(
                f"CRITICAL TOLERANCE BREACH: {n_breach:,} / {len(work_df):,} records ({breach_rate:.2f}%) exceeded permissible error threshold (batch limit: {max_batch_breach_rate_pct:.1f}%)."
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

        if breach_rate > max_batch_breach_rate_pct or (
            is_clf and resolved_actual and kpis.get("accuracy_pct", 100.0) < min_classification_accuracy_pct
        ):
            system_status = "CRITICAL_ALARM"
        elif n_breach > 0 or n_warn > 0:
            system_status = "WARNING"
        else:
            system_status = "NORMAL"

        # Extract Exact Mathematical Equations for BOTH Champion Model & Reference Baseline Model
        champion_equation = self.extract_model_equation(primary_entry, ref_df=ref_df)
        baseline_equation = self.extract_model_equation(baseline_entry, ref_df=ref_df)
        if baseline_equation is None:
            b_val = float(pd.to_numeric(pd.Series(base_preds), errors="coerce").dropna().mean()) if len(base_preds) else 0.0
            if not math.isfinite(b_val):
                b_val = 0.0
            b_label = "Training Majority-Class Prior Baseline" if is_clf else "Training Historical Mean Baseline"
            safe_t = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in target_name).strip("_") or "target"
            baseline_equation = {
                "model_name": b_label,
                "model_family": "baseline",
                "task_type": "classification" if is_clf else "regression",
                "target_name": target_name,
                "equation_type": "constant_baseline",
                "formula_plain": f"{target_name}_baseline = {b_val:.6g}  (constant historical reference)",
                "formula_latex": rf"\widehat{{\text{{{target_name}}}}}_{{\text{{base}}}} = {b_val:.6g}",
                "formula_python": f"def predict_{safe_t}_baseline(row: dict) -> float:\n    return {b_val:.8f}\n",
                "formula_excel": f"={b_val:.6f}",
                "formula_sql": f"SELECT {b_val:.6f} AS predicted_{safe_t}_baseline FROM your_table;",
                "formula_c_matlab": f"double predicted_{safe_t}_baseline = {b_val:.8f};",
                "external_usage_summary": (
                    f"CONSTANT REFERENCE BASELINE ({b_label}): Represents the historical in-control mean/prior "
                    f"({b_val:.4g}). Run Studio 03 Baselines to upgrade to a full multivariate Ridge/Logistic/GAM reference equation."
                ),
                "intercept": round(b_val, 6),
                "n_features_total": len(primary_entry["feature_cols"]),
                "n_features_active": 0,
                "coefficients_table": [],
                "notes": [
                    "No separate Studio 03 multivariate baseline was selected; using historical training mean/prior reference.",
                    "Run Studio 03 (Universal Diagnostics & Baselines) to benchmark against Ridge, Logistic, or B-Spline GAM baselines.",
                ],
            }

        # Compute Signed Root-Cause Attribution (What Caused Increase (+) or Decrease (-) vs Baseline)
        preds_scalar = self._predict_scalar_response(primary_entry["_estimator"], X_primary, is_clf)
        base_scalar: np.ndarray | None
        if baseline_entry is not None:
            X_base_rc = self._model_input(work_df, baseline_entry, ref_df, primary_entry["feature_cols"])
            base_scalar = self._predict_scalar_response(baseline_entry["_estimator"], X_base_rc, is_clf)
        else:
            b_num_rc = pd.to_numeric(pd.Series(base_preds), errors="coerce").to_numpy(dtype=float)
            base_scalar = np.nan_to_num(b_num_rc, nan=0.0) if not np.isnan(b_num_rc).all() else None
        actuals_scalar = (
            pd.to_numeric(work_df[resolved_actual], errors="coerce").to_numpy(dtype=float)
            if (resolved_actual is not None and not is_clf)
            else None
        )

        root_cause_analysis, per_row_diag = self._analyze_root_causes(
            estimator=primary_entry["_estimator"],
            X_primary=X_primary,
            ref_df=ref_df,
            preds_scalar=preds_scalar,
            base_scalar=base_scalar,
            actuals_scalar=actuals_scalar,
            is_clf=is_clf,
            target_name=target_name,
            breach_indices=breach_indices,
        )

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
                "spc_limits": spc_limits,
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
        hist = state.setdefault("uap_supervisory_history", [])
        hist.append(
            {
                "timestamp": result["timestamp_utc"],
                "source": source_label,
                "model": primary_entry["label"],
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
