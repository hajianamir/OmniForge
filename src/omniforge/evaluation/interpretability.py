"""Model Interpretability Suite: Permutation Importance, KernelSHAP, LIME Local Surrogate, PDP/ICE, Counterfactuals."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from omniforge.core.types import TaskType
from omniforge.models.base import BaseOmniForgeModel


class InterpretabilityEngine:
    """Model-agnostic global and local explainability suite compatible with every OmniForge model."""

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    def _predict_scalar(self, model: BaseOmniForgeModel, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if model.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            proba = model.predict_proba(X)
            return proba[:, 1] if proba.shape[1] == 2 else np.max(proba, axis=1)
        return model.predict(X).astype(float)

    def permutation_importance(
        self,
        model: BaseOmniForgeModel,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        n_repeats: int = 5,
    ) -> dict[str, dict[str, float]]:
        """Compute model-agnostic permutation feature importance with mean and std."""
        rng = np.random.default_rng(self.random_seed)
        baseline_score = model.score(X, y)
        results: dict[str, dict[str, float]] = {}

        for col in X.columns:
            drops: list[float] = []
            for _ in range(max(1, n_repeats)):
                X_perm = X.copy()
                X_perm[col] = rng.permutation(X_perm[col].to_numpy())
                drops.append(baseline_score - model.score(X_perm, y))
            results[str(col)] = {
                "importance_mean": round(float(np.mean(drops)), 6),
                "importance_std": round(float(np.std(drops, ddof=1) if len(drops) > 1 else 0.0), 6),
            }
        return dict(sorted(results.items(), key=lambda kv: kv[1]["importance_mean"], reverse=True))

    def explain_shap_local(
        self,
        model: BaseOmniForgeModel,
        X_background: pd.DataFrame,
        instance: pd.DataFrame | pd.Series,
        *,
        n_samples: int = 64,
    ) -> dict[str, Any]:
        """Estimate local Shapley feature attributions via Monte Carlo permutation sampling."""
        rng = np.random.default_rng(self.random_seed)
        x_row = instance.to_frame().T if isinstance(instance, pd.Series) else instance.iloc[[0]].copy()
        cols = list(X_background.columns)
        n_feats = len(cols)
        bg = X_background.sample(n=min(len(X_background), 50), random_state=self.random_seed).reset_index(drop=True)
        pred_val = float(self._predict_scalar(model, x_row)[0])

        shap_vals = np.zeros(n_feats, dtype=float)
        if n_feats == 0 or n_samples <= 0 or len(bg) == 0:
            return {
                "method": "monte_carlo_shap",
                "expected_value": round(pred_val, 6),
                "prediction": round(pred_val, 6),
                "attributions": {},
            }

        bg_idx = rng.integers(0, len(bg), size=n_samples)
        perms = [rng.permutation(n_feats) for _ in range(n_samples)]

        # Vectorized coalition batch: evaluate all n_samples * (n_feats + 1) rows in one model call
        n_steps = n_feats + 1
        batch_df = bg.iloc[np.repeat(bg_idx, n_steps)].reset_index(drop=True).copy()
        use_instance = np.zeros((n_samples, n_steps, n_feats), dtype=bool)
        for s, perm in enumerate(perms):
            for step, feat_i in enumerate(perm):
                use_instance[s, step + 1 :, int(feat_i)] = True
        mask_2d = use_instance.reshape(n_samples * n_steps, n_feats)
        for j, col in enumerate(cols):
            col_mask = mask_2d[:, j]
            if np.any(col_mask):
                inst_val = x_row[col].iloc[0]
                if (
                    isinstance(batch_df[col].dtype, pd.CategoricalDtype)
                    and pd.notna(inst_val)
                    and inst_val not in batch_df[col].cat.categories
                ):
                    batch_df[col] = batch_df[col].cat.add_categories([inst_val])
                batch_df.loc[col_mask, col] = inst_val

        all_preds = np.asarray(self._predict_scalar(model, batch_df), dtype=float).reshape(n_samples, n_steps)
        expected_val = float(np.mean(all_preds[:, 0]))
        diffs = np.diff(all_preds, axis=1)
        for s, perm in enumerate(perms):
            shap_vals[perm] += diffs[s]

        shap_vals /= max(n_samples, 1)
        attributions = {
            str(col): round(float(val), 6)
            for col, val in sorted(zip(cols, shap_vals, strict=False), key=lambda kv: abs(kv[1]), reverse=True)
        }
        return {
            "method": "monte_carlo_shap",
            "expected_value": round(expected_val, 6),
            "prediction": round(pred_val, 6),
            "attributions": attributions,
        }

    def explain_lime_local(
        self,
        model: BaseOmniForgeModel,
        X_background: pd.DataFrame,
        instance: pd.DataFrame | pd.Series,
        *,
        n_perturbations: int = 200,
        kernel_width: float | None = None,
        ridge_alpha: float = 1.0,
    ) -> dict[str, Any]:
        """Fit a scale-invariant LIME local linear surrogate (Ribeiro et al. 2016).

        Perturbs numeric features in standardized z-score units around ``instance``,
        weights perturbations by ``exp(-||z||_2^2 / sigma^2)`` with default Ribeiro
        kernel width ``sigma = 0.75 * sqrt(p)``, and fits weighted Ridge strictly in
        standardized coordinates so regularization is invariant to feature units.
        """
        rng = np.random.default_rng(self.random_seed)
        x_row = instance.to_frame().T if isinstance(instance, pd.Series) else instance.iloc[[0]].copy()
        num_cols = list(X_background.select_dtypes(include=[np.number]).columns)
        pred_x0 = float(self._predict_scalar(model, x_row)[0])
        if not num_cols:
            return {
                "method": "lime_local_surrogate",
                "prediction": round(pred_x0, 6),
                "local_fidelity_r2": 0.0,
                "kernel_width": 0.0,
                "feature_weights": {},
            }

        p = len(num_cols)
        n_pts = max(10, int(n_perturbations))
        stds = X_background[num_cols].std(ddof=1).replace(0.0, 1.0).fillna(1.0)
        std_vec = stds.to_numpy(dtype=float)

        # Anchor exact instance at row 0 (z = 0, distance = 0, weight = 1) per marcotcr/lime
        z_local = rng.normal(0.0, 0.35, size=(n_pts, p))
        z_local[0, :] = 0.0

        perturbed = pd.concat([x_row] * n_pts, ignore_index=True)
        for idx, col in enumerate(num_cols):
            perturbed[col] = float(x_row[col].iloc[0]) + z_local[:, idx] * float(std_vec[idx])

        preds = self._predict_scalar(model, perturbed)

        # Ribeiro et al. (2016) lime_tabular default kernel width: 0.75 * sqrt(p) on Euclidean distance
        kw = float(kernel_width) if (kernel_width is not None and kernel_width > 0) else (0.75 * float(np.sqrt(p)))
        distances = np.linalg.norm(z_local, axis=1)
        kernel_w = np.sqrt(np.exp(-(distances**2) / max(kw * kw, 1e-12)))

        # Fit weighted Ridge in standardized z-space so L2 penalty is strictly scale-invariant
        surrogate = Ridge(alpha=max(1e-6, float(ridge_alpha)))
        surrogate.fit(z_local, preds, sample_weight=kernel_w)
        r2_local = float(surrogate.score(z_local, preds, sample_weight=kernel_w))

        # surrogate.coef_ is already the standardized attribution (change in prediction per +1 SD of feature)
        weights = {
            str(col): round(float(coef), 6)
            for col, coef in sorted(
                zip(num_cols, surrogate.coef_, strict=False), key=lambda kv: abs(kv[1]), reverse=True
            )
        }
        return {
            "method": "lime_local_surrogate",
            "prediction": round(pred_x0, 6),
            "local_fidelity_r2": round(r2_local, 4),
            "kernel_width": round(kw, 4),
            "feature_weights": weights,
        }

    def partial_dependence(
        self,
        model: BaseOmniForgeModel,
        X: pd.DataFrame,
        feature: str,
        *,
        grid_resolution: int = 15,
    ) -> dict[str, Any]:
        """Compute Partial Dependence (PDP) and Individual Conditional Expectation (ICE) profiles."""
        sample_df = X.head(min(len(X), 200)).copy()
        vals = pd.to_numeric(sample_df[feature], errors="coerce").dropna()
        if len(vals) == 0 or len(sample_df) == 0:
            return {"feature": feature, "grid": [], "pdp_mean": [], "pdp_std": []}
        grid = np.linspace(float(vals.quantile(0.05)), float(vals.quantile(0.95)), max(1, int(grid_resolution)))
        n_rows = len(sample_df)
        stacked = pd.concat([sample_df] * len(grid), ignore_index=True)
        stacked[feature] = np.repeat(grid, n_rows)
        all_preds = np.asarray(self._predict_scalar(model, stacked), dtype=float).reshape(len(grid), n_rows)
        pdp_means = [round(float(v), 6) for v in np.mean(all_preds, axis=1)]
        pdp_stds = [round(float(v), 6) for v in np.std(all_preds, axis=1)]
        return {
            "feature": feature,
            "grid": [round(float(v), 6) for v in grid],
            "pdp_mean": pdp_means,
            "pdp_std": pdp_stds,
        }

    def generate_counterfactual(
        self,
        model: BaseOmniForgeModel,
        X_background: pd.DataFrame,
        instance: pd.DataFrame | pd.Series,
        *,
        desired_class: Any,
        n_candidates: int = 400,
        immutable_features: list[str] | None = None,
        feature_bounds: dict[str, tuple[float, float]] | None = None,
        respect_domain_bounds: bool = True,
    ) -> dict[str, Any]:
        """Search for the nearest plausible counterfactual record achieving ``desired_class``.

        Minimizes the Wachter, Mittelstadt & Russell (2017, *Harvard J. Law & Tech.* 31(2), Eq. 4)
        and Mothilal, Sharma & Tan (2020, DiCE) inverse-MAD normalized :math:`L_1` distance
        subject to actionability (``immutable_features``) and domain box constraints.
        """
        rng = np.random.default_rng(self.random_seed)
        x_row = instance.to_frame().T if isinstance(instance, pd.Series) else instance.iloc[[0]].copy()
        frozen = set(immutable_features or [])
        num_cols = [c for c in X_background.select_dtypes(include=[np.number]).columns if c not in frozen]
        if not num_cols:
            return {"found": False, "desired_class": desired_class, "changes": {}, "distance_metric": "wachter_l1_mad"}

        bg_num = X_background[num_cols].apply(pd.to_numeric, errors="coerce")
        stds = bg_num.std(ddof=1).replace(0.0, 1.0).fillna(1.0).to_numpy(dtype=float)
        medians = bg_num.median().fillna(0.0).to_numpy(dtype=float)
        raw_mad = np.nanmedian(np.abs(bg_num.to_numpy(dtype=float) - medians[None, :]), axis=0)
        mads = np.where(np.isfinite(raw_mad) & (raw_mad > 1e-9), 1.4826 * raw_mad, stds)
        mads = np.where(mads > 1e-9, mads, 1.0)

        lows = np.full(len(num_cols), -np.inf, dtype=float)
        highs = np.full(len(num_cols), np.inf, dtype=float)
        if respect_domain_bounds:
            emp_min = bg_num.min().to_numpy(dtype=float)
            emp_max = bg_num.max().to_numpy(dtype=float)
            lows = np.where(np.isfinite(emp_min), emp_min, lows)
            highs = np.where(np.isfinite(emp_max), emp_max, highs)
        if feature_bounds:
            for idx_c, col in enumerate(num_cols):
                if col in feature_bounds:
                    lo_b, hi_b = feature_bounds[col]
                    lows[idx_c] = float(lo_b)
                    highs[idx_c] = float(hi_b)

        x0 = x_row[num_cols].to_numpy(dtype=float).ravel()
        cands = pd.concat([x_row] * n_candidates, ignore_index=True)
        scales = np.linspace(0.05, 3.0, n_candidates)[:, None]
        raw_deltas = rng.normal(0.0, 1.0, size=(n_candidates, len(num_cols))) * scales * stds[None, :]
        # Encourage sparse feature edits on the first half of candidates (DiCE sparsity prior)
        if len(num_cols) > 1 and n_candidates >= 4:
            half = n_candidates // 2
            sparse_mask = rng.uniform(0.0, 1.0, size=(half, len(num_cols))) < min(1.0, 2.0 / len(num_cols))
            empty_rows = ~np.any(sparse_mask, axis=1)
            if np.any(empty_rows):
                chosen_cols = rng.integers(0, len(num_cols), size=int(np.sum(empty_rows)))
                sparse_mask[empty_rows, chosen_cols] = True
            raw_deltas[:half] *= sparse_mask.astype(float)

        cand_vals = np.clip(x0[None, :] + raw_deltas, lows[None, :], highs[None, :])
        eff_deltas = cand_vals - x0[None, :]
        cands[num_cols] = cand_vals

        preds = model.predict(cands)
        matched = np.where(preds == desired_class)[0]
        if len(matched) == 0:
            return {"found": False, "desired_class": desired_class, "changes": {}, "distance_metric": "wachter_l1_mad"}

        wachter_l1 = np.sum(np.abs(eff_deltas[matched]) / mads[None, :], axis=1)
        sparsity = np.sum(np.abs(eff_deltas[matched]) > 1e-4, axis=1)
        # Lexicographic tie-breaking favoring sparser counterfactuals at equal Wachter L1 distance
        score = wachter_l1 + 0.01 * sparsity
        best_local = int(np.argmin(score))
        best_idx = int(matched[best_local])
        best_dist = float(wachter_l1[best_local])
        changes = {
            col: {
                "from": round(float(x_row[col].iloc[0]), 6),
                "to": round(float(cands[col].iloc[best_idx]), 6),
                "delta": round(float(cands[col].iloc[best_idx] - x_row[col].iloc[0]), 6),
            }
            for col in num_cols
            if abs(float(cands[col].iloc[best_idx] - x_row[col].iloc[0])) > 1e-4
        }
        return {
            "found": True,
            "desired_class": desired_class,
            "normalized_distance": round(best_dist, 4),
            "distance_metric": "wachter_l1_mad",
            "sparsity_changed_features": len(changes),
            "changes": changes,
        }
