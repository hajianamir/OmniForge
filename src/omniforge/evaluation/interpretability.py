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
        bg_idx = rng.integers(0, len(bg), size=n_samples)
        sampled_base_preds: list[float] = []

        for s in range(n_samples):
            perm = rng.permutation(n_feats)
            base_row = bg.iloc[[int(bg_idx[s])]].copy()
            prev_pred = float(self._predict_scalar(model, base_row)[0])
            sampled_base_preds.append(prev_pred)
            for feat_i in perm:
                col = cols[feat_i]
                base_row[col] = x_row[col].values[0]
                new_pred = float(self._predict_scalar(model, base_row)[0])
                shap_vals[feat_i] += new_pred - prev_pred
                prev_pred = new_pred

        expected_val = float(np.mean(sampled_base_preds)) if sampled_base_preds else 0.0
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
        grid = np.linspace(float(vals.quantile(0.05)), float(vals.quantile(0.95)), grid_resolution)
        pdp_means: list[float] = []
        pdp_stds: list[float] = []
        for g_val in grid:
            mod = sample_df.copy()
            mod[feature] = g_val
            preds = self._predict_scalar(model, mod)
            pdp_means.append(round(float(np.mean(preds)), 6))
            pdp_stds.append(round(float(np.std(preds)), 6))
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
    ) -> dict[str, Any]:
        """Search for the nearest plausible counterfactual record achieving ``desired_class``."""
        rng = np.random.default_rng(self.random_seed)
        x_row = instance.to_frame().T if isinstance(instance, pd.Series) else instance.iloc[[0]].copy()
        num_cols = list(X_background.select_dtypes(include=[np.number]).columns)
        stds = X_background[num_cols].std().replace(0.0, 1.0).fillna(1.0).to_numpy(dtype=float)

        cands = pd.concat([x_row] * n_candidates, ignore_index=True)
        scales = np.linspace(0.1, 2.5, n_candidates)[:, None]
        deltas = rng.normal(0.0, 1.0, size=(n_candidates, len(num_cols))) * scales * stds[None, :]
        cands[num_cols] = cands[num_cols].to_numpy(dtype=float) + deltas

        preds = model.predict(cands)
        matched = np.where(preds == desired_class)[0]
        if len(matched) == 0:
            return {"found": False, "desired_class": desired_class, "changes": {}}

        dists = np.linalg.norm(deltas[matched] / stds[None, :], axis=1)
        best_idx = int(matched[int(np.argmin(dists))])
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
            "normalized_distance": round(float(np.min(dists)), 4),
            "changes": changes,
        }
