"""Enterprise Causal Inference Engine: IPW, AIPW, PSM, DML, IV-2SLS, DiD, CATE Uplift, and Refutation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

from omniforge.core.exceptions import CausalInferenceError
from omniforge.core.logging import get_logger

logger = get_logger(__name__)


class CausalEffectResult(BaseModel):
    """Structured result of causal treatment effect estimation and refutation tests."""

    method: str
    ate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_treated: int
    n_control: int
    cate_summary: dict[str, float] = Field(default_factory=dict)
    refutations: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


class CausalInferenceEngine:
    """Comprehensive causal effect estimation, CATE uplift modeling, and refutation suite."""

    def __init__(self, random_seed: int = 42, bootstrap_rounds: int = 40) -> None:
        self.random_seed = random_seed
        self.bootstrap_rounds = max(10, int(bootstrap_rounds))

    def estimate_ate(
        self,
        df: pd.DataFrame,
        *,
        treatment_col: str,
        outcome_col: str,
        confounders: list[str],
        method: str = "aipw",
        instrument_col: str | None = None,
        run_refutations: bool = True,
    ) -> CausalEffectResult:
        """Estimate Average Treatment Effect (ATE) and optionally run DoWhy-style refutation checks.

        Supported methods: ``aipw`` (Doubly Robust), ``ipw``, ``psm``, ``dml``, ``iv_2sls``,
        ``t_learner``, ``s_learner``, ``x_learner``.
        """
        required = [treatment_col, outcome_col, *confounders]
        if instrument_col:
            required.append(instrument_col)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise CausalInferenceError(f"Missing required columns for causal estimation: {missing}")

        work = df[required].dropna().copy()
        if len(work) < 20:
            raise CausalInferenceError("At least 20 complete observations are required for causal inference.")

        X = work[confounders].to_numpy(dtype=float)
        T = work[treatment_col].to_numpy(dtype=float)
        Y = work[outcome_col].to_numpy(dtype=float)
        Z = work[instrument_col].to_numpy(dtype=float) if instrument_col else None

        n_treated = int(np.sum(T == 1))
        n_control = int(np.sum(T == 0))

        est_method = method.lower()
        ate, cate, extra = self._dispatch_estimator(est_method, X, T, Y, Z)

        # Non-parametric bootstrap standard error & 95% confidence interval
        rng = np.random.default_rng(self.random_seed)
        boot_ates: list[float] = []
        n = len(Y)
        for _ in range(self.bootstrap_rounds):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(T[idx])) < 2:
                continue
            b_ate, _, _ = self._dispatch_estimator(
                est_method, X[idx], T[idx], Y[idx], Z[idx] if Z is not None else None
            )
            boot_ates.append(b_ate)

        se = float(np.std(boot_ates, ddof=1)) if len(boot_ates) >= 2 else 0.0
        ci_low = float(ate - 1.96 * se)
        ci_high = float(ate + 1.96 * se)
        from scipy.stats import norm

        p_val = float(2.0 * (1.0 - norm.cdf(abs(ate) / max(se, 1e-12)))) if se > 0 else 0.0

        cate_summary: dict[str, float] = {}
        if cate is not None and len(cate) > 0:
            cate_summary = {
                "mean": round(float(np.mean(cate)), 6),
                "std": round(float(np.std(cate)), 6),
                "min": round(float(np.min(cate)), 6),
                "p25": round(float(np.quantile(cate, 0.25)), 6),
                "median": round(float(np.median(cate)), 6),
                "p75": round(float(np.quantile(cate, 0.75)), 6),
                "max": round(float(np.max(cate)), 6),
            }

        refutations: dict[str, Any] = {}
        if run_refutations:
            refutations = self._run_refutations(est_method, X, T, Y, Z, ate)

        logger.info("causal_ate_estimated", method=est_method, ate=round(ate, 4), p_value=round(p_val, 6))
        return CausalEffectResult(
            method=est_method,
            ate=round(float(ate), 6),
            std_error=round(se, 6),
            ci_lower=round(ci_low, 6),
            ci_upper=round(ci_high, 6),
            p_value=round(p_val, 6),
            n_treated=n_treated,
            n_control=n_control,
            cate_summary=cate_summary,
            refutations=refutations,
            details=extra,
        )

    def difference_in_differences(
        self,
        df: pd.DataFrame,
        *,
        group_col: str,
        post_col: str,
        outcome_col: str,
    ) -> dict[str, float]:
        """Compute canonical 2x2 Difference-in-Differences (DiD) treatment effect."""
        for c in (group_col, post_col, outcome_col):
            if c not in df.columns:
                raise CausalInferenceError(f"Column '{c}' missing for DiD estimation.")
        g1_post = float(df.loc[(df[group_col] == 1) & (df[post_col] == 1), outcome_col].mean())
        g1_pre = float(df.loc[(df[group_col] == 1) & (df[post_col] == 0), outcome_col].mean())
        g0_post = float(df.loc[(df[group_col] == 0) & (df[post_col] == 1), outcome_col].mean())
        g0_pre = float(df.loc[(df[group_col] == 0) & (df[post_col] == 0), outcome_col].mean())
        did = (g1_post - g1_pre) - (g0_post - g0_pre)
        return {
            "did_estimate": round(did, 6),
            "treated_change": round(g1_post - g1_pre, 6),
            "control_change": round(g0_post - g0_pre, 6),
            "treated_pre": round(g1_pre, 6),
            "treated_post": round(g1_post, 6),
            "control_pre": round(g0_pre, 6),
            "control_post": round(g0_post, 6),
        }

    def _dispatch_estimator(
        self,
        method: str,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray | None,
    ) -> tuple[float, np.ndarray | None, dict[str, Any]]:
        if method == "ipw":
            ps = self._fit_propensity(X, T)
            ate = float(np.mean(T * Y / ps - (1.0 - T) * Y / (1.0 - ps)))
            return ate, None, {"mean_propensity": round(float(np.mean(ps)), 4)}

        if method == "psm":
            ps = self._fit_propensity(X, T).reshape(-1, 1)
            t_idx = np.where(T == 1)[0]
            c_idx = np.where(T == 0)[0]
            nn = NearestNeighbors(n_neighbors=1).fit(ps[c_idx])
            _, matched = nn.kneighbors(ps[t_idx])
            att = float(np.mean(Y[t_idx] - Y[c_idx[matched.ravel()]]))
            nn_rev = NearestNeighbors(n_neighbors=1).fit(ps[t_idx])
            _, matched_rev = nn_rev.kneighbors(ps[c_idx])
            atc = float(np.mean(Y[t_idx[matched_rev.ravel()]] - Y[c_idx]))
            ate = (len(t_idx) * att + len(c_idx) * atc) / len(Y)
            return ate, None, {"att": round(att, 6), "atc": round(atc, 6)}

        if method == "dml":
            # Cross-fitted Robinson partialling-out Double Machine Learning
            kf = KFold(n_splits=min(5, max(2, len(Y) // 10)), shuffle=True, random_state=self.random_seed)
            y_res = np.zeros_like(Y)
            t_res = np.zeros_like(T)
            for tr, va in kf.split(X):
                m_y = GradientBoostingRegressor(max_depth=3, n_estimators=60, random_state=self.random_seed).fit(X[tr], Y[tr])
                m_t = GradientBoostingRegressor(max_depth=3, n_estimators=60, random_state=self.random_seed).fit(X[tr], T[tr])
                y_res[va] = Y[va] - m_y.predict(X[va])
                t_res[va] = T[va] - m_t.predict(X[va])
            denom = float(np.dot(t_res, t_res))
            theta = float(np.dot(t_res, y_res) / max(denom, 1e-12))
            return theta, None, {"orthogonal_score_r2": round(float(1.0 - np.var(y_res - theta * t_res) / max(np.var(y_res), 1e-12)), 4)}

        if method == "iv_2sls":
            if Z is None:
                raise CausalInferenceError("Method 'iv_2sls' requires 'instrument_col'.")
            X_stage1 = np.column_stack([np.ones(len(X)), X, Z])
            beta1, *_ = np.linalg.lstsq(X_stage1, T, rcond=None)
            t_hat = X_stage1 @ beta1
            X_stage2 = np.column_stack([np.ones(len(X)), X, t_hat])
            beta2, *_ = np.linalg.lstsq(X_stage2, Y, rcond=None)
            f_stat_corr = float(np.corrcoef(Z, T)[0, 1]) ** 2
            return float(beta2[-1]), None, {"first_stage_r2_instrument": round(f_stat_corr, 4)}

        if method == "s_learner":
            reg = RandomForestRegressor(n_estimators=80, max_depth=6, random_state=self.random_seed)
            reg.fit(np.column_stack([X, T]), Y)
            y1 = reg.predict(np.column_stack([X, np.ones(len(X))]))
            y0 = reg.predict(np.column_stack([X, np.zeros(len(X))]))
            cate = y1 - y0
            return float(np.mean(cate)), cate, {}

        if method == "t_learner":
            m1 = Ridge(alpha=1.0).fit(X[T == 1], Y[T == 1])
            m0 = Ridge(alpha=1.0).fit(X[T == 0], Y[T == 0])
            cate = m1.predict(X) - m0.predict(X)
            return float(np.mean(cate)), cate, {}

        if method == "x_learner":
            ps = self._fit_propensity(X, T)
            m1 = Ridge(alpha=1.0).fit(X[T == 1], Y[T == 1])
            m0 = Ridge(alpha=1.0).fit(X[T == 0], Y[T == 0])
            d1 = Y[T == 1] - m0.predict(X[T == 1])
            d0 = m1.predict(X[T == 0]) - Y[T == 0]
            tau1 = Ridge(alpha=1.0).fit(X[T == 1], d1).predict(X)
            tau0 = Ridge(alpha=1.0).fit(X[T == 0], d0).predict(X)
            cate = ps * tau0 + (1.0 - ps) * tau1
            return float(np.mean(cate)), cate, {}

        # Default: Doubly Robust Augmented IPW (AIPW)
        ps = self._fit_propensity(X, T)
        m1 = Ridge(alpha=1.0).fit(X[T == 1], Y[T == 1]).predict(X)
        m0 = Ridge(alpha=1.0).fit(X[T == 0], Y[T == 0]).predict(X)
        psi = (m1 - m0) + T * (Y - m1) / ps - (1.0 - T) * (Y - m0) / (1.0 - ps)
        return float(np.mean(psi)), (m1 - m0), {"mean_propensity": round(float(np.mean(ps)), 4)}

    def _fit_propensity(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        clf = LogisticRegression(C=1.0, max_iter=500, random_state=self.random_seed)
        clf.fit(X, T.astype(int))
        return np.clip(clf.predict_proba(X)[:, 1], 0.05, 0.95)

    def _run_refutations(
        self,
        method: str,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray | None,
        original_ate: float,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(self.random_seed + 101)
        # 1. Placebo treatment test (permuted treatment should give ATE ~ 0)
        T_placebo = rng.permutation(T)
        placebo_ate, _, _ = self._dispatch_estimator(method, X, T_placebo, Y, Z)

        # 2. Random common cause (adding noise confounder should not materially shift ATE)
        X_aug = np.column_stack([X, rng.normal(0.0, 1.0, size=len(X))])
        rcc_ate, _, _ = self._dispatch_estimator(method, X_aug, T, Y, Z)

        # 3. 80% data subset stability check
        sub_idx = rng.choice(len(Y), size=int(len(Y) * 0.8), replace=False)
        sub_ate, _, _ = self._dispatch_estimator(
            method, X[sub_idx], T[sub_idx], Y[sub_idx], Z[sub_idx] if Z is not None else None
        )

        return {
            "placebo_treatment": {
                "estimated_ate": round(float(placebo_ate), 6),
                "passed": abs(placebo_ate) < max(abs(original_ate) * 0.35, 0.5),
            },
            "random_common_cause": {
                "estimated_ate": round(float(rcc_ate), 6),
                "delta": round(float(abs(rcc_ate - original_ate)), 6),
                "passed": abs(rcc_ate - original_ate) < max(abs(original_ate) * 0.15, 0.25),
            },
            "data_subset_80pct": {
                "estimated_ate": round(float(sub_ate), 6),
                "delta": round(float(abs(sub_ate - original_ate)), 6),
                "passed": abs(sub_ate - original_ate) < max(abs(original_ate) * 0.25, 0.5),
            },
        }
