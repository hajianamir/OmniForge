"""Enterprise Causal Inference Engine: IPW, AIPW, PSM, DML, IV-2SLS, DiD, CATE Uplift, and Refutation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
        ate, cate, extra, analytic_se = self._dispatch_estimator(est_method, X, T, Y, Z)

        # Use theoretically valid closed-form variance estimators when available:
        # - PSM: Abadie & Imbens (2006/2008) analytical matching variance (bootstrap is formally inconsistent for NN matching)
        # - AIPW: Robins-Rotnitzky-Zhao (1994) Efficient Influence Function (EIF) SE with cross-fitting
        # - DML: Chernozhukov et al. (2018) Neyman-orthogonal HC1 score SE
        # - IV-2SLS: HC1 heteroskedasticity-robust structural-residual 2SLS SE
        if analytic_se is not None:
            se = float(analytic_se)
        else:
            # Non-parametric bootstrap SE for IPW / meta-learners
            rng = np.random.default_rng(self.random_seed)
            boot_ates: list[float] = []
            n = len(Y)
            for _ in range(self.bootstrap_rounds):
                idx = rng.integers(0, n, size=n)
                if len(np.unique(T[idx])) < 2:
                    continue
                b_ate, _, _, _ = self._dispatch_estimator(
                    est_method, X[idx], T[idx], Y[idx], Z[idx] if Z is not None else None
                )
                if np.isfinite(b_ate):
                    boot_ates.append(b_ate)
            se = float(np.std(boot_ates, ddof=1)) if len(boot_ates) >= 2 else 0.0

        z_crit = float(norm.ppf(0.975))
        ci_low = float(ate - z_crit * se)
        ci_high = float(ate + z_crit * se)
        if se > 1e-12:
            p_val = float(2.0 * norm.sf(abs(ate) / se))
        else:
            p_val = 1.0 if abs(ate) <= 1e-12 else 0.0

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
        """Compute canonical 2x2 Difference-in-Differences (DiD) with Neyman heteroskedastic SE and p-value."""
        for c in (group_col, post_col, outcome_col):
            if c not in df.columns:
                raise CausalInferenceError(f"Column '{c}' missing for DiD estimation.")
        y11 = df.loc[(df[group_col] == 1) & (df[post_col] == 1), outcome_col].dropna().to_numpy(dtype=float)
        y10 = df.loc[(df[group_col] == 1) & (df[post_col] == 0), outcome_col].dropna().to_numpy(dtype=float)
        y01 = df.loc[(df[group_col] == 0) & (df[post_col] == 1), outcome_col].dropna().to_numpy(dtype=float)
        y00 = df.loc[(df[group_col] == 0) & (df[post_col] == 0), outcome_col].dropna().to_numpy(dtype=float)
        if min(len(y11), len(y10), len(y01), len(y00)) < 1:
            raise CausalInferenceError("All four 2x2 group x post cells must contain at least one observation.")
        g1_post, g1_pre = float(np.mean(y11)), float(np.mean(y10))
        g0_post, g0_pre = float(np.mean(y01)), float(np.mean(y00))
        did = (g1_post - g1_pre) - (g0_post - g0_pre)
        def _var_cell(arr: np.ndarray) -> float:
            return float(np.var(arr, ddof=1) / len(arr)) if len(arr) > 1 else 0.0

        se = float(np.sqrt(_var_cell(y11) + _var_cell(y10) + _var_cell(y01) + _var_cell(y00)))
        z_crit = float(norm.ppf(0.975))
        p_val = float(2.0 * norm.sf(abs(did) / se)) if se > 1e-12 else (1.0 if abs(did) <= 1e-12 else 0.0)
        return {
            "did_estimate": round(did, 6),
            "std_error": round(se, 6),
            "ci_lower": round(did - z_crit * se, 6),
            "ci_upper": round(did + z_crit * se, 6),
            "p_value": round(p_val, 6),
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
    ) -> tuple[float, np.ndarray | None, dict[str, Any], float | None]:
        if method == "ipw":
            ps_raw, ps = self._fit_propensity_with_raw(X, T)
            w1 = T / ps
            w0 = (1.0 - T) / (1.0 - ps)
            sum_w1 = max(float(np.sum(w1)), 1e-12)
            sum_w0 = max(float(np.sum(w0)), 1e-12)
            # Hájek (1971) normalized / stabilized IPW: exact shift invariance and markedly lower variance than HT
            mu1_hajek = float(np.sum(w1 * Y) / sum_w1)
            mu0_hajek = float(np.sum(w0 * Y) / sum_w0)
            ate_hajek = mu1_hajek - mu0_hajek
            ate_ht = float(np.mean(w1 * Y - w0 * Y))
            ess1 = float((sum_w1**2) / max(float(np.sum(w1**2)), 1e-12))
            ess0 = float((sum_w0**2) / max(float(np.sum(w0**2)), 1e-12))
            trimmed = float(np.mean((ps_raw < 0.05) | (ps_raw > 0.95)))
            return (
                ate_hajek,
                None,
                {
                    "ipw_variant": "hajek_stabilized",
                    "horvitz_thompson_ate": round(ate_ht, 6),
                    "mean_propensity": round(float(np.mean(ps)), 4),
                    "effective_sample_size_treated": round(ess1, 2),
                    "effective_sample_size_control": round(ess0, 2),
                    "propensity_trimmed_share": round(trimmed, 4),
                },
                None,
            )

        if method == "psm":
            ps = self._fit_propensity(X, T).reshape(-1, 1)
            n = len(Y)
            t_idx = np.where(T == 1)[0]
            c_idx = np.where(T == 0)[0]
            if len(t_idx) < 2 or len(c_idx) < 2:
                raise CausalInferenceError("PSM requires at least 2 treated and 2 control observations.")

            nn_c = NearestNeighbors(n_neighbors=1).fit(ps[c_idx])
            _, m_c = nn_c.kneighbors(ps[t_idx])
            matched_c_global = c_idx[m_c.ravel()]
            tau_treated = Y[t_idx] - Y[matched_c_global]
            att = float(np.mean(tau_treated))

            nn_t = NearestNeighbors(n_neighbors=1).fit(ps[t_idx])
            _, m_t = nn_t.kneighbors(ps[c_idx])
            matched_t_global = t_idx[m_t.ravel()]
            tau_control = Y[matched_t_global] - Y[c_idx]
            atc = float(np.mean(tau_control))

            tau_i = np.empty(n, dtype=float)
            tau_i[t_idx] = tau_treated
            tau_i[c_idx] = tau_control
            ate = float(np.mean(tau_i))

            # Abadie & Imbens (2006, Econometrica 74:235-267) analytical variance for M=1 matching with replacement
            # Match multiplicity K_M(i): number of times unit i serves as a match for opposite-arm units
            k_m = np.zeros(n, dtype=float)
            np.add.at(k_m, matched_c_global, 1.0)
            np.add.at(k_m, matched_t_global, 1.0)

            # Conditional outcome variance sigma^2(X_i, W_i) via J same-arm nearest neighbors (Eq. 14)
            sigma2_hat = np.zeros(n, dtype=float)
            for arm_idx in (t_idx, c_idx):
                j_nn = min(2, len(arm_idx) - 1)
                if j_nn >= 1:
                    nn_same = NearestNeighbors(n_neighbors=j_nn + 1).fit(ps[arm_idx])
                    _, idx_same = nn_same.kneighbors(ps[arm_idx])
                    # Exclude self (column 0 is usually self, but guard against exact propensity ties)
                    for pos, global_i in enumerate(arm_idx):
                        nbrs = [arm_idx[k] for k in idx_same[pos] if arm_idx[k] != global_i][:j_nn]
                        if nbrs:
                            y_bar_nbr = float(np.mean(Y[nbrs]))
                            j_eff = float(len(nbrs))
                            sigma2_hat[global_i] = (j_eff / (j_eff + 1.0)) * ((Y[global_i] - y_bar_nbr) ** 2)

            # Population ATE variance (Abadie & Imbens 2006, M=1):
            # V_PATE = (1/N^2) sum [ (tau_i - ate)^2 + (K_1(i)^2 + K_1(i)) * sigma^2_i ]
            # Floor at conditional SATE variance (1/N^2) sum (1 + K_1(i))^2 * sigma^2_i when small-sample heterogeneity cancels
            var_pate = float(np.sum((tau_i - ate) ** 2 + (k_m * k_m + k_m) * sigma2_hat) / (n * n))
            var_sate = float(np.sum(((1.0 + k_m) ** 2) * sigma2_hat) / (n * n))
            ai_se = float(np.sqrt(max(var_pate, var_sate, 0.0)))
            return (
                ate,
                tau_i,
                {
                    "att": round(att, 6),
                    "atc": round(atc, 6),
                    "se_method": "abadie_imbens_2006_analytical",
                    "max_match_multiplicity": int(np.max(k_m)),
                    "sate_std_error": round(float(np.sqrt(max(var_sate, 0.0))), 6),
                },
                ai_se,
            )

        if method == "dml":
            # Cross-fitted Robinson partialling-out Double Machine Learning (Chernozhukov et al. 2018)
            n = len(Y)
            kf = KFold(n_splits=min(5, max(2, n // 10)), shuffle=True, random_state=self.random_seed)
            y_res = np.zeros_like(Y)
            t_res = np.zeros_like(T)
            for tr, va in kf.split(X):
                m_y = GradientBoostingRegressor(max_depth=3, n_estimators=60, random_state=self.random_seed).fit(X[tr], Y[tr])
                m_t = GradientBoostingRegressor(max_depth=3, n_estimators=60, random_state=self.random_seed).fit(X[tr], T[tr])
                y_res[va] = Y[va] - m_y.predict(X[va])
                t_res[va] = T[va] - m_t.predict(X[va])
            denom = float(np.dot(t_res, t_res))
            theta = float(np.dot(t_res, y_res) / max(denom, 1e-12))
            # Neyman-orthogonal HC1 variance: Var(theta) = (n/(n-1)) * sum(t_res^2 * eps^2) / (sum(t_res^2))^2
            eps = y_res - theta * t_res
            dof_scale = float(n) / max(float(n - 1), 1.0)
            dml_se = float(np.sqrt(dof_scale * float(np.sum((t_res * eps) ** 2))) / max(denom, 1e-12))
            return (
                theta,
                None,
                {
                    "orthogonal_score_r2": round(float(1.0 - np.var(eps) / max(np.var(y_res), 1e-12)), 4),
                    "se_method": "neyman_orthogonal_hc1",
                },
                dml_se,
            )

        if method == "iv_2sls":
            if Z is None:
                raise CausalInferenceError("Method 'iv_2sls' requires 'instrument_col'.")
            n = len(X)
            # First stage: restricted T ~ [1, X] vs unrestricted T ~ [1, X, Z]
            X_r = np.column_stack([np.ones(n), X])
            X_u = np.column_stack([np.ones(n), X, Z])
            beta1_r, *_ = np.linalg.lstsq(X_r, T, rcond=None)
            beta1_u, *_ = np.linalg.lstsq(X_u, T, rcond=None)
            rss_r = float(np.sum((T - X_r @ beta1_r) ** 2))
            rss_u = float(np.sum((T - X_u @ beta1_u) ** 2))
            k_u = X_u.shape[1]
            df_denom = max(n - k_u, 1)
            q_excl = 1  # single excluded instrument
            partial_r2 = max(0.0, (rss_r - rss_u) / max(rss_r, 1e-12))
            partial_f = max(0.0, ((rss_r - rss_u) / float(q_excl)) / max(rss_u / float(df_denom), 1e-12))

            # Second stage using first-stage fitted T_hat
            t_hat = X_u @ beta1_u
            W_hat = np.column_stack([np.ones(n), X, t_hat])
            W_actual = np.column_stack([np.ones(n), X, T])
            beta2, *_ = np.linalg.lstsq(W_hat, Y, rcond=None)
            # Structural residuals MUST use actual T (not t_hat) for valid 2SLS variance!
            u_struct = Y - W_actual @ beta2
            k_2 = W_hat.shape[1]
            wtw_inv = np.linalg.pinv(W_hat.T @ W_hat)
            meat = (W_hat * (u_struct**2)[:, None]).T @ W_hat
            dof_adj = float(n) / max(float(n - k_2), 1.0)
            vcov_hc1 = dof_adj * (wtw_inv @ meat @ wtw_inv)
            se_2sls = float(np.sqrt(max(float(vcov_hc1[-1, -1]), 0.0)))

            # Stock & Yogo (2005) critical value for 10% maximal IV Wald size distortion (1 instrument) = 16.38
            return (
                float(beta2[-1]),
                None,
                {
                    "first_stage_partial_f": round(partial_f, 4),
                    "first_stage_partial_r2": round(partial_r2, 4),
                    "first_stage_r2_instrument": round(partial_r2, 4),
                    "stock_yogo_10pct_cv": 16.38,
                    "weak_instrument_warning": bool(partial_f < 16.38),
                    "se_method": "2sls_structural_hc1",
                },
                se_2sls,
            )

        if method == "s_learner":
            reg = RandomForestRegressor(n_estimators=80, max_depth=6, random_state=self.random_seed)
            reg.fit(np.column_stack([X, T]), Y)
            y1 = reg.predict(np.column_stack([X, np.ones(len(X))]))
            y0 = reg.predict(np.column_stack([X, np.zeros(len(X))]))
            cate = y1 - y0
            return float(np.mean(cate)), cate, {}, None

        if method == "t_learner":
            m1 = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[T == 1], Y[T == 1])
            m0 = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[T == 0], Y[T == 0])
            cate = m1.predict(X) - m0.predict(X)
            return float(np.mean(cate)), cate, {}, None

        if method == "x_learner":
            ps = self._fit_propensity(X, T)
            m1 = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[T == 1], Y[T == 1])
            m0 = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[T == 0], Y[T == 0])
            d1 = Y[T == 1] - m0.predict(X[T == 1])
            d0 = m1.predict(X[T == 0]) - Y[T == 0]
            tau1 = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[T == 1], d1).predict(X)
            tau0 = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[T == 0], d0).predict(X)
            cate = ps * tau0 + (1.0 - ps) * tau1
            return float(np.mean(cate)), cate, {}, None

        # Default: Cross-fitted Doubly Robust Augmented IPW (AIPW; Robins, Rotnitzky & Zhao 1994)
        n = len(Y)
        t_int = T.astype(int)
        min_arm = int(min(np.sum(t_int == 1), np.sum(t_int == 0)))
        n_splits = min(5, max(2, min_arm // 3))
        ps_oof = np.empty(n, dtype=float)
        m1_oof = np.empty(n, dtype=float)
        m0_oof = np.empty(n, dtype=float)

        if n_splits >= 2 and min_arm >= 4:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
            for tr, va in skf.split(X, t_int):
                ps_oof[va] = self._fit_propensity(X[tr], T[tr], X_eval=X[va])
                r1 = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[tr][T[tr] == 1], Y[tr][T[tr] == 1])
                r0 = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[tr][T[tr] == 0], Y[tr][T[tr] == 0])
                m1_oof[va] = r1.predict(X[va])
                m0_oof[va] = r0.predict(X[va])
        else:
            ps_oof = self._fit_propensity(X, T)
            m1_oof = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[T == 1], Y[T == 1]).predict(X)
            m0_oof = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[T == 0], Y[T == 0]).predict(X)

        psi = (m1_oof - m0_oof) + T * (Y - m1_oof) / ps_oof - (1.0 - T) * (Y - m0_oof) / (1.0 - ps_oof)
        ate = float(np.mean(psi))
        # Efficient Influence Function (EIF) SE: sd(psi, ddof=1) / sqrt(n)
        eif_se = float(np.std(psi, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        return (
            ate,
            (m1_oof - m0_oof),
            {
                "mean_propensity": round(float(np.mean(ps_oof)), 4),
                "cross_fit_folds": n_splits if min_arm >= 4 else 1,
                "se_method": "efficient_influence_function",
            },
            eif_se,
        )

    def _fit_propensity_with_raw(
        self, X: np.ndarray, T: np.ndarray, X_eval: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=500, random_state=self.random_seed),
        )
        clf.fit(X, T.astype(int))
        target_X = X if X_eval is None else X_eval
        raw = np.asarray(clf.predict_proba(target_X)[:, 1], dtype=float)
        return raw, np.clip(raw, 0.05, 0.95)

    def _fit_propensity(
        self, X: np.ndarray, T: np.ndarray, X_eval: np.ndarray | None = None
    ) -> np.ndarray:
        _, clipped = self._fit_propensity_with_raw(X, T, X_eval=X_eval)
        return clipped

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
        placebo_ate, _, _, _ = self._dispatch_estimator(method, X, T_placebo, Y, Z)

        # 2. Random common cause (adding noise confounder should not materially shift ATE)
        X_aug = np.column_stack([X, rng.normal(0.0, 1.0, size=len(X))])
        rcc_ate, _, _, _ = self._dispatch_estimator(method, X_aug, T, Y, Z)

        # 3. 80% data subset stability check
        sub_idx = rng.choice(len(Y), size=int(len(Y) * 0.8), replace=False)
        sub_ate, _, _, _ = self._dispatch_estimator(
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
