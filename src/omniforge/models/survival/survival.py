"""Enterprise Survival Analysis: Kaplan-Meier, Log-Rank Test, Nelson-Aalen, and Cox PH."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from omniforge.core.exceptions import ModelError


class KaplanMeierEstimator:
    """Non-parametric Kaplan-Meier survival curve estimator with Greenwood log-log 95% confidence intervals."""

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = float(alpha)
        self.survival_table_: pd.DataFrame = pd.DataFrame()
        self.median_survival_time_: float | None = None

    def fit(self, durations: pd.Series | np.ndarray, events: pd.Series | np.ndarray) -> KaplanMeierEstimator:
        """Compute Kaplan-Meier survival probabilities and Kalbfleisch-Prentice log-log Greenwood CIs."""
        t = np.asarray(durations, dtype=float).ravel()
        e = np.asarray(events, dtype=int).ravel()
        if len(t) == 0 or len(t) != len(e):
            raise ModelError("durations and events must be non-empty arrays of equal length.")

        z_crit = float(stats.norm.ppf(1.0 - self.alpha / 2.0))
        unique_times = np.sort(np.unique(t))
        rows: list[dict[str, float]] = []
        surv = 1.0
        greenwood_sum = 0.0
        cum_hazard = 0.0

        for time_pt in unique_times:
            at_risk = int(np.sum(t >= time_pt))
            deaths = int(np.sum((t == time_pt) & (e == 1)))
            censored = int(np.sum((t == time_pt) & (e == 0)))
            if deaths > 0 and at_risk > 0:
                surv *= 1.0 - deaths / at_risk
                cum_hazard += deaths / at_risk
                if at_risk > deaths:
                    greenwood_sum += deaths / (at_risk * (at_risk - deaths))
            if 0.0 < surv < 1.0 and greenwood_sum > 0.0:
                se_loglog = float(np.sqrt(greenwood_sum) / abs(np.log(surv)))
                ci_lower = float(surv ** np.exp(z_crit * se_loglog))
                ci_upper = float(surv ** np.exp(-z_crit * se_loglog))
            elif surv >= 1.0:
                ci_lower, ci_upper = 1.0, 1.0
            else:
                ci_lower, ci_upper = 0.0, 0.0
            rows.append(
                {
                    "time": float(time_pt),
                    "at_risk": float(at_risk),
                    "events": float(deaths),
                    "censored": float(censored),
                    "survival_prob": round(float(surv), 6),
                    "ci_lower": round(float(np.clip(ci_lower, 0.0, 1.0)), 6),
                    "ci_upper": round(float(np.clip(ci_upper, 0.0, 1.0)), 6),
                    "nelson_aalen_cum_hazard": round(float(cum_hazard), 6),
                }
            )

        self.survival_table_ = pd.DataFrame(rows)
        below_half = self.survival_table_[self.survival_table_["survival_prob"] <= 0.5]
        self.median_survival_time_ = float(below_half["time"].iloc[0]) if len(below_half) > 0 else None
        return self

    @staticmethod
    def log_rank_test(
        durations_a: np.ndarray, events_a: np.ndarray, durations_b: np.ndarray, events_b: np.ndarray
    ) -> dict[str, float]:
        """Perform two-group Mantel-Haenszel Log-Rank hypothesis test."""
        ta, ea = np.asarray(durations_a, dtype=float), np.asarray(events_a, dtype=int)
        tb, eb = np.asarray(durations_b, dtype=float), np.asarray(events_b, dtype=int)
        all_times = np.sort(np.unique(np.concatenate([ta[ea == 1], tb[eb == 1]])))
        obs_a, exp_a, var_a = 0.0, 0.0, 0.0
        for tm in all_times:
            n1 = float(np.sum(ta >= tm))
            n2 = float(np.sum(tb >= tm))
            n = n1 + n2
            if n <= 1:
                continue
            d1 = float(np.sum((ta == tm) & (ea == 1)))
            d2 = float(np.sum((tb == tm) & (eb == 1)))
            d = d1 + d2
            obs_a += d1
            exp_a += n1 * d / n
            var_a += (n1 * n2 * d * (n - d)) / (n * n * (n - 1.0))
        chi2 = ((obs_a - exp_a) ** 2) / max(var_a, 1e-12)
        p_val = float(1.0 - stats.chi2.cdf(chi2, df=1))
        return {"chi2_statistic": round(float(chi2), 6), "p_value": round(p_val, 6)}


class CoxPHModel:
    """Cox Proportional Hazards regression with exact Efron/Breslow tie handling and original-unit inference.

    Optimization is executed in standardized covariate coordinates for numerical conditioning,
    then coefficients, covariance matrix, standard errors, and hazard ratios are transformed
    back to original covariate units (matching R ``survival::coxph`` and ``statsmodels.PHReg``).
    """

    def __init__(
        self,
        l2_penalty: float = 1e-4,
        max_iter: int = 100,
        tol: float = 1e-7,
        ties: Literal["efron", "breslow"] = "efron",
    ) -> None:
        self.l2_penalty = max(0.0, float(l2_penalty))
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.ties: Literal["efron", "breslow"] = ties
        self._scaler = StandardScaler()
        self.coef_: np.ndarray = np.array([], dtype=float)
        self.standardized_coef_: np.ndarray = np.array([], dtype=float)
        self.standard_errors_: np.ndarray = np.array([], dtype=float)
        self.p_values_: dict[str, float] = {}
        self.feature_names_: list[str] = []
        self.hazard_ratios_: dict[str, float] = {}
        self.hazard_ratio_ci_: dict[str, tuple[float, float]] = {}
        self.log_partial_likelihood_: float = 0.0
        self.baseline_times_: np.ndarray = np.array([], dtype=float)
        self.baseline_cum_hazard_: np.ndarray = np.array([], dtype=float)
        self.concordance_index_: float = 0.5

    def _partial_loglik_grad_hess(
        self,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
        unique_event_times: np.ndarray,
        beta: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute exact Efron or Breslow penalized partial log-likelihood, gradient, and Fisher info."""
        p = X.shape[1]
        eta = np.clip(X @ beta, -30.0, 30.0)
        exp_eta = np.exp(eta)

        # Reverse cumulative sums sorted ascending by T; index at first occurrence of t_k gives R(t_k) = {j: T_j >= t_k}
        rev_w = np.cumsum(exp_eta[::-1])[::-1]
        wx = X * exp_eta[:, None]
        rev_wx = np.cumsum(wx[::-1], axis=0)[::-1]
        wxx = (X[:, :, None] * X[:, None, :]) * exp_eta[:, None, None]
        rev_wxx = np.cumsum(wxx[::-1], axis=0)[::-1]

        ll = 0.0
        grad = np.zeros(p, dtype=float)
        info = np.zeros((p, p), dtype=float)

        for t_k in unique_event_times:
            # First index in sorted T where T >= t_k defines the exact risk set R(t_k) (including ties & censored at t_k)
            risk_start = int(np.searchsorted(T, t_k, side="left"))
            w_r = float(rev_w[risk_start])
            x_r = rev_wx[risk_start]
            m_r = rev_wxx[risk_start]

            event_mask = (T == t_k) & (E == 1)
            d_k = int(np.sum(event_mask))
            if d_k == 0 or w_r <= 1e-15:
                continue

            ll += float(np.sum(eta[event_mask]))
            grad += np.sum(X[event_mask], axis=0)

            if self.ties == "breslow" or d_k == 1:
                inv_w = 1.0 / w_r
                mean_x = x_r * inv_w
                ll -= d_k * float(np.log(w_r))
                grad -= d_k * mean_x
                info += d_k * (m_r * inv_w - np.outer(mean_x, mean_x))
            else:
                # Efron (1977) partial likelihood approximation for tied event times
                w_d = float(np.sum(exp_eta[event_mask]))
                x_d = np.sum(wx[event_mask], axis=0)
                m_d = np.sum(wxx[event_mask], axis=0)
                for m in range(d_k):
                    frac = float(m) / float(d_k)
                    w_m = max(w_r - frac * w_d, 1e-15)
                    x_m = x_r - frac * x_d
                    m_m = m_r - frac * m_d
                    mean_m = x_m / w_m
                    ll -= float(np.log(w_m))
                    grad -= mean_m
                    info += (m_m / w_m) - np.outer(mean_m, mean_m)

        # L2 ridge regularization in standardized coordinates (0.5 * lambda * ||beta||^2)
        if self.l2_penalty > 0.0:
            ll -= 0.5 * self.l2_penalty * float(beta @ beta)
            grad -= self.l2_penalty * beta
            info += np.eye(p, dtype=float) * self.l2_penalty

        return ll, grad, info

    def fit(
        self,
        df: pd.DataFrame,
        *,
        duration_col: str = "duration",
        event_col: str = "event",
        feature_cols: list[str] | None = None,
    ) -> CoxPHModel:
        """Fit Cox PH via damped Newton-Raphson with exact Efron/Breslow tie handling."""
        cols = feature_cols or [c for c in df.columns if c not in (duration_col, event_col)]
        if not cols:
            raise ModelError("CoxPHModel requires at least one covariate column.")
        self.feature_names_ = [str(c) for c in cols]

        order = np.argsort(df[duration_col].to_numpy(dtype=float), kind="mergesort")
        X_raw = df[cols].to_numpy(dtype=float)[order]
        X = self._scaler.fit_transform(X_raw)
        T = df[duration_col].to_numpy(dtype=float)[order]
        E = df[event_col].to_numpy(dtype=int)[order]
        unique_event_times = np.unique(T[E == 1])

        p = X.shape[1]
        beta = np.zeros(p, dtype=float)
        ll_curr, grad, info = self._partial_loglik_grad_hess(X, T, E, unique_event_times, beta)

        for _ in range(self.max_iter):
            try:
                step = np.linalg.solve(info + np.eye(p, dtype=float) * 1e-10, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(info + np.eye(p, dtype=float) * 1e-6, grad, rcond=None)[0]

            # Step-halving line search guarantees monotonic ascent in penalized partial log-likelihood
            step_scale = 1.0
            accepted = False
            for _ls in range(20):
                cand = beta + step_scale * step
                ll_cand, grad_cand, info_cand = self._partial_loglik_grad_hess(
                    X, T, E, unique_event_times, cand
                )
                if np.isfinite(ll_cand) and ll_cand >= ll_curr - 1e-10:
                    beta = cand
                    ll_curr, grad, info = ll_cand, grad_cand, info_cand
                    accepted = True
                    break
                step_scale *= 0.5

            if not accepted or float(np.linalg.norm(step_scale * step)) < self.tol:
                break

        self.standardized_coef_ = beta.copy()
        self.log_partial_likelihood_ = float(ll_curr)

        # Unscale coefficients & covariance back to original covariate units: beta_orig = beta_std / sigma
        scales = np.where(self._scaler.scale_ > 1e-12, self._scaler.scale_, 1.0)
        beta_orig = beta / scales
        self.coef_ = beta_orig

        # Covariance matrix in original units: D^{-1} I^{-1} D^{-1}
        try:
            cov_std = np.linalg.inv(info + np.eye(p, dtype=float) * 1e-12)
        except np.linalg.LinAlgError:
            cov_std = np.linalg.pinv(info)
        cov_orig = cov_std / np.outer(scales, scales)
        se_orig = np.sqrt(np.maximum(np.diag(cov_orig), 0.0))
        self.standard_errors_ = se_orig

        z_crit = float(stats.norm.ppf(0.975))
        self.hazard_ratios_ = {}
        self.hazard_ratio_ci_ = {}
        self.p_values_ = {}
        for name, b, se in zip(self.feature_names_, beta_orig, se_orig, strict=False):
            hr = float(np.exp(np.clip(b, -30.0, 30.0)))
            lo = float(np.exp(np.clip(b - z_crit * se, -30.0, 30.0)))
            hi = float(np.exp(np.clip(b + z_crit * se, -30.0, 30.0)))
            z_stat = abs(float(b) / se) if se > 1e-12 else 0.0
            p_val = float(2.0 * stats.norm.sf(z_stat))
            self.hazard_ratios_[name] = round(hr, 6)
            self.hazard_ratio_ci_[name] = (round(lo, 6), round(hi, 6))
            self.p_values_[name] = round(p_val, 6)

        # Exact Efron / Breslow baseline cumulative hazard H_0(t) anchored at covariate mean
        risk_scores = np.exp(np.clip(X @ beta, -30.0, 30.0))
        cum_h = 0.0
        base_h: list[float] = []
        for et in unique_event_times:
            ev_mask = (T == et) & (E == 1)
            d_i = int(np.sum(ev_mask))
            r_i = float(np.sum(risk_scores[T >= et]))
            if self.ties == "efron" and d_i > 1:
                w_d = float(np.sum(risk_scores[ev_mask]))
                for m in range(d_i):
                    cum_h += 1.0 / max(r_i - (float(m) / float(d_i)) * w_d, 1e-12)
            else:
                cum_h += float(d_i) / max(r_i, 1e-12)
            base_h.append(cum_h)
        self.baseline_times_ = unique_event_times
        self.baseline_cum_hazard_ = np.asarray(base_h, dtype=float)
        self.concordance_index_ = self.compute_c_index(T, E, X @ beta)
        return self

    def predict_risk(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict mean-centered relative hazard risk score exp((X - mu) @ beta_orig)."""
        arr = X[self.feature_names_].to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        centered = arr - self._scaler.mean_
        return np.asarray(np.exp(np.clip(centered @ self.coef_, -30.0, 30.0)), dtype=float)

    def predict_survival_function(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Return predicted survival probability curves S(t | x) across baseline event times."""
        risk = self.predict_risk(X)
        surv = np.exp(-np.outer(self.baseline_cum_hazard_, risk))
        return pd.DataFrame(surv, index=self.baseline_times_)

    @staticmethod
    def compute_c_index(durations: np.ndarray, events: np.ndarray, risk_scores: np.ndarray) -> float:
        """Compute Harrell's Concordance Index (C-index) properly including tied event/censoring pairs."""
        t = np.asarray(durations, dtype=float).ravel()
        e = np.asarray(events, dtype=int).ravel()
        r = np.asarray(risk_scores, dtype=float).ravel()
        concordant = 0.0
        permissible = 0.0
        n = len(t)
        for i in range(n):
            if e[i] != 1:
                continue
            for j in range(n):
                if i == j:
                    continue
                if t[j] > t[i] or (t[j] == t[i] and e[j] == 0):
                    permissible += 1.0
                    if r[i] > r[j]:
                        concordant += 1.0
                    elif r[i] == r[j]:
                        concordant += 0.5
        return round(concordant / max(permissible, 1.0), 6)
