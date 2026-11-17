"""Enterprise Survival Analysis: Kaplan-Meier, Log-Rank Test, Nelson-Aalen, and Cox PH."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from omniforge.core.exceptions import ModelError


class KaplanMeierEstimator:
    """Non-parametric Kaplan-Meier survival curve estimator with Greenwood 95% confidence intervals."""

    def __init__(self) -> None:
        self.survival_table_: pd.DataFrame = pd.DataFrame()
        self.median_survival_time_: float | None = None

    def fit(self, durations: pd.Series | np.ndarray, events: pd.Series | np.ndarray) -> KaplanMeierEstimator:
        """Compute Kaplan-Meier survival probabilities and Greenwood standard errors."""
        t = np.asarray(durations, dtype=float).ravel()
        e = np.asarray(events, dtype=int).ravel()
        if len(t) == 0 or len(t) != len(e):
            raise ModelError("durations and events must be non-empty arrays of equal length.")

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
            se = surv * np.sqrt(greenwood_sum)
            rows.append(
                {
                    "time": float(time_pt),
                    "at_risk": float(at_risk),
                    "events": float(deaths),
                    "censored": float(censored),
                    "survival_prob": round(float(surv), 6),
                    "ci_lower": round(float(max(0.0, surv - 1.96 * se)), 6),
                    "ci_upper": round(float(min(1.0, surv + 1.96 * se)), 6),
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
    """Cox Proportional Hazards regression solved via L2-regularized Newton-Raphson partial likelihood."""

    def __init__(self, l2_penalty: float = 1e-2, max_iter: int = 50, tol: float = 1e-6) -> None:
        self.l2_penalty = float(l2_penalty)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self._scaler = StandardScaler()
        self.coef_: np.ndarray = np.array([], dtype=float)
        self.feature_names_: list[str] = []
        self.hazard_ratios_: dict[str, float] = {}
        self.baseline_times_: np.ndarray = np.array([], dtype=float)
        self.baseline_cum_hazard_: np.ndarray = np.array([], dtype=float)
        self.concordance_index_: float = 0.5

    def fit(
        self,
        df: pd.DataFrame,
        *,
        duration_col: str = "duration",
        event_col: str = "event",
        feature_cols: list[str] | None = None,
    ) -> CoxPHModel:
        """Fit Cox PH coefficients via Newton-Raphson on Breslow partial log-likelihood."""
        cols = feature_cols or [c for c in df.columns if c not in (duration_col, event_col)]
        self.feature_names_ = [str(c) for c in cols]
        order = np.argsort(df[duration_col].to_numpy(dtype=float))
        X_raw = df[cols].to_numpy(dtype=float)[order]
        X = self._scaler.fit_transform(X_raw)
        T = df[duration_col].to_numpy(dtype=float)[order]
        E = df[event_col].to_numpy(dtype=int)[order]

        n, p = X.shape
        beta = np.zeros(p, dtype=float)

        for _ in range(self.max_iter):
            eta = np.clip(X @ beta, -30.0, 30.0)
            exp_eta = np.exp(eta)
            # Reverse cumulative sums define risk sets R(t_i)
            s0 = np.cumsum(exp_eta[::-1])[::-1]
            s1 = np.cumsum((X * exp_eta[:, None])[::-1], axis=0)[::-1]

            grad = np.zeros(p, dtype=float)
            hess = np.eye(p, dtype=float) * self.l2_penalty
            for i in range(n):
                if E[i] == 1 and s0[i] > 1e-12:
                    mean_x = s1[i] / s0[i]
                    grad += X[i] - mean_x
                    # Outer product variance approximation for positive-definite Hessian
                    risk_w = exp_eta[i:] / s0[i]
                    centered = X[i:] - mean_x
                    hess += (centered * risk_w[:, None]).T @ centered

            grad -= self.l2_penalty * beta
            step = np.linalg.solve(hess + np.eye(p) * 1e-6, grad)
            beta += step
            if float(np.linalg.norm(step)) < self.tol:
                break

        self.coef_ = beta
        self.hazard_ratios_ = {
            name: round(float(np.exp(b)), 6) for name, b in zip(self.feature_names_, beta, strict=False)
        }

        # Breslow baseline cumulative hazard
        risk_scores = np.exp(np.clip(X @ beta, -30.0, 30.0))
        event_times = np.unique(T[E == 1])
        cum_h = 0.0
        base_h: list[float] = []
        for et in event_times:
            d_i = float(np.sum((T == et) & (E == 1)))
            r_i = float(np.sum(risk_scores[T >= et]))
            cum_h += d_i / max(r_i, 1e-12)
            base_h.append(cum_h)
        self.baseline_times_ = event_times
        self.baseline_cum_hazard_ = np.asarray(base_h, dtype=float)
        self.concordance_index_ = self.compute_c_index(T, E, X @ beta)
        return self

    def predict_risk(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict relative hazard risk score exp(X @ beta)."""
        arr = X[self.feature_names_].to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        X_s = self._scaler.transform(arr)
        return np.exp(np.clip(X_s @ self.coef_, -30.0, 30.0))

    def predict_survival_function(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Return predicted survival probability curves S(t | x) across baseline event times."""
        risk = self.predict_risk(X)
        surv = np.exp(-np.outer(self.baseline_cum_hazard_, risk))
        return pd.DataFrame(surv, index=self.baseline_times_)

    @staticmethod
    def compute_c_index(durations: np.ndarray, events: np.ndarray, risk_scores: np.ndarray) -> float:
        """Compute Harrell's Concordance Index (C-index)."""
        concordant = 0.0
        permissible = 0.0
        n = len(durations)
        for i in range(n):
            if events[i] != 1:
                continue
            for j in range(n):
                if durations[j] > durations[i]:
                    permissible += 1.0
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1.0
                    elif risk_scores[i] == risk_scores[j]:
                        concordant += 0.5
        return round(concordant / max(permissible, 1.0), 6)
