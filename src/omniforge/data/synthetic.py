"""Synthetic data generator supporting Gaussian Copula synthesis, DP noise, and multi-task datasets."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from omniforge.core.constants import DEFAULT_RANDOM_SEED
from omniforge.core.exceptions import DataValidationError


def _project_psd_correlation(corr: np.ndarray) -> np.ndarray:
    """Symmetrize and project a noisy matrix onto the positive semi-definite correlation cone."""
    sym = 0.5 * (corr + corr.T)
    np.fill_diagonal(sym, 1.0)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, 1e-5, None)
    psd = (eigvecs * eigvals) @ eigvecs.T
    diag_std = np.sqrt(np.maximum(np.diag(psd), 1e-12))
    psd = psd / np.outer(diag_std, diag_std)
    psd = 0.5 * (psd + psd.T)
    np.fill_diagonal(psd, 1.0)
    return np.asarray(psd, dtype=float)


def _dp_noisy_histogram_cdf(
    values: np.ndarray,
    low: float,
    high: float,
    n_bins: int,
    eps_col: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an (eps_col, 0)-DP piecewise-uniform histogram CDF via the Laplace Mechanism.

    Under bounded-replacement adjacency (D ~ D', |D| = |D'| = N differing in 1 record), a disjoint
    histogram count vector has L1 sensitivity Delta_1 = 2 (Dwork et al., 2006; Dwork & Roth, 2014,
    Theorem 3.6; NIST SP 800-226).
    """
    if high <= low:
        edges = np.array([low, high], dtype=float)
        return edges, np.array([0.0, 1.0], dtype=float)

    edges = np.linspace(low, high, n_bins + 1, dtype=float)
    clipped = np.clip(values, low, high)
    counts, _ = np.histogram(clipped, bins=edges)
    # L1 sensitivity of a disjoint histogram under 1-record bounded replacement is Delta_1 = 2.0
    noisy_counts = np.maximum(counts.astype(float) + rng.laplace(0.0, 2.0 / eps_col, size=n_bins), 0.0)
    total = float(noisy_counts.sum())
    probs = noisy_counts / total if total > 1e-12 else np.full(n_bins, 1.0 / n_bins, dtype=float)
    cdf_knots = np.concatenate(([0.0], np.cumsum(probs)))
    cdf_knots[-1] = 1.0
    return edges, cdf_knots


class SyntheticDataGenerator:
    """Generate synthetic tabular data and domain benchmarks."""

    def __init__(self, random_seed: int = DEFAULT_RANDOM_SEED, random_state: int | None = None) -> None:
        self.random_seed = random_state if random_state is not None else random_seed

    def fit_generate_copula(
        self,
        reference_df: pd.DataFrame,
        n_samples: int | None = None,
        *,
        dp_epsilon: float | None = None,
        dp_bounds: dict[str, tuple[float, float]] | None = None,
        dp_categories: dict[str, list[Any]] | None = None,
    ) -> pd.DataFrame:
        """Synthesize tabular data preserving marginal distributions and rank correlations via Gaussian Copula.

        With ``dp_epsilon``, use a Laplace mechanism with a fixed public schema,
        public column domains, and bounded-replacement adjacency at a fixed row
        count. Noise uses fresh randomness for each release. Repeated releases
        need a cumulative budget. Categorical columns use marginal draws.

        Args:
            reference_df: Real source DataFrame to model.
            n_samples: Number of synthetic rows to generate (defaults to ``len(reference_df)``).
            dp_epsilon: Optional strictly positive total privacy budget ``epsilon > 0``.
            dp_bounds: Required public ``(min, max)`` clipping bounds per numeric/datetime column for strict global sensitivity.
            dp_categories: Required public category list per non-boolean categorical column for strict domain privacy.

        Returns:
            Synthetic DataFrame matching schema and statistical structure of ``reference_df``.
        """
        if dp_epsilon is not None:
            if isinstance(dp_epsilon, bool) or not math.isfinite(float(dp_epsilon)) or float(dp_epsilon) <= 0.0:
                raise ValueError(
                    "dp_epsilon must be a finite positive float (> 0) to guarantee epsilon-differential privacy."
                )

        rng = np.random.default_rng(self.random_seed if dp_epsilon is None else None)
        if len(reference_df.columns) == 0:
            return pd.DataFrame()
        n = len(reference_df) if n_samples is None else int(n_samples)
        if n <= 0 or len(reference_df) == 0:
            return pd.DataFrame(columns=reference_df.columns)

        num_cols = [
            c
            for c in reference_df.select_dtypes(include=[np.number]).columns
            if not pd.api.types.is_bool_dtype(reference_df[c])
        ]
        other_cols = [c for c in reference_df.columns if c not in num_cols]
        out = pd.DataFrame(index=range(n))

        # Non-DP fast empirical Gaussian copula path
        if dp_epsilon is None:
            if num_cols:
                z_cols: list[np.ndarray] = []
                sorted_originals: dict[str, np.ndarray] = {}
                is_int_col: dict[str, bool] = {}

                for col in num_cols:
                    series = reference_df[col]
                    is_int_col[str(col)] = bool(pd.api.types.is_integer_dtype(series))
                    vals = series.fillna(series.median() if series.notna().any() else 0.0).to_numpy(dtype=float)
                    sorted_originals[str(col)] = np.sort(vals)
                    ranks = stats.rankdata(vals, method="average")
                    u = np.clip(ranks / (len(vals) + 1.0), 1e-6, 1.0 - 1e-6)
                    z_cols.append(stats.norm.ppf(u))

                z_matrix = np.column_stack(z_cols)
                if z_matrix.shape[1] == 1:
                    corr = np.array([[1.0]])
                else:
                    z_std = np.std(z_matrix, axis=0)
                    non_const = z_std > 1e-12
                    corr = np.eye(z_matrix.shape[1], dtype=float)
                    if int(np.sum(non_const)) >= 2:
                        sub_corr = np.corrcoef(z_matrix[:, non_const], rowvar=False)
                        sub_corr = np.nan_to_num(sub_corr, nan=0.0)
                        idx_nc = np.where(non_const)[0]
                        corr[np.ix_(idx_nc, idx_nc)] = sub_corr
                    corr = _project_psd_correlation(corr)

                sampled_z = rng.multivariate_normal(mean=np.zeros(len(num_cols)), cov=corr, size=n)
                sampled_u = stats.norm.cdf(sampled_z)

                for idx, col in enumerate(num_cols):
                    orig_sorted = sorted_originals[str(col)]
                    synth_vals = np.quantile(orig_sorted, sampled_u[:, idx])
                    if is_int_col[str(col)]:
                        out[col] = np.rint(synth_vals).astype(int)
                    else:
                        out[col] = synth_vals

            for col in other_cols:
                series = reference_df[col]
                if pd.api.types.is_datetime64_any_dtype(series):
                    non_null = series.dropna()
                    if len(non_null) == 0:
                        out[col] = pd.NaT
                    else:
                        ts_int = non_null.astype("datetime64[ns]").astype("int64").to_numpy()
                        sampled_ts = rng.uniform(float(ts_int.min()), float(ts_int.max()), size=n).astype("int64")
                        out[col] = pd.to_datetime(sampled_ts, unit="ns")
                else:
                    counts = series.value_counts(dropna=False)
                    probs = (counts / counts.sum()).to_numpy(dtype=float)
                    choices = counts.index.to_numpy()
                    sampled_idx = rng.choice(len(choices), size=n, p=probs)
                    out[col] = [choices[i] for i in sampled_idx]

            out.attrs["is_synthetic"] = True
            return out[reference_df.columns]

        # Pure (dp_epsilon, 0)-DP Gaussian Copula path (Sequential Composition + Post-Processing)
        eps_total = float(dp_epsilon)
        n_ref = len(reference_df)
        total_cols = max(len(reference_df.columns), 1)
        d_num = len(num_cols)
        n_pairs = (d_num * (d_num - 1)) // 2
        if n_pairs >= 1:
            eps_marginal_total = 0.7 * eps_total
            eps_corr_total = 0.3 * eps_total
            eps_pair = eps_corr_total / n_pairs
        else:
            eps_marginal_total = eps_total
            eps_pair = 0.0
        eps_per_col = eps_marginal_total / total_cols
        n_bins = min(32, max(4, int(math.isqrt(max(n_ref, 1)))))
        if len({str(column) for column in reference_df.columns}) != len(reference_df.columns):
            raise DataValidationError("DP synthesis requires unique public column names.")
        bounds_map = dp_bounds or {}
        cat_domain_map = dp_categories or {}
        public_bounds: dict[str, tuple[float, float]] = {}
        for col, series in reference_df.items():
            key = str(col)
            temporal = pd.api.types.is_datetime64_any_dtype(series)
            if col in num_cols or temporal:
                if key not in bounds_map:
                    raise DataValidationError("DP synthesis requires public bounds for every numeric and datetime column.")
                try:
                    low, high = map(float, bounds_map[key])
                except (ValueError, TypeError, OverflowError) as exc:
                    raise DataValidationError("DP synthesis requires a numeric pair for each public bound.") from exc
                if not math.isfinite(low) or not math.isfinite(high) or low > high or not math.isfinite(high - low):
                    raise DataValidationError("DP synthesis requires finite, ordered public bounds.")
                if pd.api.types.is_integer_dtype(series) and (
                    not low.is_integer() or not high.is_integer() or low < -(2**63) or high >= 2**63
                ):
                    raise DataValidationError("DP synthesis requires public integer bounds within int64 range.")
                if temporal and (low <= float(pd.NaT.value) or high >= float(pd.Timestamp.max.value)):
                    raise DataValidationError("DP datetime bounds must fit in the supported nanosecond timestamp range.")
                public_bounds[key] = (low, high)
            elif not pd.api.types.is_bool_dtype(series) and not cat_domain_map.get(key):
                raise DataValidationError("DP synthesis requires a public category domain for every non-boolean categorical column.")

        if eps_per_col <= 0 or not math.isfinite(2.0 / eps_per_col) or (n_pairs and eps_pair <= 0):
            raise DataValidationError("DP budget is too small for this schema.")
        if num_cols:
            dp_edges: dict[str, np.ndarray] = {}
            dp_cdfs: dict[str, np.ndarray] = {}
            dp_is_int: dict[str, bool] = {}
            z_bounded_cols: list[np.ndarray] = []

            for col in num_cols:
                col_key = str(col)
                series = reference_df[col]
                dp_is_int[col_key] = bool(pd.api.types.is_integer_dtype(series))
                lo, hi = public_bounds[col_key]
                mid = lo + 0.5 * (hi - lo)
                vals = np.clip(pd.to_numeric(series, errors="coerce").fillna(mid).to_numpy(dtype=float), lo, hi)
                edges, cdf_knots = _dp_noisy_histogram_cdf(vals, lo, hi, n_bins, eps_per_col, rng)
                dp_edges[col_key] = edges
                dp_cdfs[col_key] = cdf_knots

                u_scores = np.clip(np.interp(vals, edges, cdf_knots), 1e-6, 1.0 - 1e-6)
                z_bounded = np.clip(stats.norm.ppf(u_scores), -3.0, 3.0)
                z_bounded_cols.append(z_bounded)

            if d_num == 1:
                corr = np.array([[1.0]], dtype=float)
            else:
                z_mat = np.column_stack(z_bounded_cols)
                corr = np.eye(d_num, dtype=float)
                # Since each z_{ia} in [-3, 3], z_{ia}*z_{ib} in [-9, 9]; under 1-record replacement,
                # L1 sensitivity of (1/N) * sum_i z_{ia}*z_{ib} is exactly Delta_1 = 18 / N.
                pair_scale = (18.0 / n_ref) / eps_pair
                if not math.isfinite(pair_scale):
                    raise DataValidationError("DP budget is too small for the requested numeric relationships.")
                for a_idx in range(d_num):
                    for b_idx in range(a_idx + 1, d_num):
                        raw_cov = float(np.mean(z_mat[:, a_idx] * z_mat[:, b_idx]))
                        noisy_cov = float(np.clip(raw_cov + rng.laplace(0.0, pair_scale), -1.0, 1.0))
                        corr[a_idx, b_idx] = noisy_cov
                        corr[b_idx, a_idx] = noisy_cov
                corr = _project_psd_correlation(corr)

            sampled_z = rng.multivariate_normal(mean=np.zeros(d_num), cov=corr, size=n)
            sampled_u = np.clip(stats.norm.cdf(sampled_z), 0.0, 1.0)

            for idx, col in enumerate(num_cols):
                col_key = str(col)
                synth_vals = np.interp(sampled_u[:, idx], dp_cdfs[col_key], dp_edges[col_key])
                if dp_is_int[col_key]:
                    out[col] = np.rint(synth_vals).astype(int)
                else:
                    out[col] = synth_vals

        for col in other_cols:
            col_key = str(col)
            series = reference_df[col]
            if pd.api.types.is_datetime64_any_dtype(series):
                lo_ts, hi_ts = public_bounds[col_key]
                mid_ts = lo_ts + 0.5 * (hi_ts - lo_ts)
                timestamp_ns = pd.to_datetime(series, utc=True).astype("datetime64[ns, UTC]")
                raw_ts = timestamp_ns.astype("int64").where(timestamp_ns.notna(), int(mid_ts)).to_numpy(dtype=float)
                edges_ts, cdf_ts = _dp_noisy_histogram_cdf(raw_ts, lo_ts, hi_ts, n_bins, eps_per_col, rng)
                u_draw = rng.uniform(0.0, 1.0, size=n)
                synth_ts = np.interp(u_draw, cdf_ts, edges_ts).astype("int64")
                timezone = series.dt.tz
                samples = pd.to_datetime(synth_ts, unit="ns", utc=timezone is not None)
                out[col] = samples.tz_convert(timezone) if timezone is not None else samples
            else:
                domain_vals = list(cat_domain_map.get(col_key, [False, True]))
                if not domain_vals or len(set(domain_vals)) != len(domain_vals):
                    raise DataValidationError("A public category domain must contain unique values.")
                counts = series.value_counts(dropna=True, sort=False)
                cat_counts = counts.reindex(domain_vals, fill_value=0).to_numpy(dtype=float)
                # Disjoint category count histogram has L1 sensitivity Delta_1 = 2.0 under replacement
                noisy_cat = np.maximum(
                    cat_counts + rng.laplace(0.0, 2.0 / eps_per_col, size=len(domain_vals)),
                    0.0,
                )
                cat_sum = float(noisy_cat.sum())
                probs = noisy_cat / cat_sum if cat_sum > 1e-12 else np.full(len(domain_vals), 1.0 / len(domain_vals))
                sampled_idx = rng.choice(len(domain_vals), size=n, p=probs)
                out[col] = [domain_vals[i] for i in sampled_idx]

        out.attrs["is_synthetic"] = True
        out.attrs["privacy_accounting"] = {"epsilon": eps_total, "adjacency": "bounded_replacement"}
        return out[reference_df.columns]

    def generate_classification_dataset(
        self,
        n_samples: int = 500,
        n_features: int = 10,
        n_classes: int = 2,
        include_sensitive_attr: bool = True,
        missing_rate: float = 0.0,
    ) -> pd.DataFrame:
        """Generate a realistic tabular classification dataset with optional protected attributes."""
        rng = np.random.default_rng(self.random_seed)
        X = rng.normal(0.0, 1.0, size=(n_samples, n_features))
        weights = rng.normal(0.0, 1.2, size=(n_features, max(1, n_classes if n_classes > 2 else 1)))
        logits = X @ weights + 0.5 * (X[:, 0] * X[:, 1])[:, None]

        if n_classes == 2:
            probs = 1.0 / (1.0 + np.exp(-logits.ravel()))
            y = (rng.uniform(0.0, 1.0, size=n_samples) < probs).astype(int)
        else:
            exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_l / exp_l.sum(axis=1, keepdims=True)
            y = np.array([rng.choice(n_classes, p=p) for p in probs], dtype=int)

        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        df["segment"] = rng.choice(["A", "B", "C"], size=n_samples, p=[0.45, 0.35, 0.20])
        if include_sensitive_attr:
            df["group_attr"] = rng.choice(["group_0", "group_1"], size=n_samples, p=[0.6, 0.4])
        if missing_rate > 0:
            for col in [f"feature_{i}" for i in range(min(3, n_features))]:
                mask = rng.uniform(0.0, 1.0, size=n_samples) < missing_rate
                df.loc[mask, col] = np.nan
        df["target"] = y
        return df

    def generate_regression_dataset(
        self, n_samples: int = 500, n_features: int = 8, noise_std: float = 0.5
    ) -> pd.DataFrame:
        """Generate a nonlinear tabular regression dataset with mixed feature types."""
        rng = np.random.default_rng(self.random_seed)
        X = rng.normal(0.0, 1.0, size=(n_samples, n_features))
        coef = rng.uniform(-3.0, 3.0, size=n_features)
        y = (
            X @ coef
            + 1.5 * np.sin(X[:, 0])
            + 0.8 * (X[:, 1] ** 2)
            + rng.normal(0.0, noise_std, size=n_samples)
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        df["region"] = rng.choice(["north", "south", "east", "west"], size=n_samples)
        df["target"] = y
        return df

    def generate_time_series_dataset(
        self,
        n_periods: int = 365,
        freq: str = "D",
        seasonal_period: int = 7,
        trend_slope: float = 0.05,
        start_date: str = "2024-01-01",
    ) -> pd.DataFrame:
        """Generate a synthetic univariate/exogenous time-series with trend, seasonality, and noise."""
        rng = np.random.default_rng(self.random_seed)
        dates = pd.date_range(start=start_date, periods=n_periods, freq=freq)
        t = np.arange(n_periods, dtype=float)
        trend = 100.0 + trend_slope * t
        seasonal = 12.0 * np.sin(2.0 * np.pi * t / max(seasonal_period, 2)) + 5.0 * np.cos(
            2.0 * np.pi * t / max(seasonal_period * 4, 4)
        )
        promo = (rng.uniform(0, 1, size=n_periods) < 0.12).astype(int)
        temp = 20.0 + 10.0 * np.sin(2.0 * np.pi * t / 365.0) + rng.normal(0, 2.0, size=n_periods)
        noise = rng.normal(0.0, 3.0, size=n_periods)
        y = trend + seasonal + 15.0 * promo - 0.3 * temp + noise
        return pd.DataFrame(
            {"timestamp": dates, "y": y, "promotion": promo, "temperature": temp}
        )

    def generate_survival_dataset(self, n_samples: int = 400) -> pd.DataFrame:
        """Generate a right-censored survival dataset for Cox PH / Kaplan-Meier evaluation."""
        rng = np.random.default_rng(self.random_seed)
        age = rng.normal(55.0, 12.0, size=n_samples)
        biomarker = rng.normal(0.0, 1.0, size=n_samples)
        treatment = rng.binomial(1, 0.5, size=n_samples)
        hazard = np.exp(0.03 * (age - 55.0) + 0.7 * biomarker - 0.5 * treatment)
        event_time = rng.exponential(scale=100.0 / hazard)
        censor_time = rng.uniform(20.0, 150.0, size=n_samples)
        observed_time = np.minimum(event_time, censor_time)
        event = (event_time <= censor_time).astype(int)
        return pd.DataFrame(
            {
                "age": age,
                "biomarker": biomarker,
                "treatment": treatment,
                "duration": np.maximum(observed_time, 0.1),
                "event": event,
            }
        )

    def generate_causal_dataset(self, n_samples: int = 600, true_ate: float = 3.5) -> pd.DataFrame:
        """Generate confounded observational data with known ground-truth ATE."""
        rng = np.random.default_rng(self.random_seed)
        x1 = rng.normal(0.0, 1.0, size=n_samples)
        x2 = rng.normal(0.0, 1.0, size=n_samples)
        instrument = rng.normal(0.0, 1.0, size=n_samples)
        propensity = 1.0 / (1.0 + np.exp(-(0.8 * x1 - 0.6 * x2 + 0.9 * instrument)))
        treatment = rng.binomial(1, propensity)
        heterogeneous_effect = true_ate + 0.5 * x1
        outcome = 2.0 + 1.8 * x1 - 1.2 * x2 + heterogeneous_effect * treatment + rng.normal(0.0, 1.0, size=n_samples)
        return pd.DataFrame(
            {
                "confounder_1": x1,
                "confounder_2": x2,
                "instrument": instrument,
                "treatment": treatment,
                "outcome": outcome,
            }
        )
