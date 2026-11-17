"""Synthetic data generator supporting Gaussian Copula synthesis, DP noise, and multi-task datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from omniforge.core.constants import DEFAULT_RANDOM_SEED
from omniforge.core.logging import get_logger

logger = get_logger(__name__)


class SyntheticDataGenerator:
    """Generates privacy-preserving synthetic tabular data and domain benchmark datasets."""

    def __init__(self, random_seed: int = DEFAULT_RANDOM_SEED, random_state: int | None = None) -> None:
        self.random_seed = random_state if random_state is not None else random_seed

    def fit_generate_copula(
        self,
        reference_df: pd.DataFrame,
        n_samples: int | None = None,
        *,
        dp_epsilon: float | None = None,
    ) -> pd.DataFrame:
        """Synthesize tabular data preserving marginal distributions and rank correlations via Gaussian Copula.

        Args:
            reference_df: Real source DataFrame to model.
            n_samples: Number of synthetic rows to generate (defaults to ``len(reference_df)``).
            dp_epsilon: Optional differential-privacy epsilon adding calibrated Laplace noise.

        Returns:
            Synthetic DataFrame matching schema and statistical structure of ``reference_df``.
        """
        rng = np.random.default_rng(self.random_seed)
        n = n_samples or len(reference_df)
        if n <= 0 or len(reference_df) == 0:
            return pd.DataFrame(columns=reference_df.columns)

        num_cols = [
            c
            for c in reference_df.select_dtypes(include=[np.number]).columns
            if not pd.api.types.is_bool_dtype(reference_df[c])
        ]
        other_cols = [c for c in reference_df.columns if c not in num_cols]
        out = pd.DataFrame(index=range(n))

        if num_cols:
            # Map each numeric column to normal scores via empirical CDF
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
                corr = np.corrcoef(z_matrix, rowvar=False)
                corr = np.nan_to_num(corr, nan=0.0)
                np.fill_diagonal(corr, 1.0)
                # Ensure positive semi-definiteness via eigenvalue clipping
                eigvals, eigvecs = np.linalg.eigh(corr)
                eigvals = np.clip(eigvals, 1e-5, None)
                corr = (eigvecs * eigvals) @ eigvecs.T
                d = np.sqrt(np.diag(corr))
                corr = corr / np.outer(d, d)

            sampled_z = rng.multivariate_normal(
                mean=np.zeros(len(num_cols)), cov=corr, size=n
            )
            sampled_u = stats.norm.cdf(sampled_z)

            for idx, col in enumerate(num_cols):
                orig_sorted = sorted_originals[str(col)]
                synth_vals = np.quantile(orig_sorted, sampled_u[:, idx])
                if dp_epsilon is not None and dp_epsilon > 0:
                    sensitivity = float(np.ptp(orig_sorted)) / max(len(orig_sorted), 1)
                    noise_scale = max(sensitivity / dp_epsilon, 1e-9)
                    synth_vals = synth_vals + rng.laplace(0.0, noise_scale, size=n)
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
                    ts_int = non_null.astype("int64").to_numpy()
                    sampled_ts = rng.uniform(float(ts_int.min()), float(ts_int.max()), size=n).astype("int64")
                    out[col] = pd.to_datetime(sampled_ts)
            else:
                counts = series.value_counts(dropna=False)
                probs = (counts / counts.sum()).to_numpy(dtype=float)
                choices = counts.index.to_numpy()
                sampled_idx = rng.choice(len(choices), size=n, p=probs)
                out[col] = [choices[i] for i in sampled_idx]

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
