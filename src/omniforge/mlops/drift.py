"""Evidently-style Data, Prediction, and Concept Drift Detection (PSI, KS, Wasserstein, JS, Chi2, Page-Hinkley)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from omniforge.core.logging import get_logger
from omniforge.core.types import DriftStatus

logger = get_logger(__name__)


class FeatureDriftResult(BaseModel):
    """Drift diagnostic metrics for an individual feature."""

    column: str
    dtype: str
    status: DriftStatus
    psi: float | None = None
    ks_statistic: float | None = None
    ks_p_value: float | None = None
    wasserstein_distance: float | None = None
    js_divergence: float | None = None
    chi2_p_value: float | None = None


class DriftReport(BaseModel):
    """Dataset-level drift evaluation across all monitored columns."""

    overall_status: DriftStatus
    drifted_feature_count: int
    total_features: int
    drifted_feature_ratio: float
    features: list[FeatureDriftResult] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    evaluated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class DriftDetector:
    """Computes Population Stability Index (PSI), KS, Wasserstein, JS Divergence, Chi2, and Concept Drift."""

    def __init__(
        self,
        *,
        psi_warning: float = 0.10,
        psi_critical: float = 0.25,
        ks_alpha: float = 0.05,
        n_bins: int = 10,
    ) -> None:
        self.psi_warning = psi_warning
        self.psi_critical = psi_critical
        self.ks_alpha = ks_alpha
        self.n_bins = max(4, int(n_bins))

    def detect(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> DriftReport:
        """Compare ``current_df`` against ``reference_df`` across all shared columns."""
        shared_cols = [c for c in reference_df.columns if c in current_df.columns]
        feature_results: list[FeatureDriftResult] = []

        for col in shared_cols:
            ref_s = reference_df[col].dropna()
            cur_s = current_df[col].dropna()
            if len(ref_s) < 5 or len(cur_s) < 5:
                continue

            if pd.api.types.is_numeric_dtype(ref_s) and not pd.api.types.is_bool_dtype(ref_s):
                r_vals = ref_s.astype(float).to_numpy()
                c_vals = pd.to_numeric(cur_s, errors="coerce").dropna().to_numpy(dtype=float)
                if len(c_vals) < 5:
                    continue
                psi = self.compute_psi(r_vals, c_vals, n_bins=self.n_bins)
                ks_res = stats.ks_2samp(r_vals, c_vals)
                w_dist = float(stats.wasserstein_distance(r_vals, c_vals))
                js_div = self._compute_js_numeric(r_vals, c_vals, n_bins=self.n_bins)

                if psi >= self.psi_critical:
                    status = DriftStatus.CRITICAL
                elif psi >= self.psi_warning or float(ks_res.pvalue) < self.ks_alpha:
                    status = DriftStatus.WARNING
                else:
                    status = DriftStatus.HEALTHY

                feature_results.append(
                    FeatureDriftResult(
                        column=str(col),
                        dtype=str(ref_s.dtype),
                        status=status,
                        psi=round(psi, 6),
                        ks_statistic=round(float(ks_res.statistic), 6),
                        ks_p_value=round(float(ks_res.pvalue), 6),
                        wasserstein_distance=round(w_dist, 6),
                        js_divergence=round(js_div, 6),
                    )
                )
            else:
                r_counts = ref_s.astype(str).value_counts(normalize=True)
                c_counts = cur_s.astype(str).value_counts(normalize=True)
                cats = sorted(set(r_counts.index) | set(c_counts.index))
                p_ref = np.array([max(float(r_counts.get(k, 0.0)), 1e-6) for k in cats])
                p_cur = np.array([max(float(c_counts.get(k, 0.0)), 1e-6) for k in cats])
                p_ref /= p_ref.sum()
                p_cur /= p_cur.sum()
                psi = float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))
                # Chi-square on scaled expected counts
                obs = p_cur * len(cur_s)
                exp = p_ref * len(cur_s)
                _, p_val = stats.chisquare(obs, f_exp=exp)
                status = (
                    DriftStatus.CRITICAL
                    if psi >= self.psi_critical
                    else (DriftStatus.WARNING if (psi >= self.psi_warning or p_val < self.ks_alpha) else DriftStatus.HEALTHY)
                )
                feature_results.append(
                    FeatureDriftResult(
                        column=str(col),
                        dtype=str(ref_s.dtype),
                        status=status,
                        psi=round(psi, 6),
                        chi2_p_value=round(float(p_val), 6),
                    )
                )

        drifted = [f for f in feature_results if f.status != DriftStatus.HEALTHY]
        critical = [f for f in feature_results if f.status == DriftStatus.CRITICAL]
        total = max(len(feature_results), 1)
        ratio = len(drifted) / total

        if critical or ratio >= 0.4:
            overall = DriftStatus.CRITICAL
        elif drifted:
            overall = DriftStatus.WARNING
        else:
            overall = DriftStatus.HEALTHY

        recs: list[str] = []
        if overall == DriftStatus.CRITICAL:
            recs.append(
                f"Critical drift detected in {[f.column for f in critical] or [f.column for f in drifted[:5]]}. Trigger continuous retraining."
            )
        elif overall == DriftStatus.WARNING:
            recs.append("Moderate distribution shift detected. Increase monitoring frequency and inspect upstream feeds.")
        else:
            recs.append("No statistically material distribution drift detected.")

        logger.info("drift_evaluated", status=overall.value, drifted=len(drifted), total=len(feature_results))
        return DriftReport(
            overall_status=overall,
            drifted_feature_count=len(drifted),
            total_features=len(feature_results),
            drifted_feature_ratio=round(ratio, 4),
            features=feature_results,
            recommendations=recs,
        )

    @staticmethod
    def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """Compute Population Stability Index (PSI) using reference quantile bin edges."""
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(reference, quantiles))
        if len(edges) < 3:
            return 0.0
        edges[0] = -np.inf
        edges[-1] = np.inf
        ref_hist, _ = np.histogram(reference, bins=edges)
        cur_hist, _ = np.histogram(current, bins=edges)
        ref_p = np.clip(ref_hist / max(len(reference), 1), 1e-6, 1.0)
        cur_p = np.clip(cur_hist / max(len(current), 1), 1e-6, 1.0)
        return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))

    @staticmethod
    def _compute_js_numeric(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        lo = min(float(np.min(reference)), float(np.min(current)))
        hi = max(float(np.max(reference)), float(np.max(current)))
        if hi <= lo:
            return 0.0
        bins = np.linspace(lo, hi, n_bins + 1)
        p, _ = np.histogram(reference, bins=bins)
        q, _ = np.histogram(current, bins=bins)
        p_n = np.clip(p / max(p.sum(), 1), 1e-9, 1.0)
        q_n = np.clip(q / max(q.sum(), 1), 1e-9, 1.0)
        m = 0.5 * (p_n + q_n)
        return float(0.5 * np.sum(p_n * np.log(p_n / m)) + 0.5 * np.sum(q_n * np.log(q_n / m)))

    @staticmethod
    def detect_concept_drift_page_hinkley(
        error_stream: list[float] | np.ndarray, *, delta: float = 0.005, threshold: float = 5.0
    ) -> dict[str, Any]:
        """Run sequential Page-Hinkley change-point test over streaming model error values."""
        errors = np.asarray(error_stream, dtype=float).ravel()
        mean_err = 0.0
        cum_sum = 0.0
        min_cum = 0.0
        drift_points: list[int] = []

        for t, val in enumerate(errors, start=1):
            mean_err += (val - mean_err) / t
            cum_sum += val - mean_err - delta
            min_cum = min(min_cum, cum_sum)
            if (cum_sum - min_cum) > threshold:
                drift_points.append(t - 1)
                cum_sum = 0.0
                min_cum = 0.0

        return {
            "drift_detected": len(drift_points) > 0,
            "drift_indices": drift_points,
            "n_observations": len(errors),
        }
