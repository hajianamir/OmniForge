"""Evidently-style Data, Prediction, and Concept Drift Detection (PSI, KS, Wasserstein, JS, Chi2, Page-Hinkley)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

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
                r_raw = ref_s.astype(str).value_counts()
                c_raw = cur_s.astype(str).value_counts()
                cats = sorted(set(r_raw.index) | set(c_raw.index))
                r_freq = np.array([float(r_raw.get(k, 0)) / max(len(ref_s), 1) for k in cats], dtype=float)
                c_freq = np.array([float(c_raw.get(k, 0)) / max(len(cur_s), 1) for k in cats], dtype=float)
                p_ref = np.clip(r_freq, 1e-6, 1.0)
                p_cur = np.clip(c_freq, 1e-6, 1.0)
                p_ref /= p_ref.sum()
                p_cur /= p_cur.sum()
                psi = float(max(0.0, np.sum((p_cur - p_ref) * np.log(p_cur / p_ref))))
                # Two-sample Chi-square test of homogeneity on observed raw counts
                cont = np.vstack(
                    [
                        [float(r_raw.get(k, 0)) for k in cats],
                        [float(c_raw.get(k, 0)) for k in cats],
                    ]
                )
                if cont.shape[1] >= 2 and np.all(cont.sum(axis=1) > 0):
                    _, p_val, _, _ = stats.chi2_contingency(cont)
                else:
                    p_val = 1.0
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

        if len(feature_results) == 0:
            overall = DriftStatus.UNKNOWN
        elif critical or ratio >= 0.4:
            overall = DriftStatus.CRITICAL
        elif drifted:
            overall = DriftStatus.WARNING
        else:
            overall = DriftStatus.HEALTHY

        recs: list[str] = []
        if overall == DriftStatus.UNKNOWN:
            recs.append("Insufficient shared columns or valid samples (>= 5) to evaluate drift.")
        elif overall == DriftStatus.CRITICAL:
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
    def compute_jensen_shannon_divergence(
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Jensen-Shannon Divergence (base-2, bounded in [0, 1]) between two 1-D arrays."""
        ref = np.asarray(reference, dtype=float).ravel()
        cur = np.asarray(current, dtype=float).ravel()
        ref = ref[np.isfinite(ref)]
        cur = cur[np.isfinite(cur)]
        if len(ref) == 0 or len(cur) == 0:
            return 0.0
        combined = np.concatenate([ref, cur])
        quantiles = np.linspace(0.0, 1.0, max(2, int(n_bins)) + 1)
        edges = np.unique(np.quantile(combined, quantiles))
        if len(edges) < 2:
            return 0.0
        ref_counts, _ = np.histogram(ref, bins=edges)
        cur_counts, _ = np.histogram(cur, bins=edges)
        p = ref_counts.astype(float) + 1e-9
        q = cur_counts.astype(float) + 1e-9
        p /= p.sum()
        q /= q.sum()
        m = 0.5 * (p + q)
        jsd = 0.5 * float(np.sum(p * np.log2(p / m))) + 0.5 * float(np.sum(q * np.log2(q / m)))
        return float(np.clip(jsd, 0.0, 1.0))

    @staticmethod
    def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """Compute Population Stability Index (PSI) with open outer bins, binary/constant support, and renormalized probabilities."""
        ref = np.asarray(reference, dtype=float).ravel()
        cur = np.asarray(current, dtype=float).ravel()
        ref = ref[np.isfinite(ref)]
        cur = cur[np.isfinite(cur)]
        if len(ref) == 0 or len(cur) == 0:
            return 0.0

        uniq_ref = np.unique(ref)
        if len(uniq_ref) == 1:
            # Constant reference distribution: compare mass at constant value vs shifted values
            c0 = float(uniq_ref[0])
            cur_match = float(np.mean(np.isclose(cur, c0, rtol=1e-9, atol=1e-12)))
            ref_p = np.array([1.0, 1e-6], dtype=float)
            cur_p = np.array([max(cur_match, 1e-6), max(1.0 - cur_match, 1e-6)], dtype=float)
            ref_p /= ref_p.sum()
            cur_p /= cur_p.sum()
            return float(max(0.0, np.sum((cur_p - ref_p) * np.log(cur_p / ref_p))))

        quantiles = np.linspace(0.0, 1.0, max(2, int(n_bins)) + 1)
        edges = np.unique(np.quantile(ref, quantiles))
        if len(edges) == 2:
            # Binary / two-value numeric feature: split cleanly at midpoint so shifts are detected
            mid = 0.5 * (float(edges[0]) + float(edges[1]))
            edges = np.array([-np.inf, mid, np.inf], dtype=float)
        else:
            edges[0] = -np.inf
            edges[-1] = np.inf

        ref_hist, _ = np.histogram(ref, bins=edges)
        cur_hist, _ = np.histogram(cur, bins=edges)
        ref_p = np.clip(ref_hist / float(len(ref)), 1e-6, 1.0)
        cur_p = np.clip(cur_hist / float(len(cur)), 1e-6, 1.0)
        # Renormalize after epsilon floor so both distributions strictly sum to 1 (non-negative divergence)
        ref_p /= ref_p.sum()
        cur_p /= cur_p.sum()
        return float(max(0.0, np.sum((cur_p - ref_p) * np.log(cur_p / ref_p))))

    @staticmethod
    def _compute_js_numeric(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        ref = np.asarray(reference, dtype=float).ravel()
        cur = np.asarray(current, dtype=float).ravel()
        ref = ref[np.isfinite(ref)]
        cur = cur[np.isfinite(cur)]
        if len(ref) == 0 or len(cur) == 0:
            return 0.0
        lo = min(float(np.min(ref)), float(np.min(cur)))
        hi = max(float(np.max(ref)), float(np.max(cur)))
        if hi <= lo:
            return 0.0
        bins = np.linspace(lo, hi, max(2, int(n_bins)) + 1)
        p, _ = np.histogram(ref, bins=bins)
        q, _ = np.histogram(cur, bins=bins)
        p_n = np.clip(p / max(float(p.sum()), 1.0), 1e-9, 1.0)
        q_n = np.clip(q / max(float(q.sum()), 1.0), 1e-9, 1.0)
        p_n /= p_n.sum()
        q_n /= q_n.sum()
        m = 0.5 * (p_n + q_n)
        js = float(0.5 * np.sum(p_n * np.log(p_n / m)) + 0.5 * np.sum(q_n * np.log(q_n / m)))
        return float(np.clip(js, 0.0, np.log(2.0)))

    @staticmethod
    def detect_concept_drift_page_hinkley(
        error_stream: list[float] | np.ndarray,
        *,
        delta: float = 0.005,
        threshold: float = 5.0,
        alpha: float = 1.0,
        min_instances: int = 10,
        mode: Literal["up", "down", "both"] = "up",
    ) -> dict[str, Any]:
        """Run sequential Page-Hinkley change-point test (Page 1954; Gama et al. 2014; River).

        Resets the running mean, sample counter, and cumulative extrema upon each detected
        change-point so a single regime shift does not cause cascading false alarms.
        """
        errors = np.asarray(error_stream, dtype=float).ravel()
        alpha_f = float(np.clip(alpha, 0.9, 1.0))
        min_n = max(1, int(min_instances))

        sample_count = 0
        mean_err = 0.0
        cum_up = 0.0
        min_cum_up = 0.0
        cum_down = 0.0
        max_cum_down = 0.0
        drift_points: list[int] = []
        drift_directions: list[str] = []

        for idx, val in enumerate(errors):
            if not np.isfinite(val):
                continue
            sample_count += 1
            mean_err += (float(val) - mean_err) / float(sample_count)

            # Upward mean shift monitor: m_t = alpha * m_{t-1} + (x_t - x_bar_t - delta), PH_t = m_t - min(m)
            cum_up = alpha_f * cum_up + (float(val) - mean_err - delta)
            min_cum_up = min(min_cum_up, cum_up)
            ph_up = cum_up - min_cum_up

            # Downward mean shift monitor: d_t = alpha * d_{t-1} + (x_t - x_bar_t + delta), PH_down = max(d) - d_t
            cum_down = alpha_f * cum_down + (float(val) - mean_err + delta)
            max_cum_down = max(max_cum_down, cum_down)
            ph_down = max_cum_down - cum_down

            if sample_count < min_n:
                continue

            triggered_up = mode in ("up", "both") and ph_up > threshold
            triggered_down = mode in ("down", "both") and ph_down > threshold
            if triggered_up or triggered_down:
                drift_points.append(idx)
                drift_directions.append("up" if triggered_up else "down")
                # Full state reset (mean, counter, extrema) after confirmed regime change
                sample_count = 0
                mean_err = 0.0
                cum_up = 0.0
                min_cum_up = 0.0
                cum_down = 0.0
                max_cum_down = 0.0

        return {
            "drift_detected": len(drift_points) > 0,
            "drift_indices": drift_points,
            "drift_directions": drift_directions,
            "n_observations": len(errors),
        }

    @staticmethod
    def detect_concept_drift_adwin(
        stream: np.ndarray | list[float],
        delta: float = 0.002,
        min_window_length: int = 10,
    ) -> dict[str, Any]:
        """ADWIN (Bifet & Gavalda, 2007) adaptive windowing concept drift detector."""
        arr = np.asarray(stream, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if len(arr) < max(4, min_window_length):
            return {
                "drift_detected": False,
                "drift_indices": [],
                "n_observations": int(len(arr)),
                "final_window_size": int(len(arr)),
            }

        window: list[float] = []
        drift_indices: list[int] = []
        for idx, val in enumerate(arr):
            window.append(float(val))
            n = len(window)
            if n < min_window_length:
                continue
            w_arr = np.asarray(window, dtype=float)
            prefix_sum = np.cumsum(w_arr)
            total_sum = float(prefix_sum[-1])
            for cut in range(1, n):
                n0 = cut
                n1 = n - cut
                if n0 < 2 or n1 < 2:
                    continue
                mu0 = float(prefix_sum[cut - 1]) / n0
                mu1 = (total_sum - float(prefix_sum[cut - 1])) / n1
                m = 1.0 / (1.0 / n0 + 1.0 / n1)
                ddof = np.log(4.0 * n / max(delta, 1e-12))
                eps_cut = float(np.sqrt((1.0 / (2.0 * m)) * ddof))
                if abs(mu0 - mu1) >= eps_cut:
                    drift_indices.append(idx)
                    window = window[cut:]
                    break

        return {
            "drift_detected": len(drift_indices) > 0,
            "drift_indices": drift_indices,
            "n_observations": int(len(arr)),
            "final_window_size": int(len(window)),
        }

