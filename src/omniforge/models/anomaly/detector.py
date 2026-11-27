"""Leakage-Safe Multi-Algorithm Anomaly Detection Suite: Isolation Forest, LOF (novelty=True), OCSVM, PCA, MAD, Z-Score, Causal EWMA, Ensemble."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from omniforge.core.exceptions import ModelTrainingError
from omniforge.core.performance import parallel_n_jobs
from omniforge.core.types import TaskType
from omniforge.models.base import BaseOmniForgeModel


class AnomalyDetector(BaseOmniForgeModel):
    """Unified tabular and causal time-series anomaly detector supporting Train-fitted inductive scoring."""

    model_family = "anomaly_detection"

    def __init__(
        self,
        method: str = "ensemble",  # isolation_forest, lof, ocsvm, pca_recon, mad, zscore, ensemble
        *,
        contamination: float = 0.05,
        **kwargs: Any,
    ) -> None:
        normalized_method = method.strip().lower()
        if normalized_method not in {"isolation_forest", "lof", "ocsvm", "pca_recon", "mad", "zscore", "ensemble"}:
            raise ModelTrainingError("Unsupported anomaly detection method.")
        super().__init__(
            name=f"Anomaly_{normalized_method}",
            task_type=TaskType.ANOMALY_DETECTION,
            method=normalized_method,
            contamination=contamination,
            **kwargs,
        )
        self._constructor_params = {"method": method}
        self.method = normalized_method
        self.contamination = float(np.clip(contamination, 0.001, 0.49))
        self._scaler = StandardScaler()
        self._detectors: dict[str, Any] = {}
        self._train_score_refs: dict[str, np.ndarray] = {}
        self._pca: PCA | None = None
        self._medians: np.ndarray | None = None
        self._mads: np.ndarray | None = None
        self._means: np.ndarray | None = None
        self._stds: np.ndarray | None = None
        self.threshold_: float = 0.0

    def _fit(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs: Any) -> None:
        _ = (y, kwargs)
        X_s = self._scaler.fit_transform(X)
        if self.method == "ensemble":
            methods = ["isolation_forest", "lof", "pca_recon", "mad"]
        else:
            methods = [self.method]
        self._detectors = {}
        self._train_score_refs = {}

        for m in methods:
            if m == "isolation_forest":
                self._detectors[m] = IsolationForest(
                    contamination=self.contamination,
                    random_state=self.random_seed,
                    n_estimators=150,
                    n_jobs=parallel_n_jobs(),
                ).fit(X_s)
            elif m == "lof":
                k = min(20, max(2, len(X_s) - 1))
                self._detectors[m] = LocalOutlierFactor(
                    n_neighbors=k,
                    contamination=self.contamination,
                    novelty=True,
                    n_jobs=parallel_n_jobs(),
                ).fit(X_s)
            elif m == "ocsvm":
                fit_X = X_s
                if len(X_s) > 12_000:
                    rng = np.random.default_rng(self.random_seed)
                    fit_X = X_s[rng.choice(len(X_s), size=12_000, replace=False)]
                self._detectors[m] = OneClassSVM(nu=self.contamination, kernel="rbf").fit(fit_X)
            elif m == "pca_recon":
                n_comp = max(1, min(X_s.shape[1] - 1, max(1, X_s.shape[1] // 2)))
                self._pca = PCA(n_components=n_comp, random_state=self.random_seed).fit(X_s)
                self._detectors[m] = self._pca
            elif m == "mad":
                self._medians = np.median(X_s, axis=0)
                self._mads = np.maximum(np.median(np.abs(X_s - self._medians), axis=0), 1e-6)
                self._detectors[m] = "mad"
            elif m == "zscore":
                self._means = np.mean(X_s, axis=0)
                self._stds = np.maximum(np.std(X_s, axis=0, ddof=1 if len(X_s) > 1 else 0), 1e-6)
                self._detectors[m] = "zscore"

        # Cache sorted training score empirical CDFs for strictly inductive, zero-leakage rank calibration
        for m in self._detectors:
            raw_train = self._raw_component_score(m, X_s)
            self._train_score_refs[m] = np.sort(raw_train)

        train_scores = self._score_scaled(X_s)
        self.threshold_ = float(np.quantile(train_scores, 1.0 - self.contamination))

    def _raw_component_score(self, m: str, X_s: np.ndarray) -> np.ndarray:
        """Compute uncalibrated severity score (higher = more anomalous) for component ``m``."""
        det = self._detectors[m]
        if m in ("isolation_forest", "lof", "ocsvm"):
            return -np.asarray(det.score_samples(X_s), dtype=float)
        if m == "pca_recon" and self._pca is not None:
            recon = self._pca.inverse_transform(self._pca.transform(X_s))
            return np.asarray(np.mean((X_s - recon) ** 2, axis=1), dtype=float)
        if m == "mad" and self._medians is not None and self._mads is not None:
            z = np.abs(X_s - self._medians) / (1.4826 * self._mads)
            return np.asarray(np.max(z, axis=1), dtype=float)
        if m == "zscore" and self._means is not None and self._stds is not None:
            z = np.abs(X_s - self._means) / self._stds
            return np.asarray(np.max(z, axis=1), dtype=float)
        return np.zeros(len(X_s), dtype=float)

    def anomaly_scores(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Compute continuous Train-calibrated anomaly severity scores (higher = more anomalous)."""
        self._check_is_fitted()
        X_arr, _ = self._prepare_X(X, fitting=False)
        return self._score_scaled(self._scaler.transform(X_arr))

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Alias for :meth:`anomaly_scores` matching scikit-learn conventions."""
        return self.anomaly_scores(X)

    def _score_scaled(self, X_s: np.ndarray) -> np.ndarray:
        score_blocks: list[np.ndarray] = []
        multi = len(self._detectors) > 1
        for m in self._detectors:
            raw = self._raw_component_score(m, X_s)
            if multi:
                # Calibrate against Train Empirical CDF (zero batch/Test leakage; valid even for n=1!)
                ref = self._train_score_refs.get(m)
                if ref is not None and len(ref) > 0:
                    ecdf = np.searchsorted(ref, raw, side="right") / float(len(ref))
                    score_blocks.append(ecdf)
                else:
                    score_blocks.append(raw)
            else:
                score_blocks.append(raw)
        return np.mean(score_blocks, axis=0) if score_blocks else np.zeros(len(X_s))

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Return binary anomaly predictions (1 = anomaly, 0 = normal) using Train-calibrated threshold."""
        _ = kwargs
        scores = self._score_scaled(self._scaler.transform(X))
        return (scores > self.threshold_).astype(int)

    def detect_split(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        *,
        feature_cols: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        """Leakage-safe detection: fit detector strictly on ``df_train``, then score Train and unseen Test."""
        cols = feature_cols or list(df_train.select_dtypes(include=[np.number]).columns)
        self.fit(df_train[cols])

        tr_scores = self.anomaly_scores(df_train[cols])
        tr_flags = (tr_scores > self.threshold_).astype(bool)
        te_scores = self.anomaly_scores(df_test[cols])
        te_flags = (te_scores > self.threshold_).astype(bool)

        tr_out = df_train.copy()
        tr_out["anomaly_score"] = np.round(tr_scores, 6)
        tr_out["is_anomaly"] = tr_flags

        te_out = df_test.copy()
        te_out["anomaly_score"] = np.round(te_scores, 6)
        te_out["is_anomaly"] = te_flags

        meta = {
            "method": self.method,
            "contamination": self.contamination,
            "threshold": round(self.threshold_, 6),
            "feature_cols": cols,
            "train_anomalies": int(tr_flags.sum()),
            "test_anomalies": int(te_flags.sum()),
            "leakage_safe_fit_on_train_only": True,
        }
        return tr_out, te_out, meta

    @staticmethod
    def detect_time_series_anomalies(
        series: pd.Series | np.ndarray,
        *,
        span: int = 12,
        window: int | None = None,
        z_threshold: float = 3.0,
    ) -> pd.DataFrame:
        """Strictly causal time-series anomaly detection using backward-looking EWMA & shifted rolling MAD (zero look-ahead)."""
        raw = pd.Series(series, dtype=float)
        # Strictly causal past-only forward-fill (never bfill from future!)
        first_valid = float(raw.dropna().iloc[0]) if raw.notna().any() else 0.0
        s = raw.ffill().fillna(first_valid)

        # Predict t from strictly past EWMA(t-1) to prevent point t from masking itself
        ewma_expected = s.ewm(span=max(2, span), adjust=False).mean().shift(1).fillna(s.iloc[0] if len(s) > 0 else 0.0)
        resid = s - ewma_expected

        # Strictly backward-looking rolling/expanding median & MAD shifted by 1 step (no future peeking!)
        win = max(6, int(window or (span * 2)))
        min_p = min(4, max(2, len(s) // 5))
        past_resid = resid.shift(1)
        def _window_mad(w: np.ndarray) -> float:
            med = float(np.median(w))
            return float(np.median(np.abs(w - med)))

        roll_med = past_resid.rolling(window=win, min_periods=min_p, center=False).median()
        roll_mad = past_resid.rolling(window=win, min_periods=min_p, center=False).apply(_window_mad, raw=True)
        # Warm-up fallback using strictly expanding past residuals (never future!)
        exp_med = past_resid.expanding(min_periods=1).median().fillna(0.0)
        exp_mad = past_resid.expanding(min_periods=1).apply(_window_mad, raw=True)
        med_series = roll_med.fillna(exp_med).fillna(0.0)
        mad_series = roll_mad.fillna(exp_mad).fillna(1e-6).clip(lower=1e-6)

        mod_z = (resid - med_series).abs() / (1.4826 * mad_series)
        # Do not falsely flag t=0 warm-up point
        if len(mod_z) > 0:
            mod_z.iloc[0] = 0.0

        return pd.DataFrame(
            {
                "value": s.values,
                "ewma_expected": ewma_expected.values,
                "residual": resid.values,
                "anomaly_score": mod_z.values,
                "is_anomaly": (mod_z > z_threshold).astype(int).values,
            }
        )
