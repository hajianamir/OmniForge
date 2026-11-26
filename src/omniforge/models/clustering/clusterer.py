"""Unsupervised Clustering & Natural Segmentation Engine (KMeans, Hierarchical, GMM, DBSCAN, OPTICS, BIRCH)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AgglomerativeClustering,
    Birch,
    KMeans,
    MiniBatchKMeans,
)
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from omniforge.core.exceptions import ModelTrainingError
from omniforge.core.performance import parallel_n_jobs
from omniforge.core.types import TaskType
from omniforge.evaluation.metrics import MetricsEvaluator
from omniforge.models.base import BaseOmniForgeModel

SUPPORTED_CLUSTERING_ALGORITHMS: tuple[str, ...] = (
    "kmeans",
    "minibatch_kmeans",
    "agglomerative",
    "gmm",
    "dbscan",
    "optics",
    "birch",
)

CLUSTERING_ALGORITHM_LABELS: dict[str, str] = {
    "kmeans": "K-Means (Centroid Partitioning)",
    "minibatch_kmeans": "MiniBatch K-Means (Large-Scale Fast Partitioning)",
    "agglomerative": "Agglomerative Hierarchical (Ward / Linkage Trees)",
    "gmm": "Gaussian Mixture Model (Probabilistic Soft Clusters)",
    "dbscan": "DBSCAN (Density-Based Arbitrary Shapes + Noise)",
    "optics": "OPTICS (Variable-Density Reachability Clustering)",
    "birch": "BIRCH (Incremental CF-Tree Hierarchical Clustering)",
}


@dataclass
class ClusteringResult:
    """Comprehensive output of an unsupervised clustering and segmentation run."""

    algorithm: str
    n_clusters_found: int
    noise_count: int
    noise_ratio: float
    labels: np.ndarray
    metrics: dict[str, float]
    feature_columns: list[str]
    cluster_profiles: list[dict[str, Any]] = field(default_factory=list)
    feature_separation_tests: list[dict[str, Any]] = field(default_factory=list)
    pca_2d: pd.DataFrame | None = None
    pca_explained_variance: tuple[float, float] = (0.0, 0.0)
    k_selection_curve: list[dict[str, Any]] = field(default_factory=list)
    algorithm_comparison: list[dict[str, Any]] = field(default_factory=list)
    probabilities: np.ndarray | None = None


class ClusteringEngine(BaseOmniForgeModel):
    """Enterprise unsupervised clustering suite with Auto-K selection, algorithm benchmarking, and statistical profiling.

    Supported algorithms:
        - ``kmeans``: Lloyd/Elkan K-Means++ partitioning.
        - ``minibatch_kmeans``: Mini-Batch K-Means for high-volume datasets.
        - ``agglomerative``: Hierarchical Agglomerative Clustering (Ward / Complete / Average / Single linkage).
        - ``gmm``: Probabilistic Gaussian Mixture Model with soft probabilities & AIC/BIC.
        - ``dbscan``: Density-Based Spatial Clustering with automatic k-distance epsilon heuristic.
        - ``optics``: Multi-density reachability clustering (OPTICS).
        - ``birch``: Memory-efficient hierarchical clustering (BIRCH).
    """

    model_family = "clustering"

    def __init__(
        self,
        algorithm: str = "kmeans",
        n_clusters: int = 4,
        *,
        scaler: Literal["standard", "robust", "minmax", "none"] = "standard",
        linkage: Literal["ward", "complete", "average", "single"] = "ward",
        eps: float | None = None,
        min_samples: int = 5,
        covariance_type: Literal["full", "tied", "diag", "spherical"] = "full",
        random_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        algo = algorithm.lower().strip()
        if algo not in SUPPORTED_CLUSTERING_ALGORITHMS:
            raise ModelTrainingError(
                f"Unsupported clustering algorithm '{algorithm}'.",
                suggestions=[f"Choose one of: {', '.join(SUPPORTED_CLUSTERING_ALGORITHMS)}"],
            )
        super().__init__(
            name=f"Clustering_{algo}",
            task_type=TaskType.CLUSTERING,
            random_seed=random_seed,
            algorithm=algo,
            n_clusters=n_clusters,
            scaler=scaler,
            linkage=linkage,
            eps=eps,
            min_samples=min_samples,
            covariance_type=covariance_type,
            **kwargs,
        )
        self.algorithm = algo
        self.n_clusters = int(max(2, n_clusters))
        self.scaler_type = scaler
        self.linkage = linkage
        self.eps = eps
        self.min_samples = int(max(2, min_samples))
        self.covariance_type = covariance_type
        self._scaler: Any = None
        self._estimator: Any = None
        self._labels_: np.ndarray = np.array([], dtype=int)
        self._X_scaled_: np.ndarray = np.empty((0, 0), dtype=float)
        self._encoded_cols_: list[str] = []
        self._col_medians_: dict[str, float] = {}
        self._array_medians_: np.ndarray | None = None
        self._profile_input_columns_: list[str] = []

    def _make_scaler(self) -> Any:
        if self.scaler_type == "standard":
            return StandardScaler()
        if self.scaler_type == "robust":
            return RobustScaler()
        if self.scaler_type == "minmax":
            return MinMaxScaler()
        return None

    def _prepare_matrix(self, X: pd.DataFrame | np.ndarray, *, fit_scaler: bool) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.rename(columns=str)
            if not X.columns.is_unique:
                raise ModelTrainingError("Clustering feature names must be unique after normalization.")
            if fit_scaler:
                self._profile_input_columns_ = list(X.columns)
                self._col_medians_ = {}
                num_cols = list(X.select_dtypes(include=[np.number]).columns)
            else:
                num_cols = list(self._col_medians_)
            work = X.reindex(columns=self._profile_input_columns_).copy()
            cat_cols = [c for c in work.columns if c not in num_cols]
            for c in num_cols:
                col_vals = pd.to_numeric(work[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
                if fit_scaler:
                    med = float(col_vals.median()) if col_vals.notna().any() else 0.0
                    self._col_medians_[str(c)] = med
                else:
                    med = self._col_medians_[str(c)]
                work[c] = col_vals.fillna(med)
            if cat_cols:
                for c in cat_cols:
                    work[c] = work[c].astype("string").fillna("missing")
                work = pd.get_dummies(work, columns=cat_cols, drop_first=False, dtype=float)
            if fit_scaler:
                self._encoded_cols_ = list(work.columns)
            elif self._encoded_cols_:
                work = work.reindex(columns=self._encoded_cols_, fill_value=0.0)
            arr = work.to_numpy(dtype=float)
        else:
            arr = np.asarray(X, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if fit_scaler:
                self._profile_input_columns_ = []
                with np.errstate(all="ignore"):
                    col_med = np.nanmedian(arr, axis=0)
                col_med = np.where(np.isnan(col_med), 0.0, col_med).astype(float)
                self._array_medians_ = col_med
                if self.feature_names_ and len(self.feature_names_) == len(col_med):
                    self._col_medians_ = {str(name): float(val) for name, val in zip(self.feature_names_, col_med, strict=False)}
            else:
                if self._array_medians_ is not None and len(self._array_medians_) == arr.shape[1]:
                    col_med = self._array_medians_
                else:
                    with np.errstate(all="ignore"):
                        col_med = np.nanmedian(arr, axis=0)
                    col_med = np.where(np.isnan(col_med), 0.0, col_med).astype(float)
            if np.isnan(arr).any():
                inds = np.where(np.isnan(arr))
                arr = arr.copy()
                arr[inds] = np.take(col_med, inds[1])

        arr = np.asarray(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
        if fit_scaler:
            self._scaler = self._make_scaler()
            return np.asarray(self._scaler.fit_transform(arr), dtype=float) if self._scaler is not None else arr
        return np.asarray(self._scaler.transform(arr), dtype=float) if self._scaler is not None else arr

    def _estimate_dbscan_eps(self, X_s: np.ndarray) -> float:
        """Estimate DBSCAN eps via Kneedle curvature knee on the sorted k-NN distance curve (Ester et al. 1996; Satopää et al. 2011)."""
        n = len(X_s)
        k = min(max(self.min_samples, 2), max(n - 1, 1))
        if n <= 3:
            return 0.5
        fit_X = X_s
        if n > 15_000:
            rng = np.random.default_rng(self.random_seed)
            fit_X = X_s[rng.choice(n, size=15_000, replace=False)]
        nn = NearestNeighbors(n_neighbors=k, n_jobs=parallel_n_jobs()).fit(fit_X)
        dists, _ = nn.kneighbors(fit_X)
        k_dists = np.sort(dists[:, -1])
        # Trim extreme top 1% outlier tail so single isolated points do not skew chord geometry
        m_trim = max(3, int(np.ceil(0.99 * len(k_dists))))
        curve = k_dists[:m_trim]
        span = float(curve[-1] - curve[0])
        if span <= 1e-12:
            return max(float(curve[-1]), 1e-3)
        x_norm = np.linspace(0.0, 1.0, len(curve))
        y_norm = (curve - curve[0]) / span
        # Maximum perpendicular distance below chord from (0, 0) to (1, 1)
        knee_idx = int(np.argmax(x_norm - y_norm))
        est = float(curve[knee_idx])
        if est <= 1e-6:
            est = float(np.quantile(k_dists, 0.85))
        return max(est, 1e-3)

    def _build_and_fit(self, X_s: np.ndarray) -> tuple[Any, np.ndarray]:
        n = len(X_s)
        k = min(self.n_clusters, max(2, n - 1)) if n > 2 else 1
        if self.algorithm == "kmeans":
            if n > 25_000:
                est = MiniBatchKMeans(n_clusters=k, n_init="auto", batch_size=2048, random_state=self.random_seed)
            else:
                est = KMeans(n_clusters=k, n_init="auto", random_state=self.random_seed)
            labels = est.fit_predict(X_s)
        elif self.algorithm == "minibatch_kmeans":
            est = MiniBatchKMeans(n_clusters=k, n_init="auto", batch_size=2048, random_state=self.random_seed)
            labels = est.fit_predict(X_s)
        elif self.algorithm == "agglomerative":
            if n > 10_000:
                # Subsample for O(N^2) Agglomerative fit, then assign full dataset to nearest cluster centroid
                rng = np.random.default_rng(self.random_seed)
                sub_idx = rng.choice(n, size=10_000, replace=False)
                X_sub = X_s[sub_idx]
                est = AgglomerativeClustering(n_clusters=k, linkage=self.linkage)
                sub_labels = est.fit_predict(X_sub)
                uniq = sorted(np.unique(sub_labels))
                centroids = np.vstack([X_sub[sub_labels == c].mean(axis=0) for c in uniq])
                nn = NearestNeighbors(n_neighbors=1, n_jobs=parallel_n_jobs()).fit(centroids)
                _, idx = nn.kneighbors(X_s)
                labels = np.asarray([uniq[int(i)] for i in idx[:, 0]], dtype=int)
            else:
                est = AgglomerativeClustering(n_clusters=k, linkage=self.linkage)
                labels = est.fit_predict(X_s)
        elif self.algorithm == "gmm":
            est = GaussianMixture(
                n_components=k,
                covariance_type=self.covariance_type,
                random_state=self.random_seed,
            )
            labels = est.fit_predict(X_s)
        elif self.algorithm == "dbscan":
            eps_val = self.eps if (self.eps is not None and self.eps > 0) else self._estimate_dbscan_eps(X_s)
            self._fitted_eps_ = float(eps_val)
            est = DBSCAN(eps=eps_val, min_samples=min(self.min_samples, max(2, n // 3)), n_jobs=parallel_n_jobs())
            labels = est.fit_predict(X_s)
        elif self.algorithm == "optics":
            self._fitted_eps_ = self._estimate_dbscan_eps(X_s)
            est = OPTICS(min_samples=min(self.min_samples, max(2, n // 4)), n_jobs=parallel_n_jobs())
            labels = est.fit_predict(X_s)
        else:
            est = Birch(n_clusters=k)
            labels = est.fit_predict(X_s)
        return est, np.asarray(labels, dtype=int)

    def _fit(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs: Any) -> None:
        _ = (y, kwargs)
        X_s = self._prepare_matrix(X, fit_scaler=True)
        self._X_scaled_ = X_s
        self._estimator, self._labels_ = self._build_and_fit(X_s)

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        _ = kwargs
        return self._predict_scaled(self._prepare_matrix(X, fit_scaler=False))

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame) and getattr(self, "_profile_input_columns_", []):
            self._check_is_fitted()
            return self._predict_scaled(self._prepare_matrix(X, fit_scaler=False))
        return super().predict(X, **kwargs)

    def _predict_scaled(self, X_s: np.ndarray) -> np.ndarray:
        if hasattr(self._estimator, "predict"):
            return np.asarray(self._estimator.predict(X_s), dtype=int)
        # Density-aware core-point inductive assignment for DBSCAN / OPTICS (labels out-of-radius points -1)
        if self.algorithm in ("dbscan", "optics"):
            core_idx = getattr(self._estimator, "core_sample_indices_", None)
            if core_idx is not None and len(core_idx) > 0:
                ref_pts = self._X_scaled_[core_idx]
                ref_lbls = self._labels_[core_idx]
            else:
                non_noise = self._labels_ != -1
                if not np.any(non_noise):
                    return np.full(len(X_s), -1, dtype=int)
                ref_pts = self._X_scaled_[non_noise]
                ref_lbls = self._labels_[non_noise]
            eps_radius = float(getattr(self, "_fitted_eps_", 1.0))
            nn = NearestNeighbors(n_neighbors=1).fit(ref_pts)
            dists, idx = nn.kneighbors(X_s)
            assigned = np.where(dists[:, 0] <= eps_radius, ref_lbls[idx[:, 0]], -1)
            return np.asarray(assigned, dtype=int)
        # Nearest-centroid assignment for Agglomerative clustering
        valid_clusters = sorted(c for c in np.unique(self._labels_) if c != -1)
        if not valid_clusters:
            return np.zeros(len(X_s), dtype=int)
        centroids = np.vstack([self._X_scaled_[self._labels_ == c].mean(axis=0) for c in valid_clusters])
        dists = np.linalg.norm(X_s[:, None, :] - centroids[None, :, :], axis=2)
        nearest_idx = np.argmin(dists, axis=1)
        return np.asarray([valid_clusters[i] for i in nearest_idx], dtype=int)

    def fit_predict_split(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        *,
        feature_cols: list[str] | None = None,
        run_auto_k: bool = False,
        k_min: int = 2,
        k_max: int = 8,
    ) -> tuple[ClusteringResult, np.ndarray]:
        """Leakage-safe clustering: fit scaler & clusters strictly on ``df_train``, then inductively predict ``df_test``."""
        train_res = self.fit_predict_with_profile(
            df_train,
            feature_cols=feature_cols,
            run_auto_k=run_auto_k,
            k_min=k_min,
            k_max=k_max,
        )
        test_labels = np.asarray(self.predict(df_test), dtype=int)
        return train_res, test_labels

    def predict_proba_clusters(self, X: pd.DataFrame | np.ndarray) -> np.ndarray | None:
        """Return soft cluster membership probabilities (exact for GMM, softmax distance for centroid models)."""
        self._check_is_fitted()
        if not getattr(self, "_profile_input_columns_", []):
            X, _ = self._prepare_X(X, fitting=False)
        X_s = self._prepare_matrix(X, fit_scaler=False)
        if hasattr(self._estimator, "predict_proba"):
            return np.asarray(self._estimator.predict_proba(X_s), dtype=float)
        return None

    @staticmethod
    def auto_select_k(
        df: pd.DataFrame | np.ndarray,
        *,
        feature_cols: list[str] | None = None,
        k_min: int = 2,
        k_max: int = 9,
        algorithm: Literal["kmeans", "agglomerative", "gmm", "birch"] = "kmeans",
        scaler: Literal["standard", "robust", "minmax", "none"] = "standard",
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """Evaluate candidate K values using Silhouette, Calinski-Harabasz, Davies-Bouldin, Inertia/BIC."""
        X_in = df[feature_cols].copy() if (isinstance(df, pd.DataFrame) and feature_cols) else df
        n_rows = len(X_in)
        upper_k = min(k_max, max(k_min, n_rows - 2))
        if n_rows < 5:
            return {"best_k": 2, "best_silhouette": 0.0, "curve": []}

        helper = ClusteringEngine(algorithm=algorithm, n_clusters=k_min, scaler=scaler, random_seed=random_seed)
        X_s = helper._prepare_matrix(X_in, fit_scaler=True)

        curve: list[dict[str, Any]] = []
        for k in range(k_min, upper_k + 1):
            eng = ClusteringEngine(algorithm=algorithm, n_clusters=k, scaler="none", random_seed=random_seed)
            est, labels = eng._build_and_fit(X_s)
            m = MetricsEvaluator.clustering_metrics(X_s, labels)
            row: dict[str, Any] = {
                "k": k,
                "silhouette": m["silhouette"],
                "calinski_harabasz": m["calinski_harabasz"],
                "davies_bouldin": m["davies_bouldin"],
            }
            if hasattr(est, "inertia_"):
                row["inertia"] = round(float(est.inertia_), 4)
            if hasattr(est, "bic"):
                row["bic"] = round(float(est.bic(X_s)), 4)
                row["aic"] = round(float(est.aic(X_s)), 4)
            curve.append(row)

        if not curve:
            return {"best_k": k_min, "best_silhouette": 0.0, "curve": []}
        if algorithm == "gmm" and "bic" in curve[0]:
            best_row = min(curve, key=lambda r: float(r["bic"]))
        else:
            best_row = max(curve, key=lambda r: float(r["silhouette"]))
        return {
            "best_k": int(best_row["k"]),
            "best_silhouette": round(float(best_row["silhouette"]), 4),
            "curve": curve,
        }

    @staticmethod
    def compute_gap_statistic(
        df: pd.DataFrame | np.ndarray,
        *,
        feature_cols: list[str] | None = None,
        k_max: int = 8,
        n_refs: int = 15,
        scaler: Literal["standard", "robust", "minmax", "none"] = "standard",
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """Compute Tibshirani, Walther & Hastie (2001, JRSS-B) PCA-aligned Gap Statistic & 1-SE optimal K (including K=1)."""
        X_in = df[feature_cols].copy() if (isinstance(df, pd.DataFrame) and feature_cols) else df
        n_rows = len(X_in)
        if n_rows < 4:
            return {"optimal_k": 1, "curve": []}

        helper = ClusteringEngine(algorithm="kmeans", n_clusters=2, scaler=scaler, random_seed=random_seed)
        X_s = helper._prepare_matrix(X_in, fit_scaler=True)
        upper_k = min(max(1, int(k_max)), n_rows - 1)
        b_refs = max(3, int(n_refs))
        rng = np.random.default_rng(random_seed)

        # PCA-aligned uniform bounding box ("Gap/pc", Tibshirani et al. 2001 §3.2) for rotation invariance
        mean_vec = np.mean(X_s, axis=0, keepdims=True)
        X_centered = X_s - mean_vec
        _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
        X_proj = X_centered @ vt.T
        mins = np.min(X_proj, axis=0)
        maxs = np.max(X_proj, axis=0)

        def _within_ss(mat: np.ndarray, k_val: int, seed_val: int) -> float:
            if k_val <= 1:
                centered = mat - np.mean(mat, axis=0, keepdims=True)
                return float(np.sum(centered * centered))
            km = KMeans(n_clusters=k_val, n_init=5, random_state=seed_val)
            km.fit(mat)
            return float(km.inertia_)

        curve: list[dict[str, float | int]] = []
        for k_val in range(1, upper_k + 1):
            wk = max(_within_ss(X_s, k_val, random_seed), 1e-12)
            ref_logs: list[float] = []
            for b in range(b_refs):
                z_unif = rng.uniform(mins, maxs, size=X_proj.shape)
                x_ref = z_unif @ vt + mean_vec
                w_ref = max(_within_ss(x_ref, k_val, random_seed + b + 1), 1e-12)
                ref_logs.append(float(np.log(w_ref)))
            ref_arr = np.asarray(ref_logs, dtype=float)
            gap_k = float(np.mean(ref_arr) - np.log(wk))
            sdk = float(np.std(ref_arr, ddof=0))
            sk = float(np.sqrt(1.0 + 1.0 / b_refs) * sdk)
            curve.append(
                {
                    "k": int(k_val),
                    "log_wk": round(float(np.log(wk)), 6),
                    "expected_log_wk_null": round(float(np.mean(ref_arr)), 6),
                    "gap": round(gap_k, 6),
                    "s_k": round(sk, 6),
                }
            )

        # Tibshirani 1-SE selection rule: smallest k such that Gap(k) >= Gap(k+1) - s_{k+1}
        opt_k = int(curve[-1]["k"])
        for idx in range(len(curve) - 1):
            if float(curve[idx]["gap"]) >= float(curve[idx + 1]["gap"]) - float(curve[idx + 1]["s_k"]):
                opt_k = int(curve[idx]["k"])
                break

        return {"optimal_k": opt_k, "n_refs": b_refs, "curve": curve}

    @staticmethod
    def compare_algorithms(
        df: pd.DataFrame,
        *,
        feature_cols: list[str] | None = None,
        n_clusters: int = 4,
        scaler: Literal["standard", "robust", "minmax", "none"] = "standard",
        eps: float = 0.7,
        min_samples: int = 5,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """Benchmark KMeans, Agglomerative (Ward), GMM, BIRCH, and DBSCAN on internal validity metrics."""
        cols = feature_cols or list(df.select_dtypes(include=[np.number]).columns)
        if not cols:
            cols = list(df.columns)
        rows: list[dict[str, Any]] = []
        for algo in ("kmeans", "agglomerative", "gmm", "birch", "dbscan"):
            try:
                eng = ClusteringEngine(
                    algorithm=algo,
                    n_clusters=n_clusters,
                    scaler=scaler,
                    eps=eps,
                    min_samples=min_samples,
                    random_seed=random_seed,
                )
                res = eng.fit_predict_with_profile(df, feature_cols=cols, run_auto_k=False)
                rows.append(
                    {
                        "algorithm": algo,
                        "n_clusters_found": res.n_clusters_found,
                        "noise_pct": round(res.noise_ratio * 100.0, 2),
                        "silhouette": res.metrics.get("silhouette", 0.0),
                        "calinski_harabasz": res.metrics.get("calinski_harabasz", 0.0),
                        "davies_bouldin": res.metrics.get("davies_bouldin", 0.0),
                    }
                )
            except Exception:
                continue
        rows.sort(key=lambda r: float(r["silhouette"]), reverse=True)
        return pd.DataFrame(rows)

    def fit_predict_with_profile(
        self,
        df: pd.DataFrame,
        *,
        feature_cols: list[str] | None = None,
        run_auto_k: bool = False,
        k_min: int = 2,
        k_max: int = 8,
    ) -> ClusteringResult:
        """Fit clustering model, compute validity indices, 2D PCA coordinates, ANOVA separation & cluster profiles."""
        cols = [c for c in (feature_cols or list(df.columns)) if c in df.columns]
        if not cols:
            raise ModelTrainingError("No valid columns supplied for clustering.")

        k_curve: list[dict[str, Any]] = []
        if run_auto_k and self.algorithm in ("kmeans", "agglomerative", "gmm", "birch"):
            auto_algo: Literal["kmeans", "agglomerative", "gmm", "birch"] = (
                self.algorithm  # type: ignore[assignment]
            )
            auto_res = self.auto_select_k(
                df,
                feature_cols=cols,
                k_min=k_min,
                k_max=k_max,
                algorithm=auto_algo,
                scaler=self.scaler_type,
                random_seed=self.random_seed,
            )
            self.n_clusters = int(auto_res["best_k"])
            k_curve = list(auto_res["curve"])

        X_sub = df[cols].copy()
        X_s = self._prepare_matrix(X_sub, fit_scaler=True)
        self._X_scaled_ = X_s
        self._estimator, labels = self._build_and_fit(X_s)
        self._labels_ = labels
        self.is_fitted_ = True

        # Internal validity metrics computed on non-noise points when noise exists
        non_noise_mask = labels != -1
        if int(np.sum(non_noise_mask)) >= 3 and len(np.unique(labels[non_noise_mask])) >= 2:
            metrics = MetricsEvaluator.clustering_metrics(X_s[non_noise_mask], labels[non_noise_mask])
        else:
            metrics = MetricsEvaluator.clustering_metrics(X_s, labels)

        if hasattr(self._estimator, "inertia_"):
            metrics["inertia"] = round(float(self._estimator.inertia_), 4)
        if hasattr(self._estimator, "bic"):
            metrics["bic"] = round(float(self._estimator.bic(X_s)), 4)
            metrics["aic"] = round(float(self._estimator.aic(X_s)), 4)

        noise_count = int(np.sum(labels == -1))
        noise_ratio = float(noise_count / max(len(labels), 1))
        unique_clusters = sorted(int(c) for c in np.unique(labels) if c != -1)
        n_found = len(unique_clusters)

        # 2D PCA projection for visualization
        if X_s.shape[1] >= 2 and len(X_s) >= 2:
            pca = PCA(n_components=2, random_state=self.random_seed)
            coords = pca.fit_transform(X_s)
            exp_var = (round(float(pca.explained_variance_ratio_[0]) * 100.0, 2), round(float(pca.explained_variance_ratio_[1]) * 100.0, 2))
        else:
            coords = np.column_stack([X_s[:, 0], np.zeros(len(X_s))])
            exp_var = (100.0, 0.0)

        pca_df = pd.DataFrame(
            {
                "PC1": np.round(coords[:, 0], 4),
                "PC2": np.round(coords[:, 1], 4),
                "Cluster": ["Noise (-1)" if int(lbl) == -1 else f"Cluster {int(lbl)}" for lbl in labels],
            }
        )

        # Statistical cluster profiles + top distinguishing numeric features (z-shift from global mean)
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        global_means = {c: float(pd.to_numeric(df[c], errors="coerce").mean()) for c in num_cols}
        global_stds = {c: max(float(pd.to_numeric(df[c], errors="coerce").std(ddof=1)), 1e-9) for c in num_cols}

        profiles: list[dict[str, Any]] = []
        for cid in sorted(int(c) for c in np.unique(labels)):
            mask = labels == cid
            sub = df.loc[mask]
            row_p: dict[str, Any] = {
                "Cluster": "Noise (-1)" if cid == -1 else f"Cluster {cid}",
                "Size": int(np.sum(mask)),
                "Share %": round(float(np.sum(mask) / max(len(labels), 1) * 100.0), 2),
            }
            z_shifts: list[tuple[str, float]] = []
            for c in num_cols[:10]:
                c_mean = float(pd.to_numeric(sub[c], errors="coerce").mean())
                row_p[f"{c} (mean)"] = round(c_mean, 4)
                z_shifts.append((c, (c_mean - global_means[c]) / global_stds[c]))
            z_shifts.sort(key=lambda kv: abs(kv[1]), reverse=True)
            row_p["Key Drivers (vs. Overall)"] = ", ".join(
                f"{fname} ({'↑' if z > 0 else '↓'}{abs(z):.2f}σ)" for fname, z in z_shifts[:3]
            )
            profiles.append(row_p)

        # Cross-cluster ANOVA separation test per numeric feature
        sep_tests: list[dict[str, Any]] = []
        if n_found >= 2:
            for c in num_cols[:20]:
                grps = [
                    pd.to_numeric(df.loc[labels == cid, c], errors="coerce").dropna().to_numpy(dtype=float)
                    for cid in unique_clusters
                ]
                grps = [g for g in grps if len(g) >= 2]
                if len(grps) >= 2:
                    f_stat, p_val = stats.f_oneway(*grps)
                    sep_tests.append(
                        {
                            "Feature": c,
                            "ANOVA F-Statistic": round(float(np.nan_to_num(f_stat, nan=0.0)), 4),
                            "p-value (Sig)": round(float(np.nan_to_num(p_val, nan=1.0)), 6),
                            "Separates Clusters?": "Yes (p < 0.05)" if float(p_val) < 0.05 else "No",
                        }
                    )
            sep_tests.sort(key=lambda r: float(r["ANOVA F-Statistic"]), reverse=True)

        probs = self.predict_proba_clusters(X_sub) if self.algorithm == "gmm" else None

        return ClusteringResult(
            algorithm=self.algorithm,
            n_clusters_found=n_found,
            noise_count=noise_count,
            noise_ratio=round(noise_ratio, 4),
            labels=labels,
            metrics=metrics,
            feature_columns=cols,
            cluster_profiles=profiles,
            feature_separation_tests=sep_tests,
            pca_2d=pca_df,
            pca_explained_variance=exp_var,
            k_selection_curve=k_curve,
            probabilities=probs,
        )

    @staticmethod
    def k_prototypes_fit(
        df: pd.DataFrame,
        *,
        n_clusters: int = 3,
        gamma: float = 0.5,
        max_iter: int = 30,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """K-Prototypes (Huang, 1998) clustering for mixed numeric + categorical data."""
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        cat_cols = [c for c in df.columns if c not in num_cols]
        if not num_cols and not cat_cols:
            raise ValueError("k_prototypes_fit requires at least one column.")
        n = len(df)
        if n < n_clusters or n_clusters < 2:
            raise ValueError("n_clusters must be >= 2 and <= number of rows.")

        rng = np.random.default_rng(random_state)
        if num_cols:
            num_raw = df[num_cols].fillna(df[num_cols].median()).to_numpy(dtype=float)
            mu = num_raw.mean(axis=0)
            sd = np.where(num_raw.std(axis=0) > 1e-12, num_raw.std(axis=0), 1.0)
            X_num = (num_raw - mu) / sd
        else:
            X_num = np.zeros((n, 0), dtype=float)

        X_cat = df[cat_cols].fillna("MISSING").astype(str).to_numpy() if cat_cols else np.empty((n, 0), dtype=object)

        init_idx = rng.choice(n, size=n_clusters, replace=False)
        num_centers = X_num[init_idx].copy()
        cat_centers = X_cat[init_idx].copy()
        labels = np.zeros(n, dtype=int)

        for _ in range(max_iter):
            new_labels = np.empty(n, dtype=int)
            for i in range(n):
                best_k = 0
                best_dist = np.inf
                for k in range(n_clusters):
                    d_num = float(np.sum((X_num[i] - num_centers[k]) ** 2)) if X_num.shape[1] > 0 else 0.0
                    d_cat = float(np.sum(X_cat[i] != cat_centers[k])) if X_cat.shape[1] > 0 else 0.0
                    dist = d_num + gamma * d_cat
                    if dist < best_dist:
                        best_dist = dist
                        best_k = k
                new_labels[i] = best_k

            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            for k in range(n_clusters):
                mask = labels == k
                if not mask.any():
                    continue
                if X_num.shape[1] > 0:
                    num_centers[k] = X_num[mask].mean(axis=0)
                if X_cat.shape[1] > 0:
                    for j in range(X_cat.shape[1]):
                        vals, cnts = np.unique(X_cat[mask, j], return_counts=True)
                        cat_centers[k, j] = vals[int(np.argmax(cnts))]

        return {
            "n_clusters": int(n_clusters),
            "labels": labels.tolist(),
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols,
        }

    @staticmethod
    def bootstrap_stability_ari(
        df: pd.DataFrame,
        *,
        n_clusters: int = 3,
        n_bootstraps: int = 20,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Evaluate clustering stability via bootstrap resampling and Adjusted Rand Index (ARI) on shared points."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        from sklearn.preprocessing import StandardScaler

        num = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna()
        n = len(num)
        if n < max(15, n_clusters * 3) or num.shape[1] < 1:
            return {"evaluable": False, "mean_ari": 0.0, "std_ari": 0.0, "n_bootstraps": 0}
        X = StandardScaler().fit_transform(num.to_numpy(dtype=float))
        ref_labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto").fit_predict(X)
        rng = np.random.default_rng(random_state)
        aris: list[float] = []
        for b in range(n_bootstraps):
            idx = rng.integers(0, n, size=n)
            uniq_idx, first_pos = np.unique(idx, return_index=True)
            if len(uniq_idx) < n_clusters * 2:
                continue
            b_labels = KMeans(n_clusters=n_clusters, random_state=random_state + b + 1, n_init="auto").fit_predict(X[idx])
            ari = float(adjusted_rand_score(ref_labels[uniq_idx], b_labels[first_pos]))
            aris.append(ari)
        return {
            "evaluable": True,
            "n_clusters": int(n_clusters),
            "n_bootstraps": len(aris),
            "mean_ari": round(float(np.mean(aris)), 4) if aris else 0.0,
            "std_ari": round(float(np.std(aris)), 4) if aris else 0.0,
        }

    @staticmethod
    def hopkins_statistic(df: pd.DataFrame, *, sample_size: int = 50, random_state: int = 42) -> dict[str, Any]:
        """Compute the dimension-powered Hopkins statistic in the standardized bounding box."""
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        num = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna()
        num = num.loc[:, num.nunique() > 1]
        n, d = num.shape
        if n < 20 or d < 1:
            return {"evaluable": False, "hopkins": None, "tendency": "insufficient_data"}
        m = max(1, min(sample_size, max(5, n // 5)))
        X = StandardScaler().fit_transform(num.to_numpy(dtype=float))
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(n, size=m, replace=False)
        X_sample = X[sample_idx]
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        U = rng.uniform(mins, maxs, size=(m, d))

        nbrs = NearestNeighbors(n_neighbors=2).fit(X)
        u_dist, _ = nbrs.kneighbors(U, n_neighbors=1)
        w_dist, _ = nbrs.kneighbors(X_sample, n_neighbors=2)
        distance_scale = max(float(u_dist.max()), float(w_dist.max()))
        if distance_scale == 0.0:
            return {"evaluable": False, "hopkins": None, "tendency": "degenerate_distances"}
        u_sum = float(np.sum((u_dist[:, 0] / distance_scale) ** d))
        w_sum = float(np.sum((w_dist[:, 1] / distance_scale) ** d))
        h = u_sum / (u_sum + w_sum)
        tendency = "strong_clusters" if h >= 0.75 else ("moderate_clusters" if h >= 0.6 else "uniform_or_random")
        return {
            "evaluable": True, "hopkins": h, "tendency": tendency,
            "sample_size": int(m), "distance_power": d,
        }



