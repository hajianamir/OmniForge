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
            work = X.copy()
            # Encode categorical columns safely and impute numeric medians
            num_cols = list(work.select_dtypes(include=[np.number]).columns)
            cat_cols = [c for c in work.columns if c not in num_cols]
            for c in num_cols:
                col_vals = pd.to_numeric(work[c], errors="coerce")
                med = float(col_vals.median()) if col_vals.notna().any() else 0.0
                work[c] = col_vals.fillna(med)
            if cat_cols:
                for c in cat_cols:
                    work[c] = work[c].fillna("missing").astype(str)
                work = pd.get_dummies(work, columns=cat_cols, drop_first=False, dtype=float)
            if fit_scaler:
                self._encoded_cols_ = list(work.columns)
            elif self._encoded_cols_:
                for c in self._encoded_cols_:
                    if c not in work.columns:
                        work[c] = 0.0
                work = work[self._encoded_cols_]
            arr = work.to_numpy(dtype=float)
        else:
            arr = np.asarray(X, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if np.isnan(arr).any():
                col_med = np.nanmedian(arr, axis=0)
                col_med = np.where(np.isnan(col_med), 0.0, col_med)
                inds = np.where(np.isnan(arr))
                arr = arr.copy()
                arr[inds] = np.take(col_med, inds[1])

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if fit_scaler:
            self._scaler = self._make_scaler()
            return self._scaler.fit_transform(arr) if self._scaler is not None else arr
        return self._scaler.transform(arr) if self._scaler is not None else arr

    def _estimate_dbscan_eps(self, X_s: np.ndarray) -> float:
        """Heuristic k-distance elbow estimate for DBSCAN eps (sampled on >15k rows for O(N) speed)."""
        n = len(X_s)
        k = min(max(self.min_samples, 2), max(n - 1, 1))
        if n <= 2:
            return 0.5
        fit_X = X_s
        if n > 15_000:
            rng = np.random.default_rng(self.random_seed)
            fit_X = X_s[rng.choice(n, size=15_000, replace=False)]
        nn = NearestNeighbors(n_neighbors=k, n_jobs=parallel_n_jobs()).fit(fit_X)
        dists, _ = nn.kneighbors(fit_X)
        k_dists = np.sort(dists[:, -1])
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
        X_s = self._prepare_matrix(X, fit_scaler=False)
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
        cols = train_res.feature_columns
        test_labels = np.asarray(self.predict(df_test[cols]), dtype=int)
        return train_res, test_labels

    def predict_proba_clusters(self, X: pd.DataFrame | np.ndarray) -> np.ndarray | None:
        """Return soft cluster membership probabilities (exact for GMM, softmax distance for centroid models)."""
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
