"""Unified sklearn-compatible BaseOmniForgeModel contract with metadata, carbon tracking, and serialization."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

from omniforge.core.carbon import CarbonTracker
from omniforge.core.constants import DEFAULT_RANDOM_SEED
from omniforge.core.exceptions import ModelNotFittedError, ModelSerializationError, ModelTrainingError, SecurityError
from omniforge.core.logging import get_logger
from omniforge.core.security import ArtifactSigner, EncryptionManager, safe_joblib_load_bytes
from omniforge.core.types import ModelMetadata, TaskType
from omniforge.core.utils import compute_file_checksum, set_global_seed

_MODEL_HMAC_CONTEXT = b"omniforge:model:v1"

logger = get_logger(__name__)


class BaseOmniForgeModel(ABC):
    """Abstract base contract implemented by every supervised, unsupervised, and deep OmniForge model.

    Provides:
        - Scikit-learn compatible ``fit`` / ``predict`` / ``predict_proba`` / ``score`` API.
        - Automatic global seeding, timing, and carbon footprint measurement on every ``fit``.
        - Rich :class:`ModelMetadata` tracking hyperparameters, schema, metrics, and lineage.
        - Atomic ``save`` / ``load`` with SHA-256 artifact checksums and optional encryption at rest.
    """

    model_family: str = "generic"
    framework: str = "uap"

    def __init__(
        self,
        *,
        name: str | None = None,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        random_seed: int = DEFAULT_RANDOM_SEED,
        track_carbon: bool = True,
        country_iso: str = "DEU",
        **hyperparameters: Any,
    ) -> None:
        self.name = name or self.__class__.__name__
        self.task_type = task_type
        self.random_seed = random_seed
        self.track_carbon = track_carbon
        self.country_iso = country_iso
        self.hyperparameters: dict[str, Any] = dict(hyperparameters)
        self.is_fitted_: bool = False
        self.feature_names_: list[str] = []
        self.classes_: np.ndarray | None = None
        self.metadata_: ModelMetadata = ModelMetadata(
            name=self.name,
            model_family=self.model_family,
            task_type=self.task_type,
            framework=self.framework,
            hyperparameters=self.hyperparameters,
            random_seed=self.random_seed,
        )

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None, **kwargs: Any) -> BaseOmniForgeModel:
        """Fit the model while recording metadata, timing, and carbon emissions."""
        set_global_seed(self.random_seed)
        X_arr, feat_names = self._prepare_X(X, fitting=True)
        y_arr = None if y is None else np.asarray(y)

        if y_arr is not None and self.task_type in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        ):
            self.classes_ = np.unique(y_arr)
            if len(self.classes_) > 2:
                self.task_type = TaskType.MULTICLASS_CLASSIFICATION

        start = time.perf_counter()
        try:
            with CarbonTracker(
                f"train:{self.name}",
                country_iso=self.country_iso,
                enabled=self.track_carbon,
            ) as tracker:
                self._fit(X_arr, y_arr, **kwargs)
        except Exception as exc:
            raise ModelTrainingError(
                f"Training failed for model '{self.name}' ({self.model_family}).",
                context={"n_samples": X_arr.shape[0], "n_features": X_arr.shape[1]},
                suggestions=["Verify input feature scaling, missing values, and target alignment."],
                cause=exc,
            ) from exc

        elapsed = time.perf_counter() - start
        self.is_fitted_ = True
        self.feature_names_ = feat_names
        self.metadata_.task_type = self.task_type
        self.metadata_.feature_names = feat_names
        self.metadata_.classes = self.classes_.tolist() if self.classes_ is not None else []
        self.metadata_.training_rows = int(X_arr.shape[0])
        self.metadata_.training_features = int(X_arr.shape[1])
        self.metadata_.training_duration_seconds = round(elapsed, 6)
        self.metadata_.carbon_report = tracker.report
        logger.info(
            "model_fitted",
            model=self.name,
            family=self.model_family,
            rows=X_arr.shape[0],
            features=X_arr.shape[1],
            duration_s=round(elapsed, 4),
        )
        return self

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Generate predictions after ensuring the model has been fitted."""
        self._check_is_fitted()
        X_arr, _ = self._prepare_X(X, fitting=False)
        return np.asarray(self._predict(X_arr, **kwargs))

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict class probabilities for classification models."""
        self._check_is_fitted()
        X_arr, _ = self._prepare_X(X, fitting=False)
        return np.asarray(self._predict_proba(X_arr, **kwargs), dtype=float)

    def score(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> float:
        """Compute default scikit-learn compatible score (accuracy or R^2)."""
        preds = self.predict(X)
        y_arr = np.asarray(y)
        if self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            return float(accuracy_score(y_arr, preds))
        return float(r2_score(y_arr, preds))

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importances if exposed by the underlying estimator."""
        self._check_is_fitted()
        return {}

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return model hyperparameters (sklearn estimator API)."""
        return {"name": self.name, "task_type": self.task_type, "random_seed": self.random_seed, **self.hyperparameters}

    def set_params(self, **params: Any) -> BaseOmniForgeModel:
        """Update model hyperparameters (sklearn estimator API)."""
        for k, v in params.items():
            if k in ("name", "task_type", "random_seed", "track_carbon", "country_iso"):
                setattr(self, k, v)
            else:
                self.hyperparameters[k] = v
        self.metadata_.hyperparameters = dict(self.hyperparameters)
        return self

    def save(
        self,
        path: str | Path,
        *,
        encrypt: bool = False,
        encryption_key: str | bytes | None = None,
    ) -> Path:
        """Persist model artifact to disk with SHA-256 checksum, HMAC-SHA256 signature, and optional AES-256-GCM encryption."""
        self._check_is_fitted()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(self, target)
            if encrypt:
                enc = EncryptionManager(encryption_key)
                raw = target.read_bytes()
                target.write_bytes(enc.encrypt_bytes(raw))
            persisted_bytes = target.read_bytes()
            self.metadata_.artifact_checksum = compute_file_checksum(target)
            hmac_sig = ArtifactSigner().sign_bytes(persisted_bytes, context=_MODEL_HMAC_CONTEXT)
            meta_dict = self.metadata_.model_dump(mode="json")
            meta_dict["artifact_hmac_sha256"] = hmac_sig
            meta_path = target.with_suffix(target.suffix + ".meta.json")
            meta_path.write_text(json.dumps(meta_dict, indent=2), encoding="utf-8")
            return target
        except Exception as exc:
            raise ModelSerializationError(
                f"Failed to serialize model '{self.name}' to '{target}'.", cause=exc
            ) from exc

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        encrypted: bool = False,
        encryption_key: str | bytes | None = None,
        require_signature: bool = True,
    ) -> BaseOmniForgeModel:
        """Verify HMAC-SHA256 signature and safely load a serialized OmniForge model from disk."""
        src = Path(path)
        if not src.exists():
            raise ModelSerializationError(f"Model artifact not found: {src}")
        try:
            raw_bytes = src.read_bytes()
            if require_signature:
                meta_path = src.with_suffix(src.suffix + ".meta.json")
                if not meta_path.is_file():
                    raise SecurityError(
                        f"Missing metadata signature file '{meta_path.name}' for model artifact '{src.name}'. "
                        "Refusing to deserialize unsigned model artifact."
                    )
                meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
                expected_hmac = str(meta_payload.get("artifact_hmac_sha256") or "")
                if not ArtifactSigner().verify_bytes(raw_bytes, expected_hmac, context=_MODEL_HMAC_CONTEXT):
                    raise SecurityError(
                        f"Model artifact '{src.name}' failed HMAC-SHA256 signature verification."
                    )
            if encrypted:
                enc = EncryptionManager(encryption_key)
                payload_bytes = enc.decrypt_bytes(raw_bytes)
            else:
                payload_bytes = raw_bytes
            loaded = safe_joblib_load_bytes(payload_bytes, filename=src.name)
            if not isinstance(loaded, BaseOmniForgeModel):
                raise ModelSerializationError(f"Loaded artifact is not a BaseOmniForgeModel: {type(loaded).__name__}")
            return loaded
        except (ModelSerializationError, SecurityError):
            raise
        except Exception as exc:
            raise ModelSerializationError(f"Failed to load model artifact '{src}'.", cause=exc) from exc

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise ModelNotFittedError(
                f"Model '{self.name}' has not been fitted yet. Call .fit(X, y) first."
            )

    def _prepare_X(
        self, X: pd.DataFrame | np.ndarray, *, fitting: bool
    ) -> tuple[np.ndarray, list[str]]:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            if fitting:
                self._cat_columns_ = [
                    str(c) for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
                ]
                self._cat_mappings_: dict[str, dict[str, int]] = {}
                for col in self._cat_columns_:
                    cats = sorted(df[col].astype(str).fillna("MISSING").unique().tolist())
                    self._cat_mappings_[col] = {cat: idx for idx, cat in enumerate(cats)}
            for col in getattr(self, "_cat_columns_", []):
                if col in df.columns:
                    mapping = getattr(self, "_cat_mappings_", {}).get(col, {})
                    df[col] = df[col].astype(str).fillna("MISSING").map(lambda v, _m=mapping: _m.get(v, -1)).astype(float)
            if not fitting and self.feature_names_:
                for req in self.feature_names_:
                    if req not in df.columns:
                        df[req] = getattr(self, "_num_medians_", {}).get(req, 0.0)
                df = df[self.feature_names_]
            feat_names = [str(c) for c in df.columns]
            num_df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            if fitting:
                self._num_medians_: dict[str, float] = {
                    str(c): (float(num_df[c].median()) if num_df[c].notna().any() else 0.0)
                    for c in num_df.columns
                }
            med_map = getattr(self, "_num_medians_", {})
            for col in num_df.columns:
                if num_df[col].isna().any():
                    num_df[col] = num_df[col].fillna(med_map.get(str(col), 0.0))
            arr = num_df.to_numpy(dtype=np.float64)
            return arr, feat_names

        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        if fitting:
            col_meds = np.nanmedian(arr, axis=0)
            self._arr_medians_ = np.where(np.isfinite(col_meds), col_meds, 0.0)
        meds = getattr(self, "_arr_medians_", None)
        if np.isnan(arr).any():
            if meds is not None and len(meds) == arr.shape[1]:
                inds = np.where(np.isnan(arr))
                arr = arr.copy()
                arr[inds] = np.take(meds, inds[1])
            else:
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        feat_names = (
            self.feature_names_
            if (not fitting and self.feature_names_ and len(self.feature_names_) == arr.shape[1])
            else [f"f_{i}" for i in range(arr.shape[1])]
        )
        return arr, feat_names

    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        """Internal model fitting implementation."""

    @abstractmethod
    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Internal prediction implementation."""

    def _predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Default probability projection from discrete predictions when native probabilities are unavailable."""
        preds = self._predict(X, **kwargs)
        classes = self.classes_ if self.classes_ is not None else np.unique(preds)
        proba = np.zeros((len(preds), len(classes)), dtype=float)
        for idx, cls_val in enumerate(classes):
            proba[:, idx] = (preds == cls_val).astype(float)
        return proba


class SklearnEstimatorAdapter(BaseOmniForgeModel):
    """BaseOmniForgeModel wrapper for fitted scikit-learn estimators (enabling signed Registry & Serving)."""

    model_family = "sklearn_adapter"
    framework = "sklearn"

    def __init__(
        self,
        estimator: Any,
        *,
        name: str | None = None,
        task_type: TaskType | None = None,
    ) -> None:
        classes = getattr(estimator, "classes_", None)
        if task_type is None:
            if classes is not None:
                task_type = TaskType.BINARY_CLASSIFICATION if len(classes) <= 2 else TaskType.MULTICLASS_CLASSIFICATION
            else:
                task_type = TaskType.REGRESSION
        super().__init__(
            name=name or type(estimator).__name__,
            task_type=task_type,
            track_carbon=False,
        )
        self._estimator = estimator
        self.is_fitted_ = True
        self._cat_columns_ = []
        self._cat_mappings_ = {}
        feat_in = getattr(estimator, "feature_names_in_", None)
        if feat_in is not None:
            self.feature_names_ = [str(c) for c in feat_in]
            self.metadata_.feature_names = list(self.feature_names_)
        if classes is not None:
            self.classes_ = np.asarray(classes)

    def _to_est_input(self, X: np.ndarray) -> pd.DataFrame | np.ndarray:
        if self.feature_names_ and X.ndim == 2 and X.shape[1] == len(self.feature_names_):
            return pd.DataFrame(X, columns=self.feature_names_)
        return X

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        self._estimator.fit(X, y)

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return np.asarray(self._estimator.predict(self._to_est_input(X)))

    def _predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if hasattr(self._estimator, "predict_proba"):
            return np.asarray(self._estimator.predict_proba(self._to_est_input(X)), dtype=float)
        return super()._predict_proba(X, **kwargs)

