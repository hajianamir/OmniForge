"""Gated cross-modal deep fusion network combining tabular, text, image, and sequence modalities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from omniforge.core.types import TaskType
from omniforge.features.embeddings import EmbeddingGenerator
from omniforge.models.base import BaseOmniForgeModel
from omniforge.models.deep.mlp import MLPModel


class MultimodalFusionModel(BaseOmniForgeModel):
    """Gated cross-modal neural network fusing heterogeneous modality representations."""

    model_family = "deep_multimodal"
    framework = "uap_multimodal_fusion"

    def __init__(
        self,
        embedding_dim: int = 24,
        hidden_dims: tuple[int, ...] = (64, 32),
        epochs: int = 60,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="MultimodalFusionNet",
            task_type=task_type,
            embedding_dim=embedding_dim,
            hidden_dims=list(hidden_dims),
            epochs=epochs,
            **kwargs,
        )
        self.embedding_dim = int(embedding_dim)
        self.hidden_dims = list(hidden_dims)
        self.epochs = int(epochs)
        self._text_embedders: dict[str, EmbeddingGenerator] = {}
        self._df_cols: dict[str, list[str]] = {}
        self._df_medians: dict[str, dict[str, float]] = {}
        self._scalers: dict[str, StandardScaler] = {}
        self._modality_gates: dict[str, np.ndarray] = {}
        self._fusion_head: MLPModel | None = None
        self._has_text = False
        self._modality_keys: list[str] = []

    def fit_multimodal(
        self,
        modalities: dict[str, Any],
        y: np.ndarray | pd.Series,
    ) -> MultimodalFusionModel:
        """Fit gated fusion over named modalities (e.g. ``{"tabular": df, "text": texts, "image": arr}``)."""
        self._modality_keys = sorted(modalities.keys())
        fused = self._encode_modalities(modalities, fitting=True)
        self.fit(fused, y)
        return self

    def predict_multimodal(self, modalities: dict[str, Any]) -> np.ndarray:
        """Predict from a dictionary of aligned modality inputs."""
        self._check_is_fitted()
        fused = self._encode_modalities(modalities, fitting=False)
        return self.predict(fused)

    def predict_proba_multimodal(self, modalities: dict[str, Any]) -> np.ndarray:
        """Predict class probabilities from multimodal inputs."""
        self._check_is_fitted()
        fused = self._encode_modalities(modalities, fitting=False)
        return self.predict_proba(fused)

    def _encode_modalities(self, modalities: dict[str, Any], *, fitting: bool) -> np.ndarray:
        blocks: list[np.ndarray] = []
        rng = np.random.default_rng(self.random_seed)
        keys = self._modality_keys or sorted(modalities.keys())

        for key in keys:
            data = modalities[key]
            if key == "text" or key in self._text_embedders or (isinstance(data, list) and data and isinstance(data[0], str)):
                if fitting:
                    self._has_text = True
                    emb = EmbeddingGenerator(embedding_dim=self.embedding_dim, random_seed=self.random_seed)
                    arr = emb.fit_transform_text([str(x) for x in data]).astype(np.float64)
                    self._text_embedders[key] = emb
                else:
                    arr = self._text_embedders[key].transform_text([str(x) for x in data]).astype(np.float64)
            elif isinstance(data, pd.DataFrame):
                if fitting:
                    num_df = data.select_dtypes(include=[np.number]).copy()
                    cols = list(num_df.columns)
                    self._df_cols[key] = cols
                    meds: dict[str, float] = {}
                    for col in cols:
                        vals = pd.to_numeric(num_df[col], errors="coerce")
                        med = float(vals.median()) if vals.notna().any() else 0.0
                        meds[str(col)] = med
                        num_df[col] = vals.fillna(med)
                    self._df_medians[key] = meds
                    arr = num_df.to_numpy(dtype=np.float64)
                else:
                    cols = self._df_cols.get(key, list(data.select_dtypes(include=[np.number]).columns))
                    meds = self._df_medians.get(key, {})
                    aligned = pd.DataFrame(index=data.index)
                    for col in cols:
                        med = meds.get(str(col), 0.0)
                        if col in data.columns:
                            aligned[col] = pd.to_numeric(data[col], errors="coerce").fillna(med)
                        else:
                            aligned[col] = med
                    arr = aligned.to_numpy(dtype=np.float64)
            else:
                arr = np.asarray(data, dtype=np.float64)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                elif arr.ndim > 2:
                    arr = arr.reshape(arr.shape[0], -1)

            if fitting:
                scaler = StandardScaler()
                scaled = scaler.fit_transform(arr)
                self._scalers[key] = scaler
                # Learnable-style modality gating projection
                self._modality_gates[key] = rng.normal(
                    0.0, 1.0 / np.sqrt(max(scaled.shape[1], 1)), size=(scaled.shape[1], scaled.shape[1])
                )
            else:
                scaled = self._scalers[key].transform(arr)

            gate = 1.0 / (1.0 + np.exp(-np.clip(scaled @ self._modality_gates[key], -15.0, 15.0)))
            blocks.append(scaled * gate)

        return np.hstack(blocks)

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        self._fusion_head = MLPModel(
            hidden_dims=self.hidden_dims,
            epochs=self.epochs,
            task_type=self.task_type,
            random_seed=self.random_seed,
            track_carbon=False,
        )
        self._fusion_head.fit(X, y)

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        assert self._fusion_head is not None
        return self._fusion_head.predict(X)

    def _predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        assert self._fusion_head is not None
        return self._fusion_head.predict_proba(X)
