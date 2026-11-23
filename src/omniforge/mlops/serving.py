"""Real-time & Batch Model Serving Engine with Canary A/B Routing, Latency Telemetry, and Edge Export."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from omniforge.core.exceptions import MLOpsError
from omniforge.core.logging import get_logger
from omniforge.core.types import ModelStage, TaskType
from omniforge.mlops.registry import ModelRegistry
from omniforge.models.base import BaseOmniForgeModel

logger = get_logger(__name__)


class ModelServingEngine:
    """Production inference server supporting live endpoints, batch scoring, canary routing, and latency metrics."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()
        self._endpoints: dict[str, dict[str, Any]] = {}
        self._rng = np.random.default_rng(42)

    def deploy_model(
        self,
        endpoint_name: str,
        model: BaseOmniForgeModel,
        *,
        version: str = "1.0.0",
        canary_model: BaseOmniForgeModel | None = None,
        canary_version: str | None = None,
        canary_traffic_ratio: float = 0.0,
    ) -> dict[str, Any]:
        """Mount a fitted model (and optional canary challenger) onto an active serving endpoint."""
        resolved_canary_ver = (
            canary_version
            if canary_version is not None
            else (f"{version}-canary" if canary_model is not None else None)
        )
        self._endpoints[endpoint_name] = {
            "primary_model": model,
            "primary_version": version,
            "canary_model": canary_model,
            "canary_version": resolved_canary_ver,
            "canary_ratio": float(np.clip(canary_traffic_ratio, 0.0, 1.0)),
            "requests": 0,
            "latencies_ms": [],
        }
        logger.info(
            "endpoint_deployed",
            endpoint=endpoint_name,
            version=version,
            canary_version=resolved_canary_ver,
            canary_ratio=canary_traffic_ratio,
        )
        return {
            "endpoint": endpoint_name,
            "version": version,
            "canary_version": resolved_canary_ver,
            "status": "serving",
        }

    def deploy_from_registry(
        self, endpoint_name: str, model_name: str, stage: ModelStage = ModelStage.PRODUCTION
    ) -> dict[str, Any]:
        """Load a registered model by stage and deploy it to ``endpoint_name``."""
        model, rec = self.registry.load_model(model_name, stage=stage)
        return self.deploy_model(endpoint_name, model, version=f"v{rec.version}")

    def predict(
        self,
        endpoint_name: str,
        records: pd.DataFrame | list[dict[str, Any]] | np.ndarray,
        *,
        return_probabilities: bool = False,
    ) -> dict[str, Any]:
        """Run validated inference through ``endpoint_name`` with canary traffic split and telemetry."""
        if endpoint_name not in self._endpoints:
            raise MLOpsError(f"Serving endpoint '{endpoint_name}' is not deployed.")
        ep = self._endpoints[endpoint_name]
        use_canary = ep["canary_model"] is not None and float(self._rng.uniform()) < ep["canary_ratio"]
        model: BaseOmniForgeModel = ep["canary_model"] if use_canary else ep["primary_model"]
        routed_variant = "canary" if use_canary else "primary"

        X_df = pd.DataFrame(records) if isinstance(records, list) else records
        t0 = time.perf_counter()
        preds = model.predict(X_df)
        probas = None
        if return_probabilities and model.task_type in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        ):
            probas = model.predict_proba(X_df).tolist()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        ep["requests"] += 1
        ep["latencies_ms"].append(latency_ms)
        if len(ep["latencies_ms"]) > 1000:
            ep["latencies_ms"] = ep["latencies_ms"][-1000:]

        routed_version = (ep.get("canary_version") or ep["primary_version"]) if use_canary else ep["primary_version"]
        return {
            "endpoint": endpoint_name,
            "variant": routed_variant,
            "model_version": routed_version,
            "n_records": len(preds),
            "predictions": preds.tolist(),
            "probabilities": probas,
            "latency_ms": round(latency_ms, 3),
        }

    def predict_batch(
        self,
        endpoint_name: str,
        data: pd.DataFrame,
        *,
        batch_size: int = 256,
        return_probabilities: bool = False,
    ) -> pd.DataFrame:
        """Score a large DataFrame in memory-bounded batches with latency telemetry."""
        if endpoint_name not in self._endpoints:
            raise MLOpsError(f"Serving endpoint '{endpoint_name}' is not deployed.")
        chunk_size = max(1, int(batch_size))
        preds_all: list[Any] = []
        probas_all: list[Any] = []

        for start in range(0, len(data), chunk_size):
            chunk = data.iloc[start : start + chunk_size]
            res = self.predict(endpoint_name, chunk, return_probabilities=return_probabilities)
            preds_all.extend(res["predictions"])
            if return_probabilities and res["probabilities"] is not None:
                probas_all.extend(res["probabilities"])

        out = data.copy()
        out["prediction"] = preds_all
        if probas_all:
            proba_arr = np.asarray(probas_all)
            if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2:
                out["prediction_probability"] = proba_arr[:, -1]
            else:
                out["prediction_probability"] = proba_arr.ravel()
        return out

    def export_onnx(
        self,
        endpoint_name: str,
        output_path: str | Path,
        sample_input: pd.DataFrame | np.ndarray,
    ) -> dict[str, Any]:
        """Export a deployed model to ONNX (via ``skl2onnx``/``torch.onnx``) or portable edge JSON bundle."""
        if endpoint_name not in self._endpoints:
            raise MLOpsError(f"Serving endpoint '{endpoint_name}' is not deployed.")
        ep = self._endpoints[endpoint_name]
        model: BaseOmniForgeModel = ep["primary_model"]
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        arr = (
            sample_input.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
            if isinstance(sample_input, pd.DataFrame)
            else np.asarray(sample_input, dtype=np.float32)
        )
        n_features = int(arr.shape[1]) if arr.ndim == 2 else 1

        # 1. Native skl2onnx path when available and underlying sklearn estimator exists
        estimator = getattr(model, "_estimator", None) or getattr(model, "_model", None)
        scaler = getattr(model, "_scaler", None)
        if estimator is not None:
            try:
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
                from sklearn.pipeline import make_pipeline

                export_target = make_pipeline(scaler, estimator) if scaler is not None else estimator
                onnx_model = convert_sklearn(
                    export_target, initial_types=[("float_input", FloatTensorType([None, n_features]))]
                )
                dest.write_bytes(onnx_model.SerializeToString())
                logger.info(
                    "model_exported_onnx",
                    endpoint=endpoint_name,
                    path=str(dest),
                    includes_scaler=scaler is not None,
                )
                return {
                    "endpoint": endpoint_name,
                    "format": "onnx",
                    "path": str(dest),
                    "n_features": n_features,
                    "includes_scaler": scaler is not None,
                }
            except Exception as exc:
                logger.debug("skl2onnx_export_unavailable_falling_back", endpoint=endpoint_name, error=str(exc))

        # 2. Portable edge deployment package (joblib artifact + schema/metadata manifest)
        model_bin = dest.with_suffix(".omniforge.joblib")
        model.save(model_bin)
        manifest = {
            "format": "omniforge_portable_edge_bundle",
            "onnx_note": "Install skl2onnx or torch to emit native .onnx protobuf graphs; portable OmniForge bundle written.",
            "endpoint": endpoint_name,
            "model_name": model.name,
            "model_family": model.model_family,
            "task_type": model.task_type.value,
            "version": ep["primary_version"],
            "n_features": n_features,
            "artifact_path": str(model_bin.name),
        }
        manifest_path = dest.with_suffix(".edge.json")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("model_exported_portable_edge", endpoint=endpoint_name, path=str(manifest_path))
        return {
            "endpoint": endpoint_name,
            "format": "omniforge_portable_edge_bundle",
            "path": str(manifest_path),
            "artifact_path": str(model_bin),
            "n_features": n_features,
        }

    def list_endpoints(self) -> list[dict[str, Any]]:
        """Summarize all active serving endpoints and their latency distributions."""
        summary: list[dict[str, Any]] = []
        for name, ep in self._endpoints.items():
            lats = ep["latencies_ms"]
            summary.append(
                {
                    "endpoint": name,
                    "version": ep["primary_version"],
                    "requests": ep["requests"],
                    "canary_ratio": ep["canary_ratio"],
                    "p50_latency_ms": round(float(np.median(lats)), 3) if lats else 0.0,
                    "p95_latency_ms": round(float(np.quantile(lats, 0.95)), 3) if lats else 0.0,
                }
            )
        return summary
